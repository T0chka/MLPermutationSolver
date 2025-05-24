"""
Script to profile model prediction performance with different batch sizes.
Measures elapsed time, throughput, and memory usage for model inference.

See experiments/README.md for detailed usage instructions.
"""

import time
import torch
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys
import os
from typing import Dict, List, Any

# Add parent directory to path for src imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.random_walks import (
    create_lrx_moves,
    first_visit_random_walks,
    nbt_random_walks,
    random_walks_beam_nbt
)
from src.models.catboost_model import CatBoostModel
from src.models.xgboost_model import XGBoostModel
from src.models.mlp_model import MLPModel

def predict_with_batch_size(
    model, 
    X: torch.Tensor, 
    batch_size: int
) -> Dict[str, Any]:
    """Run prediction with a specific batch size and measure performance."""
    n_samples = X.shape[0]
    results = torch.zeros(n_samples, device=X.device)
    
    # Reset memory stats before profiling
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    start_time = time.time()
    
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        batch = X[i:end_idx]
        
        # Get predictions for this batch
        batch_preds = model.predict(batch)
        results[i:end_idx] = batch_preds
    
    elapsed_time = time.time() - start_time
    
    # Get peak memory usage
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / (1024**3)  # GB
    else:
        peak_memory = 0
    
    return {
        'batch_size': batch_size,
        'elapsed_time': elapsed_time,
        'samples_per_second': n_samples / elapsed_time,
        'peak_memory_gb': peak_memory
    }

def create_performance_plots(
    df: pd.DataFrame, 
    model_type: str,
    state_size: int,
    rw_type: str
) -> None:
    """Create and save performance visualization plots."""
    try:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot elapsed time
        ax1.plot(df['batch_size'], df['elapsed_time'], 'o-', linewidth=2, color='blue')
        ax1.set_xlabel('Batch Size')
        ax1.set_ylabel('Elapsed Time (seconds)')
        ax1.set_title('Prediction Time vs Batch Size')
        ax1.grid(True)
        ax1.set_xscale('log')
        
        # Plot samples per second (throughput)
        ax2.plot(df['batch_size'], df['samples_per_second'], 'o-', linewidth=2, color='green')
        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('Samples per Second')
        ax2.set_title('Throughput vs Batch Size')
        ax2.grid(True)
        ax2.set_xscale('log')
        
        # Plot memory usage if available
        if df['peak_memory_gb'].max() > 0:
            ax3.plot(df['batch_size'], df['peak_memory_gb'], 'o-', linewidth=2, color='red')
            ax3.set_xlabel('Batch Size')
            ax3.set_ylabel('Peak Memory (GB)')
            ax3.set_title('Memory Usage vs Batch Size')
            ax3.grid(True)
            ax3.set_xscale('log')
        else:
            ax3.text(0.5, 0.5, 'Memory data not available\n(CPU mode)', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Memory Usage vs Batch Size')
        
        plt.tight_layout()
        
        # Save the plot
        plot_filename = f'profile_model_size{state_size}_{model_type}_{rw_type}.png'
        plt.savefig(plot_filename, dpi=300)
        print(f"Performance plots saved to: {plot_filename}")
        
    except Exception as e:
        print(f"Could not create plots: {e}")

def print_results_summary(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Print a formatted summary of profiling results."""
    df = pd.DataFrame(results)
    
    # Filter out error results
    valid_df = df[~df['elapsed_time'].isin([float('inf')])]
    
    if valid_df.empty:
        print("No valid results to display.")
        return df
    
    print("\n=== MODEL PREDICTION PROFILING RESULTS ===")
    print(f"{'Batch Size':<12} {'Time (s)':<10} {'Throughput':<15} {'Memory (GB)':<12}")
    print("-" * 55)
    
    for _, row in valid_df.iterrows():
        print(f"{row['batch_size']:<12} {row['elapsed_time']:<10.4f} "
              f"{row['samples_per_second']:<15.0f} {row['peak_memory_gb']:<12.2f}")
    
    # Find optimal configurations
    best_throughput_idx = valid_df['samples_per_second'].idxmax()
    best_memory_idx = valid_df['peak_memory_gb'].idxmin() if valid_df['peak_memory_gb'].max() > 0 else None
    
    print(f"\nOptimal batch size for throughput: {valid_df.iloc[best_throughput_idx]['batch_size']}")
    print(f"  - Throughput: {valid_df.iloc[best_throughput_idx]['samples_per_second']:.0f} samples/sec")
    print(f"  - Time: {valid_df.iloc[best_throughput_idx]['elapsed_time']:.4f}s")
    
    if best_memory_idx is not None:
        print(f"\nOptimal batch size for memory: {valid_df.iloc[best_memory_idx]['batch_size']}")
        print(f"  - Memory: {valid_df.iloc[best_memory_idx]['peak_memory_gb']:.2f} GB")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Profile model prediction performance with different batch sizes')
    parser.add_argument('--state-size', type=int, default=16, help='Size of permutation vector')
    parser.add_argument('--model-type', type=str, default='xgboost', 
                        choices=['xgboost', 'catboost', 'mlp'],
                        help='Type of model to profile')
    parser.add_argument('--rw-type', type=str, default='beam_nbt',
                        choices=['first_visit', 'nbt', 'beam_nbt'],
                        help='Random walk type for data generation')
    parser.add_argument('--n-walks', type=int, default=10000, help='Number of random walks')
    parser.add_argument('--batch-sizes', type=str, default='1000,10000,50000,100000',
                        help='Comma-separated list of batch sizes to test')
    parser.add_argument('--no-plot', action='store_true', help='Disable plotting')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Parse batch sizes
    batch_sizes = [int(x) for x in args.batch_sizes.split(',')]
    
    # Calculate steps based on state size
    conj_steps = int(args.state_size * (args.state_size - 1) / 2)
    print(f"State size: {args.state_size}, Steps: {conj_steps}")
    
    # Random walk functions
    rw_functions = {
        "first_visit": first_visit_random_walks,
        "nbt": nbt_random_walks,
        "beam_nbt": random_walks_beam_nbt
    }
    
    # Generate training data
    print(f"Generating data with {args.rw_type} random walks...")
    generators = create_lrx_moves(args.state_size)
    func = rw_functions[args.rw_type]
    X, y = func(generators, n_steps=conj_steps, n_walks=args.n_walks, device=device)
    print(f"Generated dataset with {X.shape[0]} samples")
    
    # Initialize and train model
    if args.model_type == "xgboost":
        model = XGBoostModel()
    elif args.model_type == "catboost":
        model = CatBoostModel()
    elif args.model_type == "mlp":
        model = MLPModel()
    
    print(f"Training {args.model_type} model...")
    start_train = time.time()
    model.train(X, y)
    train_time = time.time() - start_train
    print(f"Training completed in {train_time:.2f} seconds")
    
    # Warm-up prediction
    _ = model.predict(X[:1000])
    
    # Profile predictions with different batch sizes
    print(f"\nProfiling {args.model_type} model prediction performance...")
    results = []
    
    # Filter batch sizes that are larger than dataset
    valid_batch_sizes = [bs for bs in batch_sizes if bs <= X.shape[0]]
    
    for batch_size in valid_batch_sizes:
        print(f"Testing batch_size = {batch_size}...", end="", flush=True)
        
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            result = predict_with_batch_size(model, X, batch_size)
            result['model_type'] = args.model_type
            results.append(result)
            
            print(f" ✓ Time: {result['elapsed_time']:.4f}s, "
                  f"Throughput: {result['samples_per_second']:.0f} samples/sec")
            
        except Exception as e:
            print(f" ✗ Error: {e}")
            results.append({
                'batch_size': batch_size,
                'elapsed_time': float('inf'),
                'samples_per_second': 0,
                'peak_memory_gb': 0,
                'model_type': args.model_type,
                'error': str(e)
            })
    
    # Print results summary
    df = print_results_summary(results)
    
    # Save results to CSV
    csv_filename = f'profile_model_size{args.state_size}_{args.model_type}_{args.rw_type}.csv'
    df.to_csv(csv_filename, index=False)
    print(f"\nDetailed results saved to: {csv_filename}")
    
    # Create plots
    if not args.no_plot:
        create_performance_plots(df, args.model_type, args.state_size, args.rw_type)

if __name__ == "__main__":
    main()
