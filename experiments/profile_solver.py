"""
Script to profile solver performance with different batch sizes and parameters.
Measures execution time, memory usage, and exploration statistics for BeamSearchSolver.

See experiments/README.md for detailed usage instructions.
"""

import torch
from time import time
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys
import os
from typing import Dict, List, Any

# Add parent directory to path for src imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.solvers.simple_solver import BeamSearchSolver

class DummyModel:
    """Dummy model for solver guidance that returns zeros for all predictions."""
    
    def __init__(self, device: torch.device):
        self.device = device
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return torch.zeros(len(X), device=self.device, dtype=torch.float32)
    
    def train(self, X: torch.Tensor, y: torch.Tensor) -> None:
        pass

def run_solver_test(
    size: int,
    beam_width: int,
    batch_size: int,
    history_window_size: int = 1,
    use_x_rule: bool = False,
    target_radius: int = 0,
    device: torch.device = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """Run a single solver test with specified parameters and return results."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create test permutation (reversed sorted state)
    perm = torch.tensor(list(range(size)), device=device)
    perm = perm.flip(0)
    
    conj_steps = int(size * (size - 1) / 2)
    max_steps = conj_steps * 2
    
    # Initialize solver
    solver = BeamSearchSolver(
        state_size=size,
        beam_width=beam_width,
        max_steps=max_steps,
        use_x_rule=use_x_rule,
        target_neighborhood_radius=target_radius,
        filter_batch_size=batch_size,
        predict_batch_size=batch_size,
        history_window_size=history_window_size,
        device=device,
        verbose=verbose
    )
    
    dummy_model = DummyModel(device)
    
    # Clean memory before test
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    # Run solver
    start_time = time()
    found, steps, solution = solver.solve(start_state=perm, model=dummy_model)
    elapsed_time = time() - start_time
    
    # Get statistics from solver
    peak_memory = solver.search_stats.get('peak_memory_gb', 0)
    total_states = solver.search_stats.get('total_states_explored', 0)
    total_hashes = solver.search_stats.get('Total hashes ever seen', 0)
    current_hashes = solver.search_stats.get('Total hashes in history', 0)
    
    return {
        'size': size,
        'beam_width': beam_width,
        'batch_size': batch_size,
        'history_window_size': history_window_size,
        'use_x_rule': use_x_rule,
        'target_radius': target_radius,
        'max_steps': max_steps,
        'conjectured_steps': conj_steps,
        'solution_found': found,
        'steps_taken': steps,
        'solution_length': len(solution.split('.')) if found and solution else 0,
        'elapsed_time': elapsed_time,
        'states_explored': total_states,
        'total_hashes': total_hashes,
        'current_hashes': current_hashes,
        'peak_memory_gb': peak_memory,
        'timestamp': datetime.now().isoformat()
    }

def create_performance_plots(
    df: pd.DataFrame,
    size: int,
    beam_width: int
) -> None:
    """Create and save performance visualization plots."""
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot elapsed time
        ax1.plot(df['batch_size'], df['elapsed_time'], 'o-', linewidth=2, color='blue')
        ax1.set_xlabel('Batch Size')
        ax1.set_ylabel('Elapsed Time (seconds)')
        ax1.set_title('Execution Time vs Batch Size')
        ax1.grid(True)
        ax1.set_xscale('log')
        
        # Plot memory usage
        ax2.plot(df['batch_size'], df['peak_memory_gb'], 'o-', linewidth=2, color='red')
        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('Peak Memory (GB)')
        ax2.set_title('Memory Usage vs Batch Size')
        ax2.grid(True)
        ax2.set_xscale('log')
        
        # Plot states explored
        ax3.plot(df['batch_size'], df['states_explored'], 'o-', linewidth=2, color='green')
        ax3.set_xlabel('Batch Size')
        ax3.set_ylabel('States Explored')
        ax3.set_title('States Explored vs Batch Size')
        ax3.grid(True)
        ax3.set_xscale('log')
        
        # Plot steps taken
        ax4.plot(df['batch_size'], df['steps_taken'], 'o-', linewidth=2, color='orange')
        ax4.set_xlabel('Batch Size')
        ax4.set_ylabel('Steps Taken')
        ax4.set_title('Solution Steps vs Batch Size')
        ax4.grid(True)
        ax4.set_xscale('log')
        
        plt.suptitle(f'Solver Performance Analysis (size={size}, beam_width={beam_width})', fontsize=16)
        plt.tight_layout()
        
        # Save the plot
        plot_filename = f'profile_solver_size{size}_beam{beam_width}.png'
        plt.savefig(plot_filename, dpi=300)
        print(f"Performance plots saved to: {plot_filename}")
        
    except Exception as e:
        print(f"Could not create plots: {e}")

def print_results_summary(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Print a formatted summary of profiling results."""
    df = pd.DataFrame(results)
    
    # Filter out error results
    valid_df = df[df['elapsed_time'] != float('inf')]
    
    if valid_df.empty:
        print("No valid results to display.")
        return df
    
    print("\n=== SOLVER PROFILING RESULTS ===")
    print(f"{'Batch Size':<12} {'Time (s)':<10} {'Memory (GB)':<12} {'States':<12} {'Steps':<8} {'Found':<6}")
    print("-" * 70)
    
    for _, row in valid_df.iterrows():
        print(f"{row['batch_size']:<12} {row['elapsed_time']:<10.2f} "
              f"{row['peak_memory_gb']:<12.2f} {row['states_explored']:<12} "
              f"{row['steps_taken']:<8} {'✓' if row['solution_found'] else '✗':<6}")
    
    # Show common parameters
    if not valid_df.empty:
        common_params = {
            'Size': valid_df.iloc[0]['size'],
            'Beam Width': valid_df.iloc[0]['beam_width'],
            'History Window': valid_df.iloc[0]['history_window_size'],
            'Max Steps': valid_df.iloc[0]['max_steps'],
            'Conjectured Steps': valid_df.iloc[0]['conjectured_steps']
        }
        
        print("\n=== COMMON PARAMETERS ===")
        for param, value in common_params.items():
            print(f"{param}: {value}")
    
    # Find optimal configurations
    best_time_idx = valid_df['elapsed_time'].idxmin()
    best_memory_idx = valid_df['peak_memory_gb'].idxmin()
    
    print(f"\nOptimal batch size for time: {valid_df.iloc[best_time_idx]['batch_size']}")
    print(f"  - Time: {valid_df.iloc[best_time_idx]['elapsed_time']:.2f}s")
    print(f"  - Memory: {valid_df.iloc[best_time_idx]['peak_memory_gb']:.2f} GB")
    
    if best_memory_idx != best_time_idx:
        print(f"\nOptimal batch size for memory: {valid_df.iloc[best_memory_idx]['batch_size']}")
        print(f"  - Memory: {valid_df.iloc[best_memory_idx]['peak_memory_gb']:.2f} GB")
        print(f"  - Time: {valid_df.iloc[best_memory_idx]['elapsed_time']:.2f}s")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Profile solver performance with different batch sizes')
    parser.add_argument('--state-size', type=int, default=16, help='Size of permutation vector')
    parser.add_argument('--beam-width', type=int, default=None, help='Beam width for search (default: 2^state_size)')
    parser.add_argument('--batch-sizes', type=str, default='10000,50000,100000,500000',
                        help='Comma-separated list of batch sizes to test')
    parser.add_argument('--history-window', type=int, default=None, 
                        help='History window size (default: conjectured_steps/5)')
    parser.add_argument('--use-x-rule', action='store_true', help='Enable X-rule optimization')
    parser.add_argument('--target-radius', type=int, default=0, help='Target neighborhood radius')
    parser.add_argument('--no-plot', action='store_true', help='Disable plotting')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose solver output')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Parse batch sizes
    batch_sizes = [int(x) for x in args.batch_sizes.split(',')]
    
    # Calculate derived parameters
    conj_steps = int(args.state_size * (args.state_size - 1) / 2)
    beam_width = args.beam_width if args.beam_width is not None else 2**args.state_size
    history_window = args.history_window if args.history_window is not None else max(1, int(conj_steps/5))
    
    print(f"Profiling solver with:")
    print(f"  - State size: {args.state_size}")
    print(f"  - Beam width: {beam_width}")
    print(f"  - History window: {history_window}")
    print(f"  - Conjectured steps: {conj_steps}")
    print(f"  - Batch sizes: {batch_sizes}")
    
    # Run tests for each batch size
    results = []
    for batch_size in batch_sizes:
        print(f"\nTesting batch_size = {batch_size}...", end="", flush=True)
        try:
            result = run_solver_test(
                size=args.state_size,
                beam_width=beam_width,
                batch_size=batch_size,
                history_window_size=history_window,
                use_x_rule=args.use_x_rule,
                target_radius=args.target_radius,
                device=device,
                verbose=args.verbose
            )
            results.append(result)
            print(f" ✓ Time: {result['elapsed_time']:.2f}s, Memory: {result['peak_memory_gb']:.2f}GB")
            
        except Exception as e:
            print(f" ✗ Error: {e}")
            results.append({
                'size': args.state_size,
                'beam_width': beam_width,
                'batch_size': batch_size,
                'history_window_size': history_window,
                'use_x_rule': args.use_x_rule,
                'target_radius': args.target_radius,
                'elapsed_time': float('inf'),
                'peak_memory_gb': float('inf'),
                'states_explored': 0,
                'steps_taken': 0,
                'solution_found': False,
                'error': str(e)
            })
    
    # Print results summary
    df = print_results_summary(results)
    
    # Save results to CSV
    csv_filename = f'profile_solver_size{args.state_size}_beam{beam_width}.csv'
    df.to_csv(csv_filename, index=False)
    print(f"\nDetailed results saved to: {csv_filename}")
    
    # Create plots
    if not args.no_plot:
        create_performance_plots(df, args.state_size, beam_width)

if __name__ == "__main__":
    main()
