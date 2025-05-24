"""
Script to find optimal parameters for random walks by testing different combinations
and evaluating model performance on BFS distances.

See README.md for detailed usage instructions and examples.
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr

# Add parent directory to path for src imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the functions
from src.data.random_walks import (
    create_lrx_moves,
    first_visit_random_walks,
    random_walks_beam_nbt,
    nbt_random_walks
)

# Import models
from src.models.xgboost_model import XGBoostModel
from src.models.mlp_model import MLPModel
from src.models.catboost_model import CatBoostModel

def compute_bfs_distances(generators: list, state_size: int) -> dict:
    """
    For starting sorted state [0, 1, ..., state_size-1],
    compute the minimal number of moves (distance) to every reachable state.
    
    The allowed moves are defined by the list 'generators', where a generator g
    defines a move: new_state[i] = state[g[i]].
    """
    sorted_state = tuple(range(state_size))
    distances = {sorted_state: 0}
    queue = deque([sorted_state])
    
    while queue:
        state = queue.popleft()
        current_distance = distances[state]
        for g in generators:
            new_state = tuple(state[i] for i in g)
            if new_state not in distances:
                distances[new_state] = current_distance + 1
                queue.append(new_state)
    return distances

def run_test(func, generators, n_steps, n_walks, device):
    """Run a single test with given parameters."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    start_time = time.time()
    X, y = func(generators, n_steps, n_walks, device)
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    data_points = X.shape[0]
    memory_used = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
    points_per_second = data_points / elapsed_time
    
    return {
        'n_steps': n_steps,
        'n_walks': n_walks,
        'time': elapsed_time,
        'data_points': data_points,
        'memory_gb': memory_used,
        'points_per_second': points_per_second,
        'X': X,
        'y': y
    }

def train_and_evaluate_model(X_train, y_train, X_test, y_test, model_type='xgboost'):
    """Train a model and evaluate its performance using existing model implementations."""
    # Initialize the model based on type
    if model_type == 'xgboost':
        model = XGBoostModel()
    elif model_type == 'mlp':
        model = MLPModel()
    elif model_type == 'catboost':
        model = CatBoostModel()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train the model
    model_start_time = time.time()
    model.train(X_train, y_train)
    fit_time = time.time() - model_start_time
    
    # Make predictions
    pred_start_time = time.time()
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    predict_time = time.time() - pred_start_time
    
    # Convert to numpy for metric calculation if needed
    y_train_np = y_train.cpu().numpy() if hasattr(y_train, "cpu") else np.asarray(y_train)
    y_test_np = y_test.cpu().numpy() if hasattr(y_test, "cpu") else np.asarray(y_test)
    y_pred_train_np = y_pred_train.cpu().numpy() if hasattr(y_pred_train, "cpu") else np.asarray(y_pred_train)
    y_pred_test_np = y_pred_test.cpu().numpy() if hasattr(y_pred_test, "cpu") else np.asarray(y_pred_test)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train_np, y_pred_train_np))
    test_rmse = np.sqrt(mean_squared_error(y_test_np, y_pred_test_np))
    
    train_r2 = r2_score(y_train_np, y_pred_train_np)
    test_r2 = r2_score(y_test_np, y_pred_test_np)
    
    train_spearman, _ = spearmanr(y_train_np, y_pred_train_np)
    test_spearman, _ = spearmanr(y_test_np, y_pred_test_np)
    
    return {
        'model': model,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_spearman': train_spearman,
        'test_spearman': test_spearman,
        'fit_time': fit_time,
        'predict_time': predict_time,
        'y_pred_test': y_pred_test_np,
        'y_pred_train': y_pred_train_np
    }

def plot_results(all_results, steps_range, walks_range, best_configs, y_test, implementations, state_size, model_type):
    """Create comprehensive plots of all results in a single file."""
    # Convert to numpy if needed
    y_test_np = y_test.cpu().numpy() if hasattr(y_test, "cpu") else np.asarray(y_test)
    
    # Create a figure with 2 columns - one for parameter effects, one for scatter plots
    plt.figure(figsize=(16, 6 * len(implementations)))
    
    # Convert all_results to DataFrame
    df = pd.DataFrame(all_results)
    
    # Plot for each implementation
    for impl_idx, impl_name in enumerate(implementations.keys()):
        # Filter results for this implementation
        impl_df = df[df['implementation'] == impl_name]
        
        # 1. Plot how parameters affect correlation
        ax1 = plt.subplot(len(implementations), 2, impl_idx*2 + 1)
        
        # Create a single plot showing both n_steps and n_walks effects
        # Group by n_steps for each n_walks
        for n_walks in walks_range:
            walk_df = impl_df[impl_df['n_walks'] == n_walks]
            if not walk_df.empty:
                walk_df = walk_df.sort_values('n_steps')
                ax1.plot(walk_df['n_steps'], walk_df['test_spearman'], 
                        marker='o', label=f'n_walks={n_walks}')
        
        ax1.set_xlabel('Number of Steps')
        ax1.set_ylabel('Spearman Correlation')
        ax1.set_title(f'{impl_name}: Effect of Parameters on Correlation')
        ax1.grid(True, alpha=0.3)
        ax1.legend(title='Number of Walks')
        
        # 2. Scatter plot of predicted vs true values for best configuration
        ax2 = plt.subplot(len(implementations), 2, impl_idx*2 + 2)
        
        if impl_name in best_configs:
            config = best_configs[impl_name]
            ax2.scatter(y_test_np, config['y_pred'], s=10, alpha=0.5)
            ax2.plot([y_test_np.min(), y_test_np.max()],
                    [y_test_np.min(), y_test_np.max()], 'r--')
            
            # Show the best parameters in the title
            ax2.set_title(f'{impl_name}: Best Configuration\n'
                         f'n_steps={config["n_steps"]}, n_walks={config["n_walks"]}\n'
                         f'Correlation={config["test_spearman"]:.3f}')
            ax2.set_xlabel('True Distance')
            ax2.set_ylabel('Predicted Distance')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No valid results', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax2.transAxes)
    
    plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.95])
    plt.suptitle(
        f"Random Walk Optimization Results - {len(implementations)} Implementations",
        fontsize=16, y=0.98
    )
    
    # Save the figure
    plot_filename = f"optimize_random_walks_size{state_size}_{model_type}.png"
    plt.savefig(plot_filename, dpi=300)
    print(f"All plots saved to {plot_filename}")

def main():
    parser = argparse.ArgumentParser(description='Optimize random walk parameters and evaluate model performance')
    parser.add_argument('--state-size', type=int, default=8, help='Size of state vector')
    parser.add_argument('--implementation', type=str, default='random_walks_beam_nbt', 
                        choices=[
                            'first_visit_random_walks', 
                            'random_walks_beam_nbt',
                            'nbt_random_walks'
                        ],
                        help='Which implementation to optimize'
                        )
    parser.add_argument('--test-all', action='store_true', help='Test all implementations')
    parser.add_argument('--model-type', type=str, default='xgboost', 
                        choices=['xgboost', 'mlp', 'catboost'],
                        help='Type of model to use for evaluation'
                        )
    parser.add_argument('--steps', type=str, default=None, 
                        help='Comma-separated list of step multipliers (e.g., "0.5,1,2")')
    parser.add_argument('--walks', type=str, default=None, 
                        help='Comma-separated list of walk counts (e.g., "500,1000,5000")')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Select the implementations to test
    all_implementations = {
        'first_visit_random_walks': first_visit_random_walks,
        'random_walks_beam_nbt': random_walks_beam_nbt,
        'nbt_random_walks': nbt_random_walks
    }
    
    if args.test_all:
        implementations = all_implementations
    else:
        if args.implementation not in all_implementations:
            raise ValueError(f"Unknown implementation: {args.implementation}")
        implementations = {args.implementation: all_implementations[args.implementation]}
    
    # Generate test data
    generators = create_lrx_moves(args.state_size)
    
    # Compute BFS distances for ground truth
    print(f"Computing BFS distances for state size {args.state_size}...")
    bfs_dists = compute_bfs_distances(generators, args.state_size)
    X_test = torch.tensor(list(bfs_dists.keys()), device=device, dtype=torch.long)
    y_test = torch.tensor(list(bfs_dists.values()), device=device, dtype=torch.long)
    print(f"Generated {len(bfs_dists)} states with BFS distances")
    
    # Define parameter ranges to test
    conj_steps = int(args.state_size * (args.state_size - 1) / 2)
    
    # Parse custom step multipliers if provided
    if args.steps:
        step_multipliers = [float(x) for x in args.steps.split(',')]
        steps_range = [int(conj_steps * m) for m in step_multipliers]
    else:
        steps_range = [
            int(conj_steps * 0.5),
            int(conj_steps * 0.75),
            int(conj_steps * 1),
            int(conj_steps * 1.5),
            int(conj_steps * 2),
            int(conj_steps * 3),
            int(conj_steps * 5)
        ]
    
    # Parse custom walk counts if provided
    if args.walks:
        walks_range = [int(x) for x in args.walks.split(',')]
    else:
        walks_range = [100, 500, 1000, 5000, 10000, 15000, 20000]
    
    print(f"Testing step counts: {steps_range}")
    print(f"Testing walk counts: {walks_range}")
    
    # Store results for all implementations and parameters
    all_results = []
    best_configs = {}
    
    # For each implementation
    for name, func in implementations.items():
        print(f"\nTesting implementation: {name}")
        implementation_results = []
        
        # Run tests for all parameter combinations
        for n_steps in steps_range:
            for n_walks in walks_range:
                print(f"Testing n_steps={n_steps}, n_walks={n_walks}...")
                try:
                    # Run the random walk
                    result = run_test(func, generators, n_steps, n_walks, device)
                    
                    # Train and evaluate model
                    X_train, y_train = result['X'], result['y']
                    model_results = train_and_evaluate_model(
                        X_train, y_train, X_test, y_test, args.model_type
                    )
                    
                    # Combine results
                    combined_result = {
                        'implementation': name,
                        'n_steps': n_steps,
                        'n_walks': n_walks,
                        'data_points': result['data_points'],
                        'memory_gb': result['memory_gb'],
                        'time': result['time'],
                        'points_per_second': result['points_per_second'],
                        'train_rmse': model_results['train_rmse'],
                        'test_rmse': model_results['test_rmse'],
                        'train_r2': model_results['train_r2'],
                        'test_r2': model_results['test_r2'],
                        'train_spearman': model_results['train_spearman'],
                        'test_spearman': model_results['test_spearman'],
                        'fit_time': model_results['fit_time'],
                        'predict_time': model_results['predict_time']
                    }
                    
                    # Print results
                    print(f"  - Data points: {result['data_points']}")
                    print(f"  - Memory: {result['memory_gb']:.2f} GB")
                    print(f"  - Time: {result['time']:.4f}s")
                    print(f"  - Points/sec: {result['points_per_second']:.0f}")
                    print(f"  - Test RMSE: {model_results['test_rmse']:.4f}")
                    print(f"  - Test R²: {model_results['test_r2']:.4f}")
                    print(f"  - Test Spearman: {model_results['test_spearman']:.4f}")
                    
                    implementation_results.append(combined_result)
                    all_results.append(combined_result)
                    
                    # Save predictions for the best model (by test spearman)
                    if name not in best_configs or model_results['test_spearman'] > best_configs[name]['test_spearman']:
                        best_configs[name] = {
                            'n_steps': n_steps,
                            'n_walks': n_walks,
                            'test_r2': model_results['test_r2'],
                            'test_rmse': model_results['test_rmse'],
                            'test_spearman': model_results['test_spearman'],
                            'y_pred': model_results['y_pred_test']
                        }
                    
                except Exception as e:
                    print(f"Error with n_steps={n_steps}, n_walks={n_walks}: {e}")
        
        # Convert results to DataFrame for this implementation
        if implementation_results:
            df = pd.DataFrame(implementation_results)
            
            # Save results
            output_file = f"optimize_{name}_size{args.state_size}_{args.model_type}.csv"
            df.to_csv(output_file, index=False)
            print(f"Results saved to {output_file}")
    
    # Create comprehensive plots of all results
    if all_results:
        plot_results(all_results, steps_range, walks_range, best_configs, y_test, implementations, args.state_size, args.model_type)
    
    # Find best configuration for each implementation
    best_results = []
    for name, config in best_configs.items():
        best_results.append({
            'implementation': name,
            'n_steps': config['n_steps'],
            'n_walks': config['n_walks'],
            'test_r2': config['test_r2'],
            'test_rmse': config['test_rmse'],
            'test_spearman': config['test_spearman']
        })
    
    # Sort by Spearman correlation (higher is better)
    df_best = pd.DataFrame(best_results)
    df_best = df_best.sort_values('test_spearman', ascending=False)
    
    # Print best configurations
    print("\nBest configurations for each implementation (sorted by Spearman correlation):")
    print(df_best[['implementation', 'n_steps', 'n_walks', 'test_spearman', 'test_r2', 'test_rmse']])
    
    # Save best configurations
    best_config_file = f"optimize_best_configurations_size{args.state_size}_{args.model_type}.csv"
    df_best.to_csv(best_config_file, index=False)
    print(f"Best configurations saved to {best_config_file}")

if __name__ == "__main__":
    main() 