"""
Benchmark script for comparing different random walk implementations.
Measures execution time, memory usage, and data generation statistics.

See README.md for detailed usage instructions and examples.
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Any, Callable

# Add parent directory to path for src imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.random_walks import (
    create_lrx_moves,
    first_visit_random_walks,
    nbt_random_walks,
    random_walks_beam_nbt
)

def run_benchmark(
    func: Callable,
    generators: List[Callable],
    n_steps: int,
    n_walks: int,
    device: torch.device,
    n_runs: int = 3
) -> Dict[str, Any]:
    """Run a benchmark for a specific random walk function."""
    times = []
    data_points = []
    peak_memory = []
    unique_states_counts = []
    unique_pairs_counts = []
    max_target_values = []
    target_percentiles = []
    target_means = []
    target_stds = []
    X_samples = []
    y_samples = []
    
    # Warm-up run
    torch.cuda.empty_cache()
    _ = func(generators, n_steps, n_walks, device)
    
    for run in range(n_runs):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        start_time = time.time()
        X, y = func(generators, n_steps, n_walks, device)
        end_time = time.time()
        
        times.append(end_time - start_time)
        data_points.append(X.shape[0])
        peak_memory.append(torch.cuda.max_memory_allocated() / (1024 ** 3))  # GB
        
        # Calculate unique states and unique state-step pairs
        X_np = X.cpu().numpy()
        y_np = y.cpu().numpy()
        
        # Count unique states
        unique_states = set(tuple(state) for state in X_np)
        unique_states_counts.append(len(unique_states))
        
        # Count unique state-step pairs
        unique_pairs = set((tuple(state), step) for state, step in zip(X_np, y_np))
        unique_pairs_counts.append(len(unique_pairs))
        
        # Target value statistics
        max_target_values.append(np.max(y_np))
        target_means.append(np.mean(y_np))
        target_stds.append(np.std(y_np))
        
        # Calculate percentiles for target distribution
        percentiles = [10, 25, 50, 75, 90]
        target_percentiles.append(np.percentile(y_np, percentiles))
        
        # Save the last run for plotting
        if run == n_runs - 1:
            X_samples.append(X)
            y_samples.append(y)
    
    return {
        'avg_time': np.mean(times),
        'std_time': np.std(times),
        'avg_data_points': np.mean(data_points),
        'std_data_points': np.std(data_points),
        'avg_memory': np.mean(peak_memory),
        'std_memory': np.std(peak_memory),
        'avg_unique_states': np.mean(unique_states_counts),
        'std_unique_states': np.std(unique_states_counts),
        'avg_unique_pairs': np.mean(unique_pairs_counts),
        'std_unique_pairs': np.std(unique_pairs_counts),
        'avg_max_target': np.mean(max_target_values),
        'std_max_target': np.std(max_target_values),
        'avg_target_mean': np.mean(target_means),
        'std_target_mean': np.std(target_means),
        'avg_target_std': np.mean(target_stds),
        'std_target_std': np.std(target_stds),
        'avg_target_percentiles': np.mean(target_percentiles, axis=0),
        'X_sample': X_samples[0],
        'y_sample': y_samples[0]
    }

def plot_state_error_bars(
    X: torch.Tensor,
    y: torch.Tensor,
    ax: plt.Axes,
    sample_size: int = 50
) -> None:
    """Plot error bars showing the range of y values for sampled states."""
    X_np = X.cpu().numpy() if hasattr(X, "cpu") else np.asarray(X)
    y_np = y.cpu().numpy() if hasattr(y, "cpu") else np.asarray(y)

    grouped = defaultdict(list)
    for i, state in enumerate(X_np):
        key = tuple(state.tolist())
        grouped[key].append(y_np[i])

    unique_keys = list(grouped.keys())
    min_y = np.array([min(grouped[k]) for k in unique_keys])
    max_y = np.array([max(grouped[k]) for k in unique_keys])
    avg_y = (min_y + max_y) / 2.0

    order = np.argsort(avg_y)
    min_y = min_y[order]
    max_y = max_y[order]
    avg_y = avg_y[order]
    indices = np.arange(len(avg_y))

    if len(avg_y) > sample_size:
        sample_idx = np.linspace(0, len(avg_y)-1, sample_size, dtype=int)
        indices = indices[sample_idx]
        avg_y = avg_y[sample_idx]
        min_y = min_y[sample_idx]
        max_y = max_y[sample_idx]

    err_lower = avg_y - min_y
    err_upper = max_y - avg_y

    ax.errorbar(indices, avg_y, yerr=[err_lower, err_upper],
                fmt='o', ecolor='gray', capsize=5)
    ax.set_xlabel("Unique state index (sorted by average y)")
    ax.set_ylabel("Step (y)")
    ax.set_title("Range of y values for each unique state (sampled)")
    ax.grid(True)

def plot_target_distribution(
    y: torch.Tensor,
    ax1: plt.Axes,
    ax2: plt.Axes
) -> None:
    """Plot histogram of target values to visualize distribution."""
    y_np = y.cpu().numpy() if hasattr(y, "cpu") else np.asarray(y)
    
    # Histogram
    ax1.hist(y_np, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel("Step Value")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Distribution of Step Values")
    ax1.grid(True, alpha=0.3)
    
    # Cumulative distribution
    ax2.hist(y_np, bins=30, alpha=0.7, color='lightgreen', edgecolor='black', cumulative=True, density=True)
    ax2.set_xlabel("Step Value")
    ax2.set_ylabel("Cumulative Probability")
    ax2.set_title("Cumulative Distribution")
    ax2.grid(True, alpha=0.3)

def main():
    parser = argparse.ArgumentParser(description='Benchmark random walk implementations')
    parser.add_argument('--state-size', type=int, default=16, help='Size of state vector')
    parser.add_argument('--n-steps', type=int, default=120, help='Number of steps')
    parser.add_argument('--n-walks', type=int, default=10000, help='Number of walks')
    parser.add_argument('--n-runs', type=int, default=3, help='Number of benchmark runs')
    parser.add_argument('--no-plot', action='store_true', help='Disable plotting')
    parser.add_argument('--save-results', action='store_true', help='Save results to CSV')
    parser.add_argument('--sort-by', type=str, default='points_per_second', 
                        choices=['time', 'memory', 'data_points', 'unique_states', 
                                'unique_pairs', 'max_target', 'target_mean', 'points_per_second'],
                        help='Metric to sort results by')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate test data
    generators = create_lrx_moves(args.state_size)
    
    # Define functions to benchmark
    functions = {
        'first_visit_random_walks': first_visit_random_walks,
        'nbt_random_walks': nbt_random_walks,
        'random_walks_beam_nbt': random_walks_beam_nbt
    }
    
    # Run benchmarks
    results = {}
    for name, func in functions.items():
        print(f"Benchmarking {name}...")
        try:
            results[name] = run_benchmark(
                func, generators, args.n_steps, args.n_walks, device, args.n_runs
            )
            print(f"  - Average time: {results[name]['avg_time']:.4f}s")
            print(f"  - Data points: {results[name]['avg_data_points']:.0f}")
            print(f"  - Peak memory: {results[name]['avg_memory']:.2f} GB")
            print(f"  - Unique states: {results[name]['avg_unique_states']:.0f}")
            print(f"  - Unique state-step pairs: {results[name]['avg_unique_pairs']:.0f}")
            print(f"  - Max target value: {results[name]['avg_max_target']:.1f}")
            print(f"  - Target mean: {results[name]['avg_target_mean']:.1f}")
            print(f"  - Target std: {results[name]['avg_target_std']:.1f}")
            print(f"  - Target percentiles (10,25,50,75,90): {results[name]['avg_target_percentiles']}")
        except Exception as e:
            print(f"Error benchmarking {name}: {e}")
            results[name] = None
    
    # Create a sorted list of results for display
    valid_results = {name: result for name, result in results.items() if result is not None}
    
    # Calculate points per second for each implementation
    for name, result in valid_results.items():
        result['points_per_second'] = result['avg_data_points'] / result['avg_time']
    
    # Sort results based on the specified metric
    sort_key = args.sort_by
    reverse = True  # Higher values are better for most metrics
    if sort_key == 'time' or sort_key == 'memory':
        reverse = False  # Lower values are better for time and memory
    
    # Map sort_key to the actual result key
    sort_key_map = {
        'time': 'avg_time',
        'memory': 'avg_memory',
        'data_points': 'avg_data_points',
        'unique_states': 'avg_unique_states',
        'unique_pairs': 'avg_unique_pairs',
        'max_target': 'avg_max_target',
        'target_mean': 'avg_target_mean',
        'points_per_second': 'points_per_second'
    }
    
    actual_sort_key = sort_key_map.get(sort_key, 'points_per_second')
    
    # Sort the results
    sorted_results = sorted(
        valid_results.items(),
        key=lambda x: x[1][actual_sort_key],
        reverse=reverse
    )
    
    # Calculate speedup compared to the slowest implementation
    if len(sorted_results) > 1:
        slowest_time = max(result['avg_time'] for _, result in sorted_results)
        fastest_time = min(result['avg_time'] for _, result in sorted_results)
        overall_speedup = slowest_time / fastest_time
        
        for name, result in valid_results.items():
            result['speedup'] = slowest_time / result['avg_time']
    
    # Print results table
    print(f"\nBenchmark Results (sorted by {sort_key}):")
    print("{:<30} {:<20} {:<25} {:<20} {:<20} {:<20} {:<15} {:<15} {:<15} {:<15}".format(
        "Implementation", "Time (s)", "Data Points", "Memory (GB)", 
        "Unique States", "Unique Pairs", "Max Target", "Target Mean", "Points/sec", "Speedup"))
    print("-" * 185)
    
    for name, result in sorted_results:
        speedup = result.get('speedup', 1.0)
        print("{:<30} {:<20} {:<25} {:<20} {:<20} {:<20} {:<15} {:<15} {:<15} {:<15}".format(
            name,
            f"{result['avg_time']:.4f} ± {result['std_time']:.4f}",
            f"{result['avg_data_points']:.0f} ± {result['std_data_points']:.0f}",
            f"{result['avg_memory']:.2f} ± {result['std_memory']:.2f}",
            f"{result['avg_unique_states']:.0f} ± {result['std_unique_states']:.0f}",
            f"{result['avg_unique_pairs']:.0f} ± {result['std_unique_pairs']:.0f}",
            f"{result['avg_max_target']:.1f} ± {result['std_max_target']:.1f}",
            f"{result['avg_target_mean']:.1f} ± {result['std_target_mean']:.1f}",
            f"{result['points_per_second']:.0f}",
            f"{speedup:.2f}x"
        ))
    
    if len(sorted_results) > 1:
        print(f"\nOverall speedup (slowest vs. fastest): {overall_speedup:.2f}x")
    
    # Save results if requested
    if args.save_results:
        import pandas as pd
        table_data = []
        headers = ["Implementation", "Time (s)", "Data Points", "Memory (GB)", 
                  "Unique States", "Unique Pairs", "Max Target", "Target Mean", 
                  "Target Std", "P10", "P25", "P50", "P75", "P90", "Points/sec", "Speedup"]
        
        for name, result in sorted_results:
            speedup = result.get('speedup', 1.0)
            percentiles = result['avg_target_percentiles']
            table_data.append([
                name,
                f"{result['avg_time']:.4f} ± {result['std_time']:.4f}",
                f"{result['avg_data_points']:.0f} ± {result['std_data_points']:.0f}",
                f"{result['avg_memory']:.2f} ± {result['std_memory']:.2f}",
                f"{result['avg_unique_states']:.0f} ± {result['std_unique_states']:.0f}",
                f"{result['avg_unique_pairs']:.0f} ± {result['std_unique_pairs']:.0f}",
                f"{result['avg_max_target']:.1f} ± {result['std_max_target']:.1f}",
                f"{result['avg_target_mean']:.1f} ± {result['std_target_mean']:.1f}",
                f"{result['avg_target_std']:.1f} ± {result['std_target_std']:.1f}",
                f"{percentiles[0]:.1f}",
                f"{percentiles[1]:.1f}",
                f"{percentiles[2]:.1f}",
                f"{percentiles[3]:.1f}",
                f"{percentiles[4]:.1f}",
                f"{result['points_per_second']:.0f}",
                f"{speedup:.2f}x"
            ])
        
        df = pd.DataFrame(table_data, columns=headers)
        csv_filename = f'benchmark_random_walks_size{args.state_size}_steps{args.n_steps}_walks{args.n_walks}.csv'
        df.to_csv(csv_filename, index=False)
        print(f"Results saved to {csv_filename}")
    
    # Plot results if not disabled
    if not args.no_plot:
        # Create performance comparison plot
        plt.figure(figsize=(15, 10))
        
        # Get sorted names for consistent ordering in plots
        sorted_names = [name for name, _ in sorted_results]
        
        # Time comparison
        plt.subplot(2, 4, 1)
        times = [results[name]['avg_time'] for name in sorted_names]
        errors = [results[name]['std_time'] for name in sorted_names]
        
        plt.bar(sorted_names, times, yerr=errors)
        plt.ylabel('Time (s)')
        plt.title('Execution Time')
        plt.xticks(rotation=45, ha='right')
        
        # Data points comparison
        plt.subplot(2, 4, 2)
        data_points = [results[name]['avg_data_points'] for name in sorted_names]
        dp_errors = [results[name]['std_data_points'] for name in sorted_names]
        
        plt.bar(sorted_names, data_points, yerr=dp_errors)
        plt.ylabel('Number of Data Points')
        plt.title('Generated Data Points')
        plt.xticks(rotation=45, ha='right')
        
        # Memory usage
        plt.subplot(2, 4, 3)
        memory = [results[name]['avg_memory'] for name in sorted_names]
        mem_errors = [results[name]['std_memory'] for name in sorted_names]
        
        plt.bar(sorted_names, memory, yerr=mem_errors)
        plt.ylabel('Memory (GB)')
        plt.title('Peak Memory Usage')
        plt.xticks(rotation=45, ha='right')
        
        # Unique states
        plt.subplot(2, 4, 4)
        unique_states = [results[name]['avg_unique_states'] for name in sorted_names]
        us_errors = [results[name]['std_unique_states'] for name in sorted_names]
        
        plt.bar(sorted_names, unique_states, yerr=us_errors)
        plt.ylabel('Count')
        plt.title('Unique States Generated')
        plt.xticks(rotation=45, ha='right')
        
        # Unique pairs
        plt.subplot(2, 4, 5)
        unique_pairs = [results[name]['avg_unique_pairs'] for name in sorted_names]
        up_errors = [results[name]['std_unique_pairs'] for name in sorted_names]
        
        plt.bar(sorted_names, unique_pairs, yerr=up_errors)
        plt.ylabel('Count')
        plt.title('Unique State-Step Pairs')
        plt.xticks(rotation=45, ha='right')
        
        # Max target value
        plt.subplot(2, 4, 6)
        max_targets = [results[name]['avg_max_target'] for name in sorted_names]
        mt_errors = [results[name]['std_max_target'] for name in sorted_names]
        
        plt.bar(sorted_names, max_targets, yerr=mt_errors)
        plt.ylabel('Value')
        plt.title('Maximum Target Value')
        plt.xticks(rotation=45, ha='right')
        
        # Target mean
        plt.subplot(2, 4, 7)
        target_means = [results[name]['avg_target_mean'] for name in sorted_names]
        tm_errors = [results[name]['std_target_mean'] for name in sorted_names]
        
        plt.bar(sorted_names, target_means, yerr=tm_errors)
        plt.ylabel('Value')
        plt.title('Mean Target Value')
        plt.xticks(rotation=45, ha='right')
        
        # Points per second (efficiency)
        plt.subplot(2, 4, 8)
        efficiency = [results[name]['points_per_second'] for name in sorted_names]
        
        plt.bar(sorted_names, efficiency)
        plt.ylabel('Points per Second')
        plt.title('Processing Efficiency')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        performance_plot = f'benchmark_random_walks_performance_size{args.state_size}_steps{args.n_steps}_walks{args.n_walks}.png'
        plt.savefig(performance_plot)
        print(f"Performance comparison plot saved to {performance_plot}")
        
        # Create a single figure with all detailed visualizations
        if len(valid_results) > 0:
            # Create a large figure with subplots for each implementation
            fig = plt.figure(figsize=(20, 5 * len(sorted_results)))
            
            # For each implementation, create a row with 3 plots:
            # 1. State-step error bars
            # 2. Target distribution histogram
            # 3. Target cumulative distribution
            for i, name in enumerate(sorted_names):
                result = results[name]
                # Create a row of 3 subplots for this implementation
                ax1 = plt.subplot(len(sorted_results), 3, i*3 + 1)
                ax2 = plt.subplot(len(sorted_results), 3, i*3 + 2)
                ax3 = plt.subplot(len(sorted_results), 3, i*3 + 3)
                
                # Add a row title (implementation name)
                ax1.text(-0.1, 0.5, name.replace('_', ' '), 
                         transform=ax1.transAxes, 
                         rotation=90, 
                         fontsize=12, 
                         fontweight='bold',
                         verticalalignment='center',
                         horizontalalignment='right')
                
                # Plot state-step error bars
                plot_state_error_bars(result['X_sample'], result['y_sample'], ax1)
                
                # Plot target distributions
                plot_target_distribution(result['y_sample'], ax2, ax3)
            
            plt.tight_layout(rect=[0.02, 0, 1, 0.98])  # Add a bit of padding on the left
            plt.suptitle("Detailed Analysis of Random Walk Implementations", fontsize=16, y=0.995)
            detailed_plot = f'benchmark_random_walks_detailed_size{args.state_size}_steps{args.n_steps}_walks{args.n_walks}.png'
            plt.savefig(detailed_plot)
            print(f"Detailed visualizations saved to {detailed_plot}")

if __name__ == "__main__":
    main() 