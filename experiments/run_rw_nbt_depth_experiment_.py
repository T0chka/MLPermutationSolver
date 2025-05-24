"""
Experiment to test the effect of NBT (non-backtracking) depth on solver performance.

Tests different history_window_size values for NBT random walks to determine how the depth
of non-backtracking affects:
- BeamSearchSolver success rates
- Solution efficiency (steps, time, memory usage)
"""

import torch
import pandas as pd
from time import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.random_walks import (
    create_lrx_moves,
    first_visit_random_walks,
    random_walks_beam_nbt
)
from src.models.catboost_model import CatBoostModel
from src.models.xgboost_model import XGBoostModel
from src.models.mlp_model import MLPModel
from src.solvers.simple_solver import BeamSearchSolver

# make directory for results
os.makedirs('experiments/BS_results', exist_ok=True)

def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

EXPERIMENT_PARAMS = {
    'state_size': 15,
    'n_runs': 5,
    'rw_type': "beam_nbt",
    'n_walks': 10000,
    'model_name': "xgboost",
    'history_window_size': 2,
    'max_steps_multiplier': 10,
    'use_x_rule': False,
    'target_neighborhood_radius': 15,
    'verbose': 0,
    'test_path': 'experiments/test_files/longest_perms.csv'
}

# Beam widths to test
if EXPERIMENT_PARAMS['state_size'] == 15:
    beam_width = 2**2
elif EXPERIMENT_PARAMS['state_size'] == 25:
    beam_width = 2**12
elif EXPERIMENT_PARAMS['state_size'] == 28:
    beam_width = 2**15
else:
    raise ValueError(f"State size {EXPERIMENT_PARAMS['state_size']} not supported")

# Calculate conjunction steps for the state size
conj_steps = int(
    EXPERIMENT_PARAMS['state_size'] * (EXPERIMENT_PARAMS['state_size'] - 1) / 2
)

if EXPERIMENT_PARAMS['history_window_size'] != 2:
    EXPERIMENT_PARAMS['history_window_size'] = int(
        round(conj_steps * EXPERIMENT_PARAMS['history_window_size'])
    )

# NBT depths to test
rw_nbt_depths = [2] + [conj_steps * f for f in [0.10, 0.30, 0.50]]
rw_nbt_depths = sorted(set(int(round(v)) for v in (rw_nbt_depths)))

print("Beam width:", beam_width)
print("History window size:", EXPERIMENT_PARAMS['history_window_size'])
print("NBT depths to test:", rw_nbt_depths)

# Make output files names based on experiment parameters
base_name = (
    f'{EXPERIMENT_PARAMS["state_size"]}_'
    f'{EXPERIMENT_PARAMS["rw_type"]}_'
    f'{EXPERIMENT_PARAMS["model_name"]}'
)

results_file = f'experiments/BS_results/rw_nbt_depth_experiments_{base_name}.csv'
stats_file = f'experiments/BS_results/rw_nbt_depth_stats_{base_name}.csv'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

model_classes = {
    "xgboost": XGBoostModel,
    "catboost": CatBoostModel
}

# Load existing results if available
existing_results = []
if os.path.isfile(results_file):
    print(f"Loading existing results from {results_file}")
    existing_df = pd.read_csv(results_file)
    existing_results = existing_df.to_dict('records')
    print(f"Loaded {len(existing_results)} existing results")

# Load test data
test_df = pd.read_csv(EXPERIMENT_PARAMS['test_path'])
test_df = test_df[(test_df["n"] == EXPERIMENT_PARAMS['state_size'])]
test_df = test_df.reset_index(drop=True)

# Run experiment
script_start = time()
results = []

for _, row in test_df.iterrows():
    size = row["n"]
    print(f"\nProcessing state size: {size}")
    print(f"Number of states (n!): {factorial(size)}")

    MAX_STEPS = int(conj_steps * EXPERIMENT_PARAMS['max_steps_multiplier'])
    print(f"conj_steps: {conj_steps}, MAX_STEPS: {MAX_STEPS}")

    perm = torch.tensor(
        [int(x) for x in row['permutation'].split(',')],
        device=DEVICE
    )

    if EXPERIMENT_PARAMS['rw_type'] == "beam_nbt":
        rw_fun = random_walks_beam_nbt
    elif EXPERIMENT_PARAMS['rw_type'] == "first_visit":
        rw_fun = first_visit_random_walks
    
    # Test all combinations of nbt depth
    for nbt_depth in rw_nbt_depths:
        print(f"\n=== Testing nbt depth: {nbt_depth} ===")

        # Check for existing results with the same parameters
        existing_for_this_config = [
            r for r in existing_results 
            if (r['n'] == size and 
                r['random_walks'] == EXPERIMENT_PARAMS['rw_type'] and
                r['n_walks'] == EXPERIMENT_PARAMS['n_walks'] and
                r['nbt_depth'] == nbt_depth and
                r['model'] == EXPERIMENT_PARAMS['model_name'] and
                r['beam_width'] == beam_width and
                r['target_neighborhood_radius'] == EXPERIMENT_PARAMS['target_neighborhood_radius'] and
                r['use_x_rule'] == EXPERIMENT_PARAMS['use_x_rule'] and
                r['history_window_size'] == EXPERIMENT_PARAMS['history_window_size'] and
                r['permutation'] == row['permutation'] and
                r['max_steps'] == MAX_STEPS)
        ]
        
        # Get the runs that have already been completed
        completed_runs = set(r['run'] for r in existing_for_this_config)
        
        # If all runs are already completed, skip this configuration
        if len(completed_runs) >= EXPERIMENT_PARAMS['n_runs']:
            print(
                f"Skipping nbt_depth={nbt_depth} - "
                f"already completed {len(completed_runs)} runs"
            )
            # Add existing results to our results collection
            results.extend(existing_for_this_config)
            continue
        
        # Track success rate
        successful_runs = sum(1 for r in existing_for_this_config if r.get('success', False))
        total_runs = len(existing_for_this_config)
        
        # Display current success rate if we have existing results
        if total_runs > 0:
            current_success_rate = (successful_runs / total_runs) * 100
            print(f"Current success rate from existing results: {current_success_rate:.2f}% ({successful_runs}/{total_runs})")
        
        for run in range(1, EXPERIMENT_PARAMS['n_runs'] + 1):
            # Skip if this run has already been completed
            if run in completed_runs:
                print(f"Skipping run {run} - already completed")
                continue
            
            torch.cuda.empty_cache()
            print(f"\nRun {run} for n={size}, nbt_depth={nbt_depth}")
            
            # Generate data and train model
            generators = create_lrx_moves(size)
            model = model_classes[EXPERIMENT_PARAMS['model_name']]()
            
            start_time = time()
            X, y = rw_fun(
                generators, 
                n_steps=conj_steps, 
                n_walks=EXPERIMENT_PARAMS['n_walks'], 
                history_window_size=nbt_depth, 
                device=DEVICE
            )
            data_gen_time = time() - start_time
            torch.cuda.empty_cache()
            
            print(
                f"Generated train data with shape {X.shape} "
                f"({conj_steps} steps, {EXPERIMENT_PARAMS['n_walks']} walks) "
                f"in {data_gen_time:.2f}s"
            )
            
            start_time = time()
            model.train(X, y)
            train_time = time() - start_time
            torch.cuda.empty_cache()
            
            print(f"Training completed in {train_time:.2f}s")
            
            # Initialize solver with current parameters
            solver = BeamSearchSolver(
                state_size=size, 
                beam_width=beam_width,
                max_steps=MAX_STEPS,
                use_x_rule=EXPERIMENT_PARAMS['use_x_rule'],
                target_neighborhood_radius=EXPERIMENT_PARAMS['target_neighborhood_radius'],
                hashes_batch_size=500_000,
                filter_batch_size=1_000_000,
                predict_batch_size=10_000_000,
                history_window_size=EXPERIMENT_PARAMS['history_window_size'],
                verbose=EXPERIMENT_PARAMS['verbose'],
                device=DEVICE
            )
            
            # Solve the problem
            solve_start = time()
            found, steps, solution = solver.solve(start_state=perm, model=model)
            solve_time = time() - solve_start
            
            # Collect statistics
            peak_memory = solver.search_stats.get('peak_memory_gb', 0)
            pruned_states_total = solver.search_stats.get('pruned_states_total', 0)
            pruned_states_min = solver.search_stats.get('pruned_states_min', 0)
            pruned_states_max = solver.search_stats.get('pruned_states_max', 0)
            pruned_states_avg = solver.search_stats.get('pruned_states_avg', 0)
            total_visited_states = solver.search_stats.get('Total hashes ever seen', 0)
            total_hashes_in_history = solver.search_stats.get('Total hashes in history', 0)
            first_pruning_step = solver.search_stats.get('first_pruning_step', 0)
            last_step_pruned = solver.search_stats.get('last_step_pruned', 0)
            
            # Create result entry
            result = {
                'random_walks': EXPERIMENT_PARAMS['rw_type'],
                'n_walks': EXPERIMENT_PARAMS['n_walks'],
                'nbt_depth': nbt_depth,
                'model': EXPERIMENT_PARAMS['model_name'],
                'permutation': row['permutation'],
                'n': size,
                'beam_width': beam_width,
                'target_neighborhood_radius': EXPERIMENT_PARAMS['target_neighborhood_radius'],
                'use_x_rule': EXPERIMENT_PARAMS['use_x_rule'],
                'history_window_size': EXPERIMENT_PARAMS['history_window_size'],
                'max_steps': MAX_STEPS,
                'run': run,
                'success': found,
                'solution': solution if found else "Not found",
                'steps': steps,
                'train_time': train_time,
                'data_generation_time': data_gen_time,
                'solve_time': solve_time,
                'peak_memory': peak_memory,
                'total_visited_states': total_visited_states,
                'total_hashes_in_history': total_hashes_in_history,
                'pruned_states_total': pruned_states_total,
                'pruned_states_min': pruned_states_min,
                'pruned_states_max': pruned_states_max,
                'pruned_states_avg': pruned_states_avg,
                'first_pruning_step': first_pruning_step,
                'last_step_pruned': last_step_pruned
            }
            
            # Add to overall results
            results.append(result)
            
            # Update success tracking
            total_runs += 1
            if found:
                successful_runs += 1
            
            # Print results and success rate
            current_success_rate = (successful_runs / total_runs) * 100
            print(
                f"Success: {found}, Steps: {steps}, Time: {solve_time:.2f}s, "
                f"Peak Memory: {peak_memory:.2f}GB"
            )
            print(
                f"Current success rate: {current_success_rate:.2f}% "
                f"({successful_runs}/{total_runs})"
            )
            
            # Save current result after each run
            results_to_save = pd.DataFrame([result])
            
            if os.path.isfile(results_file):
                saved_df = pd.read_csv(results_file)
                combined_df = pd.concat([saved_df, results_to_save])
                combined_df.to_csv(results_file, index=False)
            else:
                results_to_save.to_csv(results_file, index=False)
            
            print(f"Results saved after run {run}")
        
        # After completing all runs for this configuration
        print(f"\nCompleted all runs for nbt_depth={nbt_depth}")
        print(
            f"Final success rate: {current_success_rate:.2f}% "
            f"({successful_runs}/{total_runs})"
        )
        
        # If 100% success rate achieved, move to next history window size
        if current_success_rate == 100:
            print(
                f"Achieved 100% success rate with nbt_depth={nbt_depth}"
            )
            break

# Create final results dataframe
results_df = pd.DataFrame(results)

# Print experiment summary
total_time = time() - script_start
print(f"\nTotal execution time: {total_time:.2f} seconds")

# Calculate statistics grouped by both beam_width and history_window_size
grouped = results_df.groupby(['nbt_depth'])
success_rate = grouped['success'].mean() * 100
num_runs = grouped['run'].nunique()

solved = results_df[results_df['success']]
if not solved.empty:
    stats = solved.groupby(['nbt_depth']).agg(
        successful_runs=('steps', 'size'),
        median_steps=('steps', 'median'),
        min_steps=('steps', 'min'),
        max_steps=('steps', 'max'),
        std_steps=('steps', 'std'),
        mean_solve_time=('solve_time', 'mean'),
        mean_peak_memory=('peak_memory', 'mean')
    )
    
    # Add success rate to stats
    stats = stats.join(success_rate.rename('success_rate (%)').round(2))
    stats = stats.join(num_runs.rename('num_runs'))
    
    print("\nStatistics per nbt_depth:")
    print(stats)
    
    # Save statistics
    stats.to_csv(stats_file, index=True)
else:
    print("No successful solutions found.")

