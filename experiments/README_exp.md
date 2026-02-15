# Experiment Scripts

This folder contains scripts for benchmarking and optimizing random walk implementations.

## benchmark_random_walks.py

Compares three random walk implementations (first_visit, nbt, beam_nbt) that generate (state, step) training pairs for permutation sorting. For each implementation, measures wall-clock time, GPU memory, number of pairs, unique states, unique state-step pairs, and target value distribution. Produces comparison plots and optionally CSV.

Parameters:
- `--state-size`: Size of permutation vector (default: 16)
- `--n-steps`: Number of steps per walk (default: 120)
- `--n-walks`: Number of random walks (default: 10000)
- `--n-runs`: Number of benchmark runs (default: 3)
- `--no-plot`: Disable plotting (default: plots enabled)
- `--save-results`: Save results to CSV (default: off)
- `--sort-by`: Metric to sort results by (default: points_per_second; choices: time, memory, data_points, unique_states, unique_pairs, max_target, target_mean, points_per_second)

Output files (saved to `BS_results/benchmark_random_walks/`):
- `benchmark_random_walks_performance_size{state_size}_steps{n_steps}_walks{n_walks}.png`: Performance comparison plots
- `benchmark_random_walks_detailed_size{state_size}_steps{n_steps}_walks{n_walks}.png`: Detailed analysis plots
- `benchmark_random_walks_size{state_size}_steps{n_steps}_walks{n_walks}.csv`: Detailed results (if --save-results is used)

## optimize_random_walks.py

Finds optimal parameters (n_steps - walk length and n_walks - number of parallel walks)
for a chosen random-walk implementation and model. It uses true BFS distances so can be used
on small state_size only. The best config may not transfer to larger state_size.
The downstream goal is candidate ranking in the solver, so configs are selected by max
test Spearman correlation to BFS distances (RMSE/R2 are reported as auxiliary stats).

Note: Full BFS over permutations visits up to n! states. For the current implementation,
peak memory is approximately 12 * n! bytes. This yields (on RTX3090):
- n=11 ≈ 0.446 GiB (39,916,800 states, computed in 8 seconds),
- n=12 ≈ 5.35 GiB (479,001,600 states, computed in 126 seconds),
- n=13 ≈ 69.6 GiB (6,227,020,800 states, computed in 34 min).
- n=14 ≈ 974 GiB.

Thus, with 128 GiB RAM, full BFS can fit up to n=13.
VRAM needed is approximately 8 * n! * (n + 1) bytes. For n=11 this is ≈ 3.57 GiB (fits in 24 GiB),
but for n=12 it is ≈ 46.4 GiB (does not fit). Therefore, from n >= 12 we must subsample
a fixed number of test states (BFS_EVAL_STATES = 2_000_000 by default).

Parameters:
- `--state-size`: Size of permutation vector (default: 8)
- `--implementation`: Specific implementation to optimize (default: random_walks_beam_nbt)
- `--test-all`: Test all implementations (off by default)
- `--model-type`: Type of model to use (xgboost, mlp, catboost. default: xgboost)
- `--steps`: Comma-separated step multipliers (default: 0.5,0.75,1,1.5,2,3,5)
- `--walks`: Comma-separated walk counts (default: 100,500,1000,5000,10000,15000,20000)

Output files (saved to `BS_results/optimize_random_walks/`):
- `optimize_random_walks_size{state_size}_{model_type}.png`: Parameter analysis plots
- `optimize_best_configurations_size{state_size}_{model_type}.csv`: Best parameters for each implementation
- `optimize_{implementation}_size{state_size}_{model_type}.csv`: Detailed results for each implementation

## profile_model.py

This script benchmarks model inference speed vs batch size. It generates a
dataset via one of the random-walk generators, trains a chosen model, then
measures prediction latency, throughput (samples/sec), and peak CUDA memory
for multiple batch sizes.

Parameters:
- `--state-size`: Size of permutation vector (default: 16)
- `--model-type`: Type of model to profile (xgboost, catboost, mlp)
- `--rw-type`: Random walk type for data generation (first_visit, nbt, beam_nbt)
- `--n-walks`: Number of random walks (default: 10000)
- `--batch-sizes`: Comma-separated list of batch sizes (default: "1000,10000,50000,100000")
- `--no-plot`: Disable plotting

Output files (saved to `BS_results/profile_model/`):
- `profile_model_size{state_size}_{model_type}_{rw_type}.png`: Performance plots
- `profile_model_size{state_size}_{model_type}_{rw_type}.csv`: Detailed profiling results

## profile_solver.py

Profiles solver performance with different batch sizes, measuring execution time,
memory usage, and exploration statistics for BeamSearchSolver.

Parameters:
- `--state-size`: Size of permutation vector (default: 16)
- `--beam-width`: Beam width for search (default: 2^state_size)
- `--batch-sizes`: Comma-separated list of batch sizes (default: "10000,50000,100000,500000")
- `--history-window`: History window size (default: conjectured_steps/5)
- `--use-x-rule`: Enable X-rule optimization
- `--target-radius`: Target neighborhood radius (default: 0)
- `--no-plot`: Disable plotting
- `--verbose`: Enable verbose solver output

Output files (saved to `BS_results/profile_solver/`):
- `profile_solver_size{state_size}_beam{beam_width}.png`: Performance analysis plots
- `profile_solver_size{state_size}_beam{beam_width}.csv`: Detailed profiling results

## run_rw_nbt_depth_experiment_.py

Tests the effect of random walks NBT (non-backtracking) depth on solver performance
by varying the history_window_size parameter for data generation.

Parameters (configured in script):
- `n_runs`: Number of runs per configuration
- `state_size`: Size of permutation
- `rw_type`: Random walks implementation
- `n_walks`: Number of random walks for training data
- `rw_nbt_depths`: List of NBT depth values to test
- `model_name`: Model type (xgboost, catboost)
- `history_window_size`: BeamSearchSolver NBT depth
- `beam_width`: BeamSearchSolver beam width

Output files (saved to `BS_results/run_rw_nbt_depth_experiment_/`):
- `rw_nbt_depth_experiments_{base_name}.csv`: Detailed results for each run
- `rw_nbt_depth_stats_{base_name}.csv`: Summary statistics grouped by NBT depth

## validate_solutions.py

Validates solution files from permutation sorting experiments by verifying
that move sequences actually solve the problems correctly.

Functions:
- Applies move sequences (X, L, R) to verify solutions reach sorted state
- Detects suboptimal X moves (swapping when elements are already ordered)
- Reports validation statistics and error details

Parameters:
- `solutions_file`: Path to CSV file containing solutions to validate (command-line argument)

Output:
- Prints detailed validation summary with statistics
- Reports invalid solution details with specific error reasons
- File remains unchanged (read-only validation) 