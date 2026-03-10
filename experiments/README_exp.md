# Experiment Scripts

This folder contains scripts for benchmarking and optimizing random walks, model, and solver implementations.

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

Profiles the modern `BeamSolver` implementation and its runtime profiler.
Runs solver instances on GPU and reports both coarse statistics and
stage-level timings.

There is no CLI; configuration is done by editing two objects in the script:

- `CFG: ProfileConfig`
  - `puzzle_type`: puzzle family (`"pancake"` by default)
  - `state_sizes`: list of state sizes to test
  - `backward_modes`: list of backward modes
    (`"off"`, `"bfs"`, `"beam"`)
  - `beam_width`, `max_steps_extra`, `backward_max_states`,
    `bs_nbt_depth`, `hashes_batch_size`, `random_seed`
- `RUNS: List[Dict[str, Any]]`
  - Optional explicit run list; if non-empty, overrides the
    Cartesian grid `state_sizes × backward_modes`
  - Each dict has keys:
    - required: `puzzle_type`, `state_size`, `backward_mode`
    - optional: `beam_width`, `max_steps`

For each run the script:

- builds a `PuzzleSpec` and adapter,
- constructs `BeamSolver` via the solver factory with `profile_runtime=True`,
- samples a random start permutation on GPU,
- runs `solve()` once and records:
  - wall-clock time,
  - CUDA peak memory,
  - states explored and backward-archive size,
  - solution length and success flag.

It also reads per-stage profiling data from `solver.search_stats["profile"]`
and prints two tables:

- **Stage timing (seconds)** for stages:
  `expand`, `unique`, `history_filter`, `lower_bound`,
  `beam_prune`, `meet_lookup`, `bfs_prebuild`
- **Stage counts**: number of calls and total input states for each stage

Output files (saved to `BS_results/profile_solver/`):

- `profile_results.json`: JSON list with one record per run,
  including coarse stats and (optionally) embedded `profile` dict

## tune_rw_nbt_depth.py

Tests the effect of random walks NBT (non-backtracking) depth on solver performance
by varying the RW nbt_depth parameter for data generation.

Parameters (configured in script):
- `n_runs`: Number of runs per configuration
- `state_size`: Size of permutation
- `rw_type`: Random walks implementation
- `n_walks`: Number of random walks for training data
- `rw_nbt_depths`: List of RW NBT depth values to test
- `model_name`: Model type (xgboost, catboost)
- `bs_nbt_depth`: BeamSearchSolver NBT depth (fixed)
- `beam_width`: BeamSearchSolver beam width

Output files (saved to `BS_results/tune_rw_nbt_depth/`):
- `rw_nbt_depth_experiments_{base_name}.csv`: Detailed results for each run
- `rw_nbt_depth_stats_{base_name}.csv`: Summary statistics grouped by NBT depth