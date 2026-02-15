"""
End-to-end: generate data, train model, solve permutation(s).
Configure everything in the header below. No argparse.

Usage: uv run experiments/run_pipeline.py
"""

from pathlib import Path
from time import time
import torch
import pandas as pd

from src.data_gen.random_walks import (
    create_lrx_moves,
    first_visit_random_walks,
    random_walks_beam_nbt,
)
from src.models.catboost_model import CatBoostModel
from src.models.xgboost_model import XGBoostModel
from src.models.mlp_model import MLPModel
from src.solvers.simple_solver import BeamSearchSolver

# =============================================================================
# CONFIGURATION — edit these
# =============================================================================

# Input: single permutation OR file
# Option A: solve one permutation (list of ints 0..n-1)
STATE = None #[1, 0, 2, 5, 4, 3]

# Option B: read from file (columns: n, permutation). Set STATE = None above.
FILE_PATH = Path(__file__).parent / "test_files" / "longest_perms.csv"
N_RANGE = [20]

# Pipeline parameters
N_WALKS = 10000
RW_TYPE = "beam_nbt"  # "beam_nbt" | "first_visit"
MODEL_NAME = "xgboost"  # "xgboost" | "catboost" | "mlp"
NBT_DEPTH = 2  # history_window_size for beam_nbt

# Solver parameters
BEAM_WIDTH = 2**15
MAX_STEPS_MULTIPLIER = 10
USE_X_RULE = False
TARGET_NEIGHBORHOOD_RADIUS = 15
HISTORY_WINDOW_SIZE = 2
VERBOSE = 0

# Batch sizes (GPU memory tuning)
HASHES_BATCH_SIZE = 500_000
FILTER_BATCH_SIZE = 1_000_000
PREDICT_BATCH_SIZE = 10_000_000

# =============================================================================

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required. This script runs only on GPU.")
DEVICE = torch.device("cuda")

MODEL_CLASSES = {
    "xgboost": XGBoostModel,
    "catboost": CatBoostModel,
    "mlp": MLPModel,
}

RW_FUNCTIONS = {
    "beam_nbt": random_walks_beam_nbt,
    "first_visit": first_visit_random_walks,
}


def load_targets():
    """Load permutations to solve from config."""
    if STATE is not None:
        perm = torch.tensor(STATE, device=DEVICE, dtype=torch.int8)
        return [(len(STATE), perm, str(STATE))]
    df = pd.read_csv(FILE_PATH)
    if N_RANGE is not None:
        if len(N_RANGE) == 1:
            n_min = n_max = int(N_RANGE[0])
        elif len(N_RANGE) == 2:
            n_min, n_max = (int(N_RANGE[0]), int(N_RANGE[1]))
        else:
            raise ValueError("N_RANGE must have length 1 or 2")
        if n_min > n_max:
            raise ValueError("N_RANGE must satisfy n_min <= n_max")
        df = df[(df["n"] >= n_min) & (df["n"] <= n_max)]
    targets = []
    for _, row in df.iterrows():
        n = int(row["n"])
        perm = torch.tensor(
            [int(x) for x in row["permutation"].split(",")],
            device=DEVICE,
            dtype=torch.int8,
        )
        targets.append((n, perm, row["permutation"]))
    return targets


def run_pipeline(n: int, perm: torch.Tensor, perm_str: str) -> dict:
    """Generate data, train model, solve. Returns result dict."""
    conj_steps = n * (n - 1) // 2
    max_steps = int(conj_steps * MAX_STEPS_MULTIPLIER)
    generators = create_lrx_moves(n)
    rw_fun = RW_FUNCTIONS[RW_TYPE]

    # Generate training data
    t0 = time()
    if RW_TYPE == "beam_nbt":
        X, y = rw_fun(
            generators,
            n_steps=conj_steps,
            n_walks=N_WALKS,
            history_window_size=NBT_DEPTH,
            device=DEVICE,
        )
    else:
        X, y = rw_fun(generators, conj_steps, N_WALKS, DEVICE)
    data_time = time() - t0
    torch.cuda.empty_cache()

    # Train model
    model = MODEL_CLASSES[MODEL_NAME]()
    t0 = time()
    model.train(X, y)
    train_time = time() - t0
    torch.cuda.empty_cache()

    # Solve
    solver = BeamSearchSolver(
        state_size=n,
        beam_width=BEAM_WIDTH,
        max_steps=max_steps,
        use_x_rule=USE_X_RULE,
        target_neighborhood_radius=TARGET_NEIGHBORHOOD_RADIUS,
        hashes_batch_size=HASHES_BATCH_SIZE,
        filter_batch_size=FILTER_BATCH_SIZE,
        predict_batch_size=PREDICT_BATCH_SIZE,
        history_window_size=HISTORY_WINDOW_SIZE,
        verbose=VERBOSE,
        device=DEVICE,
    )
    t0 = time()
    found, steps, solution = solver.solve(start_state=perm, model=model)
    solve_time = time() - t0

    return {
        "n": n,
        "permutation": perm_str,
        "success": found,
        "solution": solution if found else "Not found",
        "steps": steps,
        "data_time": data_time,
        "train_time": train_time,
        "solve_time": solve_time,
        "peak_memory_gb": solver.search_stats.get("peak_memory_gb", 0),
    }


def main():
    targets = load_targets()
    if not targets:
        print("No permutations to solve.")
        return

    print(f"Solving {len(targets)} permutation(s)")
    print(f"Config: rw={RW_TYPE}, model={MODEL_NAME}, beam={BEAM_WIDTH}")
    print("-" * 60)

    results = []
    for i, (n, perm, perm_str) in enumerate(targets):
        perm_display = perm_str if len(perm_str) <= 50 else perm_str[:47] + "..."
        print(f"\n[{i+1}/{len(targets)}] n={n} perm={perm_display}")
        r = run_pipeline(n, perm, perm_str)
        results.append(r)
        print(
            f"  {'OK' if r['success'] else 'FAIL'} | steps={r['steps']} | "
            f"solve={r['solve_time']:.2f}s | mem={r['peak_memory_gb']:.2f}GB"
        )
        if r["success"]:
            print(f"  Solution: {r['solution']}")

    # Summary
    solved = sum(1 for r in results if r["success"])
    print(f"\n--- Done: {solved}/{len(results)} solved ---")


if __name__ == "__main__":
    main()
