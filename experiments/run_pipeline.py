"""
End-to-end: generate data, train model, solve permutation(s).
Configure everything in the header below. No argparse.

Usage: uv run experiments/run_pipeline.py
"""

from pathlib import Path
from time import time
import warnings
import torch
import pandas as pd

from src.data_gen.random_walks import (
    first_visit_random_walks,
    random_walks_beam_nbt,
)
from src.models.catboost_model import CatBoostModel
from src.models.xgboost_model import XGBoostModel
from src.models.mlp_model import MLPModel
from src.puzzles import make_spec
from src.solvers.simple_solver import BeamSearchSolver
from src.utils import estimate_gpu_limits

# =============================================================================
# CONFIGURATION — edit these
# =============================================================================

PUZZLE = "pancake" # "lrx"

# Input: single permutation OR file
# Option A: solve one permutation (list of ints 0..n-1)
STATE = [8,5,32,34,21,22,14,9,20,7,25,4,37,35,28,13,29,8,26,19,0,12,3,10,36,16,11,15,27,39,31,2,24,33,1,30,6,23,17,38]

# Option B: read from file (columns: n, permutation). Set STATE = None above.
FILE_PATH = Path(__file__).parent / "test_files" / "pancake_test.csv" # "longest_perms.csv"
N_RANGE = [5]

# Pipeline parameters
N_RUNS = 1  # Full re-runs per target (new data each time)
N_WALKS = 10000
RW_TYPE = "beam_nbt"  # "beam_nbt" | "first_visit"
MODEL_NAME = "xgboost"  # "xgboost" | "catboost" | "mlp"
RW_NBT_DEPTH = 2  # NBT depth for random walks (beam_nbt)

# Solver parameters
BEAM_WIDTH = 2**15
MAX_STEPS_MULTIPLIER = 10
USE_X_RULE = False
TARGET_NEIGHBORHOOD_RADIUS = 10
BS_NBT_DEPTH = 2  # NBT depth for beam search
VERBOSE = 1

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


def get_conj_length(n: int, puzzle: str) -> int:
    """Conjugation length: n - (n-1)/2 (integer)."""
    if puzzle == "pancake":
        return int(2 * n - 3)
    elif puzzle == "lrx":
        return int(n * (n - 1) / 2)
    else:
        raise ValueError(f"Unknown puzzle: {puzzle}")


def run_pipeline(n: int, perm: torch.Tensor, perm_str: str, conj_length: int) -> dict:
    """Generate data, train model, solve. Returns result dict."""
    max_steps = int(conj_length * MAX_STEPS_MULTIPLIER)
    puzzle_spec = make_spec(PUZZLE, n, DEVICE)
    generators = puzzle_spec.move_indices
    rw_fun = RW_FUNCTIONS[RW_TYPE]

    # Generate training data
    t0 = time()
    if RW_TYPE == "beam_nbt":
        X, y = rw_fun(
            generators,
            n_steps=conj_length,
            n_walks=N_WALKS,
            nbt_depth=RW_NBT_DEPTH,
            device=DEVICE,
        )
    else:
        X, y = rw_fun(generators, conj_length, N_WALKS, DEVICE)
    data_time = time() - t0
    torch.cuda.empty_cache()

    # Train model
    model = MODEL_CLASSES[MODEL_NAME]()
    t0 = time()
    model.train(X, y)
    train_time = time() - t0
    torch.cuda.empty_cache()

    # Cap beam and batch sizes to avoid OOM
    limits = estimate_gpu_limits(n, DEVICE, n_moves=puzzle_spec.move_indices.size(0))
    beam_width = min(BEAM_WIDTH, limits["max_beam_width"])
    hashes_batch = min(HASHES_BATCH_SIZE, limits["hashes_batch_size"])
    filter_batch = min(FILTER_BATCH_SIZE, limits["filter_batch_size"])
    predict_batch = min(PREDICT_BATCH_SIZE, limits["predict_batch_size"])
    if beam_width < BEAM_WIDTH:
        warnings.warn(
            f"BEAM_WIDTH {BEAM_WIDTH} exceeds GPU limit; using {beam_width} "
            f"(max_beam_width from estimate_gpu_limits)."
        )
    if hashes_batch < HASHES_BATCH_SIZE or filter_batch < FILTER_BATCH_SIZE:
        warnings.warn(
            f"Batch sizes capped by GPU memory: hashes={hashes_batch}, "
            f"filter={filter_batch}, predict={predict_batch}"
        )

    # Solve
    solver = BeamSearchSolver(
        state_size=n,
        puzzle_spec=puzzle_spec,
        beam_width=beam_width,
        max_steps=max_steps,
        use_x_rule=USE_X_RULE,
        target_neighborhood_radius=TARGET_NEIGHBORHOOD_RADIUS,
        hashes_batch_size=hashes_batch,
        filter_batch_size=filter_batch,
        predict_batch_size=predict_batch,
        nbt_depth=BS_NBT_DEPTH,
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
        "beam_width": beam_width,
        "data_time": data_time,
        "train_time": train_time,
        "solve_time": solve_time,
        "peak_memory_gb": solver.search_stats.get("peak_memory_gb", 0),
    }


def _rw_type_str() -> str:
    if RW_TYPE == "beam_nbt":
        return f"beam_nbt depth={RW_NBT_DEPTH} walks={N_WALKS}"
    return f"first_visit walks={N_WALKS}"


def _bs_type_str() -> str:
    return (
        f"use_x={USE_X_RULE}, radius={TARGET_NEIGHBORHOOD_RADIUS}, "
        f"nbt_depth={BS_NBT_DEPTH}, mult={MAX_STEPS_MULTIPLIER}"
    )


def main():
    targets = load_targets()
    if not targets:
        print("No permutations to solve.")
        return

    rw_str = _rw_type_str()
    bs_str = _bs_type_str()

    table_rows = []
    for i, (n, perm, perm_str) in enumerate(targets):
        run_results = []
        first_r = None

        conj_length = get_conj_length(n, PUZZLE)

        for run_idx in range(N_RUNS):
            r = run_pipeline(n, perm, perm_str, conj_length)
            if first_r is None:
                first_r = r
            total_time = r["data_time"] + r["train_time"] + r["solve_time"]
            run_results.append({
                "success": r["success"],
                "time": total_time,
                "steps": r["steps"] if r["success"] else None,
            })
        solved = [x for x in run_results if x["success"]]
        n_success = len(solved)
        avg_time = (
            sum(x["time"] for x in solved) / n_success if n_success else float("nan")
        )
        min_steps = (
            min(x["steps"] for x in solved) if n_success else float("nan")
        )
        table_rows.append({
            "state_size": n,
            "conj_length": conj_length,
            "success_rate": n_success / N_RUNS,
            "runs": N_RUNS,
            "rw_type": rw_str,
            "bs_type": bs_str,
            "beam_width": first_r["beam_width"],
            "model": MODEL_NAME,
            "time_avg": avg_time,
            "min_steps": min_steps,
        })

    df = pd.DataFrame(table_rows)
    df["success_rate"] = df["success_rate"].apply(lambda x: f"{x:.1%}")
    df["time_avg"] = df["time_avg"].apply(lambda x: "NA" if pd.isna(x) else f"{x:.2f}s")
    df["min_steps"] = df["min_steps"].apply(lambda x: "NA" if pd.isna(x) else str(int(x)))
    cols = [
        "state_size", "conj_length", "min_steps", "success_rate", "runs",
        "model", "rw_type", "bs_type", "beam_width", "time_avg",
    ]
    col_headers = [
        "state size", "conj length", "min steps", "success rate", "runs",
        "model", "RW type", "BS type", "beam width", "time (avg)",
    ]
    out = df[cols].rename(columns=dict(zip(cols, col_headers)))
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
