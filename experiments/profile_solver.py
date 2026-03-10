"""
Profile BeamSolver: coarse benchmark and stage-level timing.

Level 1 — coarse: total time, memory, states explored, solution length, steps.
Level 2 — stage timing: expand, unique, history_filter, lower_bound, beam_prune,
          meet_lookup, bfs_prebuild (from search_stats["profile"]).

Usage: uv run experiments/profile_solver.py

Config: edit CFG and RUNS below. RUNS overrides the grid (state_sizes x backward_modes).
"""

from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Any, Dict, List, Optional

import torch

from src.puzzles import make_spec
from src.solvers.factory import make_solver
from src.solvers.puzzle_adapters import get_adapter

################################################################################
# CONFIG — edit parameters here
################################################################################


@dataclass
class ProfileConfig:
    """Default configuration for profiling runs."""

    puzzle_type: str = "lrx" # "lrx", "pancake"
    state_sizes: List[int] = field(default_factory=lambda: [15, 20, 50])
    backward_modes: List[str] = field(default_factory=lambda: ["off", "bfs", "beam"])
    beam_width: int = 2**12
    max_steps_extra: int = 0
    backward_max_states: int = 2_000_000
    bs_nbt_depth: int = 2
    hashes_batch_size: int = 50_000_000
    random_seed: int = 42


CFG = ProfileConfig()

# Explicit run configs; if non-empty, overrides grid of state_sizes x backward_modes.
# Each dict: puzzle_type, state_size, backward_mode; optional: beam_width, max_steps.
RUNS: List[Dict[str, Any]] = []

RESULTS_DIR = Path(__file__).resolve().parent / "BS_results" / "profile_solver"


class DummyModel:
    """Dummy model returning zeros."""

    def __init__(self, device: torch.device) -> None:
        self.device = device

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return torch.zeros(len(X), device=self.device, dtype=torch.float32)

    def train(self, X: torch.Tensor, y: torch.Tensor) -> None:
        pass


def run_one(
    puzzle_type: str,
    state_size: int,
    backward_mode: str,
    device: torch.device,
    *,
    beam_width: Optional[int] = None,
    max_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """Run solver once; return coarse stats and profile."""
    spec = make_spec(puzzle_type, state_size, device)
    if max_steps is None:
        max_steps = spec.conj_length if puzzle_type == "lrx" else state_size + CFG.max_steps_extra
    bw = beam_width if beam_width is not None else CFG.beam_width
    adapter = get_adapter(puzzle_type, spec, device)
    solver = make_solver(
        spec,
        "beam",
        adapter,
        beam_width=bw,
        max_steps=max_steps,
        backward_mode=backward_mode,
        backward_max_states=CFG.backward_max_states if backward_mode == "bfs" else 0,
        bs_nbt_depth=CFG.bs_nbt_depth,
        hashes_batch_size=CFG.hashes_batch_size,
        profile_runtime=True,
    )
    model = DummyModel(device)
    gen = torch.Generator(device=device).manual_seed(CFG.random_seed + state_size)
    start = torch.randperm(state_size, device=device, generator=gen).to(spec.state_dtype)

    import gc
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)

    t0 = time()
    found, steps, solution = solver.solve(start, model)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time() - t0

    sol_len = len(solution.split(".")) if found and solution else 0
    peak_gb = solver._cuda_allocated_gb() if device.type == "cuda" else 0.0

    out: Dict[str, Any] = {
        "puzzle_type": puzzle_type,
        "state_size": state_size,
        "backward_mode": backward_mode,
        "found": found,
        "steps": steps,
        "solution_len": sol_len,
        "elapsed_s": elapsed,
        "peak_memory_gb": peak_gb,
        "states_explored": solver.search_stats.get("total_states_explored", 0),
        "backward_archive_states": solver.search_stats.get("backward_archive_states", 0),
    }
    profile = solver.search_stats.get("profile")
    if profile:
        out["profile"] = profile
    return out


def print_coarse_table(results: List[Dict[str, Any]]) -> None:
    """Print coarse benchmark table."""
    print("\n=== COARSE BENCHMARK ===")
    print(f"{'puzzle':<8} {'n':<4} {'mode':<6} {'time(s)':<10} {'mem(GB)':<10} {'states':<12} {'steps':<6} {'len':<4} {'found':<6}")
    print("-" * 80)
    for r in results:
        pz = r.get("puzzle_type", "?")
        print(
            f"{pz:<8} {r['state_size']:<4} {r['backward_mode']:<6} {r['elapsed_s']:<10.3f} "
            f"{r['peak_memory_gb']:<10.2f} {r['states_explored']:<12} "
            f"{r['steps']:<6} {r['solution_len']:<4} {'✓' if r['found'] else '✗':<6}"
        )


_STAGE_COL_WIDTH = 12


def print_stage_table(results: List[Dict[str, Any]]) -> None:
    """Print stage timing table.

    Note: bfs_prebuild is a non-exclusive total (includes expand, unique,
    meet_lookup, lower_bound inside it); do not sum with sub-stages.
    """
    stages = ["expand", "unique", "history_filter", "lower_bound", "beam_prune", "meet_lookup", "bfs_prebuild"]
    print("\n=== STAGE TIMING (seconds) ===")
    header = f"{'puzzle':<8} {'n':<4} {'mode':<6} " + " ".join(f"{s[:10]:<{_STAGE_COL_WIDTH}}" for s in stages)
    print(header)
    print("-" * len(header))
    for r in results:
        profile = r.get("profile")
        if not profile:
            continue
        pz = r.get("puzzle_type", "?")
        row = f"{pz:<8} {r['state_size']:<4} {r['backward_mode']:<6} "
        row += " ".join(f"{profile.get(s, {}).get('time', 0.0):<{_STAGE_COL_WIDTH}.3f}" for s in stages)
        print(row)


def print_stage_counts(results: List[Dict[str, Any]]) -> None:
    """Print stage call counts and states (states = input size per stage)."""
    stages = ["expand", "unique", "history_filter", "lower_bound", "beam_prune", "meet_lookup", "bfs_prebuild"]
    print("\n=== STAGE COUNTS (calls / states) ===")
    for r in results:
        profile = r.get("profile")
        if not profile:
            continue
        pz = r.get("puzzle_type", "?")
        print(f"\n  {pz} n={r['state_size']} mode={r['backward_mode']}:")
        for s in stages:
            c = profile.get(s, {})
            calls = c.get("calls", 0)
            states = c.get("states", 0)
            if calls > 0 or states > 0:
                print(f"    {s}: calls={calls} states={states}")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for profiling.")
    device = torch.device("cuda")

    print("Profiling BeamSolver")
    if RUNS:
        runs = RUNS
        print(f"  runs: {len(runs)} explicit configs")
    else:
        runs = [
            {
                "puzzle_type": CFG.puzzle_type,
                "state_size": n,
                "backward_mode": mode,
            }
            for n in CFG.state_sizes
            for mode in CFG.backward_modes
            if not (mode == "bfs" and n > 14)
        ]
        print(f"  puzzle: {CFG.puzzle_type}, state_sizes: {CFG.state_sizes}")
        print(f"  backward_modes: {CFG.backward_modes}, beam_width: {CFG.beam_width}")

    if runs:
        rc = runs[0]
        print(f"\n  Warmup {rc.get('puzzle_type', '?')} n={rc['state_size']} mode={rc['backward_mode']}...", end=" ", flush=True)
        run_one(
            rc["puzzle_type"],
            rc["state_size"],
            rc["backward_mode"],
            device,
            beam_width=rc.get("beam_width"),
            max_steps=rc.get("max_steps"),
        )
        print("done")

    results: List[Dict[str, Any]] = []
    for rc in runs:
        n, mode = rc["state_size"], rc["backward_mode"]
        pz = rc.get("puzzle_type", CFG.puzzle_type)
        print(f"\n  {pz} n={n} mode={mode}...", end=" ", flush=True)
        try:
            r = run_one(
                pz,
                n,
                mode,
                device,
                beam_width=rc.get("beam_width"),
                max_steps=rc.get("max_steps"),
            )
            results.append(r)
            print(f"✓ {r['elapsed_s']:.2f}s")
        except Exception as e:
            print(f"✗ {e}")
            results.append({
                "puzzle_type": pz,
                "state_size": n,
                "backward_mode": mode,
                "found": False,
                "elapsed_s": float("inf"),
                "error": str(e),
            })

    print_coarse_table(results)
    print_stage_table(results)
    print_stage_counts(results)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    import json
    out_path = RESULTS_DIR / "profile_results.json"
    serializable = []
    for r in results:
        s = {k: v for k, v in r.items() if k != "profile"}
        if "profile" in r:
            s["profile"] = r["profile"]
        serializable.append(s)
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
