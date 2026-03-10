"""Solve one configured puzzle instance from a given or random permutation.

Usage: uv run solve_puzzle.py
"""

import logging
import random

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch

from src.data_gen.random_walks import first_visit_random_walks, random_walks_beam_nbt 
from src.models.greedy_gap_model import GreedyGapModel
from src.puzzles import make_spec
from src.solvers.factory import make_solver
from src.solvers.puzzle_adapters import get_adapter
from src.utils import assert_solution_sorts_state


################################################################################
# CONFIG — edit parameters here
################################################################################

@dataclass
class InputConfig:
    puzzle: str = "lrx"
    # "pancake" | "lrx"

    state: Optional[list[int]] = field(
        default_factory=lambda: [1, 0, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2]
    )
    # None = generate a random permutation.
    # list[int] = solve exactly this permutation.

    random_state_size: int = 20
    # Used only when state is None.

    random_seed: int = 1
    # Seed for Python, NumPy, Torch, and random state generation.


@dataclass
class ModelConfig:
    model_name: str = "xgboost"
    # "greedy_gap" | "xgboost" | "catboost" | "mlp"
    # greedy_gap is pancake-only.
    # other models work for pancake and lrx after training on random walks
    # (add imports to use them).

    greedy_mode: str = "gap"
    # "gap" | "rokicki" when model_name == "greedy_gap"

    greedy_gap_weights: tuple[float, float] = (0.0, 0.0)
    # (w2, w3) for higher-order gap terms in greedy_gap mode="gap".

    greedy_locked_gap_add: int = 0
    # 0 | 1
    # Adds one extra unit in locked non-goal states.

    greedy_use_dual_max: bool = False
    # If True, use max(score(state), score(inverse(state))).


@dataclass
class TrainConfig:
    rw_type: str = "beam_nbt"
    # "beam_nbt" | "first_visit"
    # Used only when model_name == "xgboost" | "catboost" | "mlp".

    n_walks: int = 10_000
    # Number of random walks for training data generation.

    rw_nbt_depth: int = 100
    # Non-backtracking depth for rw_type == "beam_nbt".

    rw_steps_multiplier: float = 2.0
    # Random-walk depth = int(conj_length * rw_steps_multiplier).


@dataclass
class AdapterConfig:
    pancake_max_moves: int = 0
    # Pancake only.
    # 0 means "use adapter default full move budget".
    # Internally this still passes through adapter policy logic:
    # in forward search the adapter does not simply allow every move;
    # it ranks/filter moves by delta-gap policy and may keep only
    # decreasing moves, or decreasing + some neutral moves.

    neutral_only_if_next_decreasing: bool = True
    # Pancake only.
    # If True, a neutral move is allowed only when the resulting child
    # has at least one decreasing move available next.


@dataclass
class SearchConfig:
    solver_type: str = "beam"
    # "beam" | "pancake_exact"
    # pancake_exact is supported only for puzzle == "pancake".

    beam_width: int = 2**18
    # Beam width for beam-based search.

    max_steps: int = None
    # Search depth limit, if None, max_speps = puzzle_spec.conj_length.

    bs_nbt_depth: int = 2
    # Beam-search non-backtracking depth.

    backward_mode: str = "off"
    # "off" | "bfs" | "beam"

    backward_max_states: int = 0
    # set only for bfs mode.

    hashes_batch_size: int = 10_000_000
    # Batch size for hash-heavy internal filtering.

    randomize_ties: bool = True
    # Randomize tie order inside beam pruning/scoring where supported.


@dataclass
class ExactConfig:
    exact_verify_margin: int = 2
    # Currently implemented: 0 | 1 | 2
    # Values >= 3 are not implemented.

    exact_incumbent_solver_type: str = "beam"
    # Currently this should remain "beam".


@dataclass
class RuntimeConfig:
    verbose: int = 1
    # 0 = warnings, 1 = info, 2 = debug


@dataclass
class Config:
    input: InputConfig = field(default_factory=InputConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    exact: ExactConfig = field(default_factory=ExactConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


cfg = Config()

RW_FUNCTIONS = {
    "beam_nbt": random_walks_beam_nbt,
    "first_visit": first_visit_random_walks,
}


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and Torch seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return CUDA device and fail loudly if CUDA is unavailable."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required. This repository provides GPU-only solvers."
        )
    return torch.device("cuda")


def make_start_state(config: Config, device: torch.device) -> torch.Tensor:
    """Return configured state or a random permutation."""
    if config.input.state is not None:
        values = config.input.state
    else:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.input.random_seed)
        values = torch.randperm(
            config.input.random_state_size,
            generator=generator,
        ).tolist()
    return torch.tensor(values, device=device)


def create_model(state_size: int, config: Config) -> Any:
    """Create the configured model."""
    model_name = config.model.model_name
    if model_name == "greedy_gap":
        if config.input.puzzle != "pancake":
            raise ValueError(
                "model_name='greedy_gap' is supported only for puzzle='pancake'."
            )
        return GreedyGapModel(
            state_size=state_size,
            w2=config.model.greedy_gap_weights[0],
            w3=config.model.greedy_gap_weights[1],
            mode=config.model.greedy_mode,
            locked_gap_add=config.model.greedy_locked_gap_add,
            use_dual_max=config.model.greedy_use_dual_max,
        )
    if model_name == "xgboost":
        from src.models.xgboost_model import XGBoostModel

        return XGBoostModel()
    if model_name == "catboost":
        from src.models.catboost_model import CatBoostModel

        return CatBoostModel()
    if model_name == "mlp":
        from src.models.mlp_model import MLPModel

        return MLPModel(
            input_size=state_size,
            verbose=config.runtime.verbose,
        )
    raise ValueError(
        "model.model_name must be 'greedy_gap', 'xgboost', 'catboost', or "
        f"'mlp'; got {model_name!r}"
    )


def train_model_if_needed(
    model: Any,
    puzzle_spec: Any,
    config: Config,
    device: torch.device,
) -> None:
    """Train the selected model when training is required."""
    if config.model.model_name not in {"xgboost", "catboost", "mlp"}:
        return

    rw_type = config.train.rw_type
    if rw_type not in RW_FUNCTIONS:
        raise ValueError(
            "train.rw_type must be one of "
            f"{list(RW_FUNCTIONS)}, got {rw_type!r}"
        )

    rw_steps = max(
        1,
        int(puzzle_spec.conj_length * config.train.rw_steps_multiplier),
    )
    rw_fun = RW_FUNCTIONS[rw_type]

    if rw_type == "beam_nbt":
        X, y = rw_fun(
            puzzle_spec.move_indices,
            n_steps=rw_steps,
            n_walks=config.train.n_walks,
            device=device,
            nbt_depth=config.train.rw_nbt_depth,
        )
    else:
        X, y = rw_fun(
            puzzle_spec.move_indices,
            rw_steps,
            config.train.n_walks,
            device,
        )

    model.train(X, y)
    torch.cuda.empty_cache()


def create_solver(
    puzzle_spec: Any,
    config: Config,
    device: torch.device,
) -> Any:
    """Create the configured solver."""
    adapter_options: dict[str, Any] = {}
    if config.input.puzzle == "pancake":
        adapter_options["pancake_max_moves"] = config.adapter.pancake_max_moves
        adapter_options["neutral_only_if_next_decreasing"] = (
            config.adapter.neutral_only_if_next_decreasing
        )

    return make_solver(
        puzzle_spec,
        config.search.solver_type,
        get_adapter(config.input.puzzle, puzzle_spec, device, **adapter_options),
        beam_width=config.search.beam_width,
        max_steps=config.search.max_steps,
        backward_mode=config.search.backward_mode,
        backward_max_states=config.search.backward_max_states,
        bs_nbt_depth=config.search.bs_nbt_depth,
        hashes_batch_size=config.search.hashes_batch_size,
        randomize_ties=config.search.randomize_ties,
        verbose=config.runtime.verbose,
        exact_verify_margin=config.exact.exact_verify_margin,
        exact_incumbent_solver_type=(
            config.exact.exact_incumbent_solver_type
        ),
    )


def print_config(config: Config) -> None:
    """Print the current configuration."""
    parts = [
        f"puzzle={config.input.puzzle}",
        f"input.state={config.input.state}",
        f"input.random_state_size={config.input.random_state_size}",
        f"input.random_seed={config.input.random_seed}",
        f"model.model_name={config.model.model_name}",
        f"search.solver_type={config.search.solver_type}",
        f"search.beam_width={config.search.beam_width}",
        f"search.max_steps={config.search.max_steps}",
        f"search.bs_nbt_depth={config.search.bs_nbt_depth}",
        f"search.backward_mode={config.search.backward_mode}",
        "search.backward_max_states="
        f"{config.search.backward_max_states}",
        "search.hashes_batch_size="
        f"{config.search.hashes_batch_size}",
        f"search.randomize_ties={config.search.randomize_ties}",
        f"runtime.verbose={config.runtime.verbose}",
    ]

    if config.model.model_name == "greedy_gap":
        parts.extend(
            [
                f"model.greedy_mode={config.model.greedy_mode}",
                "model.greedy_gap_weights="
                f"{config.model.greedy_gap_weights}",
                "model.greedy_locked_gap_add="
                f"{config.model.greedy_locked_gap_add}",
                "model.greedy_use_dual_max="
                f"{config.model.greedy_use_dual_max}",
            ]
        )

    if config.model.model_name in {"xgboost", "catboost", "mlp"}:
        parts.extend(
            [
                f"train.rw_type={config.train.rw_type}",
                f"train.n_walks={config.train.n_walks}",
                f"train.rw_nbt_depth={config.train.rw_nbt_depth}",
                "train.rw_steps_multiplier="
                f"{config.train.rw_steps_multiplier}",
            ]
        )

    if config.input.puzzle == "pancake":
        parts.extend(
            [
                "adapter.pancake_max_moves="
                f"{config.adapter.pancake_max_moves}",
                "adapter.neutral_only_if_next_decreasing="
                f"{config.adapter.neutral_only_if_next_decreasing}",
            ]
        )

    if config.search.solver_type == "pancake_exact":
        parts.extend(
            [
                "exact.exact_verify_margin="
                f"{config.exact.exact_verify_margin}",
                "exact.exact_incumbent_solver_type="
                f"{config.exact.exact_incumbent_solver_type}",
            ]
        )

    print("config: " + " | ".join(parts), flush=True)


def main() -> None:
    set_seed(cfg.input.random_seed)
    log_level = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG,
    }.get(cfg.runtime.verbose, logging.INFO)
    logging.basicConfig(level=log_level, format="%(message)s")
    device = get_device()

    start_state = make_start_state(cfg, device)
    state_size = int(start_state.numel())
    puzzle_spec = make_spec(cfg.input.puzzle, state_size, device)
    start_state = start_state.to(dtype=puzzle_spec.state_dtype).contiguous()

    if cfg.search.max_steps is None:
        cfg.search.max_steps = puzzle_spec.conj_length

    print_config(cfg)

    model = create_model(state_size, cfg)
    train_model_if_needed(model, puzzle_spec, cfg, device)
    solver = create_solver(puzzle_spec, cfg, device)

    lower_bound = int(
        solver.adapter.lower_bound(
            start_state.unsqueeze(0),
            "forward",
        )[0].item()
    )
    name_to_code = {
        name: code for code, name in enumerate(puzzle_spec.move_names)
    }

    torch.cuda.reset_peak_memory_stats(device)
    found, steps, solution = solver.solve(start_state=start_state, model=model)
    peak_memory_gb = torch.cuda.max_memory_allocated(device) / (1024**3)

    if found:
        assert_solution_sorts_state(
            start_state=start_state,
            solution=solution,
            name_to_code=name_to_code,
            move_indices=puzzle_spec.move_indices,
        )

    print(flush=True)
    print(f"puzzle: {cfg.input.puzzle}", flush=True)
    print(f"n: {state_size}", flush=True)
    print(f"state: {start_state.tolist()}", flush=True)
    print(f"model: {cfg.model.model_name}", flush=True)
    print(f"solver: {cfg.search.solver_type}", flush=True)
    print(f"lower_bound: {lower_bound}", flush=True)
    print(f"found: {found}", flush=True)
    print(f"steps: {steps}", flush=True)
    print(
        f"solution: {solution if found else 'NOT FOUND'}",
        flush=True,
    )
    print(
        f"search_time: {solver.search_stats.get('search_time', 0.0):.3f}s",
        flush=True,
    )
    print(
        "states_explored: "
        f"{solver.search_stats.get('total_states_explored', 0)}",
        flush=True,
    )
    print(
        "termination_reason: "
        f"{solver.search_stats.get('termination_reason', '')}",
        flush=True,
    )
    print(f"peak_memory_gb: {peak_memory_gb:.2f}", flush=True)


if __name__ == "__main__":
    main()