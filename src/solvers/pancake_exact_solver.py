"""Backward-compatible pancake name for the universal exact verifier.

The algorithm no longer lives here. New code should construct
``ExactVerifier`` with ``make_pancake_exact_kernel``. This wrapper only keeps
old imports/call sites working.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from src.puzzles import PuzzleSpec
from src.solvers.base_solver import BaseSolver
from src.solvers.exact_kernels import make_pancake_exact_kernel
from src.solvers.exact_verifier import ExactVerifier


class PancakeExactSolver(ExactVerifier):
    """Compatibility wrapper around ``ExactVerifier``."""

    def __init__(
        self,
        *,
        puzzle_spec: PuzzleSpec,
        device: torch.device,
        adapter: Any,
        incumbent_solver: Optional[BaseSolver],
        exact_verify_margin: int = 2,
        exact_tt_capacity: int = 100_000_000,
        verbose: int = 0,
    ) -> None:
        super().__init__(
            puzzle_spec=puzzle_spec,
            device=device,
            exact_kernel=make_pancake_exact_kernel(puzzle_spec),
            adapter=adapter,
            incumbent_solver=incumbent_solver,
            exact_verify_margin=exact_verify_margin,
            exact_tt_capacity=exact_tt_capacity,
            verbose=verbose,
        )
