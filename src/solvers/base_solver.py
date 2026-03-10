from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Tuple, Union
import logging

import torch

from src.puzzles import PuzzleSpec, get_state_dtype


MovesLike = Union[int, torch.Tensor, Iterable[Union[int, torch.Tensor]]]


class BaseSolver(ABC):
    """
    Base class for GPU permutation solvers. Subclasses implement reset() and solve().

    Solvers are puzzle-agnostic: they only implement beam/search logic (expand,
    filter, prune, intersect). All puzzle-specific behaviour (moves, bounds,
    heuristics, lookahead) belongs in the adapter. Do not add puzzle logic to
    solvers—change the adapter instead.

    Usage
    -----
    - Construct with puzzle_spec (from make_pancake_spec() or make_lrx_spec()) and device.
    - Call solve(start_state, model) with start_state a 1D tensor of shape (state_size,)
      on the same device; model is used for scoring (can be None if adapter handles it).
    - solve() returns (found: bool, steps: int, solution: str). steps is the effective
      search limit on failure; on success it is the
      solution length. solution is a move sequence string (e.g. "f2.f3") or "".

    Contract
    --------
    - puzzle_spec tensors (move_indices, inverse_moves, solved_state) are on `device`
      and contiguous.
    - start_state: 1D permutation of length state_size, same device, dtype from
      get_state_dtype(state_size).
    - Subclasses use inverse_move_indices for backward apply (precomputed here).
    """

    def __init__(
        self,
        puzzle_spec: PuzzleSpec,
        device: torch.device,
        model: Any = None,
        verbose: int = 0,
    ) -> None:
        """Init shared state: device, move indices, logger, search_stats."""
        self.puzzle_spec = puzzle_spec
        self.state_size = int(puzzle_spec.state_size)
        self.state_dtype = get_state_dtype(self.state_size)
        self.device = device
        self.model = model
        self.verbose = int(verbose)

        self.move_names = list(puzzle_spec.move_names)
        self.move_indices = puzzle_spec.move_indices
        self.inverse_moves = puzzle_spec.inverse_moves
        
        # Precompute inverse move indices for backward expansion (hot path)
        self.inverse_move_indices = self.move_indices[self.inverse_moves].contiguous()
        self.solved_state = puzzle_spec.solved_state
        self.n_moves = int(self.move_indices.size(0))

        self.logger = self._make_logger()
        self.search_stats: Dict[str, Any] = {
            "total_states_explored": 0,
            "search_time": 0.0,
            "path_found": False,
        }

    def _make_logger(self) -> logging.Logger:
        """Per-class logger; verbosity 0=WARNING, 1=INFO, 2=DEBUG."""
        log_level = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG,
        }.get(self.verbose, logging.DEBUG)

        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(log_level)
        logger.propagate = False

        if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
            handler = logging.StreamHandler()
            handler.setLevel(log_level)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def log_warning(self, message: str) -> None:
        """Log a warning message."""
        self.logger.warning(message)

    def log_info(self, message: str) -> None:
        """Log an info message."""
        self.logger.info(message)

    def log_debug(self, message: str) -> None:
        """Log a debug message."""
        self.logger.debug(message)

    def _cuda_allocated_gb(self) -> float:
        """Return allocated CUDA memory in GiB."""
        if self.device.type != "cuda":
            return 0.0
        return float(torch.cuda.memory_allocated(self.device) / (1024**3))

    def _get_inverse_move(self, move: Union[int, torch.Tensor]) -> int:
        """Return inverse move code."""
        move_code = int(move) if isinstance(move, int) else int(move.item())
        return int(self.inverse_moves[move_code].item())

    def _apply_moves(self, state: torch.Tensor, moves: MovesLike) -> torch.Tensor:
        """Apply a single move or sequence of moves to 1D state; returns new state."""
        if isinstance(moves, int):
            idx = self.move_indices[int(moves)]
            return state.index_select(0, idx)

        if isinstance(moves, torch.Tensor) and moves.ndim == 0:
            idx = self.move_indices[int(moves.item())]
            return state.index_select(0, idx)

        next_state = state
        for move in moves:
            move_code = int(move) if isinstance(move, int) else int(move.item())
            idx = self.move_indices[move_code]
            next_state = next_state.index_select(0, idx)
        return next_state

    def cleanup(self) -> None:
        """Release model reference and clear CUDA cache."""
        self.model = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _debug_build_ref_states(
        self, start_state: torch.Tensor, solution_str: str
    ) -> List[torch.Tensor]:
        """Temporary debug: build [s0, s1, ...] by applying solution moves. Remove when not needed."""
        parts = [p.strip() for p in solution_str.strip().split(".") if p.strip()]
        indices: List[int] = []
        for name in parts:
            if name not in self.move_names:
                return []
            indices.append(self.move_names.index(name))
        ref: List[torch.Tensor] = [start_state.clone()]
        for m in indices:
            ref.append(self._apply_moves(ref[-1], m))
        return ref

    @abstractmethod
    def reset(self) -> None:
        """Reset solver state for a new problem."""
        raise NotImplementedError

    @abstractmethod
    def solve(
        self,
        start_state: torch.Tensor,
        model: Any,
    ) -> Tuple[bool, int, str]:
        """Run search; return (found, steps, solution_string). steps = effective limit if not found."""
        raise NotImplementedError