"""Puzzle adapters used by solvers."""

from typing import Any, Optional, Tuple

import torch

from src.puzzles import PuzzleSpec


def _gap_count_batch(states: torch.Tensor) -> torch.Tensor:
    """Plate gap count for pancake states; admissible lower bound to identity."""
    n = states.size(1)
    gap = (states[:, :-1] - states[:, 1:]).abs().ne(1).sum(dim=1)
    gap = gap + (states[:, -1] - n).abs().ne(1)
    return gap


def _lrx_lower_bound_batch(states: torch.Tensor) -> torch.Tensor:
    """Admissible lower bound for LRX."""
    if states.size(0) == 0:
        return torch.empty(0, dtype=torch.long, device=states.device)

    n = states.size(1)
    if n <= 1:
        return torch.zeros(states.size(0), dtype=torch.long, device=states.device)

    states_long = states.to(torch.long)
    batch_size = int(states.size(0))

    inv = torch.zeros(batch_size, dtype=torch.long, device=states.device)
    for idx in range(n - 1):
        inv = inv + (
            states_long[:, idx : idx + 1] > states_long[:, idx + 1 :]
        ).long().sum(dim=1)
    inv_lb = (inv + n - 2) // (n - 1)

    positions = torch.empty(
        (batch_size, n),
        dtype=torch.long,
        device=states.device,
    )
    label_ids = torch.arange(n, dtype=torch.long, device=states.device)
    positions.scatter_(1, states_long, label_ids.unsqueeze(0).expand(
        batch_size, -1
    ))

    target_positions = label_ids.unsqueeze(0)
    delta = (positions - target_positions).abs()
    circ_delta = torch.minimum(delta, n - delta)
    disp_lb = circ_delta.max(dim=1).values

    return torch.maximum(inv_lb, disp_lb)


def get_adapter(
    puzzle_type: str,
    puzzle_spec: PuzzleSpec,
    device: torch.device,
    **options: Any,
) -> Any:
    """Create adapter from puzzle type."""
    if puzzle_type == "pancake":
        return PancakeAdapter(
            puzzle_spec=puzzle_spec,
            device=device,
            pancake_max_moves=options.get("pancake_max_moves", 0),
            neutral_only_if_next_decreasing=options.get(
                "neutral_only_if_next_decreasing", True
            ),
            logger=options.get("logger"),
        )
    if puzzle_type == "lrx":
        return LRXAdapter(puzzle_spec=puzzle_spec, device=device)
    raise ValueError(
        f"Unknown puzzle_type={puzzle_type!r}; expected 'pancake' or 'lrx'"
    )


class PancakeAdapter:
    """Contract-based pancake adapter without legacy modes."""

    def __init__(
        self,
        puzzle_spec: PuzzleSpec,
        device: torch.device,
        pancake_max_moves: int = 0,
        neutral_only_if_next_decreasing: bool = True,
        logger: Optional[Any] = None,
    ) -> None:
        self.puzzle_spec = puzzle_spec
        self.device = device
        self.logger = logger
        self.state_size = int(puzzle_spec.state_size)
        self.move_indices = puzzle_spec.move_indices
        self.n_moves = int(self.move_indices.size(0))
        self.moves_per_state = (
            self.n_moves
            if int(pancake_max_moves) <= 0
            else min(int(pancake_max_moves), self.n_moves)
        )
        self.neutral_only_if_next_decreasing = bool(
            neutral_only_if_next_decreasing
        )
        self._move_lengths = (
            torch.arange(self.n_moves, device=self.device, dtype=torch.long)
            + 2
        )
        self._start_state: Optional[torch.Tensor] = None
        self._start_inv: Optional[torch.Tensor] = None

    def prepare_search(self, start_state: torch.Tensor) -> None:
        """Store start_state and inverse (value -> position) for backward."""
        start_state = start_state.contiguous()
        self._start_inv = torch.empty(
            self.state_size,
            dtype=torch.long,
            device=self.device,
        )
        self._start_inv[start_state.to(torch.long)] = torch.arange(
            self.state_size,
            dtype=torch.long,
            device=self.device,
        )

    def lower_bound(self, states: torch.Tensor, direction: str) -> torch.Tensor:
        """Admissible lower bound: gap count to goal (identity or start)."""
        if states.size(0) == 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        if direction == "forward":
            return _gap_count_batch(states)
        if self._start_inv is None:
            return _gap_count_batch(states)
        rel = self._start_inv[states.to(torch.long)]
        return _gap_count_batch(rel)

    def set_logger(self, logger: Any) -> None:
        """Inject solver logger."""
        self.logger = logger

    def _delta_gap_masks(
        self,
        states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_states = int(states.size(0))
        if n_states == 0:
            shape = (0, self.n_moves)
            empty = torch.empty(shape, dtype=torch.bool, device=self.device)
            return empty, empty, empty
        move_lengths = self._move_lengths.unsqueeze(0).expand(n_states, -1)
        prefix_last = states.gather(1, move_lengths - 1)
        suffix_idx = torch.clamp(move_lengths, max=self.state_size - 1)
        suffix = states.gather(1, suffix_idx)
        suffix[:, -1] = int(self.state_size)
        top = states[:, :1]
        old_cut = (prefix_last - suffix).abs() != 1
        new_cut = (top - suffix).abs() != 1
        delta = new_cut.to(torch.int16) - old_cut.to(torch.int16)
        return delta == -1, delta == 0, delta == 1

    def _has_decreasing_move(self, states: torch.Tensor) -> torch.Tensor:
        dec, _, _ = self._delta_gap_masks(states)
        return dec.any(dim=1)

    def move_codes_and_mask(
        self,
        states: torch.Tensor,
        direction: str,
        **policy_options: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        max_moves = int(policy_options.get("max_moves", self.moves_per_state))
        max_moves = min(max(max_moves, 1), self.n_moves)
        dec, neu, inc = self._delta_gap_masks(states)
        if direction == "forward":
            base = dec
            if max_moves > 2:
                neutral = neu
                if self.neutral_only_if_next_decreasing:
                    neutral_pairs = neutral.nonzero(as_tuple=False)
                    if neutral_pairs.numel() > 0:
                        p_idx = neutral_pairs[:, 0]
                        m_idx = neutral_pairs[:, 1]
                        src = states.index_select(0, p_idx)
                        child_idx = self.move_indices.index_select(0, m_idx)
                        children = torch.gather(src, 1, child_idx)
                        dec_child = self._has_decreasing_move(children)
                        filtered = torch.zeros_like(neutral)
                        keep_pairs = neutral_pairs[dec_child]
                        if keep_pairs.numel() > 0:
                            filtered[keep_pairs[:, 0], keep_pairs[:, 1]] = True
                        neutral = filtered
                base = base | neutral
        else:
            base = inc
        key = base.to(torch.int8)
        top_vals, top_idx = torch.topk(
            key, k=max_moves, dim=1, largest=True, sorted=False
        )
        return top_idx.to(torch.long), top_vals.to(torch.bool)


class LRXAdapter:
    """LRX adapter with all moves enabled."""

    def __init__(self, puzzle_spec: PuzzleSpec, device: torch.device) -> None:
        self.puzzle_spec = puzzle_spec
        self.device = device
        self.logger = None
        self.move_indices = puzzle_spec.move_indices
        self.n_moves = int(self.move_indices.size(0))
        self.moves_per_state = self.n_moves
        self.state_size = int(puzzle_spec.state_size)
        self._anchor_inv: Optional[torch.Tensor] = None

    def prepare_search(self, start_state: torch.Tensor) -> None:
        """Store anchor inverse (value -> position) for backward."""
        start_state = start_state.contiguous()
        self._anchor_inv = torch.empty(
            self.state_size,
            dtype=torch.long,
            device=self.device,
        )
        self._anchor_inv[start_state.to(torch.long)] = torch.arange(
            self.state_size,
            dtype=torch.long,
            device=self.device,
        )

    def lower_bound(self, states: torch.Tensor, direction: str) -> torch.Tensor:
        """Admissible lower bound for LRX."""
        if states.size(0) == 0:
            return torch.empty(0, dtype=torch.long, device=self.device)

        if direction == "forward":
            rel = states
        elif self._anchor_inv is None:
            rel = states
        else:
            rel = self._anchor_inv[states.to(torch.long)]

        return _lrx_lower_bound_batch(rel)

    def set_logger(self, logger: Any) -> None:
        """Inject solver logger."""
        self.logger = logger

    def move_codes_and_mask(
        self,
        states: torch.Tensor,
        direction: str,
        **policy_options: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_states = int(states.size(0))
        max_moves = int(policy_options.get("max_moves", self.n_moves))
        max_moves = min(max(max_moves, 1), self.n_moves)
        move_codes = torch.arange(
            max_moves, device=self.device, dtype=torch.long
        ).unsqueeze(0).expand(n_states, -1)
        valid_mask = torch.ones(
            (n_states, max_moves), device=self.device, dtype=torch.bool
        )
        return move_codes, valid_mask
