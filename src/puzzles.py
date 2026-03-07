from dataclasses import dataclass
from typing import List
import torch


def get_state_dtype(state_size: int) -> torch.dtype:
    """int8 holds 0..n-1 only for n<=128; int16 for larger."""
    return torch.int8 if state_size <= 128 else torch.int16


@dataclass(frozen=True)
class PuzzleSpec:
    state_size: int
    move_names: List[str]
    move_indices: torch.Tensor
    inverse_moves: torch.Tensor
    solved_state: torch.Tensor
    conj_length: int  # conjugation length for steps/depth (puzzle-dependent)
    puzzle_type: str = ""  # e.g. "pancake" | "lrx" for adapter selection

    @property
    def state_dtype(self) -> torch.dtype:
        return get_state_dtype(self.state_size)


def make_lrx_spec(state_size: int, device: torch.device) -> PuzzleSpec:
    move_names = ["X", "L", "R"]
    idx_x = torch.tensor(
        [1, 0] + list(range(2, state_size)),
        device=device,
        dtype=torch.long,
    )
    idx_l = torch.roll(torch.arange(state_size, device=device), -1)
    idx_r = torch.roll(torch.arange(state_size, device=device), 1)
    move_indices = torch.stack([idx_x, idx_l, idx_r]).contiguous()
    inverse_moves = torch.tensor([0, 2, 1], device=device, dtype=torch.long)
    state_dtype = get_state_dtype(state_size)
    solved_state = torch.arange(state_size, device=device, dtype=state_dtype)
    conj_length = state_size * (state_size - 1) // 2
    return PuzzleSpec(
        state_size=state_size,
        move_names=move_names,
        move_indices=move_indices,
        inverse_moves=inverse_moves,
        solved_state=solved_state,
        conj_length=conj_length,
        puzzle_type="lrx",
    )


def make_pancake_spec(state_size: int, device: torch.device) -> PuzzleSpec:
    move_names = [f"R{k}" for k in range(2, state_size + 1)]
    indices = []
    for k in range(2, state_size + 1):
        prefix = list(range(k - 1, -1, -1))
        suffix = list(range(k, state_size))
        indices.append(
            torch.tensor(prefix + suffix, device=device, dtype=torch.long)
        )
    move_indices = torch.stack(indices).contiguous()
    inverse_moves = torch.arange(len(move_names), device=device, dtype=torch.long)
    state_dtype = get_state_dtype(state_size)
    solved_state = torch.arange(state_size, device=device, dtype=state_dtype)
    conj_length = (18 * state_size + 10) // 11
    return PuzzleSpec(
        state_size=state_size,
        move_names=move_names,
        move_indices=move_indices,
        inverse_moves=inverse_moves,
        solved_state=solved_state,
        conj_length=conj_length,
        puzzle_type="pancake",
    )


def make_spec(puzzle: str, state_size: int, device: torch.device) -> PuzzleSpec:
    if puzzle == "lrx":
        return make_lrx_spec(state_size, device)
    if puzzle == "pancake":
        return make_pancake_spec(state_size, device)
    raise ValueError(f"Unknown puzzle: {puzzle}")