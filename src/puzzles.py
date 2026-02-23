from dataclasses import dataclass
from typing import List
import torch


@dataclass(frozen=True)
class PuzzleSpec:
    state_size: int
    move_names: List[str]
    move_indices: torch.Tensor
    inverse_moves: torch.Tensor
    solved_state: torch.Tensor
    conj_length: int  # conjugation length for steps/depth (puzzle-dependent)


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
    solved_state = torch.arange(state_size, device=device, dtype=torch.int8)
    conj_length = state_size * (state_size - 1) // 2
    return PuzzleSpec(
        state_size=state_size,
        move_names=move_names,
        move_indices=move_indices,
        inverse_moves=inverse_moves,
        solved_state=solved_state,
        conj_length=conj_length,
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
    solved_state = torch.arange(state_size, device=device, dtype=torch.int8)
    conj_length = (18 * state_size + 10) // 11
    return PuzzleSpec(
        state_size=state_size,
        move_names=move_names,
        move_indices=move_indices,
        inverse_moves=inverse_moves,
        solved_state=solved_state,
        conj_length=conj_length,
    )


def make_spec(puzzle: str, state_size: int, device: torch.device) -> PuzzleSpec:
    if puzzle == "lrx":
        return make_lrx_spec(state_size, device)
    if puzzle == "pancake":
        return make_pancake_spec(state_size, device)
    raise ValueError(f"Unknown puzzle: {puzzle}")