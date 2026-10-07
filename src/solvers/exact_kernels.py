"""Numba kernels used by the universal exact verifier.

This module contains only puzzle/lower-bound-specific operations. The exact
search traversal itself lives in :mod:`src.solvers.exact_search`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numba import njit

from src.puzzles import PuzzleSpec


KERNEL_PANCAKE_GAP = 0
KERNEL_ORBIT_SUBSETS = 1
KERNEL_CUBE555_STRUCTURAL = 2


@dataclass(frozen=True)
class ExactKernel:
    """Static NumPy inputs consumed by the shared Numba exact-search core."""

    kind: int
    move_indices: np.ndarray
    inverse_moves: np.ndarray
    solved_state: np.ndarray
    piece_distance: np.ndarray
    piece_orbit: np.ndarray
    subset_masks: np.ndarray
    subset_capacities: np.ndarray
    n_orbits: int
    corner_dist: np.ndarray
    corner_ori_code_to_idx: np.ndarray
    corner_slot0_positions: np.ndarray
    corner_piece_to_cubie: np.ndarray
    corner_orientation_lookup: np.ndarray
    inner_cut_positions: np.ndarray
    inner_cut_lengths: np.ndarray
    inner_goal_piece_mask: np.ndarray
    inner_dual_cut_indices: np.ndarray
    inner_dual_weights_x4: np.ndarray
    inner_parity_orbit_positions: np.ndarray
    inner_parity_orbit_lengths: np.ndarray
    inner_parity_goal_local_by_piece: np.ndarray
    inner_parity_move_toggle: np.ndarray
    corner_orientation_count: int


def _as_int16_c(array: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(array, dtype=np.int16)


def _empty_cube555_kernel_fields() -> dict[str, Any]:
    return {
        "corner_dist": np.empty(0, dtype=np.uint8),
        "corner_ori_code_to_idx": np.empty(0, dtype=np.int32),
        "corner_slot0_positions": np.empty(0, dtype=np.int16),
        "corner_piece_to_cubie": np.empty(0, dtype=np.int16),
        "corner_orientation_lookup": np.empty((0, 0), dtype=np.int8),
        "inner_cut_positions": np.empty((0, 0), dtype=np.int16),
        "inner_cut_lengths": np.empty(0, dtype=np.int16),
        "inner_goal_piece_mask": np.empty((0, 0), dtype=np.uint8),
        "inner_dual_cut_indices": np.empty((0, 2), dtype=np.int16),
        "inner_dual_weights_x4": np.empty((0, 2), dtype=np.int16),
        "inner_parity_orbit_positions": np.empty((0, 0), dtype=np.int16),
        "inner_parity_orbit_lengths": np.empty(0, dtype=np.int16),
        "inner_parity_goal_local_by_piece": np.empty((0, 0), dtype=np.int16),
        "inner_parity_move_toggle": np.empty(0, dtype=np.uint8),
        "corner_orientation_count": 0,
    }


def make_pancake_exact_kernel(puzzle_spec: PuzzleSpec) -> ExactKernel:
    """Build the optimized gap kernel for a pancake ``PuzzleSpec``."""
    if puzzle_spec.puzzle_type != "pancake":
        raise ValueError(
            "make_pancake_exact_kernel requires puzzle_type='pancake'; "
            f"got {puzzle_spec.puzzle_type!r}"
        )
    if puzzle_spec.state_size > np.iinfo(np.int16).max:
        raise ValueError("ExactVerifier requires state_size <= 32767")

    return ExactKernel(
        kind=KERNEL_PANCAKE_GAP,
        move_indices=_as_int16_c(
            puzzle_spec.move_indices.detach().cpu().numpy()
        ),
        inverse_moves=_as_int16_c(
            puzzle_spec.inverse_moves.detach().cpu().numpy()
        ),
        solved_state=_as_int16_c(
            puzzle_spec.solved_state.detach().cpu().numpy()
        ),
        piece_distance=np.empty((0, 0), dtype=np.int16),
        piece_orbit=np.empty(0, dtype=np.int16),
        subset_masks=np.empty(0, dtype=np.int16),
        subset_capacities=np.empty(0, dtype=np.int16),
        n_orbits=0,
        **_empty_cube555_kernel_fields(),
    )


def make_orbit_subset_exact_kernel(
    puzzle_spec: PuzzleSpec,
    orbit_subset_bound: Any,
) -> ExactKernel:
    """Build a Numba kernel equivalent to ``OrbitSubsetLowerBound``.

    The bound object is duck-typed deliberately: reusable solver code does not
    import a competition package. It only consumes arrays exposed by the
    generic piece-work/orbit-subset implementation.
    """
    if puzzle_spec.state_size > np.iinfo(np.int16).max:
        raise ValueError("ExactVerifier requires state_size <= 32767")

    base = orbit_subset_bound.base_adapter
    state_size = int(puzzle_spec.state_size)
    n_orbits = len(base.orbits)
    if n_orbits <= 0 or n_orbits > 15:
        raise ValueError(
            "orbit-subset kernel currently requires 1..15 orbits; "
            f"got {n_orbits}"
        )

    piece_orbit = np.full(state_size, -1, dtype=np.int16)
    for orbit_idx, piece_ids in enumerate(base.orbit_piece_ids):
        ids = piece_ids.detach().cpu().numpy().astype(np.int64, copy=False)
        piece_orbit[ids] = np.int16(orbit_idx)
    if np.any(piece_orbit < 0):
        raise ValueError("Every piece must belong to exactly one invariant orbit")

    masks: list[int] = []
    capacities: list[int] = []
    for orbit_ids, capacity in zip(
        orbit_subset_bound.subset_orbit_ids,
        orbit_subset_bound.subset_capacities,
    ):
        ids = orbit_ids.detach().cpu().numpy().astype(np.int64, copy=False)
        mask = 0
        for orbit_idx in ids:
            mask |= 1 << int(orbit_idx)
        masks.append(mask)
        capacities.append(int(capacity))

    piece_distance = base.piece_distance.detach().cpu().numpy()
    if piece_distance.shape != (state_size, state_size):
        raise ValueError(
            "piece_distance must have shape "
            f"({state_size}, {state_size}); got {piece_distance.shape}"
        )

    return ExactKernel(
        kind=KERNEL_ORBIT_SUBSETS,
        move_indices=_as_int16_c(
            puzzle_spec.move_indices.detach().cpu().numpy()
        ),
        inverse_moves=_as_int16_c(
            puzzle_spec.inverse_moves.detach().cpu().numpy()
        ),
        solved_state=_as_int16_c(
            puzzle_spec.solved_state.detach().cpu().numpy()
        ),
        piece_distance=_as_int16_c(piece_distance),
        piece_orbit=_as_int16_c(piece_orbit),
        subset_masks=np.ascontiguousarray(masks, dtype=np.int16),
        subset_capacities=np.ascontiguousarray(capacities, dtype=np.int16),
        n_orbits=n_orbits,
        **_empty_cube555_kernel_fields(),
    )


def make_cube555_structural_exact_kernel(
    puzzle_spec: PuzzleSpec,
    structural_bound: Any,
) -> ExactKernel:
    """Build the Numba kernel equivalent to ``Cube555FastLowerBound``.

    The competition-specific bound is duck-typed.  Shared exact-search code
    consumes only immutable NumPy arrays and remains unaware of CSV/state-table
    orchestration.
    """
    if puzzle_spec.state_size > np.iinfo(np.int16).max:
        raise ValueError("ExactVerifier requires state_size <= 32767")

    arrays = structural_bound.exact_kernel_arrays()
    return ExactKernel(
        kind=KERNEL_CUBE555_STRUCTURAL,
        move_indices=_as_int16_c(
            puzzle_spec.move_indices.detach().cpu().numpy()
        ),
        inverse_moves=_as_int16_c(
            puzzle_spec.inverse_moves.detach().cpu().numpy()
        ),
        solved_state=_as_int16_c(
            puzzle_spec.solved_state.detach().cpu().numpy()
        ),
        piece_distance=np.empty((0, 0), dtype=np.int16),
        piece_orbit=np.empty(0, dtype=np.int16),
        subset_masks=np.empty(0, dtype=np.int16),
        subset_capacities=np.empty(0, dtype=np.int16),
        n_orbits=0,
        corner_dist=np.ascontiguousarray(arrays["corner_dist"], dtype=np.uint8),
        corner_ori_code_to_idx=np.ascontiguousarray(
            arrays["corner_ori_code_to_idx"], dtype=np.int32
        ),
        corner_slot0_positions=np.ascontiguousarray(
            arrays["corner_slot0_positions"], dtype=np.int16
        ),
        corner_piece_to_cubie=np.ascontiguousarray(
            arrays["corner_piece_to_cubie"], dtype=np.int16
        ),
        corner_orientation_lookup=np.ascontiguousarray(
            arrays["corner_orientation_lookup"], dtype=np.int8
        ),
        inner_cut_positions=np.ascontiguousarray(
            arrays["inner_cut_positions"], dtype=np.int16
        ),
        inner_cut_lengths=np.ascontiguousarray(
            arrays["inner_cut_lengths"], dtype=np.int16
        ),
        inner_goal_piece_mask=np.ascontiguousarray(
            arrays["inner_goal_piece_mask"], dtype=np.uint8
        ),
        inner_dual_cut_indices=np.ascontiguousarray(
            arrays["inner_dual_cut_indices"], dtype=np.int16
        ),
        inner_dual_weights_x4=np.ascontiguousarray(
            arrays["inner_dual_weights_x4"], dtype=np.int16
        ),
        inner_parity_orbit_positions=np.ascontiguousarray(
            arrays["inner_parity_orbit_positions"], dtype=np.int16
        ),
        inner_parity_orbit_lengths=np.ascontiguousarray(
            arrays["inner_parity_orbit_lengths"], dtype=np.int16
        ),
        inner_parity_goal_local_by_piece=np.ascontiguousarray(
            arrays["inner_parity_goal_local_by_piece"], dtype=np.int16
        ),
        inner_parity_move_toggle=np.ascontiguousarray(
            arrays["inner_parity_move_toggle"], dtype=np.uint8
        ),
        corner_orientation_count=int(arrays["corner_orientation_count"]),
    )


@njit(cache=False)  # cache=True + torch in process can segfault on a 2nd call
def gap_count_numba(state: np.ndarray) -> int:
    """Return the standard pancake plate-gap lower bound."""
    n = state.shape[0]
    total = 0
    for idx in range(n - 1):
        if abs(int(state[idx]) - int(state[idx + 1])) != 1:
            total += 1
    if abs(int(state[n - 1]) - n) != 1:
        total += 1
    return total


@njit(cache=False)
def _delta_gap_for_move_numba(state: np.ndarray, move_code: int) -> int:
    n = state.shape[0]
    k = move_code + 2
    prefix_last = int(state[k - 1])
    suffix_val = n if k == n else int(state[k])
    old_cut = 1 if abs(prefix_last - suffix_val) != 1 else 0
    new_cut = 1 if abs(int(state[0]) - suffix_val) != 1 else 0
    return new_cut - old_cut


@njit(cache=False)
def build_pos_numba(state: np.ndarray) -> np.ndarray:
    """Build inverse positions for the pancake fast candidate kernel."""
    pos = np.empty(state.shape[0], dtype=np.int16)
    for idx in range(state.shape[0]):
        pos[int(state[idx])] = idx
    return pos


@njit(cache=False)
def _pancake_decreasing_candidates_numba(
    state: np.ndarray,
    pos: np.ndarray,
    out_moves: np.ndarray,
) -> int:
    """Write every gap-decreasing pancake move; at most three exist."""
    n = state.shape[0]
    top = int(state[0])
    count = 0

    if top > 0:
        idx = int(pos[top - 1])
        if idx >= 2 and abs(int(state[idx - 1]) - int(state[idx])) != 1:
            out_moves[count] = np.int16(idx - 2)
            count += 1

    if top + 1 < n:
        idx = int(pos[top + 1])
        if idx >= 2 and abs(int(state[idx - 1]) - int(state[idx])) != 1:
            move_code = idx - 2
            duplicate = False
            for j in range(count):
                if int(out_moves[j]) == move_code:
                    duplicate = True
                    break
            if not duplicate:
                out_moves[count] = np.int16(move_code)
                count += 1

    if top == n - 1:
        move_code = n - 2
        if abs(int(state[n - 1]) - n) != 1:
            duplicate = False
            for j in range(count):
                if int(out_moves[j]) == move_code:
                    duplicate = True
                    break
            if not duplicate:
                out_moves[count] = np.int16(move_code)
                count += 1

    return count


@njit(cache=False)
def is_solved_numba(state: np.ndarray, solved_state: np.ndarray) -> bool:
    for idx in range(state.shape[0]):
        if state[idx] != solved_state[idx]:
            return False
    return True


@njit(cache=False)
def _orbit_subset_lower_bound_numba(
    state: np.ndarray,
    piece_distance: np.ndarray,
    piece_orbit: np.ndarray,
    subset_masks: np.ndarray,
    subset_capacities: np.ndarray,
    n_orbits: int,
) -> int:
    """Integer implementation of the cheap all-orbit-subsets bound."""
    orbit_work = np.zeros(n_orbits, dtype=np.int32)
    single = 0

    # state[position] = piece. piece_distance[piece, position] is the exact
    # isolated-piece distance to that piece's goal position.
    for position in range(state.shape[0]):
        piece = int(state[position])
        distance = int(piece_distance[piece, position])
        if distance < 0:
            return 32767
        if distance > single:
            single = distance
        orbit_idx = int(piece_orbit[piece])
        orbit_work[orbit_idx] += distance

    best = single
    for subset_idx in range(subset_masks.shape[0]):
        mask = int(subset_masks[subset_idx])
        work = 0
        for orbit_idx in range(n_orbits):
            if mask & (1 << orbit_idx):
                work += int(orbit_work[orbit_idx])
        capacity = int(subset_capacities[subset_idx])
        bound = (work + capacity - 1) // capacity
        if bound > best:
            best = bound
    return best

@njit(cache=False)
def _cube555_inner_parity_numba(
    state: np.ndarray,
    inner_parity_orbit_positions: np.ndarray,
    inner_parity_orbit_lengths: np.ndarray,
    inner_parity_goal_local_by_piece: np.ndarray,
) -> int:
    """Required parity of remaining inner moves for one cube555 state."""
    if inner_parity_orbit_lengths.shape[0] == 0:
        return 0
    max_orbit = inner_parity_orbit_positions.shape[1]
    permutation = np.empty(max_orbit, dtype=np.int16)
    seen = np.empty(max_orbit, dtype=np.uint8)
    required_parity = 0
    for parity_row in range(inner_parity_orbit_lengths.shape[0]):
        length = int(inner_parity_orbit_lengths[parity_row])
        for i in range(length):
            position = int(inner_parity_orbit_positions[parity_row, i])
            piece = int(state[position])
            local_goal = int(inner_parity_goal_local_by_piece[parity_row, piece])
            if local_goal < 0:
                return -1
            permutation[i] = np.int16(local_goal)
            seen[i] = np.uint8(0)

        cycles = 0
        for i in range(length):
            if seen[i] != 0:
                continue
            cycles += 1
            j = i
            while seen[j] == 0:
                seen[j] = np.uint8(1)
                j = int(permutation[j])
        required_parity ^= (length - cycles) & 1
    return required_parity


@njit(cache=False)
def _cube555_structural_lower_bound_numba(
    state: np.ndarray,
    corner_dist: np.ndarray,
    corner_ori_code_to_idx: np.ndarray,
    corner_slot0_positions: np.ndarray,
    corner_piece_to_cubie: np.ndarray,
    corner_orientation_lookup: np.ndarray,
    inner_cut_positions: np.ndarray,
    inner_cut_lengths: np.ndarray,
    inner_goal_piece_mask: np.ndarray,
    inner_dual_cut_indices: np.ndarray,
    inner_dual_weights_x4: np.ndarray,
    required_inner_parity: int,
    corner_orientation_count: int,
) -> int:
    # Exact full-corner quotient distance.  Reading one canonical facelet per
    # physical corner slot is sufficient to identify both cubie permutation
    # and orientation.
    permutation = np.empty(8, dtype=np.int16)
    orientation = np.empty(8, dtype=np.int16)
    for slot in range(8):
        position = int(corner_slot0_positions[slot])
        piece = int(state[position])
        cubie = int(corner_piece_to_cubie[piece])
        ori = int(corner_orientation_lookup[slot, piece])
        if cubie < 0 or ori < 0:
            return 32767
        permutation[slot] = np.int16(cubie)
        orientation[slot] = np.int16(ori)

    factors = (5040, 720, 120, 24, 6, 2, 1, 1)
    rank = 0
    for i in range(8):
        smaller = 0
        for j in range(i + 1, 8):
            if permutation[j] < permutation[i]:
                smaller += 1
        rank += smaller * factors[i]

    ori_code = 0
    multiplier = 1
    for slot in range(8):
        ori_code += int(orientation[slot]) * multiplier
        multiplier *= 3
    ori_idx = int(corner_ori_code_to_idx[ori_code])
    if ori_idx < 0:
        return 32767
    corner = int(corner_dist[rank * corner_orientation_count + ori_idx])
    if corner == 255:
        return 32767

    # Fast inner-move lower bound from 16 sparse dual certificates over 11
    # outer-invariant slice cuts.
    n_cuts = inner_cut_lengths.shape[0]
    deficits = np.empty(n_cuts, dtype=np.int16)
    for cut_idx in range(n_cuts):
        length = int(inner_cut_lengths[cut_idx])
        inside = 0
        for j in range(length):
            position = int(inner_cut_positions[cut_idx, j])
            piece = int(state[position])
            if inner_goal_piece_mask[cut_idx, piece] != 0:
                inside += 1
        deficits[cut_idx] = np.int16(length - inside)

    best_scaled = 0
    for dual_idx in range(inner_dual_cut_indices.shape[0]):
        cut_a = int(inner_dual_cut_indices[dual_idx, 0])
        cut_b = int(inner_dual_cut_indices[dual_idx, 1])
        weight_a = int(inner_dual_weights_x4[dual_idx, 0])
        weight_b = int(inner_dual_weights_x4[dual_idx, 1])
        score = (
            weight_a * int(deficits[cut_a])
            + weight_b * int(deficits[cut_b])
        )
        if score > best_scaled:
            best_scaled = score
    inner = (best_scaled + 3) // 4

    # The exact-search engine carries this one-bit invariant down the DFS:
    # inner moves toggle it, outer moves do not.  This avoids recomputing a
    # 24-position permutation parity at every node.
    if (inner & 1) != (required_inner_parity & 1):
        inner += 1
    return corner + inner


@njit(cache=False)
def lower_bound_numba(
    kernel_kind: int,
    state: np.ndarray,
    piece_distance: np.ndarray,
    piece_orbit: np.ndarray,
    subset_masks: np.ndarray,
    subset_capacities: np.ndarray,
    n_orbits: int,
    corner_dist: np.ndarray,
    corner_ori_code_to_idx: np.ndarray,
    corner_slot0_positions: np.ndarray,
    corner_piece_to_cubie: np.ndarray,
    corner_orientation_lookup: np.ndarray,
    inner_cut_positions: np.ndarray,
    inner_cut_lengths: np.ndarray,
    inner_goal_piece_mask: np.ndarray,
    inner_dual_cut_indices: np.ndarray,
    inner_dual_weights_x4: np.ndarray,
    required_inner_parity: int,
    corner_orientation_count: int,
) -> int:
    if kernel_kind == KERNEL_PANCAKE_GAP:
        return gap_count_numba(state)
    if kernel_kind == KERNEL_ORBIT_SUBSETS:
        return _orbit_subset_lower_bound_numba(
            state,
            piece_distance,
            piece_orbit,
            subset_masks,
            subset_capacities,
            n_orbits,
        )
    if kernel_kind == KERNEL_CUBE555_STRUCTURAL:
        return _cube555_structural_lower_bound_numba(
            state,
            corner_dist,
            corner_ori_code_to_idx,
            corner_slot0_positions,
            corner_piece_to_cubie,
            corner_orientation_lookup,
            inner_cut_positions,
            inner_cut_lengths,
            inner_goal_piece_mask,
            inner_dual_cut_indices,
            inner_dual_weights_x4,
            required_inner_parity,
            corner_orientation_count,
        )
    return 32767


@njit(cache=False)
def apply_pancake_move_inplace_numba(
    state: np.ndarray,
    pos: np.ndarray,
    move_code: int,
) -> None:
    left = 0
    right = move_code + 1
    while left < right:
        left_val = int(state[left])
        right_val = int(state[right])
        state[left] = np.int16(right_val)
        state[right] = np.int16(left_val)
        pos[left_val] = np.int16(right)
        pos[right_val] = np.int16(left)
        left += 1
        right -= 1


@njit(cache=False)
def apply_permutation_move_numba(
    parent: np.ndarray,
    child: np.ndarray,
    move_indices: np.ndarray,
    move_code: int,
) -> None:
    for idx in range(parent.shape[0]):
        child[idx] = parent[int(move_indices[move_code, idx])]


@njit(cache=False)
def fill_candidate_moves_numba(
    kernel_kind: int,
    state: np.ndarray,
    pos: np.ndarray,
    slack: int,
    previous_move: int,
    inverse_moves: np.ndarray,
    n_moves: int,
    out_moves: np.ndarray,
) -> int:
    """Fill exact-search candidates in a puzzle-specific efficient order."""
    count = 0

    if kernel_kind == KERNEL_PANCAKE_GAP:
        # Preserve the old fast path: at zero slack generate only the at-most
        # three gap-decreasing prefix reversals without scanning all moves.
        count = _pancake_decreasing_candidates_numba(state, pos, out_moves)

        write = 0
        for idx in range(count):
            move_code = int(out_moves[idx])
            if previous_move >= 0 and move_code == previous_move:
                continue
            out_moves[write] = np.int16(move_code)
            write += 1
        count = write

        if slack <= 0:
            return count

        # Same order/cost model as the old solver:
        # decreasing cost 0, neutral cost 1, increasing cost 2.
        for wanted_delta in range(0, 2):
            cost = wanted_delta + 1
            if slack < cost:
                continue
            for move_code in range(n_moves):
                if previous_move >= 0 and move_code == previous_move:
                    continue
                if _delta_gap_for_move_numba(state, move_code) != wanted_delta:
                    continue
                out_moves[count] = np.int16(move_code)
                count += 1
        return count

    # Generic permutation kernel. Child LB is checked on entry by the shared
    # search engine. Immediate inverse moves are never needed on a shortest path.
    for move_code in range(n_moves):
        if previous_move >= 0 and move_code == int(inverse_moves[previous_move]):
            continue
        out_moves[count] = np.int16(move_code)
        count += 1
    return count


def exact_kernel_lower_bound(state: np.ndarray, kernel: ExactKernel) -> int:
    """Evaluate one kernel lower bound outside the search loop."""
    state_i16 = _as_int16_c(np.asarray(state))
    required_inner_parity = 0
    if int(kernel.kind) == KERNEL_CUBE555_STRUCTURAL:
        required_inner_parity = _cube555_inner_parity_numba(
            state_i16,
            kernel.inner_parity_orbit_positions,
            kernel.inner_parity_orbit_lengths,
            kernel.inner_parity_goal_local_by_piece,
        )
        if required_inner_parity < 0:
            return 32767
    return int(
        lower_bound_numba(
            int(kernel.kind),
            state_i16,
            kernel.piece_distance,
            kernel.piece_orbit,
            kernel.subset_masks,
            kernel.subset_capacities,
            int(kernel.n_orbits),
            kernel.corner_dist,
            kernel.corner_ori_code_to_idx,
            kernel.corner_slot0_positions,
            kernel.corner_piece_to_cubie,
            kernel.corner_orientation_lookup,
            kernel.inner_cut_positions,
            kernel.inner_cut_lengths,
            kernel.inner_goal_piece_mask,
            kernel.inner_dual_cut_indices,
            kernel.inner_dual_weights_x4,
            int(required_inner_parity),
            int(kernel.corner_orientation_count),
        )
    )
