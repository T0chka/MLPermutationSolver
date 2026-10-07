"""Shared Numba exact-search engine for permutation puzzles."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numba import njit

from src.solvers.exact_kernels import (
    KERNEL_CUBE555_STRUCTURAL,
    KERNEL_PANCAKE_GAP,
    _cube555_inner_parity_numba,
    apply_pancake_move_inplace_numba,
    apply_permutation_move_numba,
    build_pos_numba,
    fill_candidate_moves_numba,
    is_solved_numba,
    lower_bound_numba,
)


@njit(cache=False)
def _pack_state_key_numba(
    state: np.ndarray,
    bits_per_value: int,
    key_words: np.ndarray,
) -> None:
    for word_idx in range(key_words.shape[0]):
        key_words[word_idx] = np.uint64(0)

    for idx in range(state.shape[0]):
        value = np.uint64(state[idx])
        bit_pos = idx * bits_per_value
        word_idx = bit_pos >> 6
        bit_shift = bit_pos & 63
        key_words[word_idx] |= value << np.uint64(bit_shift)
        if bit_shift + bits_per_value > 64:
            spill_shift = np.uint64(64 - bit_shift)
            key_words[word_idx + 1] |= value >> spill_shift


@njit(cache=False)
def _hash_key_words_numba(key_words: np.ndarray) -> np.uint64:
    value = np.uint64(0x9E3779B97F4A7C15)
    for idx in range(key_words.shape[0]):
        value ^= key_words[idx] + np.uint64(0x9E3779B97F4A7C15)
        value ^= value >> np.uint64(30)
        value *= np.uint64(0xBF58476D1CE4E5B9)
        value ^= value >> np.uint64(27)
        value *= np.uint64(0x94D049BB133111EB)
        value ^= value >> np.uint64(31)
    return value


@njit(cache=False)
def _tt_find_slot_numba(
    key_words: np.ndarray,
    tt_used: np.ndarray,
    tt_keys: np.ndarray,
    tt_best_remaining: np.ndarray,
    tt_mask: int,
) -> Tuple[bool, int, int]:
    slot = np.int64(_hash_key_words_numba(key_words) & np.uint64(tt_mask))
    while True:
        if tt_used[slot] == 0:
            return False, int(slot), -1

        is_match = True
        for word_idx in range(key_words.shape[0]):
            if tt_keys[slot, word_idx] != key_words[word_idx]:
                is_match = False
                break
        if is_match:
            return True, int(slot), int(tt_best_remaining[slot])
        slot = np.int64((slot + 1) & tt_mask)


@njit(cache=False)
def verify_shorter_paths_numba(
    start_state: np.ndarray,
    first_length: int,
    last_length: int,
    kernel_kind: int,
    move_indices: np.ndarray,
    inverse_moves: np.ndarray,
    solved_state: np.ndarray,
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
    inner_parity_orbit_positions: np.ndarray,
    inner_parity_orbit_lengths: np.ndarray,
    inner_parity_goal_local_by_piece: np.ndarray,
    inner_parity_move_toggle: np.ndarray,
    corner_orientation_count: int,
    tt_capacity: int,
    tt_slots: int,
    bits_per_value: int,
) -> Tuple[bool, np.ndarray, int, int, int, int, int, int, bool]:
    """Exhaust every solution length in ``[first_length, last_length]``.

    Lengths are searched in increasing order, so a returned path is globally
    shortest. If no path is returned, every shorter length in the range has
    been excluded exactly.
    """
    n = start_state.shape[0]
    n_moves = move_indices.shape[0]
    max_depth = last_length
    n_key_words = (n * bits_per_value + 63) // 64

    path = np.empty(max_depth, dtype=np.int16)
    state_stack = np.empty((max_depth + 1, n), dtype=np.int16)
    # Only pancake uses inverse positions, but one shallow shared stack keeps
    # the traversal itself identical for every kernel.
    pos_stack = np.empty((max_depth + 1, n), dtype=np.int16)
    candidate_moves = np.empty((max_depth + 1, n_moves), dtype=np.int16)
    candidate_count = np.zeros(max_depth + 1, dtype=np.int16)
    next_index = np.zeros(max_depth + 1, dtype=np.int16)
    entered = np.zeros(max_depth + 1, dtype=np.uint8)
    previous_move = np.full(max_depth + 1, -1, dtype=np.int16)
    inner_parity_stack = np.zeros(max_depth + 1, dtype=np.uint8)

    key_words = np.zeros(n_key_words, dtype=np.uint64)
    tt_used = np.zeros(tt_slots, dtype=np.uint8)
    tt_keys = np.zeros((tt_slots, n_key_words), dtype=np.uint64)
    tt_best_remaining = np.full(tt_slots, -1, dtype=np.int16)
    tt_mask = tt_slots - 1
    tt_size = 0
    tt_overflow = False

    total_nodes = 0
    pruned_by_lower_bound = 0
    pruned_by_transposition = 0
    start_pos = build_pos_numba(start_state)
    root_inner_parity = 0
    if kernel_kind == KERNEL_CUBE555_STRUCTURAL:
        root_inner_parity = _cube555_inner_parity_numba(
            start_state,
            inner_parity_orbit_positions,
            inner_parity_orbit_lengths,
            inner_parity_goal_local_by_piece,
        )
        if root_inner_parity < 0:
            root_inner_parity = 0

    for target_length in range(first_length, last_length + 1):
        state_stack[0, :] = start_state
        pos_stack[0, :] = start_pos
        entered[0] = 0
        next_index[0] = 0
        previous_move[0] = -1
        inner_parity_stack[0] = np.uint8(root_inner_parity)
        depth = 0

        while depth >= 0:
            if entered[depth] == 0:
                total_nodes += 1
                current_state = state_stack[depth]

                if is_solved_numba(current_state, solved_state):
                    return (
                        True,
                        path,
                        depth,
                        target_length,
                        total_nodes,
                        pruned_by_lower_bound,
                        pruned_by_transposition,
                        tt_size,
                        tt_overflow,
                    )

                remaining = target_length - depth
                lower_bound = lower_bound_numba(
                    kernel_kind,
                    current_state,
                    piece_distance,
                    piece_orbit,
                    subset_masks,
                    subset_capacities,
                    n_orbits,
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
                    int(inner_parity_stack[depth]),
                    corner_orientation_count,
                )
                if lower_bound > remaining:
                    pruned_by_lower_bound += 1
                    depth -= 1
                    continue

                if remaining <= 0:
                    depth -= 1
                    continue

                _pack_state_key_numba(current_state, bits_per_value, key_words)
                seen, slot, seen_remaining = _tt_find_slot_numba(
                    key_words,
                    tt_used,
                    tt_keys,
                    tt_best_remaining,
                    tt_mask,
                )
                if seen and seen_remaining >= remaining:
                    pruned_by_transposition += 1
                    depth -= 1
                    continue

                if seen:
                    tt_best_remaining[slot] = np.int16(remaining)
                elif tt_size < tt_capacity:
                    tt_used[slot] = 1
                    for word_idx in range(n_key_words):
                        tt_keys[slot, word_idx] = key_words[word_idx]
                    tt_best_remaining[slot] = np.int16(remaining)
                    tt_size += 1
                else:
                    # TT exhaustion changes only speed, never exactness.
                    tt_overflow = True

                slack = remaining - lower_bound
                count = fill_candidate_moves_numba(
                    kernel_kind,
                    current_state,
                    pos_stack[depth],
                    slack,
                    int(previous_move[depth]),
                    inverse_moves,
                    n_moves,
                    candidate_moves[depth],
                )
                candidate_count[depth] = np.int16(count)
                next_index[depth] = 0
                entered[depth] = 1

            idx = int(next_index[depth])
            if idx >= int(candidate_count[depth]):
                entered[depth] = 0
                depth -= 1
                continue

            move_code = int(candidate_moves[depth, idx])
            next_index[depth] = np.int16(idx + 1)

            state_stack[depth + 1, :] = state_stack[depth, :]
            pos_stack[depth + 1, :] = pos_stack[depth, :]
            if kernel_kind == KERNEL_PANCAKE_GAP:
                apply_pancake_move_inplace_numba(
                    state_stack[depth + 1],
                    pos_stack[depth + 1],
                    move_code,
                )
            else:
                apply_permutation_move_numba(
                    state_stack[depth],
                    state_stack[depth + 1],
                    move_indices,
                    move_code,
                )

            inner_parity_stack[depth + 1] = inner_parity_stack[depth]
            if inner_parity_move_toggle.shape[0] > 0:
                inner_parity_stack[depth + 1] ^= inner_parity_move_toggle[move_code]

            path[depth] = np.int16(move_code)
            previous_move[depth + 1] = np.int16(move_code)
            entered[depth + 1] = 0
            next_index[depth + 1] = 0
            depth += 1

    return (
        False,
        path,
        -1,
        -1,
        total_nodes,
        pruned_by_lower_bound,
        pruned_by_transposition,
        tt_size,
        tt_overflow,
    )
