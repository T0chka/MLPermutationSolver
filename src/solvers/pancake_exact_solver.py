"""
The solver finds and validates the candidate solution and proves its optimality by
excluding all admissible shorter solutions down to the gap lower bound.

Proof procedure:
1. Validate the candidate solution.
2. If its length equals the gap count, certify it as optimal immediately.
3. Otherwise, search shorter solutions by increasing slack over the gap bound.
4. If all shorter lengths are excluded, certify it as optimal.

The exact search runs in the expanded state space (perm, slack_left), where the
move cost is 0 for a gap-decreasing move, 1 for a neutral move, and 2 for a
gap-increasing move.
"""

from time import time
from typing import Any, Optional, Tuple

import numpy as np
from numba import njit

import torch

from src.models.base_model import BaseModel
from src.puzzles import PuzzleSpec
from src.solvers.base_solver import BaseSolver


@njit(cache=False)  # cache=True + torch in process causes segfault on 2nd call

def _gap_count_numba(state: np.ndarray) -> int:
    """Return plate gap count."""
    n = state.shape[0]
    total = 0
    for idx in range(n - 1):
        if abs(state[idx] - state[idx + 1]) != 1:
            total += 1
    if abs(state[n - 1] - n) != 1:
        total += 1
    return total


@njit(cache=False)
def _is_solved_numba(state: np.ndarray) -> bool:
    """Return True for identity."""
    for idx in range(state.shape[0]):
        if state[idx] != idx:
            return False
    return True


@njit(cache=False)
def _build_pos_numba(state: np.ndarray) -> np.ndarray:
    """Build inverse positions."""
    pos = np.empty(state.shape[0], dtype=np.int16)
    for idx in range(state.shape[0]):
        pos[state[idx]] = idx
    return pos


@njit(cache=False)
def _apply_move_inplace_numba(
    state: np.ndarray,
    pos: np.ndarray,
    move_code: int,
) -> None:
    """Apply one prefix reversal in place."""
    left = 0
    right = move_code + 1
    while left < right:
        left_val = state[left]
        right_val = state[right]
        state[left] = right_val
        state[right] = left_val
        pos[left_val] = right
        pos[right_val] = left
        left += 1
        right -= 1


@njit(cache=False)
def _delta_gap_for_move_numba(state: np.ndarray, move_code: int) -> int:
    """Return gap delta for one move."""
    n = state.shape[0]
    k = move_code + 2
    prefix_last = state[k - 1]
    suffix_val = n if k == n else state[k]
    old_cut = 1 if abs(prefix_last - suffix_val) != 1 else 0
    new_cut = 1 if abs(state[0] - suffix_val) != 1 else 0
    return new_cut - old_cut


@njit(cache=False)
def _decreasing_candidates_numba(
    state: np.ndarray,
    pos: np.ndarray,
) -> Tuple[int, int, int, int]:
    """Return all decreasing moves."""
    n = state.shape[0]
    top = state[0]
    count = 0
    move0 = -1
    move1 = -1
    move2 = -1

    if top > 0:
        idx = pos[top - 1]
        if idx >= 2 and abs(state[idx - 1] - state[idx]) != 1:
            move0 = idx - 2
            count = 1

    if top + 1 < n:
        idx = pos[top + 1]
        if idx >= 2 and abs(state[idx - 1] - state[idx]) != 1:
            move_code = idx - 2
            if move_code != move0:
                if count == 0:
                    move0 = move_code
                else:
                    move1 = move_code
                count += 1

    if top == n - 1:
        move_code = n - 2
        if abs(state[n - 1] - n) != 1:
            if move_code != move0 and move_code != move1:
                if count == 0:
                    move0 = move_code
                elif count == 1:
                    move1 = move_code
                else:
                    move2 = move_code
                count += 1

    return count, move0, move1, move2


@njit(cache=False)
def _pack_state_key_numba(
    state: np.ndarray,
    bits_per_value: int,
    key_words: np.ndarray,
) -> None:
    """Pack a state into fixed 64-bit words."""
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
    """Hash packed state words."""
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
    tt_best_slack: np.ndarray,
    tt_mask: int,
) -> Tuple[bool, int, int]:
    """Find TT slot for a packed key."""
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
            return True, int(slot), int(tt_best_slack[slot])

        slot = np.int64((slot + 1) & tt_mask)


@njit(cache=False)
def _verify_shorter_paths_numba(
    start_state: np.ndarray,
    max_shorter_slack: int,
    tt_capacity: int,
    tt_slots: int,
    bits_per_value: int,
) -> Tuple[bool, np.ndarray, int, int, int, int, bool]:
    """Return existence of any shorter path within the slack range."""
    n = start_state.shape[0]
    gap0 = _gap_count_numba(start_state)
    max_depth = gap0 + max_shorter_slack
    n_moves = n - 1
    n_key_words = (n * bits_per_value + 63) // 64

    path = np.empty(max_depth, dtype=np.int16)
    start_pos = _build_pos_numba(start_state)
    state_stack = np.empty((max_depth + 1, n), dtype=np.int16)
    pos_stack = np.empty((max_depth + 1, n), dtype=np.int16)
    stage_stack = np.zeros(max_depth + 1, dtype=np.int8)
    next_index_stack = np.zeros(max_depth + 1, dtype=np.int16)
    dec_count_stack = np.zeros(max_depth + 1, dtype=np.int8)
    dec_moves_stack = np.full((max_depth + 1, 3), -1, dtype=np.int16)
    slack_stack = np.zeros(max_depth + 1, dtype=np.int16)
    prev_move_stack = np.full(max_depth + 1, -1, dtype=np.int16)
    key_words = np.zeros(n_key_words, dtype=np.uint64)
    tt_used = np.zeros(tt_slots, dtype=np.uint8)
    tt_keys = np.zeros((tt_slots, n_key_words), dtype=np.uint64)
    tt_best_slack = np.full(tt_slots, -1, dtype=np.int16)

    tt_mask = tt_slots - 1
    tt_size = 0
    tt_overflow = False
    total_nodes = 0
    total_pruned = 0

    for root_slack in range(max_shorter_slack + 1):
        state_stack[0, :] = start_state
        pos_stack[0, :] = start_pos
        stage_stack[0] = 0
        next_index_stack[0] = 0
        slack_stack[0] = root_slack
        prev_move_stack[0] = -1
        depth = 0

        while True:
            stage = stage_stack[depth]

            if stage == 0:
                total_nodes += 1
                current_state = state_stack[depth]

                if _is_solved_numba(current_state):
                    return (
                        True,
                        path,
                        depth,
                        total_nodes,
                        total_pruned,
                        tt_size,
                        tt_overflow,
                    )

                _pack_state_key_numba(
                    current_state,
                    bits_per_value,
                    key_words,
                )
                seen, slot, seen_slack = _tt_find_slot_numba(
                    key_words,
                    tt_used,
                    tt_keys,
                    tt_best_slack,
                    tt_mask,
                )
                slack_left = int(slack_stack[depth])

                if seen and seen_slack >= slack_left:
                    total_pruned += 1
                    stage_stack[depth] = 4
                    continue

                if seen:
                    tt_best_slack[slot] = np.int16(slack_left)
                elif tt_size < tt_capacity:
                    tt_used[slot] = 1
                    for word_idx in range(n_key_words):
                        tt_keys[slot, word_idx] = key_words[word_idx]
                    tt_best_slack[slot] = np.int16(slack_left)
                    tt_size += 1
                else:
                    tt_overflow = True

                count, move0, move1, move2 = _decreasing_candidates_numba(
                    current_state,
                    pos_stack[depth],
                )
                dec_count_stack[depth] = count
                dec_moves_stack[depth, 0] = move0
                dec_moves_stack[depth, 1] = move1
                dec_moves_stack[depth, 2] = move2
                next_index_stack[depth] = 0
                stage_stack[depth] = 1
                continue

            if stage == 1:
                next_index = int(next_index_stack[depth])
                dec_count = int(dec_count_stack[depth])
                moved = False

                while next_index < dec_count:
                    move_code = int(dec_moves_stack[depth, next_index])
                    next_index += 1
                    if move_code == int(prev_move_stack[depth]):
                        continue

                    next_index_stack[depth] = np.int16(next_index)
                    state_stack[depth + 1, :] = state_stack[depth, :]
                    pos_stack[depth + 1, :] = pos_stack[depth, :]
                    _apply_move_inplace_numba(
                        state_stack[depth + 1],
                        pos_stack[depth + 1],
                        move_code,
                    )
                    path[depth] = move_code
                    slack_stack[depth + 1] = slack_stack[depth]
                    prev_move_stack[depth + 1] = np.int16(move_code)
                    stage_stack[depth + 1] = 0
                    next_index_stack[depth + 1] = 0
                    depth += 1
                    moved = True
                    break

                if moved:
                    continue

                if slack_stack[depth] == 0:
                    stage_stack[depth] = 4
                else:
                    stage_stack[depth] = 2
                    next_index_stack[depth] = 0
                continue

            if stage == 2:
                next_index = int(next_index_stack[depth])
                moved = False

                while next_index < n_moves:
                    move_code = next_index
                    next_index += 1
                    if move_code == int(prev_move_stack[depth]):
                        continue
                    if _delta_gap_for_move_numba(
                        state_stack[depth],
                        move_code,
                    ) != 0:
                        continue

                    next_index_stack[depth] = np.int16(next_index)
                    state_stack[depth + 1, :] = state_stack[depth, :]
                    pos_stack[depth + 1, :] = pos_stack[depth, :]
                    _apply_move_inplace_numba(
                        state_stack[depth + 1],
                        pos_stack[depth + 1],
                        move_code,
                    )
                    path[depth] = move_code
                    slack_stack[depth + 1] = np.int16(slack_stack[depth] - 1)
                    prev_move_stack[depth + 1] = np.int16(move_code)
                    stage_stack[depth + 1] = 0
                    next_index_stack[depth + 1] = 0
                    depth += 1
                    moved = True
                    break

                if moved:
                    continue

                if slack_stack[depth] == 1:
                    stage_stack[depth] = 4
                else:
                    stage_stack[depth] = 3
                    next_index_stack[depth] = 0
                continue

            if stage == 3:
                next_index = int(next_index_stack[depth])
                moved = False

                while next_index < n_moves:
                    move_code = next_index
                    next_index += 1
                    if move_code == int(prev_move_stack[depth]):
                        continue
                    if _delta_gap_for_move_numba(
                        state_stack[depth],
                        move_code,
                    ) != 1:
                        continue

                    next_index_stack[depth] = np.int16(next_index)
                    state_stack[depth + 1, :] = state_stack[depth, :]
                    pos_stack[depth + 1, :] = pos_stack[depth, :]
                    _apply_move_inplace_numba(
                        state_stack[depth + 1],
                        pos_stack[depth + 1],
                        move_code,
                    )
                    path[depth] = move_code
                    slack_stack[depth + 1] = np.int16(slack_stack[depth] - 2)
                    prev_move_stack[depth + 1] = np.int16(move_code)
                    stage_stack[depth + 1] = 0
                    next_index_stack[depth + 1] = 0
                    depth += 1
                    moved = True
                    break

                if moved:
                    continue

                stage_stack[depth] = 4
                continue

            if depth == 0:
                break

            depth -= 1

    return False, path, -1, total_nodes, total_pruned, tt_size, tt_overflow


class PancakeExactSolver(BaseSolver):
    """
    The solver finds and certifies optimal pancake solutions by validating an
    incumbent and excluding all admissible shorter solutions down to the gap
    lower bound.

    The exact search is formulated over the expanded state space
    (perm, slack_left), where slack_left is the remaining budget above the
    current gap lower bound. A gap-decreasing move consumes no slack, a neutral
    move consumes one slack unit, and a gap-increasing move consumes two.
    """

    def __init__(
        self,
        *,
        puzzle_spec: PuzzleSpec,
        device: torch.device,
        adapter: Any,
        incumbent_solver: BaseSolver,
        exact_verify_margin: int = 2,
        exact_tt_capacity: int = 100_000_000,
        verbose: int = 0,
    ) -> None:
        """Store exact-solver dependencies."""
        if puzzle_spec.puzzle_type != "pancake":
            raise ValueError(
                "PancakeExactSolver only supports puzzle_type='pancake'; "
                f"got {puzzle_spec.puzzle_type!r}"
            )
        if exact_verify_margin < 0:
            raise ValueError("exact_verify_margin must be non-negative.")
        if exact_tt_capacity <= 0:
            raise ValueError("exact_tt_capacity must be positive.")
        if puzzle_spec.state_size > np.iinfo(np.int16).max:
            raise ValueError(
                "PancakeExactSolver requires state_size <= 32767."
            )
        super().__init__(puzzle_spec, device, model=None, verbose=verbose)
        self.adapter = adapter
        self.incumbent_solver = incumbent_solver
        self.exact_progress = ""
        self.exact_verify_margin = exact_verify_margin
        self.exact_tt_capacity = exact_tt_capacity
        self._exact_tt_slots = _next_power_of_two(exact_tt_capacity * 2)
        self._bits_per_value = max(1, (puzzle_spec.state_size - 1).bit_length())
        self._name_to_code = {
            name: code for code, name in enumerate(self.move_names)
        }

    def reset(self) -> None:
        """Reset search statistics."""
        self.search_stats.update({
            "incumbent_found": False,
            "incumbent_len": -1,
            "incumbent_solution": "",
            "incumbent_time": 0.0,
            "incumbent_step": -1,
            "incumbent_meet_source": "",
            "exact_time": 0.0,
            "gap0": -1,
            "exact_attempted": False,
            "exact_verify_margin": self.exact_verify_margin,
            "exact_tt_capacity": self.exact_tt_capacity,
            "exact_tt_size": 0,
            "exact_tt_overflow": False,
            "exact_checked_from": -1,
            "exact_checked_to": -1,
            "exact_nodes_expanded": 0,
            "exact_pruned_by_gap": 0,
            "exact_pruned_by_transposition": 0,
            "exact_found": False,
            "exact_found_len": -1,
            "exact_status": "",
            "path_found": False,
        })

    def _parse_solution_codes(self, solution: str) -> list[int]:
        """Parse move names into move codes."""
        if not solution:
            return []
        out: list[int] = []
        for part in solution.split("."):
            token = part.strip()
            if not token:
                continue
            code = self._name_to_code.get(token)
            if code is None:
                return []
            out.append(code)
        return out

    def _validate_solution(
        self,
        start_state: torch.Tensor,
        solution: str,
        expected_len: int,
    ) -> None:
        """Validate an incumbent solution."""
        codes = self._parse_solution_codes(solution)
        if len(codes) != expected_len:
            raise ValueError(
                f"Incumbent solution length mismatch: expected {expected_len}, "
                f"got {len(codes)}."
            )
        final_state = self._apply_moves(start_state, codes)
        if not torch.equal(final_state, self.solved_state):
            raise ValueError("Incumbent solution does not solve the puzzle.")

    def _codes_to_solution(self, path: np.ndarray, path_len: int) -> str:
        """Convert move codes to a solution string."""
        return ".".join(self.move_names[path[idx]] for idx in range(path_len))

    def solve(
        self,
        start_state: torch.Tensor,
        model: BaseModel,
        *,
        initial_solution: Optional[str] = None,
    ) -> Tuple[bool, int, str]:
        """Validate incumbent and certify shorter lengths when requested."""
        self.reset()
        self.exact_progress = ""
        self.model = model
        stats = self.search_stats

        if initial_solution is not None:
            best_solution = initial_solution.strip()
            if not best_solution:
                raise ValueError("initial_solution is empty in verify_only mode.")
            best_len = best_solution.count(".") + 1
            found = True
            stats["incumbent_time"] = 0.0
            stats["incumbent_found"] = True
            stats["incumbent_len"] = best_len
            stats["incumbent_solution"] = best_solution
            stats["incumbent_step"] = -1
            stats["incumbent_meet_source"] = "verify_only"
        else:
            t0 = time()
            found, best_len, best_solution = self.incumbent_solver.solve(
                start_state,
                model,
            )
            stats["incumbent_time"] = time() - t0
            incumbent_stats = self.incumbent_solver.search_stats
            stats["incumbent_found"] = found
            stats["incumbent_len"] = best_len
            stats["incumbent_solution"] = best_solution or ""
            stats["incumbent_step"] = incumbent_stats.get(
                "intersect_at_step",
                -1,
            )
            stats["incumbent_meet_source"] = incumbent_stats.get(
                "meet_source",
                "",
            )

        if not found:
            stats["exact_status"] = "skipped_no_incumbent"
            return False, best_len, best_solution or ""

        if best_solution:
            self._validate_solution(start_state, best_solution, best_len)

        start_state_cpu = start_state.detach().cpu().contiguous()
        start_np = np.array(
            start_state_cpu.numpy(),
            dtype=np.int16,
            copy=True,
            order="C",
        )
        gap0 = _gap_count_numba(start_np)
        stats["gap0"] = gap0

        if best_len < gap0:
            raise ValueError(
                f"Incumbent length {best_len} is below gap lower bound {gap0}."
            )

        if best_len == gap0:
            stats["exact_status"] = "incumbent_at_lower_bound"
            stats["path_found"] = True
            return True, best_len, best_solution

        if best_len > gap0 + self.exact_verify_margin:
            stats["exact_status"] = "skipped_margin"
            stats["path_found"] = True
            return True, best_len, best_solution

        stats["exact_attempted"] = True
        stats["exact_checked_from"] = gap0
        stats["exact_checked_to"] = best_len - 1
        exact_start = time()
        max_shorter_slack = best_len - gap0 - 1
        (
            found_shorter,
            path,
            path_len,
            nodes,
            pruned,
            tt_size,
            overflowed,
        ) = _verify_shorter_paths_numba(
            start_np,
            max_shorter_slack,
            self.exact_tt_capacity,
            self._exact_tt_slots,
            self._bits_per_value,
        )
        stats["exact_time"] = time() - exact_start
        stats["exact_nodes_expanded"] = nodes
        stats["exact_pruned_by_transposition"] = pruned
        stats["exact_tt_size"] = tt_size
        stats["exact_tt_overflow"] = overflowed

        if self.verbose > 0 and max_shorter_slack > 0 and nodes > 0:
            elapsed = time() - exact_start
            self.exact_progress = (
                f"| ex {gap0}..{best_len - 1} {max_shorter_slack + 1} steps "
                f"{nodes:,}n {elapsed:.1f}s"
            )

        if found_shorter:
            stats["exact_found"] = True
            stats["exact_found_len"] = path_len
            stats["exact_status"] = "found_exact_improvement"
            stats["path_found"] = True
            return True, path_len, self._codes_to_solution(path, path_len)

        stats["exact_status"] = "proved_optimal_in_range"
        stats["path_found"] = True
        return True, best_len, best_solution


def _next_power_of_two(value: int) -> int:
    """Return the next power of two."""
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()
