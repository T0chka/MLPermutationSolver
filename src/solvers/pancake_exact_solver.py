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


def _state_key(state: np.ndarray) -> int | tuple[int, ...]:
    """Pack a state into 64-bit words for TT lookup."""
    word = 0
    shift = 0
    words: list[int] = []

    for value in state:
        word |= int(value) << shift
        shift += 16
        if shift == 64:
            words.append(word)
            word = 0
            shift = 0

    if shift != 0:
        words.append(word)

    if len(words) == 1:
        return words[0]
    return tuple(words)


def _verify_with_slack(
    start_state: np.ndarray,
    start_pos: np.ndarray,
    work_state: np.ndarray,
    work_pos: np.ndarray,
    path: np.ndarray,
    slack: int,
    tt_capacity: int,
    best_slack_by_state: dict[object, int],
    tt_overflow: bool,
) -> Tuple[bool, int, int, int, bool]:
    """Return existence of a path within the given slack budget."""
    np.copyto(work_state, start_state)
    np.copyto(work_pos, start_pos)
    nodes = 0
    pruned = 0

    def dfs(depth: int, slack_left: int, prev_move: int) -> int:
        nonlocal nodes, pruned, tt_overflow
        nodes += 1

        if _is_solved_numba(work_state):
            return depth

        state_key = _state_key(work_state)
        seen_slack = best_slack_by_state.get(state_key)
        if seen_slack is not None and seen_slack >= slack_left:
            pruned += 1
            return -1

        if seen_slack is None:
            if len(best_slack_by_state) < tt_capacity:
                best_slack_by_state[state_key] = slack_left
            else:
                tt_overflow = True
        else:
            best_slack_by_state[state_key] = slack_left

        count, move0, move1, move2 = _decreasing_candidates_numba(
            work_state,
            work_pos,
        )

        if count >= 1 and move0 != prev_move:
            _apply_move_inplace_numba(work_state, work_pos, move0)
            path[depth] = move0
            found_depth = dfs(depth + 1, slack_left, move0)
            _apply_move_inplace_numba(work_state, work_pos, move0)
            if found_depth >= 0:
                return found_depth

        if count >= 2 and move1 != prev_move:
            _apply_move_inplace_numba(work_state, work_pos, move1)
            path[depth] = move1
            found_depth = dfs(depth + 1, slack_left, move1)
            _apply_move_inplace_numba(work_state, work_pos, move1)
            if found_depth >= 0:
                return found_depth

        if count >= 3 and move2 != prev_move:
            _apply_move_inplace_numba(work_state, work_pos, move2)
            path[depth] = move2
            found_depth = dfs(depth + 1, slack_left, move2)
            _apply_move_inplace_numba(work_state, work_pos, move2)
            if found_depth >= 0:
                return found_depth

        if slack_left == 0:
            return -1

        n_moves = work_state.shape[0] - 1
        for move_code in range(n_moves):
            if move_code == prev_move:
                continue
            if _delta_gap_for_move_numba(work_state, move_code) != 0:
                continue

            _apply_move_inplace_numba(work_state, work_pos, move_code)
            path[depth] = move_code
            found_depth = dfs(depth + 1, slack_left - 1, move_code)
            _apply_move_inplace_numba(work_state, work_pos, move_code)
            if found_depth >= 0:
                return found_depth

        if slack_left == 1:
            return -1

        for move_code in range(n_moves):
            if move_code == prev_move:
                continue
            if _delta_gap_for_move_numba(work_state, move_code) != 1:
                continue

            _apply_move_inplace_numba(work_state, work_pos, move_code)
            path[depth] = move_code
            found_depth = dfs(depth + 1, slack_left - 2, move_code)
            _apply_move_inplace_numba(work_state, work_pos, move_code)
            if found_depth >= 0:
                return found_depth

        return -1

    path_len = dfs(0, slack, -1)
    return path_len >= 0, path_len, nodes, pruned, tt_overflow


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
        exact_tt_capacity: int = 10_000_000,
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
        self.exact_progress = ""

        max_shorter_slack = best_len - gap0 - 1
        total_slack_steps = max_shorter_slack + 1
        start_pos = _build_pos_numba(start_np)
        work_state = start_np.copy()
        work_pos = start_pos.copy()
        path = np.empty(best_len - 1, dtype=np.int16)
        best_slack_by_state: dict[object, int] = {}
        tt_overflow = False

        for slack in range(total_slack_steps):
            found_shorter, path_len, nodes, pruned, tt_overflow = (
                _verify_with_slack(
                    start_np,
                    start_pos,
                    work_state,
                    work_pos,
                    path,
                    slack,
                    self.exact_tt_capacity,
                    best_slack_by_state,
                    tt_overflow,
                )
            )
            if self.verbose > 0 and slack > 0:
                elapsed = time() - exact_start
                n_exp = stats["exact_nodes_expanded"] + nodes
                self.exact_progress = (
                    f"| ex {gap0}..{best_len - 1} {slack + 1}/{total_slack_steps} "
                    f"{n_exp:,}n {elapsed:.1f}s"
                )
            stats["exact_nodes_expanded"] += nodes
            stats["exact_pruned_by_transposition"] += pruned
            stats["exact_tt_size"] = len(best_slack_by_state)
            stats["exact_tt_overflow"] = tt_overflow

            if found_shorter:
                stats["exact_time"] = time() - exact_start
                stats["exact_found"] = True
                stats["exact_found_len"] = path_len
                stats["exact_status"] = "found_exact_improvement"
                stats["path_found"] = True
                return True, path_len, self._codes_to_solution(path, path_len)

        stats["exact_time"] = time() - exact_start
        stats["exact_status"] = "proved_optimal_in_range"
        stats["path_found"] = True
        return True, best_len, best_solution
