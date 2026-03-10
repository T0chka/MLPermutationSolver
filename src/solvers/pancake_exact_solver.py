"""
The solver finds and validates the candidate solution and proves its optimality by
excluding all admissible shorter solutions down to the gap lower bound.

Proof procedure:
1. Validate the candidate solution.
2. If its length equals the gap count, certify it as optimal immediately.
3. Otherwise, prove that no solution of length gap count exists.
4. If needed, prove that no solution of length gap count + 1 exists.
5. If all shorter lengths are excluded, certify the candidate as optimal.

Currently, certification is supported only up to gap count + 2.
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
def _has_decreasing_move_numba(state: np.ndarray, pos: np.ndarray) -> bool:
    """Return True if some move decreases gap."""
    n = state.shape[0]
    top = state[0]

    if top > 0:
        idx = pos[top - 1]
        if idx >= 2 and abs(state[idx - 1] - state[idx]) != 1:
            return True

    if top + 1 < n:
        idx = pos[top + 1]
        if idx >= 2 and abs(state[idx - 1] - state[idx]) != 1:
            return True

    if top == n - 1:
        if abs(state[n - 1] - n) != 1:
            return True

    return False


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
def _dfs_gap_numba(
    state: np.ndarray,
    pos: np.ndarray,
    depth: int,
    target_len: int,
    path: np.ndarray,
    nodes: np.ndarray,
) -> bool:
    """DFS over decreasing moves only."""
    nodes[0] += 1

    if depth == target_len:
        return _is_solved_numba(state)

    count, move0, move1, move2 = _decreasing_candidates_numba(state, pos)

    if count >= 1:
        _apply_move_inplace_numba(state, pos, move0)
        path[depth] = move0
        found = _dfs_gap_numba(state, pos, depth + 1, target_len, path, nodes)
        _apply_move_inplace_numba(state, pos, move0)
        if found:
            return True

    if count >= 2:
        _apply_move_inplace_numba(state, pos, move1)
        path[depth] = move1
        found = _dfs_gap_numba(state, pos, depth + 1, target_len, path, nodes)
        _apply_move_inplace_numba(state, pos, move1)
        if found:
            return True

    if count >= 3:
        _apply_move_inplace_numba(state, pos, move2)
        path[depth] = move2
        found = _dfs_gap_numba(state, pos, depth + 1, target_len, path, nodes)
        _apply_move_inplace_numba(state, pos, move2)
        if found:
            return True

    return False


@njit(cache=False)
def _dfs_gap_plus_one_numba(
    state: np.ndarray,
    pos: np.ndarray,
    depth: int,
    target_len: int,
    path: np.ndarray,
    nodes: np.ndarray,
    neutral_used: bool,
) -> bool:
    """DFS over paths with exactly one neutral move."""
    nodes[0] += 1

    if depth == target_len:
        return neutral_used and _is_solved_numba(state)

    count, move0, move1, move2 = _decreasing_candidates_numba(state, pos)

    if count >= 1:
        _apply_move_inplace_numba(state, pos, move0)
        path[depth] = move0
        found = _dfs_gap_plus_one_numba(
            state,
            pos,
            depth + 1,
            target_len,
            path,
            nodes,
            neutral_used,
        )
        _apply_move_inplace_numba(state, pos, move0)
        if found:
            return True

    if count >= 2:
        _apply_move_inplace_numba(state, pos, move1)
        path[depth] = move1
        found = _dfs_gap_plus_one_numba(
            state,
            pos,
            depth + 1,
            target_len,
            path,
            nodes,
            neutral_used,
        )
        _apply_move_inplace_numba(state, pos, move1)
        if found:
            return True

    if count >= 3:
        _apply_move_inplace_numba(state, pos, move2)
        path[depth] = move2
        found = _dfs_gap_plus_one_numba(
            state,
            pos,
            depth + 1,
            target_len,
            path,
            nodes,
            neutral_used,
        )
        _apply_move_inplace_numba(state, pos, move2)
        if found:
            return True

    if neutral_used:
        return False

    n_moves = state.shape[0] - 1
    for move_code in range(n_moves):
        if _delta_gap_for_move_numba(state, move_code) != 0:
            continue

        _apply_move_inplace_numba(state, pos, move_code)
        if _has_decreasing_move_numba(state, pos):
            path[depth] = move_code
            found = _dfs_gap_numba(
                state,
                pos,
                depth + 1,
                target_len,
                path,
                nodes,
            )
            _apply_move_inplace_numba(state, pos, move_code)
            if found:
                return True
        else:
            _apply_move_inplace_numba(state, pos, move_code)

    return False


@njit(cache=False)
def _verify_gap_exists_numba(start_state: np.ndarray) -> Tuple[bool, np.ndarray, int]:
    """Return existence of a gap-length path."""
    state = start_state.copy()
    pos = _build_pos_numba(state)
    target_len = _gap_count_numba(state)
    path = np.empty(target_len, dtype=np.int16)
    nodes = np.zeros(1, dtype=np.int64)
    found = _dfs_gap_numba(state, pos, 0, target_len, path, nodes)
    return found, path, nodes[0]


@njit(cache=False)
def _verify_gap_plus_one_exists_numba(
    start_state: np.ndarray,
) -> Tuple[bool, np.ndarray, int]:
    """Return existence of a gap+1 path."""
    state = start_state.copy()
    pos = _build_pos_numba(state)
    target_len = _gap_count_numba(state) + 1
    path = np.empty(target_len, dtype=np.int16)
    nodes = np.zeros(1, dtype=np.int64)
    found = _dfs_gap_plus_one_numba(
        state,
        pos,
        0,
        target_len,
        path,
        nodes,
        False,
    )
    return found, path, nodes[0]


class PancakeExactSolver(BaseSolver):
    """
    The solver finds and certifies optimal pancake solutions with length equal to
    the gap count, or gap count + n (where n is the exact_verify_margin, max is 2).

    The current solution is validated first. If its length equals the gap count,
    it is already optimal. Otherwise the solver runs exhaustive depth-first search
    over all admissible shorter paths. It first checks all decreasing-gap paths of
    gap-count length. If the current solution has length gap count + 2, it then
    checks all paths of length gap count + 1 with exactly one neutral move and
    all remaining moves decreasing. If no path is found, then no shorter solution
    exists in the verified range.
    """

    def __init__(
        self,
        *,
        puzzle_spec: PuzzleSpec,
        device: torch.device,
        adapter: Any,
        incumbent_solver: BaseSolver,
        exact_verify_margin: int = 2,
        verbose: int = 0,
    ) -> None:
        """Store exact-solver dependencies."""
        if puzzle_spec.puzzle_type != "pancake":
            raise ValueError(
                "PancakeExactSolver only supports puzzle_type='pancake'; "
                f"got {puzzle_spec.puzzle_type!r}"
            )
        if exact_verify_margin >= 3:
            raise NotImplementedError(
                "exact_verify_margin >= 3 is not implemented yet."
            )
        super().__init__(puzzle_spec, device, model=None, verbose=verbose)
        self.adapter = adapter
        self.incumbent_solver = incumbent_solver
        self.exact_verify_margin = exact_verify_margin
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
            dtype=np.int64,
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
        stats["exact_checked_to"] = min(best_len - 1, gap0 + 1)
        exact_start = time()

        found_gap, gap_path, gap_nodes = _verify_gap_exists_numba(start_np)
        stats["exact_nodes_expanded"] += gap_nodes

        if found_gap:
            stats["exact_time"] = time() - exact_start
            stats["exact_found"] = True
            stats["exact_found_len"] = gap0
            stats["exact_status"] = "found_exact_improvement"
            stats["path_found"] = True
            return True, gap0, self._codes_to_solution(gap_path, len(gap_path))

        if best_len == gap0 + 1:
            stats["exact_time"] = time() - exact_start
            stats["exact_status"] = "proved_optimal_in_range"
            stats["path_found"] = True
            return True, best_len, best_solution

        if best_len != gap0 + 2:
            raise RuntimeError("Internal error: unexpected exact branch.")

        found_gap1, gap1_path, gap1_nodes = _verify_gap_plus_one_exists_numba(
            start_np
        )
        stats["exact_nodes_expanded"] += gap1_nodes
        stats["exact_time"] = time() - exact_start

        if found_gap1:
            stats["exact_found"] = True
            stats["exact_found_len"] = gap0 + 1
            stats["exact_status"] = "found_exact_improvement"
            stats["path_found"] = True
            return True, gap0 + 1, self._codes_to_solution(
                gap1_path,
                len(gap1_path),
            )

        stats["exact_status"] = "proved_optimal_in_range"
        stats["path_found"] = True
        return True, best_len, best_solution