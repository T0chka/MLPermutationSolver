"""Universal exact optimality verifier for permutation puzzles.

Competition-specific persistence and target selection intentionally live outside
this module. ``ExactVerifier`` receives one permutation, one incumbent and one
proven root lower bound; it either finds a shorter exact optimum or certifies
that the incumbent is globally optimal.
"""

from __future__ import annotations

from time import time
from typing import Any, Optional, Tuple

import numpy as np
import torch

from src.models.base_model import BaseModel
from src.puzzles import PuzzleSpec
from src.solvers.base_solver import BaseSolver
from src.solvers.exact_kernels import (
    ExactKernel,
    KERNEL_PANCAKE_GAP,
    exact_kernel_lower_bound,
)
from src.solvers.exact_search import verify_shorter_paths_numba


class ExactVerifier(BaseSolver):
    """Find an exact improvement or certify an incumbent as globally optimal."""

    def __init__(
        self,
        *,
        puzzle_spec: PuzzleSpec,
        device: torch.device,
        exact_kernel: ExactKernel,
        adapter: Any = None,
        incumbent_solver: Optional[BaseSolver] = None,
        exact_verify_margin: int = 2,
        exact_tt_capacity: int = 100_000_000,
        verbose: int = 0,
    ) -> None:
        if exact_verify_margin < 0:
            raise ValueError("exact_verify_margin must be non-negative")
        if exact_tt_capacity <= 0:
            raise ValueError("exact_tt_capacity must be positive")
        if puzzle_spec.state_size > np.iinfo(np.int16).max:
            raise ValueError("ExactVerifier requires state_size <= 32767")
        expected_shape = (
            int(puzzle_spec.move_indices.size(0)),
            int(puzzle_spec.state_size),
        )
        if exact_kernel.move_indices.shape != expected_shape:
            raise ValueError(
                "exact_kernel move_indices do not match PuzzleSpec: "
                f"expected={expected_shape}, got={exact_kernel.move_indices.shape}"
            )
        spec_moves = np.ascontiguousarray(
            puzzle_spec.move_indices.detach().cpu().numpy(), dtype=np.int16
        )
        spec_inverse = np.ascontiguousarray(
            puzzle_spec.inverse_moves.detach().cpu().numpy(), dtype=np.int16
        )
        spec_solved = np.ascontiguousarray(
            puzzle_spec.solved_state.detach().cpu().numpy(), dtype=np.int16
        )
        if not np.array_equal(exact_kernel.move_indices, spec_moves):
            raise ValueError("exact_kernel generators differ from PuzzleSpec")
        if not np.array_equal(exact_kernel.inverse_moves, spec_inverse):
            raise ValueError("exact_kernel inverse moves differ from PuzzleSpec")
        if not np.array_equal(exact_kernel.solved_state, spec_solved):
            raise ValueError("exact_kernel solved state differs from PuzzleSpec")

        super().__init__(puzzle_spec, device, model=None, verbose=verbose)
        self.adapter = adapter
        self.incumbent_solver = incumbent_solver
        self.exact_kernel = exact_kernel
        self.exact_verify_margin = int(exact_verify_margin)
        self.exact_tt_capacity = int(exact_tt_capacity)
        self._exact_tt_slots = _next_power_of_two(self.exact_tt_capacity * 2)
        self._bits_per_value = max(1, (puzzle_spec.state_size - 1).bit_length())
        self._name_to_code = {
            name: code for code, name in enumerate(self.move_names)
        }
        self.exact_progress = ""
        self.reset()

    def reset(self) -> None:
        self.search_stats.update({
            "incumbent_found": False,
            "incumbent_len": -1,
            "incumbent_solution": "",
            "incumbent_time": 0.0,
            "incumbent_step": -1,
            "incumbent_meet_source": "",
            "exact_time": 0.0,
            "lower_bound0": -1,
            # Existing pancake reports expect this field.
            "gap0": -1,
            "exact_attempted": False,
            "exact_verify_margin": self.exact_verify_margin,
            "exact_tt_capacity": self.exact_tt_capacity,
            "exact_tt_size": 0,
            "exact_tt_overflow": False,
            "exact_checked_from": -1,
            "exact_checked_to": -1,
            "exact_nodes_expanded": 0,
            "exact_pruned_by_lower_bound": 0,
            "exact_pruned_by_gap": 0,
            "exact_pruned_by_transposition": 0,
            "exact_found": False,
            "exact_found_len": -1,
            "exact_proved_optimal": False,
            "proved_lower_bound": -1,
            "exact_status": "",
            "path_found": False,
        })

    def _parse_solution_codes(self, solution: str) -> list[int]:
        if not solution:
            return []
        out: list[int] = []
        for part in solution.split("."):
            token = part.strip()
            if not token:
                continue
            code = self._name_to_code.get(token)
            if code is None:
                raise ValueError(f"Unknown move in incumbent solution: {token!r}")
            out.append(code)
        return out

    def _validate_solution(
        self,
        start_state: torch.Tensor,
        solution: str,
        expected_len: int,
    ) -> None:
        codes = self._parse_solution_codes(solution)
        if len(codes) != expected_len:
            raise ValueError(
                f"Incumbent solution length mismatch: expected {expected_len}, "
                f"got {len(codes)}"
            )
        final_state = self._apply_moves(start_state, codes)
        if not torch.equal(final_state, self.solved_state):
            raise ValueError("Incumbent solution does not solve the puzzle")

    def _codes_to_solution(self, path: np.ndarray, path_len: int) -> str:
        return ".".join(
            self.move_names[int(path[idx])] for idx in range(path_len)
        )

    def _kernel_root_lower_bound(self, start_state: torch.Tensor) -> int:
        state_np = np.ascontiguousarray(
            start_state.detach().cpu().numpy(), dtype=np.int16
        )
        return exact_kernel_lower_bound(state_np, self.exact_kernel)

    def _load_incumbent(
        self,
        start_state: torch.Tensor,
        model: Any,
        initial_solution: Optional[str],
    ) -> tuple[bool, int, str]:
        stats = self.search_stats
        if initial_solution is not None:
            solution = str(initial_solution).strip()
            if not solution:
                raise ValueError("initial_solution is empty in verify-only mode")
            length = len(self._parse_solution_codes(solution))
            stats["incumbent_found"] = True
            stats["incumbent_len"] = length
            stats["incumbent_solution"] = solution
            stats["incumbent_step"] = -1
            stats["incumbent_meet_source"] = "verify_only"
            return True, length, solution

        if self.incumbent_solver is None:
            raise ValueError(
                "ExactVerifier requires initial_solution or incumbent_solver"
            )

        t0 = time()
        found, length, solution = self.incumbent_solver.solve(start_state, model)
        stats["incumbent_time"] = time() - t0
        incumbent_stats = self.incumbent_solver.search_stats
        stats["incumbent_found"] = bool(found)
        stats["incumbent_len"] = int(length)
        stats["incumbent_solution"] = str(solution or "")
        stats["incumbent_step"] = incumbent_stats.get("intersect_at_step", -1)
        stats["incumbent_meet_source"] = incumbent_stats.get("meet_source", "")
        return bool(found), int(length), str(solution or "")

    def solve(
        self,
        start_state: torch.Tensor,
        model: BaseModel | Any = None,
        *,
        initial_solution: Optional[str] = None,
        initial_lower_bound: Optional[int] = None,
    ) -> Tuple[bool, int, str]:
        """Return the exact optimum when verification completes.

        ``initial_lower_bound`` may be stronger than the Numba per-state kernel,
        for example a root-only ILP proof stored by a competition pipeline. It
        determines the first length that must be excluded. Descendant pruning
        uses only the configured Numba kernel.
        """
        self.reset()
        self.exact_progress = ""
        self.model = model
        stats = self.search_stats

        found, best_len, best_solution = self._load_incumbent(
            start_state, model, initial_solution
        )
        if not found:
            stats["exact_status"] = "skipped_no_incumbent"
            return False, best_len, best_solution

        self._validate_solution(start_state, best_solution, best_len)

        kernel_lb = self._kernel_root_lower_bound(start_state)
        root_lb = kernel_lb
        if initial_lower_bound is not None:
            external_lb = int(initial_lower_bound)
            if external_lb < 0:
                raise ValueError("initial_lower_bound must be non-negative")
            root_lb = max(root_lb, external_lb)

        stats["lower_bound0"] = root_lb
        if self.exact_kernel.kind == KERNEL_PANCAKE_GAP:
            stats["gap0"] = kernel_lb

        if best_len < root_lb:
            raise ValueError(
                f"Incumbent length {best_len} is below proven lower bound {root_lb}"
            )

        if best_len == root_lb:
            stats["exact_status"] = "incumbent_at_lower_bound"
            stats["exact_proved_optimal"] = True
            stats["proved_lower_bound"] = best_len
            stats["path_found"] = True
            return True, best_len, best_solution

        if best_len - root_lb > self.exact_verify_margin:
            stats["exact_status"] = "skipped_margin"
            stats["path_found"] = True
            return True, best_len, best_solution

        stats["exact_attempted"] = True
        stats["exact_checked_from"] = root_lb
        stats["exact_checked_to"] = best_len - 1

        start_np = np.ascontiguousarray(
            start_state.detach().cpu().numpy(), dtype=np.int16
        )
        exact_start = time()
        (
            found_shorter,
            path,
            path_len,
            found_at_length,
            nodes,
            pruned_lb,
            pruned_tt,
            tt_size,
            overflowed,
        ) = verify_shorter_paths_numba(
            start_np,
            root_lb,
            best_len - 1,
            int(self.exact_kernel.kind),
            self.exact_kernel.move_indices,
            self.exact_kernel.inverse_moves,
            self.exact_kernel.solved_state,
            self.exact_kernel.piece_distance,
            self.exact_kernel.piece_orbit,
            self.exact_kernel.subset_masks,
            self.exact_kernel.subset_capacities,
            int(self.exact_kernel.n_orbits),
            self.exact_kernel.corner_dist,
            self.exact_kernel.corner_ori_code_to_idx,
            self.exact_kernel.corner_slot0_positions,
            self.exact_kernel.corner_piece_to_cubie,
            self.exact_kernel.corner_orientation_lookup,
            self.exact_kernel.inner_cut_positions,
            self.exact_kernel.inner_cut_lengths,
            self.exact_kernel.inner_goal_piece_mask,
            self.exact_kernel.inner_dual_cut_indices,
            self.exact_kernel.inner_dual_weights_x4,
            self.exact_kernel.inner_parity_orbit_positions,
            self.exact_kernel.inner_parity_orbit_lengths,
            self.exact_kernel.inner_parity_goal_local_by_piece,
            self.exact_kernel.inner_parity_move_toggle,
            int(self.exact_kernel.corner_orientation_count),
            self.exact_tt_capacity,
            self._exact_tt_slots,
            self._bits_per_value,
        )
        stats["exact_time"] = time() - exact_start
        stats["exact_nodes_expanded"] = int(nodes)
        stats["exact_pruned_by_lower_bound"] = int(pruned_lb)
        # For pancakes the generic lower-bound pruning is exactly gap pruning.
        stats["exact_pruned_by_gap"] = int(pruned_lb)
        stats["exact_pruned_by_transposition"] = int(pruned_tt)
        stats["exact_tt_size"] = int(tt_size)
        stats["exact_tt_overflow"] = bool(overflowed)

        if self.verbose > 0 and nodes > 0:
            self.exact_progress = (
                f"| ex {root_lb}..{best_len - 1} "
                f"{int(nodes):,}n {stats['exact_time']:.1f}s"
            )

        if found_shorter:
            optimal_len = int(path_len)
            if optimal_len != int(found_at_length):
                raise RuntimeError(
                    "Internal exact-search inconsistency: path length does not "
                    "match the target length where it was found"
                )
            solution = self._codes_to_solution(path, optimal_len)
            self._validate_solution(start_state, solution, optimal_len)
            stats["exact_found"] = True
            stats["exact_found_len"] = optimal_len
            stats["exact_proved_optimal"] = True
            stats["proved_lower_bound"] = optimal_len
            stats["exact_status"] = "found_exact_improvement"
            stats["path_found"] = True
            return True, optimal_len, solution

        stats["exact_proved_optimal"] = True
        stats["proved_lower_bound"] = best_len
        stats["exact_status"] = "proved_optimal_in_range"
        stats["path_found"] = True
        return True, best_len, best_solution


def _next_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()
