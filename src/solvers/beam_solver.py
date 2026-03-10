"""Unified approximate solver with backward search modes."""

from contextlib import contextmanager
from dataclasses import dataclass
from time import time
from typing import Callable, Optional, Sequence, Tuple

import torch

from src.models.base_model import BaseModel
from src.puzzles import PuzzleSpec
from src.solvers.base_solver import BaseSolver
from src.solvers.hash_history import HashHistory


class _SolverProfiler:
    """Encapsulates runtime profiling for BeamSolver; no-op when disabled."""

    STAGES = (
        "expand",
        "unique",
        "history_filter",
        "lower_bound",
        "beam_prune",
        "meet_lookup",
        "bfs_prebuild",
    )

    def __init__(
        self,
        device: torch.device,
        enabled: bool,
        stats_ref: dict,
    ) -> None:
        self.device = device
        self.enabled = enabled
        self._stats = stats_ref

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def sync_for_timing(self) -> None:
        """Sync GPU before/after total search_time; always runs for accurate timing."""
        self._sync()

    def init_profile(self) -> None:
        if self.enabled:
            self._stats["profile"] = {
                s: {"time": 0.0, "calls": 0, "states": 0}
                for s in self.STAGES
            }

    @contextmanager
    def section(
        self,
        name: str,
        get_states: Optional[Callable[[], int]] = None,
        get_states_exit: Optional[Callable[[], int]] = None,
    ):
        """Profile a section. get_states at enter, or get_states_exit at exit."""
        if self.enabled:
            self._sync()
            t0 = time()
            states = int(get_states()) if get_states else 0
        else:
            t0 = None
            states = 0
        yield
        if self.enabled and t0 is not None:
            self._sync()
            if get_states_exit:
                states = int(get_states_exit())
            delta = time() - t0
            p = self._stats.get("profile")
            if p and name in p:
                p[name]["time"] += delta
                p[name]["calls"] += 1
                p[name]["states"] += states


@dataclass(frozen=True)
class SolverConfig:
    """Public configuration for approximate (beam) solver."""

    solver_type: str = "beam"
    beam_width: int = 2**20
    max_steps: int = 0
    backward_mode: str = "off"
    backward_max_states: int = 0
    bs_nbt_depth: int = 2
    randomize_ties: bool = True


class SolutionObjective:
    """Solution objective with explicit stop policy.

    TODO: support additional objectives such as "longest".
    """

    def __init__(self, objective: str = "shortest") -> None:
        if objective != "shortest":
            raise ValueError(f"Only objective='shortest' is supported; got {objective!r}")
        
        self.objective = objective
        self.best_solution_len: Optional[int] = None

    def solution_better_than(self, candidate_len: int) -> bool:
        if self.best_solution_len is None:
            return True
        return int(candidate_len) < int(self.best_solution_len)

    def update_best(self, candidate_len: int) -> None:
        """Update best solution length if candidate is better."""
        if self.solution_better_than(candidate_len):
            self.best_solution_len = int(candidate_len)


@dataclass(frozen=True)
class SchedulerContext:
    """Runtime context visible to scheduler."""

    step: int
    forward_depth: int
    backward_depth: int
    path_len_limit: int
    fwd_exhausted: bool = False
    bwd_exhausted: bool = False
    

class SearchScheduler:
    """Scheduler policy interface."""

    def make_plan(self, context: SchedulerContext) -> Tuple[str, ...]:
        raise NotImplementedError


class IndependentFrontierScheduler(SearchScheduler):
    """Independent-side scheduler for off/bfs/beam."""

    def __init__(self, backward_mode: str) -> None:
        self.backward_mode = backward_mode

    def make_plan(self, context: SchedulerContext) -> Tuple[str, ...]:
        pl = context.path_len_limit
        fwd_ok = not context.fwd_exhausted and context.forward_depth + 1 < pl
        bwd_ok = not context.bwd_exhausted and context.backward_depth + 1 < pl

        if self.backward_mode in {"off", "bfs"}:
            return ("forward",) if fwd_ok else ()
        if not fwd_ok and not bwd_ok:
            return ()
        if context.step % 2 == 1:
            return tuple(a for a in ("forward", "backward") if (a == "forward" and fwd_ok) or (a == "backward" and bwd_ok))
        return tuple(a for a in ("backward", "forward") if (a == "backward" and bwd_ok) or (a == "forward" and fwd_ok))


class ExpansionKernel:
    """Forward/backward expansion kernel."""

    def __init__(
        self,
        *,
        adapter,
        move_indices: torch.Tensor,
        inverse_move_indices: torch.Tensor,
        device: torch.device,
        state_size: int,
        state_dtype: torch.dtype,
    ) -> None:
        self.adapter = adapter
        self.move_indices = move_indices
        self.inverse_move_indices = inverse_move_indices
        self.device = device
        self.state_size = state_size
        self.state_dtype = state_dtype

    def expand(
        self,
        states: torch.Tensor,
        direction: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        max_moves = getattr(self.adapter, "moves_per_state", 0)
        move_codes, valid_mask = self.adapter.move_codes_and_mask(
            states,
            direction,
            max_moves=max_moves,
        )
        pairs = valid_mask.nonzero(as_tuple=False)
        if pairs.numel() == 0:
            empty_states = torch.empty(
                (0, self.state_size),
                dtype=self.state_dtype,
                device=self.device,
            )
            empty_idx = torch.empty(0, dtype=torch.long, device=self.device)
            empty_codes = torch.empty(0, dtype=torch.long, device=self.device)
            return empty_states, empty_idx, empty_codes
        parent_ids = pairs[:, 0]
        codes = move_codes[parent_ids, pairs[:, 1]]
        src = states.index_select(0, parent_ids)
        if direction == "forward":
            child_idx = self.move_indices[codes]
        else:
            child_idx = self.inverse_move_indices[codes]
        children = torch.gather(src, 1, child_idx).contiguous()
        return children, parent_ids, codes


class LayerArchive:
    """Layered archive of states and parent/move links."""

    def __init__(self, root_state: torch.Tensor) -> None:
        self.states_by_depth = [root_state.unsqueeze(0)]
        self.parents_by_depth: list[torch.Tensor] = []
        self.moves_by_depth: list[torch.Tensor] = []

    @property
    def depth(self) -> int:
        return len(self.states_by_depth) - 1

    @property
    def frontier(self) -> torch.Tensor:
        return self.states_by_depth[-1]

    def add_layer(
        self,
        states: torch.Tensor,
        parents: torch.Tensor,
        moves: torch.Tensor,
    ) -> None:
        self.states_by_depth.append(states)
        self.parents_by_depth.append(parents)
        self.moves_by_depth.append(moves)

    def get_state(self, depth: int, idx: int) -> torch.Tensor:
        return self.states_by_depth[depth][idx]


class MeetArchive:
    """Meeting hash archive with chunked append."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.hash_chunks: list[torch.Tensor] = []
        self.depth_chunks: list[torch.Tensor] = []
        self.idx_chunks: list[torch.Tensor] = []
        self.size = 0
        self.version = 0

    def add_layer_hashes(self, hashes: torch.Tensor, depth: int) -> None:
        if hashes.numel() == 0:
            return
        idxs = torch.arange(hashes.size(0), dtype=torch.long, device=self.device)
        depths = torch.full_like(idxs, int(depth))
        self.hash_chunks.append(hashes)
        self.depth_chunks.append(depths)
        self.idx_chunks.append(idxs)
        self.size += int(hashes.size(0))
        self.version += 1


class MeetLookup:
    """Lookup helper with lazy cached sorted view.

    Note:
    Matching is hash-based and then state-equality is checked at meet time.
    This is approximate-search infrastructure, not a collision-proof index.
    """

    def __init__(self, archive: MeetArchive) -> None:
        self.archive = archive
        self._cache_version = -1
        self._sorted_hashes = torch.empty(0, dtype=torch.int64, device=archive.device)
        self._sorted_depths = torch.empty(0, dtype=torch.long, device=archive.device)
        self._sorted_idxs = torch.empty(0, dtype=torch.long, device=archive.device)

    def _ensure_cache(self) -> None:
        if self._cache_version == self.archive.version:
            return
        if self.archive.size == 0:
            self._sorted_hashes = torch.empty(
                0, dtype=torch.int64, device=self.archive.device
            )
            self._sorted_depths = torch.empty(
                0, dtype=torch.long, device=self.archive.device
            )
            self._sorted_idxs = torch.empty(
                0, dtype=torch.long, device=self.archive.device
            )
            self._cache_version = self.archive.version
            return

        # Vectorized rebuild/update: concatenate and sort once.
        if self._cache_version < 0:
            start_chunk = 0
        else:
            start_chunk = self._cache_version

        if start_chunk >= self.archive.version:
            self._cache_version = self.archive.version
            return

        new_h = torch.cat(self.archive.hash_chunks[start_chunk:], dim=0)
        new_d = torch.cat(self.archive.depth_chunks[start_chunk:], dim=0)
        new_i = torch.cat(self.archive.idx_chunks[start_chunk:], dim=0)

        if self._cache_version < 0 or self._sorted_hashes.numel() == 0:
            all_h, all_d, all_i = new_h, new_d, new_i
        else:
            all_h = torch.cat([self._sorted_hashes, new_h], dim=0)
            all_d = torch.cat([self._sorted_depths, new_d], dim=0)
            all_i = torch.cat([self._sorted_idxs, new_i], dim=0)

        order = torch.argsort(all_h)
        self._sorted_hashes = all_h[order]
        self._sorted_depths = all_d[order]
        self._sorted_idxs = all_i[order]
        self._cache_version = self.archive.version

    def membership_mask(self, query_hashes: torch.Tensor) -> torch.Tensor:
        self._ensure_cache()
        n = query_hashes.size(0)
        if n == 0 or self._sorted_hashes.numel() == 0:
            return torch.zeros(n, dtype=torch.bool, device=self.archive.device)
        pos = torch.searchsorted(self._sorted_hashes, query_hashes)
        valid = pos < self._sorted_hashes.size(0)
        out = torch.zeros(n, dtype=torch.bool, device=self.archive.device)
        if valid.any():
            qi = valid.nonzero(as_tuple=True)[0]
            p = pos[qi]
            out[qi] = self._sorted_hashes[p] == query_hashes[qi]
        return out

    def lookup_rows(
        self,
        query_hashes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._ensure_cache()
        if query_hashes.numel() == 0 or self._sorted_hashes.numel() == 0:
            empty = torch.empty(0, dtype=torch.long, device=self.archive.device)
            return empty, empty
        pos = torch.searchsorted(self._sorted_hashes, query_hashes)
        valid = pos < self._sorted_hashes.size(0)
        if not valid.any():
            empty = torch.empty(0, dtype=torch.long, device=self.archive.device)
            return empty, empty
        q_idx = valid.nonzero(as_tuple=True)[0]
        pos = pos[q_idx]
        eq = self._sorted_hashes[pos] == query_hashes[q_idx]
        if not eq.any():
            empty = torch.empty(0, dtype=torch.long, device=self.archive.device)
            return empty, empty
        return q_idx[eq], pos[eq]

    def refs_for_rows(self, row_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self._ensure_cache()
        return self._sorted_depths[row_idx], self._sorted_idxs[row_idx]


class PathAssembler:
    """Tensor-only path reconstruction."""

    def __init__(
        self,
        *,
        device: torch.device,
        inverse_moves: torch.Tensor,
    ) -> None:
        self.device = device
        self.inverse_moves = inverse_moves

    def path_root_to_node(
        self,
        parents_by_depth: Sequence[torch.Tensor],
        moves_by_depth: Sequence[torch.Tensor],
        depth: int,
        node_idx: int,
    ) -> torch.Tensor:
        if depth == 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        out = torch.empty(depth, dtype=torch.long, device=self.device)
        idx = int(node_idx)
        for d in range(depth - 1, -1, -1):
            out[d] = moves_by_depth[d][idx]
            idx = int(parents_by_depth[d][idx].item())
        return out

    def build_solution_codes(
        self,
        fwd_archive: LayerArchive,
        fwd_depth: int,
        fwd_idx: int,
        bwd_archive: LayerArchive,
        bwd_depth: int,
        bwd_idx: int,
    ) -> torch.Tensor:
        fwd = self.path_root_to_node(
            fwd_archive.parents_by_depth,
            fwd_archive.moves_by_depth,
            fwd_depth,
            fwd_idx,
        )
        bwd_root_to_meet = self.path_root_to_node(
            bwd_archive.parents_by_depth,
            bwd_archive.moves_by_depth,
            bwd_depth,
            bwd_idx,
        )
        if bwd_root_to_meet.numel() == 0:
            return fwd
        # Backward codes c used inverse_move_indices[c]; to undo we need forward
        # codes c (not inverse_moves[c]). Reverse order only.
        tail = torch.flip(bwd_root_to_meet, dims=[0])
        return torch.cat([fwd, tail], dim=0)


class BeamSolver(BaseSolver):
    """Approximate solver with modes off/bfs/beam for backward search."""

    def __init__(
        self,
        *,
        puzzle_spec: PuzzleSpec,
        adapter,
        scheduler: SearchScheduler,
        objective: SolutionObjective,
        beam_width: int = 2**20,
        max_steps: int = 0,
        backward_mode: str = "off",
        backward_max_states: int = 0,
        bs_nbt_depth: int = 2,
        randomize_ties: bool = True,
        verbose: int = 0,
        hashes_batch_size: int = 10_000_000,
        profile_runtime: bool = False,
    ) -> None:
        device = puzzle_spec.solved_state.device
        super().__init__(puzzle_spec, device, model=None, verbose=verbose)
        if backward_mode not in {"off", "bfs", "beam"}:
            raise ValueError(f"backward_mode must be one of: 'off', 'bfs', 'beam'; got {backward_mode!r}")
        if backward_mode == "bfs" and int(backward_max_states) <= 0:
            raise ValueError("backward_max_states must be > 0 for backward_mode='bfs'")
        if backward_mode in {"off", "beam"} and int(backward_max_states) != 0:
            raise ValueError(f"backward_max_states must be 0 for backward_mode in {'off','beam'}")

        self.adapter = adapter
        self.adapter.set_logger(self.logger)
        self.scheduler = scheduler
        self.objective = objective
        self.backward_mode = backward_mode
        self.backward_max_states = int(backward_max_states)
        self.beam_width = int(beam_width)
        self.bs_nbt_depth = int(bs_nbt_depth)
        self.randomize_ties = bool(randomize_ties)
        self.hashes_batch_size = int(hashes_batch_size)
        if int(max_steps) <= 0:
            raise ValueError("max_steps must be > 0")
        self.max_steps = int(max_steps)

        self.kernel = ExpansionKernel(
            adapter=self.adapter,
            move_indices=self.move_indices,
            inverse_move_indices=self.inverse_move_indices,
            device=self.device,
            state_size=self.state_size,
            state_dtype=self.state_dtype,
        )
        self.path_assembler = PathAssembler(device=self.device, inverse_moves=self.inverse_moves)
        max_hist = self.beam_width * max(1, self.bs_nbt_depth)
        self.fwd_history = HashHistory(
            device=self.device,
            max_history_size=max_hist,
            nbt_depth=self.bs_nbt_depth,
            hashes_batch_size=self.hashes_batch_size,
            warn_fn=self.log_warning,
        )
        self.bwd_history = HashHistory(
            device=self.device,
            max_history_size=max_hist,
            nbt_depth=self.bs_nbt_depth,
            hashes_batch_size=self.hashes_batch_size,
            warn_fn=self.log_warning,
        )
        self.hash_vec = torch.randint(
            low=-(2**62),
            high=2**62,
            size=(self.state_size,),
            dtype=torch.int64,
            device=self.device,
        ).contiguous()
        self._hash_vec_2d = self.hash_vec.unsqueeze(0)
        self._best_solution_codes: Optional[torch.Tensor] = None
        self._profiler = _SolverProfiler(
            self.device,
            bool(profile_runtime),
            self.search_stats,
        )

    def _path_len_limit(self) -> int:
        if self.objective.best_solution_len is None:
            return self.max_steps + 1
        return int(self.objective.best_solution_len)

    def reset(self) -> None:
        self.fwd_history.reset()
        self.bwd_history.reset()
        self.objective.best_solution_len = None
        self._best_solution_codes = None
        self.search_stats.clear()
        self.search_stats.update({
            "total_states_explored": 0,
            "search_time": 0.0,
            "path_found": False,
            "beam_width": self.beam_width,
            "max_steps": self.max_steps,
            "backward_mode": self.backward_mode,
            "backward_max_states": self.backward_max_states,
            "objective": self.objective.objective,
            "termination_reason": "",
            "backward_archive_states": 0,
            "best_solution_len": -1,
        })
        self._profiler.init_profile()

    def _compute_state_hashes(self, states: torch.Tensor) -> torch.Tensor:
        if states.size(0) == 0:
            return torch.empty(0, dtype=torch.int64, device=self.device)
        return torch.sum(states.to(torch.int64) * self._hash_vec_2d, dim=1)

    def _unique_by_hash(
        self,
        states: torch.Tensor,
        parents: torch.Tensor,
        moves: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hashes = self._compute_state_hashes(states)
        if hashes.numel() == 0:
            return states, parents, moves, hashes
        order = torch.argsort(hashes)
        sorted_hashes = hashes[order]
        keep = torch.ones(sorted_hashes.size(0), dtype=torch.bool, device=self.device)
        keep[1:] = sorted_hashes[1:] != sorted_hashes[:-1]
        keep_idx = order[keep]
        return (
            states.index_select(0, keep_idx).contiguous(),
            parents.index_select(0, keep_idx),
            moves.index_select(0, keep_idx),
            hashes.index_select(0, keep_idx),
        )

    def _filter_new(
        self,
        states: torch.Tensor,
        parents: torch.Tensor,
        moves: torch.Tensor,
        hashes: torch.Tensor,
        history: HashHistory,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if hashes.numel() == 0:
            return states, parents, moves, hashes
        seen = history.check(hashes)
        keep = (~seen).nonzero(as_tuple=True)[0]
        return (
            states.index_select(0, keep).contiguous(),
            parents.index_select(0, keep),
            moves.index_select(0, keep),
            hashes.index_select(0, keep),
        )

    def _prune_by_lower_bound(
        self,
        states: torch.Tensor,
        parents: torch.Tensor,
        moves: torch.Tensor,
        hashes: torch.Tensor,
        new_depth: int,
        direction: str,
        path_len_limit: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if states.numel() == 0:
            return states, parents, moves, hashes
        lb = self.adapter.lower_bound(states, direction)
        keep = (new_depth + lb) < path_len_limit
        keep_idx = keep.nonzero(as_tuple=True)[0]
        return (
            states.index_select(0, keep_idx).contiguous(),
            parents.index_select(0, keep_idx),
            moves.index_select(0, keep_idx),
            hashes.index_select(0, keep_idx),
        )

    def _prune_frontier(
        self,
        states: torch.Tensor,
        parents: torch.Tensor,
        moves: torch.Tensor,
        hashes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n = int(states.size(0))
        if n <= self.beam_width:
            return states, parents, moves, hashes
        if self.model is None:
            scores = torch.zeros(n, dtype=torch.float32, device=self.device)
        else:
            scores = self.model.predict(states)
        if self.randomize_ties:
            scores = scores + torch.rand(n, device=self.device) * 1e-4
        _, idx = torch.topk(scores, k=self.beam_width, largest=False, sorted=False)
        return (
            states.index_select(0, idx).contiguous(),
            parents.index_select(0, idx),
            moves.index_select(0, idx),
            hashes.index_select(0, idx),
        )

    def _codes_to_solution(self, codes: torch.Tensor) -> str:
        if codes.numel() == 0:
            return ""
        return ".".join(self.move_names[int(c.item())] for c in codes)

    def _accept_candidate(self, codes: torch.Tensor, reason: str) -> None:
        """Update best found solution if candidate is better."""
        cand_len = int(codes.numel())
        if not self.objective.solution_better_than(cand_len):
            return
        self.objective.update_best(cand_len)
        self._best_solution_codes = codes
        self.search_stats["best_solution_len"] = cand_len
        self.search_stats["path_found"] = True
        self.search_stats["termination_reason"] = reason

    def _find_valid_meeting(
        self,
        query_states: torch.Tensor,
        query_hashes: torch.Tensor,
        other_archive: LayerArchive,
        other_lookup: MeetLookup,
    ) -> Optional[Tuple[int, int, int]]:
        q_idx, row_idx = other_lookup.lookup_rows(query_hashes)
        if q_idx.numel() == 0:
            return None
        depths, idxs = other_lookup.refs_for_rows(row_idx)
        for i in range(q_idx.numel()):
            qi = int(q_idx[i].item())
            d = int(depths[i].item())
            j = int(idxs[i].item())
            if torch.equal(query_states[qi], other_archive.get_state(d, j)):
                return qi, d, j
        return None

    def _prebuild_bfs_archive(
        self,
    ) -> Tuple[LayerArchive, MeetArchive, MeetLookup]:
        with self._profiler.section("bfs_prebuild", get_states_exit=lambda: total_states):
            bwd_archive = LayerArchive(self.solved_state)
            meet_archive = MeetArchive(device=self.device)
            meet_lookup = MeetLookup(meet_archive)
            root_hash = self._compute_state_hashes(self.solved_state.unsqueeze(0))
            meet_archive.add_layer_hashes(root_hash, depth=0)
            total_states = 1

            for _ in range(1, self.max_steps + 1):
                if bwd_archive.depth >= self.max_steps:
                    break
                frontier = bwd_archive.frontier
                with self._profiler.section("expand", get_states=lambda: int(frontier.size(0))):
                    children, parents, moves = self.kernel.expand(frontier, "backward")
                with self._profiler.section("unique", get_states=lambda: int(children.size(0))):
                    children, parents, moves, hashes = self._unique_by_hash(children, parents, moves)
                with self._profiler.section("meet_lookup", get_states=lambda: int(hashes.size(0))):
                    seen = meet_lookup.membership_mask(hashes)
                keep = (~seen).nonzero(as_tuple=True)[0]
                if keep.numel() == 0:
                    break
                children = children.index_select(0, keep).contiguous()
                parents = parents.index_select(0, keep)
                moves = moves.index_select(0, keep)
                hashes = hashes.index_select(0, keep)
                new_depth = bwd_archive.depth + 1
                with self._profiler.section("lower_bound", get_states=lambda: int(children.size(0))):
                    children, parents, moves, hashes = self._prune_by_lower_bound(
                        children, parents, moves, hashes,
                        new_depth, "backward", self.max_steps + 1,
                    )
                if children.numel() == 0:
                    break
                next_size = int(children.size(0))
                if total_states + next_size > self.backward_max_states:
                    break
                bwd_archive.add_layer(children, parents, moves)
                meet_archive.add_layer_hashes(hashes, depth=bwd_archive.depth)
                total_states += next_size

        self.search_stats["backward_archive_states"] = total_states
        return bwd_archive, meet_archive, meet_lookup

    def solve(
        self,
        start_state: torch.Tensor,
        model: BaseModel,
    ) -> Tuple[bool, int, str]:
        self.reset()
        self.model = model
        self._profiler.sync_for_timing()
        solve_start = time()
        self.adapter.prepare_search(start_state)

        fwd_archive = LayerArchive(start_state)
        fwd_meet = MeetArchive(device=self.device)
        fwd_lookup = MeetLookup(fwd_meet)
        start_hash = self._compute_state_hashes(start_state.unsqueeze(0))
        fwd_meet.add_layer_hashes(start_hash, depth=0)
        self.fwd_history.add(start_hash)

        if torch.equal(start_state, self.solved_state):
            self._accept_candidate(torch.empty(0, dtype=torch.long, device=self.device), "already_solved")
            self._profiler.sync_for_timing()
            self.search_stats["search_time"] = time() - solve_start
            return True, 0, ""

        if self.backward_mode == "off":
            bwd_archive = LayerArchive(self.solved_state)
            bwd_meet = MeetArchive(device=self.device)
            bwd_lookup = MeetLookup(bwd_meet)
            bwd_meet.add_layer_hashes(
                self._compute_state_hashes(self.solved_state.unsqueeze(0)),
                depth=0,
            )
        elif self.backward_mode == "bfs":
            bwd_archive, bwd_meet, bwd_lookup = self._prebuild_bfs_archive()
        else:
            bwd_archive = LayerArchive(self.solved_state)
            bwd_meet = MeetArchive(device=self.device)
            bwd_lookup = MeetLookup(bwd_meet)
            solved_hash = self._compute_state_hashes(self.solved_state.unsqueeze(0))
            bwd_meet.add_layer_hashes(solved_hash, depth=0)
            self.bwd_history.add(solved_hash)

        fwd_exhausted = False
        bwd_exhausted = False

        for step in range(1, self.max_steps + 1):
            context = SchedulerContext(
                step=step,
                forward_depth=fwd_archive.depth,
                backward_depth=bwd_archive.depth,
                path_len_limit=self._path_len_limit(),
                fwd_exhausted=fwd_exhausted,
                bwd_exhausted=bwd_exhausted,
            )
            actions = self.scheduler.make_plan(context)
            if not actions:
                break

            for action in actions:
                path_len_limit = self._path_len_limit()
                if action == "forward" and fwd_archive.depth + 1 >= path_len_limit:
                    continue
                if action == "backward" and bwd_archive.depth + 1 >= path_len_limit:
                    continue
                if action == "forward":
                    frontier = fwd_archive.frontier
                    with self._profiler.section("expand", get_states=lambda: int(frontier.size(0))):
                        children, parents, moves = self.kernel.expand(frontier, "forward")
                    self.search_stats["total_states_explored"] += int(children.size(0))
                    with self._profiler.section("unique", get_states=lambda: int(children.size(0))):
                        children, parents, moves, hashes = self._unique_by_hash(children, parents, moves)
                    with self._profiler.section("history_filter", get_states=lambda: int(children.size(0))):
                        children, parents, moves, hashes = self._filter_new(children, parents, moves, hashes, self.fwd_history)
                    new_depth = fwd_archive.depth + 1
                    with self._profiler.section("lower_bound", get_states=lambda: int(children.size(0))):
                        children, parents, moves, hashes = self._prune_by_lower_bound(
                            children, parents, moves, hashes,
                            new_depth, "forward", path_len_limit,
                        )
                    with self._profiler.section("beam_prune", get_states=lambda: int(children.size(0))):
                        children, parents, moves, hashes = self._prune_frontier(children, parents, moves, hashes)

                    if children.numel() == 0:
                        fwd_exhausted = True
                        continue
                    
                    fwd_archive.add_layer(children, parents, moves)
                    fwd_meet.add_layer_hashes(hashes, depth=fwd_archive.depth)
                    self.fwd_history.add(hashes)

                    if self.backward_mode == "off":
                        solved = (children == self.solved_state).all(dim=1)
                        if solved.any():
                            idx = int(solved.nonzero(as_tuple=True)[0][0])
                            codes = self.path_assembler.path_root_to_node(
                                fwd_archive.parents_by_depth,
                                fwd_archive.moves_by_depth,
                                fwd_archive.depth,
                                idx,
                            )
                            self._accept_candidate(codes, "solved_on_forward")
                    else:
                        with self._profiler.section("meet_lookup", get_states=lambda: int(children.size(0))):
                            meet = self._find_valid_meeting(children, hashes, bwd_archive, bwd_lookup)
                        if meet is not None:
                            f_idx, b_depth, b_idx = meet
                            if fwd_archive.depth + b_depth >= path_len_limit:
                                continue
                            codes = self.path_assembler.build_solution_codes(
                                fwd_archive,
                                fwd_archive.depth,
                                f_idx,
                                bwd_archive,
                                b_depth,
                                b_idx,
                            )
                            self._accept_candidate(codes, "meeting_found")

                if action == "backward" and self.backward_mode == "beam":
                    frontier = bwd_archive.frontier
                    with self._profiler.section("expand", get_states=lambda: int(frontier.size(0))):
                        children, parents, moves = self.kernel.expand(frontier, "backward")
                    self.search_stats["total_states_explored"] += int(children.size(0))
                    with self._profiler.section("unique", get_states=lambda: int(children.size(0))):
                        children, parents, moves, hashes = self._unique_by_hash(children, parents, moves)
                    with self._profiler.section("history_filter", get_states=lambda: int(children.size(0))):
                        children, parents, moves, hashes = self._filter_new(children, parents, moves, hashes, self.bwd_history)
                    new_depth = bwd_archive.depth + 1
                    with self._profiler.section("lower_bound", get_states=lambda: int(children.size(0))):
                        children, parents, moves, hashes = self._prune_by_lower_bound(
                            children, parents, moves, hashes,
                            new_depth, "backward", path_len_limit,
                        )
                    with self._profiler.section("beam_prune", get_states=lambda: int(children.size(0))):
                        children, parents, moves, hashes = self._prune_frontier(children, parents, moves, hashes)
                    
                    if children.numel() == 0:
                        bwd_exhausted = True
                        continue
                    bwd_archive.add_layer(children, parents, moves)
                    bwd_meet.add_layer_hashes(hashes, depth=bwd_archive.depth)
                    self.bwd_history.add(hashes)
                    with self._profiler.section("meet_lookup", get_states=lambda: int(children.size(0))):
                        meet = self._find_valid_meeting(children, hashes, fwd_archive, fwd_lookup)

                    if meet is not None:
                        b_idx, f_depth, f_idx = meet
                        if f_depth + bwd_archive.depth >= path_len_limit:
                            continue
                        codes = self.path_assembler.build_solution_codes(
                            fwd_archive,
                            f_depth,
                            f_idx,
                            bwd_archive,
                            bwd_archive.depth,
                            b_idx,
                        )
                        self._accept_candidate(codes, "meeting_found")

        self._profiler.sync_for_timing()
        self.search_stats["search_time"] = time() - solve_start
        if self._best_solution_codes is None:
            if fwd_exhausted and (self.backward_mode == "off" or bwd_exhausted):
                self.search_stats["termination_reason"] = "both_frontiers_exhausted"
            else:
                self.search_stats["termination_reason"] = "max_steps_reached"
            return False, self.max_steps, ""
        best_len = int(self.objective.best_solution_len or 0)
        if not self.search_stats["termination_reason"]:
            self.search_stats["termination_reason"] = "best_after_budget"
        return True, best_len, self._codes_to_solution(self._best_solution_codes)
