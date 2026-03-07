"""Simple beam search solver using ML guidance."""

import os
import logging

import numpy as np
import torch

from typing import List, Optional, Tuple, Dict, Union
from time import time

from src.models.base_model import BaseModel
from src.solvers.base_solver import BaseSolver
from src.solvers.hash_history import HashHistory
from src.puzzles import PuzzleSpec

logger = logging.getLogger(__name__)


class BeamSearchSolver(BaseSolver):
    """Simple beam search solver using ML guidance."""
    
    def __init__(
        self,
        state_size: int,
        device: torch.device,
        puzzle_spec: PuzzleSpec,
        beam_width: int,
        max_steps: int,
        use_x_rule: bool = False,
        target_neighborhood_radius: int = 0,
        hashes_batch_size: int = 1_000_000,
        filter_batch_size: int = 1_000_000,
        predict_batch_size: int = 1e10,
        nbt_depth: int = 5,
        pancake_max_moves: int = 0,
        randomize_ties: bool = True,
        gap_lb_only: bool = False,
        diversity_buckets: int = 0,
        verbose: int = 0,
        *,
        adapter,
    ):
        """Initialize beam search solver."""
        super().__init__(puzzle_spec, device, model=None, verbose=verbose)
        self.adapter = adapter
        self.beam_width = beam_width
        self.max_steps = max_steps
        self.use_x_rule = use_x_rule
        self.target_neighborhood_radius = target_neighborhood_radius
        self.hashes_batch_size = hashes_batch_size
        self.filter_batch_size = filter_batch_size
        self.predict_batch_size = predict_batch_size
        self.nbt_depth = nbt_depth
        self.pancake_max_moves = pancake_max_moves
        self.gap_lb_only = bool(gap_lb_only)
        self.randomize_ties = bool(randomize_ties)
        self.diversity_buckets = int(diversity_buckets)
        self.lb_gap_start = None

        self.puzzle_spec = puzzle_spec
        self.move_names = list(puzzle_spec.move_names)
        self.move_indices = puzzle_spec.move_indices
        self.inverse_moves = puzzle_spec.inverse_moves
        self.solved_state = puzzle_spec.solved_state
        self.n_moves = int(self.move_indices.size(0))

        self.x_move_code = (
            self.move_names.index("X") if "X" in self.move_names else None
        )
        
        # Pre-compute hash vector for efficient state hashing
        max_int = int(2**62)
        self.hash_vec = torch.randint(
            low=-max_int,
            high=max_int + 1,
            size=(self.state_size,),
            dtype=torch.int64,
            device=device
        ).contiguous()
        
        # Hash history: NBT component, puzzle-agnostic
        max_hashes_per_step = beam_width
        if self.nbt_depth > 0:
            self.max_history_size = max(
                max_hashes_per_step, self.nbt_depth * max_hashes_per_step
            )
        else:
            self.max_history_size = 0
        self.hash_history = HashHistory(
            device=device,
            max_history_size=self.max_history_size,
            nbt_depth=self.nbt_depth,
            hashes_batch_size=self.hashes_batch_size,
        )
        
        # Memory tracking
        self.initial_memory = 0
        self.peak_memory = 0
        
        # Precompute target neighborhood if radius > 0
        if target_neighborhood_radius > 0:
            self.target_neighborhood, self.target_paths = self._get_target_neighborhood(
                target_neighborhood_radius
            )
            self.log_info(f"Got target neighborhood with {len(self.target_neighborhood)} states")
        else:
            self.target_neighborhood = None
            self.target_paths = None
            
        # Search statistics
        self.search_stats.update({
            'beam_width': beam_width,
            'max_steps': max_steps,
            'use_x_rule': use_x_rule,
            'target_neighborhood_radius': target_neighborhood_radius,
            'nbt_depth': nbt_depth,
            'Total hashes in history': 0,
            'Total hashes ever seen': 0,
            'peak_memory_gb': 0,
            'Termination reason': '',
            'pruned_states_total': 0,
            'pruned_states_min': float('inf'),
            'pruned_states_max': 0,
            'pruned_states_avg': 0,
            'pruning_steps': 0,
            'first_pruning_step': -1,  # Step when pruning first occurs (-1 means no pruning yet)
            'last_step_pruned': 0,     # Number of states pruned in the last step
        })

        # Counter for total hashes ever seen
        self.total_hashes_ever_seen = 0

    def _track_memory(self):
        """Track current GPU memory usage and update peak memory."""
        current = torch.cuda.memory_allocated() / (1024**3)  # GB
        self.peak_memory = max(self.peak_memory, current)
        self.search_stats['peak_memory_gb'] = self.peak_memory
        return current

    def reset(self) -> None:
        """Reset solver state for a new problem."""
        self.search_stats.update({
            'total_states_explored': 0,
            'search_time': 0,
            'path_found': False,
            'Total hashes in history': 0,
            'Total hashes ever seen': 0,
            'peak_memory_gb': 0,
            'pruned_states_total': 0,
            'pruned_states_min': float('inf'),
            'pruned_states_max': 0,
            'pruned_states_avg': 0,
            'pruning_steps': 0,
            'first_pruning_step': -1,  # Step when pruning first occurs (-1 means no pruning yet)
            'last_step_pruned': 0,     # Number of states pruned in the last step
        })
        # Reset hash history
        self.hash_history.reset()
        self.total_hashes_ever_seen = 0
        
        # Reset memory tracking
        self.initial_memory = torch.cuda.memory_allocated() / (1024**3)  # GB
        self.peak_memory = self.initial_memory

    def solve(self, start_state: torch.Tensor, model: BaseModel) -> Tuple[bool, int, str]:
        """
        Perform beam search using ML guidance. For each step:
        1. Expand states with all possible moves
        2. Check if solution is found
        3. Filter states by hash and rules (in batches)
        4. Prune beam to keep only top beam_width states
        """
        self.reset()
        self.model = model
        self.start_state = start_state.clone().to(dtype=self.state_dtype)
        
        max_steps = int(self.max_steps)
        if self.gap_lb_only:
            if not self.adapter.use_pruned_hashes_for_nbt():
                raise ValueError("gap_lb_only is supported only for pancake puzzles")
            if int(self.target_neighborhood_radius) != 0:
                raise ValueError("gap_lb_only requires target_neighborhood_radius=0")
            gap0 = int(
                self.adapter.lower_bound(
                    self.start_state.unsqueeze(0),
                    self.solved_state,
                    "forward",
                ).item()
            )
            if max_steps < gap0:
                raise ValueError(
                    f"gap_lb_only requires max_steps >= gap_lb ({max_steps} < {gap0})"
                )
            self.lb_gap_start = gap0
            max_steps = gap0
        
        # Initialize with start state
        start_hash = self._compute_state_hashes(self.start_state.unsqueeze(0))
        
        # Add initial hash to history if nbt_depth > 0
        if self.nbt_depth > 0:
            self.hash_history.add(start_hash)
        
        current_states = self.start_state.unsqueeze(0)
        self.log_info(f"Initial state: {start_state.cpu().numpy()}")
        
        parent_indices = []
        move_indices = []
        search_start = time()
        
        for step in range(1, max_steps + 1):
            self.log_info(f"\n{'='*10} Step {step} {'='*10}")
            
            # Store current step in search_stats
            self.search_stats['current_step'] = step
            
            # 1. Expand all states at once - use GPU efficiently
            next_states, parents, next_moves = self._bulk_expand(current_states)
            
            self._track_memory()  # Track memory after expansion
            
            self.search_stats['total_states_explored'] += len(next_states)
            
            # Find unique states in this expansion to avoid duplicates
            unique_states, unique_indices = self._get_unique_states(next_states)
            
            # Update related tensors accordingly
            if unique_states.size(0) < next_states.size(0):
                self.log_info(
                    f"Removed {next_states.size(0) - unique_states.size(0)} "
                    f"duplicate states in expansion"
                )
                
                next_states = unique_states
                parents = parents[unique_indices]
                next_moves = next_moves[unique_indices]
                
                del unique_states, unique_indices
                self._track_memory()  # Track memory after deduplication
            
            # 2. Check for solution
            found, solution_idx, target_path = self._check_solution(next_states)
            
            if found:
                parent_indices.append(parents)
                move_indices.append(next_moves)
                additional_steps = target_path.numel() if target_path is not None else 0
                
                self.search_stats.update({
                    'search_time': time() - search_start,
                    'path_found': True,
                    'solution_length': step + additional_steps,
                    'Termination reason': "Solution found!"
                })
                
                if additional_steps > 0:
                    self.log_info(f"Solution found in target neighborhood after {step} steps")
                    self.log_info(f"Additional {additional_steps} steps to reach solved state")
                
                self._display_search_stats()
                return True, step + additional_steps, self.reconstruct_solution(
                    parent_indices, move_indices, solution_idx, target_path
                )
            
            # 3. Filter states in batches
            all_filtered_states = []
            all_filtered_parents = []
            all_filtered_moves = []
            all_new_hashes = []

            remaining_steps = max_steps - step
            
            for batch_start in range(0, len(next_states), self.filter_batch_size):
                batch_end = min(batch_start + self.filter_batch_size, len(next_states))
                
                filtered_states, filtered_parents, filtered_moves, new_hashes = self._filter_states(
                    next_states[batch_start:batch_end],
                    next_moves[batch_start:batch_end],
                    parents[batch_start:batch_end],
                    current_states,
                    remaining_steps,
                )
                
                self._track_memory()  # Track memory after filtering batch
                
                if filtered_states.shape[0] > 0:
                    all_filtered_states.append(filtered_states)
                    all_filtered_parents.append(filtered_parents)
                    all_filtered_moves.append(filtered_moves)
                    if new_hashes.numel() > 0:
                        all_new_hashes.append(new_hashes)
            
            # Total hashes ever seen (for stats); keep for history when not gap_lb_only
            step_hashes_from_filter = (
                torch.cat(all_new_hashes) if all_new_hashes else
                torch.empty(0, dtype=torch.int64, device=self.device)
            )
            if step_hashes_from_filter.numel() > 0:
                self.total_hashes_ever_seen += step_hashes_from_filter.size(0)
            
            # Free memory
            del next_states, next_moves, parents
            if 'all_new_hashes' in locals():
                del all_new_hashes

            # Combine filtered results
            if all_filtered_states:
                next_states = torch.cat(all_filtered_states)
                parents = torch.cat(all_filtered_parents)
                next_moves = torch.cat(all_filtered_moves)
                
                del all_filtered_states, all_filtered_parents, all_filtered_moves
                self._track_memory()  # Track memory after combining filtered results
            else:
                next_states = torch.empty(
                    (0, self.state_size), device=self.device, dtype=self.state_dtype
                )
                parents = torch.empty(0, device=self.device, dtype=torch.long)
                next_moves = torch.empty(0, device=self.device, dtype=torch.int16)
            
            # 4. Prune beam if needed
            if next_states.shape[0] > self.beam_width:
                if self.search_stats['first_pruning_step'] == -1:
                    self.search_stats['first_pruning_step'] = step
                    self.log_info(f"First pruning at step {step} with {next_states.shape[0]} states")
                
                current_states, parents, next_moves = self._prune_beam(
                    next_states, parents, next_moves, current_states
                )
                self._track_memory()  # Track memory after pruning
            else:
                current_states = next_states
            
            # Update hash history: when adapter says so, add only kept states
            if self.nbt_depth > 0:
                if self.adapter.use_pruned_hashes_for_nbt():
                    step_hashes = self._compute_state_hashes(current_states)
                else:
                    step_hashes = step_hashes_from_filter
                if step_hashes.numel() > 0:
                    self.log_info(f"Before updating history: buffer_size={self.hash_history.buffer_size}")
                    self.hash_history.add(step_hashes)
                    self.search_stats['Total hashes in history'] = self.hash_history.buffer_size
                    self.log_info(f"After updating history: buffer_size={self.hash_history.buffer_size}")
                self._track_memory()
            else:
                self.search_stats['Total hashes in history'] = 0
            self.search_stats['Total hashes ever seen'] = self.total_hashes_ever_seen

            parent_indices.append(parents)
            move_indices.append(next_moves)
            
            self.log_info(f"End of step:")
            self.log_info(f"  Number of states: {current_states.shape[0]}")
            self.log_info(f"  Current hash history size: {self.search_stats['Total hashes in history']}")
            self.log_info(f"  Current memory usage: {self._track_memory():.2f} GB")
        
        self.log_info(f"\nSearch terminated after reaching max steps: {self.max_steps}")
        self.search_stats['search_time'] = time() - search_start
        self.search_stats['Termination reason'] = "Max steps reached"
        self._display_search_stats()
        return False, self.max_steps, ""

    def _filter_states(
        self,
        next_states: torch.Tensor,
        next_moves: torch.Tensor,
        parents: torch.Tensor,
        current_states: torch.Tensor,
        remaining_steps: int,
        apply_budget: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Filter states based on hash (already visited states) or rules (X rule, etc.).
        apply_budget: if False, skip step-budget pruning (e.g. for backward direction
        in bidirectional, where gaps are distance to solved, not to start).
        """
        if apply_budget:
            budget_mask = self.adapter.filter_budget(
                next_states,
                remaining_steps,
                gap_lb_only=self.gap_lb_only,
                target_neighborhood_radius=self.target_neighborhood_radius,
                state_size=self.state_size,
            )
            keep_idx = torch.where(budget_mask)[0]
            if keep_idx.size(0) == 0:
                empty_states = torch.empty(
                    (0, self.state_size), dtype=next_states.dtype, device=self.device
                )
                empty_parents = torch.empty(0, dtype=parents.dtype, device=self.device)
                empty_moves = torch.empty(0, dtype=next_moves.dtype, device=self.device)
                empty_hashes = torch.empty(0, dtype=torch.int64, device=self.device)
                return empty_states, empty_parents, empty_moves, empty_hashes

            if keep_idx.size(0) < next_states.size(0):
                next_states = torch.index_select(next_states, 0, keep_idx).contiguous()
                next_moves = torch.index_select(next_moves, 0, keep_idx)
                parents = torch.index_select(parents, 0, keep_idx)

        # Calculate hashes for this batch
        state_hashes = self._compute_state_hashes(next_states)

        # Check if hashes are in history
        is_in_history = self.hash_history.check(state_hashes)
        is_new = ~is_in_history

        valid_moves = is_new

        # Filter states based on rules (X rule, etc.)
        if self.use_x_rule and self.x_move_code is not None:
            is_x_move = next_moves == self.x_move_code

            # Optimized vectorized operation
            parent_indices = parents % current_states.size(0)
            first_vals = torch.gather(current_states[:, 0], 0, parent_indices)
            second_vals = torch.gather(current_states[:, 1], 0, parent_indices)
            first_smaller = first_vals < second_vals

            valid_moves &= ~(is_x_move & first_smaller)

        # Return only new hashes for adding to history
        new_hashes = (
            state_hashes[valid_moves]
            if torch.any(valid_moves)
            else torch.empty(0, dtype=torch.int64, device=self.device)
        )

        # Log filtering details if verbose
        if self.verbose > 1:
            self._log_move_filtering(
                current_states, next_states, parents,
                next_moves, valid=valid_moves, is_visited=~is_new
            )

        # Select valid states efficiently
        valid_indices = torch.where(valid_moves)[0]
        if valid_indices.numel() > 0:
            filtered_states = torch.index_select(
                next_states, 0, valid_indices
            ).contiguous()
            filtered_parents = torch.index_select(parents, 0, valid_indices)
            filtered_moves = torch.index_select(next_moves, 0, valid_indices)
        else:
            filtered_states = torch.empty(
                (0, self.state_size), dtype=next_states.dtype, device=self.device
            )
            filtered_parents = torch.empty(0, dtype=parents.dtype, device=self.device)
            filtered_moves = torch.empty(0, dtype=next_moves.dtype, device=self.device)

        del is_in_history, is_new, valid_moves, valid_indices
        return filtered_states, filtered_parents, filtered_moves, new_hashes
        
    def _prune_beam(
        self,
        next_states: torch.Tensor,
        parents: torch.Tensor,
        moves: torch.Tensor,
        current_states: torch.Tensor,
        keep_width: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prune beam to keep only top states according to model predictions.
        keep_width: if set, keep at most this many (for chunked expansion).
        """
        k_target = int(keep_width) if keep_width is not None else self.beam_width
        self.log_info(
            f"Pruning beam from {len(next_states)} to "
            f"{min(k_target, len(next_states))} states"
        )

        pruned_count = len(next_states) - min(k_target, len(next_states))
        self.search_stats["pruned_states_total"] += pruned_count
        if pruned_count > 0:
            self.search_stats["pruned_states_min"] = min(
                self.search_stats["pruned_states_min"], pruned_count
            )
        self.search_stats["pruned_states_max"] = max(
            self.search_stats["pruned_states_max"], pruned_count
        )
        self.search_stats["pruning_steps"] += 1
        if self.search_stats["pruning_steps"] > 0:
            self.search_stats["pruned_states_avg"] = (
                self.search_stats["pruned_states_total"]
                / self.search_stats["pruning_steps"]
            )
        self.search_stats["last_step_pruned"] = pruned_count

        num_states = int(next_states.size(0))
        chunk_size = min(int(self.predict_batch_size), num_states)

        scores = torch.empty(num_states, device=next_states.device,
                            dtype=torch.float32)

        with torch.no_grad():
            locked_mask = (
                self.adapter.locked_mask(next_states)
                if self.verbose > 1
                else None
            )

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                for i in range(0, num_states, chunk_size):
                    end = min(i + chunk_size, num_states)
                    s = next_states[i:end]
                    p = parents[i:end]
                    m = moves[i:end]
                    scores[i:end] = self._score_candidates(
                        s, p, m, current_states
                    ).to(torch.float32)
                    if chunk_size > 100000:
                        del s, p, m

            if self.gap_lb_only:
                n_plus = num_states + 1
                scores = scores * n_plus + torch.rand(
                    num_states, device=scores.device, dtype=torch.float32
                )

        k = int(min(k_target, num_states))
        if self.diversity_buckets > 0:
            div_key = self.adapter.diversity_key(next_states)
            bucket = div_key
            rank = torch.empty(num_states, device=next_states.device, dtype=torch.int64)
            rank[torch.argsort(scores)] = torch.arange(
                num_states, device=next_states.device, dtype=torch.int64
            )
            key = bucket * (num_states + 1) + rank
            top_indices = torch.topk(key, k=k, largest=False).indices
        elif not self.randomize_ties:
            top_indices = torch.topk(scores, k=k, largest=False, sorted=False).indices
        else:
            topk_out = torch.topk(scores, k=k, largest=False, sorted=True)
            cutoff_value = topk_out.values[-1]
            strict_mask = scores < cutoff_value
            strict_indices = torch.nonzero(strict_mask).squeeze(1)
            n_strict = int(strict_indices.numel())
            need = int(k - n_strict)
            if need <= 0:
                top_indices = strict_indices[:k]
            else:
                tie_mask = scores == cutoff_value
                tie_indices = torch.nonzero(tie_mask).squeeze(1)
                perm = torch.randperm(int(tie_indices.numel()),
                                    device=tie_indices.device)
                chosen_ties = tie_indices.index_select(0, perm[:need])
                top_indices = torch.cat([strict_indices, chosen_ties], dim=0)

        if self.verbose > 1:
            self._log_pruning_decisions(
                next_states, parents, moves, scores, top_indices, current_states
            )

        with torch.no_grad():
            if locked_mask is not None:
                kept_locked = locked_mask.index_select(0, top_indices)
                self.log_info(
                    f"Locked states kept: {kept_locked.sum().item()}/{top_indices.size(0)} "
                    f"(before prune: {locked_mask.sum().item()}/{next_states.size(0)})"
                )

        pruned_states = torch.index_select(next_states, 0, top_indices)
        pruned_parents = torch.index_select(parents, 0, top_indices)
        pruned_moves = torch.index_select(moves, 0, top_indices)

        del scores, top_indices
        return pruned_states, pruned_parents, pruned_moves

    def _bulk_expand(
        self, states: torch.Tensor, force_all_moves: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expand states: adapter policy + core apply via gather."""
        max_moves = getattr(self.adapter, "moves_per_state", self.n_moves)
        policy_options = {
            "max_moves": max_moves,
            "only_decreasing": self.gap_lb_only and not force_all_moves,
            "only_increasing": False,
        }
        if force_all_moves:
            policy_options["max_moves"] = self.n_moves
            policy_options["only_decreasing"] = False

        move_codes, valid_mask = self.adapter.move_codes_and_mask(
            states, "forward", **policy_options
        )
        return self._apply_move_codes(states, move_codes, valid_mask, inverse=False)

    def _apply_move_codes(
        self,
        states: torch.Tensor,
        move_codes: torch.Tensor,
        valid_mask: torch.Tensor,
        inverse: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply move codes via gather. Core logic, puzzle-agnostic."""
        pairs = valid_mask.nonzero(as_tuple=False)
        parent_ids = pairs[:, 0].to(dtype=torch.long)
        col_ids = pairs[:, 1]
        codes = move_codes[parent_ids, col_ids]

        if inverse:
            indices = self.move_indices[self.inverse_moves[codes]]
        else:
            indices = self.move_indices[codes]

        src = states.index_select(0, parent_ids)
        children = torch.gather(src, 1, indices).contiguous()
        return children, parent_ids, codes.to(dtype=torch.int16)

    def _score_candidates(
        self,
        next_states: torch.Tensor,
        parents: torch.Tensor,
        moves: torch.Tensor,
        current_states: torch.Tensor,
    ) -> torch.Tensor:
        """Base score from model + adapter.score_extra."""
        base = (
            self.model.predict(next_states).to(torch.float32)
            if self.model is not None
            else torch.zeros(
                next_states.size(0),
                device=next_states.device,
                dtype=torch.float32,
            )
        )
        extra = self.adapter.score_extra(
            next_states, parents, moves, current_states
        )
        return base + extra

    def _compute_state_hashes(self, states: torch.Tensor) -> torch.Tensor:
        """Compute unique hashes for states using vectorized operations."""
        n_states = states.size(0)
        hashes = torch.empty(n_states, dtype=torch.int64, device=self.device)
        batch_size = self.hashes_batch_size

        i = 0
        while i < n_states:
            end = min(i + batch_size, n_states)
            batch = states[i:end]
            try:
                hashes[i:end] = torch.sum(
                    batch.to(dtype=torch.int64) * self.hash_vec.unsqueeze(0), dim=1
                )
                i = end
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                new_batch_size = max(10_000, batch_size // 2)
                if new_batch_size >= batch_size:
                    logger.error(
                        "OOM in _compute_state_hashes with batch_size=%s, n_states=%s",
                        batch_size, n_states,
                    )
                    raise
                logger.warning(
                    "OOM in _compute_state_hashes: reducing batch %s -> %s (n_states=%s)",
                    batch_size, new_batch_size, n_states,
                )
                self.hashes_batch_size = new_batch_size
                batch_size = new_batch_size
        return hashes
    
    def _get_unique_states(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return only unique states from input tensor by using efficient hashing.
        Much faster than torch.unique(states, dim=0).
        
        Returns:
            Tuple of (unique_states, unique_indices)
            - unique_states: Tensor containing only unique states
            - unique_indices: Indices of the first occurrence of each unique state
        """
        if states.size(0) == 0:
            # Handle empty tensor case
            return states, torch.empty(0, dtype=torch.long, device=self.device)
        
        # Compute hash for each state using our precomputed hash vector
        hashes = self._compute_state_hashes(states)
        n = hashes.size(0)

        # Sort hashes to identify unique elements (argsort + gather uses less memory)
        sort_indices = torch.argsort(hashes)
        sorted_hashes = hashes[sort_indices]
        mask = torch.ones(n, dtype=torch.bool, device=self.device)
        mask[1:] = sorted_hashes[1:] != sorted_hashes[:-1]
        unique_indices = sort_indices[mask].clone()
        num_unique = unique_indices.size(0)

        del hashes, sorted_hashes, sort_indices, mask

        # Build unique_states: copy in chunks to avoid OOM on large single index
        unique_states = torch.empty(
            (num_unique, states.size(1)),
            dtype=states.dtype,
            device=self.device,
        )
        copy_batch = self.hashes_batch_size
        for start in range(0, num_unique, copy_batch):
            end = min(start + copy_batch, num_unique)
            unique_states[start:end] = states[unique_indices[start:end]]

        return unique_states, unique_indices

    def _apply_moves(
        self, state: torch.Tensor, moves: Union[torch.Tensor, int]
    ) -> torch.Tensor:
        """Apply a sequence of moves or single move to a state."""
        new_state = state

        # Handle single move (int or 0-d tensor)
        if isinstance(moves, (int, torch.Tensor)) and not isinstance(
            moves, (list, tuple)
        ):
            move_code = int(moves) if isinstance(moves, int) else int(moves.item())
            idx = self.move_indices[move_code]
            return new_state.index_select(0, idx)

        # Sequence of moves
        for move in moves:
            move_code = int(move) if isinstance(move, int) else int(move.item())
            idx = self.move_indices[move_code]
            new_state = new_state.index_select(0, idx)

        return new_state
    
    def _check_solution(self, states: torch.Tensor) -> Tuple[bool, int, torch.Tensor]:
        """Check if any states match solution criteria."""
        if self.target_neighborhood is not None:
            # Check if any states are in target neighborhood
            state_hashes = self._compute_state_hashes(states)
            matches = torch.isin(state_hashes, self.target_neighborhood)
            found = torch.any(matches)
            
            if found:
                idx = torch.where(matches)[0][0].item()
                hash_val = state_hashes[idx].item()
                stored_state, stored_path = self.target_paths[hash_val]

                self.log_info(f"Found state in target neighborhood: {stored_state}")
                
                # Verify the state matches (in case of hash collision)
                if not torch.all(states[idx] == stored_state):
                    self.log_warning(f"Hash collision detected for state: {states[idx]}")
                    # Hash collision - search for actual matching state
                    for h, (s, p) in self.target_paths.items():
                        if torch.all(states[idx] == s):
                            return True, idx, p
                    return False, -1, torch.empty(
                        0, dtype=self.state_dtype, device=self.device
                    )
                
                return True, idx, stored_path
            
            return False, -1, torch.empty(
                0, dtype=self.state_dtype, device=self.device
            )
        else:
            # Check for exact match with solved state
            matches = torch.all(states == self.solved_state, dim=1)
            found = torch.any(matches)
            idx = torch.where(matches)[0][0].item() if found else -1
            return found, idx, torch.empty(
                0, dtype=self.state_dtype, device=self.device
            )
        
    def reconstruct_solution(
        self,
        parent_indices: List[torch.Tensor],
        move_indices: List[torch.Tensor],
        solution_idx: int,
        target_path: torch.Tensor = None
    ) -> str:
        """Reconstruct solution path from stored moves and verify it."""
        # Get moves in reverse order
        reverse_moves = []
        current_idx = solution_idx
        
        # Get the last move
        reverse_moves.append(move_indices[-1][current_idx].item())
        current_idx = parent_indices[-1][current_idx].item()
        
        # Get previous moves
        for step in range(len(parent_indices)-1, 0, -1):
            reverse_moves.append(move_indices[step-1][current_idx].item())
            current_idx = parent_indices[step-1][current_idx].item()
        
        # Convert to tensor and reverse to get chronological order
        moves = torch.tensor(
            reverse_moves[::-1], dtype=self.state_dtype, device=self.device
        )
        
        # If we found a state in target neighborhood, append its path
        if target_path is not None and target_path.numel() > 0:
            moves = torch.cat([moves, target_path])
        
        # Verify solution
        if self.verbose > 0:
            current_state = self.start_state.clone()
            self.log_info("\nVerifying solution:")
            self.log_info(f"Start state: {current_state}")
            
            for i, move in enumerate(moves):
                move_name = self.move_names[move.item()]
                current_state = self._apply_moves(current_state, move)
                self.log_info(f"After move {i+1} ({move_name}): {current_state}")
            
            if torch.all(current_state == self.solved_state):
                self.log_info("Solution verified!")
            else:
                self.log_warning(f"Invalid solution! Final state {current_state} != {self.solved_state}")
        
        # Convert moves to string representation
        return '.'.join(self.move_names[m.item()] for m in moves)
        
    def _get_target_neighborhood(self, radius: int) -> Tuple[torch.Tensor, Dict[int, Tuple[torch.Tensor, torch.Tensor]]]:
        """Precompute states within given radius of target state using BFS."""
        initial_state = self.solved_state.unsqueeze(0)
        initial_hash = int(self._compute_state_hashes(initial_state).item())

        states_dict: Dict[int, Tuple[torch.Tensor, List[int]]] = {
            initial_hash: (initial_state[0].clone(), [])
        }
        frontier_states = initial_state
        frontier_paths: List[List[int]] = [[]]
        inverse_moves_np = self.inverse_moves.cpu().numpy()

        self.log_info("Starting BFS from solved state")

        for depth in range(radius):
            if frontier_states.size(0) == 0:
                break

            next_states, parents, next_moves = self._bulk_expand(
                frontier_states, force_all_moves=True
            )
            n_next = int(next_states.size(0))
            if n_next == 0:
                break

            hashes_batch = self._compute_state_hashes(next_states)
            hashes_np = hashes_batch.cpu().numpy()
            parents_np = parents.cpu().numpy()
            next_moves_np = next_moves.cpu().numpy()

            visited_arr = np.fromiter(states_dict.keys(), dtype=np.int64, count=len(states_dict))
            is_new = ~np.isin(hashes_np, visited_arr)
            new_indices = np.flatnonzero(is_new)
            n_new = int(new_indices.size)

            if n_new == 0:
                break

            new_paths = []
            for k in range(n_new):
                idx = int(new_indices[k])
                hash_val = int(hashes_np[idx])
                parent_idx = int(parents_np[idx])
                move = int(next_moves_np[idx])
                inverse_move = int(inverse_moves_np[move])
                new_path = [inverse_move] + frontier_paths[parent_idx].copy()
                state_row = next_states[idx].clone()
                states_dict[hash_val] = (state_row, new_path)
                new_paths.append(new_path)

            idx_t = torch.as_tensor(new_indices, device=self.device, dtype=torch.long)
            frontier_states = next_states[idx_t].contiguous()
            frontier_paths = new_paths
            self.log_info(f"Target neighborhood depth {depth + 1}: {len(states_dict)} states")
        self.log_warning(f"Overall states in target neighborhood: {len(states_dict)}")
        
        # Convert paths to tensors
        final_states_dict = {}
        for hash_val, (state, path) in states_dict.items():
            path_tensor = torch.tensor(
                path, dtype=self.state_dtype, device=self.device
            )
            final_states_dict[hash_val] = (state, path_tensor)
        
        hashes = torch.tensor(
            list(final_states_dict.keys()), dtype=torch.int64, device=self.device
        )
        
        return hashes, final_states_dict

    def _get_inverse_move(self, move: int) -> int:
        """Return inverse move code."""
        return int(self.inverse_moves[int(move)].item())

    def _log_move_filtering(
        self,
        current_states: torch.Tensor,
        new_states: torch.Tensor,
        parents: torch.Tensor,
        moves: torch.Tensor,
        valid: torch.Tensor,
        is_visited: torch.Tensor
    ) -> None:
        """Log detailed move filtering information."""
        self.log_info(f"Filtering {len(new_states)} new states:")
        for j in range(len(new_states)):
            parent_idx = parents[j].item()
            parent_state = current_states[parent_idx % len(current_states)]
            move_code = moves[j].item()
            move_name = self.move_names[move_code]
            new_state = new_states[j]
            
            # Verify move application using a single-move tensor
            expected_state = self._apply_moves(parent_state, move_code)
            
            if not torch.all(expected_state == new_state):
                self.log_debug(
                    f"WARNING: Move application mismatch for {move_name}:"
                    f"\n  Parent:   {parent_state.cpu().numpy()}"
                    f"\n  Got:      {new_state.cpu().numpy()}"
                    f"\n  Expected: {expected_state.cpu().numpy()}"
                )
            
            if not valid[j]:
                reason = "visited state" if is_visited[j] else "X rule"
                self.log_debug(
                    f"From parent {parent_idx} {parent_state.cpu().numpy()} → "
                    f"Move {move_name} → {new_state.cpu().numpy()} ❌ - {reason}"
                )
            else:
                self.log_debug(
                    f"From parent {parent_idx} {parent_state.cpu().numpy()} → "
                    f"Move {move_name} → {new_state.cpu().numpy()} ✅"
                )

    def _log_pruning_decisions(
        self,
        next_states: torch.Tensor,
        parents: torch.Tensor,
        moves: torch.Tensor,
        model_distances: torch.Tensor,
        top_indices: torch.Tensor,
        current_states: torch.Tensor
    ) -> None:
        """Log detailed information about pruning decisions efficiently."""
        # Only transfer to CPU what we need for logging
        with torch.no_grad():
            # Get min/max distances first - single GPU operation
            min_dist, max_dist = model_distances.min().item(), model_distances.max().item()
            
            # Convert indices to set once
            kept_indices = top_indices.cpu().numpy()
            kept_set = set(kept_indices)
            
            self.log_debug("\nPruning decisions:")
            self.log_debug(f"Total states: {len(next_states)}, Keeping: {len(kept_indices)}")
            self.log_debug(f"Distance range: {min_dist:.1f} to {max_dist:.1f}")
            
            # Transfer to CPU in batches to avoid memory spikes
            batch_size = 1000
            for j in range(0, len(next_states), batch_size):
                end_idx = min(j + batch_size, len(next_states))
                
                # Get batch of states and their info
                next_states_np = next_states[j:end_idx].cpu().numpy()
                parents_np = parents[j:end_idx].cpu().numpy()
                moves_np = moves[j:end_idx].cpu().numpy()
                distances_np = model_distances[j:end_idx].cpu().numpy()
                
                # Log details for each state in batch
                for k in range(end_idx - j):
                    idx = j + k
                    parent_idx = parents_np[k]
                    move_type = moves_np[k]
                    move = self.move_names[move_type]
                    parent_state = current_states[parent_idx % len(current_states)].cpu().numpy()
                    new_state = next_states_np[k]
                    pred = distances_np[k]
                    kept = idx in kept_set
                        
                    self.log_debug(
                    f"From parent {parent_idx} {parent_state} → "
                    f"Move {move} → {new_state} "
                        f"pred {pred:.1f} {'✅' if kept else '❌'}"
                    )
                
    def _display_search_stats(self) -> None:
        """Display detailed search statistics."""
        self.log_info("\nSearch Statistics:")
        self.log_info("=" * 40)
        self.log_info(f"Total states explored: {self.search_stats['total_states_explored']}")
        self.log_info(f"Total hashes in history: {self.search_stats['Total hashes in history']}")
        self.log_info(f"Total hashes ever seen: {self.search_stats['Total hashes ever seen']}")
        self.log_info(f"Search time: {self.search_stats['search_time']:.3f} seconds")
        self.log_info(f"Termination reason: {self.search_stats['Termination reason']}")
        
        if self.search_stats['path_found']:
            self.log_info(f"Solution length: {self.search_stats['solution_length']}")
        
        self.log_info(f"Beam width: {self.search_stats['beam_width']}")
        self.log_info(f"Max steps: {self.search_stats['max_steps']}")
        self.log_info(f"X rule enabled: {self.search_stats['use_x_rule']}")
        self.log_info(f"Target neighborhood radius: {self.search_stats['target_neighborhood_radius']}")
        self.log_info(f"NBT depth: {self.search_stats['nbt_depth']}")
        if self.search_stats['pruning_steps'] > 0:
            self.log_info(f"Pruning statistics:")
            self.log_info(f"  Total states pruned: {self.search_stats['pruned_states_total']}")
            self.log_info(f"  Min states pruned per step: {self.search_stats['pruned_states_min']}")
            self.log_info(f"  Max states pruned per step: {self.search_stats['pruned_states_max']}")
            self.log_info(f"  Avg states pruned per step: {self.search_stats['pruned_states_avg']:.2f}")
            self.log_info(f"  First pruning at step: {self.search_stats['first_pruning_step'] if self.search_stats['first_pruning_step'] != -1 else 'N/A'}")
            self.log_info(f"  Last step pruned states: {self.search_stats['last_step_pruned']}")
        self.log_info("=" * 40)

