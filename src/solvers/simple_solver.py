import torch
from typing import List, Tuple, Dict, Any, Union
from time import time
from ..models.base_model import BaseModel
from .base_solver import BaseSolver
import numpy as np
import os

class BeamSearchSolver(BaseSolver):
    """Simple beam search solver using ML guidance."""
    
    def __init__(
        self,
        state_size: int,
        device: torch.device,
        beam_width: int,
        max_steps: int,
        use_x_rule: bool = False,
        target_neighborhood_radius: int = 0, # 0 for exact match
        hashes_batch_size: int = 1_000_000,
        filter_batch_size: int = 1_000_000,
        predict_batch_size: int = 1e10,
        history_window_size: int = 5,  # Store only last N steps of hash history
        verbose: int = 0 # 0=WARNING, 1=INFO, 2=DEBUG
    ):
        """Initialize beam search solver."""
        super().__init__(state_size, device, verbose=verbose)
        
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,garbage_collection_threshold:0.8'
        
        self.beam_width = beam_width
        self.max_steps = max_steps
        self.use_x_rule = use_x_rule
        self.target_neighborhood_radius = target_neighborhood_radius
        self.hashes_batch_size = hashes_batch_size
        self.filter_batch_size = filter_batch_size
        self.predict_batch_size = predict_batch_size
        self.history_window_size = history_window_size
        
        # Move codes: X=0, L=1, R=2
        self.move_names = ['X', 'L', 'R']  # Keep for logging only
        self.MOVE_X = 0
        self.MOVE_L = 1
        self.MOVE_R = 2
        
        # Pre-compute indices for state transformations in _bulk_state_transform
        self.idx_x = torch.tensor([1, 0] + list(range(2, state_size)), device=device)
        self.idx_l = torch.roll(torch.arange(state_size, device=device), -1)
        self.idx_r = torch.roll(torch.arange(state_size, device=device), 1)
        
        # Pre-compute hash vector for efficient state hashing
        max_int = int(2**62)
        self.hash_vec = torch.randint(
            low=-max_int,
            high=max_int + 1,
            size=(self.state_size,),
            dtype=torch.int64,
            device=device
        ).contiguous()
        
        # Get solved state as sorted permutation
        self.solved_state = torch.arange(self.state_size, device=device, dtype=torch.int8)
        
        max_hashes_per_step = beam_width * 3  
        
        # Single buffer for all hashes in history
        if self.history_window_size > 0:
            # Ensure there's enough space for at least one full step of hashes
            self.max_history_size = max(
                max_hashes_per_step, self.history_window_size * max_hashes_per_step
            )
            self.hash_history_buffer = torch.empty(self.max_history_size, dtype=torch.int64, device=device)
        else:
            self.max_history_size = 0
            self.hash_history_buffer = torch.empty(0, dtype=torch.int64, device=device)
        
        self.buffer_head = 0  # Position for writing new hashes
        self.buffer_size = 0  # Current size of filled buffer
        
        # tracking for each step - where the hashes of this step start/end
        self.step_boundaries = []  # [(start_idx, count), ...]
        
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
            'history_window_size': history_window_size,
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
        self.buffer_head = 0
        self.buffer_size = 0
        self.step_boundaries = []
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
        self.start_state = start_state.clone().to(dtype=torch.int8)
        
        # Initialize with start state
        start_hash = self._compute_state_hashes(self.start_state.unsqueeze(0))
        
        # Add initial hash to history if history_window_size > 0
        if self.history_window_size > 0:
            self._update_hash_history(start_hash)
        
        current_states = self.start_state.unsqueeze(0)
        self.log_info(f"Initial state: {start_state.cpu().numpy()}")
        
        parent_indices = []
        move_indices = []
        search_start = time()
        
        for step in range(1, self.max_steps + 1):
            self.log_info(f"\n{'='*10} Step {step} {'='*10}")
            
            # Store current step in search_stats
            self.search_stats['current_step'] = step
            
            # 1. Expand all states at once - use GPU efficiently
            next_states, next_moves = self._bulk_state_transform(current_states)
            parents = torch.arange(len(current_states), device=self.device).repeat(3)
            
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
                
                # Free memory
                del unique_states, unique_indices
                torch.cuda.empty_cache()
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
                    self.log_warning(f"Solution found in target neighborhood after {step} steps")
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
            
            for batch_start in range(0, len(next_states), self.filter_batch_size):
                batch_end = min(batch_start + self.filter_batch_size, len(next_states))
                
                filtered_states, filtered_parents, filtered_moves, new_hashes = self._filter_states(
                    next_states[batch_start:batch_end],
                    next_moves[batch_start:batch_end],
                    parents[batch_start:batch_end],
                    current_states
                )
                
                self._track_memory()  # Track memory after filtering batch
                
                if filtered_states.shape[0] > 0:
                    all_filtered_states.append(filtered_states)
                    all_filtered_parents.append(filtered_parents)
                    all_filtered_moves.append(filtered_moves)
                    if new_hashes.numel() > 0:
                        all_new_hashes.append(new_hashes)
            
            # Update hash history with new hashes
            if all_new_hashes:
                step_hashes = torch.cat(all_new_hashes)
                
                # Update total count of all hashes ever seen
                self.total_hashes_ever_seen += step_hashes.size(0)
                
                # Only add to history if we're tracking it
                if self.history_window_size > 0:
                    self.log_info(f"Before updating history: buffer_size={self.buffer_size}, boundaries={self.step_boundaries}")
                    self._update_hash_history(step_hashes)
                    # Update history stats
                    current_hashes = self.buffer_size
                    self.search_stats['Total hashes in history'] = current_hashes
                    self.log_info(f"After updating history: buffer_size={self.buffer_size}, boundaries={self.step_boundaries}")
                else:
                    # No history tracking, so hashes in history is always 0
                    self.search_stats['Total hashes in history'] = 0
                
                # Total hashes ever seen should be updated regardless
                self.search_stats['Total hashes ever seen'] = self.total_hashes_ever_seen
                
                self._track_memory()  # Track memory after updating hash history
            
            # Free memory
            del next_states, next_moves, parents
            if 'all_new_hashes' in locals():
                del all_new_hashes
            torch.cuda.empty_cache()
            
            # Combine filtered results
            if all_filtered_states:
                next_states = torch.cat(all_filtered_states)
                parents = torch.cat(all_filtered_parents)
                next_moves = torch.cat(all_filtered_moves)
                
                # Free memory
                del all_filtered_states, all_filtered_parents, all_filtered_moves
                torch.cuda.empty_cache()
                self._track_memory()  # Track memory after combining filtered results
            else:
                next_states = torch.empty((0, self.state_size), device=self.device, dtype=torch.int8)
                parents = torch.empty(0, device=self.device, dtype=torch.int8)
                next_moves = torch.empty(0, device=self.device, dtype=torch.int8)
            
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
        current_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Filter states based on hash (already visited states) or rules (X rule, etc.)."""
        # Calculate hashes for this batch
        state_hashes = self._compute_state_hashes(next_states)
        
        # Check if hashes are in history
        is_in_history = self._check_history(state_hashes)
        is_new = ~is_in_history
        
        # Return only new hashes for adding to history
        new_hashes = state_hashes[is_new] if torch.any(is_new) else torch.empty(0, dtype=torch.int64, device=self.device)
        
        valid_moves = is_new

        # Filter states based on rules (X rule, etc.)
        if self.use_x_rule:
            is_x_move = next_moves == 0
            
            # Optimized vectorized operation
            parent_indices = parents % current_states.size(0)
            first_vals = torch.gather(current_states[:, 0], 0, parent_indices)
            second_vals = torch.gather(current_states[:, 1], 0, parent_indices)
            first_smaller = first_vals < second_vals
            
            valid_moves &= ~(is_x_move & first_smaller)
            
        # Log filtering details if verbose
        if self.verbose > 1:
            self._log_move_filtering(
                current_states, next_states, parents, 
                next_moves, valid=valid_moves, is_visited=~is_new
            )

        # Select valid states efficiently
        valid_indices = torch.where(valid_moves)[0]
        if valid_indices.numel() > 0:
            filtered_states = torch.index_select(next_states, 0, valid_indices).contiguous()
            filtered_parents = torch.index_select(parents, 0, valid_indices)
            filtered_moves = torch.index_select(next_moves, 0, valid_indices)
        else:
            filtered_states = torch.empty((0, self.state_size), dtype=next_states.dtype, device=self.device)
            filtered_parents = torch.empty(0, dtype=parents.dtype, device=self.device)
            filtered_moves = torch.empty(0, dtype=next_moves.dtype, device=self.device)
        
        # Free memory
        del is_in_history, is_new, valid_moves, valid_indices
        torch.cuda.empty_cache()
        
        return filtered_states, filtered_parents, filtered_moves, new_hashes
        
    def _prune_beam(
        self,
        next_states: torch.Tensor,
        parents: torch.Tensor,
        moves: torch.Tensor,
        current_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prune beam to keep only top states according to model predictions."""
        self.log_info(f"Pruning beam from {len(next_states)} to {self.beam_width} states")
        
        # Track pruning statistics
        pruned_count = len(next_states) - min(self.beam_width, len(next_states))
        
        # Update pruning statistics
        self.search_stats['pruned_states_total'] += pruned_count
        self.search_stats['pruned_states_min'] = min(self.search_stats['pruned_states_min'], pruned_count) if pruned_count > 0 else self.search_stats['pruned_states_min']
        self.search_stats['pruned_states_max'] = max(self.search_stats['pruned_states_max'], pruned_count)
        self.search_stats['pruning_steps'] += 1
        if self.search_stats['pruning_steps'] > 0:
            self.search_stats['pruned_states_avg'] = self.search_stats['pruned_states_total'] / self.search_stats['pruning_steps']
        
        # Always update last step pruned (will hold the final step's value when search ends)
        self.search_stats['last_step_pruned'] = pruned_count
        
        # Process model predictions in chunks for memory efficiency
        num_states = next_states.shape[0]
        chunk_size = min(int(self.predict_batch_size), num_states)
        
        # Pre-allocate tensor for model distances
        model_distances = torch.empty(num_states, device=next_states.device, dtype=torch.float32)
        
        # Use half-precision for faster GPU computation
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            for i in range(0, num_states, chunk_size):
                end = min(i + chunk_size, num_states)
                chunk = next_states[i:end]
                
                # This is where GPU utilization should be highest
                distances_chunk = self.model.predict(chunk)
                model_distances[i:end] = distances_chunk
                
                # Only free for very large chunks to avoid fragmentation
                if chunk_size > 100000:
                    del chunk, distances_chunk
                    torch.cuda.empty_cache()
        
        # Use optimized GPU topk operation
        k = min(self.beam_width, num_states)
        top_values, top_indices = torch.topk(model_distances, k=k, largest=False, sorted=False)
        
        # Debug logging with optimized function
        if self.verbose > 1:
            self._log_pruning_decisions(
                next_states, parents, moves, model_distances, 
                top_indices, current_states
            )
        
        # Apply pruning with efficient indexing
        pruned_states = torch.index_select(next_states, 0, top_indices)
        pruned_parents = torch.index_select(parents, 0, top_indices)
        pruned_moves = torch.index_select(moves, 0, top_indices)
        
        # Free memory
        del model_distances, top_values, top_indices
        torch.cuda.empty_cache()
        
        return pruned_states, pruned_parents, pruned_moves

    def _bulk_state_transform(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply all possible moves to states efficiently, return expanded states and move types."""
        n_states = states.size(0)
        
        # Pre-allocate result tensor
        result = torch.empty((n_states * 3, self.state_size), dtype=torch.int8, device=self.device)
        
        # Copy states to different sections
        result[:n_states] = states  # X moves section
        result[n_states:2*n_states] = states  # L moves section
        result[2*n_states:] = states  # R moves section
        
        # Apply moves using pre-computed indices - all operations stay on GPU
        result[:n_states] = result[:n_states].index_select(1, self.idx_x)  # X moves
        result[n_states:2*n_states] = result[n_states:2*n_states].index_select(1, self.idx_l)  # L moves
        result[2*n_states:] = result[2*n_states:].index_select(1, self.idx_r)  # R moves
        
        # Generate move types (0=X, 1=L, 2=R)
        move_types = torch.arange(3, device=states.device).repeat_interleave(n_states)
        
        return result.contiguous(), move_types

    def _compute_state_hashes(self, states: torch.Tensor) -> torch.Tensor:
        """Compute unique hashes for states using vectorized operations."""
        n_states = states.size(0)
        hashes = torch.empty(n_states, dtype=torch.int64, device=self.device)

        for i in range(0, n_states, self.hashes_batch_size):
            end = min(i + self.hashes_batch_size, n_states)
            batch = states[i:end]
            
            try:
                hashes[i:end] = torch.sum(
                    batch.to(dtype=torch.int64) * self.hash_vec.unsqueeze(0), dim=1
                )
            except torch.cuda.OutOfMemoryError as e:
                # Print diagnostic information
                batch_size = end - i
                print(f"Out of memory error in _compute_state_hashes!")
                print(f"Trying to compute hashes for {batch_size} states at once")
                print(f"Total number of states: {n_states}")
                print(f"Current batch: {i}")
                print(f"GPU memory allocated: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
                print(f"GPU memory reserved: {torch.cuda.memory_reserved() / (1024**3):.2f} GB")
                
                # Re-raise the exception after printing diagnostic info
                raise e
            
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
        
        # Sort hashes to identify unique elements
        sorted_hashes, sort_indices = torch.sort(hashes)
        
        # Create mask for unique elements (first element and elements different from previous)
        mask = torch.ones(sorted_hashes.size(0), dtype=torch.bool, device=self.device)
        mask[1:] = sorted_hashes[1:] != sorted_hashes[:-1]
        
        # Get indices of unique elements in original tensor
        unique_indices = sort_indices[mask]
        
        # Get unique states
        unique_states = states[unique_indices]
        
        return unique_states, unique_indices

    def _apply_moves(self, state: torch.Tensor, moves: Union[torch.Tensor, int]) -> torch.Tensor:
        """Apply a sequence of moves or single move to a state."""
        new_state = state.clone()
        
        # Handle single move (int or 0-d tensor)
        if isinstance(moves, (int, torch.Tensor)) and not isinstance(moves, (list, tuple)):
            if isinstance(moves, torch.Tensor):
                code = moves.item()
            else:
                code = moves
            
            if code == 0:  # X
                new_state[[0, 1]] = new_state[[1, 0]]
            elif code == 1:  # L
                new_state = torch.roll(new_state, shifts=-1)
            elif code == 2:  # R
                new_state = torch.roll(new_state, shifts=1)
        else:
            # Sequence of moves
            for move in moves:
                code = move.item() if isinstance(move, torch.Tensor) else move
                if code == 0:  # X
                    new_state[[0, 1]] = new_state[[1, 0]]
                elif code == 1:  # L
                    new_state = torch.roll(new_state, shifts=-1)
                elif code == 2:  # R
                    new_state = torch.roll(new_state, shifts=1)
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
                    return False, -1, torch.empty(0, dtype=torch.int8, device=self.device)
                
                return True, idx, stored_path
            
            return False, -1, torch.empty(0, dtype=torch.int8, device=self.device)
        else:
            # Check for exact match with solved state
            matches = torch.all(states == self.solved_state, dim=1)
            found = torch.any(matches)
            idx = torch.where(matches)[0][0].item() if found else -1
            return found, idx, torch.empty(0, dtype=torch.int8, device=self.device)
        
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
        moves = torch.tensor(reverse_moves[::-1], dtype=torch.int8, device=self.device)
        
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
        initial_hash = self._compute_state_hashes(initial_state).item()
        
        # Track states and their paths TO solved state
        states_dict = {initial_hash: (initial_state[0], [])}
        frontier = [(initial_state[0], [])]
        
        self.log_info(f"Starting BFS from solved state {initial_state[0]}")
        
        for depth in range(radius):
            if not frontier:
                break
            
            current_states = torch.stack([state for state, _ in frontier])
            current_paths = [path for _, path in frontier]
            
            # Generate all possible next states
            next_states, next_moves = self._bulk_state_transform(current_states)
            
            # Process states
            next_frontier = []
            for i, (state, move) in enumerate(zip(next_states, next_moves)):
                parent_idx = i % len(current_states)
                hash_val = self._compute_state_hashes(state.unsqueeze(0)).item()
                
                if hash_val not in states_dict:
                    # Create path TO solved state by prepending inverse move to parent's path
                    inverse_move = self._get_inverse_move(move.item())
                    new_path = [inverse_move] + current_paths[parent_idx].copy()
                    states_dict[hash_val] = (state, new_path)
                    next_frontier.append((state, new_path))
            
            frontier = next_frontier
            self.log_info(f"Target neighborhood depth {depth + 1}: {len(states_dict)} states") 
        self.log_warning(f"Overall states in target neighborhood: {len(states_dict)}")
        
        # Convert paths to tensors
        final_states_dict = {}
        for hash_val, (state, path) in states_dict.items():
            path_tensor = torch.tensor(path, dtype=torch.int8, device=self.device)
            final_states_dict[hash_val] = (state, path_tensor)
        
        hashes = torch.tensor(
            list(final_states_dict.keys()), dtype=torch.int64, device=self.device
        )
        
        return hashes, final_states_dict

    def _get_inverse_move(self, move: int) -> int:
        """Get the inverse of a move (X->X, L->R, R->L)."""
        if move == 0:  # X is self-inverse
            return 0
        elif move == 1:  # L inverse is R
            return 2
        else:  # R inverse is L
            return 1

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
        self.log_info(f"History window size: {self.search_stats['history_window_size']}")
        if self.search_stats['pruning_steps'] > 0:
            self.log_info(f"Pruning statistics:")
            self.log_info(f"  Total states pruned: {self.search_stats['pruned_states_total']}")
            self.log_info(f"  Min states pruned per step: {self.search_stats['pruned_states_min']}")
            self.log_info(f"  Max states pruned per step: {self.search_stats['pruned_states_max']}")
            self.log_info(f"  Avg states pruned per step: {self.search_stats['pruned_states_avg']:.2f}")
            self.log_info(f"  First pruning at step: {self.search_stats['first_pruning_step'] if self.search_stats['first_pruning_step'] != -1 else 'N/A'}")
            self.log_info(f"  Last step pruned states: {self.search_stats['last_step_pruned']}")
        self.log_info("=" * 40)

    def _update_hash_history(self, new_hashes):
        """Add new hashes to circular buffer."""
        if self.history_window_size <= 0:
            self.step_boundaries.clear()
            self.buffer_size = 0
            self.buffer_head = 0
            return

        n_hashes = new_hashes.size(0)

        while self.buffer_size + n_hashes > self.max_history_size and self.step_boundaries:
            old_start, old_count = self.step_boundaries.pop(0)
            self.buffer_size = max(0, self.buffer_size - old_count)

        step_start = self.buffer_head

        if self.buffer_head + n_hashes > self.max_history_size:
            first_part = self.max_history_size - self.buffer_head
            second_part = n_hashes - first_part
            self.hash_history_buffer[self.buffer_head:] = new_hashes[:first_part]
            self.hash_history_buffer[:second_part] = new_hashes[first_part:]
            self.buffer_head = second_part
        else:
            self.hash_history_buffer[self.buffer_head:self.buffer_head + n_hashes] = new_hashes
            self.buffer_head = (self.buffer_head + n_hashes) % self.max_history_size

        self.step_boundaries.append((step_start, n_hashes))
        self.buffer_size = min(self.buffer_size + n_hashes, self.max_history_size)

        while len(self.step_boundaries) > self.history_window_size:
            old_start, old_count = self.step_boundaries.pop(0)
            self.buffer_size = max(0, self.buffer_size - old_count)

    def _check_history(self, state_hashes):
        """Check if hashes have been seen in history."""
        # If history window size is 0 or buffer is empty, nothing is in history
        if self.history_window_size == 0 or self.buffer_size == 0:
            return torch.zeros(state_hashes.size(0), dtype=torch.bool, device=self.device)
        
        # Unified approach for all positive history window sizes
        is_in_history = torch.zeros(state_hashes.size(0), dtype=torch.bool, device=self.device)
        
        # Process each step in history window
        for start_idx, count in self.step_boundaries:
            end_idx = start_idx + count
            
            if end_idx <= self.max_history_size:
                # linear case
                is_in_history |= torch.isin(state_hashes, self.hash_history_buffer[start_idx:end_idx])
            else:
                # circular transition
                first_part = self.max_history_size - start_idx
                second_part = count - first_part
                
                is_in_history |= torch.isin(state_hashes, self.hash_history_buffer[start_idx:])
                is_in_history |= torch.isin(state_hashes, self.hash_history_buffer[:second_part])
        
        return is_in_history
