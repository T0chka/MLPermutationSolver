import torch
import random
import numpy as np
from typing import Tuple, List

def create_lrx_moves(state_size: int) -> List[List[int]]:
    """Create basic LRX moves: X (swap), L (left shift), R (right shift)"""
    identity = list(range(state_size))
    
    X = identity.copy()
    X[0], X[1] = X[1], X[0]
    
    L = identity[1:] + [identity[0]]
    R = [identity[-1]] + identity[:-1]
    
    return [X, L, R]

def first_visit_random_walks(
    generators: list,
    n_steps: int,
    n_walks: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates random walks from identity permutation,
    tracks when each state was first visited,
    returns the sequence of visited states and their first occurrence step.
    """
    state_size = len(generators[0])
    all_moves = torch.tensor(generators, device=device, dtype=torch.long)

    # initialize
    total_states = n_steps * n_walks
    X = torch.empty((total_states, state_size), device=device, dtype=torch.long)

    # precompute step numbers for each row in X
    steps = torch.arange(
        n_steps, device=device
    ).unsqueeze(1).expand(n_steps, n_walks).reshape(-1)
    
    # starting states
    current_states = torch.arange(
        state_size, device=device
    ).unsqueeze(0).expand(n_walks, state_size).clone()
    X[:n_walks] = current_states

    # simulate random walks
    for step in range(1, n_steps):
        chosen_moves = torch.randint(0, len(generators), (n_walks,), device=device)
        current_states = torch.gather(current_states, 1, all_moves[chosen_moves])
        idx_start = step * n_walks
        idx_end = (step + 1) * n_walks
        X[idx_start:idx_end] = current_states
        
    # hash vector for state tracking
    hash_vec = torch.randint(
        low=-(2**30),
        high=(2**30),
        size=(state_size,),
        device=device, dtype=torch.long
    )
    flat_hashes = torch.sum(X * hash_vec, dim=1)
    
    # group states by hash and determine the first occurrence step per unique state
    unique_hashes, inverse_indices = torch.unique(flat_hashes, return_inverse=True)

    # initialize with a value larger than any possible step
    # (n_steps is safe because steps are 0-indexed)
    init_val = n_steps  
    first_occurrence = torch.full(
        (unique_hashes.size(0),), init_val, device=device, dtype=steps.dtype
    )
    first_occurrence = first_occurrence.scatter_reduce(
        0, inverse_indices, steps, reduce='amin', include_self=False
    )
    y = first_occurrence[inverse_indices]
    
    return X, y

def nbt_random_walks(
    generators: list,
    n_steps: int,
    n_walks: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate non-backtracking random walks from identity permutation"""
    state_size = len(generators[0])
    all_moves = torch.tensor(generators, device=device, dtype=torch.long)
    
    # initialize starting states
    current_states = torch.arange(state_size, device=device).repeat(n_walks, 1)
    
    # create hash vector for state tracking
    hash_vec = torch.randint(
        low=-(2**30), high=(2**30),
        size=(state_size,),
        device=device, dtype=torch.long
    )
    
    # track valid walks and their lengths
    valid_states = []
    valid_lengths = []
    
    # initialize state tracking for each walk
    current_hashes = torch.sum(hash_vec * current_states, dim=1)
    seen_hashes = [{h.item()} for h in current_hashes]
    
    # store initial states
    valid_states.append(current_states.clone())
    valid_lengths.extend([0] * n_walks)
    
    # random walks
    for step in range(1, n_steps):
        # Try all possible moves for current states
        expanded_states = torch.cat([
            torch.gather(current_states, 1, move.repeat(current_states.size(0), 1))
            for move in all_moves
        ])
        expanded_hashes = torch.sum(hash_vec * expanded_states, dim=1)
        
        # reshape for per-walk processing
        expanded_hashes = expanded_hashes.view(len(generators), -1)
        expanded_states = expanded_states.view(len(generators), -1, state_size)
        
        # select valid moves for each walk
        new_states = []
        active_walks = []
        
        for walk_idx in range(current_states.size(0)):
            # find moves that lead to unseen states
            walk_hashes = expanded_hashes[:, walk_idx]
            valid_moves = [
                i for i in range(len(generators))
                if walk_hashes[i].item() not in seen_hashes[walk_idx]
            ]
            
            if valid_moves:
                # randomly choose one of the valid moves
                chosen_move = random.choice(valid_moves)
                new_state = expanded_states[chosen_move, walk_idx]
                new_hash = walk_hashes[chosen_move].item()
                
                new_states.append(new_state)
                seen_hashes[walk_idx].add(new_hash)
                active_walks.append(walk_idx)
        
        if not new_states:
            break
            
        # update current states
        current_states = torch.stack(new_states)
        valid_states.append(current_states.clone())
        valid_lengths.extend([step] * len(new_states))
        
        if len(active_walks) == 0:
            break
    
    # combine all valid states and their lengths
    X = torch.cat(valid_states)
    y = torch.tensor(valid_lengths, device=device, dtype=torch.long)
    
    return X, y

def random_walks_beam_nbt(
    generators: list,
    n_steps: int,
    n_walks: int,
    device: torch.device,
    nbt_depth: int = None,
    dtype: str = 'auto',
    verbose: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate non-backtracking random walks from identity permutation.    
    Uses beam search approach where states visited by any trajectory are banned for all.
    """
    if not nbt_depth:
        nbt_depth = n_steps
    
    state_size = len(generators[0])
    n_generators = len(generators)
    
    # Convert generators to tensor
    tensor_generators = torch.tensor(generators, device=device, dtype=torch.int64)
    
    # Determine appropriate dtype based on state size
    if isinstance(dtype, str) and dtype.lower() == 'auto':
        dtype = torch.uint8 if state_size <= 256 else torch.uint16
    
    # Create initial state (identity permutation)
    initial_state = torch.arange(
        state_size, device=device, dtype=dtype
    ).reshape(-1, state_size)
    
    # Create hash vector for fast state comparison
    hash_vector = torch.randint(
        -(2**62), 2**62,
        size=(state_size,),
        device=device,
        dtype=torch.int64
    )
    
    # Initialize current states by duplicating the initial state
    current_states = initial_state.view(1, state_size).expand(n_walks, state_size).clone()
    
    # Allocate memory for output
    X = torch.zeros(n_walks * n_steps, state_size, device=device, dtype=dtype)
    y = torch.zeros(n_walks * n_steps, device=device, dtype=torch.uint32)
    
    # Store initial states in output
    X[:n_walks] = current_states
    y[:n_walks] = 0
    
    # Initialize hash history if using non-backtracking
    if nbt_depth > 0:
        initial_hash = torch.sum(initial_state.view(-1, state_size) * hash_vector, dim=1)
        hash_history = initial_hash.expand(n_walks * n_generators, nbt_depth).clone()
        history_index = 0  # Cyclic index for hash storage
    
    # Track effective step count (may differ from loop index due to backtracking avoidance)
    effective_step = 0
    
    if verbose:
        print(f"Starting random walks with {n_walks} walks for {n_steps} steps")
        print(f"Using device: {device}, state size: {state_size}, history depth: {nbt_depth}")
    
    # Main random walk loop
    for step in range(1, n_steps):
        if verbose and step % max(1, n_steps // 10) == 0:
            print(f"Processing step {step}/{n_steps} (effective step: {effective_step})")
        
        # 1. Generate new states by applying all generators to all current states
        expanded_states = current_states.unsqueeze(1).expand(
            current_states.size(0),
            tensor_generators.shape[0],
            current_states.size(1)
        )
        expanded_moves = tensor_generators.unsqueeze(0).expand(
            current_states.size(0),
            tensor_generators.shape[0],
            tensor_generators.size(1)
        )
        new_states = torch.gather(expanded_states, 2, expanded_moves).flatten(end_dim=1)
        
        # 2. Compute hashes for new states
        new_hashes = torch.sum(new_states * hash_vector, dim=1)
        
        # 3. Handle non-backtracking if nbt_depth > 0
        if nbt_depth > 0:
            # Filter out states that have been visited before
            is_new_mask = ~torch.isin(new_hashes, hash_history.view(-1), assume_unique=False)
            new_state_count = is_new_mask.sum().item()
            
            if new_state_count >= n_walks:
                # Enough new states available
                new_states = new_states[is_new_mask]
                effective_step += 1
                
                if verbose and new_state_count > n_walks * 1.5:
                    print(f"  Found {new_state_count} new states (using {n_walks})")
            else:
                # Handle edge case: not enough new states
                if new_state_count > 0:
                    # Use available new states with repetition
                    repeat_factor = int(np.ceil(n_walks / new_state_count))
                    new_states = new_states[is_new_mask].repeat(repeat_factor, 1)[:n_walks]
                    effective_step += 1
                    
                    if verbose:
                        print(f"  Warning: Only {new_state_count} new states found, repeating some")
                else:
                    # No new states available, stay at current states
                    new_states = current_states
                    
                    if verbose:
                        print(f"  Warning: No new states found, staying at current states")
        else:
            # If nbt_depth is 0, allow any move (no non-backtracking)
            effective_step = step
        
        # 4. Randomly select n_walks states from available new states
        if new_states.size(0) > n_walks:
            perm = torch.randperm(new_states.size(0), device=device)
            current_states = new_states[perm][:n_walks]
        else:
            current_states = new_states
        
        # 5. Store results in output arrays
        start_idx = step * n_walks
        end_idx = (step + 1) * n_walks
        y[start_idx:end_idx] = effective_step
        X[start_idx:end_idx] = current_states
        
        # 6. Update hash history if using non-backtracking
        if nbt_depth > 0:
            history_index = (history_index + 1) % nbt_depth
            hash_history[:, history_index] = new_hashes
    
    if verbose:
        print(f"Random walks completed. Final effective step: {effective_step}")
    
    return X, y