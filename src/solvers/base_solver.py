from abc import ABC, abstractmethod
import torch
from typing import Tuple, List, Dict, Any
import logging

class BaseSolver(ABC):
    """Base class for all LRX solvers."""
    
    def __init__(
        self,
        state_size: int,
        device: torch.device,
        model: Any = None,
        verbose: int = 0
    ):
        """Initialize common solver attributes."""
        self.state_size = state_size
        self.device = device
        self.model = model
        self.verbose = verbose
        
        # Configure logging based on verbosity
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        log_level = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }.get(verbose, logging.DEBUG)
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        # self.logger.setLevel(log_level)
        
        # Common initialization - use contiguous tensors for better GPU performance
        self.move_names = ['X', 'L', 'R']
        self.solved_state = torch.arange(state_size, dtype=torch.int64, device=device).contiguous()
        
        # For detecting visited states - use smaller range for better numerical stability
        self.hash_vec = torch.randint(
            low=1,
            high=2**15,
            size=(self.state_size,),
            dtype=torch.int32,
            device=device
        ).contiguous()
        
        # Performance tracking
        self.search_stats: Dict[str, Any] = {
            'total_states_explored': 0,
            'search_time': 0,
            'path_found': False
        }

    def log_warning(self, message: str) -> None:
        """Log warning information based on verbosity level."""
        self.logger.warning(message)

    def log_info(self, message: str) -> None:
        """Log information based on verbosity level."""
        self.logger.info(message)

    def log_debug(self, message: str) -> None:
        """Log debug information based on verbosity level."""
        self.logger.debug(message)

    @torch.jit.script
    def _bulk_state_transform(self, states: torch.Tensor) -> torch.Tensor:
        """Efficient batched state transformations using PyTorch operations."""
        n_states = states.size(0)
        state_size = states.size(1)
        
        # Pre-allocate result tensor with contiguous memory
        result = torch.empty((n_states * 3, state_size), 
                           dtype=states.dtype, 
                           device=states.device).contiguous()
        
        # Swap [0,1] - use vectorized operations
        result[0::3] = states.clone()
        result[0::3, [0,1]] = result[0::3, [1,0]]
        
        # Left and right rolls - use single roll operation for entire batch
        result[1::3] = torch.roll(states, shifts=-1, dims=1)
        result[2::3] = torch.roll(states, shifts=1, dims=1)
        
        return result

    def validate_path(self, start: torch.Tensor, moves: List[str]) -> bool:
        """Apply moves in order to start. Check if we reach solved_state."""
        current = start.view(-1).clone().contiguous()
        target = self.solved_state.view(-1)
        
        self.log_debug(f"Validating path from state: {current.cpu().numpy()}")
        
        # Pre-compute indices for efficient moves
        swap_idx = torch.tensor([1, 0] + list(range(2, len(current))), 
                              device=self.device)
        left_idx = torch.roll(torch.arange(len(current)), -1, dims=0)
        right_idx = torch.roll(torch.arange(len(current)), 1, dims=0)
        
        for m in moves:
            if m == 'X':
                current = current[swap_idx]
            elif m == 'L':
                current = current[left_idx]
            else:  # 'R'
                current = current[right_idx]
                
            self.log_debug(f"After {m}: {current.cpu().numpy()}")
        
        is_valid = torch.all(current == target)
        self.log_debug(f"Target: {target.cpu().numpy()}")
        self.log_debug(f"Path is valid: {is_valid}")
        
        return is_valid

    @torch.jit.script
    def count_inversions(self, state: torch.Tensor) -> int:
        """Count the number of inversions in a permutation."""
        n = len(state)
        # Use broadcasting for vectorized comparison
        i_vals = state.unsqueeze(1).expand(-1, n)
        j_vals = state.unsqueeze(0).expand(n, -1)
        # Count inversions using upper triangular mask
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=state.device), diagonal=1)
        inversions = torch.sum(torch.logical_and(i_vals > j_vals, mask)).item()
        return inversions

    def min_cyclic_inversions(self, state: torch.Tensor) -> int:
        """Calculate minimum inversions among all cyclic shifts."""
        n = len(state)
        # Create all cyclic shifts at once
        shifts = torch.stack([torch.roll(state, i) for i in range(n)])
        # Calculate inversions for all shifts vectorized
        inv_counts = torch.tensor([self.count_inversions(s) for s in shifts])
        return torch.min(inv_counts).item()

    def calculate_move_benefit(
        self,
        current_state: torch.Tensor,
        next_state: torch.Tensor,
        move_type: int
    ) -> float:
        """Calculate the benefit of a move based on inversion reduction."""
        current_inv = self.min_cyclic_inversions(current_state)
        next_inv = self.min_cyclic_inversions(next_state)
        inversion_reduction = current_inv - next_inv
        
        # Simplified distance calculation
        move_distance = element_distance = 1
        return 2 * inversion_reduction - element_distance - 2 * move_distance

    @abstractmethod
    def solve(self, start_state: torch.Tensor) -> Tuple[bool, int, str]:
        """
        Search for solution path from start_state to solved_state.
        
        Returns:
            Tuple[bool, int, str]: (found, steps, solution)
                - found: Whether a solution was found
                - steps: Number of steps in solution (or explored if not found)
                - solution: Solution path as string (or empty if not found)
        """
        pass

    def cleanup(self) -> None:
        """Clean up resources after search."""
        if hasattr(self, 'model'):
            del self.model
        torch.cuda.empty_cache()

    @abstractmethod
    def reset(self) -> None:
        """Reset solver state for a new problem"""
        pass 