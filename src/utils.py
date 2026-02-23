import torch
from typing import Dict, Optional


def apply_move_solution(
    start_state: torch.Tensor,
    solution: str,
    name_to_code: Dict[str, int],
    move_indices: torch.Tensor,
) -> torch.Tensor:
    """Apply a dot-separated solution to a 1D state tensor."""
    if start_state.ndim != 1:
        raise ValueError(f"start_state must be 1D, got: {start_state.ndim}D")
    if move_indices.ndim != 2:
        raise ValueError(f"move_indices must be 2D, got: {move_indices.ndim}D")
    if int(move_indices.size(1)) != int(start_state.numel()):
        raise ValueError(
            "move_indices width must match state_size: "
            f"{int(move_indices.size(1))} vs {int(start_state.numel())}"
        )
    if start_state.device != move_indices.device:
        raise ValueError(
            "Device mismatch: "
            f"state={start_state.device}, moves={move_indices.device}"
        )

    tokens = [t.strip() for t in str(solution).split(".") if t.strip()]
    state = start_state
    for token in tokens:
        if token not in name_to_code:
            raise ValueError(f"Unknown move token: {token}")
        move_code = int(name_to_code[token])
        idx = move_indices[move_code]
        state = state.index_select(0, idx)
    return state


def assert_solution_sorts_state(
    start_state: torch.Tensor,
    solution: str,
    name_to_code: Dict[str, int],
    move_indices: torch.Tensor,
    *,
    state_id: Optional[int] = None,
    unsolved_token: str = "UNSOLVED",
) -> None:
    """Raise ValueError if solution does not sort the state to identity."""
    if str(solution) == unsolved_token:
        raise ValueError(f"Unsolved token for id={state_id}")

    final_state = apply_move_solution(
        start_state=start_state,
        solution=solution,
        name_to_code=name_to_code,
        move_indices=move_indices,
    )
    target = torch.arange(
        int(final_state.numel()),
        device=final_state.device,
        dtype=final_state.dtype,
    )
    if torch.equal(final_state, target):
        return

    n = int(final_state.numel())
    tokens = [t.strip() for t in str(solution).split(".") if t.strip()]
    head = final_state[: min(20, n)].tolist()
    raise ValueError(
        "Solution does not sort state to identity. "
        f"id={state_id}, n={n}, moves={len(tokens)}, final_head={head}"
    )


def estimate_gpu_limits(
    state_size: int,
    device: torch.device,
    n_moves: int = 3,
    reserve_gb: float = 4.0,
    frac_for_hash_batch: float = 0.25,
    frac_for_beam: float = 0.2,
) -> Dict[str, int]:
    """
    Estimate safe batch sizes and max beam width from current free GPU memory.
    Call after torch.cuda.empty_cache() and with model/data already on GPU if applicable.
    Returns dict with: hashes_batch_size, filter_batch_size, predict_batch_size, max_beam_width.
    """
    torch.cuda.empty_cache()
    if hasattr(torch.cuda, "mem_get_info"):
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    else:
        total_bytes = torch.cuda.get_device_properties(device).total_memory
        free_bytes = total_bytes - torch.cuda.memory_reserved(device)
    reserve_bytes = int(reserve_gb * (1024**3))
    available = max(0, free_bytes - reserve_bytes)
    
    # One hash batch: batch_size * state_size * 8 (int64) * 2 (temp) bytes
    bytes_per_hash_state = state_size * 8 * 2
    safe_hashes = int(available * frac_for_hash_batch / bytes_per_hash_state)
    safe_hashes = max(10_000, min(safe_hashes, 10_000_000))
    
    # Filter/predict batches: similar scale
    safe_filter = safe_hashes
    safe_predict = min(safe_hashes * 2, 20_000_000)
    
    # Max beam: after expand we have beam*3 states; need 3*beam*state_size bytes + headroom
    bytes_per_beam_state = n_moves * state_size + 24  # states + hashes
    max_beam = int(available * frac_for_beam / bytes_per_beam_state)
    max_beam = max(2**4, min(max_beam, 2**20))
    return {
        "hashes_batch_size": safe_hashes,
        "filter_batch_size": safe_filter,
        "predict_batch_size": safe_predict,
        "max_beam_width": max_beam,
    }