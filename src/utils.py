import torch
from typing import Dict

def estimate_gpu_limits(
    state_size: int,
    device: torch.device,
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
    bytes_per_beam_state = 3 * state_size + 24  # states + hashes
    max_beam = int(available * frac_for_beam / bytes_per_beam_state)
    max_beam = max(2**4, min(max_beam, 2**20))
    return {
        "hashes_batch_size": safe_hashes,
        "filter_batch_size": safe_filter,
        "predict_batch_size": safe_predict,
        "max_beam_width": max_beam,
    }