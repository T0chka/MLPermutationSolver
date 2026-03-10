"""NBT hash history: puzzle-agnostic component for beam solver."""

from typing import Callable, List, Optional, Tuple

import torch


class HashHistory:
    """Circular buffer of state hashes for NBT (no-backtrack) pruning.

    Tracks last nbt_depth steps; evicts oldest step when full.
    """

    def __init__(
        self,
        device: torch.device,
        max_history_size: int,
        nbt_depth: int,
        hashes_batch_size: int = 1_000_000,
        warn_fn: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Initialize hash history buffer."""
        self.device = device
        self.max_history_size = max_history_size
        self.nbt_depth = nbt_depth
        self.hashes_batch_size = hashes_batch_size
        self._warn_fn = warn_fn

        if nbt_depth > 0 and max_history_size > 0:
            self._buffer = torch.empty(max_history_size, dtype=torch.int64, device=device)
        else:
            self._buffer = torch.empty(0, dtype=torch.int64, device=device)

        self._buffer_head = 0
        self._buffer_size = 0
        self._step_boundaries: List[Tuple[int, int]] = []

    def add(self, new_hashes: torch.Tensor) -> None:
        """Add new hashes to circular buffer."""
        if self.nbt_depth <= 0:
            self._step_boundaries.clear()
            self._buffer_size = 0
            self._buffer_head = 0
            return

        n_hashes = new_hashes.size(0)
        if n_hashes > self.max_history_size:
            new_hashes = new_hashes[-self.max_history_size :]
            n_hashes = self.max_history_size

        # Evict oldest step if buffer overflows
        while (
            self._buffer_size + n_hashes > self.max_history_size
            and self._step_boundaries
        ):
            old_start, old_count = self._step_boundaries.pop(0)
            self._buffer_size = max(0, self._buffer_size - old_count)

        step_start = self._buffer_head

        # Write hashes (wrap around if needed)
        if self._buffer_head + n_hashes > self.max_history_size:
            first_part = self.max_history_size - self._buffer_head
            second_part = n_hashes - first_part
            self._buffer[self._buffer_head:] = new_hashes[:first_part]
            self._buffer[:second_part] = new_hashes[first_part:]
            self._buffer_head = second_part
        else:
            self._buffer[self._buffer_head : self._buffer_head + n_hashes] = new_hashes
            self._buffer_head = (self._buffer_head + n_hashes) % self.max_history_size

        self._step_boundaries.append((step_start, n_hashes))
        self._buffer_size = min(self._buffer_size + n_hashes, self.max_history_size)

        # Evict steps beyond nbt_depth
        while len(self._step_boundaries) > self.nbt_depth:
            old_start, old_count = self._step_boundaries.pop(0)
            self._buffer_size = max(0, self._buffer_size - old_count)

    def check(self, state_hashes: torch.Tensor) -> torch.Tensor:
        """Check which hashes have been seen. Returns bool mask of same length."""
        n = int(state_hashes.size(0))
        if self.nbt_depth == 0 or self._buffer_size == 0:
            return torch.zeros(n, dtype=torch.bool, device=self.device)

        capacity = int(self.max_history_size)
        size = int(self._buffer_size)
        head = int(self._buffer_head)
        tail = (head - size) % capacity if size > 0 else head

        # Extract active window (contiguous for sort)
        if tail < head:
            active = self._buffer[tail:head].contiguous()
        else:
            active = torch.cat(
                [self._buffer[tail:], self._buffer[:head]],
                dim=0,
            )
        sorted_buf = torch.sort(active)[0]

        batch_size = int(self.hashes_batch_size)
        is_in_history = torch.zeros(n, dtype=torch.bool, device=self.device)
        active_size = sorted_buf.size(0)
        i = 0
        while i < n:
            end = min(i + batch_size, n)
            chunk = state_hashes[i:end]
            try:
                idx = torch.searchsorted(sorted_buf, chunk, side="left")
                valid = idx < active_size
                safe_idx = idx.clamp(max=active_size - 1)
                is_in_history[i:end] = valid & (sorted_buf[safe_idx] == chunk)
                i = end
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                new_batch = max(5_000, batch_size // 2)
                if new_batch >= batch_size:
                    raise
                if self._warn_fn is not None:
                    self._warn_fn(
                        "OOM in HashHistory.check: reducing batch "
                        f"{batch_size} -> {new_batch} (n={n})"
                    )
                self.hashes_batch_size = new_batch
                batch_size = new_batch

        return is_in_history

    def reset(self) -> None:
        """Clear history (e.g. for new solve)."""
        self._buffer_head = 0
        self._buffer_size = 0
        self._step_boundaries.clear()

    @property
    def buffer_size(self) -> int:
        """Current number of hashes in buffer."""
        return self._buffer_size
