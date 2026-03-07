"""Compute BFS distances for a given set of generators."""

from typing import List, Sequence, Tuple

import numpy as np
from numba import njit



@njit(cache=True)
def _factorials_u64(state_size: int) -> np.ndarray:
    factorials = np.empty(state_size + 1, dtype=np.uint64)
    factorials[0] = 1
    for value in range(1, state_size + 1):
        factorials[value] = factorials[value - 1] * np.uint64(value)
    return factorials


@njit(cache=True)
def _build_mask_tables(state_size: int) -> Tuple[np.ndarray, np.ndarray]:
    n_masks = 1 << state_size
    popcount = np.empty(n_masks, dtype=np.uint8)
    select = np.full((n_masks, state_size), -1, dtype=np.int16)

    for mask in range(n_masks):
        count = 0
        for bit in range(state_size):
            if mask & (1 << bit):
                select[mask, count] = np.int16(bit)
                count += 1
        popcount[mask] = np.uint8(count)

    return popcount, select


@njit(cache=True)
def _unrank_perm_inplace_mask(
    rank: np.int64,
    state_size: int,
    factorials: np.ndarray,
    select: np.ndarray,
    perm: np.ndarray,
) -> None:
    mask = (1 << state_size) - 1
    remaining = np.uint64(rank)

    for position in range(state_size):
        f = factorials[state_size - 1 - position]
        idx = int(remaining // f)
        remaining = remaining - np.uint64(idx) * f

        value = int(select[mask, idx])
        perm[position] = np.int16(value)
        mask ^= 1 << value


@njit(cache=True)
def _rank_from_perm_and_gen(
    perm: np.ndarray,
    g: np.ndarray,
    state_size: int,
    factorials: np.ndarray,
    popcount: np.ndarray,
) -> np.int64:
    mask = (1 << state_size) - 1
    rank = np.uint64(0)

    for position in range(state_size):
        value = int(perm[int(g[position])])
        less_mask = mask & ((1 << value) - 1)
        rank += np.uint64(popcount[less_mask]) * factorials[state_size - 1 - position]
        mask ^= 1 << value

    return np.int64(rank)


@njit(cache=True)
def _bfs_distances(generators: np.ndarray, state_size: int) -> np.ndarray:
    factorials = _factorials_u64(state_size)
    popcount, select = _build_mask_tables(state_size)

    n_states = int(factorials[state_size])
    dist = np.full(n_states, -1, dtype=np.int32)
    queue = np.empty(n_states, dtype=np.int64)

    perm = np.empty(state_size, dtype=np.int16)

    head = 0
    tail = 0
    dist[0] = 0
    queue[tail] = np.int64(0)
    tail += 1

    n_gens = int(generators.shape[0])

    while head < tail:
        state_rank = queue[head]
        head += 1
        base_dist = int(dist[int(state_rank)]) + 1

        _unrank_perm_inplace_mask(
            state_rank, state_size, factorials, select, perm
        )

        for gen_index in range(n_gens):
            nxt = _rank_from_perm_and_gen(
                perm, generators[gen_index], state_size, factorials, popcount
            )
            nxt_i = int(nxt)
            if dist[nxt_i] == -1:
                dist[nxt_i] = base_dist
                queue[tail] = nxt
                tail += 1

    return dist


@njit(cache=True)
def _unrank_many(ranks: np.ndarray, state_size: int) -> np.ndarray:
    factorials = _factorials_u64(state_size)
    _, select = _build_mask_tables(state_size)

    n = int(ranks.size)
    perms = np.empty((n, state_size), dtype=np.int16)
    perm = np.empty(state_size, dtype=np.int16)

    for i in range(n):
        _unrank_perm_inplace_mask(
            ranks[i], state_size, factorials, select, perm
        )
        perms[i, :] = perm

    return perms


def compute_bfs_distances(
    generators: List[Sequence[int]],
    state_size: int,
) -> np.ndarray:
    generators_arr = np.asarray(generators, dtype=np.int16)
    if generators_arr.ndim != 2 or generators_arr.shape[1] != state_size:
        raise ValueError("generators must have shape (n_gens, state_size)")
    dist = _bfs_distances(generators_arr, state_size)
    if (dist < 0).any():
        raise RuntimeError("BFS did not reach all states.")
    return dist


def sample_bfs_eval_set(
    dist: np.ndarray,
    state_size: int,
    n_eval: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_states = int(dist.size)
    n_eval = int(min(n_eval, n_states))
    ranks = rng.integers(0, n_states, size=n_eval, dtype=np.int64)
    X = _unrank_many(ranks, state_size)
    y = dist[ranks]
    return X, y