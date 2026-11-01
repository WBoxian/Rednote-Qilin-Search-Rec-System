"""Preserve a learned coarse pool while prioritizing a fixed prior-ranked head.

Input sequence order is rank order. This module does not retrieve candidates,
fit models, select parameters, or activate a serving strategy.
"""
from __future__ import annotations

import numpy as np

_MAX_ID = np.iinfo(np.int64).max


def _nonnegative_integer(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be a nonnegative int64 integer")
    result = int(value)
    if result < 0 or result > _MAX_ID:
        raise ValueError(name + " must be a nonnegative int64 integer")
    return result


def _ordered_ids(values, name: str) -> np.ndarray:
    if isinstance(values, np.ndarray):
        if values.ndim != 1 or values.dtype.kind not in "iu":
            raise ValueError(name + " must be a one-dimensional integer array")
        if values.size and (int(values.min()) < 0 or int(values.max()) > _MAX_ID):
            raise ValueError(name + " contains an item outside nonnegative int64")
        result = values.astype(np.int64, copy=True)
    elif isinstance(values, (list, tuple)):
        result = np.asarray([_nonnegative_integer(value, name + " item") for value in values],
                            dtype=np.int64)
    else:
        raise ValueError(name + " must be an ordered list, tuple, or integer array")
    if len(np.unique(result)) != len(result):
        raise ValueError(name + " requires unique item IDs")
    return result


def guarded_head_order(learned_ids, prior_ids, *, coarse_pool: int = 100,
                       head_size: int = 10) -> np.ndarray:
    """Return a fresh int64 ranking; preserve coarse/full sets and the tail.

    Only prior items already inside the learned first ``coarse_pool`` positions
    can move to the front. Others retain their learned relative order. Item0 is
    valid; input order and contents are never modified. Inputs must be unique,
    ordered list/tuple or one-dimensional integer arrays, not scores/sets.
    """
    pool = _nonnegative_integer(coarse_pool, "coarse_pool")
    head = _nonnegative_integer(head_size, "head_size")
    if head > pool:
        raise ValueError("head_size must not exceed coarse_pool")
    learned = _ordered_ids(learned_ids, "learned_ids")
    prior = _ordered_ids(prior_ids, "prior_ids")
    if not head or not len(learned):
        return learned
    coarse = set(map(int, learned[:pool]))
    priority = []
    for item in prior:
        if int(item) in coarse:
            priority.append(int(item))
            if len(priority) == head:
                break
    selected = set(priority)
    remainder = learned[[int(item) not in selected for item in learned]]
    return np.concatenate((np.asarray(priority, dtype=np.int64), remainder))


def ordinal_rank_scores(count: int) -> np.ndarray:
    """FP64 display-order scores ``1/(60+rank)``; not model probabilities."""
    count = _nonnegative_integer(count, "count")
    return 1.0 / (60.0 + np.arange(1, count + 1, dtype=np.float64))
