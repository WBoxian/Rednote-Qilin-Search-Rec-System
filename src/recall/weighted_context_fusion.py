"""Validation-selected weighting of the original three Rec routes."""
from __future__ import annotations

from collections.abc import Mapping
import math

import numpy as np
import pandas as pd

from training.history import recent_history_ids

ROUTES = ("ann", "swing", "usercf")


def fuse_weighted_request_routes(routes: Mapping, history, weights: Mapping,
                                topk: int = 500, rrf_k: int = 60):
    """Filter original Top1000 arrays, compact ranks, then apply weighted RRF.

    No labels enter this function. Zero-weight channels contribute no items;
    filtering never retrieves replacement candidates. Real note0 is valid.
    """
    if set(routes) - set(ROUTES) or set(weights) != set(ROUTES):
        raise ValueError("Only the declared DSSM/Swing/UserCF routes are supported")
    if isinstance(topk, bool) or not isinstance(topk, int) or not 1 <= topk <= 500 or rrf_k != 60:
        raise ValueError("Unsupported fixed fusion budget or RRF constant")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or value < 0 for value in weights.values()) or not any(weights.values()):
        raise ValueError("Fusion weights must be finite nonnegative numbers with positive mass")
    consumed = set(recent_history_ids(history, 20))
    scores = {}
    for name in ROUTES:
        values = np.asarray(routes.get(name, []))
        if values.ndim != 1 or len(values) > 1000:
            raise ValueError("Routes must contain only their original rank-ordered Top1000 IDs")
        if len(values) and not np.issubdtype(values.dtype, np.integer):
            raise ValueError("Route item IDs must be integers")
        ids = list(map(int, values))
        if len(set(ids)) != len(ids) or any(value < 0 for value in ids):
            raise ValueError("Route identities must be unique and nonnegative")
        if not weights[name]:
            continue
        filtered = (item for item in ids if item not in consumed)
        for rank, item in enumerate(filtered, 1):
            scores[item] = scores.get(item, 0.) + float(weights[name]) / (rrf_k + rank)
    order = sorted(scores, key=lambda item: (-scores[item], item))[:topk]
    return pd.DataFrame({"note_idx": pd.Series(order, dtype="int64"),
                         "score": pd.Series([scores[item] for item in order], dtype="float64"),
                         "rank": pd.Series(np.arange(1, len(order)+1), dtype="int64")})
