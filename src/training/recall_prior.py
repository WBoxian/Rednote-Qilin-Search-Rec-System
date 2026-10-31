"""Versioned residual recall-rank prior, shared by validation and online ranking."""
from __future__ import annotations
import math
import numpy as np

RECALL_PRIOR_CONTRACT = "reciprocal_recall_prior_v1"


def recall_prior_terms(recall_ranks, learned_scores, k=60):
    """Return reusable original/learned reciprocal terms for one or many requests.

    The final axis contains candidates; rows are independent requests. Rank zero
    is optional padding, never an item identity. Learned-score ties use the
    original recall rank. Existing item ID zero is unaffected by this operation.
    """
    ranks = np.asarray(recall_ranks, dtype=np.float64)
    scores = np.asarray(learned_scores, dtype=np.float64)
    if ranks.ndim not in {1, 2} or scores.shape != ranks.shape:
        raise ValueError("Recall ranks and learned scores must align in one or two dimensions")
    if not math.isfinite(float(k)) or int(k) != k or k <= 0:
        raise ValueError("Reciprocal rank offset must be a positive integer")
    if not np.isfinite(ranks).all() or (ranks < 0).any() or (ranks != np.floor(ranks)).any():
        raise ValueError("Recall ranks must be nonnegative finite integers")
    valid = ranks > 0
    if not np.isfinite(scores[valid]).all():
        raise ValueError("Valid candidates require finite learned scores")
    original = np.where(valid, ranks, np.inf)
    sorted_ranks = np.sort(original, axis=-1)
    if ((sorted_ranks[..., 1:] == sorted_ranks[..., :-1]) & np.isfinite(sorted_ranks[..., 1:])).any():
        raise ValueError("Original recall ranks must be unique per request")
    safe_scores = np.where(valid, scores, -np.inf)
    order = np.lexsort((original, -safe_scores), axis=-1)
    learned_rank = np.empty_like(order, dtype=np.int32)
    ordinal = np.broadcast_to(np.arange(1, ranks.shape[-1]+1, dtype=np.int32), order.shape)
    np.put_along_axis(learned_rank, order, ordinal, axis=-1)
    original_term = np.full(ranks.shape, -np.inf, dtype=np.float64)
    learned_term = np.zeros(ranks.shape, dtype=np.float64)
    np.divide(1., float(k)+ranks, out=original_term, where=valid)
    np.divide(1., float(k)+learned_rank, out=learned_term, where=valid)
    return original_term, learned_term


def blend_recall_prior(recall_ranks, learned_scores, alpha, k=60):
    """1/(k+original_rank) + alpha/(k+learned_rank), preserving float64."""
    alpha = float(alpha)
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("Recall-prior alpha must be finite and nonnegative")
    original, learned = recall_prior_terms(recall_ranks, learned_scores, k)
    return original + alpha*learned