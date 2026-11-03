"""Candidate-wise multi-interest IP objective for the actual retrieval vectors."""
from __future__ import annotations

import math
from numbers import Real

import torch


def candidatewise_max_ip(query_heads, items, valid_heads, *, temperature=.07):
    """Score [B,H,D] heads against [C,D] items, then max heads before CE.

    Callers supply the same scaled, concatenated blocks used by retrieval.
    Do not normalize those blocks again or drop a frozen scoring branch.
    """
    if (not isinstance(temperature, Real) or isinstance(temperature, bool)
            or not math.isfinite(temperature) or temperature <= 0):
        raise ValueError("Temperature must be finite and positive")
    if (query_heads.ndim != 3 or items.ndim != 2
            or query_heads.shape[2] != items.shape[1]
            or min(query_heads.shape) < 1 or len(items) < 1
            or query_heads.dtype != torch.float32 or items.dtype != torch.float32
            or valid_heads.dtype != torch.bool
            or tuple(valid_heads.shape) != tuple(query_heads.shape[:2])
            or query_heads.device != items.device or query_heads.device != valid_heads.device):
        raise ValueError("Expected aligned FP32 heads/items and boolean head mask")
    if (not bool(torch.isfinite(query_heads).all()) or not bool(torch.isfinite(items).all())
            or not bool(valid_heads.any(1).all())):
        raise ValueError("Every request needs a valid head and finite input vectors")
    scores = (query_heads.reshape(-1, query_heads.shape[-1]) @ items.T
              / temperature).reshape(*query_heads.shape[:2], len(items))
    return scores.masked_fill(~valid_heads[:, :, None], -torch.inf).max(1).values
