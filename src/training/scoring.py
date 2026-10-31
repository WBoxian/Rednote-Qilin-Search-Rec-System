"""Versioned coarse/fine score fusion shared by evaluation and serving."""
from __future__ import annotations
import numpy as np
from training.rank_stack import group_minmax


def blend_rank_scores(coarse, fine, groups, weight, contract='request_minmax_v1'):
    weight=float(weight)
    base=np.asarray(coarse,dtype=np.float32)
    if base.ndim!=1 or not np.isfinite(base).all() or not 0<=weight<=1:
        raise ValueError('Invalid ranking scores or blend weight')
    if contract=='probability_v2':
        if ((base<0)|(base>1)).any():
            raise ValueError('Probability fusion requires a binary-classifier teacher')
        if weight==0:return base.copy()
        raw=np.asarray(fine,dtype=np.float32)
        if raw.shape!=base.shape or not np.isfinite(raw).all():raise ValueError('Invalid neural logits')
        prob=np.exp(-np.logaddexp(0.,-raw)).astype(np.float32)
        return ((1-weight)*base+weight*prob).astype(np.float32)
    if contract=='request_minmax_v1':
        normalized=group_minmax(base,groups)
        if weight==0:return normalized
        raw=np.asarray(fine,dtype=np.float32)
        if raw.shape!=base.shape or not np.isfinite(raw).all():raise ValueError('Invalid neural logits')
        return ((1-weight)*normalized+weight*group_minmax(raw,groups)).astype(np.float32)
    raise ValueError(f'Unsupported score contract: {contract}')