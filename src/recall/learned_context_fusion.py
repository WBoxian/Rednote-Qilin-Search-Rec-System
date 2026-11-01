"""Frozen 23-feature Rec fusion and alpha=.5 rank blending.

The prior is deliberately the float32 rrf211 feature from the validated
prototype. Recomputing it in float64 changes near-tie ranks. This module consumes
only request context, original PCA vectors and past support; it never reads data
files or labels, fits a model, or falls back to another recall strategy.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from types import MappingProxyType

import numpy as np

from training.history import recent_history_ids

ROUTE_NAMES = ("ann", "swing", "usercf")
FEATURE_NAMES = (
    "ann_present", "ann_rank", "ann_rr",
    "swing_present", "swing_rank", "swing_rr",
    "usercf_present", "usercf_rank", "usercf_rr",
    "swing_norm", "usercf_norm", "route_count", "rrf211",
    "pca_query_cos", "pca_recent_cos", "pca_hist_max", "pca_hist_mean",
    "history_len", "history_unique_ratio", "history_coherence",
    "history_graph_support", "log_item_support", "log_user_support",
)
_PRIOR_COLUMN = FEATURE_NAMES.index("rrf211")
_MAX_ID = np.iinfo(np.int64).max


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a nonnegative integer")
    if not 0 <= int(value) <= _MAX_ID:
        raise ValueError(f"{name} is outside the int64 ID/count range")
    return int(value)


def _ids(values, name, *, maximum=None):
    if isinstance(values, (list, tuple)) and any(isinstance(v, (bool, np.bool_)) for v in values):
        raise ValueError(f"{name} cannot contain boolean IDs")
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a one-dimensional integer sequence") from error
    if array.ndim != 1 or (maximum is not None and len(array) > maximum):
        raise ValueError(f"{name} has an invalid shape or budget")
    if len(array) and (array.dtype.kind not in "iu" or np.any(array < 0)
                       or int(array.max()) > _MAX_ID):
        raise ValueError(f"{name} must contain nonnegative int64 IDs")
    result = array.astype(np.int64, copy=True)
    result.setflags(write=False)
    return result


def _real(values, name, *, dtype):
    if isinstance(values, (list, tuple)) and any(isinstance(v, (bool, np.bool_)) for v in values):
        raise ValueError(f"{name} cannot contain booleans")
    array = np.asarray(values)
    if array.dtype.kind not in "iuf" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite real numbers")
    with np.errstate(over="ignore", invalid="ignore"):
        result = array.astype(dtype, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} is not representable as {np.dtype(dtype)}")
    return result


def _schema(names):
    try:
        matches = tuple(names) == FEATURE_NAMES
    except TypeError:
        matches = False
    if not matches:
        raise ValueError("Learned fusion requires the exact ordered 23-feature schema")


def _cpu_threads(value):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
            or int(value) not in (2, 3)):
        raise ValueError("Learned fusion CPU prediction requires 2 or 3 threads")
    return int(value)


@dataclass(frozen=True)
class RouteCandidates:
    """One original rank-ordered Top1000 route; native ANN scores are not features."""
    ids: np.ndarray
    scores: np.ndarray

    def __post_init__(self):
        ids = _ids(self.ids, "Route IDs", maximum=1000)
        scores = _real(self.scores, "Route scores", dtype=np.float32)
        if scores.ndim != 1 or len(scores) != len(ids) or len(np.unique(ids)) != len(ids):
            raise ValueError("Route IDs must be unique and aligned with one-dimensional scores")
        scores.setflags(write=False)
        object.__setattr__(self, "ids", ids)
        object.__setattr__(self, "scores", scores)


@dataclass(frozen=True)
class PastSupport:
    """Immutable click counts and graph membership prepared from the caller's past."""
    item_counts: Mapping
    user_counts: Mapping
    graph_items: frozenset

    def __post_init__(self):
        for field in ("item_counts", "user_counts"):
            values = getattr(self, field)
            if not isinstance(values, Mapping):
                raise ValueError(f"{field} must be an ID-to-count mapping")
            copied = {_integer(key, field + " key"): _integer(value, field + " count")
                      for key, value in values.items()}
            object.__setattr__(self, field, MappingProxyType(copied))
        try:
            graph = _ids(tuple(self.graph_items), "Graph item IDs")
        except TypeError as error:
            raise ValueError("Graph items must be a collection of integer IDs") from error
        object.__setattr__(self, "graph_items", frozenset(map(int, graph)))


@dataclass(frozen=True)
class CandidateFeatures:
    note_ids: np.ndarray
    values: np.ndarray
    route_ranks: np.ndarray
    feature_names: tuple = FEATURE_NAMES

    def __post_init__(self):
        _schema(self.feature_names)
        ids = _ids(self.note_ids, "Union IDs", maximum=3000)
        values = np.asarray(self.values)
        ranks = np.asarray(self.route_ranks)
        if len(ids) and np.any(ids[1:] <= ids[:-1]):
            raise ValueError("Union IDs must be unique and sorted ascending")
        if (values.shape != (len(ids), len(FEATURE_NAMES))
                or values.dtype != np.dtype(np.float32) or not np.isfinite(values).all()):
            raise ValueError("Features must be finite float32[N,23]")
        if (ranks.shape != (len(ids), 3) or ranks.dtype.kind not in "iu"
                or np.any(ranks < 1) or np.any(ranks > 1001)):
            raise ValueError("Route ranks must be integer[N,3] in [1,1001]")
        values = values.copy()
        ranks = ranks.astype(np.int32, copy=True)
        values.setflags(write=False)
        ranks.setflags(write=False)
        object.__setattr__(self, "note_ids", ids)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "route_ranks", ranks)
        object.__setattr__(self, "feature_names", FEATURE_NAMES)


def _pca_vectors(lookup, ids):
    vectors = _real(lookup(ids), "Original item PCA vectors", dtype=np.float32)
    if vectors.shape != (len(ids), 128):
        raise ValueError("Original item PCA lookup must return [N,128]")
    return vectors


def build_candidate_features(routes: Mapping[str, RouteCandidates], *, history,
                             user_idx, q0_pca, item_pca_lookup: Callable,
                             support: PastSupport) -> CandidateFeatures:
    """Apply canonical recent20 filtering and the prototype's exact feature math."""
    if not isinstance(routes, Mapping) or set(routes) - set(ROUTE_NAMES):
        raise ValueError("Only ann, swing and usercf routes are supported")
    if any(not isinstance(route, RouteCandidates) for route in routes.values()):
        raise ValueError("Each route must be an explicit RouteCandidates")
    if not isinstance(support, PastSupport) or not callable(item_pca_lookup):
        raise ValueError("Explicit past support and original PCA lookup are required")
    uid = _integer(user_idx, "User ID")
    query = _real(q0_pca, "Canonical query PCA", dtype=np.float32)
    if query.shape != (128,):
        raise ValueError("Canonical query PCA must have shape [128]")
    try:
        hist = recent_history_ids(history, 20)
    except TypeError as error:
        raise ValueError("History must follow the canonical item sequence contract") from error
    hist_ids = _ids(hist, "Canonical history")
    consumed = set(hist)
    parts = []
    for name in ROUTE_NAMES:
        route = routes.get(name)
        if route is None:
            parts.append((np.empty(0, np.int64), np.empty(0, np.float32)))
        else:
            keep = np.array([int(item) not in consumed for item in route.ids], dtype=bool)
            parts.append((route.ids[keep], route.scores[keep]))
    union = np.unique(np.concatenate([ids for ids, _ in parts]))
    size = len(union)
    if not size:
        return CandidateFeatures(union, np.empty((0, 23), np.float32),
                                 np.empty((0, 3), np.int32))
    ranks = np.full((size, 3), 1001, np.int32)
    norms = np.zeros((size, 2), np.float32)
    for channel, (ids, scores) in enumerate(parts):
        where = np.searchsorted(union, ids)
        ranks[where, channel] = np.arange(1, len(ids) + 1)
        if channel and len(scores):
            with np.errstate(over="ignore", invalid="ignore"):
                norms[where, channel - 1] = scores / max(float(scores.max()), 1e-12)
    present = ranks < 1001
    rr = np.where(present, 1 / (60 + ranks), 0)
    columns = []
    for channel in range(3):
        columns.extend([present[:, channel], ranks[:, channel], rr[:, channel]])
    vectors = _pca_vectors(item_pca_lookup, union)
    history_vectors = (_pca_vectors(item_pca_lookup, hist_ids) if hist
                       else np.empty((0, 128), np.float32))
    cos = (vectors @ history_vectors.T if hist
           else np.zeros((size, 1), np.float32))
    coherence = float(np.linalg.norm(history_vectors.mean(0))) if hist else 0.
    columns.extend([
        norms[:, 0], norms[:, 1], present.sum(1), rr @ np.array([2, 1, 1]),
        vectors @ query, cos[:, 0], cos.max(1), cos.mean(1),
        np.full(size, len(hist)), np.full(size, len(consumed) / max(len(hist), 1)),
        np.full(size, coherence),
        np.full(size, sum(item in support.graph_items for item in hist) / max(len(hist), 1)),
        np.log1p([support.item_counts.get(int(item), 0) for item in union]),
        np.full(size, np.log1p(support.user_counts.get(uid, 0))),
    ])
    return CandidateFeatures(union, np.stack(columns, axis=1).astype(np.float32), ranks)


@dataclass(frozen=True)
class RankedCandidates:
    note_idx: np.ndarray
    score: np.ndarray
    rank: np.ndarray

    def __post_init__(self):
        ids = _ids(self.note_idx, "Ranked IDs", maximum=500)
        scores = _real(self.score, "Blend scores", dtype=np.float64)
        ranks = _ids(self.rank, "Final ranks")
        if (scores.shape != (len(ids),) or len(np.unique(ids)) != len(ids)
                or not np.array_equal(ranks, np.arange(1, len(ids) + 1))):
            raise ValueError("Final candidates must have aligned scores and contiguous ranks")
        scores.setflags(write=False)
        object.__setattr__(self, "note_idx", ids)
        object.__setattr__(self, "score", scores)
        object.__setattr__(self, "rank", ranks)

    def to_frame(self):
        import pandas as pd
        return pd.DataFrame({"note_idx": self.note_idx, "score": self.score, "rank": self.rank})


def rank_candidates(features: CandidateFeatures, predictions, *, head_guard=False) -> RankedCandidates:
    """Fixed alpha=.5/RRF60/top500; float32 prior ties are intentional."""
    if type(head_guard) is not bool:
        raise ValueError("head_guard must be an explicit boolean")
    if not isinstance(features, CandidateFeatures):
        raise ValueError("Validated CandidateFeatures are required")
    _schema(features.feature_names)
    ids = features.note_ids
    pred = _real(predictions, "Learned predictions", dtype=np.float64)
    if pred.shape != (len(ids),):
        raise ValueError("Predictions must have shape [N]")
    learned_order = np.lexsort((ids, -pred))
    learned_ranks = np.empty(len(ids), dtype=int)
    learned_ranks[learned_order] = np.arange(1, len(ids) + 1)
    prior = features.values[:, _PRIOR_COLUMN]  # Deliberately retain prototype float32.
    prior_order = np.lexsort((ids, -prior))
    prior_ranks = np.empty(len(ids), dtype=int)
    prior_ranks[prior_order] = np.arange(1, len(ids) + 1)
    blend = .5 / (60 + learned_ranks) + .5 / (60 + prior_ranks)
    take = np.lexsort((ids, -blend))[:500]
    if head_guard:
        from recall.head_guard import guarded_head_order, ordinal_rank_scores
        # The head control is the existing FP64 weighted-RRF Top500, whereas
        # the learned blend above intentionally retains its trained FP32 prior.
        head_prior = np.zeros(len(ids), np.float64)
        for channel, weight in enumerate((2., 1., 1.)):
            present = features.route_ranks[:, channel] < 1001
            head_prior[present] += weight / (60 + features.route_ranks[present, channel])
        head_order = np.lexsort((ids, -head_prior))[:500]
        ordered = guarded_head_order(ids[take], ids[head_order], coarse_pool=100, head_size=10)
        return RankedCandidates(ordered, ordinal_rank_scores(len(ordered)),
                                np.arange(1, len(ordered) + 1))
    return RankedCandidates(ids[take], blend[take], np.arange(1, len(take) + 1))


class LearnedContextFusion:
    """An explicitly supplied schema-checked predictor; no fallback model/path."""
    def __init__(self, model, *, num_threads=2):
        self.num_threads = _cpu_threads(num_threads)
        self.model = model
        self._validate_model()

    def _validate_model(self):
        if (not callable(getattr(self.model, "feature_name", None))
                or not callable(getattr(self.model, "num_feature", None))
                or not callable(getattr(self.model, "predict", None))):
            raise ValueError("Predictor must expose feature_name, num_feature and predict")
        _schema(self.model.feature_name())
        if _integer(self.model.num_feature(), "Model feature count") != len(FEATURE_NAMES):
            raise ValueError("Learned fusion model must have exactly 23 features")

    @classmethod
    def from_model_path(cls, path, *, num_threads=2):
        threads = _cpu_threads(num_threads)
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(source)
        import lightgbm as lgb
        return cls(lgb.Booster(model_file=str(source), params={"num_threads": threads, "device_type": "cpu"}),
                   num_threads=threads)

    def predict(self, features: CandidateFeatures):
        if not isinstance(features, CandidateFeatures):
            raise ValueError("Validated CandidateFeatures are required")
        self._validate_model()
        _schema(features.feature_names)
        if not len(features.note_ids):
            return np.empty(0, np.float64)
        predictions = _real(self.model.predict(features.values, num_threads=self.num_threads),
                            "Learned predictions", dtype=np.float64)
        if predictions.shape != (len(features.note_ids),):
            raise ValueError("Predictor returned an invalid shape")
        return predictions

    def fuse(self, routes, *, history, user_idx, q0_pca, item_pca_lookup, support, head_guard=False):
        features = build_candidate_features(routes, history=history, user_idx=user_idx,
            q0_pca=q0_pca, item_pca_lookup=item_pca_lookup, support=support)
        return rank_candidates(features, self.predict(features), head_guard=head_guard)

