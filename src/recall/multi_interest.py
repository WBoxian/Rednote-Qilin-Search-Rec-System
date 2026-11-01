"""Deterministic K3 history clustering; no model, retrieval, or policy loading.

The caller selects the provided recent20 in source order. A required callback
supplies the existing, normalized PCA token representation; using adapted query
vectors here changes the recipe. This function never truncates or deduplicates
members and never replaces zero/invalid rows with a valid item such as note0.
"""
from __future__ import annotations

import numpy as np


def _normalize(values):
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


def k3_history_means(raw_tokens, *, cluster_transform, valid_mask=None):
    """Return L2-normalized raw member means [K,D] float32 and source metadata.

    Exactly first-valid/farthest-first <=3 seeds, four spherical-kmeans updates,
    and earliest-token/lowest-head tie rules match the frozen K3 recipe. The
    transform receives a copy of finite, nonzero raw float32 rows and must return
    finite normalized float32 [valid_rows,P] PCA representations, including zero
    projected rows if applicable. It is not applied to the output raw means.

    A separate boolean valid_mask can exclude padding regardless of its values.
    All-invalid/empty input returns [0,D], metadata fallback=True; the caller
    retains its original global query. Global prefixes and query adapters are
    outside this pure operation. Inputs are never modified.
    """
    values = np.asarray(raw_tokens)
    if values.ndim != 2 or values.shape[1] == 0 or values.dtype.kind not in "fiu":
        raise ValueError("raw_tokens must be a two-dimensional numeric matrix with D>0")
    if not callable(cluster_transform):
        raise ValueError("An explicit PCA cluster_transform callback is required")
    mask = np.ones(len(values), dtype=bool)
    if valid_mask is not None:
        supplied = np.asarray(valid_mask)
        if supplied.dtype != np.bool_ or supplied.shape != (len(values),):
            raise ValueError("valid_mask must be a separate boolean vector of length N")
        mask &= supplied
    raw = values.astype(np.float32)
    valid = np.isfinite(raw).all(1) & (np.linalg.norm(raw, axis=1) > 0) & mask
    offsets = np.flatnonzero(valid)
    raw = raw[valid]
    if not len(raw):
        return np.empty((0, values.shape[1]), np.float32), {
            "source_tokens": 0, "heads": 0, "fallback": True,
            "valid_indices": [], "seed_indices": [], "assignments": [-1] * len(values),
            "cluster_ids": [], "member_indices": [],
        }
    projected = np.asarray(cluster_transform(raw.copy()))
    if (projected.ndim != 2 or projected.shape[0] != len(raw) or projected.shape[1] == 0
            or projected.dtype != np.float32 or not np.isfinite(projected).all()):
        raise ValueError("cluster_transform must return finite float32 [valid_rows,P] PCA tokens")
    seeds = [0]
    for _ in range(2):
        distance = 1 - np.max(projected @ projected[seeds].T, axis=1)
        selected = int(np.argmax(distance))
        if distance[selected] <= 1e-6:
            break
        seeds.append(selected)
    centers = projected[seeds].copy()
    for _ in range(4):
        labels = np.argmax(projected @ centers.T, axis=1)
        for head in range(len(centers)):
            members = projected[labels == head]
            if len(members):
                centers[head] = _normalize(members.mean(0))
    labels = np.argmax(projected @ centers.T, axis=1)
    means, member_indices, cluster_ids = [], [], []
    for head in range(len(centers)):
        selected = labels == head
        if not selected.any():
            continue
        # Preserve raw_history_text_mean operand order: raw float32 mean, scalar
        # norm, scalar max, division. Averaging normalized tokens is different.
        mean = raw[selected].astype(np.float32).mean(0)
        length = np.linalg.norm(mean)
        means.append((mean / max(float(length), 1e-12)).astype(np.float32))
        member_indices.append(offsets[selected].tolist())
        cluster_ids.append(head)
    assignments = np.full(len(values), -1, dtype=np.int64)
    assignments[offsets] = labels
    return np.asarray(means, np.float32), {
        "source_tokens": len(raw), "heads": len(means), "fallback": False,
        "valid_indices": offsets.tolist(), "seed_indices": offsets[seeds].tolist(),
        "assignments": assignments.tolist(), "cluster_ids": cluster_ids,
        "member_indices": member_indices,
    }
