"""Versioned source-history order shared by retrieval and ranking.

Qilin snapshots are interpreted as most recent first. This is an explicit input
contract supported by observed transitions, not proof of per-event timestamps.
Existing checkpoint/embedding files must retain their recorded legacy contract.
"""
from __future__ import annotations

import math

HISTORY_ORDER = "most_recent_first_v1"
SEQUENCE_ORDER = "oldest_first_v1"


def recent_history_ids(values, max_len: int | None = 20) -> list[int]:
    """Return the most recent prefix, preserving valid item zero and repeats."""
    if max_len is not None and (not isinstance(max_len, int) or max_len < 0):
        raise ValueError("History maximum length must be a nonnegative integer or None")
    if values is None or max_len == 0:
        return []
    if isinstance(values, (str, bytes)):
        raise ValueError("History must be a sequence of item IDs")
    ids = []
    for value in values:
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("History item IDs must be numeric") from exc
        if not math.isfinite(numeric) or numeric < 0:
            continue
        item = int(numeric)
        if numeric != item:
            raise ValueError("History item IDs must be integral")
        ids.append(item)
        if max_len is not None and len(ids) == max_len:
            break
    return ids


def history_ids_oldest_first(values, max_len: int | None = 20) -> list[int]:
    """Select recent history first, then reverse once for chronological RNN input."""
    return recent_history_ids(values, max_len)[::-1]