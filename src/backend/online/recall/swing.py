from __future__ import annotations

import pandas as pd


def pick_swing_route(recall: pd.DataFrame) -> pd.DataFrame:
    if recall.empty:
        return recall
    return recall[pd.to_numeric(recall.get("from_swing", 0), errors="coerce").fillna(0).astype(int) > 0].copy()
