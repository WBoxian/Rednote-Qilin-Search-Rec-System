from __future__ import annotations

import pandas as pd


def pick_usercf_route(recall: pd.DataFrame) -> pd.DataFrame:
    if recall.empty:
        return recall
    return recall[pd.to_numeric(recall.get("from_usercf", 0), errors="coerce").fillna(0).astype(int) > 0].copy()
