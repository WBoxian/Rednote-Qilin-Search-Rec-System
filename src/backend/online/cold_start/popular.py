"""Cold candidates from an explicit training-derived catalog pool."""
from __future__ import annotations
import numpy as np
import pandas as pd


def build_hot_candidates(feat_req: pd.DataFrame, topk: int, note_ids: list[int] | None = None) -> pd.DataFrame:
    # Request exposure rows are never a global popularity index or fallback candidate universe.
    ids = list(dict.fromkeys(int(note) for note in (note_ids or []) if int(note) >= 0))[:max(0, int(topk))]
    if not ids or feat_req.empty:
        return pd.DataFrame()
    context = feat_req.iloc[0]
    fields = [name for name in ("search_idx", "request_idx", "user_idx", "query", "recent_clicked_note_idxs",
        "gender_enc", "platform_enc", "age_enc", "location_enc") if name in context]
    cand = pd.DataFrame([{**{name: context[name] for name in fields}, "note_idx": note} for note in ids])
    cand["rank"] = np.arange(1, len(cand) + 1)
    cand["recall_score"] = (1.0 / cand["rank"]).astype(np.float32)
    for name in ("score_ann", "score_swing", "score_usercf", "from_ann", "from_swing", "from_usercf"):
        cand[name] = 0
    cand["from_hot"] = 1
    cand["first_route"] = "hot"
    return cand