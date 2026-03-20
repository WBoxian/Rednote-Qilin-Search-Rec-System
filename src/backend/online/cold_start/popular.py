from __future__ import annotations

import numpy as np
import pandas as pd


def build_hot_candidates(feat_req: pd.DataFrame, topk: int) -> pd.DataFrame:
    if feat_req.empty:
        return pd.DataFrame()
    cand = feat_req.copy()

    def _num_col(name: str) -> pd.Series:
        if name in cand.columns:
            return pd.to_numeric(cand[name], errors="coerce").fillna(0)
        return pd.Series(np.zeros(len(cand), dtype=np.float32), index=cand.index)

    hot_score = (
        1.0 * _num_col("accum_like_num")
        + 1.2 * _num_col("accum_collect_num")
        + 0.8 * _num_col("accum_comment_num")
        + 0.2 * _num_col("imp_num")
    )
    cand["hot_score"] = hot_score.astype(np.float32)
    cand = cand.sort_values("hot_score", ascending=False, kind="mergesort").head(max(0, int(topk))).copy()
    cand["rank"] = np.arange(1, len(cand) + 1)
    cand["recall_score"] = cand["hot_score"].astype(np.float32)
    cand["score_ann"] = 0.0
    cand["score_swing"] = 0.0
    cand["score_usercf"] = 0.0
    cand["from_ann"] = 0
    cand["from_swing"] = 0
    cand["from_usercf"] = 0
    cand["from_hot"] = 1
    cand["first_route"] = "hot"
    return cand
