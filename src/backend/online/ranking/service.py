"""在线精排服务编排层。"""

from __future__ import annotations

import pandas as pd

from backend.online.ranking.dien import apply_dien_scores

RANK_TOPN = 500


def run_ranking(
    cand: pd.DataFrame,
    page: int,
    page_size: int,
    predict_dien,
    history_note_ids: list[int] | None = None,
    enable_dien: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """DIEN 精排服务：按 GBDT 候选顺序进入 DIEN，再按 DIEN 分数排序。"""
    if cand.empty:
        return cand, cand

    if "rank" in cand.columns:
        cand = cand.sort_values(["rank"], ascending=[True], kind="mergesort")
    cand = cand.drop_duplicates(subset=["note_idx"], keep="first").reset_index(drop=True)
    cand = cand.head(int(RANK_TOPN)).copy()

    if bool(enable_dien) and predict_dien is not None:
        cand = apply_dien_scores(
            cand,
            page_size=page_size,
            predict_dien=predict_dien,
            history_note_ids=history_note_ids,
        )
    else:
        cand["dien_score"] = pd.to_numeric(cand.get("gbdt_score", 0.0), errors="coerce").fillna(0.0)

    if "linkage_boost" in cand.columns:
        cand["dien_score"] = (
            pd.to_numeric(cand["dien_score"], errors="coerce").fillna(0.0)
            + pd.to_numeric(cand["linkage_boost"], errors="coerce").fillna(0.0) * 0.05
        )

    cand = cand.sort_values(["dien_score", "rank"], ascending=[False, True], kind="mergesort").reset_index(drop=True)

    start = max(0, (int(page) - 1) * int(page_size))
    end = start + int(page_size)
    page_df = cand.iloc[start:end].copy()
    return cand, page_df
