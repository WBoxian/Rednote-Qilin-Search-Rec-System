"""在线精排服务编排层。"""

from __future__ import annotations

import pandas as pd

from backend.online.ranking.dien import apply_dien_scores


def run_ranking(
    cand: pd.DataFrame,
    page: int,
    page_size: int,
    predict_dien,
    history_note_ids: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """DIEN 精排服务：注入 DIEN 打分并按分数排序。"""
    if cand.empty:
        return cand, cand

    if "rank" in cand.columns:
        cand = cand.sort_values(["rank"], ascending=[True], kind="mergesort")
    cand = cand.drop_duplicates(subset=["note_idx"], keep="first").reset_index(drop=True)

    cand = apply_dien_scores(
        cand,
        page_size=page_size,
        predict_dien=predict_dien,
        history_note_ids=history_note_ids,
    )

    # 直接按 dien_score 排序
    cand = cand.sort_values(["dien_score", "rank"], ascending=[False, True], kind="mergesort").reset_index(drop=True)

    start = max(0, (int(page) - 1) * int(page_size))
    end = start + int(page_size)
    page_df = cand.iloc[start:end].copy()
    return cand, page_df
