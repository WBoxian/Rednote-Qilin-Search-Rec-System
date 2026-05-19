"""在线精排服务编排层。"""

from __future__ import annotations

import os

import pandas as pd

from backend.online.ranking.dien import apply_dien_scores

RANK_TOPN = int(os.getenv("QILIN_ONLINE_RANK_TOPN", "300"))


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

    if "query_match_score" in cand.columns or "query_exact_hit" in cand.columns:
        exact = pd.to_numeric(cand.get("query_exact_hit", 0.0), errors="coerce").fillna(0.0)
        cover = pd.to_numeric(cand.get("query_term_cover", 0.0), errors="coerce").fillna(0.0)
        lexical = pd.to_numeric(cand.get("query_match_score", 0.0), errors="coerce").fillna(0.0)
        pre_score = pd.to_numeric(cand.get("preranking_score", cand.get("gbdt_score", 0.0)), errors="coerce").fillna(0.0)
        strong = ((exact > 0) | (cover >= 0.45) | (lexical >= 1.25)).astype("float32")
        hard_penalty = ((exact <= 0) & (cover < 0.20) & (lexical < 0.80)).astype("float32") * 2.0
        cand["query_strong_match"] = strong.astype("int8")
        base_gap = (
            pd.to_numeric(cand.get("dien_score", 0.0), errors="coerce").fillna(0.0)
            - pre_score
        )
        adjusted_dien = (
            pd.to_numeric(cand["dien_score"], errors="coerce").fillna(0.0)
            + lexical * 0.18
            + exact * 1.05
            + cover * 1.25
            + strong * 0.90
            + base_gap * 0.16
            - hard_penalty
        )
        weak_penalty = (
            (exact <= 0)
            & (cover < 0.34)
        ).astype("float32") * 0.65
        adjusted_dien = pd.to_numeric(adjusted_dien, errors="coerce").fillna(0.0) - weak_penalty
        cand["dien_score"] = adjusted_dien.astype("float32")
        cand["final_score"] = (
            pre_score * 0.72
            + adjusted_dien * 0.28
            + lexical * 0.04
            + cover * 0.08
            + exact * 0.12
        ).astype("float32")
    else:
        cand["final_score"] = pd.to_numeric(cand.get("dien_score", 0.0), errors="coerce").fillna(0.0).astype("float32")
    sort_cols = ["final_score", "rank"]
    sort_asc = [False, True]
    if "query_term_cover" in cand.columns:
        sort_cols = ["query_strong_match", "final_score", "query_exact_hit", "query_term_cover", "rank"]
        sort_asc = [False, False, False, False, True]
    cand = cand.sort_values(sort_cols, ascending=sort_asc, kind="mergesort").reset_index(drop=True)
    if "query_strong_match" in cand.columns:
        strong_mask = pd.to_numeric(cand.get("query_strong_match", 0), errors="coerce").fillna(0).astype(int) > 0
        strong_df = cand[strong_mask].copy()
        weak_df = cand[~strong_mask].copy()
        if len(strong_df) >= max(10, min(int(page_size) * 2, 20)):
            cand = strong_df.reset_index(drop=True)
        elif len(strong_df) > 0:
            weak_keep = weak_df[
                (pd.to_numeric(weak_df.get("query_term_cover", 0.0), errors="coerce").fillna(0.0) >= 0.24)
                | (pd.to_numeric(weak_df.get("query_match_score", 0.0), errors="coerce").fillna(0.0) >= 0.90)
            ].copy()
            cand = pd.concat([strong_df, weak_keep.head(max(0, len(cand) - len(strong_df)))], ignore_index=True)

    start = max(0, (int(page) - 1) * int(page_size))
    end = start + int(page_size)
    page_df = cand.iloc[start:end].copy()
    return cand, page_df
