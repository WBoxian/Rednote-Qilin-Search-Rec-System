"""在线 preranking 服务编排层。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.online.preranking.gbdt import run_preranking_gbdt

_USER_SCALAR_COLS = (
    ["gender_enc", "platform_enc", "age_enc", "location_enc", "fans_num", "follows_num"]
    + [f"dense_feat{i}" for i in range(1, 41)]
)


def _valid_history_value(v) -> bool:
    if v is None:
        return False
    if isinstance(v, float) and np.isnan(v):
        return False
    if isinstance(v, np.ndarray):
        return v.size > 0
    if isinstance(v, (list, tuple)):
        return len(v) > 0
    return True


def run_preranking(
    user_idx: int,
    query: str,
    scene: str,
    group_key: str,
    recall_cand: pd.DataFrame,
    feat_req: pd.DataFrame,
    gbdt_topn: int,
    fetch_notes,
    predict_gbdt,
) -> pd.DataFrame:
    # 召回候选拼接特征与内容元信息，再进入 GBDT preranking
    if recall_cand.empty:
        return pd.DataFrame()

    cand = recall_cand.merge(
        feat_req,
        on=[group_key, "note_idx"],
        how="left",
        suffixes=("", "_feat"),
    )
    cand["note_idx"] = pd.to_numeric(cand.get("note_idx"), errors="coerce")
    cand = cand[cand["note_idx"].notna()].copy()
    if cand.empty:
        return pd.DataFrame()
    cand["note_idx"] = cand["note_idx"].astype("int64")
    cand = cand[cand["note_idx"] >= 0].copy()
    if cand.empty:
        return pd.DataFrame()

    if "rank" in cand.columns:
        cand = cand.sort_values([group_key, "rank"], ascending=[True, True], kind="mergesort")
    elif "recall_score" in cand.columns:
        cand = cand.sort_values([group_key, "recall_score"], ascending=[True, False], kind="mergesort")
    cand = cand.drop_duplicates(subset=[group_key, "note_idx"], keep="first").reset_index(drop=True)
    if "user_idx" not in cand.columns:
        cand["user_idx"] = int(user_idx)
    cand["user_idx"] = cand["user_idx"].fillna(int(user_idx)).astype("int64")

    if len(feat_req) > 0:
        user_profile = feat_req.iloc[0]
        for col in _USER_SCALAR_COLS:
            val = user_profile.get(col)
            if val is None:
                continue
            try:
                fill_val = float(val) if not isinstance(val, (int, float)) else val
                if isinstance(fill_val, float) and np.isnan(fill_val):
                    continue
            except Exception:
                continue
            if col not in cand.columns:
                cand[col] = fill_val
            else:
                cand[col] = pd.to_numeric(cand[col], errors="coerce").fillna(fill_val)
        if "recent_clicked_note_idxs" in user_profile:
            hist_val = user_profile["recent_clicked_note_idxs"]
            if "recent_clicked_note_idxs" not in cand.columns:
                cand["recent_clicked_note_idxs"] = [hist_val] * len(cand)
            else:
                cand["recent_clicked_note_idxs"] = [
                    v if _valid_history_value(v) else hist_val
                    for v in cand["recent_clicked_note_idxs"]
                ]

    note_cols = ["accum_like_num", "accum_collect_num", "accum_comment_num", "image_path", "note_title", "note_content"]
    need_note_meta = any((col not in cand.columns) or cand[col].isna().all() for col in note_cols)
    if need_note_meta:
        note_meta = fetch_notes(cand["note_idx"].drop_duplicates().astype(int).tolist(), include_text=False)
        if not note_meta.empty:
            cand = cand.merge(note_meta, on="note_idx", how="left", suffixes=("", "_note"))

    if scene == "search":
        cand["query"] = str(query or "").strip()

    for col in note_cols:
        note_col = f"{col}_note"
        if note_col in cand.columns:
            if col in cand.columns:
                cand[col] = cand[col].where(cand[col].notna(), cand[note_col])
            else:
                cand[col] = cand[note_col]

    ann_score = pd.to_numeric(cand.get("score_ann", 0.0), errors="coerce").fillna(0.0)
    recall_score = pd.to_numeric(cand.get("recall_score", ann_score), errors="coerce").fillna(0.0)
    cand["dssm_score"] = ann_score.where(ann_score >= recall_score, recall_score).astype("float32")

    return run_preranking_gbdt(cand=cand, gbdt_topn=gbdt_topn, predict_gbdt=predict_gbdt)
