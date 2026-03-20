"""在线召回服务编排层。"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import duckdb
import numpy as np
import pandas as pd

from backend.online.cold_start.popular import build_hot_candidates
from backend.online.recall.dssm import DSSMRecaller


def snake_merge_routes(recall: pd.DataFrame, hot: pd.DataFrame | None, topk: int) -> pd.DataFrame:
    # 蛇形融合：多路轮询取候选，避免单路召回垄断
    if recall.empty and (hot is None or hot.empty):
        return pd.DataFrame()

    def _route_df(df: pd.DataFrame, route: str) -> pd.DataFrame:
        if df.empty:
            return df
        if route == "ann":
            x = df[pd.to_numeric(df.get("from_ann", 0), errors="coerce").fillna(0).astype(int) > 0].copy()
        elif route == "swing":
            x = df[pd.to_numeric(df.get("from_swing", 0), errors="coerce").fillna(0).astype(int) > 0].copy()
        elif route == "usercf":
            x = df[pd.to_numeric(df.get("from_usercf", 0), errors="coerce").fillna(0).astype(int) > 0].copy()
        else:
            x = df.copy()
        if "rank" in x.columns:
            x = x.sort_values("rank", ascending=True, kind="mergesort")
        return x.reset_index(drop=True)

    pools: dict[str, pd.DataFrame] = {
        "ann": _route_df(recall, "ann"),
        "swing": _route_df(recall, "swing"),
        "usercf": _route_df(recall, "usercf"),
        "hot": (hot.copy().reset_index(drop=True) if hot is not None and len(hot) > 0 else pd.DataFrame()),
    }
    order = ["ann", "swing", "usercf", "hot"]
    idx = {k: 0 for k in order}
    picked: list[pd.Series] = []
    seen: set[int] = set()

    while len(picked) < int(topk):
        moved = False
        for r in order:
            df = pools[r]
            i = idx[r]
            while i < len(df):
                row = df.iloc[i]
                i += 1
                nid = int(row.get("note_idx", -1))
                if nid < 0 or nid in seen:
                    continue
                seen.add(nid)
                picked.append(row)
                moved = True
                break
            idx[r] = i
            if len(picked) >= int(topk):
                break
        if not moved:
            break

    if not picked:
        return pd.DataFrame()
    out = pd.DataFrame(picked).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def run_recall(
    request_id: int,
    user_idx: int,
    feat_req: pd.DataFrame,
    is_cold: bool,
    recall_rank_cap: int,
    hot_route_topk: int,
    fetch_recall_candidates,
    group_key: str,
) -> pd.DataFrame:
    # 冷启动优先热榜兜底，非冷启动走多路召回融合
    hot = build_hot_candidates(feat_req, topk=min(int(hot_route_topk), int(recall_rank_cap)))
    if is_cold:
        if hot.empty:
            return pd.DataFrame()
        return hot.drop_duplicates(subset=["note_idx"], keep="first").reset_index(drop=True)

    recall = fetch_recall_candidates(request_id, max_rank=int(recall_rank_cap))
    merged = snake_merge_routes(recall, hot=None, topk=int(recall_rank_cap))
    if not merged.empty:
        return merged.drop_duplicates(subset=["note_idx"], keep="first").reset_index(drop=True)

    if not hot.empty:
        return hot.drop_duplicates(subset=["note_idx"], keep="first").reset_index(drop=True)

    if feat_req.empty:
        return pd.DataFrame()

    cand = feat_req.copy()
    cand["rank"] = np.arange(1, len(cand) + 1)
    cand["recall_score"] = 0.0
    cand["score_ann"] = 0.0
    cand["score_swing"] = 0.0
    cand["score_usercf"] = 0.0
    cand["from_ann"] = 0
    cand["from_swing"] = 0
    cand["from_usercf"] = 0
    cand["from_hot"] = 1
    cand["first_route"] = "hot"
    if group_key not in cand.columns:
        cand[group_key] = int(request_id)
    if "user_idx" not in cand.columns:
        cand["user_idx"] = int(user_idx)
    return cand.drop_duplicates(subset=["note_idx"], keep="first").reset_index(drop=True)


def fetch_recall_candidates(
    request_id: int,
    max_rank: int,
    group_key: str,
    req_df: pd.DataFrame,
    get_feat_req: Callable[[int], pd.DataFrame],
    dssm_recaller: DSSMRecaller | None,
    recall_test_path: Path | None,
) -> pd.DataFrame:
    if dssm_recaller is not None:
        dssm_df = dssm_recaller.fetch_candidates(
            request_id=int(request_id),
            max_rank=int(max_rank),
            req_df=req_df,
            get_feat_req=get_feat_req,
            group_key=group_key,
        )
        if not dssm_df.empty:
            return dssm_df

    return pd.DataFrame()
