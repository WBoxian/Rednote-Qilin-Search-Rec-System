"""在线召回服务编排层。"""

from __future__ import annotations

import pickle
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from backend.online.cold_start.popular import build_hot_candidates
from backend.online.recall.dssm import DSSMRecaller

BASE_DIR = Path(__file__).resolve().parents[4]
INDEX_DIR = BASE_DIR / "outputs" / "index"
ROUTE_STRATEGY: dict[str, dict] = {
    "rec": {
        "route_min_quota": 120,
        "route_max_share": 0.55,
        "merge_order": ["usercf", "swing", "ann"],
        "route_weights": {"ann": 0.9, "swing": 1.1, "usercf": 1.2},
        "rrf_k": 60,
        "min_candidates": {"swing": 20, "usercf": 20},
    },
    "search": {
        "route_min_quota": 100,
        "route_max_share": 0.60,
        "merge_order": ["ann", "usercf", "swing"],
        "route_weights": {"ann": 1.0, "swing": 0.8, "usercf": 1.1},
        "rrf_k": 80,
        "min_candidates": {"swing": 10, "usercf": 15},
    },
}


def _to_list(x) -> list[int]:
    if isinstance(x, np.ndarray):
        return [int(v) for v in x.tolist()]
    if isinstance(x, list):
        return [int(v) for v in x]
    return []


def _resolve_existing(paths: list[Path]) -> Path:
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError(f"None of these files exists: {[str(p) for p in paths]}")


@lru_cache(maxsize=4)
def _load_route_assets(scene: str):
    swing_path = _resolve_existing(
        [
            INDEX_DIR / f"swing_{scene}_train_i2i_topk.parquet",
            INDEX_DIR / f"swing_{scene}_i2i_topk.parquet",
        ]
    )
    usercf_path = _resolve_existing(
        [
            INDEX_DIR / f"usercf_{scene}_train_u2u_topk.parquet",
            INDEX_DIR / f"usercf_{scene}_u2u_topk.parquet",
        ]
    )
    user_items_path = _resolve_existing([INDEX_DIR / f"cf_{scene}_train_user_item_index.pkl"])

    swing_df = pd.read_parquet(swing_path)
    swing_i2i: dict[int, list[list[float]]] = defaultdict(list)
    for i, j, s in swing_df[["item_idx", "sim_item_idx", "score"]].itertuples(index=False):
        swing_i2i[int(i)].append([int(j), float(s)])
    swing_i2i = {k: np.asarray(v, dtype=np.float32) for k, v in swing_i2i.items()}

    usercf_df = pd.read_parquet(usercf_path)
    usercf_u2u: dict[int, list[list[float]]] = defaultdict(list)
    for u, v, s in usercf_df[["user_idx", "sim_user_idx", "score"]].itertuples(index=False):
        usercf_u2u[int(u)].append([int(v), float(s)])
    usercf_u2u = {k: np.asarray(v, dtype=np.float32) for k, v in usercf_u2u.items()}

    with open(user_items_path, "rb") as f:
        user_items_obj = pickle.load(f)
    user_items = user_items_obj.get("user_items", {})
    return swing_i2i, usercf_u2u, user_items


def _normalize_route_scores(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    keys = list(scores.keys())
    vals = np.asarray([float(scores[k]) for k in keys], dtype=np.float32)
    s_min = float(np.min(vals))
    s_max = float(np.max(vals))
    if s_max - s_min <= 1e-12:
        norm_vals = np.ones_like(vals, dtype=np.float32)
    else:
        norm_vals = (vals - s_min) / (s_max - s_min)
    out = {int(k): float(v) for k, v in zip(keys, norm_vals.tolist())}
    return dict(sorted(out.items(), key=lambda x: x[1], reverse=True))


def _resolve_route_strategy(scene: str, topk: int) -> dict:
    cfg = ROUTE_STRATEGY.get(scene, {})
    return {
        "route_min_quota": int(max(0, cfg.get("route_min_quota", 100))),
        "route_max_share": float(min(1.0, max(0.5, cfg.get("route_max_share", 0.8)))),
        "merge_order": [str(x) for x in cfg.get("merge_order", ["ann", "swing", "usercf"])],
        "route_weights": {
            str(k): float(v)
            for k, v in cfg.get("route_weights", {"ann": 1.0, "swing": 1.0, "usercf": 1.0}).items()
        },
        "rrf_k": int(max(10, cfg.get("rrf_k", 60))),
        "min_candidates": {
            str(k): int(max(0, v))
            for k, v in cfg.get("min_candidates", {}).items()
        },
        "topk": int(max(1, topk)),
    }


def _auto_learn_route_weights(
    ann_scores: dict[int, float],
    swing_scores: dict[int, float],
    usercf_scores: dict[int, float],
    base_weights: dict[str, float],
) -> dict[str, float]:
    route_scores = {
        "ann": ann_scores,
        "swing": swing_scores,
        "usercf": usercf_scores,
    }

    learned_raw: dict[str, float] = {}
    for route, scores in route_scores.items():
        prior = float(base_weights.get(route, 1.0))
        if not scores:
            learned_raw[route] = 0.0
            continue
        vals = np.asarray(list(scores.values()), dtype=np.float32)
        if vals.size == 0:
            learned_raw[route] = 0.0
            continue
        top1 = float(vals[0])
        top_mean = float(np.mean(vals[: min(50, vals.size)]))
        p90 = float(np.percentile(vals, 90)) if vals.size >= 5 else top1
        coverage = float(min(1.0, vals.size / 200.0))
        sharpness = float(max(0.0, top1 - top_mean))
        quality = 0.40 * top1 + 0.25 * p90 + 0.20 * top_mean + 0.10 * coverage + 0.05 * sharpness
        learned_raw[route] = float(prior * max(0.05, 0.5 + quality))

    total = float(sum(learned_raw.values()))
    if total <= 1e-12:
        return {"ann": 1.0, "swing": 1.0, "usercf": 1.0}

    route_cnt = len(learned_raw)
    return {
        r: float((v / total) * route_cnt)
        for r, v in learned_raw.items()
    }


def _recall_swing(hist_items: list[int], swing_i2i: dict[int, np.ndarray], topk: int) -> dict[int, float]:
    if not hist_items:
        return {}
    scores: dict[int, float] = {}
    for idx, item in enumerate(reversed(hist_items)):
        sim_arr = swing_i2i.get(int(item))
        if sim_arr is None or len(sim_arr) == 0:
            continue
        decay = 1.0 / (1.0 + idx)
        for row in sim_arr:
            item_id = int(row[0])
            scores[item_id] = scores.get(item_id, 0.0) + decay * float(row[1])
    if not scores:
        return {}
    import heapq
    return dict(heapq.nlargest(int(topk), scores.items(), key=lambda x: x[1]))


def _recall_usercf(
    user_idx: int,
    usercf_u2u: dict[int, np.ndarray],
    user_items: dict[int, dict[int, float]],
    topk: int,
) -> dict[int, float]:
    sim_users = usercf_u2u.get(int(user_idx))
    if sim_users is None or len(sim_users) == 0:
        return {}
    scores: dict[int, float] = {}
    for row in sim_users:
        sim_uid = int(row[0])
        sim_score = float(row[1])
        items = user_items.get(sim_uid)
        if not items:
            continue
        for item, pref in items.items():
            iid = int(item)
            scores[iid] = scores.get(iid, 0.0) + sim_score * float(pref)
    if not scores:
        return {}
    import heapq
    return dict(heapq.nlargest(int(topk), scores.items(), key=lambda x: x[1]))


def _merge_scores(
    ann_scores: dict[int, float],
    swing_scores: dict[int, float],
    usercf_scores: dict[int, float],
    topk: int,
    route_min_quota: int,
    route_max_share: float,
    merge_order: list[str],
    route_weights: dict[str, float],
    rrf_k: int,
):
    route2scores = {
        "ann": ann_scores,
        "swing": swing_scores,
        "usercf": usercf_scores,
    }
    route_order = [r for r in merge_order if r in route2scores]
    if not route_order:
        route_order = ["ann", "swing", "usercf"]
    route2list = {r: list(route2scores[r].items()) for r in route_order}
    route_cursor = {r: 0 for r in route_order}
    route_take_cnt = {r: 0 for r in route_order}

    active_routes = [r for r in route_order if len(route2list[r]) > 0]
    active_cnt = max(len(active_routes), 1)

    min_quota_floor = int(np.ceil(0.1 * float(topk)))
    base_quota = max(max(0, int(route_min_quota)), min_quota_floor)
    quota_cap = max(0, int(int(topk) // max(active_cnt, 1)))
    min_quota = min(base_quota, quota_cap)
    if base_quota > 0 and min_quota <= 0 and active_cnt > 1:
        min_quota = 1

    route_cap = max(1, int(int(topk) * max(0.0, min(1.0, float(route_max_share)))))
    if active_cnt <= 1:
        route_cap = int(topk)

    rank_maps: dict[str, dict[int, int]] = {}
    for route in route_order:
        rank_maps[route] = {
            int(item): int(idx)
            for idx, (item, _) in enumerate(route2list[route], start=1)
        }

    merged: list[tuple[int, float, float, float, float, int, int, int, str]] = []
    seen: set[int] = set()

    def _route_has_unseen(route: str) -> bool:
        items = route2list[route]
        p = route_cursor[route]
        while p < len(items) and int(items[p][0]) in seen:
            p += 1
        return p < len(items)

    def _append_from_route(route: str, enforce_cap: bool) -> bool:
        if enforce_cap and route_take_cnt[route] >= route_cap:
            if any(_route_has_unseen(r) for r in route_order if r != route):
                return False
        items = route2list[route]
        p = route_cursor[route]
        while p < len(items):
            item, route_score = items[p]
            p += 1
            item = int(item)
            if item in seen:
                continue
            seen.add(item)
            route_cursor[route] = p
            route_take_cnt[route] += 1
            s_ann = route2scores["ann"].get(item, 0.0)
            s_sw = route2scores["swing"].get(item, 0.0)
            s_ucf = route2scores["usercf"].get(item, 0.0)

            w_ann = float(route_weights.get("ann", 1.0))
            w_sw = float(route_weights.get("swing", 1.0))
            w_ucf = float(route_weights.get("usercf", 1.0))
            rrk = float(max(1, int(rrf_k)))
            r_ann = rank_maps.get("ann", {}).get(item)
            r_sw = rank_maps.get("swing", {}).get(item)
            r_ucf = rank_maps.get("usercf", {}).get(item)

            rrf_score = 0.0
            if r_ann is not None:
                rrf_score += w_ann / (rrk + float(r_ann))
            if r_sw is not None:
                rrf_score += w_sw / (rrk + float(r_sw))
            if r_ucf is not None:
                rrf_score += w_ucf / (rrk + float(r_ucf))

            score_part = (w_ann * float(s_ann)) + (w_sw * float(s_sw)) + (w_ucf * float(s_ucf))
            fused_score = float(rrf_score + 0.15 * score_part)

            merged.append(
                (
                    item,
                    fused_score,
                    float(s_ann),
                    float(s_sw),
                    float(s_ucf),
                    int(item in route2scores["ann"]),
                    int(item in route2scores["swing"]),
                    int(item in route2scores["usercf"]),
                    route,
                )
            )
            return True
        route_cursor[route] = p
        return False

    quota_order = [r for r in route_order if r != "ann"] + ["ann"]
    if min_quota > 0:
        for route in quota_order:
            while len(merged) < int(topk) and route_take_cnt[route] < min_quota:
                if not _append_from_route(route, enforce_cap=False):
                    break

    while len(merged) < int(topk):
        progressed = False
        for route in route_order:
            if len(merged) >= int(topk):
                break
            if _append_from_route(route, enforce_cap=True):
                progressed = True
        if not progressed:
            break
    return merged[: int(topk)]


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
    if not recall.empty:
        out = recall.copy()
        if "rank" in out.columns:
            out = out.sort_values("rank", ascending=True, kind="mergesort")
        out = out.drop_duplicates(subset=["note_idx"], keep="first").head(int(recall_rank_cap)).reset_index(drop=True)
        if "rank" not in out.columns:
            out["rank"] = np.arange(1, len(out) + 1, dtype=np.int64)
        return out

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
    scene = "search" if str(group_key) == "search_idx" else "rec"

    ann_df = pd.DataFrame()
    if dssm_recaller is not None:
        ann_df = dssm_recaller.fetch_candidates(
            request_id=int(request_id),
            max_rank=int(max_rank),
            req_df=req_df,
            get_feat_req=get_feat_req,
            group_key=group_key,
        )

    feat_req = get_feat_req(int(request_id))
    req_cur = req_df[req_df[group_key] == int(request_id)]
    if len(req_cur) > 0:
        user_idx = int(req_cur.iloc[0].get("user_idx", -1))
    elif len(feat_req) > 0 and "user_idx" in feat_req.columns:
        user_idx = int(pd.to_numeric(feat_req.iloc[0].get("user_idx", -1), errors="coerce") or -1)
    else:
        user_idx = -1

    hist_items: list[int] = []
    if len(feat_req) > 0 and "recent_clicked_note_idxs" in feat_req.columns:
        hist_items = _to_list(feat_req.iloc[0].get("recent_clicked_note_idxs", []))

    ann_scores = {}
    if len(ann_df) > 0:
        src = "score_ann" if "score_ann" in ann_df.columns else "recall_score"
        ann_scores = {
            int(r.note_idx): float(getattr(r, src))
            for r in ann_df[["note_idx", src]].itertuples(index=False)
            if int(r.note_idx) >= 0
        }

    swing_scores = {}
    usercf_scores = {}
    try:
        swing_i2i, usercf_u2u, user_items = _load_route_assets(scene)
        if hist_items:
            swing_scores = _recall_swing(hist_items=hist_items, swing_i2i=swing_i2i, topk=int(max_rank))
        if int(user_idx) >= 0:
            usercf_scores = _recall_usercf(
                user_idx=int(user_idx),
                usercf_u2u=usercf_u2u,
                user_items=user_items,
                topk=int(max_rank),
            )
    except Exception:
        swing_scores = {}
        usercf_scores = {}

    ann_scores = _normalize_route_scores(ann_scores)
    swing_scores = _normalize_route_scores(swing_scores)
    usercf_scores = _normalize_route_scores(usercf_scores)

    policy = _resolve_route_strategy(scene=scene, topk=int(max_rank))
    min_candidates = policy.get("min_candidates", {})
    if len(swing_scores) < int(min_candidates.get("swing", 0)):
        swing_scores = {}
    if len(usercf_scores) < int(min_candidates.get("usercf", 0)):
        usercf_scores = {}

    learned_weights = _auto_learn_route_weights(
        ann_scores=ann_scores,
        swing_scores=swing_scores,
        usercf_scores=usercf_scores,
        base_weights=dict(policy["route_weights"]),
    )

    merged = _merge_scores(
        ann_scores=ann_scores,
        swing_scores=swing_scores,
        usercf_scores=usercf_scores,
        topk=int(max_rank),
        route_min_quota=int(policy["route_min_quota"]),
        route_max_share=float(policy["route_max_share"]),
        merge_order=list(policy["merge_order"]),
        route_weights=learned_weights,
        rrf_k=int(policy["rrf_k"]),
    )
    if not merged:
        return pd.DataFrame()

    rows = []
    for rk, (item, s, s_ann, s_sw, s_ucf, f_ann, f_sw, f_ucf, first_route) in enumerate(merged, start=1):
        rows.append(
            {
                group_key: int(request_id),
                "user_idx": int(user_idx),
                "note_idx": int(item),
                "rank": int(rk),
                "recall_score": float(s),
                "score_ann": float(s_ann),
                "score_swing": float(s_sw),
                "score_usercf": float(s_ucf),
                "from_ann": int(f_ann),
                "from_swing": int(f_sw),
                "from_usercf": int(f_ucf),
                "first_route": str(first_route),
            }
        )
    return pd.DataFrame(rows)
