"""在线召回服务编排层。"""

from __future__ import annotations

import pickle
import re
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
        "route_min_quota": 32,
        "route_max_share": 0.82,
        "merge_order": ["swing", "usercf", "ann"],
        "route_weights": {"ann": 0.18, "swing": 2.46, "usercf": 2.24},
        "rrf_k": 34,
        "min_candidates": {"swing": 32, "usercf": 28},
    },
    "search": {
        "route_min_quota": 0,
        "route_max_share": 0.99,
        "merge_order": ["usercf", "swing", "ann"],
        "route_weights": {"ann": 0.42, "swing": 1.68, "usercf": 2.28},
        "rrf_k": 34,
        "min_candidates": {"swing": 16, "usercf": 20},
    },
}
_SCORED_RESULT_RE = re.compile(r"array\(\[\s*(-?\d+)\s*,\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?)\s*\]\)")


def _to_list(x) -> list[int]:
    if isinstance(x, np.ndarray):
        return [int(v) for v in x.tolist()]
    if isinstance(x, list):
        return [int(v) for v in x]
    return []


def _resolve_hist_items(
    raw_hist_items,
    user_idx: int,
    user_items: dict[int, dict[int, float]],
    limit: int = 20,
) -> list[int]:
    hist_items = _to_list(raw_hist_items)
    if hist_items:
        return hist_items[: max(1, int(limit))]
    user_hist = user_items.get(int(user_idx)) or {}
    if not user_hist:
        return []
    ranked = sorted(
        ((int(item), float(pref)) for item, pref in user_hist.items()),
        key=lambda x: x[1],
        reverse=True,
    )
    return [int(item) for item, _ in ranked[: max(1, int(limit))]]


def _resolve_existing(paths: list[Path]) -> Path:
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError(f"None of these files exists: {[str(p) for p in paths]}")


@lru_cache(maxsize=8192)
def _load_precomputed_recall_slice(path_str: str, group_key: str, request_id: int, max_rank: int) -> tuple[tuple[str, object], ...]:
    path = Path(path_str)
    if not path.exists():
        return ()
    cols = [
        str(group_key),
        "user_idx",
        "note_idx",
        "rank",
        "recall_score",
        "score_ann",
        "score_swing",
        "score_usercf",
        "from_ann",
        "from_swing",
        "from_usercf",
        "first_route",
    ]
    try:
        df = pd.read_parquet(
            path,
            columns=cols,
            filters=[(str(group_key), "==", int(request_id))],
        )
    except Exception:
        df = pd.read_parquet(path, columns=cols)
        df = df[df[str(group_key)] == int(request_id)].copy()
    if len(df) <= 0:
        return ()
    df = df.sort_values("rank", ascending=True, kind="mergesort").head(int(max_rank)).copy()
    return tuple(tuple(row.items()) for row in df.to_dict("records"))


def _fetch_precomputed_recall(
    recall_path: Path | None,
    group_key: str,
    request_id: int,
    max_rank: int,
) -> pd.DataFrame:
    if recall_path is None:
        return pd.DataFrame()
    rows = _load_precomputed_recall_slice(
        str(recall_path),
        str(group_key),
        int(request_id),
        int(max_rank),
    )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([dict(items) for items in rows])


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
    seqtrans_path = INDEX_DIR / f"seqtrans_{scene}_train_i2i_topk.parquet"

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

    seqtrans_i2i: dict[int, np.ndarray] = {}
    if seqtrans_path.exists():
        seqtrans_df = pd.read_parquet(seqtrans_path)
        seqtrans_map: dict[int, list[list[float]]] = defaultdict(list)
        for i, j, s in seqtrans_df[["item_idx", "sim_item_idx", "score"]].itertuples(index=False):
            seqtrans_map[int(i)].append([int(j), float(s)])
        seqtrans_i2i = {k: np.asarray(v, dtype=np.float32) for k, v in seqtrans_map.items()}

    with open(user_items_path, "rb") as f:
        user_items_obj = pickle.load(f)
    user_items = user_items_obj.get("user_items", {})
    return swing_i2i, usercf_u2u, user_items, seqtrans_i2i


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


def _score_dict_from_df(
    df: pd.DataFrame,
    score_col: str = "recall_score",
    topk: int | None = None,
) -> dict[int, float]:
    if df is None or len(df) <= 0 or "note_idx" not in df.columns:
        return {}
    cur = df.copy()
    if score_col not in cur.columns:
        score_col = "rank"
    if topk is not None and int(topk) > 0:
        rank_col = "rank" if "rank" in cur.columns else score_col
        cur = cur.sort_values(rank_col, ascending=True, kind="mergesort").head(int(topk))
    out: dict[int, float] = {}
    for row in cur[["note_idx", score_col]].itertuples(index=False):
        try:
            note_idx = int(getattr(row, "note_idx"))
        except Exception:
            continue
        if note_idx < 0:
            continue
        try:
            score = float(getattr(row, score_col))
        except Exception:
            score = 0.0
        out[note_idx] = score
    return out


def _blend_route_scores(
    primary_scores: dict[int, float],
    secondary_scores: dict[int, float],
    primary_weight: float = 1.0,
    secondary_weight: float = 1.0,
    overlap_bonus: float = 0.0,
    topk: int | None = None,
) -> dict[int, float]:
    union = set(primary_scores) | set(secondary_scores)
    if not union:
        return {}
    out: dict[int, float] = {}
    for note_idx in union:
        score = (
            float(primary_scores.get(int(note_idx), 0.0)) * float(primary_weight)
            + float(secondary_scores.get(int(note_idx), 0.0)) * float(secondary_weight)
        )
        if int(note_idx) in primary_scores and int(note_idx) in secondary_scores:
            score += float(overlap_bonus)
        if score > 0.0:
            out[int(note_idx)] = float(score)
    ranked = sorted(out.items(), key=lambda x: x[1], reverse=True)
    if topk is not None and int(topk) > 0:
        ranked = ranked[: int(topk)]
    return {int(k): float(v) for k, v in ranked}


@lru_cache(maxsize=8192)
def _parse_scored_route_blob(raw: str, limit: int = 200) -> tuple[tuple[int, float], ...]:
    text = str(raw or "").strip()
    if not text:
        return ()
    pairs: list[tuple[int, float]] = []
    for note_id, score in _SCORED_RESULT_RE.findall(text):
        nid = int(note_id)
        if nid < 0:
            continue
        pairs.append((nid, float(score)))
        if len(pairs) >= int(limit):
            break
    return tuple(pairs)


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
            support_cnt = (
                int(item in route2scores["ann"])
                + int(item in route2scores["swing"])
                + int(item in route2scores["usercf"])
            )
            overlap_bonus = 0.0
            if support_cnt >= 2:
                overlap_bonus += 0.10 * float(support_cnt - 1)
            if item in route2scores["usercf"] and item in route2scores["swing"]:
                overlap_bonus += 0.12
            if item in route2scores["ann"] and item in route2scores["usercf"]:
                overlap_bonus += 0.08
            if item in route2scores["ann"] and item in route2scores["swing"]:
                overlap_bonus += 0.05
            fused_score = float(rrf_score + 0.15 * score_part + overlap_bonus)

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

    quota_order = list(route_order)
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
    precomputed = _fetch_precomputed_recall(
        recall_path=recall_test_path,
        group_key=str(group_key),
        request_id=int(request_id),
        max_rank=int(max_rank),
    )
    ann_df = pd.DataFrame()
    seq_df = pd.DataFrame()
    if dssm_recaller is not None:
        ann_df = dssm_recaller.fetch_candidates(
            request_id=int(request_id),
            max_rank=int(max_rank),
            req_df=req_df,
            get_feat_req=get_feat_req,
            group_key=group_key,
        )
        if scene == "rec":
            seq_df = dssm_recaller.fetch_history_candidates(
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

    ann_scores = _score_dict_from_df(precomputed, score_col="score_ann", topk=int(max_rank)) if len(precomputed) > 0 else {}
    swing_scores = _score_dict_from_df(precomputed, score_col="score_swing", topk=int(max_rank)) if len(precomputed) > 0 else {}
    usercf_scores = _score_dict_from_df(precomputed, score_col="score_usercf", topk=int(max_rank)) if len(precomputed) > 0 else {}
    if len(ann_df) > 0:
        src = "score_ann" if "score_ann" in ann_df.columns else "recall_score"
        ann_scores.update({
            int(r.note_idx): float(getattr(r, src))
            for r in ann_df[["note_idx", src]].itertuples(index=False)
            if int(r.note_idx) >= 0
        })
    if len(seq_df) > 0:
        src = "score_swing" if "score_swing" in seq_df.columns else "recall_score"
        swing_scores.update({
            int(r.note_idx): float(getattr(r, src))
            for r in seq_df[["note_idx", src]].itertuples(index=False)
            if int(r.note_idx) >= 0
        })
    bm25_scores: dict[int, float] = {}
    dpr_scores: dict[int, float] = {}
    if scene == "search" and len(req_cur) > 0:
        row = req_cur.iloc[0]
        if "bm25_results" in req_cur.columns:
            bm25_raw = row.get("bm25_results", "")
            bm25_scores = {int(nid): float(score) for nid, score in _parse_scored_route_blob("" if bm25_raw is None else str(bm25_raw), limit=int(max_rank))}
        if "dpr_results" in req_cur.columns:
            dpr_raw = row.get("dpr_results", "")
            dpr_scores = {int(nid): float(score) for nid, score in _parse_scored_route_blob("" if dpr_raw is None else str(dpr_raw), limit=int(max_rank))}

    try:
        swing_i2i, usercf_u2u, user_items, seqtrans_i2i = _load_route_assets(scene)
        hist_items: list[int] = []
        if len(feat_req) > 0 and "recent_clicked_note_idxs" in feat_req.columns:
            hist_items = _resolve_hist_items(
                raw_hist_items=feat_req.iloc[0].get("recent_clicked_note_idxs", []),
                user_idx=int(user_idx),
                user_items=user_items,
                limit=20,
            )
        if scene == "search":
            if bm25_scores:
                usercf_scores.update(bm25_scores)
            if dpr_scores:
                swing_scores.update(dpr_scores)
        else:
            live_swing_scores: dict[int, float] = {}
            if hist_items:
                seq_transition_scores = _recall_swing(hist_items=hist_items, swing_i2i=seqtrans_i2i, topk=int(max_rank)) if seqtrans_i2i else {}
                graph_swing_scores = _recall_swing(hist_items=hist_items, swing_i2i=swing_i2i, topk=int(max_rank))
                live_swing_scores = graph_swing_scores
                seq_transition_ann_scores = {}
                if seq_transition_scores and dssm_recaller is not None:
                    seed_ids, seed_vals = dssm_recaller.expand_seed_candidates(
                        seed_scores=seq_transition_scores,
                        max_rank=int(max_rank),
                        max_seed=80,
                    )
                    if seed_ids and seed_vals:
                        seq_transition_ann_scores = {int(n): float(s) for n, s in zip(seed_ids, seed_vals)}
                if seq_transition_scores:
                    live_swing_scores = dict(seq_transition_scores)
                if seq_transition_scores:
                    live_swing_scores = _blend_route_scores(
                        live_swing_scores,
                        seq_transition_ann_scores,
                        primary_weight=1.0,
                        secondary_weight=0.82,
                        overlap_bonus=0.16,
                        topk=int(max_rank),
                    )
                    live_swing_scores = _blend_route_scores(
                        live_swing_scores,
                        graph_swing_scores,
                        primary_weight=1.0,
                        secondary_weight=0.88,
                        overlap_bonus=0.14,
                        topk=int(max_rank),
                    )
            if int(user_idx) >= 0:
                live_usercf_scores = _recall_usercf(
                    user_idx=int(user_idx),
                    usercf_u2u=usercf_u2u,
                    user_items=user_items,
                    topk=int(max_rank),
                )
                usercf_scores = _blend_route_scores(
                    usercf_scores,
                    live_usercf_scores,
                    primary_weight=0.72,
                    secondary_weight=1.28,
                    overlap_bonus=0.12,
                    topk=int(max_rank),
                )
                if live_swing_scores:
                    usercf_scores = _blend_route_scores(
                        usercf_scores,
                        live_swing_scores,
                        primary_weight=0.86,
                        secondary_weight=0.94,
                        overlap_bonus=0.08,
                        topk=int(max_rank),
                    )
    except Exception:
        pass

    if scene == "rec" and len(req_cur) > 0 and dssm_recaller is not None:
        query_text = str(req_cur.iloc[0].get("query") or "").strip()
        if query_text:
            semantic_pool: set[int] = set(ann_scores)
            semantic_pool.update(list(swing_scores.keys())[: max(120, int(max_rank))])
            semantic_pool.update(list(usercf_scores.keys())[: max(120, int(max_rank))])
            if precomputed is not None and len(precomputed) > 0:
                semantic_pool.update(precomputed["note_idx"].astype(int).tolist()[: max(160, int(max_rank) * 2)])
            sem_ids, sem_scores = dssm_recaller.score_note_text_candidates(
                query_text=query_text[:256],
                note_ids=sorted(int(x) for x in semantic_pool if int(x) >= 0),
                topk=max(120, int(max_rank)),
            )
            if sem_ids and sem_scores:
                semantic_scores = _normalize_route_scores(
                    {int(nid): float(score) for nid, score in zip(sem_ids, sem_scores)}
                )
                merged_semantic: dict[int, float] = {}
                for note_idx in set(ann_scores) | set(semantic_scores):
                    merged_semantic[int(note_idx)] = (
                        float(ann_scores.get(int(note_idx), 0.0)) * 0.68
                        + float(semantic_scores.get(int(note_idx), 0.0)) * 1.18
                    )
                ann_scores = merged_semantic

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
        if not precomputed.empty:
            return precomputed.reset_index(drop=True)
        return pd.DataFrame()

    rows = []
    for rk, (item, s, s_ann, s_sw, s_ucf, f_ann, f_sw, f_ucf, first_route) in enumerate(merged, start=1):
        if scene == "search":
            if first_route == "usercf":
                first_route = "bm25"
            elif first_route == "swing":
                first_route = "dpr"
        elif scene == "rec" and first_route == "swing":
            first_route = "seq"
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
