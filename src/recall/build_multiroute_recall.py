"""
澶氳矾鍙洖鐢熸垚鑴氭湰锛圓NN + Swing + UserCF锛夛紝杈撳嚭鍙敤浜庣矖鎺掞紙LambdaMART锛夌殑鍊欓€夎〃銆?
杈撳叆:
- 鐢ㄦ埛璇锋眰鏉ヨ嚜 features/{scene}_{split}_features.parquet
- 绱㈠紩鏉ヨ嚜 outputs/index:
    - dssm_{scene}_{tag}_ivfpq.faiss
    - dssm_{scene}_{tag}_row2note.npy
    - dssm_{scene}_{tag}_item_emb.bin / item_meta.json
    - (search鍙€? dssm_search_{tag}_{split}_query_emb.bin / query_map.json
    - swing_{scene}*_i2i_topk.parquet
    - usercf_{scene}*_u2u_topk.parquet
    - cf_{scene}_train_user_item_index.pkl

铻嶅悎绛栫暐:
- 姣忎竴璺厛鍚勮嚜鍙洖 topk锛圓NN / Swing / UserCF锛?- 鍚勮矾鍒嗘暟鍏堝湪璇锋眰鍐呭仛褰掍竴鍖栵紙Min-Max 鍒?[0,1]锛夛紝鍐嶈繘鍏ヨ瀺鍚堬紝淇濊瘉閲忕翰鍙瘮
- 閫氳繃姣忚矾鏈€灏忛厤棰?+ 鍗曡矾鍗犳瘮涓婇檺锛屾姂鍒跺崟璺富瀵硷紝鏈€鍚庢寜鎬?topk 鎴柇

杈撳嚭:
- outputs/data/recall_{scene}_{split}_{tag}_multiroute_top{topk}.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:
    pa = None
    pq = None

try:
    import faiss
except Exception as e:
    raise SystemExit(f"faiss import failed: {e}") from e


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("build_multiroute_recall")
DISABLE_TQDM = os.getenv("QILIN_DISABLE_TQDM", "0") == "1"

BASE_DIR = Path(__file__).resolve().parents[2]  # Qilin/
FEAT_DIR = BASE_DIR / "features"
INDEX_DIR = BASE_DIR / "outputs" / "index"
OUT_DATA_DIR = BASE_DIR / "outputs" / "data"
OUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from backend.online.search_query import detect_query_intents, normalize_query_text, seed_query_terms

ROW_FLUSH_SIZE = 200_000
SEARCH_ROUTE_FUSE_WEIGHTS = {
    "ann": 0.14,
    "dpr": 0.34,
    "bm25": 0.52,
}
SEARCH_SHORT_QUERY_WEIGHTS = {
    "ann": 0.12,
    "dpr": 0.24,
    "bm25": 0.64,
}
SEARCH_LONG_QUERY_WEIGHTS = {
    "ann": 0.16,
    "dpr": 0.52,
    "bm25": 0.32,
}
SEARCH_RECOMMEND_QUERY_WEIGHTS = {
    "ann": 0.20,
    "dpr": 0.38,
    "bm25": 0.42,
}
SEARCH_MERGE_ROUTE_WEIGHTS = {"ann": 0.42, "swing": 1.68, "usercf": 2.28}
REC_MERGE_ROUTE_WEIGHTS = {"ann": 0.18, "swing": 2.46, "usercf": 2.24}


def _normalize_weight_map(weights: dict[str, float]) -> dict[str, float]:
    total = float(sum(max(0.0, float(v)) for v in weights.values()))
    if total <= 1e-12:
        return {"ann": 1.0, "dpr": 1.0, "bm25": 1.0}
    return {str(k): float(max(0.0, float(v)) / total) for k, v in weights.items()}


def _resolve_search_route_weights(query: str) -> dict[str, float]:
    q = normalize_query_text(query)
    compact = q.replace(" ", "")
    terms = [t for t in seed_query_terms(q) if str(t).strip()]
    intents = set(detect_query_intents(q))
    if len(compact) <= 6 or len(terms) <= 2:
        return _normalize_weight_map(dict(SEARCH_SHORT_QUERY_WEIGHTS))
    if len(compact) >= 12 or intents.intersection({"tutorial", "compare", "review"}):
        return _normalize_weight_map(dict(SEARCH_LONG_QUERY_WEIGHTS))
    if intents.intersection({"recommend", "fashion"}):
        return _normalize_weight_map(dict(SEARCH_RECOMMEND_QUERY_WEIGHTS))
    return _normalize_weight_map(dict(SEARCH_ROUTE_FUSE_WEIGHTS))


def _resolve_merge_route_weights(scene: str, hist_len: int, has_request_vec: bool) -> dict[str, float]:
    if scene == "search":
        return dict(SEARCH_MERGE_ROUTE_WEIGHTS)
    weights = dict(REC_MERGE_ROUTE_WEIGHTS)
    if hist_len <= 2 and has_request_vec:
        weights["ann"] = 0.48
        weights["usercf"] = 1.82
        weights["swing"] = 2.08
    elif hist_len >= 8:
        weights["ann"] = 0.08
        weights["usercf"] = 2.46
        weights["swing"] = 2.82
    elif hist_len >= 4:
        weights["ann"] = 0.12
        weights["usercf"] = 2.34
        weights["swing"] = 2.68
    return weights


def _to_list(x: Any) -> list[int]:
    if isinstance(x, np.ndarray):
        return [int(v) for v in x.tolist()]
    if isinstance(x, list):
        return [int(v) for v in x]
    return []


def _resolve_hist_items(
    raw_hist_items: list[int] | np.ndarray | None,
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


def _index_candidates(filename: str) -> list[Path]:
    return [INDEX_DIR / filename]


def _get_parquet_columns(path: Path) -> list[str]:
    """Read only the parquet schema to avoid loading full tables into memory."""
    if pq is not None:
        return list(pq.ParquetFile(path).schema.names)
    # Fallback: if pyarrow is unavailable, read a light dataframe header once.
    return pd.read_parquet(path, columns=None).columns.tolist()


def _safe_intish(x: Any, default: int = -1) -> int:
    try:
        if x is None:
            return int(default)
        if isinstance(x, (np.integer, int)):
            return int(x)
        if isinstance(x, float):
            if np.isnan(x):
                return int(default)
            return int(x)
        s = str(x).strip()
        if not s:
            return int(default)
        return int(float(s))
    except Exception:
        return int(default)


def _request_dataset_path(scene: str, split: str) -> Path | None:
    folder = "search" if scene == "search" else "recommendation"
    path = BASE_DIR / "datasets" / f"{folder}_{split}" / "train-00000-of-00001.parquet"
    return path if path.exists() else None


def _parse_scored_results(raw: Any, limit: int = 200) -> dict[int, float]:
    if raw is None:
        return {}
    pairs: list[tuple[int, float]] = []
    if isinstance(raw, np.ndarray):
        iterable = raw.tolist()
    elif isinstance(raw, list):
        iterable = raw
    else:
        iterable = []
    for obj in iterable[: max(1, int(limit))]:
        if isinstance(obj, np.ndarray):
            obj = obj.tolist()
        if isinstance(obj, (list, tuple)) and len(obj) >= 2:
            note_idx = _safe_intish(obj[0], -1)
            if note_idx < 0:
                continue
            try:
                score = float(obj[1])
            except Exception:
                score = 0.0
            if np.isfinite(score):
                pairs.append((int(note_idx), float(score)))
        else:
            note_idx = _safe_intish(obj, -1)
            if note_idx >= 0:
                pairs.append((int(note_idx), 1.0))
    out: dict[int, float] = {}
    for note_idx, score in pairs:
        if note_idx not in out:
            out[note_idx] = float(score)
    return out


def _load_requests(scene: str, split: str) -> tuple[pd.DataFrame, str]:
    path = FEAT_DIR / f"{scene}_{split}_features.parquet"
    cols = ["user_idx", "recent_clicked_note_idxs", "session_idx"]
    candidate_req_cols = ["request_idx", "search_idx"]
    parquet_cols = _get_parquet_columns(path)
    req_col = next((c for c in candidate_req_cols if c in parquet_cols), "session_idx")
    use_cols = cols + ([req_col] if req_col not in cols else [])
    df = pd.read_parquet(path, columns=use_cols)
    df["recent_clicked_note_idxs"] = df["recent_clicked_note_idxs"].apply(_to_list)
    # Keep a single row per request.
    req_df = df.drop_duplicates(subset=[req_col]).reset_index(drop=True)
    req_data_path = _request_dataset_path(scene, split)
    if req_data_path is not None:
        req_cols = [req_col, "query"]
        if scene == "search":
            req_cols += ["bm25_results", "dpr_results", "search_results"]
        else:
            req_cols += ["rec_results"]
        try:
            req_meta = pd.read_parquet(req_data_path, columns=req_cols).drop_duplicates(subset=[req_col])
        except Exception:
            available = [c for c in req_cols if c in _get_parquet_columns(req_data_path)]
            req_meta = pd.read_parquet(req_data_path, columns=available).drop_duplicates(subset=[req_col]) if available else None
        if req_meta is not None and len(req_meta) > 0:
            req_df = req_df.merge(req_meta, how="left", on=req_col)
    return req_df, req_col


def _load_swing_index(scene: str) -> dict[int, np.ndarray]:
    """Load swing item-to-item index into compact numpy arrays."""
    p = _resolve_existing(
        [
            *_index_candidates(f"swing_{scene}_train_i2i_topk.parquet"),
            *_index_candidates(f"swing_{scene}_i2i_topk.parquet"),
        ]
    )
    df = pd.read_parquet(p)
    out: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for i, j, s in df[["item_idx", "sim_item_idx", "score"]].itertuples(index=False):
        out[int(i)].append([int(j), float(s)])
    # Convert to compact numpy arrays to reduce memory and lookup cost.
    return {k: np.array(v, dtype=np.float32) for k, v in out.items()}




def _load_seq_transition_index(scene: str) -> dict[int, np.ndarray]:
    p = _resolve_existing([INDEX_DIR / f"seqtrans_{scene}_train_i2i_topk.parquet"])
    df = pd.read_parquet(p)
    out: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for i, j, s in df[["item_idx", "sim_item_idx", "score"]].itertuples(index=False):
        out[int(i)].append([int(j), float(s)])
    return {k: np.array(v, dtype=np.float32) for k, v in out.items()}


def _load_usercf_index(scene: str) -> dict[int, np.ndarray]:
    """Load usercf user-to-user index into compact numpy arrays."""
    p = _resolve_existing(
        [
            *_index_candidates(f"usercf_{scene}_train_u2u_topk.parquet"),
            *_index_candidates(f"usercf_{scene}_u2u_topk.parquet"),
        ]
    )
    df = pd.read_parquet(p)
    out: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for u, v, s in df[["user_idx", "sim_user_idx", "score"]].itertuples(index=False):
        out[int(u)].append([int(v), float(s)])
    return {k: np.array(v, dtype=np.float32) for k, v in out.items()}


def _load_user_items(scene: str) -> dict[int, dict[int, float]]:
    p = _resolve_existing(_index_candidates(f"cf_{scene}_train_user_item_index.pkl"))
    with open(p, "rb") as f:
        obj = pickle.load(f)
    return obj["user_items"]


def _load_ann_assets(scene: str, tag: str):
    """Load ANN index, id mapping, and memmap-backed item embeddings."""
    idx_path = _resolve_existing(_index_candidates(f"dssm_{scene}_{tag}_ivfpq.faiss"))
    row2note_path = _resolve_existing(_index_candidates(f"dssm_{scene}_{tag}_row2note.npy"))
    emb_meta_path = _resolve_existing(_index_candidates(f"dssm_{scene}_{tag}_item_meta.json"))
    emb_path = _resolve_existing(_index_candidates(f"dssm_{scene}_{tag}_item_emb.bin"))

    index = faiss.read_index(str(idx_path))
    row2note = np.load(row2note_path)
    note2row = {int(n): int(i) for i, n in enumerate(row2note.tolist())}
    with open(emb_meta_path, "r") as f:
        emeta = json.load(f)
    n_items = int(emeta["num_items"])
    dim = int(emeta["dim"])
    # Use memmap to avoid loading the full item embedding matrix into RAM.
    emb = np.memmap(emb_path, dtype="float32", mode="r", shape=(n_items, dim))
    return index, row2note, note2row, emb, dim


def _load_request_ann_vecs(scene: str, tag: str, split: str, dim: int):
    candidates = [
        (
            INDEX_DIR / f"dssm_{scene}_{tag}_{split}_request_meta.json",
            INDEX_DIR / f"dssm_{scene}_{tag}_{split}_request_emb.bin",
            INDEX_DIR / f"dssm_{scene}_{tag}_{split}_request_map.json",
        )
    ]
    if scene == "search":
        candidates.append(
            (
                INDEX_DIR / f"dssm_search_{tag}_{split}_query_meta.json",
                INDEX_DIR / f"dssm_search_{tag}_{split}_query_emb.bin",
                INDEX_DIR / f"dssm_search_{tag}_{split}_query_map.json",
            )
        )
    for meta_path, emb_path, map_path in candidates:
        if not (meta_path.exists() and emb_path.exists() and map_path.exists()):
            continue
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        qn = int(meta.get("num_requests", meta.get("num_queries", 0)))
        qd = int(meta.get("dim", dim))
        if qn <= 0:
            continue
        if qd != dim:
            logger.warning(f"Request vec dim mismatch: req_dim={qd}, item_dim={dim}. fallback to history ANN.")
            continue
        qemb = np.memmap(emb_path, dtype="float32", mode="r", shape=(qn, qd))
        with open(map_path, "r", encoding="utf-8") as f:
            req2row_raw = json.load(f)
        req2row = {int(k): int(v) for k, v in req2row_raw.items()}
        return qemb, req2row
    return None, None


def _fuse_search_sources(
    ann_scores: dict[int, float],
    bm25_scores: dict[int, float],
    dpr_scores: dict[int, float],
    topk: int,
    route_weights: dict[str, float] | None = None,
) -> dict[int, float]:
    weights = _normalize_weight_map(dict(route_weights or SEARCH_ROUTE_FUSE_WEIGHTS))
    norm_ann = _normalize_route_scores(ann_scores)
    norm_bm25 = _normalize_route_scores(bm25_scores)
    norm_dpr = _normalize_route_scores(dpr_scores)
    union = set(norm_ann) | set(norm_bm25) | set(norm_dpr)
    if not union:
        return {}
    fused: dict[int, float] = {}
    for note_idx in union:
        score = (
            weights["ann"] * float(norm_ann.get(note_idx, 0.0))
            + weights["dpr"] * float(norm_dpr.get(note_idx, 0.0))
            + weights["bm25"] * float(norm_bm25.get(note_idx, 0.0))
        )
        if note_idx in norm_bm25 and note_idx in norm_dpr:
            score += 0.14
        if note_idx in norm_ann and note_idx in norm_bm25:
            score += 0.08
        if note_idx in norm_ann and note_idx in norm_dpr:
            score += 0.04
        if note_idx in norm_ann and note_idx in norm_bm25 and note_idx in norm_dpr:
            score += 0.08
        if score > 0.0:
            fused[int(note_idx)] = float(score)
    if not fused:
        return {}
    ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[: max(1, int(topk))]
    return {int(k): float(v) for k, v in ranked}


def _recall_swing(
    hist_items: list[int],
    swing_i2i: dict[int, np.ndarray],
    topk: int,
) -> dict[int, float]:
    """
    # Use vectorized numpy accumulation to avoid repeated dictionary lookups.
    """
    if not hist_items:
        return {}
    
    scores = {}
    n = len(hist_items)
    
    # Apply stronger weight to more recent history items.
    for idx, it in enumerate(reversed(hist_items)):
        sim_arr = swing_i2i.get(int(it))
        if sim_arr is None or len(sim_arr) == 0:
            continue
        decay = 1.0 / (1.0 + idx)
        # sim_arr shape: [n, 2], col0=item_idx, col1=score
        for row in sim_arr:
            item_id = int(row[0])
            scores[item_id] = scores.get(item_id, 0.0) + decay * float(row[1])
    
    if not scores:
        return {}
    # Use heapq.nlargest to avoid a full global sort for TopK.
    import heapq
    return dict(heapq.nlargest(topk, scores.items(), key=lambda x: x[1]))


def _recall_usercf(
    user_idx: int,
    usercf_u2u: dict[int, np.ndarray],
    user_items: dict[int, dict[int, float]],
    topk: int,
) -> dict[int, float]:
    """
    # Use numpy arrays and heapq to reduce sorting overhead.
    """
    scores = {}
    sim_users = usercf_u2u.get(int(user_idx))
    if sim_users is None or len(sim_users) == 0:
        return {}
    
    # sim_users shape: [n, 2], col0=sim_user_idx, col1=score
    for row in sim_users:
        v = int(row[0])
        sim_uv = float(row[1])
        v_items = user_items.get(v)
        if v_items is None:
            continue
        for item, pref in v_items.items():
            scores[int(item)] = scores.get(int(item), 0.0) + sim_uv * float(pref)
    
    if not scores:
        return {}
    import heapq
    return dict(heapq.nlargest(topk, scores.items(), key=lambda x: x[1]))


def _recall_ann(
    scene: str,
    hist_items: list[int],
    ann_index,
    row2note: np.ndarray,
    note2row: dict[int, int],
    item_emb: np.ndarray,
    request_vec: np.ndarray | None,
    topk: int,
):
    req_q = None
    if request_vec is not None:
        rv = np.asarray(request_vec, dtype=np.float32).reshape(1, -1)
        if rv.shape[1] == item_emb.shape[1] and float(np.linalg.norm(rv)) > 1e-12:
            req_q = rv

    hist_q = None
    rows = [note2row[it] for it in hist_items if it in note2row]
    if rows:
        hist_q = np.asarray(item_emb[rows], dtype=np.float32).mean(axis=0, keepdims=True)
        if float(np.linalg.norm(hist_q)) <= 1e-12:
            hist_q = None

    if req_q is not None and hist_q is not None:
        req_q = req_q / np.maximum(np.linalg.norm(req_q, axis=1, keepdims=True), 1e-12)
        hist_q = hist_q / np.maximum(np.linalg.norm(hist_q, axis=1, keepdims=True), 1e-12)
        hist_len = len(rows)
        if scene == "rec":
            if hist_len >= 8:
                hist_w = 0.96
            elif hist_len >= 4:
                hist_w = 0.88
            elif hist_len >= 2:
                hist_w = 0.76
            else:
                hist_w = 0.62
        else:
            hist_w = 0.18 if hist_len >= 6 else 0.10
        req_w = 1.0 - hist_w
        q = (req_w * req_q) + (hist_w * hist_q)
    elif req_q is not None:
        q = req_q
    elif hist_q is not None:
        q = hist_q
    else:
        return {}

    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    D, I = ann_index.search(q.astype(np.float32), topk)
    scores = {}
    for idx, s in zip(I[0].tolist(), D[0].tolist()):
        if idx < 0 or idx >= len(row2note):
            continue
        scores[int(row2note[idx])] = float(s)
    return scores


def _recall_ann_item_history(
    hist_items: list[int],
    ann_index,
    row2note: np.ndarray,
    note2row: dict[int, int],
    item_emb: np.ndarray,
    topk: int,
    per_item_topk: int = 180,
) -> dict[int, float]:
    if not hist_items:
        return {}
    scores: dict[int, float] = {}
    hist_set = {int(x) for x in hist_items}
    for idx, item in enumerate(reversed(hist_items[:20])):
        row = note2row.get(int(item))
        if row is None:
            continue
        q = np.asarray(item_emb[int(row)], dtype=np.float32).reshape(1, -1)
        q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
        D, I = ann_index.search(q.astype(np.float32), int(max(20, per_item_topk)))
        decay = 1.0 / (1.0 + idx)
        for ann_idx, score in zip(I[0].tolist(), D[0].tolist()):
            if ann_idx < 0 or ann_idx >= len(row2note):
                continue
            note_idx = int(row2note[ann_idx])
            if note_idx in hist_set:
                continue
            scores[note_idx] = scores.get(note_idx, 0.0) + decay * float(score)
    if not scores:
        return {}
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[: max(1, int(topk))]
    return {int(k): float(v) for k, v in ranked}




def _recall_ann_from_seed_scores(
    seed_scores: dict[int, float],
    ann_index,
    row2note: np.ndarray,
    note2row: dict[int, int],
    item_emb: np.ndarray,
    topk: int,
    max_seed: int = 80,
) -> dict[int, float]:
    if not seed_scores:
        return {}
    ranked_seed = sorted(seed_scores.items(), key=lambda x: x[1], reverse=True)[: max(1, int(max_seed))]
    rows = []
    weights = []
    seed_set = set()
    for note_idx, score in ranked_seed:
        row = note2row.get(int(note_idx))
        if row is None:
            continue
        rows.append(int(row))
        weights.append(float(max(0.0, score)))
        seed_set.add(int(note_idx))
    if not rows:
        return {}
    vecs = np.asarray(item_emb[rows], dtype=np.float32)
    ws = np.asarray(weights, dtype=np.float32)
    if float(ws.sum()) <= 1e-12:
        ws = np.ones_like(ws, dtype=np.float32)
    ws = ws / np.maximum(ws.sum(), 1e-12)
    q = (vecs * ws.reshape(-1, 1)).sum(axis=0, keepdims=True)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    dists, idxs = ann_index.search(q.astype(np.float32), int(max(20, topk)))
    out: dict[int, float] = {}
    for row_idx, score in zip(idxs[0].tolist(), dists[0].tolist()):
        if row_idx < 0 or row_idx >= len(row2note):
            continue
        note_idx = int(row2note[row_idx])
        if note_idx in seed_set:
            continue
        out[note_idx] = float(score)
    if not out:
        return {}
    ranked = sorted(out.items(), key=lambda x: x[1], reverse=True)[: max(1, int(topk))]
    return {int(k): float(v) for k, v in ranked}

def _normalize_route_scores(scores: dict[int, float]) -> dict[int, float]:
    """Apply request-local min-max normalization for a single route."""
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

    norm_scores = {int(k): float(v) for k, v in zip(keys, norm_vals.tolist())}
    # Sort explicitly by normalized score so later route merging is stable.
    return dict(sorted(norm_scores.items(), key=lambda x: x[1], reverse=True))


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


def _merge_scores(
    ann_scores: dict[int, float],
    swing_scores: dict[int, float],
    usercf_scores: dict[int, float],
    merge_order: list[str],
    topk: int,
    route_min_quota: int,
    route_max_share: float,
    route_weights: dict[str, float] | None = None,
):
    all_routes = ["ann", "swing", "usercf"]
    route2scores = {
        "ann": ann_scores,
        "swing": swing_scores,
        "usercf": usercf_scores,
    }
    weights = {"ann": 1.0, "swing": 1.0, "usercf": 1.0}
    if route_weights:
        for key, value in route_weights.items():
            if key in weights:
                weights[key] = float(value)
    route_order = list(dict.fromkeys([r for r in merge_order if r in all_routes] + all_routes))
    route2list = {r: list(route2scores[r].items()) for r in all_routes}
    route_cursor = {r: 0 for r in all_routes}
    route_take_cnt = {r: 0 for r in all_routes}
    route_cap = max(1, int(topk * max(0.0, min(1.0, route_max_share))))

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
            if any(_route_has_unseen(r) for r in all_routes if r != route):
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
            fused_score = (
                weights["ann"] * float(s_ann)
                + weights["swing"] * float(s_sw)
                + weights["usercf"] * float(s_ucf)
                + 0.12 * float(route_score)
                + overlap_bonus
            )
            merged.append(
                (
                    item,
                    float(fused_score),
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

    min_quota = max(0, int(route_min_quota))
    quota_order = [r for r in route_order if r != "ann"] + [r for r in route_order if r == "ann"]
    if min_quota > 0:
        for route in quota_order:
            while len(merged) < topk and route_take_cnt[route] < min_quota:
                if not _append_from_route(route, enforce_cap=False):
                    break

    while len(merged) < topk:
        progressed = False
        for route in route_order:
            if len(merged) >= topk:
                break
            if _append_from_route(route, enforce_cap=True):
                progressed = True
        if not progressed:
            break

    return merged[:topk]


class StreamingParquetWriter:
    """Stream parquet output to avoid materializing the full recall table in memory."""

    def __init__(self, out_path: Path, columns: list[str], flush_size: int = ROW_FLUSH_SIZE):
        self.out_path = out_path
        self.columns = columns
        self.flush_size = max(10_000, int(flush_size))
        self.buffer: list[tuple[Any, ...]] = []
        self.writer = None
        self.total_rows = 0

    def add(self, row: tuple[Any, ...]) -> None:
        self.buffer.append(row)
        if len(self.buffer) >= self.flush_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        df_chunk = pd.DataFrame(self.buffer, columns=self.columns)
        if pq is not None and pa is not None:
            table = pa.Table.from_pandas(df_chunk, preserve_index=False)
            if self.writer is None:
                self.writer = pq.ParquetWriter(str(self.out_path), table.schema)
            self.writer.write_table(table)
        else:
            # Fallback when pyarrow is unavailable: keep a small in-memory append buffer.
            if self.writer is None:
                self.writer = []
            self.writer.append(df_chunk)
        self.total_rows += len(df_chunk)
        self.buffer = []

    def close(self) -> pd.DataFrame | None:
        self.flush()
        if pq is not None and pa is not None:
            if self.writer is not None:
                self.writer.close()
            return None
        # Fallback branch: if no writer exists, concatenate once before flushing.
        if self.writer is None:
            pd.DataFrame(columns=self.columns).to_parquet(self.out_path, index=False)
            return pd.DataFrame(columns=self.columns)
        out_df = pd.concat(self.writer, ignore_index=True)
        out_df.to_parquet(self.out_path, index=False)
        return out_df


def build_multiroute_recall(
    scene: str,
    split: str,
    tag: str,
    topk: int,
    ann_topk: int,
    swing_topk: int,
    usercf_topk: int,
    route_min_quota: int,
    route_max_share: float,
    merge_order: list[str],
) -> Path:
    allowed_routes = {"ann", "swing", "usercf"}
    if not merge_order:
        raise ValueError("--merge-order cannot be empty")
    invalid = [x for x in merge_order if x not in allowed_routes]
    if invalid:
        raise ValueError(f"--merge-order contains invalid routes: {invalid}, allowed={sorted(allowed_routes)}")

    req_df, req_col = _load_requests(scene, split)
    swing_i2i = _load_swing_index(scene)
    seqtrans_i2i = _load_seq_transition_index(scene) if scene == "rec" else {}
    usercf_u2u = _load_usercf_index(scene)
    user_items = _load_user_items(scene)
    ann_index, row2note, note2row, item_emb, ann_dim = _load_ann_assets(scene, tag)
    query_ann_emb = None
    req2row = None
    query_ann_emb, req2row = _load_request_ann_vecs(scene=scene, tag=tag, split=split, dim=ann_dim)
    if query_ann_emb is None:
        logger.warning(f"{scene} request ANN vectors not found. Fallback to history-based ANN query.")

    output_columns = [
        req_col,
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
    out_path = OUT_DATA_DIR / f"recall_{scene}_{split}_{tag}_multiroute_top{int(topk)}.parquet"
    writer = StreamingParquetWriter(out_path=out_path, columns=output_columns, flush_size=ROW_FLUSH_SIZE)

    for row in tqdm(req_df.itertuples(index=False), total=len(req_df), desc=f"Recall {scene}_{split}", disable=DISABLE_TQDM):
        req_id = int(getattr(row, req_col))
        user_idx = int(getattr(row, "user_idx"))
        hist_items = _resolve_hist_items(
            raw_hist_items=getattr(row, "recent_clicked_note_idxs"),
            user_idx=int(user_idx),
            user_items=user_items,
            limit=20,
        )
        has_query_vec = req2row is not None and req_id in req2row
        if (not hist_items) and (not has_query_vec):
            continue

        req_vec = None
        if has_query_vec and query_ann_emb is not None and req2row is not None:
            req_vec = np.asarray(query_ann_emb[req2row[req_id]], dtype=np.float32)
        ann_scores = _recall_ann(
            scene,
            [],
            ann_index,
            row2note,
            note2row,
            item_emb,
            request_vec=req_vec,
            topk=ann_topk,
        )
        route_weights = _resolve_merge_route_weights(scene, hist_len=len(hist_items), has_request_vec=has_query_vec)
        if scene == "search":
            bm25_scores = _parse_scored_results(getattr(row, "bm25_results", None), limit=ann_topk)
            dpr_scores = _parse_scored_results(getattr(row, "dpr_results", None), limit=ann_topk)
            search_route_weights = _resolve_search_route_weights(getattr(row, "query", ""))
            route_weights = {
                "ann": float(search_route_weights.get("ann", 0.2)),
                "swing": float(search_route_weights.get("dpr", 0.3)),
                "usercf": float(search_route_weights.get("bm25", 0.5)),
            }
            swing_scores = dpr_scores
            usercf_scores = bm25_scores
        else:
            seq_transition_scores = _recall_swing(hist_items, seqtrans_i2i, topk=swing_topk) if seqtrans_i2i else {}
            seq_transition_ann_scores = _recall_ann_from_seed_scores(
                seed_scores=seq_transition_scores,
                ann_index=ann_index,
                row2note=row2note,
                note2row=note2row,
                item_emb=item_emb,
                topk=swing_topk,
                max_seed=80,
            ) if seq_transition_scores else {}
            seq_scores = _recall_ann_item_history(
                hist_items,
                ann_index,
                row2note,
                note2row,
                item_emb,
                topk=swing_topk,
            )
            seq_mean_scores = _recall_ann(
                scene,
                hist_items,
                ann_index,
                row2note,
                note2row,
                item_emb,
                request_vec=None,
                topk=swing_topk,
            )
            swing_scores = _blend_route_scores(
                seq_transition_scores,
                seq_transition_scores,
                seq_transition_ann_scores,
                primary_weight=1.0,
                secondary_weight=0.82,
                overlap_bonus=0.16,
                topk=swing_topk,
            )
            swing_scores = _blend_route_scores(
                swing_scores,
                seq_scores,
                primary_weight=1.0,
                secondary_weight=0.78,
                overlap_bonus=0.14,
                topk=swing_topk,
            )
            swing_scores = _blend_route_scores(
                swing_scores,
                seq_mean_scores,
                primary_weight=1.0,
                secondary_weight=0.36,
                overlap_bonus=0.10,
                topk=swing_topk,
            )
            usercf_scores = _recall_usercf(user_idx, usercf_u2u, user_items, topk=usercf_topk)
            graph_scores = _recall_swing(hist_items, swing_i2i, topk=swing_topk)
            if graph_scores:
                usercf_scores = _blend_route_scores(
                    usercf_scores,
                    graph_scores,
                    primary_weight=0.86,
                    secondary_weight=0.94,
                    overlap_bonus=0.08,
                    topk=usercf_topk,
                )
            if swing_scores:
                swing_scores = _blend_route_scores(
                    swing_scores,
                    ann_scores,
                    primary_weight=1.0,
                    secondary_weight=0.18,
                    overlap_bonus=0.04,
                    topk=swing_topk,
                )

        # Normalize route scores onto a common scale before merging routes.
        ann_scores = _normalize_route_scores(ann_scores)
        swing_scores = _normalize_route_scores(swing_scores)
        usercf_scores = _normalize_route_scores(usercf_scores)

        if scene == "search":
            effective_merge_order = ["usercf", "swing", "ann"]
        elif scene == "rec":
            effective_merge_order = ["swing", "usercf", "ann"]
        else:
            effective_merge_order = list(merge_order)
        merged = _merge_scores(
            ann_scores=ann_scores,
            swing_scores=swing_scores,
            usercf_scores=usercf_scores,
            merge_order=effective_merge_order,
            topk=topk,
            route_min_quota=32 if scene == "rec" else route_min_quota,
            route_max_share=0.82 if scene == "rec" else route_max_share,
            route_weights=route_weights,
        )
        for rank, (item, s, s_ann, s_sw, s_ucf, f_ann, f_sw, f_ucf, first_route) in enumerate(merged, start=1):
            if scene == "search":
                if first_route == "usercf":
                    first_route = "bm25"
                elif first_route == "swing":
                    first_route = "dpr"
            elif scene == "rec" and first_route == "swing":
                first_route = "seq"
            writer.add(
                (
                    req_id,
                    user_idx,
                    int(item),
                    int(rank),
                    float(s),
                    float(s_ann),
                    float(s_sw),
                    float(s_ucf),
                    f_ann,
                    f_sw,
                    f_ucf,
                    first_route,
                )
            )

    fallback_df = writer.close()
    if fallback_df is None:
        logger.info(f"Saved multi-route recall: {out_path}, rows={writer.total_rows}")
    else:
        logger.info(f"Saved multi-route recall: {out_path}, shape={fallback_df.shape}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=["search", "rec"], required=True)
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--tag", default="easy")
    parser.add_argument("--topk", type=int, default=1000)
    parser.add_argument("--ann-topk", type=int, default=1000)
    parser.add_argument("--swing-topk", type=int, default=1000)
    parser.add_argument("--usercf-topk", type=int, default=1000)
    parser.add_argument("--route-min-quota", type=int, default=100, help="minimum kept items per route after deduplication")
    parser.add_argument("--route-max-share", type=float, default=0.6, help="maximum share allowed for one route while others still have candidates")
    parser.add_argument(
        "--merge-order",
        type=str,
        default="ann,swing,usercf",
        help="merge order for routes, comma separated, e.g. ann,swing,usercf",
    )
    args = parser.parse_args()
    merge_order = [x.strip().lower() for x in args.merge_order.split(",") if x.strip()]

    build_multiroute_recall(
        scene=args.scene,
        split=args.split,
        tag=args.tag,
        topk=args.topk,
        ann_topk=args.ann_topk,
        swing_topk=args.swing_topk,
        usercf_topk=args.usercf_topk,
        route_min_quota=args.route_min_quota,
        route_max_share=args.route_max_share,
        merge_order=merge_order,
    )


if __name__ == "__main__":
    main()

