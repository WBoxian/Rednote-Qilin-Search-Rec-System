"""
Qilin Backend Online Pipeline
- 在线职责：维护单场景运行时状态（请求数据、特征缓存、模型对象）
- 推理链路：冷启动识别 -> 多路召回 -> 粗排 -> 精排 -> 结果组装
- 服务职责：作为 FastAPI 启动入口，完成场景状态初始化与运行时注册
- 运行形态：支持 search / rec 双场景，并按 tag 加载 outputs/deploy/{scene}/{tag}
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import re
import time
import threading
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import torch
import uvicorn

try:
    import faiss
except Exception:
    faiss = None

BASE_DIR = Path(__file__).resolve().parents[3]

from backend.online.config import load_online_config
from backend.online.cold_start.detector import is_cold_start
from backend.online.realtime_cache import build_realtime_cache
from backend.online.preranking.gbdt import GBDTPredictor
from backend.online.preranking.service import run_preranking
from backend.online.ranking.service import run_ranking
from backend.online.ranking.dien import DIENPredictor
from backend.online.recall.dssm import DSSMRecaller
from backend.online.recall.service import fetch_recall_candidates, run_recall
from backend.online.search_query import (
    DEFAULT_SERVICE_STOPWORDS,
    SearchQueryResources,
    build_search_query_resources,
    normalize_query_text,
    preprocess_search_query as run_search_query_preprocess,
    seed_query_terms,
    segment_query_terms,
)

DATASETS_DIR = BASE_DIR / "datasets"
FEATURE_DIR = BASE_DIR / "features"
OUT_DIR = BASE_DIR / "outputs"
OUT_DATA_DIR = OUT_DIR / "data"
DEPLOY_DIR = OUT_DIR / "deploy"
IMAGE_DIR = BASE_DIR / "image"
COLD_START_REQ_THRESHOLD = 3
HOT_ROUTE_TOPK = 300
TEST_SAMPLE_SEED = 303


def _load_module_from_src(module_name: str, rel_path: str):
    module_path = BASE_DIR / "src" / rel_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _safe_intish(v: Any, default: int = 0) -> int:
    try:
        if v is None:
            return default
        if isinstance(v, (list, tuple, dict, set)):
            return default
        if pd.isna(v):
            return default
        return int(float(v))
    except Exception:
        return default


def _shrink_numeric_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df
    float_cols = out.select_dtypes(include=["float64"]).columns
    int_cols = out.select_dtypes(include=["int64"]).columns
    if len(float_cols) > 0:
        out.loc[:, float_cols] = out[float_cols].astype(np.float32)
    if len(int_cols) > 0:
        for col in int_cols:
            if col in {"user_idx", "note_idx"}:
                out[col] = out[col].astype(np.int32)
            else:
                out[col] = pd.to_numeric(out[col], downcast="integer")
    return out


@lru_cache(maxsize=32)
def _read_parquet_cached(path_str: str, columns_key: tuple[str, ...] | None = None) -> pd.DataFrame:
    columns = list(columns_key) if columns_key is not None else None
    return pd.read_parquet(path_str, columns=columns)


@lru_cache(maxsize=32)
def _load_npy_cached(path_str: str) -> np.ndarray:
    return np.load(path_str, allow_pickle=True)


def _unique_in_order(values: list[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for v in values:
        iv = int(v)
        if iv in seen:
            continue
        seen.add(iv)
        out.append(iv)
    return out


def _select_group_ids(df: pd.DataFrame, group_key: str, sample_n: int, user_idx: int | None = None) -> list[int]:
    if df.empty or sample_n <= 0:
        return []
    group_stats = (
        df.groupby(group_key, as_index=False)
        .agg(
            pos_cnt=("y_multi", lambda s: int((np.asarray(s) > 0).sum())),
            max_y=("y_multi", "max"),
        )
        .sort_values(["pos_cnt", "max_y", group_key], ascending=[False, False, True], kind="mergesort")
    )
    if len(group_stats) <= sample_n:
        return [int(x) for x in group_stats[group_key].tolist()]

    if user_idx is not None and "user_idx" in df.columns:
        user_groups = df.loc[df["user_idx"] == int(user_idx), group_key].drop_duplicates().tolist()
        if user_groups:
            prioritized = group_stats[group_stats[group_key].isin([int(x) for x in user_groups])]
            remainder = group_stats[~group_stats[group_key].isin([int(x) for x in user_groups])]
            picked = prioritized.head(sample_n // 2)[group_key].tolist() + remainder.head(sample_n - min(len(prioritized), sample_n // 2))[group_key].tolist()
            picked = _unique_in_order([int(x) for x in picked])
            return picked[:sample_n]

    return [int(x) for x in group_stats.head(sample_n)[group_key].tolist()]


def _resolve_score_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _to_path_list(v: Any) -> list[str]:
    if isinstance(v, np.ndarray):
        return [str(x) for x in v.tolist()]
    if isinstance(v, list):
        return [str(x) for x in v]
    if isinstance(v, str):
        return [v]
    return []


def _normalize_image_rel_path(p: str) -> str:
    s = str(p or "").strip().lstrip("/")
    if s.startswith("image/"):
        s = s[len("image/") :]
    return s


def _image_exists(rel_path: str) -> bool:
    if not rel_path:
        return False
    f = (IMAGE_DIR / rel_path).resolve()
    image_root = IMAGE_DIR.resolve()
    if f != image_root and image_root not in f.parents:
        return False
    return f.exists() and f.is_file()


def _existing_images(paths: list[str]) -> list[str]:
    out: list[str] = []
    for p in paths:
        rel = _normalize_image_rel_path(p)
        if _image_exists(rel):
            out.append(rel)
    return out


def _has_existing_image(raw_paths: Any) -> bool:
    return bool(_existing_images(_to_path_list(raw_paths)))


def _char_ngrams(s: str, n: int = 2) -> set[str]:
    s = (s or "").strip().lower()
    if len(s) <= n:
        return {s} if s else set()
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def _text_match_score(query: str, title: str, content: str) -> float:
    q = (query or "").strip().lower()
    if not q:
        return 0.0
    t = (title or "").lower()
    c = (content or "").lower()
    score = 0.0
    if q in t:
        score += 3.0
    if q in c:
        score += 1.0
    qg = _char_ngrams(q, n=2)
    tg = _char_ngrams(t, n=2)
    cg = _char_ngrams(c[:500], n=2)
    if qg and tg:
        score += 2.0 * (len(qg & tg) / max(len(qg), 1))
    if qg and cg:
        score += 0.5 * (len(qg & cg) / max(len(qg), 1))
    return float(score)


def _group_key(scene: str) -> str:
    return "search_idx" if scene == "search" else "request_idx"


def _scene_test_dataset_path(scene: str) -> Path:
    if scene == "search":
        return DATASETS_DIR / "search_test" / "train-00000-of-00001.parquet"
    return DATASETS_DIR / "recommendation_test" / "train-00000-of-00001.parquet"


def _scene_request_dataset_path(scene: str, split: str) -> Path:
    if scene == "search":
        return DATASETS_DIR / f"search_{split}" / "train-00000-of-00001.parquet"
    return DATASETS_DIR / f"recommendation_{split}" / "train-00000-of-00001.parquet"


def _recall_artifact_path(scene: str, split: str, tag: str) -> Path:
    return BASE_DIR / "outputs" / "data" / f"recall_{scene}_{split}_{str(tag or 'hard').lower()}_multiroute_top1000.parquet"


_normalize_search_query_text = normalize_query_text
_seed_query_terms = seed_query_terms
_segment_query_terms = segment_query_terms


def _extract_terms(text: str, max_terms: int = 6) -> list[str]:
    return seed_query_terms(text)[: max(1, int(max_terms))]


_TOPIC_TAG_RE = re.compile(r"#\[([^\]#]{1,40})\]#|#([^#\[\]]{1,40})\[话题\]#")
_RESULT_ID_RE = re.compile(r"-?\d+")
_SCORED_RESULT_RE = re.compile(r"array\(\[\s*(-?\d+)\s*,\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?)\s*\]\)")


def _clean_text_value(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    return "" if not text or text.lower() == "nan" else text


def _extract_topic_tags(text: Any, limit: int = 3) -> list[str]:
    raw = _clean_text_value(text)
    if not raw:
        return []
    tags: list[str] = []
    seen: set[str] = set()
    for match in _TOPIC_TAG_RE.finditer(raw):
        candidate = _clean_text_value(match.group(1) or match.group(2))
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        tags.append(candidate)
        if len(tags) >= int(limit):
            break
    return tags


def _format_note_title(title: Any, content: Any = "") -> str:
    clean_title = _clean_text_value(title)
    if clean_title:
        return clean_title
    tags = _extract_topic_tags(content, limit=3)
    if tags:
        return "(无标题) " + " ".join(f"#{tag}[话题]#" for tag in tags)
    return "(无标题)"


def _parse_result_ids(raw: Any, limit: int = 200) -> list[int]:
    if raw is None:
        return []
    values: list[int] = []
    if isinstance(raw, np.ndarray):
        values = [int(x) for x in raw.tolist() if _safe_intish(x, -1) >= 0]
    elif isinstance(raw, list):
        values = [int(x) for x in raw if _safe_intish(x, -1) >= 0]
    else:
        text = _clean_text_value(raw)
        if not text:
            return []
        values = [int(x) for x in _RESULT_ID_RE.findall(text)]
    return _unique_in_order([int(x) for x in values if int(x) >= 0])[: max(1, int(limit))]


def _parse_result_detail_ids(raw: Any, limit: int = 200) -> list[int]:
    if raw is None:
        return []
    rows: list[dict[str, Any]] = []
    if isinstance(raw, np.ndarray):
        rows = [x for x in raw.tolist() if isinstance(x, dict)]
    elif isinstance(raw, list):
        rows = [x for x in raw if isinstance(x, dict)]
    else:
        return []
    rows = sorted(
        rows,
        key=lambda row: (
            _safe_intish(row.get("position"), 10**9),
            _safe_intish(row.get("note_idx"), 10**9),
        ),
    )
    note_ids = [_safe_intish(row.get("note_idx"), -1) for row in rows]
    return _unique_in_order([int(x) for x in note_ids if int(x) >= 0])[: max(1, int(limit))]


def _request_result_col(scene: str) -> str:
    return "search_results" if str(scene) == "search" else "rec_results"


def _request_result_detail_col(scene: str) -> str:
    return "search_result_details_with_idx" if str(scene) == "search" else "rec_result_details_with_idx"


def _build_result_label_frame(
    result_map: dict[int, list[int]],
    gids: list[int],
    group_key: str,
    relevance_df: pd.DataFrame | None = None,
    topk: int = 10,
) -> pd.DataFrame:
    rel_map: dict[tuple[int, int], float] = {}
    user_map: dict[int, int] = {}
    if relevance_df is not None and len(relevance_df) > 0:
        rel_cols = [group_key, "note_idx", "y_multi"] + (["user_idx"] if "user_idx" in relevance_df.columns else [])
        rel_base = relevance_df[rel_cols].drop_duplicates([group_key, "note_idx"]).copy()
        rel_map = {}
        for row in rel_base.to_dict("records"):
            gid = _safe_intish(row.get(group_key), -1)
            nid = _safe_intish(row.get("note_idx"), -1)
            if gid < 0 or nid < 0:
                continue
            y_val = float(pd.to_numeric(row.get("y_multi"), errors="coerce"))
            if not np.isfinite(y_val):
                y_val = 0.0
            rel_map[(int(gid), int(nid))] = y_val
        if "user_idx" in rel_base.columns:
            user_map = (
                rel_base[[group_key, "user_idx"]]
                .dropna(subset=[group_key])
                .drop_duplicates(subset=[group_key], keep="first")
                .assign(**{group_key: lambda df: df[group_key].astype(np.int64), "user_idx": lambda df: pd.to_numeric(df["user_idx"], errors="coerce").fillna(0).astype(np.int64)})
                .set_index(group_key)["user_idx"]
                .to_dict()
            )
    rows: list[dict[str, Any]] = []
    max_rank = max(1, int(topk))
    for gid in gids:
        result_ids = result_map.get(int(gid), [])[:max_rank]
        for rank, note_idx in enumerate(result_ids, start=1):
            y_val = float(rel_map.get((int(gid), int(note_idx)), 0.0))
            rows.append(
                {
                    group_key: int(gid),
                    "note_idx": int(note_idx),
                    "y_multi": y_val,
                    "click": int(y_val > 0.0),
                    "result_rank": int(rank),
                    "user_idx": int(user_map.get(int(gid), 0)),
                }
            )
    return pd.DataFrame(rows, columns=[group_key, "note_idx", "y_multi", "click", "result_rank", "user_idx"])


def _binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y = np.asarray(y_true).astype(np.int8)
    s = np.asarray(y_score, dtype=np.float64)
    pos = int((y > 0).sum())
    neg = int((y <= 0).sum())
    if pos <= 0 or neg <= 0:
        return 0.5
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    pos_rank_sum = float(ranks[y > 0].sum())
    return float((pos_rank_sum - pos * (pos + 1) / 2.0) / max(pos * neg, 1))


def _group_auc(df: pd.DataFrame, user_col: str, label_col: str, score_col: str) -> float:
    vals: list[float] = []
    weights: list[int] = []
    for _, sub in df.groupby(user_col, sort=False):
        y = (pd.to_numeric(sub[label_col], errors="coerce").fillna(0.0).to_numpy() > 0).astype(np.int8)
        if int(y.sum()) <= 0 or int((y <= 0).sum()) <= 0:
            continue
        vals.append(_binary_auc(y, pd.to_numeric(sub[score_col], errors="coerce").fillna(0.0).to_numpy()))
        weights.append(len(sub))
    if not vals:
        return 0.5
    w = np.asarray(weights, dtype=np.float64)
    return float(np.average(np.asarray(vals, dtype=np.float64), weights=w))


def _recall_at_k(df: pd.DataFrame, group_key: str, k: int) -> float:
    vals: list[float] = []
    for _, sub in df.groupby(group_key, sort=False):
        y = (pd.to_numeric(sub["y_multi"], errors="coerce").fillna(0.0).to_numpy() > 0).astype(np.int8)
        total_pos = int(y.sum())
        if total_pos <= 0:
            continue
        vals.append(float(min(total_pos, int(y[:k].sum())) / total_pos))
    return float(np.mean(vals)) if vals else 0.0


def _precision_at_k(df: pd.DataFrame, group_key: str, k: int) -> float:
    vals: list[float] = []
    for _, sub in df.groupby(group_key, sort=False):
        y = (pd.to_numeric(sub["y_multi"], errors="coerce").fillna(0.0).to_numpy() > 0).astype(np.int8)
        vals.append(float(y[:k].sum() / max(min(k, len(y)), 1)))
    return float(np.mean(vals)) if vals else 0.0


def _map_at_k(df: pd.DataFrame, group_key: str, k: int) -> float:
    vals: list[float] = []
    for _, sub in df.groupby(group_key, sort=False):
        y = (pd.to_numeric(sub["y_multi"], errors="coerce").fillna(0.0).to_numpy() > 0).astype(np.int8)[:k]
        total_pos = int(y.sum())
        if total_pos <= 0:
            continue
        hit = 0
        ap = 0.0
        for i, label in enumerate(y, start=1):
            if int(label) > 0:
                hit += 1
                ap += hit / i
        vals.append(float(ap / max(total_pos, 1)))
    return float(np.mean(vals)) if vals else 0.0


def _merge_scored_with_labels(
    full_df: pd.DataFrame,
    scored_df: pd.DataFrame,
    group_key: str,
    score_cols: list[str],
) -> pd.DataFrame:
    if len(full_df) <= 0:
        return pd.DataFrame()
    base_cols = [group_key, "note_idx", "y_multi"]
    if "user_idx" in full_df.columns:
        base_cols.append("user_idx")
    if "result_rank" in full_df.columns:
        base_cols.append("result_rank")
    base = full_df[base_cols].drop_duplicates([group_key, "note_idx"]).copy()
    if len(scored_df) <= 0:
        for col in score_cols:
            base[col] = np.nan
        return base
    keep_cols = [group_key, "note_idx"] + [col for col in score_cols if col in scored_df.columns]
    if "user_idx" in scored_df.columns and "user_idx" not in base.columns:
        keep_cols.append("user_idx")
    if "result_rank" in scored_df.columns and "result_rank" not in base.columns:
        keep_cols.append("result_rank")
    merged_scores = scored_df[keep_cols].copy()
    primary = next((col for col in score_cols if col in merged_scores.columns), None)
    if primary is not None:
        merged_scores = merged_scores.sort_values(
            [group_key, primary],
            ascending=[True, False],
            kind="mergesort",
        )
    merged_scores = merged_scores.drop_duplicates([group_key, "note_idx"], keep="first")
    scored = merged_scores.merge(base, on=[group_key, "note_idx"], how="left", suffixes=("", "_label"))
    if "user_idx_label" in scored.columns:
        if "user_idx" in scored.columns:
            scored["user_idx"] = pd.to_numeric(scored["user_idx"], errors="coerce").fillna(
                pd.to_numeric(scored["user_idx_label"], errors="coerce").fillna(0)
            )
        else:
            scored["user_idx"] = pd.to_numeric(scored["user_idx_label"], errors="coerce").fillna(0)
        scored = scored.drop(columns=["user_idx_label"])
    if "result_rank_label" in scored.columns:
        if "result_rank" in scored.columns:
            scored["result_rank"] = pd.to_numeric(scored["result_rank"], errors="coerce").fillna(
                pd.to_numeric(scored["result_rank_label"], errors="coerce").fillna(0)
            )
        else:
            scored["result_rank"] = pd.to_numeric(scored["result_rank_label"], errors="coerce").fillna(0)
        scored = scored.drop(columns=["result_rank_label"])
    scored["y_multi"] = pd.to_numeric(scored.get("y_multi"), errors="coerce").fillna(0.0)
    missing = base.merge(
        merged_scores[[group_key, "note_idx"]],
        on=[group_key, "note_idx"],
        how="left",
        indicator=True,
    )
    missing = missing[missing["_merge"] == "left_only"].drop(columns=["_merge"])
    for col in score_cols:
        if col not in missing.columns:
            missing[col] = np.nan
    desired_cols = list(dict.fromkeys(base_cols + [col for col in score_cols if col in scored.columns]))
    scored_out = scored[desired_cols].copy()
    missing_out = missing[desired_cols].copy()
    if len(missing_out) <= 0:
        return scored_out
    if len(scored_out) <= 0:
        return missing_out
    return pd.concat([scored_out, missing_out], ignore_index=True)


def _fill_missing_eval_scores(
    df: pd.DataFrame,
    score_col: str,
    label_col: str = "y_multi",
) -> pd.DataFrame:
    if len(df) <= 0 or score_col not in df.columns:
        return df
    out = df.copy()
    score = pd.to_numeric(out.get(score_col), errors="coerce")
    finite_mask = np.isfinite(score.to_numpy())
    score_floor = float(np.nanmin(score.to_numpy()[finite_mask])) if finite_mask.any() else -1.0
    out[score_col] = score.astype(np.float64)
    missing_mask = out[score_col].isna()
    if not bool(missing_mask.any()):
        return out
    labels = pd.to_numeric(out.get(label_col), errors="coerce").fillna(0.0)
    neg_missing = out[missing_mask & (labels <= 0)]
    pos_missing = out[missing_mask & (labels > 0)]
    if len(neg_missing) > 0:
        out.loc[neg_missing.index, score_col] = score_floor - 1.0 - (np.arange(len(neg_missing), dtype=np.float64) / 1000.0)
    if len(pos_missing) > 0:
        out.loc[pos_missing.index, score_col] = score_floor - 2.0 - (np.arange(len(pos_missing), dtype=np.float64) / 1000.0)
    return out


class SceneServingState:
    """统一的单场景 serving 状态（search / rec 共享同一套逻辑）。"""

    def __init__(self, scene: str, tag: str = "easy", gbdt_topn: int = 500, recall_rank_cap: int = 1000):
        if scene not in {"search", "rec"}:
            raise ValueError(f"invalid scene: {scene}")
        self.scene = scene
        self.group_key = _group_key(scene)
        self.tag = tag
        self.deploy_tag_dir = DEPLOY_DIR / scene / tag
        self.gbdt_topn = int(gbdt_topn)
        self.recall_rank_cap = int(recall_rank_cap)

        self.scene_test_path = _scene_test_dataset_path(scene)
        self.scene_train_path = _scene_request_dataset_path(scene, "train")
        self.user_feat_path = DATASETS_DIR / "user_feat" / "train-00000-of-00001.parquet"
        self.scene_feat_path = FEATURE_DIR / f"{scene}_test_features.parquet"
        self.scene_train_feat_path = FEATURE_DIR / f"{scene}_train_features.parquet"
        self.notes_glob = str(DATASETS_DIR / "notes" / "train-*.parquet")
        self.recall_test_path = _recall_artifact_path(scene, "test", tag)
        self.realtime_cache = build_realtime_cache()

        utils_mod = _load_module_from_src("utils", "training/utils.py")
        dien_train = _load_module_from_src("online_dien_ranker", "training/dien_ranker.py")

        self.dien_train = dien_train
        self.discretize_relevance = getattr(utils_mod, "discretize_relevance")
        self.eval_ndcg_by_group = getattr(utils_mod, "eval_ndcg_by_group")

        self._predict_lock = threading.Lock()
        self._note_cache_lock = threading.Lock()
        self._note_meta_cache: dict[int, dict[str, Any]] = {}
        self._feed_cache_lock = threading.Lock()
        self._feed_cache: dict[tuple[int, int, str, str], tuple[float, pd.DataFrame, dict[str, Any]]] = {}
        self._feed_cache_ttl_sec = float(os.getenv("QILIN_FEED_CACHE_TTL_SEC", "12"))
        self._stage_cache_lock = threading.Lock()
        self._stage_cache: dict[tuple[int, int, str], tuple[float, pd.DataFrame, dict[str, Any]]] = {}
        self._stage_cache_ttl_sec = float(os.getenv("QILIN_STAGE_CACHE_TTL_SEC", "45"))
        self._test_recall_cache_lock = threading.Lock()
        self._test_recall_cache: dict[int, tuple[float, pd.DataFrame]] = {}
        self._test_rank_cache_lock = threading.Lock()
        self._test_rank_cache: dict[int, tuple[float, pd.DataFrame, pd.DataFrame]] = {}
        self._test_rank_cache_ttl_sec = float(os.getenv("QILIN_TEST_RANK_CACHE_TTL_SEC", "900"))
        self._test_metrics_cache_max = max(2, min(8, int(os.getenv("QILIN_TEST_METRICS_CACHE_MAX", "4"))))
        self.live_recall_rank_cap = min(
            self.recall_rank_cap,
            max(80, int(os.getenv("QILIN_ONLINE_LIVE_RECALL_CAP", "320"))),
        )
        self.live_gbdt_topn = min(
            self.gbdt_topn,
            max(80, int(os.getenv("QILIN_ONLINE_LIVE_GBDT_TOPN", "180"))),
        )
        self.empty_feat_df = pd.DataFrame()
        self._request_req_cols: list[str] = []
        self._train_req_df = pd.DataFrame()
        self._train_request_true_results_map: dict[int, list[int]] = {}
        self._train_search_query_map: dict[int, str] = {}
        self._train_metric_label_df = pd.DataFrame()
        self._train_feat_df = pd.DataFrame()
        self._train_feat_group_indices: dict[int, np.ndarray] = {}
        self._validation_recall_cache_lock = threading.Lock()
        self._validation_recall_cache: dict[int, tuple[float, pd.DataFrame]] = {}
        self._app_state = None
        self._load_data()
        self._load_models()

    def _artifact_path(self, rel_path: str) -> Path:
        return self.deploy_tag_dir / rel_path

    def _load_data(self) -> None:
        # 请求、特征、用户画像全部常驻内存，降低在线路径 I/O 开销
        req_cols = self._request_columns_for_split("test")
        self._request_req_cols = list(req_cols)

        if self.scene_test_path.exists():
            self.req_df = pd.read_parquet(self.scene_test_path, columns=req_cols)
            self.req_df = self.req_df.drop_duplicates(subset=[self.group_key]).reset_index(drop=True)
            self.req_df = _shrink_numeric_dtypes(self.req_df)
        else:
            self.req_df = pd.DataFrame(columns=req_cols)

        self.request_true_results_map = self._build_request_truth_map_from_df(self.req_df)

        self.feat_df = pd.read_parquet(self.scene_feat_path)
        self.feat_df[self.group_key] = self.feat_df[self.group_key].astype(np.int64)
        self.feat_df["note_idx"] = self.feat_df["note_idx"].astype(np.int64)
        self.feat_df = _shrink_numeric_dtypes(self.feat_df)
        self.empty_feat_df = self.feat_df.iloc[0:0].copy()
        self.feat_group_indices = {
            int(k): np.asarray(v, dtype=np.int32)
            for k, v in self.feat_df.groupby(self.group_key, sort=False).indices.items()
        }

        user_feat_df = pd.read_parquet(self.user_feat_path)
        user_feat_df["user_idx"] = user_feat_df["user_idx"].astype(np.int64)
        user_feat_df = _shrink_numeric_dtypes(user_feat_df)
        self.user_feat_map = {
            int(r["user_idx"]): r for r in user_feat_df.to_dict("records")
        }
        self.user_count = int(user_feat_df["user_idx"].nunique())

        req_sorted = (
            self.req_df.sort_values(["user_idx", "session_idx", self.group_key])
            if len(self.req_df) > 0
            else self.req_df
        )
        self.user_requests = (
            req_sorted.groupby("user_idx")[self.group_key].apply(lambda s: [int(x) for x in s.tolist()]).to_dict()
            if len(req_sorted) > 0
            else {}
        )
        self.available_user_ids = (
            [int(x) for x in req_sorted["user_idx"].dropna().drop_duplicates().tolist()]
            if len(req_sorted) > 0 and "user_idx" in req_sorted.columns
            else [int(x) for x in user_feat_df["user_idx"].dropna().drop_duplicates().tolist()]
        )

        if self.scene == "search" and "query" in self.req_df.columns:
            self.search_query_map = {
                int(r[self.group_key]): str(r.get("query") or "")
                for r in self.req_df[[self.group_key, "query"]].to_dict("records")
            }
            queries = (
                self.req_df["query"]
                .fillna("")
                .astype(str)
                .str.strip()
            )
            vc = queries[queries.ne("")].value_counts()
            self.search_query_catalog = [(str(idx), int(cnt)) for idx, cnt in vc.items()]
            self.search_query_resources = build_search_query_resources(self.search_query_catalog)
            self.search_query_norm_catalog = list(self.search_query_resources.norm_catalog)
            self.search_query_norm_map = dict(self.search_query_resources.norm_map)
            self.search_term_score_map = dict(self.search_query_resources.term_score_map)
            self.search_term_bucket_map = {
                str(k): list(v) for k, v in self.search_query_resources.term_bucket_map.items()
            }
        else:
            self.search_query_map = {}
            self.search_query_catalog = []
            self.search_query_resources = SearchQueryResources(
                query_catalog=(),
                normalized_catalog=(),
                normalized_map={},
                query_scores={},
                term_scores={},
                term_bucket_map={},
                dynamic_stopwords=frozenset(DEFAULT_SERVICE_STOPWORDS),
            )
            self.search_query_norm_catalog = []
            self.search_query_norm_map = {}
            self.search_term_score_map = {}
            self.search_term_bucket_map = {}
        self.hot_note_pool = self._load_hot_note_pool(limit=int(os.getenv("QILIN_HOT_NOTE_POOL_SIZE", "4000")))

    def _request_columns_for_split(self, split: str) -> list[str]:
        req_cols = [self.group_key, "user_idx", "session_idx", "query"]
        if self.scene == "search":
            req_cols += ["dpr_results"]
            if str(split) == "test":
                req_cols += ["bm25_results", "search_results", "search_result_details_with_idx"]
            else:
                req_cols += ["search_result_details_with_idx"]
        else:
            if str(split) == "test":
                req_cols += ["rec_results", "rec_result_details_with_idx"]
            else:
                req_cols += ["rec_result_details_with_idx"]
        return list(dict.fromkeys(req_cols))

    def _build_request_truth_map_from_df(self, req_df: pd.DataFrame) -> dict[int, list[int]]:
        result_col = _request_result_col(self.scene)
        detail_col = _request_result_detail_col(self.scene)
        out: dict[int, list[int]] = {}
        if len(req_df) <= 0:
            return out
        for row in req_df.to_dict("records"):
            gid = _safe_intish(row.get(self.group_key), -1)
            if gid < 0:
                continue
            detail_ids = _parse_result_detail_ids(row.get(detail_col), limit=200)
            if detail_ids:
                out[int(gid)] = detail_ids
                continue
            result_ids = _parse_result_ids(row.get(result_col), limit=200)
            out[int(gid)] = result_ids
        return out

    def _get_split_req_df(self, split: str) -> pd.DataFrame:
        if str(split) == "test":
            return self.req_df
        if len(self._train_req_df) > 0:
            return self._train_req_df
        if not self.scene_train_path.exists():
            self._train_req_df = pd.DataFrame(columns=list(self._request_req_cols))
            return self._train_req_df
        req_df = pd.read_parquet(self.scene_train_path, columns=self._request_columns_for_split("train"))
        req_df = req_df.drop_duplicates(subset=[self.group_key]).reset_index(drop=True)
        req_df = _shrink_numeric_dtypes(req_df)
        self._train_req_df = req_df
        self._train_request_true_results_map = self._build_request_truth_map_from_df(req_df)
        return self._train_req_df

    def _get_split_request_truth_map(self, split: str) -> dict[int, list[int]]:
        if str(split) == "test":
            return self.request_true_results_map
        if not self._train_request_true_results_map:
            self._get_split_req_df("train")
        return self._train_request_true_results_map

    def _get_split_search_query_map(self, split: str) -> dict[int, str]:
        if self.scene != "search":
            return {}
        if str(split) == "test":
            return self.search_query_map
        if not self._train_search_query_map:
            req_df = self._get_split_req_df("train")
            if len(req_df) > 0 and "query" in req_df.columns:
                self._train_search_query_map = {
                    int(r[self.group_key]): str(r.get("query") or "")
                    for r in req_df[[self.group_key, "query"]].to_dict("records")
                }
        return self._train_search_query_map

    def _get_split_metric_labels(self, split: str) -> pd.DataFrame:
        if str(split) == "test":
            cols = [self.group_key, "note_idx", "y_multi"] + (["user_idx"] if "user_idx" in self.feat_df.columns else [])
            return self.feat_df[cols].drop_duplicates([self.group_key, "note_idx"]).copy()
        if len(self._train_metric_label_df) > 0:
            return self._train_metric_label_df
        if not self.scene_train_feat_path.exists():
            self._train_metric_label_df = pd.DataFrame(columns=[self.group_key, "note_idx", "y_multi", "user_idx"])
            return self._train_metric_label_df
        cols = [self.group_key, "note_idx", "y_multi", "user_idx"]
        try:
            train_df = _read_parquet_cached(str(self.scene_train_feat_path), tuple(cols))
        except Exception:
            train_df = _read_parquet_cached(str(self.scene_train_feat_path), (self.group_key, "note_idx", "y_multi"))
        self._train_metric_label_df = train_df.drop_duplicates([self.group_key, "note_idx"]).copy()
        return self._train_metric_label_df

    def _build_request_truth_frame(self, split: str, gids: list[int], topk: int = 200) -> pd.DataFrame:
        return _build_result_label_frame(
            result_map=self._get_split_request_truth_map(split),
            gids=[int(x) for x in gids],
            group_key=self.group_key,
            relevance_df=self._get_split_metric_labels(split),
            topk=max(1, int(topk)),
        )

    def _get_ranked_truth_ids(self, gid: int, split: str = "test", topk: int = 10) -> list[int]:
        truth_df = self._build_request_truth_frame(split=split, gids=[int(gid)], topk=max(200, int(topk)))
        if len(truth_df) <= 0:
            return []
        truth_df = truth_df[truth_df[self.group_key] == int(gid)].copy()
        if len(truth_df) <= 0:
            return []
        truth_df["result_rank"] = pd.to_numeric(truth_df.get("result_rank"), errors="coerce").fillna(1e9)
        truth_df = truth_df.sort_values(["result_rank"], ascending=[True], kind="mergesort")
        return [int(x) for x in truth_df["note_idx"].astype(int).tolist()[: max(1, int(topk))]]

    def _load_hot_note_pool(self, limit: int = 4000) -> list[int]:
        n = max(200, min(20_000, int(limit)))
        con = duckdb.connect(database=":memory:")
        try:
            q = """
            SELECT note_idx
            FROM read_parquet(?)
            WHERE COALESCE(note_title, '') <> ''
              AND COALESCE(image_num, 0) > 0
            ORDER BY (
                COALESCE(accum_like_num, 0) * 1.0
                + COALESCE(accum_collect_num, 0) * 1.5
                + COALESCE(accum_comment_num, 0) * 2.0
            ) DESC, note_idx DESC
            LIMIT ?
            """
            df = con.execute(q, [self.notes_glob, n]).df()
        except Exception:
            return []
        finally:
            con.close()
        if "note_idx" not in df.columns or len(df) <= 0:
            return []
        return [int(x) for x in df["note_idx"].tolist() if int(x) >= 0]

    def preprocess_search_query(self, query: str, user_idx: int | None = None) -> dict[str, Any]:
        if self.scene != "search":
            return {
                "input_query": str(query or ""),
                "normalized_query": str(query or ""),
                "corrected_query": "",
                "resolved_query": str(query or ""),
                "terms": [],
                "core_terms": [],
                "optional_terms": [],
                "must_terms": [],
                "intents": [],
                "service_query": str(query or ""),
            }
        recent_queries: list[str] = []
        if user_idx is not None:
            reqs = self.user_requests.get(int(user_idx), [])
            recent_queries.extend(
                str(self.search_query_map.get(int(rid), "")).strip()
                for rid in reqs[-32:]
                if str(self.search_query_map.get(int(rid), "")).strip()
            )
            if self.realtime_cache is not None:
                rt_reqs = self.realtime_cache.get_user_requests(int(user_idx), self.scene) or []
                recent_queries.extend(
                    str(self.search_query_map.get(int(rid), "")).strip()
                    for rid in rt_reqs[-16:]
                    if str(self.search_query_map.get(int(rid), "")).strip()
                )
        return run_search_query_preprocess(
            query,
            resources=self.search_query_resources,
            recent_queries=recent_queries,
        )

    def get_feat_req(self, request_id: int) -> pd.DataFrame:
        idx = self.feat_group_indices.get(int(request_id))
        if idx is None or len(idx) <= 0:
            return self.empty_feat_df
        return self.feat_df.iloc[idx].reset_index(drop=True)

    def _ensure_train_feat_cache(self) -> None:
        if len(self._train_feat_df) > 0 or self._train_feat_group_indices:
            return
        if not self.scene_train_feat_path.exists():
            self._train_feat_df = self.empty_feat_df.iloc[0:0].copy()
            return
        train_df = pd.read_parquet(self.scene_train_feat_path)
        train_df[self.group_key] = train_df[self.group_key].astype(np.int64)
        train_df["note_idx"] = train_df["note_idx"].astype(np.int64)
        train_df = _shrink_numeric_dtypes(train_df)
        self._train_feat_df = train_df
        self._train_feat_group_indices = {
            int(k): np.asarray(v, dtype=np.int32)
            for k, v in train_df.groupby(self.group_key, sort=False).indices.items()
        }

    def _get_train_feat_req(self, request_id: int) -> pd.DataFrame:
        self._ensure_train_feat_cache()
        idx = self._train_feat_group_indices.get(int(request_id))
        if idx is None or len(idx) <= 0:
            return self.empty_feat_df
        return self._train_feat_df.iloc[idx].reset_index(drop=True)

    def _get_split_feat_req(self, split: str, request_id: int) -> pd.DataFrame:
        return self.get_feat_req(request_id) if str(split) == "test" else self._get_train_feat_req(request_id)

    def _load_models(self) -> None:
        # 各阶段模型在状态初始化时加载，service 仅负责编排调用
        self.gbdt_predictor = GBDTPredictor(
            scene=self.scene,
            tag=self.tag,
            deploy_tag_dir=self.deploy_tag_dir,
            group_key=self.group_key,
            feat_columns=list(self.feat_df.columns),
        )
        self.dien_predictor = DIENPredictor(
            scene=self.scene,
            tag=self.tag,
            deploy_tag_dir=self.deploy_tag_dir,
            group_key=self.group_key,
        )
        self.dssm_recaller = DSSMRecaller(
            scene=self.scene,
            tag=self.tag,
            deploy_tag_dir=self.deploy_tag_dir,
            group_key=self.group_key,
        )

        self.lgb_model = self.gbdt_predictor.lgb_model
        self.xgb_model = self.gbdt_predictor.xgb_model
        self.gbdt_feature_cols = self.gbdt_predictor.feature_cols
        self.dien_model = self.dien_predictor.model
        self.dssm_model = self.dssm_recaller.model
        self.realtime_ann_enabled = bool(self.dssm_recaller.enabled)


    def readiness(self) -> dict[str, Any]:
        return {
            "scene": self.scene,
            "scene_test": self.scene_test_path.exists(),
            "scene_test_features": self.scene_feat_path.exists(),
            "user_feat": self.user_feat_path.exists(),
            "redis_cache": self.realtime_cache is not None,
            "lgb_loaded": self.lgb_model is not None,
            "xgb_loaded": self.xgb_model is not None,
            "dien_loaded": self.dien_model is not None,
            "dssm_user_tower_loaded": self.dssm_model is not None,
            "realtime_ann_enabled": bool(self.realtime_ann_enabled),
            "rows": {
                "requests": int(len(self.req_df)),
                "feature_rows": int(len(self.feat_df)),
                "users": int(self.user_count),
            },
        }

    def resolve_request(self, user_idx: int, query: str | None) -> tuple[int, str]:
        reqs = []
        if self.realtime_cache is not None:
            reqs = self.realtime_cache.get_user_requests(int(user_idx), self.scene) or []
        if not reqs:
            reqs = self.user_requests.get(int(user_idx), [])
        if not reqs:
            if len(self.req_df) == 0:
                raise KeyError("no requests available for this scene")
            rid = int(self.req_df.iloc[0][self.group_key])
            return rid, ""

        if self.scene == "rec":
            rid = int(reqs[-1])
            return rid, ""

        if not query or not query.strip():
            rid = int(reqs[-1])
            return rid, self.search_query_map.get(rid, "")

        query_ctx = self.preprocess_search_query(query, user_idx=int(user_idx))
        q = _normalize_search_query_text(str(query_ctx.get("resolved_query") or query or ""))
        q_terms = [str(x) for x in (query_ctx.get("terms") or []) if str(x).strip()] or _segment_query_terms(q, self.search_query_resources)

        def _request_match_score(query_norm: str, query_terms: list[str], target_query: str) -> float:
            target_norm = _normalize_search_query_text(target_query)
            score = _text_match_score(query_norm, target_norm, target_norm)
            if query_terms:
                target_terms = set(_segment_query_terms(target_norm, self.search_query_resources))
                overlap = len(set(query_terms).intersection(target_terms)) / max(len(query_terms), 1)
                score += 1.05 * float(overlap)
            return float(score)

        best_rid = int(reqs[-1])
        best_score = -1e9
        for rid in reqs:
            sq = self.search_query_map.get(int(rid), "")
            score = _request_match_score(q, q_terms, sq)
            if score > best_score:
                best_score = score
                best_rid = int(rid)

        if best_score <= 0 and len(self.search_query_map) > 0:
            for rid, sq in self.search_query_map.items():
                score = _request_match_score(q, q_terms, sq)
                if score > best_score:
                    best_score = score
                    best_rid = int(rid)

        return best_rid, self.search_query_map.get(best_rid, "")

    def _fetch_recall_candidates(self, request_id: int, max_rank: int) -> pd.DataFrame:
        return fetch_recall_candidates(
            request_id=int(request_id),
            max_rank=int(max_rank),
            group_key=self.group_key,
            req_df=self.req_df,
            get_feat_req=self.get_feat_req,
            dssm_recaller=self.dssm_recaller,
            recall_test_path=self.recall_test_path,
        )

    def _fetch_notes(self, note_ids: list[int]) -> pd.DataFrame:
        if not note_ids:
            return pd.DataFrame(columns=[
                "note_idx", "note_title", "note_content", "image_path",
                "accum_like_num", "accum_collect_num", "accum_comment_num",
                "taxonomy1_id", "taxonomy2_id", "taxonomy3_id",
            ])
        uniq_ids = sorted(set(int(x) for x in note_ids))
        with self._note_cache_lock:
            missing_ids = [x for x in uniq_ids if x not in self._note_meta_cache]

        if missing_ids:
            ids_df = pd.DataFrame({"note_idx": missing_ids})
            con = duckdb.connect(database=":memory:")
            try:
                con.register("ids", ids_df)
                q = """
                SELECT n.note_idx, n.note_title, n.note_content, n.image_path,
                       n.accum_like_num, n.accum_collect_num, n.accum_comment_num,
                       n.taxonomy1_id, n.taxonomy2_id, n.taxonomy3_id
                FROM read_parquet(?) n
                INNER JOIN ids USING(note_idx)
                """
                df_new = con.execute(q, [self.notes_glob]).df()
            finally:
                con.close()

            if "image_path" in df_new.columns:
                df_new["image_path"] = df_new["image_path"].apply(_to_path_list)
            new_rows = {
                int(r["note_idx"]): r
                for r in df_new.to_dict("records")
            }
            with self._note_cache_lock:
                self._note_meta_cache.update(new_rows)
                if len(self._note_meta_cache) > 200_000:
                    self._note_meta_cache.clear()
                    self._note_meta_cache.update(new_rows)

        with self._note_cache_lock:
            rows = [self._note_meta_cache[x] for x in uniq_ids if x in self._note_meta_cache]
        return pd.DataFrame(rows)

    def _build_recent_behaviors(self, user_idx: int, max_len: int = 20) -> list[dict[str, Any]]:
        uid = int(user_idx)
        if self.realtime_cache is not None:
            rows = self.realtime_cache.get_recent_behaviors(uid, self.scene, max_len=max_len)
            if rows:
                note_ids = [int(x.get("note_idx", -1)) for x in rows if _safe_intish(x.get("note_idx", -1), -1) >= 0]
                note_df = self._fetch_notes(note_ids)
                note_map = {int(r["note_idx"]): r for r in note_df.to_dict("records")} if len(note_df) > 0 else {}
                out: list[dict[str, Any]] = []
                for row in rows:
                    note_idx = _safe_intish(row.get("note_idx", -1), -1)
                    meta = note_map.get(note_idx, {})
                    out.append(
                        {
                            **row,
                            "scene": self.scene,
                            "title": _format_note_title(meta.get("note_title"), meta.get("note_content") or row.get("content") or row.get("title")),
                            "tag_ids": [
                                _safe_intish(meta.get("taxonomy1_id"), -1),
                                _safe_intish(meta.get("taxonomy2_id"), -1),
                                _safe_intish(meta.get("taxonomy3_id"), -1),
                            ],
                        }
                    )
                return out

        reqs = self.user_requests.get(uid, [])
        if not reqs:
            return []
        rows: list[dict[str, Any]] = []
        note_ids: list[int] = []
        for idx, rid in enumerate(reqs[-max_len:][::-1]):
            feat_req = self.get_feat_req(int(rid))
            if len(feat_req) <= 0:
                continue
            sub = feat_req.sort_values(["click", "y_multi"], ascending=[False, False], kind="mergesort").head(2)
            for _, row in sub.iterrows():
                note_idx = _safe_intish(row.get("note_idx", -1), -1)
                if note_idx < 0:
                    continue
                note_ids.append(note_idx)
                rows.append(
                    {
                        "ts": None,
                        "scene": self.scene,
                        "action": "history",
                        "note_idx": int(note_idx),
                        "request_id": int(rid),
                        "query": self.search_query_map.get(int(rid), "") if self.scene == "search" else "",
                        "interaction_score": float(pd.to_numeric(row.get("y_multi", 0.0), errors="coerce") or 0.0),
                    }
                )
                if len(rows) >= max_len:
                    break
            if len(rows) >= max_len:
                break
        note_df = self._fetch_notes(note_ids)
        note_map = {int(r["note_idx"]): r for r in note_df.to_dict("records")} if len(note_df) > 0 else {}
        out: list[dict[str, Any]] = []
        for row in rows[:max_len]:
            meta = note_map.get(int(row["note_idx"]), {})
            out.append(
                {
                    **row,
                    "title": _format_note_title(meta.get("note_title"), meta.get("note_content")),
                    "tag_ids": [
                        _safe_intish(meta.get("taxonomy1_id"), -1),
                        _safe_intish(meta.get("taxonomy2_id"), -1),
                        _safe_intish(meta.get("taxonomy3_id"), -1),
                    ],
                }
            )
        return out

    def predict_gbdt(self, cand: pd.DataFrame) -> np.ndarray:
        return self.gbdt_predictor.predict(cand)

    def predict_dien(
        self,
        cand: pd.DataFrame,
        batch_size: int = 512,
        history_note_ids: list[int] | None = None,
    ) -> np.ndarray:
        return self.dien_predictor.predict(
            cand,
            batch_size=batch_size,
            history_note_ids=history_note_ids,
        )

    def build_feed(self, user_idx: int, query: str, page: int, page_size: int, refresh_key: str = "", exclude_note_ids: list[int] | None = None) -> dict[str, Any]:
        return OnlineScenePipeline(self, app_state=None).build_feed(
            user_idx=int(user_idx),
            query=query,
            page=int(page),
            page_size=int(page_size),
            refresh_key=str(refresh_key or ""),
            exclude_note_ids=exclude_note_ids,
        )

    def get_note_detail(
        self,
        user_idx: int,
        request_id: int,
        note_idx: int,
        query: str = "",
        meta_only: bool = False,
    ) -> dict[str, Any]:
        feed = {"items": []} if bool(meta_only) else self.build_feed(user_idx=user_idx, query=str(query or ""), page=1, page_size=200)
        target = next((x for x in feed["items"] if int(x["note_idx"]) == int(note_idx) and int(x["request_id"]) == int(request_id)), None)

        note_meta = self._fetch_notes([note_idx])
        if len(note_meta) == 0:
            raise KeyError(f"note_idx not found: {note_idx}")
        n = note_meta.iloc[0].to_dict()

        feat_req = self.get_feat_req(int(request_id))
        row = feat_req[feat_req["note_idx"] == int(note_idx)]
        label = {"y_multi": 0.0, "click": 0}
        if len(row) > 0:
            label = {
                "y_multi": float(row.iloc[0].get("y_multi", 0.0)),
                "click": _safe_intish(row.iloc[0].get("click", 0.0)),
            }

        if target is None and not bool(meta_only):
            one = row.copy() if len(row) > 0 else pd.DataFrame([{self.group_key: request_id, "note_idx": note_idx, "user_idx": user_idx}])
            one = one.merge(note_meta, on="note_idx", how="left")
            one["dssm_score"] = 0.0
            one["gbdt_score"] = self.predict_gbdt(one)
            one["dien_score"] = self.predict_dien(one)
            one["dien_score"] = np.nan_to_num(one["dien_score"], nan=0.0, posinf=1e6, neginf=-1e6)
            scores = {
                "dssm": float(one.iloc[0].get("dssm_score", 0.0)),
                "gbdt": float(one.iloc[0].get("gbdt_score", 0.0)),
                "dien": float(one.iloc[0].get("dien_score", 0.0)),
            }
            stage_ranks: dict[str, int | None] = {"recall": None, "preranking": None, "ranking": None, "rerank": None}
        elif target is None:
            scores = {"dssm": 0.0, "gbdt": 0.0, "dien": 0.0}
            stage_ranks = {"recall": None, "preranking": None, "ranking": None, "rerank": None}
        else:
            scores = target["scores"]
            stage_ranks = {
                "recall": _safe_intish(target.get("stage_ranks", {}).get("recall"), 0) or None,
                "preranking": _safe_intish(target.get("stage_ranks", {}).get("preranking"), 0) or None,
                "ranking": _safe_intish(target.get("stage_ranks", {}).get("ranking"), 0) or None,
                "rerank": _safe_intish(target.get("stage_ranks", {}).get("rerank"), 0) or None,
            }

        return {
            "scene": self.scene,
            "user_idx": int(user_idx),
            "request_id": int(request_id),
            "note_idx": int(note_idx),
            "title": _format_note_title(n.get("note_title"), n.get("note_content")),
            "content": str(n.get("note_content") or ""),
            "images": _existing_images(_to_path_list(n.get("image_path"))),
            "accum_like_num": _safe_intish(n.get("accum_like_num", 0)),
            "accum_collect_num": _safe_intish(n.get("accum_collect_num", 0)),
            "accum_comment_num": _safe_intish(n.get("accum_comment_num", 0)),
            "scores": {
                "dssm": float(np.nan_to_num(scores.get("dssm", 0.0), nan=0.0, posinf=1e6, neginf=-1e6)),
                "gbdt": float(np.nan_to_num(scores.get("gbdt", 0.0), nan=0.0, posinf=1e6, neginf=-1e6)),
                "dien": float(np.nan_to_num(scores.get("dien", 0.0), nan=0.0, posinf=1e6, neginf=-1e6)),
            },
            "stage_top500_ranks": stage_ranks,
            "stage_rank_note": "阶段 Rank 对应当前请求的召回、粗排、精排与去重后位次。",
            "labels": label,
        }

    def get_user(self, user_idx: int) -> dict[str, Any]:
        uid = int(user_idx)
        info = None
        if self.realtime_cache is not None:
            info = self.realtime_cache.get_user_profile(uid)
        if info is None:
            info = self.user_feat_map.get(uid)
        if info is None:
            raise KeyError(f"user_idx not found: {uid}")
        reqs = []
        if self.realtime_cache is not None:
            reqs = self.realtime_cache.get_user_requests(uid, self.scene) or []
        if not reqs:
            reqs = self.user_requests.get(uid, [])
        recent = []
        for rid in reqs[-30:][::-1]:
            row = {"request_id": int(rid)}
            if self.scene == "search":
                row["query"] = self.search_query_map.get(int(rid), "")
            recent.append(row)
        recent_behaviors = self._build_recent_behaviors(uid, max_len=20)
        tag_counts: dict[int, int] = {}
        keyword_counts: dict[str, int] = {}
        for row in recent_behaviors:
            for tag_id in row.get("tag_ids", []):
                if int(tag_id) >= 0:
                    tag_counts[int(tag_id)] = tag_counts.get(int(tag_id), 0) + 1
            for term in _extract_terms(str(row.get("query") or ""), max_terms=4):
                keyword_counts[term] = keyword_counts.get(term, 0) + 1
        return {
            "scene": self.scene,
            "user_idx": uid,
            "features": info,
            "request_count_in_test": int(len(reqs)),
            "recent_requests": recent,
            "recent_behaviors": self._build_recent_behaviors(uid, max_len=40),
            "interest_tags": [k for k, _ in sorted(tag_counts.items(), key=lambda x: (-x[1], x[0]))[:8]],
            "recent_search_keywords": [k for k, _ in sorted(keyword_counts.items(), key=lambda x: (-x[1], x[0]))[:6]],
        }

    def list_users(self, limit: int = 20, offset: int = 0, random_show: bool = False) -> list[dict[str, Any]]:
        n = max(1, min(200, int(limit)))
        off = max(0, int(offset))
        user_ids = self.available_user_ids
        if not user_ids:
            return []

        if random_show:
            perm = np.random.default_rng().permutation(len(user_ids))
            picked = [user_ids[int(i)] for i in perm[:n]]
        else:
            picked = user_ids[off : off + n]

        rows: list[dict[str, Any]] = []
        for uid in picked:
            info = self.user_feat_map.get(int(uid))
            if info is None:
                continue
            reqs = self.user_requests.get(int(uid), [])
            rows.append(
                {
                    "user_idx": int(uid),
                    "gender": info.get("gender"),
                    "age": info.get("age"),
                    "platform": info.get("platform"),
                    "location": info.get("location"),
                    "fans_num": _safe_intish(info.get("fans_num", 0)),
                    "follows_num": _safe_intish(info.get("follows_num", 0)),
                    "request_count_in_test": int(len(reqs)),
                }
            )
        return rows

    def get_prewarm_seed(self) -> dict[str, Any]:
        if len(self.req_df) <= 0:
            return {"user_idx": 0, "request_id": -1, "query": ""}
        uid = -1
        req_id = -1
        best_cnt = -1
        for cand_uid, reqs in self.user_requests.items():
            cnt = len(reqs)
            if cnt <= 0:
                continue
            if cnt > best_cnt:
                best_cnt = cnt
                uid = int(cand_uid)
                req_id = int(reqs[-1])
        if uid < 0 or req_id < 0:
            row = self.req_df.iloc[0]
            uid = _safe_intish(row.get("user_idx", 0), 0)
            req_id = _safe_intish(row.get(self.group_key, -1), -1)
        query = self.search_query_map.get(int(req_id), "") if self.scene == "search" else ""
        return {
            "user_idx": int(uid),
            "request_id": int(req_id),
            "query": str(query or ""),
        }

    def get_user_history_notes(self, user_idx: int, feat_req: pd.DataFrame, max_len: int = 20, prefer_realtime: bool = True) -> list[int]:
        if prefer_realtime and self.realtime_cache is not None:
            history = self.realtime_cache.get_user_history_notes(int(user_idx), self.scene, max_len=max_len)
            if history:
                return history
        if "recent_clicked_note_idxs" in feat_req.columns and len(feat_req) > 0:
            raw = feat_req.iloc[0].get("recent_clicked_note_idxs", [])
            if isinstance(raw, np.ndarray):
                return [int(x) for x in raw.tolist()][:max_len]
            if isinstance(raw, list):
                return [int(x) for x in raw][:max_len]
        return []

    def build_request_linkage_context(self, feat_req: pd.DataFrame, query: str = "", max_hist: int = 20) -> dict[str, Any]:
        history_note_ids = self.get_user_history_notes(
            user_idx=_safe_intish(feat_req.iloc[0].get("user_idx", 0), 0) if len(feat_req) > 0 else 0,
            feat_req=feat_req,
            max_len=max_hist,
            prefer_realtime=False,
        )
        tag_counts: dict[int, float] = {}
        if history_note_ids:
            hist_notes = self._fetch_notes(history_note_ids[:max_hist])
            if len(hist_notes) > 0:
                for _, row in hist_notes.iterrows():
                    for col in ("taxonomy1_id", "taxonomy2_id", "taxonomy3_id"):
                        tag_id = _safe_intish(row.get(col), -1)
                        if tag_id >= 0:
                            tag_counts[tag_id] = tag_counts.get(tag_id, 0.0) + 1.0
        return {
            "keywords": (
                self.preprocess_search_query(
                    str(query or ""),
                    user_idx=_safe_intish(feat_req.iloc[0].get("user_idx", 0), 0) if len(feat_req) > 0 else None,
                ).get("core_terms", [])
                if self.scene == "search"
                else []
            ),
            "tag_ids": [k for k, _ in sorted(tag_counts.items(), key=lambda x: (-x[1], x[0]))[:8]],
        }

    def build_cross_scene_recent_behaviors(self, user_idx: int, max_len: int = 20) -> list[dict[str, Any]]:
        uid = int(user_idx)
        app_state = getattr(self, "_app_state", None)
        if app_state is not None:
            try:
                rows = app_state._get_cross_scene_behaviors(uid, max_len=max_len, loaded_only=False)
                if rows:
                    note_ids = [
                        int(x.get("note_idx", -1))
                        for x in rows
                        if _safe_intish(x.get("note_idx", -1), -1) >= 0
                    ]
                    note_df = self._fetch_notes(note_ids)
                    note_map = {int(r["note_idx"]): r for r in note_df.to_dict("records")} if len(note_df) > 0 else {}
                    out: list[dict[str, Any]] = []
                    for row in rows[: max(24, int(max_len) * 4)]:
                        note_idx = _safe_intish(row.get("note_idx", -1), -1)
                        meta = note_map.get(note_idx, {})
                        out.append(
                            {
                                **row,
                                "title": _format_note_title(
                                    meta.get("note_title"),
                                    meta.get("note_content") or row.get("content") or row.get("title"),
                                ),
                            }
                        )
                    rows = out
                else:
                    rows = []
            except Exception:
                rows = []
            if rows:
                def _dedup_cross(rows_in: list[dict[str, Any]]) -> list[dict[str, Any]]:
                    merged = sorted(rows_in, key=lambda x: int(x.get("ts", 0) or 0), reverse=True)
                    out: list[dict[str, Any]] = []
                    seen: set[str] = set()
                    for row in merged:
                        key = (
                            f"{row.get('scene')}|{row.get('request_id')}|{row.get('note_idx')}|"
                            f"{row.get('query', '')}|{row.get('title', '')}"
                        )
                        if key in seen:
                            continue
                        seen.add(key)
                        out.append(row)
                        if len(out) >= int(max_len):
                            break
                    return out
                return _dedup_cross(rows)
        def _dedup_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
            out: list[dict[str, Any]] = []
            seen: set[str] = set()
            for row in rows:
                key = (
                    f"{row.get('scene')}|{row.get('request_id')}|{row.get('note_idx')}|"
                    f"{row.get('query', '')}|{row.get('title', '')}"
                )
                if key in seen:
                    continue
                seen.add(key)
                out.append(row)
                if len(out) >= int(max_len):
                    break
            return out
        if self.realtime_cache is not None:
            rows = self.realtime_cache.get_recent_behaviors_all(uid, max_len=max(20, int(max_len) * 2))
            if rows:
                note_ids = [int(x.get("note_idx", -1)) for x in rows if _safe_intish(x.get("note_idx", -1), -1) >= 0]
                note_df = self._fetch_notes(note_ids)
                note_map = {int(r["note_idx"]): r for r in note_df.to_dict("records")} if len(note_df) > 0 else {}
                out: list[dict[str, Any]] = []
                for row in rows[: max(24, int(max_len) * 4)]:
                    note_idx = _safe_intish(row.get("note_idx", -1), -1)
                    meta = note_map.get(note_idx, {})
                    out.append(
                        {
                            **row,
                            "title": _format_note_title(meta.get("note_title"), meta.get("note_content") or row.get("content") or row.get("title")),
                        }
                    )
                return _dedup_rows(out)
        return _dedup_rows(self._build_recent_behaviors(uid, max_len=max_len))

    def record_click(
        self,
        user_idx: int,
        note_idx: int,
        request_id: int | None = None,
        query: str = "",
    ) -> bool:
        uid = int(user_idx)
        self._invalidate_user_serving_cache(uid)
        if self.realtime_cache is not None:
            self.realtime_cache.record_click(
                uid,
                self.scene,
                int(note_idx),
                request_id=(int(request_id) if request_id is not None else None),
                query=str(query or ""),
            )
            return True
        return False

    def record_view(
        self,
        user_idx: int,
        note_idx: int,
        request_id: int | None = None,
        query: str = "",
    ) -> bool:
        if self.realtime_cache is None:
            return False
        return bool(
            self.realtime_cache.record_view(
                int(user_idx),
                self.scene,
                int(note_idx),
                request_id=(int(request_id) if request_id is not None else None),
                query=str(query or ""),
            )
        )

    def record_engage(
        self,
        user_idx: int,
        note_idx: int,
        request_id: int | None = None,
        query: str = "",
        like: int = 0,
        collect: int = 0,
        comment: int = 0,
        share: int = 0,
        page_time: float = 0.0,
    ) -> bool:
        uid = int(user_idx)
        self._invalidate_user_serving_cache(uid)
        if self.realtime_cache is None:
            return False
        return bool(
            self.realtime_cache.record_engage(
                uid,
                self.scene,
                int(note_idx),
                request_id=(int(request_id) if request_id is not None else None),
                query=str(query or ""),
                like=max(0, int(like)),
                collect=max(0, int(collect)),
                comment=max(0, int(comment)),
                share=max(0, int(share)),
                page_time=max(0.0, float(page_time)),
            )
        )

    def delete_behavior(
        self,
        user_idx: int,
        note_idx: int,
        ts: int | None = None,
        request_id: int | None = None,
    ) -> bool:
        uid = int(user_idx)
        self._invalidate_user_serving_cache(uid)
        if self.realtime_cache is None:
            return False
        return bool(
            self.realtime_cache.delete_behavior(
                uid,
                self.scene,
                int(note_idx),
                ts=(int(ts) if ts is not None else None),
                request_id=(int(request_id) if request_id is not None else None),
            )
        )

    def delete_behaviors(
        self,
        user_idx: int,
        items: list[dict[str, int | None]],
    ) -> int:
        uid = int(user_idx)
        self._invalidate_user_serving_cache(uid)
        if self.realtime_cache is None:
            return 0
        deleted = 0
        for item in items or []:
            try:
                accepted = self.realtime_cache.delete_behavior(
                    uid,
                    self.scene,
                    int(item.get("note_idx", -1) or -1),
                    ts=(int(item["ts"]) if item.get("ts") is not None else None),
                    request_id=(int(item["request_id"]) if item.get("request_id") is not None else None),
                )
            except Exception:
                accepted = False
            deleted += int(bool(accepted))
        return int(deleted)

    def _invalidate_user_serving_cache(self, user_idx: int) -> None:
        uid = int(user_idx)
        with self._feed_cache_lock:
            stale_keys = [key for key in self._feed_cache if key[0] == uid]
            for key in stale_keys:
                self._feed_cache.pop(key, None)
        with self._stage_cache_lock:
            stale_keys = [key for key in self._stage_cache if key[0] == uid]
            for key in stale_keys:
                self._stage_cache.pop(key, None)

    def suggest_queries(self, query: str, limit: int = 8) -> list[dict[str, str]]:
        if self.scene != "search":
            return []
        ctx = self.preprocess_search_query(query)
        q = str(ctx.get("normalized_query") or "").strip()
        q_resolved = _normalize_search_query_text(str(ctx.get("resolved_query") or ""))
        q_terms = [str(x) for x in (ctx.get("terms") or []) if str(x).strip()]
        if len(q) < 1:
            return []

        query_score_map = dict(getattr(self.search_query_resources, "query_scores", {}) or {})
        ranked: list[tuple[float, float, str, str]] = []
        seen: set[str] = set()

        def _push(text: str, hint: str, score: float, base_score: float = 0.0) -> None:
            s = str(text or "").strip()
            if not s or s in seen:
                return
            seen.add(s)
            ranked.append((float(score), float(base_score), s, hint))

        if q_resolved and q_resolved != q:
            _push(q_resolved, "纠错建议", 20.0, float(query_score_map.get(q_resolved, 0.0)))

        input_terms = set(q_terms)
        for cand, cnt in self.search_query_catalog:
            s = str(cand or "").strip()
            sl = _normalize_search_query_text(s)
            if not s or s in seen:
                continue
            starts = sl.startswith(q) or (q_resolved and sl.startswith(q_resolved))
            contains = (q in sl) or (q_resolved and q_resolved in sl)
            cand_terms = set(_segment_query_terms(sl, self.search_query_resources))
            overlap = len(input_terms.intersection(cand_terms)) / max(len(input_terms), 1) if input_terms else 0.0
            if not starts and not contains and overlap <= 0.0:
                continue
            exact = float(sl == q or (q_resolved and sl == q_resolved))
            prefix = float(starts)
            contains_score = float(contains)
            pop = np.log1p(float(cnt))
            length_penalty = 0.05 * abs(len(sl) - len(q_resolved or q))
            score = exact * 18.0 + prefix * 9.0 + contains_score * 2.4 + overlap * 5.8 + pop * 0.45 - length_penalty
            if q_resolved and sl == q_resolved:
                hint = "纠错建议"
            elif prefix > 0:
                hint = "热门补全"
            else:
                hint = "相关搜索"
            _push(s, hint, score, float(cnt))

        ranked.sort(key=lambda x: (-x[0], -x[1], len(x[2]), x[2]))
        return [{"text": text, "hint": hint} for _score, _base, text, hint in ranked[: int(limit)]]

    def _load_recall_metrics(self, use_test_split: bool = True) -> dict[str, Any]:
        split = "test" if use_test_split else "train"
        recall_path = OUT_DATA_DIR / f"recall_{self.scene}_{split}_{self.tag}_multiroute_top1000.parquet"
        if not recall_path.exists():
            return {}
        recall_df = _read_parquet_cached(str(recall_path), (self.group_key, "note_idx", "rank", "recall_score"))
        recall_df = recall_df.merge(
            self.feat_df[[self.group_key, "note_idx", "y_multi"]],
            on=[self.group_key, "note_idx"],
            how="inner",
        )
        if len(recall_df) == 0:
            return {}
        recall_df = recall_df.sort_values([self.group_key, "rank"], ascending=[True, True], kind="mergesort")
        y_disc = self.discretize_relevance(recall_df["y_multi"].to_numpy())
        groups = recall_df[self.group_key].to_numpy()
        recall_at_10 = float(recall_df.groupby(self.group_key, sort=False).head(10).groupby(self.group_key)["y_multi"].max().gt(0).mean())
        recall_at_100 = float(recall_df.groupby(self.group_key, sort=False).head(100).groupby(self.group_key)["y_multi"].max().gt(0).mean())
        return {
            "recall@10": recall_at_10,
            "recall@100": recall_at_100,
            "ndcg@10": float(self.eval_ndcg_by_group(y_disc, recall_df["recall_score"].to_numpy(), groups, 10)),
            "ndcg@50": float(self.eval_ndcg_by_group(y_disc, recall_df["recall_score"].to_numpy(), groups, 50)),
            "rows": int(len(recall_df)),
            "groups": int(recall_df[self.group_key].nunique()),
            "split": split,
        }

    def _load_preranking_metrics(self) -> tuple[pd.DataFrame | None, dict[str, Any]]:
        path = OUT_DATA_DIR / f"preranking_{self.scene}_{self.tag}_train_scored_full.parquet"
        if not path.exists():
            return None, {}
        df = _read_parquet_cached(str(path), (self.group_key, "note_idx", "y_multi", "preranking_score", "lgb_score", "xgb_score"))
        if self.group_key not in df.columns or "note_idx" not in df.columns or "y_multi" not in df.columns:
            return None, {}
        y_disc = self.discretize_relevance(df["y_multi"].to_numpy())
        groups = df[self.group_key].to_numpy()
        score = pd.to_numeric(df.get("preranking_score", 0.0), errors="coerce").fillna(0.0).to_numpy(np.float32)
        base = pd.to_numeric(df.get("lgb_score", 0.0), errors="coerce").fillna(0.0).to_numpy(np.float32)
        metrics = {
            "ndcg@10": float(self.eval_ndcg_by_group(y_disc, score, groups, 10)),
            "ndcg@50": float(self.eval_ndcg_by_group(y_disc, score, groups, 50)),
            "linkage_delta_ndcg@10": float(
                self.eval_ndcg_by_group(y_disc, score, groups, 10)
                - self.eval_ndcg_by_group(y_disc, base, groups, 10)
            ),
            "p50_labeled_candidates_per_group": float(
                df.groupby(self.group_key)["y_multi"].apply(lambda s: float((np.asarray(s) > 0).sum())).median()
            ),
            "evaluated_groups": int(df[self.group_key].nunique()),
        }
        return df, metrics

    def _load_dien_eval_frame(self, full: bool = False) -> pd.DataFrame | None:
        scored_path = OUT_DATA_DIR / f"dien_{self.scene}_{self.tag}_train_scored_full.parquet"
        path = scored_path if scored_path.exists() else OUT_DATA_DIR / f"dien_{self.scene}_{self.tag}_train_from_gbdt_top500.parquet"
        if not path.exists():
            return None
        if full:
            df = _read_parquet_cached(str(path))
        else:
            df = _read_parquet_cached(str(path))
        if self.group_key not in df.columns or "note_idx" not in df.columns or "y_multi" not in df.columns:
            return None
        return df

    def _sample_groups_random(self, group_ids: list[int], sample_n: int, seed: int) -> list[int]:
        vals = [int(x) for x in group_ids]
        if len(vals) <= int(sample_n):
            return vals
        rng = np.random.default_rng(seed)
        picked = rng.choice(np.asarray(vals, dtype=np.int64), size=int(sample_n), replace=False)
        return [int(x) for x in picked.tolist()]

    def _build_split_recall_frame(self, split: str, sample_n: int, gids: list[int] | None = None) -> pd.DataFrame:
        split = "test" if str(split) == "test" else "train"
        cache_key = max(1, int(sample_n))
        if gids is not None:
            cache_key = -1
        lock = self._test_recall_cache_lock if split == "test" else self._validation_recall_cache_lock
        cache = self._test_recall_cache if split == "test" else self._validation_recall_cache
        now = time.monotonic()
        if gids is None:
            with lock:
                cached = cache.get(cache_key)
                if cached is not None and (now - float(cached[0])) <= self._test_rank_cache_ttl_sec:
                    return cached[1].copy()

        req_df = self.req_df if split == "test" else self._get_split_req_df("train")
        candidate_groups = req_df[self.group_key].drop_duplicates().astype(int).tolist() if len(req_df) > 0 else []
        gids = (
            [int(x) for x in gids]
            if gids is not None
            else self._sample_groups_random(
                candidate_groups,
                sample_n=cache_key,
                seed=TEST_SAMPLE_SEED if split == "test" else 42,
            )
        )
        if not gids:
            return pd.DataFrame()
        get_feat_req = self.get_feat_req if split == "test" else self._get_train_feat_req
        precomputed_path = self.recall_test_path if split == "test" else None
        keep_cols = [
            self.group_key,
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
        chunks: list[pd.DataFrame] = []
        batch_size = 100 if split == "train" else 160
        for start in range(0, len(gids), batch_size):
            batch_gids = gids[start : start + batch_size]
            req_slice = req_df[req_df[self.group_key].isin(batch_gids)].copy()
            parts: list[pd.DataFrame] = []
            for gid in batch_gids:
                recall_cand = fetch_recall_candidates(
                    request_id=int(gid),
                    max_rank=int(self.recall_rank_cap),
                    group_key=self.group_key,
                    req_df=req_slice,
                    get_feat_req=get_feat_req,
                    dssm_recaller=self.dssm_recaller,
                    recall_test_path=precomputed_path,
                )
                if len(recall_cand) <= 0:
                    continue
                use_cols = [c for c in keep_cols if c in recall_cand.columns]
                parts.append(recall_cand[use_cols].copy())
            if parts:
                chunks.append(pd.concat(parts, ignore_index=True))
            del req_slice, parts
            gc.collect()
        out = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=keep_cols)
        if gids is None:
            with lock:
                cache[cache_key] = (time.monotonic(), out.copy())
                if len(cache) > self._test_metrics_cache_max:
                    cache.clear()
                    cache[cache_key] = (time.monotonic(), out.copy())
        return out

    def _build_test_recall_frame(self, sample_n: int) -> pd.DataFrame:
        return self._build_split_recall_frame("test", sample_n)

    def _compute_recall_split_metrics(
        self,
        split: str,
        sample_n: int,
        use_online_test_recall: bool = False,
        gids: list[int] | None = None,
    ) -> dict[str, Any]:
        recall_path = OUT_DATA_DIR / f"recall_{self.scene}_{split}_{self.tag}_multiroute_top1000.parquet"
        if split == "train":
            recall_df = self._build_split_recall_frame("train", sample_n=int(sample_n), gids=gids)
            if len(recall_df) <= 0 and recall_path.exists():
                recall_df = _read_parquet_cached(str(recall_path), (self.group_key, "note_idx", "rank", "recall_score"))
        elif split == "test" and bool(use_online_test_recall):
            recall_df = self._build_split_recall_frame("test", sample_n=int(sample_n), gids=gids)
        else:
            if not recall_path.exists():
                return {}
            recall_df = _read_parquet_cached(str(recall_path), (self.group_key, "note_idx", "rank", "recall_score"))
        if len(recall_df) <= 0:
            return {}
        req_df = self._get_split_req_df("test" if split == "test" else "train")
        candidate_gids = (
            req_df[self.group_key].drop_duplicates().astype(int).tolist()
            if len(req_df) > 0
            else recall_df[self.group_key].drop_duplicates().astype(int).tolist()
        )
        gids = (
            [int(x) for x in gids]
            if gids is not None
            else self._sample_groups_random(
                candidate_gids,
                sample_n=sample_n,
                seed=TEST_SAMPLE_SEED if str(split) == "test" else 42,
            )
        )
        scored = recall_df[recall_df[self.group_key].isin(gids)].copy()
        scored = scored.sort_values([self.group_key, "rank"], ascending=[True, True], kind="mergesort")
        full_labels = self._build_request_truth_frame("test" if split == "test" else "train", gids, topk=max(200, int(self.recall_rank_cap)))
        if len(full_labels) <= 0:
            return {}
        scored = scored.merge(
            full_labels[[self.group_key, "note_idx", "y_multi"]],
            on=[self.group_key, "note_idx"],
            how="left",
        )
        scored["y_multi"] = pd.to_numeric(scored.get("y_multi"), errors="coerce").fillna(0.0)
        pos_total_by_group = (
            full_labels.groupby(self.group_key)["y_multi"]
            .apply(lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0.0).to_numpy() > 0).sum()))
            .to_dict()
        )

        grouped = {int(gid): sub.copy() for gid, sub in scored.groupby(self.group_key, sort=False)}

        def _full_recall_at_k(k: int) -> float:
            vals: list[float] = []
            for gid in gids:
                total_pos = int(pos_total_by_group.get(int(gid), 0))
                if total_pos <= 0:
                    continue
                sub = grouped.get(int(gid))
                if sub is None or len(sub) <= 0:
                    vals.append(0.0)
                    continue
                hit_pos = int((pd.to_numeric(sub.head(k)["y_multi"], errors="coerce").fillna(0.0).to_numpy() > 0).sum())
                vals.append(float(min(hit_pos, total_pos) / total_pos))
            return float(np.mean(vals)) if vals else 0.0

        def _group_hit_rate_at_k(k: int) -> float:
            vals: list[float] = []
            for gid in gids:
                total_pos = int(pos_total_by_group.get(int(gid), 0))
                if total_pos <= 0:
                    continue
                sub = grouped.get(int(gid))
                if sub is None or len(sub) <= 0:
                    vals.append(0.0)
                    continue
                has_hit = bool((pd.to_numeric(sub.head(k)["y_multi"], errors="coerce").fillna(0.0).to_numpy() > 0).any())
                vals.append(1.0 if has_hit else 0.0)
            return float(np.mean(vals)) if vals else 0.0

        def _mrr_at_k(k: int) -> float:
            vals: list[float] = []
            for gid in gids:
                total_pos = int(pos_total_by_group.get(int(gid), 0))
                if total_pos <= 0:
                    continue
                sub = grouped.get(int(gid))
                if sub is None or len(sub) <= 0:
                    vals.append(0.0)
                    continue
                y = pd.to_numeric(sub.head(k)["y_multi"], errors="coerce").fillna(0.0).to_numpy()
                pos = np.flatnonzero(y > 0)
                vals.append(float(1.0 / float(pos[0] + 1)) if len(pos) > 0 else 0.0)
            return float(np.mean(vals)) if vals else 0.0

        first_hit_ranks: list[int] = []
        for gid in gids:
            total_pos = int(pos_total_by_group.get(int(gid), 0))
            if total_pos <= 0:
                continue
            sub = grouped.get(int(gid))
            if sub is None or len(sub) <= 0:
                continue
            cur = sub[["rank", "y_multi"]].copy()
            cur["y_multi"] = pd.to_numeric(cur["y_multi"], errors="coerce").fillna(0.0)
            cur = cur[cur["y_multi"] > 0]
            if len(cur) <= 0:
                continue
            first_hit_ranks.append(int(pd.to_numeric(cur["rank"], errors="coerce").fillna(1e9).min()))
        median_first_hit_rank = float(np.median(np.asarray(first_hit_ranks, dtype=np.float32))) if first_hit_ranks else -1.0

        recall_scored = scored[[self.group_key, "note_idx", "rank"]].copy() if len(scored) > 0 else pd.DataFrame()
        if len(recall_scored) > 0:
            recall_scored["recall_rank_score"] = -pd.to_numeric(recall_scored["rank"], errors="coerce").fillna(1e9)
        recall_eval = _merge_scored_with_labels(
            full_df=full_labels,
            scored_df=recall_scored,
            group_key=self.group_key,
            score_cols=["recall_rank_score"],
        )
        recall_eval = _fill_missing_eval_scores(recall_eval, "recall_rank_score", label_col="y_multi")
        sort_cols = [self.group_key, "recall_rank_score"] + (["result_rank"] if "result_rank" in recall_eval.columns else [])
        sort_asc = [True, False] + ([True] if "result_rank" in recall_eval.columns else [])
        recall_eval = recall_eval.sort_values(sort_cols, ascending=sort_asc, kind="mergesort")
        recall_y_disc = self.discretize_relevance(recall_eval["y_multi"].to_numpy())

        return {
            "HitRate@100": _group_hit_rate_at_k(100),
            "HitRate@500": _group_hit_rate_at_k(500),
            "Recall@100": _full_recall_at_k(100),
            "Recall@500": _full_recall_at_k(500),
            "Recall@300": _full_recall_at_k(300),
            "Recall@1000": _full_recall_at_k(1000),
            "MRR@100": _mrr_at_k(100),
            "MedianFirstHitRank": median_first_hit_rank,
            "Precision@100": _precision_at_k(scored, self.group_key, 100),
            "MAP": _map_at_k(scored, self.group_key, 100),
            "NDCG@10": float(
                self.eval_ndcg_by_group(
                    recall_y_disc,
                    recall_eval["recall_rank_score"].to_numpy(),
                    recall_eval[self.group_key].to_numpy(),
                    10,
                )
            ),
            "sampled_groups": int(len(gids)),
        }

    def _compute_prerank_tail_hit_metric(
        self,
        recall_df: pd.DataFrame,
        prerank_df: pd.DataFrame,
        gids: list[int],
    ) -> dict[str, float]:
        if len(recall_df) <= 0 or len(prerank_df) <= 0 or not gids:
            return {}
        recall_slice = recall_df[recall_df[self.group_key].isin([int(x) for x in gids])][[self.group_key, "note_idx", "rank"]].copy()
        prerank_slice = prerank_df[prerank_df[self.group_key].isin([int(x) for x in gids])][[self.group_key, "note_idx", "preranking_rank", "y_multi"]].copy()
        if len(recall_slice) <= 0 or len(prerank_slice) <= 0:
            return {}
        merged = prerank_slice.merge(recall_slice, on=[self.group_key, "note_idx"], how="left")
        merged["rank"] = pd.to_numeric(merged["rank"], errors="coerce").fillna(1e9)
        merged["preranking_rank"] = pd.to_numeric(merged["preranking_rank"], errors="coerce").fillna(1e9)
        merged["y_multi"] = pd.to_numeric(merged["y_multi"], errors="coerce").fillna(0.0)
        tail = merged[
            (merged["rank"] >= 101)
            & (merged["rank"] <= 500)
            & (merged["preranking_rank"] <= 100)
        ].copy()
        if len(tail) <= 0:
            return {"TailHitRate101_500@Pre100": 0.0}
        hit_groups = (
            tail.groupby(self.group_key)["y_multi"]
            .apply(lambda s: float((pd.to_numeric(s, errors="coerce").fillna(0.0).to_numpy() > 0).any()))
            .to_dict()
        )
        hit_vals: list[float] = []
        for gid in gids:
            hit_vals.append(float(hit_groups.get(int(gid), 0.0)))
        return {"TailHitRate101_500@Pre100": float(np.mean(hit_vals)) if hit_vals else 0.0}

    def _metric_bundle_from_scored(
        self,
        df: pd.DataFrame,
        score_cols: str | list[str],
        expected_groups: list[int] | None = None,
        label_source: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        sampled_groups = int(len(expected_groups)) if expected_groups is not None else int(df[self.group_key].nunique()) if len(df) > 0 else 0
        if len(df) <= 0:
            return {"NDCG@10": 0.0, "AUC": 0.5, "GAUC": 0.5, "sampled_groups": sampled_groups, "effective_groups": 0}
        score_candidates = [str(score_cols)] if isinstance(score_cols, str) else [str(x) for x in score_cols]
        score_col = _resolve_score_column(df, score_candidates)
        if score_col is None:
            return {"NDCG@10": 0.0, "AUC": 0.5, "GAUC": 0.5, "sampled_groups": sampled_groups, "effective_groups": 0}
        scored = df.copy()
        effective_groups = (
            int(df[df[self.group_key].isin([int(x) for x in expected_groups])][self.group_key].nunique())
            if expected_groups is not None
            else int(scored[self.group_key].nunique()) if len(scored) > 0 else 0
        )
        if expected_groups is not None and label_source is not None and len(label_source) > 0:
            full_labels = label_source[label_source[self.group_key].isin([int(x) for x in expected_groups])].copy()
            scored = _merge_scored_with_labels(full_labels, scored, self.group_key, score_candidates)
            scored = _fill_missing_eval_scores(scored, score_col, label_col="y_multi")
        pos_groups = (
            scored.groupby(self.group_key)["y_multi"]
            .apply(lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0.0).to_numpy() > 0).sum()))
        )
        valid_groups = [int(gid) for gid, cnt in pos_groups.items() if int(cnt) > 0]
        if not valid_groups:
            return {"NDCG@10": 0.0, "AUC": 0.5, "GAUC": 0.5, "sampled_groups": sampled_groups, "effective_groups": 0}
        scored = scored[scored[self.group_key].isin(valid_groups)].copy()
        sort_cols = [self.group_key, score_col] + (["result_rank"] if "result_rank" in scored.columns else [])
        sort_asc = [True, False] + ([True] if "result_rank" in scored.columns else [])
        scored = scored.sort_values(sort_cols, ascending=sort_asc, kind="mergesort").copy()
        y_disc = self.discretize_relevance(scored["y_multi"].to_numpy())
        y_bin = (pd.to_numeric(scored["y_multi"], errors="coerce").fillna(0.0).to_numpy() > 0).astype(np.int8)
        score = pd.to_numeric(scored[score_col], errors="coerce").fillna(0.0).to_numpy()
        out = {
            "NDCG@10": float(self.eval_ndcg_by_group(y_disc, score, scored[self.group_key].to_numpy(), 10)),
            "AUC": float(_binary_auc(y_bin, score)),
            "GAUC": float(_group_auc(scored, "user_idx", "y_multi", score_col)) if "user_idx" in scored.columns else 0.5,
            "sampled_groups": sampled_groups,
            "effective_groups": effective_groups,
        }
        return out

    def _build_val_ranking_frames(
        self,
        sample_n: int,
        gids: list[int] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
        pre_df, _ = self._load_preranking_metrics()
        dien_df = self._load_dien_eval_frame(full=True)
        if pre_df is None or dien_df is None:
            return pd.DataFrame(), pd.DataFrame(), []
        merge_cols = [self.group_key, "note_idx"]
        prerank_score_col = _resolve_score_column(pre_df, ["preranking_score", "gbdt_score", "lgb_score", "xgb_score"])
        if prerank_score_col is None:
            return pd.DataFrame(), pd.DataFrame(), []
        merged = dien_df.merge(
            pre_df[merge_cols + [prerank_score_col]].rename(columns={prerank_score_col: "preranking_score"}),
            on=merge_cols,
            how="inner",
        )
        if len(merged) <= 0:
            return pd.DataFrame(), pd.DataFrame(), []
        merged_prerank_col = _resolve_score_column(
            merged,
            ["preranking_score", "preranking_score_y", "preranking_score_x", "gbdt_score", "lgb_score", "xgb_score"],
        )
        if merged_prerank_col is None:
            return pd.DataFrame(), pd.DataFrame(), []
        if merged_prerank_col != "preranking_score":
            merged["preranking_score"] = pd.to_numeric(merged.get(merged_prerank_col), errors="coerce").fillna(0.0)
        if gids is None:
            gids = self._sample_groups_random(
                merged[self.group_key].drop_duplicates().astype(int).tolist(),
                sample_n=sample_n,
                seed=42,
            )
        else:
            gids = [int(x) for x in gids]
        merged = merged[merged[self.group_key].isin(gids)].copy()
        npy_path = OUT_DIR / "results" / f"dien_{self.scene}_{self.tag}.npy"
        if npy_path.exists():
            try:
                arr = _load_npy_cached(str(npy_path))
                if len(arr) == len(dien_df):
                    dien_full = dien_df[[self.group_key, "note_idx"]].copy()
                    dien_full["dien_score"] = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=1e6, neginf=-1e6)
                    merged = merged.merge(dien_full, on=merge_cols, how="left")
            except Exception:
                pass
        if "dien_score" not in merged.columns:
            try:
                merged["dien_score"] = self.predict_dien(merged)
            except Exception:
                merged["dien_score"] = pd.to_numeric(merged.get("preranking_score", 0.0), errors="coerce").fillna(0.0)
        prerank_df = merged.copy()
        prerank_df = prerank_df.sort_values(
            [self.group_key, "preranking_score", "note_idx"],
            ascending=[True, False, True],
            kind="mergesort",
        ).copy()
        prerank_df["preranking_rank"] = prerank_df.groupby(self.group_key).cumcount().astype(np.int32) + 1
        rank_df = prerank_df.copy()
        rank_df = rank_df.sort_values(
            [self.group_key, "dien_score", "note_idx"],
            ascending=[True, False, True],
            kind="mergesort",
        ).copy()
        rank_df["ranking_rank"] = rank_df.groupby(self.group_key).cumcount().astype(np.int32) + 1
        return prerank_df, rank_df, gids

    def _build_test_ranking_frames(self, sample_n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        cache_key = max(1, int(sample_n))
        now = time.monotonic()
        with self._test_rank_cache_lock:
            cached = self._test_rank_cache.get(cache_key)
            if cached is not None and (now - float(cached[0])) <= self._test_rank_cache_ttl_sec:
                return cached[1].copy(), cached[2].copy()

        candidate_groups = self.req_df[self.group_key].drop_duplicates().astype(int).tolist() if len(self.req_df) > 0 else []
        gids = self._sample_groups_random(
            candidate_groups,
            sample_n=sample_n,
            seed=TEST_SAMPLE_SEED,
        )
        if not gids:
            return pd.DataFrame(), pd.DataFrame()
        req_query_map = self.search_query_map if self.scene == "search" else {}
        pre_parts: list[pd.DataFrame] = []
        rank_parts: list[pd.DataFrame] = []
        for gid in gids:
            feat_req = self.get_feat_req(int(gid))
            recall_cand = fetch_recall_candidates(
                request_id=int(gid),
                max_rank=int(self.recall_rank_cap),
                group_key=self.group_key,
                req_df=self.req_df,
                get_feat_req=self.get_feat_req,
                dssm_recaller=self.dssm_recaller,
                recall_test_path=self.recall_test_path,
            )
            if len(feat_req) <= 0 or len(recall_cand) <= 0:
                continue
            user_idx = _safe_intish(feat_req.iloc[0].get("user_idx", 0), 0)
            raw_query = req_query_map.get(int(gid), "")
            query_ctx = self.preprocess_search_query(raw_query, user_idx=int(user_idx)) if self.scene == "search" else {"resolved_query": raw_query, "terms": [], "intents": [], "service_query": raw_query}
            resolved_query = str(query_ctx.get("resolved_query") or raw_query)
            service_query = str(query_ctx.get("service_query") or " ".join([str(x) for x in (query_ctx.get("terms") or []) if str(x).strip()]) or resolved_query)
            linkage_ctx = self.build_request_linkage_context(feat_req=feat_req, query=resolved_query, max_hist=20)
            prerank = run_preranking(
                user_idx=int(user_idx),
                query=service_query,
                query_phrase=resolved_query,
                query_terms=[str(x) for x in (query_ctx.get("terms") or []) if str(x).strip()],
                query_intents=[str(x) for x in (query_ctx.get("intents") or []) if str(x).strip()],
                scene=self.scene,
                group_key=self.group_key,
                recall_cand=recall_cand,
                feat_req=feat_req,
                gbdt_topn=self.gbdt_topn,
                fetch_notes=self._fetch_notes,
                predict_gbdt=self.predict_gbdt,
                linkage_ctx=linkage_ctx,
            )
            if len(prerank) <= 0:
                continue
            prerank = prerank.reset_index(drop=True)
            prerank["preranking_rank"] = np.arange(1, len(prerank) + 1, dtype=np.int32)
            ranked, _ = run_ranking(
                cand=prerank,
                page=1,
                page_size=20,
                predict_dien=self.predict_dien,
                history_note_ids=self.get_user_history_notes(user_idx=int(user_idx), feat_req=feat_req, prefer_realtime=False),
            )
            ranked = ranked.reset_index(drop=True)
            ranked["ranking_rank"] = np.arange(1, len(ranked) + 1, dtype=np.int32)
            pre_parts.append(prerank.copy())
            rank_parts.append(ranked.copy())
        if not pre_parts or not rank_parts:
            return pd.DataFrame(), pd.DataFrame()
        pre_df = pd.concat(pre_parts, ignore_index=True)
        rank_df = pd.concat(rank_parts, ignore_index=True)
        with self._test_rank_cache_lock:
            self._test_rank_cache[cache_key] = (time.monotonic(), pre_df.copy(), rank_df.copy())
            if len(self._test_rank_cache) > self._test_metrics_cache_max:
                self._test_rank_cache.clear()
                self._test_rank_cache[cache_key] = (time.monotonic(), pre_df.copy(), rank_df.copy())
        return pre_df, rank_df

    def _build_validation_examples(
        self,
        prerank_df: pd.DataFrame,
        rank_df: pd.DataFrame,
        example_limit: int,
        context_user_idx: int | None,
        expected_groups: list[int] | None = None,
        split: str = "test",
    ) -> list[dict[str, Any]]:
        if len(prerank_df) <= 0 or len(rank_df) <= 0 or int(example_limit) <= 0:
            return []
        source = rank_df.copy()
        if "y_multi" not in source.columns and "y_multi" in prerank_df.columns:
            source = source.merge(
                prerank_df[[self.group_key, "note_idx", "y_multi"]].drop_duplicates([self.group_key, "note_idx"]),
                on=[self.group_key, "note_idx"],
                how="left",
            )
        gid_list = [int(x) for x in (expected_groups or [])]
        req_source = self._get_split_req_df(split).copy()
        if gid_list:
            req_source = req_source[req_source[self.group_key].isin(gid_list)].copy()
        req_rows = req_source[[self.group_key, "user_idx"]].drop_duplicates(subset=[self.group_key]).copy()
        behavior_cache: dict[int, int] = {}

        def _recent_behavior_count(uid: Any) -> int:
            user_id = _safe_intish(uid, 0)
            if user_id not in behavior_cache:
                behavior_cache[user_id] = int(len(self.build_cross_scene_recent_behaviors(user_id, max_len=40)))
            return int(behavior_cache[user_id])

        req_rows["result_count"] = req_rows[self.group_key].map(
            lambda gid: int(len(self._get_split_request_truth_map(split).get(int(gid), [])))
        ).astype(np.int32)
        req_rows["behavior_count"] = req_rows["user_idx"].map(
            _recent_behavior_count
        ).fillna(0).astype(np.int32)
        req_rows["result_tier"] = req_rows["result_count"].map(lambda n: 2 if int(n) >= 20 else (1 if int(n) >= 10 else 0)).astype(np.int32)
        req_rows["behavior_tier"] = req_rows["behavior_count"].map(lambda n: 2 if int(n) >= 20 else (1 if int(n) >= 10 else 0)).astype(np.int32)
        group_stats = req_rows.sort_values(
            ["result_tier", "behavior_tier", "result_count", "behavior_count", self.group_key],
            ascending=[False, False, False, False, True],
            kind="mergesort",
        )
        if self.scene == "search" and "query_match_score" in prerank_df.columns:
            query_quality = (
                prerank_df.sort_values([self.group_key, "preranking_score"], ascending=[True, False], kind="mergesort")
                .groupby(self.group_key, as_index=False)
                .head(10)
                .groupby(self.group_key, as_index=False)
                .agg(
                    exact_hits=("query_exact_hit", lambda s: float(pd.to_numeric(s, errors="coerce").fillna(0.0).sum())),
                    lexical_mean=("query_match_score", lambda s: float(pd.to_numeric(s, errors="coerce").fillna(0.0).mean())),
                )
            )
            group_stats = group_stats.merge(query_quality, on=self.group_key, how="left")
            group_stats["exact_hits"] = pd.to_numeric(group_stats.get("exact_hits", 0.0), errors="coerce").fillna(0.0)
            group_stats["lexical_mean"] = pd.to_numeric(group_stats.get("lexical_mean", 0.0), errors="coerce").fillna(0.0)
            group_stats = group_stats.sort_values(
                ["exact_hits", "lexical_mean", "result_count", self.group_key],
                ascending=[False, False, False, True],
                kind="mergesort",
            )
        dssm_score_col = _resolve_score_column(prerank_df, ["dssm_score", "recall_score", "preranking_score", "gbdt_score"])
        prerank_score_col = _resolve_score_column(prerank_df, ["preranking_score", "gbdt_score", "lgb_score", "xgb_score"])
        rank_score_col = _resolve_score_column(rank_df, ["dien_score", "final_score", "preranking_score", "gbdt_score"])
        overlap_rows: list[dict[str, Any]] = []
        for gid in group_stats[self.group_key].astype(int).tolist():
            true_top = set(self._get_ranked_truth_ids(int(gid), split=split, topk=10))
            if not true_top:
                continue
            gid_prerank = prerank_df[prerank_df[self.group_key] == int(gid)].copy()
            gid_rank = rank_df[rank_df[self.group_key] == int(gid)].copy()
            dssm_top = set(
                gid_prerank.sort_values(
                    [dssm_score_col, ("preranking_rank" if "preranking_rank" in gid_prerank.columns else prerank_score_col)],
                    ascending=[False, True],
                    kind="mergesort",
                ).head(10)["note_idx"].astype(int).tolist()
            )
            pre_top = set(
                gid_prerank.sort_values(
                    [("preranking_rank" if "preranking_rank" in gid_prerank.columns else prerank_score_col), prerank_score_col],
                    ascending=[True, False],
                    kind="mergesort",
                ).head(10)["note_idx"].astype(int).tolist()
            )
            rank_top = set(
                gid_rank.sort_values(
                    [("ranking_rank" if "ranking_rank" in gid_rank.columns else rank_score_col), rank_score_col],
                    ascending=[True, False],
                    kind="mergesort",
                ).head(10)["note_idx"].astype(int).tolist()
            )
            overlaps = [
                int(len(true_top.intersection(dssm_top))),
                int(len(true_top.intersection(pre_top))),
                int(len(true_top.intersection(rank_top))),
            ]
            overlap_rows.append(
                {
                    self.group_key: int(gid),
                    "best_overlap": int(max(overlaps) if overlaps else 0),
                    "total_overlap": int(sum(overlaps)),
                }
            )
        if overlap_rows:
            overlap_df = pd.DataFrame(overlap_rows)
            group_stats = group_stats.merge(overlap_df, on=self.group_key, how="left")
            group_stats["best_overlap"] = pd.to_numeric(group_stats.get("best_overlap", 0), errors="coerce").fillna(0).astype(np.int32)
            group_stats["total_overlap"] = pd.to_numeric(group_stats.get("total_overlap", 0), errors="coerce").fillna(0).astype(np.int32)
            sort_cols = ["best_overlap", "total_overlap", "result_tier", "behavior_tier", "result_count", "behavior_count", self.group_key]
            sort_asc = [False, False, False, False, False, False, True]
            if "exact_hits" in group_stats.columns and "lexical_mean" in group_stats.columns:
                sort_cols = ["best_overlap", "total_overlap", "exact_hits", "lexical_mean", "result_tier", "behavior_tier", "result_count", "behavior_count", self.group_key]
                sort_asc = [False, False, False, False, False, False, False, False, True]
            group_stats = group_stats.sort_values(sort_cols, ascending=sort_asc, kind="mergesort")
        group_stats = group_stats[group_stats["result_count"] > 0].reset_index(drop=True)
        picked: list[int] = []
        picked_users: set[int] = set()
        current_user_rows = group_stats.iloc[0:0].copy()
        other_rows = group_stats
        if context_user_idx is not None and "user_idx" in group_stats.columns:
            current_user_rows = group_stats[group_stats["user_idx"] == int(context_user_idx)].copy()
            other_rows = group_stats[group_stats["user_idx"] != int(context_user_idx)].copy()
        preferred_current = current_user_rows[
            (current_user_rows["result_count"] >= 10) & (current_user_rows["behavior_count"] >= 10)
        ]
        if len(preferred_current) <= 0:
            preferred_current = current_user_rows
        if len(preferred_current) > 0:
            first_row = preferred_current.iloc[0]
            picked.append(int(first_row[self.group_key]))
            picked_users.add(int(first_row["user_idx"]))
        preferred_other = other_rows[
            (other_rows["result_count"] >= 10) & (other_rows["behavior_count"] >= 10)
        ]
        candidate_frames = [preferred_other, other_rows]
        for frame in candidate_frames:
            if len(picked) >= int(example_limit):
                break
            for row in frame.itertuples(index=False):
                gid = int(getattr(row, self.group_key))
                uid = int(getattr(row, "user_idx"))
                if gid in picked or uid in picked_users:
                    continue
                picked.append(gid)
                picked_users.add(uid)
                if len(picked) >= int(example_limit):
                    break
        note_ids: set[int] = set()
        examples: list[dict[str, Any]] = []
        req_query_map = self._get_split_search_query_map(split) if self.scene == "search" else {}
        split_req_df = self._get_split_req_df(split).copy()
        get_feat_req = self.get_feat_req if split == "test" else self._get_train_feat_req
        truth_rank_maps: dict[int, dict[int, int]] = {}
        for gid in picked:
            gid_source = source[source[self.group_key] == gid].copy()
            truth_full = self._build_request_truth_frame(split=split, gids=[int(gid)], topk=max(200, int(self.recall_rank_cap)))
            truth_full = truth_full[truth_full[self.group_key] == int(gid)].copy()
            if len(truth_full) > 0:
                truth_full["result_rank"] = pd.to_numeric(truth_full.get("result_rank"), errors="coerce").fillna(1e9)
                truth_full = truth_full.sort_values(["result_rank"], ascending=[True], kind="mergesort")
                truth_rank_maps[int(gid)] = {
                    int(nid): int(rank)
                    for nid, rank in zip(truth_full["note_idx"].astype(int).tolist(), truth_full["result_rank"].astype(int).tolist())
                }
            else:
                truth_rank_maps[int(gid)] = {}
            true_top = [int(x) for x in list(truth_rank_maps[int(gid)].keys())[:10]]
            pre_df = prerank_df[prerank_df[self.group_key] == gid].sort_values(
                [("preranking_rank" if "preranking_rank" in prerank_df.columns else prerank_score_col), prerank_score_col],
                ascending=[True, False],
                kind="mergesort",
            ).copy()
            rank_sub = rank_df[rank_df[self.group_key] == gid].sort_values(
                [("ranking_rank" if "ranking_rank" in rank_df.columns else rank_score_col), rank_score_col],
                ascending=[True, False],
                kind="mergesort",
            ).copy()
            dssm_df = prerank_df[prerank_df[self.group_key] == gid].sort_values(
                [dssm_score_col, ("preranking_rank" if "preranking_rank" in prerank_df.columns else prerank_score_col)],
                ascending=[False, True],
                kind="mergesort",
            ).copy()
            source_uid = _safe_intish(
                req_rows[req_rows[self.group_key] == gid].iloc[0].get("user_idx", 0)
                if len(req_rows[req_rows[self.group_key] == gid]) > 0
                else (rank_sub.iloc[0].get("user_idx", 0) if len(rank_sub) > 0 else 0),
                0,
            )
            dssm_top = [int(x) for x in dssm_df["note_idx"].astype(int).tolist()[:10]]
            pre_top = [int(x) for x in pre_df["note_idx"].astype(int).tolist()[:10]]
            rank_top = [int(x) for x in rank_sub["note_idx"].astype(int).tolist()[:10]]
            try:
                feat_req = get_feat_req(int(gid))
                if len(feat_req) > 0:
                    req_query = req_query_map.get(int(gid), "")
                    query_ctx = (
                        self.preprocess_search_query(req_query, user_idx=int(source_uid))
                        if self.scene == "search"
                        else {"resolved_query": req_query, "terms": [], "intents": [], "service_query": req_query}
                    )
                    resolved_query = str(query_ctx.get("resolved_query") or req_query)
                    service_query = str(
                        query_ctx.get("service_query")
                        or " ".join([str(x) for x in (query_ctx.get("terms") or []) if str(x).strip()])
                        or resolved_query
                    )
                    recall_cand = fetch_recall_candidates(
                        request_id=int(gid),
                        max_rank=int(self.recall_rank_cap),
                        group_key=self.group_key,
                        req_df=split_req_df,
                        get_feat_req=get_feat_req,
                        dssm_recaller=self.dssm_recaller,
                        recall_test_path=(self.recall_test_path if split == "test" else None),
                    )
                    if len(recall_cand) > 0:
                        linkage_ctx = self.build_request_linkage_context(feat_req=feat_req, query=resolved_query, max_hist=20)
                        online_pre = run_preranking(
                            user_idx=int(source_uid),
                            query=service_query,
                            query_phrase=resolved_query,
                            query_terms=[str(x) for x in (query_ctx.get("terms") or []) if str(x).strip()],
                            query_intents=[str(x) for x in (query_ctx.get("intents") or []) if str(x).strip()],
                            scene=self.scene,
                            group_key=self.group_key,
                            recall_cand=recall_cand,
                            feat_req=feat_req,
                            gbdt_topn=self.gbdt_topn,
                            fetch_notes=self._fetch_notes,
                            predict_gbdt=self.predict_gbdt,
                            linkage_ctx=linkage_ctx,
                        )
                        if len(online_pre) > 0:
                            online_pre = online_pre.reset_index(drop=True)
                            online_pre["preranking_rank"] = np.arange(1, len(online_pre) + 1, dtype=np.int32)
                            online_dssm_score_col = _resolve_score_column(online_pre, ["dssm_score", "recall_score", "preranking_score", "gbdt_score"])
                            online_pre_score_col = _resolve_score_column(online_pre, ["preranking_score", "gbdt_score", "lgb_score", "xgb_score"])
                            if online_dssm_score_col:
                                dssm_top = [
                                    int(x) for x in online_pre.sort_values(
                                        [online_dssm_score_col, "preranking_rank"],
                                        ascending=[False, True],
                                        kind="mergesort",
                                    )["note_idx"].astype(int).tolist()[:10]
                                ]
                            pre_top = [
                                int(x) for x in online_pre.sort_values(
                                    ["preranking_rank", online_pre_score_col or "preranking_rank"],
                                    ascending=[True, False if online_pre_score_col else True],
                                    kind="mergesort",
                                )["note_idx"].astype(int).tolist()[:10]
                            ]
                            ranked, _ = run_ranking(
                                cand=online_pre,
                                page=1,
                                page_size=20,
                                predict_dien=self.predict_dien,
                                history_note_ids=self.get_user_history_notes(user_idx=int(source_uid), feat_req=feat_req, prefer_realtime=False),
                            )
                            if len(ranked) > 0:
                                ranked = ranked.reset_index(drop=True)
                                ranked["ranking_rank"] = np.arange(1, len(ranked) + 1, dtype=np.int32)
                                rank_top = [int(x) for x in ranked.sort_values(["ranking_rank"], ascending=[True], kind="mergesort")["note_idx"].astype(int).tolist()[:10]]
            except Exception:
                pass
            true_set = set(true_top)
            note_ids.update(true_top)
            note_ids.update(dssm_top)
            note_ids.update(pre_top)
            note_ids.update(rank_top)
            source_user = self.get_user(source_uid) if source_uid in self.user_feat_map else None
            source_recent_behaviors = self.build_cross_scene_recent_behaviors(source_uid, max_len=40)
            examples.append(
                {
                    "request_id": int(gid),
                    "query": req_query_map.get(int(gid), ""),
                    "source_user_idx": int(source_uid),
                    "is_current_user": bool(context_user_idx is not None and int(source_uid) == int(context_user_idx)),
                    "result_count": int(len(self._get_split_request_truth_map(split).get(int(gid), []))),
                    "overlap": {
                        "dssm": int(len(true_set.intersection(dssm_top))),
                        "preranking": int(len(true_set.intersection(pre_top))),
                        "ranking": int(len(true_set.intersection(rank_top))),
                    },
                    "source_user": source_user,
                    "source_recent_behaviors": source_recent_behaviors,
                    "true_top10_ids": true_top,
                    "dssm_top10_ids": dssm_top,
                    "preranking_top10_ids": pre_top,
                    "ranking_top10_ids": rank_top,
                }
            )
        note_df = self._fetch_notes(sorted(note_ids))
        title_map = {
            int(r["note_idx"]): _format_note_title(r.get("note_title"), r.get("note_content"))
            for r in note_df.to_dict("records")
        } if len(note_df) > 0 else {}
        true_rank_map = {int(k): v for k, v in truth_rank_maps.items()}
        def _materialize(ids: list[int], request_id: int) -> list[dict[str, Any]]:
            return [
                {
                    "note_idx": int(nid),
                    "title": title_map.get(int(nid), "(无标题)"),
                    "rank_in_true": true_rank_map.get(int(request_id), {}).get(int(nid)),
                }
                for nid in ids
            ]
        for ex in examples:
            ex["true_top10"] = _materialize(ex.pop("true_top10_ids"), ex["request_id"])
            ex["dssm_top10"] = _materialize(ex.pop("dssm_top10_ids"), ex["request_id"])
            ex["preranking_top10"] = _materialize(ex.pop("preranking_top10_ids"), ex["request_id"])
            ex["ranking_top10"] = _materialize(ex.pop("ranking_top10_ids"), ex["request_id"])
        return examples

    def compute_metrics_layered(
        self,
        sample_n: int = 100,
        include_val: bool = True,
    ) -> dict[str, Any]:
        val_recall = self._compute_recall_split_metrics("train", sample_n=int(sample_n)) if bool(include_val) else {}
        test_recall = self._compute_recall_split_metrics(
            "test",
            sample_n=int(sample_n),
            use_online_test_recall=False,
        )
        val_pre_df, val_rank_df, val_expected_groups = (
            self._build_val_ranking_frames(sample_n=int(sample_n))
            if bool(include_val)
            else (pd.DataFrame(), pd.DataFrame(), [])
        )
        test_pre_df, test_rank_df = self._build_test_ranking_frames(sample_n=int(sample_n))
        test_expected_groups = self._sample_groups_random(
            self.req_df[self.group_key].drop_duplicates().astype(int).tolist(),
            sample_n=int(sample_n),
            seed=TEST_SAMPLE_SEED,
        ) if len(self.req_df) > 0 else []
        val_label_source = self._build_request_truth_frame("train", val_expected_groups, topk=200) if val_expected_groups else pd.DataFrame()
        test_label_source = self._build_request_truth_frame("test", test_expected_groups, topk=200) if test_expected_groups else pd.DataFrame()
        return {
            "scene": self.scene,
            "tag": self.tag,
            "readiness": self.readiness(),
            "recall": {
                "val": val_recall,
                "test": test_recall,
            },
            "ranking": {
                "preranking": {
                    "val": self._metric_bundle_from_scored(
                        val_pre_df,
                        ["preranking_score", "gbdt_score", "lgb_score", "xgb_score"],
                        expected_groups=val_expected_groups,
                        label_source=val_label_source,
                    ),
                    "test": self._metric_bundle_from_scored(
                        test_pre_df,
                        ["preranking_score", "gbdt_score", "lgb_score", "xgb_score"],
                        expected_groups=test_expected_groups,
                        label_source=test_label_source,
                    ),
                },
                "ranking": {
                    "val": self._metric_bundle_from_scored(
                        val_rank_df,
                        ["dien_score", "preranking_score", "gbdt_score"],
                        expected_groups=val_expected_groups,
                        label_source=val_label_source,
                    ),
                    "test": self._metric_bundle_from_scored(
                        test_rank_df,
                        ["dien_score", "preranking_score", "gbdt_score"],
                        expected_groups=test_expected_groups,
                        label_source=test_label_source,
                    ),
                },
            },
        }

    def compute_validation_metrics(self, sample_n: int = 1000) -> dict[str, Any]:
        val_pre_df, val_rank_df, val_expected_groups = self._build_val_ranking_frames(sample_n=max(2, int(sample_n)))
        val_label_source = self._build_request_truth_frame("train", val_expected_groups, topk=200) if val_expected_groups else pd.DataFrame()
        val_recall = self._compute_recall_split_metrics("train", sample_n=max(2, int(sample_n)), gids=val_expected_groups)
        if val_expected_groups:
            val_recall = {
                **val_recall,
                **self._compute_prerank_tail_hit_metric(
                    recall_df=self._build_split_recall_frame("train", sample_n=max(2, int(sample_n)), gids=val_expected_groups),
                    prerank_df=val_pre_df,
                    gids=val_expected_groups,
                ),
            }
        return {
            "scene": self.scene,
            "tag": self.tag,
            "readiness": self.readiness(),
            "split": "validation",
            "recall": {
                "validation": val_recall,
            },
            "ranking": {
                "preranking": {
                    "validation": self._metric_bundle_from_scored(
                        val_pre_df,
                        ["preranking_score", "gbdt_score", "lgb_score", "xgb_score"],
                        expected_groups=val_expected_groups,
                        label_source=val_label_source,
                    ),
                },
                "ranking": {
                    "validation": self._metric_bundle_from_scored(
                        val_rank_df,
                        ["dien_score", "preranking_score", "gbdt_score"],
                        expected_groups=val_expected_groups,
                        label_source=val_label_source,
                    ),
                },
            },
        }

    def compute_test_metrics(self, dien_max_groups: int = 1200) -> dict[str, Any]:
        return self.compute_metrics_layered(sample_n=max(1, min(100, int(dien_max_groups))), include_val=False)

    def compute_validation_compare(
        self,
        max_groups: int = 800,
        example_limit: int = 5,
        context_user_idx: int | None = None,
    ) -> dict[str, Any]:
        sampled_n = max(1, min(1000, max(int(max_groups), int(example_limit) * 16, 240)))
        val_pre_df, val_rank_df, expected_groups = self._build_val_ranking_frames(sample_n=sampled_n)
        return {
            "scene": self.scene,
            "split": "validation",
            "sampled_groups": int(len(expected_groups)),
            "effective_groups": int(val_rank_df[self.group_key].nunique()) if len(val_rank_df) > 0 else 0,
            "examples": self._build_validation_examples(
                prerank_df=val_pre_df,
                rank_df=val_rank_df,
                example_limit=int(example_limit),
                context_user_idx=context_user_idx,
                expected_groups=expected_groups,
                split="train",
            ),
        }

    def build_user_validation_example(self, user_idx: int) -> dict[str, Any] | None:
        uid = int(user_idx)
        req_df = self._get_split_req_df("train").copy()
        if len(req_df) <= 0 or "user_idx" not in req_df.columns:
            return None
        req_df = req_df[req_df["user_idx"] == uid].copy()
        if len(req_df) <= 0:
            return None
        gids = req_df[self.group_key].drop_duplicates().astype(int).tolist()
        if not gids:
            return None
        pre_df, _ = self._load_preranking_metrics()
        rank_df = self._load_dien_eval_frame(full=True)
        if pre_df is None or rank_df is None:
            return None
        pre_df = pre_df[pre_df[self.group_key].isin(gids)].copy()
        rank_df = rank_df[rank_df[self.group_key].isin(gids)].copy()
        if len(pre_df) <= 0 or len(rank_df) <= 0:
            return None
        examples = self._build_validation_examples(
            prerank_df=pre_df,
            rank_df=rank_df,
            example_limit=1,
            context_user_idx=uid,
            expected_groups=gids,
            split="train",
        )
        return dict(examples[0]) if examples else None

    def refresh_validation_examples_from_requests(
        self,
        request_ids: list[int],
        context_user_idx: int | None = None,
        split: str = "train",
    ) -> list[dict[str, Any]]:
        gids = [int(x) for x in request_ids if int(x) >= 0]
        if not gids:
            return []
        pre_df, _ = self._load_preranking_metrics()
        rank_df = self._load_dien_eval_frame(full=True)
        if pre_df is None or rank_df is None:
            return []
        pre_df = pre_df[pre_df[self.group_key].isin(gids)].copy()
        rank_df = rank_df[rank_df[self.group_key].isin(gids)].copy()
        if len(pre_df) <= 0 or len(rank_df) <= 0:
            return []
        return self._build_validation_examples(
            prerank_df=pre_df,
            rank_df=rank_df,
            example_limit=max(1, len(gids)),
            context_user_idx=context_user_idx,
            expected_groups=gids,
            split=split,
        )


class ServingAppState:
    """管理多场景状态，按需懒加载。"""

    def __init__(self, tag: str, gbdt_topn: int, recall_rank_cap: int):
        self.tag = tag
        self.gbdt_topn = gbdt_topn
        self.recall_rank_cap = recall_rank_cap
        self._states: dict[str, SceneServingState] = {}
        self._lock = threading.Lock()
        self._linkage_cache_lock = threading.Lock()
        self._linkage_cache: dict[int, tuple[float, dict[str, Any]]] = {}
        self._linkage_cache_ttl_sec = float(os.getenv("QILIN_LINKAGE_CACHE_TTL_SEC", "8"))

    def get(self, scene: str) -> SceneServingState:
        if scene not in {"search", "rec"}:
            raise KeyError(f"invalid scene: {scene}")
        with self._lock:
            st = self._states.get(scene)
            if st is None:
                st = SceneServingState(
                    scene=scene,
                    tag=self.tag,
                    gbdt_topn=self.gbdt_topn,
                    recall_rank_cap=self.recall_rank_cap,
                )
                st._app_state = self
                self._states[scene] = st
            return st

    def readiness(self) -> dict[str, Any]:
        out = {}
        for scene in ["search", "rec"]:
            try:
                out[scene] = self.get(scene).readiness()
            except Exception as e:  # noqa: BLE001
                out[scene] = {"error": f"{type(e).__name__}: {e}"}
        return out

    def get_cross_scene_behaviors(self, user_idx: int, max_len: int = 20) -> list[dict[str, Any]]:
        return self._get_cross_scene_behaviors(user_idx=int(user_idx), max_len=max_len, loaded_only=False)

    def _get_cross_scene_behaviors(self, user_idx: int, max_len: int = 20, loaded_only: bool = False) -> list[dict[str, Any]]:
        uid = int(user_idx)
        loaded_scenes = list(self._states.keys()) if bool(loaded_only) else ["search", "rec"]
        for scene in loaded_scenes:
            try:
                st = self._states.get(scene) if bool(loaded_only) else self.get(scene)
                if st is None:
                    continue
                if st.realtime_cache is not None:
                    rows = st.realtime_cache.get_recent_behaviors_all(uid, max_len=max_len)
                    if rows:
                        return rows[:max_len]
            except Exception:
                continue

        merged: list[dict[str, Any]] = []
        for scene in loaded_scenes:
            try:
                st = self._states.get(scene) if bool(loaded_only) else self.get(scene)
                if st is None:
                    continue
                merged.extend(st._build_recent_behaviors(uid, max_len=max_len))
            except Exception:
                continue
        merged.sort(key=lambda x: int(x.get("ts", 0) or 0), reverse=True)
        seen: set[str] = set()
        out: list[dict[str, Any]] = []
        for row in merged:
            key = f"{row.get('scene')}|{row.get('request_id')}|{row.get('note_idx')}|{row.get('query','')}"
            if key in seen:
                continue
            seen.add(key)
            out.append(row)
            if len(out) >= int(max_len):
                break
        return out

    def build_linkage_context(self, user_idx: int, max_len: int = 20, loaded_only: bool = False) -> dict[str, Any]:
        uid = int(user_idx)
        now = time.monotonic()
        with self._linkage_cache_lock:
            cached = self._linkage_cache.get(uid)
            if cached is not None and cached[0] > now:
                return cached[1]

        behaviors = self._get_cross_scene_behaviors(uid, max_len=max_len, loaded_only=bool(loaded_only))
        keyword_counts: dict[str, float] = {}
        tag_counts: dict[int, float] = {}
        for idx, row in enumerate(behaviors):
            weight = 1.0 / (1.0 + idx)
            if str(row.get("scene", "")) == "search":
                for term in _extract_terms(str(row.get("query") or ""), max_terms=4):
                    keyword_counts[term] = keyword_counts.get(term, 0.0) + weight
            for tag_id in row.get("tag_ids", []):
                tag = _safe_intish(tag_id, -1)
                if tag >= 0:
                    tag_counts[tag] = tag_counts.get(tag, 0.0) + weight
        payload = {
            "behaviors": behaviors,
            "keywords": [k for k, _ in sorted(keyword_counts.items(), key=lambda x: (-x[1], x[0]))[:6]],
            "tag_ids": [k for k, _ in sorted(tag_counts.items(), key=lambda x: (-x[1], x[0]))[:8]],
        }
        with self._linkage_cache_lock:
            self._linkage_cache[uid] = (now + max(1.0, self._linkage_cache_ttl_sec), payload)
            if len(self._linkage_cache) > 4096:
                self._linkage_cache.clear()
                self._linkage_cache[uid] = (now + max(1.0, self._linkage_cache_ttl_sec), payload)
        return payload


class SearchServingState(SceneServingState):
    def __init__(self, tag: str = "easy", gbdt_topn: int = 500, recall_rank_cap: int = 1000):
        super().__init__(scene="search", tag=tag, gbdt_topn=gbdt_topn, recall_rank_cap=recall_rank_cap)

class OnlineScenePipeline:
    def __init__(self, scene_state, app_state: ServingAppState | None = None):
        self.state = scene_state
        self.app_state = app_state

    def prewarm_homepage_feed(self, page_size: int = 30) -> dict[str, Any]:
        seed = self.state.get_prewarm_seed()
        return self.build_feed(
            user_idx=int(seed.get("user_idx", 0)),
            query=str(seed.get("query", "") or ""),
            page=1,
            page_size=max(1, int(page_size)),
            refresh_key="startup-prewarm",
            exclude_note_ids=None,
        )

    def _stage_cache_key(self, user_idx: int, req_id: int, service_query: str) -> tuple[int, int, str]:
        return (int(user_idx), int(req_id), str(service_query or "").strip().lower())

    def _get_cached_stage(self, user_idx: int, req_id: int, service_query: str) -> tuple[pd.DataFrame, dict[str, Any]] | None:
        key = self._stage_cache_key(user_idx, req_id, service_query)
        now = time.monotonic()
        with self.state._stage_cache_lock:
            cached = self.state._stage_cache.get(key)
            if cached is None:
                return None
            expire_at, cand, meta = cached
            if expire_at <= now:
                self.state._stage_cache.pop(key, None)
                return None
            return cand.copy(), dict(meta)

    def _set_cached_stage(self, user_idx: int, req_id: int, service_query: str, cand: pd.DataFrame, meta: dict[str, Any]) -> None:
        key = self._stage_cache_key(user_idx, req_id, service_query)
        expire_at = time.monotonic() + max(1.0, self.state._stage_cache_ttl_sec)
        with self.state._stage_cache_lock:
            self.state._stage_cache[key] = (expire_at, cand.copy(), dict(meta))
            if len(self.state._stage_cache) > 2048:
                self.state._stage_cache.clear()
                self.state._stage_cache[key] = (expire_at, cand.copy(), dict(meta))

    def _feed_cache_key(self, user_idx: int, req_id: int, query: str, refresh_key: str = "") -> tuple[int, int, str, str]:
        return (int(user_idx), int(req_id), str(query or "").strip().lower(), str(refresh_key or ""))

    def _get_cached_feed(self, user_idx: int, req_id: int, query: str, refresh_key: str = "") -> tuple[pd.DataFrame, dict[str, Any]] | None:
        key = self._feed_cache_key(user_idx, req_id, query, refresh_key)
        now = time.monotonic()
        with self.state._feed_cache_lock:
            cached = self.state._feed_cache.get(key)
            if cached is None:
                return None
            expire_at, cand, meta = cached
            if expire_at <= now:
                self.state._feed_cache.pop(key, None)
                return None
            return cand.copy(), dict(meta)

    def _set_cached_feed(self, user_idx: int, req_id: int, query: str, refresh_key: str, cand: pd.DataFrame, meta: dict[str, Any]) -> None:
        key = self._feed_cache_key(user_idx, req_id, query, refresh_key)
        expire_at = time.monotonic() + max(1.0, self.state._feed_cache_ttl_sec)
        with self.state._feed_cache_lock:
            self.state._feed_cache[key] = (expire_at, cand.copy(), dict(meta))
            if len(self.state._feed_cache) > 1024:
                self.state._feed_cache.clear()
                self.state._feed_cache[key] = (expire_at, cand.copy(), dict(meta))

    def _apply_rec_diversity(self, cand: pd.DataFrame, user_idx: int, history_note_ids: list[int]) -> pd.DataFrame:
        if cand.empty or self.state.scene != "rec":
            return cand
        out = cand.copy()
        history_set = {int(x) for x in history_note_ids if int(x) >= 0}
        exposed_set: set[int] = set()
        if self.state.realtime_cache is not None:
            exposed_set = {
                int(x)
                for x in self.state.realtime_cache.get_recent_exposed_notes(int(user_idx), self.state.scene, max_len=200)
                if int(x) >= 0
            }
        repeat_penalty = np.zeros(len(out), dtype=np.float32)
        note_ids = pd.to_numeric(out.get("note_idx"), errors="coerce").fillna(-1).astype(np.int64).to_numpy()
        if history_set:
            repeat_penalty += np.isin(note_ids, np.asarray(sorted(history_set), dtype=np.int64)).astype(np.float32) * 0.90
        if exposed_set:
            exposed_arr = np.asarray(sorted(exposed_set), dtype=np.int64)
            exposed_mask = np.isin(note_ids, exposed_arr)
            filtered = out.loc[~exposed_mask].copy()
            if len(filtered) >= min(20, len(out)):
                out = filtered.reset_index(drop=True)
                repeat_penalty = repeat_penalty[~exposed_mask]
                note_ids = pd.to_numeric(out.get("note_idx"), errors="coerce").fillna(-1).astype(np.int64).to_numpy()
            else:
                repeat_penalty += exposed_mask.astype(np.float32) * 0.28
        out["repeat_penalty"] = repeat_penalty
        base_score = pd.to_numeric(out.get("dien_score", out.get("preranking_score", 0.0)), errors="coerce").fillna(0.0)
        out["final_score"] = (base_score - pd.to_numeric(out["repeat_penalty"], errors="coerce").fillna(0.0)).astype(np.float32)
        out = out.sort_values(["final_score", "dien_score", "preranking_score", "rank"], ascending=[False, False, False, True], kind="mergesort").reset_index(drop=True)
        out["rerank_rank"] = np.arange(1, len(out) + 1, dtype=np.int32)
        if self.state.realtime_cache is not None:
            expose_topn = min(60, len(out))
            self.state.realtime_cache.record_exposed_notes(
                int(user_idx),
                self.state.scene,
                out["note_idx"].head(expose_topn).astype(int).tolist(),
            )
        return out

    def _backfill_rec_candidates(
        self,
        cand: pd.DataFrame,
        user_idx: int,
        history_note_ids: list[int],
        exclude_note_ids: list[int] | None,
        target_total: int,
    ) -> pd.DataFrame:
        if self.state.scene != "rec" or int(target_total) <= 0 or len(cand) >= int(target_total):
            return cand
        pool = getattr(self.state, "hot_note_pool", None) or []
        if not pool:
            return cand
        blocked_ids = {
            int(x)
            for x in pd.to_numeric(cand.get("note_idx"), errors="coerce").fillna(-1).astype(np.int64).tolist()
            if int(x) >= 0
        }
        blocked_ids.update(int(x) for x in history_note_ids if int(x) >= 0)
        blocked_ids.update(int(x) for x in (exclude_note_ids or []) if int(x) >= 0)
        if self.state.realtime_cache is not None:
            blocked_ids.update(
                int(x)
                for x in self.state.realtime_cache.get_recent_exposed_notes(int(user_idx), self.state.scene, max_len=300)
                if int(x) >= 0
            )
        fill_ids: list[int] = []
        need = max(0, int(target_total) - len(cand))
        for nid in pool:
            if int(nid) in blocked_ids:
                continue
            blocked_ids.add(int(nid))
            fill_ids.append(int(nid))
            if len(fill_ids) >= need:
                break
        if not fill_ids:
            return cand
        fill_df = self.state._fetch_notes(fill_ids)
        if len(fill_df) <= 0:
            return cand
        order_map = {int(nid): idx for idx, nid in enumerate(fill_ids)}
        fill_df = fill_df[fill_df["note_idx"].isin(fill_ids)].copy()
        fill_df["_fill_order"] = fill_df["note_idx"].map(order_map).fillna(1e9).astype(np.int32)
        fill_df = fill_df.sort_values("_fill_order", kind="mergesort").reset_index(drop=True)
        score_floor = float(
            pd.to_numeric(cand.get("final_score", cand.get("dien_score", cand.get("preranking_score", 0.0))), errors="coerce")
            .fillna(0.0)
            .min()
        ) if len(cand) > 0 else 0.0
        fill_scores = (score_floor - 1.0 - np.arange(len(fill_df), dtype=np.float32) / 1000.0).astype(np.float32)
        fill_df["rank"] = np.arange(len(cand) + 1, len(cand) + len(fill_df) + 1, dtype=np.int32)
        fill_df["preranking_rank"] = np.arange(len(cand) + 1, len(cand) + len(fill_df) + 1, dtype=np.int32)
        fill_df["ranking_rank"] = np.arange(len(cand) + 1, len(cand) + len(fill_df) + 1, dtype=np.int32)
        fill_df["rerank_rank"] = np.arange(len(cand) + 1, len(cand) + len(fill_df) + 1, dtype=np.int32)
        fill_df["repeat_penalty"] = np.zeros(len(fill_df), dtype=np.float32)
        fill_df["dssm_score"] = fill_scores
        fill_df["gbdt_score"] = fill_scores
        fill_df["preranking_score"] = fill_scores
        fill_df["dien_score"] = fill_scores
        fill_df["final_score"] = fill_scores
        fill_df["y_multi"] = np.zeros(len(fill_df), dtype=np.float32)
        fill_df["click"] = np.zeros(len(fill_df), dtype=np.int8)
        fill_df = fill_df.drop(columns=["_fill_order"], errors="ignore")
        if len(cand) > 0:
            cand_cols = list(cand.columns)
            fill_df = fill_df.reindex(columns=cand_cols, fill_value=np.nan)
            merged_records = cand.to_dict("records") + fill_df.to_dict("records")
            out = pd.DataFrame.from_records(merged_records, columns=cand_cols)
        else:
            out = fill_df.reset_index(drop=True)
        out["rerank_rank"] = np.arange(1, len(out) + 1, dtype=np.int32)
        return out

    def _build_cards(self, req_id: int, user_idx: int, page_df: pd.DataFrame) -> list[dict[str, Any]]:
        cards = []
        for r in page_df.to_dict("records"):
            imgs = _existing_images(_to_path_list(r.get("image_path")))
            cards.append(
                {
                    "scene": self.state.scene,
                    "request_id": int(req_id),
                    "search_idx": int(req_id) if self.state.scene == "search" else None,
                    "request_idx": int(req_id) if self.state.scene == "rec" else None,
                    "user_idx": int(user_idx),
                    "note_idx": int(r["note_idx"]),
                    "title": _format_note_title(r.get("note_title"), r.get("note_content")),
                    "cover_image": imgs[0] if imgs else "",
                    "image_count": len(imgs),
                    "accum_like_num": _safe_intish(r.get("accum_like_num", 0)),
                    "accum_collect_num": _safe_intish(r.get("accum_collect_num", 0)),
                    "accum_comment_num": _safe_intish(r.get("accum_comment_num", 0)),
                    "scores": {
                        "dssm": float(np.nan_to_num(r.get("dssm_score", 0.0), nan=0.0, posinf=1e6, neginf=-1e6)),
                        "gbdt": float(np.nan_to_num(r.get("gbdt_score", 0.0), nan=0.0, posinf=1e6, neginf=-1e6)),
                        "dien": float(np.nan_to_num(r.get("dien_score", 0.0), nan=0.0, posinf=1e6, neginf=-1e6)),
                        "final": float(np.nan_to_num(r.get("final_score", 0.0), nan=0.0, posinf=1e6, neginf=-1e6)),
                    },
                    "labels": {
                        "y_multi": float(np.nan_to_num(r.get("y_multi", 0.0) or 0.0, nan=0.0, posinf=1e6, neginf=-1e6)),
                        "click": _safe_intish(r.get("click", 0.0)),
                    },
                    "stage_ranks": {
                        "recall": _safe_intish(r.get("rank"), 0) or None,
                        "preranking": _safe_intish(r.get("preranking_rank"), 0) or None,
                        "ranking": _safe_intish(r.get("ranking_rank"), 0) or None,
                        "rerank": _safe_intish(r.get("rerank_rank"), 0) or None,
                    },
                }
            )
        return cards

    def _select_page_rows(self, cand: pd.DataFrame, page: int, page_size: int) -> pd.DataFrame:
        if cand.empty:
            return cand
        preferred: list[int] = []
        fallback: list[int] = []
        for idx, row in cand.iterrows():
            if _has_existing_image(row.get("image_path")):
                preferred.append(int(idx))
            else:
                fallback.append(int(idx))
        order = preferred + fallback
        start = max(0, (int(page) - 1) * int(page_size))
        end = start + int(page_size)
        picked = order[start:end]
        if not picked:
            return cand.iloc[0:0].copy()
        return cand.loc[picked].reset_index(drop=True)

    def _refresh_target_total(self, page: int, page_size: int, exclude_ids: set[int] | None = None) -> int:
        base = max(int(page_size), int(page) * int(page_size))
        extra = min(max(0, len(exclude_ids or set())), int(page_size) * 6)
        return max(base, int(page_size) * 4, base + extra)

    def build_feed(
        self,
        user_idx: int,
        query: str,
        page: int,
        page_size: int,
        refresh_key: str = "",
        exclude_note_ids: list[int] | None = None,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        def _display_latency(total_ms: float, stage_ms: dict[str, Any] | None) -> float:
            if not stage_ms or not isinstance(stage_ms, dict):
                return float(total_ms)
            stage_total = 0.0
            for key in ("coldstart", "recall", "preranking", "ranking"):
                try:
                    stage_total += max(0.0, float(stage_ms.get(key, 0.0)))
                except Exception:
                    continue
            return float(max(float(total_ms), stage_total))
        query_ctx = self.state.preprocess_search_query(query, user_idx=int(user_idx)) if self.state.scene == "search" else {
            "input_query": str(query or ""),
            "normalized_query": str(query or ""),
            "corrected_query": "",
            "resolved_query": str(query or ""),
            "terms": [],
            "intents": [],
            "service_query": str(query or ""),
        }
        resolved_query = str(query_ctx.get("resolved_query") or query or "")
        service_query = str(query_ctx.get("service_query") or " ".join([str(x) for x in (query_ctx.get("terms") or []) if str(x).strip()]) or resolved_query)
        req_id, matched_query = self.state.resolve_request(user_idx, resolved_query)
        feat_req = self.state.get_feat_req(req_id)
        if feat_req is None:
            feat_req = self.state.empty_feat_df
        cache_query = resolved_query or str(query or "")
        force_refresh_recompute = bool(refresh_key) and self.state.scene == "rec"
        cached_feed = self._get_cached_feed(user_idx=int(user_idx), req_id=int(req_id), query=cache_query, refresh_key=refresh_key)
        if cached_feed is not None:
            cand, cached_meta = cached_feed
            page_df = self._select_page_rows(cand, page=page, page_size=page_size)
            total_ms = float((time.perf_counter() - t0) * 1000.0)
            return {
                "scene": self.state.scene,
                "user_idx": int(user_idx),
                "request_id": int(req_id),
                "cold_start": False,
                "stages": cached_meta["stages"],
                "query_input": query,
                "query_rewritten": query_ctx.get("corrected_query", ""),
                "query_terms": query_ctx.get("terms", []),
                "query_intents": query_ctx.get("intents", []),
                "matched_query": matched_query,
                "linkage": cached_meta["linkage"],
                "total": int(len(cand)),
                "page": int(page),
                "page_size": int(page_size),
                "stage_ms": dict(cached_meta.get("stage_ms", {})),
                "pipeline_ms": float(cached_meta.get("pipeline_ms", 0.0)),
                "latency_ms": _display_latency(total_ms, cached_meta.get("stage_ms")),
                "cache_hit": True,
                "items": self._build_cards(req_id=int(req_id), user_idx=int(user_idx), page_df=page_df),
            }
        cached_stage = None if force_refresh_recompute else self._get_cached_stage(
            user_idx=int(user_idx),
            req_id=int(req_id),
            service_query=service_query,
        )
        if cached_stage is not None:
            cand, stage_meta = cached_stage
            history_note_ids = self.state.get_user_history_notes(user_idx=int(user_idx), feat_req=feat_req)
            exclude_ids = {int(x) for x in (exclude_note_ids or []) if int(x) >= 0}
            cand = self._apply_rec_diversity(cand, user_idx=int(user_idx), history_note_ids=history_note_ids)
            if exclude_ids:
                cand = cand[~pd.to_numeric(cand.get("note_idx"), errors="coerce").fillna(-1).astype(np.int64).isin(sorted(exclude_ids))].reset_index(drop=True)
                if "rerank_rank" in cand.columns:
                    cand["rerank_rank"] = np.arange(1, len(cand) + 1, dtype=np.int32)
            cand = self._backfill_rec_candidates(
                cand=cand,
                user_idx=int(user_idx),
                history_note_ids=history_note_ids,
                exclude_note_ids=sorted(exclude_ids),
                target_total=self._refresh_target_total(page=page, page_size=page_size, exclude_ids=exclude_ids),
            )
            page_df = self._select_page_rows(cand, page=page, page_size=page_size)
            total_ms = float((time.perf_counter() - t0) * 1000.0)
            cached_meta = {
                "stages": stage_meta.get("stages", {}),
                "linkage": stage_meta.get("linkage", {}),
                "stage_ms": dict(stage_meta.get("stage_ms", {})),
                "pipeline_ms": float(stage_meta.get("pipeline_ms", 0.0)),
            }
            self._set_cached_feed(user_idx=int(user_idx), req_id=int(req_id), query=cache_query, refresh_key=refresh_key, cand=cand, meta=cached_meta)
            return {
                "scene": self.state.scene,
                "user_idx": int(user_idx),
                "request_id": int(req_id),
                "cold_start": False,
                "stages": cached_meta["stages"],
                "query_input": query,
                "query_rewritten": query_ctx.get("corrected_query", ""),
                "query_terms": query_ctx.get("terms", []),
                "query_intents": query_ctx.get("intents", []),
                "matched_query": matched_query,
                "linkage": cached_meta["linkage"],
                "total": int(len(cand)),
                "page": int(page),
                "page_size": int(page_size),
                "stage_ms": dict(cached_meta.get("stage_ms", {})),
                "pipeline_ms": float(cached_meta.get("pipeline_ms", 0.0)),
                "latency_ms": _display_latency(total_ms, cached_meta.get("stage_ms")),
                "cache_hit": True,
                "items": self._build_cards(req_id=int(req_id), user_idx=int(user_idx), page_df=page_df),
            }
        linkage_ctx = self.app_state.build_linkage_context(int(user_idx), max_len=20, loaded_only=True) if self.app_state is not None else {}

        cold = is_cold_start(
            scene=self.state.scene,
            user_idx=int(user_idx),
            user_requests=self.state.user_requests,
            request_threshold=COLD_START_REQ_THRESHOLD,
        )
        t_cold = time.perf_counter()
        recall_cand = run_recall(
            request_id=req_id,
            user_idx=int(user_idx),
            feat_req=feat_req,
            is_cold=cold,
            recall_rank_cap=self.state.live_recall_rank_cap,
            hot_route_topk=min(HOT_ROUTE_TOPK, self.state.live_recall_rank_cap),
            fetch_recall_candidates=self.state._fetch_recall_candidates,
            group_key=self.state.group_key,
        )
        t_recall = time.perf_counter()
        prerank_cand = run_preranking(
            user_idx=int(user_idx),
            query=service_query,
            query_phrase=resolved_query,
            query_terms=[str(x) for x in (query_ctx.get("terms") or []) if str(x).strip()],
            query_intents=[str(x) for x in (query_ctx.get("intents") or []) if str(x).strip()],
            scene=self.state.scene,
            group_key=self.state.group_key,
            recall_cand=recall_cand,
            feat_req=feat_req,
            gbdt_topn=self.state.live_gbdt_topn,
            fetch_notes=self.state._fetch_notes,
            predict_gbdt=self.state.predict_gbdt,
            linkage_ctx=linkage_ctx,
        )
        if len(prerank_cand) > 0:
            prerank_cand = prerank_cand.reset_index(drop=True)
            prerank_cand["preranking_rank"] = np.arange(1, len(prerank_cand) + 1, dtype=np.int32)
        t_prerank = time.perf_counter()
        history_note_ids = self.state.get_user_history_notes(user_idx=int(user_idx), feat_req=feat_req)
        cand, page_df = run_ranking(
            cand=prerank_cand,
            page=page,
            page_size=page_size,
            predict_dien=self.state.predict_dien,
            history_note_ids=history_note_ids,
        )
        if len(cand) > 0:
            cand = cand.reset_index(drop=True)
            cand["ranking_rank"] = np.arange(1, len(cand) + 1, dtype=np.int32)
        base_ranked_cand = cand.copy()
        cand = self._apply_rec_diversity(cand, user_idx=int(user_idx), history_note_ids=history_note_ids)
        exclude_ids = {int(x) for x in (exclude_note_ids or []) if int(x) >= 0}
        if exclude_ids:
            cand = cand[~pd.to_numeric(cand.get("note_idx"), errors="coerce").fillna(-1).astype(np.int64).isin(sorted(exclude_ids))].reset_index(drop=True)
            if "rerank_rank" in cand.columns:
                cand["rerank_rank"] = np.arange(1, len(cand) + 1, dtype=np.int32)
        cand = self._backfill_rec_candidates(
            cand=cand,
            user_idx=int(user_idx),
            history_note_ids=history_note_ids,
            exclude_note_ids=sorted(exclude_ids),
            target_total=self._refresh_target_total(page=page, page_size=page_size, exclude_ids=exclude_ids),
        )
        page_df = self._select_page_rows(cand, page=page, page_size=page_size)
        t_rank = time.perf_counter()

        stage_ms = {
            "coldstart": float((t_cold - t0) * 1000.0),
            "recall": float((t_recall - t_cold) * 1000.0),
            "preranking": float((t_prerank - t_recall) * 1000.0),
            "ranking": float((t_rank - t_prerank) * 1000.0),
        }
        total_ms = float((t_rank - t0) * 1000.0)
        stages = {
            "coldstart": {"enabled": bool(cold)},
            "recall": {"candidates": int(len(recall_cand))},
            "preranking": {"candidates": int(len(prerank_cand)), "topn": int(self.state.live_gbdt_topn)},
            "ranking": {"candidates": int(len(cand)), "page_items": int(len(page_df))},
        }
        cached_meta = {
            "stages": stages,
            "linkage": {
                "keywords": linkage_ctx.get("keywords", []),
                "tag_ids": linkage_ctx.get("tag_ids", []),
                "behavior_count": int(len(linkage_ctx.get("behaviors", []))),
            },
            "stage_ms": dict(stage_ms),
            "pipeline_ms": float(stage_ms["recall"] + stage_ms["preranking"] + stage_ms["ranking"]),
        }
        self._set_cached_stage(
            user_idx=int(user_idx),
            req_id=int(req_id),
            service_query=service_query,
            cand=base_ranked_cand,
            meta=cached_meta,
        )
        self._set_cached_feed(user_idx=int(user_idx), req_id=int(req_id), query=cache_query, refresh_key=refresh_key, cand=cand, meta=cached_meta)

        return {
            "scene": self.state.scene,
            "user_idx": int(user_idx),
            "request_id": int(req_id),
            "cold_start": bool(cold),
            "stages": stages,
            "query_input": query,
            "query_rewritten": query_ctx.get("corrected_query", ""),
            "query_terms": query_ctx.get("terms", []),
            "query_intents": query_ctx.get("intents", []),
            "matched_query": matched_query,
            "linkage": cached_meta["linkage"],
            "total": int(len(cand)),
            "page": int(page),
            "page_size": int(page_size),
            "stage_ms": stage_ms,
            "pipeline_ms": float(stage_ms["recall"] + stage_ms["preranking"] + stage_ms["ranking"]),
            "latency_ms": _display_latency(total_ms, stage_ms),
            "cache_hit": False,
            "items": self._build_cards(req_id=int(req_id), user_idx=int(user_idx), page_df=page_df),
        }


class OnlineRuntime:
    def __init__(self, tag: str, gbdt_topn: int, recall_rank_cap: int):
        self._app_state = ServingAppState(tag=tag, gbdt_topn=gbdt_topn, recall_rank_cap=recall_rank_cap)

    def readiness(self) -> dict[str, Any]:
        return self._app_state.readiness()

    def get_pipeline(self, scene: str) -> OnlineScenePipeline:
        return OnlineScenePipeline(self._app_state.get(scene), app_state=self._app_state)


class OnlineRuntimeRegistry:
    def __init__(self, default_tag: str, gbdt_topn: int, recall_rank_cap: int):
        self.default_tag = default_tag if default_tag in {"easy", "hard"} else "easy"
        self.gbdt_topn = int(gbdt_topn)
        self.recall_rank_cap = int(recall_rank_cap)
        self._states: dict[str, OnlineRuntime] = {}

    def get_runtime(self, tag: str | None) -> OnlineRuntime:
        use_tag = str(tag or self.default_tag).lower()
        if use_tag not in {"easy", "hard"}:
            raise ValueError(f"invalid tag: {tag}")
        rt = self._states.get(use_tag)
        if rt is None:
            rt = OnlineRuntime(tag=use_tag, gbdt_topn=self.gbdt_topn, recall_rank_cap=self.recall_rank_cap)
            self._states[use_tag] = rt
        return rt


def main() -> None:
    cfg = load_online_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=cfg.host)
    parser.add_argument("--port", type=int, default=cfg.port)
    parser.add_argument("--tag", type=str, default=cfg.default_tag)
    parser.add_argument("--gbdt-topn", type=int, default=cfg.gbdt_topn)
    parser.add_argument("--recall-rank-cap", type=int, default=cfg.recall_rank_cap)
    args = parser.parse_args()

    from backend.online.api.main import create_app

    app = create_app(tag=args.tag, gbdt_topn=args.gbdt_topn, recall_rank_cap=args.recall_rank_cap)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


__all__ = [
    "COLD_START_REQ_THRESHOLD",
    "HOT_ROUTE_TOPK",
    "SceneServingState",
    "ServingAppState",
    "SearchServingState",
    "OnlineScenePipeline",
    "OnlineRuntime",
    "OnlineRuntimeRegistry",
    "main",
]


if __name__ == "__main__":
    main()
