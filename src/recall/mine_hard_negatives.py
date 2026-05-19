from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

BASE_DIR = Path(__file__).resolve().parents[2]
FEATURE_DIR = BASE_DIR / "features"
OUT_DATA_DIR = BASE_DIR / "outputs" / "data"
DATASETS_DIR = BASE_DIR / "datasets"


def _group_key(scene: str) -> str:
    return "search_idx" if scene == "search" else "request_idx"


def _request_dataset_path(scene: str, split: str) -> Path:
    folder = "search" if scene == "search" else "recommendation"
    return DATASETS_DIR / f"{folder}_{split}" / "train-00000-of-00001.parquet"


def _request_result_cols(scene: str) -> list[str]:
    if scene == "search":
        return ["search_results", "search_result_details_with_idx"]
    return ["rec_results", "rec_result_details_with_idx"]


def _safe_int(x: Any, default: int = -1) -> int:
    try:
        if x is None:
            return default
        if isinstance(x, float) and np.isnan(x):
            return default
        return int(float(x))
    except Exception:
        return default


def _parse_result_ids(raw: Any, limit: int = 80) -> list[int]:
    if raw is None:
        return []
    if isinstance(raw, np.ndarray):
        raw = raw.tolist()
    if isinstance(raw, list):
        out = []
        for item in raw[: max(1, int(limit))]:
            if isinstance(item, np.ndarray):
                item = item.tolist()
            nid = -1
            if isinstance(item, dict):
                nid = _safe_int(item.get("note_idx"), -1)
            elif isinstance(item, (list, tuple)) and len(item) >= 1:
                if len(item) >= 2 and not isinstance(item[0], (int, float, np.integer, np.floating)):
                    nid = _safe_int(item[1], -1)
                else:
                    nid = _safe_int(item[0], -1)
            else:
                nid = _safe_int(item, -1)
            if nid >= 0:
                out.append(int(nid))
        return list(dict.fromkeys(out))
    return []


def mine(scene: str, tag: str, split: str, topk: int, neg_per_request: int, teacher_results: bool) -> Path:
    group_key = _group_key(scene)
    feat_path = FEATURE_DIR / f"{scene}_{split}_features.parquet"
    recall_path = OUT_DATA_DIR / f"recall_{scene}_{split}_{tag}_multiroute_top1000.parquet"
    out_path = OUT_DATA_DIR / f"hard_neg_{scene}_{split}_{tag}.parquet"

    feat_df = pd.read_parquet(feat_path, columns=[group_key, "note_idx", "y_multi", "click"]).copy()
    feat_df[group_key] = feat_df[group_key].astype(int)
    feat_df["note_idx"] = feat_df["note_idx"].astype(int)
    pos_df = feat_df[(feat_df["click"] > 0) | (feat_df["y_multi"] > 0)].copy()
    pos_map = pos_df.groupby(group_key)["note_idx"].apply(lambda s: set(int(x) for x in s.tolist())).to_dict()

    if teacher_results:
        req_path = _request_dataset_path(scene, split)
        if req_path.exists():
            parquet_cols = set(pq.ParquetFile(req_path).schema.names)
            use_result_col = next((col for col in _request_result_cols(scene) if col in parquet_cols), None)
            if use_result_col is not None:
                req_df = pd.read_parquet(req_path, columns=[group_key, use_result_col]).drop_duplicates(subset=[group_key])
                for row in req_df.itertuples(index=False):
                    gid = int(getattr(row, group_key))
                    pos_map.setdefault(gid, set()).update(_parse_result_ids(getattr(row, use_result_col, None), limit=topk))

    recall_df = pd.read_parquet(recall_path, columns=[group_key, "note_idx", "rank", "recall_score"]).copy()
    recall_df[group_key] = recall_df[group_key].astype(int)
    recall_df["note_idx"] = recall_df["note_idx"].astype(int)
    recall_df = recall_df.sort_values([group_key, "rank", "recall_score"], ascending=[True, True, False], kind="mergesort")

    rows: list[dict[str, Any]] = []
    for gid, sub in recall_df.groupby(group_key, sort=False):
        positives = pos_map.get(int(gid), set())
        negatives: list[int] = []
        for note_idx in sub["note_idx"].tolist()[: max(neg_per_request * 4, topk)]:
            nid = int(note_idx)
            if nid in positives or nid in negatives:
                continue
            negatives.append(nid)
            if len(negatives) >= int(neg_per_request):
                break
        if negatives:
            rows.append({group_key: int(gid), "hard_neg_note_idxs": negatives})

    out_df = pd.DataFrame(rows)
    out_df.to_parquet(out_path, index=False)
    print(f"hard_neg_ready scene={scene} split={split} tag={tag} rows={len(out_df)} path={out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=["search", "rec"], required=True)
    parser.add_argument("--tag", choices=["easy", "hard"], default="easy")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--topk", type=int, default=80)
    parser.add_argument("--neg-per-request", type=int, default=24)
    parser.add_argument("--teacher-results", type=int, default=1)
    args = parser.parse_args()
    mine(args.scene, args.tag, args.split, int(args.topk), int(args.neg_per_request), bool(int(args.teacher_results)))
