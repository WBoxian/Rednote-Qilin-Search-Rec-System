"""构建推荐场景的 sequence transition i2i 索引。

目标：
- 用 train 特征里的 recent_clicked_note_idxs -> 当前正样本 note_idx
- 学到更接近 next-item recommendation 的顺序迁移关系
- 作为 rec 召回的显式顺序路由，补足纯语义 ANN 的泛化不足
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
FEAT_DIR = BASE_DIR / "features"
INDEX_DIR = BASE_DIR / "outputs" / "index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

MAX_HIST_LEN = 20
DEFAULT_TOPK = 200


def _to_hist_list(raw) -> list[int]:
    if isinstance(raw, np.ndarray):
        return [int(x) for x in raw.tolist() if int(x) >= 0]
    if isinstance(raw, list):
        return [int(x) for x in raw if int(x) >= 0]
    return []


def build_seq_transition_for_scene(
    scene: str = "rec",
    split: str = "train",
    topk: int = DEFAULT_TOPK,
    max_hist_len: int = MAX_HIST_LEN,
) -> Path:
    if scene != "rec":
        raise ValueError("sequence transition index 目前只用于 rec 场景")
    feat_path = FEAT_DIR / f"{scene}_{split}_features.parquet"
    cols = ["recent_clicked_note_idxs", "note_idx", "y_multi"]
    df = pd.read_parquet(feat_path, columns=cols)
    df = df[df["y_multi"].fillna(0.0) > 0].copy()

    transitions: dict[int, dict[int, float]] = defaultdict(dict)
    for row in df.itertuples(index=False):
        target = int(row.note_idx)
        if target < 0:
            continue
        hist_items = _to_hist_list(row.recent_clicked_note_idxs)
        if not hist_items:
            continue
        hist_items = hist_items[-max(1, int(max_hist_len)) :]
        label_gain = float(max(0.0, float(row.y_multi)))
        label_w = 1.0 + min(2.5, 0.22 * label_gain)
        for idx, src in enumerate(reversed(hist_items)):
            src = int(src)
            if src < 0 or src == target:
                continue
            recency_w = 1.0 / (1.0 + 0.32 * idx)
            if idx < 3:
                recency_w *= 1.18
            score = label_w * recency_w
            bucket = transitions[src]
            bucket[target] = bucket.get(target, 0.0) + float(score)

    rows: list[tuple[int, int, float]] = []
    for src, dst_map in transitions.items():
        ranked = sorted(dst_map.items(), key=lambda x: x[1], reverse=True)[: max(1, int(topk))]
        for dst, score in ranked:
            rows.append((int(src), int(dst), float(score)))

    out_path = INDEX_DIR / f"seqtrans_{scene}_{split}_i2i_topk.parquet"
    out_df = pd.DataFrame(rows, columns=["item_idx", "sim_item_idx", "score"])
    out_df.to_parquet(out_path, index=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=["rec"], default="rec")
    parser.add_argument("--split", choices=["train"], default="train")
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    parser.add_argument("--max-hist-len", type=int, default=MAX_HIST_LEN)
    args = parser.parse_args()
    out = build_seq_transition_for_scene(
        scene=args.scene,
        split=args.split,
        topk=args.topk,
        max_hist_len=args.max_hist_len,
    )
    print(out)


if __name__ == "__main__":
    main()
