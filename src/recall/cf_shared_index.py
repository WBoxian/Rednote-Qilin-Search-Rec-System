"""
Swing / UserCF 共享的 user-item 索引构建与缓存工具。

输出:
- outputs/index/cf_{scene}_{split}_user_item_index.pkl
- outputs/index/cf_{scene}_{split}_user_item_index.meta.json

示例:
    uv run src/recall/cf_shared_index.py --scene search
    uv run src/recall/cf_shared_index.py --scene rec
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]  # Qilin/
FEAT_DIR = BASE_DIR / "features"
RESULT_DIR = BASE_DIR / "outputs" / "index"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def _cache_paths(scene: str, split: str):
    prefix = f"cf_{scene}_{split}_user_item_index"
    return (
        RESULT_DIR / f"{prefix}.pkl",
        RESULT_DIR / f"{prefix}.meta.json",
    )


def _load_scene_df(scene: str, split: str, cols: list[str]) -> pd.DataFrame:
    path = FEAT_DIR / f"{scene}_{split}_features.parquet"
    return pd.read_parquet(path, columns=cols)


def ensure_user_item_index(
    scene: str,
    split: str = "train",
    interest_col: str = "y_multi",
    min_interest: float = 0.0,
    use_cache: bool = True,
    rebuild: bool = False,
):
    """
    构建并缓存 user-item 索引:
      - user_items: user -> {item: score}
      - item_users: item -> {user: score}
    score 默认来自 y_multi（若不存在则回退 click）。
    """
    cache_pkl, cache_meta = _cache_paths(scene, split)
    if use_cache and (not rebuild) and cache_pkl.exists():
        with open(cache_pkl, "rb") as f:
            return pickle.load(f)

    cols = ["user_idx", "note_idx", "click"]
    if interest_col not in cols:
        cols.append(interest_col)
    df = _load_scene_df(scene, split, cols=cols)

    score_col = interest_col if interest_col in df.columns else "click"
    df[score_col] = df[score_col].fillna(0.0).astype(float)
    df["score"] = np.clip(df[score_col].to_numpy(dtype=float), a_min=0.0, a_max=None)
    df = df[df["score"] > min_interest].copy()

    # 兜底：如果 y_multi 全部<=0，回退到 click=1
    if len(df) == 0:
        df = _load_scene_df(scene, split, cols=["user_idx", "note_idx", "click"])
        df["score"] = df["click"].astype(float)
        df = df[df["score"] > 0]

    # 合并重复曝光，取同一 user-item 的最大兴趣分
    grp = (
        df.groupby(["user_idx", "note_idx"], as_index=False)["score"]
        .max()
        .reset_index(drop=True)
    )

    user_items_dd: dict[int, dict[int, float]] = defaultdict(dict)
    item_users_dd: dict[int, dict[int, float]] = defaultdict(dict)
    users = grp["user_idx"].to_numpy(dtype=np.int64)
    items = grp["note_idx"].to_numpy(dtype=np.int64)
    scores = grp["score"].to_numpy(dtype=np.float32)

    for u, i, s in zip(users, items, scores):
        u_i = int(u)
        i_i = int(i)
        s_f = float(s)
        user_items_dd[u_i][i_i] = s_f
        item_users_dd[i_i][u_i] = s_f

    user_items = dict(user_items_dd)
    item_users = dict(item_users_dd)
    index = {
        "scene": scene,
        "split": split,
        "score_col": score_col,
        "min_interest": float(min_interest),
        "user_items": user_items,
        "item_users": item_users,
    }

    if use_cache:
        with open(cache_pkl, "wb") as f:
            pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(cache_meta, "w") as f:
            json.dump(
                {
                    "scene": scene,
                    "split": split,
                    "score_col": score_col,
                    "min_interest": float(min_interest),
                    "num_users": len(user_items),
                    "num_items": len(item_users),
                    "num_edges": int(len(grp)),
                },
                f,
                indent=2,
            )

    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=["search", "rec"], required=True)
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--interest-col", type=str, default="y_multi")
    parser.add_argument("--min-interest", type=float, default=0.0)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    index = ensure_user_item_index(
        scene=args.scene,
        split=args.split,
        interest_col=args.interest_col,
        min_interest=args.min_interest,
        use_cache=not args.no_cache,
        rebuild=args.rebuild,
    )
    cache_pkl, cache_meta = _cache_paths(args.scene, args.split)
    print(
        f"[OK] scene={args.scene} split={args.split} "
        f"users={len(index['user_items'])} items={len(index['item_users'])} "
        f"score_col={index['score_col']}\n"
        f"pkl={cache_pkl}\nmeta={cache_meta}"
    )


# =============================
if __name__ == "__main__":
    main()
