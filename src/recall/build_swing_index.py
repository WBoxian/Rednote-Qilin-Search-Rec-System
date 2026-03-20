"""
构建 Swing item-item 索引（带用户权重调整）。

设计:
- 与 UserCF 共用同一个 user-item 索引缓存（cf_shared_index.py）
- 默认按 scene 分开构建（search / rec）
- 相似度使用 Swing 公式（用户权重 w_u = 1/sqrt(|I_u|)）

输出:
- outputs/index/swing_{scene}_i2i_topk.parquet
- outputs/index/swing_{scene}_meta.json

示例:
    uv run src/recall/build_swing_index.py --scene search
    uv run src/recall/build_swing_index.py --scene rec
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from itertools import combinations
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from recall.cf_shared_index import RESULT_DIR, ensure_user_item_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("build_swing_index")


def _trim_user_items(
    user_items_raw: dict[int, dict[int, float]],
    max_items_per_user: int,
):
    user_items = {}
    for u, item_score in user_items_raw.items():
        items = sorted(item_score.items(), key=lambda x: x[1], reverse=True)
        if max_items_per_user > 0:
            items = items[:max_items_per_user]
        user_items[u] = dict(items)
    return user_items


def _build_item_users_from_user_items(user_items: dict[int, dict[int, float]], max_users_per_item: int):
    item_users = defaultdict(dict)
    for u, items in user_items.items():
        for i, s in items.items():
            item_users[i][u] = s

    if max_users_per_item > 0:
        for i in list(item_users.keys()):
            users = sorted(item_users[i].items(), key=lambda x: x[1], reverse=True)[:max_users_per_item]
            item_users[i] = dict(users)
    return dict(item_users)


def build_swing_for_scene(
    scene: str,
    split: str,
    topk: int,
    alpha: float,
    max_items_per_user: int,
    max_users_per_item: int,
    candidate_topn: int,
    min_common_users: int,
    interest_col: str,
    min_interest: float,
    rebuild_ui_index: bool,
):
    ui_index = ensure_user_item_index(
        scene=scene,
        split=split,
        interest_col=interest_col,
        min_interest=min_interest,
        use_cache=True,
        rebuild=rebuild_ui_index,
    )
    user_items = _trim_user_items(ui_index["user_items"], max_items_per_user=max_items_per_user)
    item_users = _build_item_users_from_user_items(user_items, max_users_per_item=max_users_per_item)

    user_item_sets = {u: set(items.keys()) for u, items in user_items.items()}
    user_weight = {u: 1.0 / sqrt(max(len(items), 1)) for u, items in user_item_sets.items()}

    item_ids = sorted(item_users.keys())
    item_neighbors: dict[int, dict[int, float]] = defaultdict(dict)

    for i in tqdm(item_ids, desc=f"Swing {scene}"):
        users_i = set(item_users[i].keys())
        if len(users_i) < min_common_users:
            continue

        # 候选召回：由与 i 相关用户交互过的物品集合组成，再按共现用户数截断
        co_user_count = defaultdict(int)
        for u in users_i:
            for j in user_items[u].keys():
                if j == i:
                    continue
                co_user_count[j] += 1

        candidates = [(j, c) for j, c in co_user_count.items() if c >= min_common_users]
        if not candidates:
            continue
        candidates.sort(key=lambda x: x[1], reverse=True)
        if candidate_topn > 0:
            candidates = candidates[:candidate_topn]

        for j, _ in candidates:
            users_j = set(item_users.get(j, {}).keys())
            common_users = list(users_i & users_j)
            if len(common_users) < min_common_users:
                continue

            sim_ij = 0.0
            # Swing 双重用户求和（u,v）:
            # sum w_u*w_v / (alpha + |Iu ∩ Iv|)
            for u, v in combinations(common_users, 2):
                inter = len(user_item_sets[u] & user_item_sets[v])
                if inter <= 0:
                    continue
                sim_ij += (user_weight[u] * user_weight[v]) / (alpha + inter)

            if sim_ij > 0:
                item_neighbors[i][j] = sim_ij

    rows = []
    for i, neigh in item_neighbors.items():
        top = sorted(neigh.items(), key=lambda x: x[1], reverse=True)[:topk]
        for rank, (j, s) in enumerate(top, start=1):
            rows.append((i, j, float(s), rank))

    out_df = pd.DataFrame(rows, columns=["item_idx", "sim_item_idx", "score", "rank"])
    out_parquet = RESULT_DIR / f"swing_{scene}_{split}_i2i_topk.parquet"
    out_meta = RESULT_DIR / f"swing_{scene}_{split}_meta.json"
    out_df.to_parquet(out_parquet, index=False)
    with open(out_meta, "w") as f:
        json.dump(
            {
                "scene": scene,
                "split": split,
                "topk": topk,
                "alpha": alpha,
                "max_items_per_user": max_items_per_user,
                "max_users_per_item": max_users_per_item,
                "candidate_topn": candidate_topn,
                "min_common_users": min_common_users,
                "interest_col": interest_col,
                "min_interest": min_interest,
                "num_item_rows": int(len(out_df)),
                "num_items_with_neighbors": int(out_df["item_idx"].nunique() if len(out_df) else 0),
            },
            f,
            indent=2,
        )
    logger.info(f"[{scene}] Swing saved: {out_parquet}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=["search", "rec", "all"], default="all")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--max-items-per-user", type=int, default=200)
    parser.add_argument("--max-users-per-item", type=int, default=200)
    parser.add_argument("--candidate-topn", type=int, default=200)
    parser.add_argument("--min-common-users", type=int, default=2)
    parser.add_argument("--interest-col", type=str, default="y_multi")
    parser.add_argument("--min-interest", type=float, default=0.0)
    parser.add_argument("--rebuild-ui-index", action="store_true")
    args = parser.parse_args()

    scenes = ["search", "rec"] if args.scene == "all" else [args.scene]
    for s in scenes:
        build_swing_for_scene(
            scene=s,
            split=args.split,
            topk=args.topk,
            alpha=args.alpha,
            max_items_per_user=args.max_items_per_user,
            max_users_per_item=args.max_users_per_item,
            candidate_topn=args.candidate_topn,
            min_common_users=args.min_common_users,
            interest_col=args.interest_col,
            min_interest=args.min_interest,
            rebuild_ui_index=args.rebuild_ui_index,
        )


# =============================
if __name__ == "__main__":
    main()

