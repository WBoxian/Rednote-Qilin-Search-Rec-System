"""
构建 UserCF user-user 索引（带物品权重调整）。

设计:
- 与 Swing 共用同一个 user-item 索引缓存（cf_shared_index.py）
- 默认按 scene 分开构建（search / rec）
- 相似度采用带热门物品惩罚的余弦：
    sim(u,v) = sum_{l in Iu∩Iv} [ r_ul*r_vl / log(1+n_l) ] / (||u||*||v||)

输出:
- outputs/index/usercf_{scene}_u2u_topk.parquet
- outputs/index/usercf_{scene}_meta.json

示例:
    uv run src/recall/build_usercf_index.py --scene search
    uv run src/recall/build_usercf_index.py --scene rec
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from itertools import combinations
from math import log, sqrt

import pandas as pd
from tqdm import tqdm

from recall.cf_shared_index import RESULT_DIR, ensure_user_item_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("build_usercf_index")


def _trim_user_items(user_items_raw: dict[int, dict[int, float]], max_items_per_user: int):
    user_items = {}
    for u, item_score in user_items_raw.items():
        items = sorted(item_score.items(), key=lambda x: x[1], reverse=True)
        if max_items_per_user > 0:
            items = items[:max_items_per_user]
        user_items[u] = dict(items)
    return user_items


def _build_item_users(user_items: dict[int, dict[int, float]], max_users_per_item: int):
    item_users = defaultdict(dict)
    for u, items in user_items.items():
        for i, s in items.items():
            item_users[i][u] = s

    if max_users_per_item > 0:
        for i in list(item_users.keys()):
            users = sorted(item_users[i].items(), key=lambda x: x[1], reverse=True)[:max_users_per_item]
            item_users[i] = dict(users)
    return dict(item_users)


def build_usercf_for_scene(
    scene: str,
    split: str,
    topk: int,
    max_items_per_user: int,
    max_users_per_item: int,
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
    item_users = _build_item_users(user_items, max_users_per_item=max_users_per_item)

    # 先计算用户范数
    user_norm_sq = defaultdict(float)
    for i, u_scores in item_users.items():
        n_i = len(u_scores)
        if n_i <= 0:
            continue
        item_w = 1.0 / max(log(1.0 + n_i), 1e-12)
        for u, r_ui in u_scores.items():
            user_norm_sq[u] += item_w * float(r_ui) * float(r_ui)

    # 用户对共现分子累计
    pair_num = defaultdict(float)
    for i, u_scores in tqdm(item_users.items(), desc=f"UserCF {scene}"):
        user_list = list(u_scores.items())
        n_i = len(user_list)
        if n_i < 2:
            continue
        item_w = 1.0 / max(log(1.0 + n_i), 1e-12)
        for (u, r_ui), (v, r_vi) in combinations(user_list, 2):
            if u == v:
                continue
            a, b = (u, v) if u < v else (v, u)
            pair_num[(a, b)] += item_w * float(r_ui) * float(r_vi)

    # 归一化为余弦并取 topk
    user_neighbors = defaultdict(dict)
    for (u, v), num in pair_num.items():
        den = sqrt(user_norm_sq[u]) * sqrt(user_norm_sq[v])
        if den <= 1e-12:
            continue
        sim = num / den
        if sim > 0:
            user_neighbors[u][v] = float(sim)
            user_neighbors[v][u] = float(sim)

    rows = []
    for u, neigh in user_neighbors.items():
        top = sorted(neigh.items(), key=lambda x: x[1], reverse=True)[:topk]
        for rank, (v, s) in enumerate(top, start=1):
            rows.append((u, v, float(s), rank))

    out_df = pd.DataFrame(rows, columns=["user_idx", "sim_user_idx", "score", "rank"])
    out_parquet = RESULT_DIR / f"usercf_{scene}_u2u_topk.parquet"
    out_meta = RESULT_DIR / f"usercf_{scene}_meta.json"
    out_df.to_parquet(out_parquet, index=False)
    with open(out_meta, "w") as f:
        json.dump(
            {
                "scene": scene,
                "split": split,
                "topk": topk,
                "max_items_per_user": max_items_per_user,
                "max_users_per_item": max_users_per_item,
                "interest_col": interest_col,
                "min_interest": min_interest,
                "num_rows": int(len(out_df)),
                "num_users_with_neighbors": int(out_df["user_idx"].nunique() if len(out_df) else 0),
            },
            f,
            indent=2,
        )
    logger.info(f"[{scene}] UserCF saved: {out_parquet}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=["search", "rec", "all"], default="all")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--max-items-per-user", type=int, default=200)
    parser.add_argument("--max-users-per-item", type=int, default=300)
    parser.add_argument("--interest-col", type=str, default="y_multi")
    parser.add_argument("--min-interest", type=float, default=0.0)
    parser.add_argument("--rebuild-ui-index", action="store_true")
    args = parser.parse_args()

    scenes = ["search", "rec"] if args.scene == "all" else [args.scene]
    for s in scenes:
        build_usercf_for_scene(
            scene=s,
            split=args.split,
            topk=args.topk,
            max_items_per_user=args.max_items_per_user,
            max_users_per_item=args.max_users_per_item,
            interest_col=args.interest_col,
            min_interest=args.min_interest,
            rebuild_ui_index=args.rebuild_ui_index,
        )


# =============================
if __name__ == "__main__":
    main()

