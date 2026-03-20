"""
多路召回生成脚本（ANN + Swing + UserCF），输出可用于粗排（LambdaMART）的候选表。

输入:
- 用户请求来自 features/{scene}_{split}_features.parquet
- 索引来自 outputs/index:
    - dssm_{scene}_{tag}_ivfpq.faiss
    - dssm_{scene}_{tag}_row2note.npy
    - dssm_{scene}_{tag}_item_emb.bin / item_meta.json
    - (search可选) dssm_search_{tag}_{split}_query_emb.bin / query_map.json
    - swing_{scene}*_i2i_topk.parquet
    - usercf_{scene}*_u2u_topk.parquet
    - cf_{scene}_train_user_item_index.pkl

融合策略:
- 每一路先各自召回 topk（ANN / Swing / UserCF）
- 各路分数先在请求内做归一化（Min-Max 到 [0,1]），再进入融合，保证量纲可比
- 通过每路最小配额 + 单路占比上限，抑制单路主导，最后按总 topk 截断

输出:
- outputs/data/recall_{scene}_{split}_{tag}_multiroute_topk.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
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

BASE_DIR = Path(__file__).resolve().parents[2]  # Qilin/
FEAT_DIR = BASE_DIR / "features"
INDEX_DIR = BASE_DIR / "outputs" / "index"
OUT_DATA_DIR = BASE_DIR / "outputs" / "data"
OUT_DATA_DIR.mkdir(parents=True, exist_ok=True)

ROW_FLUSH_SIZE = 200_000


def _to_list(x: Any) -> list[int]:
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


def _index_candidates(filename: str) -> list[Path]:
    return [INDEX_DIR / filename]


def _get_parquet_columns(path: Path) -> list[str]:
    """仅读取 parquet schema 列名，避免整表加载造成高内存占用。"""
    if pq is not None:
        return list(pq.ParquetFile(path).schema.names)
    # 兜底：若 pyarrow 不可用，退化为一次头部读取
    return pd.read_parquet(path, columns=None).columns.tolist()


def _load_requests(scene: str, split: str) -> tuple[pd.DataFrame, str]:
    path = FEAT_DIR / f"{scene}_{split}_features.parquet"
    cols = ["user_idx", "recent_clicked_note_idxs", "session_idx"]
    candidate_req_cols = ["request_idx", "search_idx"]
    parquet_cols = _get_parquet_columns(path)
    req_col = next((c for c in candidate_req_cols if c in parquet_cols), "session_idx")
    use_cols = cols + ([req_col] if req_col not in cols else [])
    df = pd.read_parquet(path, columns=use_cols)
    df["recent_clicked_note_idxs"] = df["recent_clicked_note_idxs"].apply(_to_list)
    # 每个请求只保留一行
    req_df = df.drop_duplicates(subset=[req_col]).reset_index(drop=True)
    return req_df, req_col


def _load_swing_index(scene: str) -> dict[int, np.ndarray]:
    """
    优化：返回 numpy array，减少内存和查询开销
    返回格式：{item_idx: np.array([[sim_item_idx, score], ...], dtype=float32)}
    """
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
    # 转为 numpy array，内存更紧凑，查询更快
    return {k: np.array(v, dtype=np.float32) for k, v in out.items()}


def _load_usercf_index(scene: str) -> dict[int, np.ndarray]:
    """
    优化：返回 numpy array 提升性能
    返回格式：{user_idx: np.array([[sim_user_idx, score], ...], dtype=float32)}
    """
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
    """
    优化：仅返回 index 和映射，不加载完整 embedding（按需查询）
    """
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
    # 使用 memmap 延迟加载，避免占用大量内存
    emb = np.memmap(emb_path, dtype="float32", mode="r", shape=(n_items, dim))
    return index, row2note, note2row, emb, dim


def _load_search_query_ann_vecs(tag: str, split: str, dim: int):
    try:
        meta_path = _resolve_existing(_index_candidates(f"dssm_search_{tag}_{split}_query_meta.json"))
        emb_path = _resolve_existing(_index_candidates(f"dssm_search_{tag}_{split}_query_emb.bin"))
        map_path = _resolve_existing(_index_candidates(f"dssm_search_{tag}_{split}_query_map.json"))
    except FileNotFoundError:
        return None, None
    with open(meta_path, "r") as f:
        meta = json.load(f)
    qn = int(meta.get("num_queries", 0))
    qd = int(meta.get("dim", dim))
    if qn <= 0:
        return None, None
    if qd != dim:
        logger.warning(f"Query vec dim mismatch: query_dim={qd}, item_dim={dim}. fallback to history ANN.")
        return None, None
    qemb = np.memmap(emb_path, dtype="float32", mode="r", shape=(qn, qd))
    with open(map_path, "r") as f:
        req2row_raw = json.load(f)
    req2row = {int(k): int(v) for k, v in req2row_raw.items()}
    return qemb, req2row


def _recall_swing(
    hist_items: list[int],
    swing_i2i: dict[int, np.ndarray],
    topk: int,
) -> dict[int, float]:
    """
    优化：使用 numpy 向量化操作，避免重复字典查询
    """
    if not hist_items:
        return {}
    
    scores = {}
    n = len(hist_items)
    
    # 近因加权：越新的历史权重越大
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
    # 使用 heapq.nlargest 优化 TopK，避免全局排序
    import heapq
    return dict(heapq.nlargest(topk, scores.items(), key=lambda x: x[1]))


def _recall_usercf(
    user_idx: int,
    usercf_u2u: dict[int, np.ndarray],
    user_items: dict[int, dict[int, float]],
    topk: int,
) -> dict[int, float]:
    """
    优化：使用 numpy array 和 heapq，避免全局排序
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
    hist_items: list[int],
    ann_index,
    row2note: np.ndarray,
    note2row: dict[int, int],
    item_emb: np.ndarray,
    request_vec: np.ndarray | None,
    topk: int,
):
    q = None
    if request_vec is not None:
        rv = np.asarray(request_vec, dtype=np.float32).reshape(1, -1)
        if rv.shape[1] == item_emb.shape[1] and float(np.linalg.norm(rv)) > 1e-12:
            q = rv
    if q is None:
        rows = [note2row[it] for it in hist_items if it in note2row]
        if not rows:
            return {}
        q = np.asarray(item_emb[rows], dtype=np.float32).mean(axis=0, keepdims=True)
    q_norm = np.linalg.norm(q, axis=1, keepdims=True)
    q = q / np.maximum(q_norm, 1e-12)
    D, I = ann_index.search(q.astype(np.float32), topk)
    scores = {}
    for idx, s in zip(I[0].tolist(), D[0].tolist()):
        if idx < 0 or idx >= len(row2note):
            continue
        scores[int(row2note[idx])] = float(s)
    return scores


def _normalize_route_scores(scores: dict[int, float]) -> dict[int, float]:
    """对单一路由分数做请求内 Min-Max 归一化，统一到 [0,1]。"""
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
    # 显式按归一化分数降序，确保后续轮询取值顺序稳定
    return dict(sorted(norm_scores.items(), key=lambda x: x[1], reverse=True))


def _merge_scores(
    ann_scores: dict[int, float],
    swing_scores: dict[int, float],
    usercf_scores: dict[int, float],
    merge_order: list[str],
    topk: int,
    route_min_quota: int,
    route_max_share: float,
):
    all_routes = ["ann", "swing", "usercf"]
    route2scores = {
        "ann": ann_scores,
        "swing": swing_scores,
        "usercf": usercf_scores,
    }
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
            # 若其它路由还有候选，则该路由暂缓，避免单路主导。
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
            merged.append(
                (
                    item,
                    float(route_score),  # 不加权，首命中路由分作为 recall_score
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

    # Stage-1: 先满足每路最小配额（若该路由有足够 unique 候选）
    # 弱路由优先，避免被 ANN 先占位导致配额失效。
    min_quota = max(0, int(route_min_quota))
    quota_order = [r for r in route_order if r != "ann"] + [r for r in route_order if r == "ann"]
    if min_quota > 0:
        for route in quota_order:
            while len(merged) < topk and route_take_cnt[route] < min_quota:
                if not _append_from_route(route, enforce_cap=False):
                    break

    # Stage-2: 轮询补齐（带单路占比上限约束）
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
    """流式写 parquet，避免将全量召回结果一次性堆在内存中。"""

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
            # 无 pyarrow 时退化为内存拼接（保持可运行性）
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
        # 兜底分支：合并后一次写出
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
        raise ValueError("--merge-order 不能为空")
    invalid = [x for x in merge_order if x not in allowed_routes]
    if invalid:
        raise ValueError(f"--merge-order 含非法路由: {invalid}, allowed={sorted(allowed_routes)}")

    req_df, req_col = _load_requests(scene, split)
    swing_i2i = _load_swing_index(scene)
    usercf_u2u = _load_usercf_index(scene)
    user_items = _load_user_items(scene)
    ann_index, row2note, note2row, item_emb, ann_dim = _load_ann_assets(scene, tag)
    query_ann_emb = None
    req2row = None
    if scene == "search":
        query_ann_emb, req2row = _load_search_query_ann_vecs(tag=tag, split=split, dim=ann_dim)
        if query_ann_emb is None:
            logger.warning("Search query ANN vectors not found. Fallback to history-based ANN query.")

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
    out_path = OUT_DATA_DIR / f"recall_{scene}_{split}_{tag}_multiroute_topk.parquet"
    writer = StreamingParquetWriter(out_path=out_path, columns=output_columns, flush_size=ROW_FLUSH_SIZE)

    for row in tqdm(req_df.itertuples(index=False), total=len(req_df), desc=f"Recall {scene}_{split}"):
        req_id = int(getattr(row, req_col))
        user_idx = int(getattr(row, "user_idx"))
        hist_items = _to_list(getattr(row, "recent_clicked_note_idxs"))
        has_query_vec = scene == "search" and req2row is not None and req_id in req2row
        if (not hist_items) and (not has_query_vec):
            continue

        req_vec = None
        if has_query_vec and query_ann_emb is not None and req2row is not None:
            req_vec = np.asarray(query_ann_emb[req2row[req_id]], dtype=np.float32)
        ann_scores = _recall_ann(
            hist_items,
            ann_index,
            row2note,
            note2row,
            item_emb,
            request_vec=req_vec,
            topk=ann_topk,
        )
        swing_scores = _recall_swing(hist_items, swing_i2i, topk=swing_topk)
        usercf_scores = _recall_usercf(user_idx, usercf_u2u, user_items, topk=usercf_topk)

        # 各路召回分数归一化到同一量纲后再融合
        ann_scores = _normalize_route_scores(ann_scores)
        swing_scores = _normalize_route_scores(swing_scores)
        usercf_scores = _normalize_route_scores(usercf_scores)

        merged = _merge_scores(
            ann_scores=ann_scores,
            swing_scores=swing_scores,
            usercf_scores=usercf_scores,
            merge_order=merge_order,
            topk=topk,
            route_min_quota=route_min_quota,
            route_max_share=route_max_share,
        )
        for rank, (item, s, s_ann, s_sw, s_ucf, f_ann, f_sw, f_ucf, first_route) in enumerate(merged, start=1):
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
    parser.add_argument("--ann-topk", type=int, default=500)
    parser.add_argument("--swing-topk", type=int, default=500)
    parser.add_argument("--usercf-topk", type=int, default=500)
    parser.add_argument("--route-min-quota", type=int, default=100, help="每路最少保留配额（去重后，不足则按实际）")
    parser.add_argument("--route-max-share", type=float, default=0.6, help="单路最大占比上限（其他路由仍有候选时生效）")
    parser.add_argument(
        "--merge-order",
        type=str,
        default="ann,swing,usercf",
        help="融合轮询顺序，逗号分隔，比如 ann,swing,usercf",
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
