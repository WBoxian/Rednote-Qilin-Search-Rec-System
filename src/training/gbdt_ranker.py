"""
Qilin GBDT 粗排训练（Search & Rec）
- LightGBM + XGBoost (LambdaMART)
- 默认按请求分组 Train/Val 切分（search_idx / request_idx）
- 默认使用召回候选训练：候选ID与曝光特征做半连接
- 训练完成后导出 DIEN 精排训练集（粗排 topN）
- easy 模式导出 DSSM hard neg：召回输入集合 - 粗排 topN 补集

使用示例:
    uv run python src/training/gbdt_ranker.py --scene search
    uv run python src/training/gbdt_ranker.py --scene rec
    nohup uv run tensorboard --logdir=outputs/logs > /dev/null 2>&1 &   # 默认端口 6006
"""

from __future__ import annotations

import argparse
import gc
import random
from datetime import datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from torch.utils.tensorboard import SummaryWriter

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

try:
    from .utils import (
        build_group_sizes,
        discretize_relevance,
        eval_ndcg_by_group,
        sort_by_group,
        unsort,
    )
except ImportError:
    from utils import (
        build_group_sizes,
        discretize_relevance,
        eval_ndcg_by_group,
        sort_by_group,
        unsort,
    )

# =============================
# Global config
# =============================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

TOPK = 10
NUM_BOOST_ROUND = 2000
EARLY_STOP = 10
PRINT_EVERY = 10
STREAM_BATCH = 200_000  # PyArrow iter_batches 每批行数
VALID_RATIO = 0.2
HARD_NEG_KEEP_TOPK = 10

BASE_DIR = Path(__file__).resolve().parents[2]  # Qilin/
FEATURE_DIR = BASE_DIR / "features"
OUT_DIR = BASE_DIR / "outputs"
OUT_DATA_DIR = OUT_DIR / "data"

OUT_DIR.mkdir(exist_ok=True)
(OUT_DIR / "models").mkdir(exist_ok=True)
(OUT_DIR / "results").mkdir(exist_ok=True)
(OUT_DIR / "logs").mkdir(exist_ok=True)
OUT_DATA_DIR.mkdir(parents=True, exist_ok=True)

# =============================
# Params
# =============================
lgb_params = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [TOPK],
    "learning_rate": 0.01,
    "num_leaves": 31,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 2.0,
    "verbosity": -1,
    "seed": SEED,
}

xgb_params = {
    "objective": "rank:ndcg",
    "eval_metric": f"ndcg@{TOPK}",
    "eta": 0.01,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 20,
    "lambda": 2.0,
    "tree_method": "hist",
    "seed": SEED,
}


def _group_key(scene: str) -> str:
    return "search_idx" if scene == "search" else "request_idx"


def _default_candidate_path(scene: str, recall_tag: str) -> Path:
    return OUT_DATA_DIR / f"recall_{scene}_train_{recall_tag}_multiroute_topk.parquet"


def _df_mem_mb(df: pd.DataFrame) -> float:
    if df is None or len(df) == 0:
        return 0.0
    return float(df.memory_usage(deep=True).sum() / (1024 * 1024))


def _feature_columns_for_gbdt(scene: str) -> list[str]:
    group_key = _group_key(scene)
    train_path = FEATURE_DIR / f"{scene}_train_features.parquet"
    cols = list(pq.read_schema(train_path).names)
    drop_heavy = {"recent_clicked_note_idxs"}
    keep = [c for c in cols if c not in drop_heavy]
    must_have = [group_key, "note_idx", "click", "y_multi", "user_idx"]
    for c in must_have:
        if c in cols and c not in keep:
            keep.append(c)
    return keep


def _build_training_df_from_candidates(
    scene: str,
    candidate_path: Path,
    train_candidate_topn: int,
) -> pd.DataFrame:
    """流式半连接：候选表 × 特征表 → 全程落盘中间结果，峰值内存仅为单批大小。"""
    group_key = _group_key(scene)
    train_path = FEATURE_DIR / f"{scene}_train_features.parquet"
    feature_cols = _feature_columns_for_gbdt(scene)

    if not train_path.exists():
        raise FileNotFoundError(f"Missing {train_path}. Please build features first")
    if not candidate_path.exists():
        raise FileNotFoundError(
            f"Missing recall candidate file: {candidate_path}. "
            "Please run multiroute stage before GBDT training."
        )

    _meta_want = [
        "rank", "recall_score",
        "score_ann", "score_swing", "score_usercf",
        "from_ann", "from_swing", "from_usercf",
        "first_route",
    ]

    # ---- 1) 流式读取候选表，边读边过滤 rank，落盘轻量临时文件 ----
    pf_cand = pq.ParquetFile(candidate_path)
    avail_cand = set(pf_cand.schema_arrow.names)
    read_cols = [c for c in [group_key, "note_idx"] + _meta_want if c in avail_cand]
    meta_cols = [c for c in _meta_want if c in avail_cand]

    tmp_cand = OUT_DATA_DIR / f"_tmp_{scene}_cand.parquet"
    pw_cand: pq.ParquetWriter | None = None
    n_cand_raw = 0
    try:
        for batch in pf_cand.iter_batches(batch_size=STREAM_BATCH, columns=read_cols):
            tbl = pa.Table.from_batches([batch])
            n_cand_raw += tbl.num_rows
            if "rank" in tbl.column_names and train_candidate_topn > 0:
                tbl = tbl.filter(pc.less_equal(tbl["rank"], train_candidate_topn))
            if tbl.num_rows == 0:
                continue
            if pw_cand is None:
                pw_cand = pq.ParquetWriter(str(tmp_cand), tbl.schema)
            pw_cand.write_table(tbl)
    finally:
        if pw_cand is not None:
            pw_cand.close()

    # 读回已过滤的候选（rank 截断后数据量大幅缩小），再做去重
    cand = pd.read_parquet(tmp_cand)
    cand.drop_duplicates(subset=[group_key, "note_idx"], inplace=True)
    cand.reset_index(drop=True, inplace=True)
    tmp_cand.unlink(missing_ok=True)
    print(f"[Cand] raw={n_cand_raw} -> filtered+dedup={len(cand)}, mem={_df_mem_mb(cand):.2f}MB")

    # ---- 2) 构建 (group_key, note_idx) 复合键有序数组，供 searchsorted O(n log m) 过滤 ----
    cand_gk = cand[group_key].to_numpy(dtype=np.int64)
    cand_nid = cand["note_idx"].to_numpy(dtype=np.int64)
    note_shift = int(cand_nid.max()) + 1
    cand_composite = cand_gk * note_shift + cand_nid
    sort_idx = cand_composite.argsort()
    cand_composite = cand_composite[sort_idx]
    del cand_gk, cand_nid, sort_idx

    # ---- 3) 流式读特征表，逐批 searchsorted 匹配，命中行落盘临时 parquet ----
    tmp_feat = OUT_DATA_DIR / f"_tmp_{scene}_gbdt_joined.parquet"
    pf_feat = pq.ParquetFile(train_path)
    pw_feat: pq.ParquetWriter | None = None
    n_feat_scanned, n_matched = 0, 0
    try:
        for batch in pf_feat.iter_batches(batch_size=STREAM_BATCH, columns=feature_cols):
            chunk = batch.to_pandas()
            n_feat_scanned += len(chunk)
            comp = (
                chunk[group_key].to_numpy(dtype=np.int64) * note_shift
                + chunk["note_idx"].to_numpy(dtype=np.int64)
            )
            pos = np.searchsorted(cand_composite, comp)
            np.clip(pos, 0, len(cand_composite) - 1, out=pos)
            mask = cand_composite[pos] == comp
            matched = chunk.loc[mask]
            if len(matched) == 0:
                del chunk
                continue
            out_tbl = pa.Table.from_pandas(matched, preserve_index=False)
            if pw_feat is None:
                pw_feat = pq.ParquetWriter(str(tmp_feat), out_tbl.schema)
            pw_feat.write_table(out_tbl)
            n_matched += len(matched)
            del chunk, matched, out_tbl
    finally:
        if pw_feat is not None:
            pw_feat.close()
    del cand_composite
    gc.collect()
    print(f"[Scan] feature rows scanned={n_feat_scanned}, matched={n_matched}")

    # ---- 4) 读回已筛选的特征数据，附加候选元信息 ----
    if n_matched == 0:
        print("[Warn] no rows matched between candidates and features.")
        return pd.DataFrame(columns=feature_cols + meta_cols)

    train_df = pd.read_parquet(tmp_feat)
    tmp_feat.unlink(missing_ok=True)

    if meta_cols:
        cand_meta = cand[[group_key, "note_idx"] + meta_cols]
        train_df = train_df.merge(cand_meta, on=[group_key, "note_idx"], how="left")
    del cand
    gc.collect()

    num_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        train_df[num_cols] = train_df[num_cols].fillna(0)

    print(f"[Load] matched={n_matched}, shape={train_df.shape}, mem={_df_mem_mb(train_df):.2f}MB")
    return train_df


def _prepare_training_frame(df: pd.DataFrame, scene: str) -> pd.DataFrame:
    group_key = _group_key(scene)

    if "first_route" in df.columns:
        route_id_map = {"ann": 1, "swing": 2, "usercf": 3}
        df["first_route_id"] = (
            df["first_route"].astype(str).str.lower().map(route_id_map).fillna(0).astype(np.int16)
        )

    before_groups = int(df[group_key].nunique())
    # 向量化过滤：避免 groupby().filter(lambda) 的逐组 Python 调用
    g = df.groupby(group_key, sort=False)
    valid = (g[group_key].transform("size") > 1) & (g["y_multi"].transform("max") > 0.0)
    df = df.loc[valid].reset_index(drop=True)
    after_groups = int(df[group_key].nunique())

    # 降精度降低训练峰值内存
    float_cols = df.select_dtypes(include=["float64"]).columns.tolist()
    if float_cols:
        df[float_cols] = df[float_cols].astype(np.float32)
    int_cols = [c for c in df.select_dtypes(include=["int64"]).columns.tolist() if c not in {group_key, "note_idx", "user_idx"}]
    if int_cols:
        df[int_cols] = df[int_cols].astype(np.int32)

    print(f"[Group Filter] groups: {before_groups} -> {after_groups}, rows={len(df)}")
    return df


def _downsample_negatives_for_gbdt(
    df: pd.DataFrame,
    group_key: str,
    max_neg_per_req: int,
) -> pd.DataFrame:
    if max_neg_per_req <= 0:
        return df

    sampled_parts: list[pd.DataFrame] = []
    for gid, g in df.groupby(group_key, sort=False):
        pos = g[g["y_multi"] > 0]
        neg = g[g["y_multi"] <= 0]
        if len(neg) <= max_neg_per_req:
            sampled_parts.append(g)
            continue

        if "rank" in g.columns:
            neg_keep = neg.nsmallest(max_neg_per_req, "rank")
        elif "recall_score" in g.columns:
            neg_keep = neg.nlargest(max_neg_per_req, "recall_score")
        else:
            try:
                gid_int = int(gid)
            except Exception:
                gid_int = abs(hash(gid))
            neg_keep = neg.sample(n=max_neg_per_req, random_state=SEED + (gid_int % 1000003))

        sampled_parts.append(pd.concat([pos, neg_keep], axis=0))

    if not sampled_parts:
        return df.iloc[:0].copy()
    out = pd.concat(sampled_parts, axis=0, ignore_index=True)
    if "rank" in out.columns:
        out = out.sort_values([group_key, "rank"], ascending=[True, True], kind="mergesort").reset_index(drop=True)
    return out


def _build_coarse_scored_frame(
    df: pd.DataFrame,
    group_key: str,
    oof_lgb: np.ndarray,
    oof_xgb: np.ndarray,
) -> pd.DataFrame:
    out = df.copy()
    out["lgb_score"] = oof_lgb
    out["xgb_score"] = oof_xgb
    out["coarse_score"] = 0.5 * (oof_lgb + oof_xgb)
    out.sort_values([group_key, "coarse_score"], ascending=[True, False], kind="mergesort", inplace=True)
    out["coarse_rank"] = out.groupby(group_key).cumcount() + 1
    return out


def _group_val_split(group: np.ndarray, valid_ratio: float = VALID_RATIO) -> tuple[np.ndarray, np.ndarray]:
    unique_groups = np.unique(group)
    if len(unique_groups) < 2:
        raise ValueError(f"Not enough groups for val split: group_cnt={len(unique_groups)}")

    rng = np.random.RandomState(SEED)
    shuffled = unique_groups.copy()
    rng.shuffle(shuffled)

    val_cnt = int(round(len(shuffled) * float(valid_ratio)))
    val_cnt = max(1, min(val_cnt, len(shuffled) - 1))
    val_groups = set(shuffled[:val_cnt].tolist())

    val_mask = np.isin(group, list(val_groups))
    val_idx = np.where(val_mask)[0]
    tr_idx = np.where(~val_mask)[0]
    if len(tr_idx) == 0 or len(val_idx) == 0:
        raise ValueError("invalid val split: empty train or valid index")
    return tr_idx, val_idx


def _export_dien_dataset(
    coarse_scored: pd.DataFrame,
    scene: str,
    output_tag: str,
    group_key: str,
    topn: int,
    keep_positive: bool,
) -> tuple[Path, Path]:
    out = coarse_scored.copy()

    # 导出粗排全量打分结果，供 hard neg 挖掘（coarse 淘汰集合）
    full_cols = [
        group_key,
        "note_idx",
        "click",
        "y_multi",
        "coarse_score",
        "coarse_rank",
        "lgb_score",
        "xgb_score",
    ]
    full_cols = [c for c in full_cols if c in out.columns]
    tag_suffix = f"_{output_tag}" if output_tag else ""
    full_scored_path = OUT_DATA_DIR / f"coarse_{scene}{tag_suffix}_train_scored_full.parquet"
    out[full_cols].to_parquet(full_scored_path, index=False)
    print(f"[Export] coarse full scored set: {full_scored_path}, shape={out[full_cols].shape}")

    if topn > 0:
        mask = out["coarse_rank"] <= topn
        if keep_positive and "click" in out.columns:
            mask = mask | (out["click"] > 0)
        out = out[mask].copy()

    out = out.groupby(group_key).filter(lambda x: len(x) > 1 and float(x["y_multi"].max()) > 0.0).reset_index(drop=True)
    out_path = OUT_DATA_DIR / f"dien_{scene}{tag_suffix}_train_from_gbdt_top{topn}.parquet"
    out.to_parquet(out_path, index=False)
    print(f"[Export] DIEN train set: {out_path}, shape={out.shape}")
    return out_path, full_scored_path


def _export_easy_coarse_complement_hard_neg(
    scene: str,
    output_tag: str,
    group_key: str,
    candidate_path: Path | None,
    coarse_scored: pd.DataFrame,
    coarse_topn: int,
    hard_neg_input_topn: int,
    hard_neg_per_req: int,
) -> Path | None:
    if output_tag != "easy":
        print(f"[HardNeg Export] skip: output_tag={output_tag} (only export in easy mode)")
        return None
    if candidate_path is None or not candidate_path.exists():
        print(f"[HardNeg Export] skip: candidate file missing: {candidate_path}")
        return None

    cand_cols = [group_key, "note_idx", "rank"]
    try:
        cand = pd.read_parquet(candidate_path, columns=cand_cols)
    except Exception:
        cand_all = pd.read_parquet(candidate_path)
        use_cols = [c for c in cand_cols if c in cand_all.columns]
        cand = cand_all[use_cols].copy()

    if group_key not in cand.columns or "note_idx" not in cand.columns:
        print(f"[HardNeg Export] skip: candidate file missing required cols, file={candidate_path}")
        return None

    cand = cand.drop_duplicates(subset=[group_key, "note_idx"]).copy()
    if "rank" in cand.columns and hard_neg_input_topn > 0:
        cand = cand[cand["rank"] <= hard_neg_input_topn].copy()
    if len(cand) == 0:
        print("[HardNeg Export] skip: no candidate rows after rank filter")
        return None

    keep_pairs = coarse_scored[coarse_scored["coarse_rank"] <= max(0, int(coarse_topn))][[group_key, "note_idx"]]
    keep_pairs = keep_pairs.drop_duplicates(subset=[group_key, "note_idx"]).copy()

    anti = cand.merge(keep_pairs, on=[group_key, "note_idx"], how="left", indicator=True)
    hard = anti[anti["_merge"] == "left_only"].copy()
    if len(hard) == 0:
        print("[HardNeg Export] skip: empty complement set (recall_input - coarse_topN)")
        return None

    if "rank" in hard.columns:
        hard.sort_values([group_key, "rank"], ascending=[True, True], kind="mergesort", inplace=True)
    per_req_limit = HARD_NEG_KEEP_TOPK
    if hard_neg_per_req > 0:
        per_req_limit = min(int(hard_neg_per_req), HARD_NEG_KEEP_TOPK)
    hard = hard.groupby(group_key, sort=False).head(per_req_limit).copy()

    req_hard = (
        hard.groupby(group_key)["note_idx"]
        .apply(lambda x: [int(v) for v in x.tolist()])
        .reset_index(name="hard_neg_note_idxs")
    )

    out_path = OUT_DATA_DIR / f"dssm_hard_neg_{scene}.parquet"
    req_hard.to_parquet(out_path, index=False)
    cov = float((req_hard["hard_neg_note_idxs"].apply(len) > 0).mean()) if len(req_hard) > 0 else 0.0
    avg_len = float(req_hard["hard_neg_note_idxs"].apply(len).mean()) if len(req_hard) > 0 else 0.0
    print(
        f"[HardNeg Export] path={out_path}, req_rows={len(req_hard)}, "
        f"coverage={cov:.4f}, avg_list_len={avg_len:.2f}, "
        f"input_topn={hard_neg_input_topn}, coarse_topn={coarse_topn}, per_req_limit={per_req_limit}"
    )
    return out_path


def main(
    scene: str,
    output_tag: str,
    candidate_path: Path | None,
    recall_tag: str,
    train_candidate_topn: int,
    dien_topn: int,
    keep_positive_for_dien: bool,
    skip_export_dien: bool,
    hard_neg_input_topn: int,
    hard_neg_per_req: int,
    skip_export_hard_neg: bool,
    print_every: int,
    train_max_neg_per_req: int,
):
    assert scene in ["search", "rec"]
    if not output_tag and recall_tag in {"easy", "hard"}:
        output_tag = recall_tag
    group_key = _group_key(scene)

    if candidate_path is None:
        candidate_path = _default_candidate_path(scene, recall_tag)
    df = _build_training_df_from_candidates(
        scene=scene,
        candidate_path=candidate_path,
        train_candidate_topn=train_candidate_topn,
    )

    df = _prepare_training_frame(df, scene)
    if train_max_neg_per_req > 0:
        before_rows = len(df)
        df = _downsample_negatives_for_gbdt(
            df=df,
            group_key=group_key,
            max_neg_per_req=train_max_neg_per_req,
        )
        print(f"[Train Sampling] max_neg_per_req={train_max_neg_per_req}, rows: {before_rows} -> {len(df)}")
    if len(df) == 0:
        raise ValueError("No training rows left after recall/group filtering.")

    group = df[group_key].to_numpy()
    y_cont = df["y_multi"].to_numpy()
    y_disc = discretize_relevance(y_cont)

    drop_cols = ["click", "y_multi", "note_idx", group_key, "recent_clicked_note_idxs", "first_route"]
    drop_cols = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=drop_cols)
    X = X.astype(np.float32, copy=False)
    print(f"[Mem] feature matrix shape={X.shape}, mem={_df_mem_mb(X):.2f}MB")

    tr_idx, val_idx = _group_val_split(group=group, valid_ratio=VALID_RATIO)
    print(f"[Split] val by group: train={len(tr_idx)}, valid={len(val_idx)}, valid_ratio={VALID_RATIO}")

    tag_suffix = f"_{output_tag}" if output_tag else ""
    log_dir = OUT_DIR / "logs" / f"gbdt_{scene}{tag_suffix}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"[TB] logs at: {log_dir}")

    # =============================
    # LightGBM
    # =============================
    print(f"\n[Train] LightGBM LambdaRank ({scene})")
    oof_lgb = np.zeros(len(df), dtype=np.float32)

    X_tr, y_tr, g_tr = X.iloc[tr_idx], y_disc[tr_idx], group[tr_idx]
    X_val, y_val, g_val = X.iloc[val_idx], y_disc[val_idx], group[val_idx]

    X_tr, y_tr, g_tr, _ = sort_by_group(X_tr, y_tr, g_tr)
    X_val, y_val, g_val, order_val = sort_by_group(X_val, y_val, g_val)

    dtr = lgb.Dataset(X_tr, y_tr, group=build_group_sizes(g_tr))
    dva = lgb.Dataset(X_val, y_val, group=build_group_sizes(g_val))

    def lgb_tb_callback(env):
        metric_parts = []
        for item in env.evaluation_result_list:
            dataset_name = item[0]
            metric_name = item[1]
            tag = f"LGB/{metric_name}"
            writer.add_scalars(tag, {dataset_name: item[2]}, env.iteration)
            metric_parts.append(f"{dataset_name}:{metric_name}={item[2]:.6f}")
        if (env.iteration + 1) % max(1, int(print_every)) == 0 and metric_parts:
            print(f"[LGB][Iter {env.iteration + 1}] " + " | ".join(metric_parts))

    lgb_model = lgb.train(
        lgb_params,
        dtr,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dtr, dva],
        valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOP), lgb_tb_callback],
    )

    lgb_val_pred = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
    lgb_val_pred = unsort(lgb_val_pred, order_val)
    oof_lgb[val_idx] = lgb_val_pred

    g_val_raw = unsort(g_val, order_val)
    lgb_nd10 = eval_ndcg_by_group(y_disc[val_idx], lgb_val_pred, g_val_raw, 10)
    lgb_nd100 = eval_ndcg_by_group(y_disc[val_idx], lgb_val_pred, g_val_raw, 100)
    writer.add_scalar("OOF_NDCG10/LGB", lgb_nd10, 1)
    joblib.dump(lgb_model, OUT_DIR / f"models/lgb_{scene}{tag_suffix}.pkl")
    print(f"[LGB Val] NDCG@10={lgb_nd10:.4f} NDCG@100={lgb_nd100:.4f}")

    np.save(OUT_DIR / f"results/lgb_{scene}{tag_suffix}.npy", oof_lgb)

    # =============================
    # XGBoost
    # =============================
    print(f"\n[Train] XGBoost Rank ({scene})")
    oof_xgb = np.zeros(len(df), dtype=np.float32)

    dtr = xgb.DMatrix(X_tr.to_numpy(), label=y_tr)
    dva = xgb.DMatrix(X_val.to_numpy(), label=y_val)
    dtr.set_group(build_group_sizes(g_tr))
    dva.set_group(build_group_sizes(g_val))

    class XGBTensorBoardCB(xgb.callback.TrainingCallback):
        def __init__(self, summary_writer):
            super().__init__()
            self.writer = summary_writer

        def after_iteration(self, _model, epoch, evals_log):
            _ = _model
            metrics_found = {}
            for data_name, metrics in evals_log.items():
                for metric_name, log_values in metrics.items():
                    metrics_found.setdefault(metric_name, {})
                    metrics_found[metric_name][data_name] = log_values[-1]
            for metric_name, val_dict in metrics_found.items():
                self.writer.add_scalars(f"XGB/{metric_name}", val_dict, epoch)

            if (epoch + 1) % max(1, int(print_every)) == 0 and metrics_found:
                parts = []
                for metric_name, val_dict in metrics_found.items():
                    for data_name, value in val_dict.items():
                        parts.append(f"{data_name}:{metric_name}={value:.6f}")
                print(f"[XGB][Iter {epoch + 1}] " + " | ".join(parts))
            return False

    xgb_model = xgb.train(
        xgb_params,
        dtr,
        num_boost_round=NUM_BOOST_ROUND,
        evals=[(dtr, "train"), (dva, "valid")],
        early_stopping_rounds=EARLY_STOP,
        verbose_eval=False,
        callbacks=[XGBTensorBoardCB(summary_writer=writer)],
    )

    best_iter = getattr(xgb_model, "best_iteration", None)
    if best_iter is None or int(best_iter) < 0:
        xgb_val_pred = xgb_model.predict(dva)
    else:
        xgb_val_pred = xgb_model.predict(dva, iteration_range=(0, int(best_iter) + 1))
    xgb_val_pred = unsort(xgb_val_pred, order_val)
    oof_xgb[val_idx] = xgb_val_pred

    xgb_nd10 = eval_ndcg_by_group(y_disc[val_idx], xgb_val_pred, g_val_raw, 10)
    xgb_nd100 = eval_ndcg_by_group(y_disc[val_idx], xgb_val_pred, g_val_raw, 100)
    writer.add_scalar("OOF_NDCG10/XGB", xgb_nd10, 1)
    joblib.dump(xgb_model, OUT_DIR / f"models/xgb_{scene}{tag_suffix}.pkl")
    print(f"[XGB Val] NDCG@10={xgb_nd10:.4f} NDCG@100={xgb_nd100:.4f}")

    np.save(OUT_DIR / f"results/xgb_{scene}{tag_suffix}.npy", oof_xgb)

    # =============================
    # Summary & Export
    # =============================
    lgb_m10 = float(lgb_nd10)
    xgb_m10 = float(xgb_nd10)
    print("\n[Summary]")
    print(f"LGB Mean NDCG@10={lgb_m10:.4f}")
    print(f"XGB Mean NDCG@10={xgb_m10:.4f}")

    writer.add_text("Summary/Final_NDCG10", f"LGB: {lgb_m10:.4f}, XGB: {xgb_m10:.4f}")

    writer.close()

    lgb_iter_for_pred = getattr(lgb_model, "best_iteration", None)
    if lgb_iter_for_pred is None or int(lgb_iter_for_pred) <= 0:
        full_lgb_pred = np.asarray(lgb_model.predict(X), dtype=np.float32)
    else:
        full_lgb_pred = np.asarray(lgb_model.predict(X, num_iteration=int(lgb_iter_for_pred)), dtype=np.float32)

    xgb_dall = xgb.DMatrix(X.to_numpy())
    if best_iter is None or int(best_iter) < 0:
        full_xgb_pred = np.asarray(xgb_model.predict(xgb_dall), dtype=np.float32)
    else:
        full_xgb_pred = np.asarray(xgb_model.predict(xgb_dall, iteration_range=(0, int(best_iter) + 1)), dtype=np.float32)

    del dtr, dva, xgb_dall, X_tr, X_val, y_tr, y_val, g_tr, g_val
    gc.collect()

    coarse_scored = _build_coarse_scored_frame(
        df=df,
        group_key=group_key,
        oof_lgb=full_lgb_pred,
        oof_xgb=full_xgb_pred,
    )

    if not skip_export_dien:
        _export_dien_dataset(
            coarse_scored=coarse_scored,
            scene=scene,
            output_tag=output_tag,
            group_key=group_key,
            topn=dien_topn,
            keep_positive=keep_positive_for_dien,
        )

    if not skip_export_hard_neg:
        _export_easy_coarse_complement_hard_neg(
            scene=scene,
            output_tag=output_tag,
            group_key=group_key,
            candidate_path=candidate_path,
            coarse_scored=coarse_scored,
            coarse_topn=dien_topn,
            hard_neg_input_topn=hard_neg_input_topn,
            hard_neg_per_req=hard_neg_per_req,
        )
    print(f"[OK] Training complete. Logs at {log_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True, choices=["search", "rec"])
    parser.add_argument("--output-tag", type=str, default="", help="输出文件标签（例如 easy/hard），为空则不加后缀")
    parser.add_argument("--candidate-path", type=Path, default=None, help="召回候选文件路径")
    parser.add_argument("--recall-tag", type=str, default="easy", help="当未指定 candidate-path 时，按 tag 推断默认候选文件")
    parser.add_argument("--train-candidate-topn", type=int, default=500, help="粗排训练使用的候选截断（不影响召回 topk）")
    parser.add_argument("--dien-topn", type=int, default=500, help="导出给 DIEN 的粗排 topN")
    parser.add_argument("--no-keep-positive-for-dien", action="store_true", help="导出 DIEN 数据时不强制保留正样本")
    parser.add_argument("--skip-export-dien", action="store_true")
    parser.add_argument("--hard-neg-input-topn", type=int, default=1000, help="hard neg 输入召回集合截断（通常等于召回 topk）")
    parser.add_argument("--hard-neg-per-req", type=int, default=10, help="每请求最多导出的 hard neg 数（最终不超过10）")
    parser.add_argument("--skip-export-hard-neg", action="store_true", help="跳过 hard neg 导出")
    parser.add_argument("--print-every", type=int, default=10, help="每 N 轮在终端打印一次训练指标")
    parser.add_argument("--train-max-neg-per-req", type=int, default=0, help="GBDT 训练每请求最多保留负样本数（<=0 不采样）")
    args = parser.parse_args()

    main(
        scene=args.scene,
        output_tag=args.output_tag,
        candidate_path=args.candidate_path,
        recall_tag=args.recall_tag,
        train_candidate_topn=args.train_candidate_topn,
        dien_topn=args.dien_topn,
        keep_positive_for_dien=not args.no_keep_positive_for_dien,
        skip_export_dien=args.skip_export_dien,
        hard_neg_input_topn=args.hard_neg_input_topn,
        hard_neg_per_req=args.hard_neg_per_req,
        skip_export_hard_neg=args.skip_export_hard_neg,
        print_every=args.print_every,
        train_max_neg_per_req=args.train_max_neg_per_req,
    )
