"""
Qilin Samples Project: Data Preprocessing Pipeline (Search & Rec)
---------------------------------------------------------------------------
功能描述:
    使用 DuckDB 向量化引擎，将搜索/推荐日志与用户、笔记特征进行合并，
    生成统一样本文件（samples）。

处理流程:
    1. 用户行为字段展开(UNNEST User Behavior)
    2. 关联笔记特征(Join Notes)
    3. 关联用户特征(Join User)
    4. 导出: samples/{scene}_{split}_samples.parquet

使用示例:
    uv run python src/preprocess/build_samples.py --scene search --split train
    uv run python src/preprocess/build_samples.py --scene rec --split test
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

BASE_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = BASE_DIR / "samples"
OUT_DIR.mkdir(exist_ok=True)
NOTES_PARQUET = BASE_DIR / "datasets" / "notes" / "*.parquet"
USER_PARQUET = BASE_DIR / "datasets" / "user_feat" / "*.parquet"


def build_samples(scene: str, split: str) -> None:
    folder_prefix = "recommendation" if scene == "rec" else "search"
    nested_col = "rec_result_details_with_idx" if scene == "rec" else "search_result_details_with_idx"
    time_col = "request_timestamp" if scene == "rec" else "search_timestamp"

    data_path = BASE_DIR / "datasets" / f"{folder_prefix}_{split}" / "*.parquet"
    out_file = OUT_DIR / f"{scene}_{split}_samples.parquet"

    con = duckdb.connect(database=":memory:")
    print(f"[*] Processing {scene.upper()} {split} -> {out_file.name}")

    extra_cols = "s.query, "
    if scene == "search":
        extra_cols += "s.query_from_type, s.search_idx, "
    else:
        extra_cols += "s.request_idx, "

    con.execute(
        f"""
        CREATE TABLE stage_flat AS
        SELECT
            s.user_idx,
            s.session_idx,
            s.recent_clicked_note_idxs,
            {extra_cols}
            u.note_idx,
            CAST(u.position AS INTEGER) AS position,
            u.{time_col} AS timestamp,
            CAST(u.click AS INTEGER)   AS click,
            CAST(u.like AS INTEGER)    AS like,
            CAST(u.collect AS INTEGER) AS collect,
            CAST(u.comment AS INTEGER) AS comment,
            CAST(u.share AS INTEGER)   AS share,
            u.page_time
        FROM read_parquet('{data_path}') AS s,
             UNNEST(s.{nested_col}) AS d(u)
        """
    )

    prefix = "search" if scene == "search" else "rec"
    con.execute(
        f"""
        CREATE TABLE stage_full AS
        SELECT
            f.*,
            n.note_title, n.note_content, n.note_type, n.taxonomy1_id, n.taxonomy2_id, n.taxonomy3_id,
            n.video_duration, n.content_length, n.commercial_flag,
            n.imp_num, n.click_num, n.like_num, n.view_time,
            n.{prefix}_like_num, n.{prefix}_collect_num, n.{prefix}_comment_num, n.{prefix}_share_num,
            u.gender, u.platform, u.age, u.location, u.fans_num, u.follows_num,
            u.dense_feat1, u.dense_feat2, u.dense_feat3, u.dense_feat4, u.dense_feat5,
            u.dense_feat6, u.dense_feat7, u.dense_feat8, u.dense_feat9, u.dense_feat10,
            u.dense_feat11, u.dense_feat12, u.dense_feat13, u.dense_feat14, u.dense_feat15,
            u.dense_feat16, u.dense_feat17, u.dense_feat18, u.dense_feat19, u.dense_feat20,
            u.dense_feat21, u.dense_feat22, u.dense_feat23, u.dense_feat24, u.dense_feat25,
            u.dense_feat26, u.dense_feat27, u.dense_feat28, u.dense_feat29, u.dense_feat30,
            u.dense_feat31, u.dense_feat32, u.dense_feat33, u.dense_feat34, u.dense_feat35,
            u.dense_feat36, u.dense_feat37, u.dense_feat38, u.dense_feat39, u.dense_feat40
        FROM stage_flat f
        LEFT JOIN read_parquet('{NOTES_PARQUET}') n ON f.note_idx = n.note_idx
        LEFT JOIN read_parquet('{USER_PARQUET}') u ON f.user_idx = u.user_idx
        """
    )

    con.execute(f"COPY stage_full TO '{out_file}' (FORMAT PARQUET)")
    print(f"[OK] saved: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True, choices=["rec", "search"])
    parser.add_argument("--split", required=True, choices=["train", "test"])
    args = parser.parse_args()

    build_samples(args.scene, args.split)
