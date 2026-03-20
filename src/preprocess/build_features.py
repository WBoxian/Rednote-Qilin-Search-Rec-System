"""
Qilin Samples Project: Feature Engineering Pipeline (Search & Rec)
---------------------------------------------------------------------------
统一特征构建脚本
构建树模型 + DNN 模型所需的特征，不处理 query/title/content 文本字段。

数据表：
- user_behavior：行为 & label & pos_bucket
- note：note画像统计
- user：用户画像 & dense feat

输出 wide_df：
    [idx_col] + user_behavior + note(含类别编码) + user(含类别编码)
    
使用示例:
    uv run python build_features.py --scene search --split train
    uv run python build_features.py --scene rec --split test
"""

import duckdb
import pandas as pd
import pickle
import argparse
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

# ================================================
# 路径配置
# ================================================
BASE_DIR = Path(__file__).resolve().parents[2]   # Qilin/
DATA_DIR = BASE_DIR / "samples"
FEATURE_DIR = BASE_DIR / "features"
CAT_DIR = FEATURE_DIR / "vocab_dict"

FEATURE_DIR.mkdir(parents=True, exist_ok=True)
CAT_DIR.mkdir(parents=True, exist_ok=True)

USER_CAT = ["gender", "platform", "age", "location"]
NOTE_CAT = ["taxonomy1_id", "taxonomy2_id", "taxonomy3_id"]

# ================================================
# 工具函数：构建类别映射
# ================================================
def _resolve_sample_path(scene: str, split: str) -> Path:
    new_path = DATA_DIR / f"{scene}_{split}_samples.parquet"
    if not new_path.exists():
        raise FileNotFoundError(f"Missing sample file: {new_path}")
    return new_path


def prepare_category_mappings(con):
    """全局构建一次 pkl，并注册到 duckdb 中用于 SQL JOIN"""
    all_cats = USER_CAT + NOTE_CAT
    for col in all_cats:
        map_file = CAT_DIR / f"{col}.pkl"
        
        # 1. 如果不存在，则构建 pkl (仅扫描训练集防止泄露)
        if not map_file.exists():
            print(f"[Map Build] {col}...")
            dfs = []
            for scene in ["search", "rec"]:
                p = _resolve_sample_path(scene, "train")
                dfs.append(pd.read_parquet(p, columns=[col]))
            if dfs:
                df_map = pd.concat(dfs, ignore_index=True)
                uniq = sorted(df_map[col].fillna("UNK").astype(str).unique())
                mapping = {v: i for i, v in enumerate(uniq)}
            else:
                mapping = {"UNK": 0}
            with open(map_file, "wb") as f:
                pickle.dump(mapping, f)
        else:
            with open(map_file, "rb") as f:
                mapping = pickle.load(f)
        
        # 2. 将映射注册为 DuckDB 内存表
        map_df = pd.DataFrame(list(mapping.items()), columns=["raw_val", f"{col}_enc"])
        con.register(f"map_{col}", map_df)


def _resolve_event_time_col(scene: str, in_file: Path) -> str:
    cols = set(pq.ParquetFile(in_file).schema.names)
    candidates = ["search_timestamp", "timestamp"] if scene == "search" else ["request_timestamp", "timestamp"]
    for col in candidates:
        if col in cols:
            return col
    raise ValueError(
        f"Cannot find event timestamp column for scene={scene}. "
        f"Expected one of {candidates}, available={sorted(cols)}"
    )


def _col_or(cols: set[str], name: str, fallback: str) -> str:
    return name if name in cols else fallback

# ================================================
# 主流程：构建特征并进行编码
# ================================================
def build_features(scene: str, split: str):
    assert scene in ["search", "rec"]
    assert split in ["train", "test"]

    IN_FILE = _resolve_sample_path(scene, split)
    OUT_FILE = FEATURE_DIR / f"{scene}_{split}_features.parquet"
    idx_col = "search_idx" if scene == "search" else "request_idx"
    event_time_col = _resolve_event_time_col(scene, IN_FILE)
    available_cols = set(pq.ParquetFile(IN_FILE).schema.names)

    imp_scene_col = _col_or(available_cols, f"imp_{scene}_num", "imp_num")
    click_scene_col = _col_or(available_cols, f"click_{scene}_num", "click_num")
    scene_like_col = _col_or(available_cols, f"{scene}_like_num", "0")
    scene_collect_col = _col_or(available_cols, f"{scene}_collect_num", "0")
    scene_comment_col = _col_or(available_cols, f"{scene}_comment_num", "0")
    scene_share_col = _col_or(available_cols, f"{scene}_share_num", "0")
    scene_follow_col = _col_or(available_cols, f"{scene}_follow_num", "0")

    accum_like_col = _col_or(available_cols, "accum_like_num", "like_num")
    accum_collect_col = _col_or(available_cols, "accum_collect_num", "collect_num")
    accum_comment_col = _col_or(available_cols, "accum_comment_num", "comment_num")
    scene_view_time_col = _col_or(available_cols, f"{scene}_view_time", "view_time")
    valid_view_times_col = _col_or(available_cols, "valid_view_times", "0")
    full_view_times_col = _col_or(available_cols, "full_view_times", "0")
    
    con = duckdb.connect(database=":memory:")
    
    print(f"Preparing Category Maps...")
    prepare_category_mappings(con)

    if scene == 'search':
        # 搜索场景特有字段
        scene_specific_cols = f"""
            query_from_type,
            {event_time_col} AS event_timestamp
        """
    else:
        # 推荐场景特有字段
        scene_specific_cols = f"""
            {event_time_col} AS event_timestamp
        """
        
    sql_query = f"""
        SELECT
            -- 1. [User_Behavior 模块]
            {idx_col}, {scene_specific_cols},
            recent_clicked_note_idxs, session_idx, user_idx, note_idx, 
            
            -- y_multi
            click,
            CASE WHEN click = 0 THEN 0.0 
                ELSE 
                    1.0 * click + 
                    2.0 * "like" + 
                    3.0 * collect + 
                    3.0 * comment + 
                    3.0 * share  + 
                    0.2 * LN(1 + COALESCE(page_time,0))
            END AS y_multi,
            
            -- pos_bucket
            position,
            CASE WHEN position <= 4  THEN 1 WHEN position <= 8  THEN 2 
                 WHEN position <= 12 THEN 3 WHEN position <= 20 THEN 4 
                 WHEN position <= 50 THEN 5 WHEN position <= 100 THEN 6
                 WHEN position <= 200 THEN 6 WHEN position <= 300 THEN 7
                 WHEN position <= 400 THEN 8 WHEN position <= 500 THEN 9
                 ELSE 10 END AS pos_bucket,
            
            -- 2. [NOTE 模块]
            {_col_or(available_cols, 'note_type', "''")} AS note_type,
            {_col_or(available_cols, 'video_duration', '0')} AS video_duration,
            {_col_or(available_cols, 'video_height', '0')} AS video_height,
            {_col_or(available_cols, 'video_width', '0')} AS video_width,
            {_col_or(available_cols, 'image_num', '0')} AS image_num,
            {_col_or(available_cols, 'content_length', '0')} AS content_length,
            {_col_or(available_cols, 'commercial_flag', '0')} AS commercial_flag,
            COALESCE(mt1.taxonomy1_id_enc, 0) AS taxonomy1_id_enc,
            COALESCE(mt2.taxonomy2_id_enc, 0) AS taxonomy2_id_enc,
            COALESCE(mt3.taxonomy3_id_enc, 0) AS taxonomy3_id_enc,
            {_col_or(available_cols, 'imp_num', '0')} AS imp_num,
            {imp_scene_col} AS imp_{scene}_num,
            {_col_or(available_cols, 'click_num', '0')} AS click_num,
            {click_scene_col} AS click_{scene}_num,
            {_col_or(available_cols, 'like_num', '0')} AS like_num,
            {_col_or(available_cols, 'collect_num', '0')} AS collect_num,
            {_col_or(available_cols, 'comment_num', '0')} AS comment_num,
            {_col_or(available_cols, 'share_num', '0')} AS share_num,
            {scene_like_col} AS {scene}_like_num,
            {scene_collect_col} AS {scene}_collect_num,
            {scene_comment_col} AS {scene}_comment_num,
            {scene_share_col} AS {scene}_share_num,
            {scene_follow_col} AS {scene}_follow_num,
            {accum_like_col} AS accum_like_num,
            {accum_collect_col} AS accum_collect_num,
            {accum_comment_col} AS accum_comment_num,
            {_col_or(available_cols, 'view_time', '0')} AS view_time,
            {scene_view_time_col} AS {scene}_view_time,
            {valid_view_times_col} AS valid_view_times,
            {full_view_times_col} AS full_view_times,

            -- 近一个月的行为统计
            COALESCE(click_num / NULLIF(imp_num, 0), 0.0) AS ctr,
            COALESCE(like_num / NULLIF(click_num, 0), 0.0) AS like_rate,
            COALESCE(collect_num / NULLIF(click_num, 0), 0.0) AS collect_rate,
            COALESCE(comment_num / NULLIF(click_num, 0), 0.0) AS comment_rate,
            COALESCE(share_num / NULLIF(click_num, 0), 0.0) AS share_rate,
            
            -- 近一个月的业务场景行为统计
            COALESCE({click_scene_col} / NULLIF({imp_scene_col}, 0), 0.0) AS {scene}_ctr,
            COALESCE({scene_like_col} / NULLIF({click_scene_col}, 0), 0.0) AS {scene}_like_rate,
            COALESCE({scene_collect_col} / NULLIF({click_scene_col}, 0), 0.0) AS {scene}_collect_rate,
            COALESCE({scene_share_col} / NULLIF({click_scene_col}, 0), 0.0) AS {scene}_share_rate,
            COALESCE({scene_follow_col} / NULLIF({click_scene_col}, 0), 0.0) AS {scene}_follow_rate,
            
            -- 累计行为统计
            COALESCE({scene_view_time_col} / NULLIF(view_time, 0), 0.0) AS {scene}_view_time_rate,
            COALESCE({valid_view_times_col} / NULLIF(view_time, 0), 0.0) AS valid_view_time_rate,
            COALESCE({full_view_times_col} / NULLIF(view_time, 0), 0.0) AS full_view_time_rate,
            
            -- 3. [USER 模块]
            COALESCE(mg.gender_enc, 0)   AS gender_enc,
            COALESCE(mp.platform_enc, 0) AS platform_enc,
            COALESCE(ma.age_enc, 0)      AS age_enc,
            COALESCE(ml.location_enc, 0) AS location_enc,
            fans_num, follows_num,
            COLUMNS('dense_feat.*'),
        
        FROM read_parquet('{IN_FILE}') l
        
        LEFT JOIN map_gender mg   ON CAST(gender AS VARCHAR)   = mg.raw_val
        LEFT JOIN map_platform mp ON CAST(platform AS VARCHAR) = mp.raw_val
        LEFT JOIN map_age ma      ON CAST(age AS VARCHAR)      = ma.raw_val
        LEFT JOIN map_location ml ON CAST(location AS VARCHAR) = ml.raw_val
        LEFT JOIN map_taxonomy1_id mt1 ON CAST(taxonomy1_id AS VARCHAR) = mt1.raw_val
        LEFT JOIN map_taxonomy2_id mt2 ON CAST(taxonomy2_id AS VARCHAR) = mt2.raw_val
        LEFT JOIN map_taxonomy3_id mt3 ON CAST(taxonomy3_id AS VARCHAR) = mt3.raw_val
    """

    print(f"[Streaming] Writing to {OUT_FILE}...")
    # COPY 命令确保全程不占用超额内存
    con.execute(f"COPY ({sql_query}) TO '{OUT_FILE}' (FORMAT PARQUET)")
    row_count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{OUT_FILE}')").fetchone()[0]
    col_count = len(con.execute(f"SELECT * FROM read_parquet('{OUT_FILE}') LIMIT 0").description)
    print(f"[Done] 特征构建完成: [Shape] [{row_count}, {col_count}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True, choices=["search", "rec"])
    parser.add_argument("--split", required=True, choices=["train", "test"])
    args = parser.parse_args()

    build_features(args.scene, args.split)
