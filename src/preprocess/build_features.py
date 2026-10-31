"""Build versioned Qilin features with required static inputs and finite labels."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "samples"
FEATURE_DIR = BASE_DIR / "features"
CAT_DIR = FEATURE_DIR / "vocab_dict"
USER_CAT = ["gender", "platform", "age", "location"]
NOTE_CAT = ["taxonomy1_id", "taxonomy2_id", "taxonomy3_id"]
STATIC_NUMERIC = ["note_type", "video_duration", "video_height", "video_width", "image_num", "content_length", "commercial_flag"]
FEATURE_CONTRACT = "finite_click_preserving_static_v2"


def _resolve_sample_path(scene, split, sample_dir=None):
    path = (Path(sample_dir) if sample_dir is not None else DATA_DIR) / f"{scene}_{split}_samples.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing sample file: {path}")
    return path


def prepare_category_mappings(con, sample_dir=None, vocab_dir=None):
    """Preserve existing checkpoint IDs; new UNK numbering needs a new model version."""
    destination = Path(vocab_dir) if vocab_dir is not None else CAT_DIR
    destination.mkdir(parents=True, exist_ok=True)
    for column in USER_CAT + NOTE_CAT:
        path = destination / f"{column}.pkl"
        if path.exists():
            with path.open("rb") as stream:
                mapping = pickle.load(stream)
        else:
            frames = [pd.read_parquet(_resolve_sample_path(scene, "train", sample_dir), columns=[column])
                      for scene in ["search", "rec"]]
            unique = sorted(pd.concat(frames, ignore_index=True)[column].fillna("UNK").astype(str).unique())
            mapping = {value: i for i, value in enumerate(unique)}
            with path.open("wb") as stream:
                pickle.dump(mapping, stream)
        con.register(f"map_{column}", pd.DataFrame(list(mapping.items()), columns=["raw_val", f"{column}_enc"]))


def _resolve_event_time_col(scene, in_file):
    columns = set(pq.read_schema(in_file).names)
    candidates = ["search_timestamp", "timestamp"] if scene == "search" else ["request_timestamp", "timestamp"]
    for name in candidates:
        if name in columns:
            return name
    raise ValueError(f"Missing event timestamp for {scene}; expected {candidates}")


def _finite(column):
    name = '"' + column.replace('"', '""') + '"'
    return f"CASE WHEN isfinite(TRY_CAST({name} AS DOUBLE)) THEN CAST({name} AS DOUBLE) ELSE 0.0 END"


def _optional(columns, name):
    return _finite(name) if name in columns else "0.0"


def build_features(scene, split, *, sample_dir=None, output_dir=None, vocab_dir=None):
    if scene not in {"search", "rec"} or split not in {"train", "test"}:
        raise ValueError("Expected scene search/rec and split train/test")
    source = _resolve_sample_path(scene, split, sample_dir)
    columns = set(pq.read_schema(source).names)
    key = "search_idx" if scene == "search" else "request_idx"
    time_column = _resolve_event_time_col(scene, source)
    required = {key, "user_idx", "session_idx", "note_idx", "recent_clicked_note_idxs", "click", "position", "page_time",
                "like", "collect", "comment", "share", *STATIC_NUMERIC, *USER_CAT, *NOTE_CAT}
    if scene == "search":
        required.add("query_from_type")
    missing = sorted(required - columns)
    if missing:
        raise ValueError(f"Missing required feature inputs: {missing}. Rebuild samples with the current catalog projection.")
    destination = Path(output_dir) if output_dir is not None else FEATURE_DIR
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / f"{scene}_{split}_features.parquet"
    temporary = output.with_suffix(".tmp.parquet")
    con = duckdb.connect(database=":memory:")
    try:
        prepare_category_mappings(con, sample_dir, vocab_dir)
        con.read_parquet(str(source)).create_view("samples")
        invalid = con.execute("SELECT count(*) FROM samples WHERE click IS NULL OR click NOT IN (0,1)").fetchone()[0]
        if invalid:
            raise ValueError(f"Invalid binary click labels: {invalid}")
        terms = ["1.0 * click"] + [f"{weight}.0 * GREATEST({_finite(name)}, 0.0)" for name, weight in [("like", 2), ("collect", 3), ("comment", 3), ("share", 3)]]
        terms.append(f"0.2 * LN(1.0 + GREATEST({_finite('page_time')}, 0.0))")
        projection = [key]
        if scene == "search":
            projection.append("query_from_type")
        projection += [f'"{time_column}" AS event_timestamp', "recent_clicked_note_idxs", "session_idx", "user_idx", "note_idx", "click",
            "CASE WHEN click=0 THEN 0.0 ELSE " + " + ".join(terms) + " END AS y_multi", "position",
            "CASE WHEN position<=4 THEN 1 WHEN position<=8 THEN 2 WHEN position<=12 THEN 3 WHEN position<=20 THEN 4 WHEN position<=50 THEN 5 WHEN position<=200 THEN 6 WHEN position<=300 THEN 7 WHEN position<=400 THEN 8 WHEN position<=500 THEN 9 ELSE 10 END AS pos_bucket"]
        projection += [f"{_finite(name)} AS {name}" for name in STATIC_NUMERIC]
        joins = []
        for name in NOTE_CAT + USER_CAT:
            alias = "map_" + name
            projection.append(f"COALESCE({alias}.{name}_enc, 0) AS {name}_enc")
            joins.append(f"LEFT JOIN map_{name} {alias} ON CAST(l.{name} AS VARCHAR)={alias}.raw_val")
        aggregates = ["imp_num", "click_num", "like_num", "collect_num", "comment_num", "share_num",
            f"imp_{scene}_num", f"click_{scene}_num", f"{scene}_like_num", f"{scene}_collect_num",
            f"{scene}_comment_num", f"{scene}_share_num", f"{scene}_follow_num", "accum_like_num",
            "accum_collect_num", "accum_comment_num", "view_time", f"{scene}_view_time", "valid_view_times", "full_view_times"]
        projection += [f"{_optional(columns,name)} AS {name}" for name in aggregates]
        ratios = {"ctr": ("click_num", "imp_num"), "like_rate": ("like_num", "click_num"),
            "collect_rate": ("collect_num", "click_num"), "comment_rate": ("comment_num", "click_num"),
            "share_rate": ("share_num", "click_num"), f"{scene}_ctr": (f"click_{scene}_num", f"imp_{scene}_num"),
            f"{scene}_like_rate": (f"{scene}_like_num", f"click_{scene}_num"), f"{scene}_collect_rate": (f"{scene}_collect_num", f"click_{scene}_num"),
            f"{scene}_share_rate": (f"{scene}_share_num", f"click_{scene}_num"), f"{scene}_follow_rate": (f"{scene}_follow_num", f"click_{scene}_num"),
            f"{scene}_view_time_rate": (f"{scene}_view_time", "view_time"), "valid_view_time_rate": ("valid_view_times", "view_time"), "full_view_time_rate": ("full_view_times", "view_time")}
        projection += [f"COALESCE(({_optional(columns,numerator)}) / NULLIF(({_optional(columns,denominator)}),0.0),0.0) AS {name}" for name,(numerator,denominator) in ratios.items()]
        projection += [f"{_optional(columns,name)} AS {name}" for name in ["fans_num", "follows_num"] + [f"dense_feat{i}" for i in range(1,41)]]
        query = f"SELECT {', '.join(projection)} FROM samples l {' '.join(joins)}"
        escaped = str(temporary).replace("'", "''")
        con.execute(f"COPY ({query}) TO '{escaped}' (FORMAT PARQUET)")
        con.read_parquet(str(temporary)).create_view("built")
        before = con.execute("SELECT count(*) FROM samples").fetchone()[0]
        after, bad = con.execute("SELECT count(*), count(*) FILTER(WHERE NOT isfinite(y_multi) OR (y_multi>0) IS DISTINCT FROM (click=1)) FROM built").fetchone()
        if before != after or bad:
            raise ValueError(f"Exposure/label contract failed: input={before}, output={after}, invalid={bad}")
        temporary.replace(output)
        print(f"[Features] {FEATURE_CONTRACT}: {scene}/{split}, {after} exposures -> {output}")
        return output
    finally:
        con.close()
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, choices=["search", "rec"])
    parser.add_argument("--split", required=True, choices=["train", "test"])
    parser.add_argument("--sample-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None, help="Stage features without replacing frozen artifacts")
    parser.add_argument("--vocab-dir", type=Path, default=None, help="Use existing checkpoint vocabularies without renumbering")
    args = parser.parse_args()
    build_features(args.scene, args.split, sample_dir=args.sample_dir, output_dir=args.output_dir, vocab_dir=args.vocab_dir)