"""Flatten Qilin logs without dropping catalog features or rewriting source data."""
from __future__ import annotations

import argparse
from pathlib import Path
import duckdb
import pyarrow.parquet as pq

BASE_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = BASE_DIR / "samples"
NOTES_PARQUET = BASE_DIR / "datasets" / "notes" / "*.parquet"
USER_PARQUET = BASE_DIR / "datasets" / "user_feat" / "*.parquet"

NOTE_REQUIRED = (
    "note_idx", "note_title", "note_content", "note_type", "taxonomy1_id",
    "taxonomy2_id", "taxonomy3_id", "video_duration", "video_height",
    "video_width", "image_num", "content_length", "commercial_flag",
)
# Missing snapshot aggregates remain explicitly optional; never alias one period
# or scene's count to a different field. New conservative models do not use them.
NOTE_OPTIONAL = (
    "imp_num", "click_num", "like_num", "collect_num", "comment_num", "share_num",
    "imp_rec_num", "imp_search_num", "click_rec_num", "click_search_num",
    "accum_like_num", "accum_collect_num", "accum_comment_num", "view_time",
    "rec_view_time", "search_view_time", "valid_view_times", "full_view_times",
    "rec_like_num", "rec_collect_num", "rec_comment_num", "rec_share_num", "rec_follow_num",
    "search_like_num", "search_collect_num", "search_comment_num", "search_share_num", "search_follow_num",
)
USER_COLUMNS = ("gender", "platform", "age", "location", "fans_num", "follows_num") + tuple(
    f"dense_feat{i}" for i in range(1, 41)
)


def _require_columns(paths, columns, label):
    if not paths:
        raise FileNotFoundError(f"No parquet files for {label}")
    for path in paths:
        missing = sorted(set(columns) - set(pq.read_schema(path).names))
        if missing:
            raise ValueError(f"Missing required {label} columns in {path.name}: {missing}")


def build_samples(scene: str, split: str, *, output_dir: Path | None = None,
                  dataset_dir: Path | None = None) -> Path:
    if scene not in {"search", "rec"} or split not in {"train", "test"}:
        raise ValueError("Expected scene search/rec and split train/test")
    dataset_dir = Path(dataset_dir) if dataset_dir is not None else BASE_DIR / "datasets"
    destination = Path(output_dir) if output_dir is not None else OUT_DIR
    prefix = "recommendation" if scene == "rec" else "search"
    nested = "rec_result_details_with_idx" if scene == "rec" else "search_result_details_with_idx"
    time_key = "request_timestamp" if scene == "rec" else "search_timestamp"
    key = "request_idx" if scene == "rec" else "search_idx"
    raw_paths = sorted((dataset_dir / f"{prefix}_{split}").glob("*.parquet"))
    notes = sorted((dataset_dir / "notes").glob("*.parquet"))
    users = sorted((dataset_dir / "user_feat").glob("*.parquet"))
    _require_columns(raw_paths, ["user_idx", "session_idx", "recent_clicked_note_idxs", "query", key, nested]
                     + (["query_from_type"] if scene == "search" else []), "log")
    _require_columns(notes, NOTE_REQUIRED, "static note")
    _require_columns(users, ["user_idx", *USER_COLUMNS], "user")
    note_columns = set.intersection(*(set(pq.read_schema(path).names) for path in notes))
    projection = [f"n.{name}" for name in NOTE_REQUIRED if name != "note_idx"]
    projection.extend(f"n.{name}" for name in NOTE_OPTIONAL if name in note_columns)
    projection.extend(f"u.{name}" for name in USER_COLUMNS)
    request_columns = f"s.query, s.{key}, " + ("s.query_from_type, " if scene == "search" else "")
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / f"{scene}_{split}_samples.parquet"
    temporary = output.with_suffix(".tmp.parquet")
    con = duckdb.connect(database=":memory:")
    try:
        con.read_parquet([str(path) for path in raw_paths]).create_view("raw_logs")
        con.read_parquet([str(path) for path in notes]).create_view("notes")
        con.read_parquet([str(path) for path in users]).create_view("users")
        for table, identifier in [("notes", "note_idx"), ("users", "user_idx")]:
            count, unique, nonnull = con.execute(f"SELECT count(*), count(DISTINCT {identifier}), count({identifier}) FROM {table}").fetchone()
            if count != unique or count != nonnull:
                raise ValueError(f"{table} requires unique nonnull {identifier}")
        con.execute(f"""CREATE TABLE stage_flat AS
            SELECT s.user_idx, s.session_idx, s.recent_clicked_note_idxs,
                CAST(d.ordinality - 1 AS BIGINT) AS exposure_index,
                {request_columns} u.note_idx, CAST(u.position AS INTEGER) AS position,
                u.{time_key} AS timestamp, u.click AS click,
                CAST(u.like AS INTEGER) AS like, CAST(u.collect AS INTEGER) AS collect,
                CAST(u.comment AS INTEGER) AS comment, CAST(u.share AS INTEGER) AS share, u.page_time
            FROM raw_logs AS s, UNNEST(s.{nested}) WITH ORDINALITY AS d(u, ordinality)""")
        invalid = con.execute("SELECT count(*) FROM stage_flat WHERE click IS NULL OR click NOT IN (0,1) OR note_idx IS NULL OR note_idx<0 OR note_idx<>floor(note_idx)").fetchone()[0]
        if invalid:
            raise ValueError(f"Invalid click/item identity in {invalid} exposures")
        invalid_time = con.execute("SELECT count(*) FROM stage_flat WHERE timestamp IS NULL OR NOT isfinite(TRY_CAST(timestamp AS DOUBLE)) OR timestamp<=0").fetchone()[0]
        if invalid_time:
            raise ValueError(f"Invalid event timestamp in {invalid_time} exposures")
        request_count, request_unique = con.execute(f"SELECT count(*),count(DISTINCT {key}) FROM raw_logs").fetchone()
        if request_count != request_unique:
            raise ValueError("Raw request IDs must be unique and nonnull")
        missing = con.execute("SELECT count(*) FROM stage_flat f LEFT JOIN notes n USING(note_idx) WHERE n.note_idx IS NULL").fetchone()[0]
        if missing:
            raise ValueError(f"{missing} exposures reference missing catalog items")
        con.execute(f"CREATE TABLE stage_full AS SELECT f.*, {', '.join(projection)} FROM stage_flat f LEFT JOIN notes n ON f.note_idx=n.note_idx LEFT JOIN users u ON f.user_idx=u.user_idx")
        expected = con.execute("SELECT count(*) FROM stage_flat").fetchone()[0]
        actual = con.execute("SELECT count(*) FROM stage_full").fetchone()[0]
        if expected != actual:
            raise ValueError("Feature joins changed the original exposure multiplicity")
        escaped = str(temporary).replace("'", "''")
        con.execute(f"COPY stage_full TO '{escaped}' (FORMAT PARQUET)")
        temporary.replace(output)
        print(f"[Samples] {scene}/{split}: {actual} original exposures -> {output}")
        return output
    finally:
        con.close()
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, choices=["rec", "search"])
    parser.add_argument("--split", required=True, choices=["train", "test"])
    parser.add_argument("--output-dir", type=Path, default=None, help="Stage new samples without replacing frozen artifacts")
    parser.add_argument("--dataset-dir", type=Path, default=None)
    args = parser.parse_args()
    build_samples(args.scene, args.split, output_dir=args.output_dir, dataset_dir=args.dataset_dir)
