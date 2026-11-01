"""Prepare post-selection recommendation refit data from the original train only.

No model, optimizer, validation scoring, test-file discovery, or deployment work
belongs here. A source-bound previous split supplies the already-frozen cutoff.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

_MAX_ID = 2**63 - 1
_DETAILS = "rec_result_details_with_idx"


def _fingerprint(path: Path) -> dict:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, key) != getattr(after, key) for key in fields):
        raise RuntimeError("Input changed while hashing: " + str(path))
    return {"path": str(path), "bytes": after.st_size, "sha256": digest.hexdigest(),
            "identity": [getattr(after, key) for key in fields]}


def _cutoff(value, name: str) -> int:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0 or value != int(value)):
        raise ValueError(name + " requires a finite positive integer timestamp")
    return int(value)


def _training_schema(path: Path) -> None:
    schema = pq.read_schema(path)
    if len(schema.names) != len(set(schema.names)):
        raise ValueError("Duplicate training schema column names")
    required = {"request_idx", "user_idx", "recent_clicked_note_idxs", _DETAILS}
    if not required.issubset(schema.names) or "rec_results" in schema.names:
        raise ValueError("Original recommendation training schema required; test schema is forbidden")
    for name in ("request_idx", "user_idx"):
        if not pa.types.is_integer(schema.field(name).type):
            raise ValueError(name + " must have an integer Arrow type")
    history = schema.field("recent_clicked_note_idxs").type
    details = schema.field(_DETAILS).type
    if (not (pa.types.is_list(history) or pa.types.is_large_list(history))
            or not pa.types.is_integer(history.value_type)):
        raise ValueError("History must be a list of integer item IDs")
    if (not (pa.types.is_list(details) or pa.types.is_large_list(details))
            or not pa.types.is_struct(details.value_type)):
        raise ValueError("Training details must be a list of structs")
    fields = {field.name: field.type for field in details.value_type}
    if len(fields) != len(details.value_type):
        raise ValueError("Duplicate training exposure field names")
    if not {"note_idx", "click", "request_timestamp"}.issubset(fields):
        raise ValueError("Training exposure item/click/request_timestamp fields required")
    if not pa.types.is_integer(fields["note_idx"]):
        raise ValueError("Exposure item IDs must have an integer Arrow type")
    for name in ("click", "request_timestamp"):
        if not (pa.types.is_integer(fields[name]) or pa.types.is_floating(fields[name])):
            raise ValueError(name + " must have a numeric Arrow type")


def prepare_refit_data(train_path: str | Path, split_manifest: str | Path,
                       output: str | Path, *, project_root: str | Path) -> Path:
    """Publish a new all-past data directory; reject stale inputs and overwrites.

    All exposures of a request must precede the supplied manifest's exclusive
    test cutoff. Zero-click requests remain eligible; a request without any
    timestamped exposure is rejected because its time interval is unknowable.
    """
    root = Path(project_root).resolve(strict=True)
    train = Path(train_path).resolve(strict=True)
    original = Path(split_manifest).resolve(strict=True)
    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    if any(part.lower() in {"recommendation_test", "rec_test"} for part in train.parts):
        raise ValueError("Test source is forbidden")
    source_before, manifest_before = _fingerprint(train), _fingerprint(original)
    previous = json.loads(original.read_text())
    if previous.get("scene") != "rec" or not isinstance(previous.get("inputs"), dict):
        raise ValueError("Original source-bound recommendation split manifest required")
    matched = []
    for name, digest in previous["inputs"].items():
        path = Path(name)
        path = path.resolve() if path.is_absolute() else (root / path).resolve()
        if path == train:
            matched.append(digest)
    if (len(matched) != 1 or not isinstance(matched[0], str)
            or re.fullmatch(r"[0-9a-fA-F]{64}", matched[0]) is None
            or matched[0].lower() != source_before["sha256"]):
        raise ValueError("Original train path/SHA256 does not match split manifest inputs")
    cutoff = _cutoff(previous.get("official_test_first_event"), "Official cutoff")
    old_fit = _cutoff(previous.get("fit_end_exclusive"), "Original fit boundary")
    if old_fit >= cutoff:
        raise ValueError("Original fit boundary must precede the final refit cutoff")
    _training_schema(train)
    with duckdb.connect(config={"threads": "2", "memory_limit": "2GB"}) as con:
        con.execute(f"""CREATE TABLE raw AS SELECT request_idx,user_idx,
            recent_clicked_note_idxs,{_DETAILS} FROM read_parquet(?)""", [str(train)])
        count, unique, nonnull = con.execute(
            "SELECT count(*),count(DISTINCT request_idx),count(request_idx) FROM raw").fetchone()
        if not count or count != unique or count != nonnull:
            raise ValueError("Original request IDs must be unique, nonnull, and nonempty")
        invalid = con.execute(f"""SELECT count(*) FROM raw WHERE request_idx<0
            OR request_idx>{_MAX_ID} OR user_idx IS NULL OR user_idx<0 OR user_idx>{_MAX_ID}
            OR recent_clicked_note_idxs IS NULL OR {_DETAILS} IS NULL
            OR len({_DETAILS})=0""").fetchone()[0]
        if invalid:
            raise ValueError("Invalid request/user/history or empty timestamped exposure list")
        invalid = con.execute(f"""SELECT count(*) FROM raw,unnest(recent_clicked_note_idxs) h(item)
            WHERE item IS NULL OR item<0 OR item>{_MAX_ID}""").fetchone()[0]
        if invalid:
            raise ValueError("Invalid history item IDs")
        invalid = con.execute(f"""SELECT count(*) FROM raw,unnest({_DETAILS}) t(d)
            WHERE d IS NULL OR d.note_idx IS NULL OR d.note_idx<0 OR d.note_idx>{_MAX_ID}
            OR d.click IS NULL OR NOT isfinite(d.click) OR d.click NOT IN (0,1)
            OR d.request_timestamp IS NULL OR NOT isfinite(d.request_timestamp)
            OR d.request_timestamp<=0""").fetchone()[0]
        if invalid:
            raise ValueError("Invalid exposure item, binary click, or finite positive timestamp")
        con.execute(f"""CREATE TABLE pairs AS SELECT request_idx::BIGINT gid,
            first(user_idx)::BIGINT user_idx,d.note_idx::BIGINT note_idx,
            max(d.click)::INTEGER positive,min(d.request_timestamp)::DOUBLE lo,
            max(d.request_timestamp)::DOUBLE hi
            FROM raw,unnest({_DETAILS}) t(d) GROUP BY request_idx,d.note_idx""")
        con.execute("""CREATE TABLE partitions AS SELECT gid,min(lo) lo,max(hi) hi,
            CASE WHEN max(hi)<? THEN 'fit' ELSE 'excluded' END "partition",
            CASE WHEN max(hi)<? THEN 'fit'
                 WHEN min(lo)>=? AND max(hi)<? THEN 'validation'
                 WHEN max(hi)<? THEN 'cross_fit_validation' ELSE 'excluded' END original_partition
            FROM pairs GROUP BY gid""", [cutoff, old_fit, old_fit, cutoff, cutoff])
        labels = con.execute("""SELECT p.* FROM pairs p JOIN partitions s USING(gid)
            WHERE s.partition='fit' ORDER BY p.gid,p.note_idx""").fetchdf()
        requests = con.execute(f"""SELECT r.request_idx::BIGINT gid,r.user_idx::BIGINT user_idx,
            r.recent_clicked_note_idxs AS history,
            coalesce(list_sort(list(DISTINCT p.note_idx) FILTER(WHERE p.positive>0)),[]::BIGINT[]) positives
            FROM raw r JOIN pairs p ON r.request_idx=p.gid
            JOIN partitions s ON r.request_idx=s.gid WHERE s.partition='fit'
            GROUP BY r.request_idx,r.user_idx,r.recent_clicked_note_idxs ORDER BY gid""").fetchdf()
        partitions = con.execute("SELECT * FROM partitions ORDER BY gid").fetchdf()
    if requests.empty:
        raise ValueError("No complete training request precedes the exclusive cutoff")
    if (_fingerprint(train) != source_before or _fingerprint(original) != manifest_before):
        raise RuntimeError("Source or original split manifest changed during preparation")
    eligible = partitions[partitions.partition.eq("fit")]
    if set(requests.gid) != set(eligible.gid) or set(labels.gid) != set(eligible.gid):
        raise RuntimeError("Complete-request roster was not retained")
    # The manifest is written last, so a failed/partial directory is never ready.
    destination.mkdir(parents=True, exist_ok=False)
    outputs = {"requests": destination / "requests.parquet", "labels": destination / "labels.parquet",
               "partitions": destination / "partitions.parquet"}
    for name, frame in (("requests", requests), ("labels", labels), ("partitions", partitions)):
        frame.to_parquet(outputs[name], index=False)
    if (_fingerprint(train) != source_before or _fingerprint(original) != manifest_before):
        raise RuntimeError("Source or original split manifest changed before publication")
    groups = []
    for name, rows in partitions.groupby("original_partition", sort=True):
        selected = labels[labels.gid.isin(rows.gid)]
        groups.append({"original_partition": name, "requests": len(rows),
                       "eligible_refit_requests": int(rows.partition.eq("fit").sum()),
                       "refit_pair_rows": len(selected), "refit_positive_pairs": int(selected.positive.sum())})
    manifest = {"version": 1, "scene": "rec", "stage": "postselection-final-refit",
                "fit_end_exclusive": cutoff, "official_test_first_event": cutoff,
                "original_fit_end_exclusive": old_fit, "whole_request_cutoff_rule": "max(eventtime)<cutoff",
                "selection_split": None, "original_validation_retired": True,
                "validation_policy": "Previous validation is now training; no subsequent validation/model selection on it",
                "source": source_before, "original_split_manifest": manifest_before,
                "inputs": {str(train.relative_to(root)) if train.is_relative_to(root) else str(train): source_before["sha256"]},
                "outputs": {name: _fingerprint(path) for name, path in outputs.items()},
                "raw_requests": int(len(partitions)), "refit_requests": int(len(requests)),
                "refit_exposure_pairs": int(len(labels)), "refit_positive_pairs": int(labels.positive.sum()),
                "refit_zero_click_requests": int(sum(len(items) == 0 for items in requests.positives)),
                "original_partition_counts": groups,
                "implementation_sha256": _fingerprint(Path(__file__).resolve())["sha256"],
                "official_test_read": False, "model_training_performed": False,
                "limitations": ["Provided history has no per-item timestamps; provider snapshot PIT is not independently verified",
                                "This prepares source-bound past data only, not a fresh validation or test result"]}
    path = destination / "manifest.json"
    with path.open("x") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return path
