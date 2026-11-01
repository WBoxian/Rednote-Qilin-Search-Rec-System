"""Build isolated, audited, scene-specific data for the next training iteration.

Run: python -m preprocess.release --output outputs/preprocess/<new-version>
Existing models, vocabularies, source datasets and published metrics are untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from preprocess.build_samples import build_samples
from preprocess.build_features import build_features, USER_CAT, NOTE_CAT, STATIC_NUMERIC, FEATURE_CONTRACT
from training.history import HISTORY_ORDER

ROOT = Path(__file__).resolve().parents[2]
CATEGORY_CONTRACT = "fit_only_reserved_zero_v1"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            result.update(block)
    return result.hexdigest()


def partition_requests(train: pd.DataFrame, test_start: float, validation_ratio=.2):
    """Use full request intervals, never split an exposure group across time."""
    if not 0 < validation_ratio < 1 or not np.isfinite(test_start):
        raise ValueError("Invalid temporal partition configuration")
    if train.gid.duplicated().any() or train.gid.isna().any():
        raise ValueError("Request intervals require unique nonnull IDs")
    if not np.isfinite(train[["first", "last"]].to_numpy()).all() or (train["first"] > train["last"]).any():
        raise ValueError("Invalid event intervals")
    eligible = train[train["last"] < test_start]
    if len(eligible) < 2:
        raise ValueError("Not enough historical requests")
    position = max(1, min(len(eligible)-1, int(len(eligible)*(1-validation_ratio))))
    boundary = float(np.sort(eligible["last"].to_numpy())[position])
    fit = eligible.loc[eligible["last"] < boundary, "gid"].tolist()
    validation = eligible.loc[eligible["first"] >= boundary, "gid"].tolist()
    if not fit or not validation:
        raise ValueError("No strict forward fit/validation partition")
    return {"fit_ids": fit, "validation_ids": validation, "validation_start": boundary,
            "test_start": float(test_start), "official_train_requests": len(train),
            "excluded_at_test_boundary": int((train["last"] >= test_start).sum()),
            "excluded_at_validation_boundary": len(eligible)-len(fit)-len(validation)}


def fit_vocabularies(sample: Path, key: str, fit_ids: list, destination: Path):
    """Known values start at one. ID zero has no real-category dictionary entry."""
    if not fit_ids or len(set(fit_ids)) != len(fit_ids):
        raise ValueError("Vocabulary requires nonempty unique fit request IDs")
    destination.mkdir(parents=True, exist_ok=False)
    with duckdb.connect(config={"threads": 4}) as con:
        con.read_parquet(str(sample)).create_view("sample")
        con.register("fit_ids", pd.DataFrame({key: fit_ids}))
        matched = con.execute(f"SELECT count(DISTINCT s.{key}) FROM sample s JOIN fit_ids f USING({key})").fetchone()[0]
        if matched != len(fit_ids):
            raise ValueError("Vocabulary fit requests absent from training source")
        metadata = {"fit_ids_sha256": hashlib.sha256(json.dumps(sorted(map(int,fit_ids))).encode()).hexdigest(), "contract": CATEGORY_CONTRACT, "unknown_id": 0, "fit_request_count": len(fit_ids),
                    "sample_sha256": digest(sample), "categories": {}}
        for name in USER_CAT + NOTE_CAT:
            rows = con.execute(f"""SELECT DISTINCT CAST(s.{name} AS VARCHAR) AS category_value FROM sample s
                JOIN fit_ids f USING({key}) WHERE s.{name} IS NOT NULL AND CAST(s.{name} AS VARCHAR)<>''
                ORDER BY category_value""").fetchall()
            mapping = {value: i+1 for i, (value,) in enumerate(rows)}
            path = destination / f"{name}.pkl"
            with path.open("wb") as stream:
                pickle.dump(mapping, stream)
            metadata["categories"][name] = {"known_count": len(mapping), "embedding_size": len(mapping)+1,
                                            "sha256": digest(path)}
        (destination / "contract.json").write_text(json.dumps(metadata, indent=2)+"\n")
    return metadata


def _copy(con, query, output):
    escaped = str(output).replace("'", "''")
    con.execute(f"COPY ({query}) TO '{escaped}' (FORMAT PARQUET)")


def build_release(output: Path, dataset_dir: Path | None = None):
    dataset_dir = Path(dataset_dir or ROOT / "datasets").resolve()
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Use a new output directory; releases are immutable")
    if output == dataset_dir or dataset_dir.is_relative_to(output) or output.is_relative_to(dataset_dir):
        raise ValueError("Release output must be separate from source datasets")
    output.mkdir(parents=True)
    source_paths = sorted(path for folder in ["notes", "user_feat", "search_train", "search_test", "recommendation_train", "recommendation_test"]
                          for path in (dataset_dir / folder).glob("*.parquet"))
    source_hashes = {str(path): digest(path) for path in source_paths}
    manifest = {"contract": FEATURE_CONTRACT, "category_contract": CATEGORY_CONTRACT,
                "status": "building", "completed": False, "scenes": {}, "source_sha256": source_hashes,
                "eligible_new_features": STATIC_NUMERIC + [name+"_enc" for name in NOTE_CAT] +
                    [name+"_is_missing" for name in STATIC_NUMERIC] + [name+suffix for name in NOTE_CAT for suffix in ("_is_unknown","_is_missing","_is_unseen")] +
                    ["history_length","history_unique_length","history_repeat_ratio"],
                "excluded_model_inputs": ["click","y_multi","position","pos_bucket","page_time","snapshot_aggregate_counts_and_rates","opaque_user_dense_features"],
                "selection_policy": "Train gradients and preprocessing fits use scene/fit.parquet only; scene/validation.parquet selects settings; scene/test.parquet is evaluated after freezing",
                "history_order": HISTORY_ORDER, "history_event_time_provenance": "unknown",
                "feature_point_in_time_provenance": "unknown", "training_performed": False,
                "warning": "Static snapshots are not proof of event-time availability. New category IDs require retraining; never replace active checkpoint inputs."}
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    for scene in ["search", "rec"]:
        key = "search_idx" if scene == "search" else "request_idx"
        folder = "search" if scene == "search" else "recommendation"
        nested = "search_result_details_with_idx" if scene == "search" else "rec_result_details_with_idx"
        sample_paths = {split: build_samples(scene, split, dataset_dir=dataset_dir, output_dir=output/"samples")
                        for split in ["train", "test"]}
        scene_dir = output / scene
        scene_dir.mkdir()
        with duckdb.connect(config={"threads": 4}) as con:
            train = con.execute(f"SELECT {key} gid,min(timestamp) AS time_first,max(timestamp) AS time_last FROM read_parquet(?) GROUP BY {key}", [str(sample_paths["train"])]).fetchdf().rename(columns={"time_first":"first", "time_last":"last"})
            test_start = con.execute("SELECT min(timestamp) FROM read_parquet(?)", [str(sample_paths["test"])]).fetchone()[0]
        partition = partition_requests(train, test_start)
        fit_ids, validation_ids = partition["fit_ids"], partition["validation_ids"]
        vocab = fit_vocabularies(sample_paths["train"], key, fit_ids, scene_dir/"vocab")
        pd.DataFrame({key: fit_ids}).to_parquet(scene_dir/"fit_requests.parquet", index=False)
        pd.DataFrame({key: validation_ids}).to_parquet(scene_dir/"validation_requests.parquet", index=False)
        summary = {"partition": {k:v for k,v in partition.items() if not k.endswith("_ids")},
                   "fit_requests": len(fit_ids), "validation_requests": len(validation_ids),
                   "vocabulary": vocab, "splits": {}}
        for split in ["train", "test"]:
            feature = build_features(scene, split, sample_dir=output/"samples", output_dir=scene_dir/"features", vocab_dir=scene_dir/"vocab")
            with duckdb.connect(config={"threads": 4}) as con:
                con.read_parquet(str(feature)).create_view("features")
                con.read_parquet(str(sample_paths[split])).create_view("samples")
                raw_paths = sorted((dataset_dir/f"{folder}_{split}").glob("*.parquet"))
                con.read_parquet(list(map(str,raw_paths))).create_view("original")
                con.execute(f"""CREATE TEMP VIEW original_labels AS SELECT s.{key},s.user_idx,
                    d.u.note_idx,d.u.click,CAST(d.ordinality-1 AS BIGINT) exposure_index
                    FROM original s,unnest(s.{nested}) WITH ORDINALITY d(u,ordinality)""")
                projection = f"{key},user_idx,note_idx,click,exposure_index"
                difference = con.execute(f"SELECT count(*) FROM ((SELECT {projection} FROM original_labels EXCEPT ALL SELECT {projection} FROM features) UNION ALL (SELECT {projection} FROM features EXCEPT ALL SELECT {projection} FROM original_labels))").fetchone()[0]
                if difference:
                    raise ValueError("Original exposure identity/click conservation failed")
                stats = con.execute(f"""SELECT count(*),count(DISTINCT {key}),sum(click),
                    count(*) FILTER(WHERE NOT isfinite(y_multi)),count(*) FILTER(WHERE (y_multi>0)<>(click=1)),
                    count(*) FILTER(WHERE history_length=0) FROM features""").fetchone()
                categories = {name: con.execute(f"SELECT count(*) FILTER(WHERE {name}_is_unknown=1) FROM features").fetchone()[0] for name in USER_CAT+NOTE_CAT}
                missing_categories = {name: con.execute(f"SELECT count(*) FILTER(WHERE {name}_is_missing=1) FROM features").fetchone()[0] for name in USER_CAT+NOTE_CAT}
                unseen_categories = {name: con.execute(f"SELECT count(*) FILTER(WHERE {name}_is_unseen=1) FROM features").fetchone()[0] for name in USER_CAT+NOTE_CAT}
                empty_history_requests = con.execute(f"SELECT count(DISTINCT {key}) FROM features WHERE history_length=0").fetchone()[0]
                parity = con.execute("""SELECT count(*) FROM features f JOIN samples s USING("""+key+""",exposure_index)
                    WHERE f.video_height IS DISTINCT FROM s.video_height OR f.video_width IS DISTINCT FROM s.video_width OR f.image_num IS DISTINCT FROM s.image_num""").fetchone()[0]
                if parity or stats[3] or stats[4]:
                    raise ValueError("Static or click-label parity failed")
                summary["splits"][split] = dict(zip(["exposures","requests","clicked_exposures","nonfinite_labels","lost_click_labels","empty_history_exposures"],stats))
                summary["splits"][split].update(unknown_category_exposures=categories, missing_category_exposures=missing_categories, unseen_category_exposures=unseen_categories, empty_history_requests=empty_history_requests, static_field_mismatches=parity, original_exposure_mismatches=difference)
                if split == "train":
                    for part, ids in [("fit",fit_ids),("validation",validation_ids)]:
                        con.register("selected_ids", pd.DataFrame({key: ids}))
                        _copy(con, f"SELECT f.* FROM features f JOIN selected_ids s USING({key})", scene_dir/f"{part}.parquet")
                else:
                    _copy(con, "SELECT * FROM features", scene_dir/"test.parquet")
        manifest["scenes"][scene] = summary
        manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    for path, expected in source_hashes.items():
        if digest(path) != expected:
            raise ValueError("Source changed while building release")
    paths = sorted(output.rglob("*.parquet"))+sorted(output.rglob("*.pkl"))+sorted(output.rglob("contract.json"))
    manifest["output_sha256"] = {str(path.relative_to(output)):digest(path) for path in paths}
    code_paths = [Path(__file__), Path(__file__).with_name("build_samples.py"),Path(__file__).with_name("build_features.py")]
    manifest["code_sha256"] = {str(path.relative_to(ROOT)):digest(path) for path in code_paths}
    manifest["status"] = "PASS"
    manifest["completed"] = True
    manifest["source_datasets_unchanged"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    print(f"[Release] PASS: {manifest_path}", flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path)
    args = parser.parse_args()
    build_release(args.output, args.dataset_dir)
