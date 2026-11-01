"""Boundary/provenance contract tests for original-train final refit data."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from training import rec_refit_data as refit


def exposure(note, click, timestamp):
    return dict(note_idx=note, click=click, request_timestamp=timestamp)


def request(gid, user, history, *items):
    return dict(request_idx=gid, user_idx=user, recent_clicked_note_idxs=history,
                rec_result_details_with_idx=list(items))


def schema():
    return pa.schema([("request_idx", pa.int64()), ("user_idx", pa.int64()),
                      ("recent_clicked_note_idxs", pa.list_(pa.int64())),
                      ("rec_result_details_with_idx", pa.list_(pa.struct([
                          ("note_idx", pa.int64()), ("click", pa.float64()),
                          ("request_timestamp", pa.float64())])))])


class RefitDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.train = self.root / "datasets/recommendation_train/train.parquet"
        self.train.parent.mkdir(parents=True)
        self.manifest = self.root / "old_split.json"
        self.output = self.root / "new_refit"
        self.rows = [
            request(1, 10, [0, 4, 0], exposure(0, 0, 1), exposure(0, 1, 2), exposure(7, 0, 3)),
            request(2, 11, [], exposure(1, 0, 10)),
            request(3, 12, [3], exposure(4, 1, 9), exposure(2, 0, 11)),
            request(4, 13, [0], exposure(4, 1, 19), exposure(5, 1, 20)),
            request(5, 14, [], exposure(5, 0, 20)),
            request(6, 15, [0], exposure(0, 0, 15)),
        ]
        self.write_source(self.rows)

    def write_source(self, rows, arrow_schema=None):
        pq.write_table(pa.Table.from_pylist(rows, schema=arrow_schema or schema()), self.train)
        self.write_manifest()

    def write_manifest(self, **changes):
        value = dict(version=1, scene="rec", fit_end_exclusive=10.0,
                     official_test_first_event=20.0,
                     # This unavailable test asset is never opened or discovered.
                     official_test_sha256="f" * 64,
                     inputs={str(self.train.relative_to(self.root)): hashlib.sha256(self.train.read_bytes()).hexdigest()})
        value.update(changes)
        self.manifest.write_text(json.dumps(value))

    def prepare(self):
        return refit.prepare_refit_data(self.train, self.manifest, self.output, project_root=self.root)

    def test_whole_request_boundary_duplicate_pairs_zero_click_and_note0(self):
        result = self.prepare()
        value = json.loads(result.read_text())
        requests = pd.read_parquet(self.output / "requests.parquet").set_index("gid")
        labels = pd.read_parquet(self.output / "labels.parquet").set_index(["gid", "note_idx"])
        parts = pd.read_parquet(self.output / "partitions.parquet").set_index("gid")
        self.assertEqual(set(requests.index), {1, 2, 3, 6})
        self.assertEqual(len(labels), 6)
        self.assertEqual(labels.loc[(1, 0), "positive"], 1)
        self.assertEqual(labels.loc[(1, 0), "lo"], 1)
        self.assertEqual(labels.loc[(1, 0), "hi"], 2)
        self.assertEqual(list(requests.loc[1, "history"]), [0, 4, 0])
        self.assertEqual(list(requests.loc[2, "positives"]), [])
        self.assertEqual(list(requests.loc[6, "positives"]), [])
        self.assertIn((6, 0), labels.index)
        self.assertEqual(parts.loc[3, "original_partition"], "cross_fit_validation")
        self.assertEqual(parts.loc[4, "partition"], "excluded")
        self.assertNotIn(4, labels.index.get_level_values("gid"))  # Its time19 exposure is excluded too.
        self.assertEqual(parts.loc[5, "partition"], "excluded")  # Equality at20 is excluded.
        self.assertEqual(value["fit_end_exclusive"], 20)
        self.assertEqual(value["original_fit_end_exclusive"], 10)
        self.assertEqual(value["stage"], "postselection-final-refit")
        self.assertTrue(value["original_validation_retired"])
        self.assertFalse(value["official_test_read"])
        self.assertFalse(value["model_training_performed"])
        self.assertEqual(value["refit_zero_click_requests"], 2)
        counts = {v["original_partition"]: v["requests"] for v in value["original_partition_counts"]}
        self.assertEqual(counts, {"fit": 1, "validation": 2, "cross_fit_validation": 1, "excluded": 2})
        for spec in value["outputs"].values():
            self.assertEqual(hashlib.sha256(Path(spec["path"]).read_bytes()).hexdigest(), spec["sha256"])
        # These exact columns/rosters are consumed by rec_content_adapter.prepare.
        flat = labels.reset_index()
        self.assertTrue(np.array_equal(requests.user_idx, flat.groupby("gid").user_idx.first().sort_index()))

    def test_existing_output_is_never_overwritten(self):
        self.output.mkdir()
        sentinel = self.output / "keep.txt"
        sentinel.write_text("keep")
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.assertEqual(sentinel.read_text(), "keep")

    def test_bad_or_unmatched_source_sha_rejected(self):
        for inputs in ({str(self.train.relative_to(self.root)): "0" * 64},
                       {"different.parquet": hashlib.sha256(self.train.read_bytes()).hexdigest()},
                       {str(self.train.relative_to(self.root)): "x" * 64}):
            with self.subTest(inputs=inputs):
                self.write_manifest(inputs=inputs)
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    self.prepare()
                self.assertFalse(self.output.exists())

    def test_duplicate_request_even_identical_or_conflicting_is_rejected(self):
        for duplicate in (self.rows[0], request(1, 99, [9], exposure(1, 0, 3))):
            with self.subTest(duplicate=duplicate):
                self.write_source(self.rows + [duplicate])
                with self.assertRaisesRegex(ValueError, "unique"):
                    self.prepare()

    def test_nonbinary_or_nonfinite_click_is_rejected(self):
        for value in (None, float("nan"), float("inf"), -1, 2, .5):
            with self.subTest(click=value):
                self.write_source([request(1, 0, [], exposure(0, value, 1))])
                with self.assertRaisesRegex(ValueError, "binary click"):
                    self.prepare()

    def test_invalid_event_time_is_rejected(self):
        for value in (None, float("nan"), float("inf"), 0, -1):
            with self.subTest(timestamp=value):
                self.write_source([request(1, 0, [], exposure(0, 1, value))])
                with self.assertRaisesRegex(ValueError, "timestamp"):
                    self.prepare()

    def test_empty_exposures_cannot_be_assigned_a_causal_time(self):
        self.write_source([request(1, 0, [])])
        with self.assertRaisesRegex(ValueError, "empty timestamped"):
            self.prepare()

    def test_invalid_request_user_item_and_history_ids_are_rejected(self):
        for row in (request(-1, 0, [], exposure(0, 1, 1)),
                    request(1, -1, [], exposure(0, 1, 1)),
                    request(1, 0, [], exposure(-1, 1, 1)),
                    request(1, 0, [-1], exposure(0, 1, 1)),
                    request(1, 0, [None], exposure(0, 1, 1)),
                    request(1, 0, None, exposure(0, 1, 1))):
            with self.subTest(row=row):
                self.write_source([row])
                with self.assertRaises(ValueError):
                    self.prepare()

    def test_float_identity_schema_is_rejected(self):
        arrow = schema().set(0, pa.field("request_idx", pa.float64()))
        self.write_source(self.rows, arrow)
        with self.assertRaisesRegex(ValueError, "integer Arrow"):
            self.prepare()

    def test_test_schema_and_test_directory_are_rejected_without_reading_test(self):
        arrow = schema()
        arrow = arrow.set(3, pa.field("rec_results", arrow.field(3).type))
        rows = [{**r, "rec_results": r["rec_result_details_with_idx"]} for r in self.rows]
        self.write_source(rows, arrow)
        with self.assertRaisesRegex(ValueError, "training schema"):
            self.prepare()
        self.train = self.root / "datasets/recommendation_test/train.parquet"
        self.train.parent.mkdir()
        self.write_source(self.rows)
        with mock.patch.object(refit, "_fingerprint", side_effect=AssertionError("must not hash test")):
            with self.assertRaisesRegex(ValueError, "Test source"):
                self.prepare()

    def test_bad_manifest_boundaries_and_scene_are_rejected(self):
        for change in ({"fit_end_exclusive": 20}, {"official_test_first_event": float("nan")},
                       {"fit_end_exclusive": True}, {"official_test_first_event": 20.5}, {"scene": "search"}):
            with self.subTest(change=change):
                self.write_manifest(**change)
                with self.assertRaises(ValueError):
                    self.prepare()

    def test_source_changed_after_initial_hash_is_not_published(self):
        original = refit._fingerprint
        mutated = False
        def mutate(path):
            nonlocal mutated
            result = original(path)
            if path == self.train and not mutated:
                mutated = True
                rows = [request(1, 0, [], exposure(0, 1, 2))]
                pq.write_table(pa.Table.from_pylist(rows, schema=schema()), self.train)
            return result
        with mock.patch.object(refit, "_fingerprint", side_effect=mutate):
            with self.assertRaisesRegex(RuntimeError, "changed"):
                self.prepare()
        self.assertFalse(self.output.exists())

    def test_cli_uses_only_explicit_inputs(self):
        script = Path(refit.__file__).resolve().parents[2] / "scripts/prepare_rec_final_refit.py"
        result = subprocess.run([sys.executable, str(script), "--train-path", str(self.train),
                                 "--split-manifest", str(self.manifest), "--output", str(self.output),
                                 "--project-root", str(self.root)], capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.strip(), str(self.output / "manifest.json"))
        self.assertTrue((self.output / "manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
