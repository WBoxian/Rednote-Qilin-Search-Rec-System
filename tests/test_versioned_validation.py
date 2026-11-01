import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fastapi import HTTPException

from backend.online.api.main import create_app
from backend.online.bundle import file_sha256


class VersionedValidationTests(unittest.TestCase):
    def test_bound_current_user_moves_first_without_mutating_frozen_order_or_labels(self):
        from backend.online.api.main import _personalize_validation
        original = {"bound_to_active_bundle": True, "examples": [
            {"request_id":1,"source_user_idx":8,"overlap":{"ranking":0}},
            {"request_id":2,"source_user_idx":7,"overlap":{"ranking":1}},
            {"request_id":3,"source_user_idx":9,"overlap":{"ranking":10}}]}
        result = _personalize_validation(original, context_user_idx=7)
        self.assertEqual([r["request_id"] for r in result["examples"]],[2,1,3])
        self.assertTrue(result["examples"][0]["is_current_user"])
        self.assertEqual([r["request_id"] for r in original["examples"]],[1,2,3])
        self.assertNotIn("is_current_user",original["examples"][1])

    def test_bound_missing_current_user_replays_active_model_without_changing_test_counts(self):
        from backend.online.api.main import _ensure_current_user_example
        app_ctx=SimpleNamespace(get_cached_user_validation_example=Mock(return_value={"model_bundle_id":"old"}),set_cached_user_validation_example=Mock())
        pipeline=object();runtime=SimpleNamespace(get_pipeline=lambda scene:pipeline)
        snapshot={"bound_to_active_bundle":True,"model_bundle_id":"new","sampled_groups":6192,"examples":[{"request_id":1,"source_user_idx":8}]}
        row={"request_id":2,"source_user_idx":7,"model_bundle_id":"new","scope":"current_user_request_replay","split":"train","evaluation_included":False,"sample_note":"训练请求回放"}
        with patch("backend.online.user_example.build_current_user_example",return_value=row) as replay:
            result=_ensure_current_user_example(snapshot,app_ctx,runtime,"search","hard",7,800,5)
        replay.assert_called_once_with(pipeline,7,snapshot)
        self.assertEqual(result["examples"][0],row)
        self.assertEqual(result["sampled_groups"],6192)
        self.assertEqual(snapshot["examples"],[{"request_id":1,"source_user_idx":8}])

    def test_current_user_training_replay_keeps_labels_out_of_inference(self):
        import pandas as pd
        from backend.online.user_example import build_current_user_example
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);target=root/"datasets/recommendation_train";target.mkdir(parents=True)
            pd.DataFrame({"request_idx":[2,1],"user_idx":[7,7],"query":["", ""],"recent_clicked_note_idxs":[[0],[0]],"rec_result_details_with_idx":[[{"note_idx":5,"click":1}],[{"note_idx":3,"click":1}]]}).to_parquet(target/"part.parquet",index=False)
            state=SimpleNamespace(scene="rec",bundle=SimpleNamespace(root=root,bundle_id="current"),_fetch_notes=lambda ids:pd.DataFrame({"note_idx":ids,"note_title":[str(i) for i in ids],"note_content":[""]*len(ids)}))
            feed={"model_bundle_id":"current","stage_top10_ids":{"recall":[3,5],"preranking":[],"ranking":[]},"stages":{"preranking":{"enabled":False},"ranking":{"enabled":False}}}
            pipeline=SimpleNamespace(state=state,build_feed=Mock(return_value=feed))
            row=build_current_user_example(pipeline,7,{})
            self.assertEqual(row["request_id"],1)
            self.assertEqual(row["split"],"train")
            self.assertFalse(row["evaluation_included"])
            self.assertEqual(row["dssm_top10"][0]["rank_in_true"],1)
            self.assertIsNone(row["dssm_top10"][1]["rank_in_true"])
            kwargs=pipeline.build_feed.call_args.kwargs
            self.assertFalse(kwargs["record_exposure"])
            self.assertEqual(set(kwargs["request_context"].columns),{"request_idx","user_idx","query","recent_clicked_note_idxs"})
            self.assertEqual(kwargs["request_context"].recent_clicked_note_idxs.iloc[0],[0])
            self.assertEqual(row["ranking_top10"],[])

    def fixture(self, root):
        with patch.dict(os.environ, {"QILIN_ASYNC_PREWARM": "0"}):
            app = create_app(tag="hard")
        context = app.state.ctx
        context._snapshot_root = root/"legacy"
        old = context._validation_snapshot_path("search", "hard", 800, 5)
        old.parent.mkdir(parents=True)
        old.write_text(json.dumps({"examples": [{"request_id": -1}], "scope": "stale"}))
        current = root/"current.json"
        current.write_text(json.dumps({"scene": "search", "tag": "hard", "split": "test", "scope": "frozen_full_test_examples",
            "sampled_groups": 6192, "effective_groups": 6192, "examples": [{"request_id": 123, "source_user_idx": 8}]}))
        bundle = SimpleNamespace(root=root, bundle_id="new-model", manifest={"evaluation": {"validation_snapshot": {
            "path": "current.json", "sha256": file_sha256(current)}}})
        state = SimpleNamespace(bundle_active=True, bundle=bundle, build_user_validation_example=Mock(side_effect=AssertionError("No legacy recompute")))
        context.get_runtime = Mock(return_value=SimpleNamespace(get_pipeline=lambda scene: SimpleNamespace(state=state)))
        endpoint = next(route.endpoint for route in app.routes if getattr(route, "path", "") == "/api/validation")
        context._test_note_endpoint = next(route.endpoint for route in app.routes if getattr(route, "path", "") == "/api/validation/note")
        return context, state, current, endpoint

    def test_diagnostic_note_route_only_uses_request_bound_notes_and_history(self):
        with tempfile.TemporaryDirectory() as directory:
            context, state, current, _ = self.fixture(Path(directory))
            payload = json.loads(current.read_text())
            payload["examples"][0].update(ranking_top10=[{"note_idx": 11}], source_recent_behaviors=[{"note_idx": 12}])
            payload["examples"].append({"request_id": 999, "ranking_top10_ids": [13]})
            payload["notes"] = [{"note_idx": note, "note_title": f"note {note}", "note_content": "content"} for note in [11, 12, 13]]
            current.write_text(json.dumps(payload))
            state.bundle.manifest["evaluation"]["validation_snapshot"]["sha256"] = file_sha256(current)
            for note in [11, 12]:
                result = context._test_note_endpoint(scene="search", tag="hard", request_id=123, note_idx=note)
                self.assertEqual(result["detail"]["note_idx"], note)
                self.assertEqual(result["detail"]["scope"], "frozen_full_test_examples")
            for request, note in [(123, 13), (123, 90), (888, 11)]:
                with self.assertRaises(HTTPException) as error:
                    context._test_note_endpoint(scene="search", tag="hard", request_id=request, note_idx=note)
                self.assertEqual(error.exception.status_code, 404)
            state.build_user_validation_example.assert_not_called()

    def test_actual_route_serves_bound_examples_and_never_legacy_personalization(self):
        with tempfile.TemporaryDirectory() as directory:
            context, state, current, endpoint = self.fixture(Path(directory))
            with patch.object(context, "_load_json_snapshot", wraps=context._load_json_snapshot) as loader:
                result = endpoint(scene="search", tag="hard", max_groups=800, example_limit=5, user_idx=7)
            payload = result["validation"]
            self.assertEqual(payload["model_bundle_id"], "new-model")
            self.assertEqual(payload["examples"], [{"request_id": 123, "source_user_idx": 8, "is_current_user": False}])
            self.assertEqual(payload["split"], "test")
            loader.assert_called_once_with(current)
            state.build_user_validation_example.assert_not_called()

    def test_missing_bound_snapshot_returns_503_instead_of_old_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            _, state, current, endpoint = self.fixture(Path(directory))
            for missing in ["descriptor", "file"]:
                with self.subTest(missing=missing):
                    if missing == "descriptor":
                        saved = state.bundle.manifest.pop("evaluation")
                    else:
                        state.bundle.manifest["evaluation"] = saved
                        current.unlink()
                    with self.assertRaises(HTTPException) as error:
                        endpoint(scene="search", tag="hard", max_groups=800, example_limit=5, user_idx=7)
                    self.assertEqual(error.exception.status_code, 503)

    def test_stale_explicit_bundle_id_is_rejected_for_examples_and_note_details(self):
        with tempfile.TemporaryDirectory() as directory:
            context, _, _, endpoint = self.fixture(Path(directory))
            with self.assertRaises(HTTPException) as error:
                endpoint(scene="search", tag="hard", max_groups=800, example_limit=5, user_idx=7, bundle_id="stale")
            self.assertEqual(error.exception.status_code, 409)
            with self.assertRaises(HTTPException) as error:
                context._test_note_endpoint(scene="search", tag="hard", request_id=123, note_idx=11, bundle_id="stale")
            self.assertEqual(error.exception.status_code, 409)

    def test_cache_is_separated_by_bundle_id_and_returns_detached_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            context, state, current, _ = self.fixture(Path(directory))
            first, _ = context.get_validation_cached("search", "hard", 800, 5, 7)
            first["examples"].clear()
            again, hit = context.get_validation_cached("search", "hard", 800, 5, 7)
            self.assertTrue(hit)
            self.assertEqual(len(again["examples"]), 1)
            updated = json.loads(current.read_text())
            updated["examples"][0]["request_id"] = 456
            current.write_text(json.dumps(updated))
            state.bundle.bundle_id = "second-model"
            state.bundle.manifest["evaluation"]["validation_snapshot"]["sha256"] = file_sha256(current)
            second, hit = context.get_validation_cached("search", "hard", 800, 5, 7)
            self.assertFalse(hit)
            self.assertEqual(second["examples"][0]["request_id"], 456)

    def test_tampered_snapshot_cannot_hide_behind_cached_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            context, _, current, _ = self.fixture(Path(directory))
            context.get_validation_cached("search", "hard", 800, 5, 7)
            current.write_text(current.read_text().replace("123", "9999"))
            with self.assertRaisesRegex(ValueError, "asset mismatch"):
                context.get_validation_cached("search", "hard", 800, 5, 7)


if __name__ == "__main__":
    unittest.main()
