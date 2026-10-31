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
            self.assertEqual(payload["examples"], [{"request_id": 123, "source_user_idx": 8}])
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
