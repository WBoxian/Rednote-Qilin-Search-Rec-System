import json
import os
import pickle
import tempfile
from pathlib import Path
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import torch

from backend.online.api.main import create_app
from backend.online.cold_start.popular import build_hot_candidates
from backend.online.item_features import STATIC_ITEM_FEATURES
from backend.online.live_context import LiveContextBuilder
from backend.online.pipeline import OnlineScenePipeline, SceneServingState, ServingAppState
from backend.online.ranking.dien import DIENPredictor
from backend.online.recall.service import fetch_recall_candidates


class MemoryCache:
    def __init__(self):
        self.history = []
        self.exposures = []
        self.writes = []

    def get_user_history_notes(self, *args, **kwargs):
        return list(self.history)

    def get_recent_exposed_notes(self, *args, **kwargs):
        return list(self.exposures)

    def record_exposed_notes(self, uid, scene, notes):
        self.writes.append(list(notes))
        self.exposures = list(dict.fromkeys(notes + self.exposures))

    def record_click(self, uid, scene, note, **kwargs):
        self.history = [note] + [value for value in self.history if value != note]


def serving_fixture(scene="search"):
    state = SceneServingState.__new__(SceneServingState)
    state.scene = scene
    state.group_key = "search_idx" if scene == "search" else "request_idx"
    state.bundle_active = True
    state.bundle_id = "tested-bundle"
    state.live_recall_rank_cap = 320
    state.live_gbdt_topn = 180
    state.hot_note_pool = [91, 92, 93]
    state.realtime_cache = MemoryCache()
    state._feed_cache_lock = threading.Lock()
    state._stage_cache_lock = threading.Lock()
    state._context_lock = threading.Lock()
    state._memory_history = {}
    state._feed_cache = {}
    state._stage_cache = {}
    state._feed_cache_ttl_sec = 12
    state._stage_cache_ttl_sec = 45
    state._app_state = None
    builder = LiveContextBuilder.__new__(LiveContextBuilder)
    builder.scene, builder.group_key = scene, state.group_key
    builder.profiles = {7: {"gender": "female"}}
    builder.vocabularies = {key: {"female": 1} for key in ("gender", "platform", "age", "location")}
    builder.training_contexts = {7: {"recent_clicked_note_idxs": [5, 6], "query": "train query"}}
    builder.query_catalog = [("train query", 1)]
    state.live_context_builder = builder
    state.preprocess_search_query = lambda query, user_idx=None: {"terms": [query]}
    state._fetch_notes = lambda ids: pd.DataFrame({"note_idx": ids, "note_title": ["note"] * len(ids), "note_content": [""] * len(ids), "image_path": [[]] * len(ids)})
    state.fetch_item_features = lambda ids: pd.DataFrame({"note_idx": ids, "item_features_found": [True] * len(ids), **{name: np.zeros(len(ids)) for name in STATIC_ITEM_FEATURES}})
    state.gbdt_predictor = SimpleNamespace(feature_cols=["gender_enc"], predict_with_components=lambda frame: {"lgb_score": 10-frame.note_idx.to_numpy(), "xgb_score": np.zeros(len(frame)), "gbdt_score": 10-frame.note_idx.to_numpy()})
    state.predict_dien = lambda frame, history_note_ids=None: 10-frame.note_idx.to_numpy()
    seen = []
    def recall(context, max_rank):
        seen.append(context.copy())
        row = context.iloc[0]
        return pd.DataFrame({state.group_key: [row[state.group_key]]*3, "user_idx": [row.user_idx]*3,
            "note_idx": [1, 2, 3], "rank": [1, 2, 3], "score_ann": [.9, .8, .7], "recall_score": [.9, .8, .7]})
    state.fetch_live_recall_candidates = recall
    return OnlineScenePipeline(state), seen


class LiveServingTests(unittest.TestCase):
    def test_direct_bundle_coarse_wrapper_does_not_accidentally_call_fine_blend(self):
        pipe, _ = serving_fixture()
        expected = np.asarray([.2, .8], dtype=np.float32)
        pipe.state.gbdt_predictor = Mock()
        pipe.state.gbdt_predictor.predict_with_components.return_value = {"gbdt_score": expected}
        frame = pd.DataFrame({"note_idx": [1, 2]})
        np.testing.assert_array_equal(pipe.state.predict_gbdt(frame), expected)
        pipe.state.gbdt_predictor.predict.assert_not_called()

    def test_recall_only_actual_feed_keeps_entire_pool_and_bypasses_learned_stages(self):
        for scene in ["search", "rec"]:
            with self.subTest(scene=scene):
                pipe, _ = serving_fixture(scene)
                pipe.state.bundle = SimpleNamespace(manifest={"ranking_strategy": "recall_only"})
                pipe.state.bundle_input_cap = pipe.state.bundle_coarse_cap = 500
                caps = []
                def recall(context, max_rank):
                    caps.append(max_rank)
                    return pd.DataFrame({pipe.state.group_key: [int(context.iloc[0][pipe.state.group_key])]*500,
                        "user_idx": [7]*500, "note_idx": list(range(500, 0, -1)), "rank": list(range(1, 501)),
                        "recall_score": np.linspace(1., .01, 500)})
                pipe.state.fetch_live_recall_candidates = recall
                pipe.state.gbdt_predictor = Mock()
                pipe.state.predict_dien = Mock()
                pipe.state.fetch_item_features = Mock(side_effect=AssertionError("No model features in recall-only serving"))
                response = pipe.build_feed(7, "q", 1, 500, record_exposure=False)
                self.assertEqual(caps, [500])
                self.assertEqual(response["total"], 500)
                self.assertEqual([item["note_idx"] for item in response["items"]], list(range(500, 0, -1)))
                pipe.state.gbdt_predictor.predict_with_components.assert_not_called()
                pipe.state.predict_dien.assert_not_called()
                self.assertFalse(response["stages"]["ranking"]["enabled"])
                self.assertFalse(response["stages"]["ranking"]["schema_enabled"])
                self.assertTrue(all(item["labels"]["click"] is None for item in response["items"]))

    def test_train_cutoff_excludes_future_and_entire_spanning_requests(self):
        for scene in ["search", "rec"]:
            with self.subTest(scene=scene), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                vocab = root/"features/vocab_dict"
                vocab.mkdir(parents=True)
                for name in ["gender", "platform", "age", "location"]:
                    (vocab/f"{name}.pkl").write_bytes(pickle.dumps({"UNK": 0}))
                folder = root/"datasets"/("search_train" if scene == "search" else "recommendation_train")
                folder.mkdir(parents=True)
                key = "search_idx" if scene == "search" else "request_idx"
                nested = "search_result_details_with_idx" if scene == "search" else "rec_result_details_with_idx"
                timestamp = "search_timestamp" if scene == "search" else "request_timestamp"
                pd.DataFrame({key: [1, 2, 3], "user_idx": [7, 7, 7], "query": ["past", "future", "spanning"],
                    "recent_clicked_note_idxs": [[1], [2], [3]], nested: [
                        [{"note_idx": 91, "click": 1, timestamp: 9}],
                        [{"note_idx": 92, "click": 1, timestamp: 11}],
                        [{"note_idx": 93, "click": 1, timestamp: 8}, {"note_idx": 94, "click": 1, timestamp: 12}]]}).to_parquet(folder/"train.parquet", index=False)
                builder = LiveContextBuilder(root, scene, {}, cutoff_exclusive=10)
                self.assertEqual(builder.training_request_count, 1)
                self.assertEqual(builder.seed_history(7), [1])
                self.assertEqual(builder.hot_note_ids, [91])
                self.assertEqual(builder.query_catalog, [("past", 1)] if scene == "search" else [])

    def test_explicit_replay_uses_only_supplied_context_and_never_logs_exposure(self):
        pipe, seen = serving_fixture("search")
        context = pd.DataFrame({"search_idx": [987], "user_idx": [7], "query": ["q"], "recent_clicked_note_idxs": [[30]], "event_timestamp": [10]})
        response = pipe.build_feed(7, "q", 1, 2, request_context=context, record_exposure=False)
        self.assertEqual(seen[0].search_idx.iloc[0], 987)
        self.assertEqual(seen[0].recent_clicked_note_idxs.iloc[0], [30])
        self.assertEqual(response["request_id"], 987)
        self.assertEqual(response["scope"], "explicit_context_replay")
        self.assertTrue(all(item["labels"]["click"] is None for item in response["items"]))
        self.assertEqual(pipe.state.realtime_cache.writes, [])

    def test_explicit_replay_rejects_labels_mismatched_user_and_writes(self):
        pipe, seen = serving_fixture()
        context = pd.DataFrame({"search_idx": [987], "user_idx": [7], "query": ["q"], "recent_clicked_note_idxs": [[30]]})
        with self.assertRaises(ValueError):
            pipe.build_feed(7, "q", 1, 2, request_context=context)
        with self.assertRaises(ValueError):
            pipe.build_feed(8, "q", 1, 2, request_context=context, record_exposure=False)
        with self.assertRaises(ValueError):
            pipe.build_feed(7, "q", 1, 2, request_context=context.assign(click=1), record_exposure=False)
        self.assertEqual(seen, [])

    def test_fresh_numeric_request_identity_is_exact_in_javascript(self):
        pipe, _ = serving_fixture()
        context = pipe.state.build_live_context(7, "q")
        request_id = int(context.search_idx.iloc[0])
        self.assertGreaterEqual(request_id, 0)
        self.assertLessEqual(request_id, 2**53-1)
        self.assertEqual(int(float(request_id)), request_id)

    def test_note_details_reuse_delivered_scores_without_new_feed_or_test_labels(self):
        pipe, seen = serving_fixture()
        response = pipe.build_feed(7, "q", 1, 1)
        item = response["items"][0]
        writes_before = list(pipe.state.realtime_cache.writes)
        pipe.state.build_feed = Mock(side_effect=AssertionError("detail must not rerank"))
        pipe.state.get_feat_req = Mock(side_effect=AssertionError("detail must not read test labels"))
        detail = pipe.state.get_note_detail(7, response["request_id"], item["note_idx"], query="q")
        self.assertEqual(detail["scores"]["dien"], item["scores"]["dien"])
        self.assertEqual(detail["score_scope"], "delivered_request")
        self.assertIsNone(detail["labels"]["click"])
        self.assertEqual(len(seen), 1)
        self.assertEqual(pipe.state.realtime_cache.writes, writes_before)

    def test_requested_user_never_borrows_test_user_and_unknown_has_empty_history(self):
        pipe, _ = serving_fixture()
        context = pipe.state.build_live_context(99999, "typed query")
        self.assertEqual(context.user_idx.iloc[0], 99999)
        self.assertEqual(context.recent_clicked_note_idxs.iloc[0], [])
        self.assertEqual(context["query"].iloc[0], "typed query")
        self.assertNotIn("click", context)
        self.assertNotIn("y_multi", context)

    def test_actual_feed_sends_current_query_user_history_to_recall_and_fine_rank(self):
        pipe, seen = serving_fixture()
        pipe.state.realtime_cache.history = [88, 77]
        fine = Mock(side_effect=lambda frame, **kwargs: 10-frame.note_idx.to_numpy())
        pipe.state.predict_dien = fine
        response = pipe.build_feed(7, "new typed query", 1, 2)
        self.assertEqual(seen[0]["query"].iloc[0], "new typed query")
        self.assertEqual(seen[0].user_idx.iloc[0], 7)
        self.assertEqual(seen[0].recent_clicked_note_idxs.iloc[0], [88, 77])
        self.assertEqual(fine.call_args.kwargs["history_note_ids"], [88, 77])
        self.assertEqual(fine.call_args.args[0]["query"].unique().tolist(), ["new typed query"])
        self.assertEqual(response["model_bundle_id"], "tested-bundle")
        self.assertTrue(response["trace_id"])

    def test_cache_hit_honors_new_exclusions_and_uses_fresh_live_identity(self):
        for scene in ["search", "rec"]:
            with self.subTest(scene=scene):
                pipe, seen = serving_fixture(scene)
                first = pipe.build_feed(7, "q", 1, 2)
                blocked = first["items"][0]["note_idx"]
                second = pipe.build_feed(7, "q", 1, 2, exclude_note_ids=[blocked])
                self.assertTrue(second["cache_hit"])
                self.assertEqual(len(seen), 1)
                self.assertNotEqual(first["request_id"], second["request_id"])
                self.assertNotEqual(first["trace_id"], second["trace_id"])
                self.assertNotIn(blocked, [item["note_idx"] for item in second["items"]])

    def test_only_delivered_page_items_are_exposed_on_cache_hits_and_misses(self):
        pipe, _ = serving_fixture("rec")
        first = pipe.build_feed(7, "", 1, 1)
        second = pipe.build_feed(7, "", 2, 1)
        expected = [[item["note_idx"] for item in response["items"]] for response in [first, second]]
        self.assertEqual(pipe.state.realtime_cache.writes, expected)
        self.assertEqual([len(value) for value in expected], [1, 1])

    def test_prewarm_never_writes_exposure_and_does_not_evict_candidates(self):
        pipe, _ = serving_fixture("rec")
        pipe.prewarm_homepage_feed(page_size=2)
        self.assertEqual(pipe.state.realtime_cache.writes, [])
        response = pipe.build_feed(7, "", 1, 2)
        self.assertEqual(len(response["items"]), 2)
        self.assertEqual(len(pipe.state.realtime_cache.writes), 1)

    def test_missing_image_keeps_model_order_for_search_and_rec(self):
        frame = pd.DataFrame({"note_idx": [1, 2], "image_path": [[], ["existing.jpg"]]})
        for scene in ["search", "rec"]:
            pipe, _ = serving_fixture(scene)
            with patch("backend.online.pipeline._has_existing_image", side_effect=[False, True]) as check:
                self.assertEqual(pipe._select_page_rows(frame, 1, 1).note_idx.tolist(), [1])
            check.assert_not_called()

    def test_new_click_changes_current_history_and_evicts_other_scene_caches(self):
        search, _ = serving_fixture("search")
        rec, _ = serving_fixture("rec")
        app = ServingAppState("hard", 500, 1000)
        app._states = {"search": search.state, "rec": rec.state}
        app._linkage_cache[7] = (999999999., {"keywords": ["old"]})
        for pipe in [search, rec]:
            pipe.state._app_state = app
            pipe.build_feed(7, "q", 1, 1, record_exposure=False)
        self.assertTrue(rec.state._feed_cache)
        search.state.record_click(7, 44)
        self.assertFalse(search.state._feed_cache)
        self.assertFalse(rec.state._feed_cache)
        self.assertFalse(rec.state._stage_cache)
        self.assertNotIn(7, app._linkage_cache)
        self.assertEqual(search.state.build_live_context(7, "q").recent_clicked_note_idxs.iloc[0][0], 44)

    def test_history_and_query_changes_cannot_reuse_previous_inference_pool(self):
        pipe, seen = serving_fixture()
        pipe.build_feed(7, "first", 1, 1)
        changed_query = pipe.build_feed(7, "second", 1, 1)
        pipe.state.realtime_cache.history = [99]
        changed_history = pipe.build_feed(7, "second", 1, 1)
        self.assertFalse(changed_query["cache_hit"])
        self.assertFalse(changed_history["cache_hit"])
        self.assertEqual(len(seen), 3)

    def test_missing_bundle_is_safe_train_catalog_fallback_not_old_model(self):
        pipe, seen = serving_fixture()
        pipe.state.bundle_active = False
        pipe.state.gbdt_predictor = Mock()
        response = pipe.build_feed(9999, "typed", 1, 2)
        self.assertEqual([item["note_idx"] for item in response["items"]], [91, 92])
        self.assertEqual(seen, [])
        pipe.state.gbdt_predictor.predict_with_components.assert_not_called()
        self.assertEqual(response["degraded_reason"], "validated_model_bundle_unavailable")

    def test_cold_catalog_candidates_never_reuse_exposure_rows_or_counters(self):
        context = pd.DataFrame({"request_idx": [1], "user_idx": [7], "note_idx": [1234], "click": [1], "accum_like_num": [999]})
        self.assertTrue(build_hot_candidates(context, 20).empty)
        cold = build_hot_candidates(context, 20, [91, 92])
        self.assertEqual(cold.note_idx.tolist(), [91, 92])
        self.assertNotIn("click", cold)
        self.assertNotIn("accum_like_num", cold)

    def test_new_dien_query_overrides_nonzero_reference_and_failure_is_explicit(self):
        predictor = DIENPredictor.__new__(DIENPredictor)
        predictor.device = "cpu"
        predictor.dien_train = SimpleNamespace(EMB_DIM=2)
        store = Mock()
        predictor.stores = {"q": store}
        predictor._encode_query_vec = Mock(return_value=np.asarray([[0., 1.]], dtype=np.float32))
        frame = pd.DataFrame({"query": ["actual"], "query_ref_request_id": [99]})
        self.assertEqual(predictor._get_runtime_query_tensor(frame, np.asarray([1])).tolist(), [[0., 1.]])
        store.get_tensor.assert_not_called()
        predictor._encode_query_vec.return_value = None
        with self.assertRaises(RuntimeError):
            predictor._get_runtime_query_tensor(frame, np.asarray([1]))

    def test_readiness_fails_http_status_when_any_model_missing(self):
        with patch.dict(os.environ, {"QILIN_ASYNC_PREWARM": "0"}):
            app = create_app()
        app.state.ctx.get_runtime = lambda tag: SimpleNamespace(get_pipeline=lambda scene: SimpleNamespace(state=SimpleNamespace(readiness=lambda: {"ready": False, "error": "missing"} if scene == "search" else {"ready": True})))
        health = next(route.endpoint for route in app.routes if getattr(route, "path", "") == "/api/health")
        response = health(verbose=1)
        self.assertEqual(response.status_code, 503)
        self.assertFalse(json.loads(response.body)["ok"])

    def test_readiness_checks_only_configured_default_scene_bundles(self):
        with patch.dict(os.environ, {"QILIN_ASYNC_PREWARM": "0", "QILIN_TAG_SEARCH": "hard", "QILIN_TAG_REC": "hard"}):
            app = create_app()
        requested = []
        def runtime(tag):
            requested.append(tag)
            return SimpleNamespace(get_pipeline=lambda scene: SimpleNamespace(state=SimpleNamespace(readiness=lambda: {"ready": True})))
        app.state.ctx.get_runtime = runtime
        health = next(route.endpoint for route in app.routes if getattr(route, "path", "") == "/api/health")
        self.assertTrue(health(verbose=1)["ok"])
        self.assertEqual(requested, ["hard", "hard"])

    def test_active_search_never_reads_provider_dpr_or_saved_mixture(self):
        request = pd.DataFrame({"search_idx": [1], "user_idx": [7], "query": ["q"], "recent_clicked_note_idxs": [[5]],
            "dpr_results": [[np.asarray([999., 900.])]], "bm25_results": [[np.asarray([11., .9])]]})
        ann = pd.DataFrame({"note_idx": [10], "score_ann": [.8]})
        recaller = SimpleNamespace(fetch_candidates=lambda **kwargs: ann)
        with patch("backend.online.recall.service._fetch_precomputed_recall") as precomputed:
            result = fetch_recall_candidates(1, 20, "search_idx", request, lambda gid: request, recaller, None)
        precomputed.assert_not_called()
        self.assertNotIn(999, result.note_idx.tolist())
        self.assertIn(10, result.note_idx.tolist())


if __name__ == "__main__":
    unittest.main()
