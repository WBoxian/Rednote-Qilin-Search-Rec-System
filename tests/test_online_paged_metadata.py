"""Display-only metadata pagination; inference pools and scores stay immutable."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch, call

import numpy as np
import pandas as pd

from test_online_live_context import serving_fixture


def metadata(ids):
    # Reverse return order proves keyed enrichment cannot reorder the selected page.
    return pd.DataFrame([dict(note_idx=i, note_title=f"title-{i}", note_content=f"content-{i}",
                              image_path=[f"cover-{i}.jpg", f"second-{i}.jpg"],
                              accum_like_num=i+10, accum_collect_num=i+20, accum_comment_num=i+30)
                         for i in reversed(ids)])


class PagedDisplayMetadataTests(unittest.TestCase):
    def setUp(self):
        images = patch("backend.online.pipeline._existing_images", side_effect=lambda values: list(values))
        images.start()
        self.addCleanup(images.stop)

    def fixture(self, scene="rec", provided=False):
        pipe, _ = serving_fixture(scene)
        pipe.state.bundle = SimpleNamespace(manifest={"ranking_strategy": "recall_only"})
        pipe.state.bundle_input_cap = pipe.state.bundle_coarse_cap = 500
        ids = np.arange(500, dtype=np.int64)  # Real item0 remains a displayable candidate.
        scores = np.linspace(1., .01, 500)
        def recall(context, max_rank):
            self.assertEqual(max_rank, 500)
            result = pd.DataFrame({pipe.state.group_key: [int(context.iloc[0][pipe.state.group_key])]*500,
                                   "user_idx": [7]*500, "note_idx": ids, "rank": np.arange(1, 501),
                                   "recall_score": scores, "score_ann": scores})
            if provided:
                result["note_title"] = "provided title"
                result["image_path"] = [["provided.jpg"] for _ in ids]
                result["accum_like_num"] = 991
            return result
        pipe.state.fetch_live_recall_candidates = Mock(side_effect=recall)
        pipe.state._fetch_notes = Mock(side_effect=metadata)
        pipe.state.gbdt_predictor = Mock()
        pipe.state.predict_dien = Mock()
        return pipe, ids, scores

    def assert_cards(self, response, expected):
        self.assertEqual([r["note_idx"] for r in response["items"]], list(expected))
        for card in response["items"]:
            i = card["note_idx"]
            self.assertEqual(card["title"], f"title-{i}")
            self.assertEqual(card["cover_image"], f"cover-{i}.jpg")
            self.assertEqual(card["image_count"], 2)
            self.assertEqual(card["accum_like_num"], i+10)
            self.assertEqual(card["accum_collect_num"], i+20)
            self.assertEqual(card["accum_comment_num"], i+30)
            self.assertAlmostEqual(card["scores"]["final"], np.linspace(1., .01, 500)[i])
            self.assertEqual(card["stage_ranks"]["recall"], i+1)
            self.assertEqual(card["stage_ranks"]["ranking"], i+1)
            self.assertIsNone(card["labels"]["click"])

    def cached_pool(self, pipe):
        self.assertEqual(len(pipe.state._feed_cache), 1)
        return next(iter(pipe.state._feed_cache.values()))[1]

    def test_miss_hit_page2_and_exclusions_fetch_only_each_actual_page(self):
        pipe, ids, scores = self.fixture()
        first = pipe.build_feed(7, "", 1, 20, record_exposure=False)
        self.assertFalse(first["cache_hit"])
        self.assertEqual(first["total"], 500)
        self.assert_cards(first, range(20))
        pipe.state._fetch_notes.assert_called_once_with(list(range(20)))
        before = self.cached_pool(pipe).copy(deep=True)
        np.testing.assert_array_equal(before.note_idx, ids)
        np.testing.assert_array_equal(before.recall_score, scores)
        np.testing.assert_array_equal(before.final_score, scores)
        np.testing.assert_array_equal(before["rank"], np.arange(1, 501))
        self.assertNotIn("note_title", before)  # No full-pool display enrichment in inference cache.
        for page, excluded, expected in [(1, [], range(20)), (2, [], range(20, 40)),
                                          (1, [0, 2, 4], [i for i in range(500) if i not in {0, 2, 4}][:20])]:
            pipe.state._fetch_notes.reset_mock()
            result = pipe.build_feed(7, "", page, 20, exclude_note_ids=excluded, record_exposure=False)
            self.assertTrue(result["cache_hit"])
            self.assertEqual(result["total"], 500-len(excluded))
            self.assert_cards(result, expected)
            pipe.state._fetch_notes.assert_called_once_with(list(expected))
            pd.testing.assert_frame_equal(before, self.cached_pool(pipe))
        pipe.state.fetch_live_recall_candidates.assert_called_once()
        pipe.state.gbdt_predictor.predict_with_components.assert_not_called()
        pipe.state.predict_dien.assert_not_called()

    def test_miss_with_exclusions_fetches_eligible_page_not_original_head(self):
        pipe, _, _ = self.fixture()
        result = pipe.build_feed(7, "", 1, 20, exclude_note_ids=list(range(15)), record_exposure=False)
        self.assertFalse(result["cache_hit"])
        self.assert_cards(result, range(15, 35))
        pipe.state._fetch_notes.assert_called_once_with(list(range(15, 35)))
        self.assertEqual(len(self.cached_pool(pipe)), 500)

    def test_stage_cache_hit_reuses_full_pool_but_fetches_new_page_only(self):
        pipe, _, _ = self.fixture()
        pipe.build_feed(7, "", 1, 20, record_exposure=False)
        pipe.state._feed_cache.clear()
        pipe.state._fetch_notes.reset_mock()
        result = pipe.build_feed(7, "", 2, 20, record_exposure=False)
        self.assertTrue(result["cache_hit"])
        self.assert_cards(result, range(20, 40))
        pipe.state._fetch_notes.assert_called_once_with(list(range(20, 40)))
        pipe.state.fetch_live_recall_candidates.assert_called_once()

    def test_existing_provided_fields_are_preserved_while_missing_fields_filled(self):
        pipe, _, _ = self.fixture(provided=True)
        result = pipe.build_feed(7, "", 1, 20, record_exposure=False)
        pipe.state._fetch_notes.assert_called_once_with(list(range(20)))
        for card in result["items"]:
            self.assertEqual(card["title"], "provided title")
            self.assertEqual(card["cover_image"], "provided.jpg")
            self.assertEqual(card["accum_like_num"], 991)
            self.assertEqual(card["accum_collect_num"], card["note_idx"]+20)
        self.assertEqual(self.cached_pool(pipe).note_title.unique().tolist(), ["provided title"])

    def test_complete_display_fields_require_no_additional_lookup(self):
        pipe, _ = serving_fixture("search")
        pipe.state._fetch_notes = Mock(side_effect=AssertionError("All card metadata already provided"))
        frame = metadata([0]).assign(final_score=.7, rank=1)
        before = frame.copy(deep=True)
        cards = pipe._build_cards(31, 7, frame)
        self.assertEqual(cards[0]["title"], "title-0")
        self.assertEqual(cards[0]["accum_like_num"], 10)
        pipe.state._fetch_notes.assert_not_called()
        pd.testing.assert_frame_equal(frame, before)

    def test_empty_or_exhausted_page_never_fetches_metadata(self):
        pipe, _, _ = self.fixture()
        result = pipe.build_feed(7, "", 26, 20, record_exposure=False)
        self.assertEqual(result["items"], [])
        self.assertEqual(result["total"], 500)
        pipe.state._fetch_notes.assert_not_called()

    def test_ranked_search_keeps_existing_ranking_and_card_metadata_behavior(self):
        pipe, _ = serving_fixture("search")
        pipe.state._fetch_notes = Mock(side_effect=metadata)
        fine = Mock(side_effect=pipe.state.predict_dien)
        pipe.state.predict_dien = fine
        result = pipe.build_feed(7, "q", 1, 2, record_exposure=False)
        self.assertEqual([r["note_idx"] for r in result["items"]], [1, 2])
        self.assertTrue(result["stages"]["ranking"]["enabled"])
        self.assertTrue(result["stages"]["ranking"]["schema_enabled"])
        fine.assert_called_once()
        # Ranked Search retains its existing candidate enrichment and page-count lookup.
        self.assertEqual(pipe.state._fetch_notes.call_args_list, [call([1, 2, 3]), call([1, 2])])
        for card in result["items"]:
            self.assertEqual(card["title"], f"title-{card['note_idx']}")
            self.assertEqual(card["accum_like_num"], card["note_idx"]+10)
            self.assertEqual(card["cover_image"], f"cover-{card['note_idx']}.jpg")


if __name__ == "__main__":
    unittest.main()
