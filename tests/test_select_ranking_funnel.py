import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

SPEC = importlib.util.spec_from_file_location("ranking_funnel_gate", Path(__file__).resolve().parents[1]/"scripts/select_ranking_funnel.py")
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def metrics(recall=.5, hit=.5, ndcg=.2, mrr=.1):
    return {"Recall@100": recall, "HitRate@10": hit, "HitRate@50": hit, "known_positive_NDCG@10": ndcg, "MRR@100": mrr}


class RankingFunnelGateTests(unittest.TestCase):
    def test_only_logged_ndcg_auc_gauc_nonregressing_weights_are_eligible(self):
        manifest = {"dien": {"blend_candidates": [
            {"weight": 0., "validation": {"NDCG@10": .5, "AUC": .7, "GAUC": .7}},
            {"weight": .1, "validation": {"NDCG@10": .6, "AUC": .69, "GAUC": .8}},
            {"weight": .25, "validation": {"NDCG@10": .6, "AUC": .71, "GAUC": .69}},
            {"weight": .5, "validation": {"NDCG@10": .6, "AUC": .71, "GAUC": .71}}]}}
        self.assertEqual(gate.eligible_weights(manifest), [0., .5])

    def test_no_strict_primary_gain_cannot_force_acceptance(self):
        baseline = metrics()
        grid = [{"coarse_cap": 160, "weight": 0., "metrics": metrics(recall=.5, hit=.8, ndcg=.6, mrr=.4)}]
        self.assertIsNone(gate.choose_recipe(grid, baseline, "search"))

    def test_release_gate_cannot_promote_a_rejected_validation_recipe_or_trade_off_metrics(self):
        baseline = metrics()
        self.assertFalse(gate.release_passes(metrics(recall=.8, hit=.8, ndcg=.4), baseline, "rec", False))
        self.assertFalse(gate.release_passes(metrics(recall=.8, hit=.4), baseline, "rec", True))
        self.assertFalse(gate.release_passes(metrics(recall=.8, mrr=.09), baseline, "search", True))
        self.assertTrue(gate.release_passes(baseline, baseline, "search", True))

    def test_search_guard_protects_mrr_and_recommendation_guard_does_not_invent_it(self):
        baseline = metrics()
        candidate = metrics(recall=.6, hit=.6, ndcg=.3, mrr=.09)
        self.assertFalse(gate.guard_passes(candidate, baseline, "search"))
        self.assertTrue(gate.guard_passes(candidate, baseline, "rec"))

    def test_primary_recall_cannot_trade_off_hit_or_known_ndcg(self):
        baseline = metrics()
        self.assertFalse(gate.guard_passes(metrics(recall=.8, hit=.4), baseline, "rec"))
        self.assertFalse(gate.guard_passes(metrics(recall=.8, ndcg=.1), baseline, "search"))

    def test_only_accepted_candidates_compete_by_recall_then_guards(self):
        baseline = metrics()
        grid = [{"coarse_cap": 160, "weight": .5, "metrics": metrics(recall=.8, hit=.3)},
                {"coarse_cap": 80, "weight": .1, "metrics": metrics(recall=.6, hit=.6)},
                {"coarse_cap": 120, "weight": .25, "metrics": metrics(recall=.7, hit=.6)}]
        self.assertIs(gate.choose_recipe(grid, baseline, "search"), grid[2])

    def test_unrecalled_requests_remain_zero_in_macro_denominator(self):
        requests = pd.DataFrame({"search_idx": [1, 2], "user_idx": [7, 8], "positive_note_ids": [[11], [22]]})
        candidates = pd.DataFrame({"search_idx": [1], "note_idx": [11], "rank": [1]})
        rows = gate.evaluate_recipe(candidates, requests, "search", raw=True)
        summary = gate.summarize(rows)
        self.assertEqual(summary["requests"], 2)
        self.assertEqual(summary["empty_candidate_requests"], 1)
        self.assertEqual(summary["Recall@100"], .5)
        self.assertEqual(summary["HitRate@10"], .5)

    def test_all_original_clicked_positives_define_ndcg_denominator(self):
        result = gate.metric_row([1], [1, 2])
        self.assertAlmostEqual(result["Recall@100"], .5)
        self.assertAlmostEqual(result["known_positive_NDCG@10"], 1/(1+1/np.log2(3)))
        self.assertAlmostEqual(gate.metric_row([1, 1], [1])["known_positive_NDCG@10"], 1.)
        self.assertTrue(np.isnan(gate.metric_row([], [])["Recall@100"]))

    def test_candidate_recipe_scores_all_input_but_retains_coarse_budget(self):
        requests = pd.DataFrame({"request_idx": [1], "user_idx": [7], "positive_note_ids": [[3]]})
        frame = pd.DataFrame({"request_idx": [1, 1, 1], "note_idx": [1, 2, 3], "rank": [1, 2, 3],
            "coarse_score": [.9, .8, .7], "dien_raw_score": [-2., -1., 20.]})
        small = gate.evaluate_recipe(frame, requests, "rec", coarse_cap=2, weight=1., score_contract="probability_v2")
        large = gate.evaluate_recipe(frame, requests, "rec", coarse_cap=3, weight=1., score_contract="probability_v2")
        self.assertEqual(small.candidates.iloc[0], 2)
        self.assertEqual(small["Recall@100"].iloc[0], 0.)
        self.assertEqual(large["HitRate@10"].iloc[0], 1.)
        self.assertEqual(large["MRR@100"].iloc[0], 1.)

    def test_foreign_or_duplicate_candidate_requests_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"candidates.parquet"
            pd.DataFrame({"gid": [1, 99], "note_idx": [11, 22], "rank": [1, 1]}).to_parquet(path, index=False)
            with self.assertRaisesRegex(ValueError, "foreign"):
                gate.read_candidates(path, "request_idx", {1}, 320)
            pd.DataFrame({"gid": [1, 1], "note_idx": [11, 11], "rank": [1, 2]}).to_parquet(path, index=False)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                gate.read_candidates(path, "request_idx", {1}, 320)

    def test_fit_ids_cannot_be_relabelled_as_common_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/"datasets/search_train"
            source.mkdir(parents=True)
            pd.DataFrame({"search_idx": [1], "user_idx": [7], "query": ["q"], "recent_clicked_note_idxs": [[3]],
                "search_result_details_with_idx": [[{"note_idx": 1, "click": 1}]]}).to_parquet(source/"train.parquet", index=False)
            ids = root/"validation_ids.parquet"
            pd.DataFrame({"search_idx": [1]}).to_parquet(ids, index=False)
            run = root/"model"
            run.mkdir()
            pd.DataFrame({"search_idx": [1]}).to_parquet(run/"validation.parquet", index=False)
            pd.DataFrame({"search_idx": [1], "partition": ["fit"]}).to_parquet(run/"train.parquet", index=False)
            with patch.object(gate, "ROOT", root), self.assertRaisesRegex(ValueError, "overlap"):
                gate.validation_cohort("search", ids, [run])

    def test_official_test_is_not_opened_without_a_frozen_validation_recipe(self):
        args = SimpleNamespace(scene="search", selection=Path("absent"), output=Path("unused"))
        with patch.object(gate, "verify_frozen_selection", side_effect=ValueError("not frozen")), patch.object(gate, "original_requests") as source:
            with self.assertRaises(ValueError):
                gate.test(args)
            source.assert_not_called()

    def test_a_later_test_report_does_not_change_the_frozen_model_contract(self):
        manifest = {"scene": "search", "key": "search_idx", "features": ["image_num"], "inputs": {"train": "sha"},
            "dien": {"numeric_features": ["image_num", "coarse_score"], "architecture": {"inference_precision": "fp32"}, "blend_candidates": [], "score_contract": "probability_v2"}}
        before = gate.contract_hash(gate.model_contract(manifest))
        manifest["final_test"] = {"metrics": "not selection"}
        self.assertEqual(before, gate.contract_hash(gate.model_contract(manifest)))
        manifest["dien"]["score_contract"] = "request_minmax_v1"
        self.assertNotEqual(before, gate.contract_hash(gate.model_contract(manifest)))


if __name__ == "__main__":
    unittest.main()
