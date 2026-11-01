import unittest
from pathlib import Path
from copy import deepcopy
from scripts.evaluation_comparisons import metric_delta, comparison, exposure_comparison, adapted_release_comparisons, multi_interest_refit_comparisons, attach_comparisons


class EvaluationComparisonTests(unittest.TestCase):
    def test_negative_lift_and_zero_baseline_are_not_hidden(self):
        result = metric_delta(.2, .1, [-.15, -.05])
        self.assertAlmostEqual(result['absolute_delta_pp'], -10)
        self.assertAlmostEqual(result['relative_delta_pct'], -50)
        self.assertEqual(result['ci95_delta'], [-.15, -.05])
        self.assertIsNone(metric_delta(0, .1)['relative_delta_pct'])

    def test_invalid_values_and_interval_are_rejected(self):
        for value in [float('nan'), float('inf'), -1, 2]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                metric_delta(value, .1)
        with self.assertRaises(ValueError):
            metric_delta(.1, .2, [.3, -.2])

    def test_catalog_comparison_does_not_substitute_conditional_ndcg(self):
        metrics = {'HitRate@50': .02, 'Recall@100': .01, 'known_positive_NDCG@10': .001, 'MRR@100': .003}
        result = comparison(metrics, metrics, baseline_name='frozen', current_name='released', scene='rec')
        self.assertEqual(result['scope'], 'full_catalog_known_click')
        self.assertIn('known_positive_NDCG@10', [r['name'] for r in result['metrics']])
        del metrics['known_positive_NDCG@10']
        metrics['NDCG@10'] = .6
        with self.assertRaisesRegex(ValueError, 'known_positive'):
            comparison(metrics, metrics, baseline_name='frozen', current_name='released', scene='rec')

    def test_exposure_reference_discloses_stronger_position_diagnostic(self):
        audit = {'random_expected_NDCG@10': .53, 'candidate_count_quantiles': {'0.5': 10},
                 'requests_with_at_most_10_candidates': 6, 'requests': 10, 'original_position_NDCG@10': .65}
        result = exposure_comparison(audit, {'NDCG@10': .62, 'AUC': .61, 'GAUC': .62})
        self.assertEqual(result['diagnostics']['original_position_NDCG_at10'], .65)
        self.assertEqual(result['diagnostics']['requests_at_most_10_fraction'], .6)
        self.assertAlmostEqual(result['metrics'][0]['absolute_delta'], .09)
        self.assertEqual(result['metrics'][1]['baseline'], .5)

    def test_adapted_comparisons_separate_index_model_and_merge_gains(self):
        def metrics(value):
            return {key: value for key in ("HitRate@50", "Recall@100", "known_positive_NDCG@10", "MRR@100")}
        evaluated = {"ablations": {key: {"metrics": metrics(value)} for key, value in (
            ("original_baseline", .1), ("original_flat211", .2), ("dual128_rrf", .3), ("dual128", .25))}}
        current, rows = {"comparisons": {}}, []
        adapted_release_comparisons(current, evaluated, rows, metrics(.01))
        self.assertEqual(len(rows), 4)
        self.assertEqual([row["status"] for row in rows], ["diagnostic", "diagnostic", "diagnostic", "accepted"])
        result = current["comparisons"]
        self.assertAlmostEqual(result["index_precision"]["metrics"][0]["absolute_delta"], .1)
        self.assertAlmostEqual(result["content_adapter"]["metrics"][0]["absolute_delta"], .1)
        self.assertAlmostEqual(result["learned_fusion"]["metrics"][0]["absolute_delta"], -.05)
        self.assertAlmostEqual(result["end_to_end"]["metrics"][0]["baseline"], .1)
        self.assertIn("Previous residual", result["end_to_end"]["baseline_name"])


class MultiInterestRefitComparisonTests(unittest.TestCase):
    def prepare(self):
        short = ("HR50", "R100", "KP_NDCG10", "MRR100", "R500", "HR500")
        values = {"current_global_LGB": (.1, .05, .01, .02, .15, .3),
            "oldfit_k3_lgb": (.11, .06, .011, .021, .16, .31),
            "oldfit_k3_safehead": (.111, .06, .014, .03, .16, .31),
            "finalrefit_k3_safehead": (.110, .061, .015, .031, .161, .312),
            "old_PQ211": (.08, .04, .009, .018, .12, .25)}
        proof = {"status": "PASS", "all_requests": 11115, "known_positive_pairs": 44683,
            "parameter_choice_after_test": False, "outputs": {key: {"all_requests": 11115,
                "known_positive_pairs": 44683, "metrics": dict(zip(short, value))} for key, value in values.items()},
            "paired_usercluster95_exploratory_CI": {}}
        for new, old in (("finalrefit_k3_safehead", "current_global_LGB"),
            ("oldfit_k3_safehead", "oldfit_k3_lgb"), ("finalrefit_k3_safehead", "oldfit_k3_safehead"),
            ("finalrefit_k3_safehead", "old_PQ211")):
            proof["paired_usercluster95_exploratory_CI"][new + "__minus__" + old] = {
                name: {"delta": a - b, "CI95": [a - b - .01, a - b + .01]}
                for name, a, b in zip(short, values[new], values[old])}
        long = ("HitRate@50", "Recall@100", "known_positive_NDCG@10", "MRR@100", "Recall@500", "HitRate@500")
        scene = {"pipeline": {"test": dict(zip(long, values["finalrefit_k3_safehead"]))}}
        return scene, proof

    def test_immediate_baseline_and_fixed_components_use_actual_audited_pairs(self):
        scene, proof = self.prepare()
        rows = multi_interest_refit_comparisons(scene, proof)
        main = scene["comparisons"]["end_to_end"]
        recall = next(row for row in main["metrics"] if row["name"] == "Recall@100")
        self.assertAlmostEqual(recall["relative_delta_pct"], 22.)
        self.assertIn("global single-interest", main["baseline_name"])
        original = scene["comparisons"]["original_PQ_reference"]
        self.assertAlmostEqual(original["metrics"][1]["relative_delta_pct"], 52.5)
        self.assertEqual([r["status"] for r in rows], ["baseline", "diagnostic", "diagnostic", "accepted"])
        refit = scene["comparisons"]["content_refit"]["metrics"][0]
        self.assertLess(refit["absolute_delta"], 0)
        self.assertLess(refit["ci95_delta"][0], 0)
        self.assertGreater(refit["ci95_delta"][1], 0)
        self.assertIn("327 to 408", rows[-1]["changed"])
        self.assertIn("no DIEN gain", rows[-2]["changed"])

    def test_different_pipeline_values_and_wrong_paired_evidence_are_rejected(self):
        scene, proof = self.prepare();scene["pipeline"]["test"]["Recall@100"] = .9
        with self.assertRaisesRegex(ValueError, "current released pipeline"):
            multi_interest_refit_comparisons(scene, proof)
        scene, proof = self.prepare()
        proof["paired_usercluster95_exploratory_CI"]["finalrefit_k3_safehead__minus__current_global_LGB"]["R100"]["delta"] = .8
        with self.assertRaisesRegex(ValueError, "different contrast"):
            multi_interest_refit_comparisons(scene, proof)

    def test_new_schema_early_path_never_reads_legacy_residual_results(self):
        scene, proof = self.prepare();root = Path("/fixture")
        scene["pipeline"]["release_evidence"] = {"path": "release.json"}
        scene["ranking"] = {"ranking": {"test": {"NDCG@10": .4, "AUC": .6, "GAUC": .65}}}
        exposure = {"random_expected_NDCG@10": .3, "candidate_count_quantiles": {"0.5": 12},
            "requests_with_at_most_10_candidates": 3, "requests": 10, "original_position_NDCG@10": .5}
        data = {str(root / ".qilin/final_metrics_acceptance_20261031.json"): {"scopes": {"rec_test_logged_exposure": exposure}},
            str(root / "release.json"): {"release_schema": "multi_interest_refit_v1", "sources": {"independent_verification": {"path": "proof.json"}}},
            str(root / "proof.json"): proof}
        class FakeSources:
            def read(self, path):return deepcopy(data[str(path)])
        def no_legacy(*args):raise AssertionError("Unexpected legacy catalog loading")
        attach_comparisons(root, root / "base", {"rec": scene}, FakeSources(), no_legacy)
        self.assertEqual(scene["comparisons"]["logged_exposure"]["scope"], "logged_exposure_conditional")
        self.assertEqual(scene["comparisons"]["end_to_end"]["metrics"][1]["current"], scene["pipeline"]["test"]["Recall@100"])
        self.assertEqual(len(scene["ablations"]["rows"]), 4)
