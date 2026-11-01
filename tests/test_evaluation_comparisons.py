import unittest
from scripts.evaluation_comparisons import metric_delta, comparison, exposure_comparison, adapted_release_comparisons


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
