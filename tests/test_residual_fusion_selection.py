import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from recall.context_fusion import fuse_request_routes
from recall.weighted_context_fusion import fuse_weighted_request_routes
from scripts import select_residual_fusion as experiment


class ResidualFusionSelectionTests(unittest.TestCase):
    def test_control_matches_actual_legacy_f1_over_random_routes(self):
        random = np.random.default_rng(42)
        for _ in range(64):
            routes = {name: random.choice(150, 80, replace=False) for name in ('ann', 'swing', 'usercf')}
            history = random.integers(0, 150, 30).tolist()
            expected = fuse_request_routes(routes, history, (), 'filter_history')
            actual = fuse_weighted_request_routes(routes, history, experiment.CONTROL)
            pd.testing.assert_frame_equal(actual, expected)
            experiment.independent_hand_proof(routes, history, experiment.CONTROL, actual)

    def test_filter_compacts_original_rank_keeps_zero_and_does_not_refill(self):
        routes = {'ann': [9, 0, 3], 'swing': [3, 0], 'usercf': [4]}
        actual = fuse_weighted_request_routes(routes, [9]*20+[0], {'ann': 4, 'swing': 1, 'usercf': 1})
        self.assertEqual(actual.note_idx.tolist(), [0, 3, 4])
        scores = actual.set_index('note_idx').score
        self.assertEqual(scores.loc[0], 4/61+1/62)
        self.assertEqual(scores.loc[3], 4/62+1/61)
        self.assertEqual(scores.loc[4], 1/61)
        self.assertNotIn(9, actual.note_idx.tolist())
        self.assertEqual(actual.score.dtype, np.dtype('float64'))

    def test_ties_are_item_order_and_zero_weight_routes_add_no_tail(self):
        routes = {'ann': [2], 'swing': [1], 'usercf': [99]}
        self.assertEqual(fuse_weighted_request_routes(routes, [], {'ann': 1, 'swing': 1, 'usercf': 0}).note_idx.tolist(), [1, 2])
        self.assertEqual(fuse_weighted_request_routes(routes, [], {'ann': 1, 'swing': 0, 'usercf': 0}).note_idx.tolist(), [2])

    def test_malformed_budgets_identities_and_weights_reject(self):
        for routes in [{'ann': [1, 1]}, {'ann': [1.5]}, {'ann': [-1]}, {'ann': list(range(1001))}, {'dpr': [1]}]:
            with self.assertRaises(ValueError):
                fuse_weighted_request_routes(routes, [], experiment.CONTROL)
        for weights in [{'ann': 1}, {'ann': np.nan, 'swing': 1, 'usercf': 1}, {'ann': -1, 'swing': 1, 'usercf': 1},
                        {'ann': 0, 'swing': 0, 'usercf': 0}]:
            with self.assertRaises(ValueError):
                fuse_weighted_request_routes({}, [], weights)

    def test_grid_is_exactly_eight_and_selection_never_forces_failed_guards(self):
        self.assertEqual(len(experiment.GRID), 8)
        self.assertIn(experiment.CONTROL, experiment.GRID)
        baseline = {'Recall@100': .1, 'HitRate@50': .2, 'known_positive_NDCG@10': .03}
        trials = [{'weights': weights, 'metrics': copy.deepcopy(baseline)} for weights in experiment.GRID]
        self.assertIsNone(experiment.choose(trials, baseline))
        trials[4]['metrics']['Recall@100'] = .2
        trials[4]['metrics']['HitRate@50'] = .19
        self.assertIsNone(experiment.choose(trials, baseline))
        trials[4]['metrics']['HitRate@50'] = .2
        trials[6]['metrics']['Recall@100'] = .2
        self.assertEqual(experiment.choose(trials, baseline), trials[4])

    def test_known_positive_metric_retains_misses_in_denominator(self):
        metrics = experiment.metric_values([5, 7, 9], [5, 99])
        self.assertEqual(metrics['Recall@100'], .5)
        self.assertEqual(metrics['MRR@100'], 1.)
        self.assertAlmostEqual(metrics['known_positive_NDCG@10'], 1/(1+1/np.log2(3)))
        self.assertEqual(experiment.metric_values([], [5])['Recall@100'], 0.)

    def test_unapproved_test_never_reads_official_test_truth(self):
        selection = {'accepted': True}
        with patch.object(experiment, 'verify_plan'), patch.object(experiment, 'load_json', side_effect=[selection, {'approved': False}]), patch.object(experiment, 'original_truth') as truth:
            with self.assertRaisesRegex(ValueError, 'explicit parent approval'):
                experiment.test(Path('/tmp/not-a-real-experiment'), Path('/tmp/approval.json'))
            truth.assert_not_called()

    def test_original_train_truth_is_derived_from_nested_clicks_and_fixed_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root/'datasets/recommendation_train'
            folder.mkdir(parents=True)
            pd.DataFrame({'request_idx': [2, 1], 'user_idx': [8, 7], 'recent_clicked_note_idxs': [[0, 4], []],
                'rec_result_details_with_idx': [[{'note_idx': 10, 'click': 0}, {'note_idx': 11, 'click': 1}],
                                               [{'note_idx': 20, 'click': 1}, {'note_idx': 20, 'click': 1}]]}).to_parquet(folder/'train.parquet', index=False)
            with patch.object(experiment, 'ROOT', root):
                raw = experiment.original_truth('train', {1, 2})
                self.assertEqual(raw.gid.tolist(), [1, 2])
                self.assertEqual(raw.positives.tolist(), [[20], [11]])
                with self.assertRaisesRegex(ValueError, 'entirely original-train'):
                    experiment.original_truth('train', {99})


if __name__ == '__main__':
    unittest.main()
