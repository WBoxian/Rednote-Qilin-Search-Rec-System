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


# Portable learned-fusion fixtures: no project data or trained model files.
from recall.learned_context_fusion import (
    CandidateFeatures, FEATURE_NAMES, LearnedContextFusion, PastSupport,
    RouteCandidates, build_candidate_features, rank_candidates,
)


class _FusionPredictor:
    def __init__(self, output=None, names=FEATURE_NAMES, count=23):
        self.names = names
        self.count = count
        self.output = output
        self.calls = []

    def feature_name(self):
        return self.names

    def num_feature(self):
        return self.count

    def predict(self, values, *, num_threads):
        self.calls.append((values.copy(), num_threads))
        return np.zeros(len(values)) if self.output is None else self.output


class LearnedContextFusionTests(unittest.TestCase):
    def setUp(self):
        self.vectors = np.zeros((16, 128), np.float32)
        self.vectors[0, 0] = self.vectors[5, 0] = 1.
        self.vectors[1, 1] = 1.
        self.vectors[2, :2] = [.6, .8]
        self.vectors[3, :2] = [-.8, .6]
        self.vectors[9, 9] = 1.
        self.support = PastSupport({2: 7}, {0: 3}, {0})
        self.args = dict(history=[], user_idx=0, q0_pca=self.vectors[1],
                         item_pca_lookup=lambda ids: self.vectors[ids], support=self.support)

    def build(self, routes, **changes):
        return build_candidate_features(routes, **{**self.args, **changes})

    def test_all23_features_have_hand_computed_history_and_support_semantics(self):
        routes = {name: RouteCandidates([2], [value])
                  for name, value in [('ann', 99.), ('swing', 4.), ('usercf', 8.)]}
        batch = self.build(routes, history=[0, 0, 1])
        expected = np.asarray([
            1, 1, 1/61, 1, 1, 1/61, 1, 1, 1/61, 1, 1, 3, 4/61,
            .8, .6, .8, (2*.6+.8)/3, 3, 2/3, np.sqrt(5)/3,
            2/3, np.log(8), np.log(4),
        ], np.float32)
        self.assertEqual(len(FEATURE_NAMES), 23)
        self.assertEqual(batch.values.dtype, np.dtype('float32'))
        np.testing.assert_array_equal(batch.note_ids, [2])
        np.testing.assert_allclose(batch.values[0], expected, rtol=1e-6, atol=1e-7)
        # Native ANN score is excluded, while CF scores retain their own normalization.
        changed = dict(routes, ann=RouteCandidates([2], [-99.]))
        np.testing.assert_array_equal(self.build(changed, history=[0, 0, 1]).values, batch.values)
        # Missing-text zero vectors stay in history means/coherence denominators.
        zero_history = self.build(routes, history=[4, 0]).values[0]
        np.testing.assert_allclose(zero_history[[14, 15, 16, 19, 20]],
                                   [0., .6, .3, .5, .5], rtol=1e-6, atol=1e-7)

    def test_prefix_consumption_compacts_ranks_and_cf_normalizes_after_filter(self):
        routes = {
            'ann': RouteCandidates([9, 0, 2], [100., .9, .8]),
            'swing': RouteCandidates([9, 2, 3], [10., 4., 2.]),
            'usercf': RouteCandidates([3, 4], [9., 3.]),
        }
        batch = self.build(routes, history=[9]*20+[0])
        np.testing.assert_array_equal(batch.note_ids, [0, 2, 3, 4])
        np.testing.assert_array_equal(batch.route_ranks,
                                     [[1, 1001, 1001], [2, 1, 1001],
                                      [1001, 2, 1], [1001, 1001, 2]])
        np.testing.assert_array_equal(batch.values[:, 9], [0., 1., .5, 0.])
        np.testing.assert_allclose(batch.values[:, 10], [0., 0., 1., 1/3], rtol=1e-7)
        self.assertTrue(np.all(batch.values[:, 17] == 20))

    def test_history_uses_shared_canonical_parser_and_keeps_real_zero(self):
        route = {'ann': RouteCandidates([2, 5], [1., .5])}
        normal = self.build(route, history=[0, 0, 1])
        canonical = self.build(route, history=[None, np.nan, -1, 0, 0, 1])
        np.testing.assert_array_equal(normal.values, canonical.values)
        empty = self.build(route, history=None)
        np.testing.assert_array_equal(empty.values[:, 14:21], 0.)
        for malformed in [[.5], ['bad'], '012', 3, np.array([[1, 2]])]:
            with self.subTest(history=malformed), self.assertRaises(ValueError):
                self.build(route, history=malformed)

    def test_missing_routes_and_empty_union_are_explicit_without_prediction_fallback(self):
        batch = self.build({'swing': RouteCandidates([2], [.25])})
        np.testing.assert_array_equal(batch.route_ranks, [[1001, 1, 1001]])
        self.assertEqual(float(batch.values[0, 11]), 1.)
        self.assertEqual(float(batch.values[0, 12]), float(np.float32(1/61)))
        predictor = _FusionPredictor(output=np.array([np.nan]))
        fusion = LearnedContextFusion(predictor)
        empty = fusion.fuse({}, **self.args)
        self.assertEqual(len(empty.note_idx), 0)
        self.assertEqual(predictor.calls, [])
        self.assertEqual(empty.to_frame().score.dtype, np.dtype('float64'))
        consumed = fusion.fuse({'ann': RouteCandidates([0], [1.])},
                               **{**self.args, 'history': [0]})
        self.assertEqual(len(consumed.rank), 0)

    def test_past_support_and_arrays_are_copied_and_frozen(self):
        items, users, graph = {2: 7}, {0: 3}, {0}
        support = PastSupport(items, users, graph)
        items[2] = users[0] = 99
        graph.add(1)
        self.assertEqual(support.item_counts[2], 7)
        self.assertEqual(support.user_counts[0], 3)
        self.assertEqual(support.graph_items, frozenset({0}))
        with self.assertRaises(TypeError):
            support.item_counts[2] = 99
        ids = np.array([2], np.int64)
        scores = np.array([1.], np.float32)
        route = RouteCandidates(ids, scores)
        ids[0], scores[0] = 3, 9
        self.assertEqual(route.ids.tolist(), [2])
        self.assertEqual(route.scores.tolist(), [1.])
        batch = self.build({'ann': route}, support=support)
        with self.assertRaises(ValueError):
            batch.values[0, 0] = 7

    def test_malformed_routes_scores_and_support_are_rejected(self):
        cases = [
            ([1, 1], [1., 2.]), ([1.], [1.]), ([True, 2], [1., 2.]),
            ([-1], [1.]), ([np.uint64(2**63)], [1.]),
            ([[1]], [1.]), (list(range(1001)), [1.]*1001),
            ([1], []), ([1], [[1.]]), ([1], [np.nan]),
            ([1], [np.inf]), ([1], [1e300]), ([1], [1+2j]),
        ]
        for ids, scores in cases:
            with self.subTest(ids=str(ids)[:30]), self.assertRaises(ValueError):
                RouteCandidates(ids, scores)
        for bad in [{'dpr': RouteCandidates([], [])}, {'ann': ([1], [1.])}]:
            with self.assertRaises(ValueError):
                self.build(bad)
        for item, user, graph in [({-1: 1}, {}, set()), ({1: -1}, {}, set()),
                                  ({1: np.nan}, {}, set()), ({}, {False: 1}, set()),
                                  ({}, {}, {-1}), ({}, {}, {True})]:
            with self.subTest(support=(item, user, graph)), self.assertRaises(ValueError):
                PastSupport(item, user, graph)

    def test_query_lookup_features_and_schema_mismatches_are_rejected(self):
        routes = {'ann': RouteCandidates([2], [1.])}
        for q in [np.zeros(127), np.zeros((1, 128)), np.full(128, np.nan)]:
            with self.assertRaises(ValueError):
                self.build(routes, q0_pca=q)
        for lookup in [lambda ids: np.zeros((len(ids), 256)),
                       lambda ids: np.full((len(ids), 128), np.inf)]:
            with self.assertRaises(ValueError):
                self.build(routes, item_pca_lookup=lookup)
        with self.assertRaises(ValueError):
            self.build(routes, user_idx=0.)
        batch = self.build(routes)
        with self.assertRaises(ValueError):
            CandidateFeatures(batch.note_ids, batch.values.astype(np.float64), batch.route_ranks)
        with self.assertRaises(ValueError):
            CandidateFeatures(batch.note_ids, batch.values, batch.route_ranks,
                              feature_names=FEATURE_NAMES[::-1])
        with self.assertRaises(ValueError):
            CandidateFeatures(batch.note_ids, batch.values[:, :-1], batch.route_ranks)
        with self.assertRaises(ValueError):
            CandidateFeatures(np.arange(3001), np.zeros((3001, 23), np.float32),
                              np.full((3001, 3), 1001, np.int32))
        with self.assertRaises(TypeError):
            build_candidate_features(routes, **self.args, labels=[1])

    def test_model_schema_prediction_errors_and_explicit_cpu_threads(self):
        for predictor in [_FusionPredictor(names=FEATURE_NAMES[::-1]),
                          _FusionPredictor(names=None), _FusionPredictor(count=22), object()]:
            with self.assertRaises(ValueError):
                LearnedContextFusion(predictor)
        for threads in [1, 4, True, 2.]:
            with self.assertRaises(ValueError):
                LearnedContextFusion(_FusionPredictor(), num_threads=threads)
        batch = self.build({'ann': RouteCandidates([2], [1.])})
        for values in [np.array([np.nan]), np.array([[1.]]), np.array([1., 2.])]:
            predictor = _FusionPredictor(output=values)
            with self.assertRaises(ValueError):
                LearnedContextFusion(predictor).predict(batch)
            self.assertEqual(len(predictor.calls), 1)
        predictor = _FusionPredictor()
        prediction = LearnedContextFusion(predictor, num_threads=3).predict(batch)
        np.testing.assert_array_equal(prediction, [0.])
        self.assertEqual(predictor.calls[0][1], 3)
        with patch.object(predictor, 'predict', side_effect=RuntimeError('model unavailable')):
            with self.assertRaisesRegex(RuntimeError, 'model unavailable'):
                LearnedContextFusion(predictor).predict(batch)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                LearnedContextFusion.from_model_path(Path(directory)/'missing-model.txt')

    def test_float32_prior_collision_and_ties_keep_prototype_id_order(self):
        ranks = np.asarray([[928, 695, 767], [998, 568, 870]], np.int32)
        rr64 = (1/(60+ranks)) @ np.array([2, 1, 1])
        self.assertGreater(rr64[1], rr64[0])
        self.assertEqual(np.float32(rr64[0]), np.float32(rr64[1]))
        values = np.zeros((2, 23), np.float32)
        values[:, FEATURE_NAMES.index('rrf211')] = rr64.astype(np.float32)
        features = CandidateFeatures([1, 2], values, ranks)
        ranked = rank_candidates(features, [0., 1.])
        # Learned order2,1 and float32-prior order1,2 cancel; final tie is ID ascending.
        np.testing.assert_array_equal(ranked.note_idx, [1, 2])
        np.testing.assert_array_equal(ranked.rank, [1, 2])
        np.testing.assert_array_equal(ranked.score, [.5/61+.5/62]*2)
        tied = rank_candidates(features, [1., 1.])
        np.testing.assert_array_equal(tied.note_idx, [1, 2])
        with self.assertRaises(ValueError):
            rank_candidates(features, [np.nan, 1.])

    def test_fixed500_budget_and_deterministic_large_union(self):
        lookup = lambda ids: np.zeros((len(ids), 128), np.float32)
        features = self.build({'ann': RouteCandidates(np.arange(501), np.ones(501))},
                              item_pca_lookup=lookup)
        first = rank_candidates(features, np.arange(501))
        second = rank_candidates(features, np.arange(501))
        self.assertEqual(len(first.note_idx), 500)
        self.assertEqual(len(np.unique(first.note_idx)), 500)
        np.testing.assert_array_equal(first.rank, np.arange(1, 501))
        np.testing.assert_array_equal(first.note_idx, second.note_idx)
        np.testing.assert_array_equal(first.score, second.score)


if __name__ == '__main__':
    unittest.main()
