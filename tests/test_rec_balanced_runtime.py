import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd
import torch

from backend.online.bundle import ServingBundle, bundle_identity, file_sha256, required_asset_names, validate_manifest
from backend.online.recall.dssm import DSSMRecaller
from recall.learned_context_fusion import CandidateFeatures, rank_candidates
import test_online_residual_recall as residual_fixtures
from test_serving_bundle import learned_fixture


class BalancedRuntimeTests(unittest.TestCase):
    def test_guard_changes_head_without_changing_coarse_pool_or_tail(self):
        ids = np.arange(600, dtype=np.int64)
        features = np.zeros((600, 23), np.float32)
        # Prior deliberately disagrees with the learned score, including items outside the coarse pool.
        features[:, 12] = np.linspace(0, 1, 600, dtype=np.float32)
        ranks = np.full((600, 3), 1001, np.int32)
        ranks[:, 0] = np.arange(600, 0, -1)
        bundle = CandidateFeatures(ids, features, ranks)
        predictions = np.random.default_rng(42).standard_normal(600)
        original = rank_candidates(bundle, predictions)
        actual = rank_candidates(bundle, predictions, head_guard=True)
        self.assertEqual(set(actual.note_idx[:100]), set(original.note_idx[:100]))
        np.testing.assert_array_equal(actual.note_idx[100:], original.note_idx[100:])
        prior = list(reversed(ids.tolist()))
        expected_head = [item for item in prior if item in set(original.note_idx[:100])][:10]
        self.assertEqual(actual.note_idx[:10].tolist(), expected_head)
        self.assertFalse(np.array_equal(actual.note_idx, original.note_idx))
        np.testing.assert_array_equal(actual.score, 1 / (60 + np.arange(1, 501, dtype=np.float64)))
        for bad in (1, None, 'true'):
            with self.assertRaises(ValueError):
                rank_candidates(bundle, predictions, head_guard=bad)

    def test_loaded_multi_interest_keeps_base_block_and_original_per_head_budget(self):
        fixture = residual_fixtures.OnlineAdaptedResidualRecallTests()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture.fixture(root)
            tower = fixture.load(root)
            tower.query_strategy = 'adapted_k3_v1'
            frame = fixture.request([0, 1, 0])
            global_query = tower._build_user_vec(frame, 123, 1)
            heads = tower._build_interest_queries([0, 1, 0], global_query)
            self.assertGreaterEqual(len(heads), 1)
            self.assertLessEqual(len(heads), 3)
            np.testing.assert_array_equal(heads[:, :128], np.repeat(global_query[:, :128], len(heads), axis=0))
            for history in ([], [99999]):
                np.testing.assert_array_equal(tower._build_interest_queries(history, global_query), global_query)
            # Synthetic per-head scores exercise dedup/max, global tie ordering, and original K cap.
            tower._build_interest_queries = Mock(return_value=np.zeros((3, 256), np.float32))
            tower._search_ann = Mock(side_effect=[([7, 1, 2], [1., .8, .5]),
                                                ([2, 3, 7], [1., .8, .7]),
                                                ([4, 1, 2], [1., .9, .2])])
            result = tower.fetch_candidates(123, 3, frame, lambda _: frame, 'request_idx', pure_request_ann=True)
            self.assertEqual(result.note_idx.tolist(), [2, 4, 7])
            self.assertEqual(tower._search_ann.call_count, 3)
            self.assertTrue(all(call.args[1] == 3 for call in tower._search_ann.call_args_list))
            self.assertEqual(result['rank'].tolist(), [1, 2, 3])

    def test_optional_policy_is_strict_rec_only_and_binds_both_new_runtime_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = learned_fixture(root)
            for key, strategy, filename in [('ann_query', 'adapted_k3_v1', 'multi_interest.py'),
                                            ('head_guard', 'coarse100_prior10_v1', 'head_guard.py'),
                                            ('ann_engine', 'cuda_cpu_refine_v1', 'exact_ip_filter.py')]:
                manifest['recall_policy'][key] = {'version': 1, 'strategy': strategy}
                path = root / 'src/recall' / filename
                path.write_bytes((Path(__file__).resolve().parents[1] / 'src/recall' / filename).read_bytes())
                manifest['assets'][path.relative_to(root).as_posix()] = file_sha256(path)
            manifest['bundle_id'] = bundle_identity(manifest)
            validate_manifest(manifest, 'rec', root)
            required = required_asset_names(manifest, root)
            self.assertIn('src/recall/multi_interest.py', required)
            self.assertIn('src/recall/head_guard.py', required)
            for key in ('ann_query', 'head_guard', 'ann_engine'):
                for bad in (None, 'wrong', {}, {'version': True, 'strategy': manifest['recall_policy'][key]['strategy']},
                            {**manifest['recall_policy'][key], 'extra': True}):
                    changed = copy.deepcopy(manifest)
                    changed['recall_policy'][key] = bad
                    changed['bundle_id'] = bundle_identity(changed)
                    with self.assertRaises(ValueError):
                        validate_manifest(changed, 'rec', root)
            # Runtime hashes are rechecked even after a successful model cache warm.
            serving = ServingBundle(manifest, None, root)
            serving._get_learned_fusion()
            manifest['assets']['src/recall/head_guard.py'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'asset mismatch'):
                serving._get_learned_fusion()
