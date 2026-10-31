import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import torch

from backend.online.ranking.inference import (
    REQUEST_FP32, canonical_request_logits, configure_request_inference,
)
from backend.online.ranking.stack import RankStackPredictor
from backend.online.recall.dssm import DSSMRecaller
import backend.online.shared_query_encoder as encoder
from training.dien_ranker import DIENRanker


class Context:
    def batch(self, frame, numeric, device):
        value = torch.as_tensor(frame.note_idx.to_numpy(np.float32)[:, None], device=device)
        return {'quality': value / 100,
                'note_text_emb': value.expand(-1, 4) / 10,
                'note_img_emb': value.expand(-1, 4) / 20,
                'seq_embs': torch.ones((len(frame), 3, 4), device=device)}


class RequestInferenceContractTests(unittest.TestCase):
    def setUp(self):
        self.saved = (torch.get_num_threads(), torch.backends.cuda.matmul.allow_tf32,
                      torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark)
        configure_request_inference(REQUEST_FP32)

    def tearDown(self):
        threads, matmul, cudnn, benchmark = self.saved
        torch.set_num_threads(threads)
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn
        torch.backends.cudnn.benchmark = benchmark

    def model(self):
        return DIENRanker('rec', ['quality'], embed_dim=4, hidden_dim=4,
                          mlp_dims=[4], mask_padded_history=True, inference_precision='fp32').eval()

    def frame(self):
        return pd.DataFrame({'request_idx': [1, 2, 1, 2], 'note_idx': [12, 22, 11, 21],
                             'rank': [2, 2, 1, 1]})

    def test_real_dien_is_request_local_order_independent_and_fixed_shape(self):
        model, context, frame = self.model(), Context(), self.frame()
        batches = []
        hook = model.register_forward_pre_hook(lambda _, args: batches.append(args[0]['note_text_emb'].shape[0]))
        together = canonical_request_logits(model, context, frame, ['quality'], 'cpu', 'request_idx')
        separated = np.empty(len(frame), dtype=np.float32)
        for positions in frame.groupby('request_idx').indices.values():
            separated[positions] = canonical_request_logits(model, context, frame.iloc[positions], ['quality'], 'cpu', 'request_idx')
        np.testing.assert_array_equal(together, separated)
        shuffled = frame.iloc[::-1]
        actual = canonical_request_logits(model, context, shuffled, ['quality'], 'cpu', 'request_idx')
        np.testing.assert_array_equal(actual, together[::-1])
        hook.remove()
        self.assertEqual(batches, [1024] * 6)

    def test_contract_rejects_unknown_ranks_training_and_global_drift(self):
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            configure_request_inference('unknown')
        model, frame = self.model(), self.frame()
        invalid = frame.assign(rank=1)
        with self.assertRaisesRegex(ValueError, 'unique positive'):
            canonical_request_logits(model, Context(), invalid, ['quality'], 'cpu', 'request_idx')
        with self.assertRaisesRegex(ValueError, 'evaluation model'):
            canonical_request_logits(model.train(), Context(), frame, ['quality'], 'cpu', 'request_idx')
        model.eval()
        torch.backends.cudnn.benchmark = True
        with self.assertRaisesRegex(RuntimeError, 'benchmark disabled'):
            canonical_request_logits(model, Context(), frame, ['quality'], 'cpu', 'request_idx')

    def test_actual_predict_and_public_replay_method_share_logits(self):
        ranker = RankStackPredictor.__new__(RankStackPredictor)
        from threading import RLock
        ranker._lock = RLock()
        ranker.inference_contract = REQUEST_FP32
        ranker.model, ranker.device, ranker.group_key = self.model(), 'cpu', 'request_idx'
        ranker.feature_cols, ranker.num_feats = ['quality'], ['quality']
        ranker.weight, ranker.score_contract, ranker.rank_prior = 1., 'probability_v2', None
        frame = self.frame().assign(quality=.1)
        ranker._frame = lambda candidates, *args: (candidates.copy(), Context())
        ranker.coarse = Mock()
        ranker.coarse.predict.side_effect = lambda rows: np.full(len(rows), .2)
        replay = ranker.score_request_frame(frame)
        actual = ranker.predict(frame)
        np.testing.assert_array_equal(actual, replay.final_score)
        np.testing.assert_array_equal(replay.dien_raw_score,
            canonical_request_logits(ranker.model, Context(), frame, ['quality'], 'cpu', 'request_idx'))

    def test_canonical_query_callers_share_forward_and_guard_cached_thread_drift(self):
        tokenizer = Mock(return_value={})
        model = Mock(return_value=SimpleNamespace(last_hidden_state=torch.ones((1, 1, 768))))
        recall, rank = DSSMRecaller.__new__(DSSMRecaller), RankStackPredictor.__new__(RankStackPredictor)
        for caller in [recall, rank]:
            caller.query_encoder_dir, caller.query_encoder_revision = '/tmp/pinned_query', 'pin'
            caller.inference_contract = REQUEST_FP32
        with patch.object(encoder, '_QUERY_VECTORS', encoder.OrderedDict()), patch.object(encoder, 'get_shared_query_encoder', return_value=(tokenizer, model)):
            first = recall._encode_query_vec('exact query').reshape(-1)
            second = rank._query_vector('exact query')
            np.testing.assert_array_equal(first, second)
            self.assertEqual(model.call_count, 1)
            tokenizer.assert_called_once_with(['exact query'], padding=True, truncation=True, max_length=512, return_tensors='pt')
            torch.set_num_threads(2)
            with self.assertRaisesRegex(RuntimeError, 'threads4'):
                rank._query_vector('exact query')
            with self.assertRaisesRegex(RuntimeError, 'threads4'):
                recall._encode_query_vec('exact query')


if __name__ == '__main__':
    unittest.main()
