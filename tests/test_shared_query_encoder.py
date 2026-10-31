import concurrent.futures
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

import numpy as np
import torch

import backend.online.shared_query_encoder as encoder
from backend.online.recall.dssm import DSSMRecaller
from backend.online.ranking.stack import RankStackPredictor


class SharedQueryEncoderTests(unittest.TestCase):
    def setUp(self):
        self.cached = dict(encoder._ENCODERS)
        self.failed = set(encoder._FAILED)
        self.vectors = encoder._QUERY_VECTORS.copy()
        encoder._ENCODERS.clear()
        encoder._FAILED.clear()
        encoder._QUERY_VECTORS.clear()

    def tearDown(self):
        encoder._ENCODERS.clear()
        encoder._ENCODERS.update(self.cached)
        encoder._FAILED.clear()
        encoder._FAILED.update(self.failed)
        encoder._QUERY_VECTORS.clear()
        encoder._QUERY_VECTORS.update(self.vectors)

    def test_actual_recall_and_rank_share_one_forward_and_return_independent_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            tokenizer = Mock(return_value={"input_ids": torch.ones((1, 3), dtype=torch.long)})
            model = Mock(return_value=SimpleNamespace(last_hidden_state=torch.arange(768, dtype=torch.float32).reshape(1, 1, 768)))
            encoder._ENCODERS[(str(Path(directory).resolve()), "pin")] = tokenizer, model
            recall, rank = DSSMRecaller.__new__(DSSMRecaller), RankStackPredictor.__new__(RankStackPredictor)
            for caller in [recall, rank]:
                caller.query_encoder_dir, caller.query_encoder_revision = directory, "pin"
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                values = list(pool.map(lambda index: recall._encode_query_vec(" exact query ").reshape(-1) if index % 2 else rank._query_vector("exact query"), range(12)))
            self.assertEqual(model.call_count, 1)
            self.assertEqual(tokenizer.call_count, 1)
            tokenizer.assert_called_once_with(["exact query"], padding=True, truncation=True, max_length=512, return_tensors="pt")
            np.testing.assert_array_equal(values[0], values[1])
            values[0][0] = 999
            self.assertNotEqual(encoder.encode_current_query("exact query", directory, "pin")[0], 999)
            self.assertFalse(next(iter(encoder._QUERY_VECTORS.values())).flags.writeable)

    def test_query_cache_separates_pins_and_exact_text_and_is_bounded(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(encoder, "_QUERY_CACHE_SIZE", 2):
            tokenizer, model = Mock(return_value={}), Mock(return_value=SimpleNamespace(last_hidden_state=torch.ones((1, 1, 768))))
            for revision in ["one", "two"]:
                encoder._ENCODERS[(str(Path(directory).resolve()), revision)] = tokenizer, model
            encoder.encode_current_query("Query", directory, "one")
            encoder.encode_current_query("Query", directory, "two")
            encoder.encode_current_query("query", directory, "two")
            self.assertEqual(model.call_count, 3)
            self.assertEqual(len(encoder._QUERY_VECTORS), 2)

    def test_failed_encoding_does_not_cache_zero_or_replace_subsequent_success(self):
        with tempfile.TemporaryDirectory() as directory:
            tokenizer = Mock(return_value={})
            model = Mock(side_effect=[SimpleNamespace(last_hidden_state=torch.zeros((1, 1, 768))), SimpleNamespace(last_hidden_state=torch.ones((1, 1, 768)))])
            encoder._ENCODERS[(str(Path(directory).resolve()), "pin")] = tokenizer, model
            with self.assertRaises(RuntimeError):
                encoder.encode_current_query("query", directory, "pin")
            self.assertEqual(len(encoder._QUERY_VECTORS), 0)
            result = encoder.encode_current_query("query", directory, "pin")
            self.assertAlmostEqual(float(np.linalg.norm(result)), 1., places=6)
            self.assertEqual(model.call_count, 2)

    def test_pinned_local_directory_and_revision_are_forwarded_without_network(self):
        tokenizer, model = Mock(), Mock()
        model.from_pretrained.return_value.to.return_value = model
        with tempfile.TemporaryDirectory() as directory, patch.object(encoder, "AutoTokenizer", tokenizer), patch.object(encoder, "AutoModel", model):
            first = encoder.get_shared_query_encoder(directory, "training-commit")
            second = encoder.get_shared_query_encoder(directory, "training-commit")
            self.assertIs(first, second)
            tokenizer.from_pretrained.assert_called_once_with(str(Path(directory).resolve()), local_files_only=True, revision="training-commit")
            model.from_pretrained.assert_called_once()

    def test_failed_legacy_alias_does_not_disable_a_valid_pinned_bundle(self):
        tokenizer, model = Mock(), Mock()
        tokenizer.from_pretrained.side_effect = [OSError("legacy cache absent"), object()]
        model.from_pretrained.return_value.to.return_value = model
        with tempfile.TemporaryDirectory() as directory, patch.object(encoder, "AutoTokenizer", tokenizer), patch.object(encoder, "AutoModel", model):
            self.assertEqual(encoder.get_shared_query_encoder(), (None, None))
            self.assertIsNotNone(encoder.get_shared_query_encoder(directory, "training-commit")[0])

    def test_concurrent_same_revision_is_loaded_once_and_different_revision_is_distinct(self):
        tokenizer, model = Mock(), Mock()
        model.from_pretrained.return_value.to.return_value = model
        with tempfile.TemporaryDirectory() as directory, patch.object(encoder, "AutoTokenizer", tokenizer), patch.object(encoder, "AutoModel", model):
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                values = list(pool.map(lambda _: encoder.get_shared_query_encoder(directory, "one"), range(16)))
            self.assertTrue(all(value is values[0] for value in values))
            self.assertEqual(model.from_pretrained.call_count, 1)
            encoder.get_shared_query_encoder(directory, "two")
            self.assertEqual(model.from_pretrained.call_count, 2)


if __name__ == "__main__":
    unittest.main()
