import concurrent.futures
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import backend.online.shared_query_encoder as encoder


class SharedQueryEncoderTests(unittest.TestCase):
    def setUp(self):
        self.cached = dict(encoder._ENCODERS)
        self.failed = set(encoder._FAILED)
        encoder._ENCODERS.clear()
        encoder._FAILED.clear()

    def tearDown(self):
        encoder._ENCODERS.clear()
        encoder._ENCODERS.update(self.cached)
        encoder._FAILED.clear()
        encoder._FAILED.update(self.failed)

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