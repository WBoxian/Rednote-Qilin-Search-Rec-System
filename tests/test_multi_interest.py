"""Fixed K3 edge cases and an independent frozen-producer arithmetic oracle."""
import unittest
import numpy as np
from recall.multi_interest import k3_history_means


def l2(values):
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


class MultiInterestTests(unittest.TestCase):
    def test_empty_invalid_zero_and_padding(self):
        calls = []
        def transform(raw):
            calls.append(raw.copy())
            return l2(raw)
        means, meta = k3_history_means(np.empty((0, 4), np.float32), cluster_transform=transform)
        self.assertEqual(means.shape, (0, 4))
        self.assertEqual(means.dtype, np.float32)
        self.assertTrue(meta['fallback'])
        raw = np.array([[np.nan, 0], [np.inf, 1], [0, 0], [3, 4]], np.float32)
        means, meta = k3_history_means(raw, cluster_transform=transform,
                                      valid_mask=np.array([True, True, True, False]))
        self.assertEqual(means.shape, (0, 2))
        self.assertTrue(meta['fallback'])
        self.assertEqual(calls, [])

    def test_single_identical_and_repeats_preserve_members(self):
        raw = np.array([[3, 4], [3, 4], [3, 4]], np.float32)
        means, meta = k3_history_means(raw, cluster_transform=l2)
        np.testing.assert_array_equal(means, [[np.float32(.6), np.float32(.8)]])
        self.assertEqual(meta['seed_indices'], [0])
        self.assertEqual(meta['member_indices'], [[0, 1, 2]])
        self.assertEqual(meta['assignments'], [0, 0, 0])
        self.assertFalse(meta['fallback'])
        single, info = k3_history_means(raw[:1], cluster_transform=l2)
        np.testing.assert_array_equal(single, means)
        self.assertEqual(info['source_tokens'], 1)

    def test_antipodal_heads_keep_original_cluster_order(self):
        raw = np.array([[1, 0], [-1, 0]], np.float32)
        means, meta = k3_history_means(raw, cluster_transform=l2)
        np.testing.assert_array_equal(means, raw)
        self.assertEqual(meta['seed_indices'], [0, 1])
        self.assertEqual(meta['member_indices'], [[0], [1]])

    def test_zero_projection_preserves_frozen_duplicate_seed_behavior(self):
        raw = np.array([[1, 0], [-1, 0]], np.float32)
        means, meta = k3_history_means(raw, cluster_transform=lambda r: np.zeros_like(r))
        np.testing.assert_array_equal(means, np.zeros((1, 2), np.float32))
        self.assertEqual(meta['seed_indices'], [0, 0, 0])
        self.assertEqual(meta['member_indices'], [[0, 1]])
        self.assertFalse(meta['fallback'])  # Raw history exists; its mean cancels.

    def test_ties_choose_earliest_token_and_lowest_head(self):
        raw = np.array([[1, 0], [0, 1], [0, -1], [-1, 0]], np.float32)
        means, meta = k3_history_means(raw, cluster_transform=l2)
        self.assertEqual(meta['seed_indices'], [0, 3, 1])
        self.assertEqual(meta['cluster_ids'], [0, 1, 2])
        self.assertEqual(meta['member_indices'], [[0, 2], [3], [1]])
        np.testing.assert_array_equal(means[0], l2(np.array([.5, -.5], np.float32)))
        np.testing.assert_array_equal(means[1:], [[-1, 0], [0, 1]])

    def test_raw_member_mean_is_not_normalized_token_mean(self):
        raw = np.array([[10, 0], [0, 1]], np.float32)
        means, meta = k3_history_means(raw, cluster_transform=lambda r: np.zeros_like(r))
        expected = raw.mean(0)
        expected = (expected / max(float(np.linalg.norm(expected)), 1e-12)).astype(np.float32)
        np.testing.assert_array_equal(means[0], expected)
        self.assertFalse(np.array_equal(means[0], l2(l2(raw).mean(0))))
        self.assertEqual(meta['source_tokens'], 2)

    def test_separate_padding_mask_and_invalid_filter_keep_source_offsets(self):
        raw = np.array([[np.nan, 0], [0, 0], [3, 4], [1, 2]], np.float32)
        means, meta = k3_history_means(raw, cluster_transform=l2,
                                      valid_mask=np.array([True, True, True, False]))
        np.testing.assert_array_equal(means, [[np.float32(.6), np.float32(.8)]])
        self.assertEqual(meta['valid_indices'], [2])
        self.assertEqual(meta['seed_indices'], [2])
        self.assertEqual(meta['assignments'], [-1, -1, 0, -1])
        # Offset0 can be a real token (including caller's note0), never padding ID.
        means, meta = k3_history_means(raw[2:], cluster_transform=l2)
        self.assertIn(0, meta['valid_indices'])

    def test_deterministic_and_does_not_modify_inputs_even_if_callback_mutates(self):
        raw = np.random.default_rng(42).normal(size=(20, 8)).astype(np.float32)
        original = raw.copy()
        first, a = k3_history_means(raw, cluster_transform=l2)
        second, b = k3_history_means(raw, cluster_transform=l2)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(a, b)
        np.testing.assert_array_equal(raw, original)
        def overwrite(received):
            received[:] = 0
            return received
        means, meta = k3_history_means(raw, cluster_transform=overwrite)
        expected = raw.mean(0)
        expected = (expected / max(float(np.linalg.norm(expected)), 1e-12)).astype(np.float32)
        np.testing.assert_array_equal(means[0], expected)
        np.testing.assert_array_equal(raw, original)

    def test_required_transform_and_bad_contracts_rejected(self):
        raw = np.ones((2, 3), np.float32)
        with self.assertRaises(TypeError):
            k3_history_means(raw)
        for value in ([], np.empty((2, 0), np.float32), np.ones((2, 3), bool),
                      np.ones((2, 3), complex), [['x']]):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    k3_history_means(value, cluster_transform=l2)
        for callback in (None, lambda r: np.ones((1, 3), np.float32),
                         lambda r: np.ones((2, 3), np.float64),
                         lambda r: np.full((2, 3), np.nan, np.float32)):
            with self.subTest(callback=callback):
                with self.assertRaises(ValueError):
                    k3_history_means(raw, cluster_transform=callback)
        for mask in ([1, 0], np.ones(3, bool), np.ones((2, 1), bool)):
            with self.subTest(mask=mask):
                with self.assertRaises(ValueError):
                    k3_history_means(raw, cluster_transform=l2, valid_mask=mask)


if __name__ == '__main__':
    unittest.main()
