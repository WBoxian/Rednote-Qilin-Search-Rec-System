"""Coarse-pool safety, input contracts, and an independent small oracle."""
import itertools
import unittest

import numpy as np

from recall.head_guard import guarded_head_order, ordinal_rank_scores


class HeadGuardTests(unittest.TestCase):
    def test_prior_cannot_pull_tail_or_external_items_into_coarse_pool(self):
        learned = np.arange(160, dtype=np.int64)
        prior = [999, 159, 105, 88, 0, 99, 100, 2, 50, 15, 40, 70, 80, 90, 10]
        result = guarded_head_order(learned, prior)
        np.testing.assert_array_equal(result[:10], [88, 0, 99, 2, 50, 15, 40, 70, 80, 90])
        self.assertEqual(set(result[:100]), set(learned[:100]))
        self.assertEqual(set(result), set(learned))
        np.testing.assert_array_equal(result[100:], learned[100:])
        self.assertNotIn(999, result)
        self.assertNotIn(159, result[:100])

    def test_empty_short_note0_and_absent_prior(self):
        for learned, prior, expected in [([], [], []), ([], [0], []), ([0], [], [0]),
                                         ([0, 8, 3], [99, 3, 0], [3, 0, 8])]:
            with self.subTest(learned=learned, prior=prior):
                actual = guarded_head_order(learned, prior)
                self.assertEqual(actual.dtype, np.int64)
                np.testing.assert_array_equal(actual, np.asarray(expected, dtype=np.int64))
        np.testing.assert_array_equal(guarded_head_order([2, 0], [0], coarse_pool=0, head_size=0), [2, 0])
        np.testing.assert_array_equal(guarded_head_order([2, 0], [0], head_size=0), [2, 0])

    def test_does_not_modify_or_alias_readonly_inputs(self):
        learned = np.array([7, 0, 3, 9], dtype=np.int64)
        prior = np.array([3, 0, 7], dtype=np.int64)
        learned.setflags(write=False)
        prior.setflags(write=False)
        result = guarded_head_order(learned, prior, coarse_pool=3, head_size=2)
        np.testing.assert_array_equal(result, [3, 0, 7, 9])
        result[0] = 999
        np.testing.assert_array_equal(learned, [7, 0, 3, 9])
        np.testing.assert_array_equal(prior, [3, 0, 7])
        result = guarded_head_order(learned, prior, head_size=0)
        self.assertFalse(np.shares_memory(result, learned))

    def test_int64_bounds_and_integer_array_conversion(self):
        largest = np.iinfo(np.int64).max
        np.testing.assert_array_equal(guarded_head_order([largest, 0], [0]), [0, largest])
        result = guarded_head_order(np.array([1, 0], np.int32), np.array([0, 1], np.uint32))
        self.assertEqual(result.dtype, np.int64)
        np.testing.assert_array_equal(result, [0, 1])

    def test_reject_duplicate_float_bool_negative_overflow_and_unordered_ids(self):
        bad = [[0, 0], [1., 2.], [True, 1], [np.bool_(False)], [-1], [2**63], [2**100],
               np.array([2**63], np.uint64), np.array([True], bool), np.array([], float),
               np.array([[1, 2]], np.int64), {0, 1}, {0: 1}, iter([0, 1]), "01"]
        for value in bad:
            for side in ("learned", "prior"):
                with self.subTest(value=value, side=side):
                    args = (value, []) if side == "learned" else ([], value)
                    with self.assertRaises(ValueError):
                        guarded_head_order(*args)

    def test_reject_invalid_pool_head_and_count(self):
        for kwargs in ({"coarse_pool": True}, {"coarse_pool": 1.0}, {"coarse_pool": -1},
                       {"head_size": False}, {"head_size": 1.0}, {"head_size": -1},
                       {"head_size": 2, "coarse_pool": 1}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    guarded_head_order([0], [0], **kwargs)
        for count in (-1, True, .5, float("inf"), 2**63):
            with self.subTest(count=count):
                with self.assertRaises(ValueError):
                    ordinal_rank_scores(count)

    def test_ordinal_scores_are_exact_research_fp64_not_probabilities(self):
        expected = np.array([1. / (60 + rank) for rank in range(1, 501)], dtype=np.float64)
        actual = ordinal_rank_scores(np.int64(500))
        self.assertEqual(actual.dtype, np.float64)
        np.testing.assert_array_equal(actual, expected)
        self.assertTrue(np.all(actual[:-1] > actual[1:]))
        self.assertNotAlmostEqual(float(actual.sum()), 1.)
        self.assertEqual(ordinal_rank_scores(0).dtype, np.float64)
        self.assertEqual(len(ordinal_rank_scores(0)), 0)

    def test_exhaustive_small_oracle_and_set_tail_invariants(self):
        universe = [0, 1, 2]
        sequences = [list(p) for length in range(4) for p in itertools.permutations(universe, length)]
        checked = 0
        for learned, prior, pool in itertools.product(sequences, sequences, range(4)):
            for head in range(pool + 1):
                # Independent oracle: rank eligible coarse items by prior.index,
                # then move them from back to front one at a time.
                eligible = [item for item in learned[:pool] if item in prior]
                preferred = sorted(eligible, key=prior.index)[:head]
                expected = learned.copy()
                for item in reversed(preferred):
                    expected.remove(item)
                    expected.insert(0, item)
                actual = guarded_head_order(learned, prior, coarse_pool=pool, head_size=head)
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(set(actual[:pool]), set(learned[:pool]))
                self.assertEqual(set(actual), set(learned))
                np.testing.assert_array_equal(actual[pool:], learned[pool:])
                checked += 1
        self.assertEqual(checked, 2560)


if __name__ == "__main__":
    unittest.main()
