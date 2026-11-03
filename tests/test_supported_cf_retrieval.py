import unittest

import numpy as np
from scipy import sparse

from recall.cf_retrieval import CFGraphRecaller
from recall.supported_cf_retrieval import SupportedCFGraphRecaller


class SupportedCFTests(unittest.TestCase):
    def fixture(self, threshold):
        # Item0 has large weight but only one user; item1 has two users.
        graph = sparse.csr_matrix([[100., 1., 0.], [0., .25, 1.]], dtype=np.float32)
        users, items = np.array([0, 1]), np.array([0, 1, 2])
        swing = sparse.csr_matrix([[0., 0., 0.], [100., 1., 0.], [0., 1., 0.]])
        config = {"fold_in": True, "neighbors": 2, "min_item_users": threshold}
        base = CFGraphRecaller.from_arrays(graph, users, items, swing, config)
        gated = SupportedCFGraphRecaller.from_arrays(graph, users, items, swing, config)
        return graph, base, gated

    def test_counts_users_not_weight_and_filters_before_topk(self):
        graph, base, gated = self.fixture(2)
        self.assertEqual(gated.item_user_counts.tolist(), [1, 2, 1])
        self.assertEqual(list(base.score(99, [1], topk=1)["swing"]), [0])
        self.assertEqual(list(gated.score(99, [1], topk=1)["swing"]), [1])
        self.assertEqual(graph[0, 0], 100)

    def test_history_and_user_similarity_are_preserved(self):
        _, base, gated = self.fixture(2)
        np.testing.assert_array_equal(base.normalized.toarray(), gated.normalized.toarray())
        np.testing.assert_array_equal(base.sqrt_idf, gated.sqrt_idf)
        result, support = gated.score_with_support(99, [2], topk=5)
        self.assertEqual(support["history_items_in_graph"], 1)
        for route in result.values():
            self.assertTrue(set(route).issubset({1}))

    def test_threshold_one_matches_original_for_positive_graph(self):
        _, base, gated = self.fixture(1)
        for user in [0, 1, 99]:
            self.assertEqual(base.score(user, [0, 1, 2]), gated.score(user, [0, 1, 2]))

    def test_zero_edges_and_duplicate_entries_do_not_inflate_user_count(self):
        graph = sparse.coo_matrix(([1., 2., 0.], ([0, 0, 1], [0, 0, 0])), shape=(2, 1))
        model = SupportedCFGraphRecaller.from_arrays(graph, np.array([0, 1]), np.array([0]),
                sparse.csr_matrix((1, 1)), {"fold_in": True, "neighbors": 2, "min_item_users": 2})
        self.assertEqual(model.item_user_counts.tolist(), [1])
        self.assertFalse(model.eligible_items[0])

    def test_threshold_requires_explicit_integer(self):
        for value in [None, True, 0, 1.5]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.fixture(value)


if __name__ == "__main__":
    unittest.main()
