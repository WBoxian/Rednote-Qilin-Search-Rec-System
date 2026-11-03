import unittest

import torch

from training.retrieval_objective import candidatewise_max_ip


class RetrievalObjectiveTests(unittest.TestCase):
    def test_actual_joined_vectors_match_exhaustive_retrieval_scores(self):
        torch.manual_seed(42)
        learned_query = torch.randn(2, 1, 3) * .5
        learned_item = torch.randn(5, 3) * .5
        content_query = torch.randn(2, 3, 3, requires_grad=True)
        content_item = torch.randn(5, 3, requires_grad=True)
        query = torch.cat([learned_query.expand(-1, 3, -1),
                           (.75 ** .5) * content_query], dim=-1)
        item = torch.cat([learned_item, (.75 ** .5) * content_item], dim=-1)
        valid = torch.tensor([[True, True, True], [True, False, False]])
        actual = candidatewise_max_ip(query, item, valid)
        expected = torch.stack([torch.stack([
            torch.stack([(q * t).sum() for t in item])
            for q in query[b][valid[b]]]).max(0).values for b in range(2)]) / .07
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        self.assertGreater(float(content_query.grad[0].abs().sum()), 0)
        self.assertEqual(float(content_query.grad[1, 1:].abs().sum()), 0)
        self.assertGreater(float(content_item.grad.abs().sum()), 0)

    def test_frozen_branch_can_change_candidate_order(self):
        # Content alone prefers item0, but actual joined IP prefers item1.
        query = torch.tensor([[[1., 1.]]])
        item = torch.tensor([[-2., 1.], [2., 0.]])
        self.assertEqual(int(candidatewise_max_ip(query, item,
                            torch.tensor([[True]])).argmax()), 1)

    def test_empty_or_nonfinite_heads_fail(self):
        q, t = torch.ones(1, 2, 3), torch.ones(2, 3)
        with self.assertRaises(ValueError):
            candidatewise_max_ip(q, t, torch.zeros(1, 2, dtype=torch.bool))
        q[0, 0, 0] = torch.nan
        with self.assertRaises(ValueError):
            candidatewise_max_ip(q, t, torch.ones(1, 2, dtype=torch.bool))

    def test_invalid_temperature_and_precision_fail(self):
        q, t, m = torch.ones(1, 1, 2), torch.ones(2, 2), torch.ones(1, 1, dtype=torch.bool)
        for temperature in [0, -1, True, float("nan"), float("inf")]:
            with self.subTest(temperature=temperature), self.assertRaises(ValueError):
                candidatewise_max_ip(q, t, m, temperature=temperature)
        with self.assertRaises(ValueError):
            candidatewise_max_ip(q.double(), t.double(), m)


if __name__ == "__main__":
    unittest.main()
