"""Coverage must expose missing labels without dropping failed requests."""
import unittest

import pandas as pd

from scripts.evaluation_coverage import label_coverage


class TestEvaluationCoverage(unittest.TestCase):
    def test_empty_and_unlabeled_requests_stay_in_scope(self):
        requests = pd.DataFrame({"gid": [1, 2, 3, 4]})
        exposures = pd.DataFrame({"gid": [1, 1, 1, 2, 4], "note_idx": [0, 0, 1, 2, 3], "click": [0, 1, 0, 1, 0]})
        candidates = pd.DataFrame({"gid": [1, 1, 2, 3], "note_idx": [0, 1, 99, 20]})
        got = label_coverage(requests, exposures, candidates)
        self.assertEqual(got["all_requests"], 4)
        self.assertEqual(got["exposure_rows"], 5)
        self.assertEqual(got["empty_exposure_requests"], 1)
        self.assertEqual(got["zero_click_requests"], 2)
        self.assertEqual(got["labeled_candidate_pairs"], 2)
        self.assertEqual(got["unlabeled_candidate_pairs"], 2)
        self.assertEqual(got["labeled_candidate_fraction"], .5)
        self.assertEqual(got["requests_without_exposure_overlap"], 3)
        self.assertEqual(got["requests_without_known_click_in_candidates"], 3)
        self.assertEqual(got["requests_with_both_labels_in_overlap"], 1)
        self.assertEqual(got["empty_final_candidate_requests"], 1)

    def test_no_candidates_has_unknown_coverage_not_perfect_coverage(self):
        requests = pd.DataFrame({"gid": [1]})
        exposures = pd.DataFrame({"gid": [1], "note_idx": [0], "click": [1]})
        candidates = pd.DataFrame({"gid": pd.Series(dtype="int64"), "note_idx": pd.Series(dtype="int64")})
        got = label_coverage(requests, exposures, candidates)
        self.assertIsNone(got["labeled_candidate_fraction"])
        self.assertEqual(got["requests_without_known_click_in_candidates"], 1)

    def test_foreign_requests_and_nonbinary_labels_are_rejected(self):
        requests = pd.DataFrame({"gid": [1]})
        exposures = pd.DataFrame({"gid": [1], "note_idx": [0], "click": [2]})
        candidates = pd.DataFrame({"gid": [1], "note_idx": [0]})
        with self.assertRaisesRegex(ValueError, "binary"):
            label_coverage(requests, exposures, candidates)
        exposures["click"] = 1
        candidates["gid"] = 2
        with self.assertRaisesRegex(ValueError, "Foreign"):
            label_coverage(requests, exposures, candidates)
