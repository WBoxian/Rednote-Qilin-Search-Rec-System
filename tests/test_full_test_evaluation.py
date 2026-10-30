import unittest
import numpy as np
import pandas as pd
import duckdb
from tempfile import TemporaryDirectory
from pathlib import Path
from backend.offline.evaluate_full_test import ranking_metrics, recall_summary, require, evaluate_recall


class FullTestMetricTests(unittest.TestCase):
    def test_missed_and_empty_requests_stay_in_denominator(self):
        requests = pd.DataFrame({
            "gid": [1, 2, 3], "user_idx": [1, 2, 3], "positives": [2, 1, 0],
            "candidates": [2, 0, 0], "first_hit": [2., np.nan, np.nan],
            "rr@100": [.5, 0., 0.]})
        for k in (100, 500, 1000):
            requests[f"hits@{k}"] = [1, 0, 0]
            requests[f"hit@{k}"] = [1., 0., 0.]
            requests[f"recall@{k}"] = [.5, 0., np.nan]
        result = recall_summary(requests)
        self.assertEqual(result["all_requests"], 3)
        self.assertEqual(result["positive_requests"], 2)
        self.assertEqual(result["HitRate@500"], .5)
        self.assertEqual(result["Recall@500"], .25)
        self.assertEqual(result["MicroRecall@500"], 1/3)
        self.assertEqual(result["MedianFirstHitRank_among_hits@1000"], 2.)
        self.assertEqual(result["missed_requests@1000"], 1)

    def test_real_candidate_join_preserves_null_misses(self):
        with TemporaryDirectory() as directory, duckdb.connect() as con:
            root = Path(directory)
            output = root / "out"
            output.mkdir()
            data = root / "outputs/data"
            data.mkdir(parents=True)
            index = root / "outputs/deploy/rec/hard/index"
            index.mkdir(parents=True)
            np.save(index / "dssm_rec_hard_row2note.npy", np.array([8, 9, 10]))
            pd.DataFrame({"request_idx": [1, 1], "note_idx": [8, 9], "rank": [1, 2],
                          "recall_score": [.9, .8]}).to_parquet(data / "recall_rec_test_hard_multiroute_top1000.parquet")
            con.execute("CREATE TABLE requests_test AS SELECT * FROM (VALUES (1,7,[8]),(2,8,[]),(3,9,[])) t(gid,user_idx,history)")
            con.execute("CREATE TABLE labels_test AS SELECT * FROM (VALUES (1,9,7,1),(2,10,8,1),(3,8,9,0)) t(gid,note_idx,user_idx,positive)")
            con.execute("CREATE TABLE labels_train AS SELECT * FROM labels_test WHERE gid=1")
            result, _ = evaluate_recall(con, root, "rec", "request_idx", output)
            self.assertEqual(result["all_requests"], 3)
            self.assertEqual(result["Recall@500"], .5)
            self.assertEqual(result["MRR@100"], .25)
            self.assertEqual(result["empty_candidate_requests"], 2)

    def test_auc_ties_and_ndcg_do_not_use_label_tie_breaks(self):
        frame = pd.DataFrame({"gid": [1, 1], "user_idx": [7, 7],
                              "note_idx": [9, 8], "positive": [1, 0]})
        result, _, _ = ranking_metrics(frame, np.array([.5, .5]))
        self.assertEqual(result["AUC"], .5)
        self.assertAlmostEqual(result["NDCG@10"], (1+1/np.log2(3))/2)
        reverse, _, _ = ranking_metrics(frame.iloc[::-1], np.array([.5, .5]))
        self.assertEqual(result["NDCG@10"], reverse["NDCG@10"])

    def test_gauc_excludes_single_class_users(self):
        frame = pd.DataFrame({"gid": [1, 1, 2], "user_idx": [7, 7, 8],
                              "note_idx": [1, 2, 3], "positive": [1, 0, 1]})
        result, _, _ = ranking_metrics(frame, np.array([.9, .1, .8]))
        self.assertEqual(result["GAUC"], 1.)
        self.assertEqual(result["gauc_effective_rows"], 2)
        self.assertEqual(result["gauc_excluded_users"], 1)

    def test_predictions_must_cover_all_rows_and_be_finite(self):
        frame = pd.DataFrame({"gid": [1], "user_idx": [1], "note_idx": [1], "positive": [1]})
        for scores in (np.array([]), np.array([np.nan]), np.array([np.inf])):
            with self.assertRaises(ValueError):
                ranking_metrics(frame, scores)

    def test_undefined_metric_is_not_fabricated(self):
        frame = pd.DataFrame({"gid": [1], "user_idx": [1], "note_idx": [1], "positive": [0]})
        result, _, _ = ranking_metrics(frame, np.array([.1]))
        self.assertIsNone(result["AUC"])
        self.assertIsNone(result["GAUC"])
        self.assertIsNone(result["NDCG@10"])



class AcceptanceTests(unittest.TestCase):
    def test_time_order_and_missing_lineage_are_separate_checks(self):
        from backend.offline.evaluation_contract import acceptance_checks
        scene = {"data_audit": {"train_test_request_overlap": 0, "chronological_split": False},
                 "ranking": {"dien": {"feature_audit": {
                     "exposure_position_features": ["position"], "missing_numeric_features": []}}}}
        result = acceptance_checks({"rec": scene})
        checks = {row["check"]: row["status"] for row in result["checks"]}
        self.assertEqual(checks["request_id_disjoint"], "PASS")
        self.assertEqual(checks["chronological_split"], "FAIL")
        self.assertEqual(checks["pre_exposure_features"], "FAIL")
        self.assertEqual(checks["model_index_training_lineage"], "UNKNOWN")
        self.assertEqual(result["status"], "BLOCKED")
        scene["data_audit"]["chronological_split"] = True
        scene["ranking"]["dien"]["feature_audit"]["exposure_position_features"] = []
        self.assertEqual(acceptance_checks({"rec": scene})["status"], "BLOCKED")

    def test_empty_scenes_cannot_pass(self):
        from backend.offline.evaluation_contract import acceptance_checks
        self.assertEqual(acceptance_checks({})["status"], "BLOCKED")

    def test_empty_and_invalid_label_frames_are_rejected(self):
        frame = pd.DataFrame({"gid": [1], "user_idx": [1], "note_idx": [1], "positive": [2]})
        with self.assertRaises(ValueError):
            ranking_metrics(frame, np.array([.1]))
        with self.assertRaises(ValueError):
            ranking_metrics(frame.iloc[:0], np.array([]))

    def test_raw_request_disjointness_does_not_imply_time_isolation(self):
        from backend.offline.evaluate_full_test import prepare_data
        with TemporaryDirectory() as directory, duckdb.connect() as con:
            root = Path(directory)
            for split, gid, timestamp in (("train", 1, 20.), ("test", 2, 10.)):
                folder = root / "datasets" / f"recommendation_{split}"
                folder.mkdir(parents=True)
                pd.DataFrame({"request_idx": [gid], "user_idx": [7], "query": [""],
                              "recent_clicked_note_idxs": [[8]],
                              "rec_result_details_with_idx": [[{
                                  "note_idx": 9, "click": 1, "position": 1,
                                  "request_timestamp": timestamp}]]}).to_parquet(folder / "part.parquet")
            _, audit = prepare_data(con, root, "rec")
            self.assertEqual(audit["train_test_request_overlap"], 0)
            self.assertFalse(audit["chronological_split"])
            self.assertEqual(audit["train_requests_not_strictly_before_first_test"], 1)


if __name__ == "__main__":
    unittest.main()
