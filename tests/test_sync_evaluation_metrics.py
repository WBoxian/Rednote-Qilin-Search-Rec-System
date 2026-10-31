"""Full-test denominators, frozen selection, and publication safety checks."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

SPEC = importlib.util.spec_from_file_location("sync_evaluation_metrics", Path(__file__).parents[1] / "scripts/sync_evaluation_metrics.py")
sync = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sync)


class TestSyncEvaluationMetrics(unittest.TestCase):
    def test_full_request_denominator_dedup_note_zero_and_actual_budget(self):
        requests = pd.DataFrame({"gid": [1, 2, 3, 4], "positives": [[0, 0, 4], [8], [9], []]})
        candidates = pd.DataFrame({"gid": [1, 1, 2], "note_idx": [0, 3, 7], "rank": [1, 2, 1]})
        result = sync.recall_metrics(requests, candidates)
        self.assertEqual(result["all_requests"], 4)
        self.assertEqual(result["positive_requests"], 3)
        self.assertEqual(result["effective_groups"], 3)
        self.assertEqual(result["positive_pairs"], 4)
        self.assertAlmostEqual(result["HitRate@10"], 1 / 3)
        self.assertAlmostEqual(result["Recall@100"], 1 / 6)
        self.assertEqual(result["MicroRecall@100"], 0.25)
        self.assertAlmostEqual(result["MRR@100"], 1 / 3)
        self.assertEqual(result["MedianFirstHitRank"], 1)
        self.assertEqual(result["missed_requests_at_500"], 2)
        self.assertEqual(result["missed_requests@500"], 2)
        self.assertEqual(result["empty_candidate_requests"], 2)
        self.assertFalse(any("1000" in name for name in result))

    def test_recall_rejects_invalid_candidate_contract(self):
        raw = pd.DataFrame({"gid": [1], "positives": [[2]]})
        cases = [
            [(1, 2, 1), (1, 2, 2)],  # duplicated item
            [(1, 2, 1), (1, 3, 1)],  # duplicated rank
            [(1, 2, 2)],             # missing rank1
            [(1, 2, 501)],           # not actual top500
            [(2, 2, 1)],             # foreign request
            [(1, -1, 1)],
        ]
        for values in cases:
            with self.subTest(values=values), self.assertRaises(ValueError):
                sync.recall_metrics(raw, pd.DataFrame(values, columns=["gid", "note_idx", "rank"]))

    def test_no_hit_median_is_null_and_misses_are_not_removed(self):
        raw = pd.DataFrame({"gid": [1, 2], "positives": [[2], [3]]})
        frame = pd.DataFrame(columns=["gid", "note_idx", "rank"])
        result = sync.recall_metrics(raw, frame)
        self.assertIsNone(result["MedianFirstHitRank"])
        self.assertEqual(result["HitRate@500"], 0)
        self.assertEqual(result["MRR@100"], 0)
        self.assertEqual(result["missed_requests_at_500"], 2)

    def test_sources_detect_post_read_mutation(self):
        with tempfile.TemporaryDirectory(prefix="qilin-metrics-test-") as temporary:
            root = Path(temporary)
            path = root / "fixture.json"
            path.write_text('{"a": 1}')
            sources = sync.Sources(root)
            sources.read(path)
            path.write_text('{"a": 2}')
            with self.assertRaisesRegex(ValueError, "changed"):
                sources.finish()

    def test_bundle_identity_and_release_strategy_are_required(self):
        with tempfile.TemporaryDirectory(prefix="qilin-metrics-test-") as temporary:
            root = Path(temporary)
            path = root / "bundle.json"
            bundle = {"scene": "rec", "tag": "hard", "ranking_strategy": "recall_only"}
            bundle["bundle_id"] = sync.bundle_identity(bundle)
            path.write_text(json.dumps(bundle))
            self.assertEqual(sync.verify_bundle(path, "rec", "recall_only", sync.Sources(root))["bundle_id"], bundle["bundle_id"])
            with self.assertRaisesRegex(ValueError, "strategy"):
                sync.verify_bundle(path, "rec", "learned", sync.Sources(root))
            bundle["ranking_strategy"] = "learned"
            path.write_text(json.dumps(bundle))
            with self.assertRaisesRegex(ValueError, "identity"):
                sync.verify_bundle(path, "rec", "learned", sync.Sources(root))

    def test_unpromoted_preview_cannot_publish(self):
        for extra in ([], ["--update-readme"], ["--output", str(sync.DEFAULT_OUTPUT)]):
            with self.subTest(extra=extra), patch("sys.stderr"), self.assertRaises(SystemExit):
                sync.main(["--bundle-search", "search.json", "--bundle-rec", "rec.json", *extra])

    def test_missing_active_bundle_fails_before_snapshot_or_write(self):
        with tempfile.TemporaryDirectory(prefix="qilin-metrics-test-") as temporary:
            root = Path(temporary)
            output = root / "new-snapshot.json"
            with patch.object(sync, "ROOT", root), patch("sys.stderr"), patch.object(sync, "build_snapshot") as build:
                with self.assertRaises(SystemExit):
                    sync.main(["--output", str(output)])
                build.assert_not_called()
                self.assertFalse(output.exists())

    def test_conditional_uses_validation_selected_run_and_frozen_hashes(self):
        with tempfile.TemporaryDirectory(prefix="qilin-metrics-test-") as temporary:
            root = Path(temporary)
            run = root / "selected"
            run.mkdir()
            (run / "manifest.json").write_text('{"scene": "search"}')
            manifest_sha = sync.sha256(run / "manifest.json")
            stage = {"NDCG@10": 0.4, "AUC": 0.6, "GAUC": 0.5, "rows": 5, "all_requests": 2}
            test = {"gbdt": stage, "selected": stage, "test_rows": 5, "test_requests": 2}
            (run / "test_metrics.json").write_text(json.dumps(test))
            test_sha = sync.sha256(run / "test_metrics.json")
            (run / "test_manifest.json").write_text(json.dumps({"frozen_training_manifest_sha256": manifest_sha, "metrics_sha256": test_sha}))
            (run / "independent_ranking_verification.json").write_text(json.dumps({"status": "PASS", "test_metrics_sha256": test_sha, "frozen_training_manifest_sha256": manifest_sha, "test_manifest_sha256": sync.sha256(run / "test_manifest.json")}))
            selection = {"search": {"selection_split": "train_internal_validation", "selected": {
                "run": "selected", "manifest_sha256": manifest_sha, "name": "frozen", "weight": 1},
                "choices": [{"name": "other", "test_NDCG": 0.99}]}}
            result = sync.conditional_ranking(root, "search", selection, sync.Sources(root), (2, 5, 3))
            self.assertEqual(result["ranking"]["test"]["NDCG@10"], 0.4)
            (run / "manifest.json").write_text('{"scene": "changed"}')
            with self.assertRaisesRegex(ValueError, "manifest changed"):
                sync.conditional_ranking(root, "search", selection, sync.Sources(root), (2, 5, 3))

    def test_readme_only_changes_four_metrics_and_latency_preserving_crlf(self):
        text = ("Intro unchanged\r\n- 在线服务\r\n  - 单机串行实测 oldnumbers\r\n- Search 离线指标\r\n"
                "  - 召回（测试集）：old\r\n  - 排序（DIEN，测试曝光）：old\r\n- Recommendation 离线指标\r\n"
                "  - 召回（测试集）：old\r\n  - 排序（DIEN，测试曝光）：old\r\nTail unchanged\r\n")
        recall = {"HitRate@10": 0.55, "HitRate@50": 0.02, "Recall@100": 0.01, "MRR@100": 0.003}
        rank = {"NDCG@10": 0.6, "AUC": 0.61, "GAUC": 0.62}
        scene = {"recall": {"test": recall}, "ranking": {"ranking": {"test": rank}}}
        result = sync.update_readme_text(text, {"scenes": {"search": scene, "rec": scene}})
        before, after = text.splitlines(keepends=True), result.splitlines(keepends=True)
        self.assertEqual(len(before), len(after))
        self.assertEqual(sum(a != b for a, b in zip(before, after)), 5)
        self.assertTrue(all(line.endswith("\r\n") for line in after))
        self.assertEqual(after[0], before[0])
        self.assertEqual(after[-1], before[-1])
        self.assertIn("HitRate@10 = 0.5500", result)
        self.assertIn("HitRate@50 = 0.0200", result)
        self.assertIn("接口返回总延迟与分阶段延迟", result)


if __name__ == "__main__":
    unittest.main()
