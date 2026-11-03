import os
import unittest
from pathlib import Path

os.environ["QILIN_ASYNC_PREWARM"] = "0"
from fastapi import HTTPException
from backend.online.api.main import create_app
from backend.online.metrics_snapshot import load_metrics_snapshot

ROOT = Path(__file__).resolve().parents[1]


class ProductMetricsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app(tag="hard")
        cls.endpoint = staticmethod(next(route.endpoint for route in cls.app.routes if route.path == "/api/metrics"))

    def test_api_uses_full_test_for_every_sampling_argument(self):
        for scene, count, rows in (("search", 6192, 128191), ("rec", 11115, 182567)):
            expected = load_metrics_snapshot(scene, "hard")
            for sample, include in ((2, 0), (100, 1), (1000, 1)):
                response = self.endpoint(scene=scene, tag="hard", sample_n=sample, include_val=include)
                actual = response["metrics"]
                self.assertEqual(actual, expected)
                self.assertEqual(actual["recall"]["test"]["sampled_groups"], count)
                self.assertEqual(actual["ranking"]["ranking"]["test"]["rows"], rows)
                self.assertFalse(actual["protocol"]["held_out"])
                self.assertEqual(actual["protocol"]["strict_acceptance"], "blocked")
                self.assertNotIn("validation", actual["recall"])
                self.assertNotIn("val", actual["recall"])

    def test_unknown_models_and_scenes_do_not_reuse_hard_metrics(self):
        for scene, tag in (("search", "easy"), ("unknown", "hard")):
            with self.assertRaises(HTTPException) as failure:
                self.endpoint(scene=scene, tag=tag, sample_n=1000, include_val=1)
            self.assertEqual(failure.exception.status_code, 400)

    def test_response_mutation_does_not_change_later_requests(self):
        first = load_metrics_snapshot("rec", "hard")
        expected = first["recall"]["test"]["Recall@500"]
        first["recall"]["test"]["Recall@500"] = -1.0
        self.assertEqual(load_metrics_snapshot("rec", "hard")["recall"]["test"]["Recall@500"], expected)

    def test_readme_values_match_shared_snapshot(self):
        import json
        validation = json.loads((ROOT / "src/frontend/src/data/recallValidation.json").read_text())
        self.assertFalse(validation["deployed"])
        for filename in ("README.md", "README_EN.md"):
            text = (ROOT / filename).read_text()
            search = load_metrics_snapshot("search", "hard")["pipeline"]["test"]
            for key in ("HitRate@10", "Recall@100"):
                self.assertIn(f"{key} = {search[key] * 100:.2f}%", text)
            self.assertIn(f"MRR@100 = {search['MRR@100']:.2f}", text)
            for row in validation["metrics"]:
                self.assertIn(f"{row['key']} = {row['current'] * 100:.2f}%", text)
            self.assertIn(f"{validation['requests']:,}", text)
            self.assertIn("内部验证" if filename == "README.md" else "internal validation", text)
            for scene in ("search", "rec"):
                snapshot = load_metrics_snapshot(scene, "hard")
                ranking = snapshot["ranking"]["ranking"]["test"]
                for key in ("NDCG@10", "AUC", "GAUC"):
                    self.assertIn(f"{key} = {ranking[key]:.2f}", text)
                self.assertEqual(self.endpoint(scene=scene, tag="hard", sample_n=1000, include_val=1)["metrics"]["label_coverage"], snapshot["label_coverage"])
            self.assertIn("src/backend/online/full_test_metrics.json", text)
            self.assertIn("src/frontend/src/data/recallValidation.json", text)

if __name__ == "__main__":
    unittest.main()
