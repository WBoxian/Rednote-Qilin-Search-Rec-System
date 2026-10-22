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
        first["recall"]["test"]["Recall@500"] = .99
        self.assertLess(load_metrics_snapshot("rec", "hard")["recall"]["test"]["Recall@500"], .03)

    def test_readme_values_match_shared_snapshot(self):
        for filename in ("README.md", "README_EN.md"):
            text = (ROOT / filename).read_text()
            for scene in ("search", "rec"):
                snapshot = load_metrics_snapshot(scene, "hard")
                for area in (snapshot["recall"]["test"], snapshot["ranking"]["ranking"]["test"]):
                    keys = ("HitRate@500", "Recall@500", "MRR@100") if "HitRate@500" in area else ("NDCG@10", "AUC", "GAUC")
                    for key in keys:
                        self.assertIn(f"{key} = {area[key]:.4f}", text)

if __name__ == "__main__":
    unittest.main()
