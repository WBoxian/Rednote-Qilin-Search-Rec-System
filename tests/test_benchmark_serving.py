import copy
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from io import BytesIO

from scripts.benchmark_serving import request, summarize, validate


class BenchmarkServingTests(unittest.TestCase):
    def row(self, hit=False):
        return dict(ok=True, model_bundle_id="bound", degraded_reason=None, items=20,
                    cache_hit=hit, pipeline_ms=0 if hit else 12,
                    stage_ms={} if hit else {"recall": 12}, client_ms=15, cold_user=False)

    def test_cache_and_inference_are_not_interchangeable(self):
        self.assertTrue(validate(self.row(True), "cache_hit", "bound")["ok"])
        self.assertTrue(validate(self.row(), "inference", "bound")["ok"])
        self.assertTrue(validate(self.row(True), "inference", "bound")["phase_mismatch"])
        self.assertTrue(validate(self.row(), "cache_hit", "bound")["phase_mismatch"])

    def test_wrong_bundle_degradation_and_empty_output_are_errors(self):
        for change in [dict(model_bundle_id="stale"), dict(degraded_reason="route_failure"), dict(items=0)]:
            row = self.row()
            row.update(change)
            self.assertFalse(validate(row, "inference", "bound")["ok"])

    def test_cold_user_fallback_is_not_normal_inference(self):
        row = self.row()
        row["cold_user"] = True
        self.assertFalse(validate(copy.deepcopy(row), "inference", "bound")["ok"])
        self.assertTrue(validate(row, "cold", "bound")["ok"])

    def test_cached_pipeline_cost_is_not_current_inference(self):
        row = self.row(True)
        row["pipeline_ms"] = 12
        self.assertFalse(validate(row, "cache_hit", "bound")["ok"])

    def test_failed_requests_stay_in_error_denominator(self):
        rows = [self.row(), dict(ok=False, client_ms=3000)]
        result = summarize(rows)
        self.assertEqual(result["requests"], 2)
        self.assertEqual(result["errors"], 1)
        self.assertEqual(result["error_rate"], .5)
        self.assertEqual(result["latency_ms"]["p99"], 15)

    def test_cache_expiration_is_not_an_http_failure_or_a_hit_latency(self):
        hit = validate(self.row(True), "cache_hit", "bound")
        miss = self.row()
        miss["client_ms"] = 2000
        miss = validate(miss, "cache_hit", "bound")
        result = summarize([hit, miss], expected_cache_hit=True)
        self.assertEqual(result["errors"], 0)
        self.assertEqual(result["http_errors"], 0)
        self.assertEqual(result["phase_mismatches"], 1)
        self.assertEqual(result["latency_samples"], 1)
        self.assertEqual(result["latency_ms"]["p99"], 15)

    def test_all_failed_and_empty_sets_have_no_fabricated_latency(self):
        self.assertIsNone(summarize([dict(ok=False)])["latency_ms"])
        self.assertIsNone(summarize([])["error_rate"])

    def test_http_errors_preserve_status_and_body(self):
        error = HTTPError("http://localhost", 503, "unavailable", {}, BytesIO(b"not ready"))
        with patch("scripts.benchmark_serving.urlopen", side_effect=error):
            row = request("http://localhost", {"scene": "rec"}, 1)
        self.assertFalse(row["ok"])
        self.assertEqual(row["http_status"], 503)
        self.assertIn("not ready", row["error"])


if __name__ == "__main__":
    unittest.main()
