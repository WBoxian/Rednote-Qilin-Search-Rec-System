import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location("search_attribution",Path(__file__).resolve().parents[1]/"scripts/analyze_search_recall.py")
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class SearchAttributionTests(unittest.TestCase):
    def test_all_requests_and_misses_keep_macro_denominator(self):
        summary,_=module.metrics({1:{10,11},2:{20},3:set()},{1:[0,10]})
        self.assertEqual(summary["all_requests"],3)
        self.assertEqual(summary["positive_requests"],2)
        self.assertEqual(summary["Recall@500"],.25)
        self.assertEqual(summary["HitRate@500"],.5)
        self.assertEqual(summary["MicroRecall@500"],1/3)
        self.assertEqual(summary["MRR@100"],.25)
        self.assertEqual(summary["empty_candidates"],2)
    def test_raw_candidate_order_and_unique_items(self):
        self.assertEqual(module.parse_candidates([[5,2.],[5,1.],[-1,4.],[9,0.]]),[5,9])
    def test_positive_free_metrics_are_undefined(self):
        summary,_=module.metrics({1:set()},{1:[2]})
        self.assertIsNone(summary["Recall@500"])
        self.assertIsNone(summary["MRR@100"])

if __name__=="__main__":unittest.main()
