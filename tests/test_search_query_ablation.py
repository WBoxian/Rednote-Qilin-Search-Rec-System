import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location("query_ablation",Path(__file__).resolve().parents[1]/"scripts/analyze_search_query_ablation.py")
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class QueryAblationTests(unittest.TestCase):
    def test_unmatched_requests_stay_in_denominator(self):
        summary,_=module.summarize([{10,11},{20}],[[0,10],[9,8]])
        self.assertEqual(summary["requests"],2)
        self.assertEqual(summary["Recall@500"],.25)
        self.assertEqual(summary["HitRate@500"],.5)
        self.assertEqual(summary["micro_recall500"],1/3)
        self.assertEqual(summary["MRR@100"],.25)
    def test_hit_after_100_does_not_receive_mrr_credit(self):
        summary,_=module.summarize([{9999}],[list(range(100))+[9999]])
        self.assertEqual(summary["Recall@500"],1.)
        self.assertEqual(summary["MRR@100"],0.)

if __name__=="__main__":unittest.main()
