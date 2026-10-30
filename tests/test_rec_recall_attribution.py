import unittest
import pandas as pd
from scripts.analyze_rec_recall import recall_from_counts

class RecallAttributionTests(unittest.TestCase):
    def test_miss_requests_remain_and_macro_differs_from_micro(self):
        frame=pd.DataFrame({"positives":[4,1,0],"hits":[2,0,0]})
        metric=recall_from_counts(frame,"hits")
        self.assertEqual(metric["all_requests"],3)
        self.assertEqual(metric["positive_requests"],2)
        self.assertEqual(metric["macro_recall"],.25)
        self.assertEqual(metric["micro_recall"],.4)
        self.assertEqual(metric["HitRate"],.5)
    def test_no_positives_is_undefined_not_perfect(self):
        metric=recall_from_counts(pd.DataFrame({"positives":[0],"hits":[0]}),"hits")
        self.assertIsNone(metric["macro_recall"])
        self.assertIsNone(metric["micro_recall"])

if __name__=="__main__":unittest.main()
