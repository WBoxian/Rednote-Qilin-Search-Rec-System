import unittest
import numpy as np
from scripts.replay_stage_funnel import stage_row
class FunnelTests(unittest.TestCase):
    def test_unique_known_positives_not_impressions(self):
        row=stage_row(1,2,{10,20},{"recall":[10,10,30],"coarse":[10],"fine":[]})
        self.assertEqual(row["recall_hits"],1)
        self.assertEqual(row["recall_size"],2)
        self.assertEqual(row["recall_recall"],.5)
        self.assertEqual(row["fine_recall"],0)
    def test_unknown_items_are_not_fabricated_click_labels(self):
        row=stage_row(1,2,{10},{"stage":[99,98]})
        self.assertEqual(row["stage_hits"],0)
        self.assertNotIn("AUC",row)
    def test_no_positive_request_is_undefined_not_dropped(self):
        row=stage_row(1,2,set(),{"stage":[]})
        self.assertEqual(row["positives"],0)
        self.assertTrue(np.isnan(row["stage_recall"]))
if __name__=="__main__": unittest.main()
