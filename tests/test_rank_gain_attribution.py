import importlib.util
from pathlib import Path
import unittest
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("rank_gains",ROOT/"scripts/analyze_rank_gains.py")
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class RankGainTests(unittest.TestCase):
    def test_retention_averages_ties_without_label_ordering(self):
        frame=pd.DataFrame({"gid":[1,1,2],"user_idx":[1,1,2],"positive":[1,0,1]})
        result=module.retention(frame,[.5,.5,.1],ks=(1,))
        self.assertAlmostEqual(result["1"]["macro_positive_impression_retention"],.75)
        self.assertAlmostEqual(result["1"]["expected_positive_impressions_retained"],1.5)

    def test_paired_gain_counts_all_requests(self):
        frame=pd.DataFrame({"gid":[1,2,3],"user_idx":[1,2,3],"gbdt_ndcg":[.1,.3,.5],
                            "dien_ndcg":[.2,.3,.4],"delta":[.1,0.,-.1]})
        result=module.paired_summary(frame)
        self.assertEqual((result["win_requests"],result["tie_requests"],result["loss_requests"]),(1,1,1))
        self.assertAlmostEqual(result["mean_ndcg_delta"],0.)
        self.assertEqual(result["requests"],3)

if __name__=="__main__":unittest.main()
