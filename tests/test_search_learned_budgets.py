import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from scripts.compare_search_learned_budgets import budget_guard, summarize_budget, main


class LearnedBudgetTests(unittest.TestCase):
    def test_predeclared_guard_requires_r10_gain_and_no_r20_regression(self):
        old={"Recall@10":.3,"Recall@20":.5}
        self.assertTrue(budget_guard(old,{"Recall@10":.31,"Recall@20":.5}))
        self.assertFalse(budget_guard(old,{"Recall@10":.3,"Recall@20":.6}))
        self.assertFalse(budget_guard(old,{"Recall@10":.4,"Recall@20":.49}))

    def test_more_reachable_positives_does_not_imply_ranking_gain(self):
        frame=pd.DataFrame({"gid":[1]*40,"note_idx":range(1,41),"rank":range(1,41)})
        requests=pd.DataFrame({"search_idx":[1],"user_idx":[8],"positives":[[40]]})
        old,_,_=summarize_budget(frame,requests,np.ones(20),20)
        scores=np.ones(40);scores[-1]=0.
        wide,_,_=summarize_budget(frame,requests,scores,40)
        self.assertEqual(old["pool_recall"],0.)
        self.assertEqual(wide["pool_recall"],1.)
        self.assertEqual(wide["Recall@10"],0.)
        self.assertEqual(wide["Recall@20"],0.)
        self.assertFalse(budget_guard(old,wide))

    def test_more_candidates_can_hurt_r10_without_changing_pool_recall(self):
        frame=pd.DataFrame({"gid":[1]*20,"note_idx":range(1,21),"rank":range(1,21)})
        requests=pd.DataFrame({"search_idx":[1],"user_idx":[8],"positives":[[5]]})
        scores=np.r_[np.full(10,.1),np.ones(10)];scores[4]=.9
        old,_,_=summarize_budget(frame,requests,scores[:10],10)
        wide,_,_=summarize_budget(frame,requests,scores,20)
        self.assertEqual(old["pool_recall"],wide["pool_recall"])
        self.assertEqual(old["Recall@10"],1.)
        self.assertEqual(wide["Recall@10"],0.)
        self.assertFalse(budget_guard(old,wide))

    def test_rejected_validation_budget_cannot_read_test_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/"gbdt.txt").write_text("fake")
            (root/"manifest.json").write_text(json.dumps({"scene":"search","trees":{"model_sha256":hashlib.sha256(b"fake").hexdigest()}}))
            selection=root/"selection.json"
            selection.write_text(json.dumps({"guard_passed":False,"selected_cap":160}))
            argv=["budget","--stage","test","--requests","missing.parquet",
                  "--rank-stack-run",str(root),"--new-recall-file","new.parquet",
                  "--old-recall-file","old.parquet","--selection",str(selection),"--output",str(root/"out")]
            with patch("sys.argv",argv),patch("pandas.read_parquet",side_effect=AssertionError("test labels read")):
                with self.assertRaises(ValueError):
                    main()
