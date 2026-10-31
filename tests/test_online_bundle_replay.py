import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('online_replay',Path(__file__).resolve().parents[1]/'scripts/replay_online_bundle.py')
replay=importlib.util.module_from_spec(spec);spec.loader.exec_module(replay)


class OnlineReplayMetricsTests(unittest.TestCase):
    def test_missing_candidates_keep_positive_denominator(self):
        result=replay.retention([],{7,8})
        self.assertTrue(all(value==0 for value in result.values()))

    def test_unrecalled_positive_stays_in_idcg(self):
        result=replay.retention([7,9],{7,8})
        self.assertEqual(result['Recall@100'],.5)
        self.assertAlmostEqual(result['known_positive_NDCG@10'],1/(1+1/np.log2(3)))

    def test_cutoffs_are_applied_to_ordered_output(self):
        result=replay.retention(list(range(101)),{100})
        self.assertEqual(result['Recall@100'],0)
        self.assertEqual(result['MRR@100'],0)
        self.assertEqual(result['HitRate@50'],0)


if __name__=='__main__':unittest.main()