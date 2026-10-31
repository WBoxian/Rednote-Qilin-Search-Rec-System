import unittest
import numpy as np
from training.scoring import blend_rank_scores


class ScoreFusionTests(unittest.TestCase):
    def test_coarse_bypass_preserves_global_scores(self):
        base=np.array([.2,.4,.7,.9],dtype=np.float32)
        np.testing.assert_array_equal(blend_rank_scores(base,None,[1,1,2,2],0,'probability_v2'),base)

    def test_fine_endpoint_preserves_global_order(self):
        raw=np.array([-100.,-2.,1.,100.],dtype=np.float32)
        score=blend_rank_scores([.1]*4,raw,[1,1,2,2],1,'probability_v2')
        self.assertTrue((np.diff(score)>0).all())
        self.assertTrue(np.isfinite(score).all())
        np.testing.assert_allclose(score[1:3],1/(1+np.exp(-raw[1:3])),rtol=1e-6)

    def test_fusion_does_not_depend_on_request_partition(self):
        args=([.1,.3,.5,.9],[-3.,0.,1.,3.])
        a=blend_rank_scores(*args,[1,1,2,2],.5,'probability_v2')
        b=blend_rank_scores(*args,[1,2,3,4],.5,'probability_v2')
        np.testing.assert_array_equal(a,b)

    def test_legacy_bypass_retains_normalization(self):
        np.testing.assert_array_equal(blend_rank_scores([.2,.4,.7,.9],None,[1,1,2,2],0),[0,1,0,1])

    def test_invalid_contract_or_teacher_or_logits_rejected(self):
        for args in [([2.],[0.],[1],.5,'probability_v2'),([.5],[float('nan')],[1],1,'probability_v2'),([.5],[0.],[1],.5,'unknown')]:
            with self.assertRaises(ValueError):blend_rank_scores(*args)


if __name__=='__main__':unittest.main()