import sys
import unittest
import tempfile
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from train_search_balanced_dien import balanced_losses,evaluate
from train_search_lexical_ranker import lexical_group,feature_sets
from backend.online.preranking.service import build_search_item_features,SEARCH_ITEM_FEATURES

class SearchFinalExperimentTests(unittest.TestCase):
    def test_bce_weights_requests_equally_not_exposure_count(self):
        scores=torch.tensor([0.,2.,2.,2.],requires_grad=True)
        labels=torch.tensor([1.,0.,0.,0.]);groups=torch.tensor([1,2,2,2])
        bce,_=balanced_losses(scores,labels,groups)
        expected=(F.softplus(torch.tensor(0.))+F.softplus(torch.tensor(2.)))/2
        torch.testing.assert_close(bce,expected)

    def test_lambda_loss_macro_groups_and_gradients(self):
        scores=torch.tensor([1.,0.,-1.,1.],requires_grad=True)
        labels=torch.tensor([1.,0.,1.,0.]);groups=torch.tensor([1,1,2,2])
        _,loss=balanced_losses(scores,labels,groups)
        expected=(F.softplus(torch.tensor(-1.))+F.softplus(torch.tensor(2.)))/2
        torch.testing.assert_close(loss,expected)
        loss.backward();self.assertTrue(torch.isfinite(scores.grad).all())
        self.assertLess(scores.grad[0].item(),0)
        self.assertGreater(scores.grad[1].item(),0)

    def test_pairwise_loss_ignores_swap_both_below_top10(self):
        scores=torch.arange(12.,0.,-1.,requires_grad=True)
        labels=torch.zeros(12);labels[10]=1.
        _,rank=balanced_losses(scores,labels,torch.ones(12,dtype=torch.long))
        rank.backward()
        self.assertEqual(scores.grad[-1].item(),0.)
        self.assertLess(scores.grad[10].item(),0.)

    def test_all_clicked_group_has_zero_pairwise_loss(self):
        scores=torch.tensor([1.,2.],requires_grad=True)
        _,rank=balanced_losses(scores,torch.ones(2),torch.ones(2,dtype=torch.long))
        self.assertEqual(rank.item(),0.)
        rank.backward();torch.testing.assert_close(scores.grad,torch.zeros(2))

    def test_lexical_worker_matches_shared_function_exactly(self):
        positions=np.array([4,1]);titles=['北京旅游攻略','美食'];contents=['旅行安排','北京']
        indices,actual=lexical_group(('北京旅游',positions,titles,contents,['北京','旅游']))
        expected=build_search_item_features(pd.DataFrame({'note_title':titles,'note_content':contents}),'北京旅游',query_terms=['北京','旅游'])
        np.testing.assert_array_equal(indices,positions)
        np.testing.assert_allclose(actual,expected[list(SEARCH_ITEM_FEATURES)].to_numpy(),rtol=0,atol=0)

    def test_rejected_validation_candidate_cannot_open_final_test(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder);(path/'selection.json').write_text(json.dumps({'accepted':False}))
            with self.assertRaisesRegex(RuntimeError,'official test must remain unread'):
                evaluate(path)

    def test_feature_sets_reject_position_and_label(self):
        for name in ['position','click','event_timestamp','rank']:
            with self.assertRaises(ValueError):feature_sets(['query_item_cosine',name])

if __name__=='__main__':unittest.main()