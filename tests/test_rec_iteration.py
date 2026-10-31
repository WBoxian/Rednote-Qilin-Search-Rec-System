import unittest
from unittest.mock import patch
import numpy as np,pandas as pd,torch
from scripts.optimize_rec_iteration import contrastive_loss,logged_pairwise_loss,safe_negative_lists,LoggedNegativeDataset
from scripts.optimize_rec_retrieval import multi_positive_inbatch_loss
from recall.dssm_trainer import DSSMDataset

class RecIterationTests(unittest.TestCase):
 def test_control_is_existing_v2_objective(self):
  torch.manual_seed(42);u=torch.nn.functional.normalize(torch.randn(7,8),dim=1);i=torch.nn.functional.normalize(torch.randn(7,8),dim=1);ids=torch.tensor([0,1,2,2,3,4,5]);req=torch.tensor([1,1,2,3,4,5,6]);users=torch.tensor([1,1,2,3,4,5,6]);hist=torch.full((7,20),-1)
  torch.testing.assert_close(contrastive_loss(u,i,ids,req,users,hist),multi_positive_inbatch_loss(u,i,ids,req),rtol=0,atol=0)
 def test_mask_removes_false_negatives_without_promoting_to_positive(self):
  u=torch.eye(3);i=torch.tensor([[1.,0,0],[1.,0,0],[0,0,1.]]);ids=torch.tensor([0,1,2]);req=torch.tensor([1,2,3]);users=torch.tensor([1,1,2]);hist=torch.full((3,20),-1);hist[2,0]=1
  plain=contrastive_loss(u,i,ids,req,users,hist);masked=contrastive_loss(u,i,ids,req,users,hist,True)
  self.assertLess(float(masked),float(plain));expected=((torch.logsumexp(torch.tensor([1.,0.])/.07,0)-1/.07)*2+torch.log(torch.tensor(2.)))/3
  self.assertAlmostEqual(float(masked),float(expected),places=5)
 def test_current_request_and_same_item_positives_remain_positive(self):
  u=torch.eye(2);i=torch.eye(2);ids=torch.tensor([0,1]);req=torch.tensor([1,1]);users=torch.tensor([3,3]);hist=torch.tensor([[1],[0]])
  self.assertEqual(float(contrastive_loss(u,i,ids,req,users,hist,True)),0.)
 def test_hard_negatives_remove_any_clicked_duplicate_and_current_history(self):
  frame=pd.DataFrame({'request_idx':[1]*7+[2]*2,'note_idx':[0,0,1,2,3,4,4,6,7],'click':[1,0,0,0,1,0,0,1,0],'recent_clicked_note_idxs':[[1,2]]*7+[[]]*2})
  result=safe_negative_lists(frame);self.assertEqual(result,{1:[4],2:[7]})
 def test_pairwise_gradient_and_missing_negative_mask(self):
  u=torch.tensor([[1.,0],[1.,0]],requires_grad=True);p=torch.tensor([[0.,1.],[0.,1.]],requires_grad=True);n=torch.tensor([[1.,0],[1.,0]],requires_grad=True);loss=logged_pairwise_loss(u,p,n,torch.tensor([True,False]));loss.backward();self.assertGreater(float(loss.detach()),1.);self.assertEqual(float(n.grad[1].abs().sum()),0.);self.assertGreater(float(n.grad[0].abs().sum()),0.)
  zero=logged_pairwise_loss(u,p,n,torch.tensor([False,False]));self.assertEqual(float(zero.detach()),0.)
 def test_actual_dataset_uses_only_valid_explicit_logged_negatives(self):
  row={'user_idx':0,'request_idx':10,'note_idx':0,'gender_enc':0,'platform_enc':0,'age_enc':0,'location_enc':0,'fans_num':0.,'follows_num':0.,'recent_clicked_note_idxs':[0,2],'click':1,'hard':[1]};row.update({f'dense_feat{i}':0. for i in range(1,41)})
  meta={i:{'note_idx':i,'note_type':1,'taxonomy':np.zeros(3,np.int64),'dense_stats':np.zeros(1,np.float32)} for i in [0,1]}
  dataset=DSSMDataset('rec','request_idx',pd.DataFrame([row]),meta,[0,1],torch.tensor([.5,.5]),use_hard_neg=True,hard_neg_col='hard',use_inbatch_neg=True)
  user,positive,negative,kind,label=dataset[0];self.assertEqual(user['history_ids'],[0,2]);self.assertEqual(positive['note_idx'],0);self.assertEqual(negative[0]['note_idx'],1);self.assertEqual(kind.tolist(),[1])

 def test_full_logged_pool_is_not_silently_truncated_to_trainer_top10(self):
  class Base:
   def __len__(self):return 1
   def __getitem__(self,i):return {},{'note_idx':0},[{'note_idx':0}],np.array([0]),1
   def _fetch_item(self,i):return {'note_idx':i}
  wrapped=LoggedNegativeDataset(Base(),[list(range(1,21))])
  with patch('numpy.random.randint',return_value=19):row=wrapped[0]
  self.assertEqual(row[2][0]['note_idx'],20);self.assertEqual(row[3].tolist(),[1])
if __name__=='__main__':unittest.main()
