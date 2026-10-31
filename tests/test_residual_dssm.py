import tempfile,unittest
from pathlib import Path
import numpy as np,torch
from recall import dssm_trainer as trainer
from recall.residual_dssm import ResidualDSSMModel,project_content,raw_history_text_mean,save_residual_checkpoints,model_from_checkpoint

class ResidualDSSMTests(unittest.TestCase):
 def model(self,w=.5):
  vocab={'user_idx':3,'note_idx':5,'gender':3,'platform':3,'age':3,'location':3,'tax1':3,'tax2':3,'tax3':3};components=torch.zeros(128,768);components[:,:128]=torch.eye(128)
  model=ResidualDSSMModel('rec',vocab,6,42+768+128,pca_mean=torch.arange(768)*1e-5,pca_components=components,residual_weight=w).eval()
  return model,vocab
 def test_block_inner_product_has_no_cross_terms_including_empty_content(self):
  model,_=self.model(.25);torch.manual_seed(1);u=torch.nn.functional.normalize(torch.randn(4,128),dim=1);i=torch.nn.functional.normalize(torch.randn(4,128),dim=1);cu=torch.nn.functional.normalize(torch.randn(4,128),dim=1);ci=torch.nn.functional.normalize(torch.randn(4,128),dim=1);cu[0]=0
  actual=(model.join(u,cu)*model.join(i,ci)).sum(1);expected=.25*(u*i).sum(1)+.75*(cu*ci).sum(1);torch.testing.assert_close(actual,expected,atol=1e-7,rtol=1e-6)
  self.assertAlmostEqual(float(model.join(u,cu)[0].detach().norm()),.5,places=6)
 def test_empty_content_is_zero_after_nonzero_pca_center(self):
  model,_=self.model();out=project_content(torch.zeros(2,768),model.residual_pca_mean,model.residual_pca_components);self.assertEqual(float(out.abs().sum()),0.)
 def test_history_mean_recent_prefix_keeps_zero_duplicates_and_missing(self):
  class Store:
   def get_batch_vecs(self,ids):
    x=np.zeros((len(ids),768),np.float32)
    for row,i in enumerate(ids):
     if i==0:x[row,0]=1
     if i==1:x[row,1]=1
    return torch.from_numpy(x)
  value=raw_history_text_mean([0,0,1]+[9]*17+[1],Store());np.testing.assert_allclose(value[:2],np.array([2,1])/np.sqrt(5),rtol=1e-6);self.assertEqual(float(raw_history_text_mean([9],Store()).sum()),0.)
 def test_checkpoint_full_and_both_partial_keep_pca_and_original_keys(self):
  model,vocab=self.model();meta={'scene':'rec','cat_vocabs':vocab,'item_dense_dim':6,'user_dense_dim':938,'numeric_preprocessing':None,'known_user_ids':[0,1],'unknown_user_id':2}
  with tempfile.TemporaryDirectory() as tmp:
   dest=Path(tmp)/'model';save_residual_checkpoints(model,meta,dest);full=model_from_checkpoint(dest/'dssm_rec_residual.pt');user=model_from_checkpoint(dest/'dssm_rec_residual_user_tower.pt',allow_partial=True);item=model_from_checkpoint(dest/'dssm_rec_residual_item_tower.pt',allow_partial=True)
   for name,value in model.state_dict().items():torch.testing.assert_close(full.state_dict()[name],value,rtol=0,atol=0)
   for active in [user,item]:
    self.assertEqual(active.retrieval_dim,256);torch.testing.assert_close(active.residual_pca_components,model.residual_pca_components,rtol=0,atol=0)
   payload=torch.load(dest/'dssm_rec_residual_user_tower.pt',weights_only=False);payload['meta']['residual_weight']=.75
   with self.assertRaises(ValueError):model_from_checkpoint(payload,allow_partial=True)
 def test_factory_rejects_missing_pca_or_invalid_geometry(self):
  model,vocab=self.model();meta={'scene':'rec','cat_vocabs':vocab,'item_dense_dim':6,'user_dense_dim':938,'model_family':'residual_dssm_v1','base_hidden_dim':128,'retrieval_dim':256,'pca_input_dim':768,'pca_dim':128,'residual_weight':.5};state=model.state_dict();del state['residual_version']
  with self.assertRaises(ValueError):model_from_checkpoint({'meta':meta,'state_dict':state})
 def test_actual_forward_user_requires_content_and_returns256(self):
  model,_=self.model();batch={k:torch.zeros(2,dtype=torch.long) for k in ['user_idx','gender_enc','platform_enc','age_enc','location_enc']};batch.update({'fans_num':torch.zeros(2),'follows_num':torch.zeros(2),'dense_feats':torch.zeros(2,40)})
  with self.assertRaises(ValueError):model.forward_user(batch,torch.zeros(2,768))
  batch['residual_history_text']=torch.zeros(2,768);out=model.forward_user(batch,torch.zeros(2,768));self.assertEqual(tuple(out.shape),(2,256));self.assertEqual(float(out[:,128:].detach().abs().sum()),0.)
if __name__=='__main__':unittest.main()
