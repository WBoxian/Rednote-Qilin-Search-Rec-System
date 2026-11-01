import tempfile,unittest
from pathlib import Path
import numpy as np,torch
from recall import dssm_trainer as trainer
from recall.residual_dssm import ResidualDSSMModel,AdaptedResidualDSSMModel,project_content,raw_history_text_mean,save_residual_checkpoints,model_from_checkpoint

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

class AdaptedResidualDSSMTests(unittest.TestCase):
 def model(self,dim=128):
  base,vocab=ResidualDSSMTests().model(.25)
  model=AdaptedResidualDSSMModel('rec',vocab,6,938,pca_mean=base.residual_pca_mean,pca_components=base.residual_pca_components,residual_weight=.25,content_input_dim=dim).eval()
  model.load_state_dict(base.state_dict(),strict=False)
  return model,base,vocab
 def meta(self,vocab):
  return {'scene':'rec','cat_vocabs':vocab,'item_dense_dim':6,'user_dense_dim':938,'numeric_preprocessing':None,'known_user_ids':[0,1],'unknown_user_id':2}
 def test_zero_heads_preserve_original_projection_and_raw_empty_rows(self):
  torch.manual_seed(4);raw=torch.randn(4,768);raw[0]=0
  for dim in (128,768):
   model,base,_=self.model(dim)
   for actual in (model.project_user_content(raw),model.project_item_content(raw)):
    torch.testing.assert_close(actual,base.project_user_content(raw),rtol=0,atol=0)
    self.assertEqual(float(actual[0].detach().abs().sum()),0.)
   self.assertTrue(all(not p.requires_grad for n,p in model.named_parameters() if not n.startswith('content_')))
 def test_zero_initialized_heads_have_nonzero_training_gradients(self):
  torch.manual_seed(5);raw=torch.randn(8,768);target=torch.randn(8,768)
  for dim in (128,768):
   model,_,_=self.model(dim);model.train()
   logits=model.project_user_content(raw)@model.project_item_content(target).T/.07
   torch.nn.functional.cross_entropy(logits,torch.arange(8)).backward()
   for adapter in (model.content_query_adapter,model.content_item_adapter):
    self.assertGreater(float(adapter[-1].weight.grad.norm()),0.)
   for name,module in model.named_children():
    if name not in {'content_query_adapter','content_item_adapter'}:self.assertFalse(module.training)
 def test_nonzero_raw_with_zero_pca_projection_can_learn_raw_residual(self):
  raw=torch.zeros(1,768);raw[:,128]=1
  for dim in (128,768):
   model,_,_=self.model(dim);model.residual_pca_mean.zero_()
   with torch.no_grad():model.content_query_adapter[-1].bias.fill_(.02)
   output=model.project_user_content(raw)
   self.assertAlmostEqual(float(output.detach().norm()),1. if dim==768 else 0.,places=6)
   self.assertEqual(float(model.project_user_content(torch.zeros_like(raw)).detach().abs().sum()),0.)
 def test_full_and_partial_roundtrip_preserves_real_nonzero_adapter_weights(self):
  torch.manual_seed(6);raw=torch.randn(3,768)
  for dim in (128,768):
   model,_,vocab=self.model(dim)
   with torch.no_grad():
    model.content_query_adapter[-1].bias.fill_(.02)
    model.content_item_adapter[-1].bias.fill_(-.01)
   with tempfile.TemporaryDirectory() as tmp:
    dest=Path(tmp)/'export';meta=save_residual_checkpoints(model,self.meta(vocab),dest)
    self.assertEqual(meta['model_family'],'adapted_residual_dssm_v1')
    self.assertEqual(meta['content_input_dim'],dim)
    full=model_from_checkpoint(dest/'dssm_rec_residual.pt')
    user=model_from_checkpoint(dest/'dssm_rec_residual_user_tower.pt',allow_partial=True)
    item=model_from_checkpoint(dest/'dssm_rec_residual_item_tower.pt',allow_partial=True)
    for name,value in model.state_dict().items():torch.testing.assert_close(full.state_dict()[name],value,rtol=0,atol=0)
    torch.testing.assert_close(user.project_user_content(raw),model.project_user_content(raw),rtol=0,atol=0)
    torch.testing.assert_close(item.project_item_content(raw),model.project_item_content(raw),rtol=0,atol=0)
    for tower,key in [('user','content_query_adapter.2.weight'),('item','content_item_adapter.2.weight')]:
     path=dest/f'dssm_rec_residual_{tower}_tower.pt';payload=torch.load(path,weights_only=False);del payload['state_dict'][key]
     with self.assertRaisesRegex(ValueError,'active content'):model_from_checkpoint(payload,allow_partial=True)
 def test_factory_rejects_invalid_adapter_metadata_shape_and_nonfinite_tensors(self):
  model,_,vocab=self.model()
  with tempfile.TemporaryDirectory() as tmp:
   dest=Path(tmp)/'export';save_residual_checkpoints(model,self.meta(vocab),dest)
   original=torch.load(dest/'dssm_rec_residual_user_tower.pt',weights_only=False)
   import copy
   for field,value in [('content_beta',.4),('content_input_dim',64),('content_hidden_dim',128),('content_version',2),('content_version',True)]:
    payload=copy.deepcopy(original);payload['meta'][field]=value
    with self.subTest(field=field,value=value),self.assertRaises(ValueError):model_from_checkpoint(payload,allow_partial=True)
   for key,value in [('content_version',torch.tensor(1.2)),('residual_version',torch.tensor(1.2)),('content_query_adapter.0.weight',torch.zeros(256,127)),('content_query_adapter.2.bias',torch.full((128,),float('nan')))]:
    payload=copy.deepcopy(original);payload['state_dict'][key]=value
    with self.subTest(key=key),self.assertRaises(ValueError):model_from_checkpoint(payload,allow_partial=True)
 def test_full_export_clears_inherited_partial_role_and_rejects_all_provided_nan_weights(self):
  model,_,vocab=self.model()
  with tempfile.TemporaryDirectory() as tmp:
   dest=Path(tmp)/'export';meta=save_residual_checkpoints(model,{**self.meta(vocab),'tower':'user'},dest)
   self.assertNotIn('tower',meta)
   full=model_from_checkpoint(dest/'dssm_rec_residual.pt')
   self.assertNotIn('tower',full.checkpoint_meta)
   original=torch.load(dest/'dssm_rec_residual_user_tower.pt',weights_only=False)
   import copy
   for key,value in [('content_item_adapter.2.bias',torch.full((128,),float('nan'))),('u_emb_user.weight',torch.full_like(original['state_dict']['u_emb_user.weight'],float('nan')))]:
    payload=copy.deepcopy(original);payload['state_dict'][key]=value
    with self.subTest(key=key),self.assertRaisesRegex(ValueError,'Nonfinite'):model_from_checkpoint(payload,allow_partial=True)
   payload=copy.deepcopy(original);payload['state_dict']['content_item_adapter.2.bias']=torch.zeros(127)
   with self.assertRaisesRegex(ValueError,'Invalid provided'):model_from_checkpoint(payload,allow_partial=True)
 def test_loaded_partial_cannot_export_random_inactive_tower_as_full(self):
  model,_,vocab=self.model()
  with tempfile.TemporaryDirectory() as tmp:
   dest=Path(tmp)/'original';save_residual_checkpoints(model,self.meta(vocab),dest)
   partial=model_from_checkpoint(dest/'dssm_rec_residual_user_tower.pt',allow_partial=True)
   self.assertTrue(partial.checkpoint_is_partial)
   rejected=Path(tmp)/'rejected'
   with self.assertRaisesRegex(ValueError,'complete source model'):save_residual_checkpoints(partial,partial.checkpoint_meta,rejected)
   self.assertFalse(rejected.exists())
 def test_constructor_freezes_old_dropout_modes_before_training_is_explicitly_called(self):
  _,base,vocab=self.model()
  model=AdaptedResidualDSSMModel('rec',vocab,6,938,pca_components=base.residual_pca_components,residual_weight=.25)
  self.assertTrue(model.training)
  self.assertTrue(model.content_query_adapter.training)
  for name,module in model.named_children():
   if name not in {'content_query_adapter','content_item_adapter'}:self.assertFalse(module.training)
 def test_adapted_forward_keeps_frozen_learned_block_and_empty_norm(self):
  model,base,_=self.model();batch={k:torch.zeros(2,dtype=torch.long) for k in ['user_idx','gender_enc','platform_enc','age_enc','location_enc']}
  raw=torch.randn(2,768);raw[0]=0;batch.update({'fans_num':torch.zeros(2),'follows_num':torch.zeros(2),'dense_feats':torch.zeros(2,40),'residual_history_text':raw})
  with torch.no_grad():model.content_query_adapter[-1].bias.fill_(.03)
  actual=model.forward_user(batch,torch.zeros(2,768));old=base.forward_user(batch,torch.zeros(2,768))
  torch.testing.assert_close(actual[:,:128],old[:,:128],rtol=0,atol=0)
  self.assertEqual(float(actual[0,128:].detach().abs().sum()),0.)
  self.assertAlmostEqual(float(actual[0].detach().norm()),.5,places=6)
  with self.assertRaisesRegex(ValueError,'Nonfinite'):model.project_user_content(torch.full((1,768),float('nan')))

if __name__=='__main__':unittest.main()
