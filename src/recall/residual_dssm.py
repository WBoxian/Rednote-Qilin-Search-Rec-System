"""Versioned DSSM residual content features and the shared checkpoint/query contract."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from recall import dssm_trainer as trainer
from training.history import recent_history_ids
MODEL_FAMILY='residual_dssm_v1'
PCA_KEYS=('residual_pca_mean','residual_pca_components','residual_weight','residual_version')

def raw_history_text_mean(history_ids,textstore,max_len=20):
 ids=recent_history_ids(history_ids,max_len)
 if not ids:return np.zeros(768,np.float32)
 raw=textstore.get_batch_vecs(ids)
 vectors=raw.detach().cpu().numpy() if torch.is_tensor(raw) else np.asarray(raw)
 if vectors.shape!=(len(ids),768):raise ValueError('Raw history text store shape must be [history,768]')
 valid=np.isfinite(vectors).all(1)&(np.linalg.norm(vectors,axis=1)>0)
 if not valid.any():return np.zeros(768,np.float32)
 mean=vectors[valid].astype(np.float32).mean(0);norm=np.linalg.norm(mean)
 return (mean/max(float(norm),1e-12)).astype(np.float32)

def project_content(values,mean,components):
 if values.ndim!=2 or values.shape[1]!=768 or mean.shape!=(768,) or components.shape!=(128,768):raise ValueError('PCA tensor shape mismatch')
 if not torch.isfinite(values).all():raise ValueError('Nonfinite content vectors')
 raw=F.normalize(values.float(),dim=1);valid=values.float().norm(dim=1)>0
 projected=F.normalize((raw-mean.float())@components.float().T,dim=1)
 return torch.where(valid[:,None],projected,torch.zeros_like(projected))

class ResidualDSSMModel(trainer.DSSMModel):
 model_family=MODEL_FAMILY;base_hidden_dim=128;retrieval_dim=256
 def __init__(self,scene,cat_vocabs,item_dense_dim,user_dense_dim,preprocessing=None,*,pca_mean=None,pca_components=None,residual_weight=.5,mask_sequence_padding=True,mask_user_profile=True):
  if scene!='rec' or float(residual_weight) not in {.25,.5,.75}:raise ValueError('Only predeclared Rec residual weights are deployable')
  super().__init__(scene,cat_vocabs,item_dense_dim,user_dense_dim,preprocessing=preprocessing,mask_sequence_padding=mask_sequence_padding,mask_user_profile=mask_user_profile)
  self.register_buffer('residual_pca_mean',torch.zeros(768) if pca_mean is None else torch.as_tensor(pca_mean,dtype=torch.float32).clone())
  self.register_buffer('residual_pca_components',torch.zeros(128,768) if pca_components is None else torch.as_tensor(pca_components,dtype=torch.float32).clone())
  self.register_buffer('residual_weight',torch.tensor(float(residual_weight),dtype=torch.float32));self.register_buffer('residual_version',torch.tensor(1,dtype=torch.int64))
 def join(self,learned,content):
  if learned.shape[1]!=128 or content.shape!=learned.shape:raise ValueError('Both residual blocks must be aligned128dim')
  return torch.cat((self.residual_weight.sqrt()*learned,(1.-self.residual_weight).sqrt()*content),dim=1)
 def forward_user(self,batch_data,history_vecs,seq_tokens=None,seq_mask=None):
  if 'residual_history_text' not in batch_data:raise ValueError('Residual user input requires raw recent20 mean/L2 text vector')
  learned=super().forward_user(batch_data,history_vecs,seq_tokens=seq_tokens,seq_mask=seq_mask);content=project_content(batch_data['residual_history_text'],self.residual_pca_mean,self.residual_pca_components)
  return self.join(learned,content)
 def forward_item(self,batch_data,text_vecs,img_vecs):
  learned=super().forward_item(batch_data,text_vecs,img_vecs);content=project_content(text_vecs,self.residual_pca_mean,self.residual_pca_components)
  return self.join(learned,content)

def validate_residual_state(meta,state):
 if meta.get('model_family')!=MODEL_FAMILY:return
 if (meta.get('base_hidden_dim'),meta.get('retrieval_dim'),meta.get('pca_input_dim'),meta.get('pca_dim'))!=(128,256,768,128):raise ValueError('Residual metadata dimension mismatch')
 if any(k not in state for k in PCA_KEYS):raise ValueError('Missing persistent residual buffers')
 mean,components,weight,version=[state[k] for k in PCA_KEYS]
 if mean.shape!=(768,) or components.shape!=(128,768) or weight.numel()!=1 or version.numel()!=1:raise ValueError('Malformed residual buffers')
 if not all(torch.isfinite(state[k]).all() for k in PCA_KEYS):raise ValueError('Nonfinite residual buffers')
 if int(version)!=1 or float(weight) not in {.25,.5,.75} or abs(float(weight)-float(meta.get('residual_weight',-1)))>1e-8:raise ValueError('Residual weight/version differs from metadata')
 gram=components.float()@components.float().T
 if not torch.allclose(gram,torch.eye(128,device=gram.device),atol=2e-4,rtol=2e-4):raise ValueError('Residual PCA components are not orthonormal')

def model_from_checkpoint(checkpoint,*,device='cpu',allow_partial=False,metadata=None):
 path=Path(checkpoint) if isinstance(checkpoint,(str,Path)) else None
 payload=torch.load(path,map_location='cpu',weights_only=False) if path is not None else checkpoint
 if not isinstance(payload,dict):raise ValueError('Checkpoint must contain a state dictionary')
 state=payload.get('state_dict',payload);meta=dict(payload.get('meta',metadata or {}))
 if not meta and path is not None and path.with_suffix('.meta.json').exists():meta=json.loads(path.with_suffix('.meta.json').read_text())
 if not meta or not meta.get('cat_vocabs'):raise ValueError('Checkpoint requires complete tower metadata')
 family=meta.get('model_family','dssm_v1')
 if family not in {'dssm_v1',MODEL_FAMILY}:raise ValueError('Unsupported DSSM model family')
 kwargs={'scene':meta.get('scene','rec'),'cat_vocabs':meta['cat_vocabs'],'item_dense_dim':int(meta['item_dense_dim']),'user_dense_dim':int(meta['user_dense_dim']),'preprocessing':meta.get('numeric_preprocessing')}
 if family==MODEL_FAMILY:
  validate_residual_state(meta,state);model=ResidualDSSMModel(**kwargs,pca_mean=state[PCA_KEYS[0]],pca_components=state[PCA_KEYS[1]],residual_weight=float(state[PCA_KEYS[2]]))
 else:model=trainer.DSSMModel(**kwargs)
 missing,unexpected=model.load_state_dict(state,strict=False)
 if unexpected:raise ValueError('Unexpected DSSM checkpoint keys: '+str(unexpected))
 if missing:
  tower=meta.get('tower')
  if tower is None and family=='dssm_v1':
   tower='user' if any(k.startswith('u_emb_') for k in state) and not any(k.startswith('i_emb_') for k in state) else ('item' if any(k.startswith('i_emb_') for k in state) and not any(k.startswith('u_emb_') for k in state) else None)
  if not allow_partial or tower not in {'user','item'}:raise ValueError('Incomplete DSSM checkpoint')
  opposite=('i_emb_','item_mlp','item_numeric.') if tower=='user' else ('u_emb_','user_mlp','user_numeric.','seq_input_proj','seq_gru','seq_attn')
  if any(not name.startswith(opposite) for name in missing):raise ValueError('Partial checkpoint misses active tower or persistent contract')
 model.checkpoint_meta=meta;return model.to(device).float().eval()

def save_residual_checkpoints(model,meta,destination):
 destination=Path(destination);destination.mkdir(parents=True,exist_ok=False);state={k:v.detach().cpu() for k,v in model.state_dict().items()};common={**meta,'tag':'residual','model_family':MODEL_FAMILY,'base_hidden_dim':128,'retrieval_dim':256,'pca_input_dim':768,'pca_dim':128,'residual_weight':float(model.residual_weight)};validate_residual_state(common,state)
 full=destination/'dssm_rec_residual.pt';torch.save(state,full);full.with_suffix('.meta.json').write_text(json.dumps(common,indent=2)+'\n')
 for tower,prefixes in [('user',('u_emb_','user_mlp','user_numeric.','seq_input_proj','seq_gru','seq_attn')),('item',('i_emb_','item_mlp','item_numeric.'))]:
  partial={k:v for k,v in state.items() if k.startswith(prefixes) or k in {*PCA_KEYS,'seq_mask_padding','user_mask_profile'}};torch.save({'state_dict':partial,'meta':{**common,'tower':tower}},destination/f'dssm_rec_residual_{tower}_tower.pt')
 return common

def encode_request_frame(model,frame,stores,meta,batch_size=1):
 """Canonical CPU single-request use by default; model and caller own device/TF32 flags."""
 req=frame.drop_duplicates('request_idx').sort_values('request_idx');device=next(model.parameters()).device;known=set(map(int,meta['known_user_ids']));outputs=[]
 with torch.no_grad():
  for first in range(0,len(req),batch_size):
   part=req.iloc[first:first+batch_size];batch={col:torch.tensor(part[col].to_numpy(np.int64),device=device) for col in ['user_idx','gender_enc','platform_enc','age_enc','location_enc']};batch['user_idx']=torch.tensor([int(u) if int(u) in known else int(meta['unknown_user_id']) for u in part.user_idx],device=device);batch['fans_num']=torch.tensor(part.fans_num.to_numpy(np.float32),device=device);batch['follows_num']=torch.tensor(part.follows_num.to_numpy(np.float32),device=device);batch['dense_feats']=torch.tensor(part[[f'dense_feat{i}' for i in range(1,41)]].to_numpy(np.float32),device=device);histories=part.recent_clicked_note_idxs.tolist();summary=torch.stack([trainer.build_history_summary_from_ids(ids,stores) for ids in histories]).to(device);seq=[trainer.build_history_sequence_from_ids(ids,stores) for ids in histories];tokens=torch.stack([v[0] for v in seq]).to(device);masks=torch.stack([v[1] for v in seq]).to(device)
   if isinstance(model,ResidualDSSMModel):batch['residual_history_text']=torch.from_numpy(np.stack([raw_history_text_mean(ids,stores['note_text']) for ids in histories])).to(device)
   outputs.append(model.forward_user(batch,summary,seq_tokens=tokens,seq_mask=masks).float().cpu().numpy())
 return req.request_idx.to_numpy(np.int64),np.concatenate(outputs)
