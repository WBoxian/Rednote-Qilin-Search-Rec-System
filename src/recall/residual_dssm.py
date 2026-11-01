"""Versioned DSSM residual content features and the shared checkpoint/query contract."""
from __future__ import annotations
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from recall import dssm_trainer as trainer
from training.history import recent_history_ids
MODEL_FAMILY='residual_dssm_v1'
CONTENT_MODEL_FAMILY='adapted_residual_dssm_v1'
RESIDUAL_FAMILIES=frozenset((MODEL_FAMILY,CONTENT_MODEL_FAMILY))
CONTENT_KEYS=('content_beta','content_version')
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
 def project_user_content(self,values):
  return project_content(values,self.residual_pca_mean,self.residual_pca_components)
 def project_item_content(self,values):
  return project_content(values,self.residual_pca_mean,self.residual_pca_components)
 def forward_user(self,batch_data,history_vecs,seq_tokens=None,seq_mask=None):
  if 'residual_history_text' not in batch_data:raise ValueError('Residual user input requires raw recent20 mean/L2 text vector')
  learned=super().forward_user(batch_data,history_vecs,seq_tokens=seq_tokens,seq_mask=seq_mask);content=self.project_user_content(batch_data['residual_history_text'])
  return self.join(learned,content)
 def forward_item(self,batch_data,text_vecs,img_vecs):
  learned=super().forward_item(batch_data,text_vecs,img_vecs);content=self.project_item_content(text_vecs)
  return self.join(learned,content)

class AdaptedResidualDSSMModel(ResidualDSSMModel):
    """Frozen original tower plus trained query/item content residuals."""

    model_family = CONTENT_MODEL_FAMILY

    def __init__(self, *args, content_input_dim=128, content_beta=.5, **kwargs):
        if content_input_dim not in (128, 768) or content_beta != .5:
            raise ValueError("Unsupported content adapter architecture")
        super().__init__(*args, **kwargs)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.content_input_dim = content_input_dim
        self.register_buffer("content_beta", torch.tensor(content_beta, dtype=torch.float32))
        self.register_buffer("content_version", torch.tensor(1, dtype=torch.int64))
        for name in ("content_query_adapter", "content_item_adapter"):
            adapter = torch.nn.Sequential(
                torch.nn.Linear(content_input_dim, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 128),
            )
            torch.nn.init.zeros_(adapter[-1].weight)
            torch.nn.init.zeros_(adapter[-1].bias)
            setattr(self, name, adapter)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        for name, module in self.named_children():
            if name not in {"content_query_adapter", "content_item_adapter"}:
                module.eval()
        return self

    def _adapt_content(self, values, base, adapter):
        features = base if self.content_input_dim == 128 else F.normalize(values.float(), dim=1)
        delta = adapter(features)
        projected = F.normalize(base + self.content_beta * delta, dim=1)
        valid_source = base if self.content_input_dim == 128 else values.float()
        valid = valid_source.norm(dim=1, keepdim=True) > 0
        projected = torch.where(valid, projected, torch.zeros_like(projected))
        if not self.training:
            # Preserve the original zero-head baseline without blocking SGD gradients.
            projected = torch.where(valid & (delta == 0).all(dim=1, keepdim=True), base, projected)
        return projected

    def project_user_content(self, values):
        base = super().project_user_content(values)
        return self._adapt_content(values, base, self.content_query_adapter)

    def project_item_content(self, values):
        base = super().project_item_content(values)
        return self._adapt_content(values, base, self.content_item_adapter)

    def content_metadata(self):
        return {"content_input_dim": self.content_input_dim, "content_hidden_dim": 256,
                "content_output_dim": 128, "content_beta": float(self.content_beta), "content_version": 1}


def validate_content_state(meta, state):
    if meta.get("model_family") != CONTENT_MODEL_FAMILY:
        return
    for name, value in state.items():
        if torch.is_tensor(value) and (value.is_floating_point() or value.is_complex()):
            if value.is_complex() or not torch.isfinite(value).all():
                raise ValueError("Nonfinite or complex provided model tensor: " + name)
    residual_version = state["residual_version"]
    if residual_version.dtype != torch.int64 or int(residual_version) != 1:
        raise ValueError("Adapted residual version must be the exact integer contract")
    input_dim = meta.get("content_input_dim")
    if input_dim not in (128, 768) or (meta.get("content_hidden_dim"), meta.get("content_output_dim"),
                                      meta.get("content_beta"), meta.get("content_version")) != (256, 128, .5, 1):
        raise ValueError("Invalid content adapter metadata")
    if any(key not in state or not torch.is_tensor(state[key]) for key in CONTENT_KEYS):
        raise ValueError("Missing persistent content adapter contract")
    beta, version = (state[key] for key in CONTENT_KEYS)
    if beta.numel() != 1 or version.numel() != 1 or not torch.isfinite(beta).all() or not torch.isfinite(version).all():
        raise ValueError("Malformed content adapter buffers")
    if version.dtype != torch.int64 or type(meta["content_version"]) is not int:
        raise ValueError("Content adapter version must be an integer buffer and metadata")
    if float(beta) != meta["content_beta"] or int(version) != meta["content_version"]:
        raise ValueError("Content adapter buffer and metadata mismatch")
    tower = meta.get("tower")
    prefixes = {"user": ("content_query_adapter.",), "item": ("content_item_adapter.",)}
    required = prefixes.get(tower, ("content_query_adapter.", "content_item_adapter."))
    shapes = {"0.weight": (256, input_dim), "0.bias": (256,), "2.weight": (128, 256), "2.bias": (128,)}
    for prefix in ("content_query_adapter.", "content_item_adapter."):
        for suffix, shape in shapes.items():
            key = prefix + suffix
            if key not in state:
                if prefix in required:
                    raise ValueError("Missing active content adapter tensor: " + key)
                continue
            value = state[key]
            if not torch.is_tensor(value) or tuple(value.shape) != shape:
                raise ValueError("Invalid provided content adapter tensor: " + key)


def validate_residual_state(meta,state):
 if meta.get('model_family') not in RESIDUAL_FAMILIES:return
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
 if family not in {'dssm_v1',*RESIDUAL_FAMILIES}:raise ValueError('Unsupported DSSM model family')
 kwargs={'scene':meta.get('scene','rec'),'cat_vocabs':meta['cat_vocabs'],'item_dense_dim':int(meta['item_dense_dim']),'user_dense_dim':int(meta['user_dense_dim']),'preprocessing':meta.get('numeric_preprocessing')}
 if family in RESIDUAL_FAMILIES:
  validate_residual_state(meta,state);validate_content_state(meta,state)
  residual_kwargs={'pca_mean':state[PCA_KEYS[0]],'pca_components':state[PCA_KEYS[1]],'residual_weight':float(state[PCA_KEYS[2]])}
  if family==CONTENT_MODEL_FAMILY:model=AdaptedResidualDSSMModel(**kwargs,**residual_kwargs,content_input_dim=meta['content_input_dim'],content_beta=meta['content_beta'])
  else:model=ResidualDSSMModel(**kwargs,**residual_kwargs)
 else:model=trainer.DSSMModel(**kwargs)
 missing,unexpected=model.load_state_dict(state,strict=False)
 if unexpected:raise ValueError('Unexpected DSSM checkpoint keys: '+str(unexpected))
 if missing:
  tower=meta.get('tower')
  if tower is None and family=='dssm_v1':
   tower='user' if any(k.startswith('u_emb_') for k in state) and not any(k.startswith('i_emb_') for k in state) else ('item' if any(k.startswith('i_emb_') for k in state) and not any(k.startswith('u_emb_') for k in state) else None)
  if not allow_partial or tower not in {'user','item'}:raise ValueError('Incomplete DSSM checkpoint')
  opposite=('i_emb_','item_mlp','item_numeric.') if tower=='user' else ('u_emb_','user_mlp','user_numeric.','seq_input_proj','seq_gru','seq_attn')
  if family==CONTENT_MODEL_FAMILY:opposite+=('content_item_adapter.',) if tower=='user' else ('content_query_adapter.',)
  if any(not name.startswith(opposite) for name in missing):raise ValueError('Partial checkpoint misses active tower or persistent contract')
 model.checkpoint_meta=meta;model.checkpoint_is_partial=bool(missing);model.checkpoint_loaded_tower=tower if missing else None;return model.to(device).float().eval()

def save_residual_checkpoints(model,meta,destination):
 if getattr(model,'checkpoint_is_partial',False):raise ValueError('Full checkpoint export requires a complete source model, not a loaded partial tower')
 destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
 state={k:v.detach().cpu() for k,v in model.state_dict().items()}
 common={**meta,'tag':'residual','model_family':model.model_family,'base_hidden_dim':128,'retrieval_dim':256,'pca_input_dim':768,'pca_dim':128,'residual_weight':float(model.residual_weight)}
 common.pop('tower',None)
 shared_keys={*PCA_KEYS,'seq_mask_padding','user_mask_profile'}
 if isinstance(model,AdaptedResidualDSSMModel):
  common.update(model.content_metadata());shared_keys.update(CONTENT_KEYS)
 validate_residual_state(common,state);validate_content_state(common,state)
 full=destination/'dssm_rec_residual.pt';torch.save(state,full)
 if isinstance(model,AdaptedResidualDSSMModel):
  digest=hashlib.sha256()
  with full.open('rb') as stream:
   for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
  common['exported_full_checkpoint_sha256']=digest.hexdigest()
 full.with_suffix('.meta.json').write_text(json.dumps(common,indent=2)+'\n')
 for tower,prefixes in [('user',('u_emb_','user_mlp','user_numeric.','seq_input_proj','seq_gru','seq_attn')),('item',('i_emb_','item_mlp','item_numeric.'))]:
  if isinstance(model,AdaptedResidualDSSMModel):prefixes+=('content_query_adapter.',) if tower=='user' else ('content_item_adapter.',)
  partial={k:v for k,v in state.items() if k.startswith(prefixes) or k in shared_keys}
  torch.save({'state_dict':partial,'meta':{**common,'tower':tower}},destination/f'dssm_rec_residual_{tower}_tower.pt')
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
