"""Frozen train-only Rec iteration: continued-training control, false-negative masking, logged hard negatives."""
from __future__ import annotations
import argparse,json,shutil,time
from pathlib import Path
import numpy as np,pandas as pd
from scripts import optimize_rec_retrieval as rec
from scripts import optimize_rec_northstar as north
ROOT=Path(__file__).resolve().parents[1];OLD=ROOT/'outputs/evaluation/northstar_fix_20261031/rec_v3'
VARIANTS={'continuation_control':{'mask_known':False,'hard_weight':0.},'masked_inbatch':{'mask_known':True,'hard_weight':0.},'masked_logged_hard':{'mask_known':True,'hard_weight':.5}}


NUMERIC_CONTRACT={'cuda_matmul_allow_tf32':False,'cudnn_allow_tf32':True,'cudnn_benchmark':True,'training_autocast':'CUDA float16','validation_export_autocast':False,'validation_export_tensor_dtype':'float32','lineage':'Explicitly matches original V3 train_dssm after trainer import; cuDNN TF32 allowed, not strict-all-TF32-off'}
def configure_numerics():
 import torch
 torch.backends.cuda.matmul.allow_tf32=NUMERIC_CONTRACT['cuda_matmul_allow_tf32']
 torch.backends.cudnn.allow_tf32=NUMERIC_CONTRACT['cudnn_allow_tf32']
 torch.backends.cudnn.benchmark=NUMERIC_CONTRACT['cudnn_benchmark']


def contrastive_loss(users,items,item_ids,request_ids,user_ids,history_ids,mask_known=False,temperature=.07):
 import torch
 logits=(users@items.T).float()/temperature
 positive=(item_ids[:,None]==item_ids[None,:])|(request_ids[:,None]==request_ids[None,:])
 valid=torch.ones_like(positive)
 if mask_known:
  sameuser=user_ids[:,None]==user_ids[None,:]
  history=(history_ids[:,:,None]==item_ids[None,None,:]).any(1)
  valid=~((sameuser|history)&~positive)
 return (torch.logsumexp(logits.masked_fill(~valid,-torch.inf),1)-torch.logsumexp(logits.masked_fill(~positive,-torch.inf),1)).mean()

def logged_pairwise_loss(users,positive,negative,valid,temperature=.07):
 import torch
 if valid.ndim!=1 or len(valid)!=len(users):raise ValueError('Hard-negative validity must align with users')
 if not valid.any():return users.sum()*0.
 pos=(users*positive).sum(1).float();neg=(users*negative).sum(1).float()
 return torch.nn.functional.softplus((neg[valid]-pos[valid])/temperature).mean()

def safe_negative_lists(frame):
 """Only same-request truly unclicked pairs; remove any clicked duplicate/history-known ID."""
 positive=frame[frame.click>0].groupby('request_idx').note_idx.apply(lambda x:set(map(int,x))).to_dict()
 negative=frame[frame.click==0].groupby('request_idx').note_idx.apply(lambda x:set(map(int,x))).to_dict()
 histories=frame.drop_duplicates('request_idx').set_index('request_idx').recent_clicked_note_idxs
 result={}
 for gid,ids in negative.items():
  history=histories.loc[gid];known=set(map(int,history)) if isinstance(history,(list,np.ndarray)) else set()
  result[int(gid)]=sorted(ids-positive.get(gid,set())-known)
 return result


class LoggedNegativeDataset:
 """Sample the complete verified request pool, without legacy top10 truncation."""
 def __init__(self,base,pools):
  if len(base)!=len(pools):raise ValueError('Negative pools must align with positive anchors')
  self.base=base;self.pools=pools
 def __len__(self):return len(self.base)
 def __getitem__(self,index):
  user,positive,negative,kinds,label=self.base[index];pool=self.pools[index]
  if pool:
   item=int(pool[np.random.randint(len(pool))]);negative=[self.base._fetch_item(item)];kinds=np.array([1],np.int64)
  return user,positive,negative,kinds,label


def verify_plan(out):
 plan=json.loads((out/'plan.json').read_text())
 for path,expected in {**plan['inputs_sha256'],**plan['code_sha256']}.items():
  if rec.digest(Path(path))!=expected:raise ValueError('Frozen experiment input/source changed: '+path)
 return plan

def plan(out):
 if (out/'plan.json').exists():raise FileExistsError('Experiment plan already frozen')
 diagnosis=json.loads((out/'diagnosis.json').read_text());source=ROOT/'features/rec_train_features.parquet';meta=OLD/'dssm/training_manifest.json';selected=json.loads((OLD/'dssm/selected.json').read_text());epoch=Path(selected['directory'])
 paths=[source,meta,epoch/'dssm_rec_optimized.pt',epoch/'item_vectors.bin',OLD/'dssm/item_ids.npy',OLD/'requests_fit.parquet',OLD/'requests_validation.parquet',OLD/'labels_fit.parquet',OLD/'labels_validation.parquet',OLD/'validation_selected.parquet',out/'diagnosis.json']
 paths.extend(sorted((ROOT/'datasets/notes').glob('*.parquet')));paths.extend(sorted((ROOT/'features/vocab_dict').glob('*.pkl')))
 for folder,prefix in [('note_text_emb','note_text'),('note_img_emb','note_img')]:paths.extend([ROOT/'embeddings'/folder/(prefix+'.bin'),ROOT/'embeddings'/folder/(prefix+'_map.json')])
 for graphdir in [OLD/'cf_graph_0.25',OLD/'refit_cf_pretest']:
  paths.extend(graphdir/name for name in ['graph.npz','users.npy','items.npy','swing.npz'])
 paths.extend(OLD/'validation/hist025_fold200'/name for name in ['swing.parquet','usercf.parquet'])
 code=[Path(__file__),ROOT/'scripts/diagnose_rec_iteration.py',ROOT/'scripts/optimize_rec_retrieval.py',ROOT/'scripts/optimize_rec_northstar.py',ROOT/'src/recall/dssm_trainer.py',ROOT/'src/training/history.py',ROOT/'src/backend/online/item_features.py',ROOT/'tests/test_rec_iteration.py']
 result={'scene':'rec','variants':VARIANTS,'initial_checkpoint':str((epoch/'dssm_rec_optimized.pt').resolve()),'initial_epoch':3,'additional_epochs':[4,5,6],
 'training':'Exactly original-train V3 fit55616 requests/197125 positivepairs; no validation/test gradients; observed unclicked negatives only for C',
 'numeric_contract':NUMERIC_CONTRACT,'batch_size':256,'seed':42,'learning_rate':.0001,'weight_decay':.01,'optimizer':'AdamW RESET for ALL variants; continuation control isolates optimizer restart/additional training','temperature':.07,
 'mask_semantics':'Exclude same-user other-request/current-history recent20 known-positive candidates only from denominator; retain V2 same-item/request positives; do not promote every same-user candidate to positive',
 'hard_negative':'One uniformly sampled same-request observed-unclicked item per positive, remove any clicked duplicate/currenthistory known positive; additional .5*softplus((negIP-posIP)/.07); not a CTR target for unexposed catalog items',
 'epoch_selection':'Exact fullcatalog validation maxRecall100 among epochs4/5/6 with HR50 and KP-NDCG10 >= originalV3 exactANN baseline; if none no deployable checkpoint',
 'index':'Per eligible variant selected epoch only, same fit-exposure sample100000 seed42, IVFPQ2048/m32/bits8/IP, fixednprobe32; no probegrid',
 'fusion':'Fixed V3 hist025_fold200 fitgraph, route1000/final500, ANN:Swing:UserCF=1:2:1,k60; no weightgrid',
 'release_selection':'Strict fullfusion validation Recall100 gain over originalV3, with HR50 and KP-NDCG10 nonregression; maximizeR100 thenHR50 thenKP; include unchanged originalV3 incumbent. Report same-data algorithm B/C vs continued-control A separately, with all paired user CIs, no significance claim if crosses0.',
 'official_test':'NO test until plan/selection/model/source hashes frozen and root explicitly approves; historicaltest is regression notnewblindholdout; selected recipe only plus predeclared continuationcontrol diagnostic ifapproved',
 'inputs_sha256':{str(p.resolve()):rec.digest(p) for p in paths},'code_sha256':{str(p.resolve()):rec.digest(p) for p in code},'created_before_training':True}
 (out/'source_snapshot').mkdir(exist_ok=True)
 for p in code:shutil.copyfile(p,out/'source_snapshot'/(p.parent.name+'_'+p.name))
 rec.write_json(out/'plan.json',result);print(json.dumps({'plan':str(out/'plan.json'),'variants':VARIANTS},indent=2));return result

def load_frames():
 from backend.online.item_features import get_item_feature_store
 from recall import dssm_trainer as trainer
 raw=pd.read_parquet(ROOT/'features/rec_train_features.parquet');fitids=set(pd.read_parquet(OLD/'requests_fit.parquet').gid);valids=set(pd.read_parquet(OLD/'requests_validation.parquet').gid)
 fit=rec.raw_model_frame(raw[raw.request_idx.isin(fitids)],get_item_feature_store(ROOT),trainer,'rec');val=rec.raw_model_frame(raw[raw.request_idx.isin(valids)],get_item_feature_store(ROOT),trainer,'rec')
 for target in [fit,val]:target[['fans_num','follows_num']+[f'dense_feat{i}' for i in range(1,41)]]=0.
 assert fit.request_idx.nunique()==55616 and val.request_idx.nunique()==11893 and fit.event_timestamp.max()<val.event_timestamp.min()
 return fit,val

def train(out,variant,device='cuda'):
 import torch
 from torch.utils.data import DataLoader
 from recall import dssm_trainer as trainer
 from backend.online.item_features import get_item_feature_store
 declared=verify_plan(out);directory=out/variant
 if directory.exists():raise FileExistsError('Variant training already started; retain failed evidence or use new experiment')
 directory.mkdir();config=VARIANTS[variant];trainer.set_seed(42);torch.set_num_threads(2);configure_numerics();device=torch.device(device);trainer.DEVICE=device
 fit,val=load_frames();meta=json.loads((OLD/'dssm/training_manifest.json').read_text());_,cols,_=trainer.numeric_feature_columns('rec');store=get_item_feature_store(ROOT)
 model=trainer.DSSMModel('rec',meta['cat_vocabs'],len(cols),42+trainer.EMB_DIM+trainer.SEQ_TOWER_DIM,preprocessing=meta['normalizer'],mask_sequence_padding=True,mask_user_profile=True).to(device)
 model.load_state_dict(torch.load(declared['initial_checkpoint'],map_location=device,weights_only=True));stores={'note_text':trainer.MMapStore('note_text_emb','note_text'),'note_img':trainer.MMapStore('note_img_emb','note_img')}
 positives=fit[fit.click>0].drop_duplicates(['request_idx','note_idx']).copy();assert len(positives)==197125
 negatives=safe_negative_lists(fit);positives['observed_unclicked_ids']=positives.request_idx.map(negatives).apply(lambda x:x if isinstance(x,list) else [])
 # Same item static feature contract; hard-negative supervision adds no validation items to fit.
 item_meta=trainer.build_item_meta(fit if config['hard_weight'] else positives,'rec',cols)
 dataset=trainer.DSSMDataset('rec','request_idx',positives,item_meta,list(item_meta),torch.full((len(item_meta),),1./len(item_meta)),use_inbatch_neg=True,use_hard_neg=False,hard_neg_col=None,hard_neg_ratio=1,num_neg_per_pos=1)
 if config['hard_weight']:dataset=LoggedNegativeDataset(dataset,positives.observed_unclicked_ids.tolist())
 base_collate=trainer.collate_fn_builder(stores,'rec')
 def collate(rows):
  batch=base_collate(rows);batch[0]['request_ids']=torch.tensor([row[0]['req_id'] for row in rows],dtype=torch.long)
  history=np.full((len(rows),trainer.SEQ_MAX_LEN),-1,np.int64)
  for i,row in enumerate(rows):values=row[0]['history_ids'];history[i,:len(values)]=values
  batch[0]['known_history_ids']=torch.from_numpy(history);return batch
 loader=DataLoader(dataset,batch_size=256,shuffle=True,collate_fn=collate,num_workers=2,pin_memory=device.type=='cuda',persistent_workers=True,prefetch_factor=2,generator=torch.Generator().manual_seed(42))
 optimizer=torch.optim.AdamW(model.parameters(),lr=declared['learning_rate'],weight_decay=declared['weight_decay']);scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda');itemids=np.load(OLD/'dssm/item_ids.npy');np.save(directory/'item_ids.npy',itemids);valrequests=pd.read_parquet(OLD/'requests_validation.parquet');baseline=json.loads((OLD/'dssm/epochs.json').read_text())[-1]
 manifest={**meta,'iteration_variant':variant,'iteration_config':config,'numeric_contract':NUMERIC_CONTRACT,'initial_checkpoint_sha256':rec.digest(Path(declared['initial_checkpoint'])),'plan_sha256':rec.digest(out/'plan.json'),'epochs':declared['additional_epochs'],'optimizer_reset':True,'learning_rate':declared['learning_rate'],'gradient_split':'original V3 fit only','negative_sampling':'same-request observed-unclicked minus clicked/history-known IDs' if config['hard_weight'] else 'inbatch only','source_sha256':declared['code_sha256']}
 manifest['hard_negative_support']={'usable_distinct_request_item_pairs':sum(len(ids) for ids in negatives.values()),'positive_anchors_with_valid_negative':int(positives.observed_unclicked_ids.map(len).gt(0).sum()),'positive_anchors':len(positives)}
 rec.write_json(directory/'training_manifest.json',manifest);logs=[];winner=None
 for epoch in declared['additional_epochs']:
  start=time.monotonic();configure_numerics();model.train();losses=[];iblosses=[];hardlosses=[];hardrows=0
  for step,batch in enumerate(loader):
   batch=trainer.move_batch_to_device(batch,device);user,hist,tokens,mask,pos,txt,img,neg,ntxt,nimg,ntypes,_=batch;optimizer.zero_grad(set_to_none=True)
   with torch.amp.autocast('cuda',enabled=device.type=='cuda',dtype=torch.float16):
    uv=model.forward_user(user,hist,seq_tokens=tokens,seq_mask=mask);iv=model.forward_item(pos,txt,img)
    ib=contrastive_loss(uv,iv,pos['note_idx'],user['request_ids'],user['user_idx'],user['known_history_ids'],config['mask_known'])
    hard=uv.sum()*0.
    if config['hard_weight']:
     n={name:value[:,0] for name,value in neg.items()};nv=model.forward_item(n,ntxt[:,0],nimg[:,0]);valid=ntypes[:,0]>0;hard=logged_pairwise_loss(uv,iv,nv,valid);hardrows+=int(valid.sum())
    loss=ib+config['hard_weight']*hard
   scaler.scale(loss).backward();scaler.step(optimizer);scaler.update();losses.append(float(loss.detach()));iblosses.append(float(ib.detach()));hardlosses.append(float(hard.detach()))
   if step%200==0:print(json.dumps({'variant':variant,'epoch':epoch,'batch':step,'loss':losses[-1],'inbatch':iblosses[-1],'hard':hardlosses[-1]}),flush=True)
  configure_numerics();model.float().eval();ep=directory/f'epoch_{epoch}';ep.mkdir();trainer.save_tower_checkpoints(model.state_dict(),'rec','optimized',meta['cat_vocabs'],len(cols),42+trainer.EMB_DIM+trainer.SEQ_TOWER_DIM,preprocessing=meta['normalizer'],output_dir=ep,extra_meta={'masked_user_dense':True,'loss_contract':meta['loss_contract'],'unknown_user_id':meta['unknown_user_id'],'known_user_ids':meta['known_user_ids'],'iteration_variant':variant,'initial_checkpoint_sha256':manifest['initial_checkpoint_sha256']})
  with torch.amp.autocast('cuda',enabled=False):
   vectors=rec.export_catalog(model,store,itemids,stores,trainer,'rec',ep/'item_vectors.bin');ids,queries=rec.encode_requests(model,val,stores,'rec',meta['unknown_user_id'],set(meta['known_user_ids']));np.save(ep/'validation_request_ids.npy',ids);np.save(ep/'validation_vectors.npy',queries)
   candidates=rec.exact_retrieval(ids,queries,itemids,vectors,device)
  candidates.to_parquet(ep/'validation_ann.parquet',index=False);metrics,rows=rec.metrics(valrequests,candidates);rows.to_parquet(ep/'validation_metrics.parquet',index=False)
  score={'epoch':epoch,'loss':float(np.mean(losses)),'inbatch_loss':float(np.mean(iblosses)),'hard_loss':float(np.mean(hardlosses)),'hard_rows':hardrows,'seconds':time.monotonic()-start,**metrics};score['guard_passed']=metrics['HitRate@50']>=baseline['HitRate@50'] and metrics['known_positive_NDCG@10']>=baseline['known_positive_NDCG@10'];logs.append(score);rec.write_json(directory/'epochs.json',logs)
  if score['guard_passed'] and (winner is None or (score['Recall@100'],score['HitRate@50'],score['known_positive_NDCG@10'])>(winner['validation']['Recall@100'],winner['validation']['HitRate@50'],winner['validation']['known_positive_NDCG@10'])):winner={'epoch':epoch,'directory':str(ep),'validation':score,'checkpoint_sha256':rec.digest(ep/'dssm_rec_optimized.pt'),'item_vectors_sha256':rec.digest(ep/'item_vectors.bin')}
  print(json.dumps({'variant':variant,'epoch_metrics':score}),flush=True);del vectors
 rec.write_json(directory/'selected.json',{'accepted':winner is not None,'selected':winner,'baseline_exact_ann':baseline,'no_test_read':True});return winner

def index(out,variant):
 import faiss
 verify_plan(out);directory=out/variant;choice=json.loads((directory/'selected.json').read_text())
 if not choice['accepted']:raise ValueError('No epoch passes original V3 ANN head guards')
 selected=choice['selected'];epoch=Path(selected['directory']);items=np.load(directory/'item_ids.npy');vectors=np.memmap(epoch/'item_vectors.bin',np.float32,'r',shape=(len(items),128));dest=directory/'index';dest.mkdir(exist_ok=False);fitids=np.unique(pd.read_parquet(OLD/'labels_fit.parquet',columns=['note_idx']).note_idx);pos=np.searchsorted(items,fitids);assert np.array_equal(items[pos],fitids);sampled=np.random.default_rng(42).choice(pos,min(100000,len(pos)),replace=False);faiss.omp_set_num_threads(2)
 idx=faiss.IndexIVFPQ(faiss.IndexFlatIP(128),128,2048,32,8,faiss.METRIC_INNER_PRODUCT);idx.cp.seed=42;idx.train(np.ascontiguousarray(vectors[sampled]))
 for start in range(0,len(items),65536):idx.add(np.ascontiguousarray(vectors[start:start+65536]))
 idx.nprobe=32;faiss.write_index(idx,str(dest/'ivfpq.faiss'));ids=np.load(epoch/'validation_request_ids.npy');queries=np.load(epoch/'validation_vectors.npy');rec.retrieve_index(idx,ids,queries,items,dest/'validation_ann.parquet');req=pd.read_parquet(OLD/'requests_validation.parquet');cf=OLD/'validation/hist025_fold200';recipe={'weights':[1,2,1],'rrf_k':60};north.write_candidates(req,[dest/'validation_ann.parquet',cf/'swing.parquet',cf/'usercf.parquet'],recipe,directory/'validation_selected.parquet');score,rows=rec.metrics(req,pd.read_parquet(directory/'validation_selected.parquet'));rows.to_parquet(directory/'validation_metrics.parquet',index=False)
 rec.write_json(dest/'build.json',{'metric':'inner_product','algorithm':'IVFPQ','nlist':2048,'m':32,'bits':8,'nprobe':32,'items':len(items),'fit_exposure_items':len(fitids),'training_vector_rows':len(sampled),'selected_checkpoint_sha256':selected['checkpoint_sha256'],'selected_item_vectors_sha256':selected['item_vectors_sha256'],'index_sha256':rec.digest(dest/'ivfpq.faiss')});rec.write_json(directory/'fusion_validation.json',score);return score

def select(out):
 from backend.offline.evaluate_full_test import confidence
 plan=verify_plan(out)
 if (out/'selection.json').exists():raise FileExistsError('New selection already frozen')
 requests=pd.read_parquet(OLD/'requests_validation.parquet');baseline,baserows=rec.metrics(requests,pd.read_parquet(OLD/'validation_selected.parquet'));trials=[]
 for name in VARIANTS:
  p=out/name/'fusion_validation.json'
  if not p.exists():continue
  score=json.loads(p.read_text());rows=pd.read_parquet(out/name/'validation_metrics.parquet');assert np.array_equal(rows.gid,baserows.gid);accepted=score['Recall@100']>baseline['Recall@100'] and score['HitRate@50']>=baseline['HitRate@50'] and score['known_positive_NDCG@10']>=baseline['known_positive_NDCG@10'];delta={}
  for metric,col in [('HitRate@50','hit50'),('Recall@100','recall100'),('known_positive_NDCG@10','ndcg10'),('MRR@100','mrr100')]:
   table=rows[['gid','user_idx']].copy();table['delta']=rows[col]-baserows[col];delta[metric]={'delta':float(table.delta.mean()),'CI95':confidence(table,'delta')}
  trials.append({'variant':name,'metrics':score,'guard_passed':accepted,'paired_vs_v3':delta})
 eligible=[r for r in trials if r['guard_passed']];winner=max(eligible,key=lambda r:(r['metrics']['Recall@100'],r['metrics']['HitRate@50'],r['metrics']['known_positive_NDCG@10'])) if eligible else None
 output={'accepted':winner is not None,'selected':winner,'baseline_v3':baseline,'trials':trials,'plan_sha256':rec.digest(out/'plan.json'),'frozen_before_official_test':True,'inputs_sha256':plan['inputs_sha256'],'code_sha256':plan['code_sha256'],'models':{name:{'selection_sha256':rec.digest(out/name/'selected.json'),'build_sha256':rec.digest(out/name/'index/build.json')} for name in VARIANTS if (out/name/'index/build.json').exists()},'fulltest':'ROOT_APPROVAL_REQUIRED'};rec.write_json(out/'selection.json',output);print(json.dumps(output,indent=2));return output

def full_test(out,include_control=False,device='cpu'):
 """Read frozen selected recipes only; caller must obtain root approval before invoking."""
 import torch,faiss
 from scipy import sparse
 from recall import dssm_trainer as trainer
 from backend.online.item_features import get_item_feature_store
 from backend.offline.evaluate_full_test import confidence
 declared=verify_plan(out);selection=json.loads((out/'selection.json').read_text())
 if not selection['accepted']:raise ValueError('No accepted validation recipe; never open official test')
 if rec.digest(out/'plan.json')!=selection['plan_sha256']:raise ValueError('Frozen plan changed')
 for name,bound in selection['models'].items():
  for file,key in [('selected.json','selection_sha256'),('index/build.json','build_sha256')]:
   if rec.digest(out/name/file)!=bound[key]:raise ValueError('Frozen selected model/index metadata changed')
 destination=out/'test';destination.mkdir(exist_ok=False)
 release=ROOT/'outputs/evaluation/northstar_fix_20261031/release_plan.json';release_sha=rec.digest(release) if release.exists() else None
 rawpath=ROOT/'datasets/recommendation_test/train-00000-of-00001.parquet';raw_sha=rec.digest(rawpath);expected=json.loads((OLD/'manifest.json').read_text())['official_test_sha256']
 if raw_sha!=expected:raise ValueError('Official source changed')
 raw=pd.read_parquet(rawpath,columns=['request_idx','user_idx','recent_clicked_note_idxs','rec_result_details_with_idx'])
 requests=pd.DataFrame({'gid':raw.request_idx,'user_idx':raw.user_idx,'history':raw.recent_clicked_note_idxs,'positives':raw.rec_result_details_with_idx.apply(lambda a:sorted({int(v['note_idx']) for v in a if v['click']>0}))})
 assert len(requests)==11115 and requests.gid.is_unique and sum(requests.positives.map(len))==44683
 requests.to_parquet(destination/'requests.parquet',index=False);device=torch.device(device);trainer.DEVICE=device;torch.set_num_threads(2);configure_numerics();faiss.omp_set_num_threads(2)
 frame=rec.raw_model_frame(pd.read_parquet(ROOT/'features/rec_test_features.parquet'),get_item_feature_store(ROOT),trainer,'rec');frame[['fans_num','follows_num']+[f'dense_feat{i}' for i in range(1,41)]]=0.
 names=[selection['selected']['variant']]
 if include_control and 'continuation_control' not in names:
  if not (out/'continuation_control/index/build.json').exists():raise ValueError('Control index not frozen')
  names.append('continuation_control')
 result={'selection_sha256':rec.digest(out/'selection.json'),'release_plan_sha256':release_sha,'official_test_source_sha256':raw_sha,'all_requests':len(requests),'unique_clicked_pairs':44683,'parameter_selection_after_test':False,'scope':'Frozen regression benchmark; fit-only algorithm change and additional CF historical-data effect separated; no new test recipe choices','variants':{}}
 for name in names:
  path=out/name;meta=json.loads((path/'training_manifest.json').read_text());selected=json.loads((path/'selected.json').read_text())['selected'];epoch=Path(selected['directory']);checkpoint=epoch/'dssm_rec_optimized.pt';build=json.loads((path/'index/build.json').read_text())
  if rec.digest(checkpoint)!=selected['checkpoint_sha256'] or rec.digest(path/'index/ivfpq.faiss')!=build['index_sha256']:raise ValueError('Model/index bytes changed')
  _,cols,_=trainer.numeric_feature_columns('rec');model=trainer.DSSMModel('rec',meta['cat_vocabs'],len(cols),42+trainer.EMB_DIM+trainer.SEQ_TOWER_DIM,preprocessing=meta['normalizer']).to(device);model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True));model.eval();stores={'note_text':trainer.MMapStore('note_text_emb','note_text'),'note_img':trainer.MMapStore('note_img_emb','note_img')};ids,queries=rec.encode_requests(model,frame,stores,'rec',meta['unknown_user_id'],set(meta['known_user_ids']));index=faiss.read_index(str(path/'index/ivfpq.faiss'));index.nprobe=32;items=np.load(path/'item_ids.npy');dest=destination/name;dest.mkdir();np.save(dest/'request_ids.npy',ids);np.save(dest/'query_vectors.npy',queries);rec.retrieve_index(index,ids,queries,items,dest/'ann.parquet');variant={};paired={}
  for graphname,graphpath,baselinepath in [('fit',OLD/'cf_graph_0.25',OLD/'test/selected.parquet'),('allpast',OLD/'refit_cf_pretest',OLD/'refit_cf_pretest/selected.parquet')]:
   target=dest/graphname;target.mkdir();graph=sparse.load_npz(graphpath/'graph.npz');users=np.load(graphpath/'users.npy');graphitems=np.load(graphpath/'items.npy');swing=sparse.load_npz(graphpath/'swing.npz');rec.query_cf(requests,graph,users,graphitems,swing,{'fold_in':True,'neighbors':200},target,target/'support.parquet');north.write_candidates(requests,[dest/'ann.parquet',target/'swing.parquet',target/'usercf.parquet'],{'weights':[1,2,1],'rrf_k':60},target/'selected.parquet');metrics,rows=rec.metrics(requests,pd.read_parquet(target/'selected.parquet'));rows.to_parquet(target/'metrics.parquet',index=False);base,baserows=rec.metrics(requests,pd.read_parquet(baselinepath));delta={}
   for metric,col in [('HitRate@50','hit50'),('Recall@100','recall100'),('known_positive_NDCG@10','ndcg10')]:
    table=rows[['gid','user_idx']].copy();table['delta']=rows[col]-baserows[col];delta[metric]={'delta':float(table.delta.mean()),'CI95':confidence(table,'delta')}
   variant[graphname]={'metrics':metrics,'baseline_v3':base,'paired_vs_v3':delta,'candidate_sha256':rec.digest(target/'selected.parquet')};paired[graphname]=rows
  volume={}
  for metric,col in [('HitRate@50','hit50'),('Recall@100','recall100'),('known_positive_NDCG@10','ndcg10')]:
   table=paired['fit'][['gid','user_idx']].copy();table['delta']=paired['allpast'][col]-paired['fit'][col];volume[metric]={'delta':float(table.delta.mean()),'CI95':confidence(table,'delta')}
  variant['allpast_minus_fit_data_volume']=volume;result['variants'][name]=variant
 selected=result['variants'][selection['selected']['variant']]['allpast'];score,base=selected['metrics'],selected['baseline_v3'];result['accepted_for_release']=score['Recall@100']>base['Recall@100'] and score['HitRate@50']>=base['HitRate@50'] and score['known_positive_NDCG@10']>=base['known_positive_NDCG@10'];result['release_fallback']=None if result['accepted_for_release'] else 'unchanged_v3_recall_only';rec.write_json(destination/'metrics.json',result);print(json.dumps(result,indent=2));return result

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=['plan','train','index','select','test'],required=True);p.add_argument('--output',type=Path,default=ROOT/'outputs/evaluation/northstar_fix_20261031/rec_iteration_20261031');p.add_argument('--variant',choices=list(VARIANTS));p.add_argument('--device',default='cuda',choices=['cpu','cuda']);p.add_argument('--include-control',action='store_true');a=p.parse_args()
 if a.phase in ['train','index'] and a.variant is None:p.error('variant required')
 if a.phase=='plan':plan(a.output)
 elif a.phase=='train':train(a.output,a.variant,a.device)
 elif a.phase=='index':index(a.output,a.variant)
 elif a.phase=='select':select(a.output)
 else:full_test(a.output,a.include_control,a.device)
if __name__=='__main__':main()
