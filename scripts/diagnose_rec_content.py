"""Frozen validation-only raw-BGE-history content diagnostic, never a new serving route."""
from pathlib import Path
import json,time
import numpy as np,pandas as pd,pyarrow as pa,pyarrow.parquet as pq,torch
from recall.dssm_trainer import MMapStore
from training.history import recent_history_ids
from scripts import optimize_rec_retrieval as rec
from scripts.rec_iteration_fusion import replay
ROOT=Path(__file__).resolve().parents[1];OLD=ROOT/'outputs/evaluation/northstar_fix_20261031/rec_v3';OUT=ROOT/'outputs/evaluation/northstar_fix_20261031/rec_iteration_20261031/content_only_validation'
if OUT.exists():raise FileExistsError('Content diagnostic is immutable')
OUT.mkdir();sources=[Path(__file__),ROOT/'src/training/history.py',ROOT/'src/recall/context_fusion.py',ROOT/'scripts/rec_iteration_fusion.py',ROOT/'embeddings/note_text_emb/note_text.bin',ROOT/'embeddings/note_text_emb/note_text_map.json',OLD/'requests_validation.parquet',OLD/'dssm/item_ids.npy',OLD/'dssm/epoch_3/validation_ann.parquet'];plan={'scope':'Only original-train internal validation; rawprovided text embeddings/current recent20 histories, no supervision fit/newroute/publishing/testdata','catalog':1983938,'requests':11893,'query':'L2-normalized arithmetic mean of available NONZERO raw BGE text vectors of shared recent_history_ids(history,20), preserve duplicates/note0','items':'Same fullcatalog raw BGE text vectors, L2-normalized peritem; cosine exactIP','empty_history':'No content candidate; zero metrics on fullvalidation. Additionally compare both models restricted to sameavailable-history requestsubset. No arbitrary zero-query topIDs.','models':['raw_bge_history_mean','original_v3_exact_dssm'],'policies':['raw_first500','fixed_F1_history_filter_first1000_thencompress500'],'precision':'float32 tensors, CUDA matmul andcudnn TF32False,autocastFalse; providedDSSM exact baseline keptunchanged underitsoldnumericcontract','topk_retrieval':1000,'source_sha256':{str(p.resolve()):rec.digest(p) for p in sources},'deployment_eligible':False};rec.write_json(OUT/'diagnostic_plan.json',plan)
requests=pd.read_parquet(OLD/'requests_validation.parquet').sort_values('gid').reset_index(drop=True);assert len(requests)==11893;store=MMapStore('note_text_emb','note_text');itemids=np.load(OLD/'dssm/item_ids.npy');assert len(itemids)==1983938;positions=np.array([store.id2pos[str(int(i))] for i in itemids],np.int64);queries=np.zeros((len(requests),768),np.float32);available=np.zeros(len(requests),bool);histcounts=[]
for i,row in enumerate(requests.itertuples()):
 ids=recent_history_ids(row.history,20);vec=store.get_batch_vecs(ids).numpy();valid=np.linalg.norm(vec,axis=1)>0 if len(vec) else np.zeros(0,bool);histcounts.append(int(valid.sum()))
 if valid.any():mean=vec[valid].mean(0);norm=np.linalg.norm(mean);queries[i]=mean/max(norm,1e-12);available[i]=norm>0
np.save(OUT/'request_ids.npy',requests.gid.to_numpy(np.int64));np.save(OUT/'queries.npy',queries);pd.DataFrame({'gid':requests.gid,'available_history':available,'available_history_items':histcounts}).to_parquet(OUT/'history_support.parquet',index=False)
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;device=torch.device('cuda');start=time.monotonic();catalog=torch.empty((len(itemids),768),device=device,dtype=torch.float32)
with torch.no_grad(),torch.amp.autocast('cuda',enabled=False):
 for first in range(0,len(itemids),32768):
  raw=torch.from_numpy(np.asarray(store.data[positions[first:first+32768]],np.float32).copy()).to(device);catalog[first:first+len(raw)]=torch.nn.functional.normalize(raw,dim=1)
 schema=pa.schema([('gid',pa.int64()),('note_idx',pa.int64()),('score',pa.float32()),('rank',pa.int64())])
 with pq.ParquetWriter(OUT/'content_ann.parquet',schema) as writer:
  validrows=np.flatnonzero(available)
  for first in range(0,len(validrows),128):
   rows=validrows[first:first+128];q=torch.from_numpy(queries[rows]).to(device);scores=q@catalog.T;values,indices=torch.topk(scores,1000,dim=1,sorted=True);v=values.cpu().numpy();notes=itemids[indices.cpu().numpy()];order=np.lexsort((notes,-v),axis=1);notes=np.take_along_axis(notes,order,1);v=np.take_along_axis(v,order,1);frame=pd.DataFrame({'gid':np.repeat(requests.gid.to_numpy(np.int64)[rows],1000),'note_idx':notes.ravel(),'score':v.ravel(),'rank':np.tile(np.arange(1,1001),len(rows))});writer.write_table(pa.Table.from_pandas(frame,schema=schema,preserve_index=False));del scores
   if first%2048==0:print(json.dumps({'stage':'content_exactGPU','requests_complete':min(first+128,len(validrows)),'total':len(validrows)}),flush=True)
gpu_seconds=time.monotonic()-start;del catalog,raw,q,values,indices;torch.cuda.empty_cache();print(json.dumps({'stage':'GPU_RELEASED','gpu_seconds':gpu_seconds,'available_history_requests':int(available.sum()),'nohistory_requests':int((~available).sum())}),flush=True)
results={};frames={}
for model,path in [('raw_bge_history_mean',OUT/'content_ann.parquet'),('original_v3_exact_dssm',OLD/'dssm/epoch_3/validation_ann.parquet')]:
 results[model]={}
 for label,policy in [('raw','baseline'),('F1','filter_history')]:
  metrics,rows=replay(requests,{'ann':path},set(),policy,OUT/f'{model}_{label}.parquet');rows.to_parquet(OUT/f'{model}_{label}_requests.parquet',index=False);same=rows[available];results[model][label]={'all_requests':metrics,'same_available_history_subset':{'requests':len(same),**{metric:float(same[col].mean()) for metric,col in [('HitRate@50','hit50'),('Recall@100','recall100'),('known_positive_NDCG@10','ndcg10')]}},'deployment_eligible':False};frames[model,label]=rows;print(json.dumps({'model':model,'policy':label,'metrics':metrics}),flush=True)
from backend.offline.evaluate_full_test import confidence
comparisons={}
for label in ['raw','F1']:
 new=frames['raw_bge_history_mean',label];base=frames['original_v3_exact_dssm',label];delta={}
 for metric,col in [('HitRate@50','hit50'),('Recall@100','recall100'),('known_positive_NDCG@10','ndcg10')]:
  rows=new[['gid','user_idx']].copy();rows['delta']=new[col]-base[col];delta[metric]={'delta':float(rows.delta.mean()),'CI95':confidence(rows,'delta')}
 comparisons[label]=delta
result={'scope':plan['scope'],'deployment_eligible':False,'diagnostic_plan_sha256':rec.digest(OUT/'diagnostic_plan.json'),'all_requests':11893,'unique_clicked_pairs':int(requests.positives.map(len).sum()),'available_history_requests':int(available.sum()),'empty_content_requests':int((~available).sum()),'gpu_seconds':gpu_seconds,'results':results,'paired_content_minus_dssm':comparisons,'candidate_sha256':rec.digest(OUT/'content_ann.parquet')};rec.write_json(OUT/'metrics.json',result);print(json.dumps(result,indent=2))
