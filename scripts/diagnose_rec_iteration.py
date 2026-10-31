"""Train/validation-only root-cause probes for the frozen Rec V3 retrieval model."""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'outputs/evaluation/northstar_fix_20261031/rec_v3'
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def latent_stats(x,seed=42):
 rng=np.random.default_rng(seed);take=rng.choice(len(x),min(8192,len(x)),replace=False);a=np.asarray(x[take],np.float64);norm=np.linalg.norm(a,axis=1);mean=a.mean(0);center=a-mean;ev=np.linalg.eigvalsh(center.T@center/len(a));ev=ev.clip(0);p=ev/max(ev.sum(),1e-20);effective=float(np.exp(-(p[p>0]*np.log(p[p>0])).sum()));pairs=(a[rng.integers(len(a),size=10000)]*a[rng.integers(len(a),size=10000)]).sum(1)
 return {'rows':len(x),'sample':len(a),'norm_min':float(norm.min()),'norm_max':float(norm.max()),'mean_vector_norm':float(np.linalg.norm(mean)),'centered_variance_trace':float(ev.sum()),'effective_centered_rank':effective,'dominant_eigen_fraction':float(ev.max()/max(ev.sum(),1e-20)),'random_pair_cosine_mean':float(pairs.mean()),'random_pair_cosine_q95':float(np.quantile(pairs,.95)),'dimension_std_mean':float(center.std(0).mean())}
def diagnostics(out):
 start=time.monotonic();out.mkdir(parents=True,exist_ok=True);dest=out/'diagnosis.json'
 if dest.exists():raise FileExistsError('Diagnosis already frozen; use another output directory')
 fitreq=pd.read_parquet(BASE/'requests_fit.parquet');valreq=pd.read_parquet(BASE/'requests_validation.parquet');labels=pd.read_parquet(BASE/'labels_fit.parquet');meta=json.loads((BASE/'dssm/training_manifest.json').read_text());selected=json.loads((BASE/'dssm/selected.json').read_text());epoch=Path(selected['directory']);itemids=np.load(BASE/'dssm/item_ids.npy');items=np.memmap(epoch/'item_vectors.bin',dtype=np.float32,mode='r',shape=(len(itemids),128));qids=np.load(epoch/'validation_request_ids.npy');queries=np.load(epoch/'validation_vectors.npy')
 assert len(fitreq)==55616 and len(valreq)==11893 and set(fitreq.gid).isdisjoint(valreq.gid)
 # Exact source labels, same complete-request partitions as V3; static aggregate features are irrelevant here.
 source=ROOT/'features/rec_train_features.parquet';frame=pd.read_parquet(source,columns=['request_idx','user_idx','note_idx','click','recent_clicked_note_idxs','event_timestamp']);fit=frame[frame.request_idx.isin(set(fitreq.gid))].copy();val=frame[frame.request_idx.isin(set(valreq.gid))].copy();positives=fit[fit.click>0].drop_duplicates(['request_idx','note_idx']).copy();assert len(positives)==meta['fit_positive_pairs'];assert fit.event_timestamp.max()<val.event_timestamp.min()
 result={'scope':'Train-fit and training-internal validation only; no official test opened. Diagnostics are not new CTR metrics.','source_sha256':{str(p):sha(p) for p in [source,BASE/'requests_fit.parquet',BASE/'requests_validation.parquet',BASE/'labels_fit.parquet',epoch/'validation_vectors.npy',epoch/'item_vectors.bin']},'fit_requests':len(fitreq),'validation_requests':len(valreq),'fit_positive_pairs':len(positives),'vector_user':latent_stats(queries),'vector_item':latent_stats(items)}
 uid=positives.user_idx.to_numpy(np.int64);gid=positives.request_idx.to_numpy(np.int64);iid=positives.note_idx.to_numpy(np.int64);hist=positives.recent_clicked_note_idxs.tolist();rng=np.random.default_rng(42);order=rng.permutation(len(positives));counts={k:0 for k in ['anchor_rows','inbatch_negative_pairs','sameuser_otherrequest_pairs','anchors_sameuser_conflict','history_knownpositive_negative_pairs','anchors_history_conflict']}
 for first in range(0,len(order),256):
  idx=order[first:first+256];u,g,it=uid[idx],gid[idx],iid[idx];negative=(it[:,None]!=it[None,:])&(g[:,None]!=g[None,:]);same=negative&(u[:,None]==u[None,:]);hconf=np.vstack([np.isin(it,list(hist[j])) for j in idx])&negative
  counts['anchor_rows']+=len(idx);counts['inbatch_negative_pairs']+=int(negative.sum());counts['sameuser_otherrequest_pairs']+=int(same.sum());counts['anchors_sameuser_conflict']+=int(same.any(1).sum());counts['history_knownpositive_negative_pairs']+=int(hconf.sum());counts['anchors_history_conflict']+=int(hconf.any(1).sum())
 counts['sampling']='Deterministic NumPy seed42 positive-row shuffle, batch256; estimates shuffled batch conflicts, not a replay of Torch DataLoader RNG.'
 for key in ['sameuser_otherrequest_pairs','history_knownpositive_negative_pairs']:counts[key+'_fraction_negative_pairs']=counts[key]/counts['inbatch_negative_pairs']
 for key in ['anchors_sameuser_conflict','anchors_history_conflict']:counts[key+'_fraction_anchors']=counts[key]/counts['anchor_rows']
 result['false_negative_probe']=counts
 clicked=positives.groupby('request_idx').note_idx.apply(set).to_dict();unclicked=fit[fit.click==0].drop_duplicates(['request_idx','note_idx']);conflicts=sum(int(r.note_idx in clicked.get(r.request_idx,set())) for r in unclicked.itertuples());hard=unclicked.loc[[int(r.note_idx) not in clicked.get(r.request_idx,set()) for r in unclicked.itertuples()]];hardcounts=hard.groupby('request_idx').size();result['logged_hard_negative_support']={'fit_exposure_rows':len(fit),'fit_positive_rows':int((fit.click>0).sum()),'unique_unclicked_pairs':len(unclicked),'duplicate_label_conflict_pairs_excluded':conflicts,'usable_unclicked_pairs':len(hard),'requests_with_usable_unclicked':len(hardcounts),'fraction_fit_requests_with_unclicked':len(hardcounts)/len(fitreq),'mean_unclicked_per_supported_request':float(hardcounts.mean())}
 # Evaluate frozen-model similarity of observed clicked vs observed unclicked and uniform catalog points.
 pos_by_req=val[val.click>0].groupby('request_idx').note_idx.apply(lambda x:np.unique(x.astype(np.int64))).to_dict();neg_by_req=val[val.click==0].groupby('request_idx').note_idx.apply(lambda x:np.unique(x.astype(np.int64))).to_dict();records=[];rng=np.random.default_rng(42)
 for row,g in enumerate(qids):
  pos=pos_by_req.get(int(g),np.array([],np.int64));neg=neg_by_req.get(int(g),np.array([],np.int64));neg=neg[~np.isin(neg,pos)];q=queries[row];pv=np.asarray(items[np.searchsorted(itemids,pos)])@q if len(pos) else np.array([]);nv=np.asarray(items[np.searchsorted(itemids,neg)])@q if len(neg) else np.array([]);rv=np.asarray(items[rng.integers(0,len(items),size=32)])@q
  if len(pv):records.append((int(g),float(pv.mean()),float(nv.mean()) if len(nv) else np.nan,float(rv.mean()),float(((pv[:,None]>nv)+(pv[:,None]==nv)*.5).mean()) if len(nv) else np.nan,float((pv[:,None]>rv).mean())))
 sim=pd.DataFrame(records,columns=['gid','positive_mean_cosine','logged_unclicked_mean_cosine','random_catalog_mean_cosine','positive_vs_logged_unclicked_pairwin','positive_vs_random_pairwin']);sim.to_parquet(out/'validation_similarity_requests.parquet',index=False);result['frozen_similarity_probe']={name:float(sim[name].mean()) for name in sim if name!='gid'};result['frozen_similarity_probe']['interpretation']='Pairwise similarity separability of true logged clicked/unclicked and random catalog items; random items are NOT labeled negatives for CTR/AUC.'
 # Quantify same-model item generalization on fit-click-supported vs cold validation positives.
 seen=set(map(int,positives.note_idx.unique()));ann=pd.read_parquet(epoch/'validation_ann.parquet',columns=['gid','note_idx','rank']);top=ann[ann['rank']<=100];topsets=top.groupby('gid').note_idx.apply(set).to_dict();strata={'seen_fit_click':{'pairs':0,'hits100':0},'not_seen_fit_click':{'pairs':0,'hits100':0}}
 for row in valreq.itertuples():
  truth=set(map(int,row.positives));hits=topsets.get(int(row.gid),set())
  for note in truth:s=strata['seen_fit_click' if note in seen else 'not_seen_fit_click'];s['pairs']+=1;s['hits100']+=int(note in hits)
 for s in strata.values():s['micro_Recall100']=s['hits100']/max(s['pairs'],1)
 result['validation_click_cold_strata']=strata
 result['seconds']=time.monotonic()-start;dest.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'outputs/evaluation/northstar_fix_20261031/rec_iteration_20261031');a=p.parse_args();diagnostics(a.output)
