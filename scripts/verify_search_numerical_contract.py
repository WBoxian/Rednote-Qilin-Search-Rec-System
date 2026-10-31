"""Independent original truth/stage arithmetic for request-local Search numerical replay."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/evaluation/northstar_fix_20261031/search_request_fp32_v2';OLD=ROOT/'outputs/evaluation/northstar_fix_20261031/search_rank_prior_gate';p=argparse.ArgumentParser();p.add_argument('--phase',choices=['validation','test'],required=True);p.add_argument('--run',type=Path,default=OUT);p.add_argument('--output-dir',type=Path);a=p.parse_args();OUT=a.run.resolve();phase=a.phase;dest=OUT/phase;AUDIT=(a.output_dir or OUT/'verification').resolve();AUDIT.mkdir(parents=True,exist_ok=True);proof_path=AUDIT/f'{phase}_independent_verification.json';assert not proof_path.exists(),'Refuse overwriting frozen independent proof';stored=json.loads((dest/'metrics.json').read_text());plan=json.loads((OUT/'plan.json').read_text());folder='search_train' if phase=='validation' else 'search_test';source=ROOT/f'datasets/{folder}/train-00000-of-00001.parquet';raw=pd.read_parquet(source,columns=['search_idx','user_idx','query','recent_clicked_note_idxs','search_result_details_with_idx'])
if phase=='validation':
 ids=set(map(int,pd.read_parquet(ROOT/'outputs/evaluation/northstar_fix_20261031/search_v3/requests_validation.parquet',columns=['search_idx']).search_idx));raw=raw[raw.search_idx.isin(ids)]
expected=6134 if phase=='validation' else 6192;expectedpairs=22080 if phase=='validation' else 27597;assert len(raw)==expected and raw.search_idx.is_unique;truth={int(r.search_idx):{int(i['note_idx']) for i in r.search_result_details_with_idx if i['click']>0} for r in raw.itertuples()};users=dict(zip(raw.search_idx,raw.user_idx));queries={int(r.search_idx):str(r.query or '').strip() for r in raw.itertuples()};histories={int(r.search_idx):tuple(int(n) for n in r.recent_clicked_note_idxs if int(n)>=0)[:20] for r in raw.itertuples()};assert sum(map(len,truth.values()))==expectedpairs
import pickle
profiles=pd.read_parquet(ROOT/'datasets/user_feat/train-00000-of-00001.parquet');profiles['user_idx']=profiles['user_idx'].astype(np.int64)
from backend.online.pipeline import _shrink_numeric_dtypes
profiles=_shrink_numeric_dtypes(profiles);profiles={int(r['user_idx']):r for r in profiles.to_dict('records')};vocab={}
for name in ['gender','platform','age','location']:
 with open(ROOT/f'features/vocab_dict/{name}.pkl','rb') as stream:vocab[name]=pickle.load(stream)
context=pd.read_parquet(dest/'request_context.parquet').set_index('gid');assert len(context)==expected and context.index.is_unique
for r in raw.itertuples():
 gid=int(r.search_idx);profile=profiles.get(int(r.user_idx),{});row=context.loc[gid];assert int(row.user_idx)==int(r.user_idx)
 for name in vocab:
  value=profile.get(name);cat=int(vocab[name].get(str(value) if pd.notna(value) else 'UNK',0));assert int(row[name+'_enc'])==cat
 expected_raw=list(map(int,r.recent_clicked_note_idxs));expected_hydrated=[int(n) for n in expected_raw if int(n)>=0][:20]
 assert row.raw_history_sha256==hashlib.sha256(json.dumps(expected_raw,separators=(',',':')).encode()).hexdigest()
 assert row.hydrated_history_sha256==hashlib.sha256(json.dumps(expected_hydrated,separators=(',',':')).encode()).hexdigest()
 assert row.query_sha256==hashlib.sha256(queries[gid].encode()).hexdigest()
rawc=pd.read_parquet(dest/'raw_selected.parquet');scores=pd.read_parquet(dest/'scores.parquet');assert not rawc.duplicated(['gid','note_idx']).any();assert not scores.duplicated(['search_idx','note_idx']).any();assert not any(c in scores for c in ['click','positives','positive_note_ids','search_result_details_with_idx']);rawgroups={int(g):d.sort_values('rank') for g,d in rawc.groupby('gid',sort=False)};scoregroups={int(g):d.sort_values('rank') for g,d in scores.groupby('search_idx',sort=False)};saved={stage:{int(g):d.sort_values('rank') for g,d in pd.read_parquet(dest/f'{stage}_candidates.parquet').groupby('gid',sort=False)} for stage in ['coarse','fine','final']};rows={stage:[] for stage in ['raw','input','coarse','fine','final']}
def metrics(gid,ids):
 positive=truth[gid];hits=[i for i,note in enumerate(ids,1) if int(note) in positive];first=min(hits,default=np.inf);n=len(positive);ideal=sum(1./np.log2(i+1) for i in range(1,min(10,n)+1));return {'gid':gid,'user_idx':users[gid],'positives':n,'candidates':len(ids),'known_positive_pairs_pool':len(hits),'known_positive_pairs_top10':sum(i<=10 for i in hits),'known_positive_pairs_top50':sum(i<=50 for i in hits),'known_positive_pairs_top100':sum(i<=100 for i in hits),'HitRate@10':float(first<=10),'HitRate@50':float(first<=50),'Recall@100':sum(i<=100 for i in hits)/n if n else 0.,'known_positive_NDCG@10':sum(1./np.log2(i+1) for i in hits if i<=10)/ideal if ideal else 0.,'MRR@10':1/first if first<=10 else 0.,'MRR@100':1/first if first<=100 else 0.}
for gid,positive in truth.items():
 rawg=rawgroups.get(gid,pd.DataFrame(columns=['note_idx','rank','score']));g=scoregroups.get(gid,pd.DataFrame(columns=scores.columns));assert len(rawg)<=500 and len(g)==min(160,len(rawg));rawids=rawg.note_idx.to_numpy(np.int64);rawvalues=rawg.score.to_numpy(np.float64);np.testing.assert_array_equal(np.lexsort((rawids,-rawvalues)),np.arange(len(rawg)));np.testing.assert_array_equal(rawg['rank'],np.arange(1,len(rawg)+1));np.testing.assert_array_equal(g.note_idx,rawids[:160]);np.testing.assert_array_equal(g['rank'],np.arange(1,len(g)+1));np.testing.assert_array_equal(g.originalrank,g['rank']);assert all(int(uid)==users[gid] for uid in g.user_idx);assert all(str(q)==queries[gid] for q in g['query']);assert all(str(q)==queries[gid] for q in g.raw_query);assert all(tuple(map(int,h))==histories[gid] for h in g.recent_clicked_note_idxs)
 ids=g.note_idx.to_numpy(np.int64);original=g['rank'].to_numpy(np.int64);coarse=g.coarse_score.to_numpy(np.float32);rawfine=g.dien_raw_score.to_numpy(np.float32);fine=np.exp(-np.logaddexp(np.float32(0.),-rawfine)).astype(np.float32);fineorder=np.lexsort((original,-fine));learnedrank=np.empty(len(g),np.int64);learnedrank[fineorder]=np.arange(1,len(g)+1);final=1./(60.+original)+1./(60.+learnedrank);np.testing.assert_array_equal(g.fine_score,fine);np.testing.assert_array_equal(g.final_score,final)
 orders={'coarse':np.lexsort((original,-coarse)),'fine':fineorder,'final':np.lexsort((original,-final))};values={'coarse':coarse.astype(np.float64),'fine':fine.astype(np.float64),'final':final};rows['raw'].append(metrics(gid,rawids));rows['input'].append(metrics(gid,ids))
 for stage,order in orders.items():
  actual=saved[stage].get(gid,pd.DataFrame(columns=['note_idx','rank','originalrank','score']));np.testing.assert_array_equal(actual.note_idx,ids[order]);np.testing.assert_array_equal(actual['rank'],np.arange(1,len(g)+1));np.testing.assert_array_equal(actual.originalrank,original[order]);np.testing.assert_array_equal(actual.score,values[stage][order]);rows[stage].append(metrics(gid,ids[order]))
summary={};per={};metricnames=['HitRate@10','HitRate@50','Recall@100','known_positive_NDCG@10','MRR@10','MRR@100'];countnames=['known_positive_pairs_pool','known_positive_pairs_top10','known_positive_pairs_top50','known_positive_pairs_top100']
for stage,values in rows.items():
 table=pd.DataFrame(values).sort_values('gid').reset_index(drop=True);expectedrows=pd.read_parquet(dest/f'{stage}_requests.parquet').sort_values('gid').reset_index(drop=True)
 for column in table:np.testing.assert_allclose(table[column],expectedrows[column],rtol=0,atol=1e-15)
 result={metric:float(table[metric].mean()) for metric in metricnames};result.update({c:int(table[c].sum()) for c in countnames})
 for key,value in result.items():assert abs(value-stored['stages'][stage][key])<1e-12,(stage,key,value,stored['stages'][stage][key])
 per[stage]=table;summary[stage]=result;table.to_parquet(AUDIT/f'{phase}_{stage}_independent_requests.parquet',index=False)
paired={};finalrows=per['final'];rawrows=per['raw']
for metric in metricnames:
 d=finalrows[metric]-rawrows[metric];table=pd.DataFrame({'user_idx':finalrows.user_idx,'sum':d,'n':1}).groupby('user_idx').agg({'sum':'sum','n':'sum'});sums,counts=table['sum'].to_numpy(),table.n.to_numpy();rng=np.random.default_rng(42);draws=[]
 for start in range(0,2000,100):
  pick=rng.integers(0,len(table),size=(100,len(table)));draws.extend((sums[pick].sum(1)/counts[pick].sum(1)).tolist())
 paired[metric]={'delta':float(d.mean()),'user_cluster_CI95_2000':[float(v) for v in np.quantile(draws,[.025,.975])]}
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
inputs={str(source.resolve()):sha(source),str((OUT/'plan.json').resolve()):sha(OUT/'plan.json'),str((dest/'metrics.json').resolve()):sha(dest/'metrics.json'),str(Path(__file__).resolve()):sha(Path(__file__))}
for file in dest.glob('*.parquet'):inputs[str(file.resolve())]=sha(file)

for path,expected_sha in plan['code_sha256'].items():assert sha(path)==expected_sha,('Frozen numerical source drift',path)
inputs.update(plan['code_sha256']);inputs.update({str((ROOT/'datasets/user_feat/train-00000-of-00001.parquet').resolve()):sha(ROOT/'datasets/user_feat/train-00000-of-00001.parquet'),**{str((ROOT/f'features/vocab_dict/{name}.pkl').resolve()):sha(ROOT/f'features/vocab_dict/{name}.pkl') for name in vocab}});inputs.update({str((OUT/'rank'/name).resolve()):sha(OUT/'rank'/name) for name in ['manifest.json','gbdt.txt','dien.pt']})
result={'status':'PASS','scene':'search','phase':phase,'inference_contract':'request_fp32_v1','selection_sha256':sha(OUT/'plan.json'),'original_selection_sha256':plan['original_selection_sha256'],'raw_candidate_sha256':sha(dest/'raw_selected.parquet'),'final_candidate_sha256':sha(dest/'final_candidates.parquet'),'input_contract':'LiveContextBuilder_bound_user_profile_v2','all_profile_category_and_header_fingerprints_verified':expected,'original_requests':expected,'original_positive_pairs':expectedpairs,'stages':summary,'paired_final_minus_fresh_raw':paired,'all_sorted_stage_rows_verified':len(scores),'inputs_sha256':inputs,'scope':'Independent original raw truth/currentmodelinput identity/score ordering/sigmoid plusreciprocalprior and complete original-request denominators; no project metrics/selection functions imported; arithmetic proof is distinct from model/runtime replay reproducibility','old_model_weights_and_recipe_retained':True,'parameter_selection_after_test':False};proof_path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

profile_path=AUDIT/f'{phase}_profile_input_verification.json';assert not profile_path.exists()
profile_result={'status':'PASS','scene':'search','user_profile_contract':'shared_live_context_v1','input_contract':'LiveContextBuilder_bound_user_profile_v2','profile_fields':['gender_enc','platform_enc','age_enc','location_enc'],'selection_sha256':sha(OUT/'plan.json'),'original_requests':expected,'nonzero_profile_requests':int((context[['gender_enc','platform_enc','age_enc','location_enc']]!=0).any(axis=1).sum()),'profile_mismatches':0,'raw_context_mismatches':0,'original_user_ids_preserved':True,'current_query_and_raw_history_fingerprints_verified':True,'history_normalization':'nonnegative original prefix20 exactly shared LiveContextBuilder','inputs_sha256':inputs,'scope':'Every raw request user/query/rawhistory verified against saved fingerprints and independently loaded bound user profile/vocab; known categories never defaulted away; unknown UID explicitzero vocabulary fallback retained'}
profile_path.write_text(json.dumps(profile_result,indent=2)+'\n');print(json.dumps({'profile_status':'PASS','path':str(profile_path),'nonzero_profile_requests':profile_result['nonzero_profile_requests']}))
