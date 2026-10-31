"""Frozen Search request_fp32_v1 prepare/validation/test replay; no parameter selection."""
from __future__ import annotations
import argparse,copy,json,time
from pathlib import Path
import numpy as np,pandas as pd,pyarrow as pa,pyarrow.parquet as arrowpq
from backend.offline.evaluate_full_test import confidence,sha256
from training.scoring import blend_rank_scores
ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'outputs/evaluation/northstar_fix_20261031/search_request_fp32_v2'
SOURCE_BUNDLE=ROOT/'outputs/deploy/search/hard/bundles/northstar_fix_20261031_v1/bundle_runtime_v4.json'
OLD_GATE=ROOT/'outputs/evaluation/northstar_fix_20261031/search_rank_prior_gate'
VALIDATION_IDS=ROOT/'outputs/evaluation/northstar_fix_20261031/search_v3/requests_validation.parquet'
CONTRACT='request_fp32_v1';METRICS=('HitRate@10','HitRate@50','Recall@100','known_positive_NDCG@10','MRR@10','MRR@100')

def load(path):return json.loads(Path(path).read_text())
def save(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def checked_path(root,value):
 path=(Path(root)/value).resolve()
 if not path.is_relative_to(Path(root).resolve()):raise ValueError('Replay bundle asset escapes project')
 return path

def fixed_rank_metadata(meta):
 result=copy.deepcopy(meta)
 if meta['scene']!='search' or meta['dien']['score_contract']!='probability_v2' or float(meta['dien']['chosen']['weight'])!=1. or meta['dien']['rank_prior']!={'alpha':1.,'k':60}:raise ValueError('Only original frozen Search weight1/prior1k60 is eligible')
 if meta['dien'].get('inference_contract') not in {None,CONTRACT}:raise ValueError('Unexpected old inference contract')
 result['dien']['inference_contract']=CONTRACT
 return result

PROFILE_CATEGORIES=('gender_enc','platform_enc','age_enc','location_enc')

def load_context_builder(manifest):
 from backend.online.pipeline import _shrink_numeric_dtypes
 from backend.online.live_context import LiveContextBuilder
 profiles=pd.read_parquet(ROOT/'datasets/user_feat/train-00000-of-00001.parquet')
 profiles['user_idx']=profiles['user_idx'].astype(np.int64);profiles=_shrink_numeric_dtypes(profiles)
 return LiveContextBuilder(ROOT,'search',{int(row['user_idx']):row for row in profiles.to_dict('records')},cutoff_exclusive=manifest['training_cutoff_exclusive'])

def current_context(builder,gid,uid,query,history):
 context=builder.build(int(uid),query,history);context['search_idx']=int(gid);context['raw_query']=query
 if len(context)!=1 or int(context.iloc[0].user_idx)!=int(uid) or str(context.iloc[0]['query'])!=query:raise ValueError('Shared live builder changed current request identity')
 for name in PROFILE_CATEGORIES:
  if name not in context or not np.isfinite(context[name].to_numpy(dtype=float)).all():raise ValueError('Shared live profile category missing or invalid: '+name)
 return context

def prepare(out,source_bundle):
 import shutil
 from backend.online.bundle import required_asset_names,bundle_identity,validate_manifest
 if out.exists():raise FileExistsError('Use a new request-numerical replay directory')
 source_bundle=Path(source_bundle).resolve();original=load(source_bundle);original_rank=checked_path(ROOT,original['rank_dir']);rank_meta=load(original_rank/'manifest.json');newmeta=fixed_rank_metadata(rank_meta)
 if original['scene']!='search' or original['input_cap']!=160 or original['coarse_cap']!=160 or original['recall_policy']!={'strategy':'weighted_rrf','weights':{'ann':.5,'bm25':1.},'rrf_k':60,'route_topk':1000,'topk':500,'head_cap':100}:raise ValueError('Frozen Search recall policy or budgets changed')
 old=load(OLD_GATE/'selection.json');recipe=old['selected']
 if not old['accepted'] or recipe['coarse_cap']!=160 or recipe['weight']!=1 or recipe['rank_prior']['alpha']!=1 or recipe['rank_prior']['k']!=60:raise ValueError('Original validation selection mismatch')
 out.mkdir(parents=True);rank=out/'rank';rank.mkdir()
 for name in ['gbdt.txt','dien.pt']:shutil.copyfile(original_rank/name,rank/name)
 save(rank/'manifest.json',newmeta)
 for name in ['gbdt.txt','dien.pt']:
  if sha256(rank/name)!=sha256(original_rank/name):raise ValueError('Replay must preserve exact trained weights')
 manifest=copy.deepcopy(original);manifest['rank_dir']=rank.relative_to(ROOT).as_posix();assets=manifest['assets'].copy()
 for name in ['manifest.json','gbdt.txt','dien.pt']:assets.pop((original_rank/name).relative_to(ROOT).as_posix(),None)
 for name in ['manifest.json','gbdt.txt','dien.pt']:assets[(rank/name).relative_to(ROOT).as_posix()]=sha256(rank/name)
 runtime_names={name for name in assets if name.endswith('.py')}|{'src/backend/online/ranking/inference.py'}
 for name in runtime_names:assets[name]=sha256(ROOT/name)
 manifest['assets']=assets;manifest['numerical_replay']={'contract':CONTRACT,'scope':'Offline staged replay only; no activation or recipe selection','original_bundle_sha256':sha256(source_bundle),'original_selection_sha256':sha256(OLD_GATE/'selection.json')};manifest['bundle_id']=bundle_identity(manifest)
 required=required_asset_names(manifest,ROOT)
 if not required.issubset(assets):raise ValueError('Replay omitted required immutable serving assets')
 validate_manifest(manifest,'search',ROOT,tag=manifest['tag']);save(out/'bundle.json',manifest)
 code={str((ROOT/name).resolve()):sha256(ROOT/name) for name in runtime_names};code.update({str(Path(__file__).resolve()):sha256(Path(__file__)),str((ROOT/'tests/test_search_numerical_replay.py').resolve()):sha256(ROOT/'tests/test_search_numerical_replay.py')})
 inputs={str(source_bundle):sha256(source_bundle),str((OLD_GATE/'selection.json').resolve()):sha256(OLD_GATE/'selection.json'),str(VALIDATION_IDS.resolve()):sha256(VALIDATION_IDS),str((ROOT/'datasets/search_train/train-00000-of-00001.parquet').resolve()):sha256(ROOT/'datasets/search_train/train-00000-of-00001.parquet')}
 
 for path in [ROOT/'datasets/user_feat/train-00000-of-00001.parquet',*[ROOT/'features/vocab_dict'/f'{name}.pkl' for name in ['gender','platform','age','location']]]:inputs[str(path.resolve())]=sha256(path)
 plan={'scene':'search','inference_contract':CONTRACT,'frozen_before_official_test':True,'input_contract':'LiveContextBuilder_bound_user_profile_v2','source_snapshot':str((out/'source_snapshot/manifest.json').resolve()),'profile_categories':list(PROFILE_CATEGORIES),'execution':{'query_cpu_threads':4,'fine_device':'cuda','fine_padding':1024,'fp32':True,'matmul_tf32':False,'cudnn_tf32':False},'input_cap':160,'coarse_cap':160,'raw_cap':500,'route_topk':1000,'selection_mode':'NO parameter selection: original accepted recipe and weights fixed; numerical path correction only','recipe':recipe,'recall_policy':original['recall_policy'],'guards':'Validation strictR100increase andHR10/KP-NDCG10/MRR100 nonregression vsfreshraw; test allfour nonregression aftervalPASS; originalEPS1e-12','no_test_until_validation_pass_and_root_approval':True,'original_selection_sha256':sha256(OLD_GATE/'selection.json'),'staged_bundle':str((out/'bundle.json').resolve()),'staged_bundle_sha256':sha256(out/'bundle.json'),'inputs':inputs,'code_sha256':code,'model_sha256':{name:sha256(rank/name) for name in ['gbdt.txt','dien.pt']},'asset_sha256':{str((ROOT/name).resolve()):expected for name,expected in assets.items()},'query':'Shared LiveContextBuilder with bound user profile/vocab and CURRENT rawquery/user/history everyrequest; CPU singlequery/thread4/max512dynamicpad; no cached test request vectors','fine':'Perrequest originalrecallrank order -> repeatlastrow padding1024 -> FP32/TF32off/autocastoff -> crop; shared score_request_frame; no rounded features/logits','limitations':['Historicaltest previously observed regression benchmark','Provider history/catalog PIT remains UNKNOWN','No newalgorithm/onlinegain claim from numericalcorrection']}
 save(out/'plan.json',plan)
 snapshot=out/'source_snapshot';snapshot.mkdir();snapfiles={}
 for path,expected in code.items():
  source=Path(path);target=snapshot/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
  if sha256(target)!=expected:raise ValueError('Frozen source archive mismatch')
  snapfiles[path]={'path':target.relative_to(ROOT).as_posix(),'sha256':expected}
 save(snapshot/'manifest.json',{'status':'PASS','plan_sha256':sha256(out/'plan.json'),'files':snapfiles,'scope':'Exact pre-validation source bytes including driver/tests; v1 invalid-input reference preserved'})
 print(json.dumps({'stage':'prepared','path':str(out),'plan_sha256':sha256(out/'plan.json'),'bundle_id':manifest['bundle_id']}),flush=True);return plan

def verify(out):
 plan=load(out/'plan.json')
 if (plan['scene'],plan['inference_contract'],plan['input_cap'],plan['coarse_cap'],plan['frozen_before_official_test'])!=('search',CONTRACT,160,160,True):raise ValueError('Frozen numerical plan changed')
 for path,expected in {**plan['inputs'],**plan['code_sha256']}.items():
  if sha256(Path(path))!=expected:raise ValueError('Frozen numerical replay source/input changed: '+path)
 if plan.get('input_contract')!='LiveContextBuilder_bound_user_profile_v2':raise ValueError('Missing shared live categorical input contract')
 if sha256(Path(plan['staged_bundle']))!=plan['staged_bundle_sha256']:raise ValueError('Staged bundle changed')
 return plan

def require_validation(out):
 path=out/'validation/metrics.json'
 if not path.exists():raise ValueError('Full validation must pass before official test')
 result=load(path)
 if not result['accepted_for_release'] or result['all_requests']!=6134 or result['selection_sha256']!=sha256(out/'plan.json'):raise ValueError('Validation gate did not pass the same frozen numerical plan')
 return sha256(path)

def read_original(phase):
 folder='search_train' if phase=='validation' else 'search_test';source=ROOT/f'datasets/{folder}/train-00000-of-00001.parquet';raw=pd.read_parquet(source,columns=['search_idx','user_idx','query','recent_clicked_note_idxs','search_result_details_with_idx'])
 if phase=='validation':
  wanted=pd.read_parquet(VALIDATION_IDS);key='gid' if 'gid' in wanted else 'search_idx';ids=set(map(int,wanted[key]));raw=raw[raw.search_idx.isin(ids)]
 if len(raw)!=(6134 if phase=='validation' else 6192) or raw.search_idx.duplicated().any():raise ValueError('Every original request must be retained exactly once')
 return raw.sort_values('search_idx').reset_index(drop=True),source

def metric_row(gid,uid,positive,ids):
 positive=set(map(int,positive));ids=list(map(int,ids));ranks=[i for i,note in enumerate(ids,1) if note in positive];first=min(ranks,default=np.inf);npos=len(positive);ideal=sum(1./np.log2(i+1) for i in range(1,min(npos,10)+1));hits100=sum(i<=100 for i in ranks)
 return {'gid':int(gid),'user_idx':int(uid),'positives':npos,'candidates':len(ids),'known_positive_pairs_pool':len(ranks),'known_positive_pairs_top10':sum(i<=10 for i in ranks),'known_positive_pairs_top50':sum(i<=50 for i in ranks),'known_positive_pairs_top100':hits100,'HitRate@10':float(first<=10),'HitRate@50':float(first<=50),'Recall@100':hits100/npos if npos else 0.,'known_positive_NDCG@10':sum(1./np.log2(i+1) for i in ranks if i<=10)/ideal if ideal else 0.,'MRR@10':1/first if first<=10 else 0.,'MRR@100':1/first if first<=100 else 0.}

def summarize(rows):
 frame=pd.DataFrame(rows);result={name:float(frame[name].mean()) for name in METRICS};result.update({'requests':len(frame),'positive_requests':int((frame.positives>0).sum()),'empty_candidate_requests':int((frame.candidates==0).sum()),'known_positive_pairs_pool':int(frame.known_positive_pairs_pool.sum()),'mean_candidates':float(frame.candidates.mean())})
 for name in ['known_positive_pairs_top10','known_positive_pairs_top50','known_positive_pairs_top100']:result[name]=int(frame[name].sum())
 return result

def guard(final,raw,validation):
 keys=['Recall@100','HitRate@10','known_positive_NDCG@10','MRR@100'];eps=1e-12
 if not all(np.isfinite(final.get(k,np.nan)) and np.isfinite(raw.get(k,np.nan)) for k in keys):return False
 return (final['Recall@100']>raw['Recall@100']+eps if validation else final['Recall@100']>=raw['Recall@100']-eps) and all(final[k]>=raw[k]-eps for k in keys[1:])

class ArrowSink:
 def __init__(self,path,empty_columns):self.path=Path(path);self.writer=None;self.schema=None;self.empty_columns=empty_columns
 def write(self,frame):
  if frame.empty:return
  table=pa.Table.from_pandas(frame,preserve_index=False)
  for name in ['recent_clicked_note_idxs']:
   if name in table.column_names:table=table.set_column(table.schema.get_field_index(name),name,pa.array(frame[name].tolist(),type=pa.list_(pa.int64())))
  if self.writer is None:self.schema=table.schema;self.writer=arrowpq.ParquetWriter(self.path,self.schema)
  self.writer.write_table(table.cast(self.schema))
 def close(self):
  if self.writer:self.writer.close()
  else:pd.DataFrame({name:pd.Series(dtype=dtype) for name,dtype in self.empty_columns.items()}).to_parquet(self.path,index=False)

def ordered_stage(frame,column):
 if frame.empty:return frame.copy()
 return frame.sort_values([column,'rank'],ascending=[False,True],kind='stable')

def candidate_output(frame,score_column):
 return pd.DataFrame({'search_idx':frame.search_idx.to_numpy(np.int64),'gid':frame.search_idx.to_numpy(np.int64),'note_idx':frame.note_idx.to_numpy(np.int64),'rank':np.arange(1,len(frame)+1,dtype=np.int64),'originalrank':frame['rank'].to_numpy(np.int64),'score':frame[score_column].to_numpy(np.float64)})

def run_phase(out,phase,device='cuda'):
 from backend.online.bundle import ServingBundle,validate_manifest
 from backend.online.ranking.stack import RankStackPredictor
 from backend.online.recall.dssm import DSSMRecaller
 from backend.online.recall.service import fetch_bundle_recall_candidates
 from backend.online.ranking.inference import configure_request_inference
 plan=verify(out);validation_sha=require_validation(out) if phase=='test' else None;dest=out/phase;dest.mkdir(exist_ok=False);save(dest/'phase_start.json',{'phase':phase,'plan_sha256':sha256(out/'plan.json'),'validation_sha256':validation_sha,'original_selection_sha256':plan['original_selection_sha256'],'parameter_choice_after_test':False})
 raw,source=read_original(phase);manifest=load(plan['staged_bundle']);validate_manifest(manifest,'search',ROOT,tag=manifest['tag']);configure_request_inference(CONTRACT);builder=load_context_builder(manifest)
 ranker=RankStackPredictor('search',ROOT/manifest['rank_dir'],root=ROOT,device=device,query_encoder_dir=ROOT/manifest['query_encoder_dir'],query_encoder_revision=manifest['query_encoder_revision']);bundle=ServingBundle(manifest,ranker,ROOT);bundle._get_bm25_index();dssm=DSSMRecaller('search',manifest['tag'],bundle.recall_dir,'search_idx',query_encoder_dir=ROOT/manifest['query_encoder_dir'],query_encoder_revision=manifest['query_encoder_revision'],use_shared_note_vectors=True,inference_contract=CONTRACT)
 if not dssm.enabled or ranker.inference_contract!=CONTRACT:raise ValueError('Canonical current-input serving components failed to load')
 sinkcols={'gid':'int64','note_idx':'int64','rank':'int64','score':'float64'};sinks={'raw':ArrowSink(dest/'raw_selected.parquet',sinkcols),'scores':ArrowSink(dest/'scores.parquet',{'search_idx':'int64','user_idx':'int64','note_idx':'int64','rank':'int64','originalrank':'int64','coarse_score':'float32','dien_raw_score':'float32','final_score':'float64'})}
 for stage in ['coarse','fine','final']:sinks[stage]=ArrowSink(dest/f'{stage}_candidates.parquet',{**sinkcols,'search_idx':'int64','originalrank':'int64'})
 contextrows=[];rows={stage:[] for stage in ['raw','input','coarse','fine','final']};statusrows=[];started=time.monotonic()
 try:
  for index,record in enumerate(raw.to_dict('records')):
   gid=int(record['search_idx']);uid=int(record['user_idx']);query=str(record['query'] or '').strip();history=list(map(int,record['recent_clicked_note_idxs']));positive={int(item['note_idx']) for item in record['search_result_details_with_idx'] if item['click']>0}
   context=current_context(builder,gid,uid,query,history);hydrated_history=list(map(int,context.iloc[0].recent_clicked_note_idxs));contextrows.append({'gid':gid,'user_idx':uid,**{name:int(context.iloc[0][name]) for name in PROFILE_CATEGORIES},'raw_history_sha256':__import__('hashlib').sha256(json.dumps(history,separators=(',',':')).encode()).hexdigest(),'hydrated_history_sha256':__import__('hashlib').sha256(json.dumps(hydrated_history,separators=(',',':')).encode()).hexdigest(),'query_sha256':__import__('hashlib').sha256(query.encode()).hexdigest()});candidates=fetch_bundle_recall_candidates(bundle,dssm,context,scene='search',max_rank=500);route_status=candidates.attrs.get('route_status',{})
   if route_status.get('errors'):raise RuntimeError('Fresh recall route error: '+str(route_status))
   if candidates.empty:candidates=pd.DataFrame({'search_idx':pd.Series(dtype='int64'),'user_idx':pd.Series(dtype='int64'),'note_idx':pd.Series(dtype='int64'),'rank':pd.Series(dtype='int64'),'recall_score':pd.Series(dtype='float64')})
   if candidates.note_idx.duplicated().any() or len(candidates)>500:raise ValueError('Fresh recall identity/budget violation')
   candidates=candidates.sort_values('rank',kind='stable');sinks['raw'].write(pd.DataFrame({'gid':candidates.search_idx.to_numpy(np.int64),'note_idx':candidates.note_idx.to_numpy(np.int64),'rank':candidates['rank'].to_numpy(np.int64),'score':candidates.recall_score.to_numpy(np.float64)}));rows['raw'].append(metric_row(gid,uid,positive,candidates.note_idx))
   current=candidates.head(160).copy();current['query']=query;current['raw_query']=query;current['recent_clicked_note_idxs']=[hydrated_history for _ in range(len(current))];rows['input'].append(metric_row(gid,uid,positive,current.note_idx));scored=ranker.score_request_frame(current);scored['originalrank']=scored['rank'].astype(np.int64);scored['original_recall_rank']=scored['originalrank']
   if len(scored)!=len(current) or not np.array_equal(scored.note_idx,current.note_idx):raise ValueError('Canonical score API changed input candidate order')
   scored['fine_score']=blend_rank_scores(scored.coarse_score.to_numpy(),scored.dien_raw_score.to_numpy(),scored.search_idx.to_numpy(),1.,'probability_v2')
   for col in ['coarse_score','dien_raw_score','fine_score','final_score']:
    if not np.isfinite(scored[col]).all():raise ValueError('Nonfinite numerical replay score')
   sinks['scores'].write(scored)
   for stage,col in [('coarse','coarse_score'),('fine','fine_score'),('final','final_score')]:
    ordered=ordered_stage(scored,col);rows[stage].append(metric_row(gid,uid,positive,ordered.note_idx));sinks[stage].write(candidate_output(ordered,col))
   statusrows.append({'gid':gid,'ann_candidates':int(route_status.get('counts',{}).get('ann',0)),'bm25_candidates':int(route_status.get('counts',{}).get('bm25',0)),'raw_candidates':len(candidates),'scored_candidates':len(scored)})
   if (index+1)%100==0:print(json.dumps({'phase':phase,'requests':index+1,'total':len(raw),'seconds':time.monotonic()-started}),flush=True)
 finally:
  for sink in sinks.values():sink.close()
 summaries={name:summarize(values) for name,values in rows.items()}
 for stage,values in rows.items():pd.DataFrame(values).to_parquet(dest/f'{stage}_requests.parquet',index=False)
 pd.DataFrame(contextrows).to_parquet(dest/'request_context.parquet',index=False);pd.DataFrame(statusrows).to_parquet(dest/'route_status.parquet',index=False);paired={}
 for new,previous in [('coarse','input'),('fine','coarse'),('final','fine'),('final','raw')]:
  a=pd.DataFrame(rows[new]);b=pd.DataFrame(rows[previous]);comparison={}
  for metric in METRICS:
   table=a[['gid','user_idx']].copy();table['delta']=a[metric]-b[metric];comparison[metric]={'delta':float(table.delta.mean()),'user_cluster_CI95':confidence(table,'delta')}
  paired[new+'-'+previous]=comparison
 old_diffs={};reference_inputs={}
 for stage in ['raw','coarse','fine','final']:
  path=OLD_GATE/(f'validation_{stage}_requests.parquet' if phase=='validation' else f'test/{stage}_requests.parquet');reference_inputs[str(path.resolve())]=sha256(path);reference=pd.read_parquet(path).sort_values('gid').reset_index(drop=True);fresh=pd.DataFrame(rows[stage]).sort_values('gid').reset_index(drop=True)
  if not np.array_equal(reference.gid,fresh.gid):raise ValueError('Old/new numerical diagnostic cohorts differ')
  old_diffs[stage]={metric:{'old':float(reference[metric].mean()),'new':float(fresh[metric].mean()),'new_minus_old':float((fresh[metric]-reference[metric]).mean())} for metric in METRICS}
 outputs={str(p.resolve()):sha256(p) for p in dest.glob('*.parquet')};result={'scene':'search','phase':phase,'all_requests':len(raw),'all_test_requests':len(raw) if phase=='test' else None,'original_positive_pairs':sum(len({int(d['note_idx']) for d in nested if d['click']>0}) for nested in raw.search_result_details_with_idx),'accepted_for_release':guard(summaries['final'],summaries['raw'],phase=='validation'),'selection_sha256':sha256(out/'plan.json'),'original_selection_sha256':plan['original_selection_sha256'],'validation_sha256':validation_sha,'inference_contract':CONTRACT,'input_contract':'LiveContextBuilder_bound_user_profile_v2','profile_categories':list(PROFILE_CATEGORIES),'execution':plan.get('execution',{}),'input_cap':160,'coarse_cap':160,'raw_cap':500,'recipe':plan['recipe'],'recall_policy':plan['recall_policy'],'stages':summaries,'paired_stages':paired,'numerical_new_minus_old':old_diffs,'parameter_choice_after_test':False,'model_weights_changed':False,'algorithm_policy_changed':False,'seconds':time.monotonic()-started,'inputs':{**plan['inputs'],**plan['code_sha256'],**plan.get('asset_sha256',{}),str(source.resolve()):sha256(source),str((out/'plan.json').resolve()):sha256(out/'plan.json'),str((dest/'raw_selected.parquet').resolve()):sha256(dest/'raw_selected.parquet'),**reference_inputs},'outputs':outputs,'scope':'Fresh current rawquery/user/history recall plus exact shared canonical serving score API; full raw clicked denominators; numerical contract correction, not newalgorithm gain','limitations':plan['limitations']};save(dest/'metrics.json',result);print(json.dumps({'phase':phase,'requests':len(raw),'accepted_for_release':result['accepted_for_release'],'stages':summaries}),flush=True);return result

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=['prepare','validation','test'],required=True);p.add_argument('--output',type=Path,default=DEFAULT);p.add_argument('--bundle',type=Path,default=SOURCE_BUNDLE);p.add_argument('--device',choices=['cpu','cuda'],default='cuda');a=p.parse_args();out=a.output.resolve()
 if a.phase=='prepare':prepare(out,a.bundle)
 else:run_phase(out,a.phase,a.device)
if __name__=='__main__':main()
