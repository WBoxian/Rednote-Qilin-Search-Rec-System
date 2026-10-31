"""Two predeclared validation-only request-history fusion rules on fixed Rec routes."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,pyarrow as pa,pyarrow.parquet as pq
from scripts import optimize_rec_retrieval as rec
from scripts import optimize_rec_iteration as iteration
from recall.context_fusion import POLICIES,fuse_request_routes
from backend.offline.evaluate_full_test import confidence
ROOT=iteration.ROOT;OLD=iteration.OLD;DEFAULT=ROOT/'outputs/evaluation/northstar_fix_20261031/rec_iteration_20261031/fusion_gate'

def plan(out):
 if out.exists():raise FileExistsError('Fusion output must be new')
 out.mkdir();files=[Path(__file__),ROOT/'src/recall/context_fusion.py',ROOT/'src/training/history.py',ROOT/'tests/test_context_fusion.py'];inputs=[OLD/'requests_validation.parquet',OLD/'index/validation_nprobe32.parquet',OLD/'validation/hist025_fold200/swing.parquet',OLD/'validation/hist025_fold200/usercf.parquet',OLD/'cf_graph_0.25/items.npy',OLD/'selection.json']
 result={'policies':list(POLICIES),'source_model':'unchanged_v3','conditional_extra_model':'Only if predeclared C has an accepted ANN epoch and frozen PQ index, replay SAME three policies; no newpolicy/threshold/grid','history':'shared recent_history_ids(history,20), mostrecentprefix, note0+duplicates preserved in support denominator; true positives in history remain in groundtruth','budget':'original eachrouteTop1000 then filter/compress ranks; no extra retrieval; final500','F1':'remove current recent20 history from EACH route, recompact rank, fixedRRF121/k60','F2':'F1 + graphhistory unique-supported count / recent20 length <.5 =>111 else121, no threshold search; emptyCF naturally ANN','selection':'strict Recall100 increase vsunchangedV3 baseline, HR50/KP-NDCG nonregression; selectR100 thenHR50 thenKP; no accepted candidate retainsoriginal','no_official_test_until_frozen_selection_and_root_approval':True,'code_sha256':{str(p.resolve()):rec.digest(p) for p in files},'inputs_sha256':{str(p.resolve()):rec.digest(p) for p in inputs}}
 rec.write_json(out/'fusion_plan.json',result);print(json.dumps(result,indent=2));return result

def verify(out):
 p=json.loads((out/'fusion_plan.json').read_text())
 for path,digest in {**p['code_sha256'],**p['inputs_sha256']}.items():
  if rec.digest(Path(path))!=digest:raise ValueError('Frozen fusion source/input changed')
 return p

def groups(path):
 table=pd.read_parquet(path,columns=['gid','note_idx','rank']);assert not table.duplicated(['gid','note_idx']).any()
 return {int(g):p.sort_values('rank').note_idx.to_numpy(np.int64) for g,p in table.groupby('gid',sort=False)}

def replay(requests,paths,graphitems,policy,output):
 lookup={route:groups(path) for route,path in paths.items()};rows=[];buffer=[];output.parent.mkdir(parents=True,exist_ok=True);schema=pa.schema([('gid',pa.int64()),('note_idx',pa.int64()),('score',pa.float64()),('rank',pa.int64())])
 with pq.ParquetWriter(output,schema) as writer:
  for row in requests.itertuples():
   current={route:source.get(int(row.gid),np.empty(0,np.int64)) for route,source in lookup.items()};ranked=fuse_request_routes(current,row.history,graphitems,policy,500);ids=ranked.note_idx.to_numpy(np.int64);truth=set(map(int,row.positives));positions=[i for i,note in enumerate(ids,1) if int(note) in truth];first=min(positions,default=np.inf);npos=len(truth);idcg=sum(1/np.log2(i+1) for i in range(1,min(npos,10)+1));hits=len(positions)
   rows.append({'gid':int(row.gid),'user_idx':int(row.user_idx),'positives':npos,'hits':hits,'first_hit':first,'recall500':hits/npos if npos else 0.,'recall100':sum(i<=100 for i in positions)/npos if npos else 0.,'hit50':float(first<=50),'hit500':float(first<=500),'ndcg10':sum(1/np.log2(i+1) for i in positions if i<=10)/idcg if idcg else 0.,'mrr100':1/first if first<=100 else 0.,'mrr10':1/first if first<=10 else 0.,'candidates':len(ids),'hits100':sum(i<=100 for i in positions)})
   buffer.extend({'gid':int(row.gid),'note_idx':int(note),'score':float(score),'rank':int(rank)} for note,score,rank in ranked.itertuples(index=False,name=None))
   if len(buffer)>=100000:writer.write_table(pa.Table.from_pylist(buffer,schema=schema));buffer=[]
  if buffer:writer.write_table(pa.Table.from_pylist(buffer,schema=schema))
 frame=pd.DataFrame(rows);result={'all_requests':len(frame),'positive_pairs':int(frame.positives.sum()),'positive_hits500':int(frame.hits.sum()),'positive_hits100':int(frame.hits100.sum())}
 for name,col in [('HitRate@50','hit50'),('Recall@100','recall100'),('known_positive_NDCG@10','ndcg10'),('MRR@100','mrr100'),('Recall@500','recall500'),('HitRate@500','hit500')]:result[name]=float(frame[col].mean())
 return result,frame

def validate(out,model='unchanged_v3'):
 declared=verify(out);destination=out/model
 if destination.exists():raise FileExistsError('Fusion validation recipe replay frozen')
 destination.mkdir();requests=pd.read_parquet(OLD/'requests_validation.parquet');assert len(requests)==11893
 if model=='unchanged_v3':ann=OLD/'index/validation_nprobe32.parquet'
 else:
  source=out.parent/model;selected=json.loads((source/'selected.json').read_text())
  if not selected['accepted']:raise ValueError('C model did not pass predeclared ANN guards')
  ann=source/'index/validation_ann.parquet'
 paths={'ann':ann,'swing':OLD/'validation/hist025_fold200/swing.parquet','usercf':OLD/'validation/hist025_fold200/usercf.parquet'};graphitems=set(map(int,np.load(OLD/'cf_graph_0.25/items.npy')));trials=[]
 baseline=json.loads((OLD/'selection.json').read_text())['selected'];baseframe=None
 for policy in POLICIES:
  score,frame=replay(requests,paths,graphitems,policy,destination/f'{policy}.parquet');frame.to_parquet(destination/f'{policy}_requests.parquet',index=False)
  if policy=='baseline':
   baseframe=frame
   if model=='unchanged_v3':
    for key in ['HitRate@50','Recall@100','known_positive_NDCG@10']:assert abs(score[key]-baseline[key])<1e-12,(key,score[key],baseline[key])
  guard=score['Recall@100']>baseline['Recall@100'] and score['HitRate@50']>=baseline['HitRate@50'] and score['known_positive_NDCG@10']>=baseline['known_positive_NDCG@10'];delta={}
  for metric,col in [('HitRate@50','hit50'),('Recall@100','recall100'),('known_positive_NDCG@10','ndcg10')]:
   table=frame[['gid','user_idx']].copy();table['delta']=frame[col]-baseframe[col];delta[metric]={'delta':float(table.delta.mean()),'CI95':confidence(table,'delta')}
  entry={'model':model,'policy':policy,'metrics':score,'guard_passed_vs_v3':guard,'paired_vs_same_model_baseline':delta};trials.append(entry);print(json.dumps(entry),flush=True)
 result={'model':model,'baseline_v3':baseline,'trials':trials,'input_sha256':{str(path.resolve()):rec.digest(path) for path in paths.values()},'plan_sha256':rec.digest(out/'fusion_plan.json')};rec.write_json(destination/'validation.json',result);return result

def select(out):
 declared=verify(out)
 if (out/'selection.json').exists():raise FileExistsError('Fusion selection frozen')
 trials=[];bindings={}
 for model in ['unchanged_v3','masked_logged_hard']:
  path=out/model/'validation.json'
  if path.exists():d=json.loads(path.read_text());trials.extend(d['trials']);bindings[str(path.resolve())]=rec.digest(path)
 eligible=[r for r in trials if r['guard_passed_vs_v3'] and (r['policy']!='baseline' or r['model']!='unchanged_v3')];winner=max(eligible,key=lambda r:(r['metrics']['Recall@100'],r['metrics']['HitRate@50'],r['metrics']['known_positive_NDCG@10'])) if eligible else None;result={'accepted':winner is not None,'selected':winner,'trials':trials,'selection_split':'original_train_common_validation','frozen_before_official_test':True,'validation_sha256':bindings,'plan_sha256':rec.digest(out/'fusion_plan.json'),'code_sha256':declared['code_sha256']};rec.write_json(out/'selection.json',result);print(json.dumps(result,indent=2));return result

def full_test(out):
 """Fixed selected policy only, root approval required externally; no model/policy changes."""
 verify(out);selection=json.loads((out/'selection.json').read_text())
 if not selection['accepted']:raise ValueError('No accepted validation policy; do not open test')
 if rec.digest(out/'fusion_plan.json')!=selection['plan_sha256']:raise ValueError('Plan changed')
 for p,digest in selection['validation_sha256'].items():
  if rec.digest(Path(p))!=digest:raise ValueError('Frozen validation changed')
 target=out/'test';target.mkdir(exist_ok=False);chosen=selection['selected'];model=chosen['model'];policy=chosen['policy']
 if model=='unchanged_v3':ann=OLD/'test/ann.parquet'
 else:
  selected=json.loads((out.parent/model/'selected.json').read_text())['selected'];epoch=selected['epoch'];ann=out.parent/'diagnostic_test'/f'{model}_epoch_{epoch}'/'ann.parquet'
  if not ann.exists():raise ValueError('Run already-frozen C model diagnostic test before the frozen C+fusion test')
 ann_sha=rec.digest(ann)
 release=ROOT/'outputs/evaluation/northstar_fix_20261031/release_plan.json';release_sha=rec.digest(release) if release.exists() else None
 source=ROOT/'datasets/recommendation_test/train-00000-of-00001.parquet';source_sha=rec.digest(source)
 if source_sha!=json.loads((OLD/'manifest.json').read_text())['official_test_sha256']:raise ValueError('Official source changed')
 raw=pd.read_parquet(source,columns=['request_idx','user_idx','recent_clicked_note_idxs','rec_result_details_with_idx']);requests=pd.DataFrame({'gid':raw.request_idx,'user_idx':raw.user_idx,'history':raw.recent_clicked_note_idxs,'positives':raw.rec_result_details_with_idx.apply(lambda a:sorted({int(i['note_idx']) for i in a if i['click']>0}))});assert len(requests)==11115 and sum(requests.positives.map(len))==44683
 results={}
 for graphname,graphpath in [('fit',OLD/'test'),('allpast',OLD/'refit_cf_pretest')]:
  items=OLD/('cf_graph_0.25/items.npy' if graphname=='fit' else 'refit_cf_pretest/items.npy');paths={'ann':ann,'swing':graphpath/'swing.parquet','usercf':graphpath/'usercf.parquet'};score,rows=replay(requests,paths,set(map(int,np.load(items))),policy,target/f'{graphname}_selected.parquet');rows.to_parquet(target/f'{graphname}_requests.parquet',index=False);basepath=graphpath/'selected.parquet';base,brows=rec.metrics(requests,pd.read_parquet(basepath));delta={}
  for metric,col in [('HitRate@50','hit50'),('Recall@100','recall100'),('known_positive_NDCG@10','ndcg10')]:
   table=rows[['gid','user_idx']].copy();table['delta']=rows[col]-brows[col];delta[metric]={'delta':float(table.delta.mean()),'CI95':confidence(table,'delta')}
  results[graphname]={'metrics':score,'baseline_v3':base,'paired_vs_v3':delta}
 score=results['allpast']['metrics'];base=results['allpast']['baseline_v3'];passed=score['Recall@100']>base['Recall@100'] and score['HitRate@50']>=base['HitRate@50'] and score['known_positive_NDCG@10']>=base['known_positive_NDCG@10'];result={'selected':chosen,'selection_sha256':rec.digest(out/'selection.json'),'source_sha256':source_sha,'ann_source_sha256':ann_sha,'release_plan_sha256':release_sha,'results':results,'accepted_for_release':passed,'release_fallback':None if passed else 'unchanged_v3','parameter_choice_after_test':False};rec.write_json(target/'metrics.json',result);print(json.dumps(result,indent=2));return result

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=['plan','validate','select','test'],required=True);p.add_argument('--output',type=Path,default=DEFAULT);p.add_argument('--model',choices=['unchanged_v3','masked_logged_hard'],default='unchanged_v3');a=p.parse_args()
 if a.phase=='plan':plan(a.output)
 elif a.phase=='validate':validate(a.output,a.model)
 elif a.phase=='select':select(a.output)
 else:full_test(a.output)
if __name__=='__main__':main()
