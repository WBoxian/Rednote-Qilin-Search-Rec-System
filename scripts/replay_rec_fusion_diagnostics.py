"""Frozen F1 main / F2 fixed diagnostic replay with corrected immutable V3 fit-CF paths."""
from pathlib import Path
import json
import numpy as np,pandas as pd
from scripts import rec_iteration_fusion as fusion
from scripts import optimize_rec_retrieval as rec
from backend.offline.evaluate_full_test import confidence
ROOT=fusion.ROOT;OLD=fusion.OLD;OUT=fusion.DEFAULT;DEST=OUT/'test_replay_v2'
fusion.verify(OUT);selection=json.loads((OUT/'selection.json').read_text());assert selection['accepted'] and selection['selected']['model']=='unchanged_v3';main=selection['selected']['policy'];assert main=='filter_history'
if DEST.exists():raise FileExistsError('Corrected fixed-policy replay already frozen')
DEST.mkdir();rawpath=ROOT/'datasets/recommendation_test/train-00000-of-00001.parquet';ann=OLD/'test/ann.parquet';graphs={'fit':{'cf':OLD/'test/cf/hist025_fold200','items':OLD/'cf_graph_0.25/items.npy','baseline':OLD/'test/selected.parquet'},'allpast':{'cf':OLD/'refit_cf_pretest','items':OLD/'refit_cf_pretest/items.npy','baseline':OLD/'refit_cf_pretest/selected.parquet'}};sources=[rawpath,ann,OUT/'selection.json',OUT/'fusion_plan.json',Path(__file__)]
for info in graphs.values():sources.extend([info['cf']/'swing.parquet',info['cf']/'usercf.parquet',info['items'],info['baseline']])
manifest={'scope':'Correct immutable source-path adapter only; original runner stopped at missing fit-CF path BEFORE candidate/metric creation. Frozen F1 selection remains unchanged; F2 is preapproved fixed diagnostic only. No test model/policy choice.','main_policy':main,'diagnostic_policies':['filter_history_context'],'input_sha256':{str(p.resolve()):rec.digest(p) for p in sources},'frozen_selection_sha256':rec.digest(OUT/'selection.json'),'original_source_unchanged':True};rec.write_json(DEST/'run_manifest.json',manifest)
raw=pd.read_parquet(rawpath,columns=['request_idx','user_idx','recent_clicked_note_idxs','rec_result_details_with_idx']);requests=pd.DataFrame({'gid':raw.request_idx,'user_idx':raw.user_idx,'history':raw.recent_clicked_note_idxs,'positives':raw.rec_result_details_with_idx.apply(lambda a:sorted({int(i['note_idx']) for i in a if i['click']>0}))});assert len(requests)==11115 and sum(requests.positives.map(len))==44683;results={}
for graph,info in graphs.items():
 base,brows=rec.metrics(requests,pd.read_parquet(info['baseline']));graphitems=set(map(int,np.load(info['items'])));paths={'ann':ann,'swing':info['cf']/'swing.parquet','usercf':info['cf']/'usercf.parquet'};results[graph]={'baseline':base,'policies':{}}
 for policy in [main,'filter_history_context']:
  score,rows=fusion.replay(requests,paths,graphitems,policy,DEST/f'{graph}_{policy}.parquet');rows.to_parquet(DEST/f'{graph}_{policy}_requests.parquet',index=False);delta={}
  for metric,col in [('HitRate@50','hit50'),('Recall@100','recall100'),('known_positive_NDCG@10','ndcg10'),('MRR@100','mrr100')]:
   table=rows[['gid','user_idx']].copy();table['delta']=rows[col]-brows[col];delta[metric]={'delta':float(table.delta.mean()),'CI95':confidence(table,'delta')}
  result={'metrics':score,'paired_vs_v3':delta,'deployment_eligible':policy==main};results[graph]['policies'][policy]=result;print(json.dumps({'graph':graph,'policy':policy,**result}),flush=True)
score=results['allpast']['policies'][main]['metrics'];base=results['allpast']['baseline'];passed=score['Recall@100']>base['Recall@100'] and score['HitRate@50']>=base['HitRate@50'] and score['known_positive_NDCG@10']>=base['known_positive_NDCG@10'];result={'all_requests':11115,'unique_clicked_pairs':44683,'frozen_selection_sha256':manifest['frozen_selection_sha256'],'run_manifest_sha256':rec.digest(DEST/'run_manifest.json'),'main_policy':main,'results':results,'accepted_for_release':passed,'release_fallback':None if passed else 'unchanged_v3','parameter_choice_after_test':False,'failed_original_attempt':str(OUT/'test')};rec.write_json(DEST/'metrics.json',result);print(json.dumps({'accepted_for_release':passed,'main':score}),flush=True)
