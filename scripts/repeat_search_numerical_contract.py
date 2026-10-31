"""Read-only fresh 64-request replay of a completed canonical Search phase."""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT))
from replay_search_numerical_contract import DEFAULT,CONTRACT,load,save,verify,read_original,sha256

def run(phase,run_dir=DEFAULT,output_dir=None):
 out=Path(run_dir).resolve();AUDIT=Path(output_dir or out/'verification').resolve();AUDIT.mkdir(parents=True,exist_ok=True);plan=verify(out);dest=out/phase
 target=AUDIT/f'{phase}_reproducibility.json'
 if target.exists():raise FileExistsError(target)
 metrics=load(dest/'metrics.json');raw,source=read_original(phase)
 chosen=list(np.linspace(0,len(raw)-1,64,dtype=int));ids=set(map(int,raw.iloc[chosen].search_idx))
 for gid in [55858,56477]:
  if gid in set(raw.search_idx) and gid not in ids:ids.remove(sorted(ids)[-1]);ids.add(gid)
 raw=raw[raw.search_idx.isin(ids)].sort_values('search_idx')
 raw_saved=pd.read_parquet(dest/'raw_selected.parquet',filters=[('gid','in',list(ids))]);score_saved=pd.read_parquet(dest/'scores.parquet',filters=[('search_idx','in',list(ids))])
 from backend.online.bundle import ServingBundle,validate_manifest
 from backend.online.pipeline import SceneServingState,OnlineScenePipeline
 from unittest.mock import patch
 from backend.online.ranking.stack import RankStackPredictor
 from backend.online.recall.dssm import DSSMRecaller
 from backend.online.recall.service import fetch_bundle_recall_candidates
 from backend.online.ranking.inference import configure_request_inference
 manifest=load(plan['staged_bundle']);validate_manifest(manifest,'search',ROOT,tag=manifest['tag']);configure_request_inference(CONTRACT)
 ranker=RankStackPredictor('search',ROOT/manifest['rank_dir'],root=ROOT,device='cuda',query_encoder_dir=ROOT/manifest['query_encoder_dir'],query_encoder_revision=manifest['query_encoder_revision']);bundle=ServingBundle(manifest,ranker,ROOT);bundle._get_bm25_index()
 with patch('backend.online.bundle.get_active_bundle',return_value=bundle):state=SceneServingState('search',tag=manifest['tag'])
 if not state.bundle_active:raise AssertionError(state.bundle_errors)
 pipeline=OnlineScenePipeline(state)
 dssm=DSSMRecaller('search',manifest['tag'],bundle.recall_dir,'search_idx',query_encoder_dir=ROOT/manifest['query_encoder_dir'],query_encoder_revision=manifest['query_encoder_revision'],use_shared_note_vectors=True,inference_contract=CONTRACT)
 rows=[]
 for rec in raw.to_dict('records'):
  gid=int(rec['search_idx']);query=str(rec['query'] or '').strip();hist=list(map(int,rec['recent_clicked_note_idxs']))
  context=state.live_context_builder.build(int(rec['user_idx']),query,hist);context['search_idx']=gid;context['raw_query']=query
  fresh=fetch_bundle_recall_candidates(bundle,dssm,context,scene='search',max_rank=500).sort_values('rank',kind='stable')
  if fresh.attrs.get('route_status',{}).get('errors'):raise AssertionError('Recall route failure')
  saved=raw_saved[raw_saved.gid==gid].sort_values('rank')
  for a,b in [(fresh.note_idx.to_numpy(),saved.note_idx.to_numpy()),(fresh['rank'].to_numpy(),saved['rank'].to_numpy()),(fresh.recall_score.to_numpy(np.float64),saved.score.to_numpy(np.float64))]:np.testing.assert_array_equal(a,b)
  frame=fresh.head(160).copy();frame['query']=query;frame['raw_query']=query;frame['recent_clicked_note_idxs']=[list(context.iloc[0].recent_clicked_note_idxs) for _ in range(len(frame))]
  result=ranker.score_request_frame(frame).sort_values('rank');old=score_saved[score_saved.search_idx==gid].sort_values('rank')
  np.testing.assert_array_equal(result.note_idx,old.note_idx)
  diffs={}
  for col in ['coarse_score','dien_raw_score','final_score']:
   np.testing.assert_array_equal(result[col].to_numpy(),old[col].to_numpy());diffs[col]=float(np.max(np.abs(result[col].to_numpy()-old[col].to_numpy()),initial=0))
  reverse=ranker.score_request_frame(frame.iloc[::-1].reset_index(drop=True)).sort_values('rank')
  for col in ['coarse_score','dien_raw_score','final_score']:np.testing.assert_array_equal(reverse[col].to_numpy(),old[col].to_numpy())
  captured={};original_fine=ranker._fine_logits
  def capture(frame,context):
   raw=original_fine(frame,context);captured['frame']=frame.copy();captured['raw']=raw.copy();return raw
  minimal=pd.DataFrame([{'search_idx':gid,'user_idx':int(rec['user_idx']),'query':query,'recent_clicked_note_idxs':hist}])
  with patch.object(ranker,'_fine_logits',side_effect=capture):response=pipeline.build_feed(int(rec['user_idx']),query,1,30,record_exposure=False,request_context=minimal);actual=pipeline._get_cached_feed(int(rec['user_idx']),int(response['context_id'][:15],16),query)[0];ordered=result.sort_values(['final_score','rank'],ascending=[False,True],kind='stable');np.testing.assert_array_equal(ordered.note_idx,actual.note_idx)
  a=result.set_index('note_idx');z=actual.set_index('note_idx')
  np.testing.assert_array_equal(a.final_score,z.loc[a.index,'final_score']);f=captured['frame'].copy();f['dien_raw_score']=captured['raw'];f=f.set_index('note_idx')
  for col in ['coarse_score','dien_raw_score']:np.testing.assert_array_equal(a[col],f.loc[a.index,col])
  rows.append({'actual_build_feed_full_order_scores_bitwise_equal':True,'profile_categories':{k:int(context.iloc[0][k]) for k in ['gender_enc','platform_enc','age_enc','location_enc']},'gid':gid,'raw_rows':len(saved),'score_rows':len(old),'fresh_recall_exact':True,'fresh_scores_exact':True,'reverse_input_scores_exact':True,'max_abs_errors':diffs})
  print(json.dumps({'done':len(rows),'total':len(raw),'gid':gid}),flush=True)
 inputs={str(p.resolve()):sha256(p) for p in [out/'plan.json',dest/'metrics.json',dest/'raw_selected.parquet',dest/'scores.parquet',Path(__file__),source]};inputs.update(plan['code_sha256'])
 proof={'status':'PASS','scene':'search','phase':phase,'inference_contract':CONTRACT,'selection_sha256':sha256(out/'plan.json'),'requests':len(rows),'actual_build_feed_full_order_and_scores_bitwise_equal':True,'input_contract':'LiveContextBuilder_bound_user_profile_v2','fresh_raw_and_full_score_bitwise_equal':True,'reverse_input_full_score_bitwise_equal':True,'rows':rows,'inputs_sha256':inputs,'scope':'Independent process and current raw header reconstruction; 64 full fresh recall500/canonicalfine160 repeats; no candidate or query-vector snapshot passed to model; exact array equality'}
 save(target,proof);print(json.dumps({'status':'PASS','path':str(target),'requests':len(rows)}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['validation','test'],required=True);p.add_argument('--run',type=Path,default=DEFAULT);p.add_argument('--output-dir',type=Path);a=p.parse_args();run(a.phase,a.run,a.output_dir)
