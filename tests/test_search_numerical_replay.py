import copy,tempfile,unittest,json
from pathlib import Path
from unittest.mock import patch
import numpy as np,pandas as pd
from scripts import replay_search_numerical_contract as replay

class SearchNumericalReplayTests(unittest.TestCase):
 def meta(self):return {'scene':'search','features':['f'],'dien':{'score_contract':'probability_v2','chosen':{'weight':1.,'original_field':'keep'},'rank_prior':{'alpha':1.,'k':60}}}
 def test_rank_copy_changes_only_inference_contract_and_refuses_recipe_changes(self):
  before=self.meta();after=replay.fixed_rank_metadata(before);expected=copy.deepcopy(before);expected['dien']['inference_contract']='request_fp32_v1';self.assertEqual(after,expected);self.assertNotIn('inference_contract',before['dien']);bad=self.meta();bad['dien']['chosen']['weight']=.5
  with self.assertRaises(ValueError):replay.fixed_rank_metadata(bad)
 def test_raw_truth_denominator_real_zero_and_empty_requests(self):
  hit=replay.metric_row(1,3,[0,0,9],[0]);miss=replay.metric_row(2,4,[8],[]);s=replay.summarize([hit,miss]);self.assertEqual(hit['Recall@100'],.5);self.assertEqual(s['Recall@100'],.25);self.assertEqual(s['requests'],2);self.assertEqual(s['empty_candidate_requests'],1)
 def test_validation_strict_gain_test_nondegradation_same_old_guards(self):
  baseline={k:.2 for k in ['Recall@100','HitRate@10','known_positive_NDCG@10','MRR@100']};self.assertFalse(replay.guard(baseline,baseline,True));self.assertTrue(replay.guard(baseline,baseline,False));good={**baseline,'Recall@100':.3};self.assertTrue(replay.guard(good,baseline,True));good['MRR@100']=.19;self.assertFalse(replay.guard(good,baseline,True))
 def test_missing_or_rejected_validation_prevents_any_official_raw_read(self):
  with tempfile.TemporaryDirectory() as tmp:
   out=Path(tmp)
   with patch.object(replay,'verify',return_value={}),patch.object(replay,'read_original') as source:
    with self.assertRaises(ValueError):replay.run_phase(out,'test','cpu')
    source.assert_not_called()
 def test_actual_replay_adapter_uses_current_inputs_and_shared_scores_no_labels(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);out=root/'out';out.mkdir();old=root/'old';old.mkdir();query='测'*600;raw=pd.DataFrame({'search_idx':[1,2],'user_idx':[7,8],'query':[query,'second'],'recent_clicked_note_idxs':[[],[0]],'search_result_details_with_idx':[[{'note_idx':0,'click':1},{'note_idx':9,'click':1}],[{'note_idx':99,'click':1}]]});source=root/'raw.parquet';raw.to_parquet(source,index=False);manifest={'rank_dir':'rank','query_encoder_dir':'encoder','query_encoder_revision':'pin','tag':'hard'};(out/'bundle.json').write_text(json.dumps(manifest));(out/'plan.json').write_text('{}');selection=old/'selection.json';selection.write_text('{}');plan={'staged_bundle':str(out/'bundle.json'),'original_selection_sha256':'oldsha','recipe':{'weight':1.,'rank_prior':{'alpha':1,'k':60}},'recall_policy':{'weights':{'ann':.5,'bm25':1}},'inputs':{},'code_sha256':{},'limitations':[]}
   for stage in ['raw','coarse','fine','final']:pd.DataFrame([replay.metric_row(1,7,[0,9],[0,1,2]),replay.metric_row(2,8,[99],[3])]).to_parquet(old/f'validation_{stage}_requests.parquet',index=False)
   calls=[]
   class Ranker:
    inference_contract='request_fp32_v1'
    def __init__(self,*args,**kwargs):pass
    def score_request_frame(self,frame):
     selftest.assertTrue({'search_idx','user_idx','query','raw_query','recent_clicked_note_idxs','note_idx','rank'}.issubset(frame));selftest.assertFalse(any(k in frame for k in ['click','positives','positive_note_ids','search_result_details_with_idx']));calls.append(frame[['search_idx','user_idx','query','recent_clicked_note_idxs']].copy());result=frame.copy();result['feature_example']=np.arange(len(frame),dtype=np.float32);result['coarse_score']=np.full(len(frame),.3,np.float32);result['dien_raw_score']=np.zeros(len(frame),np.float32);result['final_score']=2./(60.+result['rank'].to_numpy());return result
   class Bundle:
    def __init__(self,manifest,ranker,root):self.manifest=manifest;self.recall_dir=root/'recall';self.bundle_id='test'
    def _get_bm25_index(self):return None
   class DSSM:
    enabled=True
    def __init__(self,*args,**kwargs):selftest.assertEqual(kwargs['inference_contract'],'request_fp32_v1')
   def fetch(bundle,dssm,context,**kwargs):
    row=context.iloc[0];selftest.assertEqual(set(context),{'search_idx','user_idx','query','raw_query','recent_clicked_note_idxs'});ids=[0,1,2] if row.search_idx==1 else [3];result=pd.DataFrame({'search_idx':int(row.search_idx),'user_idx':int(row.user_idx),'note_idx':ids,'rank':np.arange(1,len(ids)+1),'recall_score':1./np.arange(1,len(ids)+1)});result.attrs['route_status']={'errors':{},'counts':{'ann':len(ids),'bm25':0}};return result
   selftest=self
   with patch.object(replay,'ROOT',root),patch.object(replay,'OLD_GATE',old),patch.object(replay,'verify',return_value=plan),patch.object(replay,'read_original',return_value=(raw,source)),patch('backend.online.bundle.validate_manifest'),patch('backend.online.bundle.ServingBundle',Bundle),patch('backend.online.ranking.stack.RankStackPredictor',Ranker),patch('backend.online.recall.dssm.DSSMRecaller',DSSM),patch('backend.online.recall.service.fetch_bundle_recall_candidates',side_effect=fetch):result=replay.run_phase(out,'validation','cpu')
   self.assertEqual(result['all_requests'],2);self.assertEqual(result['stages']['raw']['Recall@100'],.25);self.assertEqual(len(calls),2);self.assertEqual(calls[0].iloc[0]['query'],query);scored=pd.read_parquet(out/'validation/scores.parquet');self.assertEqual(scored.recent_clicked_note_idxs.iloc[-1].tolist(),[0]);self.assertIn('originalrank',scored);self.assertEqual(len(pd.read_parquet(out/'validation/final_candidates.parquet')),4)
   with patch.object(replay,'verify',return_value=plan),patch.object(replay,'read_original') as sourcecall:
    with self.assertRaises(FileExistsError):replay.run_phase(out,'validation','cpu')
    sourcecall.assert_not_called()
if __name__=='__main__':unittest.main()
