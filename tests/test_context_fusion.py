import tempfile,unittest
from pathlib import Path
import numpy as np,pandas as pd
from recall.context_fusion import fuse_request_routes
from scripts.rec_iteration_fusion import replay

class ContextFusionTests(unittest.TestCase):
 def test_filter_preserves_real_zero_identity_and_removes_consumed_only(self):
  routes={'ann':np.array([0,1,2]),'swing':np.array([0,3]),'usercf':np.array([],np.int64)}
  base=fuse_request_routes(routes,[],{0},'baseline');self.assertIn(0,base.note_idx.tolist())
  filtered=fuse_request_routes(routes,[0],{0},'filter_history');self.assertNotIn(0,filtered.note_idx.tolist());self.assertEqual(set(filtered.note_idx),{1,2,3});self.assertAlmostEqual(float(filtered.iloc[0].score),2/61)
 def test_shared_recent_prefix_not_suffix_is_consumed(self):
  result=fuse_request_routes({'ann':np.array([0,19,20,24])},list(range(25)),set(),'filter_history');self.assertEqual(result.note_idx.tolist(),[20,24])
 def test_context_coverage_threshold_and_duplicate_denominator(self):
  routes={'ann':np.array([1]),'swing':np.array([2]),'usercf':np.array([3])}
  high=fuse_request_routes(routes,[10,11],{10},'filter_history_context');self.assertEqual(high.iloc[0].note_idx,2)
  low=fuse_request_routes(routes,[10,10,11],{10},'filter_history_context');self.assertEqual(low.note_idx.tolist(),[1,2,3]);self.assertAlmostEqual(float(low.iloc[0].score),1/61)
 def test_route_budget_dedup_and_deterministic_ties(self):
  value=fuse_request_routes({'ann':np.array([2,1]),'swing':np.array([1,2])},[],set(),'baseline');self.assertEqual(value.note_idx.tolist(),[1,2]);self.assertEqual(value['rank'].tolist(),[1,2])
  with self.assertRaises(ValueError):fuse_request_routes({'ann':np.arange(1001)},[],set(),'baseline')
  with self.assertRaises(ValueError):fuse_request_routes({'ann':np.array([1,1])},[],set(),'baseline')
 def test_replay_keeps_history_target_and_empty_request_in_denominator(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);paths={}
   for route in ['ann','swing','usercf']:
    path=root/f'{route}.parquet';pd.DataFrame({'gid':[1],'note_idx':[0],'rank':[1]}).to_parquet(path,index=False);paths[route]=path
   req=pd.DataFrame({'gid':[1,2],'user_idx':[3,4],'history':[[0],[]],'positives':[[0],[99]]});score,rows=replay(req,paths,{0},'filter_history',root/'result.parquet')
   self.assertEqual(score['all_requests'],2);self.assertEqual(score['positive_pairs'],2);self.assertEqual(score['Recall@100'],0.);self.assertEqual(rows.positives.tolist(),[1,1]);self.assertEqual(rows.candidates.tolist(),[0,0])
if __name__=='__main__':unittest.main()
