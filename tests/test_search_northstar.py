"""Search recall protocol tests without official test retrieval or large training."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import optimize_search_northstar as search
from evaluate_search_index import apply_user_profile_contract


class SearchNorthstarTests(unittest.TestCase):
    def test_all_requests_and_original_positive_denominators(self):
        requests=pd.DataFrame({'gid':[1,2],'user_idx':[7,8],'positives':[[0], [99]]})
        candidates=pd.DataFrame({'gid':[1]*11,'note_idx':list(range(1,11))+[0], 'score':list(range(11,0,-1)), 'rank':list(range(1,12))})
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'candidates.parquet';candidates.to_parquet(path,index=False)
            result,frame=search.candidate_metrics(requests,path)
        self.assertEqual(result['all_requests'],2)
        self.assertEqual(result['empty_requests'],1)
        self.assertEqual(result['Recall@100'],.5)
        self.assertEqual(result['HitRate@10'],0.)
        self.assertAlmostEqual(result['MRR@100'],1./22)
        self.assertEqual(result['known_positive_NDCG@10'],0.)

    def test_candidate_duplicates_are_not_silently_deduplicated(self):
        requests=pd.DataFrame({'gid':[1],'user_idx':[7],'positives':[[0]]})
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'bad.parquet'
            pd.DataFrame({'gid':[1,1],'note_idx':[0,0],'score':[2.,1.],'rank':[1,2]}).to_parquet(path,index=False)
            with self.assertRaisesRegex(ValueError,'Duplicate'):search.candidate_metrics(requests,path)

    def test_high_recall_policy_cannot_pass_a_head_quality_regression(self):
        baseline={'policy':{'strategy':'protected_headfill'},'validation':{'Recall@100':.5,'HitRate@10':.4,'MRR@10':.2,'known_positive_NDCG@10':.3}}
        bad={'policy':{'strategy':'weighted_rrf'},'validation':{'Recall@100':.8,'HitRate@10':.39,'MRR@10':.3,'known_positive_NDCG@10':.4}}
        good={'policy':{'strategy':'weighted_rrf'},'validation':{'Recall@100':.6,'HitRate@10':.4,'MRR@10':.2,'known_positive_NDCG@10':.3}}
        self.assertIs(search.guarded_fusion_choice([baseline,bad]),baseline)
        self.assertIs(search.guarded_fusion_choice([baseline,bad,good]),good)

    def test_protected_head_and_weight_zero_route_semantics(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            pd.DataFrame({'gid':[1,1,1],'note_idx':[2,0,3],'score':[3.,2.,1.],'rank':[1,2,3]}).to_parquet(root/'ann.parquet',index=False)
            pd.DataFrame({'gid':[1,1],'note_idx':[0,1],'score':[2.,1.],'rank':[1,2]}).to_parquet(root/'bm25.parquet',index=False)
            connection=search.fusion_connection(root/'ann.parquet',root/'bm25.parquet')
            try:
                search.policy_candidates(connection,{'strategy':'protected_headfill'},root/'base.parquet')
                self.assertEqual(pd.read_parquet(root/'base.parquet').note_idx.tolist(),[0,1,2,3])
                search.policy_candidates(connection,{'strategy':'weighted_rrf','weights':{'ann':0.},'rrf_k':10},root/'zero.parquet')
                self.assertEqual(pd.read_parquet(root/'zero.parquet').note_idx.tolist(),[0,1])
                search.policy_candidates(connection,{'strategy':'weighted_rrf','weights':{'ann':4.},'rrf_k':10},root/'mixed.parquet')
                result=pd.read_parquet(root/'mixed.parquet')
                self.assertEqual(int(result.note_idx.iloc[0]),0)
                self.assertTrue(result.score.is_monotonic_decreasing)
            finally:connection.close()

    def test_fit_only_codebook_sampling_preserves_ids_and_rejects_unknowns(self):
        result=search.codebook_positions(np.array([0,2,4,6]),np.array([0,4]),max_samples=2)
        self.assertEqual(set(result),{0,2})
        with self.assertRaises(ValueError):search.codebook_positions(np.array([0,2,4]),np.array([3]))
        with self.assertRaises(ValueError):search.codebook_positions(np.array([2,0,4]),np.array([0]))

    def test_profile_mask_applies_all42_fields_without_mutating_source(self):
        columns=['fans_num','follows_num']+[f'dense_feat{i}' for i in range(1,41)]
        frame=pd.DataFrame({name:[5.] for name in columns});frame['gender_enc']=2
        result=apply_user_profile_contract(frame,{'masked_user_dense':True})
        self.assertTrue((result[columns]==0).all().all())
        self.assertTrue((frame[columns]==5).all().all())
        self.assertEqual(result.gender_enc.iloc[0],2)
        same=apply_user_profile_contract(frame,{}, {'masked_user_dense':False})
        pd.testing.assert_frame_equal(same,frame)

    def test_model_choice_requires_all_declared_epochs_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'dssm').mkdir()
            pd.DataFrame({'search_idx':[1],'user_idx':[7],'positives':[[0]]}).to_parquet(root/'requests_validation.parquet',index=False)
            (root/'dssm/training_manifest.json').write_text(json.dumps({'epochs':3}))
            with self.assertRaisesRegex(RuntimeError,'not published'):search.select_epoch(root)

    def test_frozen_inputs_are_verified_before_final_test(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'input';source.write_bytes(b'frozen')
            frozen={'inputs_sha256':{str(source):search.digest(source)},'code_sha256':{}}
            (root/'selection.json').write_text(json.dumps(frozen))
            self.assertEqual(search.verify_frozen_selection(root),frozen)
            source.write_bytes(b'edited')
            with self.assertRaisesRegex(ValueError,'changed'):search.verify_frozen_selection(root)


if __name__=='__main__':unittest.main()