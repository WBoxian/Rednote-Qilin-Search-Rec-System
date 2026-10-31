"""Residual-prior numerics and validation/test gate regression tests."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from training.recall_prior import blend_recall_prior,recall_prior_terms
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import select_recall_prior as prior
import select_ranking_funnel as old


def entry(family,recall,hit=.4):
    metrics={name:.2 for name in old.METRICS};metrics.update({'Recall@100':recall,'HitRate@10':hit,'HitRate@50':hit})
    return {'family':family,'metrics':metrics,'coarse_cap':160,'weight':.5,
            'rank_prior':None if family=='incumbent' else {'alpha':.1,'k':60}}


class RecallPriorTests(unittest.TestCase):
    def test_zero_alpha_preserves_original_recall_and_nonzero_can_rerank(self):
        ranks=np.array([1,2,3]);scores=np.array([.1,.2,.3])
        zero=blend_recall_prior(ranks,scores,0.)
        np.testing.assert_array_equal(np.argsort(-zero),[0,1,2])
        mixed=blend_recall_prior(ranks,scores,4.)
        np.testing.assert_array_equal(np.argsort(-mixed),[2,1,0])
        self.assertEqual(mixed.dtype,np.float64)

    def test_learned_ties_use_original_rank_and_padded_rows_do_not_compete(self):
        ranks=np.array([[4,1,0],[1,2,3]])
        scores=np.array([[.5,.5,np.nan],[.1,.2,.3]])
        base,learned=recall_prior_terms(ranks,scores)
        self.assertEqual(learned[0,1],1/61)
        self.assertEqual(learned[0,0],1/62)
        self.assertTrue(np.isneginf(base[0,2]));self.assertEqual(learned[0,2],0.)
        np.testing.assert_array_equal(blend_recall_prior([],[],.5),np.array([],dtype=float))

    def test_invalid_ranks_parameters_and_valid_nan_score_fail(self):
        for ranks,scores in [([1,1],[.1,.2]),([1.5],[.2]),([-1],[.2]),([1],[np.nan])]:
            with self.assertRaises(ValueError):blend_recall_prior(ranks,scores,.1)
        for alpha in [-1,np.nan,np.inf]:
            with self.assertRaises(ValueError):blend_recall_prior([1],[.2],alpha)
        with self.assertRaises(ValueError):blend_recall_prior([1],[.2],.1,k=1.5)

    def test_joint_selection_keeps_better_incumbent_and_can_reject_everything(self):
        baseline=entry('raw_control',.1)['metrics']
        incumbent=entry('incumbent',.3);weaker=entry('recall_prior',.2);badhead=entry('recall_prior',.5,hit=.39)
        self.assertIs(prior.choose_joint([incumbent,weaker,badhead],baseline,'search'),incumbent)
        self.assertIsNone(prior.choose_joint([entry('recall_prior',.1)],baseline,'search'))
        stronger=entry('recall_prior',.31)
        self.assertIs(prior.choose_joint([incumbent,stronger],baseline,'search'),stronger)

    def test_matrix_stage_metrics_match_original_operational_recipe(self):
        requests=pd.DataFrame({'search_idx':[3,1,2],'user_idx':[9,7,8],'positive_note_ids':[[7],[0],[22,33]]})
        features=pd.DataFrame({'search_idx':[1,1,1,2,2,2],'note_idx':[0,2,3,22,44,33],'rank':[1,2,3,1,2,3]})
        values=features.assign(coarse_score=[.2,.8,.4,.4,.1,.9],dien_raw_score=[2.,-2.,0.,1.,-2.,3.])
        packed=prior.pack_candidates(features,requests,'search',3)
        coarse,fine=prior.model_matrices(packed,values,'search')
        recipe={'coarse_cap':2,'weight':.5,'rank_prior':None,'score_contract':'probability_v2'}
        orders=prior.candidate_order(packed,coarse,fine,recipe)
        actual=prior.metric_frame(packed,orders['final']).set_index('gid').sort_index()
        expected=old.evaluate_recipe(values,requests,'search',2,.5,score_contract='probability_v2').set_index('gid').sort_index()
        for metric in old.METRICS:np.testing.assert_allclose(actual[metric],expected[metric],atol=1e-14,rtol=0)
        raw=prior.metric_frame(packed,np.broadcast_to(np.arange(3),packed['ids'].shape))
        self.assertEqual(raw.candidates.tolist(),[3,3,0])
        self.assertEqual(raw.known_positive_pairs_pool.tolist(),[1,2,0])

    def test_plan_is_bound_before_reading_new_candidate_metrics(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);cached=root/'cached';cached.mkdir()
            (cached/'prepare_manifest.json').write_text(json.dumps({'scene':'search','phase':'validation','input_cap':160,'coarse_caps':[80,120,160],'runs':['first','second']}))
            incumbent=entry('incumbent',.3)
            (cached/'selection.json').write_text(json.dumps({'scene':'search','accepted':True,'selected':incumbent}))
            args=SimpleNamespace(scene='search',cached=cached,output=root/'new')
            with patch.object(prior.pd,'read_parquet',side_effect=AssertionError('No new validation data before plan')):
                plan=prior.plan(args)
            self.assertFalse(plan['new_grid_metrics_read'])
            self.assertEqual(plan['incumbent'],incumbent)
            self.assertEqual(plan['alphas'],[.05,.1,.25,.5,1.,2.,4.])
            self.assertTrue(plan['code_sha256'])
            with self.assertRaises(FileExistsError):prior.plan(args)

    def test_rejected_validation_never_opens_official_test(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'selection.json'
            path.write_text(json.dumps({'scene':'search','frozen_before_official_test':True,
                'selection_split':'original_train_common_forward_validation','accepted':False,'selected':None}))
            args=SimpleNamespace(selection=path,scene='search',output=Path(folder))
            with patch.object(old,'original_requests') as original:
                with self.assertRaisesRegex(ValueError,'No accepted'):prior.test(args)
                original.assert_not_called()

    def test_final_candidate_output_preserves_real_item_zero_and_original_ranks(self):
        requests=pd.DataFrame({'search_idx':[1],'user_idx':[7],'positive_note_ids':[[0]]})
        features=pd.DataFrame({'search_idx':[1,1],'note_idx':[0,1],'rank':[1,2]})
        values=features.assign(coarse_score=[.1,.9],dien_raw_score=[0.,1.])
        packed=prior.pack_candidates(features,requests,'search',2);coarse,fine=prior.model_matrices(packed,values,'search')
        recipe={'coarse_cap':2,'weight':.5,'rank_prior':{'alpha':.1,'k':60},'score_contract':'probability_v2'}
        orders=prior.candidate_order(packed,coarse,fine,recipe)
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'candidates.parquet';prior.save_final_candidates(output,packed,orders,'search')
            frame=pd.read_parquet(output)
        self.assertIn(0,frame.note_idx.tolist())
        self.assertEqual(frame['rank'].tolist(),[1,2])
        self.assertEqual(set(frame.original_recall_rank),{1,2})


if __name__=='__main__':unittest.main()