"""Causal stacking regression tests at export/split seams without model training."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from training import gbdt_ranker as coarse
from training.dien_ranker import _stack_train_validation_split, _eligible_stack_groups, _all_request_ndcg


def frame(groups=20):
    return pd.DataFrame([{'search_idx':gid,'user_idx':7,'note_idx':gid*2+item,'click':item,'y_multi':float(item),
        'event_timestamp':1732700000.+gid*10,'image_num':float(item+1)} for gid in range(groups) for item in [0,1]])


class ForwardStackingTests(unittest.TestCase):
    def build(self):
        data=frame();calls=[]
        def predict(past,future,features,key):
            self.assertLess(past.event_timestamp.max(),future.event_timestamp.min())
            self.assertFalse(set(past[key])&set(future[key]))
            calls.append((set(past[key]),set(future[key])))
            return future.image_num.to_numpy(),future.image_num.to_numpy()*2
        scored,audit,first,second=coarse.build_forward_stacking_frame(
            data,'search_idx',['image_num'],np.arange(32),np.arange(32,40),np.ones(8),np.ones(8)*2,fit_predict=predict)
        return data,scored,audit,first,second,calls

    def test_forward_scores_exclude_warmup_and_preserve_heldout_partition(self):
        data,scored,audits,first,second,calls=self.build()
        self.assertEqual(len(calls),3)
        self.assertEqual(len(scored),32)
        self.assertEqual(set(scored.loc[scored.stack_partition=='fit','search_idx']),set(range(4,16)))
        self.assertEqual(set(scored.loc[scored.stack_partition=='validation','search_idx']),set(range(16,20)))
        self.assertTrue(np.isnan(first[:8]).all())
        self.assertTrue((scored.coarse_fit_end<scored.event_timestamp).all())
        self.assertTrue((scored.stack_contract=='forward_coarse_v1').all())
        fitting,validation=_stack_train_validation_split(scored,'search_idx')
        self.assertEqual(set(scored.iloc[validation].search_idx),set(range(16,20)))
        self.assertTrue(set(scored.iloc[fitting].search_idx).isdisjoint(scored.iloc[validation].search_idx))

    def test_bad_coarse_lineage_and_missing_partition_are_rejected(self):
        _,scored,*_=self.build()
        bad=scored.copy();bad.loc[bad.index[0],'coarse_fit_end']=bad.loc[bad.index[0],'event_timestamp']
        with self.assertRaisesRegex(ValueError,'in-sample'):_stack_train_validation_split(bad,'search_idx')
        with self.assertRaisesRegex(ValueError,'Missing forward'):_stack_train_validation_split(scored.drop(columns=['stack_partition']),'search_idx')
        bad=scored.copy();bad.loc[bad.index[0],'stack_partition']='validation'
        with self.assertRaisesRegex(ValueError,'crosses'):_stack_train_validation_split(bad,'search_idx')

    def test_validation_requests_are_not_refiltered_by_positive_or_group_size(self):
        data=pd.DataFrame({'search_idx':[1,1,2], 'y_multi':[1.,0.,0.], 'stack_partition':['fit','fit','validation']})
        retained=_eligible_stack_groups(data,'search_idx')
        self.assertEqual(set(retained.search_idx),{1,2})
        score=_all_request_ndcg(np.array([1.,0.,0.]),np.array([1.,0.,0.]),np.array([1,1,2]),10)
        self.assertAlmostEqual(score,.5)

    def test_dien_export_never_injects_validation_positive_beyond_budget(self):
        data=pd.DataFrame({'search_idx':[1,1,2,2], 'note_idx':[10,11,20,21], 'click':[0,1,0,1], 'y_multi':[0.,1.,0.,1.],
            'preranking_rank':[1,2,1,2], 'preranking_score':[2.,1.,2.,1.], 'lgb_score':[2.,1.,2.,1.], 'xgb_score':[2.,1.,2.,1.],
            'stack_partition':['fit','fit','validation','validation'], 'stack_contract':['forward_coarse_v1']*4,
            'coarse_fit_end':[1.,1.,2.,2.], 'event_timestamp':[2.,2.,3.,3.]})
        with tempfile.TemporaryDirectory() as folder,patch.object(coarse,'OUT_DATA_DIR',Path(folder)):
            output,_=coarse._export_dien_dataset(data,'search','easy','search_idx',1,True)
            saved=pd.read_parquet(output)
        self.assertEqual(set(saved[saved.stack_partition=='fit'].note_idx),{10,11})
        self.assertEqual(saved[saved.stack_partition=='validation'].note_idx.tolist(),[20])

    def test_complete_source_interval_controls_official_cutoff(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'datasets/search_test').mkdir(parents=True);(root/'features').mkdir()
            pd.DataFrame({'search_result_details_with_idx':[[{'search_timestamp':200.}]]}).to_parquet(root/'datasets/search_test/train-00000-of-00001.parquet',index=False)
            pd.DataFrame({'search_idx':[1,1,2], 'event_timestamp':[100.,300.,90.]}).to_parquet(root/'features/search_train_features.parquet',index=False)
            candidate=pd.DataFrame({'search_idx':[1,2], 'event_timestamp':[100.,90.]})
            with patch.object(coarse,'BASE_DIR',root),patch.object(coarse,'FEATURE_DIR',root/'features'):
                safe=coarse._filter_before_official_test(candidate,'search')
        self.assertEqual(safe.search_idx.tolist(),[2])

    def test_prepare_preserves_epoch_precision_and_nan_clicks(self):
        data=frame(2);data.loc[1,'y_multi']=np.nan;data.loc[1,'event_timestamp']+=1
        prepared=coarse._prepare_training_frame(data.copy(),'search')
        np.testing.assert_array_equal(prepared.event_timestamp,data.event_timestamp)
        self.assertEqual(prepared.loc[prepared.note_idx==1,'y_multi'].iloc[0],1.)

    def test_default_fold_trainers_only_receive_past_labels(self):
        past=frame(6);future=frame(2);future['search_idx']+=10;future['event_timestamp']+=10000;future['y_multi']=999999.
        seen=[]
        class LGB:
            def predict(self,data):return np.ones(len(data))
        class XGB:
            def predict(self,data,**kwargs):return np.ones(data.num_row())
        def lgb_train(params,dataset,**kwargs):
            seen.append(np.asarray(dataset.label))
            self.assertEqual(params['num_threads'],2)
            return LGB()
        def xgb_train(params,dataset,**kwargs):
            seen.append(dataset.get_label());self.assertEqual(params['nthread'],2)
            return XGB()
        with patch.object(coarse.lgb,'train',side_effect=lgb_train),patch.object(coarse.xgb,'train',side_effect=xgb_train):
            first,second=coarse._fit_predict_coarse_fold(past,future,['image_num'],'search_idx')
        self.assertEqual(len(first),len(future));self.assertEqual(len(second),len(future))
        self.assertTrue(all(np.max(labels)<=5 for labels in seen))

    def test_hard_negatives_exclude_validation_and_known_positive_pairs(self):
        scored=pd.DataFrame({'search_idx':[1,1,2,2], 'note_idx':[10,11,20,21], 'click':[0,1,0,1],
            'preranking_rank':[1,2,1,2], 'stack_partition':['fit','fit','validation','validation']})
        candidates=pd.DataFrame({'search_idx':[1,1,1,2,2], 'note_idx':[10,11,12,20,22], 'rank':[1,2,3,1,2]})
        with tempfile.TemporaryDirectory() as folder,patch.object(coarse,'OUT_DATA_DIR',Path(folder)):
            source=Path(folder)/'candidates.parquet';candidates.to_parquet(source,index=False)
            output=coarse._export_easy_preranking_complement_hard_neg('search','easy','search_idx',source,scored,1,1000,10)
            saved=pd.read_parquet(output)
        self.assertEqual(saved.search_idx.tolist(),[1]);self.assertEqual(saved.hard_neg_note_idxs.iloc[0].tolist(),[12])


if __name__=='__main__':unittest.main()