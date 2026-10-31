"""Regressions at offline recall build and cache call sites."""
import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import faiss
import numpy as np
import pandas as pd
from recall import build_multiroute_recall as builder
from recall import cf_shared_index as cf
from recall.fusion import weighted_rrf

class MemoryWriter:
    latest=None
    def __init__(self,*args,**kwargs):
        self.rows=[];self.total_rows=0;MemoryWriter.latest=self
    def add(self,row):self.rows.append(row);self.total_rows+=1
    def close(self):return pd.DataFrame(self.rows)

class RecallRebuildContracts(unittest.TestCase):
    def run_builder(self,scene,history,request_vector=True):
        key='search_idx' if scene=='search' else 'request_idx'
        frame=pd.DataFrame([{key:1,'user_idx':7,'session_idx':3,
            'recent_clicked_note_idxs':history,'query':'test',
            'bm25_results':[[103,2.],[104,1.]]}])
        vectors=np.eye(2,dtype=np.float32);index=faiss.IndexFlatIP(2);index.add(vectors)
        replacements={
            '_load_requests':lambda *args:(frame,key),
            '_load_swing_index':lambda *args:{101:np.array([[102,1.]])},
            '_load_usercf_index':lambda *args:{7:np.array([[8,1.]])},
            '_load_user_items':lambda *args:{8:{102:1.}},
            '_load_ann_assets':lambda *args:(index,np.array([101,102]),{101:0,102:1},vectors,2),
            '_load_request_ann_vecs':lambda **kwargs:(vectors[:1],{1:0}) if request_vector else (None,None),
            'StreamingParquetWriter':MemoryWriter,'DISABLE_TQDM':True}
        with patch.multiple(builder,**replacements):
            builder.build_multiroute_recall(scene,'test','hard',500,1000,1000,1000,0,1.,['ann','swing','usercf'])
        return MemoryWriter.latest.rows

    def test_recommendation_builder_completes_three_real_routes(self):
        self.assertTrue(self.run_builder('rec',[101]))

    def test_search_provider_survives_absent_ann_and_history(self):
        rows=self.run_builder('search',[],request_vector=False)
        self.assertEqual({row[2] for row in rows},{103,104})

    def test_builder_uses_history_when_request_vectors_are_missing(self):
        rows=self.run_builder('search',[101],request_vector=False)
        self.assertTrue(any(row[8] for row in rows),'ANN history fallback must yield actual ANN source flags')

    def test_learned_request_vector_is_not_overridden_by_history_average(self):
        rows=self.run_builder('rec',[102],request_vector=True)
        ann_scores={row[2]:row[5] for row in rows}
        self.assertGreater(ann_scores[101],ann_scores[102])
    def test_l2_distance_is_converted_to_higher_is_better(self):
        vectors=np.eye(2,dtype=np.float32);index=faiss.IndexFlatL2(2);index.add(vectors)
        scores=builder._recall_ann('search',[],index,np.array([101,102]),{101:0,102:1},vectors,vectors[0],2)
        self.assertEqual([r[0] for r in weighted_rrf(scores,{},{},topk=2)],[101,102])

class CFCacheContracts(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.features=self.root/'features';self.index=self.root/'index'
        self.features.mkdir();self.index.mkdir()
        self.path=self.features/'rec_train_features.parquet'
        pd.DataFrame({'user_idx':[1,1],'note_idx':[0,20],'click':[1,0],
            'y_multi':[1.,2.],'event_timestamp':[10.,20.]}).to_parquet(self.path,index=False)
        p=patch.multiple(cf,FEAT_DIR=self.features,RESULT_DIR=self.index);p.start();self.addCleanup(p.stop)
    def test_parameters_invalidate_cache(self):
        cf.ensure_user_item_index('rec',interest_col='y_multi')
        current=cf.ensure_user_item_index('rec',interest_col='click')
        self.assertEqual(current['score_col'],'click');self.assertEqual(set(current['user_items'][1]),{0})
    def test_source_content_invalidates_cache(self):
        cf.ensure_user_item_index('rec',interest_col='click')
        pd.DataFrame({'user_idx':[1],'note_idx':[30],'click':[1],
            'y_multi':[3.],'event_timestamp':[30.]}).to_parquet(self.path,index=False)
        self.assertEqual(set(cf.ensure_user_item_index('rec',interest_col='click')['user_items'][1]),{30})
    def test_explicit_cutoff_excludes_future_edges_and_preserves_note_zero(self):
        current=cf.ensure_user_item_index('rec',interest_col='y_multi',cutoff_exclusive=15.)
        self.assertEqual(set(current['user_items'][1]),{0})

    def test_default_train_cutoff_is_applied(self):
        with patch.object(cf,'_official_cutoff',return_value=15.) as resolver:
            current=cf.ensure_user_item_index('rec')
        resolver.assert_called_once_with('rec','train')
        self.assertEqual(current['cache_contract']['cutoff_exclusive'],15.)
        self.assertEqual(set(current['user_items'][1]),{0})
    def test_minimum_interest_invalidates_cache(self):
        cf.ensure_user_item_index('rec')
        current=cf.ensure_user_item_index('rec',min_interest=1.5)
        self.assertEqual(set(current['user_items'][1]),{20})
if __name__=='__main__':unittest.main()