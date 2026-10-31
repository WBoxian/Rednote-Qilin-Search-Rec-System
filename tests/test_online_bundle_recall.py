import unittest
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pandas as pd

from backend.online.recall.dssm import DSSMRecaller
from backend.online.recall.service import fetch_bundle_recall_candidates


class BundleRecallTests(unittest.TestCase):
    def test_f1_filters_only_recent20_after_original_route_budget_and_uses_shared_fusion(self):
        from recall.context_fusion import fuse_request_routes
        context=pd.DataFrame({'request_idx':[123],'user_idx':[7],'query':[''], 'recent_clicked_note_idxs':[list(range(20))+[999]]})
        ann=pd.DataFrame({'note_idx':[0,20,999],'score_ann':[3.,2.,1.]})
        tower=SimpleNamespace(fetch_candidates=Mock(return_value=ann))
        bundle=SimpleNamespace(bundle_id='frozen',manifest={'recall_policy':{'strategy':'weighted_rrf',
            'weights':{'ann':1.,'swing':2.,'usercf':1.},'route_topk':1000,'topk':500,'rrf_k':60,'context_policy':'filter_history'}},
            cf_candidates=Mock(return_value={'swing':{1:3.,20:2.,31:1.},'usercf':{2:3.,40:2.,999:1.}}))
        actual=fetch_bundle_recall_candidates(bundle,tower,context,scene='rec',max_rank=500)
        expected=fuse_request_routes({'ann':np.array([0,20,999]),'swing':np.array([1,20,31]),'usercf':np.array([2,40,999])},list(range(20)),(), 'filter_history')
        self.assertEqual(actual.note_idx.tolist(),expected.note_idx.tolist())
        np.testing.assert_array_equal(actual.recall_score.to_numpy(),expected.score.to_numpy())
        self.assertIn(999,actual.note_idx.tolist());self.assertNotIn(0,actual.note_idx.tolist())
        self.assertAlmostEqual(actual.loc[actual.note_idx==20,'recall_score'].iloc[0],3/61)
        self.assertEqual(tower.fetch_candidates.call_args.args[1],1000)
        self.assertEqual(bundle.cf_candidates.call_args.args[2],1000)
        self.assertEqual(tower.fetch_candidates.call_count,1);self.assertEqual(bundle.cf_candidates.call_count,1)
        self.assertEqual(actual.attrs['route_status']['context_policy'],'filter_history')
        self.assertEqual(actual.attrs['route_status']['history_filter_removed'],{'ann':1,'swing':1,'usercf':1})

    def test_rec_complete_route_union_is_fused_before_small_input_budget(self):
        context = pd.DataFrame({"request_idx": [123], "user_idx": [7], "query": [""], "recent_clicked_note_idxs": [[8, 9]]})
        ann = pd.DataFrame({"note_idx": [10, 20], "score_ann": [.9, .8]})
        tower = SimpleNamespace(fetch_candidates=Mock(return_value=ann))
        bundle = SimpleNamespace(bundle_id="frozen", manifest={"recall_policy": {
            "strategy": "weighted_rrf", "weights": {"ann": 1., "swing": 1., "usercf": 1.}, "route_topk": 1000, "topk": 500, "rrf_k": 60}},
            cf_candidates=Mock(return_value={"swing": {20: 9.}, "usercf": {20: 8.}}))
        result = fetch_bundle_recall_candidates(bundle, tower, context, scene="rec", max_rank=1)
        self.assertEqual(result.note_idx.tolist(), [20])
        self.assertAlmostEqual(result.recall_score.iloc[0], 1/62+2/61)
        self.assertEqual(tower.fetch_candidates.call_args.args[1], 1000)
        self.assertTrue(tower.fetch_candidates.call_args.kwargs["pure_request_ann"])
        bundle.cf_candidates.assert_called_once_with(7, [8, 9], 1000)
        self.assertEqual(result.attrs["route_status"]["merged_before_input_budget"], 2)

    def test_search_uses_own_current_query_bm25_and_never_provider_dpr(self):
        context = pd.DataFrame({"search_idx": [123], "user_idx": [7], "query": ["current query"],
            "recent_clicked_note_idxs": [[8]], "dpr_results": [[np.array([999., 10000.])]]})
        tower = SimpleNamespace(fetch_candidates=Mock(return_value=pd.DataFrame({"note_idx": [10, 20], "score_ann": [.9, .8]})))
        bundle = SimpleNamespace(bundle_id="frozen", manifest={"recall_policy": {
            "strategy": "weighted_rrf", "weights": {"ann": 1., "bm25": 2.}, "route_topk": 1000, "topk": 500, "rrf_k": 60}},
            bm25_candidates=Mock(return_value=pd.DataFrame({"note_idx": [20], "score": [3.]})), cf_candidates=Mock())
        result = fetch_bundle_recall_candidates(bundle, tower, context, scene="search", max_rank=1)
        self.assertEqual(result.note_idx.tolist(), [20])
        self.assertEqual(result.first_route.iloc[0], "bm25")
        self.assertNotIn(999, result.note_idx)
        bundle.bm25_candidates.assert_called_once_with("current query", 1000)
        bundle.cf_candidates.assert_not_called()

    def test_zero_weight_ann_route_cannot_append_ann_only_items(self):
        context = pd.DataFrame({"search_idx": [123], "user_idx": [7], "query": ["q"], "recent_clicked_note_idxs": [[]]})
        tower = SimpleNamespace(fetch_candidates=Mock(return_value=pd.DataFrame({"note_idx": [999], "score_ann": [.9]})))
        bundle = SimpleNamespace(bundle_id="frozen", manifest={"recall_policy": {
            "strategy": "weighted_rrf", "weights": {"ann": 0., "bm25": 1.}}},
            bm25_candidates=Mock(return_value=pd.DataFrame({"note_idx": [11], "score": [1.]})))
        result = fetch_bundle_recall_candidates(bundle, tower, context, scene="search", max_rank=160)
        self.assertEqual(result.note_idx.tolist(), [11])

    def test_pure_request_ann_does_not_invoke_legacy_auxiliary_fusion(self):
        tower = DSSMRecaller.__new__(DSSMRecaller)
        tower.scene = "search"
        tower.enabled = True
        tower.ann_index = object()
        tower.row2note = [10]
        tower._build_user_vec = Mock(return_value=np.ones((1, 128), dtype=np.float32))
        tower._search_ann = Mock(return_value=([10], [.8]))
        tower._build_query_only_vec = Mock()
        tower._fuse_search_query_rank = Mock()
        frame = pd.DataFrame({"search_idx": [123], "user_idx": [7], "query": ["actual query"], "recent_clicked_note_idxs": [[8]]})
        result = tower.fetch_candidates(123, 1000, frame, lambda _: frame, "search_idx", pure_request_ann=True)
        self.assertEqual(result.note_idx.tolist(), [10])
        tower._build_query_only_vec.assert_not_called()
        tower._fuse_search_query_rank.assert_not_called()

    def test_route_failure_is_exposed_without_historical_snapshot_fallback(self):
        context = pd.DataFrame({"search_idx": [123], "user_idx": [7], "query": ["q"], "recent_clicked_note_idxs": [[]]})
        tower = SimpleNamespace(fetch_candidates=Mock(return_value=pd.DataFrame({"note_idx": [10], "score_ann": [.8]})))
        bundle = SimpleNamespace(bundle_id="frozen", manifest={"recall_policy": {
            "strategy": "weighted_rrf", "weights": {"ann": 1., "bm25": 2.}}},
            bm25_candidates=Mock(side_effect=RuntimeError("index unavailable")))
        result = fetch_bundle_recall_candidates(bundle, tower, context, scene="search", max_rank=10)
        self.assertEqual(result.note_idx.tolist(), [10])
        self.assertEqual(result.attrs["route_status"]["errors"], {"bm25": "RuntimeError"})


if __name__ == "__main__":
    unittest.main()
