import unittest
import numpy as np
import pandas as pd
from backend.online.preranking.service import (
    SEARCH_ITEM_FEATURES, build_search_item_features,
    apply_search_preranking_policy, run_preranking,
)


class SearchPolicyTests(unittest.TestCase):
    def test_shared_features_ignore_labels_and_preserve_input(self):
        source=pd.DataFrame({"note_title":["北京旅游","其他"],"note_content":["北京路线","无关"],
                             "click":[1,0]},index=[9,3])
        first=build_search_item_features(source,"北京")
        source["click"]=[0,1]
        second=build_search_item_features(source,"北京")
        pd.testing.assert_frame_equal(first[list(SEARCH_ITEM_FEATURES)],second[list(SEARCH_ITEM_FEATURES)])
        self.assertEqual(first.query_exact_hit.tolist(),[1,0])
        self.assertNotIn("query_exact_hit",source)
        empty=build_search_item_features(source.drop(columns=["note_title","note_content"]),"北京")
        self.assertFalse(empty[list(SEARCH_ITEM_FEATURES)].isna().any().any())

    def test_soft_keeps_same_budget_without_hard_deletion(self):
        source=pd.DataFrame({"note_idx":[1,2,3],"rank":[1,2,3],"gbdt_score":[.9,.8,.7],
                             "preranking_score":[.9,.8,.7],"query_exact_hit":[1,0,1],
                             "query_match_score":[3.,0.,3.],"query_term_cover":[1.,0.,1.],
                             "query_intent_match":[0.,0.,0.]})
        legacy=apply_search_preranking_policy(source,2,"legacy")
        soft=apply_search_preranking_policy(source,2,"soft")
        self.assertEqual(legacy.note_idx.tolist(),[1])
        self.assertEqual(soft.note_idx.tolist(),[1,2])
        self.assertNotIn(3,soft.note_idx)
        self.assertEqual(soft.preranking_score.iloc[0],legacy.preranking_score.iloc[0])
        with self.assertRaises(ValueError):
            apply_search_preranking_policy(source,2,"typo")

    def test_formal_context_ignores_snapshot_quality_statistics(self):
        feat=pd.DataFrame({"search_idx":[1,1],"note_idx":[1,2],
                           "note_title":["北京旅游","北京路线"],"note_content":["北京","北京"],
                           "accum_like_num":[0.,10000.],"accum_collect_num":[0.,0.],"accum_comment_num":[0.,0.]})
        recall=pd.DataFrame({"search_idx":[1,1],"note_idx":[1,2],"rank":[1,2],
                             "score_ann":[.5,.5],"recall_score":[.5,.5]})
        def call(frame, allow):
            return run_preranking(1,"北京","search","search_idx",recall,frame,2,
                                  lambda ids:frame.drop(columns="search_idx"),lambda rows:np.zeros(len(rows)),
                                  linkage_ctx={"keywords":[]},search_filter_policy="soft",
                                  allow_snapshot_quality_boost=allow).sort_values("note_idx")
        a=call(feat,False)
        changed=feat.copy();changed["accum_like_num"]=[10000.,0.]
        b=call(changed,False)
        np.testing.assert_array_equal(a.linkage_boost,b.linkage_boost)
        self.assertTrue((a.linkage_boost==0).all())
        self.assertGreater(call(feat,True).linkage_boost.iloc[1],0)
