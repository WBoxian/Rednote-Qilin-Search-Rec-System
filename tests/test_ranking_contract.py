import unittest
import numpy as np
import pandas as pd
from training.utils import select_ranking_features, split_groups_temporally
from training.dien_ranker import _binary_auc_local, DIENRanker
from training.gbdt_ranker import _group_val_split as gbdt_split
from training.dien_ranker import _group_val_split as dien_split


class RankingContractTests(unittest.TestCase):
    def test_unknown_outcomes_ids_positions_and_aggregates_are_excluded(self):
        names = ["note_type","rank","score_ann","lgb_score","preranking_rank",
                 "click","y_multi","position","pos_bucket","request_idx","user_idx",
                 "session_idx","event_timestamp","ctr","click_num","dense_feat1","future_label"]
        frame = pd.DataFrame({name: [1.] for name in names})
        self.assertEqual(select_ranking_features(frame,"gbdt"),["note_type","rank","score_ann"])
        self.assertEqual(select_ranking_features(frame,"dien"),["note_type","rank","score_ann","lgb_score","preranking_rank"])
        self.assertEqual(select_ranking_features(frame[names[::-1]],"dien"),select_ranking_features(frame,"dien"))

    def test_unknown_only_feature_frame_fails_closed(self):
        with self.assertRaises(ValueError):
            select_ranking_features(pd.DataFrame({"future_label":[1]}),"dien")

    def test_tied_auc_is_order_independent(self):
        for y in ([0,1],[1,0]):
            self.assertEqual(_binary_auc_local(np.array(y),np.array([.5,.5])),.5)
        self.assertEqual(_binary_auc_local(np.array([0,1,1,0]),np.array([.1,.5,.5,.5])),.75)

    def test_legacy_checkpoint_shape_is_unchanged(self):
        import torch
        first = DIENRanker("rec", ["position","ctr","user_idx"])
        state = first.state_dict()
        names = [key.split(".")[1] for key in state if key.startswith("num_proj.") and key.endswith(".weight")]
        second = DIENRanker("rec", names)
        second.load_state_dict(state, strict=True)
        self.assertEqual(names,["position","ctr","user_idx"])

    def test_forward_split_keeps_whole_groups_and_drops_crossing_requests(self):
        frame = pd.DataFrame({"request_idx":[1,1,2,2,3,3,4,4],
                              "event_timestamp":[1,2,3,4,2,10,11,12]},index=range(10,18))
        for splitter in (split_groups_temporally,gbdt_split,dien_split):
            fit,val = splitter(frame,"request_idx",.5)
            self.assertTrue(set(frame.iloc[fit].request_idx).isdisjoint(frame.iloc[val].request_idx))
            self.assertLess(frame.iloc[fit].event_timestamp.max(),frame.iloc[val].event_timestamp.min())
            self.assertNotIn(3,set(frame.iloc[fit].request_idx)|set(frame.iloc[val].request_idx))
            self.assertEqual(list(fit),[0,1,2,3])
            self.assertEqual(list(val),[6,7])

    def test_invalid_times_do_not_fall_back_to_random_split(self):
        for frame in (pd.DataFrame({"request_idx":[1,2]}),
                      pd.DataFrame({"request_idx":[1,2],"event_timestamp":[1,np.nan]}),
                      pd.DataFrame({"request_idx":[1,2],"event_timestamp":[1,1]})):
            with self.assertRaises(ValueError):
                split_groups_temporally(frame,"request_idx")
