import unittest
from unittest.mock import patch, Mock
import numpy as np
import pandas as pd
import torch
from backend.online.preranking.gbdt import run_preranking_gbdt, GBDTPredictor
from backend.online.ranking.service import run_ranking
from backend.online.ranking.dien import DIENPredictor

class OnlineRankingTests(unittest.TestCase):
    def test_ranking_budget_keeps_best_preranked_candidates(self):
        frame=pd.DataFrame({"note_idx":[1,2,3],"rank":[1,2,3],
                            "preranking_rank":[3,2,1],"gbdt_score":[.1,.5,.9]})
        seen=[]
        def predict(rows, **kwargs):
            seen.extend(rows.note_idx.tolist())
            return rows.gbdt_score.to_numpy()
        with patch("backend.online.ranking.service.RANK_TOPN",2):
            full,_=run_ranking(frame,1,2,predict)
        self.assertEqual(seen,[3,2])
        self.assertEqual(full.note_idx.tolist(),[3,2])

    def test_business_fusion_preserves_dien_training_features(self):
        frame=pd.DataFrame({"note_idx":[1,2],"rank":[1,2],"dssm_score":[1.,0.],
                            "first_route":["ann","swing"]})
        output=run_preranking_gbdt(frame,2,lambda rows:np.array([.1,.2]))
        by_id=output.set_index("note_idx")
        self.assertGreater(by_id.loc[1,"preranking_score"],by_id.loc[2,"preranking_score"])
        self.assertEqual(by_id.loc[1,"dien_preranking_rank"],2)
        self.assertEqual(by_id.loc[2,"dien_preranking_rank"],1)
        self.assertAlmostEqual(by_id.loc[1,"dien_preranking_score"],.1)
        self.assertEqual(by_id.loc[2,"first_route_id"],2)

    def test_state_wrapper_passes_components_without_double_inference(self):
        predictor=GBDTPredictor.__new__(GBDTPredictor)
        predictor.predict_with_components=Mock(return_value={
            "lgb_score":np.array([.2]),"xgb_score":np.array([.4]),"gbdt_score":np.array([.3])})
        class State:
            gbdt_predictor=predictor
            def predict_gbdt(self, frame): return self.gbdt_predictor.predict(frame)
        frame=pd.DataFrame({"note_idx":[1],"rank":[1],"dssm_score":[.5]})
        result=run_preranking_gbdt(frame,1,State().predict_gbdt)
        self.assertAlmostEqual(result.lgb_score.iloc[0],.2)
        self.assertAlmostEqual(result.xgb_score.iloc[0],.4)
        predictor.predict_with_components.assert_called_once()

    def test_dien_uses_training_alias_without_mutating_business_frame(self):
        predictor=DIENPredictor.__new__(DIENPredictor)
        predictor.device="cpu"; predictor.scene="rec"; predictor.group_key="request_idx"
        predictor.num_feats=["preranking_score","preranking_rank"]
        captured={}
        class Model:
            def __call__(self, features):
                captured.update(features)
                return torch.zeros((1,1)),None
        class Store:
            def __init__(self, shape): self.shape=shape
            def get_tensor(self, ids): return torch.zeros((len(ids),)+self.shape)
        predictor.model=Model()
        predictor.stores={"s":Store((20,768)),"t":Store((768,)),"i":Store((768,))}
        frame=pd.DataFrame({"request_idx":[1],"note_idx":[2],"preranking_score":[.99],
                            "preranking_rank":[1],"dien_preranking_score":[.1],"dien_preranking_rank":[9]})
        predictor.predict(frame)
        self.assertAlmostEqual(captured["preranking_score"].item(),.1,places=6)
        self.assertEqual(captured["preranking_rank"].item(),9)
        self.assertEqual(frame.preranking_score.iloc[0],.99)

if __name__=="__main__": unittest.main()
