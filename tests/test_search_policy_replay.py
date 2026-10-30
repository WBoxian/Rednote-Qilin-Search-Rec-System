import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock
import numpy as np
import pandas as pd

from scripts.compare_search_policies import main, RankStackPredictor
from backend.online.preranking.gbdt import run_preranking_gbdt


class SearchPolicyReplayTests(unittest.TestCase):
    def test_final_test_requires_frozen_selection_before_reading_labels(self):
        with patch("sys.argv",["compare","--requests","missing.parquet","--output","unused","--split","test"]):
            with patch("pandas.read_parquet",side_effect=AssertionError("test labels read")):
                with self.assertRaises(ValueError):
                    main()

    def test_sampling_final_test_is_rejected(self):
        with patch("sys.argv",["compare","--requests","missing.parquet","--output","unused","--split","test",
                               "--selection","missing.json","--max-groups","2"]):
            with self.assertRaises(ValueError):
                main()

    def test_test_partition_cannot_be_used_for_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/"requests.parquet"
            pd.DataFrame({"partition":["test"]}).to_parquet(source,index=False)
            output=Path(directory)/"out"
            with patch("sys.argv",["compare","--requests",str(source),"--output",str(output)]):
                with self.assertRaises(ValueError):
                    main()
            self.assertFalse(output.exists())

    def test_single_thread_switch_requires_64_bitwise_equal_validation_requests(self):
        predictor=RankStackPredictor.__new__(RankStackPredictor)
        predictor.thread_checks=0
        predictor.thread_verified=False
        model=Mock()
        model.predict.side_effect=lambda values,num_threads:np.array([.1,.2],dtype=np.float64)
        frame=pd.DataFrame({"value":[1.,2.]})
        for _ in range(63):
            predictor._score_tree(model,frame,["value"])
        self.assertFalse(predictor.thread_verified)
        predictor._score_tree(model,frame,["value"])
        self.assertTrue(predictor.thread_verified)
        self.assertEqual(predictor.thread_checks,64)
        predictor.thread_checks=0
        predictor.thread_verified=False
        model.predict.side_effect=lambda values,num_threads:np.array([.1 + (1e-12 if num_threads==1 else 0),.2])
        with self.assertRaises(ValueError):
            predictor._score_tree(model,frame,["value"])
        self.assertFalse(predictor.thread_verified)

    def test_new_single_tree_adapter_preserves_real_components(self):
        predictor=RankStackPredictor.__new__(RankStackPredictor)
        predictor.static_store=object()
        predictor.context=Mock()
        predictor.context.feature_frame.side_effect=lambda frame,store:frame
        predictor.feature_cols=["content_length"]
        predictor.model=object()
        predictor.score_tree=lambda model,frame,features:np.array([.2,.8],dtype=np.float32)
        frame=pd.DataFrame({"note_idx":[1,2],"rank":[1,2],"content_length":[50.,100.],"dssm_score":[0.,0.]})
        result=run_preranking_gbdt(frame,2,predictor.predict).sort_values("note_idx")
        np.testing.assert_allclose(result.lgb_score,[.2,.8])
        np.testing.assert_array_equal(result.xgb_score,[0.,0.])
        np.testing.assert_allclose(result.gbdt_score,[.2,.8])
        np.testing.assert_allclose(result.dien_preranking_score,[.2,.8])
        predictor.context.feature_frame.assert_called_once()
