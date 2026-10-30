import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from recall.dssm_trainer import (DSSMModel, NumericPreprocessor, fit_numeric_preprocessing,
    numeric_feature_columns, save_tower_checkpoints, split_by_time, EMB_DIM)
from recall.query_encoding import QueryTextStore, build_search_query_store

class DSSMContractTests(unittest.TestCase):
    def test_fit_only_stats_finite_and_count_log(self):
        fit=pd.DataFrame({"fans_num":[0.,3.,np.nan],"dense_feat1":[1.,2.,3.]})
        stats=fit_numeric_preprocessing(fit,["fans_num"],list(fit))
        layer=NumericPreprocessor(list(fit),stats)
        expected=np.log1p([0.,3.,0.])
        self.assertAlmostEqual(stats["columns"]["fans_num"]["mean"],expected.mean())
        output=layer(torch.tensor([[np.inf,99.]],dtype=torch.float32))
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(stats["fit_rows"],3)
        self.assertTrue(pd.isna(fit.iloc[2,0]))

    def test_checkpoint_reload_applies_transform_without_constructor_metadata(self):
        user,item,logs=numeric_feature_columns("search")
        fit=pd.DataFrame({c:[0.,2.,8.] for c in user+item})
        stats=fit_numeric_preprocessing(fit,logs,user+item)
        vocabs={key:4 for key in ["user_idx","note_idx","gender","platform","age","location","tax1","tax2","tax3"]}
        model=DSSMModel("search",vocabs,len(item),42+2*EMB_DIM,preprocessing=stats).eval()
        batch={key:torch.tensor([1]) for key in ["user_idx","gender_enc","platform_enc","age_enc","location_enc"]}
        batch.update(fans_num=torch.tensor([8.]),follows_num=torch.tensor([2.]),dense_feats=torch.ones(1,40),query_vec=torch.ones(1,EMB_DIM))
        history=torch.zeros(1,EMB_DIM)
        expected=model.forward_user(batch,history)
        with tempfile.TemporaryDirectory() as directory:
            save_tower_checkpoints(model.state_dict(),"search","easy",vocabs,len(item),42+2*EMB_DIM,preprocessing=stats,output_dir=Path(directory))
            ckpt=torch.load(Path(directory)/"dssm_search_easy_user_tower.pt",weights_only=False)
            restored=DSSMModel("search",vocabs,len(item),42+2*EMB_DIM).eval()
            restored.load_state_dict(ckpt["state_dict"],strict=False)
            torch.testing.assert_close(expected,restored.forward_user(batch,history),rtol=0,atol=0)
            self.assertEqual(ckpt["meta"]["numeric_preprocessing"],stats)
            self.assertIn("user_numeric.mean",ckpt["state_dict"])
            self.assertNotIn("item_numeric.mean",ckpt["state_dict"])

    def test_legacy_checkpoint_identity(self):
        transform=NumericPreprocessor(["fans_num","dense_feat1"])
        values=torch.tensor([[3.,-4.]])
        torch.testing.assert_close(transform(values),values,rtol=0,atol=0)

    def test_chronological_split_keeps_sessions_and_excludes_future(self):
        frame=pd.DataFrame({"search_idx":[1,2,3,4,5,6],"session_idx":[1,1,2,3,4,5],
                            "event_timestamp":[1.,2.,3.,4.,9.,10.]})
        fit,val=split_by_time(frame,"search_idx",test_start=10.,val_ratio=.4)
        self.assertFalse(set(fit.session_idx)&set(val.session_idx))
        self.assertLess(fit.event_timestamp.max(),val.event_timestamp.min())
        self.assertNotIn(6,set(fit.search_idx)|set(val.search_idx))

    def test_query_missing_ids_never_silently_zero(self):
        store=QueryTextStore([12],np.ones((1,768),dtype=np.float32))
        with self.assertRaises(KeyError):store.get_batch_vecs([13])
        torch.testing.assert_close(store.get_batch_vecs([12]),torch.ones(1,768))
        with self.assertRaises(ValueError):QueryTextStore([12],np.zeros((1,768)))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):build_search_query_store(pd.DataFrame({"search_idx":[1],"query":[""]}),directory)

if __name__=="__main__":unittest.main()