import pickle
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from backend.online.item_features import (
    ITEM_NUMERIC_FEATURES, ITEM_CATEGORY_FEATURES, STATIC_ITEM_FEATURES,
    StaticItemFeatureStore, encode_static_items,
)
from backend.online.preranking.service import _resolve_prerank_feat_cols, run_preranking


class ItemFeatureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.notes = self.root / "notes"; self.notes.mkdir()
        self.vocab = self.root / "vocab"; self.vocab.mkdir()
        self.raw = pd.DataFrame({"note_idx": [7, 0], **{name: [3., 12.] for name in ITEM_NUMERIC_FEATURES},
                                 **{name: ["new-category", "train-category"] for name in ITEM_CATEGORY_FEATURES}})
        self.raw.to_parquet(self.notes / "part.parquet", index=False)
        self.maps = {name: {"train-category": 9} for name in ITEM_CATEGORY_FEATURES}
        for name, mapping in self.maps.items():
            with (self.vocab / (name + ".pkl")).open("wb") as stream: pickle.dump(mapping, stream)
        self.store = StaticItemFeatureStore(self.notes, self.vocab)

    def tearDown(self):
        self.temp.cleanup()

    def test_offline_online_encoding_and_non_exposed_zero_id(self):
        request = pd.DataFrame({"note_idx":[0,7,0,999], "request_idx":[1]*4, "position":[98]*4})
        enriched = self.store.attach(request)
        self.assertEqual(enriched.note_idx.tolist(), request.note_idx.tolist())
        self.assertEqual(enriched.content_length.tolist(), [12.,3.,12.,0.])
        self.assertEqual(enriched.item_features_found.tolist(), [True,True,True,False])
        expected = encode_static_items(self.raw,self.maps).sort_values("note_idx")
        pd.testing.assert_frame_equal(self.store.get([0,7]).drop(columns="item_features_found"),expected.reset_index(drop=True))
        self.assertNotIn("position",self.store.get([0]).columns)
        self.assertEqual(self.store.get([0]).taxonomy1_id_enc.iloc[0],9)
        self.assertEqual(self.store.get([7]).taxonomy1_id_enc.iloc[0],0)
        self.assertEqual(self.store.nbytes,2*(8+4*len(STATIC_ITEM_FEATURES)))

    def test_null_category_uses_zero_like_training_sql(self):
        raw=self.raw.copy()
        for name in ITEM_CATEGORY_FEATURES:
            raw[name]=[None, "UNK"]
        maps={name:{"UNK":12} for name in ITEM_CATEGORY_FEATURES}
        encoded=encode_static_items(raw,maps)
        self.assertEqual(encoded.taxonomy1_id_enc.tolist(),[0.,12.])

    def test_fractional_item_id_is_rejected(self):
        with self.assertRaises(ValueError):
            self.store.get([.5])

    def test_missing_vocab_does_not_rebuild_from_catalog(self):
        (self.vocab/"taxonomy1_id.pkl").unlink()
        with self.assertRaises(FileNotFoundError):
            StaticItemFeatureStore(self.notes,self.vocab)

    def test_wrapper_projection_includes_real_predictor_and_static_columns(self):
        class Predictor:
            feature_cols = ["custom_checkpoint_feature", "note_type", "position", "pos_bucket", "event_timestamp", "ctr", "accum_like_num", "click"]
        class State:
            gbdt_predictor = Predictor()
            def predict_gbdt(self, rows): return np.zeros(len(rows))
        frame=pd.DataFrame({"request_idx":[1],"note_idx":[7],"custom_checkpoint_feature":[8.],
                            "note_type":[3.],"content_length":[12.],"position":[5.],"pos_bucket":[1.],"event_timestamp":[100.],"ctr":[.9],"accum_like_num":[400.],"click":[1]})
        columns=_resolve_prerank_feat_cols(State().predict_gbdt,frame,"request_idx")
        self.assertIn("custom_checkpoint_feature",columns)
        self.assertIn("content_length",columns)
        for forbidden in ("position","pos_bucket","event_timestamp","ctr","accum_like_num","click"):
            self.assertNotIn(forbidden,columns)

    def test_non_exposed_candidate_gets_catalog_features_before_prediction(self):
        observed=pd.DataFrame({"request_idx":[1],"note_idx":[7],"note_type":[3.],"content_length":[3.]})
        recalled=pd.DataFrame({"request_idx":[1],"note_idx":[0],"rank":[1],"dssm_score":[.5],"score_ann":[.5],"score_swing":[0.],"score_usercf":[0.]})
        seen=[]
        def predict(frame):
            seen.append(frame[["note_idx","note_type","content_length","taxonomy1_id_enc"]].copy())
            return np.zeros(len(frame))
        result=run_preranking(1,"","rec","request_idx",recalled,observed,10,
                             lambda ids:pd.DataFrame(),predict,fetch_item_features=self.store.get)
        self.assertEqual(result.note_idx.tolist(),[0])
        self.assertEqual(seen[0].iloc[0].tolist(),[0.,12.,12.,9.])
