import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import faiss
import numpy as np
import pandas as pd
import torch

from backend.online.recall.dssm import DSSMRecaller
from recall.residual_dssm import ResidualDSSMModel, encode_request_frame, model_from_checkpoint, save_residual_checkpoints
from training.history import HISTORY_ORDER


class TextStore:
    def get_batch_vecs(self, ids):
        values=np.zeros((len(ids),768),np.float32)
        for row,note in enumerate(ids):
            if note in {0,1}:values[row,int(note)]=1.
        return torch.from_numpy(values)


class OnlineResidualRecallTests(unittest.TestCase):
    def fixture(self, root, index_dim=256):
        vocab={'user_idx':3,'note_idx':5,'gender':3,'platform':3,'age':3,'location':3,'tax1':3,'tax2':3,'tax3':3}
        components=torch.zeros(128,768);components[:,:128]=torch.eye(128)
        model=ResidualDSSMModel('rec',vocab,6,938,pca_components=components,residual_weight=.25).eval()
        meta={'scene':'rec','cat_vocabs':vocab,'item_dense_dim':6,'user_dense_dim':938,
            'known_user_ids':[0],'unknown_user_id':2,'history_contract':HISTORY_ORDER,'masked_user_dense':True}
        exported=root/'export';save_residual_checkpoints(model,meta,exported)
        (root/'models').mkdir();(root/'index').mkdir()
        path=root/'models/dssm_rec_hard_user_tower.pt'
        path.write_bytes((exported/'dssm_rec_residual_user_tower.pt').read_bytes())
        (root/'index/dssm_rec_hard_item_map.json').write_text(json.dumps({'0':0,'7':1}))
        np.save(root/'index/dssm_rec_hard_row2note.npy',np.asarray([0,7],np.int64))
        index=faiss.IndexFlatIP(index_dim);items=np.zeros((2,index_dim),np.float32);items[:,0]=[1.,-.5];index.add(items)
        faiss.write_index(index,str(root/'index/dssm_rec_hard_ivfpq.faiss'))
        return path

    def load(self, root):
        with patch('recall.dssm_trainer.MMapStore',return_value=TextStore()):
            return DSSMRecaller('rec','hard',root,'request_idx')

    def request(self, history):
        return pd.DataFrame({'request_idx':[123],'user_idx':[1],'query':[''],'recent_clicked_note_idxs':[history],
            'gender_enc':[0],'platform_enc':[0],'age_enc':[0],'location_enc':[0],'fans_num':[0.],'follows_num':[0.],
            **{f'dense_feat{i}':[0.] for i in range(1,41)}})

    def test_real_loaded_family_matches_shared_encoder_and_preserves_missing_content_norm(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=self.fixture(root);tower=self.load(root)
            self.assertTrue(tower.enabled);self.assertEqual(tower.retrieval_dim,256)
            offline=model_from_checkpoint(path,allow_partial=True)
            for history in [[0,0,1]+[99]*17+[1],[]]:
                frame=self.request(history)
                actual=tower._build_user_vec(frame,123,1)
                _,expected=encode_request_frame(offline,frame,{'note_text':TextStore(),'note_img':TextStore()},offline.checkpoint_meta)
                np.testing.assert_array_equal(actual,expected)
                self.assertEqual(actual.shape,(1,256))
                if not history:
                    self.assertAlmostEqual(float(np.linalg.norm(actual)),.5,places=6)
                    self.assertEqual(float(np.abs(actual[:,128:]).sum()),0.)
            probe=np.zeros((1,256),np.float32);probe[0,0]=.5
            ids,scores=tower._search_ann(probe,2)
            self.assertEqual(ids,[0,7]);self.assertEqual(scores,[.5,-.25])

    def test_unknown_family_and_missing_persistent_mask_reject_in_actual_loader(self):
        for mutation in ['unknown_family','missing_mask','false_mask']:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root=Path(directory);path=self.fixture(root)
                payload=torch.load(path,map_location='cpu',weights_only=False)
                if mutation=='unknown_family':payload['meta']['model_family']='unsupported_family'
                elif mutation=='missing_mask':payload['state_dict'].pop('seq_mask_padding')
                else:payload['state_dict']['user_mask_profile']=torch.tensor(False)
                torch.save(payload,path)
                with self.assertRaises(ValueError):self.load(root)

    def test_bound_ann_dimension_mismatch_rejects_actual_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.fixture(root,index_dim=128)
            with self.assertRaisesRegex(ValueError,'dimensions/metric differ'):self.load(root)


if __name__=='__main__':unittest.main()
