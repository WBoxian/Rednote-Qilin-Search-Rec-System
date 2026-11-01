import hashlib
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
from recall.residual_dssm import AdaptedResidualDSSMModel, ResidualDSSMModel, encode_request_frame, model_from_checkpoint, save_residual_checkpoints
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



class OnlineAdaptedResidualRecallTests(unittest.TestCase):
    load = OnlineResidualRecallTests.load
    request = OnlineResidualRecallTests.request

    @staticmethod
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def fixture(self, root, dim=128, index_kind='FlatIP'):
        vocab = {'user_idx': 3, 'note_idx': 5, 'gender': 3, 'platform': 3,
                 'age': 3, 'location': 3, 'tax1': 3, 'tax2': 3, 'tax3': 3}
        components = torch.zeros(128, 768)
        components[:, :128] = torch.eye(128)
        torch.manual_seed(42)
        model = AdaptedResidualDSSMModel('rec', vocab, 6, 938,
            pca_components=components, residual_weight=.25, content_input_dim=dim).eval()
        with torch.no_grad():
            model.content_query_adapter[-1].weight.normal_(0, .01)
            model.content_query_adapter[-1].bias.fill_(.03)
            model.content_item_adapter[-1].weight.normal_(0, .01)
            model.content_item_adapter[-1].bias.fill_(-.02)
        meta = {'scene': 'rec', 'cat_vocabs': vocab, 'item_dense_dim': 6, 'user_dense_dim': 938,
                'known_user_ids': [0], 'unknown_user_id': 2, 'history_contract': HISTORY_ORDER,
                'masked_user_dense': True}
        exported = root / 'export'
        save_residual_checkpoints(model, meta, exported)
        (root / 'models').mkdir()
        (root / 'index').mkdir()
        prefix = root / 'index/dssm_rec_hard'
        paths = {name: Path(str(prefix) + suffix) for name, suffix in {
            'map': '_item_map.json', 'rows': '_row2note.npy', 'vectors': '_item_emb.bin',
            'index': '_ivfpq.faiss', 'ann_meta': '_ivfpq_meta.json', 'item_meta': '_item_meta.json'}.items()}
        paths['user'] = root / 'models/dssm_rec_hard_user_tower.pt'
        paths['full'] = exported / 'dssm_rec_residual.pt'
        paths['item'] = exported / 'dssm_rec_residual_item_tower.pt'
        paths['user'].write_bytes((exported / 'dssm_rec_residual_user_tower.pt').read_bytes())
        paths['map'].write_text(json.dumps({'0': 0, '1': 1}))
        np.save(paths['rows'], np.asarray([0, 1], np.int64))
        batch = {'note_idx': torch.tensor([0, 1]), 'note_type': torch.tensor([1, 2]),
                 'taxonomy': torch.zeros(2, 3, dtype=torch.long), 'dense_stats': torch.zeros(2, 6)}
        text = TextStore().get_batch_vecs([0, 1])
        with torch.no_grad():
            # Canonical B1 item vectors exercise both real learned and adapted blocks.
            vectors = np.concatenate([model.forward_item({key: value[i:i+1] for key, value in batch.items()},
                text[i:i+1], torch.zeros(1, 768)).numpy() for i in range(2)])
        vectors.astype(np.float32).tofile(paths['vectors'])
        if index_kind == 'FlatIP':
            index = faiss.IndexFlatIP(256)
        else:
            quantizer = faiss.IndexFlatIP(256)
            if index_kind == 'IVFFlat':
                index = faiss.IndexIVFFlat(quantizer, 256, 2, faiss.METRIC_INNER_PRODUCT)
            elif index_kind == 'IVFPQ':
                index = faiss.IndexIVFPQ(quantizer, 256, 2, 8, 2, faiss.METRIC_INNER_PRODUCT)
            else:
                raise AssertionError(index_kind)
            previous_threads = faiss.omp_get_max_threads()
            try:
                faiss.omp_set_num_threads(2)
                index.train(np.random.default_rng(42).standard_normal((256, 256)).astype(np.float32))
            finally:
                faiss.omp_set_num_threads(previous_threads)
        index.add(vectors)
        faiss.write_index(index, str(paths['index']))
        full_sha = self.sha(paths['full'])
        payload = torch.load(paths['user'], map_location='cpu', weights_only=False)
        self.assertEqual(payload['meta']['exported_full_checkpoint_sha256'], full_sha)
        common = {'model_family': 'adapted_residual_dssm_v1', 'dim': 256, 'num_items': 2,
                  'dtype': 'float32', 'metric': 'ip', 'history_contract': HISTORY_ORDER,
                  'selected_checkpoint_sha256': full_sha,
                  'selected_item_vectors_sha256': self.sha(paths['vectors']), 'index_type': index_kind}
        paths['item_meta'].write_text(json.dumps(common))
        ann_meta = {**common, 'index_sha256': self.sha(paths['index'])}
        if index_kind != 'FlatIP':
            ann_meta.update(nprobe=2, nlist=2)
        paths['ann_meta'].write_text(json.dumps(ann_meta))
        return paths

    @staticmethod
    def update_meta(path, field, value):
        document = json.loads(path.read_text())
        document[field] = value
        path.write_text(json.dumps(document))

    def test_both_adapters_real_full_partial_roundtrip_and_online_b1_bitwise(self):
        for dim in (128, 768):
            with self.subTest(dim=dim), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                paths = self.fixture(root, dim)
                online = self.load(root)
                full = model_from_checkpoint(paths['full'])
                user = model_from_checkpoint(paths['user'], allow_partial=True)
                item = model_from_checkpoint(paths['item'], allow_partial=True)
                self.assertTrue(online.enabled)
                self.assertEqual(online.model.content_input_dim, dim)
                self.assertEqual(user.checkpoint_loaded_tower, 'user')
                self.assertEqual(item.checkpoint_loaded_tower, 'item')
                self.assertIsNone(full.checkpoint_loaded_tower)
                raw = TextStore().get_batch_vecs([0, 1])
                torch.testing.assert_close(user.project_user_content(raw), full.project_user_content(raw), rtol=0, atol=0)
                torch.testing.assert_close(item.project_item_content(raw), full.project_item_content(raw), rtol=0, atol=0)
                histories = [[0], [0, 0, 1] + [99] * 17 + [1], [], np.array([0, 1, 0])]
                for history in histories:
                    for user_id in (0, 1, 999):
                        frame = self.request(history)
                        frame['user_idx'] = user_id
                        frame['fans_num'] = 999.
                        frame['follows_num'] = 11.
                        frame['gender_enc'] = 1
                        frame['dense_feat1'] = 123.
                        actual = online._build_user_vec(frame, 123, user_id)
                        _, expected = encode_request_frame(full, frame,
                            {'note_text': TextStore(), 'note_img': TextStore()}, full.checkpoint_meta)
                        np.testing.assert_array_equal(actual, expected)
                        self.assertTrue(np.isfinite(actual).all())
                        self.assertEqual(actual.shape, (1, 256))
                        if len(history) == 0:
                            self.assertEqual(float(np.abs(actual[:, 128:]).sum()), 0.)
                            self.assertAlmostEqual(float(np.linalg.norm(actual)), .5, places=6)
                query = online._build_user_vec(self.request([0, 0, 1]), 123, 1)
                equivalent_unknown = self.request([0, 0, 1])
                equivalent_unknown['user_idx'] = 999
                np.testing.assert_array_equal(query, online._build_user_vec(equivalent_unknown, 123, 999))
                ids, scores = online._search_ann(query, 2)
                vectors = np.fromfile(paths['vectors'], np.float32).reshape(2, 256)
                expected_scores = (query @ vectors.T)[0]
                expected_order = np.argsort(-expected_scores)
                self.assertEqual(ids, expected_order.tolist())
                np.testing.assert_allclose(scores, expected_scores[expected_order], rtol=1e-6, atol=1e-7)
                with self.assertRaisesRegex(ValueError, 'without legacy normalized expansion'):
                    online._search_ann_by_history_items([0], 2)
                with self.assertRaisesRegex(ValueError, 'without legacy normalized expansion'):
                    online.expand_seed_candidates({0: 1.}, 2)

    def test_item_partial_and_missing_active_adapter_are_rejected(self):
        for mutation in ('item_partial', 'missing_adapter', 'missing_mask', 'unknown_anchor', 'bad_known_users'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                paths = self.fixture(root)
                if mutation == 'item_partial':
                    paths['user'].write_bytes(paths['item'].read_bytes())
                else:
                    payload = torch.load(paths['user'], map_location='cpu', weights_only=False)
                    if mutation == 'missing_adapter':
                        del payload['state_dict']['content_query_adapter.2.weight']
                    elif mutation == 'missing_mask':
                        del payload['state_dict']['seq_mask_padding']
                    elif mutation == 'unknown_anchor':
                        del payload['meta']['exported_full_checkpoint_sha256']
                    else:
                        payload['meta']['known_user_ids'] = [2]
                    torch.save(payload, paths['user'])
                with self.assertRaises(ValueError):
                    self.load(root)

    def test_required_metadata_and_bound_files_cannot_fall_back(self):
        for missing in ('ann_meta', 'item_meta', 'vectors', 'rows', 'index', 'map'):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                paths = self.fixture(root)
                paths[missing].unlink()
                with self.assertRaises(ValueError):
                    self.load(root)
        for document in ('ann_meta', 'item_meta'):
            for invalid in ('{bad json', '[]'):
                with self.subTest(document=document, invalid=invalid), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    paths = self.fixture(root)
                    paths[document].write_text(invalid)
                    with self.assertRaises(ValueError):
                        self.load(root)

    def test_metadata_contract_mismatch_is_rejected_for_both_documents(self):
        mutations = {'model_family': 'residual_dssm_v1', 'dim': 128, 'num_items': 3,
                     'dtype': 'float16', 'metric': 'cosine', 'history_contract': 'legacy_raw_order_v0',
                     'selected_checkpoint_sha256': '0' * 64,
                     'selected_item_vectors_sha256': '1' * 64, 'index_type': 'IVFPQ'}
        # Reuse immutable actual model/vector assets; each subtest restores metadata.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.fixture(root)
            for document in ('ann_meta', 'item_meta'):
                original = paths[document].read_bytes()
                for field, value in mutations.items():
                    with self.subTest(document=document, field=field):
                        self.update_meta(paths[document], field, value)
                        with self.assertRaises(ValueError):
                            self.load(root)
                        paths[document].write_bytes(original)
            for field, value in [('dim', True), ('num_items', True), ('num_items', 2.),
                                 ('nprobe', True), ('nprobe', 0), ('index_type', [])]:
                with self.subTest(field=field, value=value):
                    original = paths['ann_meta'].read_bytes()
                    self.update_meta(paths['ann_meta'], field, value)
                    with self.assertRaises(ValueError):
                        self.load(root)
                    paths['ann_meta'].write_bytes(original)

    def test_vector_and_index_actual_bytes_are_verified(self):
        for asset in ('vectors', 'index'):
            with self.subTest(asset=asset), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                paths = self.fixture(root)
                data = bytearray(paths[asset].read_bytes())
                data[-1] ^= 1
                paths[asset].write_bytes(data)
                with self.assertRaisesRegex(ValueError, 'asset mismatch'):
                    self.load(root)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.fixture(root)
            with paths['vectors'].open('ab') as stream:
                stream.write(b'\0\0\0\0')
            digest = self.sha(paths['vectors'])
            for document in ('ann_meta', 'item_meta'):
                self.update_meta(paths[document], 'selected_item_vectors_sha256', digest)
            with self.assertRaisesRegex(ValueError, 'byte length'):
                self.load(root)

    def test_bound_index_actual_dimension_metric_count_and_rows_are_verified(self):
        for mutation in ('dimension', 'metric', 'count', 'duplicate_row', 'negative_row'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                paths = self.fixture(root)
                if mutation.endswith('_row'):
                    np.save(paths['rows'], np.asarray([0, 0] if mutation == 'duplicate_row' else [0, -1], np.int64))
                else:
                    dim = 128 if mutation == 'dimension' else 256
                    index = faiss.IndexFlatL2(dim) if mutation == 'metric' else faiss.IndexFlatIP(dim)
                    index.add(np.zeros((1 if mutation == 'count' else 2, dim), np.float32))
                    faiss.write_index(index, str(paths['index']))
                    self.update_meta(paths['ann_meta'], 'index_sha256', self.sha(paths['index']))
                with self.assertRaises(ValueError):
                    self.load(root)

    def test_actual_supported_ivf_types_and_strict_nprobe(self):
        for kind in ('IVFFlat', 'IVFPQ'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                paths = self.fixture(root, index_kind=kind)
                online = self.load(root)
                self.assertTrue(online.enabled)
                self.assertEqual(online.ann_index.nprobe, 2)
                original = paths['ann_meta'].read_bytes()
                for field, value in [('nprobe', 0), ('nprobe', 3), ('nprobe', True),
                                     ('nprobe', 1.), ('nlist', 3), ('nlist', True)]:
                    with self.subTest(field=field, value=value):
                        self.update_meta(paths['ann_meta'], field, value)
                        with self.assertRaises(ValueError):
                            self.load(root)
                        paths['ann_meta'].write_bytes(original)
                document = json.loads(original)
                del document['nprobe']
                paths['ann_meta'].write_text(json.dumps(document))
                with self.assertRaisesRegex(ValueError, 'nprobe'):
                    self.load(root)


if __name__ == '__main__':
    unittest.main()
