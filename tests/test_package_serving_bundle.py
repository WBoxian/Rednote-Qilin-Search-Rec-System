"""Offline bundle packaging tests; every source/output is inside a new /tmp fixture workspace."""
import json,tempfile,unittest
from pathlib import Path
import faiss,numpy as np,torch
from scipy import sparse
from scripts.package_serving_bundle import package_bundle,parser
from backend.online.bundle import file_sha256,required_asset_names,validate_manifest
from training.history import HISTORY_ORDER


def fixture(fixture_root:Path,scene='rec'):
    # This function receives only TemporaryDirectory roots, never the real project path.
    assert fixture_root.resolve().is_relative_to(Path(tempfile.gettempdir()).resolve())
    rank=fixture_root/'artifacts/rank';experiment=fixture_root/'artifacts/experiment';epoch=experiment/'dssm/epoch_3'
    rank.mkdir(parents=True);epoch.mkdir(parents=True);(experiment/'index').mkdir()
    features=['image_num','history_mean_cosine']
    (rank/'manifest.json').write_text(json.dumps({'scene':scene,'key':'search_idx' if scene=='search' else 'request_idx','features':features,
        'trees':{'chosen':{'config':{'kind':'classifier'}}},'dien':{'score_contract':'probability_v2','numeric_features':features+['coarse_score'],'architecture':{'inference_precision':'fp32'},'chosen':{'weight':.5}}}))
    (rank/'gbdt.txt').write_bytes(b'fixture tree');(rank/'dien.pt').write_bytes(b'fixture fine weights')
    user_state={'u_emb_user.weight':torch.arange(6,dtype=torch.float32).reshape(3,2),'user_mask_profile':torch.tensor(True)}
    if scene=='rec':user_state['seq_mask_padding']=torch.tensor(True)
    item_state={'i_emb_item.weight':torch.eye(3)}
    full={**user_state,**item_state}
    meta={'scene':scene,'history_contract':HISTORY_ORDER,'sequence_padding_contract':'packed_masked_v1' if scene=='rec' else 'legacy_attention_only_v0',
        'masked_user_dense':True,'known_user_ids':[0,1],'unknown_user_id':2,'cat_vocabs':{'user_idx':3,'note_idx':3},'hidden_dim':2}
    prefix=f'dssm_{scene}_optimized'
    torch.save(full,epoch/f'{prefix}.pt');torch.save({'state_dict':user_state,'meta':meta},epoch/f'{prefix}_user_tower.pt')
    torch.save({'state_dict':item_state,'meta':meta},epoch/f'{prefix}_item_tower.pt')
    vectors=np.array([[1.,0.],[0.,1.],[.7,.7]],np.float32);vectors.tofile(epoch/'item_vectors.bin')
    np.save(experiment/'dssm/item_ids.npy',np.arange(3,dtype=np.int64))
    selected={'epoch':3,'directory':str(epoch),'validation':{'Recall@100':.1},'checkpoint_sha256':file_sha256(epoch/f'{prefix}.pt'),
        'item_vectors_sha256':file_sha256(epoch/'item_vectors.bin')}
    (experiment/'dssm/selected.json').write_text(json.dumps(selected))
    index=faiss.IndexFlatIP(2);index.add(vectors);faiss.write_index(index,str(experiment/'index/ivfpq.faiss'))
    (experiment/'index/selected.json').write_text(json.dumps({'nprobe':1,'index_sha256':file_sha256(experiment/'index/ivfpq.faiss'),
        'selected_item_vectors_sha256':selected['item_vectors_sha256'],'selected_checkpoint_sha256':selected['checkpoint_sha256']}))
    policy={'strategy':'weighted_rrf','weights':{'ann':1,'bm25':2} if scene=='search' else {'ann':1,'swing':2,'usercf':1},'rrf_k':60,'route_topk':1000,'topk':500}
    extra=[]
    if scene=='rec':
        graph=fixture_root/'artifacts/graph';graph.mkdir()
        sparse.save_npz(graph/'graph.npz',sparse.csr_matrix([[1.,0.,1.],[0.,1.,1.]]))
        sparse.save_npz(graph/'swing.npz',sparse.eye(3,format='csr'))
        np.save(graph/'users.npy',np.array([0,1]));np.save(graph/'items.npy',np.array([0,1,2]))
        (graph/'northstar_metrics.json').write_text(json.dumps({'last_train_event':9.,'test_start':10,'train_requests':2}))
        policy.update(cf_graph_dir='artifacts/graph',cf_config={'fold_in':True,'history_weight':.25,'neighbors':2})
    else:
        lexical=fixture_root/'artifacts/lexical.sqlite';lexical.write_bytes(b'immutable lexical fixture')
        lexical.with_suffix('.sqlite.manifest.json').write_text(json.dumps({'index_sha256':file_sha256(lexical)}))
        encoder=fixture_root/'artifacts/encoder';encoder.mkdir()
        for name in ['config.json','tokenizer_config.json','tokenizer.json','model.safetensors']:(encoder/name).write_bytes(b'{}')
        extra=['--bm25-index',str(lexical),'--query-encoder-dir',str(encoder),'--query-encoder-revision','frozen-commit']
    policy_path=fixture_root/'artifacts/policy.json';policy_path.write_text(json.dumps(policy))
    for modality in ['text','img']:
        folder=fixture_root/f'embeddings/note_{modality}_emb';folder.mkdir(parents=True)
        (folder/f'note_{modality}_map.json').write_bytes(b'opaque vector map fixture');(folder/f'note_{modality}.bin').write_bytes(b'opaque vector bytes fixture')
    vocab=fixture_root/'features/vocab_dict';vocab.mkdir(parents=True)
    for name in ['gender','platform','age','location','taxonomy1_id','taxonomy2_id','taxonomy3_id']:(vocab/f'{name}.pkl').write_bytes(b'opaque fixed vocab fixture')
    for folder in ['notes','user_feat','search_train' if scene=='search' else 'recommendation_train']:
        source=fixture_root/'datasets'/folder;source.mkdir(parents=True);(source/'fixture.parquet').write_bytes(b'opaque immutable source fixture')
    argv=['--scene',scene,'--rank-dir',str(rank),'--recall-experiment-dir',str(experiment),'--policy-file',str(policy_path),
        '--training-cutoff-exclusive','10','--input-cap','2','--coarse-cap','1','--output-dir',str(fixture_root/'release/new')]+extra
    return parser().parse_args(argv),experiment,epoch


class PackageServingBundleTests(unittest.TestCase):
    def test_search_and_rec_packages_bind_all_assets_without_activation(self):
        for scene in ['search','rec']:
            with self.subTest(scene=scene),tempfile.TemporaryDirectory(prefix='qilin-package-test-') as temporary_test_dir:
                fixture_root=Path(temporary_test_dir);args,experiment,epoch=fixture(fixture_root,scene)
                before=file_sha256(epoch/f'dssm_{scene}_optimized_user_tower.pt')
                output,manifest=package_bundle(args,fixture_root)
                self.assertTrue(output.is_file());validate_manifest(manifest,scene,fixture_root,tag='hard')
                self.assertTrue(required_asset_names(manifest,fixture_root).issubset(manifest['assets']))
                self.assertFalse(list(fixture_root.rglob('active_bundle.json')))
                self.assertEqual(before,file_sha256(epoch/f'dssm_{scene}_optimized_user_tower.pt'))
                self.assertEqual(json.loads((fixture_root/manifest['recall_dir']/f'index/dssm_{scene}_hard_item_map.json').read_text())['0'],0)
                if scene=='rec':
                    self.assertFalse((fixture_root/'artifacts/graph/meta.json').exists())
                    self.assertTrue((fixture_root/manifest['recall_policy']['cf_graph_dir']/'meta.json').is_file())
    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory(prefix='qilin-package-test-') as temporary_test_dir:
            fixture_root=Path(temporary_test_dir);args,_,_=fixture(fixture_root)
            output,manifest=package_bundle(args,fixture_root);before=output.read_bytes()
            with self.assertRaises(FileExistsError):package_bundle(args,fixture_root)
            self.assertEqual(output.read_bytes(),before)
    def test_legacy_unmasked_or_unpacked_checkpoints_are_rejected(self):
        for flag in ['user_mask_profile','seq_mask_padding']:
            with self.subTest(flag=flag),tempfile.TemporaryDirectory(prefix='qilin-package-test-') as temporary_test_dir:
                fixture_root=Path(temporary_test_dir);args,_,epoch=fixture(fixture_root)
                path=epoch/'dssm_rec_optimized_user_tower.pt';payload=torch.load(path,map_location='cpu',weights_only=True)
                del payload['state_dict'][flag];torch.save(payload,path)
                with self.assertRaises(ValueError):package_bundle(args,fixture_root)
                self.assertFalse(args.output_dir.exists())
    def test_escape_symlink_and_model_index_mismatch_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix='qilin-package-test-') as temporary_test_dir,tempfile.TemporaryDirectory(prefix='qilin-package-outside-') as outside_test_dir:
            fixture_root=Path(temporary_test_dir);args,experiment,_=fixture(fixture_root)
            (fixture_root/'escape').symlink_to(Path(outside_test_dir),target_is_directory=True)
            args.output_dir=fixture_root/'escape/bundle'
            with self.assertRaisesRegex(ValueError,'escapes'):package_bundle(args,fixture_root)
            self.assertFalse((Path(outside_test_dir)/'bundle').exists())
            args.output_dir=fixture_root/'release/new'
            path=experiment/'index/selected.json';payload=json.loads(path.read_text());payload['selected_item_vectors_sha256']='0'*64;path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError,'different selected model'):package_bundle(args,fixture_root)
    def test_rank_gate_override_is_copied_and_later_source_tampering_is_detected(self):
        with tempfile.TemporaryDirectory(prefix='qilin-package-test-') as temporary_test_dir:
            fixture_root=Path(temporary_test_dir);args,_,_=fixture(fixture_root)
            alternate=fixture_root/'artifacts/rank_override.json';meta=json.loads((args.rank_dir/'manifest.json').read_text());meta['dien']['chosen']['weight']=0.;alternate.write_text(json.dumps(meta));args.rank_manifest=alternate
            output,manifest=package_bundle(args,fixture_root)
            copied=json.loads((fixture_root/manifest['rank_dir']/'manifest.json').read_text());self.assertEqual(copied['dien']['chosen']['weight'],0.);self.assertEqual(copied['dien']['score_contract'],'probability_v2')
            source=fixture_root/'embeddings/note_text_emb/note_text.bin';source.write_bytes(b'tampered fixture')
            with self.assertRaisesRegex(ValueError,'asset mismatch'):validate_manifest(manifest,'rec',fixture_root)

    def test_runtime_code_is_bound_and_recall_only_budget_is_explicit(self):
        with tempfile.TemporaryDirectory(prefix='qilin-package-test-') as temporary_test_dir:
            fixture_root=Path(temporary_test_dir);args,_,_=fixture(fixture_root)
            source=fixture_root/'src/backend/online/pipeline.py';source.parent.mkdir(parents=True);source.write_text('original runtime')
            args.ranking_strategy='recall_only';args.input_cap=args.coarse_cap=500
            _,manifest=package_bundle(args,fixture_root)
            self.assertIn('src/backend/online/pipeline.py',manifest['assets'])
            self.assertFalse(manifest['schema_enabled'])
            source.write_text('modified runtime')
            with self.assertRaisesRegex(ValueError,'asset mismatch'):validate_manifest(manifest,'rec',fixture_root)

if __name__=='__main__':unittest.main()