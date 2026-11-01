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


# Real small trained heads exercise the same release format as production artifacts.
from training.content_adapter_release import export_content_adapter_release
from tests.test_rec_residual_driver import _TrainedReleaseFixture
from recall.learned_context_fusion import FEATURE_NAMES
from recall.residual_dssm import model_from_checkpoint
import lightgbm as lgb


def adapted_fixture(root, dim=128, learned=False):
    args, experiment, _ = fixture(root)
    trained = root / 'artifacts/trained'
    trained.mkdir()
    source = _TrainedReleaseFixture(trained, dim)
    base_meta_path = source.base.with_suffix('.meta.json')
    base_meta = json.loads(base_meta_path.read_text())
    base_meta['history_contract'] = HISTORY_ORDER
    base_meta_path.write_text(json.dumps(base_meta))
    release = export_content_adapter_release(**source.arguments()).parent
    args.model_prefix = 'dssm_rec_residual'
    vectors = np.zeros((3, 256), np.float32)
    vectors[:, :3] = np.eye(3)
    vectors.tofile(release / 'item_vectors.bin')
    selected = dict(epoch=2, directory=str(release),
        checkpoint_sha256=file_sha256(release / 'dssm_rec_residual.pt'),
        item_vectors_sha256=file_sha256(release / 'item_vectors.bin'))
    (experiment / 'dssm/selected.json').write_text(json.dumps(selected))
    index = faiss.IndexFlatIP(256)
    index.add(vectors)
    faiss.write_index(index, str(experiment / 'index/ivfpq.faiss'))
    (experiment / 'index/selected.json').write_text(json.dumps(dict(
        index_sha256=file_sha256(experiment / 'index/ivfpq.faiss'),
        selected_item_vectors_sha256=selected['item_vectors_sha256'],
        selected_checkpoint_sha256=selected['checkpoint_sha256'])))
    if learned:
        fusion = root / 'artifacts/learned'
        fusion.mkdir()
        rng = np.random.default_rng(42)
        data = lgb.Dataset(rng.normal(size=(32, 23)).astype(np.float32),
            label=np.arange(32) % 2, feature_name=list(FEATURE_NAMES))
        model = lgb.train(dict(objective='binary', num_leaves=3, min_data_in_leaf=1,
            verbosity=-1, num_threads=2), data, num_boost_round=2)
        model.save_model(str(fusion / 'model.txt'))
        np.save(fusion / 'ids.npy', np.arange(3, dtype=np.int64))
        np.save(fusion / 'mean.npy', np.load(trained / 'pca/mean.npy'))
        np.save(fusion / 'components.npy', np.load(trained / 'pca/components.npy'))
        np.eye(3, 128, dtype=np.float32).tofile(fusion / 'item_pca.bin')
        paths = dict(model=fusion / 'model.txt', support=fusion / 'support.json',
            pca_vectors=fusion / 'item_pca.bin', pca_ids=fusion / 'ids.npy',
            pca_mean=fusion / 'mean.npy', pca_components=fusion / 'components.npy')
        support = dict(version=1, history_contract=HISTORY_ORDER,
            training_cutoff_exclusive=10, num_items=3, dim=128,
            item_counts=[[0, 2], [1, 1]], user_counts=[[0, 2]], graph_items=[0, 1, 2])
        support.update({key + '_sha256': file_sha256(path)
                        for key, path in paths.items() if key != 'support'})
        paths['support'].write_text(json.dumps(support))
        policy = json.loads(args.policy_file.read_text())
        policy.update(strategy='learned_rrf', context_policy='learned_filter_history',
            weights=dict(ann=2, swing=1, usercf=1), learned_fusion=dict(version=1,
                feature_schema='rec_context_23_v1', alpha=.5,
                prior_precision='float32_rrf211_v1',
                **{key: path.relative_to(root).as_posix() for key, path in paths.items()}))
        args.policy_file.write_text(json.dumps(policy))
        rank_path = args.rank_dir / 'manifest.json'
        rank = json.loads(rank_path.read_text())
        rank['dien']['inference_contract'] = 'request_fp32_v1'
        rank_path.write_text(json.dumps(rank))
        runtime = root / 'src/recall/learned_context_fusion.py'
        runtime.parent.mkdir(parents=True)
        runtime.write_bytes((Path(__file__).resolve().parents[1] / 'src/recall/learned_context_fusion.py').read_bytes())
    return args, experiment, release


class AdaptedServingPackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_real_full_partials_and_exact_flatip_without_nprobe_are_packaged(self):
        for dim in (128, 768):
            with self.subTest(dim=dim), tempfile.TemporaryDirectory(prefix='qilin-package-test-') as tmp:
                root = Path(tmp)
                args, _, source = adapted_fixture(root, dim)
                _, bundle = package_bundle(args, root)
                folder = root / bundle['recall_dir'] / 'models'
                full = folder / 'dssm_rec_hard.pt'
                self.assertEqual(file_sha256(full), file_sha256(source / 'dssm_rec_residual.pt'))
                model = model_from_checkpoint(full)
                self.assertEqual(model.checkpoint_meta['content_input_dim'], dim)
                for role in ('user', 'item'):
                    partial = torch.load(folder / f'dssm_rec_hard_{role}_tower.pt', weights_only=True)
                    self.assertEqual(partial['meta']['tower'], role)
                    self.assertEqual(partial['meta']['exported_full_checkpoint_sha256'], file_sha256(full))
                self.assertIn(full.relative_to(root).as_posix(), bundle['assets'])
                ann = json.loads((root / bundle['recall_dir'] / 'index/dssm_rec_hard_ivfpq_meta.json').read_text())
                self.assertEqual(ann['index_type'], 'FlatIP')
                self.assertNotIn('nprobe', ann)
                from backend.online.recall.dssm import DSSMRecaller
                recaller = DSSMRecaller.__new__(DSSMRecaller)
                recaller.scene, recaller.tag = 'rec', 'hard'
                recaller.model_family = model.checkpoint_meta['model_family']
                indexes = root / bundle['recall_dir'] / 'index'
                ann_path = indexes / 'dssm_rec_hard_ivfpq.faiss'
                ann_meta, item_meta = recaller._adapted_asset_metadata(model.checkpoint_meta,
                    indexes / 'dssm_rec_hard_ivfpq_meta.json', indexes / 'dssm_rec_hard_item_meta.json',
                    ann_path, indexes / 'dssm_rec_hard_item_emb.bin', indexes / 'dssm_rec_hard_row2note.npy')
                self.assertEqual(recaller._adapted_index_contract(faiss.read_index(str(ann_path)),
                    ann_meta, item_meta, 3), 'FlatIP')
                self.assertFalse(list(root.rglob('active_bundle.json')))

    def test_selected_interest_and_head_policies_bind_actual_runtime_sources(self):
        with tempfile.TemporaryDirectory(prefix='qilin-package-test-') as tmp:
            root = Path(tmp)
            args, _, _ = adapted_fixture(root, learned=True)
            policy = json.loads(args.policy_file.read_text())
            policy['ann_query'] = dict(version=1, strategy='adapted_k3_v1')
            policy['head_guard'] = dict(version=1, strategy='coarse100_prior10_v1')
            args.policy_file.write_text(json.dumps(policy))
            for filename in ('multi_interest.py', 'head_guard.py'):
                target = root/'src/recall'/filename
                target.write_bytes((Path(__file__).resolve().parents[1]/'src/recall'/filename).read_bytes())
            _, bundle = package_bundle(args, root)
            for filename in ('multi_interest.py', 'head_guard.py'):
                name = 'src/recall/' + filename
                self.assertEqual(bundle['assets'][name], file_sha256(root/name))
            self.assertEqual(bundle['recall_policy']['ann_query'], policy['ann_query'])
            self.assertEqual(bundle['recall_policy']['head_guard'], policy['head_guard'])
            self.assertFalse(list(root.rglob('active_bundle.json')))

    def test_corrupt_active_heads_roles_or_full_binding_are_rejected(self):
        for mutation in ('missing_head', 'shape', 'role', 'full_sha', 'nonfinite', 'dtype'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory(prefix='qilin-package-test-') as tmp:
                root = Path(tmp)
                args, _, source = adapted_fixture(root)
                path = source / 'dssm_rec_residual_user_tower.pt'
                payload = torch.load(path, weights_only=True)
                if mutation == 'missing_head':
                    del payload['state_dict']['content_query_adapter.2.bias']
                elif mutation == 'shape':
                    payload['state_dict']['content_query_adapter.2.bias'] = torch.zeros(129)
                elif mutation == 'role':
                    payload['meta']['tower'] = 'item'
                elif mutation == 'full_sha':
                    payload['meta']['exported_full_checkpoint_sha256'] = '0' * 64
                elif mutation == 'nonfinite':
                    payload['state_dict']['content_query_adapter.2.bias'][0] = float('nan')
                else:
                    payload['state_dict']['content_query_adapter.2.bias'] = payload['state_dict']['content_query_adapter.2.bias'].double()
                torch.save(payload, path)
                with self.assertRaises((ValueError, RuntimeError)):
                    package_bundle(args, root)
                self.assertFalse(args.output_dir.exists())

    def test_six_learned_assets_are_copied_bound_and_independent_of_source(self):
        with tempfile.TemporaryDirectory(prefix='qilin-package-test-') as tmp:
            root = Path(tmp)
            args, _, _ = adapted_fixture(root, learned=True)
            source_policy = json.loads(args.policy_file.read_text())
            _, bundle = package_bundle(args, root)
            provenance = bundle['packaging_provenance']['learned_fusion_sources']
            self.assertEqual(set(provenance), {'model','support','pca_vectors','pca_ids','pca_mean','pca_components'})
            for key, source in provenance.items():
                packaged = bundle['recall_policy']['learned_fusion'][key]
                self.assertNotEqual(packaged, source_policy['learned_fusion'][key])
                self.assertEqual(bundle['assets'][packaged], source['sha256'])
                self.assertTrue((root / packaged).is_file())
            from backend.online.bundle import ServingBundle
            from backend.online.ranking.inference import REQUEST_FP32, configure_request_inference
            previous = (torch.get_num_threads(), torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark)
            try:
                configure_request_inference(REQUEST_FP32)
                ready = ServingBundle(bundle, None, root)._get_learned_fusion()
                np.testing.assert_array_equal(ready.ids, np.arange(3, dtype=np.int64))
            finally:
                torch.set_num_threads(previous[0])
                torch.backends.cuda.matmul.allow_tf32 = previous[1]
                torch.backends.cudnn.allow_tf32 = previous[2]
                torch.backends.cudnn.benchmark = previous[3]
            original_model = root / source_policy['learned_fusion']['model']
            original_model.write_text('source changed after packaging')
            validate_manifest(bundle, 'rec', root)
            copied_model = root / bundle['recall_policy']['learned_fusion']['model']
            copied_model.write_text('packaged model corrupted')
            with self.assertRaisesRegex(ValueError, 'asset mismatch'):
                validate_manifest(bundle, 'rec', root)

    def test_ivf_requires_frozen_nprobe_while_flatip_does_not(self):
        with tempfile.TemporaryDirectory(prefix='qilin-package-test-') as tmp:
            root = Path(tmp)
            args, experiment, epoch = fixture(root)
            vectors = np.fromfile(epoch / 'item_vectors.bin', dtype=np.float32).reshape(3, 2)
            threads = faiss.omp_get_max_threads()
            try:
                faiss.omp_set_num_threads(1)
                index = faiss.IndexIVFFlat(faiss.IndexFlatIP(2), 2, 1, faiss.METRIC_INNER_PRODUCT)
                index.train(np.random.default_rng(42).normal(size=(64, 2)).astype(np.float32))
                index.add(vectors)
            finally:
                faiss.omp_set_num_threads(threads)
            path = experiment / 'index/ivfpq.faiss'
            faiss.write_index(index, str(path))
            selection_path = experiment / 'index/selected.json'
            selection = json.loads(selection_path.read_text())
            selection['index_sha256'] = file_sha256(path)
            del selection['nprobe']
            selection_path.write_text(json.dumps(selection))
            with self.assertRaisesRegex(ValueError, 'nprobe'):
                package_bundle(args, root)
            self.assertFalse(args.output_dir.exists())
            selection['nprobe'] = 1
            selection_path.write_text(json.dumps(selection))
            _, bundle = package_bundle(args, root)
            meta = json.loads((root / bundle['recall_dir'] / 'index/dssm_rec_hard_ivfpq_meta.json').read_text())
            self.assertEqual(meta['nprobe'], 1)

    def test_wrong_support_cross_sha_and_wrong_model_schema_are_rejected(self):
        for mutation in ('support', 'model', 'cutoff'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory(prefix='qilin-package-test-') as tmp:
                root = Path(tmp)
                args, _, _ = adapted_fixture(root, learned=True)
                config = json.loads(args.policy_file.read_text())['learned_fusion']
                support_path = root / config['support']
                support = json.loads(support_path.read_text())
                if mutation == 'support':
                    support['model_sha256'] = '0' * 64
                elif mutation == 'cutoff':
                    support['training_cutoff_exclusive'] = 11
                else:
                    path = root / config['model']
                    path.write_text(path.read_text().replace('feature_names=', 'feature_names=wrong_', 1))
                    support['model_sha256'] = file_sha256(path)
                support_path.write_text(json.dumps(support))
                with self.assertRaises(ValueError):
                    package_bundle(args, root)
                self.assertFalse(args.output_dir.exists())


if __name__ == '__main__':
    unittest.main()
