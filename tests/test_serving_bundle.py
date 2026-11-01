from concurrent.futures import ThreadPoolExecutor

import copy
import json
import os
from types import SimpleNamespace
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from backend.online.bundle import ServingBundle, bundle_identity, file_sha256, get_active_bundle, required_asset_names, resolve_asset, validate_manifest, verify_asset
from backend.online.ranking.stack import RankStackPredictor, shared_mapped_vectors
from backend.online.ranking.service import run_ranking
from training.recall_prior import blend_recall_prior
import backend.online.bundle as bundle_module
from training.history import HISTORY_ORDER


def fixture(root, scene="search"):
    folder = "search_train" if scene == "search" else "recommendation_train"
    for name in (f"datasets/{folder}/train.parquet", "datasets/notes/notes.parquet", "datasets/user_feat/users.parquet"):
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"immutable source fixture")
    manifest = {"version": 1, "scene": scene, "tag": "hard", "history_order": HISTORY_ORDER,
        "selection_split": "train_internal_validation", "allow_dpr": False, "training_cutoff_exclusive": 10,
        "rank_dir": "artifacts/rank", "recall_dir": "artifacts/recall", "input_cap": 160, "coarse_cap": 80,
        "recall_policy": {"strategy": "weighted_rrf", "weights": {"ann": 1., "bm25": 2.} if scene == "search" else {"ann": 1., "swing": 4., "usercf": 1.}, "route_topk": 1000, "topk": 500, "rrf_k": 60}}
    if scene == "search":
        manifest["bm25_index"] = "artifacts/lexical/index.sqlite"
        manifest["query_encoder_dir"] = "artifacts/query_encoder/commit"
        manifest["query_encoder_revision"] = "frozen-training-commit"
        encoder = root/manifest["query_encoder_dir"]
        encoder.mkdir(parents=True)
        for name in ["config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors"]:
            (encoder/name).write_bytes(b"frozen encoder fixture")
    else:
        manifest["recall_policy"].update(cf_graph_dir="artifacts/graph", cf_config={"history_weight": .25, "fold_in": True, "neighbors": 200})
    for name in required_asset_names(manifest, root):
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"immutable model fixture")
    features = ["image_num", "history_mean_cosine"]
    (root/manifest["rank_dir"]/"manifest.json").write_text(json.dumps({"scene": scene,
        "key": "search_idx" if scene == "search" else "request_idx", "features": features,
        "dien": {"numeric_features": features+["coarse_score"], "architecture": {"inference_precision": "fp32"}}}))
    manifest["assets"] = {name: file_sha256(root/name) for name in required_asset_names(manifest, root)}
    manifest["bundle_id"] = bundle_identity(manifest)
    return manifest


def resign(manifest):
    manifest["bundle_id"] = bundle_identity(manifest)
    return manifest


def learned_fixture(root):
    import lightgbm as lgb
    from recall.learned_context_fusion import FEATURE_NAMES
    from backend.online.ranking.inference import REQUEST_FP32, configure_request_inference
    import recall.learned_context_fusion as core
    manifest = fixture(root, 'rec')
    policy = manifest['recall_policy']
    paths = {key: 'artifacts/fusion/' + filename for key, filename in {
        'model': 'model.txt', 'support': 'support.json', 'pca_vectors': 'vectors.bin',
        'pca_ids': 'ids.npy', 'pca_mean': 'mean.npy', 'pca_components': 'components.npy'}.items()}
    policy.update(strategy='learned_rrf', context_policy='learned_filter_history',
        weights={'ann': 2, 'swing': 1, 'usercf': 1}, learned_fusion={**paths, 'version': 1,
            'feature_schema': 'rec_context_23_v1', 'alpha': .5, 'prior_precision': 'float32_rrf211_v1'})
    folder = root/'artifacts/fusion'
    folder.mkdir()
    inputs = np.random.default_rng(42).normal(size=(12, 23)).astype(np.float32)
    model = lgb.train({'objective': 'regression', 'num_threads': 2, 'verbosity': -1,
        'min_data_in_leaf': 1, 'min_data_in_bin': 1, 'seed': 42},
        lgb.Dataset(inputs, label=np.arange(12) % 2, feature_name=list(FEATURE_NAMES)), num_boost_round=2)
    model.save_model(str(root/paths['model']))
    np.save(root/paths['pca_ids'], np.array([0, 7, 20, 30], np.int64))
    np.save(root/paths['pca_mean'], np.zeros(768, np.float32))
    components = np.zeros((128, 768), np.float32)
    components[:, :128] = np.eye(128)
    np.save(root/paths['pca_components'], components)
    np.eye(128, dtype=np.float32)[:4].tofile(root/paths['pca_vectors'])
    runtime = root/'src/recall/learned_context_fusion.py'
    runtime.parent.mkdir(parents=True)
    runtime.write_bytes(Path(core.__file__).read_bytes())
    support = {'version': 1, 'history_contract': HISTORY_ORDER, 'training_cutoff_exclusive': 10,
        'num_items': 4, 'dim': 128, 'item_counts': [[0, 2], [7, 1]], 'user_counts': [[7, 4]],
        'graph_items': [0, 7], **{key+'_sha256': file_sha256(root/paths[key]) for key in
            ['model', 'pca_vectors', 'pca_ids', 'pca_mean', 'pca_components']}}
    (root/paths['support']).write_text(json.dumps(support))
    rank = root/manifest['rank_dir']/'manifest.json'
    metadata = json.loads(rank.read_text())
    metadata['dien']['inference_contract'] = REQUEST_FP32
    rank.write_text(json.dumps(metadata))
    manifest['assets'] = {name: file_sha256(root/name) for name in required_asset_names(manifest, root)}
    configure_request_inference(REQUEST_FP32)
    return resign(manifest)


class LearnedServingBundleTests(unittest.TestCase):
    def refresh(self, root, manifest, key):
        paths = manifest['recall_policy']['learned_fusion']
        if key != 'support':
            support = json.loads((root/paths['support']).read_text())
            support[key+'_sha256'] = file_sha256(root/paths[key])
            (root/paths['support']).write_text(json.dumps(support))
        for name in required_asset_names(manifest, root):
            manifest['assets'][name] = file_sha256(root/name)
        resign(manifest)

    def test_policy_requires_fixed_schema_paths_budgets_and_all_required_bindings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = learned_fixture(root)
            validate_manifest(manifest, 'rec', root)
            config = manifest['recall_policy']['learned_fusion']
            for field, value in [('version', True), ('alpha', 1.), ('feature_schema', 'other'),
                                 ('prior_precision', 'float64'), ('model', str(root/config['model']))]:
                changed = copy.deepcopy(manifest)
                changed['recall_policy']['learned_fusion'][field] = value
                with self.subTest(field=field), self.assertRaises(ValueError):
                    validate_manifest(resign(changed), 'rec', root)
            for name in [*(config[k] for k in bundle_module.LEARNED_FUSION_PATHS), 'src/recall/learned_context_fusion.py']:
                changed = copy.deepcopy(manifest)
                del changed['assets'][name]
                with self.subTest(missing=name), self.assertRaisesRegex(ValueError, 'omits required'):
                    validate_manifest(resign(changed), 'rec', root)
            for changed in [{'weights': {'ann': 1, 'swing': 2, 'usercf': 1}}, {'rrf_k': 20},
                            {'route_topk': 500}, {'topk': 600}, {'context_policy': None}]:
                mutation = copy.deepcopy(manifest)
                mutation['recall_policy'].update(changed)
                with self.assertRaises(ValueError):
                    validate_manifest(resign(mutation), 'rec', root)

    def test_startup_warms_actual_model_once_and_fuses_original_q0_under_contract(self):
        import torch
        from backend.online.ranking.inference import REQUEST_FP32, check_request_inference
        from recall.learned_context_fusion import LearnedContextFusion, RouteCandidates
        from recall.residual_dssm import raw_history_text_mean, project_content
        class Store:
            def get_batch_vecs(self, ids):
                values = np.zeros((len(ids), 768), np.float32)
                for row, item in enumerate(ids): values[row, 0 if item == 0 else 1] = 1.
                return torch.from_numpy(values)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = learned_fixture(root)
            active = root/'outputs/deploy/rec/hard/active_bundle.json'
            active.parent.mkdir(parents=True)
            active.write_text(json.dumps(manifest))
            with patch.object(bundle_module, 'RankStackPredictor'), patch.object(LearnedContextFusion, 'from_model_path', wraps=LearnedContextFusion.from_model_path) as constructor:
                serving = get_active_bundle('rec', 'hard', root)
                state = serving._get_learned_fusion()
                self.assertEqual(constructor.call_count, 1)
                routes = {'ann': RouteCandidates([0, 7, 20], [3., 2., 1.]),
                    'swing': RouteCandidates([30, 20], [2., 1.]), 'usercf': RouteCandidates([], [])}
                with patch.object(state.predictor, 'fuse', wraps=state.predictor.fuse) as shared:
                    actual = serving.fuse_learned_routes(routes, [0, 0, 7], 7, Store())
                q0 = project_content(torch.from_numpy(raw_history_text_mean([0, 0, 7], Store()).reshape(1, 768)), state.mean, state.components).numpy()[0]
                np.testing.assert_array_equal(shared.call_args.kwargs['q0_pca'], q0)
                self.assertEqual(actual.note_idx.tolist(), [20, 30])
                self.assertEqual(actual['rank'].tolist(), [1, 2])
                self.assertEqual(actual.score.dtype, np.dtype('float64'))
                self.assertFalse(state.ids.flags.writeable)
                self.assertFalse(state.vectors.flags.writeable)
                with self.assertRaises(TypeError): state.support.item_counts[0] = 99
                check_request_inference(REQUEST_FP32)
                self.assertEqual(torch.get_num_threads(), 4)
                serving.fuse_learned_routes({}, [], 7, Store())
                self.assertEqual(constructor.call_count, 1)

    def test_support_cutoff_hash_counts_and_history_reject_without_fallback(self):
        mutations = [('training_cutoff_exclusive', 9), ('version', True), ('dim', 64),
            ('history_contract', 'old'), ('num_items', True), ('model_sha256', '0'*64),
            ('item_counts', [[0, 1], [0, 2]]), ('user_counts', [[True, 1]]),
            ('graph_items', [0, 0]), ('graph_items', [99]), ('item_counts', [[99, 1]])]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = learned_fixture(root)
            path = root/manifest['recall_policy']['learned_fusion']['support']
            original = path.read_bytes()
            for field, value in mutations:
                with self.subTest(field=field, value=value):
                    support = json.loads(original)
                    support[field] = value
                    path.write_text(json.dumps(support))
                    self.refresh(root, manifest, 'support')
                    with self.assertRaises(ValueError): ServingBundle(manifest, None, root)._get_learned_fusion()
                    path.write_bytes(original)
            path.write_text('{bad json')
            self.refresh(root, manifest, 'support')
            with self.assertRaises(ValueError): ServingBundle(manifest, None, root)._get_learned_fusion()

    def test_actual_hash_and_wrong_model_columns_are_rejected(self):
        import lightgbm as lgb
        from backend.online.ranking.inference import REQUEST_FP32, configure_request_inference
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = learned_fixture(root)
            serving = ServingBundle(manifest, None, root)
            serving._get_learned_fusion()
            paths = manifest['recall_policy']['learned_fusion']
            model_path = root/paths['model']
            content = model_path.read_bytes()
            model_path.write_bytes(bytes([content[0]^1]) + content[1:])
            with self.assertRaisesRegex(ValueError, 'asset mismatch'): serving._get_learned_fusion()
            model_path.write_bytes(content)
            from recall.learned_context_fusion import FEATURE_NAMES
            names = ['wrong_column', *FEATURE_NAMES[1:]]
            model = lgb.train({'objective': 'regression', 'verbosity': -1, 'num_threads': 2},
                lgb.Dataset(np.zeros((8, 23), np.float32), label=np.arange(8), feature_name=names), num_boost_round=1)
            model.save_model(str(model_path))
            self.refresh(root, manifest, 'model')
            configure_request_inference(REQUEST_FP32)
            with self.assertRaisesRegex(ValueError, '23-feature schema'):
                ServingBundle(manifest, None, root)._get_learned_fusion()

    def test_pca_dtype_shape_basis_and_bytes_reject_after_cross_hash_is_updated(self):
        for mutation in ['mean_float64', 'basis_nonorthogonal', 'basis_nan', 'ids_duplicate', 'ids_int32', 'vector_length', 'vector_nan']:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = learned_fixture(root)
                paths = manifest['recall_policy']['learned_fusion']
                if mutation == 'mean_float64':
                    key = 'pca_mean'; np.save(root/paths[key], np.zeros(768, np.float64))
                elif mutation.startswith('basis'):
                    key = 'pca_components'; values = np.load(root/paths[key]); values[1] = values[0]
                    if mutation == 'basis_nan': values[0, 0] = np.nan
                    np.save(root/paths[key], values)
                elif mutation.startswith('ids'):
                    key = 'pca_ids'; np.save(root/paths[key], np.array([0, 0, 20, 30], np.int64) if mutation == 'ids_duplicate' else np.array([0, 7, 20, 30], np.int32))
                else:
                    key = 'pca_vectors'; values = np.fromfile(root/paths[key], np.float32)
                    if mutation == 'vector_length': values = values[:-1]
                    else: values[0] = np.nan
                    values.tofile(root/paths[key])
                self.refresh(root, manifest, key)
                with self.assertRaises(ValueError): ServingBundle(manifest, None, root)._get_learned_fusion()


class ServingBundleTests(unittest.TestCase):
    def test_weighted_f1_accepts_only_the_declared_eight_weight_recipes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);manifest=fixture(root,'rec')
            policy=manifest['recall_policy'];policy['context_policy']='weighted_filter_history'
            for ann in [1,2,4,8]:
                for swing in [1,2]:
                    policy['weights']={'ann':ann,'swing':swing,'usercf':1}
                    validate_manifest(resign(manifest),'rec',root)
            for changed in [{'weights':{'ann':3,'swing':1,'usercf':1}},
                            {'weights':{'ann':2,'swing':4,'usercf':1}},
                            {'weights':{'ann':2,'swing':1,'usercf':0}},
                            {'weights':{'ann':True,'swing':1,'usercf':1}},
                            {'route_topk':500},{'topk':600},{'rrf_k':20}]:
                mutation=copy.deepcopy(manifest);mutation['recall_policy'].update(changed)
                with self.assertRaisesRegex(ValueError,'weighted request-context'):
                    validate_manifest(resign(mutation),'rec',root)

    def test_request_numerical_contract_rejects_unknown_execution_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);manifest=fixture(root)
            path=root/manifest['rank_dir']/'manifest.json'
            metadata=json.loads(path.read_text())
            for contract in ['request_fp32_v1', 'unknown', {'padding':1024}]:
                metadata['dien']['inference_contract']=contract
                path.write_text(json.dumps(metadata))
                manifest['assets'][path.relative_to(root).as_posix()]=file_sha256(path)
                if contract == 'request_fp32_v1':
                    validate_manifest(resign(manifest),'search',root)
                else:
                    with self.assertRaisesRegex(ValueError,'request inference contract'):
                        validate_manifest(resign(manifest),'search',root)

    def test_optional_f1_schema_only_accepts_the_fixed_rec_history_filter_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);manifest=fixture(root,'rec')
            policy=manifest['recall_policy']
            policy.update(context_policy='filter_history',weights={'ann':1.,'swing':2.,'usercf':1.})
            validate_manifest(resign(manifest),'rec',root)
            for changed in [{'context_policy':'filter_history_context'},{'route_topk':500},{'rrf_k':10},
                            {'weights':{'ann':1.,'swing':4.,'usercf':1.}}]:
                mutation=copy.deepcopy(manifest);mutation['recall_policy'].update(changed)
                with self.assertRaisesRegex(ValueError,'context fusion contract'):
                    validate_manifest(resign(mutation),'rec',root)

    def test_diagnostic_snapshot_requires_consistent_content_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = fixture(root)
            snapshot = root/"artifacts/examples.json"
            snapshot.write_text('{}')
            manifest["evaluation"] = {"validation_snapshot": {"path": "artifacts/examples.json", "sha256": file_sha256(snapshot)}}
            with self.assertRaisesRegex(ValueError, "omits required"):
                validate_manifest(resign(manifest), "search", root)
            manifest["assets"]["artifacts/examples.json"] = file_sha256(snapshot)
            validate_manifest(resign(manifest), "search", root)
            manifest["evaluation"]["validation_snapshot"]["sha256"] = "0"*64
            with self.assertRaisesRegex(ValueError, "Diagnostic snapshot hash"):
                validate_manifest(resign(manifest), "search", root)

    def test_verified_bm25_bytes_are_hashed_once_and_tampering_still_rejects(self):
        from recall.lexical import build_index
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            notes = root/"notes"
            notes.mkdir()
            pd.DataFrame({"note_idx": [0, 1], "note_title": ["可爱猫", "城市旅行"], "note_content": ["小猫", "旅行"]}).to_parquet(notes/"notes.parquet", index=False)
            path = root/"bm25.sqlite"
            build_index(notes, path, cache_mib=16, memory_limit_mib=256)
            sidecar = path.with_suffix('.sqlite.manifest.json')
            assets = {item.name: file_sha256(item) for item in [path, sidecar]}
            serving = ServingBundle({"bm25_index": path.name, "assets": assets}, None, root)
            bundle_module._VERIFIED_ASSETS.clear()
            with patch.object(bundle_module, "file_sha256", wraps=file_sha256) as digest, patch("recall.lexical._sha256", side_effect=AssertionError("Reader must not hash already-verified DB bytes")):
                verify_asset(path, assets[path.name])
                verify_asset(sidecar, assets[sidecar.name])
                self.assertEqual(serving.bm25_candidates("可爱猫", 2).note_idx.tolist(), [0])
                serving.bm25_candidates("城市旅行", 2)
                self.assertEqual(sum(call.args[0] == path for call in digest.call_args_list), 1)
                with path.open('ab') as stream:
                    stream.write(b'tamper')
                with self.assertRaisesRegex(ValueError, "asset mismatch"):
                    serving.bm25_candidates("可爱猫", 2)
            serving._lexical.close()

    def test_optional_recall_prior_manifest_rejects_invalid_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = fixture(root)
            path = root/manifest["rank_dir"]/"manifest.json"
            metadata = json.loads(path.read_text())
            for prior in [{"alpha": -.1, "k": 60}, {"alpha": None, "k": 60}, {"alpha": .1, "k": 10}, {"alpha": True, "k": 60}]:
                metadata["dien"]["rank_prior"] = prior
                path.write_text(json.dumps(metadata))
                manifest["assets"][path.relative_to(root).as_posix()] = file_sha256(path)
                with self.assertRaisesRegex(ValueError, "recall-rank prior"):
                    validate_manifest(resign(manifest), "search", root)

    def test_actual_rank_predictor_applies_prior_per_request_and_service_retains_float64(self):
        predictor = RankStackPredictor.__new__(RankStackPredictor)
        predictor._lock = threading.RLock()
        predictor.group_key, predictor.weight, predictor.feature_cols = "request_idx", 0., []
        predictor.score_contract = "probability_v2"
        predictor.rank_prior = {"alpha": .1, "k": 60}
        predictor.coarse = SimpleNamespace(predict=lambda frame: np.linspace(.1, .9, len(frame), dtype=np.float32))
        predictor._frame = lambda frame, history: (frame.copy().reset_index(drop=True), None)
        predictor.model = Mock()
        frame = pd.DataFrame({"request_idx": [1, 1, 2, 2], "note_idx": [1, 2, 3, 4], "rank": [1, 2, 2, 1]})
        result = predictor.predict(frame)
        raw = np.linspace(.1, .9, 4, dtype=np.float32)
        expected = np.concatenate([blend_recall_prior([1, 2], raw[:2], .1), blend_recall_prior([2, 1], raw[2:], .1)])
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(result.dtype, np.dtype("float64"))
        ranked, _ = run_ranking(frame.iloc[:2], 1, 2, predictor.predict, ranking_policy="learned")
        self.assertEqual(ranked.final_score.dtype, np.dtype("float64"))
        expected_service = blend_recall_prior([1, 2], [.1, .9], .1)
        self.assertEqual(ranked.note_idx.tolist(), np.asarray([1, 2])[np.argsort(-expected_service)].tolist())
        predictor.model.assert_not_called()

    def test_search_encoder_load_failure_prevents_model_readiness(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = fixture(root)
            active = root/"outputs/deploy/search/hard/active_bundle.json"
            active.parent.mkdir(parents=True)
            active.write_text(json.dumps(manifest))
            with patch("backend.online.shared_query_encoder.get_shared_query_encoder", return_value=(None, None)), patch.object(bundle_module, "RankStackPredictor") as ranker:
                with self.assertRaisesRegex(RuntimeError, "before serving readiness"):
                    get_active_bundle("search", "hard", root)
                ranker.assert_not_called()

    def test_recall_only_requires_full_pool_and_explicit_schema_bypass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = fixture(root)
            manifest.update(ranking_strategy="recall_only", input_cap=500, coarse_cap=500, schema_enabled=False)
            validate_manifest(resign(manifest), "search", root)
            for mutation in [{"input_cap": 160, "coarse_cap": 80}, {"schema_enabled": True}, {"ranking_strategy": "arbitrary"}]:
                changed = {**manifest, **mutation}
                with self.assertRaisesRegex(ValueError, "Recall-only|Unsupported ranking"):
                    validate_manifest(resign(changed), "search", root)

    def test_valid_fully_bound_search_and_rec_bundles(self):
        for scene in ["search", "rec"]:
            with self.subTest(scene=scene), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = fixture(root, scene)
                validate_manifest(manifest, scene, root, tag="hard")

    def test_unrelated_asset_subset_cannot_authorize_unbound_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = fixture(root)
            (root/"unrelated.txt").write_text("unrelated")
            manifest["assets"] = {"unrelated.txt": file_sha256(root/"unrelated.txt")}
            with self.assertRaisesRegex(ValueError, "omits required"):
                validate_manifest(resign(manifest), "search", root)

    def test_mutating_checkpoint_with_same_size_rejects_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = fixture(root)
            target = root/manifest["rank_dir"]/"dien.pt"
            content = target.read_bytes()
            target.write_bytes(bytes([content[0]^1])+content[1:])
            with self.assertRaisesRegex(ValueError, "asset mismatch"):
                validate_manifest(manifest, "search", root)

    def test_unsafe_feature_schema_rejected_even_if_hash_and_identity_updated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = fixture(root)
            path = root/manifest["rank_dir"]/"manifest.json"
            metadata = json.loads(path.read_text())
            metadata["features"] = ["position"]
            metadata["dien"]["numeric_features"] = ["position", "coarse_score"]
            path.write_text(json.dumps(metadata))
            manifest["assets"][path.relative_to(root).as_posix()] = file_sha256(path)
            with self.assertRaisesRegex(ValueError, "feature schema"):
                validate_manifest(resign(manifest), "search", root)

    def test_wrong_history_test_selection_dpr_and_tag_rejected(self):
        mutations = [{"history_order": "legacy_raw_order_v0"}, {"selection_split": "official_test"}, {"allow_dpr": True}, {"tag": "easy"}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = fixture(root)
            for mutation in mutations:
                with self.subTest(mutation=mutation):
                    manifest = copy.deepcopy(original)
                    manifest.update(mutation)
                    with self.assertRaises(ValueError):
                        validate_manifest(resign(manifest), "search", root, tag="hard")

    def test_outside_bm25_or_cf_path_rejected_before_loading(self):
        for scene in ["search", "rec"]:
            with self.subTest(scene=scene), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = fixture(root, scene)
                if scene == "search":
                    manifest["bm25_index"] = "../../outside.sqlite"
                else:
                    manifest["recall_policy"]["cf_graph_dir"] = "../../outside_graph"
                with self.assertRaisesRegex(ValueError, "escapes"):
                    validate_manifest(resign(manifest), scene, root)

    def test_symlink_cannot_escape_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent/"workspace"
            root.mkdir()
            (parent/"outside").write_text("outside")
            (root/"alias").symlink_to(parent/"outside")
            with self.assertRaisesRegex(ValueError, "escapes"):
                resolve_asset(root, "alias")

    def test_missing_required_fine_or_index_binding_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = fixture(root)
            for missing in ["artifacts/rank/dien.pt", "artifacts/recall/index/dssm_search_hard_ivfpq.faiss", "features/vocab_dict/gender.pkl"]:
                with self.subTest(missing=missing):
                    manifest = copy.deepcopy(original)
                    del manifest["assets"][missing]
                    with self.assertRaisesRegex(ValueError, "omits required"):
                        validate_manifest(resign(manifest), "search", root)

    def test_invalid_route_weights_and_shorter_final_budget_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = fixture(root)
            mutations = [{"weights": {"ann": float("nan"), "bm25": 1.}}, {"weights": {"ann": 1., "dpr": 1.}}, {"topk": 100}]
            for mutation in mutations:
                with self.subTest(mutation=mutation):
                    manifest = copy.deepcopy(original)
                    manifest["recall_policy"].update(mutation)
                    with self.assertRaises(ValueError):
                        validate_manifest(resign(manifest), "search", root)

    def test_wrong_precision_rejected_before_instantiating_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = fixture(root)
            path = root/manifest["rank_dir"]/"manifest.json"
            metadata = json.loads(path.read_text())
            metadata["dien"]["architecture"]["inference_precision"] = "bf16"
            path.write_text(json.dumps(metadata))
            manifest["assets"][path.relative_to(root).as_posix()] = file_sha256(path)
            with self.assertRaisesRegex(ValueError, "precision"):
                validate_manifest(resign(manifest), "search", root)

    def test_unsupported_score_contract_and_nonclassifier_probability_teacher_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = fixture(root)
            path = root/original["rank_dir"]/"manifest.json"
            metadata = json.loads(path.read_text())
            for contract in ["unsupported", "probability_v2"]:
                with self.subTest(contract=contract):
                    manifest = copy.deepcopy(original)
                    changed = copy.deepcopy(metadata)
                    changed["dien"]["score_contract"] = contract
                    path.write_text(json.dumps(changed))
                    manifest["assets"][path.relative_to(root).as_posix()] = file_sha256(path)
                    with self.assertRaises(ValueError):
                        validate_manifest(resign(manifest), "search", root)

    def test_no_active_manifest_returns_none_without_model_load(self):
        with tempfile.TemporaryDirectory() as directory, patch("backend.online.bundle.RankStackPredictor") as model:
            self.assertIsNone(get_active_bundle("search", "hard", Path(directory)))
            model.assert_not_called()

    def test_shared_asset_hashes_are_reused_with_full_stat_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = fixture(root)
            bundle_module._VERIFIED_ASSETS.clear()
            with patch.object(bundle_module, "file_sha256", wraps=file_sha256) as digest:
                validate_manifest(manifest, "search", root)
                once = digest.call_count
                self.assertEqual(once, len(manifest["assets"]))
                validate_manifest(manifest, "search", root)
                self.assertEqual(digest.call_count, once)

    def test_cached_same_size_tamper_with_restored_mtime_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = fixture(root)
            validate_manifest(manifest, "search", root)
            path = root/manifest["rank_dir"]/"dien.pt"
            original = path.stat()
            content = path.read_bytes()
            path.write_bytes(bytes([content[0]^1])+content[1:])
            os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
            self.assertEqual(path.stat().st_mtime_ns, original.st_mtime_ns)
            with self.assertRaisesRegex(ValueError, "asset mismatch"):
                validate_manifest(manifest, "search", root)

    def test_readonly_vector_snapshots_are_shared_but_changed_content_metadata_reloads(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder/"note_text_map.json").write_text("{}")
            binary = folder/"note_text.bin"
            binary.write_bytes(b"one")
            def create(*args):
                return SimpleNamespace(ids=np.asarray([1]), positions=np.asarray([0]))
            with patch("backend.online.ranking.stack.MappedVectors", side_effect=create) as constructor:
                first = shared_mapped_vectors(folder, "note_text")
                second = shared_mapped_vectors(folder, "note_text")
                self.assertIs(first, second)
                self.assertEqual(constructor.call_count, 1)
                self.assertFalse(first.ids.flags.writeable)
                self.assertFalse(first.positions.flags.writeable)
                old = binary.stat()
                binary.write_bytes(b"two-with-new-size")
                os.utime(binary, ns=(old.st_atime_ns, old.st_mtime_ns))
                self.assertIsNot(shared_mapped_vectors(folder, "note_text"), first)
                self.assertEqual(constructor.call_count, 2)

    def test_zero_neural_weight_still_uses_frozen_request_normalization(self):
        predictor = RankStackPredictor.__new__(RankStackPredictor)
        predictor._lock = threading.RLock()
        predictor.group_key = "request_idx"
        predictor.weight = 0.
        predictor.feature_cols = []
        predictor.coarse = SimpleCoarse()
        predictor._frame = lambda frame, history: (frame.copy(), None)
        predictor.model = Mock()
        result = predictor.predict(pd.DataFrame({"request_idx": [1, 1, 2, 2]}))
        np.testing.assert_array_equal(result, [0., 1., 0., 1.])
        predictor.model.assert_not_called()


class SimpleCoarse:
    def predict(self, frame):
        return np.asarray([1., 2., 100., 200.], dtype=np.float32)




class ParallelAssetIntegrityTests(unittest.TestCase):
    def test_distinct_files_hash_concurrently(self):
        with tempfile.TemporaryDirectory() as directory:
            files = [Path(directory)/name for name in ('a.bin', 'b.bin')]
            for index, path in enumerate(files):
                path.write_bytes(bytes([index])*1024)
            expected = {path: file_sha256(path) for path in files}
            original = bundle_module.file_sha256
            both_entered = threading.Barrier(2)
            def observed_hash(path):
                both_entered.wait(timeout=3)
                return original(path)
            with patch.object(bundle_module, 'file_sha256', side_effect=observed_hash):
                with ThreadPoolExecutor(max_workers=2) as pool:
                    results = [pool.submit(verify_asset, path, expected[path]) for path in files]
                    for result in results:
                        result.result(timeout=4)

    def test_same_file_eight_waiters_hash_exactly_once(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'one.bin'
            path.write_bytes(b'one complete immutable file')
            expected, original = file_sha256(path), bundle_module.file_sha256
            ready = threading.Barrier(8)
            count = 0
            count_lock = threading.Lock()
            def observed_hash(value):
                nonlocal count
                with count_lock:
                    count += 1
                time.sleep(.02)
                return original(value)
            def worker():
                ready.wait(timeout=3)
                verify_asset(path, expected)
            with patch.object(bundle_module, 'file_sha256', side_effect=observed_hash):
                with ThreadPoolExecutor(max_workers=8) as pool:
                    list(pool.map(lambda _: worker(), range(8)))
            self.assertEqual(count, 1)

    def test_completed_cache_rejects_same_size_mutation_with_restored_mtime(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'mutable.bin'
            path.write_bytes(b'old!')
            expected = file_sha256(path)
            verify_asset(path, expected)
            before = path.stat()
            time.sleep(.002)
            path.write_bytes(b'new!')
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
            self.assertEqual(path.stat().st_size, before.st_size)
            self.assertEqual(path.stat().st_mtime_ns, before.st_mtime_ns)
            with self.assertRaisesRegex(ValueError, 'asset mismatch'):
                verify_asset(path, expected)

    def test_mid_hash_mutation_with_restored_mtime_never_seeds_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'mutating.bin'
            path.write_bytes(b'old!')
            expected, original = file_sha256(path), bundle_module.file_sha256
            before = path.stat()
            def mutate_after_read(value):
                result = original(value)
                time.sleep(.002)
                value.write_bytes(b'new!')
                os.utime(value, ns=(before.st_atime_ns, before.st_mtime_ns))
                return result
            with patch.object(bundle_module, 'file_sha256', side_effect=mutate_after_read):
                with self.assertRaisesRegex(ValueError, 'asset mismatch'):
                    verify_asset(path, expected)
            self.assertFalse(any(identity[0] == str(path.resolve()) for identity in bundle_module._VERIFIED_ASSETS))
            verify_asset(path, original(path))

    def test_invalid_digest_never_seeds_or_poison_locks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'digest.bin'
            path.write_bytes(b'actual bytes')
            for wrong in (None, 'short', '0'*64):
                with self.assertRaisesRegex(ValueError, 'asset mismatch'):
                    verify_asset(path, wrong)
            self.assertFalse(any(identity[0] == str(path.resolve()) for identity in bundle_module._VERIFIED_ASSETS))
            verify_asset(path, file_sha256(path))

    def test_manifest_uses_bounded_parallel_hash_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = fixture(root, 'rec')
            original = bundle_module.file_sha256
            lock, first_pair = threading.Lock(), threading.Barrier(2)
            active = maximum = calls = 0
            def observed_hash(path):
                nonlocal active, maximum, calls
                with lock:
                    active += 1
                    maximum = max(maximum, active)
                    calls += 1
                    ordinal = calls
                try:
                    if ordinal <= 2:
                        first_pair.wait(timeout=3)
                    time.sleep(.005)
                    return original(path)
                finally:
                    with lock:
                        active -= 1
            with patch.object(bundle_module, 'file_sha256', side_effect=observed_hash):
                validate_manifest(manifest, 'rec', root)
            self.assertEqual(calls, len(manifest['assets']))
            self.assertGreaterEqual(maximum, 2)
            self.assertLessEqual(maximum, 4)

if __name__ == "__main__":
    unittest.main()
