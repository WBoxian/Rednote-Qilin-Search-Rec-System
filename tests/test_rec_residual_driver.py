import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np,pandas as pd,torch
from scripts.optimize_rec_residual import exact_route

class ResidualDriverTests(unittest.TestCase):
 def test_actual_streaming_exact_route_writer_and_zero_query_with_cpu_backend(self):
  rng=np.random.default_rng(42);items=rng.normal(size=(1001,128)).astype(np.float32);query=rng.normal(size=(2,128)).astype(np.float32);query[1]=0;ids=np.array([7,8]);itemids=np.arange(1001,dtype=np.int64);original_empty=torch.empty;original_to=torch.Tensor.to
  def empty(*args,**kwargs):
   if kwargs.get('device')=='cuda':kwargs['device']='cpu'
   return original_empty(*args,**kwargs)
  def to(tensor,*args,**kwargs):
   if args and args[0]=='cuda':args=('cpu',*args[1:])
   if kwargs.get('device')=='cuda':kwargs['device']='cpu'
   return original_to(tensor,*args,**kwargs)
  with tempfile.TemporaryDirectory() as tmp,patch('torch.empty',side_effect=empty),patch.object(torch.Tensor,'to',new=to):
   path=Path(tmp)/'ann.parquet';exact_route(ids,query,items,itemids,path);result=pd.read_parquet(path)
  self.assertEqual(len(result),1000);self.assertEqual(result.gid.unique().tolist(),[7]);self.assertEqual(result['rank'].tolist(),list(range(1,1001)));expected=(torch.from_numpy(query[:1])@torch.from_numpy(items).T).numpy()[0];order=np.lexsort((itemids,-expected))[:1000];np.testing.assert_array_equal(result.note_idx,itemids[order]);np.testing.assert_allclose(result.score,expected[order],atol=0,rtol=0)

# Export tests use real toy optimizer steps; no project dataset or large weights.
import contextlib
import hashlib
import io
import json
from recall.residual_dssm import (ResidualDSSMModel, PCA_KEYS, model_from_checkpoint,
                                 save_residual_checkpoints)
from training.content_adapter_release import export_content_adapter_release, main as export_main


def _release_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _release_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


class _TrainedReleaseFixture:
    def __init__(self, root, dim=128):
        self.root = Path(root)
        self.dim = dim
        self.vocab = dict(user_idx=3, note_idx=5, gender=3, platform=3,
                          age=3, location=3, tax1=3, tax2=3, tax3=3)
        components = torch.zeros(128, 768)
        components[:, :128] = torch.eye(128)
        torch.manual_seed(42)
        base = ResidualDSSMModel('rec', self.vocab, 6, 938,
            pca_mean=torch.arange(768) * 1e-5, pca_components=components,
            residual_weight=.25).eval()
        self.learned = {key: tensor.clone() for key, tensor in base.state_dict().items()
                        if key not in PCA_KEYS}
        self.original = self.root / 'original.pt'
        torch.save(self.learned, self.original)
        self.meta = dict(scene='rec', cat_vocabs=self.vocab, item_dense_dim=6,
            user_dense_dim=938, numeric_preprocessing=None, known_user_ids=[0, 1],
            unknown_user_id=2, history_contract='provided_recent20_newest_first',
            sequence_padding_contract='packed_masked_v1', masked_user_dense=True,
            gradient_updates=0, source_model_sha256=_release_sha(self.original))
        save_residual_checkpoints(base, self.meta, self.root / 'base')
        self.base = self.root / 'base/dssm_rec_residual.pt'
        pca = self.root / 'pca'
        pca.mkdir()
        np.save(pca / 'mean.npy', base.residual_pca_mean.numpy())
        np.save(pca / 'components.npy', base.residual_pca_components.numpy())
        self.plan_path = self.root / 'plan.json'
        self.contract_path = self.root / 'data_contract.json'
        self.summary_path = self.root / 'training_summary.json'
        self.towers_path = self.root / 'towers.pt'
        self.plan = dict(training=dict(beta=.5, epochs=2, updates_per_epoch=1),
            source_sha256={str(path): _release_sha(path) for path in pca.iterdir()})
        _release_json(self.plan_path, self.plan)
        _release_json(self.contract_path, dict(official_test_read=False, fit_requests=8))
        self.identity_key = 'variant' if dim == 128 else 'arm'
        self.identity = 'dual_tower' if dim == 128 else 'main_only'
        self.plan_key = 'training_plan_sha256' if dim == 128 else 'plan_sha256'
        self.step_key = 'batches' if dim == 128 else 'updates'
        query = torch.nn.Sequential(torch.nn.Linear(dim, 256), torch.nn.ReLU(),
                                     torch.nn.Linear(256, 128))
        item = torch.nn.Sequential(torch.nn.Linear(dim, 256), torch.nn.ReLU(),
                                    torch.nn.Linear(256, 128))
        for head in (query, item):
            torch.nn.init.zeros_(head[-1].weight)
            torch.nn.init.zeros_(head[-1].bias)
        optimizer = torch.optim.AdamW(list(query.parameters()) + list(item.parameters()), lr=.001)
        q = torch.nn.functional.normalize(torch.randn(8, dim), dim=1)
        i = torch.nn.functional.normalize(torch.randn(8, dim), dim=1)
        trials = []
        for epoch in (1, 2):
            optimizer.zero_grad()
            qbase = q[:, :128]
            ibase = i[:, :128]
            qout = torch.nn.functional.normalize(qbase + .5 * query(q), dim=1)
            iout = torch.nn.functional.normalize(ibase + .5 * item(i), dim=1)
            loss = torch.nn.functional.cross_entropy(qout @ iout.T / .07, torch.arange(8))
            loss.backward()
            assert float(query[-1].weight.grad.norm()) > 0
            assert float(item[-1].weight.grad.norm()) > 0
            optimizer.step()
            self.heads = dict(query={'net.' + k: v.detach().clone() for k, v in query.state_dict().items()},
                item={'net.' + k: v.detach().clone() for k, v in item.state_dict().items()},
                epoch=epoch, **{self.identity_key: self.identity,
                    self.plan_key: _release_sha(self.plan_path),
                    'data_contract_sha256': _release_sha(self.contract_path)})
            weights = self.root / f'epoch_{epoch}.pt'
            torch.save(self.heads, weights)
            trials.append({self.identity_key: self.identity, 'epoch': epoch,
                           self.step_key: 1, 'weights_sha256': _release_sha(weights)})
        self.summary = dict(official_test_read=False, trials=trials)
        self.refresh()

    def refresh(self):
        _release_json(self.plan_path, self.plan)
        self.heads[self.plan_key] = _release_sha(self.plan_path)
        self.heads['data_contract_sha256'] = _release_sha(self.contract_path)
        torch.save(self.heads, self.towers_path)
        self.summary['trials'][-1]['weights_sha256'] = _release_sha(self.towers_path)
        _release_json(self.summary_path, self.summary)

    def arguments(self, name='release'):
        return dict(base_checkpoint=self.base, original_learned_checkpoint=self.original,
            towers_checkpoint=self.towers_path, training_plan=self.plan_path,
            data_contract=self.contract_path, training_summary=self.summary_path,
            destination=self.root / name)


class ContentAdapterReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def assert_rejected(self, fixture, error=ValueError):
        kwargs = fixture.arguments()
        with self.assertRaises(error):
            export_content_adapter_release(**kwargs)
        self.assertFalse(kwargs['destination'].exists())
        self.assertEqual(list(fixture.root.glob('.release-*')), [])

    def test_actual_128_and_raw768_heads_export_exactly_without_new_updates(self):
        for dim in (128, 768):
            with self.subTest(dim=dim), tempfile.TemporaryDirectory() as tmp:
                fixture = _TrainedReleaseFixture(tmp, dim)
                before = {path: _release_sha(path) for path in Path(tmp).rglob('*') if path.is_file()}
                manifest_path = export_content_adapter_release(**fixture.arguments())
                manifest = json.loads(manifest_path.read_text())
                self.assertEqual(manifest['gradient_updates'], 2)  # Not four for two towers.
                self.assertEqual(manifest['packaging_gradient_updates'], 0)
                self.assertEqual(manifest['status'], 'packaged_not_deployed')
                self.assertFalse(manifest['training_performed_by_exporter'])
                self.assertFalse(manifest['deployed'])
                full = model_from_checkpoint(manifest_path.parent / 'dssm_rec_residual.pt')
                for key, tensor in fixture.learned.items():
                    self.assertTrue(torch.equal(tensor, full.state_dict()[key]), key)
                for role, prefix in [('query', 'content_query_adapter.'), ('item', 'content_item_adapter.')]:
                    self.assertGreater(float(fixture.heads[role]['net.2.weight'].norm()), 0.)
                    for key, tensor in fixture.heads[role].items():
                        self.assertTrue(torch.equal(tensor, full.state_dict()[prefix + key.removeprefix('net.')]))
                for role in ('user', 'item'):
                    path = manifest_path.parent / f'dssm_rec_residual_{role}_tower.pt'
                    payload = torch.load(path, weights_only=False)
                    partial = model_from_checkpoint(path, allow_partial=True)
                    self.assertTrue(partial.checkpoint_is_partial)
                    self.assertEqual(payload['meta']['exported_full_checkpoint_sha256'],
                                     manifest['exported_full_checkpoint_sha256'])
                    for key, tensor in payload['state_dict'].items():
                        self.assertTrue(torch.equal(tensor, full.state_dict()[key]))
                        self.assertTrue(torch.equal(tensor, partial.state_dict()[key]))
                meta = full.checkpoint_meta
                for key in ('known_user_ids', 'unknown_user_id', 'history_contract',
                            'sequence_padding_contract', 'masked_user_dense', 'numeric_preprocessing'):
                    self.assertEqual(meta[key], fixture.meta[key])
                self.assertEqual(meta['gradient_update_scope'], 'content_adapter_stage_only')
                self.assertEqual(meta['content_input_dim'], dim)
                for name, info in manifest['exported_files'].items():
                    self.assertEqual(_release_sha(manifest_path.parent / name), info['sha256'])
                for info in manifest['sources'].values():
                    self.assertEqual(_release_sha(info['path']), info['sha256'])
                for path, digest in before.items():
                    self.assertEqual(_release_sha(path), digest)

    def test_cli_uses_same_validated_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _TrainedReleaseFixture(tmp)
            args = [part for name, value in fixture.arguments().items()
                    for part in ('--' + name.replace('_', '-'), str(value))]
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                export_main(args)
            result = json.loads(stream.getvalue())
            self.assertEqual(result['gradient_updates'], 2)
            self.assertFalse(result['activated'])
            self.assertTrue(Path(result['manifest']).is_file())

    def test_missing_extra_bad_dtype_shape_or_nonfinite_head_is_rejected(self):
        for mutation in ('missing_item', 'extra_key', 'dtype', 'shape', 'nonfinite'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                fixture = _TrainedReleaseFixture(tmp)
                if mutation == 'missing_item':
                    del fixture.heads['item']
                elif mutation == 'extra_key':
                    fixture.heads['query']['net.4.bias'] = torch.zeros(128)
                elif mutation == 'dtype':
                    fixture.heads['query']['net.2.bias'] = torch.zeros(128, dtype=torch.float64)
                elif mutation == 'shape':
                    fixture.heads['query']['net.2.bias'] = torch.zeros(129)
                else:
                    fixture.heads['query']['net.2.bias'][0] = float('nan')
                fixture.refresh()
                self.assert_rejected(fixture)

    def test_wrong_plan_data_or_summary_weights_hash_is_rejected(self):
        for mutation in ('plan', 'data', 'weights'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                fixture = _TrainedReleaseFixture(tmp)
                if mutation == 'plan':
                    fixture.plan_path.write_text(fixture.plan_path.read_text() + ' ')
                elif mutation == 'data':
                    fixture.contract_path.write_text(fixture.contract_path.read_text() + ' ')
                else:
                    fixture.summary['trials'][-1]['weights_sha256'] = '0' * 64
                    _release_json(fixture.summary_path, fixture.summary)
                self.assert_rejected(fixture)

    def test_wrong_pca_even_with_rebound_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _TrainedReleaseFixture(tmp)
            path = Path(tmp) / 'pca/mean.npy'
            values = np.load(path)
            values[0] += 1
            np.save(path, values)
            fixture.plan['source_sha256'][str(path)] = _release_sha(path)
            fixture.refresh()
            self.assert_rejected(fixture)

    def test_partial_or_changed_learned_base_cannot_be_exported(self):
        for mutation in ('partial', 'base_tensor', 'original_sha', 'new_family'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                fixture = _TrainedReleaseFixture(tmp)
                if mutation == 'partial':
                    fixture.base = Path(tmp) / 'base/dssm_rec_residual_user_tower.pt'
                elif mutation == 'original_sha':
                    state = dict(fixture.learned)
                    name = next(k for k, v in state.items() if v.is_floating_point())
                    state[name] = state[name] + 1
                    torch.save(state, fixture.original)
                elif mutation == 'new_family':
                    path = fixture.base.with_suffix('.meta.json')
                    meta = json.loads(path.read_text())
                    meta['model_family'] = 'adapted_residual_dssm_v1'
                    _release_json(path, meta)
                else:
                    state = torch.load(fixture.base, weights_only=False)
                    name = next(k for k, v in state.items() if v.is_floating_point() and k not in PCA_KEYS)
                    state[name] = state[name] + 1
                    torch.save(state, fixture.base)
                self.assert_rejected(fixture)

    def test_training_lineage_requires_contiguous_real_positive_steps_and_no_test_access(self):
        for mutation in ('gap', 'duplicate', 'zero', 'conflicting', 'test', 'nested_test', 'beta'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                fixture = _TrainedReleaseFixture(tmp)
                if mutation == 'gap':
                    fixture.summary['trials'] = fixture.summary['trials'][1:]
                elif mutation == 'duplicate':
                    fixture.summary['trials'].insert(0, dict(fixture.summary['trials'][0]))
                elif mutation == 'zero':
                    fixture.summary['trials'][0]['batches'] = 0
                elif mutation == 'conflicting':
                    fixture.summary['trials'][0]['updates'] = 2
                elif mutation == 'test':
                    fixture.summary['official_test_read'] = True
                elif mutation == 'nested_test':
                    fixture.summary['audit'] = dict(test_read=True)
                else:
                    fixture.plan['training']['beta'] = .6
                fixture.refresh()
                self.assert_rejected(fixture)

    def test_failed_staging_is_cleaned_without_a_complete_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _TrainedReleaseFixture(tmp)
            def fail_after_write(model, meta, destination):
                destination.mkdir()
                (destination / 'incomplete.pt').write_bytes(b'incomplete')
                raise RuntimeError('simulated interrupted serialization')
            with patch('training.content_adapter_release.save_residual_checkpoints',
                       side_effect=fail_after_write):
                self.assert_rejected(fixture, RuntimeError)

    def test_existing_destination_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _TrainedReleaseFixture(tmp)
            dest = fixture.arguments()['destination']
            dest.mkdir()
            sentinel = dest / 'sentinel'
            sentinel.write_text('keep')
            with self.assertRaises(FileExistsError):
                export_content_adapter_release(**fixture.arguments())
            self.assertEqual(sentinel.read_text(), 'keep')


if __name__ == '__main__':
    unittest.main()
