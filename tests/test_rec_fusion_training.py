"""Portable tests for exact prepared fit-row fusion training, with no evaluation data."""
from pathlib import Path
import contextlib
import io
import json
import tempfile
import unittest
import os
import subprocess
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from scripts.train_rec_learned_fusion import (
    PARAMETERS, FEATURE_NAMES, LABEL_SEMANTICS, GROUP_SELECTION, sha256,
    train_rec_learned_fusion, main,
)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


class PreparedFitFixture:
    def __init__(self, root):
        self.root = Path(root)
        self.paths = {key: self.root / name for key, name in {
            'training_npz': 'fit.npz', 'group_manifest': 'groups.parquet',
            'historical_plan': 'past_plan.json', 'past_split': 'split.json',
            'training_summary': 'summary.json', 'prefix_requests': 'prefix.parquet',
            'suffix_requests': 'suffix.parquet', 'preparation_script': 'archived_prepare.py'}.items()}
        self.paths['preparation_script'].write_text('# toy historical PREPARE fixture; never executed\n')
        rng = np.random.default_rng(42)
        self.X = rng.normal(size=(160, 23)).astype(np.float32)
        self.y = np.zeros(160, np.int32)
        for start in (0, 80):
            self.y[start + np.argsort(self.X[start:start+80, 0])[-4:]] = 1
        self.group = np.array([80, 80], np.int32)
        self.save_npz()
        self.groups = pd.DataFrame(dict(gid=[3,5], user_idx=[10,12],
            union_candidates=[100,110], union_positive_pairs=[4,4], sample_candidates=[80,80]))
        self.groups.to_parquet(self.paths['group_manifest'], index=False)
        pd.DataFrame(dict(gid=[1,2], user_idx=[10,11], lo=[0.,1.], hi=[.2,1.2])).to_parquet(self.paths['prefix_requests'], index=False)
        pd.DataFrame(dict(gid=[3,4,5], user_idx=[10,11,12], lo=[2.,3.,4.], hi=[2.2,3.2,4.2])).to_parquet(self.paths['suffix_requests'], index=False)
        self.historical = dict(features=list(FEATURE_NAMES), model=PARAMETERS,
            P_T_boundary=2., code_sha256=sha256(self.paths['preparation_script']),
            label_semantics='known train clicks versus unjudged implicit retrieval contrast items; zeros do not mean observed CTR negatives',
            training_sampling='fixed prepared rows; zero-positive groups excluded gradient only')
        write_json(self.paths['historical_plan'], self.historical)
        write_json(self.paths['past_split'], dict(prefix_requests=2, suffix_requests=3,
            prefix_max_hi=1.2, suffix_min_lo=2., suffix_max_hi=4.2,
            excluded_requests=1, suffix_positive_pairs=12))
        write_json(self.paths['training_summary'], dict(requests=3, zero_positive_union_requests=1,
            union_positive_pairs=8, total_positive_pairs=12, union_candidates=300, sampled_candidates=160))
        self.manifest = dict(version=1, kind='rec_learned_fusion_train_v1',
            training_scope='original_fit_prefix_suffix_only', feature_names=list(FEATURE_NAMES),
            label_semantics=LABEL_SEMANTICS, positive_group_selection=GROUP_SELECTION,
            resample_prepared_rows=False, parameters=PARAMETERS,
            preparation=dict(status='historical_prepared_candidates', entrypoint='preparation_script',
                clean_checkout_end_to_end_prepare=False))
        self.manifest_path = self.root / 'training_manifest.json'
        self.bind()

    def save_npz(self):
        np.savez(self.paths['training_npz'], X=self.X, y=self.y, group=self.group)

    def bind(self):
        self.manifest['assets'] = {key: dict(path=path.name, sha256=sha256(path))
                                  for key, path in self.paths.items()}
        write_json(self.manifest_path, self.manifest)

    def train(self):
        return train_rec_learned_fusion(self.manifest_path, self.root / 'trained')


class RecFusionTrainingTests(unittest.TestCase):
    def test_exact_prepared_rows_reproduce_toy_model_without_feedback_or_resampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = PreparedFitFixture(tmp)
            before = {path: sha256(path) for path in fixture.paths.values()}
            lineage_path = fixture.train()
            lineage = json.loads(lineage_path.read_text())
            reference = lgb.LGBMRanker(**PARAMETERS).fit(fixture.X, fixture.y,
                group=fixture.group, feature_name=list(FEATURE_NAMES)).booster_
            actual = lgb.Booster(model_file=str(lineage_path.parent / 'model.txt'))
            self.assertEqual(actual.dump_model()['tree_info'], reference.dump_model()['tree_info'])
            np.testing.assert_array_equal(actual.predict(fixture.X, num_threads=3),
                                          reference.predict(fixture.X, num_threads=3))
            self.assertEqual(lineage['data']['selected_positive_groups'], 2)
            self.assertEqual(lineage['data']['excluded_zero_positive_groups'], 1)
            self.assertEqual(lineage['data']['positive_rows'], 8)
            self.assertEqual(lineage['actual_trees'], actual.num_trees())
            self.assertFalse(lineage['quality_evaluation_performed'])
            self.assertFalse(lineage['official_test_read'])
            self.assertFalse(lineage['validation_feedback_read'])
            self.assertFalse(lineage['resampled_rows'])
            self.assertTrue(lineage['original_row_order_preserved'])
            self.assertFalse(lineage['preparation']['clean_checkout_end_to_end_prepare'])
            self.assertEqual(lineage['model_sha256'], sha256(lineage_path.parent / 'model.txt'))
            self.assertTrue(all(sha256(path) == digest for path, digest in before.items()))

    def test_wrong_source_sha_rejects_before_any_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = PreparedFitFixture(tmp)
            fixture.paths['training_npz'].write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                fixture.train()
            self.assertFalse((Path(tmp) / 'trained').exists())

    def test_wrong_schema_label_meaning_or_new_parameters_are_rejected(self):
        for mutation in ('order', 'CTR', 'selection', 'parameter', 'validation_asset', 'prepare_claim'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                fixture = PreparedFitFixture(tmp)
                if mutation == 'order':
                    fixture.manifest['feature_names'] = list(reversed(FEATURE_NAMES))
                elif mutation == 'CTR':
                    fixture.manifest['label_semantics'] = 'observed_CTR_negatives'
                elif mutation == 'selection':
                    fixture.manifest['positive_group_selection'] = 'all_groups'
                elif mutation == 'parameter':
                    fixture.manifest['parameters'] = {**PARAMETERS, 'n_estimators': 201}
                elif mutation == 'validation_asset':
                    fixture.manifest['assets']['validation'] = fixture.manifest['assets']['training_npz']
                else:
                    fixture.manifest['preparation']['clean_checkout_end_to_end_prepare'] = True
                write_json(fixture.manifest_path, fixture.manifest)
                with self.assertRaises(ValueError):
                    fixture.train()
                self.assertFalse((Path(tmp) / 'trained').exists())

    def test_npz_finiteness_binary_labels_and_group_sizes_are_validated(self):
        for mutation in ('nan_feature', 'nan_label', 'label2', 'group_size', 'group0', 'shape'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                fixture = PreparedFitFixture(tmp)
                if mutation == 'nan_feature':
                    fixture.X[0, 0] = np.nan
                elif mutation == 'nan_label':
                    fixture.y = fixture.y.astype(float)
                    fixture.y[0] = np.nan
                elif mutation == 'label2':
                    fixture.y[0] = 2
                elif mutation == 'group_size':
                    fixture.group[0] = 79
                elif mutation == 'group0':
                    fixture.group[0] = 0
                else:
                    fixture.X = fixture.X[:, :-1]
                fixture.save_npz();fixture.bind()
                with self.assertRaises(ValueError):
                    fixture.train()
                self.assertFalse((Path(tmp) / 'trained').exists())

    def test_group_positive_counts_request_identity_and_temporal_split_are_bound(self):
        for mutation in ('positive_count', 'user', 'prefix_overlap', 'boundary', 'summary'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                fixture = PreparedFitFixture(tmp)
                if mutation == 'positive_count':
                    fixture.groups.loc[0, 'union_positive_pairs'] = 3
                    fixture.groups.to_parquet(fixture.paths['group_manifest'], index=False)
                elif mutation == 'user':
                    fixture.groups.loc[0, 'user_idx'] = 99
                    fixture.groups.to_parquet(fixture.paths['group_manifest'], index=False)
                elif mutation == 'prefix_overlap':
                    prefix = pd.read_parquet(fixture.paths['prefix_requests'])
                    prefix.loc[0, 'gid'] = 3
                    prefix.to_parquet(fixture.paths['prefix_requests'], index=False)
                elif mutation == 'boundary':
                    fixture.historical['P_T_boundary'] = 1.
                    write_json(fixture.paths['historical_plan'], fixture.historical)
                else:
                    summary = json.loads(fixture.paths['training_summary'].read_text())
                    summary['zero_positive_union_requests'] = 0
                    write_json(fixture.paths['training_summary'], summary)
                fixture.bind()
                with self.assertRaises(ValueError):
                    fixture.train()
                self.assertFalse((Path(tmp) / 'trained').exists())

    def test_existing_destination_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = PreparedFitFixture(tmp)
            dest = Path(tmp) / 'trained'
            dest.mkdir();(dest / 'sentinel').write_bytes(b'keep')
            with self.assertRaises(FileExistsError):
                fixture.train()
            self.assertEqual((dest / 'sentinel').read_bytes(), b'keep')

    def test_direct_script_help_works_without_pythonpath_from_another_directory(self):
        script = Path(__file__).resolve().parents[1] / 'scripts/train_rec_learned_fusion.py'
        environment = dict(os.environ)
        environment.pop('PYTHONPATH', None)
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable, str(script), '--help'], cwd=tmp,
                env=environment, text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--training-manifest', result.stdout)
        self.assertIn('--output-dir', result.stdout)

    def test_cli_exposes_only_manifest_and_output_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = PreparedFitFixture(tmp)
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                main(['--training-manifest', str(fixture.manifest_path), '--output-dir', str(Path(tmp) / 'trained')])
            result = json.loads(stream.getvalue())
            self.assertFalse(result['evaluated'])
            self.assertFalse(result['preparation_included'])
            self.assertFalse(result['activated'])


if __name__ == '__main__':
    unittest.main()
