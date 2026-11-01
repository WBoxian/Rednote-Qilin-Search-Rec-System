"""Numerical runtime revisions must preserve every learned weight and recipe."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from scripts import rebind_serving_runtime as runtime
from scripts.package_serving_bundle import package_bundle
from tests.test_package_serving_bundle import fixture, adapted_fixture


class RuntimeContractTests(unittest.TestCase):
    def prepare(self, root, learned=False):
        args, _, _ = adapted_fixture(root, learned=True) if learned else fixture(root, 'search')
        original, bundle = package_bundle(args, root)
        target = root / 'revision/rank'
        shutil.copytree(root / bundle['rank_dir'], target)
        metadata = json.loads((target / 'manifest.json').read_text())
        metadata['dien']['inference_contract'] = 'request_fp32_v1'
        (target / 'manifest.json').write_text(json.dumps(metadata))
        for name in ['recall/dssm_trainer.py', 'recall/cf_retrieval.py', 'recall/fusion.py', 'recall/lexical.py',
                     'recall/context_fusion.py', 'recall/residual_dssm.py', 'recall/weighted_context_fusion.py', 'recall/learned_context_fusion.py', 'training/rank_stack.py', 'training/history.py',
                     'training/dien_ranker.py', 'training/scoring.py', 'training/recall_prior.py']:
            path = root / 'src' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# immutable runtime fixture\n')
        return original, bundle, target

    def test_contract_only_rebind_binds_evidence_without_activation(self):
        with tempfile.TemporaryDirectory(prefix='qilin-rebind-test-') as directory:
            root = Path(directory)
            original, before, rank = self.prepare(root)
            evidence = root / 'revision/evidence.json'
            evidence.write_text('{"scope":"canonical inference replay"}')
            output = root / 'revision/bundle.json'
            with patch.object(runtime, 'ROOT', root), patch.object(runtime.subprocess, 'check_output', return_value='fixture-revision\n'):
                runtime.rebind(original, output, rank_dir=rank, numerical_evidence=evidence)
            after = json.loads(output.read_text())
            self.assertNotEqual(before['bundle_id'], after['bundle_id'])
            self.assertEqual(after['recall_policy'], before['recall_policy'])
            self.assertFalse(after['runtime_rebind']['changed_model_or_original_data_assets'])
            self.assertEqual(after['assets']['revision/evidence.json'], runtime.file_sha256(evidence))
            self.assertFalse(list(root.rglob('active_bundle.json')))

    def test_weight_or_recipe_changes_cannot_hide_in_runtime_revision(self):
        for change in ('weight', 'recipe'):
            with self.subTest(change=change), tempfile.TemporaryDirectory(prefix='qilin-rebind-test-') as directory:
                root = Path(directory)
                original, _, rank = self.prepare(root)
                if change == 'weight':
                    (rank / 'dien.pt').write_bytes(b'unapproved replacement')
                else:
                    path = rank / 'manifest.json'
                    metadata = json.loads(path.read_text())
                    metadata['dien']['chosen']['weight'] = .9
                    path.write_text(json.dumps(metadata))
                output = root / 'revision/bundle.json'
                with patch.object(runtime, 'ROOT', root), self.assertRaises(ValueError):
                    runtime.rebind(original, output, rank_dir=rank)
                self.assertFalse(output.exists())

    def test_learned_recipe_rebind_preserves_six_assets_and_binds_new_core(self):
        with tempfile.TemporaryDirectory(prefix='qilin-rebind-test-') as directory:
            root = Path(directory)
            original, before, _ = self.prepare(root, learned=True)
            output = root / 'revision/bundle.json'
            with patch.object(runtime, 'ROOT', root), patch.object(runtime.subprocess, 'check_output', return_value='fixture-revision\n'):
                runtime.rebind(original, output)
            after = json.loads(output.read_text())
            self.assertEqual(before['recall_policy'], after['recall_policy'])
            for key in ('model','support','pca_vectors','pca_ids','pca_mean','pca_components'):
                name = before['recall_policy']['learned_fusion'][key]
                self.assertEqual(before['assets'][name], after['assets'][name])
            name = 'src/recall/learned_context_fusion.py'
            self.assertEqual(after['assets'][name], runtime.file_sha256(root / name))
            self.assertFalse(after['runtime_rebind']['changed_model_or_original_data_assets'])
            self.assertFalse(list(root.rglob('active_bundle.json')))

    def test_learned_recipe_rebind_cannot_hide_changed_fusion_weights(self):
        with tempfile.TemporaryDirectory(prefix='qilin-rebind-test-') as directory:
            root = Path(directory)
            original, before, _ = self.prepare(root, learned=True)
            model = root / before['recall_policy']['learned_fusion']['model']
            model.write_bytes(b'changed learned fusion model')
            output = root / 'revision/bundle.json'
            with patch.object(runtime, 'ROOT', root), self.assertRaises(ValueError):
                runtime.rebind(original, output)
            self.assertFalse(output.exists())
