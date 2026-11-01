"""CPU contract tests for fit-only content adaptation; no real dataset/model needed."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from training.rec_content_adapter import (ContentAdapter, TextStore, batch_masks,
    checked_inputs, configure, contrastive_loss, make_training_rows, sha)


class ContentTrainingTests(unittest.TestCase):
    def setUp(self):
        configure()

    def test_duplicate_pool_note0_and_false_negative_mask(self):
        entries = pd.DataFrame([
            dict(positives=[0, 2], hardneg=[3], blocked=[1]),
            dict(positives=[1], hardneg=[3], blocked=[0, 2]),
        ])
        ids, positive, allowed = batch_masks(entries, np.array([0, 1]))
        np.testing.assert_array_equal(ids, [0, 1, 2, 3])
        np.testing.assert_array_equal(positive, [[1, 0, 1, 0], [0, 1, 0, 0]])
        np.testing.assert_array_equal(allowed, [[1, 0, 1, 1], [0, 1, 0, 1]])
        self.assertTrue(allowed[positive].all())

    def test_request_equal_uniform_positive_loss_and_blocked_gradient(self):
        query = torch.eye(2, requires_grad=True)
        target = torch.tensor([[1., 0.], [0., 1.], [.3, .4]], requires_grad=True)
        positive = np.array([[True, False, True], [False, True, False]])
        allowed = np.array([[True, False, True], [True, True, True]])
        actual = contrastive_loss(query, target, positive, allowed)
        logits = query @ target.T / .07
        expected = (-torch.log_softmax(logits[0, [0, 2]], dim=0).mean()
                    -torch.log_softmax(logits[1], dim=0)[1])/2
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.backward()
        self.assertTrue(torch.isfinite(query.grad).all())
        # Removing the blocked item altogether leaves this request's loss unchanged.
        single = contrastive_loss(query[:1], target, positive[:1], allowed[:1])
        removed = contrastive_loss(query[:1], target[[0, 2]], np.array([[True, True]]), np.ones((1, 2), bool))
        torch.testing.assert_close(single, removed, rtol=0, atol=0)

    def test_zero_initialization_is_noop_in_eval_but_has_training_gradient(self):
        adapter = ContentAdapter()
        x = torch.nn.functional.normalize(torch.randn(4, 128), dim=1)
        adapter.eval()
        self.assertTrue(torch.equal(adapter(x), x))
        self.assertTrue(torch.equal(adapter(torch.zeros_like(x)), torch.zeros_like(x)))
        adapter.train()
        adapter(x)[:, 0].sum().backward()
        self.assertGreater(float(adapter.net[-1].weight.grad.norm()), 0.)
        with torch.no_grad():
            adapter.net[-1].bias.fill_(1.)
        self.assertTrue(torch.equal(adapter(torch.zeros_like(x)), torch.zeros_like(x)))

    def test_real_exposure_hardneg_own_positive_override_and_user_mask(self):
        requests = pd.DataFrame([dict(gid=4, user_idx=8, positives=[0, 2], history=[0, 1]),
                                 dict(gid=5, user_idx=8, positives=[1], history=[])])
        labels = pd.DataFrame([(4, 8, 0, 1), (4, 8, 2, 1), (4, 8, 1, 0),
                               (4, 8, 3, 0), (4, 8, 4, 0), (5, 8, 1, 1)],
                              columns=["gid", "user_idx", "note_idx", "positive"])
        vectors = np.zeros((6, 128), np.float32)
        vectors[:, 0] = [1, 1, 1, .5, .5, 100]
        queries = np.zeros((2, 128), np.float32)
        queries[0, 0] = 1
        result = make_training_rows(requests, labels, queries, vectors)
        self.assertEqual(result.iloc[0].positives, [0, 2])
        self.assertEqual(result.iloc[0].blocked, [1])
        self.assertEqual(result.iloc[0].hardneg, [3, 4])
        self.assertNotIn(5, result.iloc[0].hardneg)  # unexposed high scorer never becomes hardnegative
        self.assertEqual(result.iloc[1].blocked, [0, 2])
        self.assertTrue(np.linalg.norm(queries[1]) == 0)  # row retained in preparation

    def test_all_positive_mask_is_finite_zero_loss(self):
        loss = contrastive_loss(torch.tensor([[1., 0.]]), torch.tensor([[1., 0.]]),
                                np.ones((1, 1), bool), np.ones((1, 1), bool))
        self.assertEqual(float(loss), 0.)
        with self.assertRaises(ValueError):
            contrastive_loss(torch.ones(1, 2), torch.ones(1, 2),
                             np.ones((1, 1), bool), np.zeros((1, 1), bool))

    def test_text_map_keeps_note0_repeat_and_missing_zero(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            np.ones((1, 768), np.float16).tofile(temp/"text.bin")
            (temp/"map.json").write_text(json.dumps({"0": 0}))
            store = TextStore(temp/"text.bin", temp/"map.json")
            x = store.get_batch_vecs([0, 0, 999])
            self.assertTrue(torch.equal(x[0], x[1]))
            self.assertEqual(float(x[0].sum()), 768.)
            self.assertEqual(float(x[2].sum()), 0.)

    def test_input_config_rejects_unbound_or_wrong_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            (temp/"x").write_bytes(b"frozen")
            from training.rec_content_adapter import INPUTS
            inputs = {k: dict(path="x", sha256=sha(temp/"x")) for k in INPUTS}
            config = dict(version=1, training_cutoff_exclusive=12, inputs=inputs)
            (temp/"config.json").write_text(json.dumps(config))
            _, loaded = checked_inputs(temp/"config.json")
            self.assertEqual(len(loaded), 10)
            (temp/"x").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                checked_inputs(temp/"config.json")


    def test_toy_prepare_train_export_has_true_native_lineage(self):
        from contextlib import redirect_stdout
        import io
        from training.rec_content_adapter import prepare, train, export, INPUTS
        from training.content_adapter_release import _heads, _training_steps
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            raw = np.zeros((6, 768), np.float16)
            raw[np.arange(6), np.arange(6)] = 1
            raw.tofile(temp/"text.bin")
            (temp/"map.json").write_text(json.dumps({str(i): i for i in range(6)}))
            np.save(temp/"ids.npy", np.arange(6, dtype=np.int64))
            np.save(temp/"mean.npy", np.zeros(768, np.float32))
            np.save(temp/"components.npy", np.eye(128, 768, dtype=np.float32))
            raw[:, :128].astype(np.float32).tofile(temp/"items.bin")
            pd.DataFrame([dict(gid=1, user_idx=8, positives=[2, 3], history=[0, 0]),
                          dict(gid=2, user_idx=8, positives=[4], history=[1]),
                          dict(gid=3, user_idx=9, positives=[5], history=[])]).to_parquet(temp/"requests.parquet", index=False)
            labels = pd.DataFrame([(1, 8, 2, 1), (1, 8, 3, 1), (1, 8, 1, 0),
                                   (1, 8, 4, 0), (2, 8, 4, 1), (2, 8, 2, 0),
                                   (2, 8, 5, 0), (3, 9, 5, 1), (3, 9, 0, 0)],
                                  columns=["gid", "user_idx", "note_idx", "positive"])
            labels["lo"] = 10.
            labels["hi"] = 11.
            labels.to_parquet(temp/"labels.parquet", index=False)
            pd.DataFrame(dict(gid=[1, 2, 3, 4], lo=[10., 10., 10., 12.],
                              hi=[11., 11., 11., 13.], partition=["fit", "fit", "fit", "validation"])).to_parquet(temp/"partitions.parquet", index=False)
            (temp/"split.json").write_text(json.dumps(dict(fit_end_exclusive=12)))
            files = dict(requests="requests.parquet", labels="labels.parquet", partitions="partitions.parquet",
                         split_manifest="split.json", text_vectors="text.bin", text_map="map.json",
                         pca_mean="mean.npy", pca_components="components.npy", pca_vectors="items.bin", item_ids="ids.npy")
            self.assertEqual(set(files), INPUTS)
            config = dict(version=1, training_cutoff_exclusive=12,
                          inputs={k: dict(path=v, sha256=sha(temp/v)) for k, v in files.items()})
            (temp/"config.json").write_text(json.dumps(config))
            prepare(temp/"config.json", temp/"run")
            data = json.loads((temp/"run/data_contract.json").read_text())
            self.assertEqual(data["fit_requests"], 3)
            self.assertEqual(data["fit_empty_excluded_SGD_only"], 1)
            with redirect_stdout(io.StringIO()):
                train(temp/"run", "cpu")
            export(temp/"run", 3, temp/"heads")
            heads = torch.load(temp/"heads/towers.pt", map_location="cpu", weights_only=False)
            dim, state = _heads(heads)
            self.assertEqual(dim, 128)
            self.assertEqual(len(state), 8)
            summary = json.loads((temp/"heads/training_summary.json").read_text())
            plan = json.loads((temp/"heads/plan.json").read_text())
            identity, epoch, steps = _training_steps(heads, summary, plan,
                dict(training_plan=sha(temp/"heads/plan.json"), data_contract=sha(temp/"heads/data_contract.json"),
                     towers_checkpoint=sha(temp/"heads/towers.pt")))
            self.assertEqual((identity, epoch, steps), ("dual_tower", 3, [1, 1, 1]))
            self.assertFalse(summary["official_test_read"])
            self.assertEqual(summary["optimizer_updates"], 3)
            with self.assertRaises(FileExistsError):
                export(temp/"run", 3, temp/"heads")


if __name__ == "__main__":
    unittest.main()
