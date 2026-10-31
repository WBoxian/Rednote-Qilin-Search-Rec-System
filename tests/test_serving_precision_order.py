import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from training.dien_ranker import DIENRanker


class ServingPrecisionOrderTests(unittest.TestCase):
    def setUp(self):
        self.flags=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32,torch.backends.cudnn.benchmark)

    def tearDown(self):
        torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32,torch.backends.cudnn.benchmark=self.flags

    def test_actual_ranker_then_lazy_dssm_module_load_preserves_precision_flags(self):
        DIENRanker('rec', [], embed_dim=4, hidden_dim=4, mlp_dims=[4], inference_precision='fp32')
        torch.backends.cudnn.benchmark=False
        path=Path(__file__).resolve().parents[1]/'src/recall/dssm_trainer.py'
        spec=importlib.util.spec_from_file_location('_dssm_serving_load_order_test',path)
        module=importlib.util.module_from_spec(spec)
        with patch('torch.cuda.is_available',return_value=True):
            spec.loader.exec_module(module)
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        self.assertFalse(torch.backends.cudnn.allow_tf32)
        self.assertFalse(torch.backends.cudnn.benchmark)

    def test_actual_fp32_forward_rejects_global_precision_drift_before_gpu_work(self):
        model=DIENRanker('rec', [], embed_dim=4, hidden_dim=4, mlp_dims=[4], inference_precision='fp32')
        for matmul,cudnn in [(True,False),(False,True)]:
            torch.backends.cuda.matmul.allow_tf32=matmul
            torch.backends.cudnn.allow_tf32=cudnn
            with self.assertRaisesRegex(RuntimeError,'TF32 to remain disabled'):
                model({'note_text_emb':SimpleNamespace(is_cuda=True)})


if __name__=='__main__':
    unittest.main()
