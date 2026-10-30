import importlib.util
from pathlib import Path
import unittest
from copy import deepcopy
import numpy as np
import pandas as pd

spec=importlib.util.spec_from_file_location("fp32_finalizer",
    Path(__file__).resolve().parents[1]/"scripts/finalize_rank_precision.py")
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class PrecisionFinalizationTests(unittest.TestCase):
    def test_parent_contract_is_copied_without_weight_or_parent_mutation(self):
        parent={"scene":"rec","dien":{"architecture":{"hidden_dim":64},"chosen":{"weight":.5}},
            "final_test":{"old":True}}
        original=deepcopy(parent)
        child=module.make_fp32_manifest(parent,"rec")
        self.assertEqual(parent,original)
        self.assertNotIn("final_test",child)
        self.assertEqual(child["dien"]["architecture"]["inference_precision"],"fp32")
        self.assertFalse(child["dien"]["architecture"]["mask_missing_aux_negatives"])
        self.assertEqual(child["precision_finalization"]["gradient_updates"],0)
        self.assertFalse(child["precision_finalization"]["teacher_changed"])

    def test_v2_auxiliary_contract_remains_v2_under_fp32(self):
        parent={"scene":"search","dien":{"architecture":{},"auxiliary_contract":"v2: invalid auxiliary negatives masked"}}
        child=module.make_fp32_manifest(parent,"search")
        self.assertTrue(child["dien"]["architecture"]["mask_missing_aux_negatives"])
        self.assertEqual(child["dien"]["auxiliary_contract"],parent["dien"]["auxiliary_contract"])

    def test_validation_blend_tie_chooses_lower_weight(self):
        choices=[{"weight":weight,"validation":{"NDCG@10":.5}} for weight in module.BLEND_GRID]
        self.assertEqual(module.choose_blend(choices)["weight"],0.)
        with self.assertRaises(ValueError):module.choose_blend(choices[:2])

    def test_native_comparison_reports_error_without_claiming_cross_shape_identity(self):
        reference=np.arange(12,dtype=float)
        changed=reference.copy();changed[[0,11]]=changed[[11,0]]
        result=module.compare_profiles(reference,changed,np.ones(12,dtype=int),np.arange(12))
        self.assertEqual(result["max_absolute_score_difference"],11.)
        self.assertEqual(result["changed_rank_rows"],2)
        self.assertEqual(result["top10_exact_set_requests"],0)
        self.assertLess(result["top10_mean_set_jaccard"],1.)
        self.assertIn("no guarantee",result["claim_scope"])

    def test_canonical_callback_uses_1024_and_repeats_same_profile(self):
        import torch
        from train_rank_stack import score_dien
        class Model(torch.nn.Module):
            inference_precision="fp32"
            def forward(self,features):
                return features["x"].reshape(-1)*2,torch.tensor(0.)
        class Context:
            def __init__(self):self.sizes=[]
            def batch(self,frame,numeric,device):
                self.sizes.append(len(frame))
                return {"x":torch.tensor(frame.x.to_numpy(np.float32),device=device)}
        frame=pd.DataFrame({"x":np.arange(2050,dtype=np.float32)})
        context=Context();model=Model()
        first=score_dien(model,frame,context,["x"],torch.device("cpu"))
        second=score_dien(model,frame,context,["x"],torch.device("cpu"))
        self.assertEqual(context.sizes,[1024,1024,2,1024,1024,2])
        np.testing.assert_array_equal(first,second)

    def test_fp32_constructor_disables_both_tf32_paths(self):
        import torch
        from training.dien_ranker import DIENRanker
        old_matmul=torch.backends.cuda.matmul.allow_tf32
        old_cudnn=torch.backends.cudnn.allow_tf32
        try:
            model=DIENRanker("rec",["x"],embed_dim=4,hidden_dim=3,mlp_dims=[8],inference_precision="fp32")
            self.assertEqual(model.inference_precision,"fp32")
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            self.assertFalse(torch.backends.cudnn.allow_tf32)
        finally:
            torch.backends.cuda.matmul.allow_tf32=old_matmul
            torch.backends.cudnn.allow_tf32=old_cudnn


if __name__=="__main__":unittest.main()
