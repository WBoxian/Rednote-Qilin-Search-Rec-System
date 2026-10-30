import importlib.util
from pathlib import Path
import unittest
import tempfile,json
import numpy as np
import pandas as pd

spec=importlib.util.spec_from_file_location("rec_stage_gate",
    Path(__file__).resolve().parents[1]/"scripts/calibrate_rec_stage_blends.py")
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def requests_frame(values):
    return pd.DataFrame({"gid":np.arange(len(values)),"user_idx":np.arange(len(values)),
        "recall10":values})


class StageBlendGateTests(unittest.TestCase):
    def test_bypass_exactly_preserves_original_recall_order(self):
        notes=np.array([40,10,30,20]);rank=np.array([4,1,3,2])
        coarse,final=module.blend_stages(notes,rank,[1.,4.,2.,3.],[9.,-4.,7.,20.],0.,0.,
            input_cap=4,coarse_cap=3)
        self.assertEqual(notes[coarse].tolist(),[10,20,30])
        self.assertEqual(notes[final].tolist(),[10,20,30])

    def test_coarse_selection_does_not_replace_raw_tree_feature(self):
        tree=np.array([.123,.005,.9]);original=tree.copy()
        coarse,_=module.blend_stages(np.array([1,2,3]),np.array([1,2,3]),tree,[0.,0.,0.],
            1.,0.,input_cap=3,coarse_cap=2)
        np.testing.assert_array_equal(tree,original)
        self.assertEqual(coarse.tolist(),[2,0])

    def test_invalid_input_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            module.blend_stages([1,2],[1,2],[0.,1.],[0.,1.],0,0,input_cap=1)
        with self.assertRaises(ValueError):
            module.blend_stages([1,2],[1,1],[0.,1.],[0.,1.],0,0)

    def test_hit_recall_guard_rejects_r10_only_improvement(self):
        baseline={"Recall@10":.1,"Recall@20":.2,"HitRate@10":.2}
        candidate={"Recall@10":.3,"Recall@20":.1,"HitRate@10":.2}
        summary={(0.,0.):baseline,(1.,1.):candidate}
        frames={(0.,0.):requests_frame([.1]*10),(1.,1.):requests_frame([.3]*10)}
        selected=module.select_with_gate(summary,frames)
        self.assertEqual(selected["selected"]["alpha"],0.)
        self.assertEqual(selected["selected"]["beta"],0.)

    def test_ci_gate_keeps_bypass_when_positive_point_gain_is_uncertain(self):
        summary={(0.,0.):{"Recall@10":0.,"Recall@20":.1,"HitRate@10":0.},
            (.5,0.):{"Recall@10":.005,"Recall@20":.1,"HitRate@10":.05}}
        frames={(0.,0.):requests_frame([0.]*20),(.5,0.):requests_frame([.1]+[0.]*19)}
        selected=module.select_with_gate(summary,frames)
        self.assertFalse(selected["gate_passed"])
        self.assertEqual(selected["candidate_minus_bypass"]["user_cluster_CI95"][0],0.)

    def test_clear_paired_gain_passes_gate(self):
        summary={(0.,0.):{"Recall@10":.1,"Recall@20":.2,"HitRate@10":.1},
            (.5,1.):{"Recall@10":.2,"Recall@20":.2,"HitRate@10":.2}}
        frames={(0.,0.):requests_frame([.1]*20),(.5,1.):requests_frame([.2]*20)}
        selected=module.select_with_gate(summary,frames)
        self.assertTrue(selected["gate_passed"])
        self.assertEqual(selected["selected"]["alpha"],.5)
        self.assertEqual(selected["selected"]["beta"],1.)

    def test_missing_request_stays_in_known_positive_retention_denominator(self):
        requests=pd.DataFrame({"gid":[1,2],"user_idx":[8,9],"positives":[[99],[100]]})
        scores=pd.DataFrame({"gid":[1,1],"note_idx":[7,99],"rank":[1,2],
            "coarse_score":[20.,-1.],"dien_raw_score":[1.,-1.]})
        result,_=module.grid_metrics(requests,scores,[(0.,0.)])
        self.assertEqual(result[(0.,0.)]["all_requests"],2)
        self.assertEqual(result[(0.,0.)]["Recall@10"],.5)
        self.assertEqual(result[(0.,0.)]["hits10"],1)



    def test_selection_cannot_change_after_freeze(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"selection.json"
            original={"alpha":0.,"beta":0.}
            module.freeze_selection(path,original)
            self.assertEqual(module.freeze_selection(path,original),original)
            with self.assertRaises(ValueError):
                module.freeze_selection(path,{"alpha":1.,"beta":0.})
            self.assertEqual(json.loads(path.read_text()),original)

    def test_cached_scores_require_unchanged_raw_context(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder)
            raw=folder/"raw";raw.write_bytes(b"fixed request")
            scores=folder/"scores";scores.write_bytes(b"fixed scores")
            class Static:
                note_ids=np.array([1],np.int64)
                values=np.array([[.2]],np.float32)
            h=module.hashlib.sha256()
            h.update(Static.note_ids.tobytes());h.update(Static.values.tobytes())
            metadata={"inputs":{"model":"fixed"},"scores_sha256":module.digest(scores),
                "raw_context_sha256":module.digest(raw),"embedding_sha256":{},
                "encoded_static_sha256":h.hexdigest()}
            module.verify_cached_inputs(scores,metadata,{"model":"fixed"},raw,Static())
            raw.write_bytes(b"changed request")
            with self.assertRaises(ValueError):
                module.verify_cached_inputs(scores,metadata,{"model":"fixed"},raw,Static())


if __name__=="__main__":unittest.main()
