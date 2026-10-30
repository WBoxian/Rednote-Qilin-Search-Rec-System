import importlib.util,os,unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np,pandas as pd,torch
spec=importlib.util.spec_from_file_location("funnel",Path(__file__).resolve().parents[1]/"scripts/replay_stage_funnel.py")
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
class BatchTests(unittest.TestCase):
    def test_batch_preserves_each_history_and_alias_and_budget(self):
        class Store:
            def get_tensor(self,ids):return torch.tensor(ids,dtype=torch.float32).reshape(-1,1)
        class Model:
            def __call__(self,f):
                return f["preranking_score"]+f["seq_embs"][:,0,0].reshape(-1,1)*.01+f["note_text_emb"]*.001,None
        model=SimpleNamespace(device="cpu",scene="rec",num_feats=["preranking_score"],
                              model=Model(),stores={"t":Store(),"i":Store()},
                              _get_history_seq_base_tensor=lambda h:torch.tensor([[float(h[0])]]))
        def frame(gid):
            return pd.DataFrame({"request_idx":[gid,gid],"note_idx":[gid*10,gid*10+1],
                                 "preranking_rank":[2,1],"preranking_score":[99.,99.],
                                 "dien_preranking_score":[.1,.2]})
        queue=[(1,frame(1),[3]),(2,frame(2),[9])]
        with patch.dict(os.environ,{"QILIN_ONLINE_DIEN_TOPN":"1"}):
            one=module.score_queued(model,queue,"request_idx",batch_size=1)
            many=module.score_queued(model,queue,"request_idx",batch_size=512)
        self.assertEqual(set(many),{(1,11),(2,21)})
        for key in one:self.assertAlmostEqual(one[key],many[key],places=7)
        self.assertAlmostEqual(many[(1,11)],.2+.03+.011,places=6)
        self.assertAlmostEqual(many[(2,21)],.2+.09+.021,places=6)
if __name__=="__main__":unittest.main()
