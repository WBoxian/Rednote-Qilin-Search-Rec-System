import importlib.util
from pathlib import Path
import unittest
import tempfile
import duckdb
import numpy as np
import pandas as pd
from scipy import sparse

spec=importlib.util.spec_from_file_location("rec_optimizer",Path(__file__).resolve().parents[1]/"scripts/optimize_rec_retrieval.py")
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class RetrievalProtocolTests(unittest.TestCase):
    def test_whole_requests_exclude_boundary_crossing_and_future_train(self):
        intervals=pd.DataFrame({"gid":[1,2,3,4,5],"lo":[0,2,3,4,12],"hi":[1,3,8,5,13]})
        with self.assertRaisesRegex(ValueError,"strictly forward"):
            module.temporal_partition(intervals,10,.5)
        intervals=pd.DataFrame({"gid":[1,2,3,4,5],"lo":[0,2,4,7,12],"hi":[1,3,6,8,13]})
        result,boundary=module.temporal_partition(intervals,10,.5)
        self.assertEqual(boundary,6)
        self.assertEqual(dict(zip(result.gid,result.partition)),{1:"fit",2:"fit",3:"excluded",4:"validation",5:"excluded"})

    def test_graph_uses_only_passed_fit_edges_and_history(self):
        clicks=pd.DataFrame({"user_idx":[1,2,2],"note_idx":[10,20,99],"positive":[1,1,0],"hi":[1,2,2]})
        requests=pd.DataFrame({"user_idx":[1,2],"history":[[30],[40]],"hi":[1,2]})
        graph,users,items=module.build_graph(clicks,requests,.25)
        self.assertEqual(set(items),{10,20,30,40})
        self.assertEqual(graph.nnz,4)
        self.assertNotIn(99,items)
        self.assertEqual(graph[0,list(items).index(30)],.25)

    def test_no_hit_requests_remain_in_recall_denominator(self):
        requests=pd.DataFrame({"gid":[1,2],"user_idx":[8,9],"positives":[[10,11],[12]]})
        cand=pd.DataFrame({"gid":[1],"note_idx":[10],"score":[1.]})
        result,_=module.metrics(requests,cand)
        self.assertEqual(result["all_requests"],2)
        self.assertEqual(result["Recall@500"],.25)
        self.assertEqual(result["HitRate@500"],.5)

    def test_swing_requires_two_shared_users_not_single_user_fallback(self):
        graph=sparse.csr_matrix([[1,1,0],[1,1,1]],dtype=np.float32)
        swing=module.build_swing(graph,np.array([7,8]),np.array([10,11,12]))
        self.assertGreater(swing[0,1],0)
        self.assertEqual(swing[0,2],0)


    def test_sql_metric_keeps_null_first_hits_as_zero_mrr(self):
        con=duckdb.connect()
        requests=pd.DataFrame({"gid":[1,2],"user_idx":[8,9],"positives":[[10],[20]]})
        con.register("requests",requests)
        con.execute("CREATE TABLE pool AS SELECT 1 gid,10 note_idx,1.0 a,0.0 s,0.0 u")
        result,frame=module.sql_route_metrics(con,"a+s+u")
        self.assertEqual(result["Recall@500"],.5)
        self.assertEqual(result["MRR@100"],.5)
        self.assertEqual(frame.loc[frame.gid==2,"mrr100"].iloc[0],0.)

    def test_fold_in_serves_unknown_user_from_fixed_fit_graph(self):
        requests=pd.DataFrame({"gid":[90],"user_idx":[999],"history":[[10]],"positives":[[20]]})
        graph=sparse.csr_matrix([[1.,0.],[1.,1.]],dtype=np.float32)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)
            module.query_cf(requests,graph,np.array([1,2]),np.array([10,20]),
                sparse.csr_matrix((2,2)),{"fold_in":True,"neighbors":2},path)
            result=pd.read_parquet(path/"usercf.parquet")
            self.assertIn(20,set(result.note_idx))
            self.assertEqual(graph.nnz,3)  # Query history is not inserted into fitted edges.



    def test_multi_click_request_is_not_a_false_negative(self):
        import torch
        users=torch.tensor([[1.,0.],[1.,0.]],requires_grad=True)
        items=torch.tensor([[1.,0.],[0.,1.]],requires_grad=True)
        labels=torch.tensor([10,20])
        ordinary=module.multi_positive_inbatch_loss(users,items,labels,torch.tensor([1,2]))
        multi=module.multi_positive_inbatch_loss(users,items,labels,torch.tensor([1,1]))
        self.assertGreater(float(ordinary.detach()),0.)
        self.assertAlmostEqual(float(multi.detach()),0.)
        multi.backward()
        self.assertTrue(torch.isfinite(users.grad).all())
        self.assertTrue(torch.isfinite(items.grad).all())

    def test_duplicate_item_is_positive_across_different_requests(self):
        import torch
        vectors=torch.eye(2)
        value=module.multi_positive_inbatch_loss(vectors,vectors,torch.tensor([10,10]),torch.tensor([1,2]))
        self.assertAlmostEqual(float(value),0.)



    def test_tail_quota_preserves_head_deduplicates_and_fills_budget(self):
        baseline=list(range(1,1001))
        routes=[[1,101,1001],[101,1002],[1001,1003]]
        scores={item:1./item for item in range(1,1004)}
        selected,fulfilled=module.protected_tail_quota(baseline,routes,scores,(2,1,1))
        self.assertEqual(selected[:100],baseline[:100])
        self.assertEqual(len(selected),500)
        self.assertEqual(len(set(selected)),500)
        self.assertEqual(fulfilled,[2,1,1])
        self.assertTrue({1001,1002,1003}.issubset(selected))
        self.assertEqual(selected[100:],sorted(selected[100:],key=lambda item:(-scores[item],item)))

    def test_empty_quota_replays_original_top500(self):
        baseline=list(range(1,1001))
        selected,fulfilled=module.protected_tail_quota(baseline,[[],[],[]],
            {item:1./item for item in baseline},(0,0,0))
        self.assertEqual(selected,baseline[:500])
        self.assertEqual(fulfilled,[0,0,0])
        with self.assertRaises(ValueError):
            module.protected_tail_quota(baseline,[[],[],[]],{},(401,0,0))


if __name__=="__main__":unittest.main()
