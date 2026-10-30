import ast
import unittest
from pathlib import Path
import numpy as np
from recall.fusion import weighted_rrf

ROOT=Path(__file__).resolve().parents[1]

class RecallFusionTests(unittest.TestCase):
    def test_full_union_is_scored_before_truncation(self):
        result=weighted_rrf({1:.9,2:.8},{3:.9,2:.8},{4:.9,2:.8},topk=1)
        self.assertEqual(result[0][0],2)
        self.assertEqual(result[0][5:8],(1,1,1))

    def test_dictionary_insertion_order_is_irrelevant(self):
        first=weighted_rrf({2:.1,1:.9},{},{},topk=2)
        second=weighted_rrf({1:.9,2:.1},{},{},topk=2)
        self.assertEqual(first,second)
        self.assertEqual([row[0] for row in first],[1,2])

    def test_weights_change_top_candidate(self):
        self.assertEqual(weighted_rrf({1:1.},{2:1.},{},topk=1,route_weights={"ann":1,"swing":2})[0][0],2)

    def test_ties_dedup_and_empty_inputs(self):
        result=weighted_rrf({4:1.,2:1.},{2:1.},{},topk=10)
        self.assertEqual([x[0] for x in result],[2,4])
        self.assertEqual(weighted_rrf({},{},{},topk=10),[])
        self.assertEqual(weighted_rrf({1:1},{},{},topk=0),[])

    def test_invalid_weights_fail_explicitly(self):
        for weight in (-1.,float("nan"),float("inf")):
            with self.assertRaises(ValueError):
                weighted_rrf({1:1},{},{},topk=1,route_weights={"ann":weight})

    def test_offline_online_use_identical_explicit_rrf(self):
        outputs=[]
        for name in ("src/recall/build_multiroute_recall.py","src/backend/online/recall/service.py"):
            tree=ast.parse((ROOT/name).read_text(encoding="utf-8-sig"))
            function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=="_merge_scores")
            namespace={"np":np,"weighted_rrf":weighted_rrf}
            exec(compile(ast.Module(body=[function],type_ignores=[]),name,"exec"),namespace)
            kwargs={"ann_scores":{1:.9,2:.8},"swing_scores":{3:.9,2:.8},"usercf_scores":{},
                    "merge_order":["ann","swing","usercf"],"topk":2,"route_min_quota":0,
                    "route_max_share":1.,"route_weights":{"ann":1,"swing":1,"usercf":1},
                    "fusion_strategy":"rrf"}
            if "online" in name:
                kwargs["rrf_k"]=60
            outputs.append(namespace["_merge_scores"](**kwargs))
        self.assertEqual(outputs[0],outputs[1])
        self.assertEqual(outputs[0][0][0],2)

if __name__=="__main__":
    unittest.main()
