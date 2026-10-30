import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from select_search_fusion import protected_candidates,protected_order,final_protected

class SearchOptimizationTests(unittest.TestCase):
    def test_head_order_survives_ann_overlap_and_budget_is_fixed(self):
        channels=({7:[9,8]}, {}, {7:[8,9]+list(range(1000,2000))})
        order=protected_order(7,channels)
        self.assertEqual(order[:2],[9,8])
        self.assertEqual(len(order),500)
        self.assertEqual(len(set(order)),500)
        frame=protected_candidates([7],channels)
        self.assertTrue(np.all(np.diff(frame.score)<0))
        self.assertEqual(frame.iloc[0].from_dpr,1)
        self.assertEqual(frame.iloc[-1].from_ann,1)

    def test_test_reference_relative_order_is_explicit_and_protected(self):
        channels=({7:[9,8]},{7:[8,10]},{7:[9,10,11]})
        self.assertEqual(protected_order(7,channels,{7:[10,8,9]}),[10,8,9,11])

    def test_test_refuses_to_overwrite_frozen_results(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);(out/"fusion").mkdir();(out/"fusion/test_results.json").write_text("{}")
            with self.assertRaisesRegex(RuntimeError,"already frozen"):final_protected(out)

if __name__=="__main__":unittest.main()