import unittest
import numpy as np
from scripts.analyze_ann_fidelity import topk

class ANNTopKTests(unittest.TestCase):
    def test_block_merge_equals_full_dot_sort_with_ties(self):
        scores=np.array([.8,.9,.9,.1,.9]);ids=np.array([8,9,3,1,2])
        a,b=topk(scores[:2],ids[:2],3)
        merged,actual=topk(np.r_[a,scores[2:]],np.r_[b,ids[2:]],3)
        expected=ids[np.lexsort((ids,-scores))[:3]]
        np.testing.assert_array_equal(actual,expected)
    def test_empty_and_oversized_k(self):
        _,ids=topk(np.array([]),np.array([],dtype=int),500)
        self.assertEqual(len(ids),0)
        _,ids=topk(np.array([1]),np.array([2]),500)
        self.assertEqual(ids.tolist(),[2])
if __name__=="__main__":unittest.main()
