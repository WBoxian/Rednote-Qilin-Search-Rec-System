import unittest
import numpy as np
from scipy import sparse
from recall.cf_retrieval import CFGraphRecaller

class SharedCFRecallerTests(unittest.TestCase):
    def recaller(self):
        graph=sparse.csr_matrix(np.array([[1.,1.,0.,0.],[0.,1.,1.,0.],[0.,0.,1.,1.]],np.float32))
        swing=sparse.csr_matrix((np.array([1.,1.],np.float32),([0,1],[2,3])),shape=(4,4))
        return CFGraphRecaller.from_arrays(graph,np.array([0,1,2]),np.array([0,10,20,30]),swing,
            {'fold_in':True,'neighbors':2})
    def test_current_history_note_zero_recency_and_no_self_neighbor(self):
        recaller=self.recaller()
        actual=recaller.score(0,[0,10],1000)
        self.assertEqual(actual['swing'],{20:1.,30:.5})
        self.assertNotIn(0,actual['usercf'])
        self.assertGreater(actual['usercf'][20],0.)
        reversed_history=recaller.score(0,[10,0],1000)
        self.assertEqual(reversed_history['swing'],{30:1.,20:.5})
    def test_repeat_empty_and_zero_budget_are_stable(self):
        recaller=self.recaller();before=recaller.score(0,[0,10],1000)
        self.assertEqual(recaller.score(99,[],1000),{'swing':{},'usercf':{}})
        self.assertEqual(recaller.score(0,[0,10],0),{'swing':{},'usercf':{}})
        self.assertEqual(recaller.score(0,[0,10],1000),before)

if __name__=='__main__':unittest.main()