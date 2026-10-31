import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np,pandas as pd,torch
from scripts.optimize_rec_residual import exact_route

class ResidualDriverTests(unittest.TestCase):
 def test_actual_streaming_exact_route_writer_and_zero_query_with_cpu_backend(self):
  rng=np.random.default_rng(42);items=rng.normal(size=(1001,128)).astype(np.float32);query=rng.normal(size=(2,128)).astype(np.float32);query[1]=0;ids=np.array([7,8]);itemids=np.arange(1001,dtype=np.int64);original_empty=torch.empty;original_to=torch.Tensor.to
  def empty(*args,**kwargs):
   if kwargs.get('device')=='cuda':kwargs['device']='cpu'
   return original_empty(*args,**kwargs)
  def to(tensor,*args,**kwargs):
   if args and args[0]=='cuda':args=('cpu',*args[1:])
   if kwargs.get('device')=='cuda':kwargs['device']='cpu'
   return original_to(tensor,*args,**kwargs)
  with tempfile.TemporaryDirectory() as tmp,patch('torch.empty',side_effect=empty),patch.object(torch.Tensor,'to',new=to):
   path=Path(tmp)/'ann.parquet';exact_route(ids,query,items,itemids,path);result=pd.read_parquet(path)
  self.assertEqual(len(result),1000);self.assertEqual(result.gid.unique().tolist(),[7]);self.assertEqual(result['rank'].tolist(),list(range(1,1001)));expected=(torch.from_numpy(query[:1])@torch.from_numpy(items).T).numpy()[0];order=np.lexsort((itemids,-expected))[:1000];np.testing.assert_array_equal(result.note_idx,itemids[order]);np.testing.assert_allclose(result.score,expected[order],atol=0,rtol=0)
if __name__=='__main__':unittest.main()
