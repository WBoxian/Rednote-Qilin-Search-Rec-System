"""Versioned request-context filtering and fixed reciprocal-rank fusion for Rec."""
from __future__ import annotations
from collections.abc import Mapping,Container
import numpy as np,pandas as pd
from training.history import recent_history_ids
POLICIES=('baseline','filter_history','filter_history_context')

def fuse_request_routes(routes:Mapping[str,object],history,graph_items:Container[int],policy:str,topk:int=500):
 """Fuse ORIGINAL Top1000 ID arrays in rank order; filtering never retrieves extra items.

 History is the shared most-recent-first recent20 snapshot. Duplicate history
 engagements remain in the coverage denominator, matching CFGraphRecaller.
 Routes may be empty; real item0 is valid. Output scores are float64/higher-better.
 """
 if policy not in POLICIES or topk<1:raise ValueError('Unsupported context-fusion policy/budget')
 if set(routes)-{'ann','swing','usercf'}:raise ValueError('Only declared DSSM/Swing/UserCF routes are eligible')
 recent=recent_history_ids(history,20);consumed=set(recent);coverage=len({item for item in recent if item in graph_items})/max(len(recent),1)
 weights={'ann':1.,'swing':2.,'usercf':1.}
 if policy=='filter_history_context' and coverage<.5:weights['swing']=1.
 scores={}
 for name in ['ann','swing','usercf']:
  values=np.asarray(routes.get(name,[]))
  if values.ndim!=1 or len(values)>1000:raise ValueError('Routes must be rank-ordered arrays with original budget<=1000')
  if not np.issubdtype(values.dtype,np.integer) and len(values):raise ValueError('Item IDs must be integers')
  ids=list(map(int,values))
  if len(set(ids))!=len(ids) or any(i<0 for i in ids):raise ValueError('Duplicate/negative route item identity')
  if policy!='baseline':ids=[i for i in ids if i not in consumed]
  for rank,item in enumerate(ids,1):scores[item]=scores.get(item,0.)+weights[name]/(60.+rank)
 order=sorted(scores,key=lambda item:(-scores[item],item))[:topk]
 return pd.DataFrame({'note_idx':pd.Series(order,dtype='int64'),'score':pd.Series([scores[item] for item in order],dtype='float64'),'rank':pd.Series(np.arange(1,len(order)+1),dtype='int64')})
