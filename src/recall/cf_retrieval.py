"""Fixed-graph Swing and UserCF retrieval shared by offline evaluation and serving."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy import sparse
from training.history import recent_history_ids

class CFGraphRecaller:
    def __init__(self, graph_dir: str | Path, config: dict):
        directory=Path(graph_dir)
        self._initialize(sparse.load_npz(directory/'graph.npz'),np.load(directory/'users.npy',allow_pickle=False),
            np.load(directory/'items.npy',allow_pickle=False),sparse.load_npz(directory/'swing.npz'),config)

    @classmethod
    def from_arrays(cls,graph,users,items,swing,config):
        result=cls.__new__(cls);result._initialize(graph,users,items,swing,config);return result

    def _initialize(self,graph,users,items,swing,config):
        self.graph=graph.tocsr();self.users=np.asarray(users,np.int64);self.items=np.asarray(items,np.int64)
        self.swing=swing.tocsr();self.config=dict(config)
        if self.graph.shape!=(len(self.users),len(self.items)) or self.swing.shape!=(len(self.items),len(self.items)):
            raise ValueError('CF graph dimensions do not match ID mappings')
        if np.any(self.users<0) or np.any(self.items<0) or np.any(np.diff(self.users)<=0) or np.any(np.diff(self.items)<=0):
            raise ValueError('CF mappings must be sorted unique nonnegative IDs')
        if any(np.any(~np.isfinite(matrix.data)) or np.any(matrix.data<0) for matrix in (self.graph,self.swing)):
            raise ValueError('CF graph weights must be finite nonnegative')
        if not isinstance(self.config.get('fold_in'),bool) or int(self.config.get('neighbors',0))<1:
            raise ValueError('CF config requires fold_in and positive neighbors')
        self.user_pos={int(user):index for index,user in enumerate(self.users)}
        self.item_pos={int(item):index for index,item in enumerate(self.items)}
        popularity=np.diff(self.graph.tocsc().indptr)
        idf=1./np.maximum(np.log1p(popularity),1e-6)
        self.sqrt_idf=np.sqrt(idf)
        weighted=self.graph.multiply(self.sqrt_idf).tocsr()
        norms=np.sqrt(weighted.power(2).sum(axis=1)).A1
        self.normalized=(sparse.diags(1./np.maximum(norms,1e-12))@weighted).tocsr()

    @staticmethod
    def _top(ids,values,k):
        return np.lexsort((ids,-values))[:max(0,int(k))]

    def score(self,user_idx:int,history,topk:int=1000)->dict[str,dict[int,float]]:
        return self.score_with_support(user_idx,history,topk)[0]

    def score_with_support(self,user_idx:int,history,topk:int=1000):
        ids=recent_history_ids(history);profile={}
        for offset,item in enumerate(ids):
            position=self.item_pos.get(item)
            if position is not None:profile[position]=profile.get(position,0.)+1./(1.+offset)
        cols=np.asarray(sorted(profile),np.int32)
        query=sparse.csr_matrix((np.asarray([profile[position] for position in cols],np.float32),
            (np.zeros(len(cols),np.int32),cols)),shape=(1,len(self.items)))
        sw=(query@self.swing).tocsr()
        if self.config['fold_in']:
            weighted=query.multiply(self.sqrt_idf).tocsr()
            norm=np.sqrt(weighted.power(2).sum())
            similarities=(weighted@self.normalized.T).tocsr()
            if norm>0:similarities.data/=norm
        elif int(user_idx) in self.user_pos:
            similarities=(self.normalized[self.user_pos[int(user_idx)]]@self.normalized.T).tocsr()
        else:similarities=sparse.csr_matrix((1,len(self.users)),dtype=np.float32)
        own=self.user_pos.get(int(user_idx))
        if own is not None:
            similarities.data[similarities.indices==own]=0.;similarities.eliminate_zeros()
        chosen=self._top(self.users[similarities.indices],similarities.data,self.config['neighbors'])
        neighbors=similarities.indices[chosen];weights=similarities.data[chosen]
        user_query=sparse.csr_matrix((weights,(np.zeros(len(neighbors),np.int32),neighbors)),shape=(1,len(self.users)))
        uc=(user_query@self.graph).tocsr()
        result={}
        for name,scores in (('swing',sw),('usercf',uc)):
            order=self._top(self.items[scores.indices],scores.data,topk)
            result[name]={int(self.items[scores.indices[pos]]):float(scores.data[pos]) for pos in order}
        support={'history_items':len(ids),'history_items_in_graph':len(profile),
            'history_items_with_swing_edges':sum(self.swing.indptr[pos+1]>self.swing.indptr[pos] for pos in profile),
            'user_in_fit_graph':int(user_idx) in self.user_pos,'user_neighbors':len(neighbors),
            'swing_pool_size':sw.nnz,'usercf_pool_size':uc.nnz}
        return result,support