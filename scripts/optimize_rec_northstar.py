"""Frozen train-only Recommendation north-star optimization; never writes deploy assets."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from scripts import optimize_rec_retrieval as rec

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'outputs/evaluation/northstar_fix_20261031/rec'
CF_GRID=(
 {'name':'click_fold50','history_weight':0.,'fold_in':True,'neighbors':50},
 {'name':'hist025_fold50','history_weight':.25,'fold_in':True,'neighbors':50},
 {'name':'hist025_fold200','history_weight':.25,'fold_in':True,'neighbors':200},
)
WEIGHTS=((1,1,1),(2,1,1),(4,1,1),(1,2,1),(1,4,1),(1,8,1),(2,4,1))
RRF_K=(10,60)

def prepare(out,packed_sequence=False):
    rec.prepare(out)
    plan={'scene':'rec','training':'original recommendation_train only, forward complete-request split',
        'selection':'validation primary Recall@100; guard HR@50 and known-positive NDCG@10 >= baseline; ties HR50 thenMRR100; retain baseline if no candidate passes',
        'baseline':'new canonical-history DSSM + hist025_fold200 + RRF1:8:1/k60; prior frozen candidates reported separately as historical lineage',
        'packed_sequence':bool(packed_sequence),'epochs':[1,2,3],'seed':42,'temperature':.07,'batch_size':256,
        'loss':'same_request_or_same_item_v2','history_contract':'most_recent_first_v1',
        'masked_profile_fields':['fans_num','follows_num']+[f'dense_feat{i}' for i in range(1,41)],
        'cf_grid':CF_GRID,'fusion_weights':WEIGHTS,'rrf_k':RRF_K,
        'catalog':1983938,'route_budget':1000,'candidate_budget':500,
        'no_test_selection':True,'no_transition_or_popular_route':True,
        'reporting':'all original official test requests; full catalog snapshot; unexposed candidates not labeled negatives',
        'fit_last_event':1732696815,'validation_first_event':1732696816,'official_test_first_event':1732706088}
    rec.write_json(out/'northstar_plan.json',plan)

def train(out):
    rec.train(out,epochs=3,loss_contract='same_request_or_same_item_v2',
        selection_objective='recall100',mask_unverified_user_dense=True,
        mask_sequence_padding=bool(json.loads((out/'northstar_plan.json').read_text()).get('packed_sequence',False)))



def route_groups(path):
    import numpy as np
    import pandas as pd
    frame=pd.read_parquet(path,columns=['gid','note_idx','rank'])
    if frame[['gid','note_idx']].duplicated().any() or (frame['rank']<1).any():
        raise ValueError('Invalid or duplicate route candidates')
    return {int(gid):(part.note_idx.to_numpy(np.int64),part['rank'].to_numpy(np.int64))
        for gid,part in frame.groupby('gid',sort=False)}


def pool_for(gid,groups):
    import numpy as np
    sources=[group.get(int(gid),(np.empty(0,np.int64),np.empty(0,np.int64))) for group in groups]
    items=np.unique(np.concatenate([source[0] for source in sources]))
    ranks=np.full((3,len(items)),np.inf)
    for route,(ids,order) in enumerate(sources):
        ranks[route,np.searchsorted(items,ids)]=order
    return items,ranks


def evaluate_recipes(requests,paths,recipes):
    """Exact positive ranks with item-ID ties; avoids sorting an entire grid repeatedly."""
    import numpy as np
    import pandas as pd
    groups=[route_groups(path) for path in paths]
    values=np.zeros((len(recipes),len(requests),7),np.float64)
    ks=np.asarray([trial['rrf_k'] for trial in recipes],np.float64)
    weights=np.asarray([trial['weights'] for trial in recipes],np.float64)
    for number,row in enumerate(requests.itertuples()):
        truth=set(map(int,row.positives));items,ranks=pool_for(row.gid,groups)
        if not truth or not len(items):continue
        scores=np.sum(weights[:,:,None]/(ks[:,None,None]+ranks[None,:,:]),axis=1)
        truth_positions=np.flatnonzero(np.isin(items,np.fromiter(truth,dtype=np.int64)))
        first=np.full(len(recipes),np.inf);h100=np.zeros(len(recipes));h500=np.zeros(len(recipes));dcg=np.zeros(len(recipes))
        for position in truth_positions:
            candidate=scores[:,position,None]
            rank=1+np.count_nonzero((scores>candidate)|((scores==candidate)&(items[None,:]<items[position])),axis=1)
            first=np.minimum(first,rank);h100+=(rank<=100);h500+=(rank<=500)
            dcg+=np.where(rank<=10,1./np.log2(rank+1.),0.)
        ideal=sum(1./np.log2(rank+1.) for rank in range(1,min(10,len(truth))+1))
        values[:,number,0]=(first<=50);values[:,number,1]=h100/len(truth)
        values[:,number,2]=dcg/ideal;values[:,number,3]=np.where(first<=100,1./first,0.)
        values[:,number,4]=h500/len(truth);values[:,number,5]=(first<=500)
        values[:,number,6]=h100
    names=['HitRate@50','Recall@100','known_positive_NDCG@10','MRR@100','Recall@500','HitRate@500','hits100']
    records=[]
    for index,recipe in enumerate(recipes):
        record={"weights":list(recipe["weights"]),"rrf_k":int(recipe["rrf_k"])}
        record.update({name:float(values[index,:,column].mean()) for column,name in enumerate(names[:-1])})
        record.update({'all_requests':len(requests),'positive_pairs':sum(len(set(map(int,row.positives))) for row in requests.itertuples()),
                       'positive_hits100':int(values[index,:,6].sum())})
        records.append(record)
    return records,values


def write_candidates(requests,paths,recipe,destination):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    groups=[route_groups(path) for path in paths];buffer=[]
    schema=pa.schema([('gid',pa.int64()),('note_idx',pa.int64()),('score',pa.float64()),('rank',pa.int64())])
    destination.parent.mkdir(parents=True,exist_ok=True)
    with pq.ParquetWriter(destination,schema) as writer:
        for row in requests.itertuples():
            items,ranks=pool_for(row.gid,groups)
            scores=np.sum(np.asarray(recipe['weights'],np.float64)[:,None]/(recipe['rrf_k']+ranks),axis=0)
            order=np.lexsort((items,-scores))[:500]
            buffer.extend({'gid':int(row.gid),'note_idx':int(items[pos]),'score':float(scores[pos]),'rank':rank}
                for rank,pos in enumerate(order,1))
            if len(buffer)>=100000:
                writer.write_table(pa.Table.from_pylist(buffer,schema=schema));buffer=[]
        if buffer:writer.write_table(pa.Table.from_pylist(buffer,schema=schema))


def select(out):
    import numpy as np
    import pandas as pd
    if (out/'selection.json').exists():raise RuntimeError('Selection is frozen')
    index=json.loads((out/'index/selected.json').read_text())
    ann=out/f"index/validation_nprobe{index['nprobe']}.parquet"
    requests=pd.read_parquet(out/'requests_validation.parquet')
    trials=[{'weights':list(weights),'rrf_k':k} for weights in WEIGHTS for k in RRF_K]
    records=[];frames={}
    for config in CF_GRID:
        directory=out/'validation'/config['name']
        rows,values=evaluate_recipes(requests,[ann,directory/'swing.parquet',directory/'usercf.parquet'],trials)
        for number,row in enumerate(rows):
            row['cf']=config;row['trial_id']=len(records);records.append(row)
            frames[row['trial_id']]=values[number]
        print('grid complete',config['name'],flush=True)
    baseline=next(row for row in records if row['cf']['name']=='hist025_fold200' and row['weights']==[1,8,1] and row['rrf_k']==60)
    for row in records:
        row['guard_passed']=(row['HitRate@50']>=baseline['HitRate@50'] and
            row['known_positive_NDCG@10']>=baseline['known_positive_NDCG@10'])
    winner=max((row for row in records if row['guard_passed']),
        key=lambda row:(row['Recall@100'],row['HitRate@50'],row['MRR@100'],-row['trial_id']))
    rec.write_json(out/'northstar_validation_grid.json',{'baseline':baseline,'trials':records})
    selection={'scene':'rec','selection_split':'original-train forward validation only',
        'objective':'macro Recall@100','guards':'HR@50 and known-positiveNDCG10 >= fixedbaseline',
        'baseline':baseline,'selected':winner,'index':index,
        'selected_dssm':json.loads((out/'dssm/selected.json').read_text()),
        'plan_sha256':rec.digest(out/'northstar_plan.json'),
        'grid_sha256':rec.digest(out/'northstar_validation_grid.json'),
        'frozen_before_own_official_test':True,
        'historical_test_status':'previously observed regression benchmark, not a fresh blind holdout',
        'changed_from_baseline':winner['trial_id']!=baseline['trial_id']}
    rec.write_json(out/'selection.json',selection)
    directory=out/'validation'/winner['cf']['name']
    write_candidates(requests,[ann,directory/'swing.parquet',directory/'usercf.parquet'],winner,out/'validation_selected.parquet')
    baseframe=requests[['gid','user_idx']].copy()
    names=['hit50','recall100','ndcg10','mrr100','recall500','hit500','hits100']
    for label,recipe in [('baseline',baseline),('selected',winner)]:
        evidence=baseframe.copy()
        for column,name in enumerate(names):evidence[name]=frames[recipe['trial_id']][:,column]
        evidence.to_parquet(out/f'validation_{label}_northstar_requests.parquet',index=False)
    print(json.dumps(selection,indent=2),flush=True)


def prepare_test(out):
    import duckdb
    import numpy as np
    import pandas as pd
    import faiss,torch
    from backend.online.item_features import get_item_feature_store
    from recall import dssm_trainer as trainer
    selected=json.loads((out/'selection.json').read_text())
    raw=ROOT/'datasets/recommendation_test/train-00000-of-00001.parquet'
    manifest=json.loads((out/'manifest.json').read_text())
    if rec.digest(raw)!=manifest['official_test_sha256']:raise ValueError('Test source changed')
    destination=out/'test';destination.mkdir(exist_ok=True)
    with duckdb.connect(config={'threads':'2','memory_limit':'2GB'}) as con:
        requests=con.execute('''SELECT r.request_idx gid,r.user_idx,r.recent_clicked_note_idxs history,
         list_sort(list(DISTINCT d.note_idx) FILTER(WHERE d.click>0)) positives
         FROM read_parquet(?) r,unnest(r.rec_result_details_with_idx) u(d)
         GROUP BY r.request_idx,r.user_idx,r.recent_clicked_note_idxs ORDER BY gid''',[str(raw)]).fetchdf()
    requests.positives=requests.positives.apply(lambda value:list(value) if isinstance(value,(list,np.ndarray)) else [])
    requests.to_parquet(destination/'requests.parquet',index=False)
    if len(requests)!=11115:raise ValueError('Incomplete full test')
    meta=json.loads((out/'dssm/training_manifest.json').read_text())
    epoch=Path(selected['selected_dssm']['directory']);checkpoint=epoch/'dssm_rec_optimized.pt'
    if rec.digest(checkpoint)!=selected['selected_dssm']['checkpoint_sha256']:raise ValueError('Checkpoint changed')
    device=torch.device('cpu');torch.set_num_threads(2);trainer.DEVICE=device
    _,item_cols,_=trainer.numeric_feature_columns('rec')
    model=trainer.DSSMModel('rec',meta['cat_vocabs'],len(item_cols),42+768+128,preprocessing=meta['normalizer']).to(device)
    model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True));model.eval()
    stores={'note_text':trainer.MMapStore('note_text_emb','note_text'),'note_img':trainer.MMapStore('note_img_emb','note_img')}
    frame=rec.raw_model_frame(pd.read_parquet(ROOT/'features/rec_test_features.parquet'),get_item_feature_store(ROOT),trainer,'rec')
    if meta.get('masked_user_dense'):
        frame[['fans_num','follows_num']+[f'dense_feat{i}' for i in range(1,41)]]=0.
    ids,queries=rec.encode_requests(model,frame,stores,'rec',meta['unknown_user_id'],set(meta['known_user_ids']))
    if set(map(int,ids))!=set(map(int,requests.gid)):raise ValueError('Incomplete request vectors')
    np.save(destination/'request_ids.npy',ids);np.save(destination/'query_vectors.npy',queries)
    items=np.load(out/'dssm/item_ids.npy');index=faiss.read_index(str(out/'index/ivfpq.faiss'))
    index.nprobe=selected['index']['nprobe'];faiss.omp_set_num_threads(2)
    rec.retrieve_index(index,ids,queries,items,destination/'ann.parquet')
    return requests


def test(out):
    import numpy as np
    import pandas as pd
    from scipy import sparse
    destination=out/'test'
    if (destination/'northstar_metrics.json').exists():raise RuntimeError('Official test is frozen; no retuning')
    selected=json.loads((out/'selection.json').read_text())
    requests=prepare_test(out)
    recipes=[selected['baseline'],selected['selected']]
    grid=[]
    for config in CF_GRID:
        directory=destination/'cf'/config['name'];directory.mkdir(parents=True,exist_ok=True)
        cache=out/f"cf_graph_{config['history_weight']:g}"
        rec.query_cf(requests,sparse.load_npz(cache/'graph.npz'),np.load(cache/'users.npy'),np.load(cache/'items.npy'),
            sparse.load_npz(cache/'swing.npz'),config,directory,directory/'support.parquet')
        paths=[destination/'ann.parquet',directory/'swing.parquet',directory/'usercf.parquet']
        rows,values=evaluate_recipes(requests,paths,recipes)
        for row in rows:row['cf']=config
        grid.extend(rows)
        if config==selected['selected']['cf']:
            write_candidates(requests,paths,selected['selected'],destination/'selected.parquet')
            write_candidates(requests,paths,selected['baseline'],destination/'same_cf_baseline_weights.parquet')
        for route in ('swing','usercf'):
            metrics,_=rec.metrics(requests,pd.read_parquet(directory/f'{route}.parquet'))
            grid.append({'cf':config,'route':route,**metrics})
        print('test fixed ablation',config['name'],flush=True)
    metrics,per_request=rec.metrics(requests,pd.read_parquet(destination/'selected.parquet'))
    per_request.to_parquet(destination/'selected_metrics.parquet',index=False)
    result={'scene':'rec','scope':'full original11115 test requests, fullcatalog1983938, known-clickretention',
        'selected':metrics,'selected_parameters':selected['selected'],
        'ann_only':rec.metrics(requests,pd.read_parquet(destination/'ann.parquet'))[0],
        'fixed_test_ablations':grid,'selection_sha256':rec.digest(out/'selection.json'),
        'outputs':{p.name:rec.digest(p) for p in destination.glob('*.parquet')},
        'limitations':['No untouched holdout claim','CF/history time lineage incomplete','No candidate publication timestamps',
            'Known-positiveNDCG is not exposure-poolNDCG or onlineCTR; unexposed items are not negative labels']}
    rec.write_json(destination/'northstar_metrics.json',result);print(json.dumps(result,indent=2),flush=True)


def refit(out):
    """Predeclared data-volume ablation; no hyperparameters are selected on test."""
    import time
    import duckdb,numpy as np,pandas as pd
    from scipy import sparse
    selected=json.loads((out/'selection.json').read_text())
    manifest=json.loads((out/'manifest.json').read_text())
    destination=out/'refit_cf_pretest';destination.mkdir(exist_ok=True)
    if (destination/'northstar_metrics.json').exists():raise RuntimeError('Refit ablation is frozen')
    raw=ROOT/'datasets/recommendation_train/train-00000-of-00001.parquet'
    if rec.digest(raw)!=manifest['inputs'][str(raw.relative_to(ROOT))]:raise ValueError('Train source changed')
    with duckdb.connect(config={'threads':'2','memory_limit':'2GB'}) as con:
        con.execute('''CREATE TABLE labels AS SELECT r.request_idx gid,first(r.user_idx) user_idx,
         d.note_idx,max(d.click)::INT positive,min(d.request_timestamp) lo,max(d.request_timestamp) hi
         FROM read_parquet(?) r,unnest(r.rec_result_details_with_idx) u(d) GROUP BY r.request_idx,d.note_idx''',[str(raw)])
        con.execute('CREATE TABLE eligible AS SELECT gid,min(lo) lo,max(hi) hi FROM labels GROUP BY gid HAVING max(hi)<?',[manifest['official_test_first_event']])
        labels=con.execute('SELECT l.* FROM labels l JOIN eligible USING(gid)').fetchdf()
        requests=con.execute('''SELECT r.request_idx gid,r.user_idx,r.recent_clicked_note_idxs history,
         first(e.hi) hi,list_sort(list(DISTINCT l.note_idx) FILTER(WHERE l.positive>0)) positives
         FROM read_parquet(?) r JOIN eligible e ON r.request_idx=e.gid JOIN labels l ON l.gid=r.request_idx
         GROUP BY r.request_idx,r.user_idx,r.recent_clicked_note_idxs ORDER BY gid''',[str(raw)]).fetchdf()
    if labels.hi.max()>=manifest['official_test_first_event']:raise ValueError('Refit includes future label')
    labels.to_parquet(destination/'train_labels.parquet',index=False);requests.to_parquet(destination/'train_requests.parquet',index=False)
    config=selected['selected']['cf'];started=time.monotonic()
    graph,users,items=rec.build_graph(labels,requests,config['history_weight'])
    swing=rec.build_swing(graph,users,items)
    sparse.save_npz(destination/'graph.npz',graph);sparse.save_npz(destination/'swing.npz',swing)
    np.save(destination/'users.npy',users);np.save(destination/'items.npy',items)
    test_requests=pd.read_parquet(out/'test/requests.parquet')
    rec.query_cf(test_requests,graph,users,items,swing,config,destination,destination/'support.parquet')
    paths=[out/'test/ann.parquet',destination/'swing.parquet',destination/'usercf.parquet']
    write_candidates(test_requests,paths,selected['selected'],destination/'selected.parquet')
    score,per_request=rec.metrics(test_requests,pd.read_parquet(destination/'selected.parquet'))
    per_request.to_parquet(destination/'selected_metrics.parquet',index=False)
    main=pd.read_parquet(out/'test/selected_metrics.parquet')
    paired=per_request.merge(main,on=['gid','user_idx'],suffixes=('_refit','_main'),validate='one_to_one')
    changes={}
    for column,label in [('hit50','HitRate@50'),('recall100','Recall@100'),('ndcg10','known_positive_NDCG@10'),('recall500','Recall@500')]:
        delta=paired[f'{column}_refit']-paired[f'{column}_main']
        blocks=paired.assign(delta=delta).groupby('user_idx').delta.agg(['sum','count']).to_numpy()
        rng=np.random.default_rng(42);draws=[]
        for _ in range(1000):
            sample=blocks[rng.integers(0,len(blocks),len(blocks))];draws.append(float(sample[:,0].sum()/sample[:,1].sum()))
        changes[label]={'paired_macro_delta':float(delta.mean()),'user_cluster_CI95':list(map(float,np.quantile(draws,[.025,.975])))}
    result={'scope':'fixed-parameter whole-original-train pretest CF refit; data-volume ablation, not same-data algorithm gain',
        'selection_sha256':rec.digest(out/'selection.json'),'train_requests':len(requests),'train_positive_pairs':int(labels.positive.sum()),
        'last_train_event':float(labels.hi.max()),'test_start':manifest['official_test_first_event'],
        'graph':{'users':len(users),'items':len(items),'edges':graph.nnz,'swing_edges':swing.nnz},
        'DSSM':'same frozen selected epoch3','index':'same frozen selected PQ nprobe32',
        'selected':score,'delta_from_fit_only':changes,'seconds':time.monotonic()-started,
        'outputs':{p.name:rec.digest(p) for p in destination.glob('*.parquet')},
        'limitations':['Only original training-source labels fitted; validation labels reclaimed after parameter freeze',
          'History snapshot timestamps and item publication times remain unknown','Not a fresh blind holdout','No online CTR or latency claim']}
    rec.write_json(destination/'northstar_metrics.json',result);print(json.dumps(result,indent=2),flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--packed-sequence',action='store_true')
    parser.add_argument('--stage',choices=['prepare','train','cf','index','select','test','refit'],required=True)
    args=parser.parse_args();out=args.output.resolve()
    if args.stage=='prepare':prepare(out,packed_sequence=args.packed_sequence)
    elif args.stage=='train':train(out)
    elif args.stage=='index':rec.index_dssm(out,selection_objective='recall100')
    elif args.stage=='select':select(out)
    elif args.stage=='test':test(out)
    elif args.stage=='refit':refit(out)
    elif args.stage=='cf':
        rec.CF_CONFIGS=CF_GRID;rec.cf(out)

if __name__=='__main__':main()