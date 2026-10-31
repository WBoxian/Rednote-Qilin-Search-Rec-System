"""Validation-only Search recall selection using local BM25 and train-only DSSM.

No DPR/provider candidate route is accepted. Official test query retrieval starts
only after a hash-bound model/index/fusion recipe has been frozen.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import time
import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from recall.lexical import BM25Index

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/evaluation/northstar_fix_20261031'
ROUTE_TOPK=1000
TOPK=500
HEAD_CAP=100
ANN_WEIGHTS=(0.,.25,.5,1.,2.,4.)
RRF_K=(10,60)
GUARDS=('HitRate@10','MRR@10','known_positive_NDCG@10')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def write_json(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    temporary=Path(str(path)+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


def request_frame(path):
    frame=pd.read_parquet(path).rename(columns={'search_idx':'gid'})
    if not frame.gid.is_unique:raise ValueError('Request IDs must be unique')
    frame['positives']=frame.positives.map(lambda ids:sorted(set(map(int,ids))))
    return frame


def candidate_metrics(requests,path):
    """All-request known-click retention using canonical stored candidate ranks."""
    original=requests.copy()
    original['npos']=original.positives.map(len)
    original['idcg']=original.npos.map(lambda n:sum(1./np.log2(rank+1) for rank in range(1,min(int(n),10)+1)))
    connection=duckdb.connect(config={'threads':'2','memory_limit':'1GB'})
    try:
        connection.register('requests',original[['gid','user_idx','positives','npos','idcg']])
        connection.execute("CREATE TABLE truth AS SELECT gid,unnest(positives)::BIGINT note_idx FROM requests")
        connection.read_parquet(str(path)).create_view('candidates')
        invalid=connection.execute("SELECT count(*) FROM candidates c LEFT JOIN requests r USING(gid) WHERE r.gid IS NULL OR c.note_idx<0 OR c.rank<1 OR NOT isfinite(c.score)").fetchone()[0]
        if invalid:raise ValueError('Invalid/foreign ranked candidates')
        bad=connection.execute("SELECT count(*) FROM (SELECT gid FROM candidates GROUP BY gid HAVING min(rank)<>1 OR max(rank)<>count(*) OR count(DISTINCT note_idx)<>count(*) OR count(DISTINCT rank)<>count(*))").fetchone()[0]
        if bad:raise ValueError('Duplicate/gapped candidate ranks or item IDs')
        frame=connection.execute("""WITH hits AS (
            SELECT c.gid,min(c.rank) first_hit,count(*) FILTER(WHERE c.rank<=500) hits,
                count(*) FILTER(WHERE c.rank<=100) hits100,
                sum(CASE WHEN c.rank<=10 THEN 1./log2(c.rank+1.) ELSE 0. END) dcg10
            FROM candidates c JOIN truth t USING(gid,note_idx) GROUP BY c.gid)
            SELECT r.gid,r.user_idx,r.npos positives,r.idcg,coalesce(h.hits,0) hits,
                coalesce(h.hits100,0) hits100,h.first_hit,coalesce(h.dcg10,0.) dcg10
            FROM requests r LEFT JOIN hits h USING(gid) ORDER BY r.gid""").fetchdf()
        counts=connection.execute('SELECT gid,count(*) candidates FROM candidates GROUP BY gid').fetchdf()
    finally:connection.close()
    frame['first_hit']=frame.first_hit.astype('float64').fillna(np.inf)
    denominator=np.maximum(frame.positives.to_numpy(),1)
    frame['recall500']=frame.hits.to_numpy()/denominator
    frame['recall100']=frame.hits100.to_numpy()/denominator
    frame['ndcg10']=frame.dcg10.to_numpy()/np.maximum(frame.idcg.to_numpy(),1e-12)
    frame['mrr100']=np.where(frame.first_hit<=100,1./frame.first_hit,0.)
    frame['mrr10']=np.where(frame.first_hit<=10,1./frame.first_hit,0.)
    frame['hit10']=(frame.first_hit<=10).astype(int)
    frame['hit50']=(frame.first_hit<=50).astype(int)
    frame['hit500']=(frame.first_hit<=500).astype(int)
    result={'all_requests':len(frame),'positive_requests':int((frame.positives>0).sum()),
        'Recall@500':float(frame.recall500.mean()),'HitRate@500':float(frame.hit500.mean()),
        'Recall@100':float(frame.recall100.mean()),'HitRate@10':float(frame.hit10.mean()),
        'HitRate@50':float(frame.hit50.mean()),'MRR@10':float(frame.mrr10.mean()),'MRR@100':float(frame.mrr100.mean()),
        'known_positive_NDCG@10':float(frame.ndcg10.mean()),'positive_hits':int(frame.hits.sum()),
        'positive_pairs':int(frame.positives.sum()),'empty_requests':len(frame)-len(counts),
        'mean_candidates':float(counts.candidates.sum()/len(frame)) if len(frame) else 0.}
    return result,frame


def lexical_routes(lexical_run,requests_path,split,*,workers=4):
    """Memoize unique queries; retrieve without passing any click label to BM25."""
    lexical_run=Path(lexical_run);target=lexical_run/'lexical';target.mkdir(parents=True,exist_ok=True)
    output=target/f'{split}.parquet';manifest_path=target/f'{split}_manifest.json'
    source=ROOT/'datasets'/f'search_{"train" if split=="validation" else "test"}'/'train-00000-of-00001.parquet'
    requests=request_frame(requests_path)
    canonical=lexical_run/'requests_validation.parquet' if split=='validation' else Path(requests_path)
    canonical_requests=request_frame(canonical)
    if set(requests.gid)!=set(canonical_requests.gid):raise ValueError('Shared lexical validation request cohort changed')
    raw=pd.read_parquet(source,columns=['search_idx','query']).rename(columns={'search_idx':'gid'})
    if (raw.groupby('gid')['query'].nunique(dropna=False)>1).any():raise ValueError('Conflicting queries under one request ID')
    raw=raw.drop_duplicates('gid')
    chosen=requests[['gid']].merge(raw,on='gid',how='left',validate='one_to_one')
    if chosen['query'].isna().any():raise ValueError('Missing original query context')
    index_path=lexical_run/'bm25.sqlite'
    recipe={'split':split,'requests':len(chosen),'unique_queries':int(chosen['query'].nunique()),'route_topk':ROUTE_TOPK,
            'request_ids_sha256':hashlib.sha256(np.sort(chosen.gid.to_numpy(np.int64)).tobytes()).hexdigest(),
            'inputs':{str(path):digest(path) for path in [source,canonical,index_path.with_suffix('.sqlite.manifest.json')]},
            'DPR':False,'query_labels_used':False}
    if output.exists() or manifest_path.exists():
        if not(output.exists() and manifest_path.exists()):raise FileExistsError('Incomplete lexical route artifacts')
        recorded=json.loads(manifest_path.read_text())
        if recorded['recipe']!=recipe or recorded['candidate_sha256']!=digest(output):raise ValueError('Frozen lexical route inputs/content changed')
        return output
    source_copy=target/f'{split}_source.py'
    shutil.copyfile(__file__,source_copy)
    start=time.monotonic();index=BM25Index(index_path);cache={};unique=chosen['query'].drop_duplicates().tolist()
    def lookup(query):
        result=index.search(query,ROUTE_TOPK)
        return query,(result.note_idx.to_numpy(np.int64),result.score.to_numpy(np.float64))
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures=[pool.submit(lookup,query) for query in unique]
            for i,future in enumerate(as_completed(futures),1):
                query,values=future.result();cache[query]=values
                if i%100==0 or i==len(unique):print(json.dumps({'stage':'lexical_'+split,'unique_queries':i,'total_unique':len(unique),'seconds':time.monotonic()-start}),flush=True)
    finally:index.close()
    schema=pa.schema([('gid',pa.int64()),('note_idx',pa.int64()),('score',pa.float64()),('rank',pa.int32())])
    temporary=Path(str(output)+'.tmp');count=0
    with pq.ParquetWriter(temporary,schema) as writer:
        for start_row in range(0,len(chosen),128):
            parts=[]
            for row in chosen.iloc[start_row:start_row+128].itertuples(index=False):
                ids,scores=cache[row.query];parts.append(pa.Table.from_pydict({'gid':np.full(len(ids),int(row.gid),dtype=np.int64),
                    'note_idx':ids,'score':scores,'rank':np.arange(1,len(ids)+1,dtype=np.int32)},schema=schema));count+=len(ids)
            if parts:writer.write_table(pa.concat_tables(parts))
    temporary.replace(output)
    write_json(manifest_path,{'recipe':recipe,'rows':count,'candidate_sha256':digest(output),
        'source_snapshot':str(source_copy),'source_sha256':digest(source_copy),'seconds':time.monotonic()-start})
    print(json.dumps({'stage':'lexical_'+split,'status':'ready','rows':count,'requests':len(chosen),'path':str(output)}),flush=True)
    return output


def select_epoch(run):
    """Select exact full-catalog validation Recall100 with explicit epoch1 guards."""
    run=Path(run);requests=request_frame(run/'requests_validation.parquet');entries=[]
    training=json.loads((run/'dssm/training_manifest.json').read_text())
    if not (run/'dssm/selected.json').exists():raise RuntimeError('Training has not published its completion selection')
    for directory in sorted((run/'dssm').glob('epoch_*')):
        route=directory/'validation_ann.parquet'
        if not route.exists():continue
        metrics,_=candidate_metrics(requests,route)
        entries.append({'epoch':int(directory.name.split('_')[-1]),'directory':str(directory),
            'validation':metrics,'candidate_sha256':digest(route),
            'checkpoint_sha256':digest(directory/'dssm_search_optimized.pt'),
            'user_tower_sha256':digest(directory/'dssm_search_optimized_user_tower.pt'),
            'item_vectors_sha256':digest(directory/'item_vectors.bin')})
    entries.sort(key=lambda entry:entry['epoch'])
    if [entry['epoch'] for entry in entries]!=list(range(1,int(training['epochs'])+1)):
        raise RuntimeError('All declared training epochs must finish before model selection')
    if not entries or entries[0]['epoch']!=1:raise ValueError('Complete epoch1 validation control required')
    baseline=entries[0]['validation']
    eligible=[entry for entry in entries if all(entry['validation'][key]>=baseline[key]-1e-12 for key in GUARDS)]
    chosen=max(eligible,key=lambda entry:(entry['validation']['Recall@100'],entry['validation']['HitRate@10'],
        entry['validation']['MRR@10'],entry['validation']['known_positive_NDCG@10'],-entry['epoch']))
    result={'objective':'Recall@100','guards_vs_epoch1':list(GUARDS),'entries':entries,'selected':chosen,
            'validation_requests':len(requests),'test_used':False}
    output=run/'index/model_selection.json';output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():
        if json.loads(output.read_text())!=result:raise ValueError('Model selection already frozen with different inputs')
    else:write_json(output,result)
    return result


def ensure_plan(run):
    plan={'scene':'search','DPR':False,'route_topk':ROUTE_TOPK,'topk':TOPK,'head_cap':HEAD_CAP,
        'baseline':'protected local BM25 first100, then frozen ANN rank fill to500',
        'primary':'Recall@100','guards':list(GUARDS),'ann_weights':list(ANN_WEIGHTS),'bm25_weight':1.,'rrf_k':list(RRF_K),
        'index':{'nlist':2048,'pq_m':32,'pq_bits':8,'nprobe':[32,128],'seed':42,'fit_only_codebook':True},
        'selection':'validation only; epoch guards vs1, nprobe guards vs32, fusion guards vs protected baseline',
        'test_ablations':['local_bm25','selected_ann','protected_headfill','selected_fusion'],
        'known_positive_NDCG':'all original unique clicked positives form IDCG; absent positives receive zero credit',
        'catalog_point_in_time':'UNKNOWN','test_query_generation':'only after hash-bound selection.json exists'}
    path=Path(run)/'search_recall_plan.json'
    if path.exists():
        if json.loads(path.read_text())!=plan:raise ValueError('Search recall plan already frozen with different settings')
    else:write_json(path,plan)
    return plan


def codebook_positions(item_ids,fit_ids,max_samples=200000):
    item_ids=np.asarray(item_ids,dtype=np.int64);fit_ids=np.unique(np.asarray(fit_ids,dtype=np.int64))
    if len(item_ids)==0 or np.any(np.diff(item_ids)<=0):raise ValueError('Catalog IDs must be strictly sorted')
    positions=np.searchsorted(item_ids,fit_ids)
    if np.any(positions>=len(item_ids)) or not np.array_equal(item_ids[positions],fit_ids):raise ValueError('Fit item missing from the catalog')
    return np.random.default_rng(42).choice(positions,min(max_samples,len(positions)),replace=False)


def ann_validation(run):
    import faiss
    from optimize_rec_retrieval import retrieve_index
    ensure_plan(run);choice=select_epoch(run);selected=choice['selected'];epoch=Path(selected['directory'])
    directory=Path(run)/'index';directory.mkdir(parents=True,exist_ok=True)
    item_ids=np.load(Path(run)/'dssm/item_ids.npy')
    vectors=np.memmap(epoch/'item_vectors.bin',dtype=np.float32,mode='r',shape=(len(item_ids),128))
    requests=request_frame(Path(run)/'requests_validation.parquet')
    qids=np.load(epoch/'validation_request_ids.npy');queries=np.load(epoch/'validation_vectors.npy')
    if not np.array_equal(np.sort(qids),np.sort(requests.gid.to_numpy())) or len(np.unique(qids))!=len(requests):raise ValueError('ANN query cohort changed')
    fit_path=Path(run)/'features_fit.parquet'
    recipe={'geometry':{'nlist':2048,'m':32,'bits':8,'metric':'ip'},'seed':42,'codebook_fit_only':True,
        'item_vectors_sha256':selected['item_vectors_sha256'],'fit_features_sha256':digest(fit_path),
        'item_ids_sha256':digest(Path(run)/'dssm/item_ids.npy'),'catalog_items':len(item_ids)}
    index_path=directory/'ivfpq.faiss';manifest_path=directory/'manifest.json'
    faiss.omp_set_num_threads(2)
    if index_path.exists() or manifest_path.exists():
        if not(index_path.exists() and manifest_path.exists()):raise FileExistsError('Incomplete ANN index')
        recorded=json.loads(manifest_path.read_text())
        if recorded['recipe']!=recipe or recorded['index_sha256']!=digest(index_path):raise ValueError('Frozen ANN index/model recipe changed')
        index=faiss.read_index(str(index_path))
    else:
        fit_ids=pd.read_parquet(fit_path,columns=['note_idx']).note_idx.to_numpy(np.int64)
        positions=codebook_positions(item_ids,fit_ids)
        if len(positions)<81920:raise ValueError('Not enough fit-only vectors for the declared2048-cell codebook')
        index=faiss.IndexIVFPQ(faiss.IndexFlatIP(128),128,2048,32,8,faiss.METRIC_INNER_PRODUCT)
        start=time.monotonic();print(json.dumps({'stage':'pq_train','fit_sample':len(positions),'catalog':len(item_ids)}),flush=True)
        index.train(np.ascontiguousarray(vectors[positions]))
        index.add(vectors);index.nprobe=32
        temporary=Path(str(index_path)+'.tmp');faiss.write_index(index,str(temporary));temporary.replace(index_path)
        write_json(manifest_path,{'recipe':recipe,'index_sha256':digest(index_path),'serialized_bytes':index_path.stat().st_size,
            'fit_sample_items':len(positions),'seconds':time.monotonic()-start,'source_sha256':digest(__file__)})
    comparisons=[]
    for probe in [32,128]:
        path=directory/f'validation_nprobe{probe}.parquet'
        if not path.exists():
            index.nprobe=probe;retrieve_index(index,qids,queries,item_ids,path)
        result,per_request=candidate_metrics(requests,path)
        per_request.to_parquet(directory/f'validation_nprobe{probe}_requests.parquet',index=False)
        comparisons.append({'nprobe':probe,'path':str(path),'candidate_sha256':digest(path),'validation':result})
        print(json.dumps({'stage':'ann_validation','nprobe':probe,**result}),flush=True)
    baseline=comparisons[0]['validation']
    eligible=[entry for entry in comparisons if all(entry['validation'][key]>=baseline[key]-1e-12 for key in GUARDS)]
    chosen=max(eligible,key=lambda entry:(entry['validation']['Recall@100'],entry['validation']['HitRate@10'],
        entry['validation']['MRR@10'],entry['validation']['known_positive_NDCG@10'],-entry['nprobe']))
    flat_metrics,flat_requests=candidate_metrics(requests,epoch/'validation_ann.parquet')
    flat_requests.to_parquet(directory/'validation_flat_requests.parquet',index=False)
    result={'selected':chosen,'comparisons':comparisons,'flat_control':{'validation':flat_metrics,
        'path':str(epoch/'validation_ann.parquet'),'scope':'stored exact full-catalog IP validation route; topk boundary ties follow recorded torch protocol'},
        'model_selection':choice,'test_used':False}
    path=directory/'selected.json'
    if path.exists():
        if json.loads(path.read_text())!=result:raise ValueError('ANN nprobe selection already frozen with different inputs')
    else:write_json(path,result)
    return result


def fusion_connection(ann_path,bm25_path):
    connection=duckdb.connect(config={'threads':'2','memory_limit':'1GB'})
    connection.execute('CREATE TABLE ann AS SELECT gid,note_idx,rank,score FROM read_parquet(?)',[str(ann_path)])
    connection.execute('CREATE TABLE lexical AS SELECT gid,note_idx,rank,score FROM read_parquet(?)',[str(bm25_path)])
    return connection


def policy_candidates(connection,policy,output):
    if policy['strategy']=='protected_headfill':
        query="""WITH head AS (SELECT gid,note_idx,rank FROM lexical WHERE rank<=100),
            tail AS (SELECT a.gid,a.note_idx,a.rank FROM ann a LEFT JOIN head h USING(gid,note_idx) WHERE h.note_idx IS NULL),
            assembled AS (SELECT gid,note_idx,1 priority,rank upstream FROM head
                UNION ALL SELECT gid,note_idx,2 priority,rank upstream FROM tail),
            ordered AS (SELECT gid,note_idx,row_number() OVER(PARTITION BY gid ORDER BY priority,upstream,note_idx) rank FROM assembled)
            SELECT gid,gid search_idx,note_idx,rank::INT rank,1./rank::DOUBLE score,1./rank::DOUBLE recall_score
            FROM ordered WHERE rank<=500 ORDER BY gid,rank"""
    elif policy['strategy']=='weighted_rrf':
        weight=float(policy['weights']['ann']);k=int(policy['rrf_k'])
        if weight not in ANN_WEIGHTS or k not in RRF_K:raise ValueError('Policy is outside the declared validation grid')
        query=f"""WITH joined AS (
            SELECT coalesce(a.gid,b.gid) gid,coalesce(a.note_idx,b.note_idx) note_idx,
                coalesce({weight}/({k}+a.rank),0.)+coalesce(1./({k}+b.rank),0.) fused_score
            FROM ann a FULL JOIN lexical b USING(gid,note_idx)
            WHERE b.note_idx IS NOT NULL OR ({weight}>0 AND a.note_idx IS NOT NULL)),
            ordered AS (SELECT *,row_number() OVER(PARTITION BY gid ORDER BY fused_score DESC,note_idx) rank FROM joined)
            SELECT gid,gid search_idx,note_idx,rank::INT rank,fused_score score,fused_score recall_score
            FROM ordered WHERE rank<=500 ORDER BY gid,rank"""
    else:raise ValueError('Unknown Search fusion policy')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    temporary=Path(str(output)+'.tmp')
    connection.execute('COPY ('+query+") TO '"+str(temporary).replace("'","''")+"' (FORMAT PARQUET)")
    temporary.replace(output)
    return output


def guarded_fusion_choice(rows):
    baseline=rows[0]
    eligible=[row for row in rows if all(row['validation'][key]>=baseline['validation'][key]-1e-12 for key in GUARDS)]
    return max(eligible,key=lambda row:(row['validation']['Recall@100'],row['validation']['HitRate@10'],
        row['validation']['MRR@10'],row['validation']['known_positive_NDCG@10'],row['policy']['strategy']=='protected_headfill'))


def select_fusion(run,lexical_run):
    run,lexical_run=Path(run),Path(lexical_run);ensure_plan(run)
    if (run/'selection.json').exists():raise FileExistsError('Search recall selection is frozen')
    requests=request_frame(run/'requests_validation.parquet')
    bm25=lexical_routes(lexical_run,run/'requests_validation.parquet','validation')
    index=json.loads((run/'index/selected.json').read_text());ann=Path(index['selected']['path'])
    if digest(ann)!=index['selected']['candidate_sha256']:raise ValueError('Frozen validation ANN changed')
    target=run/'fusion_validation';target.mkdir(exist_ok=True)
    base_policy={'strategy':'protected_headfill','weights':{'ann':1.,'bm25':1.},'rrf_k':60,
        'route_topk':ROUTE_TOPK,'topk':TOPK,'head_cap':HEAD_CAP}
    policies=[base_policy]+[{'strategy':'weighted_rrf','weights':{'ann':weight,'bm25':1.},'rrf_k':k,
        'route_topk':ROUTE_TOPK,'topk':TOPK,'head_cap':HEAD_CAP} for weight in ANN_WEIGHTS for k in RRF_K]
    rows=[];connection=fusion_connection(ann,bm25)
    try:
        for i,policy in enumerate(policies):
            path=target/f'candidate_{i:02d}.parquet'
            if path.exists():raise FileExistsError('Use a fresh run rather than replacing frozen fusion validation candidates')
            policy_candidates(connection,policy,path)
            result,per_request=candidate_metrics(requests,path)
            per_request.to_parquet(target/f'candidate_{i:02d}_requests.parquet',index=False)
            row={'policy':policy,'validation':result,'path':str(path),'candidate_sha256':digest(path)};rows.append(row)
            write_json(target/'grid.json',rows)
            print(json.dumps({'stage':'fusion_validation','policy':policy,**result}),flush=True)
    finally:connection.close()
    chosen=guarded_fusion_choice(rows)
    source=Path(chosen['path']);shutil.copyfile(source,run/'validation_selected.parquet')
    snapshots=run/'source_snapshot';snapshots.mkdir(exist_ok=True)
    dependencies=[Path(__file__),ROOT/'scripts/optimize_rec_retrieval.py',ROOT/'scripts/evaluate_search_index.py',
        ROOT/'src/recall/dssm_trainer.py',ROOT/'src/recall/query_encoding.py',ROOT/'src/recall/lexical.py',ROOT/'src/training/history.py',ROOT/'src/backend/online/item_features.py']
    code={str(path):digest(path) for path in dependencies}
    for path in dependencies:shutil.copyfile(path,snapshots/(path.parent.name+'_'+path.name))
    epoch=Path(index['model_selection']['selected']['directory'])
    inputs=[run/'requests_fit.parquet',run/'requests_validation.parquet',run/'features_fit.parquet',run/'features_validation.parquet',
        run/'dssm/training_manifest.json',run/'dssm/item_ids.npy',epoch/'dssm_search_optimized_user_tower.pt',epoch/'dssm_search_optimized.pt',
        epoch/'item_vectors.bin',run/'index/ivfpq.faiss',run/'index/manifest.json',run/'index/selected.json',
        bm25,lexical_run/'lexical/validation_manifest.json',lexical_run/'bm25.sqlite.manifest.json',run/'validation_selected.parquet',
        ROOT/'datasets/search_train/train-00000-of-00001.parquet',ROOT/'datasets/search_test/train-00000-of-00001.parquet',
        ROOT/'features/search_test_features.parquet']
    for folder,name in [('note_text_emb','note_text'),('note_img_emb','note_img')]:
        inputs += [ROOT/'embeddings'/folder/f'{name}.bin',ROOT/'embeddings'/folder/f'{name}_map.json']
    inputs += sorted((ROOT/'features/vocab_dict').glob('*.pkl'))
    inputs += sorted((ROOT/'datasets/notes').glob('*.parquet'))
    from transformers import AutoConfig
    encoder_commit=AutoConfig.from_pretrained('BAAI/bge-base-zh',local_files_only=True)._commit_hash
    frozen={'query_encoder_commit':encoder_commit,'scene':'search','DPR':False,'selection_source':'all original train-internal validation requests; no officialtest metrics',
        'primary':'Recall@100','guards':list(GUARDS),'baseline':rows[0],'selected':chosen,'recall_policy':chosen['policy'],
        'index':index,'grid':rows,'code_sha256':code,'inputs_sha256':{str(path):digest(path) for path in inputs},
        'limits':['Static catalog/history/profile source timestamps incomplete','Frozen test already diagnosed historically, not a new blind holdout'],
        'test_not_run':True}
    write_json(run/'selection.json',frozen)
    print(json.dumps({'stage':'selection_frozen','policy':chosen['policy'],'validation':chosen['validation']}),flush=True)
    return frozen


def verify_frozen_selection(run):
    frozen=json.loads((Path(run)/'selection.json').read_text())
    for key in ['inputs_sha256','code_sha256']:
        for path,expected in frozen[key].items():
            if digest(path)!=expected:raise ValueError('Frozen Search input/source content changed: '+str(path))
    return frozen


def frozen_test(run,lexical_run,workers=4):
    """Compute predeclared final test ablations; never choose parameters here."""
    import faiss
    import torch
    from backend.online.item_features import get_item_feature_store
    from optimize_rec_retrieval import raw_model_frame,encode_requests,retrieve_index
    from recall import dssm_trainer as trainer
    from recall.query_encoding import build_search_query_store
    from evaluate_search_index import apply_user_profile_contract
    run,lexical_run=Path(run),Path(lexical_run)
    if (run/'test_metrics.json').exists():raise FileExistsError('Final Search test is already frozen')
    frozen=verify_frozen_selection(run);epoch=Path(frozen['index']['model_selection']['selected']['directory'])
    from transformers import AutoConfig
    revision=AutoConfig.from_pretrained('BAAI/bge-base-zh',local_files_only=True)._commit_hash
    if revision!=frozen['query_encoder_commit']:raise ValueError('Query encoder revision changed after validation selection')
    raw=pd.read_parquet(ROOT/'datasets/search_test/train-00000-of-00001.parquet',columns=['search_idx','user_idx','query','search_result_details_with_idx'])
    requests=raw[['search_idx','user_idx']].rename(columns={'search_idx':'gid'})
    requests['positives']=[sorted({int(d['note_idx']) for d in details if d['click']>0}) for details in raw.search_result_details_with_idx]
    requests.to_parquet(run/'requests_test.parquet',index=False)
    bm25=lexical_routes(lexical_run,run/'requests_test.parquet','test',workers=workers)
    checkpoint=torch.load(epoch/'dssm_search_optimized_user_tower.pt',map_location='cpu',weights_only=False)
    meta=checkpoint['meta'];training=json.loads((run/'dssm/training_manifest.json').read_text())
    model=trainer.DSSMModel('search',meta['cat_vocabs'],meta['item_dense_dim'],meta['user_dense_dim'],preprocessing=meta['numeric_preprocessing']).eval()
    missing,unexpected=model.load_state_dict(checkpoint['state_dict'],strict=False)
    if unexpected or any(name.startswith(('u_emb_','user_mlp','user_numeric')) for name in missing):raise ValueError('User tower checkpoint contract mismatch')
    query_store=build_search_query_store(raw,run/'query_cache/test',device='cpu')
    stores={'note_text':trainer.MMapStore('note_text_emb','note_text'),'note_img':trainer.MMapStore('note_img_emb','note_img'),'search_query':query_store}
    frame=raw_model_frame(pd.read_parquet(ROOT/'features/search_test_features.parquet'),get_item_feature_store(ROOT),trainer,'search')
    frame=apply_user_profile_contract(frame,meta,training)
    torch.set_num_threads(2);faiss.omp_set_num_threads(2)
    qids,queries=encode_requests(model,frame,stores,'search',int(meta.get('unknown_user_id',training['unknown_user_id'])),
        set(meta.get('known_user_ids',training['known_user_ids'])))
    if not np.array_equal(np.sort(qids),np.sort(requests.gid)):raise ValueError('Incomplete official Search test queries')
    np.save(run/'index/test_request_ids.npy',qids);np.save(run/'index/test_request_vectors.npy',queries)
    index=faiss.read_index(str(run/'index/ivfpq.faiss'));index.nprobe=int(frozen['index']['selected']['nprobe'])
    item_ids=np.load(run/'dssm/item_ids.npy');ann_path=run/'index/test_selected_ann.parquet'
    retrieve_index(index,qids,queries,item_ids,ann_path)
    connection=fusion_connection(ann_path,bm25)
    try:
        baseline_path=run/'test_protected_headfill.parquet';policy_candidates(connection,frozen['baseline']['policy'],baseline_path)
        selected_path=run/'test_selected.parquet';policy_candidates(connection,frozen['recall_policy'],selected_path)
    finally:connection.close()
    scores={};hashes={}
    for name,path in [('local_bm25',bm25),('selected_ann',ann_path),('protected_headfill',baseline_path),('selected_fusion',selected_path)]:
        result,per_request=candidate_metrics(requests,path);per_request.to_parquet(run/f'test_{name}_requests.parquet',index=False)
        scores[name]=result;hashes[str(path)]=digest(path)
    output={'scene':'search','selection_sha256':digest(run/'selection.json'),'all_requests':len(requests),'DPR':False,
        'recall_policy':frozen['recall_policy'],'selected_epoch':frozen['index']['model_selection']['selected']['epoch'],
        'nprobe':index.nprobe,'metrics':scores,'candidate_sha256':hashes,'parameter_selection_after_test':False,
        'scope':'Fixed full-catalog local BM25+fit-only DSSM routes; raw knownclicked truth, all6192 original requests, no unexposed negative labels'}
    write_json(run/'test_metrics.json',output);print(json.dumps(output,indent=2),flush=True)
    return output

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=BASE/'search')
    parser.add_argument('--lexical-run',type=Path,default=BASE/'search')
    parser.add_argument('--phase',choices=['lexical-val','epoch-val','ann-val','select','test'],required=True)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();run=args.run.resolve();lexical_run=args.lexical_run.resolve()
    ensure_plan(run)
    if args.phase=='lexical-val':lexical_routes(lexical_run,run/'requests_validation.parquet','validation',workers=args.workers)
    elif args.phase=='epoch-val':print(json.dumps(select_epoch(run),indent=2),flush=True)
    elif args.phase=='ann-val':ann_validation(run)
    elif args.phase=='select':select_fusion(run,lexical_run)
    elif args.phase=='test':frozen_test(run,lexical_run,args.workers)


if __name__=='__main__':main()