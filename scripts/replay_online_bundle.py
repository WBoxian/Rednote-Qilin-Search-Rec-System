"""Replay actual current-context service calls against an unpromoted frozen bundle."""
from __future__ import annotations
import argparse,json,os,time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from backend.offline.evaluate_full_test import sha256
from backend.online.bundle import validate_manifest,ServingBundle
from backend.online.ranking.stack import RankStackPredictor
from backend.online.pipeline import SceneServingState,OnlineScenePipeline
from training.scoring import blend_rank_scores

ROOT=Path(__file__).resolve().parents[1]


def retention(ids,positive):
    ranks=[i+1 for i,note in enumerate(ids) if note in positive]
    denom=sum(1/np.log2(i+2) for i in range(min(10,len(positive))))
    return {'HitRate@10':float(any(r<=10 for r in ranks)), 'HitRate@50':float(any(r<=50 for r in ranks)),
        'Recall@100':sum(r<=100 for r in ranks)/len(positive) if positive else 0.,
        'known_positive_NDCG@10':sum(1/np.log2(r+1) for r in ranks if r<=10)/denom if denom else 0.,
        'MRR@10':1/min(ranks) if ranks and min(ranks)<=10 else 0.,
        'MRR@100':1/min(ranks) if ranks and min(ranks)<=100 else 0.}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference-scores',type=Path)
    parser.add_argument('--reference-recall',type=Path)
    parser.add_argument('--page-size',type=int,default=30)
    parser.add_argument('--max-requests',type=int,default=0,help='0=complete original test; otherwise fixed uniformly spaced regression smoke')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    os.environ['QILIN_REDIS_URL']='';torch.set_num_threads(4)
    manifest=json.loads(args.bundle.read_text());scene=manifest['scene'];key='search_idx' if scene=='search' else 'request_idx'
    folder='search' if scene=='search' else 'recommendation'
    nested='search_result_details_with_idx' if scene=='search' else 'rec_result_details_with_idx'
    source=ROOT/f'datasets/{folder}_test/train-00000-of-00001.parquet'
    raw=pd.read_parquet(source)
    complete=len(raw)
    if args.max_requests:
        raw=raw.iloc[np.linspace(0,len(raw)-1,min(args.max_requests,len(raw)),dtype=int)]
    reference={}
    rank_meta=json.loads((ROOT/manifest['rank_dir']/'manifest.json').read_text())
    if args.reference_scores:
        saved=pd.read_parquet(args.reference_scores)
        for gid,part in saved.groupby(key,sort=False):
            chosen=part.sort_values(['coarse_score','rank'],ascending=[False,True],kind='stable').head(manifest['coarse_cap']).copy()
            chosen['final_score']=blend_rank_scores(chosen.coarse_score,chosen.dien_raw_score,chosen[key],rank_meta['dien']['chosen']['weight'],rank_meta['dien'].get('score_contract','request_minmax_v1'))
            if rank_meta['dien'].get('rank_prior'):
                from training.recall_prior import blend_recall_prior
                prior=rank_meta['dien']['rank_prior']
                chosen['final_score']=blend_recall_prior(chosen['rank'].to_numpy(),chosen['final_score'].to_numpy(),prior['alpha'],prior.get('k',60))
            reference[int(gid)]=chosen.sort_values(['final_score','rank'],ascending=[False,True],kind='stable')
    if args.reference_recall:
        saved=pd.read_parquet(args.reference_recall)
        if key not in saved:saved=saved.rename(columns={'gid':key})
        reference={int(gid):part.sort_values('rank',kind='stable') for gid,part in saved.groupby(key,sort=False)}
    started=time.monotonic();validate_manifest(manifest,scene,ROOT,tag=manifest['tag'])
    ranker=RankStackPredictor(scene,ROOT/manifest['rank_dir'],root=ROOT,
        query_encoder_dir=ROOT/manifest['query_encoder_dir'] if scene=='search' else None,query_encoder_revision=manifest.get('query_encoder_revision'))
    bundle=ServingBundle(manifest,ranker,ROOT)
    # Override only the active pointer: no disk promotion and all production code paths remain active.
    with patch('backend.online.bundle.get_active_bundle',return_value=bundle):
        state=SceneServingState(scene,tag=manifest['tag'])
    if not state.bundle_active:raise RuntimeError(state.bundle_errors)
    pipeline=OnlineScenePipeline(state)
    startup=time.monotonic()-started;rows=[];predictions=[];mismatches=[]
    for index,record in enumerate(raw.to_dict('records')):
        gid=int(record[key]);uid=int(record['user_idx']);query=str(record.get('query') or '').strip() if scene=='search' else ''
        context=pd.DataFrame([{key:gid,'user_idx':uid,'query':query,'recent_clicked_note_idxs':list(record['recent_clicked_note_idxs'])}])
        response=pipeline.build_feed(uid,query,1,args.page_size,record_exposure=False,request_context=context)
        if response['degraded_reason'] or response['route_status'].get('errors'):raise RuntimeError(response)
        cards=response['items']
        cached=pipeline._get_cached_feed(uid,int(response['context_id'][:15],16),query)
        if cached is None:raise RuntimeError('Production inference pool was not cached')
        pool=cached[0];ids=pool.note_idx.astype(int).tolist()
        if [int(card['note_idx']) for card in cards]!=ids[:args.page_size]:raise ValueError('Delivered cards changed model order')
        if len(ids)!=len(set(ids)):raise ValueError('Duplicate service candidates')
        positive={int(d['note_idx']) for d in record[nested] if d['click']>0}
        row={'gid':gid,'user_idx':uid,'positives':len(positive),'candidates':len(ids),'latency_ms':response['latency_ms'],'cache_hit':response['cache_hit'],**retention(ids,positive)}
        if reference:
            expected=reference.get(gid)
            expected_ids=[] if expected is None else expected.note_idx.astype(int).tolist()
            row['top10_order_match']=ids[:10]==expected_ids[:10]
            row['top100_set_match']=set(ids[:100])==set(expected_ids[:100])
            if not row['top10_order_match'] or not row['top100_set_match']:
                mismatches.append({'gid':gid,'actual10':ids[:10],'expected10':expected_ids[:10]})
        rows.append(row)
        predictions.extend({'gid':gid,'note_idx':note,'rank':rank+1,'score':float(pool.iloc[rank]['final_score'])} for rank,note in enumerate(ids))
        if (index+1)%100==0:print(scene,index+1,'seconds',round(time.monotonic()-started,2),flush=True)
    requests=pd.DataFrame(rows);requests.to_parquet(args.output/'requests.parquet',index=False)
    pd.DataFrame(predictions).to_parquet(args.output/'predictions.parquet',index=False)
    result={'status':'FAIL' if mismatches else 'PASS','scene':scene,'requests':len(rows),'original_test_requests':complete,'complete_test':len(rows)==complete,
        'scope':'Actual production build_feed with original current user/query/history, fresh dynamic recall and no exposure writes; active pointer injected in-memory before promotion',
        'bundle_id':bundle.bundle_id,'startup_seconds':startup,'metrics':{name:float(requests[name].mean()) for name in retention([],set())},
        'latency_ms':{'p50':float(requests.latency_ms.quantile(.5)),'p95':float(requests.latency_ms.quantile(.95)),'max':float(requests.latency_ms.max())},
        'latency_scope':'Local sequential replay including metadata/card construction; shared models progressively warm; not concurrent production SLO',
        'page_size':args.page_size,'cache_hits':int(requests.cache_hit.sum()),'reference_order_mismatches':mismatches,
        'inputs':{str(path):sha256(path) for path in [source,args.bundle,Path(__file__),*([args.reference_scores] if args.reference_scores else []),*([args.reference_recall] if args.reference_recall else [])]},'seconds':time.monotonic()-started,
        'limitations':['Provider histories/catalog/static snapshots lack complete point-in-time lineage','Test has been observed before; regression benchmark only','No unexposed candidates treated as negatives; no AUC claim for this output']}
    (args.output/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['inputs','reference_order_mismatches']}),flush=True)
    if mismatches:raise SystemExit(1)


if __name__=='__main__':main()