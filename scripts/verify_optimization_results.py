"""Independently recompute full-test ranking and retrieval metrics from saved predictions."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score,roc_auc_score

ROOT=Path(__file__).resolve().parents[1]


def verify_ranking(run):
    run=run.resolve();manifest=json.loads((run/'manifest.json').read_text());scene=manifest['scene']
    folder='search' if scene=='search' else 'recommendation';key=manifest['key']
    nested='search_result_details_with_idx' if scene=='search' else 'rec_result_details_with_idx'
    recorded=json.loads((run/'test_metrics.json').read_text());result={}
    for stage in ('gbdt','dien_raw','selected'):
        frame=pd.read_parquet(run/f'test_{stage}_predictions.parquet')
        with duckdb.connect() as con:
            con.register('pred',frame)
            mismatch=con.execute(f'''WITH raw AS (
                SELECT r.{key} gid,r.user_idx,d.note_idx,d.click positive,count(*) n
                FROM read_parquet(?) r,unnest(r.{nested}) u(d) GROUP BY ALL
            ), actual AS (SELECT gid,user_idx,note_idx,positive,count(*) n FROM pred GROUP BY ALL)
            SELECT count(*) FROM raw FULL JOIN actual USING(gid,user_idx,note_idx,positive)
            WHERE raw.n IS DISTINCT FROM actual.n''',[str(ROOT/f'datasets/{folder}_test/*.parquet')]).fetchone()[0]
        if mismatch:raise ValueError('Predictions differ from original raw-test impression/label multiset')
        values=[]
        for _,group in frame.groupby('gid',sort=False):
            if group.positive.sum()>0:
                values.append(float(ndcg_score(group.positive.to_numpy()[None,:],group.score.to_numpy()[None,:],k=10)) if len(group)>1 else 1.)
        eligible=[]
        for _,group in frame.groupby('user_idx',sort=False):
            if group.positive.nunique()==2:eligible.append((roc_auc_score(group.positive,group.score),len(group)))
        computed={'NDCG@10':float(np.mean(values)),'AUC':float(roc_auc_score(frame.positive,frame.score)),
                  'GAUC':float(np.average([value for value,_ in eligible],weights=[count for _,count in eligible]))}
        for metric,value in computed.items():
            if not np.isclose(value,recorded[stage][metric],rtol=0,atol=1e-9):raise ValueError(f'{stage} {metric} mismatch')
        result[stage]={'raw_label_mismatches':int(mismatch),'rows':len(frame),'requests':int(frame.gid.nunique()),**computed}
    return result


def verify_recall(scene, candidates):
    folder='search' if scene=='search' else 'recommendation';key='search_idx' if scene=='search' else 'request_idx'
    nested='search_result_details_with_idx' if scene=='search' else 'rec_result_details_with_idx'
    frame=pd.read_parquet(candidates)
    if 'gid' not in frame:frame=frame.rename(columns={key:'gid'})
    with duckdb.connect() as con:
        con.register('candidates',frame)
        duplicate=con.execute('SELECT count(*) FROM (SELECT gid,note_idx,count(*) n FROM candidates GROUP BY ALL HAVING n>1)').fetchone()[0]
        if duplicate:raise ValueError('Duplicate candidate request-item pairs')
        raw=con.execute(f'''SELECT r.{key} gid,r.user_idx,list_distinct(list(d.note_idx) FILTER(WHERE d.click>0)) positives
            FROM read_parquet(?) r,unnest(r.{nested}) u(d) GROUP BY r.{key},r.user_idx ORDER BY gid''',
            [str(ROOT/f'datasets/{folder}_test/*.parquet')]).fetchdf()
    groups={int(gid):part.sort_values('rank',kind='stable') for gid,part in frame.groupby('gid',sort=False)}
    if not set(groups).issubset(set(raw.gid)):raise ValueError('Foreign test requests in retrieval output')
    rows=[]
    for record in raw.itertuples(index=False):
        positive=set(record.positives) if record.positives is not None else set()
        group=groups.get(int(record.gid),pd.DataFrame(columns=['note_idx','rank']))
        chosen=group[group['rank']<=500]
        if len(chosen)>500:raise ValueError('Candidate budget exceeded')
        ranks=[int(rank) for note,rank in zip(chosen.note_idx,chosen['rank']) if int(note) in positive]
        first=min(ranks) if ranks else None
        rows.append({'gid':record.gid,'user_idx':record.user_idx,'positive_pairs':len(positive),
                     'hits':len(ranks),'recall':len(ranks)/len(positive) if positive else np.nan,
                     'hit':float(bool(ranks)),'mrr100':1/first if first is not None and first<=100 else 0.,'first_hit500':first})
    requests=pd.DataFrame(rows);valid=requests[requests.positive_pairs>0]
    return {'all_requests':len(requests),'positive_requests':len(valid),'positive_pairs':int(valid.positive_pairs.sum()),
            'HitRate@500':float(valid.hit.mean()),'Recall@500':float(valid.recall.mean()),'MRR@100':float(valid.mrr100.mean()),
            'MedianFirstHitRank_among_hits_at500':float(valid.first_hit500.median()) if valid.first_hit500.notna().any() else None,
            'median_scope':'Only requests with a hit in top500; misses not included; also report hit rate',
            'positive_hits':int(valid.hits.sum())}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--recall',type=Path,required=True);args=parser.parse_args()
    manifest=json.loads((args.run/'manifest.json').read_text())
    result={'status':'PASS','scene':manifest['scene'],'ranking':verify_ranking(args.run),
            'recall':verify_recall(manifest['scene'],args.recall)}
    (args.run/'root_independent_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()