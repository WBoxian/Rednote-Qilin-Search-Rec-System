"""Verify fixed-candidate funnel statistics against raw positives and stored selected item scores."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from backend.offline.evaluate_full_test import sha256

ROOT=Path(__file__).resolve().parents[1]


def verify(run,recall):
    run=run.resolve();recall=recall.resolve()
    summary=json.loads((run/'metrics.json').read_text());scene=summary['scene']
    key='search_idx' if scene=='search' else 'request_idx'
    folder='search' if scene=='search' else 'recommendation'
    nested='search_result_details_with_idx' if scene=='search' else 'rec_result_details_with_idx'
    source=ROOT/f'datasets/{folder}_test/train-00000-of-00001.parquet'
    raw=pd.read_parquet(source,columns=[key,'user_idx',nested])
    truth={int(row[key]):{int(d['note_idx']) for d in row[nested] if d['click']>0} for row in raw.to_dict('records')}
    users=raw.set_index(key).user_idx.to_dict()
    columns=pq.read_schema(recall).names;source_key=key if key in columns else 'gid'
    candidates=pd.read_parquet(recall,columns=[source_key,'note_idx','rank']).rename(columns={source_key:key})
    candidates=candidates[candidates['rank']<=500].sort_values([key,'rank'],kind='stable')
    pools={int(gid):group for gid,group in candidates.groupby(key,sort=False)}
    scores=pd.read_parquet(run/'scores.parquet')
    if not np.isfinite(scores[['coarse_score','selected_score']].to_numpy()).all():raise ValueError('Nonfinite ranking score')
    selected={int(gid):group for gid,group in scores.groupby(key,sort=False)}
    saved=pd.read_parquet(run/'requests.parquet').set_index('gid')
    if not saved.index.is_unique or set(saved.index)!=set(truth):raise ValueError('Funnel does not cover every original test request once')
    if not set(pools).issubset(truth) or not set(selected).issubset(truth):raise ValueError('Foreign funnel request IDs')
    rows=[];input_cap=summary['budgets']['input'];coarse_cap=summary['budgets']['coarse']
    for gid,positive in truth.items():
        pool=pools.get(gid,pd.DataFrame(columns=[key,'note_idx','rank']))
        scored=selected.get(gid,pd.DataFrame(columns=scores.columns))
        if pool.note_idx.duplicated().any() or scored.note_idx.duplicated().any():raise ValueError('Duplicate candidate items')
        input_ids=pool.note_idx.head(input_cap).tolist()
        if len(scored)>coarse_cap or not set(scored.note_idx).issubset(input_ids):raise ValueError('Scored candidates violate fixed budgets/subset')
        coarse=scored.sort_values(['coarse_score','rank'],ascending=[False,True],kind='stable')
        fine=scored.sort_values(['selected_score','rank'],ascending=[False,True],kind='stable')
        stages={'recall500':pool.note_idx.tolist(),'input_budget':input_ids,'recall10':pool.note_idx.head(10).tolist(),
                'coarse_budget':coarse.note_idx.tolist(),'coarse10':coarse.note_idx.head(10).tolist(),
                'fine_budget':fine.note_idx.tolist(),'fine10':fine.note_idx.head(10).tolist(),'fine20':fine.note_idx.head(20).tolist()}
        if int(saved.loc[gid,'user_idx'])!=int(users[gid]) or int(saved.loc[gid,'positives'])!=len(positive):raise ValueError('Raw user/positive denominator mismatch')
        row={'gid':gid,'user_idx':int(users[gid]),'positives':len(positive)}
        for stage,ids in stages.items():
            hits=len(set(ids)&positive);size=len(ids);value=hits/len(positive) if positive else np.nan
            if int(saved.loc[gid,stage+'_hits'])!=hits or int(saved.loc[gid,stage+'_size'])!=size:
                raise ValueError(f'{stage} saved per-request counts mismatch')
            if not np.isclose(saved.loc[gid,stage+'_recall'],value,atol=1e-12,rtol=0,equal_nan=True):raise ValueError('Per-request recall mismatch')
            row[stage+'_hits']=hits;row[stage+'_recall']=value;row[stage+'_size']=size
        rows.append(row)
    frame=pd.DataFrame(rows)
    for stage in summary['stages']:
        reported=summary['stages'][stage]
        if not np.isclose((frame[stage+'_hits']>0).mean(),reported['HitRate'],atol=1e-12,rtol=0):raise ValueError('Hit rate mismatch')
        if not np.isclose(frame[stage+'_size'].mean(),reported['mean_candidates'],atol=1e-12,rtol=0):raise ValueError('Mean candidate count mismatch')
        if not np.isclose(frame[stage+'_recall'].mean(),reported['Recall'],atol=1e-12,rtol=0):raise ValueError('Macro recall mismatch')
        if int(frame[stage+'_hits'].sum())!=reported['known_positive_pairs']:raise ValueError('Positive pair count mismatch')
        if int((frame[stage+'_size']<10).sum())!=reported['requests_below_10']:raise ValueError('Candidate shortage mismatch')
    result={'status':'PASS','scene':scene,'requests':len(frame),'positive_pairs':int(frame.positives.sum()),
            'scope':'Independent raw-positive/candidate/score-table reconstruction; not a fresh neural inference',
            'checks':'All official requests, unique items, budgets, nested candidate sets, Top10/20 ranking, positive counts, macro recall, shortage counts',
            'inputs':{str(p):sha256(p) for p in [source,recall,run/'scores.parquet',run/'requests.parquet',run/'metrics.json',Path(__file__)]}}
    (run/'independent_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--recall',type=Path,required=True);args=parser.parse_args()
    result=verify(args.run,args.recall);print(json.dumps({k:v for k,v in result.items() if k!='inputs'}))

if __name__=='__main__':main()