"""Replay an isolated trained stack on fixed full-catalog candidate lists; never invent negatives."""
from __future__ import annotations
import argparse,importlib.util,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import torch
import pyarrow.parquet as pq
from backend.online.item_features import get_item_feature_store
from backend.offline.evaluate_full_test import confidence,sha256
from training.dien_ranker import DIENRanker

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('rank_experiment',ROOT/'scripts/train_rank_stack.py')
experiment=importlib.util.module_from_spec(spec);spec.loader.exec_module(experiment)
spec=importlib.util.spec_from_file_location('funnel_groups',ROOT/'scripts/replay_stage_funnel.py')
funnel=importlib.util.module_from_spec(spec);spec.loader.exec_module(funnel)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--recall',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--query-cache',type=Path)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((args.run/'manifest.json').read_text());scene=manifest['scene'];key=manifest['key']
    folder='search' if scene=='search' else 'recommendation'
    nested='search_result_details_with_idx' if scene=='search' else 'rec_result_details_with_idx'
    rawpath=ROOT/f'datasets/{folder}_test/train-00000-of-00001.parquet'
    raw=pd.read_parquet(rawpath,columns=[key,'user_idx',nested])
    truth={int(row[key]):{int(d['note_idx']) for d in row[nested] if d['click']>0} for row in raw.to_dict('records')}
    users=raw.set_index(key).user_idx.to_dict()
    context=experiment.contexts(scene,'test',args.query_cache)
    static=get_item_feature_store(ROOT)
    coarse=lgb.Booster(model_file=str(args.run/'gbdt.txt'))
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');torch.set_num_threads(4)
    fine=DIENRanker(scene,manifest['dien']['numeric_features'],**manifest['dien']['architecture']).to(device)
    fine.load_state_dict(torch.load(args.run/'dien.pt',map_location=device,weights_only=True));fine.eval()
    weight=manifest['dien']['chosen']['weight'];input_cap=160 if scene=='search' else 320
    coarse_cap=80 if scene=='search' else 180
    schema=pq.read_schema(args.recall).names
    source_key=key if key in schema else 'gid'
    recall_frame=pd.read_parquet(args.recall,columns=[source_key,'note_idx','rank'])
    if key not in recall_frame and 'gid' in recall_frame:recall_frame=recall_frame.rename(columns={'gid':key})
    required={key,'note_idx','rank'}
    if not required.issubset(recall_frame):raise ValueError('Recall file missing request/item/rank')
    if not set(recall_frame[key]).issubset(truth):raise ValueError('Recall file has foreign test requests')
    recall_frame=recall_frame[recall_frame['rank']<=500].sort_values([key,'rank'],kind='stable')
    groups={int(gid):part for gid,part in recall_frame.groupby(key,sort=False)}
    rows=[];parts=[];started=time.monotonic()
    for index,(gid,positive) in enumerate(truth.items()):
        pool=groups.get(gid,pd.DataFrame(columns=[key,'note_idx','rank']))
        if pool.note_idx.duplicated().any() or len(pool)>500:raise ValueError('Invalid fixed-budget recall pool')
        candidate=pool.head(input_cap)[[key,'note_idx','rank']].copy().reset_index(drop=True)
        stages={'recall500':pool.note_idx.tolist(),'input_budget':candidate.note_idx.tolist(),'recall10':pool.note_idx.head(10).tolist()}
        if len(candidate):
            features=context.feature_frame(candidate,static)
            features['coarse_score']=experiment.tree_predict(coarse,features,manifest['features'])
            selected=features.sort_values(['coarse_score','rank'],ascending=[False,True],kind='stable').head(coarse_cap).copy()
            stages['coarse_budget']=selected.note_idx.tolist();stages['coarse10']=selected.note_idx.head(10).tolist()
            if weight:
                rawscore=experiment.score_dien(fine,selected,context,manifest['dien']['numeric_features'],device)
                selected['dien_raw_score']=rawscore
                selected['selected_score']=(1-weight)*experiment.group_minmax(selected.coarse_score,selected[key])+weight*experiment.group_minmax(rawscore,selected[key])
            else:
                selected['dien_raw_score']=np.nan;selected['selected_score']=selected.coarse_score
            ranked=selected.sort_values(['selected_score','rank'],ascending=[False,True],kind='stable')
            stages.update(fine_budget=ranked.note_idx.tolist(),fine10=ranked.note_idx.head(10).tolist(),fine20=ranked.note_idx.head(20).tolist())
            parts.append(ranked[[key,'note_idx','rank','coarse_score','dien_raw_score','selected_score']])
        else:
            stages.update({name:[] for name in ['coarse_budget','coarse10','fine_budget','fine10','fine20']})
        rows.append(funnel.stage_row(gid,int(users[gid]),positive,stages))
        if (index+1)%1000==0:print(scene,index+1,'requests',time.monotonic()-started,flush=True)
    requests=pd.DataFrame(rows);requests.to_parquet(args.output/'requests.parquet',index=False)
    if parts:pd.concat(parts,ignore_index=True).to_parquet(args.output/'scores.parquet',index=False)
    summary={'scene':scene,'requests':len(requests),'positive_pairs':int(requests.positives.sum()),
             'scope':'Fixed candidate replay of isolated experimental stack; known-positive retention only; no online CTR/AUC claim',
             'selected_dien_weight':weight,'budgets':{'recall':500,'input':input_cap,'coarse':coarse_cap},'stages':{}}
    for stage in stages:
        summary['stages'][stage]={'Recall':float(requests[stage+'_recall'].mean()),'HitRate':float(requests[stage+'_hit'].mean()),
                                 'known_positive_pairs':int(requests[stage+'_hits'].sum()),'mean_candidates':float(requests[stage+'_size'].mean()),
                                 'requests_below_10':int((requests[stage+'_size']<10).sum())}
    summary['paired_deltas']={}
    for after,before in [('coarse10','recall10'),('fine10','coarse10'),('fine10','recall10')]:
        requests['delta']=requests[after+'_recall']-requests[before+'_recall']
        summary['paired_deltas'][after+'-'+before]={'mean':float(requests.delta.mean()),'user_cluster_ci95':confidence(requests,'delta')}
    (args.output/'metrics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    inputs=[args.recall,rawpath,args.run/'manifest.json',args.run/'gbdt.txt',args.run/'dien.pt',Path(__file__)]
    (args.output/'manifest.json').write_text(json.dumps({'inputs':{str(p):sha256(p) for p in inputs},'seconds':time.monotonic()-started,
        'limits':['Provided static/history/vector snapshots lack point-in-time provenance','Frozen test previously diagnosed','No dynamic recall, business lexical ordering, postrank diversity or live feedback']},indent=2))
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()