"""Build versioned diagnostic examples from frozen full-test stage outputs."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
from backend.offline.evaluate_full_test import sha256
from backend.online.pipeline import _existing_images,_to_path_list,_format_note_title
from training.history import recent_history_ids

ROOT=Path(__file__).resolve().parents[1]


def build(scene,recall,funnel,output,example_limit=24,*,recall_only=False):
    if output.exists():raise FileExistsError('Diagnostic snapshots are immutable; choose a new output')
    key='search_idx' if scene=='search' else 'request_idx';folder='search' if scene=='search' else 'recommendation'
    nested='search_result_details_with_idx' if scene=='search' else 'rec_result_details_with_idx'
    source=ROOT/f'datasets/{folder}_test/train-00000-of-00001.parquet'
    raw=pd.read_parquet(source,columns=[key,'user_idx','query','recent_clicked_note_idxs',nested])
    metrics=json.loads((funnel/'metrics.json').read_text());accepted=bool(metrics['accepted_for_release']) and not recall_only
    if metrics.get('all_test_requests',metrics.get('all_requests'))!=len(raw) or len(raw)!=(6192 if scene=='search' else 11115):raise ValueError('Incomplete original test cohort')
    examples=raw.iloc[np.linspace(0,len(raw)-1,min(example_limit,len(raw)),dtype=int)].to_dict('records')
    wanted={int(row[key]) for row in examples}
    candidates=pd.read_parquet(recall)
    if key not in candidates:candidates=candidates.rename(columns={'gid':key})
    recall_groups={int(g):part.sort_values('rank',kind='stable') for g,part in candidates[candidates[key].isin(wanted)].groupby(key)}
    if recall_only:
        scores={}
        final=recall_groups
    else:
        scores=pd.read_parquet(funnel/'scores.parquet')
        scores={int(g):part for g,part in scores[scores[key].isin(wanted)].groupby(key)}
        final=pd.read_parquet(funnel/'final_candidates.parquet')
        final={int(g):part.sort_values('rank',kind='stable') for g,part in final[final[key].isin(wanted)].groupby(key)}
    staged=[];note_ids=set();counts=raw.groupby('user_idx').size().to_dict()
    for record in examples:
        gid=int(record[key]);positive=list(dict.fromkeys(int(d['note_idx']) for d in record[nested] if d['click']>0))
        original=recall_groups.get(gid,pd.DataFrame(columns=['note_idx'])).note_idx.astype(int).tolist()[:10]
        coarse=scores.get(gid,pd.DataFrame(columns=['note_idx','coarse_score','rank'])).sort_values(['coarse_score','rank'],ascending=[False,True],kind='stable').note_idx.astype(int).tolist()[:10]
        ranked=final.get(gid,pd.DataFrame(columns=['note_idx'])).note_idx.astype(int).tolist()[:10]
        if not accepted:coarse=ranked=original.copy()
        history=recent_history_ids(record['recent_clicked_note_idxs'],max_len=20)
        lists={'true_top10':positive[:10],'dssm_top10':original,'preranking_top10':coarse,'ranking_top10':ranked}
        for ids in lists.values():note_ids.update(ids)
        note_ids.update(history)
        staged.append((record,positive,history,lists))
    ids=pd.DataFrame({'note_idx':sorted(note_ids)})
    with duckdb.connect(config={'threads':'2','memory_limit':'2GB'}) as con:
        con.register('wanted_notes',ids)
        metadata=con.execute('SELECT n.note_idx,n.note_title,n.note_content,n.image_path,n.accum_like_num,n.accum_collect_num,n.accum_comment_num FROM read_parquet(?) n INNER JOIN wanted_notes USING(note_idx)',[str(ROOT/'datasets/notes/*.parquet')]).fetchdf()
    notes={}
    def number(value):
        try:return int(value) if np.isfinite(float(value)) else None
        except (ValueError,TypeError):return None
    for record in metadata.sort_values('note_idx',kind='stable').to_dict('records'):
        images=_existing_images(_to_path_list(record.get('image_path')))
        notes[int(record['note_idx'])]={'note_idx':int(record['note_idx']),'title':_format_note_title(record.get('note_title'),record.get('note_content')),
            'content':record['note_content'] if isinstance(record.get('note_content'),str) else '','images':images,'cover_image':images[0] if images else '',
            **{name:number(record.get(name)) for name in ['accum_like_num','accum_collect_num','accum_comment_num']}}
    result=[]
    for record,positive,history,lists in staged:
        truth={note:index+1 for index,note in enumerate(positive)};uid=int(record['user_idx'])
        entry={'request_id':int(record[key]),'query':str(record['query'] or '') if scene=='search' else '',
            'source_user_idx':uid,'is_current_user':False,'scene':scene,'scope':'frozen_full_test_examples','split':'test',
            'result_count':len(record[nested]),'stage_enabled':{'preranking':accepted,'ranking':accepted},
            'overlap':{name:len(set(lists[field])&set(positive)) for name,field in [('dssm','dssm_top10'),('preranking','preranking_top10'),('ranking','ranking_top10')]},
            'source_user':{'user_idx':uid,'request_count_in_test':int(counts[uid])},
            'source_recent_behaviors':[{'note_idx':int(note),'title':notes.get(int(note),{}).get('title',f'note {note}'),'scene':'unknown','ts':None,'request_id':None,'query':'','action':'provided_history'} for note in history]}
        for field,values in lists.items():
            entry[field+'_ids']=values
            entry[field]=[{'note_idx':note,'title':notes.get(note,{}).get('title',f'note {note}'),'rank_in_true':truth.get(note)} for note in values]
        result.append(entry)
    payload={'scene':scene,'tag':'hard','split':'test','scope':'frozen_full_test_examples','sampled_groups':len(raw),'effective_groups':len(raw),
        'sampling':'Uniformly spaced original request rows; examples never selected by hits; displayed examples are not the metric denominator',
        'examples':result,'notes':list(notes.values()),'accepted_ranking_for_release':accepted,
        'input_hashes':{str(path):sha256(path) for path in ([source,recall,funnel/'metrics.json',Path(__file__)] + ([] if recall_only else [funnel/'scores.parquet',funnel/'final_candidates.parquet']))}}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(payload,ensure_ascii=False,allow_nan=False,indent=2)+'\n')
    print(json.dumps({'snapshot':str(output),'scene':scene,'examples':len(result),'notes':len(notes),'sha256':sha256(output)}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--scene',choices=['search','rec'],required=True)
    for name in ['recall','funnel','output']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--recall-only',action='store_true',help='Use actual full recall order and mark learned stages disabled')
    parser.add_argument('--examples',type=int,default=24);args=parser.parse_args()
    if not 1<=args.examples<=40:raise ValueError('Examples must be between1and40')
    build(args.scene,args.recall,args.funnel,args.output,args.examples,recall_only=args.recall_only)