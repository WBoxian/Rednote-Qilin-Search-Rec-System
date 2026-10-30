"""Audit that every fitted ranking row originates in train and precedes validation/test."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import duckdb
import pandas as pd
from backend.offline.evaluate_full_test import sha256

ROOT=Path(__file__).resolve().parents[1]
FORBIDDEN={'click','y_multi','position','pos_bucket','event_timestamp','user_idx','note_idx',
           'accum_like_num','accum_collect_num','accum_comment_num'}


def audit(path):
    path=Path(path).resolve()
    manifest=json.loads((path/'manifest.json').read_text());scene=manifest['scene'];key=manifest['key']
    folder='search' if scene=='search' else 'recommendation'
    nested='search_result_details_with_idx' if scene=='search' else 'rec_result_details_with_idx'
    source=ROOT/f'features/{scene}_train_features.parquet'
    target=ROOT/f'features/{scene}_test_features.parquet'
    raw=ROOT/f'datasets/{folder}_train/train-00000-of-00001.parquet'
    prepared=pd.read_parquet(path/'train.parquet')
    if sha256(path/'train.parquet')!=manifest['training_artifact_sha256']:raise ValueError('Prepared train hash mismatch')
    with duckdb.connect(config={'threads':'2','memory_limit':'2GB'}) as con:
        con.register('prepared',prepared)
        mismatch=con.execute(f'''WITH raw AS (
            SELECT r.{key} gid,d.note_idx,d.click,count(*) n
            FROM read_parquet(?) r,unnest(r.{nested}) u(d) GROUP BY r.{key},d.note_idx,d.click
        ), feat AS (SELECT {key} gid,note_idx,click,count(*) n FROM read_parquet(?) GROUP BY {key},note_idx,click)
        SELECT count(*) FROM raw FULL JOIN feat USING(gid,note_idx,click) WHERE raw.n IS DISTINCT FROM feat.n''',
        [str(raw),str(source)]).fetchone()[0]
        if mismatch:raise ValueError('Raw training labels/exposures differ from train features')
        # Every retained row must be a true multiset subset of official train.
        extra=con.execute(f'''WITH src AS (
            SELECT {key},note_idx,click,event_timestamp,count(*) n FROM read_parquet(?) GROUP BY ALL
        ), used AS (SELECT {key},note_idx,click,event_timestamp,count(*) n FROM prepared GROUP BY ALL)
        SELECT count(*) FROM used LEFT JOIN src USING({key},note_idx,click,event_timestamp)
        WHERE src.n IS NULL OR used.n>src.n''',[str(source)]).fetchone()[0]
        overlap=con.execute(f'''SELECT count(DISTINCT p.{key}) FROM prepared p JOIN read_parquet(?) t USING({key})''',[str(target)]).fetchone()[0]
    if extra or overlap:raise ValueError('Prepared training rows overlap test or do not originate in train')
    bad=set(manifest['features'])&FORBIDDEN
    if bad:raise ValueError(f'Outcome/position/ID fields in model features: {bad}')
    fit=prepared[prepared.partition=='fit'];val=prepared[prepared.partition=='validation']
    if set(fit[key])&set(val[key]):raise ValueError('Fit/validation request overlap')
    if not fit.event_timestamp.max()<val.event_timestamp.min()<=val.event_timestamp.max()<manifest['official_test_first_event']:
        raise ValueError('Temporal boundary failure')
    oof=pd.read_parquet(path/'dien_fit_oof.parquet',columns=[key,'note_idx','coarse_score'])
    if not set(oof[key]).issubset(set(fit[key])):raise ValueError('OOF consumes validation/test requests')
    for block in manifest['trees']['forward_oof_blocks']:
        if not block['fit_last_event']<block['prediction_first_event']:
            raise ValueError('OOF temporal leakage')
    files=[source,target,raw,path/'train.parquet',path/'dien_fit_oof.parquet',path/'validation.parquet',path/'gbdt.txt',Path(__file__)]
    result={'status':'PASS','scene':scene,'raw_training_label_multiset_mismatches':int(mismatch),
            'prepared_rows_outside_train':int(extra),'prepared_test_request_overlap':int(overlap),
            'fit_rows':len(fit),'validation_rows':len(val),'oof_training_rows':len(oof),
            'features':manifest['features'],'forward_oof_blocks':manifest['trees']['forward_oof_blocks'],
            'checks':'Original train provenance, request separation, strict chronology, OOF direction, forbidden features',
            'limitations':manifest['limitations'],
            'inputs':{str(p.relative_to(ROOT)):sha256(p) for p in files}}
    (path/'training_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();result=audit(args.run);print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()