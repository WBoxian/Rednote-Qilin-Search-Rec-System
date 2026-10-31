"""Select a versioned ranking blend on saved training-internal validation scores."""
from __future__ import annotations
import argparse,importlib.util,json
from pathlib import Path
import pandas as pd
from training.scoring import blend_rank_scores
from training.selection import choose_rank_blend
from backend.offline.evaluate_full_test import sha256

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('rank_validation',ROOT/'scripts/train_rank_stack.py')
training=importlib.util.module_from_spec(spec);spec.loader.exec_module(training)


def select(run):
    run=Path(run).resolve()
    if (run/'test_metrics.json').exists():raise ValueError('Blend selection must precede official test evaluation')
    path=run/'manifest.json';manifest=json.loads(path.read_text());dien=manifest['dien']
    original=sha256(path)
    if sha256(run/'dien.pt')!=dien['model_sha256']:raise ValueError('Neural weights changed')
    frame=pd.read_parquet(run/'validation.parquet')
    if not {'coarse_score','dien_raw_score',manifest['key']}.issubset(frame):raise ValueError('Missing validation scores')
    key=manifest['key'];folder='search' if manifest['scene']=='search' else 'recommendation'
    original=pd.read_parquet(ROOT/f'datasets/{folder}_train/train-00000-of-00001.parquet',columns=[key])
    fit=pd.read_parquet(run/'dien_fit_oof.parquet',columns=[key])
    if not set(frame[key]).issubset(set(original[key])) or set(frame[key])&set(fit[key]):raise ValueError('Validation is not held-out original training requests')
    if frame.event_timestamp.max()>=manifest['official_test_first_event']:raise ValueError('Validation crosses official test cutoff')
    choices=[]
    for weight in [0.,.1,.25,.5,1.]:
        score=blend_rank_scores(frame.coarse_score,frame.dien_raw_score,frame[manifest['key']],weight,'probability_v2')
        metrics,_,_=training.metrics(frame,score,manifest['key'])
        choices.append({'weight':weight,'validation':metrics})
    selected=choose_rank_blend(choices)
    if dien.get('score_contract','request_minmax_v1')!='probability_v2':
        dien['legacy_blend_ablation']={'contract':'request_minmax_v1','candidates':dien['blend_candidates'],'chosen':dien['chosen']}
    dien.update(score_contract='probability_v2',blend_candidates=choices,chosen=selected)
    dien['post_training_validation_selection']={'selection_split':'train_internal_validation','gradient_steps':0,'original_manifest_sha256':original,
        'validation_sha256':sha256(run/'validation.parquet'),'script_sha256':sha256(Path(__file__)),'scoring_sha256':sha256(ROOT/'src/training/scoring.py')}
    path.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'run':str(run),'score_contract':'probability_v2','chosen':selected}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    select(parser.parse_args().run)