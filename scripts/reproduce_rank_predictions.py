"""Repeat canonical full-test inference and compare every saved score and request rank."""
import argparse,importlib.util,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
from backend.online.item_features import get_item_feature_store
from backend.offline.evaluate_full_test import sha256
from training.dien_ranker import DIENRanker

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('rank_training',ROOT/'scripts/train_rank_stack.py')
training=importlib.util.module_from_spec(spec);spec.loader.exec_module(training)


def verify_input_files(root, records):
    """Content identifies data across machines; touching/copying a file is harmless."""
    for name, record in records.items():
        path = Path(root) / name
        if not path.is_file() or path.stat().st_size != record['bytes'] or sha256(path) != record['sha256']:
            raise ValueError(f'Frozen data content changed: {name}')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--query-cache',type=Path);parser.add_argument('--input-manifest',type=Path,required=True)
    args=parser.parse_args();run=args.run.resolve();manifest=json.loads((run/'manifest.json').read_text())
    frozen=json.loads(args.input_manifest.read_text());started=time.monotonic()
    verify_input_files(ROOT, frozen['inputs'])

    scene=manifest['scene'];key=manifest['key']
    frame=pd.read_parquet(Path(manifest.get('feature_dir',ROOT/'features'))/f'{scene}_test_features.parquet',columns=[key,'user_idx','note_idx','event_timestamp','click'])
    context=training.contexts(scene,'test',args.query_cache)
    frame=context.feature_frame(frame,get_item_feature_store(ROOT))
    coarse=lgb.Booster(model_file=str(run/'gbdt.txt'))
    base=training.tree_predict(coarse,frame,manifest['features']);frame['coarse_score']=base
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');torch.set_num_threads(4)
    fine=DIENRanker(scene,manifest['dien']['numeric_features'],**manifest['dien']['architecture']).to(device)
    fine.load_state_dict(torch.load(run/'dien.pt',map_location=device,weights_only=True))
    raw=training.score_dien(fine,frame,context,manifest['dien']['numeric_features'],device)
    weight=manifest['dien']['chosen']['weight'];groups=frame[key].to_numpy()
    combined=(1-weight)*training.group_minmax(base,groups)+weight*training.group_minmax(raw,groups)
    result={'status':'PASS','scene':scene,'rows':len(frame),'requests':int(frame[key].nunique()),
            'data_manifest_sha256':sha256(args.input_manifest),'model_sha256':{'gbdt':sha256(run/'gbdt.txt'),'dien':sha256(run/'dien.pt')},
            'inference_precision':'canonical batch1024; '+str(manifest['dien']['architecture'].get('inference_precision','bf16'))+', FP32 score storage',
            'comparison':{},'limitations':['Inference reproducibility does not prove point-in-time availability of provider history/catalog snapshots']}
    for stage,score in [('gbdt',base),('dien_raw',raw),('selected',combined)]:
        saved=pd.read_parquet(run/f'test_{stage}_predictions.parquet')
        if not np.array_equal(frame[key].to_numpy(),saved.gid.to_numpy()) or not np.array_equal(frame.note_idx.to_numpy(),saved.note_idx.to_numpy()):
            raise ValueError('Canonical original exposure row order changed')
        delta=np.abs(score.astype(float)-saved.score.to_numpy())
        ranks_new=pd.Series(score).groupby(groups,sort=False).rank(method='min',ascending=False).to_numpy()
        ranks_old=saved.score.groupby(saved.gid,sort=False).rank(method='min',ascending=False).to_numpy()
        changed=int((ranks_new!=ranks_old).sum())
        if delta.max()>1e-6 or changed:raise ValueError(f'{stage} reproducibility failed: max_abs={delta.max()}, rank_rows={changed}')
        result['comparison'][stage]={'max_absolute_score_difference':float(delta.max()),'changed_rank_rows':changed,'bitwise_same_scores':bool(np.array_equal(score,saved.score.to_numpy()))}
    result['seconds']=time.monotonic()-started
    (run/'full_inference_reproduction.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()