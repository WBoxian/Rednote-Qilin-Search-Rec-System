"""Bounded Search DIEN objective comparison on the frozen forward-OOF teacher."""
from __future__ import annotations
import argparse,json,subprocess,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from training.dien_ranker import DIENRanker
from train_rank_stack import contexts,request_batches,score_dien,metrics,group_minmax,digest,write_json
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/evaluation/optimization_20261031/search_rank'
DEFAULT=ROOT/'outputs/evaluation/optimization_20261031/search_rank_balanced_dien'
CACHE=ROOT/'outputs/evaluation/optimization_20261031/search/query_cache/train'


from training.losses import balanced_losses

def train(output):
    output.mkdir(parents=True,exist_ok=True)
    if (output/'selection.json').exists():raise RuntimeError('Balanced DIEN selection already frozen')
    used=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()[0])
    if used>=10240:raise RuntimeError(f'GPU has {used} MiB allocated; defer training until below 10 GiB')
    torch.set_num_threads(3)
    original=json.loads((BASE/'manifest.json').read_text())
    fit=pd.read_parquet(BASE/'dien_fit_oof.parquet');val=pd.read_parquet(BASE/'validation.parquet')
    if fit.event_timestamp.max()>=val.event_timestamp.min():raise ValueError('OOF/validation time boundary overlap')
    numeric=[*original['features'],'coarse_score'];architecture=original['dien']['architecture']
    context=contexts('search','train',CACHE);device=torch.device('cuda')
    stats={name:{'mean':float(fit[name].mean()),'std':float(fit[name].std(ddof=0))} for name in numeric}
    manifest={'scene':'search','seed':42,'numeric_features':numeric,'architecture':architecture,
      'grid':{'lambdas':[.25,1.],'max_epochs':4,'patience':2,'blend_weights':[0.,.1,.25,.5,1.]},
      'objective':'request-macro BCE + lambda * request-macro top10 LambdaRank with detached normalized delta-NDCG weights + .01 aux',
      'auxiliary_contract':'v2: same-request observed unclicked, all-click/no-valid-negative auxiliary rows masked',
      'attribution':'Objective and aux v2 change jointly; not standalone causal attribution',
      'teacher':'Original 13-feature GBDT strict forward OOF; no lexical-model teacher substitution',
      'fit_rows':len(fit),'validation_rows':len(val),'fit_requests':int(fit.search_idx.nunique()),'validation_requests':int(val.search_idx.nunique()),
      'normalization_fit':'OOF fit rows only','fit_last_event':float(fit.event_timestamp.max()),'validation_first_event':float(val.event_timestamp.min()),
      'inputs':{str(path.relative_to(ROOT)):digest(path) for path in [BASE/'dien_fit_oof.parquet',BASE/'validation.parquet',BASE/'gbdt.txt',BASE/'manifest.json',Path(__file__),ROOT/'src/training/dien_ranker.py']}}
    write_json(output/'manifest.json',manifest)
    choices=[]
    for coefficient in (.25,1.):
        torch.manual_seed(42);torch.cuda.manual_seed_all(42);np.random.seed(42)
        target=output/f'lambda_{coefficient:g}';target.mkdir(exist_ok=True)
        model=DIENRanker('search',numeric,**architecture).to(device);model.set_zscore_stats(stats)
        optimizer=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-4)
        best=-np.inf;stale=0;history=[];start=time.monotonic()
        for epoch in range(4):
            model.train();losses=[]
            for batch in request_batches(fit,'search_idx',512,True,42+epoch):
                features=context.batch(batch,numeric,device)
                labels=torch.as_tensor(batch.click.to_numpy(np.float32),device=device)
                groups=torch.as_tensor(batch.search_idx.to_numpy(np.int64),device=device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                    scores,aux=model(features,labels,groups)
                    bce,rank=balanced_losses(scores,labels,groups)
                    loss=bce+coefficient*rank+.01*aux
                if not torch.isfinite(loss):raise ValueError('Nonfinite balanced training loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
                losses.append(float(loss.detach()))
            raw=score_dien(model,val,context,numeric,device);result,_,_=metrics(val,raw,'search_idx')
            history.append({'epoch':epoch+1,'mean_loss':float(np.mean(losses)),'validation':result})
            write_json(target/'epochs.json',history)
            print('balanced',coefficient,'epoch',epoch+1,json.dumps(result),flush=True)
            if result['NDCG@10']>best+1e-6:
                best=result['NDCG@10'];stale=0;best_epoch=epoch+1;torch.save(model.state_dict(),target/'dien.pt')
            else:
                stale+=1
                if stale>=2:break
        model.load_state_dict(torch.load(target/'dien.pt',map_location=device,weights_only=True))
        raw=score_dien(model,val,context,numeric,device)
        pd.DataFrame({'search_idx':val.search_idx,'note_idx':val.note_idx,'score':raw}).to_parquet(target/'validation_raw_predictions.parquet',index=False)
        for weight in (0.,.1,.25,.5,1.):
            score=(1-weight)*group_minmax(val.coarse_score,val.search_idx)+weight*group_minmax(raw,val.search_idx)
            result,requests,predictions=metrics(val,score,'search_idx')
            name=f'weight_{weight:g}';requests.to_parquet(target/(name+'_validation_requests.parquet'),index=False)
            predictions.to_parquet(target/(name+'_validation_predictions.parquet'),index=False)
            choices.append({'lambda':coefficient,'weight':weight,'best_epoch':best_epoch,'directory':str(target),
                            'validation':result,'checkpoint_sha256':digest(target/'dien.pt')})
        write_json(target/'training.json',{'history':history,'seconds':time.monotonic()-start,'best_epoch':best_epoch})
        del model,optimizer;torch.cuda.empty_cache()
    best=max(choices,key=lambda value:(value['validation']['NDCG@10'],-value['weight'],-value['lambda']))
    baseline=original['dien']['chosen']['validation']
    selection={'chosen':best,'candidates':choices,'original_stack_validation':baseline,
               'accepted':best['validation']['NDCG@10']>baseline['NDCG@10']+1e-12,
               'test_policy':'Only evaluate official test if validation beats original stack; no test selection',
               'model_change':'request-balanced objective + auxiliary negative mask v2 jointly'}
    write_json(output/'selection.json',selection);print('SELECTED',json.dumps(selection),flush=True)


def evaluate(output):
    """Evaluate exactly the validation-frozen candidate with its original OOF teacher."""
    import shutil
    from types import SimpleNamespace
    from train_rank_stack import evaluate as evaluate_frozen_stack
    selection=json.loads((output/'selection.json').read_text())
    if not selection['accepted']:
        raise RuntimeError('Validation did not beat incumbent: official test must remain unread')
    own=json.loads((output/'manifest.json').read_text());chosen=selection['chosen']
    teacher=BASE/'gbdt.txt'
    if digest(teacher)!=own['inputs'][str(teacher.relative_to(ROOT))]:raise ValueError('OOF teacher changed')
    target=output/'selected_final';target.mkdir(exist_ok=False)
    shutil.copyfile(teacher,target/'gbdt.txt')
    shutil.copyfile(Path(chosen['directory'])/'dien.pt',target/'dien.pt')
    bundle={'scene':'search','key':'search_idx','features':own['numeric_features'][:-1],
      'parent_selection_sha256':digest(output/'selection.json'),
      'dien':{'numeric_features':own['numeric_features'],'architecture':own['architecture'],
              'chosen':{'weight':chosen['weight']},'model_sha256':chosen['checkpoint_sha256'],
              'objective':own['objective'],'auxiliary_contract':own['auxiliary_contract']}}
    write_json(target/'manifest.json',bundle)
    evaluate_frozen_stack(SimpleNamespace(scene='search',output=target,
        test_query_cache=ROOT/'outputs/evaluation/optimization_20261031/search/query_cache/test'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--stage',choices=['train','evaluate'],default='train')
    args=parser.parse_args();{'train':train,'evaluate':evaluate}[args.stage](args.output)