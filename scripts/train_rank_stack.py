"""Train isolated temporal GBDT/DIEN candidates using official training data only."""
from __future__ import annotations
import argparse, json, time, hashlib, sys
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import torch
from backend.online.item_features import get_item_feature_store, STATIC_ITEM_FEATURES
from backend.offline.evaluate_full_test import ranking_metrics, confidence
from training.rank_stack import RequestContext, MappedVectors, SEMANTIC_FEATURES, forward_blocks, group_minmax

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from train_temporal_baseline import temporal_training_indices


def write_json(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(16*1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def metrics(frame, score, key, split='validation'):
    result,requests,predictions=ranking_metrics(frame.rename(columns={key:'gid','click':'positive'}),score)
    result['scope']=f'complete original {split} logged exposures; repeated impressions retained; NOT serving funnel'
    return result,requests,predictions


def contexts(scene, split, query_cache=None):
    folder='search' if scene=='search' else 'recommendation'
    key='search_idx' if scene=='search' else 'request_idx'
    source=ROOT/f'datasets/{folder}_{split}/train-00000-of-00001.parquet'
    requests=pd.read_parquet(source,columns=[key,'recent_clicked_note_idxs'])
    text=MappedVectors(ROOT/'embeddings/note_text_emb','note_text')
    image=MappedVectors(ROOT/'embeddings/note_img_emb','note_img')
    queries=None
    if scene=='search':
        if query_cache is None:
            raise ValueError('Search requires an explicit raw-query cache; zero fallback forbidden')
        # Trainer cache contract: request IDs plus normalized raw-query vectors.
        metadata_files=[query_cache] if query_cache.is_file() else list(query_cache.glob('*.json'))
        if len(metadata_files)!=1:raise ValueError('Select one explicit query cache metadata file')
        metadata_path=metadata_files[0]
        metadata=json.loads(metadata_path.read_text())
        values=np.load(metadata_path.with_suffix('.npy'),mmap_mode='r')
        index={int(gid):i for i,gid in enumerate(metadata['request_ids'])}
        queries=np.array([values[index[int(gid)]] for gid in requests[key]],dtype=np.float32)
        if not np.isfinite(queries).all() or (np.linalg.norm(queries,axis=1)<1e-6).any():
            raise ValueError('Missing or nonfinite raw-query vectors')
    return RequestContext(requests,key,text,image,queries)


def tree_predict(model, frame, features):
    score=model.predict(frame[features])
    if score.ndim==2:
        score=score[:,1]
    return score.astype(np.float32)


def fit_tree(frame, features, key, config, validation=None):
    params=dict(config)
    kind=params.pop('kind')
    model=(lgb.LGBMClassifier if kind=='classifier' else lgb.LGBMRanker)(**params)
    frame=frame.sort_values(key,kind='stable')
    kwargs={}
    if kind=='ranker':
        kwargs['group']=frame.groupby(key,sort=False).size().to_numpy()
    if validation is not None:
        validation=validation.sort_values(key,kind='stable')
        kwargs['eval_set']=[(validation[features],validation.click)]
        kwargs['callbacks']=[lgb.early_stopping(30,verbose=False)]
        if kind=='ranker':
            kwargs['eval_group']=[validation.groupby(key,sort=False).size().to_numpy()]
            kwargs['eval_at']=[10]
        else:
            kwargs['eval_metric']='auc'
    model.fit(frame[features],frame.click,**kwargs)
    return model.booster_


def prepare(args):
    out=args.output
    out.mkdir(parents=True,exist_ok=False)
    key='search_idx' if args.scene=='search' else 'request_idx'
    source=ROOT/f'features/{args.scene}_train_features.parquet'
    target=ROOT/f'features/{args.scene}_test_features.parquet'
    cols=[key,'user_idx','note_idx','event_timestamp','click']
    train=pd.read_parquet(source,columns=cols)
    # Only the official-test time boundary/IDs are read before final evaluation.
    boundary=pd.read_parquet(target,columns=[key,'event_timestamp'])
    if set(train[key]) & set(boundary[key]):
        raise ValueError('Train/test request overlap')
    historical,fit,val,test_start=temporal_training_indices(train,boundary,key)
    context=contexts(args.scene,'train',args.train_query_cache)
    historical=context.feature_frame(historical,get_item_feature_store(ROOT))
    historical['partition']='excluded'
    historical.loc[fit,'partition']='fit'
    historical.loc[val,'partition']='validation'
    historical.to_parquet(out/'train.parquet',index=False)
    features=[*STATIC_ITEM_FEATURES,*SEMANTIC_FEATURES]
    if args.scene=='search':features.append('query_item_cosine')
    manifest={'scene':args.scene,'key':key,'features':features,'seed':42,
              'train_rows':len(train),'fit_rows':len(fit),'validation_rows':len(val),
              'fit_requests':int(historical.iloc[fit][key].nunique()),
              'validation_requests':int(historical.iloc[val][key].nunique()),
              'fit_last_event':float(historical.iloc[fit].event_timestamp.max()),
              'validation_first_event':float(historical.iloc[val].event_timestamp.min()),
              'validation_last_event':float(historical.iloc[val].event_timestamp.max()),
              'official_test_first_event':test_start,
              'inputs':{str(source.relative_to(ROOT)):digest(source),str(target.relative_to(ROOT)):digest(target)},
              'training_artifact_sha256':digest(out/'train.parquet'),
              'limitations':['Provided catalog/vocabulary/history snapshots lack event-time provenance; benchmark assumes these fixed snapshots',
                             'Official test previously diagnosed; this is a full regression benchmark, not a new untouched holdout',
                             'Logged-exposure ranking does not measure full-catalog retrieval quality'],
              'query_cache':str(args.train_query_cache) if args.train_query_cache else None}
    assert manifest['fit_last_event']<manifest['validation_first_event']<=manifest['validation_last_event']<test_start
    write_json(out/'manifest.json',manifest)
    print(json.dumps(manifest),flush=True)


def train_trees(args):
    out=args.output; manifest=json.loads((out/'manifest.json').read_text())
    if manifest['scene']!=args.scene:raise ValueError('Scene does not match training manifest')
    if digest(out/'train.parquet')!=manifest['training_artifact_sha256']:raise ValueError('Training artifact changed')
    frame=pd.read_parquet(out/'train.parquet'); key=manifest['key']; features=manifest['features']
    fit=frame[frame.partition=='fit'].reset_index(drop=True)
    val=frame[frame.partition=='validation'].reset_index(drop=True)
    candidates=[]
    for kind in ('classifier','ranker'):
        for leaves in (15,31):
            config={'kind':kind,'n_estimators':400,'learning_rate':.04,'num_leaves':leaves,
                    'min_child_samples':100,'reg_lambda':2.,'random_state':42,'n_jobs':4,'verbosity':-1}
            model=fit_tree(fit,features,key,config,val)
            score=tree_predict(model,val,features)
            result,_,_=metrics(val,score,key)
            name=f'{kind}_{leaves}'
            model.save_model(str(out/(name+'.txt')))
            candidates.append({'name':name,'config':config,'validation':result,'iterations':model.current_iteration()})
            print(name,json.dumps(result),flush=True)
    # Pick only on train-internal validation NDCG; AUC is the deterministic tie breaker.
    chosen=max(candidates,key=lambda c:(c['validation']['NDCG@10'],c['validation']['AUC']))
    model=lgb.Booster(model_file=str(out/(chosen['name']+'.txt')))
    model.save_model(str(out/'gbdt.txt'))
    manifest['trees']={'candidates':candidates,'chosen':chosen,'model_sha256':digest(out/'gbdt.txt')}
    write_json(out/'manifest.json',manifest)
    train_oof(args)


def train_oof(args):
    out=args.output;manifest=json.loads((out/'manifest.json').read_text())
    if manifest['scene']!=args.scene:raise ValueError('Scene mismatch')
    if 'dien' in manifest:raise ValueError('OOF is frozen after DIEN training')
    frame=pd.read_parquet(out/'train.parquet');key=manifest['key'];features=manifest['features']
    fit=frame[frame.partition=='fit'].reset_index(drop=True)
    val=frame[frame.partition=='validation'].reset_index(drop=True)
    model=lgb.Booster(model_file=str(out/'gbdt.txt'))
    chosen=manifest['trees']['chosen']
    config={**chosen['config'],'n_estimators':chosen['iterations']}
    oof=np.full(len(fit),np.nan,dtype=np.float32); blocks=[]
    for block,(past,future) in enumerate(forward_blocks(fit,key)):
        fold_model=fit_tree(fit.iloc[past],features,key,config)
        oof[future]=tree_predict(fold_model,fit.iloc[future],features)
        blocks.append({'block':block,'fit_rows':len(past),'prediction_rows':len(future),
                       'fit_last_event':float(fit.iloc[past].event_timestamp.max()),
                       'prediction_first_event':float(fit.iloc[future].event_timestamp.min())})
    fit['coarse_score']=oof
    fit=fit[np.isfinite(fit.coarse_score)].reset_index(drop=True)
    val['coarse_score']=tree_predict(model,val,features)
    fit.to_parquet(out/'dien_fit_oof.parquet',index=False)
    val.to_parquet(out/'validation.parquet',index=False)
    manifest['trees'].update({'forward_oof_blocks':blocks,'oof_rows':len(fit)})
    write_json(out/'manifest.json',manifest)


def ablate_features(args):
    out=args.output;manifest=json.loads((out/'manifest.json').read_text())
    if manifest['scene']!=args.scene or 'dien' in manifest:raise ValueError('Feature selection must precede fine training')
    frame=pd.read_parquet(out/'train.parquet');key=manifest['key']
    fit=frame[frame.partition=='fit'].reset_index(drop=True)
    val=frame[frame.partition=='validation'].reset_index(drop=True)
    full_features=list(manifest['features']);original=dict(manifest['trees']['chosen'])
    original['features']=full_features
    choices=[original]
    subsets={'content_only':list(STATIC_ITEM_FEATURES)}
    if args.scene=='search':subsets['content_history']=[*STATIC_ITEM_FEATURES,*SEMANTIC_FEATURES]
    for name,features in subsets.items():
        config=dict(original['config'])
        model=fit_tree(fit,features,key,config,val)
        result,_,_=metrics(val,tree_predict(model,val,features),key)
        model.save_model(str(out/(name+'.txt')))
        choice={'name':name,'features':features,'config':config,'validation':result,'iterations':model.current_iteration()}
        choices.append(choice)
        print('Feature ablation',name,json.dumps(result),flush=True)
    selected=max(choices,key=lambda c:(c['validation']['NDCG@10'],c['validation']['AUC']))
    model=lgb.Booster(model_file=str(out/(selected['name']+'.txt')))
    model.save_model(str(out/'gbdt.txt'))
    manifest['feature_ablations']=choices
    manifest['features']=selected['features']
    manifest['trees']['chosen']=selected
    manifest['trees']['model_sha256']=digest(out/'gbdt.txt')
    write_json(out/'manifest.json',manifest)
    train_oof(args)


def request_batches(frame, key, budget=512, shuffle=False, seed=42):
    groups=[indices for indices in frame.groupby(key,sort=False).indices.values()]
    if shuffle:np.random.default_rng(seed).shuffle(groups)
    pending=[];size=0
    for indices in groups:
        if pending and size+len(indices)>budget:
            yield frame.iloc[np.concatenate(pending)]
            pending=[];size=0
        pending.append(indices);size+=len(indices)
    if pending:yield frame.iloc[np.concatenate(pending)]


def score_dien(model, frame, context, numeric, device):
    model.eval();parts=[]
    with torch.inference_mode():
        for start in range(0,len(frame),1024):
            features=context.batch(frame.iloc[start:start+1024],numeric,device)
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda' and getattr(model,'inference_precision','bf16')=='bf16'):
                score,_=model(features)
            parts.append(score.float().cpu().numpy())
    return np.concatenate(parts)


def train_dien(args):
    from training.dien_ranker import DIENRanker
    import torch.nn.functional as F
    torch.manual_seed(42);np.random.seed(42);torch.set_num_threads(4)
    out=args.output;manifest=json.loads((out/'manifest.json').read_text())
    if manifest['scene']!=args.scene:raise ValueError('Scene does not match training manifest')
    key=manifest['key'];numeric=[*manifest['features'],'coarse_score']
    fit=pd.read_parquet(out/'dien_fit_oof.parquet');val=pd.read_parquet(out/'validation.parquet')
    context=contexts(args.scene,'train',args.train_query_cache)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=DIENRanker(args.scene,numeric,hidden_dim=64,mlp_dims=[128,64],dropout=.15,mask_padded_history=True,auxiliary_gradients=True,mask_missing_aux_negatives=not args.legacy_aux_negatives,inference_precision=args.inference_precision).to(device)
    stats={name:{'mean':float(fit[name].mean()),'std':float(fit[name].std(ddof=0))} for name in numeric}
    model.set_zscore_stats(stats)
    optimizer=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-4)
    best=-np.inf;stale=0;history=[];start=time.monotonic()
    for epoch in range(args.epochs):
        model.train();losses=[]
        for batch in request_batches(fit,key,512,True,42+epoch):
            features=context.batch(batch,numeric,device)
            labels=torch.as_tensor(batch.click.to_numpy(np.float32),device=device)
            groups=torch.as_tensor(batch[key].to_numpy(np.int64),device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                score,aux=model(features,labels,groups)
                pair=(groups[:,None]==groups[None,:]) & (labels[:,None]>labels[None,:])
                differences=score[:,None]-score[None,:]
                pair_loss=F.softplus(-differences[pair]).mean() if pair.any() else score.sum()*0
                loss=F.binary_cross_entropy_with_logits(score,labels)+.25*pair_loss+.01*aux
            if not torch.isfinite(loss):raise ValueError('Non-finite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step();losses.append(float(loss.detach()))
        raw=score_dien(model,val,context,numeric,device)
        result,_,_=metrics(val,raw,key)
        history.append({'epoch':epoch+1,'loss':float(np.mean(losses)),'validation':result})
        print('DIEN',epoch+1,'loss',history[-1]['loss'],json.dumps(result),flush=True)
        if result['NDCG@10']>best+1e-6:
            best=result['NDCG@10'];stale=0
            torch.save(model.state_dict(),out/'dien.pt')
        else:
            stale+=1
            if stale>=2:break
    model.load_state_dict(torch.load(out/'dien.pt',map_location=device,weights_only=True))
    raw=score_dien(model,val,context,numeric,device)
    base=val.coarse_score.to_numpy();groups=val[key].to_numpy()
    base_norm=group_minmax(base,groups);fine_norm=group_minmax(raw,groups)
    choices=[]
    for weight in (0.,.1,.25,.5,1.):
        score=(1-weight)*base_norm+weight*fine_norm
        result,_,_=metrics(val,score,key)
        choices.append({'weight':weight,'validation':result})
    # Include bypass: a negative increment is never forced into the pipeline.
    chosen=max(choices,key=lambda c:(c['validation']['NDCG@10'],-c['weight']))
    val['dien_raw_score']=raw
    val.to_parquet(out/'validation.parquet',index=False)
    manifest['dien']={'numeric_features':numeric,'normalization_fit':'forward OOF rows only','aux_negative_contract':'same-request unclicked; missing negatives masked (v2)' if not args.legacy_aux_negatives else 'legacy batch-zero fallback (v1)',
                      'architecture':{'hidden_dim':64,'mlp_dims':[128,64],'dropout':.15,'mask_padded_history':True,'auxiliary_gradients':True,'mask_missing_aux_negatives':not args.legacy_aux_negatives,'inference_precision':args.inference_precision},
                      'training_history':history,'blend_candidates':choices,'chosen':chosen,
                      'seconds':time.monotonic()-start,'model_sha256':digest(out/'dien.pt')}
    write_json(out/'manifest.json',manifest)
    print('Chosen validation blend',json.dumps(chosen),flush=True)


def evaluate(args):
    from training.dien_ranker import DIENRanker
    import duckdb
    out=args.output
    if (out/'test_metrics.json').exists():raise ValueError('Final test already evaluated in this run; use saved artifacts')
    manifest=json.loads((out/'manifest.json').read_text());key=manifest['key']
    if manifest['scene']!=args.scene:raise ValueError('Scene does not match training manifest')
    source=ROOT/f'features/{args.scene}_test_features.parquet'
    test=pd.read_parquet(source,columns=[key,'user_idx','note_idx','event_timestamp','click'])
    folder='search' if args.scene=='search' else 'recommendation'
    nested='search_result_details_with_idx' if args.scene=='search' else 'rec_result_details_with_idx'
    with duckdb.connect() as con:
        con.register('features',test)
        mismatch=con.execute(f'''WITH raw AS (
            SELECT r.{key} gid,d.note_idx,d.click,count(*) n
            FROM read_parquet(?) r,unnest(r.{nested}) u(d) GROUP BY r.{key},d.note_idx,d.click
        ), feat AS (SELECT {key} gid,note_idx,click,count(*) n FROM features GROUP BY {key},note_idx,click)
        SELECT count(*) FROM raw FULL JOIN feat USING(gid,note_idx,click) WHERE raw.n IS DISTINCT FROM feat.n''',
        [str(ROOT/f'datasets/{folder}_test/*.parquet')]).fetchone()[0]
        if mismatch:raise ValueError('Official test exposure/label multiset mismatch')
    context=contexts(args.scene,'test',args.test_query_cache)
    test=context.feature_frame(test,get_item_feature_store(ROOT))
    model=lgb.Booster(model_file=str(out/'gbdt.txt'))
    base=tree_predict(model,test,manifest['features']);test['coarse_score']=base
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    arch=manifest['dien']['architecture'];numeric=manifest['dien']['numeric_features']
    dien=DIENRanker(args.scene,numeric,**arch).to(device)
    dien.load_state_dict(torch.load(out/'dien.pt',map_location=device,weights_only=True))
    fine=score_dien(dien,test,context,numeric,device)
    weight=manifest['dien']['chosen']['weight'];groups=test[key].to_numpy()
    combined=(1-weight)*group_minmax(base,groups)+weight*group_minmax(fine,groups)
    results={};requests={}
    for name,score in [('gbdt',base),('dien_raw',fine),('selected',combined)]:
        result,req,pred=metrics(test,score,key,split="official test")
        results[name]=result;requests[name]=req
        req.to_parquet(out/f'test_{name}_requests.parquet',index=False)
        pred.to_parquet(out/f'test_{name}_predictions.parquet',index=False)
    paired=requests['gbdt'][['gid','user_idx','ndcg@10']].rename(columns={'ndcg@10':'gbdt_ndcg'})
    paired=paired.merge(requests['selected'][['gid','ndcg@10']],on='gid',validate='one_to_one')
    paired['delta']=paired['ndcg@10']-paired.gbdt_ndcg
    paired.to_parquet(out/'test_paired.parquet',index=False)
    results['selected_minus_gbdt']={'mean_ndcg_delta':float(paired.delta.mean()),'user_cluster_ci95':confidence(paired,'delta')}
    results['feature_ablations']={}
    for choice in manifest.get('feature_ablations',[]):
        variant=lgb.Booster(model_file=str(out/(choice['name']+'.txt')))
        values=tree_predict(variant,test,choice['features'])
        result,req,pred=metrics(test,values,key,split='official test')
        results['feature_ablations'][choice['name']]=result
        req.to_parquet(out/f'test_ablation_{choice["name"]}_requests.parquet',index=False)
        pred.to_parquet(out/f'test_ablation_{choice["name"]}_predictions.parquet',index=False)
    results['test_rows']=len(test);results['test_requests']=int(test[key].nunique())
    results['selected_weight']=weight
    results['scope']='All original test exposures; conditional ranking, not full-catalog retrieval or online CTR'
    write_json(out/'test_metrics.json',results)
    manifest['final_test']={'input_sha256':digest(source),'rows':len(test),'requests':int(test[key].nunique()),
                            'metrics_sha256':digest(out/'test_metrics.json'),'selection':'Frozen before official test labels scored'}
    write_json(out/'manifest.json',manifest)
    print(json.dumps(results),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene',choices=['search','rec'],required=True)
    parser.add_argument('--phase',choices=['prepare','trees','oof','ablate','dien','evaluate'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--train-query-cache',type=Path)
    parser.add_argument('--test-query-cache',type=Path)
    parser.add_argument('--epochs',type=int,default=6)
    parser.add_argument('--inference-precision',choices=['fp32','bf16'],default='fp32')
    parser.add_argument('--legacy-aux-negatives',action='store_true',help='Reproduce the frozen v1 training objective; new training masks missing negatives')
    args=parser.parse_args()
    args.output=args.output.resolve()
    {'prepare':prepare,'trees':train_trees,'oof':train_oof,'ablate':ablate_features,'dien':train_dien,'evaluate':evaluate}[args.phase](args)

if __name__=='__main__':main()