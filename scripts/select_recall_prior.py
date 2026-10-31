"""Predeclared validation-only residual recall-prior gate and frozen test runner."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
import select_ranking_funnel as old
from backend.offline.evaluate_full_test import confidence, sha256
from training.recall_prior import RECALL_PRIOR_CONTRACT, recall_prior_terms
from training.scoring import blend_rank_scores

ROOT=Path(__file__).resolve().parents[1]
ALPHAS=(.05,.1,.25,.5,1.,2.,4.)
OFFSET=60


def plan(args):
    if args.output.exists():raise FileExistsError('Use a new prior-gate output directory')
    prepare=old.load_json(args.cached/'prepare_manifest.json')
    incumbent=old.load_json(args.cached/'selection.json')
    if prepare['scene']!=args.scene or prepare['phase']!='validation' or incumbent['scene']!=args.scene:
        raise ValueError('Only the original common training-validation cache is eligible')
    args.output.mkdir(parents=True)
    result={'scene':args.scene,'phase':'validation','cached':str(args.cached.resolve()),'input_cap':prepare['input_cap'],
        'coarse_caps':prepare['coarse_caps'],'alphas':list(ALPHAS),'k':OFFSET,'contract':RECALL_PRIOR_CONTRACT,
        'runs':prepare['runs'],'raw_control':{'alpha':0.,'coarse_cap':prepare['input_cap']},
        'incumbent':incumbent['selected'] if incumbent['accepted'] else None,
        'incumbent_contract':'Existing accepted model-only recipe, rank_prior=None, not an infinite-alpha proxy',
        'objective':'STRICT Recall100 increase vs raw, old HR10Search/HR50Rec+KP-NDCG10 guards; Search MRR100 guard; logged NDCG/AUC/GAUC eligible fineweights only',
        'tie_break':'Recall100,targetHR,KP-NDCG10,MRR100,preferincumbent,smalleralpha,smallercap,smallerfineweight',
        'no_test_before_frozen_selection':True,
        'parent_prepare_sha256':sha256(args.cached/'prepare_manifest.json'),
        'parent_selection_sha256':sha256(args.cached/'selection.json'),
        'new_grid_metrics_read':False,
        'code_sha256':{str(path.resolve()):sha256(path) for path in [Path(__file__),ROOT/'src/training/recall_prior.py',ROOT/'scripts/select_ranking_funnel.py',ROOT/'src/training/scoring.py']}}
    old.save_json(args.output/'plan.json',result)
    print(json.dumps({'stage':'plan_frozen','path':str(args.output),'alphas':list(ALPHAS),'incumbent':result['incumbent']}),flush=True)
    return result


def pack_candidates(features,requests,scene,input_cap):
    key=old.scene_key(scene);requests=requests.sort_values(key).reset_index(drop=True)
    if requests[key].duplicated().any():raise ValueError('Duplicate original request IDs')
    features=features.sort_values([key,'rank'],kind='stable').reset_index(drop=True)
    if not set(features[key]).issubset(set(requests[key])) or features.duplicated([key,'note_idx']).any() or features.duplicated([key,'rank']).any():
        raise ValueError('Foreign/duplicate candidate identities or recall ranks')
    if (features['rank']<1).any() or (features['rank']>input_cap).any():raise ValueError('Candidate rank violates frozen input cap')
    counts=features.groupby(key).size().reindex(requests[key],fill_value=0).to_numpy(np.int64)
    if (counts>input_cap).any():raise ValueError('Too many candidates in a request')
    row=np.searchsorted(requests[key].to_numpy(),features[key].to_numpy())
    column=features.groupby(key,sort=False).cumcount().to_numpy(np.int64)
    ids=np.full((len(requests),input_cap),-1,np.int64);ranks=np.zeros_like(ids)
    ids[row,column]=features.note_idx.to_numpy(np.int64);ranks[row,column]=features['rank'].to_numpy(np.int64)
    truth=[set(map(int,values)) for values in requests.positive_note_ids]
    hits=np.zeros(ids.shape,bool)
    for index,positive in enumerate(truth):hits[index]=np.isin(ids[index],list(positive))&(ranks[index]>0)
    npos=np.array([len(positive) for positive in truth],np.int64)
    discounts=1./np.log2(np.arange(2,12));idcg=np.array([discounts[:min(int(n),10)].sum() for n in npos])
    return {'requests':requests,'features':features,'row':row,'column':column,'ids':ids,'ranks':ranks,
        'hits':hits,'npos':npos,'idcg':idcg,'valid':ranks>0}


def model_matrices(packed,values,scene):
    key=old.scene_key(scene);values=values.sort_values([key,'rank'],kind='stable').reset_index(drop=True)
    features=packed['features']
    if len(values)!=len(features) or any(not np.array_equal(values[name],features[name]) for name in [key,'note_idx','rank']):
        raise ValueError('Model score identities do not match immutable candidate rows')
    coarse=np.zeros(packed['ids'].shape,np.float32);fine=np.zeros_like(coarse)
    coarse[packed['row'],packed['column']]=values.coarse_score.to_numpy(np.float32)
    fine[packed['row'],packed['column']]=values.dien_raw_score.to_numpy(np.float32)
    if not np.isfinite(coarse).all() or not np.isfinite(fine).all() or ((coarse<0)|(coarse>1)).any():
        raise ValueError('Probability-v2 requires finite sigmoid logits and binary-classifier teacher probabilities')
    return coarse,fine


def metric_frame(packed,indices):
    # Indices already encode the actual deterministic candidate order.
    hits=np.take_along_axis(packed['hits'],indices,axis=1)
    valid=np.take_along_axis(packed['valid'],indices,axis=1)
    hits=hits&valid;count=valid.sum(1);positions=np.argmax(hits,axis=1)+1 if hits.shape[1] else np.zeros(len(hits),int)
    first=np.where(hits.any(1),positions,np.inf)
    npos=packed['npos'];denom=np.maximum(npos,1)
    top=hits[:,:10];discounts=1./np.log2(np.arange(2,top.shape[1]+2))
    frame=pd.DataFrame({'gid':packed['requests'][old.scene_key('search')].to_numpy() if 'search_idx' in packed['requests'] else packed['requests'].request_idx.to_numpy(),
        'user_idx':packed['requests'].user_idx.to_numpy(),'positives':npos,'candidates':count,
        'known_positive_pairs_pool':hits.sum(1),'known_positive_pairs_top10':hits[:,:10].sum(1),
        'known_positive_pairs_top50':hits[:,:50].sum(1),'known_positive_pairs_top100':hits[:,:100].sum(1),
        'HitRate@10':(first<=10).astype(float),'HitRate@50':(first<=50).astype(float),
        'Recall@100':hits[:,:100].sum(1)/denom,
        'known_positive_NDCG@10':(top*discounts).sum(1)/np.maximum(packed['idcg'],1e-12),
        'MRR@10':np.where(first<=10,1./first,0.),'MRR@100':np.where(first<=100,1./first,0.)})
    return frame


def summary(frame):
    result=old.summarize(frame)
    result.update({'known_positive_pairs_pool':int(frame.known_positive_pairs_pool.sum()),
        'known_positive_pairs_top10':int(frame.known_positive_pairs_top10.sum()),
        'known_positive_pairs_top50':int(frame.known_positive_pairs_top50.sum()),
        'known_positive_pairs_top100':int(frame.known_positive_pairs_top100.sum()),
        'mean_candidates':float(frame.candidates.mean())})
    return result


def coarse_and_fine(packed,coarse,fine,cap,weight,contract='probability_v2'):
    if contract!='probability_v2':raise ValueError('This new experiment only supports the frozen probability_v2 models')
    original=np.where(packed['valid'],packed['ranks'],np.iinfo(np.int64).max)
    order=np.lexsort((original,-np.where(packed['valid'],coarse,-np.inf)),axis=1)[:,:cap]
    c=np.take_along_axis(coarse,order,axis=1);f=np.take_along_axis(fine,order,axis=1)
    ranks=np.take_along_axis(packed['ranks'],order,axis=1)
    learned=blend_rank_scores(c.ravel(),f.ravel(),np.zeros(c.size,np.int8),weight,contract).reshape(c.shape)
    valid=ranks>0
    fine_order=np.lexsort((np.where(valid,ranks,np.iinfo(np.int64).max),-np.where(valid,learned,-np.inf)),axis=1)
    return order,ranks,learned,fine_order


def candidate_order(packed,coarse,fine,recipe):
    chosen,ranks,learned,fine_order=coarse_and_fine(packed,coarse,fine,int(recipe['coarse_cap']),float(recipe['weight']),recipe['score_contract'])
    prior=recipe.get('rank_prior')
    if prior is None:final_order=fine_order;final_scores=learned.astype(float)
    else:
        base,term=recall_prior_terms(ranks,learned,int(prior['k']));final_scores=base+float(prior['alpha'])*term
        final_order=np.lexsort((np.where(ranks>0,ranks,np.iinfo(np.int64).max),-final_scores),axis=1)
    return {'coarse':chosen,'fine':np.take_along_axis(chosen,fine_order,axis=1),
        'final':np.take_along_axis(chosen,final_order,axis=1),
        'final_scores':np.take_along_axis(final_scores,final_order,axis=1)}


def choose_joint(grid,baseline,scene):
    eligible=[entry for entry in grid if entry['family']!='raw_control' and old.guard_passes(entry['metrics'],baseline,scene)]
    if not eligible:return None
    return max(eligible,key=lambda entry:(entry['metrics']['Recall@100'],entry['metrics']['HitRate@10' if scene=='search' else 'HitRate@50'],
        entry['metrics']['known_positive_NDCG@10'],entry['metrics']['MRR@100'],entry['family']=='incumbent',
        -float((entry.get('rank_prior') or {}).get('alpha',0.)),-entry['coarse_cap'],-entry['weight']))


def validate_model(model):
    run=Path(model['run']);manifest=old.load_json(run/'manifest.json')
    if (sha256(run/'gbdt.txt')!=model['gbdt_sha256'] or sha256(run/'dien.pt')!=model['dien_sha256']
            or old.contract_hash(old.model_contract(manifest))!=model['contract_sha256']):
        raise ValueError('Frozen trained model/contract changed')
    if old.eligible_weights(manifest)!=model['weights'] or model['score_contract']!='probability_v2':
        raise ValueError('Logged-validation fineweight eligibility or probability contract changed')
    return manifest


def select(args):
    out=args.output
    if (out/'selection.json').exists():raise FileExistsError('New prior selection is frozen; do not retune')
    declared=old.load_json(out/'plan.json');cached=Path(declared['cached'])
    if declared['scene']!=args.scene or declared['alphas']!=list(ALPHAS) or declared['k']!=OFFSET:raise ValueError('Predeclared prior grid changed')
    for name,key in [('prepare_manifest.json','parent_prepare_sha256'),('selection.json','parent_selection_sha256')]:
        if sha256(cached/name)!=declared[key]:raise ValueError('Parent cache/accepted-incumbent selection changed')
    old.verify_files(declared['code_sha256'])
    preparation=old.load_json(cached/'prepare_manifest.json');scores=old.load_json(cached/'score_manifest.json')
    if scores['scene']!=args.scene or scores['phase']!='validation' or scores['prepare_sha256']!=declared['parent_prepare_sha256']:
        raise ValueError('Complete matching validation score cache required')
    old.verify_files({**preparation['inputs'],**preparation['prepared'],**{m['score_path']:m['score_sha256'] for m in scores['models']}})
    if len(scores['models'])!=len(preparation['runs']):raise ValueError('Both declared model score caches must finish')
    code=[Path(__file__),ROOT/'src/training/recall_prior.py',ROOT/'scripts/select_ranking_funnel.py',ROOT/'src/training/scoring.py',
        ROOT/'src/training/rank_stack.py',ROOT/'src/training/dien_ranker.py',ROOT/'src/backend/online/ranking/stack.py']
    code_hashes={str(path.resolve()):sha256(path) for path in code}
    source_dir=out/'source_snapshot';source_dir.mkdir(exist_ok=True)
    for path in code:shutil.copyfile(path,source_dir/(path.parent.name+'_'+path.name))
    models=[]
    for index,model in enumerate(scores['models']):
        manifest=validate_model(model);frozen_path=out/f'frozen_model_{index}.json'
        old.save_json(frozen_path,{'model':model,'manifest_snapshot':manifest,'original_manifest_sha256':sha256(Path(model['run'])/'manifest.json')})
        models.append({**model,'snapshot':str(frozen_path.resolve()),'snapshot_sha256':sha256(frozen_path)})
    requests=pd.read_parquet(cached/'requests.parquet')
    features=pd.read_parquet(cached/'features.parquet',columns=[old.scene_key(args.scene),'note_idx','rank'])
    packed=pack_candidates(features,requests,args.scene,declared['input_cap'])
    raw_order=np.broadcast_to(np.arange(declared['input_cap']),packed['ids'].shape)
    raw_rows=metric_frame(packed,raw_order);raw_rows.to_parquet(out/'validation_raw_requests.parquet',index=False)
    baseline=summary(raw_rows);grid=[{'family':'raw_control','model_index':None,'run':None,'coarse_cap':declared['input_cap'],
        'weight':0.,'rank_prior':{'alpha':0.,'k':OFFSET,'contract':RECALL_PRIOR_CONTRACT},'score_contract':'probability_v2','metrics':baseline,'guard_passed':False}]
    cache={};incumbent=declared['incumbent'];start=time.monotonic()
    for index,model in enumerate(models):
        values=pd.read_parquet(model['score_path']);coarse,fine=model_matrices(packed,values,args.scene);cache[index]=(coarse,fine)
        if incumbent is not None and int(incumbent['model_index'])==index:
            recipe={**incumbent,'family':'incumbent','rank_prior':None}
            orders=candidate_order(packed,coarse,fine,recipe);observed=summary(metric_frame(packed,orders['final']))
            if any(abs(observed[name]-incumbent['metrics'][name])>1e-12 for name in old.METRICS):raise ValueError('Accepted incumbent failed independent matrix replay')
            recipe['metrics']=observed;recipe['guard_passed']=old.guard_passes(observed,baseline,args.scene);grid.append(recipe)
        for cap in declared['coarse_caps']:
            for weight in model['weights']:
                chosen,ranks,learned,_=coarse_and_fine(packed,coarse,fine,int(cap),float(weight),model['score_contract'])
                base,term=recall_prior_terms(ranks,learned,OFFSET)
                for alpha in ALPHAS:
                    combined=base+alpha*term
                    order=np.lexsort((np.where(ranks>0,ranks,np.iinfo(np.int64).max),-combined),axis=1)
                    selected=np.take_along_axis(chosen,order,axis=1)
                    metrics=summary(metric_frame(packed,selected))
                    entry={'family':'recall_prior','model_index':index,'run':model['run'],'coarse_cap':int(cap),'weight':float(weight),
                        'rank_prior':{'alpha':alpha,'k':OFFSET,'contract':RECALL_PRIOR_CONTRACT},'score_contract':model['score_contract'],
                        'metrics':metrics,'guard_passed':old.guard_passes(metrics,baseline,args.scene)}
                    grid.append(entry)
        print(json.dumps({'stage':'prior_validation','model':index,'recipes_complete':len(grid),'seconds':time.monotonic()-start}),flush=True)
    old.save_json(out/'validation_grid.json',{'scene':args.scene,'baseline':baseline,'candidates':grid})
    selected=choose_joint(grid,baseline,args.scene)
    result={'scene':args.scene,'accepted':selected is not None,'selected':selected,'test_recipe':selected,
        'selection_split':'original_train_common_forward_validation','frozen_before_official_test':True,
        'baseline':baseline,'input_cap':declared['input_cap'],'query_encoder_commit':preparation['query_encoder_commit'],
        'plan_sha256':sha256(out/'plan.json'),'grid_sha256':sha256(out/'validation_grid.json'),
        'parent_score_manifest_sha256':sha256(cached/'score_manifest.json'),'models':models,'code_sha256':code_hashes,
        'prepared_inputs':{**preparation['inputs'],**preparation['prepared']},
        'scope':'Same fixed input/caps/models/eligible fineweights; residual recall prior; no test model/parameter choice',
        'best_diagnostic':max(grid,key=lambda entry:entry['metrics']['Recall@100'])}
    if selected is not None:
        coarse,fine=cache[selected['model_index']];orders=candidate_order(packed,coarse,fine,selected)
        stage={'raw':raw_rows,'coarse':metric_frame(packed,orders['coarse']),'fine':metric_frame(packed,orders['fine']),
               'final':metric_frame(packed,orders['final'])}
        result['validation_stages']={name:summary(rows) for name,rows in stage.items()}
        result['paired_vs_raw']={}
        for name,rows in stage.items():rows.to_parquet(out/f'validation_{name}_requests.parquet',index=False)
        for name in old.METRICS:
            delta=stage['final'][name]-raw_rows[name];table=stage['final'][['gid','user_idx']].copy();table['delta']=delta
            result['paired_vs_raw'][name]={'delta':float(delta.mean()),'user_cluster_CI95':confidence(table,'delta')}
    old.save_json(out/'selection.json',result)
    print(json.dumps({'accepted':result['accepted'],'selected':selected,'baseline':baseline}),flush=True)
    return result


def verify_selection(path,scene):
    selected=old.load_json(path);folder=Path(path).parent
    if selected['scene']!=scene or not selected['frozen_before_official_test'] or selected['selection_split']!='original_train_common_forward_validation':
        raise ValueError('Official test needs a frozen train-validation prior selection')
    if not selected['accepted'] or selected['selected'] is None:raise ValueError('No accepted recipe: retain raw recall, do not force a diagnostic model into service')
    declared=old.load_json(folder/'plan.json');cached=Path(declared['cached'])
    old.verify_files({**selected['prepared_inputs'],**selected['code_sha256'],
        str(cached/'prepare_manifest.json'):declared['parent_prepare_sha256'],
        str(cached/'selection.json'):declared['parent_selection_sha256'],
        str(cached/'score_manifest.json'):selected['parent_score_manifest_sha256'],
        **{model['score_path']:model['score_sha256'] for model in selected['models']}})
    for name,key in [('plan.json','plan_sha256'),('validation_grid.json','grid_sha256')]:
        if sha256(folder/name)!=selected[key]:raise ValueError('Frozen validation artifacts changed')
    if not old.guard_passes(selected['selected']['metrics'],selected['baseline'],scene):raise ValueError('Recipe did not pass the declared validation gate')
    for model in selected['models']:
        if sha256(Path(model['snapshot']))!=model['snapshot_sha256']:raise ValueError('Frozen complete model manifest snapshot changed')
        validate_model(model)
    return selected


def save_final_candidates(path,packed,orders,scene):
    key=old.scene_key(scene);indices=orders['final'];valid=np.take_along_axis(packed['valid'],indices,axis=1)
    gid=np.broadcast_to(packed['requests'][key].to_numpy()[:,None],indices.shape)
    note=np.take_along_axis(packed['ids'],indices,axis=1);recall_rank=np.take_along_axis(packed['ranks'],indices,axis=1)
    ranks=np.broadcast_to(np.arange(1,indices.shape[1]+1),indices.shape)
    frame=pd.DataFrame({key:gid[valid],'gid':gid[valid],'note_idx':note[valid],'rank':ranks[valid],
        'original_recall_rank':recall_rank[valid],'score':orders['final_scores'][valid]})
    frame.to_parquet(path,index=False)


def test(args):
    # No official test source is opened until every validation recipe/source check passes.
    selection=verify_selection(args.selection,args.scene)
    release=ROOT/'outputs/evaluation/northstar_fix_20261031/release_plan.json'
    release_hash=sha256(release) if release.exists() else None
    output=args.output/'test';output.mkdir(parents=True,exist_ok=False)
    import torch
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    raw,nested,sources=old.original_requests(args.scene,'test');expected=6192 if args.scene=='search' else 11115
    if len(raw)!=expected:raise ValueError('Every original official test request must be retained')
    requests=old.request_table(raw,old.scene_key(args.scene),nested)
    features=old.read_candidates(args.recall,old.scene_key(args.scene),set(requests[old.scene_key(args.scene)]),selection['input_cap'])
    context,context_inputs,commit=old.make_context(args.scene,requests,args.query_cache)
    if commit!=selection['query_encoder_commit']:raise ValueError('Raw-query encoder commit differs from frozen validation')
    from backend.online.item_features import get_item_feature_store
    features=context.feature_frame(features,get_item_feature_store(ROOT))
    recipe=selection['selected'];values,model=old.score_run(Path(recipe['run']),features,requests,args.scene,args.query_cache,args.device,args.batch_size)
    features.to_parquet(output/'features.parquet',index=False);requests.to_parquet(output/'requests.parquet',index=False)
    values.to_parquet(output/'scores.parquet',index=False)
    packed=pack_candidates(features,requests,args.scene,selection['input_cap']);coarse,fine=model_matrices(packed,values,args.scene)
    orders=candidate_order(packed,coarse,fine,recipe)
    raw_indices=np.broadcast_to(np.arange(selection['input_cap']),packed['ids'].shape)
    stages={'raw':metric_frame(packed,raw_indices),'coarse':metric_frame(packed,orders['coarse']),
        'fine':metric_frame(packed,orders['fine']),'final':metric_frame(packed,orders['final'])}
    for name,rows in stages.items():rows.to_parquet(output/f'{name}_requests.parquet',index=False)
    save_final_candidates(output/'final_candidates.parquet',packed,orders,args.scene)
    result={'scene':args.scene,'all_test_requests':len(raw),'accepted_on_validation':True,'recipe':recipe,
        'selection_sha256':sha256(args.selection),'stages':{name:summary(rows) for name,rows in stages.items()},
        'model':model,'paired_stages':{},'parameter_choice_after_test':False,
        'inputs':{str(path.resolve()):sha256(path) for path in [*sources,args.recall,*context_inputs]},
        'outputs':{str(path.resolve()):sha256(path) for path in [output/'scores.parquet',output/'final_candidates.parquet']},
        'scope':'Complete original known-click retention; no labels created for unexposed candidates'}
    result['accepted_for_release']=old.release_passes(result['stages']['final'],result['stages']['raw'],args.scene,True)
    result['release_fallback']=None if result['accepted_for_release'] else 'recall_only'
    for new,previous in [('coarse','raw'),('fine','coarse'),('final','fine'),('final','raw')]:
        comparison={}
        for metric in old.METRICS:
            rows=stages[new][['gid','user_idx']].copy();rows['delta']=stages[new][metric]-stages[previous][metric]
            comparison[metric]={'delta':float(rows.delta.mean()),'user_cluster_CI95':confidence(rows,'delta')}
        result['paired_stages'][new+'-'+previous]=comparison
    result['release_plan_sha256']=release_hash
    old.save_json(output/'metrics.json',result)
    print(json.dumps({'scene':args.scene,'accepted_for_release':result['accepted_for_release'],'stages':result['stages']}),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene',choices=['search','rec'],required=True)
    parser.add_argument('--phase',choices=['plan','select','test'],required=True)
    parser.add_argument('--cached',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--selection',type=Path)
    parser.add_argument('--recall',type=Path)
    parser.add_argument('--query-cache',type=Path)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--batch-size',type=int,default=1024)
    args=parser.parse_args();args.output=args.output.resolve()
    if args.phase=='plan' and args.cached is None:parser.error('plan requires --cached old immutable validation score directory')
    if args.phase=='test' and (args.selection is None or args.recall is None):parser.error('test requires frozen --selection and --recall')
    if args.phase=='test' and args.scene=='search' and args.query_cache is None:parser.error('Search test requires its raw-query cache')
    if args.batch_size<1:parser.error('batch size must be positive')
    {'plan':plan,'select':select,'test':test}[args.phase](args)


if __name__=='__main__':main()