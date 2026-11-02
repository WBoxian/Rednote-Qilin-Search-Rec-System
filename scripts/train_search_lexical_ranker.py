"""Search lexical ranking: train-only temporal selection, legal feature ablations."""
from __future__ import annotations
import argparse, json, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from backend.online.preranking.service import build_search_item_features,SEARCH_ITEM_FEATURES
from backend.online.search_query import normalize_query_text,segment_query_terms,build_search_query_resources
from train_rank_stack import fit_tree, tree_predict, metrics, digest, write_json
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/evaluation/optimization_20261031/search_rank'
DEFAULT=ROOT/'outputs/evaluation/optimization_20261031/search_rank_lexical'
PROFILE_FEATURES=('gender_enc','platform_enc','age_enc','location_enc','fans_num','follows_num',*[f'dense_feat{i}' for i in range(1,41)])


def lexical_group(task):
    query,positions,titles,contents,terms=task
    result=build_search_item_features(pd.DataFrame({'note_title':titles,'note_content':contents}),query,query_terms=terms)
    return positions,result[list(SEARCH_ITEM_FEATURES)].to_numpy(np.float32)


def attach_profile(frame,source):
    columns=['search_idx',*PROFILE_FEATURES]
    raw=pd.read_parquet(source,columns=columns)
    if raw.groupby('search_idx')[list(PROFILE_FEATURES)].nunique(dropna=False).gt(1).any().any():
        raise ValueError('User snapshot changes within a request; cannot use a request profile join')
    profile=raw.drop_duplicates('search_idx')
    out=frame.merge(profile,on='search_idx',how='left',validate='many_to_one',sort=False)
    for col in PROFILE_FEATURES:
        out[col]=pd.to_numeric(out[col],errors='coerce').replace([np.inf,-np.inf],np.nan).fillna(0).astype(np.float32)
    return out


def attach_lexical(frame,requests,workers=3,resources=None):
    """Compute unique normalized query-item pairs using only query/catalog text."""
    if requests.search_idx.duplicated().any():raise ValueError('Duplicate raw query request IDs')
    queries=requests.set_index('search_idx')['query']
    if not set(frame.search_idx).issubset(queries.index):raise ValueError('Missing original raw request query')
    out=frame.copy();out['_row']=np.arange(len(out),dtype=np.int64)
    out['query']=out.search_idx.map(queries).fillna('').astype(str).map(normalize_query_text)
    pairs=out[['query','note_idx']].drop_duplicates().reset_index(drop=True)
    con=duckdb.connect(config={'threads':'2','memory_limit':'3GB'})
    con.register('needed',pairs[['note_idx']].drop_duplicates())
    texts=con.execute('''SELECT n.note_idx,coalesce(n.note_title,'') note_title,
       coalesce(n.note_content,'') note_content FROM read_parquet(?) n SEMI JOIN needed USING(note_idx)''',
       [str(ROOT/'datasets/notes/*.parquet')]).fetchdf()
    con.close()
    if texts.note_idx.duplicated().any() or set(texts.note_idx)!=set(pairs.note_idx):raise ValueError('Incomplete/duplicate item text catalog')
    texts=texts.set_index('note_idx')
    pairs['note_title']=pairs.note_idx.map(texts.note_title)
    pairs['note_content']=pairs.note_idx.map(texts.note_content)
    values=np.empty((len(pairs),len(SEARCH_ITEM_FEATURES)),dtype=np.float32)
    groups=pairs.groupby('query',sort=True).indices
    def tasks():
        for query,indices in groups.items():
            part=pairs.iloc[indices]
            terms=segment_query_terms(query,resources) if resources is not None else None
            yield query,indices,part.note_title.tolist(),part.note_content.tolist(),terms
    start=time.monotonic()
    if workers>1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for number,(indices,array) in enumerate(pool.map(lexical_group,tasks(),chunksize=32)):
                values[indices]=array
                if number%5000==0:print('lexical query groups',number,'/',len(groups),flush=True)
    else:
        for task in tasks():
            indices,array=lexical_group(task);values[indices]=array
    for i,col in enumerate(SEARCH_ITEM_FEATURES):pairs[col]=values[:,i]
    out=out.merge(pairs[['query','note_idx',*SEARCH_ITEM_FEATURES]],on=['query','note_idx'],how='left',validate='many_to_one',sort=False)
    out=out.sort_values('_row').drop(columns='_row').reset_index(drop=True)
    if len(out)!=len(frame) or not np.isfinite(out[list(SEARCH_ITEM_FEATURES)].to_numpy()).all():raise ValueError('Lexical row alignment/finiteness failure')
    return out,{'unique_queries':len(groups),'unique_query_item_pairs':len(pairs),'rows':len(out),'seconds':time.monotonic()-start,'workers':workers}


def prepare(output,workers=3,use_resources=False):
    output.mkdir(parents=True,exist_ok=True)
    if (output/'manifest.json').exists():raise RuntimeError('Lexical feature preparation already frozen')
    original=json.loads((BASE/'manifest.json').read_text())
    if digest(BASE/'train.parquet')!=original['training_artifact_sha256']:raise ValueError('Base ranking training file changed')
    frame=pd.read_parquet(BASE/'train.parquet')
    frame=frame[frame.partition.isin(['fit','validation'])].reset_index(drop=True)
    raw_path=ROOT/'datasets/search_train/train-00000-of-00001.parquet'
    source=ROOT/'features/search_train_features.parquet'
    requests=pd.read_parquet(raw_path,columns=['search_idx','query'])
    resources=None
    if use_resources:
        from dataclasses import asdict
        fit_ids=set(frame.loc[frame.partition=='fit','search_idx'])
        fit_queries=requests[requests.search_idx.isin(fit_ids)]
        resources=build_search_query_resources(fit_queries['query'])
        (output/'query_resources.json').write_text(json.dumps(asdict(resources),ensure_ascii=False,sort_keys=True,
            default=lambda value:sorted(value) if isinstance(value,(set,frozenset)) else list(value)))
    frame,feature_audit=attach_lexical(frame,requests,workers,resources)
    frame=attach_profile(frame,source)
    frame.to_parquet(output/'train.parquet',index=False)
    if frame[frame.partition=='fit'].event_timestamp.max()>=frame[frame.partition=='validation'].event_timestamp.min():raise ValueError('Non-forward inherited ranking split')
    manifest={'scene':'search','seed':42,'base_features':original['features'],'lexical_features':list(SEARCH_ITEM_FEATURES),
      'profile_features':list(PROFILE_FEATURES),'feature_audit':feature_audit,'base_partition':original,
      'inputs':{str(path.relative_to(ROOT)):digest(path) for path in [BASE/'train.parquet',raw_path,source,ROOT/'src/backend/online/preranking/service.py']},
      'limitations':['Profile/history/catalog/vocab provided snapshots lack independently verified event-time provenance',
       'Anonymous dense40 semantics unknown; profile-enhanced models are conditional snapshot experiments, not a strict PIT-certified production whitelist',
       'Logged-exposure ranking, not full-catalog or online ranking; no position/time/aggregate predictor',
       'Official test previously diagnosed; final evaluation is a complete regression acceptance benchmark'],
      'artifact_sha256':digest(output/'train.parquet'),'tokenization':'fit-only query resources' if use_resources else 'default resources=None',
      'query_resource_fit_requests':len(fit_queries) if use_resources else 0,
      'query_resource_sha256':digest(output/'query_resources.json') if use_resources else None}
    write_json(output/'manifest.json',manifest)
    print('prepared',feature_audit,flush=True)


def feature_sets(base):
    lexical=list(SEARCH_ITEM_FEATURES);profile=list(PROFILE_FEATURES)
    if set(base)&{'position','pos_bucket','event_timestamp','click','positive','y_multi','rank'}:
        raise ValueError('Unsafe base predictor')
    return {'base':list(base),'lexical_only':lexical,'base_lexical':list(base)+lexical,
            'base_profile':list(base)+profile,'all':list(base)+lexical+profile}


def train(output):
    manifest=json.loads((output/'manifest.json').read_text())
    if 'selection' in manifest:raise RuntimeError('Search lexical selection already frozen')
    if digest(output/'train.parquet')!=manifest['artifact_sha256']:raise ValueError('Prepared lexical features changed')
    frame=pd.read_parquet(output/'train.parquet')
    fit=frame[frame.partition=='fit'].reset_index(drop=True)
    val=frame[frame.partition=='validation'].reset_index(drop=True)
    subsets=feature_sets(manifest['base_features']);candidates=[]
    for subset in ('base_lexical','all'):
        kinds=('ranker',) if subset=='all' and manifest.get('query_resource_fit_requests',0) else ('classifier','ranker')
        for kind in kinds:
            for leaves in (15,31):
                config={'kind':kind,'n_estimators':400,'learning_rate':.04,'num_leaves':leaves,
                        'min_child_samples':100,'reg_lambda':2.,'random_state':42,'n_jobs':3,'verbosity':-1,
                        'deterministic':True,'force_col_wise':True}
                name=f'{subset}_{kind}_{leaves}';start=time.monotonic()
                model=fit_tree(fit,subsets[subset],'search_idx',config,val)
                score=tree_predict(model,val,subsets[subset]);result,per_request,predictions=metrics(val,score,'search_idx')
                model.save_model(str(output/(name+'.txt')))
                per_request.to_parquet(output/(name+'_validation_requests.parquet'),index=False)
                predictions.to_parquet(output/(name+'_validation_predictions.parquet'),index=False)
                entry={'name':name,'subset':subset,'features':subsets[subset],'config':config,
                       'iterations':model.current_iteration(),'validation':result,'seconds':time.monotonic()-start}
                candidates.append(entry);write_json(output/'validation_grid.json',candidates)
                print(name,json.dumps(result),flush=True)
    best=max(candidates,key=lambda value:(value['validation']['NDCG@10'],value['validation']['AUC']))
    # Block-drop ablations use one common hyperparameter configuration, selected above.
    ablations=[]
    ablation_subsets=() if manifest.get('query_resource_fit_requests',0) else ('base','lexical_only','base_profile')
    for subset in ablation_subsets:
        name='ablation_'+subset
        model=fit_tree(fit,subsets[subset],'search_idx',best['config'],val)
        result,per_request,predictions=metrics(val,tree_predict(model,val,subsets[subset]),'search_idx')
        model.save_model(str(output/(name+'.txt')))
        per_request.to_parquet(output/(name+'_validation_requests.parquet'),index=False)
        predictions.to_parquet(output/(name+'_validation_predictions.parquet'),index=False)
        entry={'name':name,'subset':subset,'features':subsets[subset],'config':best['config'],
               'iterations':model.current_iteration(),'validation':result}
        ablations.append(entry);print(name,json.dumps(result),flush=True)
    original=lgb.Booster(model_file=str(BASE/'gbdt.txt'))
    original_result,original_requests,_=metrics(val,tree_predict(original,val,manifest['base_features']),'search_idx')
    original_requests.to_parquet(output/'original_validation_requests.parquet',index=False)
    all_options=candidates+ablations
    best=max(all_options,key=lambda value:(value['validation']['NDCG@10'],value['validation']['AUC']))
    no_profile=max([value for value in all_options if not (set(value['features'])&set(PROFILE_FEATURES))],
                  key=lambda value:(value['validation']['NDCG@10'],value['validation']['AUC']))
    selected=lgb.Booster(model_file=str(output/(best['name']+'.txt')));selected.save_model(str(output/'gbdt.txt'))
    manifest['selection']={'objective':'validation NDCG@10, tie AUC; no official test selection',
      'chosen':best,'best_without_profile':no_profile,'original_baseline':original_result,
      'all_validation_candidates':all_options,'model_sha256':digest(output/'gbdt.txt'),
      'predictor_features':best['features'],'profile_features_used':bool(set(best['features'])&set(PROFILE_FEATURES)),
      'accepted':best['validation']['NDCG@10']>original_result['NDCG@10']+1e-12,
      'incumbent_retained_when_rejected':str(BASE/'gbdt.txt'),
      'test_policy':'Do not score official test for a rejected validation candidate'}
    write_json(output/'manifest.json',manifest)
    write_json(output/'selection.json',manifest['selection'])
    print('SELECTED',best['name'],best['validation'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--stage',choices=['prepare','train'],default='prepare')
    parser.add_argument('--fit-query-resources',action='store_true')
    args=parser.parse_args()
    if args.stage=='prepare':prepare(args.output,args.workers,args.fit_query_resources)
    else:train(args.output)