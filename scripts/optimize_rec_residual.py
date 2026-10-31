"""Bounded train-only PCA/DSSM block-residual validation; no SGD or test-based recipe choice."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np,pandas as pd,pyarrow as pa,pyarrow.parquet as arrowpq,torch
from scripts import optimize_rec_retrieval as rec
from scripts.rec_iteration_fusion import replay
from recall import dssm_trainer as trainer
from recall.residual_dssm import ResidualDSSMModel,model_from_checkpoint,raw_history_text_mean,project_content,save_residual_checkpoints,encode_request_frame
from backend.online.item_features import get_item_feature_store
from backend.offline.evaluate_full_test import confidence
ROOT=Path(__file__).resolve().parents[1];OLD=ROOT/'outputs/evaluation/northstar_fix_20261031/rec_v3';ITER=ROOT/'outputs/evaluation/northstar_fix_20261031/rec_iteration_20261031';DEFAULT=ROOT/'outputs/evaluation/northstar_fix_20261031/rec_residual_20261031_v2';WEIGHTS=(.25,.5,.75)

def configure():
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

def weight_name(w):return f'w_{int(round(w*100)):03d}'

def plan(out):
 if out.exists():raise FileExistsError('Residual experiment must be a new immutable directory')
 out.mkdir();paths=[OLD/'dssm/epoch_3/dssm_rec_optimized.pt',OLD/'dssm/epoch_3/dssm_rec_optimized_user_tower.pt',OLD/'dssm/epoch_3/item_vectors.bin',OLD/'dssm/item_ids.npy',OLD/'labels_fit.parquet',OLD/'requests_validation.parquet',ROOT/'features/rec_train_features.parquet',OLD/'validation/hist025_fold200/swing.parquet',OLD/'validation/hist025_fold200/usercf.parquet',OLD/'cf_graph_0.25/items.npy',ITER/'fusion_gate/unchanged_v3/filter_history.parquet',ITER/'content_only_validation/metrics.json']
 paths+=list(sorted((ROOT/'datasets/notes').glob('*.parquet')))+list(sorted((ROOT/'features/vocab_dict').glob('*.pkl')))
 for folder,prefix in [('note_text_emb','note_text'),('note_img_emb','note_img')]:paths +=[ROOT/'embeddings'/folder/(prefix+'.bin'),ROOT/'embeddings'/folder/(prefix+'_map.json')]
 code=[Path(__file__),ROOT/'src/recall/residual_dssm.py',ROOT/'src/recall/context_fusion.py',ROOT/'src/training/history.py',ROOT/'src/recall/dssm_trainer.py',ROOT/'src/backend/online/item_features.py',ROOT/'scripts/rec_iteration_fusion.py',ROOT/'scripts/optimize_rec_retrieval.py',ROOT/'tests/test_residual_dssm.py',ROOT/'tests/test_rec_residual_driver.py']
 declared={'scope':'No SGD; only original-fit unsupervised PCA plus bounded validation weight selection; three original DSSM/Swing/UserCF routes, no DPR/fourthroute','PCA':{'dim':128,'fit_items':'Unique original V3 fit-exposure item IDs with valid nonzero BGEtext, atmost100000 seeded42; no val/test labels','input':'L2 raw768 item vectors; query available recent20 textmean thenL2; same fit itemcenter/project/L2','solver':'sklearn randomized PCA128,whitenFalse,random_state42,CPU2threads','empty':'Zero PCA querybranch for noavailable history; pure PCAcontrol retrievesnothing forzeroquery; residualkeeps learnedbranch'},'weights':list(WEIGHTS),'controls':{'0':'purePCA128 diagnostic, never deployment','1':'unchangedlearned128 exact canonical CPUquery baseline, never newlyselected'},'block_join':'q/item=[sqrt(w)*learned128,sqrt(1-w)*PCA128]; no overallnormalization, no cross terms','query_encoding':'CPU single-request batch1, shared factory/history helper, float32, bothTF32False/autocastFalse; oldcatalog learned128 export reused unchanged','exact_objective':'Fullsame11893val/40894clickedpairs; fixedF1 recent20filter/perroute1000/final500/RRF121k60; strictR100 increase andHR50/KP nonregression vsw1 SAMEexactANN baseline','stop':'If purePCA128 F1 standaloneANN Recall100 <= learned128 F1 standaloneANN, stop; if no eligible residual fullfusion, stop; no extraweights/SGD/thresholdsearch','PQ':'ONLY exactVALwinner:256dim,IVFPQ2048,m64,bits8,subdim4,nprobe32,fitexposure sample100k seed42; gateallhead/R100 vs currentV3 PQ F1; no probegrid','test':'NOofficialtest until exact/PQ selection and everyartifactSHA frozen+rootapproval; selectedrecipeonly, samefit/allpastCF separate andcomparedcurrentV3F1; no testweightchoice','code_sha256':{str(p.resolve()):rec.digest(p) for p in code},'inputs_sha256':{str(p.resolve()):rec.digest(p) for p in paths}}
 rec.write_json(out/'plan.json',declared);print(json.dumps({'plan':str(out/'plan.json'),'weights':list(WEIGHTS)}));return declared

def verify(out):
 d=json.loads((out/'plan.json').read_text())
 for file,expected in {**d['code_sha256'],**d['inputs_sha256']}.items():
  if rec.digest(Path(file))!=expected:raise ValueError('Frozen residual input/source changed: '+file)
 return d

def stores():return {'note_text':trainer.MMapStore('note_text_emb','note_text'),'note_img':trainer.MMapStore('note_img_emb','note_img')}

def fit(out):
 from sklearn.decomposition import PCA
 from threadpoolctl import threadpool_limits
 verify(out);configure();dest=out/'pca';dest.mkdir(exist_ok=False);text=stores()['note_text'];ids=np.unique(pd.read_parquet(OLD/'labels_fit.parquet',columns=['note_idx']).note_idx.to_numpy(np.int64));valid=[]
 for first in range(0,len(ids),8192):
  part=ids[first:first+8192];raw=text.get_batch_vecs(part.tolist()).numpy();good=np.isfinite(raw).all(1)&(np.linalg.norm(raw,axis=1)>0);valid.extend(part[good].tolist())
 sampled=np.random.default_rng(42).choice(np.asarray(valid,np.int64),min(100000,len(valid)),replace=False);raw=text.get_batch_vecs(sampled.tolist()).numpy();raw=raw/np.maximum(np.linalg.norm(raw,axis=1,keepdims=True),1e-12);start=time.monotonic()
 with threadpool_limits(limits=2):pca=PCA(n_components=128,svd_solver='randomized',whiten=False,random_state=42).fit(raw)
 np.save(dest/'sampled_fit_item_ids.npy',sampled);np.save(dest/'mean.npy',pca.mean_.astype(np.float32));np.save(dest/'components.npy',pca.components_.astype(np.float32));result={'fit_exposure_unique_items':len(ids),'valid_fit_text_items':len(valid),'sampled_items':len(sampled),'explained_variance_ratio_sum':float(pca.explained_variance_ratio_.sum()),'solver':'randomized','random_state':42,'seconds':time.monotonic()-start,'mean_sha256':rec.digest(dest/'mean.npy'),'components_sha256':rec.digest(dest/'components.npy'),'sampled_sha256':rec.digest(dest/'sampled_fit_item_ids.npy'),'supervised_gradient_updates':0,'validation_test_fitted':False};rec.write_json(dest/'fit.json',result);print(json.dumps(result,indent=2));return result

def load_meta():return torch.load(OLD/'dssm/epoch_3/dssm_rec_optimized_user_tower.pt',map_location='cpu',weights_only=False)['meta']

def validate_queries(out):
 configure();req=pd.read_parquet(OLD/'requests_validation.parquet').sort_values('gid');ids=set(req.gid);frame=pd.read_parquet(ROOT/'features/rec_train_features.parquet');frame=rec.raw_model_frame(frame[frame.request_idx.isin(ids)],get_item_feature_store(ROOT),trainer,'rec');meta=load_meta();model=model_from_checkpoint(OLD/'dssm/epoch_3/dssm_rec_optimized.pt',metadata=meta);start=time.monotonic();sources=stores();qids,learned=encode_request_frame(model,frame,sources,meta,batch_size=1);assert np.array_equal(qids,req.gid);raw=np.stack([raw_history_text_mean(v,sources['note_text']) for v in req.history]);mean=torch.from_numpy(np.load(out/'pca/mean.npy'));components=torch.from_numpy(np.load(out/'pca/components.npy'))
 with torch.no_grad():content=np.concatenate([project_content(torch.from_numpy(raw[i:i+1]),mean,components).numpy() for i in range(len(raw))])
 np.save(out/'validation_request_ids.npy',qids);np.save(out/'validation_learned_queries.npy',learned);np.save(out/'validation_pca_queries.npy',content);rec.write_json(out/'query_encoding.json',{'rows':len(qids),'batch_size':1,'device':'cpu','dtype':'float32','bothTF32False':True,'empty_content_queries':int((np.linalg.norm(content,axis=1)==0).sum()),'seconds':time.monotonic()-start});return req,model,meta,sources

def project_catalog(out,sources):
 configure();items=np.load(OLD/'dssm/item_ids.npy');dest=out/'pca/item_vectors.bin';data=np.memmap(dest,np.float32,'w+',shape=(len(items),128));mean=torch.from_numpy(np.load(out/'pca/mean.npy')).to('cuda');comp=torch.from_numpy(np.load(out/'pca/components.npy')).to('cuda')
 with torch.no_grad(),torch.amp.autocast('cuda',enabled=False):
  for first in range(0,len(items),32768):
   raw=sources['note_text'].get_batch_vecs(items[first:first+32768].tolist()).to('cuda');data[first:first+len(raw)]=project_content(raw,mean,comp).cpu().numpy()
 data.flush();del mean,comp,raw;torch.cuda.empty_cache();return data

def exact_route(ids,queries,vectors,itemids,destination):
 configure();start=time.monotonic();catalog=torch.empty(vectors.shape,device='cuda',dtype=torch.float32);valid=np.flatnonzero(np.linalg.norm(queries,axis=1)>0);schema=pa.schema([('gid',pa.int64()),('note_idx',pa.int64()),('score',pa.float32()),('rank',pa.int64())])
 with torch.no_grad(),torch.amp.autocast('cuda',enabled=False):
  for first in range(0,len(vectors),32768):catalog[first:first+32768]=torch.from_numpy(np.array(vectors[first:first+32768],copy=True)).to('cuda')
  with arrowpq.ParquetWriter(destination,schema) as writer:
   for first in range(0,len(valid),128):
    rows=valid[first:first+128];q=torch.from_numpy(queries[rows]).to('cuda');scores=q@catalog.T;values,positions=torch.topk(scores,1000,1,sorted=True);notes=itemids[positions.cpu().numpy()];v=values.cpu().numpy();order=np.lexsort((notes,-v),axis=1);notes=np.take_along_axis(notes,order,1);v=np.take_along_axis(v,order,1);frame=pd.DataFrame({'gid':np.repeat(ids[rows],1000),'note_idx':notes.ravel(),'score':v.ravel(),'rank':np.tile(np.arange(1,1001),len(rows))});writer.write_table(pa.Table.from_pandas(frame,schema=schema,preserve_index=False));del scores
 del catalog;torch.cuda.empty_cache();return time.monotonic()-start

def joint_vectors(out,w,base,content):
 dest=out/weight_name(w);dest.mkdir(exist_ok=False);vectors=np.memmap(dest/'item_vectors.bin',np.float32,'w+',shape=(len(base),256));a=float(torch.tensor(w).sqrt());b=float(torch.tensor(1.-w).sqrt())
 for first in range(0,len(base),65536):vectors[first:first+65536,:128]=a*np.asarray(base[first:first+65536]);vectors[first:first+65536,128:]=b*np.asarray(content[first:first+65536])
 vectors.flush();return dest,vectors,a,b

def exact(out):
 verify(out);configure()
 if (out/'exact_selection.json').exists():raise FileExistsError('Exact validation is frozen')
 req,base_model,meta,sources=validate_queries(out);itemids=np.load(OLD/'dssm/item_ids.npy');learneditems=np.memmap(OLD/'dssm/epoch_3/item_vectors.bin',np.float32,'r',shape=(len(itemids),128));contentitems=project_catalog(out,sources);learnedq=np.load(out/'validation_learned_queries.npy');contentq=np.load(out/'validation_pca_queries.npy');cf=OLD/'validation/hist025_fold200';graphitems=set(map(int,np.load(OLD/'cf_graph_0.25/items.npy')));records=[];control={};mean=np.load(out/'pca/mean.npy');components=np.load(out/'pca/components.npy')
 for label,q,vectors in [('learned_control',learnedq,learneditems),('pca_control',contentq,contentitems)]:
  dest=out/label;dest.mkdir();seconds=exact_route(req.gid.to_numpy(),q,vectors,itemids,dest/'ann.parquet');pure,rows=replay(req,{'ann':dest/'ann.parquet'},graphitems,'filter_history',dest/'ann_F1.parquet');rows.to_parquet(dest/'ann_F1_requests.parquet',index=False);full,rows=replay(req,{'ann':dest/'ann.parquet','swing':cf/'swing.parquet','usercf':cf/'usercf.parquet'},graphitems,'filter_history',dest/'fusion_F1.parquet');rows.to_parquet(dest/'fusion_F1_requests.parquet',index=False);control[label]={'standalone_ANN_F1':pure,'fusion_F1':full,'exact_seconds':seconds};print(json.dumps({'control':label,**control[label]}),flush=True)
 rec.write_json(out/'controls.json',control)
 if control['pca_control']['standalone_ANN_F1']['Recall@100']<=control['learned_control']['standalone_ANN_F1']['Recall@100']:
  result={'accepted':False,'reason':'PCA128 content control does not beat learned128 on F1 ANN R100; predeclared stop','controls':control,'weights_evaluated':[]};rec.write_json(out/'exact_selection.json',result);return result
 baseline=control['learned_control']['fusion_F1']
 for w in WEIGHTS:
  dest,vectors,a,b=joint_vectors(out,w,learneditems,contentitems);q=np.concatenate((a*learnedq,b*contentq),axis=1).astype(np.float32);np.save(dest/'validation_queries.npy',q);model=ResidualDSSMModel('rec',meta['cat_vocabs'],meta['item_dense_dim'],meta['user_dense_dim'],preprocessing=meta.get('numeric_preprocessing'),pca_mean=mean,pca_components=components,residual_weight=w);base_state=torch.load(OLD/'dssm/epoch_3/dssm_rec_optimized.pt',map_location='cpu',weights_only=True);missing,unexpected=model.load_state_dict(base_state,strict=False);assert set(missing)=={'residual_pca_mean','residual_pca_components','residual_weight','residual_version'} and not unexpected
  savedmeta=save_residual_checkpoints(model,{**meta,'PCA_fit_json_sha256':rec.digest(out/'pca/fit.json'),'source_model_sha256':rec.digest(OLD/'dssm/epoch_3/dssm_rec_optimized.pt'),'gradient_updates':0},dest/'checkpoints');loaded=model_from_checkpoint(dest/'checkpoints/dssm_rec_residual.pt');assert all(torch.equal(loaded.state_dict()[k],v) for k,v in base_state.items());seconds=exact_route(req.gid.to_numpy(),q,vectors,itemids,dest/'ann_exact.parquet');score,rows=replay(req,{'ann':dest/'ann_exact.parquet','swing':cf/'swing.parquet','usercf':cf/'usercf.parquet'},graphitems,'filter_history',dest/'fusion_exact.parquet');rows.to_parquet(dest/'fusion_exact_requests.parquet',index=False);passed=score['Recall@100']>baseline['Recall@100'] and score['HitRate@50']>=baseline['HitRate@50'] and score['known_positive_NDCG@10']>=baseline['known_positive_NDCG@10'];entry={'weight':w,'directory':str(dest),'metrics':score,'guard_passed':passed,'exact_seconds':seconds,'checkpoint_sha256':rec.digest(dest/'checkpoints/dssm_rec_residual.pt'),'item_vectors_sha256':rec.digest(dest/'item_vectors.bin')};records.append(entry);print(json.dumps(entry),flush=True);del vectors,model,loaded
 eligible=[r for r in records if r['guard_passed']];winner=max(eligible,key=lambda r:(r['metrics']['Recall@100'],r['metrics']['HitRate@50'],r['metrics']['known_positive_NDCG@10'])) if eligible else None;result={'accepted':winner is not None,'selected':winner,'baseline_exact_F1':baseline,'controls':control,'trials':records,'plan_sha256':rec.digest(out/'plan.json'),'code_sha256':json.loads((out/'plan.json').read_text())['code_sha256'],'no_test_seen':True};rec.write_json(out/'exact_selection.json',result);return result

def pq(out):
 import faiss
 verify(out);choice=json.loads((out/'exact_selection.json').read_text())
 if not choice['accepted']:raise ValueError('No eligible exact residual; stop')
 winner=choice['selected'];dest=out/'index';dest.mkdir(exist_ok=False);vectors=np.memmap(Path(winner['directory'])/'item_vectors.bin',np.float32,'r',shape=(1983938,256));items=np.load(OLD/'dssm/item_ids.npy');fitids=np.unique(pd.read_parquet(OLD/'labels_fit.parquet',columns=['note_idx']).note_idx);positions=np.searchsorted(items,fitids);sample=np.random.default_rng(42).choice(positions,min(100000,len(positions)),replace=False);faiss.omp_set_num_threads(2);start=time.monotonic();index=faiss.IndexIVFPQ(faiss.IndexFlatIP(256),256,2048,64,8,faiss.METRIC_INNER_PRODUCT);index.cp.seed=42;index.train(np.ascontiguousarray(vectors[sample]))
 for first in range(0,len(items),65536):index.add(np.ascontiguousarray(vectors[first:first+65536]))
 index.nprobe=32;faiss.write_index(index,str(dest/'ivfpq.faiss'));queries=np.load(Path(winner['directory'])/'validation_queries.npy');ids=np.load(out/'validation_request_ids.npy');rec.retrieve_index(index,ids,queries,items,dest/'validation_ann.parquet');req=pd.read_parquet(OLD/'requests_validation.parquet');cf=OLD/'validation/hist025_fold200';score,rows=replay(req,{'ann':dest/'validation_ann.parquet','swing':cf/'swing.parquet','usercf':cf/'usercf.parquet'},set(map(int,np.load(OLD/'cf_graph_0.25/items.npy'))),'filter_history',dest/'validation_selected.parquet');rows.to_parquet(dest/'validation_requests.parquet',index=False);baseline=json.loads((ITER/'fusion_gate/unchanged_v3/validation.json').read_text());baseline=next(r['metrics'] for r in baseline['trials'] if r['policy']=='filter_history');passed=score['Recall@100']>baseline['Recall@100'] and score['HitRate@50']>=baseline['HitRate@50'] and score['known_positive_NDCG@10']>=baseline['known_positive_NDCG@10'];build={'algorithm':'IVFPQ','dim':256,'nlist':2048,'m':64,'bits':8,'subquantizer_dim':4,'nprobe':32,'fit_samples':len(sample),'coarse_seed':42,'pq_seed':int(index.pq.cp.seed),'seconds':time.monotonic()-start,'index_bytes':(dest/'ivfpq.faiss').stat().st_size,'index_sha256':rec.digest(dest/'ivfpq.faiss'),'selected_checkpoint_sha256':winner['checkpoint_sha256'],'selected_item_vectors_sha256':winner['item_vectors_sha256']};rec.write_json(dest/'build.json',build)
 result={'accepted':passed,'selected':winner if passed else None,'diagnostic_exact_winner':winner,'PQ_validation':score,'baseline_v3_PQ_F1':baseline,'plan_sha256':rec.digest(out/'plan.json'),'exact_selection_sha256':rec.digest(out/'exact_selection.json'),'index_build_sha256':rec.digest(dest/'build.json'),'model_checkpoint_sha256':winner['checkpoint_sha256'],'index_sha256':build['index_sha256'],'code_sha256':json.loads((out/'plan.json').read_text())['code_sha256'],'frozen_before_official_test':True,'scope':'Validation-only fixedresidual; no further model/index params aftertest'};rec.write_json(out/'selection.json',result);print(json.dumps(result,indent=2));return result

def test(out):
 import faiss
 verify(out);selection=json.loads((out/'selection.json').read_text())
 if not selection['accepted']:raise ValueError('No accepted residual, never open test')
 winner=selection['selected'];modelpath=Path(winner['directory'])/'checkpoints/dssm_rec_residual.pt';indexpath=out/'index/ivfpq.faiss'
 if rec.digest(modelpath)!=selection['model_checkpoint_sha256'] or rec.digest(indexpath)!=selection['index_sha256'] or rec.digest(out/'exact_selection.json')!=selection['exact_selection_sha256'] or rec.digest(out/'index/build.json')!=selection['index_build_sha256']:raise ValueError('Frozen selected residual artifacts changed')
 configure();dest=out/'test';dest.mkdir(exist_ok=False);rawpath=ROOT/'datasets/recommendation_test/train-00000-of-00001.parquet';rawsha=rec.digest(rawpath)
 if rawsha!=json.loads((OLD/'manifest.json').read_text())['official_test_sha256']:raise ValueError('Official test source changed')
 raw=pd.read_parquet(rawpath,columns=['request_idx','user_idx','recent_clicked_note_idxs','rec_result_details_with_idx']);requests=pd.DataFrame({'gid':raw.request_idx,'user_idx':raw.user_idx,'history':raw.recent_clicked_note_idxs,'positives':raw.rec_result_details_with_idx.apply(lambda a:sorted({int(v['note_idx']) for v in a if v['click']>0}))});assert len(requests)==11115 and sum(requests.positives.map(len))==44683;model=model_from_checkpoint(modelpath);meta=model.checkpoint_meta;frame=rec.raw_model_frame(pd.read_parquet(ROOT/'features/rec_test_features.parquet'),get_item_feature_store(ROOT),trainer,'rec');ids,queries=encode_request_frame(model,frame,stores(),meta,batch_size=1);np.save(dest/'query_ids.npy',ids);np.save(dest/'query_vectors.npy',queries);index=faiss.read_index(str(indexpath));index.nprobe=32;faiss.omp_set_num_threads(2);items=np.load(OLD/'dssm/item_ids.npy');rec.retrieve_index(index,ids,queries,items,dest/'ann.parquet');results={};tables={}
 for graph,cf,itempath,baselinepath in [('fit',OLD/'test/cf/hist025_fold200',OLD/'cf_graph_0.25/items.npy',ITER/'fusion_gate/test_replay_v2/fit_filter_history.parquet'),('allpast',OLD/'refit_cf_pretest',OLD/'refit_cf_pretest/items.npy',ITER/'fusion_gate/test_replay_v2/allpast_filter_history.parquet')]:
  score,rows=replay(requests,{'ann':dest/'ann.parquet','swing':cf/'swing.parquet','usercf':cf/'usercf.parquet'},set(map(int,np.load(itempath))),'filter_history',dest/f'{graph}_selected.parquet');rows.to_parquet(dest/f'{graph}_requests.parquet',index=False);baseline,brows=rec.metrics(requests,pd.read_parquet(baselinepath));delta={}
  for metric,col in [('HitRate@50','hit50'),('Recall@100','recall100'),('known_positive_NDCG@10','ndcg10'),('MRR@100','mrr100')]:
   t=rows[['gid','user_idx']].copy();t['delta']=rows[col]-brows[col];delta[metric]={'delta':float(t.delta.mean()),'CI95':confidence(t,'delta')}
  results[graph]={'metrics':score,'baseline_v3_F1':baseline,'paired_vs_v3_F1':delta};tables[graph]=rows
 m,b=results['allpast']['metrics'],results['allpast']['baseline_v3_F1'];passed=m['Recall@100']>b['Recall@100'] and m['HitRate@50']>=b['HitRate@50'] and m['known_positive_NDCG@10']>=b['known_positive_NDCG@10'];result={'accepted_for_release':passed,'release_fallback':None if passed else 'previously_accepted_v3_F1','selected_weight':winner['weight'],'all_requests':11115,'unique_clicked_pairs':44683,'selection_sha256':rec.digest(out/'selection.json'),'source_sha256':rawsha,'results':results,'parameter_choice_after_test':False,'gradient_updates':0};rec.write_json(dest/'metrics.json',result);print(json.dumps(result,indent=2));return result

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=['plan','fit','exact','pq','test'],required=True);p.add_argument('--output',type=Path,default=DEFAULT);a=p.parse_args();globals()[a.phase](a.output)
if __name__=='__main__':main()
