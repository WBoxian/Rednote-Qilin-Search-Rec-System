"""Fixed-duration Rec diagnostic controls; NEVER eligible for deployment or test-based selection."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from scripts import optimize_rec_iteration as iteration
from scripts import optimize_rec_retrieval as rec
from scripts import optimize_rec_northstar as north
ROOT=iteration.ROOT;OLD=iteration.OLD

def plan(out):
 path=out/'diagnostic_plan.json'
 if path.exists():raise FileExistsError('Diagnostic plan already frozen')
 parent=iteration.verify_plan(out)
 result={'scope':'Fixed-duration algorithm ablation, diagnostic only, never deployment/selection','targets':[{'variant':v,'epoch':6} for v in iteration.VARIANTS],
 'extra_targets_rule':'If original validation-gated winner epoch differs from6, add that fixed selected epoch and continuation_control at SAME epoch. No test-based recipe choice.',
 'index':'Original V3 IVFPQ2048/m32/bits8/IP sample100000fit-exposurevectors seed42,nprobe32; no grid',
 'fusion':'Fixed ANN:Swing:UserCF=1:2:1/k60 route1000/final500, same frozen fit graph and allpast graph separately',
 'official_test_policy':'Only after all train/validation/model/index selections frozen and root approval; all11115requests/44683unique clicked pairs; never pick testbest',
 'train_plan_sha256':rec.digest(out/'plan.json'),'source_sha256':{str(p.resolve()):rec.digest(p) for p in [Path(__file__),ROOT/'scripts/optimize_rec_iteration.py',ROOT/'scripts/optimize_rec_retrieval.py',ROOT/'scripts/optimize_rec_northstar.py',ROOT/'src/recall/dssm_trainer.py']},'deployment_eligible':False}
 rec.write_json(path,result);print(json.dumps(result,indent=2));return result

def verify(out):
 p=json.loads((out/'diagnostic_plan.json').read_text());iteration.verify_plan(out)
 if rec.digest(out/'plan.json')!=p['train_plan_sha256']:raise ValueError('Training plan changed')
 for file,digest in p['source_sha256'].items():
  if rec.digest(Path(file))!=digest:raise ValueError('Frozen diagnostic source changed')
 return p

def targets(out,include_selected=False):
 declared=verify(out);result=declared['targets'].copy()
 if include_selected:
  selection=json.loads((out/'selection.json').read_text())
  if selection['accepted']:
   name=selection['selected']['variant'];chosen=json.loads((out/name/'selected.json').read_text())['selected'];epoch=chosen['epoch']
   for v in [name,'continuation_control']:
    if not any(t['variant']==v and t['epoch']==epoch for t in result):result.append({'variant':v,'epoch':epoch})
 return result

def targetdir(out,variant,epoch):return out/'diagnostics'/f'{variant}_epoch_{epoch}'

def build(out,variant,epoch):
 import faiss
 verify(out)
 if not any(t['variant']==variant and t['epoch']==epoch for t in targets(out,include_selected=(out/'selection.json').exists())):raise ValueError('Target outside frozen fixed-duration plan')
 src=out/variant/f'epoch_{epoch}';dest=targetdir(out,variant,epoch);dest.mkdir(parents=True,exist_ok=False);meta=json.loads((out/variant/'training_manifest.json').read_text());itemids=np.load(OLD/'dssm/item_ids.npy');vectors=np.memmap(src/'item_vectors.bin',np.float32,'r',shape=(len(itemids),128));fitids=np.unique(pd.read_parquet(OLD/'labels_fit.parquet',columns=['note_idx']).note_idx);positions=np.searchsorted(itemids,fitids);assert np.array_equal(itemids[positions],fitids);sample=np.random.default_rng(42).choice(positions,min(100000,len(positions)),replace=False);faiss.omp_set_num_threads(2)
 idx=faiss.IndexIVFPQ(faiss.IndexFlatIP(128),128,2048,32,8,faiss.METRIC_INNER_PRODUCT);idx.cp.seed=42;idx.train(np.ascontiguousarray(vectors[sample]))
 for start in range(0,len(itemids),65536):idx.add(np.ascontiguousarray(vectors[start:start+65536]))
 idx.nprobe=32;faiss.write_index(idx,str(dest/'ivfpq.faiss'));ids=np.load(src/'validation_request_ids.npy');queries=np.load(src/'validation_vectors.npy');rec.retrieve_index(idx,ids,queries,itemids,dest/'validation_ann.parquet');req=pd.read_parquet(OLD/'requests_validation.parquet');cf=OLD/'validation/hist025_fold200';north.write_candidates(req,[dest/'validation_ann.parquet',cf/'swing.parquet',cf/'usercf.parquet'],{'weights':[1,2,1],'rrf_k':60},dest/'validation_selected.parquet');score,rows=rec.metrics(req,pd.read_parquet(dest/'validation_selected.parquet'));rows.to_parquet(dest/'validation_metrics.parquet',index=False)
 manifest={'variant':variant,'epoch':epoch,'checkpoint':str(src/'dssm_rec_optimized.pt'),'checkpoint_sha256':rec.digest(src/'dssm_rec_optimized.pt'),'item_vectors_sha256':rec.digest(src/'item_vectors.bin'),'index_sha256':rec.digest(dest/'ivfpq.faiss'),'training_manifest':str(out/variant/'training_manifest.json'),'training_manifest_sha256':rec.digest(out/variant/'training_manifest.json'),'validation':score,'index':{'nlist':2048,'m':32,'bits':8,'metric':'inner_product','nprobe':32,'training_vector_rows':len(sample)},'deployment_eligible':False,'scope':'Fixed-duration diagnostic; cannot select/promote using this manifest'};rec.write_json(dest/'build.json',manifest);print(json.dumps(manifest,indent=2));return manifest

def freeze(out):
 verify(out);path=out/'diagnostic_selection.json'
 if path.exists():raise FileExistsError('Diagnostic test targets already frozen')
 required=targets(out,True);models=[]
 for t in required:
  dest=targetdir(out,t['variant'],t['epoch']);manifest=json.loads((dest/'build.json').read_text());models.append({**t,'manifest':str(dest/'build.json'),'manifest_sha256':rec.digest(dest/'build.json'),**manifest})
 result={'deployment_eligible':False,'all_targets_fixed_before_test':True,'diagnostic_plan_sha256':rec.digest(out/'diagnostic_plan.json'),'main_selection_sha256':rec.digest(out/'selection.json'),'models':models,'scope':'No deployment/selection; fixedepoch6ABC plus deterministic matched selected-epoch control'};rec.write_json(path,result);print(json.dumps(result,indent=2));return result

def full_test(out,device='cpu'):
 import torch,faiss
 from scipy import sparse
 from recall import dssm_trainer as trainer
 from backend.online.item_features import get_item_feature_store
 from backend.offline.evaluate_full_test import confidence
 verify(out);frozen=json.loads((out/'diagnostic_selection.json').read_text())
 if not frozen['all_targets_fixed_before_test'] or frozen['deployment_eligible']:raise ValueError('Frozen diagnostic-only targets required')
 if rec.digest(out/'selection.json')!=frozen['main_selection_sha256'] or rec.digest(out/'diagnostic_plan.json')!=frozen['diagnostic_plan_sha256']:raise ValueError('Selections changed')
 for m in frozen['models']:
  if rec.digest(Path(m['manifest']))!=m['manifest_sha256'] or rec.digest(Path(m['checkpoint']))!=m['checkpoint_sha256']:raise ValueError('Diagnostic model changed')
 dest=out/'diagnostic_test';dest.mkdir(exist_ok=False);source=ROOT/'datasets/recommendation_test/train-00000-of-00001.parquet';source_sha=rec.digest(source)
 if source_sha!=json.loads((OLD/'manifest.json').read_text())['official_test_sha256']:raise ValueError('Official data changed')
 raw=pd.read_parquet(source,columns=['request_idx','user_idx','recent_clicked_note_idxs','rec_result_details_with_idx']);requests=pd.DataFrame({'gid':raw.request_idx,'user_idx':raw.user_idx,'history':raw.recent_clicked_note_idxs,'positives':raw.rec_result_details_with_idx.apply(lambda a:sorted({int(x['note_idx']) for x in a if x['click']>0}))});assert len(requests)==11115 and sum(requests.positives.map(len))==44683;requests.to_parquet(dest/'requests.parquet',index=False)
 torch.set_num_threads(2);device=torch.device(device);trainer.DEVICE=device;iteration.configure_numerics();faiss.omp_set_num_threads(2);store=get_item_feature_store(ROOT);frame=rec.raw_model_frame(pd.read_parquet(ROOT/'features/rec_test_features.parquet'),store,trainer,'rec');frame[['fans_num','follows_num']+[f'dense_feat{i}' for i in range(1,41)]]=0.;itemids=np.load(OLD/'dssm/item_ids.npy');results={};tables={}
 for m in frozen['models']:
  name=f"{m['variant']}_epoch_{m['epoch']}";target=dest/name;target.mkdir();meta=json.loads(Path(m['training_manifest']).read_text());_,cols,_=trainer.numeric_feature_columns('rec');model=trainer.DSSMModel('rec',meta['cat_vocabs'],len(cols),42+trainer.EMB_DIM+trainer.SEQ_TOWER_DIM,preprocessing=meta['normalizer']).to(device);model.load_state_dict(torch.load(m['checkpoint'],map_location=device,weights_only=True));model.eval();stores={'note_text':trainer.MMapStore('note_text_emb','note_text'),'note_img':trainer.MMapStore('note_img_emb','note_img')}
  with torch.amp.autocast('cuda',enabled=False):ids,queries=rec.encode_requests(model,frame,stores,'rec',meta['unknown_user_id'],set(meta['known_user_ids']))
  index=faiss.read_index(str(Path(m['manifest']).parent/'ivfpq.faiss'))
  if rec.digest(Path(m['manifest']).parent/'ivfpq.faiss')!=m['index_sha256']:raise ValueError('Index bytes changed')
  index.nprobe=32;rec.retrieve_index(index,ids,queries,itemids,target/'ann.parquet');metrics={}
  for label,gpath in [('fit',OLD/'cf_graph_0.25'),('allpast',OLD/'refit_cf_pretest')]:
   directory=target/label;directory.mkdir();graph=sparse.load_npz(gpath/'graph.npz');users=np.load(gpath/'users.npy');items=np.load(gpath/'items.npy');swing=sparse.load_npz(gpath/'swing.npz');rec.query_cf(requests,graph,users,items,swing,{'fold_in':True,'neighbors':200},directory,directory/'support.parquet');north.write_candidates(requests,[target/'ann.parquet',directory/'swing.parquet',directory/'usercf.parquet'],{'weights':[1,2,1],'rrf_k':60},directory/'selected.parquet');score,rows=rec.metrics(requests,pd.read_parquet(directory/'selected.parquet'));rows.to_parquet(directory/'metrics.parquet',index=False);metrics[label]=score;tables[name,label]=rows
  results[name]={'metrics':metrics,'deployment_eligible':False};print(json.dumps({'target':name,'metrics':metrics}),flush=True)
 comparisons={}
 for name in results:
  epoch=name.rsplit('_',1)[-1];control=f'continuation_control_epoch_{epoch}'
  if control not in results or control==name:continue
  for graph in ['fit','allpast']:
   rows=tables[name,graph];base=tables[control,graph];delta={}
   for metric,col in [('HitRate@50','hit50'),('Recall@100','recall100'),('known_positive_NDCG@10','ndcg10')]:
    t=rows[['gid','user_idx']].copy();t['delta']=rows[col]-base[col];delta[metric]={'delta':float(t.delta.mean()),'CI95':confidence(t,'delta')}
   comparisons[f'{name}-{control}-{graph}']=delta
 result={'deployment_eligible':False,'diagnostic_selection_sha256':rec.digest(out/'diagnostic_selection.json'),'source_sha256':source_sha,'all_requests':11115,'unique_clicked_pairs':44683,'models':results,'matched_duration_comparisons':comparisons,'parameter_choice_after_test':False};rec.write_json(dest/'metrics.json',result);return result

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=['plan','index','freeze','test'],required=True);p.add_argument('--output',type=Path,default=ROOT/'outputs/evaluation/northstar_fix_20261031/rec_iteration_20261031');p.add_argument('--variant',choices=list(iteration.VARIANTS));p.add_argument('--epoch',type=int,default=6);p.add_argument('--device',choices=['cpu','cuda'],default='cpu');a=p.parse_args()
 if a.phase=='plan':plan(a.output)
 elif a.phase=='index':build(a.output,a.variant,a.epoch)
 elif a.phase=='freeze':freeze(a.output)
 else:full_test(a.output,a.device)
if __name__=='__main__':main()
