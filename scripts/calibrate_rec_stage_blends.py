"""Validation-only gain gates for fixed Recommendation coarse/fine blend weights.

Model weights stay frozen. Unexposed candidates stay unlabeled; metrics measure
retention of original clicked positives, not NDCG, CTR, or AUC of the catalog.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
ALPHAS=(0.,.5,1.)
BETAS=(0.,.5,1.)
INPUT_CAP=320
COARSE_CAP=180
SEED=42


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(16*1024*1024),b""):h.update(block)
    return h.hexdigest()


def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n")


def minmax(values):
    values=np.asarray(values,dtype=float)
    if not len(values):return values.astype(np.float32)
    if not np.isfinite(values).all():raise ValueError("Nonfinite ranking scores")
    return ((values-values.min())/max(float(values.max()-values.min()),1e-8)).astype(np.float32)


def blend_stages(notes,ranks,tree,raw_dien,alpha,beta,input_cap=INPUT_CAP,coarse_cap=COARSE_CAP):
    """Only sort/select. The raw tree feature consumed by DIEN is never replaced."""
    notes=np.asarray(notes);ranks=np.asarray(ranks);tree=np.asarray(tree);raw_dien=np.asarray(raw_dien)
    if not (len(notes)==len(ranks)==len(tree)==len(raw_dien)) or len(notes)>input_cap:
        raise ValueError("Candidate shape violates the fixed input contract")
    if len(np.unique(notes))!=len(notes) or len(np.unique(ranks))!=len(ranks) or (ranks<1).any():
        raise ValueError("Candidate identities/ranks must be unique positive values")
    if not len(notes):return np.array([],dtype=int),np.array([],dtype=int)
    coarse=alpha*minmax(tree)+(1-alpha)*minmax(1./(60.+ranks))
    chosen=np.lexsort((notes,ranks,-coarse))[:coarse_cap]
    fine=beta*minmax(raw_dien[chosen])+(1-beta)*coarse[chosen]
    order=np.lexsort((notes[chosen],ranks[chosen],-fine))
    return chosen,chosen[order]


def metric_summary(frame):
    valid=frame[frame.positives>0]
    return {"all_requests":len(frame),"positive_requests":len(valid),
        "positive_pairs":int(valid.positives.sum()),"Recall@10":float(valid.recall10.mean()),
        "Recall@20":float(valid.recall20.mean()),"HitRate@10":float(valid.hit10.mean()),
        "hits10":int(valid.hits10.sum()),"hits20":int(valid.hits20.sum()),
        "scope":"known clicked-positive retention; unknown unexposed candidates never used as negatives"}


def grid_metrics(requests,scores,grid):
    groups={int(gid):part for gid,part in scores.groupby("gid",sort=False)}
    rows={pair:[] for pair in grid}
    for req in requests.itertuples():
        gid=int(req.gid);truth=set(map(int,req.positives))
        part=groups.get(gid)
        if part is None:
            notes=np.array([],dtype=np.int64);ranks=notes;tree=np.array([],float);dien=tree
        else:
            notes=part.note_idx.to_numpy();ranks=part["rank"].to_numpy()
            tree=part.coarse_score.to_numpy();dien=part.dien_raw_score.to_numpy()
        for alpha,beta in grid:
            coarse,order=blend_stages(notes,ranks,tree,dien,alpha,beta)
            h10=sum(int(item) in truth for item in notes[order[:10]])
            h20=sum(int(item) in truth for item in notes[order[:20]])
            hc=sum(int(item) in truth for item in notes[coarse])
            rows[(alpha,beta)].append((gid,int(req.user_idx),len(truth),h10,h20,
                h10/len(truth) if truth else np.nan,h20/len(truth) if truth else np.nan,
                int(h10>0),hc/len(truth) if truth else np.nan))
    frames={pair:pd.DataFrame(values,columns=["gid","user_idx","positives","hits10","hits20",
        "recall10","recall20","hit10","coarse_recall180"]) for pair,values in rows.items()}
    return {pair:metric_summary(frame) for pair,frame in frames.items()},frames


def paired_interval(after,before,repetitions=1000,split="validation"):
    paired=after[["gid","user_idx","recall10"]].merge(before[["gid","recall10"]],on="gid",
        suffixes=("_after","_before"),validate="one_to_one")
    paired["delta"]=paired.recall10_after-paired.recall10_before
    grouped=paired.groupby("user_idx").delta.agg(["sum","count"]).to_numpy()
    rng=np.random.default_rng(SEED);values=[]
    for _ in range(repetitions):
        selected=grouped[rng.integers(0,len(grouped),len(grouped))]
        values.append(float(selected[:,0].sum()/selected[:,1].sum()))
    return {"macro_delta":float(paired.delta.mean()),
        "user_cluster_CI95":[float(v) for v in np.quantile(values,[.025,.975])],
        "repetitions":repetitions,"seed":SEED,
        "interpretation":("Exploratory validation selection gate; same validation used for finite-grid choice, not an untouched holdout"
            if split=="validation" else "Fixed-parameter official-test regression comparison; no parameter selection on these test results")},paired


def select_with_gate(summary,frames):
    baseline=summary[(0.,0.)]
    eligible=[pair for pair,value in summary.items() if value["Recall@20"]>=baseline["Recall@20"]-1e-15
        and value["HitRate@10"]>=baseline["HitRate@10"]-1e-15]
    candidate=max(eligible,key=lambda pair:(summary[pair]["Recall@10"],-pair[0]-pair[1],-pair[0],-pair[1]))
    interval,_=paired_interval(frames[candidate],frames[(0.,0.)])
    chosen=candidate if candidate!=(0.,0.) and interval["user_cluster_CI95"][0]>0 else (0.,0.)
    return {"candidate":{"alpha":candidate[0],"beta":candidate[1],"metrics":summary[candidate]},
        "selected":{"alpha":chosen[0],"beta":chosen[1],"metrics":summary[chosen]},
        "gate_passed":chosen!=(0.,0.),"candidate_minus_bypass":interval,
        "reason":"Paired validation Recall10 CI lower bound >0 and Recall20/HR10 guards passed"
            if chosen!=(0.,0.) else "No positive lower confidence bound after validation selection; retain recall-order bypass"}


def verify_cached_inputs(destination, metadata, inputs, raw_path, static_store):
    """An existing score cache is reusable only while every consumed snapshot matches."""
    if metadata["inputs"]!=inputs or metadata["scores_sha256"]!=digest(destination):
        raise ValueError("Cached inference inputs or scores changed")
    if metadata["raw_context_sha256"]!=digest(raw_path):
        raise ValueError("Cached raw request context changed")
    for path,expected in metadata["embedding_sha256"].items():
        if digest(ROOT/Path(path))!=expected:raise ValueError("Cached embedding snapshot changed")
    encoded=hashlib.sha256()
    encoded.update(static_store.note_ids.tobytes());encoded.update(static_store.values.tobytes())
    if encoded.hexdigest()!=metadata["encoded_static_sha256"]:
        raise ValueError("Cached static feature snapshot changed")


def freeze_selection(path,value):
    """Never overwrite a winner after official-test inference may have started."""
    path=Path(path)
    if path.exists():
        previous=json.loads(path.read_text())
        if previous!=value:raise ValueError("Frozen validation selection changed; use a new output directory")
        return previous
    save(path,value)
    return value


def cache_scores(run,recall,split,destination,device,batch_size=512):
    import torch
    import lightgbm as lgb
    from train_rank_stack import contexts,tree_predict
    from backend.online.item_features import get_item_feature_store
    from training.dien_ranker import DIENRanker
    metadata=json.loads((run/"manifest.json").read_text())
    if metadata["scene"]!="rec":raise ValueError("Only Recommendation is in this protocol")
    key=metadata["key"]
    inputs={"recall":digest(recall),"gbdt":digest(run/"gbdt.txt"),"dien":digest(run/"dien.pt"),
        "run_manifest":digest(run/"manifest.json"),"feature_code":digest(ROOT/"src/training/rank_stack.py"),
        "inference_code":digest(ROOT/"scripts/train_rank_stack.py"),"model_code":digest(ROOT/"src/training/dien_ranker.py")}
    fingerprint=destination.with_suffix(".manifest.json")
    if destination.exists():
        recorded=json.loads(fingerprint.read_text())
        raw=ROOT/f"datasets/recommendation_{split}/train-00000-of-00001.parquet"
        verify_cached_inputs(destination,recorded,inputs,raw,get_item_feature_store(ROOT))
        return
    candidates=pd.read_parquet(recall)
    if key not in candidates:candidates=candidates.rename(columns={"gid":key})
    candidates=candidates[candidates["rank"]<=INPUT_CAP][[key,"note_idx","rank"]]
    if candidates.duplicated([key,"note_idx"]).any() or (candidates["rank"]<1).any():
        raise ValueError("Invalid recall pool")
    if (candidates.groupby(key).size()>INPUT_CAP).any():raise ValueError("Input budget exceeded")
    candidates=candidates.sort_values([key,"rank"],kind="stable").reset_index(drop=True)
    context=contexts("rec",split)
    if not set(candidates[key]).issubset(context.positions):raise ValueError("Foreign request context")
    static=get_item_feature_store(ROOT)
    tree=lgb.Booster(model_file=str(run/"gbdt.txt"))
    fine=DIENRanker("rec",metadata["dien"]["numeric_features"],**metadata["dien"]["architecture"]).to(device)
    fine.load_state_dict(torch.load(run/"dien.pt",map_location=device,weights_only=True));fine.eval()
    writer=None;started=time.monotonic()
    with torch.inference_mode():
        for start in range(0,len(candidates),8192):
            frame=context.feature_frame(candidates.iloc[start:start+8192].copy().reset_index(drop=True),static)
            raw_tree=tree_predict(tree,frame,metadata["features"])
            frame["coarse_score"]=raw_tree
            parts=[]
            for offset in range(0,len(frame),batch_size):
                features=context.batch(frame.iloc[offset:offset+batch_size],metadata["dien"]["numeric_features"],device)
                with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):
                    score,_=fine(features)
                parts.append(score.float().cpu().numpy())
            fine_scores=np.concatenate(parts)
            if not np.isfinite(raw_tree).all() or not np.isfinite(fine_scores).all():
                raise ValueError("Nonfinite cached model score")
            result=frame[[key,"note_idx","rank"]].rename(columns={key:"gid"})
            result["coarse_score"]=raw_tree;result["dien_raw_score"]=fine_scores
            table=pa.Table.from_pandas(result,preserve_index=False)
            if writer is None:writer=pq.ParquetWriter(destination,table.schema,compression="zstd")
            writer.write_table(table)
            if start%262144==0:print(split,"inference",start,"/",len(candidates),"seconds",time.monotonic()-started,flush=True)
    if writer is not None:writer.close()
    raw=ROOT/f"datasets/recommendation_{split}/train-00000-of-00001.parquet"
    inputs["raw_request_context"]=digest(raw)
    # These map/vector snapshots are immutable shared inputs; bind their bytes.
    embedding_inputs={}
    for store in (context.text,context.image):
        for path in store.files:embedding_inputs[str(path.relative_to(ROOT))]=digest(path)
    static_hash=hashlib.sha256()
    static_hash.update(static.note_ids.tobytes());static_hash.update(static.values.tobytes())
    save(fingerprint,{"inputs":{k:v for k,v in inputs.items() if k!="raw_request_context"},
        "raw_context_sha256":inputs["raw_request_context"],"embedding_sha256":embedding_inputs,
        "encoded_static_sha256":static_hash.hexdigest(),"rows":len(candidates),
        "requests":int(candidates[key].nunique()),"input_cap":INPUT_CAP,"model_input_coarse_score":"unchanged raw tree score",
        "inference_batch_size":batch_size,"device":str(device),"seconds":time.monotonic()-started,
        "scores_sha256":digest(destination)})
    del fine
    if device.type=="cuda":torch.cuda.empty_cache()


def write_selected(scores,alpha,beta,path):
    parts=[]
    for gid,frame in scores.groupby("gid",sort=False):
        coarse,order=blend_stages(frame.note_idx.to_numpy(),frame["rank"].to_numpy(),frame.coarse_score.to_numpy(),
            frame.dien_raw_score.to_numpy(),alpha,beta)
        chosen=frame.iloc[order].copy()
        chosen["final_rank"]=np.arange(1,len(chosen)+1,dtype=np.int32)
        parts.append(chosen)
    pd.concat(parts,ignore_index=True).to_parquet(path,index=False)


def main():
    import torch
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rec",type=Path,default=ROOT/"outputs/evaluation/optimization_20261031/rec")
    parser.add_argument("--run",type=Path,default=ROOT/"outputs/evaluation/optimization_20261031/rec_rank")
    parser.add_argument("--output",type=Path,default=ROOT/"outputs/evaluation/optimization_20261031/rec_stage_blends")
    parser.add_argument("--device",choices=["cuda","cpu"],default="cuda")
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/"metrics.json").exists():raise RuntimeError("Stage-gain gate already evaluated")
    device=torch.device(args.device)
    if device.type=="cuda":
        free,total=torch.cuda.mem_get_info()
        if free<3*1024**3:raise RuntimeError("Insufficient free GPU memory for light inference")
    torch.set_num_threads(2)
    grid=[(alpha,beta) for alpha in ALPHAS for beta in BETAS]
    protocol={"created_utc":datetime.now(timezone.utc).isoformat(),"scene":"rec","grid":[list(v) for v in grid],
        "input_cap":INPUT_CAP,"coarse_cap":COARSE_CAP,"final_topk":[10,20],
        "coarse_formula":"alpha*minmax(tree over320)+(1-alpha)*minmax(1/(60+original recall rank) over320)",
        "fine_formula":"beta*minmax(raw DIEN over selected180)+(1-beta)*coarse blended score",
        "model_input":"DIEN coarse_score always unchanged raw tree score; same fixed model weights",
        "selection":"validation macro Recall10 only; Recall20>=original order and HR10>=original order; alpha=beta=0 bypass included",
        "gate":"If best paired user-bootstrap Recall10 delta CI lower bound<=0, keep bypass; exploratory reused-validation gate",
        "metric_scope":"Known-positive retention only; no labels for unexposed candidates, no full-catalog AUC/NDCG claim",
        "limitations":["Official test already diagnosed; regression benchmark, not untouched holdout",
            "Provider history and full content/vector snapshots lack verified event-time provenance"]}
    if not (out/"protocol.json").exists():save(out/"protocol.json",protocol)
    cache_scores(args.run,args.rec/"validation_selected.parquet","train",out/"validation_scores.parquet",device)
    requests=pd.read_parquet(args.rec/"requests_validation.parquet")
    scores=pd.read_parquet(out/"validation_scores.parquet")
    summary,frames=grid_metrics(requests,scores,grid)
    save(out/"validation_grid.json",{f"alpha{a:g}_beta{b:g}":value for (a,b),value in summary.items()})
    selection=select_with_gate(summary,frames)
    selection["protocol_sha256"]=digest(out/"protocol.json")
    selection["validation_scores_sha256"]=digest(out/"validation_scores.parquet")
    selection["frozen_before_test"]=True
    selection=freeze_selection(out/"selection.json",selection)
    for pair,frame in frames.items():
        frame.to_parquet(out/f"validation_alpha{pair[0]:g}_beta{pair[1]:g}_requests.parquet",index=False)
    print("Frozen validation selection",json.dumps(selection),flush=True)
    # No official test labels have been read by selection. Final test uses only this frozen pair.
    cache_scores(args.run,args.rec/"refit_cf_pretest/selected.parquet","test",out/"test_scores.parquet",device)
    requests=pd.read_parquet(args.rec/"test/requests.parquet")
    scores=pd.read_parquet(out/"test_scores.parquet")
    chosen=(selection["selected"]["alpha"],selection["selected"]["beta"])
    test_grid=list(dict.fromkeys([(0.,0.),chosen]))
    values,frames=grid_metrics(requests,scores,test_grid)
    interval,paired=paired_interval(frames[chosen],frames[(0.,0.)],split="official_test")
    paired.to_parquet(out/"test_paired_requests.parquet",index=False)
    for pair,frame in frames.items():
        frame.to_parquet(out/f"test_alpha{pair[0]:g}_beta{pair[1]:g}_requests.parquet",index=False)
    write_selected(scores,*chosen,out/"test_selected.parquet")
    previous=ROOT/"outputs/evaluation/optimization_20261031/rec_funnel_refit/metrics.json"
    previous_metrics=json.loads(previous.read_text()) if previous.exists() else None
    logged=json.loads((args.run/"test_metrics.json").read_text())
    result={"scene":"rec","selection":selection,"official_test":{
        "recall_order_bypass":values[(0.,0.)],"frozen_selected":values[chosen],"paired_delta_from_bypass":interval},
        "previous_pure_stack_funnel":previous_metrics,
        "logged_exposure_ranking":{"gbdt":logged["gbdt"],"dien_raw":logged["dien_raw"],
            "original_logged_validation_selected_blend":logged["selected"],
            "original_blend_weight":logged["selected_weight"],"scope":logged["scope"],
            "not_the_stage_gate_selection_objective":True},
        "interpretation":"Exposure-pool NDCG estimates conditional ranking; this experiment gates known-positive retention of full-catalog candidates separately",
        "model_weights_changed":False,
        "ranked_output_schema":"test_selected.parquet rank is original recall rank; final_rank is calibrated stage order",
        "scope":protocol["metric_scope"],"limitations":protocol["limitations"]}
    save(out/"metrics.json",result)
    save(out/"source_manifest.json",{"script_sha256":digest(Path(__file__)),"selection_sha256":digest(out/"selection.json"),
        "validation_grid_sha256":digest(out/"validation_grid.json"),"test_scores_sha256":digest(out/"test_scores.parquet")})
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":main()
