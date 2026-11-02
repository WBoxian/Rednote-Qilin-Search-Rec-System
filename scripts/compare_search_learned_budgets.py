"""Predeclared Search learned-only 160-vs-500 input-budget ablation, coarse budget fixed at 80."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from train_rank_stack import contexts
from backend.online.item_features import get_item_feature_store

CAPS=(160,500)
COARSE=80


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(16*1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def budget_guard(baseline, wider):
    return wider["Recall@10"]>baseline["Recall@10"] and wider["Recall@20"]>=baseline["Recall@20"]


def load_pool(path, requests):
    with duckdb.connect(config={"threads":"2","memory_limit":"3GB"}) as con:
        con.register("requests",requests[["search_idx"]])
        schema={row[0] for row in con.execute("DESCRIBE SELECT * FROM read_parquet(?)",[str(path)]).fetchall()}
        key="gid" if "gid" in schema else "search_idx"
        frame=con.execute(f"""
            SELECT c.{key} gid,c.note_idx,c.rank
            FROM read_parquet(?) c SEMI JOIN requests r ON c.{key}=r.search_idx
            WHERE c.rank<=500 ORDER BY c.{key},c.rank
        """,[str(path)]).fetchdf()
    if frame.duplicated(["gid","note_idx"]).any() or frame.duplicated(["gid","rank"]).any():
        raise ValueError("Duplicate candidates")
    groups=frame.groupby("gid")["rank"].agg(["size","min","max"])
    if set(groups.index)!=set(requests.search_idx) or not ((groups["size"]==500)&(groups["min"]==1)&(groups["max"]==500)).all():
        raise ValueError("Every official request must have the frozen unique Top500 pool")
    return frame


def summarize_budget(frame, requests, scores, cap, coarse=COARSE):
    selected=frame[frame["rank"]<=cap].copy()
    if len(selected)!=len(scores):
        raise ValueError("Predictions do not cover candidate budget")
    selected["score"]=scores
    pairs=[(int(row.search_idx),int(item)) for row in requests.itertuples(index=False) for item in row.positives]
    positive_index=pd.MultiIndex.from_tuples(pairs,names=["gid","note_idx"])
    selected["known_positive"]=pd.MultiIndex.from_frame(selected[["gid","note_idx"]]).isin(positive_index)
    pool=selected.groupby("gid").known_positive.sum()
    ordered=selected.sort_values(["gid","score","rank","note_idx"],ascending=[True,False,True,True],kind="stable")
    ordered["learned_rank"]=ordered.groupby("gid").cumcount()+1
    per_request=requests[["search_idx","user_idx"]].rename(columns={"search_idx":"gid"}).copy()
    per_request["positive_count"]=[len(set(int(item) for item in values)) for values in requests.positives]
    denominator=per_request.positive_count.replace(0,np.nan)
    per_request["pool_recall"]=per_request.gid.map(pool).fillna(0)/denominator
    for k in (10,20,coarse):
        hits=ordered[ordered.learned_rank<=k].groupby("gid").known_positive.sum()
        per_request[f"Recall@{k}"]=per_request.gid.map(hits).fillna(0)/denominator
    summary={"all_requests":len(per_request),"positive_requests":int((per_request.positive_count>0).sum()),
             "candidate_cap":cap,"coarse_budget":coarse,"pool_recall":float(per_request.pool_recall.mean()),
             **{f"Recall@{k}":float(per_request[f"Recall@{k}"].mean()) for k in (10,20,coarse)}}
    return summary,per_request,selected


def write_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=["validation","test"],default="validation")
    parser.add_argument("--requests",type=Path,required=True)
    parser.add_argument("--rank-stack-run",type=Path,required=True)
    parser.add_argument("--new-recall-file",type=Path,required=True)
    parser.add_argument("--old-recall-file",type=Path)
    parser.add_argument("--query-cache",type=Path)
    parser.add_argument("--selection",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    started=time.monotonic()
    model_path=args.rank_stack_run/"gbdt.txt"
    model_manifest=json.loads((args.rank_stack_run/"manifest.json").read_text())
    model_hash=digest(model_path)
    if model_manifest["scene"]!="search" or model_hash!=model_manifest["trees"]["model_sha256"]:
        raise ValueError("Search model fingerprint mismatch")
    selection=None
    if args.stage=="test":
        if args.selection is None or args.old_recall_file is None:
            raise ValueError("Test requires frozen validation selection and both old/new candidate pools")
        selection=json.loads(args.selection.read_text())
        if not selection["guard_passed"] or selection["selected_cap"]!=500:
            raise ValueError("Validation rejected the larger budget; test evaluation forbidden")
        if selection["model_sha256"]!=model_hash or selection["features"]!=model_manifest["features"]:
            raise ValueError("Model changed after validation")
        if selection["script_sha256"]!=digest(Path(__file__)):
            raise ValueError("Budget procedure changed after validation")
    requests=pd.read_parquet(args.requests).sort_values("search_idx").reset_index(drop=True)
    if requests.search_idx.duplicated().any() or not requests.partition.eq(args.stage).all():
        raise ValueError("Complete requests must belong to the explicitly selected partition")
    if args.stage=="validation":
        train=pd.read_parquet(args.rank_stack_run/"train.parquet",columns=["search_idx","partition"])
        if not set(requests.search_idx)<=set(train.loc[train.partition=="validation","search_idx"]):
            raise ValueError("Retrieval validation must be a subset of rank-stack validation")
    else:
        raw_ids=pd.read_parquet(ROOT/"datasets/search_test/train-00000-of-00001.parquet",columns=["search_idx"])
        if set(raw_ids.search_idx)!=set(requests.search_idx):
            raise ValueError("All official test requests must be present")
    args.output.mkdir(parents=True,exist_ok=False)
    cache=args.query_cache or (Path(model_manifest["query_cache"]) if args.stage=="validation" else None)
    if cache is None:
        raise ValueError("Explicit final-test raw-query cache required")
    context=contexts("search","train" if args.stage=="validation" else "test",cache)
    store=get_item_feature_store(ROOT)
    model=lgb.Booster(model_file=str(model_path))
    features=model_manifest["features"]
    report={"stage":args.stage,"scope":"learned GBDT only; NO lexical soft rules, hard gate or DIEN",
            "caps_predeclared":[160,500],"score_dtype":"float32 matching root tree_predict","coarse_budget":80,"requests":len(requests),
            "model_sha256":model_hash,"features":features,"pools":{},"inputs":{}}
    paths=[args.requests,model_path,args.rank_stack_run/"manifest.json",Path(__file__),
           ROOT/"src/training/rank_stack.py",ROOT/"scripts/train_rank_stack.py",
           ROOT/"src/backend/online/item_features.py",*context.text.files,*store.input_files]
    paths+=list(cache.glob("*.json"))+list(cache.glob("*.npy")) if cache.is_dir() else [cache,cache.with_suffix(".npy")]
    pools=[("new",args.new_recall_file)]
    if args.stage=="test":
        pools.append(("old",args.old_recall_file))
        paths.append(args.selection)
    for name,path in pools:
        paths.append(path)
        frame=load_pool(path,requests)
        print(f"{args.stage}/{name}: shared feature extraction for {len(frame)} candidates",flush=True)
        feature_started=time.monotonic()
        enriched=context.feature_frame(frame.rename(columns={"gid":"search_idx"}),store)
        feature_seconds=time.monotonic()-feature_started
        if not np.isfinite(enriched[features].to_numpy()).all():
            raise ValueError("Nonfinite model features")
        caps=CAPS if args.stage=="validation" else (500,)
        results={}
        predictions={}
        for cap in caps:
            mask=frame["rank"]<=cap
            score_started=time.monotonic()
            scores=np.asarray(model.predict(enriched.loc[mask,features],num_threads=4),dtype=np.float32)
            if not np.isfinite(scores).all():
                raise ValueError("Nonfinite tree predictions")
            score_seconds=time.monotonic()-score_started
            summary,per_request,pred=summarize_budget(frame,requests,scores,cap)
            summary["tree_predict_seconds"]=score_seconds
            summary["shared_full500_feature_seconds"]=feature_seconds
            results[str(cap)]=summary
            predictions[cap]=scores
            per_request.to_parquet(args.output/f"{name}_cap{cap}_requests.parquet",index=False)
            if cap==500:
                pred.to_parquet(args.output/f"{name}_full500_predictions.parquet",index=False)
            print(f"{args.stage}/{name}/cap{cap}: {json.dumps(summary)}",flush=True)
        if args.stage=="validation":
            if not np.array_equal(predictions[160],predictions[500][frame["rank"].to_numpy()<=160]):
                raise ValueError("Changing candidate budget changed overlapping item scores")
            report["overlap_scores_bitwise_equal"]=True
        report["pools"][name]=results
        del enriched,frame
    if args.stage=="validation":
        passed=budget_guard(report["pools"]["new"]["160"],report["pools"]["new"]["500"])
        selected={"selection_split":"validation","guard":"R10_500 > R10_160 AND R20_500 >= R20_160",
                  "guard_passed":bool(passed),"selected_cap":500 if passed else 160,
                  "coarse_budget":80,"features":features,"model_sha256":model_hash,
                  "script_sha256":digest(Path(__file__)),"validation_requests":len(requests)}
        write_json(args.output/"selection.json",selected)
        report["selection"]=selected
    else:
        report["selection"]=selection
    report["inputs"]={str(path):digest(path) for path in dict.fromkeys(paths)}
    report["total_elapsed_seconds"]=time.monotonic()-started
    report["timing_scope"]="Shared full500 feature extraction measured once, tree timing per budget; not online request latency"
    report["limitations"]=["Provided catalog/history snapshots and provider candidates lack complete event-time lineage",
                           "Official test was previously diagnosed; this is full regression, not a newly untouched holdout",
                           "Known-positive pool reachability and model ranking are separate; no unknown-item negatives fabricated"]
    write_json(args.output/"metrics.json",report)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    main()
