"""Validation-only diagnostic: compare Search hard deletion with identical-budget soft policy."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import sys
from pathlib import Path
from unittest.mock import patch

import duckdb
import numpy as np
import pandas as pd

from backend.online.item_features import get_item_feature_store
from backend.online.preranking.gbdt import GBDTPredictor
from backend.online.preranking import service
from recall.build_multiroute_recall import _parse_scored_results, _normalize_route_scores

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    result=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(16*1024*1024),b""):
            result.update(chunk)
    return result.hexdigest()


def supplied_dpr_candidates(raw_requests):
    rows=[]
    for row in raw_requests.itertuples(index=False):
        scores=_normalize_route_scores(_parse_scored_results(row.dpr_results,limit=100))
        for rank,(item,score) in enumerate(scores.items(),1):
            rows.append({"search_idx":int(row.search_idx),"user_idx":int(row.user_idx),"note_idx":item,
                         "rank":rank,"recall_score":score,"score_ann":0.,"score_swing":score,"score_usercf":0.,
                         "from_ann":0,"from_swing":1,"from_usercf":0,"first_route":"dpr"})
    return pd.DataFrame(rows)


def canonical_candidates(frame, requests):
    frame=frame.rename(columns={"gid":"search_idx"}).copy()
    if "user_idx" not in frame:
        frame=frame.merge(requests[["search_idx","user_idx"]],on="search_idx",validate="many_to_one")
    if "recall_score" not in frame:
        frame["recall_score"]=frame["score"]
    for name in ("score_ann","score_swing","score_usercf","from_ann","from_swing","from_usercf"):
        if name not in frame:
            frame[name]=frame["recall_score"] if name=="score_ann" else (1 if name=="from_ann" else 0)
    if "first_route" not in frame:
        frame["first_route"]="ann"
    return frame


class RankStackPredictor(GBDTPredictor):
    """单树新模型adapter：不伪装成双树训练，不把零XGB作为第二模型。"""
    def __init__(self, run, split, query_cache, static_store):
        import lightgbm as lgb
        sys.path.insert(0,str(ROOT/"scripts"))
        from train_rank_stack import contexts, tree_predict
        self.manifest=json.loads((run/"manifest.json").read_text())
        if self.manifest["scene"]!="search":
            raise ValueError("Search policy requires Search rank stack")
        self.feature_cols=list(self.manifest["features"])
        self.model=lgb.Booster(model_file=str(run/"gbdt.txt"))
        self.model_path=run/"gbdt.txt"
        if digest(self.model_path)!=self.manifest["trees"]["model_sha256"]:
            raise ValueError("Rank-stack tree fingerprint mismatch")
        self.context=contexts("search","train" if split=="validation" else "test",query_cache)
        self.static_store=static_store
        self.thread_checks=0
        self.thread_verified=False
        self.score_tree=self._score_tree

    def _score_tree(self, model, frame, features):
        values=frame[features]
        single=model.predict(values,num_threads=1)
        if not self.thread_verified:
            reference=model.predict(values,num_threads=4)
            if not np.array_equal(single,reference):
                raise ValueError("Single-thread tree scores differ bitwise from 4-thread reference")
            self.thread_checks+=1
            self.thread_verified=self.thread_checks>=64
            if not self.thread_verified:
                single=reference
        if single.ndim==2:
            single=single[:,1]
        return np.asarray(single,dtype=np.float32)

    def predict_with_components(self, frame):
        features=self.context.feature_frame(frame,self.static_store)
        score=self.score_tree(self.model,features,self.feature_cols)
        return {"lgb_score":score,"xgb_score":np.zeros(len(score),np.float32),"gbdt_score":score}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--topn",type=int,default=80)
    parser.add_argument("--recall-cap",type=int,default=160)
    parser.add_argument("--max-groups",type=int,default=0)
    parser.add_argument("--recall-file",type=Path)
    parser.add_argument("--rank-stack-run",type=Path)
    parser.add_argument("--query-cache",type=Path)
    parser.add_argument("--split",choices=["validation","test"],default="validation")
    parser.add_argument("--selection",type=Path)
    args=parser.parse_args()
    if args.topn<=0 or args.recall_cap<args.topn:
        raise ValueError("A fixed positive candidate budget is required")
    selection=None
    if args.split=="test":
        if args.selection is None or args.max_groups:
            raise ValueError("Full final test requires a frozen validation --selection and no request sampling")
        selection=json.loads(args.selection.read_text())
        if selection["selection_split"]!="validation" or not selection.get("full_validation") or selection["topn"]!=args.topn or selection.get("recall_cap")!=args.recall_cap:
            raise ValueError("Invalid validation policy selection")
        if selection["candidate_mode"]!=("file" if args.recall_file else "dpr"):
            raise ValueError("Candidate source type differs from validation")
        if selection["service_sha256"]!=digest(ROOT/"src/backend/online/preranking/service.py"):
            raise ValueError("Search policy implementation changed after validation")
    requests=pd.read_parquet(args.requests)
    if "partition" not in requests or not requests.partition.eq(args.split).all():
        raise ValueError("Request partition must match the explicitly selected split")
    if args.max_groups>0 and len(requests)>args.max_groups:
        requests=requests.sample(args.max_groups,random_state=42)
    requests=requests.sort_values("search_idx").reset_index(drop=True)
    args.output.mkdir(parents=True,exist_ok=False)
    folder="search_train" if args.split=="validation" else "search_test"
    candidate_path=args.recall_file or next((ROOT/f"datasets/{folder}").glob("*.parquet"))
    feature_path=ROOT/("features/search_train_features.parquet" if args.split=="validation" else "features/search_test_features.parquet")
    with duckdb.connect(config={"threads":"2","memory_limit":"3GB"}) as con:
        con.register("requests",requests)
        if args.split=="test":
            ids=con.execute("SELECT search_idx FROM read_parquet(?)",[str(ROOT/"datasets/search_test/*.parquet")]).fetchnumpy()["search_idx"]
            if set(ids)!=set(requests.search_idx) or requests.search_idx.duplicated().any():
                raise ValueError("Final evaluation must cover every official test request")
        if args.rank_stack_run and args.split=="validation":
            train=pd.read_parquet(args.rank_stack_run/"train.parquet",columns=["search_idx","partition"])
            valid_ids=set(train.loc[train.partition=="validation","search_idx"])
            if not set(requests.search_idx)<=valid_ids:
                raise ValueError("Retrieval validation is not a subset of rank-stack validation")
        if args.recall_file:
            schema={row[0] for row in con.execute("DESCRIBE SELECT * FROM read_parquet(?)",[str(candidate_path)]).fetchall()}
            key="search_idx" if "search_idx" in schema else "gid"
            loaded=con.execute(f"SELECT c.* FROM read_parquet(?) c SEMI JOIN requests r ON c.{key}=r.search_idx ORDER BY c.{key},c.rank",[str(candidate_path)]).fetchdf()
            candidates=canonical_candidates(loaded,requests)
        else:
            raw=con.execute("SELECT c.search_idx,c.user_idx,c.dpr_results FROM read_parquet(?) c SEMI JOIN requests r USING(search_idx)",[str(candidate_path)]).fetchdf()
            candidates=supplied_dpr_candidates(raw)
        candidates=candidates[candidates["rank"]<=args.recall_cap].copy()
        features=con.execute("SELECT f.* FROM read_parquet(?) f SEMI JOIN requests r USING(search_idx)",[str(feature_path)]).fetchdf()
        ids=candidates[["note_idx"]].drop_duplicates()
        con.register("ids",ids)
        notes=con.execute("SELECT n.note_idx,n.note_title,n.note_content FROM read_parquet(?) n SEMI JOIN ids USING(note_idx)",[str(ROOT/"datasets/notes/*.parquet")]).fetchdf().set_index("note_idx")
    if candidates.search_idx.nunique()<.95*len(requests):
        raise ValueError("Insufficient validation candidate coverage; refusing sparse replay")
    if candidates.duplicated(["search_idx","note_idx"]).any():
        raise ValueError("Duplicate candidates")
    candidate_groups=dict(tuple(candidates.groupby("search_idx",sort=False)))
    feature_groups=dict(tuple(features.groupby("search_idx",sort=False)))
    store=get_item_feature_store(ROOT)
    model_paths=[]
    if args.rank_stack_run:
        model_manifest=json.loads((args.rank_stack_run/"manifest.json").read_text())
        query_cache=args.query_cache
        if query_cache is None and args.split=="validation":
            query_cache=Path(model_manifest["query_cache"])
        if query_cache is None:
            raise ValueError("New Search stack requires explicit query cache for this split")
        predictor=RankStackPredictor(args.rank_stack_run,args.split,query_cache,store)
        model_paths=[args.rank_stack_run/"manifest.json",predictor.model_path]
        if selection and selection["model_sha256"]!=digest(predictor.model_path):
            raise ValueError("Model changed after validation")
        if selection:
            if not selection.get("thread_probe",{}).get("verified"):
                raise ValueError("New model thread configuration was not verified on validation")
            predictor.thread_checks=selection["thread_probe"]["requests"]
            predictor.thread_verified=True
    else:
        predictor=GBDTPredictor("search","hard",ROOT/"outputs/deploy/search/hard","search_idx",list(features))
        if predictor.lgb_model is None or predictor.xgb_model is None:
            raise ValueError("This fixed legacy-model comparison requires both tree models")
        model_paths=[ROOT/"outputs/deploy/search/hard/models/lgb_search_hard.pkl",
                     ROOT/"outputs/deploy/search/hard/models/xgb_search_hard.pkl"]
        if selection and selection["model_sha256"]!="legacy":
            raise ValueError("Model differs from validation")
    original=service.run_preranking_gbdt
    rows=[]
    started=time.monotonic()
    for index,request in requests.iterrows():
        gid=int(request.search_idx)
        recalled=candidate_groups.get(gid,candidates.iloc[:0])
        profile=feature_groups.get(gid,features.iloc[:0])
        captured=[]
        def capture(*args,**kwargs):
            out=original(*args,**kwargs)
            captured.append(out.copy())
            return out
        def fetch_notes(note_ids):
            return notes.reindex(note_ids).reset_index()
        with patch.object(service,"run_preranking_gbdt",side_effect=capture):
            soft=service.run_preranking(
                int(request.user_idx),str(request.query or ""),"search","search_idx",
                recalled,profile,args.topn,fetch_notes,predictor.predict,
                fetch_item_features=store.get,search_filter_policy="soft",allow_snapshot_quality_boost=False)
        if captured and str(request.query or "").strip():
            legacy=service.apply_search_preranking_policy(captured[0],args.topn,"legacy")
        else:
            legacy=soft
        positives=set(int(x) for x in request.positives)
        for policy,frame in (("legacy",legacy),("soft",soft)):
            row={"search_idx":gid,"user_idx":int(request.user_idx),"policy":policy,
                 "positives":len(positives),"returned":len(frame),"budget":min(args.topn,len(recalled))}
            for k in (10,20):
                hits=len(positives&set(int(x) for x in frame.head(k).get("note_idx",pd.Series(dtype="int64"))))
                row[f"Recall@{k}"]=hits/len(positives) if positives else np.nan
                row[f"HitRate@{k}"]=float(hits>0) if positives else np.nan
            rows.append(row)
        if index%100==0:
            print(f"{args.split} requests={index+1}/{len(requests)}",flush=True)
    frame=pd.DataFrame(rows)
    frame.to_parquet(args.output/"requests.parquet",index=False)
    summaries={}
    for policy,group in frame.groupby("policy"):
        summaries[policy]={"requests":len(group),"positive_requests":int((group.positives>0).sum()),
                          "mean_returned":float(group.returned.mean()),
                          "shortfall_requests":int((group.returned<group.budget).sum()),
                          **{key:float(group[key].mean()) for key in ("Recall@10","Recall@20","HitRate@10","HitRate@20")}}
    paired=frame.pivot(index=["search_idx","user_idx"],columns="policy",values=["Recall@10","Recall@20"])
    intervals={}
    for k in (10,20):
        delta=(paired[(f"Recall@{k}","soft")]-paired[(f"Recall@{k}","legacy")]).dropna()
        grouped=delta.groupby(level="user_idx").agg(["sum","count"]).to_numpy()
        rng=np.random.default_rng(42)
        estimates=[]
        for _ in range(500):
            sampled=grouped[rng.integers(0,len(grouped),len(grouped))]
            estimates.append(float(sampled[:,0].sum()/sampled[:,1].sum()))
        intervals[f"Recall@{k}"]={"macro_delta":float(delta.mean()),"user_cluster_CI95":[float(x) for x in np.quantile(estimates,[.025,.975])]}
    chosen=max(summaries,key=lambda policy:(summaries[policy]["Recall@20"],summaries[policy]["Recall@10"],policy=="soft")) if args.split=="validation" else selection["selected_policy"]
    model_hash=digest(predictor.model_path) if args.rank_stack_run else "legacy"
    if args.split=="validation":
        frozen={"selection_split":"validation","selected_policy":chosen,"criterion":"Recall@20, then Recall@10, then soft",
                "topn":args.topn,"recall_cap":args.recall_cap,"thread_probe":{"requests":predictor.thread_checks,"verified":predictor.thread_verified} if args.rank_stack_run else None,"full_validation":args.max_groups==0,"candidate_mode":"file" if args.recall_file else "dpr",
                "model_sha256":model_hash,"service_sha256":digest(ROOT/"src/backend/online/preranking/service.py"),
                "validation_metrics":str(args.output/"metrics.json")}
        (args.output/"selection.json").write_text(json.dumps(frozen,ensure_ascii=False,indent=2)+"\n")
    report={"scope":f"{args.split} policy diagnostic; no model deployment",
            "selected_policy":selection["selected_policy"] if selection else chosen,
            "selection_split":"validation","rank_stack_run":str(args.rank_stack_run) if args.rank_stack_run else None,
            "candidate_budget":args.topn,"recall_cap":args.recall_cap,"candidate_source":str(args.recall_file) if args.recall_file else f"provided {folder} DPR100 only; BM25 excluded from this diagnostic","request_sampling":f"all {args.split}" if args.max_groups==0 else f"at most {args.max_groups}, seed=42",
            "thread_probe":{"requests":predictor.thread_checks,"verified":predictor.thread_verified} if args.rank_stack_run else None,
            "seconds":time.monotonic()-started,"policies":summaries,"paired_recall_gain":intervals,
            "inputs":{str(path):digest(path) for path in [
                args.requests,candidate_path,feature_path,Path(__file__),
                ROOT/"src/backend/online/preranking/service.py",
                *model_paths]},
            "limitations":["Provided content/history snapshots lack event-time provenance; legacy trees are diagnostic, new-tree lineage is in its manifest",
                           "Only hard deletion changes; upstream GBDT topn truncation and lexical soft scoring stay identical",
                           "Production policy is not activated by this script; test cannot select or overwrite validation policy"]}
    (args.output/"metrics.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    main()
