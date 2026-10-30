"""Validation-only Search DPR/ANN RRF selection, then frozen full-test fusion."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from optimize_rec_retrieval import metrics,write_json,digest
from recall.fusion import weighted_rrf
ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/"outputs/evaluation/optimization_20261031/search"
WEIGHTS=(0.,.1,.25,.5,1.)

def routes(raw,ann):
    dense={int(row.search_idx):[int(pair[0]) for pair in row.dpr_results] for row in raw.itertuples()}
    lexical={int(row.search_idx):[int(pair[0]) for pair in row.bm25_results] for row in raw.itertuples()} if "bm25_results" in raw else {}
    neural={int(gid):part.sort_values("rank").note_idx.astype(int).tolist() for gid,part in ann.groupby("gid",sort=False)}
    return dense,lexical,neural

def fused(gid,channels,weight):
    dpr,bm25,ann=channels
    def rank_scores(items):
        result={}
        for rank,item in enumerate(items,1):
            result.setdefault(item,1./rank)
        return result
    rows=weighted_rrf(rank_scores(ann.get(int(gid),[])) if weight else {},
        rank_scores(dpr.get(int(gid),[])),rank_scores(bm25.get(int(gid),[])),
        topk=500,route_weights={"ann":weight,"swing":1.,"usercf":1.},rrf_k=60)
    scores={row[0]:row[1] for row in rows}
    flags={row[0]:{name for name,flag in zip(("ann","dpr","bm25"),row[5:8]) if flag} for row in rows}
    return [row[0] for row in rows],scores,flags


def select(output):
    target=output/"fusion";target.mkdir(exist_ok=True)
    if (target/"selection.json").exists():raise RuntimeError("Fusion selection is already frozen")
    req=pd.read_parquet(output/"requests_validation.parquet").rename(columns={"search_idx":"gid"})
    raw=pd.read_parquet(ROOT/"datasets/search_train/train-00000-of-00001.parquet")
    raw=raw[raw.search_idx.isin(req.gid)]
    ann_path=output/"index/validation_ivfpq.parquet"
    channels=routes(raw,pd.read_parquet(ann_path))
    if set(channels[2])!=set(req.gid):raise ValueError("ANN validation coverage incomplete")
    records=[];grid={};best=None
    for weight in WEIGHTS:
        totals=[]
        for row in req.itertuples():
            order,_,_=fused(row.gid,channels,weight);truth=set(row.positives)
            positions={item:rank for rank,item in enumerate(order,1)}
            hits=[positions[item] for item in truth if item in positions]
            first=min(hits,default=10**12)
            totals.append((len(hits)/len(truth),bool(hits),1./first if first<=100 else 0.))
        a=np.array(totals);score={"Recall@500":float(a[:,0].mean()),"HitRate@500":float(a[:,1].mean()),"MRR@100":float(a[:,2].mean())}
        grid[str(weight)]=score
        if weight==0:baseline_mrr=score["MRR@100"]
        if score["MRR@100"]>=baseline_mrr-1e-12:
            objective=(score["Recall@500"],score["MRR@100"],-weight)
            if best is None or objective>best:best=objective;chosen=weight
    selection={"ann_weight":chosen,"dpr_weight":1.,"test_bm25_weight":1.,"rrf_k":60,"budget":500,
      "selection":"full train-only validation macro Recall500; MRR100 must not regress vs fixed DPR100; tie smaller ANN weight",
      "limitations":["Official train provides DPR only; BM25 absent on validation, test BM25 is a fixed additive reference with weight1, never tuned on test","Provider retrieval lists have unverified upstream training lineage"],
      "grid":grid,"validation_requests":len(req),"ann_candidate_sha256":digest(ann_path)}
    write_json(target/"selection.json",selection)
    save_candidates(req.gid,channels,chosen,target/"validation_selected.parquet")
    print(json.dumps(selection,indent=2),flush=True)

def save_candidates(gids,channels,weight,path):
    records=[]
    for gid in gids:
        order,scores,flags=fused(gid,channels,weight)
        records.extend((int(gid),item,scores[item],rank,int("ann" in flags[item]),int("dpr" in flags[item]),int("bm25" in flags[item])) for rank,item in enumerate(order,1))
    result=pd.DataFrame(records,columns=["gid","note_idx","score","rank","from_ann","from_dpr","from_bm25"])
    result.to_parquet(path,index=False)
    return result

def protected_order(gid,channels,heads=None):
    dpr,bm25,ann=channels
    head=(heads or dpr).get(int(gid),[])
    result=list(dict.fromkeys(head))
    if len(result)>500:raise ValueError("Protected reference head exceeds final budget")
    seen=set(result)
    for item in ann.get(int(gid),[]):
        if item not in seen:
            result.append(item);seen.add(item)
        if len(result)==500:break
    return result


def protected_candidates(gids,channels,heads=None):
    rows=[];dpr,bm25,ann=channels
    for gid in gids:
        sources=[set(route.get(int(gid),[])) for route in (ann,dpr,bm25)]
        rows.extend((int(gid),item,1./rank,rank,*[int(item in source) for source in sources])
                    for rank,item in enumerate(protected_order(gid,channels,heads),1))
    return pd.DataFrame(rows,columns=["gid","note_idx","score","rank","from_ann","from_dpr","from_bm25"])


def select_protected(output):
    target=output/"fusion"
    path=target/"selection_headfill_v2.json"
    if path.exists():raise RuntimeError("Protected-head policy is already frozen")
    req=pd.read_parquet(output/"requests_validation.parquet").rename(columns={"search_idx":"gid"})
    raw=pd.read_parquet(ROOT/"datasets/search_train/train-00000-of-00001.parquet")
    raw=raw[raw.search_idx.isin(req.gid)]
    channels=routes(raw,pd.read_parquet(output/"index/validation_ivfpq.parquet"))
    candidates=protected_candidates(req.gid,channels)
    score,per_request=metrics(req,candidates)
    grid=json.loads((target/"selection.json").read_text())
    baseline=grid["grid"]["0.0"]
    if score["MRR@100"]<baseline["MRR@100"]-1e-12 or score["Recall@500"]<=baseline["Recall@500"]:
        raise RuntimeError("Protected-head policy did not pass predeclared validation guardrails")
    selection={"strategy":"protected_provider_head_then_ann_fill","budget":500,
      "validation":score,"validation_provider_baseline":baseline,"ann_index":"IVFPQ2048,PQ32x8,nprobe32",
      "test_reference_order":"Frozen original provider BM25/DPR-membership candidates in original saved rank order; ANN only deduplicated tail fill",
      "score_semantics":"1/rank of the protected ordering, not raw cosine or provider score",
      "provider_validation_lineage":"UNKNOWN; operational comparison conditional on supplied provider candidates, not a clean independently held-out provider validation",
      "selection":"Only train-internal temporal validation; protected-provider prefix preserves MRR100 and ANN fills remaining candidate budget",
      "rrf_grid_rejected":"All nonzero ANN weights reduced validation MRR100; original grid and failed metrics retained in selection.json",
      "ann_candidate_sha256":digest(output/"index/validation_ivfpq.parquet")}
    write_json(path,selection)
    candidates.to_parquet(target/"validation_selected.parquet",index=False)
    candidates.to_parquet(target/"validation_headfill_selected.parquet",index=False)
    per_request.to_parquet(target/"validation_headfill_metrics.parquet",index=False)
    print(json.dumps(selection,indent=2),flush=True)


def final_protected(output):
    target=output/"fusion";selection=target/"selection_headfill_v2.json"
    if (target/"test_results.json").exists():raise RuntimeError("Final Search test is already frozen; use a new experiment directory")
    if not selection.exists():raise RuntimeError("Freeze validation head-fill policy before test")
    raw=pd.read_parquet(ROOT/"datasets/search_test/train-00000-of-00001.parquet")
    ann=pd.read_parquet(output/"index/test_ivfpq.parquet");channels=routes(raw,ann)
    original_path=ROOT/"outputs/data/recall_search_test_hard_multiroute_top1000.parquet"
    frozen=pd.read_parquet(original_path,columns=["search_idx","note_idx","rank","from_swing","from_usercf"]).rename(columns={"search_idx":"gid"})
    provider=frozen[(frozen.from_swing>0)|(frozen.from_usercf>0)].sort_values(["gid","rank"])
    heads={int(gid):part.note_idx.astype(int).tolist() for gid,part in provider.groupby("gid",sort=False)}
    for row in raw.itertuples():
        expected=set(channels[0][int(row.search_idx)])|set(channels[1][int(row.search_idx)])
        if set(heads.get(int(row.search_idx),[]))!=expected:raise ValueError("Frozen provider head differs from raw provider union")
    req=raw[["search_idx","user_idx"]].rename(columns={"search_idx":"gid"})
    req["positives"]=[sorted({int(item["note_idx"]) for item in details if item["click"]>0}) for details in raw.search_result_details_with_idx]
    selected=protected_candidates(req.gid,channels,heads);selected.to_parquet(target/"test_selected.parquet",index=False)
    references={"dpr_only":protected_candidates(req.gid,(channels[0],{},{})),
                "provider_head":protected_candidates(req.gid,(channels[0],channels[1],{}),heads),"new_ann_ivfpq":ann,
                "original_frozen":frozen[frozen["rank"]<=500].assign(score=lambda frame:1./frame["rank"]),"new_headfill":selected}
    result={"selection_sha256":digest(selection),"original_reference_sha256":digest(original_path),"metrics":{}}
    for name,candidates in references.items():
        score,per_request=metrics(req,candidates);result["metrics"][name]=score
        per_request.to_parquet(target/f"test_{name}_metrics.parquet",index=False)
        if name in ("dpr_only","provider_head"):candidates.to_parquet(target/f"test_{name}.parquet",index=False)
    result["selected_candidates_sha256"]=digest(target/"test_selected.parquet")
    write_json(target/"test_results.json",result);print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,default=DEFAULT)
    parser.add_argument("--stage",choices=["select","select-protected","test-protected"],required=True);args=parser.parse_args()
    {"select":select,"select-protected":select_protected,"test-protected":final_protected}[args.stage](args.output)