"""Quantify GBDT-to-DIEN gains on complete logged-exposure pools, separately by scene."""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import duckdb,numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score
from backend.offline.evaluate_full_test import ranking_metrics,confidence

ROOT=Path(__file__).resolve().parents[1]
def paired_summary(frame):
    valid=frame.dropna(subset=["delta"])
    return {"requests":len(valid),"mean_ndcg_delta":float(valid.delta.mean()),
            "delta_user_cluster_ci95":confidence(valid,"delta"),
            "win_requests":int((valid.delta>1e-10).sum()),"tie_requests":int((valid.delta.abs()<=1e-10).sum()),
            "loss_requests":int((valid.delta< -1e-10).sum()),
            "gbdt_ndcg10":float(valid.gbdt_ndcg.mean()),"dien_ndcg10":float(valid.dien_ndcg.mean())}

def retention(frame,score,ks=(5,10,20,50,100)):
    out=frame[["gid","user_idx","positive"]].copy();out["score"]=score
    out["start"]=out.groupby("gid").score.rank(method="min",ascending=False)
    out["ties"]=out.groupby(["gid","score"]).score.transform("size")
    grouped=out.groupby("gid").agg(user_idx=("user_idx","first"),positives=("positive","sum"))
    result={}
    for k in ks:
        out["gain"]=out.positive*np.clip((k-out.start+1)/out.ties,0,1)
        grouped["hits"]=out.groupby("gid").gain.sum()
        denominator=grouped.positives.replace(0,np.nan)
        result[str(k)]={"macro_positive_impression_retention":float((grouped.hits/denominator).mean()),
                        "micro_positive_impression_retention":float(grouped.hits.sum()/grouped.positives.sum()),
                        "expected_positive_impressions_retained":float(grouped.hits.sum())}
    return result

def metric_block(frame):
    scores={}
    for stage in ("gbdt","dien"):
        y=frame.positive.to_numpy();p=frame[stage+"_score"].to_numpy()
        auc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None
        user=[]
        for _,g in frame.groupby("user_idx"):
            if g.positive.nunique()==2:user.append((roc_auc_score(g.positive,g[stage+"_score"]),len(g)))
        scores[stage]={"rows":len(frame),"positive_impressions":int(frame.positive.sum()),
                      "AUC":auc,"GAUC":float(np.average([x[0] for x in user],weights=[x[1] for x in user])) if user else None}
    return scores

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,default=ROOT/"outputs/evaluation/full_test_20261030_contract_d")
    parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    summary={"scope":"complete original logged exposure pools; NOT actual full-catalog serving funnel",
             "limitations":["Conditional association is not causal feature importance",
                            "Unrecalled logged rows have explicit zero recall-route inputs",
                            "Pre-exposure availability and model training lineage are unverified",
                            "Repeated request-item exposures remain separate impression samples"],
             "scenes":{}}
    for scene in ("search","rec"):
        key="search_idx" if scene=="search" else "request_idx";folder="search" if scene=="search" else "recommendation"
        nested="search_result_details_with_idx" if scene=="search" else "rec_result_details_with_idx"
        p=args.run/scene
        gb=pd.read_parquet(p/"gbdt_predictions.parquet").rename(columns={"score":"gbdt_score"})
        di=pd.read_parquet(p/"dien_predictions.parquet").rename(columns={"score":"dien_score"})
        frame=gb.merge(di[["gid","exposure_index","dien_score"]],on=["gid","exposure_index"],validate="one_to_one")
        assert len(frame)==len(gb)==len(di)
        req_gb=pd.read_parquet(p/"gbdt_requests.parquet").rename(columns={"ndcg@10":"gbdt_ndcg"})
        req_di=pd.read_parquet(p/"dien_requests.parquet").rename(columns={"ndcg@10":"dien_ndcg"})
        req=req_gb.merge(req_di[["gid","dien_ndcg"]],on="gid",validate="one_to_one")
        req["delta"]=req.dien_ndcg-req.gbdt_ndcg
        metadata=pd.read_parquet(ROOT/f"datasets/{folder}_test/train-00000-of-00001.parquet",columns=[key,"recent_clicked_note_idxs"])
        metadata=metadata.rename(columns={key:"gid"});metadata["history_length"]=metadata.recent_clicked_note_idxs.apply(lambda x:len(x) if x is not None else 0)
        req=req.merge(metadata[["gid","history_length"]],on="gid",validate="one_to_one")
        with duckdb.connect(config={"threads":"2","memory_limit":"3GB"}) as con:
            con.register("pred",frame)
            routes=con.execute(f"""SELECT p.gid,p.exposure_index,(r.note_idx IS NOT NULL) recalled
                FROM pred p LEFT JOIN read_parquet(?) r ON r.{key}=p.gid AND r.note_idx=p.note_idx""",
                [str(ROOT/f"outputs/data/recall_{scene}_test_hard_multiroute_top1000.parquet")]).fetchdf()
            raw=con.execute(f"""SELECT r.{key} gid,ord exposure_index,d.position
                FROM read_parquet(?) r,unnest(r.{nested}) WITH ORDINALITY u(d,ord)""",
                [str(ROOT/f"datasets/{folder}_test/*.parquet")]).fetchdf()
            seen=con.execute("SELECT DISTINCT note_idx FROM read_parquet(?) WHERE click>0",
                             [str(ROOT/f"features/{scene}_train_features.parquet")]).fetchnumpy()["note_idx"]
        frame=frame.merge(routes,on=["gid","exposure_index"],validate="one_to_one").merge(raw,on=["gid","exposure_index"],validate="one_to_one")
        frame["seen_item"]=frame.note_idx.isin(seen)
        frame["position_bucket"]=pd.cut(frame.position,[-np.inf,3,10,20,np.inf],labels=["1-3","4-10","11-20",">20"])
        req["history_bucket"]=pd.cut(req.history_length,[-np.inf,0,5,20,np.inf],labels=["empty","1-5","6-20",">20"])
        req["pool_bucket"]=pd.cut(req.exposures,[0,10,20,50,np.inf],labels=["1-10","11-20","21-50",">50"])
        req["positive_bucket"]=pd.cut(req.positives,[0,1,3,10,np.inf],labels=["1","2-3","4-10",">10"])
        result={"paired_ndcg":paired_summary(req),"global_auc":metric_block(frame),
                "stage_retention":{stage:retention(frame,frame[stage+"_score"]) for stage in ("gbdt","dien")},
                "strata":{}}
        for field in ("history_bucket","pool_bucket","positive_bucket"):
            result["strata"][field]={str(value):paired_summary(group) for value,group in req.groupby(field,observed=True)}
        for field in ("recalled","seen_item","position_bucket"):
            result["strata"][field]={str(value):metric_block(group) for value,group in frame.groupby(field,observed=True)}
        position_metric,_,_=ranking_metrics(frame,-frame.position.to_numpy(dtype=float))
        result["historical_position_only_baseline"]={k:position_metric[k] for k in ("NDCG@10","AUC","GAUC")}
        result["pool_stats"]={"median_exposures":float(req.exposures.median()),"mean_exposures":float(req.exposures.mean()),
                              "requests_with_at_most_10_exposures":int((req.exposures<=10).sum()),
                              "requests_with_at_most_20_exposures":int((req.exposures<=20).sum()),
                              "random_expected_ndcg10":float(req["random_ndcg@10"].mean())}
        req.to_parquet(args.output/f"{scene}_paired_requests.parquet",index=False)
        summary["scenes"][scene]=result
        (args.output/"metrics.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
        print(scene,json.dumps(result["paired_ndcg"]),flush=True)
    files=[]
    for scene in ("search","rec"):
        folder="search" if scene=="search" else "recommendation"
        files += [args.run/scene/f"{stage}_{kind}.parquet" for stage in ("gbdt","dien") for kind in ("predictions","requests")]
        files += list((ROOT/f"datasets/{folder}_test").glob("*.parquet"))
        files += [ROOT/f"features/{scene}_train_features.parquet",
                  ROOT/f"outputs/data/recall_{scene}_test_hard_multiroute_top1000.parquet"]
    files += [args.run/"manifest.json",args.run/"metrics.json"]
    manifest={"source":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "inputs":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
if __name__=="__main__":main()
