"""Search DSSM train-only chronological preparation and isolated experiment."""
from pathlib import Path
import argparse,hashlib,json
import duckdb
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/"outputs/evaluation/optimization_20261031/search"

def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(16*1024*1024),b""):h.update(chunk)
    return h.hexdigest()

def prepare(output):
    output.mkdir(parents=True,exist_ok=True)
    if (output/"manifest.json").exists():raise FileExistsError("Refuse to replace Search manifest")
    train_path=ROOT/"datasets/search_train/train-00000-of-00001.parquet"
    test_path=ROOT/"datasets/search_test/train-00000-of-00001.parquet"
    features_path=ROOT/"features/search_train_features.parquet"
    raw=pd.read_parquet(train_path);test=pd.read_parquet(test_path)
    if set(raw.search_idx)&set(test.search_idx):raise ValueError("Official train/test request overlap")
    def interval(details):
        times=np.array([x["search_timestamp"] for x in details],dtype=float)
        if not len(times) or not np.isfinite(times).all():raise ValueError("Missing raw request event times")
        return float(times.min()),float(times.max())
    times=np.array([interval(items) for items in raw.search_result_details_with_idx])
    requests=raw[["search_idx","session_idx","user_idx","query","recent_clicked_note_idxs"]].copy()
    requests["lo"],requests["hi"]=times[:,0],times[:,1]
    test_start=min(interval(items)[0] for items in test.search_result_details_with_idx)
    sessions=requests.groupby("session_idx").agg(lo=("lo","min"),hi=("hi","max")).reset_index()
    past=sessions[sessions.hi<test_start].sort_values(["hi","session_idx"])
    index=max(1,min(len(past)-1,int(.8*len(past))))
    boundary=float(past.iloc[index].hi)
    sessions["partition"]="excluded"
    sessions.loc[sessions.hi<boundary,"partition"]="fit"
    sessions.loc[(sessions.lo>=boundary)&(sessions.hi<test_start),"partition"]="validation"
    requests=requests.merge(sessions[["session_idx","partition"]],on="session_idx",validate="many_to_one")
    requests["positives"]=[sorted({int(x["note_idx"]) for x in row if x["click"]>0}) for row in raw.search_result_details_with_idx]
    connection=duckdb.connect(config={"threads":"2","memory_limit":"2GB"})
    mismatches=connection.execute("""WITH r AS (
      SELECT search_idx,n.note_idx,max(n.click)::INT click FROM read_parquet(?) t,
      unnest(t.search_result_details_with_idx) a(n) GROUP BY search_idx,n.note_idx),
      f AS (SELECT search_idx,note_idx,max(click)::INT click FROM read_parquet(?) GROUP BY search_idx,note_idx)
      SELECT count(*) FROM r FULL JOIN f USING(search_idx,note_idx)
      WHERE r.click IS DISTINCT FROM f.click""",[str(train_path),str(features_path)]).fetchone()[0]
    if mismatches:raise ValueError("Raw/feature pair label mismatch")
    features=pd.read_parquet(features_path)
    if set(features.search_idx)!=set(raw.search_idx):raise ValueError("Raw/feature request mismatch")
    manifest={"scene":"search","raw_feature_label_mismatches":mismatches,"selection":"train-only chronological complete-session fit/validation; official test final acceptance only",
              "official_test_start":test_start,"validation_boundary":boundary,
              "limitations":["Provider raw profile/history snapshot event lineage unverified","Item cumulative count/rate features disabled","No candidate publication timestamps"],
              "inputs":{str(p.relative_to(ROOT)):digest(p) for p in [train_path,test_path,features_path]},"partitions":{}}
    for split in ("fit","validation"):
        frame=requests[requests.partition==split].sort_values("search_idx")
        path=output/f"requests_{split}.parquet";frame.to_parquet(path,index=False)
        f=features[features.search_idx.isin(frame.search_idx)]
        feat_path=output/f"features_{split}.parquet";f.to_parquet(feat_path,index=False)
        manifest["partitions"][split]={"requests":len(frame),"sessions":frame.session_idx.nunique(),"feature_rows":len(f),
          "positive_rows":int((f.click>0).sum()),"time_min":float(frame.lo.min()),"time_max":float(frame.hi.max()),
          "requests_path":str(path),"features_path":str(feat_path),"requests_sha256":digest(path),"features_sha256":digest(feat_path)}
    if manifest["partitions"]["fit"]["time_max"]>=manifest["partitions"]["validation"]["time_min"]:raise ValueError("Temporal overlap")
    requests.to_parquet(output/"partitions.parquet",index=False)
    manifest["excluded_requests"]=int((requests.partition=="excluded").sum())
    manifest["excluded_sessions"]=int((sessions.partition=="excluded").sum())
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(manifest,indent=2))

def train(output, training_source=None):
    if training_source is None:
        from optimize_rec_retrieval import train_dssm
    else:
        import importlib.util
        spec=importlib.util.spec_from_file_location("frozen_search_training",training_source)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.ROOT=ROOT
        train_dssm=module.train_dssm
    from recall.query_encoding import build_search_query_store
    fit=pd.read_parquet(output/"features_fit.parquet")
    validation=pd.read_parquet(output/"features_validation.parquet")
    raw=pd.read_parquet(ROOT/"datasets/search_train/train-00000-of-00001.parquet",columns=["search_idx","query"])
    query_store=build_search_query_store(raw,output/"query_cache/train")
    selected=train_dssm("search",fit,validation,output/"dssm",query_store=query_store)
    print(json.dumps(selected,indent=2),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=DEFAULT)
    p.add_argument("--stage",choices=["prepare","train"],required=True)
    p.add_argument("--training-source",type=Path,default=None,
                   help="Optional preserved training_source.py for exact original loss-version replay")
    args=p.parse_args()
    if args.stage=="train":train(args.output.resolve(),args.training_source)
    else:prepare(args.output.resolve())
