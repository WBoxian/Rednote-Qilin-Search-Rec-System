"""全量Search召回归因；数据提供方列表和冻结候选成员贡献分开报告。"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(16*1024*1024),b""): h.update(block)
    return h.hexdigest()

def metrics(positives, ranked):
    rows=[]
    for gid, truth in positives.items():
        ordered=ranked.get(gid,[])
        positions={item:index+1 for index,item in enumerate(ordered)}
        first=min((positions[x] for x in truth if x in positions),default=None)
        row={"gid":gid,"positives":len(truth),"candidates":len(ordered)}
        for k in (100,500,1000):
            hits=sum(positions.get(item,k+1)<=k for item in truth)
            row[f"hits@{k}"]=hits
            row[f"Recall@{k}"]=hits/len(truth) if truth else None
            row[f"HitRate@{k}"]=int(hits>0) if truth else None
        row["MRR@100"]=1/first if first is not None and first<=100 else 0.
        rows.append(row)
    frame=pd.DataFrame(rows)
    summary={"all_requests":len(frame),"positive_requests":int((frame.positives>0).sum()),
             "positive_pairs":int(frame.positives.sum()),"empty_candidates":int((frame.candidates==0).sum()),
             "candidate_count_median":float(frame.candidates.median())}
    valid=frame[frame.positives>0]
    for key in ("HitRate@100","HitRate@500","HitRate@1000","Recall@100","Recall@500","Recall@1000","MRR@100"):
        summary[key]=float(valid[key].mean()) if len(valid) else None
    for k in (100,500,1000):
        summary[f"MicroRecall@{k}"]=float(valid[f"hits@{k}"].sum()/valid.positives.sum()) if len(valid) else None
    return summary,frame

def parse_candidates(values):
    seen=set(); result=[]
    for pair in values:
        item=int(pair[0])
        if item>=0 and item not in seen:
            seen.add(item);result.append(item)
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    paths=[*sorted((ROOT/"datasets/search_test").glob("*.parquet")),
           *sorted((ROOT/"datasets/search_train").glob("*.parquet")),
           ROOT/"outputs/data/recall_search_test_hard_multiroute_top1000.parquet"]
    fingerprints={str(p.relative_to(ROOT)):{"bytes":p.stat().st_size,"sha256":digest(p)} for p in paths}
    requests=pd.read_parquet(paths[0])
    require=len(requests)==requests.search_idx.nunique()==6192
    if not require:raise ValueError("Unexpected or duplicated Search request population")
    positives={}
    raw={"bm25":{},"dpr":{},"bm25_dpr_union":{}}
    for row in requests.itertuples():
        gid=int(row.search_idx)
        positives[gid]={int(x["note_idx"]) for x in row.search_result_details_with_idx if x["click"]>0}
        raw["bm25"][gid]=parse_candidates(row.bm25_results)
        raw["dpr"][gid]=parse_candidates(row.dpr_results)
        raw["bm25_dpr_union"][gid]=list(dict.fromkeys(raw["bm25"][gid]+raw["dpr"][gid]))
    output={"scene":"search","scope":"all official test requests; raw binary click; no sampling",
            "limitations":["Raw BM25/DPR lists are supplied by dataset; training/publication-time lineage unverified",
                           "Frozen route flags measure retained membership, NOT independently regenerated full single-route TopK",
                           "Union coverage is an oracle upper bound, not a deployable ordering metric"],
            "supplied_routes":{},"frozen_candidates":{}}
    raw_frames={}
    for name,ranked in raw.items():
        summary,frame=metrics(positives,ranked)
        raw_frames[name]=frame
        output["supplied_routes"][name]=summary
        frame.to_parquet(args.output/f"supplied_{name}_requests.parquet",index=False)
    with duckdb.connect(config={"threads":"2","memory_limit":"3GB"}) as con:
        con.execute("CREATE VIEW c AS SELECT * FROM read_parquet('"+str(paths[-1])+"')")
        con.execute("""CREATE TABLE labels AS SELECT r.search_idx gid,d.note_idx,max(d.click)>0 positive
            FROM read_parquet(?) r,unnest(r.search_result_details_with_idx) u(d)
            GROUP BY r.search_idx,d.note_idx""",[str(paths[0])])
        con.execute("CREATE TABLE p AS SELECT * FROM labels WHERE positive")
        if con.execute("SELECT count(*)-count(DISTINCT(search_idx,note_idx)) FROM c").fetchone()[0]:
            raise ValueError("Duplicate frozen request-item candidates")
        if con.execute("SELECT count(*)-count(DISTINCT(search_idx,rank)) FROM c").fetchone()[0]:
            raise ValueError("Duplicate frozen request ranks")
        if con.execute("SELECT count(*) FROM c WHERE rank<1 OR rank>1000 OR note_idx<0 OR NOT isfinite(recall_score)").fetchone()[0]:
            raise ValueError("Invalid frozen candidate values")
        known=set(positives)
        unknown=con.execute("SELECT DISTINCT search_idx FROM c").fetchnumpy()["search_idx"]
        if not set(int(x) for x in unknown)<=known:raise ValueError("Unknown candidate request")
        output["frozen_candidates"]["candidate_rows"]=con.execute("SELECT count(*) FROM c").fetchone()[0]
        hits=con.execute("""SELECT p.gid,p.note_idx,c.rank,c.from_ann,c.from_swing AS from_dpr,c.from_usercf AS from_bm25,
            c.first_route FROM p LEFT JOIN c ON c.search_idx=p.gid AND c.note_idx=p.note_idx ORDER BY p.gid,p.note_idx""").fetchdf()
        hits["rank"]=hits["rank"].astype(float)
        hits.to_parquet(args.output/"positive_hits.parquet",index=False)
        full=con.execute("""SELECT search_idx,list(note_idx ORDER BY rank) items FROM c GROUP BY search_idx""").fetchall()
        merged={int(gid):list(items) for gid,items in full}
        summary,per_request=metrics(positives,merged)
        output["frozen_candidates"]["merged"]=summary
        per_request.to_parquet(args.output/"merged_requests.parquet",index=False)
        positive_rows=len(hits)
        contributions={}
        for route in ("ann","dpr","bm25"):
            flag=hits["from_"+route].fillna(0).astype(bool)
            other=[hits["from_"+r].fillna(0).astype(bool) for r in ("ann","dpr","bm25") if r!=route]
            exclusive=flag & ~other[0] & ~other[1]
            by_group=hits.assign(hit=flag.astype(int)).groupby("gid").agg(hit=("hit","sum"),positives=("note_idx","size"))
            exclusive_by=hits.assign(hit=exclusive.astype(int)).groupby("gid").agg(hit=("hit","sum"),positives=("note_idx","size"))
            contributions[route]={"retained_positive_pairs":int(flag.sum()),"exclusive_positive_pairs":int(exclusive.sum()),
                "retained_macro_positive_coverage":float((by_group["hit"]/by_group.positives).mean()),
                "exclusive_macro_positive_coverage":float((exclusive_by["hit"]/exclusive_by.positives).mean())}
        output["frozen_candidates"]["retained_route_membership"]=contributions
        output["frozen_candidates"]["membership_patterns"]=con.execute("""
            SELECT from_ann,from_swing AS from_dpr,from_usercf AS from_bm25,count(*) candidates
            FROM c GROUP BY ALL ORDER BY candidates DESC""").fetchdf().to_dict("records")
        output["frozen_candidates"]["positive_membership_patterns"]=hits.groupby(
            ["from_ann","from_dpr","from_bm25"],dropna=False).size().reset_index(name="positive_pairs").fillna(-1).to_dict("records")
        inv=con.execute("""SELECT count(*) transitions,count(*) FILTER(WHERE recall_score>previous) score_increases,
            count(DISTINCT search_idx) FILTER(WHERE recall_score>previous) affected_requests
            FROM (SELECT search_idx,recall_score,lag(recall_score) OVER(PARTITION BY search_idx ORDER BY rank) previous FROM c)
            WHERE previous IS NOT NULL""").fetchone()
        output["frozen_candidates"]["fused_score_order_audit"]=dict(zip(("transitions","score_increases","affected_requests"),inv))
        train_seen=con.execute("""SELECT DISTINCT d.note_idx FROM read_parquet(?) r,
            unnest(r.search_result_details_with_idx) u(d) WHERE d.click>0""",[str(paths[1])]).fetchnumpy()["note_idx"]
        hits["seen_train_click"]=hits.note_idx.isin(train_seen)
        output["positive_item_strata"]={}
        for seen,group in hits.groupby("seen_train_click"):
            output["positive_item_strata"]["seen" if seen else "unseen"]={
                "positive_pairs":len(group),"micro_hit500":float((group["rank"]<=500).mean()),
                "micro_hit1000":float((group["rank"]<=1000).mean())}
        meta=requests[["search_idx","user_idx","query","recent_clicked_note_idxs"]].copy().rename(columns={"search_idx":"gid"})
        meta["history_length"]=meta.recent_clicked_note_idxs.map(len)
        meta["query_length"]=meta["query"].str.len()
        merged_frame=per_request.merge(meta[["gid","user_idx","history_length","query_length"]],on="gid",validate="one_to_one")
        output["paired_gain_CI95"]={}
        user_map=meta.set_index("gid").user_idx
        for label,left,right in (
            ("DPR100_minus_BM25_100",raw_frames["dpr"],raw_frames["bm25"]),
            ("union_minus_DPR100",raw_frames["bm25_dpr_union"],raw_frames["dpr"]),
            ("frozen500_minus_supplied_union",per_request,raw_frames["bm25_dpr_union"]),
        ):
            joined=left[["gid","Recall@500"]].merge(right[["gid","Recall@500"]],on="gid",suffixes=("_left","_right"),validate="one_to_one")
            joined["user_idx"]=joined.gid.map(user_map)
            joined["delta"]=joined["Recall@500_left"]-joined["Recall@500_right"]
            grouped=joined.groupby("user_idx").delta.agg(["sum","count"]).to_numpy()
            rng=np.random.default_rng(42); estimates=[]
            for _ in range(500):
                sample=grouped[rng.integers(0,len(grouped),len(grouped))]
                estimates.append(float(sample[:,0].sum()/sample[:,1].sum()))
            output["paired_gain_CI95"][label]={"macro_delta":float(joined.delta.mean()),
                "user_bootstrap_CI95":[float(x) for x in np.quantile(estimates,[.025,.975])],"replicates":500,"seed":42}
        output["request_strata"]={}
        for name,mask in (("empty_history",merged_frame.history_length==0),("history_1_5",merged_frame.history_length.between(1,5)),
                          ("history_6_plus",merged_frame.history_length>5),("query_len_le4",merged_frame.query_length<=4),
                          ("query_len_5_8",merged_frame.query_length.between(5,8)),("query_len_ge9",merged_frame.query_length>=9)):
            frame=merged_frame[mask & (merged_frame.positives>0)]
            output["request_strata"][name]={"requests":len(frame),"Recall@500":float(frame["Recall@500"].mean()) if len(frame) else None,
                                         "HitRate@500":float(frame["HitRate@500"].mean()) if len(frame) else None}
        # Removing supplied routes is evaluated as lost positive coverage of the fixed merged pool, not regenerated recall.
        for route in ("bm25","dpr"):
            joined=raw[route]
            recovered=sum(len(positives[gid]&set(items)) for gid,items in joined.items())
            output["supplied_routes"].setdefault(route,{})["positive_pairs_hit"]=recovered
    if fingerprints!={str(p.relative_to(ROOT)):{"bytes":p.stat().st_size,"sha256":digest(p)} for p in paths}:
        raise ValueError("inputs changed during attribution")
    manifest={"inputs":fingerprints,"source_sha256":digest(Path(__file__)),"sampling":"none"}
    (args.output/"metrics.json").write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    (args.output/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(output,ensure_ascii=False,indent=2))

if __name__=="__main__":main()
