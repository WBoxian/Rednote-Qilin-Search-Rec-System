"""Recommendation recall attribution: frozen pure-three-route and deployed hybrid are separate."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import duckdb
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
ROUTES=("ann","swing","usercf")

def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(16*1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def recall_from_counts(frame, hit_column, count_column="positives"):
    selected=frame[frame[count_column]>0]
    if selected.empty:
        return {"all_requests":len(frame),"positive_requests":0,"HitRate":None,"macro_recall":None,"micro_recall":None,"hit_positive_pairs":0,"positive_pairs":0}
    return {"all_requests":len(frame),"positive_requests":len(selected),
            "HitRate":float((selected[hit_column]>0).mean()),
            "macro_recall":float((selected[hit_column]/selected[count_column]).mean()),
            "micro_recall":float(selected[hit_column].sum()/selected[count_column].sum()),
            "hit_positive_pairs":int(selected[hit_column].sum()),
            "positive_pairs":int(selected[count_column].sum())}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--leave-one-out",action="store_true",help="Fixed RRF weights; diagnostic removal of one route, no tuning")
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    base=ROOT/"outputs/evaluation/three_route/rec"
    raw_test=ROOT/"datasets/recommendation_test/train-00000-of-00001.parquet"
    raw_train=ROOT/"datasets/recommendation_train/train-00000-of-00001.parquet"
    deployed=ROOT/"outputs/data/recall_rec_test_hard_multiroute_top1000.parquet"
    inputs=[raw_test,raw_train,deployed,base/"manifest.json",Path(__file__)]
    inputs += [base/f"test_{route}.parquet" for route in ROUTES]
    before={str(p.relative_to(ROOT)):digest(p) for p in inputs}
    con=duckdb.connect(config={"threads":"2","memory_limit":"3GB"})
    con.execute("""CREATE TABLE req AS SELECT request_idx gid,user_idx,
        array_length(recent_clicked_note_idxs) history_length FROM read_parquet(?)""",[str(raw_test)])
    assert con.execute("SELECT count(*)=count(DISTINCT gid) FROM req").fetchone()[0]
    for split,path in (("test",raw_test),("train",raw_train)):
        con.execute(f"""CREATE TABLE labels_{split} AS SELECT r.request_idx gid,d.note_idx,
            first(r.user_idx) user_idx,max(d.click)::INT positive,
            min(d.request_timestamp) lo,max(d.request_timestamp) hi
            FROM read_parquet(?) r,unnest(r.rec_result_details_with_idx) u(d)
            GROUP BY r.request_idx,d.note_idx""",[str(path)])
    con.execute("CREATE TABLE positives AS SELECT * FROM labels_test WHERE positive=1")
    cutoff=json.loads((base/"manifest.json").read_text())["partition"]["fit_end_exclusive"]
    con.execute("CREATE TABLE fit_ids AS SELECT gid FROM labels_train GROUP BY gid HAVING max(hi)<?",[cutoff])
    con.execute("CREATE TABLE train_items AS SELECT DISTINCT note_idx FROM labels_train WHERE positive=1")
    con.execute("CREATE TABLE fit_items AS SELECT DISTINCT note_idx FROM labels_train JOIN fit_ids USING(gid) WHERE positive=1")
    con.execute("CREATE TABLE train_users AS SELECT DISTINCT user_idx FROM labels_train")
    for route in ROUTES:
        con.execute(f"""CREATE TABLE route_{route} AS SELECT gid,note_idx,score,
            row_number() OVER(PARTITION BY gid ORDER BY score DESC,note_idx) rank
            FROM read_parquet(?)""",[str(base/f"test_{route}.parquet")])
        assert con.execute(f"SELECT count(*)=count(DISTINCT(gid,note_idx)) FROM route_{route}").fetchone()[0]
    con.execute("""CREATE TABLE all_routes AS
        SELECT *,1 route FROM route_ann UNION ALL SELECT *,2 FROM route_swing
        UNION ALL SELECT *,4 FROM route_usercf""")
    con.execute("""CREATE TABLE pool AS SELECT gid,note_idx,
        sum(CASE route WHEN 2 THEN 4.0 ELSE 1.0 END/(60+rank)) score,
        bit_or(route) mask,\n        sum(CASE WHEN route=1 THEN 1.0/(60+rank) ELSE 0 END) ann_score,\n        sum(CASE WHEN route=2 THEN 4.0/(60+rank) ELSE 0 END) swing_score,\n        sum(CASE WHEN route=4 THEN 1.0/(60+rank) ELSE 0 END) usercf_score\n        FROM all_routes GROUP BY gid,note_idx""")
    con.execute("""CREATE TABLE selected AS SELECT gid,note_idx,
        row_number() OVER(PARTITION BY gid ORDER BY score DESC,note_idx) rank FROM pool""")
    con.execute("""CREATE TABLE deployed AS SELECT request_idx gid,note_idx,rank
        FROM read_parquet(?)""",[str(deployed)])
    con.execute("""CREATE TABLE hits AS SELECT p.gid,p.note_idx,p.user_idx,
        a.rank ann_rank,s.rank swing_rank,u.rank usercf_rank,c.rank fused_rank,d.rank deployed_rank,
        coalesce(pool.mask,0) mask,
        ti.note_idx IS NOT NULL seen_full_train_click,
        fi.note_idx IS NOT NULL seen_fit_click,
        tu.user_idx IS NOT NULL user_seen_full_train
        FROM positives p LEFT JOIN route_ann a USING(gid,note_idx)
        LEFT JOIN route_swing s USING(gid,note_idx)
        LEFT JOIN route_usercf u USING(gid,note_idx)
        LEFT JOIN selected c USING(gid,note_idx)
        LEFT JOIN pool USING(gid,note_idx)
        LEFT JOIN deployed d USING(gid,note_idx)
        LEFT JOIN train_items ti USING(note_idx)
        LEFT JOIN fit_items fi USING(note_idx)
        LEFT JOIN train_users tu USING(user_idx)""")
    expressions=["count(*) positives","count(*) FILTER(WHERE mask>0) union_hits"]
    for name in (*ROUTES,"fused","deployed"):
        expressions += [f"count(*) FILTER(WHERE {name}_rank<={k}) {name}_{k}" for k in (100,500,1000)]
    con.execute("CREATE TABLE counts AS SELECT gid,"+",".join(expressions)+" FROM hits GROUP BY gid")
    frame=con.execute("""SELECT r.*,coalesce(c.positives,0) positives,c.* EXCLUDE(gid,positives)
        FROM req r LEFT JOIN counts c USING(gid) ORDER BY gid""").fetchdf()
    for name in frame:
        if name not in ("gid","user_idx","history_length"): frame[name]=frame[name].fillna(0)
    frame.to_parquet(args.output/"request_counts.parquet",index=False)
    con.execute("COPY (SELECT * FROM hits ORDER BY gid,note_idx) TO ? (FORMAT PARQUET)",[str(args.output/"positive_hits.parquet")])
    result={"scene":"rec","scope":"all official test positive requests",
            "pure_three_route":{},"deployed_hybrid":{},"strata":{},
            "limitations":["Frozen route training lineage not independently rebuilt",
              "Union coverage uses up to 3000 candidates, not a same-budget Top500 metric",
              "Deployed hybrid is a different artifact/model protocol; do not compare its scores as causal improvement",
              "Full-train seen strata can include events later than test; fit strata are separate",
              "History arrays lack per-event timestamps; strata are descriptive, not causal"]}
    for route in (*ROUTES,"fused"):
        result["pure_three_route"][route]={f"Top{k}":recall_from_counts(frame,f"{route}_{k}") for k in (100,500,1000)}
    result["pure_three_route"]["union_max3000"]=recall_from_counts(frame,"union_hits")
    for k in (100,500,1000):result["deployed_hybrid"][f"Top{k}"]=recall_from_counts(frame,f"deployed_{k}")
    result["pure_three_route"]["source_stats"]={}
    for route in ROUTES:
        n,groups=con.execute(f"SELECT count(*),count(DISTINCT gid) FROM route_{route}").fetchone()
        result["pure_three_route"]["source_stats"][route]={"candidate_rows":n,"requests_with_candidates":groups,"empty_requests":len(frame)-groups}
    result["pure_three_route"]["positive_hit_masks"]=dict(con.execute("SELECT mask,count(*) FROM hits GROUP BY mask ORDER BY mask").fetchall())
    result["pure_three_route"]["exclusive_positive_pairs"]={
        route:con.execute("SELECT count(*) FROM hits WHERE mask=?",[1<<i]).fetchone()[0] for i,route in enumerate(ROUTES)}
    result["pure_three_route"]["candidate_union"]=con.execute("SELECT count(*) FROM pool").fetchone()[0]
    result["pure_three_route"]["candidate_sum"]=con.execute("SELECT count(*) FROM all_routes").fetchone()[0]
    result["pure_three_route"]["candidate_duplicates_removed"]=result["pure_three_route"]["candidate_sum"]-result["pure_three_route"]["candidate_union"]
    # Disjoint positive-pair miss decomposition, keeping macro and micro scopes separate.
    miss=con.execute("""SELECT count(*),
        count(*) FILTER(WHERE mask=0),
        count(*) FILTER(WHERE mask>0 AND (fused_rank>500 OR fused_rank IS NULL)),
        count(*) FILTER(WHERE fused_rank<=500) FROM hits""").fetchone()
    result["pure_three_route"]["positive_pair_loss_decomposition"]=dict(zip(("positive_pairs","not_in_any_route","in_union_but_below_fused500","fused500_hits"),miss))
    assert miss[0]==sum(miss[1:])
    for col in ("seen_full_train_click","seen_fit_click","user_seen_full_train"):
        result["strata"][col]=[]
        for value in (False,True):
            row=con.execute(f"""SELECT count(*) n,count(*) FILTER(WHERE fused_rank<=500) fused_hits,
                count(*) FILTER(WHERE deployed_rank<=500) deployed_hits,
                count(*) FILTER(WHERE mask>0) union_hits FROM hits WHERE {col}=?""",[value]).fetchone()
            result["strata"][col].append({"value":value,"positive_pairs":row[0],
                "pure_fused500_micro_recall":row[1]/row[0] if row[0] else None,
                "deployed500_micro_recall":row[2]/row[0] if row[0] else None,
                "union_micro_coverage":row[3]/row[0] if row[0] else None})
    frame["history_bin"]=pd.cut(frame.history_length.fillna(0),[-1,0,5,20,float("inf")],labels=["empty","1-5","6-20",">20"])
    result["strata"]["history_length"]={str(group):{
        "pure_fused500":recall_from_counts(part,"fused_500"),
        "deployed500":recall_from_counts(part,"deployed_500")}
        for group,part in frame.groupby("history_bin",observed=True)}
    # Pairwise request/item overlap at each route's full stored budget.
    result["pure_three_route"]["route_pair_overlap"]={}
    for a,b in (("ann","swing"),("ann","usercf"),("swing","usercf")):
        overlap=con.execute(f"SELECT count(*) FROM route_{a} JOIN route_{b} USING(gid,note_idx)").fetchone()[0]
        result["pure_three_route"]["route_pair_overlap"][f"{a}/{b}"]=overlap
    if args.leave_one_out:
        result["pure_three_route"]["leave_one_out_fixed_weights"]={}
        for index, route in enumerate(ROUTES):
            con.execute(f"""CREATE OR REPLACE TABLE without_route AS
                SELECT gid,note_idx,row_number() OVER(PARTITION BY gid ORDER BY
                    score-{route}_score DESC,note_idx) rank FROM pool WHERE (mask & {7^(1<<index)})>0""")
            remaining=con.execute("""SELECT r.gid,coalesce(t.n,0) positives,coalesce(t.h,0) hits
                FROM req r LEFT JOIN (
                    SELECT p.gid,count(*) n,count(*) FILTER(WHERE c.rank<=500) h
                    FROM positives p LEFT JOIN without_route c USING(gid,note_idx) GROUP BY p.gid
                ) t USING(gid) ORDER BY r.gid""").fetchdf()
            row=recall_from_counts(remaining,"hits")
            row["full_minus_without_macro"]=result["pure_three_route"]["fused"]["Top500"]["macro_recall"]-row["macro_recall"]
            result["pure_three_route"]["leave_one_out_fixed_weights"][route]=row
    assert before=={str(p.relative_to(ROOT)):digest(p) for p in inputs},"inputs drifted"
    (args.output/"metrics.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    (args.output/"manifest.json").write_text(json.dumps({"inputs":before,"selection":"fixed prior validation selection, no tuning"},indent=2)+"\n")
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=="__main__":main()
