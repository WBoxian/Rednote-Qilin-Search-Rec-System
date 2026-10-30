"""独立核验阶段留存的全量请求、原始正例与最终候选，不重新推理或改模型。"""
import argparse, json
from pathlib import Path
import duckdb, numpy as np, pandas as pd

ROOT=Path(__file__).resolve().parents[1]
def require(value,message):
    if not value:raise ValueError(message)
def verify(scene,run):
    folder,key,nested=("search","search_idx","search_result_details_with_idx") if scene=="search" else ("recommendation","request_idx","rec_result_details_with_idx")
    rows=pd.read_parquet(run/"requests.parquet")
    metrics=json.loads((run/"metrics.json").read_text())
    require(not rows.gid.duplicated().any(),"duplicate request ledger")
    with duckdb.connect(config={"threads":"2","memory_limit":"2GB"}) as con:
        con.execute(f"CREATE TABLE requests AS SELECT {key} gid,user_idx FROM read_parquet(?)",[str(ROOT/f"datasets/{folder}_test/*.parquet")])
        con.execute(f"""CREATE TABLE positives AS SELECT r.{key} gid,d.note_idx
            FROM read_parquet(?) r,unnest(r.{nested}) u(d)
            GROUP BY r.{key},d.note_idx HAVING max(d.click)>0""",[str(ROOT/f"datasets/{folder}_test/*.parquet")])
        con.register("ledger",rows)
        mismatch=con.execute("""SELECT count(*) FROM requests r FULL JOIN ledger l USING(gid)
            WHERE r.gid IS NULL OR l.gid IS NULL OR r.user_idx<>l.user_idx""").fetchone()[0]
        require(mismatch==0,"request/user coverage differs from raw data")
        truth=con.execute("SELECT gid,count(*) positives FROM positives GROUP BY gid").fetchdf().set_index("gid").positives
        require((rows.positives.to_numpy()==rows.gid.map(truth).fillna(0).to_numpy()).all(),"positive denominators differ")
        candidate_path=ROOT/f"outputs/data/recall_{scene}_test_hard_multiroute_top1000.parquet"
        con.execute(f"CREATE VIEW recall AS SELECT {key} gid,note_idx,rank FROM read_parquet('{candidate_path}')")
        for stage,limit in (("recall1000",1000),("recall_budget",metrics["recall_cap"]),("recall10",10)):
            hits=con.execute("SELECT p.gid,count(*) n FROM positives p JOIN recall c USING(gid,note_idx) WHERE c.rank<=? GROUP BY p.gid",[limit]).fetchdf().set_index("gid").n
            require((rows[stage+"_hits"].to_numpy()==rows.gid.map(hits).fillna(0).to_numpy()).all(),stage+" counts differ")
        con.execute(f"CREATE TEMP TABLE ranked AS SELECT *,{key} gid FROM read_parquet(?)",[str(run/"ranked_candidates.parquet")])
        require(con.execute("SELECT count(*)=count(DISTINCT(gid,note_idx)) FROM ranked").fetchone()[0],"duplicate final candidates")
        for stage,where in (("fine_budget","true"),("fine10","c.ranking_rank<=10"),("fine20","c.ranking_rank<=20")):
            hits=con.execute(f"SELECT p.gid,count(*) n FROM positives p JOIN ranked c USING(gid,note_idx) WHERE {where} GROUP BY p.gid").fetchdf().set_index("gid").n
            require((rows[stage+"_hits"].to_numpy()==rows.gid.map(hits).fillna(0).to_numpy()).all(),stage+" final candidates differ")
        valid=rows[rows.positives>0]
        for stage,value in metrics["stages"].items():
            expected=float((valid[stage+"_hits"]/valid.positives).mean())
            require(np.isclose(expected,value["macro_recall"],atol=1e-12,rtol=1e-12),stage+" macro mismatch")
            require(int(valid[stage+"_hits"].sum())==value["positive_pairs"],stage+" total hits mismatch")
        for before,after in (("recall1000","recall_budget"),("recall_budget","coarse_budget"),("coarse_budget","fine_budget"),("fine_budget","fine10")):
            require((rows[before+"_hits"]>=rows[after+"_hits"]).all(),"non-nested funnel positives")
    return {"scene":scene,"requests":len(rows),"positive_pairs":int(rows.positives.sum()),"status":"PASS",
            "coverage":"raw IDs/users/labels, recall counts, final candidate positives, all macro summaries, nested count invariants"}
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--search",type=Path,required=True);p.add_argument("--rec",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    result=[verify("search",a.search),verify("rec",a.rec)]
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
