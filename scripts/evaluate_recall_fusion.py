import argparse, ast, json, hashlib, time
import pandas as pd
from pathlib import Path
import duckdb, numpy as np
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description="Frozen three-route fusion ablation; validation-only parameter selection.")
parser.add_argument("--output",type=Path,required=True)
args=parser.parse_args()
OUTPUT=args.output
OUTPUT.mkdir(parents=True,exist_ok=False)
BASE=ROOT/"outputs/evaluation/three_route/rec"
ns={"np":np}
for name,path in (("legacy_offline","src/recall/build_multiroute_recall.py"),("legacy_online","src/backend/online/recall/service.py")):
    tree=ast.parse((ROOT/path).read_text(encoding="utf-8-sig"))
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="_merge_scores")
    function.name=name
    class Prefix(ast.NodeTransformer):
        def visit_Compare(self,node):
            self.generic_visit(node)
            if isinstance(node.left,ast.Call) and isinstance(node.left.func,ast.Name) and node.left.func.id=="len":
                if any(isinstance(arg,ast.Name) and arg.id=="merged" for arg in node.left.args):
                    node.comparators=[ast.Constant(value=500) if isinstance(n,(ast.Name,ast.Call)) else n for n in node.comparators]
            return node
    function=Prefix().visit(function)
    ast.fix_missing_locations(function)
    exec(compile(ast.Module(body=[function],type_ignores=[]),path,"exec"),ns)
con=duckdb.connect(config={"threads":"4","memory_limit":"5GB"})
out={"selection_split":"validation","protocol":"frozen temporal_three_route_v1 route replay","limitations":["Existing training/index lineage not independently rebuilt","Static catalog publication availability and history timestamps unverified","Compared on frozen pure three-route candidates; NOT original deployed hybrid route files"],"validation":{},"test":{}}
def metric_sql(expression):
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE metric_requests AS
    WITH ranked AS (SELECT gid,note_idx,row_number() OVER(PARTITION BY gid ORDER BY {expression} DESC,note_idx) rk FROM pool)
    SELECT p.gid,p.n,count(*) FILTER(WHERE c.rk<=500) h,min(c.rk) first_hit
    FROM (SELECT gid,array_length(positives) n,unnest(positives) note_idx FROM req) p
    LEFT JOIN ranked c USING(gid,note_idx) GROUP BY p.gid,p.n
    """)
    row=con.execute("""
    SELECT count(*),avg((h>0)::INT),avg(h/n),
       avg(CASE WHEN first_hit<=100 THEN 1.0/first_hit ELSE 0 END) FROM metric_requests
    """).fetchone()
    return dict(zip(("requests","HitRate@500","Recall@500","MRR@100"),row))
def legacy(split):
    # Fetch request aggregates in bounded batches instead of materializing all Python candidate dictionaries.
    tables=[]
    for route in ("ann","swing","usercf"):
        con.execute(f"CREATE OR REPLACE TEMP TABLE grouped_{route} AS SELECT gid,list(note_idx ORDER BY rank) items,list(score ORDER BY rank) scores FROM route_{route} GROUP BY gid")
    cursor=con
    query="""SELECT r.gid,r.positives,a.items,a.scores,s.items,s.scores,u.items,u.scores FROM req r
        LEFT JOIN grouped_ann a USING(gid) LEFT JOIN grouped_swing s USING(gid) LEFT JOIN grouped_usercf u USING(gid) ORDER BY gid"""
    cursor.execute(query)
    sums={name:np.zeros(3) for name in ("legacy_offline","legacy_online")}
    n=0
    records=[]
    while True:
        batch=cursor.fetchmany(64)
        if not batch: break
        for row in batch:
            gid,positives,*values=row
            routes=[dict(zip(values[i] or [],values[i+1] or [])) for i in (0,2,4)]
            recalls={}
            for name in sums:
                options=dict(ann_scores=routes[0],swing_scores=routes[1],usercf_scores=routes[2],
                    merge_order=["swing","usercf","ann"],topk=1000,route_min_quota=32,route_max_share=.82,
                    route_weights={"ann":1.,"swing":1.,"usercf":1.})
                if name=="legacy_online": options["rrf_k"]=60
                result=ns[name](**options)
                ranks={item[0]:rank for rank,item in enumerate(result,1)}
                hits=[ranks[p] for p in positives if p in ranks]
                h=sum(rank<=500 for rank in hits)
                first=min(hits,default=1001)
                recalls[name]=h/len(positives)
                sums[name]+=np.array([h>0,h/len(positives),1/first if first<=100 else 0])
            records.append((gid,recalls["legacy_offline"],recalls["legacy_online"]))
            n+=1
        if n%1024<64: print("baseline",n,flush=True)
    pd.DataFrame(records,columns=["gid","legacy_offline","legacy_online"]).to_parquet(OUTPUT/f"{split}_baseline_requests.parquet",index=False)
    return {name:{"requests":n,**dict(zip(("HitRate@500","Recall@500","MRR@100"),(value/n).tolist()))} for name,value in sums.items()}
inputs=sorted(BASE.glob("*_ann.parquet"))+sorted(BASE.glob("*_swing.parquet"))+sorted(BASE.glob("*_usercf.parquet"))+sorted(BASE.glob("requests_*.parquet"))
inputs+=[BASE/"manifest.json",ROOT/"src/recall/fusion.py",ROOT/"src/recall/build_multiroute_recall.py",ROOT/"src/backend/online/recall/service.py",Path(__file__)]
def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):h.update(block)
    return h.hexdigest()
manifest={str(p.relative_to(ROOT)):digest(p) for p in inputs}
(OUTPUT/"manifest.json").write_text(json.dumps(manifest,indent=2))
selected=None
for split in ("validation","test"):
    print("split",split,flush=True)
    con.execute("CREATE OR REPLACE TEMP TABLE req AS SELECT request_idx gid,user_idx,positives FROM read_parquet(?)",[str(BASE/f"requests_{split}.parquet")])
    raw_split="train" if split=="validation" else "test"
    raw_path=ROOT/f"datasets/recommendation_{raw_split}/*.parquet"
    con.execute("""CREATE OR REPLACE TEMP TABLE raw_labels AS
        SELECT r.request_idx gid,list_sort(list(DISTINCT d.note_idx) FILTER(WHERE d.click>0)) positives
        FROM read_parquet(?) r,unnest(r.rec_result_details_with_idx) u(d)
        GROUP BY r.request_idx""",[str(raw_path)])
    assert con.execute("SELECT count(*) FROM req r LEFT JOIN raw_labels l USING(gid) WHERE l.gid IS NULL OR list_sort(r.positives)<>l.positives").fetchone()[0]==0
    assert con.execute("SELECT count(*) FROM req WHERE array_length(positives)=0").fetchone()[0]==0
    if split=="test":
        assert con.execute("SELECT count(*) FROM req").fetchone()[0]==con.execute("SELECT count(*) FROM raw_labels").fetchone()[0]
    for route in ("ann","swing","usercf"):
        con.execute(f"CREATE OR REPLACE TEMP TABLE route_{route} AS SELECT * FROM read_parquet(?)",[str(BASE/f"{split}_{route}.parquet")])
    # Re-rank tied route scores deterministically, matching weighted_rrf().
    con.execute("""CREATE OR REPLACE TEMP TABLE pool AS WITH all_routes AS (
        SELECT gid,note_idx,score,'ann' route FROM route_ann
        UNION ALL SELECT gid,note_idx,score,'swing' FROM route_swing
        UNION ALL SELECT gid,note_idx,score,'usercf' FROM route_usercf),
        ordered AS (SELECT *,row_number() OVER(PARTITION BY gid,route ORDER BY score DESC,note_idx) rk FROM all_routes)
        SELECT gid,note_idx,
          max(CASE WHEN route='ann' THEN 1.0/(60+rk) ELSE 0 END) a,
          max(CASE WHEN route='swing' THEN 1.0/(60+rk) ELSE 0 END) s,
          max(CASE WHEN route='usercf' THEN 1.0/(60+rk) ELSE 0 END) u
        FROM ordered GROUP BY gid,note_idx""")
    cache=OUTPUT/f"legacy_{split}.json"
    if cache.exists():
        out[split].update(json.loads(cache.read_text()))
    else:
        baseline=legacy(split)
        cache.write_text(json.dumps(baseline))
        out[split].update(baseline)
    grid=[(1,1,1),(2,1,1),(1,2,1),(1,1,2),(4,1,1),(1,4,1),(1,1,4)] if split=="validation" else [selected]
    for a,s,u in grid:
        print("weights",a,s,u,flush=True)
        result=metric_sql(f"{a}*a+{s}*s+{u}*u")
        out[split][f"rrf_{a}_{s}_{u}"]=result
        print(result,flush=True)
    if split=="validation":
        selected=max(grid,key=lambda w:(out[split][f"rrf_{w[0]}_{w[1]}_{w[2]}"]["Recall@500"],
             out[split][f"rrf_{w[0]}_{w[1]}_{w[2]}"]["HitRate@500"],
             out[split][f"rrf_{w[0]}_{w[1]}_{w[2]}"]["MRR@100"]))
        out["selected_weights"]=selected
    a,s,u=selected
    metric_sql(f"{a}*a+{s}*s+{u}*u")
    selected_requests=con.execute("SELECT m.gid,r.user_idx,m.h/m.n selected FROM metric_requests m JOIN req r USING(gid) ORDER BY gid").fetchdf()
    baseline_requests=pd.read_parquet(OUTPUT/f"{split}_baseline_requests.parquet")
    paired=selected_requests.merge(baseline_requests,on="gid",validate="one_to_one")
    assert len(paired)==con.execute("SELECT count(*) FROM req").fetchone()[0]
    paired.to_parquet(OUTPUT/f"{split}_paired_requests.parquet",index=False)
    intervals={}
    for baseline in ("legacy_offline","legacy_online"):
        paired["delta"]=paired["selected"]-paired[baseline]
        grouped=paired.groupby("user_idx").delta.agg(["sum","count"]).to_numpy()
        rng=np.random.default_rng(42)
        values=[]
        for _ in range(500):
            sampled=grouped[rng.integers(0,len(grouped),len(grouped))]
            values.append(float(sampled[:,0].sum()/sampled[:,1].sum()))
        intervals[baseline]={"paired_macro_delta":float(paired.delta.mean()),
          "user_cluster_CI95":[float(x) for x in np.quantile(values,[.025,.975])],
          "bootstrap_repetitions":500,"seed":42}
    out[split]["paired_delta"]=intervals
    (OUTPUT/"metrics.json").write_text(json.dumps(out,ensure_ascii=False,indent=2))
    print("finished",split,flush=True)
assert manifest=={str(p.relative_to(ROOT)):digest(p) for p in inputs},"Fusion inputs changed during evaluation"
print(json.dumps(out,indent=2),flush=True)
