"""Repair only five normalized search-query cache misses; derive a new full result without editing originals."""
import argparse,copy,json,hashlib,time
from pathlib import Path
import duckdb,numpy as np,pandas as pd,torch
from backend.online.pipeline import SceneServingState
from backend.online.preranking.service import run_preranking
from backend.online.ranking.service import run_ranking
from backend.online.search_query import normalize_query_text
from backend.offline.evaluate_full_test import confidence,sha256
from replay_stage_funnel import stage_row

ROOT=Path(__file__).resolve().parents[1]
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(42);torch.backends.cudnn.benchmark=False
    key="search_idx";nested="search_result_details_with_idx"
    raw=pd.concat([pd.read_parquet(p) for p in sorted((ROOT/"datasets/search_test").glob("*.parquet"))],ignore_index=True)
    cache={str(q or "").strip() for q in raw["query"]}
    misses=raw[raw["query"].map(lambda q:bool(normalize_query_text(q)) and normalize_query_text(q) not in cache)]
    if len(misses)!=5:raise ValueError(f"Expected fixed 5 misses, got {len(misses)}")
    base_manifest=json.loads((args.base/"manifest.json").read_text())
    sources={name:value for name,value in base_manifest["inputs"].items() if name.endswith(".py") or "models--BAAI" in name or "/models/" in name or "/datasets/search_test/" in name or "/features/search_test_" in name or "/datasets/notes/" in name}
    for name,value in sources.items():
        if sha256(Path(name))!=value:raise ValueError("Source/input changed: "+name)
    own_sha=sha256(Path(__file__))
    base_files={name:sha256(args.base/name) for name in ("manifest.json","metrics.json","requests.parquet","ranked_candidates.parquet")}
    state=SceneServingState("search","hard",500,1000);state.realtime_cache=None
    predictor=state.dien_predictor;predictor.query_model.to("cpu")
    if predictor.query_model is None:raise ValueError("Missing CPU query encoder")
    cap=min(state.live_recall_rank_cap,160);pre_cap=min(state.live_gbdt_topn,80)
    ids=misses[key].astype(int).tolist()
    with duckdb.connect(config={"threads":"2","memory_limit":"2GB"}) as con:
        con.register("wanted",pd.DataFrame({key:ids}))
        candidates=con.execute(f"SELECT c.* FROM read_parquet(?) c JOIN wanted USING({key}) WHERE rank<=? ORDER BY {key},rank",
                              [str(ROOT/"outputs/data/recall_search_test_hard_multiroute_top1000.parquet"),cap]).fetchdf()
    rows=[];scores=[];queries=[]
    for record in misses.to_dict("records"):
        gid=int(record[key]);user=int(record["user_idx"]);query=str(record["query"] or "")
        normalized=normalize_query_text(query)
        vector=predictor._encode_query_vec(normalized)
        if vector is None or not np.isfinite(vector).all() or np.linalg.norm(vector)<=0:raise ValueError("Correct CPU query encoding failed")
        queries.append({"gid":gid,"raw_query":query,"normalized_query":normalized,"vector_norm":float(np.linalg.norm(vector))})
        recall=candidates[candidates[key]==gid].copy()
        features=state.get_feat_req(gid)
        pre=run_preranking(user,query,"search",key,recall,features,pre_cap,state._fetch_notes,state.predict_gbdt)
        pre=pre.reset_index(drop=True);pre["preranking_rank"]=np.arange(1,len(pre)+1)
        history=[int(x) for x in record["recent_clicked_note_idxs"] if int(x)>=0] or [-1]
        raw_scores=state.predict_dien(pre,history_note_ids=history)
        pre["_offline_dien_score"]=raw_scores
        ranked,_=run_ranking(pre,1,20,lambda df,**kwargs:df["_offline_dien_score"].to_numpy(np.float32),history_note_ids=history)
        positive={int(item["note_idx"]) for item in record[nested] if item["click"]>0}
        stages={"recall_budget":recall.note_idx.tolist(),"recall10":recall.note_idx.head(10).tolist(),
                "coarse_budget":pre.note_idx.tolist(),"coarse10":pre.note_idx.head(10).tolist(),
                "fine_budget":ranked.note_idx.tolist(),"fine10":ranked.note_idx.head(10).tolist(),"fine20":ranked.note_idx.head(20).tolist()}
        rows.append(stage_row(gid,user,positive,stages))
        compact=ranked[[key,"note_idx","preranking_rank","gbdt_score","dien_score","final_score","_offline_dien_score"]].copy()
        compact["ranking_rank"]=np.arange(1,len(compact)+1);scores.append(compact)
        print(json.dumps({"patched_gid":gid,"normalized":normalized}),flush=True)
    original_rows=pd.read_parquet(args.base/"requests.parquet")
    patch_rows=pd.DataFrame(rows)
    reference=original_rows.set_index("gid")
    for suffix in ("hits","recall","hit","size"):
        patch_rows["recall1000_"+suffix]=patch_rows.gid.map(reference["recall1000_"+suffix])
    updated=original_rows.copy().set_index("gid")
    patch_indexed=patch_rows.set_index("gid")
    for col in patch_indexed.columns:updated.loc[patch_indexed.index,col]=patch_indexed[col]
    updated=updated.reset_index()[original_rows.columns]
    unchanged=~original_rows.gid.isin(ids)
    pd.testing.assert_frame_equal(original_rows.loc[unchanged].reset_index(drop=True),updated.loc[unchanged].reset_index(drop=True))
    original_scores=pd.read_parquet(args.base/"ranked_candidates.parquet")
    patch_scores=pd.concat(scores,ignore_index=True)
    updated_scores=pd.concat([original_scores[~original_scores[key].isin(ids)],patch_scores],ignore_index=True).sort_values([key,"ranking_rank"],kind="stable").reset_index(drop=True)
    patch_rows.to_parquet(args.output/"patched_requests.parquet",index=False)
    patch_scores.to_parquet(args.output/"patched_ranked_candidates.parquet",index=False)
    updated.to_parquet(args.output/"requests.parquet",index=False)
    updated_scores.to_parquet(args.output/"ranked_candidates.parquet",index=False)
    result=copy.deepcopy(json.loads((args.base/"metrics.json").read_text()))
    result["query_encoding_repair"]={"requests":5,"unmodified_requests":len(updated)-5,"encoder":"CPU BAAI/bge-base-zh, normalized query","queries":queries}
    for stage in result["stages"]:
        result["stages"][stage]={"macro_recall":float(updated[stage+"_recall"].mean()),"micro_recall":float(updated[stage+"_hits"].sum()/updated.positives.sum()),
                                "hit_rate":float(updated.loc[updated.positives>0,stage+"_hit"].mean()),"positive_pairs":int(updated[stage+"_hits"].sum()),
                                "candidate_rows":int(updated[stage+"_size"].sum())}
    for name in result["paired_gains"]:
        after,before=name.split("-")
        work=updated[["user_idx"]].copy();work["delta"]=updated[after+"_recall"]-updated[before+"_recall"]
        result["paired_gains"][name]={"macro_delta":float(work.delta.mean()),"user_cluster_CI95":confidence(work,"delta")}
    original=json.loads((args.base/"metrics.json").read_text())
    result["repair_effect_on_full_test"]={stage:{"macro_recall_delta":result["stages"][stage]["macro_recall"]-original["stages"][stage]["macro_recall"],
                                              "positive_pair_delta":result["stages"][stage]["positive_pairs"]-original["stages"][stage]["positive_pairs"]}
                                         for stage in result["stages"]}
    (args.output/"metrics.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    for name,value in sources.items():
        if sha256(Path(name))!=value:raise ValueError("Input drift: "+name)
    for name,value in base_files.items():
        if sha256(args.base/name)!=value:raise ValueError("Original result was modified")
    if sha256(Path(__file__))!=own_sha:raise ValueError("Patch script drift")
    manifest={"scene":"search","derived_from":str(args.base),"original_files_sha256":base_files,"patch_script_sha256":own_sha,
              "verified_source_inputs":sources,"patch_request_ids":ids,"unchanged_request_count":len(updated)-5,
              "outputs":{p.name:sha256(p) for p in args.output.glob("*.parquet")}}
    (args.output/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    print("COMPLETE",flush=True)
if __name__=="__main__":main()
