"""全候选固定漏斗重放：只量化已知点击正例留存，不把未曝光物品当负例。"""
import argparse, gc, hashlib, json, os, time
from pathlib import Path
import duckdb, numpy as np, pandas as pd, torch
from backend.online.pipeline import SceneServingState
from backend.online.preranking.service import run_preranking
from backend.online.ranking.service import run_ranking, RANK_TOPN
from backend.offline.evaluate_full_test import confidence, sha256

ROOT=Path(__file__).resolve().parents[1]
def groups(path,key,cap):
    with duckdb.connect(config={"threads":"2","memory_limit":"2GB"}) as con:
        reader=con.execute(f"SELECT * FROM read_parquet(?) WHERE rank<=? ORDER BY {key},rank",[str(path),cap]).fetch_record_batch(64000)
        carry=pd.DataFrame()
        for batch in reader:
            frame=pd.concat([carry,batch.to_pandas()],ignore_index=True)
            last=frame[key].iloc[-1]
            carry=frame[frame[key]==last]
            for gid,group in frame[frame[key]!=last].groupby(key,sort=False): yield int(gid),group
        if len(carry): yield int(carry[key].iloc[0]),carry

def stage_row(gid,user,positive,ids):
    row={"gid":gid,"user_idx":user,"positives":len(positive)}
    for stage,items in ids.items():
        values=list(dict.fromkeys(int(x) for x in items))
        hits=len(positive.intersection(values))
        row[stage+"_hits"]=hits
        row[stage+"_recall"]=hits/len(positive) if positive else np.nan
        row[stage+"_hit"]=float(hits>0)
        row[stage+"_size"]=len(values)
    return row

def score_queued(model, queued, key, *, batch_size=512):
    """Batch only model inference; each request retains raw history and serving budget."""
    chunks=[]
    histories={}
    for gid,pre,history in queued:
        chosen=pre.sort_values("preranking_rank",kind="mergesort").drop_duplicates("note_idx")
        chosen=chosen.head(int(RANK_TOPN)).head(max(0,int(os.getenv("QILIN_ONLINE_DIEN_TOPN","240"))))
        if len(chosen):
            chunks.append(chosen)
            histories[gid]=history or [-1]
    if not chunks:
        return {}
    frame=pd.concat(chunks,ignore_index=True)
    gids=frame[key].to_numpy(np.int64)
    unique=list(dict.fromkeys(int(x) for x in gids))
    sequence=torch.stack([model._get_history_seq_base_tensor(histories[gid]) for gid in unique])
    seq_index={gid:index for index,gid in enumerate(unique)}
    numeric={}
    for name in model.num_feats:
        source=f"dien_{name}" if name in ("preranking_score","preranking_rank") and f"dien_{name}" in frame else name
        if source in frame:
            numeric[name]=pd.to_numeric(frame[source],errors="coerce").replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(np.float32)
        else:
            numeric[name]=np.zeros(len(frame),dtype=np.float32)
    scores=np.empty(len(frame),dtype=np.float32)
    with torch.no_grad():
        for start in range(0,len(frame),batch_size):
            batch=frame.iloc[start:start+batch_size]
            indices=torch.tensor([seq_index[int(g)] for g in gids[start:start+len(batch)]],device=model.device)
            feats={name:torch.tensor(values[start:start+len(batch)],device=model.device).unsqueeze(-1) for name,values in numeric.items()}
            feats["seq_embs"]=sequence[indices]
            nids=batch.note_idx.to_numpy(np.int64)
            feats["note_text_emb"]=model.stores["t"].get_tensor(nids)
            feats["note_img_emb"]=model.stores["i"].get_tensor(nids)
            if model.scene=="search":
                feats["query_emb"]=model._get_runtime_query_tensor(batch,batch[key].to_numpy(np.int64))
            pred,_=model.model(feats)
            scores[start:start+len(batch)]=pred.detach().float().cpu().numpy().reshape(-1)
    if not np.isfinite(scores).all(): raise ValueError("Nonfinite batched DIEN scores")
    return {(int(gid),int(note)):float(score) for gid,note,score in zip(gids,frame.note_idx,scores)}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene",choices=["search","rec"],required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--queue-groups",type=int,default=64)
    p.add_argument("--max-groups",type=int,default=0,help="debug only, never full-test evidence")
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(42);torch.backends.cudnn.benchmark=False
    def log(**values):
        print(json.dumps({"utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),**values}),flush=True)
    folder="search" if args.scene=="search" else "recommendation"
    key="search_idx" if args.scene=="search" else "request_idx"
    nested="search_result_details_with_idx" if args.scene=="search" else "rec_result_details_with_idx"
    paths=list((ROOT/"datasets"/f"{folder}_test").glob("*.parquet"))
    requests=pd.concat([pd.read_parquet(path,columns=[key,"user_idx","query","recent_clicked_note_idxs",nested]) for path in paths],ignore_index=True)
    if requests[key].duplicated().any(): raise ValueError("duplicate request IDs")
    raw={int(row[key]):row for row in requests.to_dict("records")}
    truth={gid:{int(item["note_idx"]) for item in row[nested] if item["click"]>0} for gid,row in raw.items()}
    state=SceneServingState(args.scene,"hard",500,1000)
    state.realtime_cache=None
    if state.lgb_model is None or state.xgb_model is None or state.dien_predictor.model is None: raise ValueError("missing models")
    recall_cap=min(state.live_recall_rank_cap,160) if args.scene=="search" else state.live_recall_rank_cap
    pre_cap=min(state.live_gbdt_topn,80) if args.scene=="search" else state.live_gbdt_topn
    manifest=json.loads((ROOT/"outputs/evaluation/full_test_20261030_contract_d/manifest.json").read_text())
    input_paths=[Path(name) if Path(name).is_absolute() else ROOT/name for name in manifest["inputs"]]
    reference_path=ROOT/"outputs/evaluation/full_test_20261030_contract_d"/args.scene/"recall_requests.parquet"
    input_paths+=list((ROOT/"datasets/notes").glob("*.parquet"))+[Path(__file__),reference_path]
    inputs={str(path):sha256(path) for path in sorted(set(input_paths))}
    (args.output/"manifest.json").write_text(json.dumps({"inputs":inputs,"scene":args.scene,"recall_cap":recall_cap,"pre_cap":pre_cap},indent=2))
    # Metadata projection is identical to SceneServingState._fetch_notes; index once rather than rescan 2M notes per request.
    log(stage="load_note_metadata")
    columns=["note_idx","note_title","note_content","image_path","accum_like_num","accum_collect_num","accum_comment_num","taxonomy1_id","taxonomy2_id","taxonomy3_id"]
    with duckdb.connect(config={"threads":"2","memory_limit":"3GB"}) as con:
        notes=con.execute("SELECT "+",".join(columns)+" FROM read_parquet(?)",[state.notes_glob]).fetchdf().set_index("note_idx",drop=False)
    if not notes.index.is_unique: raise ValueError("duplicate note IDs")
    def fetch_notes(ids):
        return notes.reindex(sorted(set(int(x) for x in ids))).dropna(subset=["note_idx"]).reset_index(drop=True)
    state._fetch_notes=fetch_notes
    if args.scene=="search":
        model=state.dien_predictor
        if model.query_model is None: raise ValueError("query encoder missing")
        model.query_model.to(model.device)
        texts=sorted(set(str(q or "").strip() for q in requests["query"])-{""})
        with torch.no_grad():
            for start in range(0,len(texts),64):
                batch=texts[start:start+64]
                tokens=model.query_tokenizer(batch,padding=True,truncation=True,max_length=512,return_tensors="pt").to(model.device)
                vectors=model.query_model(**tokens).last_hidden_state[:,0]
                vectors=vectors/vectors.norm(dim=-1,keepdim=True).clamp_min(1e-12)
                for text,vector in zip(batch,vectors.cpu().numpy()):model._query_cache[text]=vector.reshape(1,-1)
    missing={};observed_rows=0;candidate_rows=0
    original=state.gbdt_predictor.predict_with_components
    def audit(frame):
        for feature in state.gbdt_predictor.feature_cols:
            missing[feature]=missing.get(feature,0)+(int(frame[feature].isna().sum()) if feature in frame else len(frame))
        return original(frame)
    state.gbdt_predictor.predict_with_components=audit
    recall_path=ROOT/f"outputs/data/recall_{args.scene}_test_hard_multiroute_top1000.parquet"
    rows=[];score_parts=[];seen=set(); started=time.perf_counter()
    log(stage="replay",requests=len(raw),recall_cap=recall_cap,pre_cap=pre_cap)
    queued=[];verified=0;prepared=0
    def flush():
        nonlocal verified
        # Main result preserves the original per-request batch shape and expanded-history layout.
        # Cross-request GEMM batching remains a separate sensitivity helper: allclose alone cannot prove rank equivalence.
        score_map={}
        for gid,candidates,pre,history in queued:
            chosen=pre.sort_values("preranking_rank",kind="mergesort").drop_duplicates("note_idx")
            chosen=chosen.head(int(RANK_TOPN)).head(max(0,int(os.getenv("QILIN_ONLINE_DIEN_TOPN","240"))))
            if len(chosen):
                values=state.predict_dien(chosen,history_note_ids=history or [-1])
                score_map.update({(gid,int(note)):float(score) for note,score in zip(chosen.note_idx,values)})
        for gid,candidates,pre,history in queued:
            request=raw[gid];positive=truth[gid]
            pre=pre.copy()
            pre["_offline_dien_score"]=[score_map.get((gid,int(note)),np.nan) for note in pre.note_idx]
            if verified<5:
                chosen=pre.head(min(int(RANK_TOPN),max(0,int(os.getenv("QILIN_ONLINE_DIEN_TOPN","240")))))
                if len(chosen):
                    original_scores=state.predict_dien(chosen,history_note_ids=history or [-1])
                    np.testing.assert_allclose(chosen["_offline_dien_score"].to_numpy(),original_scores,rtol=1e-4,atol=1e-5)
                verified+=1
            def cached_predict(frame, **kwargs):
                values=frame["_offline_dien_score"].to_numpy(np.float32)
                if not np.isfinite(values).all():raise ValueError("Serving requested an unscored DIEN candidate")
                return values
            ranked,_=run_ranking(pre,1,20,cached_predict,history_note_ids=history or [-1])
            ids={"recall_budget":candidates.note_idx.tolist(),"recall10":candidates.note_idx.head(10).tolist(),
                 "coarse_budget":pre.note_idx.tolist(),"coarse10":pre.note_idx.head(10).tolist(),
                 "fine_budget":ranked.note_idx.tolist(),"fine10":ranked.note_idx.head(10).tolist(),
                 "fine20":ranked.note_idx.head(20).tolist()}
            rows.append(stage_row(gid,int(request["user_idx"]),positive,ids))
            compact=ranked[[key,"note_idx","preranking_rank","gbdt_score","dien_score","final_score","_offline_dien_score"]].copy()
            compact["ranking_rank"]=np.arange(1,len(compact)+1);score_parts.append(compact)
        pd.DataFrame(rows).to_parquet(args.output/"requests.partial.parquet",index=False)
        elapsed=time.perf_counter()-started
        log(stage="progress",completed=len(rows),total=len(raw),seconds=elapsed,requests_per_second=len(rows)/elapsed)
        queued.clear()
    for gid,candidates in groups(recall_path,key,recall_cap):
        if gid not in raw:raise ValueError("unknown request")
        seen.add(gid); request=raw[gid]
        features=state.get_feat_req(gid)
        if features.empty:raise ValueError("missing request features")
        candidate_rows+=len(candidates)
        observed_rows+=int(candidates.note_idx.isin(features.note_idx).sum())
        query=str(request["query"] or "") if args.scene=="search" else ""
        # No realtime writes, request substitution, cross-scene linkage or post-ranking backfill.
        pre=run_preranking(int(request["user_idx"]),query,args.scene,key,candidates,features,pre_cap,fetch_notes,state.predict_gbdt)
        pre=pre.reset_index(drop=True);pre["preranking_rank"]=np.arange(1,len(pre)+1)
        history=[int(x) for x in request["recent_clicked_note_idxs"] if int(x)>=0]
        queued.append((gid,candidates,pre,history));prepared+=1
        if len(queued)>=args.queue_groups:flush()
        if args.max_groups and prepared>=args.max_groups:break
    if queued:flush()
    ids={stage:[] for stage in ("recall_budget","recall10","coarse_budget","coarse10","fine_budget","fine10","fine20")}
    if not args.max_groups:
        # Missing candidates remain zero-recall requests.
        for gid in set(raw)-seen:rows.append(stage_row(gid,int(raw[gid]["user_idx"]),truth[gid],{stage:[] for stage in ids}))
    frame=pd.DataFrame(rows)
    reference=pd.read_parquet(reference_path).set_index("gid")
    if set(reference.index)!=set(raw):raise ValueError("Top1000 reference requests differ")
    for gid,value in truth.items():
        if int(reference.loc[gid,"positives"])!=len(value):raise ValueError("Top1000 positive denominator differs")
    frame["recall1000_hits"]=frame.gid.map(reference["hits@1000"]).astype(int)
    frame["recall1000_recall"]=frame.gid.map(reference["recall@1000"]).astype(float)
    frame["recall1000_hit"]=frame.gid.map(reference["hit@1000"]).astype(float)
    frame["recall1000_size"]=frame.gid.map(reference["candidates"]).astype(int)
    ids={"recall1000":[],**ids}
    frame.to_parquet(args.output/"requests.parquet",index=False)
    if score_parts:pd.concat(score_parts,ignore_index=True).to_parquet(args.output/"ranked_candidates.parquet",index=False)
    result={"scene":args.scene,"all_requests":len(raw),"evaluated_requests":len(frame),
            "scope":"frozen live-budget candidates replayed through current coarse/fine functions; known-positive retention only",
            "excluded":"dynamic/cold recall replacement, realtime/cross-scene linkage, query rewriting, diversity/backfill",
            "sampling":"debug prefix" if args.max_groups else "none","recall_cap":recall_cap,"coarse_cap":pre_cap,
            "verified_original_requests":verified,"batch_size":"original per-request <=512","inference_mode":"native per-request, cross-request scores not used","queue_groups":args.queue_groups,"candidate_rows":candidate_rows,"logged_feature_match_rows":observed_rows,"missing_feature_cells":missing,
            "stages":{},"paired_gains":{},"strict_acceptance":"BLOCKED"}
    for stage in ids:
        result["stages"][stage]={"macro_recall":float(frame[stage+"_recall"].mean()),
            "micro_recall":float(frame[stage+"_hits"].sum()/frame.positives.sum()),
            "hit_rate":float(frame.loc[frame.positives>0,stage+"_hit"].mean()),
            "positive_pairs":int(frame[stage+"_hits"].sum()),"candidate_rows":int(frame[stage+"_size"].sum())}
    for after,before in [("recall_budget","recall1000"),("coarse_budget","recall_budget"),("fine_budget","coarse_budget"),("coarse10","recall10"),("fine10","coarse10")]:
        name=after+"-"+before;frame["delta"]=frame[after+"_recall"]-frame[before+"_recall"]
        result["paired_gains"][name]={"macro_delta":float(frame.delta.mean()),"user_cluster_CI95":confidence(frame,"delta")}
    changed=[name for name,value in inputs.items() if sha256(Path(name))!=value]
    if changed:raise ValueError("Input drift: "+str(changed))
    (args.output/"metrics.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
    log(stage="complete",requests=len(frame),seconds=time.perf_counter()-started)
if __name__=="__main__":main()
