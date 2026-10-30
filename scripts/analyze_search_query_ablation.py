"""Search query-input mechanism ablation: fixed random sample, never a full-test score."""
from __future__ import annotations
import argparse,hashlib,importlib.util,json
from pathlib import Path
import faiss,numpy as np,pandas as pd,torch
from transformers import AutoModel,AutoTokenizer
ROOT=Path(__file__).resolve().parents[1]

def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(16*1024*1024),b""):h.update(chunk)
    return h.hexdigest()

def summarize(positives, ranked):
    values=[]
    for truth,order in zip(positives,ranked):
        positions={int(item):i+1 for i,item in enumerate(order) if item>=0}
        first=min((positions[item] for item in truth if item in positions),default=1001)
        count=sum(positions.get(item,1001)<=500 for item in truth)
        values.append((count/len(truth),int(count>0),1/first if first<=100 else 0.,count,len(truth)))
    a=np.asarray(values)
    return {"requests":len(a),"Recall@500":float(a[:,0].mean()),"HitRate@500":float(a[:,1].mean()),
            "MRR@100":float(a[:,2].mean()),"positive_pairs":int(a[:,4].sum()),
            "micro_recall500":float(a[:,3].sum()/a[:,4].sum())},a

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(42);faiss.omp_set_num_threads(4)
    raw_path=ROOT/"datasets/search_test/train-00000-of-00001.parquet"
    feature_path=ROOT/"features/search_test_features.parquet"
    deploy=ROOT/"outputs/deploy/search/hard"
    raw=pd.read_parquet(raw_path).sort_values("search_idx").reset_index(drop=True)
    selected=np.random.default_rng(42).choice(len(raw),128,replace=False)
    requests=raw.iloc[selected].copy()
    positive=[{int(item["note_idx"]) for item in details if item["click"]>0} for details in requests.search_result_details_with_idx]
    if any(not truth for truth in positive):raise ValueError("Undefined positive-free denominator")
    frame=pd.read_parquet(feature_path).drop_duplicates("search_idx").set_index("search_idx").loc[requests.search_idx].reset_index()
    gid=requests.search_idx.astype(int).tolist()
    spec=importlib.util.spec_from_file_location("readonly_search_dssm",ROOT/"src/recall/dssm_trainer.py")
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);mod.DEVICE=torch.device("cpu")
    tower=deploy/"models/dssm_search_hard_user_tower.pt"
    checkpoint=torch.load(tower,map_location="cpu");meta=checkpoint["meta"]
    if meta["scene"]!="search" or meta["hidden_dim"]!=128:raise ValueError("Unexpected tower metadata")
    model=mod.DSSMModel(scene="search",cat_vocabs=meta["cat_vocabs"],item_dense_dim=meta["item_dense_dim"],user_dense_dim=meta["user_dense_dim"])
    missing,unexpected=model.load_state_dict(checkpoint["state_dict"],strict=False)
    if any(key.startswith(("u_emb_","user_mlp")) for key in missing) or unexpected:raise ValueError("User tower checkpoint mismatch")
    model.eval()
    stores={"note_text":mod.MMapStore("note_text_emb","note_text"),"note_img":mod.MMapStore("note_img_emb","note_img")}
    batch={key:torch.tensor(frame[key].to_numpy(np.int64),dtype=torch.long) for key in ("user_idx","gender_enc","platform_enc","age_enc","location_enc")}
    batch.update({key:torch.tensor(frame[key].to_numpy(np.float32)) for key in ("fans_num","follows_num")})
    batch["dense_feats"]=torch.tensor(frame[[f"dense_feat{i}" for i in range(1,41)]].to_numpy(np.float32))
    history=torch.stack([mod.build_history_summary_from_ids(list(items),stores) if len(items) else torch.zeros(768) for items in frame.recent_clicked_note_idxs])
    batch["query_vec"]=torch.zeros((len(frame),768))
    with torch.no_grad():zero=model.forward_user(batch,history).numpy()
    print("zero-query user vectors ready",flush=True)
    tokenizer=AutoTokenizer.from_pretrained("BAAI/bge-base-zh",local_files_only=True)
    encoder=AutoModel.from_pretrained("BAAI/bge-base-zh",local_files_only=True).eval().to("cpu")
    queries=[str(text or "").strip() for text in requests["query"]]
    real=[]
    with torch.no_grad():
        for start in range(0,len(queries),8):
            tokens=tokenizer(queries[start:start+8],padding=True,truncation=True,max_length=512,return_tensors="pt")
            encoded=encoder(**tokens).last_hidden_state[:,0]
            encoded=encoded/encoded.norm(dim=-1,keepdim=True).clamp_min(1e-12)
            real.append(encoded.half().float())
    batch["query_vec"]=torch.cat(real)
    with torch.no_grad():actual=model.forward_user(batch,history).numpy()
    print("raw-query user vectors ready",flush=True)
    index_path=deploy/"index/dssm_search_hard_ivfpq.faiss"
    map_path=deploy/"index/dssm_search_hard_row2note.npy"
    index=faiss.read_index(str(index_path));items=np.load(map_path)
    if index.ntotal!=len(items) or index.d!=zero.shape[1]:raise ValueError("ANN/tower/mapping shape mismatch")
    req_meta=json.load(open(deploy/"index/dssm_search_hard_test_request_meta.json"))
    req_map=json.load(open(deploy/"index/dssm_search_hard_test_request_map.json"))
    frozen=np.memmap(deploy/"index/dssm_search_hard_test_request_emb.bin",mode="r",dtype=np.float32,shape=(req_meta["num_requests"],req_meta["dim"]))
    existing=np.asarray(frozen[[req_map[str(x)] for x in gid]])
    def retrieve(vectors):
        _,indices=index.search(np.ascontiguousarray(vectors,dtype=np.float32),1000)
        return np.array([[int(items[i]) if i>=0 else -1 for i in row] for row in indices])
    zero_items,real_items,frozen_items=retrieve(zero),retrieve(actual),retrieve(existing)
    output={"scene":"search","sample":"128 request IDs selected uniformly without replacement from sorted official test IDs",
            "seed":42,"selection_uses_labels":False,"tuning":"none","index_ntotal":index.ntotal,
            "persisted_nprobe":int(index.nprobe),"source":"same deployed hard user tower + same existing FAISS + same row2note; only query input differs",
            "limitations":["Small mechanism sample, NOT full-test results or deployable lift",
                           "No provenance certificate for historical tower/item-index build; metadata/shape and actual reproduction checked",
                           "CPU BGE normalized CLS converted via float16, matching preprocessing storage contract",
                           "User-cluster confidence interval over this small selected sample does not establish population performance"]}
    rows=[]
    arrays={}
    for label,ranked in (("frozen_request_vectors",frozen_items),("zero_query",zero_items),("raw_query_bge",real_items)):
        output[label],arrays[label]=summarize(positive,ranked)
        for request_id,order in zip(gid,ranked):
            rows.extend((label,request_id,int(item),rank+1) for rank,item in enumerate(order))
    output["zero_vs_frozen"]={
        "mean_vector_cosine":float(np.mean(np.sum(zero*existing,axis=1)/(np.linalg.norm(zero,axis=1)*np.linalg.norm(existing,axis=1)))),
        "max_abs_vector_difference":float(np.max(np.abs(zero-existing))),
        "mean_top500_set_overlap":float(np.mean([len(set(a[:500])&set(b[:500]))/500 for a,b in zip(zero_items,frozen_items)])),
        "identical_top500_order_requests":sum(np.array_equal(a[:500],b[:500]) for a,b in zip(zero_items,frozen_items))}
    per=pd.DataFrame({"gid":gid,"user_idx":requests.user_idx.astype(int).to_numpy(),
                      "query":queries,"positive_count":[len(t) for t in positive],
                      "zero_recall500":arrays["zero_query"][:,0],"raw_query_recall500":arrays["raw_query_bge"][:,0]})
    per["delta"]=per.raw_query_recall500-per.zero_recall500
    by_user=per.groupby("user_idx").delta.agg(["sum","count"]).to_numpy()
    rng=np.random.default_rng(42);estimates=[]
    for _ in range(500):
        sampled=by_user[rng.integers(0,len(by_user),len(by_user))]
        estimates.append(float(sampled[:,0].sum()/sampled[:,1].sum()))
    output["paired_raw_minus_zero"]={"Recall@500_delta":float(per.delta.mean()),
        "user_cluster_CI95":[float(x) for x in np.quantile(estimates,[.025,.975])]}
    per.to_parquet(args.output/"requests.parquet",index=False)
    pd.DataFrame(rows,columns=["variant","gid","note_idx","rank"]).to_parquet(args.output/"ann_candidates.parquet",index=False)
    important=[raw_path,feature_path,tower,index_path,map_path,ROOT/"src/recall/dssm_trainer.py",Path(__file__)]
    output["input_sha256"]={str(path):digest(path) for path in important}
    (args.output/"metrics.json").write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(output,ensure_ascii=False,indent=2),flush=True)
if __name__=="__main__":main()
