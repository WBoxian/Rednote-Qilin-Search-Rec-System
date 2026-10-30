"""Read-only sampled ANN fidelity diagnostic for the frozen temporal Rec model."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import faiss

ROOT=Path(__file__).resolve().parents[1]
def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for x in iter(lambda:f.read(16*1024*1024),b""):h.update(x)
    return h.hexdigest()
def topk(scores,ids,k):
    k=min(k,len(scores))
    if k==0:return scores[:0],ids[:0]
    threshold=np.partition(scores,len(scores)-k)[len(scores)-k]
    greater=np.flatnonzero(scores>threshold)
    ties=np.flatnonzero(scores==threshold)
    ties=ties[np.argsort(ids[ties],kind="stable")[:k-len(greater)]]
    picked=np.r_[greater,ties]
    picked=picked[np.lexsort((ids[picked],-scores[picked]))]
    return scores[picked],ids[picked]
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--requests",type=int,default=128)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    base=ROOT/"outputs/evaluation/three_route/rec"; idxdir=base/"outputs/index"
    prefix="dssm_rec_hard"
    paths=[idxdir/f"{prefix}_{suffix}" for suffix in
        ("item_meta.json","item_emb.bin","row2note.npy","ivfpq.faiss","ivfpq_meta.json",
         "test_request_meta.json","test_request_map.json","test_request_emb.bin")]
    paths += [base/"test_ann.parquet",base/"requests_test.parquet",Path(__file__)]
    before={str(x.relative_to(ROOT)):digest(x) for x in paths}
    meta=json.loads((idxdir/f"{prefix}_item_meta.json").read_text())
    reqmeta=json.loads((idxdir/f"{prefix}_test_request_meta.json").read_text())
    mapping=json.loads((idxdir/f"{prefix}_test_request_map.json").read_text())
    reqs=pd.read_parquet(base/"requests_test.parquet")
    gids=np.sort(reqs.request_idx.to_numpy())
    chosen=np.sort(np.random.default_rng(42).choice(gids,min(a.requests,len(gids)),replace=False))
    assert all(str(gid) in mapping for gid in chosen)
    queries=np.memmap(idxdir/f"{prefix}_test_request_emb.bin",dtype="float32",mode="r",
                      shape=(reqmeta["num_requests"],reqmeta["dim"]))
    q=np.array(queries[[mapping[str(gid)] for gid in chosen]],dtype=np.float32)
    item=np.memmap(idxdir/f"{prefix}_item_emb.bin",dtype="float32",mode="r",
                   shape=(meta["num_items"],meta["dim"]))
    itemids=np.load(idxdir/f"{prefix}_row2note.npy")
    assert meta["dim"]==reqmeta["dim"] and len(itemids)==len(item)
    index=faiss.read_index(str(idxdir/f"{prefix}_ivfpq.faiss"))
    faiss.omp_set_num_threads(2)
    assert index.ntotal==len(item) and index.metric_type==faiss.METRIC_INNER_PRODUCT
    _,annrows=index.search(q,1000)
    annids=itemids[annrows]
    frozen=pd.read_parquet(base/"test_ann.parquet",filters=[("gid","in",chosen.tolist())])
    replay=[len(set(ids)&set(frozen.loc[frozen.gid==gid,"note_idx"]))/1000 for ids,gid in zip(annids,chosen)]
    assert min(replay)==1.0,"UNVERIFIABLE: local ANN assets do not reproduce stored route candidates"
    bestscores=[np.array([],dtype=np.float32) for _ in chosen]
    bestids=[np.array([],dtype=np.int64) for _ in chosen]
    started=time.perf_counter()
    for start in range(0,len(item),65536):
        block=np.asarray(item[start:start+65536])
        assert np.isfinite(block).all()
        scores=q@block.T
        for row in range(len(chosen)):
            bestscores[row],bestids[row]=topk(np.r_[bestscores[row],scores[row]],
                np.r_[bestids[row],itemids[start:start+len(block)]],500)
        if start%(65536*8)==0:print("exact_items",start+len(block),flush=True)
    truth={int(row.request_idx):set(row.positives) for row in reqs.itertuples()}
    rows=[]
    for gid,exact,ann in zip(chosen,bestids,annids):
        positives=truth[int(gid)]
        er=len(positives&set(exact))/len(positives) if positives else np.nan
        ar=len(positives&set(ann[:500]))/len(positives) if positives else np.nan
        rows.append({"gid":int(gid),"positive_count":len(positives),
                     "ann_exact_overlap500":len(set(exact)&set(ann[:500]))/500,
                     "ann_business_recall500":ar,"exact_business_recall500":er,
                     "ann_hits":len(positives&set(ann[:500])),"exact_hits":len(positives&set(exact))})
    frame=pd.DataFrame(rows);frame.to_parquet(a.output/"requests.parquet",index=False)
    result={"scene":"rec","protocol":"frozen_temporal_three_route_model_only","requests":len(frame),"seed":42,
            "index_nprobe":index.nprobe,"index_type":"IVFPQ","catalog_size":len(item),
            "stored_top1000_replay_min":min(replay),"ANN_vs_exact_Top500_overlap_mean":float(frame.ann_exact_overlap500.mean()),
            "ANN_vs_exact_Top500_overlap_p10":float(frame.ann_exact_overlap500.quantile(.1)),
            "ann_business_macro_recall500":float(frame.ann_business_recall500.mean()),
            "exact_business_macro_recall500":float(frame.exact_business_recall500.mean()),
            "ann_positive_hits":int(frame.ann_hits.sum()),"exact_positive_hits":int(frame.exact_hits.sum()),
            "positive_pairs":int(frame.positive_count.sum()),"exact_cpu_seconds":time.perf_counter()-started,
            "limitations":["Sample of fixed 128 requests, not full-test model evaluation",
                           "ANN overlap is index approximation fidelity, not clicked-item recall",
                           "Exact search can gain or lose business positives; it optimizes learned-vector dot product",
                           "Training lineage/time availability remain unverified"]}
    assert before=={str(x.relative_to(ROOT)):digest(x) for x in paths},"input drift"
    (a.output/"metrics.json").write_text(json.dumps(result,indent=2)+"\n")
    (a.output/"manifest.json").write_text(json.dumps({"inputs":before},indent=2)+"\n")
    print(json.dumps(result,indent=2),flush=True)
if __name__=="__main__":main()
