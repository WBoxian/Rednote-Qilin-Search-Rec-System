"""Frozen Search DSSM: same-model Flat/production-geometry IVFPQ comparison."""
from pathlib import Path
import argparse,gc,json,time
import faiss
import numpy as np
import pandas as pd
from optimize_rec_retrieval import metrics,retrieve_index,write_json,raw_model_frame,encode_requests,digest
from recall.query_encoding import build_search_query_store
from recall import dssm_trainer as trainer
ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/"outputs/evaluation/optimization_20261031/search"

def benchmark(index,queries):
    # Validation-only warm single-query measurements, without encoder/network time.
    chosen=np.random.default_rng(42).choice(len(queries),min(64,len(queries)),replace=False)
    values=[]
    for row in chosen:
        query=np.ascontiguousarray(queries[row:row+1],dtype=np.float32)
        index.search(query,1000)
        start=time.perf_counter();index.search(query,1000);values.append((time.perf_counter()-start)*1000)
    return {"queries":len(values),"single_query_mean_ms":float(np.mean(values)),
            "single_query_p50_ms":float(np.percentile(values,50)),"single_query_p95_ms":float(np.percentile(values,95)),
            "scope":"warm retrieval only, top1000, threads2; excludes raw query/user encoding"}

def inputs(output):
    selected=json.loads((output/"dssm/selected.json").read_text())
    epoch=Path(selected["directory"]);ids=np.load(output/"dssm/item_ids.npy")
    if digest(epoch/"item_vectors.bin")!=selected["item_vectors_sha256"]:raise ValueError("Selected vectors changed")
    vectors=np.memmap(epoch/"item_vectors.bin",dtype=np.float32,mode="r",shape=(len(ids),128))
    return selected,epoch,ids,vectors

def validation(output):
    faiss.omp_set_num_threads(2)
    selected,epoch,ids,vectors=inputs(output)
    target=output/"index";target.mkdir(exist_ok=True)
    path=target/"ivfpq.faiss"
    if not path.exists():
        fit_ids=np.unique(pd.read_parquet(output/"features_fit.parquet",columns=["note_idx"]).note_idx.to_numpy(np.int64))
        positions=np.searchsorted(ids,fit_ids);valid=positions<len(ids)
        valid[valid]=ids[positions[valid]]==fit_ids[valid]
        positions=positions[valid]
        sample=np.random.default_rng(42).choice(positions,min(200000,len(positions)),replace=False)
        if len(sample)<81920:raise ValueError("Insufficient fit-only IVF training item vectors")
        index=faiss.IndexIVFPQ(faiss.IndexFlatIP(128),128,2048,32,8,faiss.METRIC_INNER_PRODUCT)
        start=time.monotonic();index.train(np.ascontiguousarray(vectors[sample]));index.add(vectors);index.nprobe=32
        faiss.write_index(index,str(path))
        write_json(target/"manifest.json",{"metric":"ip","nlist":2048,"m":32,"pq_bits":8,"nprobe":32,
          "sampling_seed":42,"codebook_training":"only item vectors observed in fit exposures",
          "codebook_fit_items":len(positions),"codebook_sample_items":len(sample),"build_seconds":time.monotonic()-start,
          "catalog_items":len(ids),"selected_epoch":selected["epoch"],"item_vectors_sha256":selected["item_vectors_sha256"],
          "production_geometry_source":"outputs/deploy/search/hard/index/dssm_search_hard_ivfpq_meta.json",
          "index_sha256":digest(path),"serialized_bytes":path.stat().st_size})
    index=faiss.read_index(str(path));index.nprobe=32
    qids=np.load(epoch/"validation_request_ids.npy");queries=np.load(epoch/"validation_vectors.npy")
    requests=pd.read_parquet(output/"requests_validation.parquet").rename(columns={"search_idx":"gid"})
    start=time.monotonic();ann=retrieve_index(index,qids,queries,ids,target/"validation_ivfpq.parquet")
    score,_=metrics(requests,ann)
    result={"ivfpq":{"metrics":score,"batch_seconds":time.monotonic()-start,"latency":benchmark(index,queries),"serialized_bytes":path.stat().st_size}}
    exact=pd.read_parquet(epoch/"validation_ann.parquet")
    result["flat"]={"metrics":metrics(requests,exact)[0],"vector_payload_bytes":len(ids)*128*4}
    flat=faiss.IndexFlatIP(128);flat.add(vectors)
    faiss.write_index(flat,str(target/"flat.faiss"))
    result["flat"].update(latency=benchmark(flat,queries),serialized_bytes=(target/"flat.faiss").stat().st_size)
    write_json(target/"validation_comparison.json",result)
    print(json.dumps(result,indent=2),flush=True)

def apply_user_profile_contract(frame, checkpoint_meta, training_manifest=None):
    """Honor the trained42-column mask even in old states without persistent flags."""
    result=frame.copy()
    training_manifest=training_manifest or {}
    masked=bool(checkpoint_meta.get("masked_user_dense",training_manifest.get("masked_user_dense",False)))
    if masked:
        columns=["fans_num","follows_num"]+[f"dense_feat{i}" for i in range(1,41)]
        result[columns]=0.
    return result


def final_test(output):
    """Final frozen-model acceptance; never chooses model, index, or weights here."""
    import torch
    from backend.online.item_features import get_item_feature_store
    faiss.omp_set_num_threads(2);torch.set_num_threads(2)
    if (output/"index/test_comparison.json").exists():
        raise RuntimeError("Final Search index test is already frozen; use a new experiment directory")
    if not (output/"index/validation_comparison.json").exists():
        raise RuntimeError("Complete validation index comparison before final test")
    selected,epoch,item_ids,vectors=inputs(output)
    training=json.loads((output/"dssm/training_manifest.json").read_text())
    checkpoint=torch.load(epoch/"dssm_search_optimized_user_tower.pt",map_location="cpu")
    meta=checkpoint["meta"]
    model=trainer.DSSMModel("search",meta["cat_vocabs"],meta["item_dense_dim"],meta["user_dense_dim"],
                           preprocessing=meta["numeric_preprocessing"]).eval()
    missing,unexpected=model.load_state_dict(checkpoint["state_dict"],strict=False)
    if unexpected or any(name.startswith(("u_emb_","user_mlp","user_numeric")) for name in missing):
        raise ValueError("Frozen Search user tower loading mismatch")
    raw=pd.read_parquet(ROOT/"datasets/search_test/train-00000-of-00001.parquet")
    query_store=build_search_query_store(raw,output/"query_cache/test")
    stores={"note_text":trainer.MMapStore("note_text_emb","note_text"),
            "note_img":trainer.MMapStore("note_img_emb","note_img"),"search_query":query_store}
    frame=raw_model_frame(pd.read_parquet(ROOT/"features/search_test_features.parquet"),get_item_feature_store(ROOT),trainer,"search")
    frame=apply_user_profile_contract(frame,meta,training)
    request_ids,queries=encode_requests(model,frame,stores,"search",int(meta.get("unknown_user_id",training["unknown_user_id"])),set(meta.get("known_user_ids",training["known_user_ids"])))
    if set(request_ids)!=set(raw.search_idx):raise ValueError("Incomplete official test requests")
    requests=raw[["search_idx","user_idx"]].rename(columns={"search_idx":"gid"})
    requests["positives"]=[sorted({int(item["note_idx"]) for item in details if item["click"]>0})
                            for details in raw.search_result_details_with_idx]
    target=output/"index"
    np.save(target/"test_request_ids.npy",request_ids);np.save(target/"test_request_vectors.npy",queries)
    results={"scene":"search","selection":"all parameters fixed on train-only validation before official test",
             "selected_epoch":selected["epoch"],"known_labels":"official raw positive request-item pairs; unlabeled candidates not treated as negative"}
    for name in ("ivfpq","flat"):
        index=faiss.read_index(str(target/f"{name}.faiss"))
        if name=="ivfpq":index.nprobe=32
        start=time.monotonic()
        candidates=retrieve_index(index,request_ids,queries,item_ids,target/f"test_{name}.parquet")
        score,per_request=metrics(requests,candidates)
        per_request.to_parquet(target/f"test_{name}_metrics.parquet",index=False)
        results[name]={"metrics":score,"total_retrieval_and_write_seconds":time.monotonic()-start,
                       "index_sha256":digest(target/f"{name}.faiss")}
        del candidates,index;gc.collect()
    write_json(target/"test_comparison.json",results)
    print(json.dumps(results,indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,default=DEFAULT)
    parser.add_argument("--stage",choices=["validation","test"],default="validation")
    args=parser.parse_args();{"validation":validation,"test":final_test}[args.stage](args.output)