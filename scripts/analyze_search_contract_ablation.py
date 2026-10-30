"""Validation-only sensitivity of a frozen Search model to its serving contract."""
from pathlib import Path
import argparse,json
import faiss,numpy as np,pandas as pd,torch
from evaluate_search_index import inputs,DEFAULT,ROOT
from optimize_rec_retrieval import raw_model_frame,encode_requests,retrieve_index,metrics,write_json
from recall import dssm_trainer as trainer
from recall.query_encoding import build_search_query_store
from backend.online.item_features import get_item_feature_store

class ZeroQueryAblation:
    def __init__(self,store):self.store=store
    def get_batch_vecs(self,ids):
        self.store.get_batch_vecs(ids) # Validate IDs; this is deliberate ablation, not a missing-ID fallback.
        return torch.zeros(len(ids),768)

def main(output):
    faiss.omp_set_num_threads(2);torch.set_num_threads(2)
    selected,epoch,item_ids,_=inputs(output)
    checkpoint=torch.load(epoch/"dssm_search_optimized_user_tower.pt",map_location="cpu");meta=checkpoint["meta"]
    model=trainer.DSSMModel("search",meta["cat_vocabs"],meta["item_dense_dim"],meta["user_dense_dim"],preprocessing=meta["numeric_preprocessing"]).eval()
    model.load_state_dict(checkpoint["state_dict"],strict=False)
    manifest=json.loads((output/"dssm/training_manifest.json").read_text())
    raw=pd.read_parquet(ROOT/"datasets/search_train/train-00000-of-00001.parquet",columns=["search_idx","query"])
    query=build_search_query_store(raw,output/"query_cache/train")
    stores={"note_text":trainer.MMapStore("note_text_emb","note_text"),"note_img":trainer.MMapStore("note_img_emb","note_img"),"search_query":query}
    frame=raw_model_frame(pd.read_parquet(output/"features_validation.parquet"),get_item_feature_store(ROOT),trainer,"search")
    requests=pd.read_parquet(output/"requests_validation.parquet").rename(columns={"search_idx":"gid"})
    index=faiss.read_index(str(output/"index/ivfpq.faiss"));index.nprobe=32
    target=output/"contract_ablation";target.mkdir(exist_ok=True)
    normalizer={key:value.clone() for key,value in model.user_numeric.state_dict().items()}
    results={"scope":"all 6134 train-only validation requests, frozen selected model and fixed IVFPQ32",
      "limitations":"Inference perturbation sensitivity; not separately retrained feature attribution and not a deployed/test gain"}
    for variant in ("correct_contract","zero_query","raw_numeric_without_transform"):
        model.user_numeric.load_state_dict(normalizer)
        stores["search_query"]=ZeroQueryAblation(query) if variant=="zero_query" else query
        if variant=="raw_numeric_without_transform":
            model.user_numeric.mean.zero_();model.user_numeric.scale.fill_(1.);model.user_numeric.log_mask.fill_(False)
        ids,vectors=encode_requests(model,frame,stores,"search",manifest["unknown_user_id"],set(manifest["known_user_ids"]))
        candidates=retrieve_index(index,ids,vectors,item_ids,target/f"{variant}.parquet")
        results[variant]=metrics(requests,candidates)[0]
        print(variant,results[variant],flush=True)
    write_json(target/"metrics.json",results)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,default=DEFAULT)
    main(parser.parse_args().output)