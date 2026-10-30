"""Raw Search query encoding with explicit cache and missing-ID checks."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

class QueryTextStore:
    def __init__(self, request_ids, vectors):
        self.id2pos = {str(int(key)):i for i,key in enumerate(request_ids)}
        self.data = np.asarray(vectors,dtype=np.float32)
        if len(self.id2pos)!=len(request_ids) or self.data.shape!=(len(request_ids),768):
            raise ValueError("Invalid query request mapping/vector shape")
        if not np.isfinite(self.data).all() or np.any(np.linalg.norm(self.data,axis=1)<1e-6):
            raise ValueError("Query embeddings must be finite and nonzero")
    def get_batch_vecs(self, request_ids):
        missing=[key for key in request_ids if str(int(key)) not in self.id2pos]
        if missing:raise KeyError(f"Missing query embedding: first={missing[0]}, count={len(missing)}")
        return torch.from_numpy(self.data[[self.id2pos[str(int(key))] for key in request_ids]].copy())

def build_search_query_store(requests,cache_dir,batch_size=32,device="cpu",model_name="BAAI/bge-base-zh"):
    """Raw text, normalized CLS, float32, max512: same encoding as serving."""
    frame=requests[["search_idx","query"]].copy()
    frame["query"]=frame["query"].fillna("").astype(str).str.strip()
    if frame["query"].eq("").any():raise ValueError("Empty raw Search query text")
    if frame.groupby("search_idx")["query"].nunique().gt(1).any():raise ValueError("Conflicting request query text")
    frame=frame.drop_duplicates("search_idx").sort_values("search_idx")
    contract={"model":model_name,"pooling":"normalized_cls","max_length":512,"dtype":"float32"}
    payload={"contract":contract,"requests":frame.to_dict("records")}
    key=hashlib.sha256(json.dumps(payload,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    cache_dir=Path(cache_dir);cache_dir.mkdir(parents=True,exist_ok=True)
    vector_path,meta_path=cache_dir/f"{key}.npy",cache_dir/f"{key}.json"
    if vector_path.exists() and meta_path.exists():
        metadata=json.loads(meta_path.read_text())
        if metadata.get("key")!=key or metadata.get("contract")!=contract:raise ValueError("Query cache contract mismatch")
        vectors=np.load(vector_path,allow_pickle=False)
        if metadata.get("vectors_sha256") != hashlib.sha256(vectors.tobytes()).hexdigest():
            raise ValueError("Query cache vector checksum mismatch")
        from transformers import AutoConfig
        revision=AutoConfig.from_pretrained(model_name,local_files_only=True)._commit_hash
        if metadata.get("model_commit") != revision:
            raise ValueError("Cached query encoder revision differs from the local encoder")
    else:
        from transformers import AutoModel,AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(model_name,local_files_only=True)
        encoder=AutoModel.from_pretrained(model_name,local_files_only=True).eval().to(device)
        texts=frame["query"].drop_duplicates().tolist();encoded=[]
        with torch.no_grad():
            for start in range(0,len(texts),batch_size):
                tokens=tokenizer(texts[start:start+batch_size],padding=True,truncation=True,max_length=512,return_tensors="pt").to(device)
                encoded.append(F.normalize(encoder(**tokens).last_hidden_state[:,0].float(),p=2,dim=-1).cpu().numpy())
        table=np.concatenate(encoded);mapping={text:i for i,text in enumerate(texts)}
        vectors=table[[mapping[text] for text in frame["query"]]]
        np.save(vector_path,vectors,allow_pickle=False)
        meta_path.write_text(json.dumps({"key":key,"contract":contract,"request_ids":frame["search_idx"].astype(int).tolist(),"model_commit":getattr(encoder.config,"_commit_hash",None),"vectors_sha256":hashlib.sha256(vectors.tobytes()).hexdigest()},indent=2))
        del encoder
        if str(device).startswith("cuda"):torch.cuda.empty_cache()
    return QueryTextStore(frame["search_idx"].tolist(),vectors)