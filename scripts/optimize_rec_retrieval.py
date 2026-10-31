"""Temporal Recommendation retrieval training and validation-only CF/fusion selection.

Never writes deployment assets. Raw histories are provider snapshots without
per-event timestamps; fit histories may expand CF edges, never validation/test
histories or labels. Query-time fold-in only reads the fixed fit graph.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path
import time

import duckdb
import numpy as np
import pandas as pd
from scipy import sparse
from training.history import HISTORY_ORDER, recent_history_ids

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/evaluation/optimization_20261031/rec"
OLD = ROOT / "outputs/evaluation/three_route/rec"
SEED = 42
ROUTE_K = 1000
CF_CONFIGS = (
    {"name": "click_id50", "history_weight": 0., "fold_in": False, "neighbors": 50},
    {"name": "click_fold50", "history_weight": 0., "fold_in": True, "neighbors": 50},
    {"name": "hist025_fold50", "history_weight": .25, "fold_in": True, "neighbors": 50},
    {"name": "hist1_fold50", "history_weight": 1., "fold_in": True, "neighbors": 50},
    {"name": "hist025_fold200", "history_weight": .25, "fold_in": True, "neighbors": 200},
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def temporal_partition(intervals, test_start, validation_fraction=.2):
    """Whole-request forward split; requests spanning a boundary are excluded."""
    if not 0 < validation_fraction < 1:
        raise ValueError("invalid validation_fraction")
    if not np.isfinite(intervals[["lo", "hi"]].to_numpy(float)).all():
        raise ValueError("nonfinite request intervals")
    if intervals.gid.duplicated().any() or (intervals.lo > intervals.hi).any():
        raise ValueError("invalid request intervals")
    historical = intervals[intervals.hi < test_start]
    if len(historical) < 2:
        raise ValueError("insufficient historical requests")
    index = max(1, min(len(historical) - 1, int(len(historical) * (1-validation_fraction))))
    boundary = float(np.sort(historical.hi.to_numpy())[index])
    part = np.full(len(intervals), "excluded", dtype=object)
    part[(intervals.hi < boundary).to_numpy()] = "fit"
    part[((intervals.lo >= boundary) & (intervals.hi < test_start)).to_numpy()] = "validation"
    result = intervals.assign(partition=part)
    fit, val = (result[result.partition == split] for split in ("fit", "validation"))
    if fit.empty or val.empty or fit.hi.max() >= val.lo.min() or val.hi.max() >= test_start:
        raise ValueError("time split is not strictly forward")
    return result, boundary


def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"prepare refuses to overwrite {manifest_path}")
    train = ROOT / "datasets/recommendation_train/train-00000-of-00001.parquet"
    test = ROOT / "datasets/recommendation_test/train-00000-of-00001.parquet"
    feat = ROOT / "features/rec_train_features.parquet"
    # Official test is read here only to establish its first timestamp and ID separation.
    con = duckdb.connect(config={"threads": "2", "memory_limit": "3GB"})
    con.execute("""CREATE TABLE labels AS SELECT r.request_idx gid,first(r.user_idx) user_idx,
        d.note_idx,max(d.click)::INT positive,min(d.request_timestamp) lo,max(d.request_timestamp) hi
        FROM read_parquet(?) r,unnest(r.rec_result_details_with_idx) u(d)
        GROUP BY r.request_idx,d.note_idx""", [str(train)])
    intervals = con.execute("SELECT gid,min(lo) lo,max(hi) hi FROM labels GROUP BY gid ORDER BY gid").fetchdf()
    test_start = float(con.execute("""SELECT min(d.request_timestamp)
        FROM read_parquet(?) r,unnest(r.rec_result_details_with_idx) u(d)""", [str(test)]).fetchone()[0])
    overlap = con.execute("""SELECT count(*) FROM
        (SELECT DISTINCT request_idx FROM read_parquet(?)) a
        JOIN (SELECT DISTINCT request_idx FROM read_parquet(?)) b USING(request_idx)""", [str(train),str(test)]).fetchone()[0]
    if overlap:
        raise ValueError("official train/test request overlap")
    partitions, boundary = temporal_partition(intervals, test_start)
    partitions.to_parquet(out / "partitions.parquet", index=False)
    con.register("partitions", partitions)
    con.execute("""CREATE TABLE raw_requests AS SELECT r.request_idx gid,r.user_idx,
        r.recent_clicked_note_idxs history,
        list_sort(list(DISTINCT l.note_idx) FILTER(WHERE l.positive>0)) positives
        FROM read_parquet(?) r JOIN labels l ON r.request_idx=l.gid
        GROUP BY r.request_idx,r.user_idx,r.recent_clicked_note_idxs""", [str(train)])
    for split in ("fit", "validation"):
        requests = con.execute("""SELECT r.* FROM raw_requests r JOIN partitions p USING(gid)
            WHERE p.partition=? ORDER BY gid""", [split]).fetchdf()
        requests["positives"] = requests.positives.apply(lambda x: [] if x is None else list(x))
        requests.to_parquet(out / f"requests_{split}.parquet", index=False)
        con.execute("""SELECT l.* FROM labels l JOIN partitions p USING(gid)
            WHERE p.partition=? ORDER BY l.gid,l.note_idx""", [split]).fetchdf().to_parquet(out/f"labels_{split}.parquet",index=False)
    # Feature row/label audit is exhaustive on the official train before any fit.
    mismatch = con.execute("""WITH f AS (
        SELECT request_idx gid,note_idx,max(click)::INT positive
        FROM read_parquet(?) GROUP BY request_idx,note_idx)
        SELECT count(*) FROM labels FULL JOIN f USING(gid,note_idx)
        WHERE labels.positive IS DISTINCT FROM f.positive""", [str(feat)]).fetchone()[0]
    if mismatch:
        raise ValueError("train feature labels disagree with raw train")
    counts = con.execute("""SELECT p.partition,count(*) exposure_pairs,count(DISTINCT p.gid) requests,
        sum(l.positive) positives,min(l.lo) first_event,max(l.hi) last_event
        FROM labels l JOIN partitions p USING(gid) GROUP BY p.partition""").fetchdf().to_dict("records")
    inputs = [train, feat, Path(__file__), ROOT/"src/recall/dssm_trainer.py"]
    manifest = {"version":1, "scene":"rec", "seed":SEED,
        "fit_end_exclusive":boundary, "official_test_first_event":test_start,
        "official_test_use":"timestamp cutoff and ID separation only during preparation; labels final once",
        "selection_split":"validation", "selection_objective":"macro Recall@500; MRR@100 >= baseline validation",
        "candidate_budget":500,"route_budget":ROUTE_K,"partition":counts,
        "cf_grid":CF_CONFIGS, "inputs":{str(p.relative_to(ROOT)):digest(p) for p in inputs},
        "official_test_sha256":digest(test),
        "limitations":["Raw request histories are provider snapshots without per-event timestamps; historical legality unverified",
            "Full content snapshot lacks publication time; candidate availability unverified",
            "No validation/test labels or histories enter fitted CF edges or numeric preprocessing",
            "Validation used for model selection; final test is not a tuning set"]}
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2), flush=True)


def top_indices(ids, scores, k):
    if not len(ids):
        return np.array([], dtype=np.int64)
    # Stable tie policy shared by every route.
    return np.lexsort((ids, -scores))[:k]


def build_graph(clicks, requests, history_weight, max_items=200):
    """Only fit requests enter this weighted implicit-feedback graph."""
    edges = defaultdict(dict)
    for row in clicks[clicks.positive > 0].itertuples():
        user, item = int(row.user_idx), int(row.note_idx)
        previous = edges[user].get(item, (0., -np.inf))
        edges[user][item] = max(previous, (1., float(row.hi)))
    if history_weight > 0:
        for row in requests.itertuples():
            for item in row.history:
                item = int(item)
                if item < 0:
                    continue
                previous = edges[int(row.user_idx)].get(item, (0., -np.inf))
                edges[int(row.user_idx)][item] = max(previous, (history_weight,float(row.hi)))
    trimmed = {user:dict((item,value[0]) for item,value in
        sorted(items.items(), key=lambda pair:(-pair[1][0],-pair[1][1],pair[0]))[:max_items])
        for user,items in edges.items()}
    users = np.array(sorted(trimmed), dtype=np.int64)
    items = np.array(sorted({i for values in trimmed.values() for i in values}),dtype=np.int64)
    item_pos = {int(i):pos for pos,i in enumerate(items)}
    row_ids, col_ids, values = [], [], []
    for row,user in enumerate(users):
        for item,value in trimmed[int(user)].items():
            row_ids.append(row); col_ids.append(item_pos[item]); values.append(value)
    graph = sparse.csr_matrix((np.array(values,np.float32),(row_ids,col_ids)),shape=(len(users),len(items)))
    graph.sort_indices()
    return graph,users,items


def build_swing(graph, users, items, topk=200, min_common=2, max_users=200):
    """Canonical two-user Swing; history support expands edges, formula unchanged."""
    binary = graph.copy()
    binary.data[:] = 1.
    csc = graph.tocsc()
    item_users = {}
    user_items = [set(binary.indices[binary.indptr[u]:binary.indptr[u+1]]) for u in range(len(users))]
    for item in range(len(items)):
        start,end = csc.indptr[item:item+2]
        us, vals = csc.indices[start:end],csc.data[start:end]
        order = np.lexsort((users[us],-vals))[:max_users]
        item_users[item] = set(map(int,us[order]))
    weight = 1. / np.sqrt(np.maximum(np.diff(binary.indptr),1))
    @lru_cache(maxsize=200000)
    def pair_score(u,v):
        return float(weight[u]*weight[v]/(1.+len(user_items[u]&user_items[v])))
    row_ids,col_ids,values=[],[],[]
    for item in range(len(items)):
        us=item_users[item]
        if len(us)<min_common:
            continue
        counts=defaultdict(int)
        for u in us:
            for other in user_items[u]:
                if other!=item: counts[other]+=1
        candidates=sorted(((other,n) for other,n in counts.items() if n>=min_common),
            key=lambda pair:(-pair[1],int(items[pair[0]])))[:topk]
        for other,_ in candidates:
            common=sorted(us&item_users[other])
            if len(common)<min_common: continue
            score=sum(pair_score(u,v) for u,v in itertools.combinations(common,2))
            if score>0:
                row_ids.append(item);col_ids.append(other);values.append(score)
        if item and item%50000==0:
            print("swing items",item,"edges",len(values),flush=True)
    return sparse.csr_matrix((np.array(values,np.float32),(row_ids,col_ids)),shape=(len(items),len(items)))


def query_cf(requests, graph, users, items, swing, config, output, support_path=None):
    """Offline and serving share exactly the same fixed-graph CF implementation."""
    from recall.cf_retrieval import CFGraphRecaller
    recaller = CFGraphRecaller.from_arrays(graph, users, items, swing, config)
    rows = {"swing": [], "usercf": []}
    support = []
    for request in requests.itertuples():
        routes, info = recaller.score_with_support(int(request.user_idx), request.history, ROUTE_K)
        for route, scores in routes.items():
            rows[route].extend((int(request.gid), item, score, rank)
                for rank, (item, score) in enumerate(scores.items(), 1))
        support.append({"gid": int(request.gid), **info})
    for route, records in rows.items():
        pd.DataFrame(records, columns=["gid", "note_idx", "score", "rank"]).to_parquet(
            output / f"{route}.parquet", index=False)
    if support_path:
        pd.DataFrame(support).to_parquet(support_path, index=False)


def metrics(requests, candidates, k=500):
    """Known-click retention; all original requests retained, not CTR on unexposed negatives."""
    ranked = defaultdict(dict)
    for row in candidates.itertuples():
        gid, item, score = int(row.gid), int(row.note_idx), float(row.score)
        if item < 0 or not np.isfinite(score):
            raise ValueError("Invalid candidate")
        ranked[gid][item] = max(score, ranked[gid].get(item, -np.inf))
    records = []
    for row in requests.itertuples():
        truth = set(map(int, row.positives))
        order = sorted(ranked[int(row.gid)].items(), key=lambda pair: (-pair[1], pair[0]))
        ranks = [rank for rank, (item, _) in enumerate(order, 1) if item in truth]
        first = min(ranks, default=np.inf)
        hit = sum(rank <= k for rank in ranks)
        npos = len(truth)
        dcg = sum(1. / np.log2(rank + 1.) for rank in ranks if rank <= 10)
        idcg = sum(1. / np.log2(rank + 1.) for rank in range(1, min(npos, 10) + 1))
        records.append((int(row.gid), int(row.user_idx), npos, hit, first,
            hit / npos if npos else 0., 1. / first if first <= 100 else 0.,
            int(first <= 50), sum(rank <= 100 for rank in ranks) / npos if npos else 0.,
            dcg / idcg if idcg else 0., 1. / first if first <= 10 else 0.,
            sum(rank <= 500 for rank in ranks) / npos if npos else 0., int(first <= 500)))
    frame = pd.DataFrame(records, columns=["gid", "user_idx", "positives", "hits", "first_hit",
        "recall", "mrr100", "hit50", "recall100", "ndcg10", "mrr10", "recall500", "hit500"])
    result = {"all_requests": len(frame), "positive_requests": int((frame.positives > 0).sum()),
        "Recall@500": float(frame.recall500.mean()), "HitRate@500": float(frame.hit500.mean()),
        "MRR@100": float(frame.mrr100.mean()), "MRR@10": float(frame.mrr10.mean()),
        "HitRate@50": float(frame.hit50.mean()), "Recall@100": float(frame.recall100.mean()),
        "known_positive_NDCG@10": float(frame.ndcg10.mean()),
        "positive_hits": int(frame.hits.sum()), "positive_pairs": int(frame.positives.sum())}
    if k != 500:
        result[f"Recall@{k}"] = float(frame.recall.mean())
    return result, frame


def cf(out):
    if (out/"selection.json").exists():
        raise RuntimeError("CF grid locked after selection")
    fit=pd.read_parquet(out/"labels_fit.parquet")
    requests=pd.read_parquet(out/"requests_fit.parquet")
    partitions=pd.read_parquet(out/"partitions.parquet")
    requests=requests.merge(partitions[["gid","hi"]],on="gid",validate="one_to_one")
    val=pd.read_parquet(out/"requests_validation.parquet")
    result={}
    for history_weight in sorted({c["history_weight"] for c in CF_CONFIGS}):
        cache=out/f"cf_graph_{history_weight:g}"
        cache.mkdir(exist_ok=True)
        if not (cache/"graph.npz").exists():
            graph,users,items=build_graph(fit,requests,history_weight)
            sparse.save_npz(cache/"graph.npz",graph)
            np.save(cache/"users.npy",users);np.save(cache/"items.npy",items)
        graph=sparse.load_npz(cache/"graph.npz")
        users=np.load(cache/"users.npy");items=np.load(cache/"items.npy")
        print("CF graph",history_weight,graph.shape,graph.nnz,flush=True)
        if not (cache/"swing.npz").exists():
            start=time.monotonic()
            swing=build_swing(graph,users,items)
            sparse.save_npz(cache/"swing.npz",swing)
            write_json(cache/"meta.json",{"history_weight":history_weight,"users":len(users),"items":len(items),
                "edges":graph.nnz,"swing_edges":swing.nnz,"seconds":time.monotonic()-start,
                "builder_contract":"max-score-then-latest-time-v2",
                "fit_requests_sha256":digest(out/"requests_fit.parquet"),
                "fit_labels_sha256":digest(out/"labels_fit.parquet")})
        swing=sparse.load_npz(cache/"swing.npz")
        for config in CF_CONFIGS:
            if config["history_weight"]!=history_weight: continue
            path=out/"validation"/config["name"]
            path.mkdir(parents=True,exist_ok=True)
            if not (path/"usercf.parquet").exists():
                query_cf(val,graph,users,items,swing,config,path,path/"support.parquet")
            result[config["name"]]={name:metrics(val,pd.read_parquet(path/f"{name}.parquet"))[0]
                for name in ("swing","usercf")}
            support=pd.read_parquet(path/"support.parquet")
            result[config["name"]]["support"]={"requests":len(support),
                "no_history_graph_support":int((support.history_items_in_graph==0).sum()),
                "no_swing_trigger":int((support.history_items_with_swing_edges==0).sum()),
                "no_user_neighbors":int((support.user_neighbors==0).sum()),
                "empty_swing":int((support.swing_pool_size==0).sum()),
                "empty_usercf":int((support.usercf_pool_size==0).sum())}
            write_json(out/"cf_validation.json",result)
            print(config["name"],result[config["name"]],flush=True)



def raw_model_frame(frame, store, trainer, scene):
    """Shared static content contract; unverifiable aggregate counters are disabled."""
    result=store.attach(frame)
    user_cols,item_cols,log_cols=trainer.numeric_feature_columns(scene)
    static={"video_duration","video_height","video_width","image_num","content_length","commercial_flag"}
    for col in item_cols:
        if col not in static: result[col]=0.
    for col in user_cols+item_cols:
        result[col]=pd.to_numeric(result[col],errors="coerce").replace([np.inf,-np.inf],np.nan).fillna(0)
    result["recent_clicked_note_idxs"]=result.recent_clicked_note_idxs.apply(
        lambda value:list(value) if isinstance(value,(list,np.ndarray)) else [])
    return result


def encode_requests(model, frame, stores, scene, unknown_user_id, known_users, batch_size=256):
    import torch
    from recall import dssm_trainer as trainer
    key="search_idx" if scene=="search" else "request_idx"
    req=frame.drop_duplicates(key).sort_values(key)
    device=next(model.parameters()).device
    result=[]
    with torch.no_grad():
        for start in range(0,len(req),batch_size):
            part=req.iloc[start:start+batch_size]
            batch={col:torch.tensor(part[col].to_numpy(np.int64),device=device)
                for col in ["user_idx","gender_enc","platform_enc","age_enc","location_enc"]}
            batch["user_idx"]=torch.tensor([int(u) if int(u) in known_users else unknown_user_id
                for u in part.user_idx],device=device)
            batch["fans_num"]=torch.tensor(part.fans_num.to_numpy(np.float32),device=device)
            batch["follows_num"]=torch.tensor(part.follows_num.to_numpy(np.float32),device=device)
            batch["dense_feats"]=torch.tensor(part[[f"dense_feat{i}" for i in range(1,41)]].to_numpy(np.float32),device=device)
            histories=part.recent_clicked_note_idxs.tolist()
            summary=torch.stack([trainer.build_history_summary_from_ids(ids,stores) for ids in histories]).to(device)
            sequences=[trainer.build_history_sequence_from_ids(ids,stores) for ids in histories]
            tokens=torch.stack([pair[0] for pair in sequences]).to(device)
            masks=torch.stack([pair[1] for pair in sequences]).to(device)
            if scene=="search":
                batch["query_vec"]=stores["search_query"].get_batch_vecs(part[key].tolist()).to(device)
            vectors=model.forward_user(batch,summary,seq_tokens=tokens,seq_mask=masks)
            result.append(vectors.detach().float().cpu().numpy())
    return req[key].to_numpy(np.int64),np.concatenate(result)


def export_catalog(model, store, item_ids, stores, trainer, scene, destination, batch_size=2048):
    import torch
    device=next(model.parameters()).device
    _,item_cols,_=trainer.numeric_feature_columns(scene)
    vectors=np.memmap(destination,dtype=np.float32,mode="w+",shape=(len(item_ids),trainer.HIDDEN_DIM))
    model.eval()
    with torch.no_grad():
        for start in range(0,len(item_ids),batch_size):
            ids=item_ids[start:start+batch_size]
            raw=store.get(ids)
            for col in item_cols:
                if col not in raw: raw[col]=0.
            batch={"note_idx":torch.tensor(ids,device=device),
                "note_type":torch.tensor(raw.note_type.to_numpy(np.int64),device=device),
                "taxonomy":torch.tensor(raw[[f"taxonomy{i}_id_enc" for i in range(1,4)]].to_numpy(np.int64),device=device),
                "dense_stats":torch.tensor(raw[item_cols].to_numpy(np.float32),device=device)}
            text=stores["note_text"].get_batch_vecs(ids.tolist()).to(device)
            image=stores["note_img"].get_batch_vecs(ids.tolist()).to(device)
            vectors[start:start+len(ids)]=model.forward_item(batch,text,image).detach().float().cpu().numpy()
    vectors.flush()
    return vectors


def exact_retrieval(query_ids, queries, item_ids, items, device, k=1000, query_batch=128, item_batch=65536):
    """Exact full-catalog IP scores, bounded memory; topk boundary ties follow torch policy."""
    import torch
    records=[]
    # Within retained candidates score ties are ordered by item ID. Topk boundary ties may choose any equal-score ID.
    with torch.no_grad():
        for start in range(0,len(queries),query_batch):
            query=torch.from_numpy(queries[start:start+query_batch].copy()).to(device)
            best_scores=torch.empty((len(query),0),device=device)
            best_rows=torch.empty((len(query),0),device=device,dtype=torch.long)
            for offset in range(0,len(items),item_batch):
                matrix=torch.from_numpy(np.asarray(items[offset:offset+item_batch]).copy()).to(device)
                scores=query@matrix.T
                take=min(k,scores.shape[1])
                # torch.topk retains a correct score TopK; arbitrary equal-score IDs
                # can be omitted at the boundary. This is not a catalog-wide ID tie oracle.
                values,rows=torch.topk(scores,take,dim=1,sorted=True)
                rows+=offset
                merged_scores=torch.cat((best_scores,values),dim=1)
                merged_rows=torch.cat((best_rows,rows),dim=1)
                values,order=torch.topk(merged_scores,min(k,merged_scores.shape[1]),dim=1,sorted=True)
                best_scores=values;best_rows=torch.gather(merged_rows,1,order)
            cpu_scores=best_scores.cpu().numpy()
            cpu_rows=best_rows.cpu().numpy()
            for local,gid in enumerate(query_ids[start:start+len(query)]):
                ids=item_ids[cpu_rows[local]]
                order=top_indices(ids,cpu_scores[local],k)
                records.extend((int(gid),int(ids[i]),float(cpu_scores[local,i]),rank) for rank,i in enumerate(order,1))
    return pd.DataFrame(records,columns=["gid","note_idx","score","rank"])


def multi_positive_inbatch_loss(user_vectors, item_vectors, item_ids, request_ids=None, temperature=.07):
    """InfoNCE with multiple clicked positives from the same request or item ID."""
    import torch
    if temperature<=0 or user_vectors.shape!=item_vectors.shape or user_vectors.ndim!=2:
        raise ValueError("Invalid contrastive loss inputs")
    size=len(user_vectors)
    if item_ids.ndim!=1 or len(item_ids)!=size or (request_ids is not None and
            (request_ids.ndim!=1 or len(request_ids)!=size)):
        raise ValueError("Loss IDs must align with batch rows")
    logits=(user_vectors@item_vectors.T).float()/temperature
    positive=item_ids[:,None]==item_ids[None,:]
    if request_ids is not None:
        positive=positive|(request_ids[:,None]==request_ids[None,:])
    return (torch.logsumexp(logits,dim=1)-
        torch.logsumexp(logits.masked_fill(~positive,-torch.inf),dim=1)).mean()


def train_dssm(scene, fit_frame, validation_frame, output, epochs=3, query_store=None, batch_size=256,
        loss_contract="same_request_or_same_item_v2", selection_objective="recall500",
        mask_unverified_user_dense=False, mask_sequence_padding=False):
    """Fit only on supplied train fold; choose the caller-declared full-catalog validation objective.

    Returns selected metadata. The exact full-catalog validation route is retained;
    a deployable FAISS index is built later from the selected vectors.
    """
    import torch
    from torch.utils.data import DataLoader
    from backend.online.item_features import get_item_feature_store
    from recall import dssm_trainer as trainer
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/"selected.json").exists(): raise RuntimeError("DSSM selection is already frozen")
    if selection_objective not in {"recall500", "recall100"}:
        raise ValueError("Unsupported DSSM validation objective")
    if loss_contract not in {"same_request_or_same_item_v2","same_item_only_v1"}:
        raise ValueError("Unsupported loss contract")
    trainer.set_seed(SEED)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    trainer.DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device=trainer.DEVICE
    key="search_idx" if scene=="search" else "request_idx"
    if set(fit_frame[key])&set(validation_frame[key]):
        raise ValueError("fit/validation overlap")
    if fit_frame.event_timestamp.max()>=validation_frame.event_timestamp.min():
        raise ValueError("non-forward model folds")
    store=get_item_feature_store(ROOT)
    fit=raw_model_frame(fit_frame,store,trainer,scene)
    val=raw_model_frame(validation_frame,store,trainer,scene)
    if mask_unverified_user_dense:
        for target in (fit, val):
            target[["fans_num", "follows_num"] + [f"dense_feat{i}" for i in range(1, 41)]] = 0.
    user_cols,item_cols,log_cols=trainer.numeric_feature_columns(scene)
    preprocessing=trainer.fit_numeric_preprocessing(fit,log_cols,user_cols+item_cols)
    stores={"note_text":trainer.MMapStore("note_text_emb","note_text"),
        "note_img":trainer.MMapStore("note_img_emb","note_img")}
    if scene=="search":
        if query_store is None: raise ValueError("search requires raw-query embedding store")
        stores["search_query"]=query_store
    item_ids=np.array(sorted(map(int,stores["note_text"].id2pos)),dtype=np.int64)
    np.save(output/"item_ids.npy",item_ids)
    # Training batches require only clicked item metadata; complete catalog uses the same raw store during export.
    positives=fit[fit.click>0].drop_duplicates([key,"note_idx"]).copy()
    item_meta=trainer.build_item_meta(positives,scene,item_cols)
    unknown_user=int(fit.user_idx.max())+1
    known_users=set(map(int,fit.user_idx.unique()))
    cat_sizes=trainer.load_category_sizes()
    vocabs={"user_idx":unknown_user+1,"note_idx":int(item_ids.max())+1}
    for name,col in [("gender","gender_enc"),("platform","platform_enc"),("age","age_enc"),("location","location_enc"),
        ("tax1","taxonomy1_id_enc"),("tax2","taxonomy2_id_enc"),("tax3","taxonomy3_id_enc")]:
        observed=int(fit[col].max())+1
        # Category dictionaries are fixed source preprocessing artifacts, never refitted on validation/test.
        vocabs[name]=max(observed,cat_sizes.get(name,0))
    model=trainer.DSSMModel(scene,vocabs,len(item_cols),42+trainer.EMB_DIM+
        (trainer.EMB_DIM if scene=="search" else trainer.SEQ_TOWER_DIM),preprocessing=preprocessing,
        mask_sequence_padding=mask_sequence_padding,mask_user_profile=mask_unverified_user_dense).to(device)
    # ID-only cold entities stay neutral. Fit entities learn offsets from content.
    with torch.no_grad():
        model.u_emb_user.weight.zero_();model.i_emb_item.weight.zero_()
    dataset=trainer.DSSMDataset(scene,key,positives,item_meta,list(item_meta),
        torch.full((len(item_meta),),1./len(item_meta)),num_neg_per_pos=1,use_inbatch_neg=True)
    original_collate=trainer.collate_fn_builder(stores,scene)
    def collate_with_request_ids(rows):
        batch=original_collate(rows)
        batch[0]["request_ids"]=torch.tensor([row[0]["req_id"] for row in rows],dtype=torch.long)
        return batch
    loader=DataLoader(dataset,batch_size=batch_size,shuffle=True,
        collate_fn=collate_with_request_ids,num_workers=2,pin_memory=device.type=="cuda",
        persistent_workers=True,prefetch_factor=2,generator=torch.Generator().manual_seed(SEED))
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=.01)
    scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda")
    val_requests=val.groupby(key,sort=True).agg(user_idx=("user_idx","first"),
        positives=("note_idx",lambda series:[])).reset_index().rename(columns={key:"gid"})
    val_labels=val[val.click>0].groupby(key).note_idx.apply(lambda series:sorted(set(map(int,series))))
    val_requests["positives"]=val_requests.gid.map(val_labels).apply(lambda value:value if isinstance(value,list) else [])
    logs=[];best=None
    source_manifest={"scene":scene,"seed":SEED,"epochs":epochs,"batch_size":batch_size,
        "fit_requests":int(fit[key].nunique()),"fit_rows":len(fit),"fit_positive_pairs":len(positives),
        "validation_requests":len(val_requests),"fit_last_event":float(fit.event_timestamp.max()),
        "validation_first_event":float(val.event_timestamp.min()),
        "input_contract":"raw numeric; saved fit-only transform; shared full-catalog static fields; aggregate statistics disabled",
        "loss":"multi-positive in-batch InfoNCE; duplicate item IDs always positive; request positives enabled only for v2; temperature=.07",
        "loss_contract":loss_contract, "history_contract":HISTORY_ORDER,
        "masked_user_dense":bool(mask_unverified_user_dense), "selection_objective":selection_objective,
        "sequence_padding_contract":"packed_masked_v1" if mask_sequence_padding else "legacy_attention_only_v0",
        "cold_id_policy":"user/item ID tables initialize zero; unseen user mapped to reserved zero-trained ID",
        "unknown_user_id":unknown_user,"known_user_ids":sorted(known_users),"cat_vocabs":vocabs,
        "normalizer":preprocessing,"selection":f"validation full-catalog exact {selection_objective}; documented north-star ties",
        "catalog_availability":"snapshot; publication timestamp unavailable",
        "history_time_legality":"provider snapshots without event-level timestamps, not independently verified",
        "source_hashes":{str(path.relative_to(ROOT)):digest(path) for path in
            [Path(__file__),ROOT/"src/recall/dssm_trainer.py",ROOT/"src/backend/online/item_features.py"]}}
    write_json(output/"training_manifest.json",source_manifest)
    for epoch in range(1,epochs+1):
        start=time.monotonic();model.train();losses=[]
        for step,batch in enumerate(loader):
            batch=trainer.move_batch_to_device(batch,device)
            user,hist,tokens,mask,pos,txt,img,_,_,_,_,_=batch
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda",enabled=device.type=="cuda"):
                uv=model.forward_user(user,hist,seq_tokens=tokens,seq_mask=mask)
                iv=model.forward_item(pos,txt,img)
                request_ids=user["request_ids"] if loss_contract=="same_request_or_same_item_v2" else None
                loss=multi_positive_inbatch_loss(uv,iv,pos["note_idx"],request_ids)
            scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
            losses.append(float(loss.detach()))
            if step%100==0:print(scene,"epoch",epoch,"batch",step,"loss",losses[-1],flush=True)
        model.eval()
        epoch_dir=output/f"epoch_{epoch}";epoch_dir.mkdir(exist_ok=True)
        trainer.save_tower_checkpoints(model.state_dict(),scene,"optimized",vocabs,len(item_cols),
            42+trainer.EMB_DIM+(trainer.EMB_DIM if scene=="search" else trainer.SEQ_TOWER_DIM),
            preprocessing=preprocessing,output_dir=epoch_dir,
            extra_meta={"masked_user_dense":bool(mask_unverified_user_dense),"loss_contract":loss_contract,
                "unknown_user_id":unknown_user,"known_user_ids":sorted(known_users)})
        vectors=export_catalog(model,store,item_ids,stores,trainer,scene,epoch_dir/"item_vectors.bin")
        ids,queries=encode_requests(model,val,stores,scene,unknown_user,known_users)
        np.save(epoch_dir/"validation_request_ids.npy",ids);np.save(epoch_dir/"validation_vectors.npy",queries)
        candidates=exact_retrieval(ids,queries,item_ids,vectors,device)
        candidates.to_parquet(epoch_dir/"validation_ann.parquet",index=False)
        score,per_request=metrics(val_requests,candidates)
        per_request.to_parquet(epoch_dir/"validation_metrics.parquet",index=False)
        entry={"epoch":epoch,"mean_loss":float(np.mean(losses)),"seconds":time.monotonic()-start,**score}
        logs.append(entry);write_json(output/"epochs.json",logs)
        print(scene,"epoch result",entry,flush=True)
        objective=((score["Recall@100"],score["HitRate@50"],score["known_positive_NDCG@10"],score["MRR@100"])
            if selection_objective == "recall100" else (score["Recall@500"],score["MRR@100"]))
        if best is None or objective>best:
            best=objective
            selected={"epoch":epoch,"directory":str(epoch_dir),"validation":score,
                "checkpoint_sha256":digest(epoch_dir/f"dssm_{scene}_optimized.pt"),
                "item_vectors_sha256":digest(epoch_dir/"item_vectors.bin")}
        del vectors
    write_json(output/"selected.json",selected)
    return selected


def train(out,epochs=3,loss_contract="same_request_or_same_item_v2", feature_path=None,
        selection_objective="recall500",mask_unverified_user_dense=False,mask_sequence_padding=False):
    parts=pd.read_parquet(out/"partitions.parquet")
    frame=pd.read_parquet(feature_path or ROOT/"features/rec_train_features.parquet")
    selected={}
    for split in ("fit","validation"):
        ids=set(parts.loc[parts.partition==split,"gid"])
        selected[split]=frame[frame.request_idx.isin(ids)].copy()
    train_dssm("rec",selected["fit"],selected["validation"],out/"dssm",epochs=epochs,loss_contract=loss_contract,
        selection_objective=selection_objective,mask_unverified_user_dense=mask_unverified_user_dense,
        mask_sequence_padding=mask_sequence_padding)


def retrieve_index(index, request_ids, query_vectors, item_ids, output):
    records=[]
    for start in range(0,len(request_ids),128):
        scores,rows=index.search(np.ascontiguousarray(query_vectors[start:start+128],dtype=np.float32),ROUTE_K)
        for local,gid in enumerate(request_ids[start:start+128]):
            valid=rows[local]>=0
            ids=item_ids[rows[local][valid]];values=scores[local][valid]
            order=top_indices(ids,values,ROUTE_K)
            records.extend((int(gid),int(ids[i]),float(values[i]),rank) for rank,i in enumerate(order,1))
    result=pd.DataFrame(records,columns=["gid","note_idx","score","rank"])
    result.to_parquet(output,index=False)
    return result


def index_dssm(out, selection_objective="recall500"):
    """Index centroids train on fit exposure item vectors; nprobe selected on validation."""
    import faiss
    selected=json.loads((out/"dssm/selected.json").read_text())
    epoch=Path(selected["directory"])
    items=np.load(out/"dssm/item_ids.npy")
    vectors=np.memmap(epoch/"item_vectors.bin",dtype=np.float32,mode="r",shape=(len(items),128))
    directory=out/"index";directory.mkdir(exist_ok=True)
    faiss.omp_set_num_threads(2)
    path=directory/"ivfpq.faiss"
    if not path.exists():
        fit_ids=np.unique(pd.read_parquet(out/"labels_fit.parquet",columns=["note_idx"]).note_idx.to_numpy(np.int64))
        positions=np.searchsorted(items,fit_ids)
        valid=positions<len(items)
        positions=positions[valid]
        positions=positions[items[positions]==fit_ids[valid]]
        # Every centroid training example is a fit-exposure item; never validation/test labels.
        rng=np.random.default_rng(SEED)
        sampled=rng.choice(positions,min(100000,len(positions)),replace=False)
        index=faiss.IndexIVFPQ(faiss.IndexFlatIP(128),128,2048,32,8,faiss.METRIC_INNER_PRODUCT)
        index.cp.seed=SEED
        index.train(np.ascontiguousarray(vectors[sampled]))
        for start in range(0,len(items),65536):
            index.add(np.ascontiguousarray(vectors[start:start+65536]))
        faiss.write_index(index,str(path))
        write_json(directory/"build.json",{"algorithm":"IVFPQ","nlist":2048,"m":32,"bits":8,"metric":"inner_product",
            "centroid_training":"random seed42 sample up to100000 fit-exposure item vectors only",
            "fit_exposure_items":len(fit_ids),"training_vector_rows":len(sampled),"items":len(items),
            "selected_checkpoint_sha256":selected["checkpoint_sha256"],
            "selected_item_vectors_sha256":selected["item_vectors_sha256"],
            "index_sha256":digest(path)})
    else:
        index=faiss.read_index(str(path))
    ids=np.load(epoch/"validation_request_ids.npy")
    queries=np.load(epoch/"validation_vectors.npy")
    req=pd.read_parquet(out/"requests_validation.parquet")
    values=[]
    for nprobe in (32,64,128):
        index.nprobe=nprobe
        destination=directory/f"validation_nprobe{nprobe}.parquet"
        candidates=pd.read_parquet(destination) if destination.exists() else retrieve_index(index,ids,queries,items,destination)
        metric,_=metrics(req,candidates)
        values.append({"nprobe":nprobe,**metric})
        write_json(directory/"validation_grid.json",values)
        print("index nprobe",nprobe,metric,flush=True)
    choice=max(values,key=lambda row: ((row["Recall@100"],row["HitRate@50"],row["known_positive_NDCG@10"],-row["nprobe"])
        if selection_objective == "recall100" else (row["Recall@500"],row["MRR@100"],-row["nprobe"])))
    write_json(directory/"selected.json",choice)


def sql_route_metrics(con, expression):
    """Macro labels from raw request table; route-less requests remain in scope."""
    con.execute(f"""CREATE OR REPLACE TEMP TABLE selected AS SELECT gid,note_idx,
        row_number() OVER(PARTITION BY gid ORDER BY {expression} DESC,note_idx) rank FROM pool""")
    frame=con.execute("""SELECT r.gid,r.user_idx,array_length(r.positives) positives,
        count(*) FILTER(WHERE s.rank<=500) hits,min(s.rank) first_hit
        FROM requests r LEFT JOIN unnest(r.positives) u(note_idx) ON true
        LEFT JOIN selected s ON r.gid=s.gid AND u.note_idx=s.note_idx
        GROUP BY r.gid,r.user_idx,r.positives ORDER BY r.gid""").fetchdf()
    positive=frame[frame.positives>0]
    frame["recall"]=frame.hits/np.maximum(frame.positives,1)
    frame["first_hit"]=frame.first_hit.to_numpy(dtype=float,na_value=np.inf)
    frame["mrr100"]=np.where(frame.first_hit<=100,1./frame.first_hit,0.)
    return {"all_requests":len(frame),"positive_requests":len(positive),
        "Recall@500":float(frame.loc[positive.index,"recall"].mean()),
        "HitRate@500":float((positive.hits>0).mean()),"MRR@100":float(frame.loc[positive.index,"mrr100"].mean()),
        "positive_hits":int(positive.hits.sum()),"positive_pairs":int(positive.positives.sum())},frame


def set_pool(con, paths):
    for name,path in zip(("ann","swing","usercf"),paths):
        con.execute(f"CREATE OR REPLACE TEMP TABLE route_{name} AS SELECT gid,note_idx,score FROM read_parquet(?)",[str(path)])
    con.execute("""CREATE OR REPLACE TEMP TABLE pool AS WITH combined AS (
        SELECT *,'ann' route FROM route_ann UNION ALL SELECT *,'swing' FROM route_swing
        UNION ALL SELECT *,'usercf' FROM route_usercf),
        ranked AS (SELECT *,row_number() OVER(PARTITION BY gid,route ORDER BY score DESC,note_idx) rank FROM combined)
        SELECT gid,note_idx,
        max(CASE WHEN route='ann' THEN 1./(60+rank) ELSE 0 END) a,
        max(CASE WHEN route='swing' THEN 1./(60+rank) ELSE 0 END) s,
        max(CASE WHEN route='usercf' THEN 1./(60+rank) ELSE 0 END) u
        FROM ranked GROUP BY gid,note_idx""")


def select_fusion(out):
    if (out/"selection.json").exists():
        raise RuntimeError("Selection is frozen; do not retune after official test")
    con=duckdb.connect(config={"threads":"2","memory_limit":"4GB"})
    con.execute("CREATE TEMP TABLE requests AS SELECT * FROM read_parquet(?)",
        [str(out/"requests_validation.parquet")])
    set_pool(con,[OLD/f"validation_{name}.parquet" for name in ("ann","swing","usercf")])
    baseline,_=sql_route_metrics(con,"a+4*s+u")
    index_choice=json.loads((out/"index/selected.json").read_text())
    ann=out/f"index/validation_nprobe{index_choice['nprobe']}.parquet"
    # Predefined scale-distinct grid. No route or label budget changes.
    weights=((1,1,1),(1,2,1),(1,4,1),(1,8,1),(1,4,2),(1,4,4),
        (2,4,1),(4,4,1),(2,2,4),(1,2,4),(1,1,4),(1,1,8))
    records=[]
    for config in CF_CONFIGS:
        directory=out/"validation"/config["name"]
        set_pool(con,[ann,directory/"swing.parquet",directory/"usercf.parquet"])
        for a,s,u in weights:
            result,frame=sql_route_metrics(con,f"{a}*a+{s}*s+{u}*u")
            records.append({"cf":config,"weights":[a,s,u],**result,
                "guard_passed":result["MRR@100"]>=baseline["MRR@100"]})
        print("fusion completed",config["name"],flush=True)
        write_json(out/"fusion_validation_grid.json",{"baseline":baseline,"trials":records})
    eligible=[r for r in records if r["guard_passed"]]
    choice=max(eligible,key=lambda row:(row["Recall@500"],row["MRR@100"])) if eligible else None
    if choice is None:
        write_json(out/"selection_failed.json",{"reason":"No new validation policy passes baseline MRR guard","baseline":baseline})
        raise RuntimeError("Validation guard rejected all fusion policies")
    selection={"selection_split":"validation","main_objective":"macro Recall500",
        "guard":"MRR100 >= frozen previous three-route RRF1:4:1 baseline on same validation requests",
        "baseline":baseline,"selected":choice,"index":index_choice,
        "selected_dssm":json.loads((out/"dssm/selected.json").read_text()),
        "validation_grid_sha256":digest(out/"fusion_validation_grid.json"),
        "frozen_before_official_test":True}
    write_json(out/"selection.json",selection)
    paths=[ann,out/"validation"/choice["cf"]["name"]/"swing.parquet",out/"validation"/choice["cf"]["name"]/"usercf.parquet"]
    set_pool(con,paths)
    a,s,u=choice["weights"];_,per_request=sql_route_metrics(con,f"{a}*a+{s}*s+{u}*u")
    per_request.to_parquet(out/"validation_selected_metrics.parquet",index=False)
    con.execute("SELECT * FROM selected WHERE rank<=1000 ORDER BY gid,rank").fetchdf().to_parquet(out/"validation_selected.parquet",index=False)
    print(json.dumps(selection,indent=2),flush=True)


def final_test(out):
    """Single frozen official-test pass, never modifies a selected parameter."""
    import faiss
    import torch
    from recall import dssm_trainer as trainer
    from backend.online.item_features import get_item_feature_store
    destination=out/"test"
    if (destination/"metrics.json").exists(): raise RuntimeError("Official-test result already exists; no retuning")
    destination.mkdir(exist_ok=True)
    selection=json.loads((out/"selection.json").read_text())
    manifest=json.loads((out/"manifest.json").read_text())
    raw=ROOT/"datasets/recommendation_test/train-00000-of-00001.parquet"
    if digest(raw)!=manifest["official_test_sha256"]:raise ValueError("Official test changed")
    con=duckdb.connect(config={"threads":"2","memory_limit":"4GB"})
    req=con.execute("""SELECT r.request_idx gid,r.user_idx,r.recent_clicked_note_idxs history,
        list_sort(list(DISTINCT d.note_idx) FILTER(WHERE d.click>0)) positives
        FROM read_parquet(?) r,unnest(r.rec_result_details_with_idx) u(d)
        GROUP BY r.request_idx,r.user_idx,r.recent_clicked_note_idxs ORDER BY gid""",[str(raw)]).fetchdf()
    req.positives=req.positives.apply(lambda value:list(value) if isinstance(value,(list,np.ndarray)) else [])
    req.to_parquet(destination/"requests.parquet",index=False)
    train_meta=json.loads((out/"dssm/training_manifest.json").read_text())
    epoch=Path(selection["selected_dssm"]["directory"])
    device=torch.device("cpu")
    trainer.DEVICE=device
    user_cols,item_cols,_=trainer.numeric_feature_columns("rec")
    model=trainer.DSSMModel("rec",train_meta["cat_vocabs"],len(item_cols),42+768+128,
        preprocessing=train_meta["normalizer"]).to(device)
    checkpoint=epoch/"dssm_rec_optimized.pt"
    if digest(checkpoint)!=selection["selected_dssm"]["checkpoint_sha256"]:raise ValueError("Selected checkpoint changed")
    model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=False));model.eval()
    stores={"note_text":trainer.MMapStore("note_text_emb","note_text"),"note_img":trainer.MMapStore("note_img_emb","note_img")}
    frame=raw_model_frame(pd.read_parquet(ROOT/"features/rec_test_features.parquet"),
        get_item_feature_store(ROOT),trainer,"rec")
    ids,queries=encode_requests(model,frame,stores,"rec",train_meta["unknown_user_id"],set(train_meta["known_user_ids"]))
    if set(map(int,ids))!=set(map(int,req.gid)):
        raise ValueError("Test feature requests do not cover every raw official request")
    np.save(destination/"request_ids.npy",ids);np.save(destination/"query_vectors.npy",queries)
    del model
    if device.type=="cuda":torch.cuda.empty_cache()
    items=np.load(out/"dssm/item_ids.npy")
    index=faiss.read_index(str(out/"index/ivfpq.faiss"));index.nprobe=selection["index"]["nprobe"]
    faiss.omp_set_num_threads(2)
    start=time.monotonic()
    retrieve_index(index,ids,queries,items,destination/"ann.parquet")
    timing={"selected_pq_replay_seconds":time.monotonic()-start,"query_batch":128,"threads":2,
        "scope":"batch candidate replay including Python rows/parquet; not online latency"}
    index.nprobe=32
    start=time.monotonic()
    retrieve_index(index,ids,queries,items,destination/"ann_nprobe32.parquet")
    timing["production_pq32_replay_seconds"]=time.monotonic()-start
    # Full-catalog exact is a diagnostic only; the selected serving route stays IVFPQ.
    item_vectors=np.memmap(epoch/"item_vectors.bin",dtype=np.float32,mode="r",shape=(len(items),128))
    start=time.monotonic()
    if not (destination/"ann_exact.parquet").exists():
        exact_retrieval(ids,queries,items,item_vectors,device).to_parquet(destination/"ann_exact.parquet",index=False)
        timing["exact_cpu_replay_seconds"]=time.monotonic()-start
    else:
        timing["exact_cpu_replay_seconds"]=json.loads((destination/"retrieval_timing.json").read_text())["exact_cpu_replay_seconds"]
    write_json(destination/"retrieval_timing.json",timing)
    config=selection["selected"]["cf"];cache=out/f"cf_graph_{config['history_weight']:g}"
    query_cf(req,sparse.load_npz(cache/"graph.npz"),np.load(cache/"users.npy"),
        np.load(cache/"items.npy"),sparse.load_npz(cache/"swing.npz"),config,destination,destination/"support.parquet")
    con.execute("CREATE TABLE requests AS SELECT * FROM read_parquet(?)",[str(destination/"requests.parquet")])
    set_pool(con,[destination/f"{name}.parquet" for name in ("ann","swing","usercf")])
    a,s,u=selection["selected"]["weights"]
    result,per_request=sql_route_metrics(con,f"{a}*a+{s}*s+{u}*u")
    per_request.to_parquet(destination/"selected_metrics.parquet",index=False)
    con.execute("SELECT * FROM selected WHERE rank<=1000 ORDER BY gid,rank").fetchdf().to_parquet(destination/"selected.parquet",index=False)
    # Disjoint route/fusion losses: unknown non-exposure candidates are never negative labels.
    losses=con.execute("""WITH positive AS (SELECT gid,unnest(positives) note_idx FROM requests)
        SELECT count(*) positive_pairs,count(*) FILTER(WHERE p2.note_idx IS NULL) unreachable_in_top1000_union,
        count(*) FILTER(WHERE p2.note_idx IS NOT NULL AND s.rank>500) lost_by_fusion_cutoff,
        count(*) FILTER(WHERE s.rank<=500) selected_hits500
        FROM positive p LEFT JOIN pool p2 USING(gid,note_idx) LEFT JOIN selected s USING(gid,note_idx)""").fetchdf().iloc[0].to_dict()
    result={"scene":"rec","selection_sha256":digest(out/"selection.json"),"fused":result,
        "routes":{name:metrics(req,pd.read_parquet(destination/f"{name}.parquet"))[0] for name in ("ann","swing","usercf")},
        "loss_decomposition":losses,"candidate_budget":500,"route_budget":1000,
        "limitations":manifest["limitations"]}
    result["index_diagnostic"]={name:metrics(req,pd.read_parquet(destination/name))[0]
        for name in ("ann_nprobe32.parquet","ann_exact.parquet")}
    result["timing"]=timing
    # Freeze-selected weights in every final ablation; official-test outputs never feed selection.
    ablations={}
    for variant in CF_CONFIGS:
        folder=destination/"ablations"/variant["name"]
        folder.mkdir(parents=True,exist_ok=True)
        cache=out/f"cf_graph_{variant['history_weight']:g}"
        query_cf(req,sparse.load_npz(cache/"graph.npz"),np.load(cache/"users.npy"),
            np.load(cache/"items.npy"),sparse.load_npz(cache/"swing.npz"),variant,folder,folder/"support.parquet")
        set_pool(con,[destination/"ann.parquet",folder/"swing.parquet",folder/"usercf.parquet"])
        score,per_request=sql_route_metrics(con,f"{a}*a+{s}*s+{u}*u")
        per_request.to_parquet(folder/"fused_metrics.parquet",index=False)
        ablations[variant["name"]]={"fused_fixed_selected_weights":score,
            "routes":{name:metrics(req,pd.read_parquet(folder/f"{name}.parquet"))[0] for name in ("swing","usercf")}}
    result["cf_ablations"]=ablations
    write_json(destination/"metrics.json",result)
    write_json(destination/"source_manifest.json",{"selection_sha256":digest(out/"selection.json"),
        "model_training_source_snapshot_sha256":digest(out/"dssm/training_source.py"),
        "final_script_sha256":digest(Path(__file__)),"raw_test_sha256":digest(raw),
        "outputs":{path.name:digest(path) for path in destination.glob("*.parquet")}})
    print(json.dumps(result,indent=2),flush=True)


def refit_cf_pretest(out):
    """Frozen-hyperparameter CF refit on all complete official-train past requests."""
    selection=json.loads((out/"selection.json").read_text())
    original=json.loads((out/"manifest.json").read_text())
    destination=out/"refit_cf_pretest"
    if (destination/"metrics.json").exists():raise RuntimeError("Pretest refit result already exists")
    destination.mkdir(exist_ok=True)
    con=duckdb.connect(config={"threads":"2","memory_limit":"4GB"})
    raw=ROOT/"datasets/recommendation_train/train-00000-of-00001.parquet"
    if digest(raw)!=original["inputs"][str(raw.relative_to(ROOT))]:
        raise ValueError("Official train source changed")
    con.execute("""CREATE TABLE labels AS SELECT r.request_idx gid,first(r.user_idx) user_idx,
        d.note_idx,max(d.click)::INT positive,min(d.request_timestamp) lo,max(d.request_timestamp) hi
        FROM read_parquet(?) r,unnest(r.rec_result_details_with_idx) u(d) GROUP BY r.request_idx,d.note_idx""",[str(raw)])
    con.execute("""CREATE TABLE eligible AS SELECT gid,min(lo) lo,max(hi) hi FROM labels
        GROUP BY gid HAVING max(hi)<?""",[original["official_test_first_event"]])
    labels=con.execute("SELECT l.* FROM labels l JOIN eligible USING(gid)").fetchdf()
    requests=con.execute("""SELECT r.request_idx gid,r.user_idx,r.recent_clicked_note_idxs history,
        first(e.hi) hi,list_sort(list(DISTINCT l.note_idx) FILTER(WHERE l.positive>0)) positives
        FROM read_parquet(?) r JOIN eligible e ON r.request_idx=e.gid
        JOIN labels l ON l.gid=r.request_idx GROUP BY r.request_idx,r.user_idx,r.recent_clicked_note_idxs
        ORDER BY gid""",[str(raw)]).fetchdf()
    labels.to_parquet(destination/"train_labels.parquet",index=False)
    requests.to_parquet(destination/"train_requests.parquet",index=False)
    config=selection["selected"]["cf"]
    started=time.monotonic()
    graph,users,items=build_graph(labels,requests,config["history_weight"])
    swing=build_swing(graph,users,items)
    sparse.save_npz(destination/"graph.npz",graph);sparse.save_npz(destination/"swing.npz",swing)
    np.save(destination/"users.npy",users);np.save(destination/"items.npy",items)
    test=pd.read_parquet(out/"test/requests.parquet")
    query_cf(test,graph,users,items,swing,config,destination,destination/"support.parquet")
    con.execute("CREATE TABLE requests AS SELECT * FROM read_parquet(?)",[str(out/"test/requests.parquet")])
    set_pool(con,[out/"test/ann.parquet",destination/"swing.parquet",destination/"usercf.parquet"])
    a,s,u=selection["selected"]["weights"]
    score,per_request=sql_route_metrics(con,f"{a}*a+{s}*s+{u}*u")
    per_request.to_parquet(destination/"selected_metrics.parquet",index=False)
    con.execute("SELECT * FROM selected WHERE rank<=1000 ORDER BY gid,rank").fetchdf().to_parquet(destination/"selected.parquet",index=False)
    # Exact same model, index, route budgets, weights; only the fitted CF history interval changes.
    main=pd.read_parquet(out/"test/selected_metrics.parquet")
    paired=per_request[["gid","user_idx","recall"]].merge(main[["gid","recall"]],on="gid",suffixes=("_refit","_main"),validate="one_to_one")
    paired["delta"]=paired.recall_refit-paired.recall_main
    grouped=paired.groupby("user_idx").delta.agg(["sum","count"]).to_numpy()
    rng=np.random.default_rng(SEED);samples=[]
    for _ in range(1000):
        draw=grouped[rng.integers(0,len(grouped),len(grouped))]
        samples.append(float(draw[:,0].sum()/draw[:,1].sum()))
    support=pd.read_parquet(destination/"support.parquet")
    result={"scene":"rec","purpose":"Frozen-parameter train-only pretest CF refit ablation; no parameter selection on test",
        "selected_configuration_sha256":digest(out/"selection.json"),
        "source_train_sha256":digest(raw),"source_script_sha256":digest(Path(__file__)),
        "fit_rule":"whole original-train request max(timestamp) < official-test first timestamp",
        "test_start":original["official_test_first_event"],
        "train_requests":len(requests),"train_positive_pairs":int(labels.positive.sum()),
        "first_train_event":float(labels.lo.min()),"last_train_event":float(labels.hi.max()),
        "users":len(users),"items":len(items),"edges":graph.nnz,"swing_edges":swing.nnz,
        "validation_requests_recovered":int(len(set(requests.gid)&set(pd.read_parquet(out/"requests_validation.parquet").gid))),
        "config":config,"DSSM":"original frozen selected epoch2 unchanged","index":"same selected PQ128 unchanged",
        "test_fused":score,"test_routes":{name:metrics(test,pd.read_parquet(destination/f"{name}.parquet"))[0]
            for name in ("swing","usercf")},
        "paired_macro_delta_from_fit_only":float(paired.delta.mean()),
        "user_cluster_CI95":[float(value) for value in np.quantile(samples,[.025,.975])],
        "test_support":{"empty_swing":int((support.swing_pool_size==0).sum()),
            "empty_usercf":int((support.usercf_pool_size==0).sum())},
        "seconds":time.monotonic()-started,"limitations":[
            item for item in original["limitations"] if not item.startswith("No validation/test labels")]
            +["Validation labels/histories reclaimed only after hyperparameters were frozen; no official-test data fitted"]}
    write_json(destination/"metrics.json",result)
    print(json.dumps(result,indent=2),flush=True)


TAIL_QUOTA_GRID=((100,100,100),(200,100,100),(300,50,50),(100,150,150),(0,200,200))


def protected_tail_quota(baseline, route_items, global_scores, quotas, budget=500, protect=100):
    """Keep the original head; reserve distinct tail items, then fill by original RRF."""
    if len(quotas)!=3 or any(q<0 for q in quotas) or sum(quotas)>budget-protect:
        raise ValueError("Tail reservations exceed the unchanged candidate budget")
    head=list(baseline[:protect]);chosen=set(head);tail=[]
    if len(chosen)!=len(head):raise ValueError("Duplicate protected head items")
    # Fixed route order ANN, Swing, UserCF; overlap never consumes another reserved slot.
    fulfilled=[]
    for items,quota in zip(route_items,quotas):
        count=0
        for item in items:
            if count>=quota:break
            if item not in chosen:
                chosen.add(item);tail.append(item);count+=1
        fulfilled.append(count)
    for item in baseline[protect:]:
        if len(head)+len(tail)>=budget:break
        if item not in chosen:
            chosen.add(item);tail.append(item)
    tail.sort(key=lambda item:(-global_scores[item],item))
    result=head+tail
    if len(result)>budget or result[:len(head)]!=head:
        raise AssertionError("Protected-head or fixed-budget contract failed")
    return result,fulfilled


def quota_route_frames(paths):
    con=duckdb.connect(config={"threads":"2","memory_limit":"3GB"})
    values=[]
    for path in paths:
        frame=con.execute("""SELECT gid,list(note_idx ORDER BY score DESC,note_idx) items
            FROM read_parquet(?) GROUP BY gid ORDER BY gid""",[str(path)]).fetchdf()
        values.append({int(row.gid):list(map(int,row.items)) for row in frame.itertuples()})
    con.close()
    return values


def quota_replay(requests, baseline_path, route_paths, weights, quota_grid, candidate_output=None):
    con=duckdb.connect(config={"threads":"2","memory_limit":"2GB"})
    base=con.execute("""SELECT gid,list(note_idx ORDER BY rank) items FROM read_parquet(?)
        GROUP BY gid ORDER BY gid""",[str(baseline_path)]).fetchdf()
    con.close()
    baseline={int(row.gid):list(map(int,row.items)) for row in base.itertuples()}
    routes=quota_route_frames(route_paths)
    metric_rows={quotas:[] for quotas in quota_grid}
    candidates=[]
    for request in requests.itertuples():
        gid=int(request.gid);truth=set(map(int,request.positives))
        order=baseline.get(gid,[])
        items_by_route=[route.get(gid,[]) for route in routes]
        scores=defaultdict(float)
        for items,weight in zip(items_by_route,weights):
            for rank,item in enumerate(items,1):scores[item]+=weight/(60.+rank)
        for quotas in quota_grid:
            selected,fulfilled=protected_tail_quota(order,items_by_route,scores,quotas)
            ranks=[rank for rank,item in enumerate(selected,1) if item in truth]
            base_first=min((rank for rank,item in enumerate(order[:100],1) if item in truth),default=10**12)
            first=min(ranks,default=10**12)
            mrr=1./first if first<=100 else 0.
            if mrr!=(1./base_first if base_first<=100 else 0.):
                raise AssertionError("MRR100 changed under protected head")
            metric_rows[quotas].append((gid,int(request.user_idx),len(truth),len(ranks),first,
                len(ranks)/len(truth) if truth else np.nan,mrr,*fulfilled))
            if candidate_output is not None:
                if len(quota_grid)!=1:raise ValueError("Only materialize the frozen selected quota")
                candidates.extend((gid,item,float(scores[item]),rank) for rank,item in enumerate(selected,1))
    summary={};frames={}
    for quotas,rows in metric_rows.items():
        frame=pd.DataFrame(rows,columns=["gid","user_idx","positives","hits","first_hit","recall","mrr100",
            "ann_tail_reserved","swing_tail_reserved","usercf_tail_reserved"])
        valid=frame[frame.positives>0]
        summary["_".join(map(str,quotas))]={"quotas":list(quotas),"all_requests":len(frame),
            "Recall@500":float(valid.recall.mean()),"HitRate@500":float((valid.hits>0).mean()),
            "MRR@100":float(valid.mrr100.mean()),"positive_hits":int(valid.hits.sum()),
            "positive_pairs":int(valid.positives.sum()),
            "mean_fulfilled_reservations":frame[["ann_tail_reserved","swing_tail_reserved","usercf_tail_reserved"]].mean().tolist()}
        frames[quotas]=frame
    if candidate_output is not None:
        pd.DataFrame(candidates,columns=["gid","note_idx","score","rank"]).to_parquet(candidate_output,index=False)
    return summary,frames


def quota_tail_ablation(out):
    """Finite validation-only protected-head quota ablation, with frozen final test."""
    destination=out/"tail_quota_ablation"
    destination.mkdir(exist_ok=True)
    if (destination/"selection.json").exists() or (destination/"rejected.json").exists():
        raise RuntimeError("Quota ablation has already been frozen")
    original=json.loads((out/"selection.json").read_text())
    name=original["selected"]["cf"]["name"]
    ann=out/f"index/validation_nprobe{original['index']['nprobe']}.parquet"
    val_paths=[ann,out/"validation"/name/"swing.parquet",out/"validation"/name/"usercf.parquet"]
    protocol={"grid":TAIL_QUOTA_GRID,"head":"original selected RRF top100, same order",
        "tail":"ANN then Swing then UserCF distinct reserved items, unused reservations filled by original RRF; tail sorted by original RRF",
        "weights":original["selected"]["weights"],"route_topk":1000,"final_budget":500,
        "selection":"macro validation Recall500 only; all protected-head MRR100 must equal frozen baseline",
        "baseline_validation":original["selected"],"original_selection_sha256":digest(out/"selection.json"),
        "inputs":{str(path.relative_to(ROOT)):digest(path) for path in [out/"validation_selected.parquet",*val_paths]}}
    write_json(destination/"protocol.json",protocol)
    req=pd.read_parquet(out/"requests_validation.parquet")
    summary,frames=quota_replay(req,out/"validation_selected.parquet",val_paths,
        original["selected"]["weights"],TAIL_QUOTA_GRID)
    write_json(destination/"validation_grid.json",summary)
    best=max(summary.values(),key=lambda row:row["Recall@500"])
    for key,value in summary.items():
        if abs(value["MRR@100"]-original["selected"]["MRR@100"])>1e-12:
            raise AssertionError("MRR100 guard failed")
        frames[tuple(value["quotas"])].to_parquet(destination/f"validation_{key}_requests.parquet",index=False)
    if best["Recall@500"]<=original["selected"]["Recall@500"]+1e-15:
        write_json(destination/"rejected.json",{"reason":"No finite-grid quota improves frozen validation macro Recall500",
            "best":best,"baseline":original["selected"]})
        print("Quota rejected on validation",best,flush=True)
        return
    chosen=tuple(best["quotas"])
    frozen={"selected_quotas":chosen,"validation":best,"original_selection_sha256":digest(out/"selection.json"),
        "protocol_sha256":digest(destination/"protocol.json"),"validation_grid_sha256":digest(destination/"validation_grid.json"),
        "frozen_before_test":True}
    write_json(destination/"selection.json",frozen)
    # Selection is written before even reading official test positive requests below.
    quota_replay(req,out/"validation_selected.parquet",val_paths,original["selected"]["weights"],[chosen],
        destination/"validation_selected.parquet")
    result={"validation":best,"test":{}}
    requests=pd.read_parquet(out/"test/requests.parquet")
    for scene,base,paths in [
        ("fit_only",out/"test/selected.parquet",[out/"test"/f"{route}.parquet" for route in ("ann","swing","usercf")]),
        ("pretest_refit",out/"refit_cf_pretest/selected.parquet",
            [out/"test/ann.parquet",out/"refit_cf_pretest/swing.parquet",out/"refit_cf_pretest/usercf.parquet"])]:
        values,per_request=quota_replay(requests,base,paths,original["selected"]["weights"],[chosen],
            destination/f"{scene}_selected.parquet")
        selected_frame=per_request[chosen]
        baseline_metrics=pd.read_parquet(out/("test/selected_metrics.parquet" if scene=="fit_only"
            else "refit_cf_pretest/selected_metrics.parquet"))
        paired=selected_frame.merge(baseline_metrics[["gid","recall"]],on="gid",suffixes=("_quota","_baseline"),validate="one_to_one")
        paired["delta"]=paired.recall_quota-paired.recall_baseline
        blocks=paired.groupby("user_idx").delta.agg(["sum","count"]).to_numpy()
        rng=np.random.default_rng(SEED);draws=[]
        for _ in range(1000):
            sample=blocks[rng.integers(0,len(blocks),len(blocks))]
            draws.append(float(sample[:,0].sum()/sample[:,1].sum()))
        paired.to_parquet(destination/f"{scene}_paired_requests.parquet",index=False)
        result["test"][scene]={"metrics":next(iter(values.values())),"macro_delta":float(paired.delta.mean()),
            "user_cluster_CI95":[float(value) for value in np.quantile(draws,[.025,.975])]}
    result["scope"]="Additional frozen quota ablation; original main and refit results remain unchanged"
    write_json(destination/"metrics.json",result)
    print(json.dumps(result,indent=2),flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--stage",choices=["prepare","cf","train","index","select","test","refit-cf","quota-tail"],required=True)
    parser.add_argument("--loss-contract",choices=["same_request_or_same_item_v2","same_item_only_v1"],
        default="same_request_or_same_item_v2",help="v1 only reproduces the frozen historical training experiment")
    args=parser.parse_args()
    if args.stage=="train":
        train(args.output.resolve(),loss_contract=args.loss_contract)
        return
    {"prepare":prepare,"cf":cf,"train":train,"index":index_dssm,"select":select_fusion,"test":final_test,"refit-cf":refit_cf_pretest,"quota-tail":quota_tail_ablation}[args.stage](args.output.resolve())


if __name__=="__main__":
    main()
