"""Full official-test snapshot audit; does not train, tune, or change serving artifacts."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from backend.offline.evaluation_contract import acceptance_checks

ROOT = Path(__file__).resolve().parents[3]
KS = (100, 500, 1000)
SEED = 42


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def confidence(frame, column, repetitions=500):
    """User-cluster bootstrap of a request-macro mean."""
    valid = frame.loc[frame[column].notna(), ["user_idx", column]]
    if valid.empty:
        return None
    grouped = valid.groupby("user_idx")[column].agg(["sum", "count"]).to_numpy()
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(repetitions):
        sample = grouped[rng.integers(0, len(grouped), len(grouped))]
        values.append(float(sample[:, 0].sum() / sample[:, 1].sum()))
    return [float(x) for x in np.quantile(values, [.025, .975])]


def recall_summary(requests):
    valid = requests[requests.positives > 0]
    result = {
        "all_requests": len(requests), "positive_requests": len(valid),
        "zero_positive_requests": int((requests.positives == 0).sum()),
        "empty_candidate_requests": int((requests.candidates == 0).sum()),
        "positive_pairs": int(requests.positives.sum()),
        "macro_denominator": "all positive requests, including empty and missed requests",
    }
    for k in KS:
        result[f"HitRate@{k}"] = float(valid[f"hit@{k}"].mean()) if len(valid) else None
        result[f"Recall@{k}"] = float(valid[f"recall@{k}"].mean()) if len(valid) else None
        result[f"MicroRecall@{k}"] = float(valid[f"hits@{k}"].sum() / valid.positives.sum()) if len(valid) else None
    result["MRR@100"] = float(valid["rr@100"].mean()) if len(valid) else None
    first = valid.first_hit.dropna()
    result["MedianFirstHitRank_among_hits@1000"] = float(first.median()) if len(first) else None
    result["hit_requests@1000"] = len(first)
    result["missed_requests@1000"] = len(valid) - len(first)
    result["user_cluster_CI95"] = {
        name: confidence(valid, column)
        for name, column in [("HitRate@500", "hit@500"), ("Recall@500", "recall@500"), ("MRR@100", "rr@100")]
    }
    return result


def ranking_metrics(frame, score):
    require(len(frame) == len(score) and np.isfinite(score).all(), "prediction count or finiteness failed")
    require(not frame.empty, "ranking frame must contain original exposures")
    require(frame.positive.isin([0, 1]).all(), "ranking labels must be binary")
    scored = frame[["gid", "user_idx", "note_idx", "positive"] + (["exposure_index"] if "exposure_index" in frame else [])].copy()
    scored["score"] = np.asarray(score, dtype=np.float64)
    rows = []
    for gid, group in scored.groupby("gid", sort=True):
        y = group.positive.to_numpy(dtype=float)
        s = group.score.to_numpy()
        n, p = len(y), int(y.sum())
        discount = 1 / np.log2(np.arange(2, min(n, 10) + 2))
        # Expected DCG over permutations inside equal-score blocks (no label-based tie breaking).
        order = np.argsort(-s, kind="stable")
        y, s = y[order], s[order]
        dcg = 0.
        starts = np.r_[0, np.flatnonzero(s[1:] != s[:-1]) + 1]
        ends = np.r_[starts[1:], n]
        for start, end in zip(starts, ends):
            if start >= 10:
                break
            dcg += float(y[start:end].mean() * discount[start:min(end, 10)].sum())
        ideal = float(discount[:min(p, 10)].sum())
        rows.append((gid, int(group.user_idx.iloc[0]), n, p,
                     dcg / ideal if ideal else np.nan,
                     (p / n) * float(discount.sum()) / ideal if ideal else np.nan))
    per_request = pd.DataFrame(rows, columns=["gid", "user_idx", "exposures", "positives", "ndcg@10", "random_ndcg@10"])
    y = scored.positive.to_numpy(dtype=int)
    auc = float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None
    user_aucs, weights = [], []
    for _, group in scored.groupby("user_idx"):
        if group.positive.nunique() == 2:
            user_aucs.append(roc_auc_score(group.positive, group.score))
            weights.append(len(group))
    result = {
        "scope": "all original official test logged exposures, repeated impressions retained; NOT serving funnel",
        "label": "binary click", "rows": len(scored), "all_requests": len(per_request),
        "NDCG@10": float(per_request["ndcg@10"].mean()) if per_request.positives.sum() else None,
        "AUC": auc, "GAUC": float(np.average(user_aucs, weights=weights)) if weights else None,
        "gauc_group": "user_idx", "gauc_weight": "exposure count of users with both classes",
        "gauc_effective_users": len(weights), "gauc_effective_rows": sum(weights),
        "gauc_excluded_users": int(scored.user_idx.nunique()) - len(weights),
        "zero_positive_requests": int((per_request.positives == 0).sum()),
        "random_expected_NDCG@10": float(per_request["random_ndcg@10"].mean()) if per_request.positives.sum() else None,
        "NDCG@10_user_cluster_CI95": confidence(per_request, "ndcg@10"),
        "tie_policy": "average DCG over equal-score permutations; AUC ties count 0.5",
    }
    return result, per_request, scored


def prepare_data(con, root, scene):
    key = "search_idx" if scene == "search" else "request_idx"
    folder = "search" if scene == "search" else "recommendation"
    nested = "search_result_details_with_idx" if scene == "search" else "rec_result_details_with_idx"
    timestamp = "search_timestamp" if scene == "search" else "request_timestamp"
    for split in ("train", "test"):
        path = str(root / "datasets" / f"{folder}_{split}" / "*.parquet")
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE requests_{split} AS
            SELECT {key} gid,user_idx,recent_clicked_note_idxs history,query FROM read_parquet(?)
        """, [path])
        require(con.execute(f"SELECT count(*)=count(DISTINCT gid) FROM requests_{split}").fetchone()[0],
                f"{scene}/{split}: duplicate request IDs")
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE raw_{split} AS
            SELECT r.{key} gid,r.user_idx,d.note_idx,d.click,d.{timestamp} event_time,d.position,exposure_index
            FROM read_parquet(?) r,unnest(r.{nested}) WITH ORDINALITY u(d,exposure_index)
        """, [path])
        require(con.execute(f"SELECT count(*) FROM raw_{split} WHERE gid IS NULL OR user_idx IS NULL OR note_idx IS NULL OR note_idx<0 OR click IS NULL OR click NOT IN (0,1) OR event_time IS NULL OR NOT isfinite(event_time)").fetchone()[0] == 0,
                f"{scene}/{split}: invalid IDs, click labels or event times")
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE labels_{split} AS
            SELECT gid,note_idx,first(user_idx) user_idx,max(click)::INT positive,
                   min(event_time) lo,max(event_time) hi
            FROM raw_{split} GROUP BY gid,note_idx
        """)
    overlap = con.execute("SELECT count(*) FROM requests_train JOIN requests_test USING(gid)").fetchone()[0]
    require(overlap == 0, f"{scene}: train/test request IDs overlap")
    test_start, test_end = con.execute("SELECT min(lo),max(hi) FROM labels_test").fetchone()
    train_start, train_end = con.execute("SELECT min(lo),max(hi) FROM labels_train").fetchone()
    require(all(value is not None and np.isfinite(value) for value in
                (test_start, test_end, train_start, train_end)), "train/test must contain finite event times")
    future = con.execute("SELECT count(DISTINCT gid) FROM labels_train WHERE hi>=?", [test_start]).fetchone()[0]
    counts = {}
    for split in ("train", "test"):
        counts[split] = {
            "requests": con.execute(f"SELECT count(*) FROM requests_{split}").fetchone()[0],
            "raw_exposures": con.execute(f"SELECT count(*) FROM raw_{split}").fetchone()[0],
            "deduplicated_exposures": con.execute(f"SELECT count(*) FROM labels_{split}").fetchone()[0],
        }
    return key, {"counts": counts, "train_test_request_overlap": overlap,
                 "train_event_range": [train_start, train_end], "test_event_range": [test_start, test_end],
                 "train_requests_not_strictly_before_first_test": future,
                 "chronological_split": train_end < test_start}


def evaluate_recall(con, root, scene, key, output):
    path = root / f"outputs/data/recall_{scene}_test_hard_multiroute_top1000.parquet"
    con.execute(f"""CREATE OR REPLACE TEMP VIEW candidates AS
        SELECT {key} gid,note_idx,rank,recall_score FROM read_parquet(\'{path}\')""")
    require(con.execute("SELECT count(*) FROM candidates WHERE gid IS NULL OR note_idx IS NULL OR note_idx<0 OR rank IS NULL OR rank<1 OR rank>1000 OR NOT isfinite(recall_score)").fetchone()[0] == 0, "invalid candidates")
    require(con.execute("SELECT count(*)-count(DISTINCT(gid,note_idx)) FROM candidates").fetchone()[0] == 0, "duplicate candidate items")
    require(con.execute("SELECT count(*)-count(DISTINCT(gid,rank)) FROM candidates").fetchone()[0] == 0, "duplicate candidate ranks")
    require(con.execute("SELECT count(*) FROM candidates ANTI JOIN requests_test USING(gid)").fetchone()[0] == 0, "unknown candidate requests")
    require(con.execute("SELECT count(*) FROM (SELECT gid,min(rank) lo,max(rank) hi,count(*) n FROM candidates GROUP BY gid) WHERE lo<>1 OR hi<>n").fetchone()[0] == 0, "candidate rank gaps")
    con.execute("""CREATE OR REPLACE TEMP TABLE positive_hits AS
        SELECT l.*,c.rank hit_rank FROM labels_test l LEFT JOIN candidates c USING(gid,note_idx)
        WHERE l.positive=1""")
    hits = con.execute("SELECT * FROM positive_hits ORDER BY gid,note_idx").fetchdf()
    hits["hit_rank"] = hits["hit_rank"].astype("float64")
    hits.to_parquet(output / "positive_hits.parquet", index=False)
    requests = con.execute("""
        SELECT r.gid,r.user_idx,coalesce(p.n,0) positives,p.first_hit,coalesce(c.n,0) candidates,
               array_length(r.history) history_length
        FROM requests_test r
        LEFT JOIN (SELECT gid,count(*) n,min(hit_rank) first_hit FROM positive_hits GROUP BY gid) p USING(gid)
        LEFT JOIN (SELECT gid,count(*) n FROM candidates GROUP BY gid) c USING(gid) ORDER BY gid
    """).fetchdf()
    for k in KS:
        count = hits[hits.hit_rank <= k].groupby("gid").size()
        requests[f"hits@{k}"] = requests.gid.map(count).fillna(0).astype(int)
        requests[f"hit@{k}"] = (requests[f"hits@{k}"] > 0).astype(float)
        requests[f"recall@{k}"] = requests[f"hits@{k}"] / requests.positives.replace(0, np.nan)
    requests["first_hit"] = requests["first_hit"].astype("float64")
    requests["rr@100"] = np.where(requests.first_hit <= 100, 1 / requests.first_hit, 0.)
    requests.to_parquet(output / "recall_requests.parquet", index=False)
    result = recall_summary(requests)
    result["candidate_rows"] = con.execute("SELECT count(*) FROM candidates").fetchone()[0]
    catalog_path = root / f"outputs/deploy/{scene}/hard/index/dssm_{scene}_hard_row2note.npy"
    catalog = np.load(catalog_path)
    require(len(np.unique(catalog)) == len(catalog), "duplicate ANN catalog IDs")
    result["ann_catalog_items"] = len(catalog)
    result["positive_pair_coverage_in_ann_catalog"] = float(hits.note_idx.isin(catalog).mean())
    train_items = con.execute("SELECT DISTINCT note_idx FROM labels_train WHERE positive=1").fetchnumpy()["note_idx"]
    hits["seen_in_train_clicks"] = hits.note_idx.isin(train_items)
    result["positive_item_strata"] = {}
    for seen, group in hits.groupby("seen_in_train_clicks"):
        result["positive_item_strata"]["seen" if seen else "unseen"] = {
            "positive_pairs": len(group), "positive_pair_hit@500": float((group.hit_rank <= 500).mean())}
    result["note"] = "Frozen merged candidates, not regenerated; catalog timestamp and training lineage unverified."
    return result, path


def predict_ranking(con, root, scene, key, output, log):
    import torch
    import xgboost as xgb
    from backend.online.preranking.gbdt import GBDTPredictor
    from backend.online.ranking.dien import DIENPredictor

    labels = con.execute("SELECT gid,note_idx,user_idx,click,click positive,event_time event_timestamp,position,exposure_index FROM raw_test ORDER BY gid,exposure_index").fetchdf()
    feature_path = root / f"features/{scene}_test_features.parquet"
    raw = pd.read_parquet(feature_path)
    duplicates = int(raw.duplicated([key, "note_idx"]).sum())
    features = raw.rename(columns={key: "gid"}).drop(columns=["user_idx", "y_multi", "recent_clicked_note_idxs"], errors="ignore")
    join_keys = ["gid", "note_idx", "event_timestamp", "position", "click"]
    # Distinct exposure identity is preserved. Repeated request-item impressions are NOT dropped.
    labels["_occurrence"] = labels.groupby(join_keys, dropna=False).cumcount()
    features["_occurrence"] = features.groupby(join_keys, dropna=False).cumcount()
    require(len(labels) == len(features), "raw test / feature exposure counts differ")
    frame = labels.merge(features, on=join_keys + ["_occurrence"], how="left", validate="one_to_one", indicator=True)
    require((frame["_merge"] == "both").all(), "missing exposure-aligned features")
    frame.drop(columns=["_merge", "_occurrence", "click"], inplace=True)
    frame[key] = frame.gid
    if scene == "search":
        frame = frame.drop(columns="query", errors="ignore").merge(
            con.execute("SELECT gid,query FROM requests_test").fetchdf(), on="gid", validate="many_to_one")
    recall_path = root / f"outputs/data/recall_{scene}_test_hard_multiroute_top1000.parquet"
    route = con.execute(f"""SELECT c.* EXCLUDE(user_idx),c.{key} gid FROM read_parquet(?) c
        SEMI JOIN labels_test l ON c.{key}=l.gid AND c.note_idx=l.note_idx""", [str(recall_path)]).fetchdf()
    extra = [name for name in route if name not in frame or name in ("gid", "note_idx")]
    frame = frame.merge(route[extra], on=["gid", "note_idx"], how="left", validate="many_to_one")
    route_missing = frame["rank"].isna()
    route_fields = ["rank", "recall_score", "score_ann", "score_swing", "score_usercf", "from_ann", "from_swing", "from_usercf"]
    require(all(name in frame for name in route_fields), "missing recall schema fields")
    frame[route_fields] = frame[route_fields].fillna(0.)
    frame["first_route_id"] = frame.first_route.map({"ann": 1, "swing": 2, "usercf": 3}).fillna(0).astype(np.int16)
    deploy = root / f"outputs/deploy/{scene}/hard"
    gbdt = GBDTPredictor(scene, "hard", deploy, key, list(frame.columns))
    require(gbdt.lgb_model is not None and gbdt.xgb_model is not None, "missing GBDT model")
    missing_gbdt = sorted(set(gbdt.feature_cols) - set(frame))
    require(not missing_gbdt, f"missing GBDT features: {missing_gbdt}")
    require(not ({"positive", "click", "y_multi"} & set(gbdt.feature_cols)), "label used as feature")
    X = gbdt._prepare_scoring_frame(frame)
    frame["lgb_score"] = gbdt.lgb_model.predict(X, num_iteration=gbdt.lgb_model.best_iteration or None)
    iteration = getattr(gbdt.xgb_model, "best_iteration", None)
    options = {"iteration_range": (0, int(iteration) + 1)} if iteration is not None else {}
    frame["xgb_score"] = gbdt.xgb_model.predict(xgb.DMatrix(X.to_numpy()), **options)
    frame["preranking_score"] = .5 * (frame.lgb_score + frame.xgb_score)
    # Conditional logged-pool rank, matching the training feature formula; not serving-pool rank.
    order = frame.sort_values(["gid", "preranking_score", "note_idx"], ascending=[True, False, True], kind="stable")
    frame.loc[order.index, "preranking_rank"] = order.groupby("gid").cumcount().to_numpy() + 1
    gbdt_result, gbdt_requests, gbdt_predictions = ranking_metrics(frame, frame.preranking_score.to_numpy())
    gbdt_predictions.to_parquet(output / "gbdt_predictions.parquet", index=False)
    gbdt_requests.to_parquet(output / "gbdt_requests.parquet", index=False)
    model = DIENPredictor(scene, "hard", deploy, key)
    require(model.model is not None, "missing DIEN model")
    require(not ({"positive", "click", "y_multi"} & set(model.num_feats)), "DIEN uses labels")
    missing = sorted(set(model.num_feats) - set(frame))
    require(not missing, f"missing DIEN features: {missing}")
    bad_numeric = {}
    numeric = {}
    for name in model.num_feats:
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float32)
        bad_numeric[name] = int((~np.isfinite(values)).sum())
        require(bad_numeric[name] == 0, f"invalid numeric feature: {name}")
        numeric[name] = values
    requests = con.execute("SELECT gid,history FROM requests_test ORDER BY gid").fetchdf()
    history = {int(gid): list(items) for gid, items in requests.itertuples(index=False, name=None)}
    unique = frame.gid.drop_duplicates().astype(int).tolist()
    mapping = {gid: index for index, gid in enumerate(unique)}
    sequences = torch.stack([model._get_history_seq_base_tensor(history[gid] or [-1]) for gid in unique])
    scores = np.empty(len(frame), dtype=np.float32)
    missing_text = 0
    missing_image = 0
    missing_query = 0
    if scene == "search":
        queries = sorted(set(str(q or "").strip() for q in frame["query"]))
        queries = [q for q in queries if q]
        require(model.query_model is not None and model.query_tokenizer is not None, "test query encoder unavailable")
        model.query_model.to(model.device)
        with torch.no_grad():
            for start in range(0, len(queries), 64):
                texts = queries[start:start+64]
                inputs = model.query_tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(model.device)
                encoded = model.query_model(**inputs).last_hidden_state[:, 0]
                encoded = encoded / encoded.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                for text, vector in zip(texts, encoded.detach().cpu().numpy()):
                    model._query_cache[text] = vector.reshape(1, -1)
                if start % 1024 == 0:
                    log("query_encoding_progress", completed=min(start+64, len(queries)), total=len(queries))
    with torch.no_grad():
        for start in range(0, len(frame), 512):
            batch = frame.iloc[start:start+512]
            gids = batch.gid.to_numpy(np.int64)
            nids = batch.note_idx.to_numpy(np.int64)
            feats = {name: torch.tensor(values[start:start+len(batch)], device=model.device).unsqueeze(-1)
                     for name, values in numeric.items()}
            indices = torch.tensor([mapping[int(gid)] for gid in gids], device=model.device)
            feats["seq_embs"] = sequences[indices]
            for label, store in (("note_text_emb", "t"), ("note_img_emb", "i")):
                feats[label] = model.stores[store].get_tensor(nids)
            missing_text += int((feats["note_text_emb"].norm(dim=-1) <= 1e-12).sum().item())
            missing_image += int((feats["note_img_emb"].norm(dim=-1) <= 1e-12).sum().item())
            if scene == "search":
                feats["query_emb"] = model._get_runtime_query_tensor(batch, gids)
                missing_query += int((feats["query_emb"].norm(dim=-1) <= 1e-12).sum().item())
            pred, _ = model.model(feats)
            scores[start:start+len(batch)] = pred.detach().cpu().numpy().reshape(-1)
            if start % 51200 == 0:
                log("ranking_progress", scene=scene, completed=start+len(batch), total=len(frame))
    for gid in unique[:5]:
        group = frame[frame.gid == gid]
        reference = model.predict(group, history_note_ids=history[gid] or [-1])
        np.testing.assert_allclose(scores[group.index], reference, rtol=1e-4, atol=1e-5)
    result, per_request, scored = ranking_metrics(frame, scores)
    per_request.to_parquet(output / "dien_requests.parquet", index=False)
    scored["lgb_score"] = frame.lgb_score
    scored["xgb_score"] = frame.xgb_score
    scored["preranking_rank"] = frame.preranking_rank
    scored.to_parquet(output / "dien_predictions.parquet", index=False)
    result["feature_audit"] = {
        "missing_numeric_features": missing, "invalid_numeric_values": bad_numeric,
        "repeated_request_item_feature_rows_retained": duplicates,
        "unrecalled_logged_rows_route_fields_explicitly_zero": int(route_missing.sum()),
        "zero_text_embedding_rows": missing_text, "zero_image_embedding_rows": missing_image,
        "zero_query_embedding_rows": missing_query,
        "exposure_position_features": sorted({"position", "pos_bucket"} & set(model.num_feats)),
        "preranking_score_definition": "0.5*lgb_score+0.5*xgb_score, training export contract",
        "preranking_rank_scope": "logged exposures only; not actual full retrieval pool",
        "numeric_features": model.num_feats,
    }
    del model, gbdt, frame, raw, sequences, numeric
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"gbdt": gbdt_result, "dien": result}, feature_path


def manifest_paths(root, scene):
    folder = "search" if scene == "search" else "recommendation"
    paths = list((root / "datasets" / f"{folder}_train").glob("*.parquet"))
    paths += list((root / "datasets" / f"{folder}_test").glob("*.parquet"))
    paths += [root / f"features/{scene}_test_features.parquet",
              root / f"outputs/data/recall_{scene}_test_hard_multiroute_top1000.parquet"]
    deploy = root / f"outputs/deploy/{scene}/hard"
    paths += list((deploy / "models").glob("*"))
    paths += list((deploy / "index").glob("*"))
    for directory in ("note_text_emb", "note_img_emb"):
        paths += list((root / "embeddings" / directory).glob("*.bin"))
        paths += list((root / "embeddings" / directory).glob("*map.json"))
    if scene == "search":
        paths += list((root / "embeddings/query_text_emb").glob("search_query*"))
    return sorted(set(path for path in paths if path.is_file()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene", choices=["search", "rec", "both"], default="both")
    parser.add_argument("--require-accepted", action="store_true", help="完整计算后，严格验收未通过则退出 2")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    log_path = args.output / "events.jsonl"
    def log(event, **fields):
        row = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "event": event, **fields}
        with log_path.open("a") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)
    import torch
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    scenes = ("search", "rec") if args.scene == "both" else (args.scene,)
    manifest = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_diff": subprocess.check_output(["git", "diff", "--stat"], cwd=ROOT, text=True),
        "python": platform.python_version(), "platform": platform.platform(),
        "cuda": torch.version.cuda, "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "packages": {name: importlib.metadata.version(name) for name in
                     ["numpy", "pandas", "duckdb", "scikit-learn", "torch", "xgboost", "lightgbm", "pyarrow", "faiss-cpu"]},
        "seed": SEED, "bootstrap": {"unit": "user_idx", "repetitions": 500},
        "command": f"PYTHONPATH=src OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python -m backend.offline.evaluate_full_test --output {args.output} --scene {args.scene}",
        "inputs": {}, "outputs": {},
    }
    paths = sorted(set(path for scene in scenes for path in manifest_paths(ROOT, scene)))
    paths += list((ROOT / "src/backend/offline").glob("evaluation_contract.py"))
    paths += [Path(__file__), ROOT / "scripts/verify_full_test.py", ROOT / "pyproject.toml", ROOT / "uv.lock"]
    if "search" in scenes:
        paths += sorted((Path.home() / ".cache/huggingface/hub/models--BAAI--bge-base-zh/snapshots").rglob("*"))
        paths = [p for p in paths if p.is_file()]
    paths += sorted((ROOT / "src/backend/online").rglob("*.py"))
    paths += sorted((ROOT / "src/training").glob("*.py"))
    # Persist exact evaluator source alongside the result; no dependency on historical Git objects.
    import shutil
    shutil.copy2(__file__, args.output / "evaluate_full_test.py")
    try:
        for path in paths:
            key = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
            log("fingerprint", path=key)
            manifest["inputs"][key] = {"size_bytes": path.stat().st_size, "sha256": sha256(path)}
        write_json(args.output / "manifest.json", manifest)
        results = {"protocol": {
            "name": "official_full_test_frozen_snapshot_v1", "sampling": "none",
            "selection": "no test tuning; existing hard artifacts are frozen",
            "labels": "raw binary clicks; duplicate request-item labels use max(click)",
            "recall_scope": "all official test requests, frozen full-catalog multi-route Top1000",
            "ranking_scope": "all original test exposure rows, including repeated request-item impressions; separate from serving funnel",
            "zero_positive_policy": "retain in coverage counts; exclude undefined recall/NDCG denominators",
            "strict_acceptance": "PENDING"},
            "scenes": {}}
        for scene in scenes:
            directory = args.output / scene
            directory.mkdir()
            log("scene_started", scene=scene)
            with duckdb.connect(config={"threads": "2", "memory_limit": "3GB"}) as con:
                key, audit = prepare_data(con, ROOT, scene)
                recall, _ = evaluate_recall(con, ROOT, scene, key, directory)
                log("recall_completed", scene=scene, requests=recall["all_requests"], recall500=recall["Recall@500"])
                ranking, _ = predict_ranking(con, ROOT, scene, key, directory, log)
                results["scenes"][scene] = {"data_audit": audit, "recall": recall, "ranking": ranking}
                results["acceptance"] = acceptance_checks(results["scenes"])
                results["protocol"]["strict_acceptance"] = results["acceptance"]["status"]
                write_json(args.output / "metrics.json", results)
            log("scene_completed", scene=scene)
        changed_inputs = []
        for name, entry in manifest["inputs"].items():
            path = Path(name) if Path(name).is_absolute() else ROOT / name
            if sha256(path) != entry["sha256"]:
                changed_inputs.append(name)
        require(not changed_inputs, f"Inputs changed during evaluation: {changed_inputs}")
        log("input_stability_verified", files=len(manifest["inputs"]))
        for path in sorted(args.output.rglob("*.parquet")):
            manifest["outputs"][str(path.relative_to(args.output))] = {"size_bytes": path.stat().st_size, "sha256": sha256(path)}
        manifest["outputs"]["metrics.json"] = {"sha256": sha256(args.output / "metrics.json")}
        write_json(args.output / "manifest.json", manifest)
        write_json(args.output / "status.json", {"execution": "COMPLETE", "strict_acceptance": results["acceptance"]["status"]})
        log("completed", strict_acceptance=results["acceptance"]["status"])
    except Exception as exc:
        write_json(args.output / "status.json", {"execution": "FAILED", "error": repr(exc)})
        log("failed", error=repr(exc))
        raise

    if args.require_accepted and results["acceptance"]["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
