"""Validation-only ranking funnel selection; frozen recipes precede official test access.

Unjudged catalog items are never converted into CTR/AUC negatives. NDCG here
measures retention of original clicked positives, with IDCG over all such positives.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from backend.offline.evaluate_full_test import confidence, sha256
from backend.online.item_features import get_item_feature_store
from backend.online.ranking.stack import shared_mapped_vectors
from training.rank_stack import RequestContext, group_minmax

ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-12
METRICS = ("HitRate@10", "HitRate@50", "Recall@100", "known_positive_NDCG@10", "MRR@10", "MRR@100")
LOGGED_GUARDS = ("NDCG@10", "AUC", "GAUC")


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def load_json(path):
    return json.loads(Path(path).read_text())


def scene_key(scene):
    return "search_idx" if scene == "search" else "request_idx"


def finite(value):
    return isinstance(value, (float, int)) and math.isfinite(value)


def eligible_weights(manifest):
    choices = manifest.get("dien", {}).get("blend_candidates", [])
    baseline = [choice for choice in choices if choice["weight"] == 0]
    if len(baseline) != 1:
        raise ValueError("Completed training must supply exactly one logged-validation weight0 baseline")
    floor = baseline[0]["validation"]
    if not all(finite(floor.get(name)) for name in LOGGED_GUARDS):
        raise ValueError("Logged-validation baseline needs finite NDCG/AUC/GAUC")
    eligible = []
    for choice in choices:
        weight = float(choice["weight"])
        metrics = choice["validation"]
        if not math.isfinite(weight) or not 0 <= weight <= 1:
            raise ValueError("Invalid fine fusion weight")
        if all(finite(metrics.get(name)) and metrics[name] >= floor[name]-EPS for name in LOGGED_GUARDS):
            eligible.append(weight)
    return sorted(set(eligible))


def model_contract(manifest):
    dien = manifest["dien"]
    # Appending a test report does not mutate the trained architecture, weights or validation gate.
    return {"scene": manifest["scene"], "key": manifest["key"], "features": manifest["features"],
            "numeric_features": dien["numeric_features"], "architecture": dien["architecture"],
            "score_contract": dien.get("score_contract", "request_minmax_v1"),
            "logged_validation_blends": dien["blend_candidates"], "training_inputs": manifest["inputs"]}


def contract_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify_files(records):
    for path, expected in records.items():
        if not Path(path).is_file() or sha256(Path(path)) != expected:
            raise ValueError(f"Frozen input content changed: {path}")


def original_requests(scene, split):
    folder = "search" if scene == "search" else "recommendation"
    key = scene_key(scene)
    nested = "search_result_details_with_idx" if scene == "search" else "rec_result_details_with_idx"
    paths = sorted((ROOT/"datasets"/f"{folder}_{split}").glob("*.parquet"))
    if not paths:
        raise ValueError("Original dataset split not found")
    frame = pd.concat([pd.read_parquet(path, columns=[key, "user_idx", "query", "recent_clicked_note_idxs", nested]) for path in paths], ignore_index=True)
    if frame[key].duplicated().any():
        raise ValueError("Original request identifiers must be unique")
    return frame, nested, paths


def validation_cohort(scene, request_path, runs):
    key = scene_key(scene)
    columns = pq.ParquetFile(request_path).schema.names
    source_key = key if key in columns else "gid"
    ids = pd.read_parquet(request_path, columns=[source_key])[source_key]
    if ids.isna().any() or ids.duplicated().any() or not len(ids):
        raise ValueError("Common validation requests must be unique and nonempty")
    original, nested, paths = original_requests(scene, "train")
    wanted = set(map(int, ids))
    if not wanted.issubset(set(original[key])):
        raise ValueError("Validation cohort must be entirely from original training requests")
    for run in runs:
        validation = pd.read_parquet(Path(run)/"validation.parquet", columns=[key])
        if not wanted.issubset(set(validation[key])):
            raise ValueError("Common validation includes a ranking-trained request")
        training = pd.read_parquet(Path(run)/"train.parquet", columns=[key, "partition"])
        if wanted.intersection(training.loc[training.partition == "fit", key]):
            raise ValueError("Ranking fit/validation request overlap")
    return original[original[key].isin(wanted)].copy().reset_index(drop=True), nested, paths


def request_table(raw, key, nested):
    out = raw[[key, "user_idx", "query", "recent_clicked_note_idxs"]].copy()
    out["positive_note_ids"] = [sorted({int(item["note_idx"]) for item in details if float(item["click"]) > 0}) for details in raw[nested]]
    return out


def read_candidates(path, key, allowed_ids, cap):
    columns = pq.ParquetFile(path).schema.names
    source_key = key if key in columns else "gid"
    with duckdb.connect(config={"threads": "2", "memory_limit": "2GB"}) as connection:
        foreign = connection.execute(f"SELECT DISTINCT {source_key} FROM read_parquet(?)", [str(path)]).fetchnumpy()[source_key]
        if not set(map(int, foreign)).issubset(allowed_ids):
            raise ValueError("Candidate file contains foreign request IDs")
        frame = connection.execute(f"SELECT {source_key} AS {key},note_idx,rank FROM read_parquet(?) WHERE rank<=? ORDER BY {source_key},rank", [str(path), cap]).fetchdf()
    for name in [key, "note_idx", "rank"]:
        values = pd.to_numeric(frame[name], errors="raise")
        if not np.isfinite(values).all() or (values != np.floor(values)).any():
            raise ValueError("Candidate IDs and ranks must be finite integers")
        frame[name] = values.astype(np.int64)
    if (frame.note_idx < 0).any() or (frame["rank"] < 1).any() or frame.duplicated([key, "note_idx"]).any() or frame.duplicated([key, "rank"]).any():
        raise ValueError("Invalid or duplicate candidate item/rank")
    return frame


def make_context(scene, requests, query_cache):
    key = scene_key(scene)
    text = shared_mapped_vectors(ROOT/"embeddings/note_text_emb", "note_text")
    image = shared_mapped_vectors(ROOT/"embeddings/note_img_emb", "note_img")
    vectors, inputs, commit = None, list(text.files)+list(image.files), None
    if scene == "search":
        cache = Path(query_cache)
        metadata = [cache] if cache.is_file() else list(cache.glob("*.json"))
        if len(metadata) != 1:
            raise ValueError("Search requires one explicit raw-query cache")
        path = metadata[0]
        meta = load_json(path)
        values = np.load(path.with_suffix(".npy"), mmap_mode="r")
        mapping = {int(gid): index for index, gid in enumerate(meta["request_ids"])}
        if not set(requests[key]).issubset(mapping):
            raise ValueError("Missing raw-query cache requests")
        vectors = np.asarray([values[mapping[int(gid)]] for gid in requests[key]], dtype=np.float32)
        if vectors.shape != (len(requests), 768) or not np.isfinite(vectors).all() or (np.linalg.norm(vectors, axis=1) < 1e-6).any():
            raise ValueError("Invalid cached raw-query vectors")
        if hashlib.sha256(memoryview(values).cast("B")).hexdigest() != meta["vectors_sha256"]:
            raise ValueError("Raw-query cache content mismatch")
        inputs += [path, path.with_suffix(".npy")]
        commit = meta.get("model_commit")
    return RequestContext(requests[[key, "query", "recent_clicked_note_idxs"]], key, text, image, vectors), inputs, commit


def prepare(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    requests, nested, sources = validation_cohort(args.scene, args.validation_requests, args.run)
    table = request_table(requests, scene_key(args.scene), nested)
    candidates = read_candidates(args.recall, scene_key(args.scene), set(table[scene_key(args.scene)]), 160 if args.scene == "search" else 320)
    context, context_inputs, commit = make_context(args.scene, table, args.query_cache)
    static = get_item_feature_store(ROOT)
    features = context.feature_frame(candidates, static)
    table.to_parquet(out/"requests.parquet", index=False)
    features.to_parquet(out/"features.parquet", index=False)
    inputs = {str(path.resolve()): sha256(path) for path in [*sources, args.validation_requests, args.recall, *context_inputs, *static.input_files]}
    plan = {"scene": args.scene, "phase": "validation", "input_cap": 160 if args.scene == "search" else 320,
            "coarse_caps": [80, 120, 160] if args.scene == "search" else [180, 240, 320],
            "runs": [str(path.resolve()) for path in args.run], "query_cache": str(args.query_cache.resolve()) if args.query_cache else None,
            "query_encoder_commit": commit, "requests": len(table), "candidate_rows": len(features), "inputs": inputs,
            "prepared": {str((out/name).resolve()): sha256(out/name) for name in ["requests.parquet", "features.parquet"]},
            "selection_rule": "STRICT Recall100 gain vs raw recall; HR10Search/HR50Rec and knownpositiveNDCG10 >= raw; Search MRR100 >=raw; logged ND/AUC/GAUC nonregression vsweight0"}
    save_json(out/"prepare_manifest.json", plan)
    print(json.dumps({"prepared": str(out), "requests": len(table), "candidate_rows": len(features)}), flush=True)


def score_run(run, features, requests, scene, query_cache, device, batch_size):
    import lightgbm as lgb
    import torch
    from training.dien_ranker import DIENRanker
    manifest = load_json(run/"manifest.json")
    if manifest["scene"] != scene or manifest["key"] != scene_key(scene):
        raise ValueError("Ranking model scene mismatch")
    weights = eligible_weights(manifest)
    arch = manifest["dien"]["architecture"]
    if arch.get("inference_precision") != "fp32":
        raise ValueError("Funnel selection requires frozen FP32 inference")
    context, _, _ = make_context(scene, requests, query_cache)
    frame = features.copy()
    tree = lgb.Booster(model_file=str(run/"gbdt.txt"))
    if tree.feature_name() != manifest["features"]:
        raise ValueError("Teacher feature order differs from manifest")
    frame["coarse_score"] = tree.predict(frame[manifest["features"]], num_threads=4).astype(np.float32)
    numeric = manifest["dien"]["numeric_features"]
    if numeric != [*manifest["features"], "coarse_score"]:
        raise ValueError("Fine feature schema differs from the selected teacher")
    model = DIENRanker(scene, numeric, **arch).to(device)
    model.load_state_dict(torch.load(run/"dien.pt", map_location=device, weights_only=True))
    model.eval()
    pieces = []
    with torch.inference_mode():
        for start in range(0, len(frame), batch_size):
            batch = frame.iloc[start:start+batch_size]
            raw, _ = model(context.batch(batch, numeric, device))
            pieces.append(raw.detach().float().cpu().numpy().reshape(-1))
            if (start//batch_size+1) % 200 == 0:
                print(json.dumps({"model": str(run), "scored": min(start+batch_size, len(frame)), "total": len(frame)}), flush=True)
    raw = np.concatenate(pieces) if pieces else np.asarray([], dtype=np.float32)
    if not np.isfinite(raw).all() or not np.isfinite(frame.coarse_score).all():
        raise ValueError("Nonfinite model scores")
    frame["dien_raw_score"] = raw
    return frame[[scene_key(scene), "note_idx", "rank", "coarse_score", "dien_raw_score"]], {
        "run": str(run.resolve()), "weights": weights, "contract": model_contract(manifest),
        "contract_sha256": contract_hash(model_contract(manifest)), "manifest_sha256": sha256(run/"manifest.json"),
        "gbdt_sha256": sha256(run/"gbdt.txt"), "dien_sha256": sha256(run/"dien.pt"),
        "device": str(device), "batch_size": batch_size, "precision": "FP32, TF32 disabled",
        "score_contract": manifest["dien"].get("score_contract", "request_minmax_v1")}


def score(args):
    import torch
    plan = load_json(args.output/"prepare_manifest.json")
    if plan["scene"] != args.scene or plan["phase"] != "validation":
        raise ValueError("Score phase is restricted to the prepared training-validation cohort")
    verify_files({**plan["inputs"], **plan["prepared"]})
    if (args.output/"score_manifest.json").exists():
        raise ValueError("Frozen model scores already exist")
    features = pd.read_parquet(args.output/"features.parquet")
    requests = pd.read_parquet(args.output/"requests.parquet")
    torch.set_num_threads(4)
    models = []
    for index, source in enumerate(plan["runs"]):
        run = Path(source)
        scores, model = score_run(run, features, requests, args.scene, Path(plan["query_cache"]) if plan["query_cache"] else None, args.device, args.batch_size)
        path = args.output/f"model_{index}_scores.parquet"
        if path.exists():
            raise ValueError("Refusing to overwrite prior candidate scores")
        scores.to_parquet(path, index=False)
        model["score_path"], model["score_sha256"] = str(path.resolve()), sha256(path)
        models.append(model)
    save_json(args.output/"score_manifest.json", {"scene": args.scene, "phase": "validation", "models": models,
        "prepare_sha256": sha256(args.output/"prepare_manifest.json"), "completed_before_test": True})


def metric_row(items, positive):
    positive = set(positive)
    if not positive:
        return {name: np.nan for name in METRICS}
    items = list(dict.fromkeys(map(int, items)))[:100]
    hits = np.asarray([note in positive for note in items], dtype=np.float64)
    ranks = np.flatnonzero(hits)+1
    top = hits[:10]
    ideal = (1/np.log2(np.arange(1, min(10, len(positive))+1)+1)).sum()
    return {"HitRate@10": float(hits[:10].any()), "HitRate@50": float(hits[:50].any()),
            "Recall@100": float(hits[:100].sum()/len(positive)),
            "known_positive_NDCG@10": float((top/np.log2(np.arange(1, len(top)+1)+1)).sum()/ideal),
            "MRR@10": float(1/ranks[0]) if len(ranks) and ranks[0] <= 10 else 0.,
            "MRR@100": float(1/ranks[0]) if len(ranks) and ranks[0] <= 100 else 0.}


def evaluate_recipe(scores, requests, scene, coarse_cap=None, weight=0., raw=False, score_contract="request_minmax_v1"):
    key = scene_key(scene)
    groups = {int(gid): {name: frame[name].to_numpy() for name in
        ["note_idx", "rank", *([] if raw else ["coarse_score", "dien_raw_score"])]}
        for gid, frame in scores.groupby(key, sort=False)}
    rows = []
    for request in requests.to_dict("records"):
        gid = int(request[key])
        frame = groups.get(gid)
        items = []
        if frame is not None and len(frame["note_idx"]):
            if raw:
                items = frame["note_idx"][np.argsort(frame["rank"], kind="stable")].tolist()
            else:
                from training.scoring import blend_rank_scores
                chosen = np.lexsort((frame["rank"], -frame["coarse_score"]))[:coarse_cap]
                blend = blend_rank_scores(frame["coarse_score"][chosen], frame["dien_raw_score"][chosen],
                    np.zeros(len(chosen), dtype=np.int8), weight, score_contract)
                order = np.lexsort((frame["rank"][chosen], -blend))
                items = frame["note_idx"][chosen[order]].tolist()
        rows.append({"gid": gid, "user_idx": int(request["user_idx"]), "positives": len(request["positive_note_ids"]),
                     "candidates": len(items), **metric_row(items, request["positive_note_ids"])})
    return pd.DataFrame(rows)


def summarize(frame):
    return {**{name: float(frame[name].mean()) if frame[name].notna().any() else None for name in METRICS},
            "requests": len(frame), "positive_requests": int((frame.positives > 0).sum()),
            "empty_candidate_requests": int((frame.candidates == 0).sum())}


def guard_passes(metrics, baseline, scene):
    guards = ["HitRate@10" if scene == "search" else "HitRate@50", "known_positive_NDCG@10"]
    if scene == "search":
        guards.append("MRR@100")
    return (finite(metrics.get("Recall@100")) and finite(baseline.get("Recall@100")) and
            metrics["Recall@100"] > baseline["Recall@100"]+EPS and all(
                finite(metrics.get(name)) and finite(baseline.get(name)) and metrics[name] >= baseline[name]-EPS for name in guards))


def choose_recipe(grid, baseline, scene):
    eligible = [entry for entry in grid if guard_passes(entry["metrics"], baseline, scene)]
    order = lambda entry: (entry["metrics"]["Recall@100"], entry["metrics"]["HitRate@10" if scene == "search" else "HitRate@50"],
        entry["metrics"]["known_positive_NDCG@10"], entry["metrics"]["MRR@100"], -entry["coarse_cap"], -entry["weight"])
    return max(eligible, key=order) if eligible else None


def release_passes(metrics, baseline, scene, accepted_on_validation):
    names = ["Recall@100", "HitRate@10" if scene == "search" else "HitRate@50", "known_positive_NDCG@10"]
    if scene == "search":
        names.append("MRR@100")
    return accepted_on_validation and all(finite(metrics.get(name)) and finite(baseline.get(name)) and
        metrics[name] >= baseline[name]-EPS for name in names)


def select(args):
    out = args.output
    if (out/"selection.json").exists():
        raise ValueError("Validation selection is frozen; refusing to retune")
    plan, scoring = load_json(out/"prepare_manifest.json"), load_json(out/"score_manifest.json")
    if plan["scene"] != args.scene or scoring["scene"] != args.scene or scoring["phase"] != "validation" or scoring["prepare_sha256"] != sha256(out/"prepare_manifest.json"):
        raise ValueError("Selection requires matching frozen validation scores")
    verify_files({**plan["inputs"], **plan["prepared"], **{model["score_path"]: model["score_sha256"] for model in scoring["models"]}})
    requests = pd.read_parquet(out/"requests.parquet")
    features = pd.read_parquet(out/"features.parquet", columns=[scene_key(args.scene), "note_idx", "rank"])
    baseline_rows = evaluate_recipe(features, requests, args.scene, raw=True)
    baseline_rows.to_parquet(out/"validation_raw_requests.parquet", index=False)
    baseline = summarize(baseline_rows)
    grid = []
    for model_index, model in enumerate(scoring["models"]):
        values = pd.read_parquet(model["score_path"])
        for cap in plan["coarse_caps"]:
            for weight in model["weights"]:
                rows = evaluate_recipe(values, requests, args.scene, cap, weight, score_contract=model["score_contract"])
                summary = summarize(rows)
                grid.append({"model_index": model_index, "run": model["run"], "coarse_cap": cap, "weight": weight,
                             "score_contract": model["score_contract"], "metrics": summary, "guard_passed": guard_passes(summary, baseline, args.scene)})
    save_json(out/"validation_grid.json", {"scene": args.scene, "baseline": baseline, "candidates": grid})
    selected = choose_recipe(grid, baseline, args.scene)
    diagnostic = max(grid, key=lambda item: (item["metrics"]["Recall@100"], item["metrics"]["known_positive_NDCG@10"])) if grid else None
    result = {"scene": args.scene, "selection_split": "original_train_common_forward_validation", "accepted": selected is not None,
        "selection_rule": plan["selection_rule"], "selected": selected, "best_diagnostic": diagnostic,
        "test_recipe": selected or diagnostic, "fallback": "raw_recall" if selected is None else None,
        "baseline": baseline, "input_cap": plan["input_cap"], "query_encoder_commit": plan["query_encoder_commit"],
        "models": scoring["models"], "score_manifest_sha256": sha256(out/"score_manifest.json"),
        "grid_sha256": sha256(out/"validation_grid.json"), "prepare_sha256": sha256(out/"prepare_manifest.json"),
        "scope": "Known-click retention, every common validation request; unjudged catalog items are not CTR negatives",
        "frozen_before_official_test": True}
    chosen = result["test_recipe"]
    if chosen is not None:
        rows = evaluate_recipe(pd.read_parquet(scoring["models"][chosen["model_index"]]["score_path"]), requests, args.scene, chosen["coarse_cap"], chosen["weight"], score_contract=chosen["score_contract"])
        rows.to_parquet(out/"validation_frozen_recipe_requests.parquet", index=False)
        result["paired_vs_raw"] = {}
        for name in METRICS:
            rows["delta"] = rows[name]-baseline_rows[name]
            result["paired_vs_raw"][name] = {"mean": float(rows.delta.mean()), "user_cluster_CI95": confidence(rows, "delta")}
    save_json(out/"selection.json", result)
    print(json.dumps({"accepted": result["accepted"], "selected": selected, "best_diagnostic": diagnostic}), flush=True)


def verify_frozen_selection(path, scene):
    selection = load_json(path)
    if selection.get("scene") != scene or not selection.get("frozen_before_official_test") or selection.get("selection_split") != "original_train_common_forward_validation":
        raise ValueError("Official test requires a frozen training-validation selection")
    if selection.get("test_recipe") is None:
        raise ValueError("No frozen diagnostic or accepted recipe is available")
    folder = Path(path).parent
    for name, expected in [("prepare_manifest.json", selection["prepare_sha256"]),
                           ("score_manifest.json", selection["score_manifest_sha256"]),
                           ("validation_grid.json", selection["grid_sha256"])]:
        if sha256(folder/name) != expected:
            raise ValueError("Frozen validation selection artifact changed")
    grid = load_json(folder/"validation_grid.json")
    if selection["test_recipe"] not in grid["candidates"]:
        raise ValueError("Test recipe was not evaluated on frozen validation")
    if selection["accepted"] and not guard_passes(selection["test_recipe"]["metrics"], selection["baseline"], scene):
        raise ValueError("Selected recipe did not pass the predeclared validation gate")
    for model in selection["models"]:
        run = Path(model["run"])
        if sha256(run/"gbdt.txt") != model["gbdt_sha256"] or sha256(run/"dien.pt") != model["dien_sha256"] or contract_hash(model_contract(load_json(run/"manifest.json"))) != model["contract_sha256"]:
            raise ValueError("Frozen ranking model/contract changed")
    return selection


def test(args):
    # This validation happens BEFORE opening any original test labels or candidates.
    selection = verify_frozen_selection(args.selection, args.scene)
    release_plan = ROOT/"outputs/evaluation/northstar_fix_20261031/release_plan.json"
    release_plan_hash = sha256(release_plan)
    out = args.output/"test"
    out.mkdir(parents=True, exist_ok=False)
    raw, nested, sources = original_requests(args.scene, "test")
    expected = 6192 if args.scene == "search" else 11115
    if len(raw) != expected:
        raise ValueError("Official test must retain every original request")
    requests = request_table(raw, scene_key(args.scene), nested)
    features = read_candidates(args.recall, scene_key(args.scene), set(requests[scene_key(args.scene)]), selection["input_cap"])
    context, context_inputs, commit = make_context(args.scene, requests, args.query_cache)
    if commit != selection["query_encoder_commit"]:
        raise ValueError("Official test query encoder differs from frozen validation")
    features = context.feature_frame(features, get_item_feature_store(ROOT))
    recipe = selection["test_recipe"]
    values, model = score_run(Path(recipe["run"]), features, requests, args.scene, args.query_cache, args.device, args.batch_size)
    values.to_parquet(out/"scores.parquet", index=False)
    raw_rows = evaluate_recipe(values, requests, args.scene, raw=True)
    rows = evaluate_recipe(values, requests, args.scene, recipe["coarse_cap"], recipe["weight"], score_contract=recipe["score_contract"])
    raw_rows.to_parquet(out/"raw_requests.parquet", index=False)
    rows.to_parquet(out/"recipe_requests.parquet", index=False)
    result = {"scene": args.scene, "accepted_on_validation": selection["accepted"], "recipe": recipe,
        "selection_sha256": sha256(args.selection), "raw_recall": summarize(raw_rows), "frozen_recipe": summarize(rows),
        "all_test_requests": len(raw), "model": model, "paired_vs_raw": {},
        "inputs": {str(path.resolve()): sha256(path) for path in [*sources, args.recall, *context_inputs]},
        "scope": "Complete original test known-click retention; frozen diagnostic remains unapproved if validation gate failed"}
    result["release_plan_sha256"] = release_plan_hash
    result["accepted_for_release"] = release_passes(result["frozen_recipe"], result["raw_recall"], args.scene, selection["accepted"])
    result["release_fallback"] = None if result["accepted_for_release"] else "recall_only"
    for name in METRICS:
        rows["delta"] = rows[name]-raw_rows[name]
        result["paired_vs_raw"][name] = {"mean": float(rows.delta.mean()), "user_cluster_CI95": confidence(rows, "delta")}
    save_json(out/"metrics.json", result)
    print(json.dumps({"scene": args.scene, "raw_recall": result["raw_recall"], "frozen_recipe": result["frozen_recipe"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=["search", "rec"], required=True)
    parser.add_argument("--phase", choices=["prepare", "score", "select", "test"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recall", type=Path)
    parser.add_argument("--validation-requests", type=Path)
    parser.add_argument("--run", type=Path, action="append", default=[])
    parser.add_argument("--query-cache", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.batch_size < 1:
        raise ValueError("Inference batch size must be positive")
    if args.phase == "prepare" and (not args.recall or not args.validation_requests or not args.run):
        parser.error("prepare needs recall, validation-requests and one or more run directories")
    if args.scene == "search" and args.phase in {"prepare", "test"} and args.query_cache is None:
        parser.error("Search prepare/test needs its explicit raw-query cache")
    if args.phase == "test" and (not args.recall or not args.selection):
        parser.error("test needs recall and an already frozen selection")
    {"prepare": prepare, "score": score, "select": select, "test": test}[args.phase](args)


if __name__ == "__main__":
    main()
