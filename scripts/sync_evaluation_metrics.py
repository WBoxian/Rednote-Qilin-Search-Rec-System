"""Publish reproducible full-test metrics bound to explicitly promoted serving bundles.

Explicit --bundle-search/--bundle-rec paths create a preview only; they cannot
update the canonical snapshot or documentation. No training or model selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from backend.online.bundle import bundle_identity
from scripts.evaluation_comparisons import attach_comparisons
from scripts.evaluation_coverage import label_coverage

DEFAULT_BASE = ROOT / "outputs/evaluation/northstar_fix_20261031"
DEFAULT_OUTPUT = ROOT / "src/backend/online/full_test_metrics.json"
EXPECTED = {"search": (6192, 128191, 27597), "rec": (11115, 182567, 44683)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Sources:
    """Bind every directly consumed file, checking that it remains immutable."""
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.files = {}

    def add(self, path: Path) -> Path:
        path = path.resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise ValueError(f"Missing or non-project source: {path}")
        name = path.relative_to(self.root).as_posix()
        stat = path.stat()
        fingerprint = {"sha256": sha256(path), "bytes": stat.st_size}
        if name in self.files and self.files[name] != fingerprint:
            raise ValueError(f"Source changed while building snapshot: {path}")
        self.files[name] = fingerprint
        return path

    def read(self, path: Path) -> dict:
        return json.loads(self.add(path).read_text(encoding="utf-8"))

    def finish(self) -> dict:
        # Re-hash inputs after all calculations; do not publish a mixed snapshot.
        for name, before in list(self.files.items()):
            self.add(self.root / name)
            if self.files[name] != before:
                raise ValueError(f"Source changed while building snapshot: {name}")
        return dict(self.files)


def recall_metrics(requests: pd.DataFrame, candidates: pd.DataFrame) -> dict:
    """Unique clicked-item truth per original request; misses stay in denominators."""
    if requests.gid.duplicated().any() or requests.empty:
        raise ValueError("Raw request IDs must be unique and nonempty")
    candidates = candidates.copy()
    required = {"gid", "note_idx", "rank"}
    if not required.issubset(candidates.columns):
        raise ValueError("Candidates require gid,note_idx,rank")
    for name in required:
        values = pd.to_numeric(candidates[name], errors="raise")
        if values.isna().any() or not np.isfinite(values).all() or (values != values.astype(np.int64)).any():
            raise ValueError(f"Nonintegral candidate {name}")
        candidates[name] = values.astype(np.int64)
    if (candidates.note_idx < 0).any() or not candidates["rank"].between(1, 500).all():
        raise ValueError("Recall snapshot requires actual top500 candidates only")
    if candidates.duplicated(["gid", "note_idx"]).any() or candidates.duplicated(["gid", "rank"]).any():
        raise ValueError("Duplicate request-item or request-rank candidates")
    if not set(candidates.gid).issubset(set(requests.gid)):
        raise ValueError("Foreign request in candidate output")
    groups = {}
    for gid, group in candidates.groupby("gid", sort=False):
        group = group.sort_values("rank")
        if group["rank"].tolist() != list(range(1, len(group) + 1)):
            raise ValueError("Candidate ranks must be contiguous from one")
        groups[int(gid)] = list(zip(group.note_idx, group["rank"]))
    rows = []
    for record in requests.itertuples(index=False):
        truth = set(record.positives) if record.positives is not None else set()
        entries = groups.get(int(record.gid), [])
        ranks = sorted(int(rank) for note, rank in entries if int(note) in truth)
        first = ranks[0] if ranks else None
        positives = len(truth)
        row = {"positives": positives, "first": first, "candidates": len(entries)}
        for k in (10, 50, 100, 500):
            hits = sum(rank <= k for rank in ranks)
            row[f"hits{k}"] = hits
            row[f"hr{k}"] = float(hits > 0)
            row[f"recall{k}"] = hits / positives if positives else 0.0
        for k in (10, 100):
            row[f"mrr{k}"] = 1.0 / first if first is not None and first <= k else 0.0
        ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(positives, 10) + 1))
        row["ndcg10"] = sum(1 / math.log2(rank + 1) for rank in ranks if rank <= 10) / ideal if ideal else 0.0
        rows.append(row)
    all_rows = pd.DataFrame(rows)
    valid = all_rows[all_rows.positives > 0]
    if valid.empty:
        raise ValueError("No positive test requests")
    result = {
        "all_requests": len(all_rows), "positive_requests": len(valid),
        "zero_positive_requests": len(all_rows) - len(valid),
        "positive_pairs": int(valid.positives.sum()),
        "sampled_groups": len(all_rows), "effective_groups": len(valid),
        "empty_candidate_requests": int((all_rows.candidates == 0).sum()),
        "candidate_pairs": len(candidates), "candidate_budget": 500,
        "mean_candidates": float(all_rows.candidates.mean()),
        "scope": "All original test requests; unique clicked request-item pairs; actual top500 retrieval; no forced positives",
        "denominator": "All positive original test requests, including zero-hit and empty-candidate requests",
        "label": "binary click > 0; repeated positive impressions deduplicated per request-item",
        "known_positive_NDCG@10": float(valid.ndcg10.mean()),
    }
    for k in (10, 50, 100, 500):
        result[f"HitRate@{k}"] = float(valid[f"hr{k}"].mean())
        result[f"known_positive_pairs_top{k}"] = int(valid[f"hits{k}"].sum())
    for k in (100, 500):
        result[f"Recall@{k}"] = float(valid[f"recall{k}"].mean())
        result[f"MicroRecall@{k}"] = float(valid[f"hits{k}"].sum() / valid.positives.sum())
    for k in (10, 100):
        result[f"MRR@{k}"] = float(valid[f"mrr{k}"].mean())
    hit_ranks = valid["first"].dropna()
    result["MedianFirstHitRank"] = float(hit_ranks.median()) if len(hit_ranks) else None
    result["MedianFirstHitRank_among_hits_at500"] = result["MedianFirstHitRank"]
    result["median_scope"] = "Only requests with a hit in actual top500; misses excluded from median and reported separately"
    result["missed_requests_at_500"] = int((valid.hits500 == 0).sum())
    result["missed_requests@500"] = result["missed_requests_at_500"]
    return result


def load_raw(root: Path, scene: str, sources: Sources):
    folder, key, nested = (("search", "search_idx", "search_result_details_with_idx") if scene == "search"
                           else ("recommendation", "request_idx", "rec_result_details_with_idx"))
    paths = sorted((root / "datasets" / f"{folder}_test").glob("*.parquet"))
    if not paths:
        raise ValueError(f"Missing original {scene} test dataset")
    for path in paths:
        sources.add(path)
    with duckdb.connect() as con:
        count, unique = con.execute(f"SELECT count(*),count(DISTINCT {key}) FROM read_parquet(?)", [list(map(str, paths))]).fetchone()
        if count != unique:
            raise ValueError("Original request IDs are not unique")
        raw = con.execute(f"""SELECT r.{key} gid,
            coalesce(list_distinct(list(d.note_idx) FILTER(WHERE d.click>0)), []::BIGINT[]) positives
            FROM read_parquet(?) r LEFT JOIN unnest(r.{nested}) u(d) ON true
            GROUP BY r.{key} ORDER BY gid""", [list(map(str, paths))]).fetchdf()
        exposure_rows = con.execute(f"SELECT count(*) FROM read_parquet(?) r,unnest(r.{nested}) u(d)", [list(map(str, paths))]).fetchone()[0]
    return raw, exposure_rows


def verify_bundle(path: Path, scene: str, strategy: str, sources: Sources) -> dict:
    bundle = sources.read(path)
    if bundle.get("scene") != scene or bundle.get("tag") != "hard" or bundle.get("bundle_id") != bundle_identity(bundle):
        raise ValueError(f"Wrong scene/tag or invalid bundle identity: {path}")
    if bundle.get("ranking_strategy") != strategy:
        raise ValueError(f"Promoted {scene} strategy differs from frozen test release decision")
    return bundle


def conditional_ranking(root: Path, scene: str, selection: dict, sources: Sources, counts: tuple) -> dict:
    chosen = selection[scene]["selected"]
    if selection[scene].get("selection_split") != "train_internal_validation":
        raise ValueError("Exposure model was not chosen by training-only validation")
    run = (root / chosen["run"]).resolve()
    manifest = sources.add(run / "manifest.json")
    if sha256(manifest) != chosen["manifest_sha256"]:
        raise ValueError("Frozen validation-selected rank manifest changed")
    test_manifest = sources.read(run / "test_manifest.json")
    test_path = sources.add(run / "test_metrics.json")
    if test_manifest["frozen_training_manifest_sha256"] != chosen["manifest_sha256"] or test_manifest["metrics_sha256"] != sha256(test_path):
        raise ValueError("Ranking test metrics are not bound to frozen training manifest")
    test = json.loads(test_path.read_text())
    if (test["test_requests"], test["test_rows"]) != counts[:2]:
        raise ValueError("Ranking metrics do not cover all original test exposures")
    proof = sources.read(run / "independent_ranking_verification.json")
    if (proof.get("status") != "PASS" or proof.get("test_metrics_sha256") != sha256(test_path)
            or proof.get("frozen_training_manifest_sha256") != chosen["manifest_sha256"]
            or proof.get("test_manifest_sha256") != sha256(run / "test_manifest.json")):
        raise ValueError("Missing current independent raw-label ranking verification")
    result = {}
    for output_name, stage in (("preranking", "gbdt"), ("ranking", "selected")):
        metrics = dict(test[stage])
        if metrics["rows"] != counts[1] or metrics["all_requests"] != counts[0]:
            raise ValueError("Conditional ranking stage scope mismatch")
        metrics.update(sampled_groups=counts[0], effective_groups=counts[0], selection_split="train_internal_validation")
        result[output_name] = {"test": metrics}
    result["selection"] = {"name": chosen["name"], "run": run.relative_to(root).as_posix(),
                           "weight": chosen["weight"], "manifest_sha256": chosen["manifest_sha256"]}
    return result


def bind_recall_provenance(root, base, scene, candidate_path, bundle, sources):
    experiment = base / f"{scene}_v3"
    selection_path = sources.add(experiment / "selection.json")
    selection = json.loads(selection_path.read_text())
    policy = bundle["recall_policy"]
    if scene == "search":
        test = sources.read(experiment / "test_metrics.json")
        if (test["selection_sha256"] != sha256(selection_path)
                or test["parameter_selection_after_test"] is not False
                or test["candidate_sha256"].get(str(candidate_path)) != sha256(candidate_path)):
            raise ValueError("Search recall test does not bind frozen selection and actual candidates")
        expected_policy = selection["recall_policy"]
        if policy != expected_policy or test["recall_policy"] != expected_policy:
            raise ValueError("Search bundle policy differs from frozen evaluated recall policy")
        nprobe = test["nprobe"]
    else:
        test = sources.read(experiment / "refit_cf_pretest/northstar_metrics.json")
        if (not selection["frozen_before_own_official_test"]
                or test["selection_sha256"] != sha256(selection_path)
                or test["outputs"]["selected.parquet"] != sha256(candidate_path)):
            raise ValueError("Rec recall test does not bind frozen selection and actual candidates")
        chosen = selection["selected"]
        expected_weights = dict(zip(("ann", "swing", "usercf"), chosen["weights"]))
        expected_cf = {key: chosen["cf"][key] for key in ("fold_in", "neighbors", "history_weight")}
        if (policy["strategy"] != "weighted_rrf" or policy["weights"] != expected_weights
                or policy["rrf_k"] != chosen["rrf_k"] or policy["cf_config"] != expected_cf):
            raise ValueError("Rec bundle policy differs from frozen evaluated recall policy")
        nprobe = selection["index"]["nprobe"]
    if policy.get("topk") != 500 or policy.get("route_topk") != 1000:
        raise ValueError("Evaluated route/final budgets differ from serving")
    index_meta = sources.read(root / bundle["recall_dir"] / "index" / f"dssm_{scene}_hard_ivfpq_meta.json")
    if index_meta["nprobe"] != nprobe:
        raise ValueError("Serving ANN probe count differs from evaluated selection")
    return {"selection_sha256": sha256(selection_path), "nprobe": nprobe,
            "route_topk": 1000, "final_topk": 500}

def load_adapted_rec_release(root, bundle, release, sources):
    """Bind a trained content adapter and learned fusion to one frozen full-test run."""
    entries = release["sources"]
    required = {"selection", "test_metrics", "independent_verification", "model_selection",
                "model_metadata", "index_selection", "candidates", "raw_test"}
    if not required.issubset(entries):
        raise ValueError("Adapted Recommendation release sources are incomplete")
    docs = {key: sources.read(root / entries[key]["path"]) for key in required - {"candidates", "raw_test"}}
    plan, test, proof = (docs[key] for key in ("selection", "test_metrics", "independent_verification"))
    if (plan.get("scene") != "rec" or plan.get("selection_split") != "train_internal_validation"
            or plan.get("frozen_before_official_test") is not True
            or plan.get("inference_contract") != "request_fp32_v1"
            or test.get("parameter_choice_after_test") is not False
            or test.get("accepted_for_release") is not True
            or (test.get("all_requests"), test.get("unique_clicked_pairs")) != EXPECTED["rec"][::2]):
        raise ValueError("Adapted Recommendation selection or full-test scope is invalid")
    bindings = {"selection_sha256": entries["selection"]["sha256"],
                "candidate_sha256": entries["candidates"]["sha256"],
                "raw_test_sha256": entries["raw_test"]["sha256"]}
    if any(document.get(key) != value for document in (test, proof) for key, value in bindings.items()):
        raise ValueError("Adapted Recommendation proof binds different selection, truth or candidates")
    if (proof.get("status") != "PASS"
            or (proof.get("original_requests"), proof.get("original_positive_pairs")) != EXPECTED["rec"][::2]):
        raise ValueError("Adapted Recommendation lacks independent full-test verification")
    selected = docs["model_selection"].get("selected", docs["model_selection"])
    metadata, recipe = docs["model_metadata"], plan["recipe"]
    if (metadata.get("model_family") != "adapted_residual_dssm_v1"
            or metadata.get("gradient_update_scope") != "content_adapter_stage_only"
            or type(metadata.get("gradient_updates")) is not int or metadata["gradient_updates"] <= 0
            or metadata.get("packaging_gradient_updates") != 0
            or selected["checkpoint_sha256"] != recipe["model_checkpoint_sha256"]
            or selected["item_vectors_sha256"] != recipe["item_vectors_sha256"]
            or metadata.get("exported_full_checkpoint_sha256") != selected["checkpoint_sha256"]
            or bundle["packaging_provenance"]["full_checkpoint_source_sha256"] != selected["checkpoint_sha256"]):
        raise ValueError("Serving adapted DSSM differs from trained and evaluated checkpoint")
    if (entries.get("model_checkpoint", {}).get("sha256") != selected["checkpoint_sha256"]
            or entries.get("item_vectors", {}).get("sha256") != selected["item_vectors_sha256"]):
        raise ValueError("Adapted Recommendation omits actual selected checkpoint or vector bytes")
    for name in ("plan", "data_contract", "summary"):
        if entries.get("content_training_" + name, {}).get("sha256") != metadata.get("content_training_" + name + "_sha256"):
            raise ValueError("Adapted Recommendation omits the actual training " + name)
    steps = metadata.get("content_training_steps_by_epoch")
    if (not isinstance(steps, list) or not steps
            or any(type(value) is not int or value < 1 for value in steps)
            or sum(steps) != metadata["gradient_updates"]):
        raise ValueError("Adapted Recommendation training steps are inconsistent")
    policy = bundle["recall_policy"]
    from backend.online.bundle import _validate_learned_policy, LEARNED_FUSION_PATHS
    config = _validate_learned_policy(policy, "rec")
    expected = plan["recall_policy"]
    for key in ("strategy", "context_policy", "weights", "rrf_k", "route_topk", "topk", "cf_config"):
        if policy.get(key) != expected.get(key):
            raise ValueError("Serving learned fusion differs from frozen evaluation policy")
    if (policy["route_topk"], policy["topk"]) != (1000, 500):
        raise ValueError("Adapted Recommendation candidate budgets changed")
    for key in LEARNED_FUSION_PATHS:
        path = sources.add(root / config[key])
        digest = sha256(path)
        if (digest != recipe["fusion_assets_sha256"][key]
                or bundle["assets"].get(config[key]) != digest):
            raise ValueError("Serving learned fusion asset differs from evaluated source: " + key)
    if bundle["training_cutoff_exclusive"] != plan["training_cutoff_exclusive"]:
        raise ValueError("Serving Recommendation past-data cutoff differs from evaluation")
    index = sources.read(root / bundle["recall_dir"] / "index/dssm_rec_hard_ivfpq_meta.json")
    frozen_index = docs["index_selection"]
    for key in ("index_sha256", "index_type", "nprobe"):
        if index.get(key) != frozen_index.get(key) or index.get(key) != recipe.get(key):
            raise ValueError("Serving adapted ANN differs from frozen evaluated index")
    if (index.get("selected_checkpoint_sha256") != selected["checkpoint_sha256"]
            or index.get("selected_item_vectors_sha256") != selected["item_vectors_sha256"]):
        raise ValueError("Serving adapted ANN binds different towers or item vectors")
    for filename in ("graph.npz", "users.npy", "items.npy", "swing.npz"):
        path = sources.add(root / policy["cf_graph_dir"] / filename)
        if sha256(path) != entries["cf_" + filename]["sha256"]:
            raise ValueError("Serving CF differs from full-test evaluated past-only graph")
    release["lineage"] = dict(selection_sha256=entries["selection"]["sha256"],
        index_type=index["index_type"], nprobe=index.get("nprobe"), route_topk=1000, final_topk=500,
        inference_contract=plan["inference_contract"], model_family=metadata["model_family"],
        residual_learned_weight=metadata["residual_weight"], gradient_updates=metadata["gradient_updates"],
        gradient_update_scope=metadata["gradient_update_scope"], packaging_gradient_updates=0,
        content_training_plan_sha256=metadata["content_training_plan_sha256"],
        content_training_data_contract_sha256=metadata["content_training_data_contract_sha256"],
        pca_fit_split="original_train_fit", fusion_strategy="learned_rrf")
    return release


def load_rec_release(root, bundle, sources):
    bound = bundle.get("evaluation", {}).get("release_evidence")
    if bound is None:
        return None
    evidence_path = sources.add(root / bound["path"])
    if sha256(evidence_path) != bound["sha256"] or bundle["assets"].get(bound["path"]) != bound["sha256"]:
        raise ValueError("Release evidence is not bound to serving bundle")
    release = json.loads(evidence_path.read_text())
    if release.get("scene") != "rec" or release.get("ranking_strategy") != "recall_only" or not release.get("accepted_for_release"):
        raise ValueError("Invalid Recommendation release decision")
    for entry in release["sources"].values():
        path = sources.add(root / entry["path"])
        if sha256(path) != entry["sha256"]:
            raise ValueError("Recommendation release input changed")
    if release.get("release_schema") == "adapted_content_v1":
        return load_adapted_rec_release(root, bundle, release, sources)
    docs = {name: json.loads((root / release["sources"][name]["path"]).read_text())
            for name in ("selection", "test_metrics", "independent_verification", "model_selection", "index_selection")}
    selection, test = docs["selection"], docs["test_metrics"]
    if (not selection.get("accepted") or not selection.get("frozen_before_official_test")
            or not test.get("accepted_for_release") or test.get("parameter_choice_after_test") is not False
            or test.get("selection_sha256") != release["sources"]["selection"]["sha256"]
            or (test.get("all_requests"), test.get("unique_clicked_pairs")) != (11115, 44683)):
        raise ValueError("Recommendation release does not bind frozen full-test selection")
    fusion = "fusion_selection" in release["sources"]
    proof = docs["independent_verification"]
    model_candidate_key = "residual_candidates" if fusion else "candidates"
    if (proof.get("selection_sha256") != release["sources"]["selection"]["sha256"]
            or proof.get("source_sha256") != release["sources"]["raw_test"]["sha256"]
            or proof.get("graphs", {}).get("allpast", {}).get("candidate_sha256", {}).get("new") != release["sources"][model_candidate_key]["sha256"]):
        raise ValueError("Independent residual proof binds different selection, raw truth or candidates")
    if docs["independent_verification"].get("status") != "PASS":
        raise ValueError("Recommendation release lacks independent verification")
    model = docs["model_selection"].get("selected", docs["model_selection"])
    if (model["checkpoint_sha256"] != selection["model_checkpoint_sha256"]
            or model["item_vectors_sha256"] != selection["selected"]["item_vectors_sha256"]
            or bundle["packaging_provenance"]["full_checkpoint_source_sha256"] != model["checkpoint_sha256"]):
        raise ValueError("Serving residual DSSM differs from evaluated model")
    index_meta = sources.read(root / bundle["recall_dir"] / "index/dssm_rec_hard_ivfpq_meta.json")
    if index_meta["index_sha256"] != selection["index_sha256"] or index_meta["nprobe"] != 32:
        raise ValueError("Serving residual ANN differs from evaluated index")
    expected_weights = {"ann": 1, "swing": 2, "usercf": 1}
    expected_context = "filter_history"
    if fusion:
        selected = sources.read(root / release["sources"]["fusion_selection"]["path"])
        evaluated = sources.read(root / release["sources"]["fusion_test_metrics"]["path"])
        verified = sources.read(root / release["sources"]["fusion_independent_verification"]["path"])
        selection_sha = release["sources"]["fusion_selection"]["sha256"]
        if (not selected.get("accepted") or not selected.get("frozen_before_official_test")
                or not evaluated.get("accepted_for_release") or evaluated.get("parameter_choice_after_test") is not False
                or (evaluated.get("all_requests"), evaluated.get("clicked_pairs")) != (11115, 44683)
                or evaluated.get("selection_sha256") != selection_sha
                or evaluated.get("approval_sha256") != release["sources"]["fusion_approval"]["sha256"]):
            raise ValueError("Fusion release is not a frozen validation-selected full-test decision")
        if (verified.get("status") != "PASS" or verified.get("fusion_selection_sha256") != selection_sha
                or verified.get("frozen_model_selection_sha256") != release["sources"]["selection"]["sha256"]
                or verified.get("raw_test_sha256") != release["sources"]["raw_test"]["sha256"]
                or verified.get("graphs", {}).get("allpast", {}).get("candidate_sha256") != release["sources"]["candidates"]["sha256"]
                or evaluated["results"]["allpast"]["winner_candidates_sha256"] != release["sources"]["candidates"]["sha256"]):
            raise ValueError("Fusion proof does not bind current model, raw truth and candidate output")
        expected_weights = selected["selected"]["weights"]
        if evaluated["selected_weights"] != expected_weights or verified["winner_weights"] != expected_weights:
            raise ValueError("Fusion release weight choice differs from validation")
        expected_context = "weighted_filter_history"
    policy = bundle["recall_policy"]
    if (policy.get("context_policy") != expected_context or policy.get("weights") != expected_weights
            or policy.get("rrf_k") != 60 or policy.get("topk") != 500 or policy.get("route_topk") != 1000
            or policy.get("cf_config") != {"fold_in": True, "neighbors": 200, "history_weight": .25}):
        raise ValueError("Serving residual fusion differs from frozen evaluated recipe")
    for filename in ("graph.npz", "users.npy", "items.npy", "swing.npz"):
        bundled = sources.add(root / policy["cf_graph_dir"] / filename)
        if sha256(bundled) != release["sources"]["cf_" + filename]["sha256"]:
            raise ValueError("Serving CF graph differs from full-test evaluated graph")
    release["lineage"] = dict(selection_sha256=release["sources"]["selection"]["sha256"],
                              nprobe=32, route_topk=1000, final_topk=500,
                              model_family="residual_dssm_v1", residual_learned_weight=.25,
                              gradient_updates=0, pca_fit_split="original_train_fit")
    if fusion:
        release["lineage"]["fusion_selection_sha256"] = release["sources"]["fusion_selection"]["sha256"]
    return release


def load_search_numerical_replay(root, bundle, sources):
    bound = bundle.get("evaluation", {}).get("numerical_replay")
    if bound is None:
        return None
    path = sources.add(root / bound["path"])
    if sha256(path) != bound["sha256"] or bundle["assets"].get(bound["path"]) != bound["sha256"]:
        raise ValueError("Numerical replay evidence is not bound to serving bundle")
    release = json.loads(path.read_text())
    for entry in release["sources"].values():
        source = sources.add(root / entry["path"])
        if sha256(source) != entry["sha256"]:
            raise ValueError("Numerical replay source changed")
    plan = sources.read(root / release["sources"]["plan"]["path"])
    test = sources.read(root / release["sources"]["test_metrics"]["path"])
    proof = sources.read(root / release["sources"]["independent_verification"]["path"])
    if (plan.get("scene") != "search" or plan.get("inference_contract") != "request_fp32_v1"
            or not plan.get("frozen_before_official_test") or plan.get("input_cap") != 160 or plan.get("coarse_cap") != 160
            or not test.get("accepted_for_release") or test.get("all_test_requests") != 6192
            or test.get("parameter_choice_after_test") is not False
            or test.get("selection_sha256") != release["sources"]["plan"]["sha256"]):
        raise ValueError("Numerical replay violates frozen model inference contract")
    if (proof.get("status") != "PASS" or proof.get("selection_sha256") != release["sources"]["plan"]["sha256"]
            or proof.get("raw_candidate_sha256") != release["sources"]["candidates"]["sha256"]
            or proof.get("final_candidate_sha256") != release["sources"]["final_candidates"]["sha256"]
            or (proof.get("original_requests"), proof.get("original_positive_pairs")) != (6192, 27597)):
        raise ValueError("Numerical replay lacks matching independent full-test proof")
    if (plan.get("input_contract") != "LiveContextBuilder_bound_user_profile_v2"
            or plan.get("profile_categories") != ["gender_enc", "platform_enc", "age_enc", "location_enc"]):
        raise ValueError("Numerical replay omitted the shared live user-profile input contract")
    profile_entry = release["sources"].get("profile_input_verification")
    if profile_entry is None:
        raise ValueError("Numerical replay lacks full-request profile-input verification")
    profile = sources.read(root / profile_entry["path"])
    if (profile.get("status") != "PASS" or profile.get("original_requests") != 6192
            or profile.get("profile_mismatches") != 0 or profile.get("raw_context_mismatches") != 0
            or profile.get("nonzero_profile_requests", 0) <= 0):
        raise ValueError("Numerical replay has an invalid or neutral-only profile reference")
    if (profile.get("selection_sha256") != release["sources"]["plan"]["sha256"]
            or profile.get("input_contract") != plan["input_contract"]
            or profile.get("profile_fields") != plan["profile_categories"]
            or profile.get("user_profile_contract") != "shared_live_context_v1"):
        raise ValueError("Profile verification belongs to a different input contract or plan")
    for logical in ("plan", "test_metrics", "request_context", "raw_test"):
        entry = release["sources"].get(logical)
        if entry is None or profile.get("inputs_sha256", {}).get(str((root / entry["path"]).resolve())) != entry["sha256"]:
            raise ValueError("Profile verification does not bind the current test phase and request context")
    profile_vocab = {"features/vocab_dict/" + name + ".pkl" for name in ("gender", "platform", "age", "location")}
    for name, digest in bundle.get("assets", {}).items():
        if name in profile_vocab or name.startswith("datasets/user_feat/"):
            if profile["inputs_sha256"].get(str((root / name).resolve())) != digest:
                raise ValueError("Profile verification used different user features or vocabulary")
    rank_meta = sources.read(root / bundle["rank_dir"] / "manifest.json")
    if rank_meta["dien"].get("inference_contract") != "request_fp32_v1":
        raise ValueError("Serving bundle did not enable the evaluated numerical contract")
    return release


def build_snapshot(base: Path, bundle_paths: dict[str, Path], *, root: Path = ROOT, preview=False) -> dict:
    root, base = root.resolve(), base.resolve()
    sources = Sources(root)
    selection = sources.read(base / "exposure_model_selection.json")
    scenes = {}
    for scene, counts in EXPECTED.items():
        prior_dir = base / f"{scene}_rank_prior_gate"
        numerical = load_search_numerical_replay(root, sources.read(bundle_paths[scene]), sources) if scene == "search" else None
        gate_selection_path = sources.add(root / numerical["sources"]["plan"]["path"] if numerical else prior_dir / "selection.json")
        gate_selection = json.loads(gate_selection_path.read_text())
        gate = sources.read(root / numerical["sources"]["test_metrics"]["path"] if numerical else prior_dir / "test/metrics.json")
        if gate["selection_sha256"] != sha256(gate_selection_path) or not gate_selection["frozen_before_official_test"] or gate.get("parameter_choice_after_test") is not False:
            raise ValueError("Pipeline test recipe is not frozen before test scoring")
        if gate["all_test_requests"] != counts[0] or not isinstance(gate.get("accepted_for_release"), bool):
            raise ValueError("Missing full-test release decision")
        strategy = "learned" if gate["accepted_for_release"] else "recall_only"
        bundle = verify_bundle(bundle_paths[scene], scene, strategy, sources)
        release = load_rec_release(root, bundle, sources) if scene == "rec" else None
        raw, exposures = load_raw(root, scene, sources)
        candidate_path = sources.add(base / ("search_v3/test_selected.parquet" if scene == "search" else "rec_v3/refit_cf_pretest/selected.parquet"))
        original_candidate_path = candidate_path
        if release or numerical:
            current_evidence = release or numerical
            candidate_path = sources.add(root / current_evidence["sources"]["candidates"]["path"])
        candidates = pd.read_parquet(candidate_path)
        recall = recall_metrics(raw, candidates)
        if release and release.get("release_schema") == "adapted_content_v1":
            evaluated = sources.read(root / release["sources"]["test_metrics"]["path"])["metrics"]
            for name in ("HitRate@50", "Recall@100", "MRR@100", "known_positive_NDCG@10"):
                if not math.isclose(evaluated[name], recall[name], rel_tol=0, abs_tol=1e-12):
                    raise ValueError("Published adapted metrics differ from raw-test recomputation: " + name)
        if (len(raw), exposures, recall["positive_pairs"]) != counts:
            raise ValueError(f"{scene} original test counts changed")
        baseline_recall = recall_metrics(raw, pd.read_parquet(original_candidate_path)) if release else recall
        gate_candidate_path = candidate_path if numerical else original_candidate_path
        for path in [gate_candidate_path, *sorted((root / "datasets" / ("search_test" if scene == "search" else "recommendation_test")).glob("*.parquet"))]:
            if gate["inputs"].get(str(path.resolve())) != sha256(path):
                raise ValueError("Pipeline test uses a different raw dataset or recall snapshot")
        for name in ("HitRate@10", "HitRate@50", "Recall@100", "MRR@10", "MRR@100", "known_positive_NDCG@10"):
            if not math.isclose(gate["stages"]["raw"][name], baseline_recall[name], abs_tol=1e-12, rel_tol=0):
                raise ValueError(f"Pipeline raw order differs from actual recall output: {scene} {name}")
        recall_lineage = release["lineage"] if release else bind_recall_provenance(root, base, scene, original_candidate_path, bundle, sources)
        ranking = conditional_ranking(root, scene, selection, sources, counts)
        chosen_run = root / ranking["selection"]["run"]
        for name in ("gbdt.txt", "dien.pt"):
            original = sources.add(chosen_run / name)
            bundled = sources.add(root / bundle["rank_dir"] / name)
            relative = bundled.relative_to(root).as_posix()
            if sha256(original) != sha256(bundled) or bundle["assets"].get(relative) != sha256(bundled):
                raise ValueError("Published rank model differs from validation-selected exposure model")
        dssm_selection = base / ("search_v3/index/model_selection.json" if scene == "search" else "rec_v3/dssm/selected.json")
        if release:
            dssm_selection = root / release["sources"]["model_selection"]["path"]
        sources.add(dssm_selection)
        if bundle["packaging_provenance"]["selected_dssm_sha256"] != sha256(dssm_selection):
            raise ValueError("Bundle DSSM selection differs from evaluated recall experiment")
        pipeline_test = dict(gate["stages"]["final"]) if strategy == "learned" else dict(recall)
        pipeline_test["scope"] = ("Full original test known-click retention after selected pipeline; all test positives form IDCG; unexposed candidates have no invented click labels" if strategy == "learned" else recall["scope"])
        pipeline = {
            "test": pipeline_test, "serving_strategy": strategy,
            "input_cap": bundle["input_cap"], "coarse_cap": bundle["coarse_cap"],
            "test_release_passed": bool(release) or gate["accepted_for_release"],
            "learned_ranking_release_passed": gate["accepted_for_release"],
            "evaluated_recipe_stage": "final" if strategy == "learned" else "raw",
            "diagnostic_raw_pool_cap": gate_selection["input_cap"],
            "label_scope": "Known positives from all original test requests; not conditional exposure NDCG or online CTR",
        }
        proof_path = root / f".qilin/fix_20261031_{scene}_prior_verification.json"
        if release:
            pipeline["independent_verification_sha256"] = release["sources"]["fusion_independent_verification" if "fusion_selection" in release["sources"] else "independent_verification"]["sha256"]
            pipeline["release_evidence"] = bundle["evaluation"]["release_evidence"]
        if numerical:
            pipeline["independent_verification_sha256"] = numerical["sources"]["independent_verification"]["sha256"]
            pipeline["numerical_replay"] = bundle["evaluation"]["numerical_replay"]
            recall_lineage["inference_contract"] = "request_fp32_v1"
            pipeline["inference_contract"] = "request_fp32_v1"
            pipeline["accepted_execution_scope"] = numerical.get("execution_scope")
        if proof_path.is_file() and not release and not numerical:
            proof = sources.read(proof_path)
            if (proof.get("status") != "PASS" or proof["selection_sha256"] != sha256(gate_selection_path)
                    or proof["original_requests"] != counts[0] or proof["original_positive_pairs"] != counts[2]):
                raise ValueError("Independent pipeline proof does not match frozen request/truth scope")
            for value, digest in proof["inputs_sha256"].items():
                path = sources.add(root / value)
                if sha256(path) != digest:
                    raise ValueError("Independent pipeline proof input changed")
            pipeline["independent_verification_sha256"] = sha256(proof_path)
        ranking["serving_enabled"] = strategy == "learned"
        ranking["inference_contract"] = "frozen_logged_exposure_fp32"
        ranking["scope_note"] = "Frozen complete logged-exposure diagnostic; not rerun under the request-local serving numerical contract; not end-to-end serving quality."
        final_path = (sources.add(root / numerical["sources"]["final_candidates"]["path"]) if numerical
                      else candidate_path if strategy == "recall_only"
                      else sources.add(prior_dir / "test/final_candidates.parquet"))
        folder, key, nested = (("search", "search_idx", "search_result_details_with_idx") if scene == "search"
                               else ("recommendation", "request_idx", "rec_result_details_with_idx"))
        paths = sorted((root / "datasets" / f"{folder}_test").glob("*.parquet"))
        with duckdb.connect(config={"threads": 2}) as con:
            exposure_labels = con.execute(f"""SELECT r.{key} gid,d.note_idx,d.click
                FROM read_parquet(?) r,unnest(r.{nested}) u(d)""", [list(map(str, paths))]).fetchdf()
        coverage = label_coverage(raw, exposure_labels, pd.read_parquet(final_path))
        pipeline["candidate_path"] = final_path.relative_to(root).as_posix()
        scenes[scene] = {"label_coverage": coverage, "recall": {"test": recall, "lineage": recall_lineage}, "ranking": ranking,
                         "pipeline": pipeline, "model_bundle_id": bundle["bundle_id"]}
    def load_catalog_metrics(scene, path):
        raw, _ = load_raw(root, scene, sources)
        return recall_metrics(raw, pd.read_parquet(path))
    attach_comparisons(root, base, scenes, sources, load_catalog_metrics)
    sources.add(root / "scripts/evaluation_comparisons.py")
    sources.add(root / "scripts/evaluation_coverage.py")
    sources.add(Path(__file__))
    return {
        "tag": "hard",
        "protocol": {"name": "official_full_test_bundle_bound_v2", "full_test": True,
                     "held_out": False, "strict_acceptance": "blocked", "sampling": "none",
                     "label": "binary_click", "evaluated_at": "2026-10-31",
                     "publication_status": "preview_unpromoted" if preview else "active_bundle_bound",
                     "strict_acceptance_reasons": ["feature_point_in_time_lineage_unknown", "official_test_previously_observed"],
                     "warning": "全量测试集，测试集已被观察；特征时间隔离未完成核验。排序为完整日志曝光条件评估，端到端为已知正例保留评估。"},
        "scenes": scenes,
        "provenance": {"sources": sources.finish(), "producer": "scripts/sync_evaluation_metrics.py",
                       "selection": "Frozen original-train validation choices; no test-based model or weight selection"},
    }


def update_readme_text(text: str, snapshot: dict, *, english=False) -> str:
    """Change four existing metric lines plus the unsupported latency detail only."""
    scene = None
    counts = {"recall": 0, "ranking": 0, "latency": 0}
    lines = []
    for line in text.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        ending = line[len(body):]
        if body.startswith("- Search "):
            scene = "search"
        elif body.startswith("- Recommendation "):
            scene = "rec"
        if body.startswith(("  - Single-machine serial measurements", "  - 单机串行实测", "  - APIs return total and per-stage latency", "  - 接口返回总延迟与分阶段延迟")) or (body.startswith("  - ") and "~150ms" in body):
            body = "  - APIs return total and per-stage latency" if english else "  - 接口返回总延迟与分阶段延迟"
            counts["latency"] += 1
        elif scene and body.startswith("  - ") and (body.startswith("  - Recall (") or body.startswith("  - 召回（") or body.startswith("  - End-to-end (") or body.startswith("  - 端到端（")):
            current = snapshot["scenes"][scene]
            values = current.get("pipeline", {}).get("test", current["recall"]["test"])
            names = ("HitRate@10", "Recall@100", "MRR@100") if scene == "search" else ("HitRate@50", "Recall@100")
            names += ("known_positive_NDCG@10",) if "known_positive_NDCG@10" in values else ()
            prefix = "  - End-to-end (full test, known clicks): " if english else "  - 端到端（全量测试，已知点击）："
            body = prefix + (", " if english else "，").join(f"`{name} = {values[name]:.4f}`" for name in names)
            comparison = next((item for item in current.get("comparisons", {}).get("end_to_end", {}).get("metrics", [])
                               if item["name"] == "Recall@100"), None)
            if comparison and comparison.get("relative_delta_pct") is not None:
                if not math.isclose(comparison["current"], values["Recall@100"], abs_tol=1e-12, rel_tol=0):
                    raise ValueError("README comparison does not match current pipeline")
                suffix = "Recall@100 relative lift vs frozen baseline" if english else "Recall@100 较冻结基线相对提升"
                body += f"; {suffix} {comparison['relative_delta_pct']:+.2f}%"
            counts["recall"] += 1
        elif scene and body.startswith("  - ") and (body.startswith("  - Ranking (") or body.startswith("  - 排序（")):
            values = snapshot["scenes"][scene]["ranking"]["ranking"]["test"]
            prefix = "  - Ranking (logged-exposure diagnostic): " if english else "  - 排序（历史曝光内诊断）："
            body = prefix + (", " if english else "，").join(f"`{name} = {values[name]:.4f}`" for name in ("NDCG@10", "AUC", "GAUC"))
            counts["ranking"] += 1
        lines.append(body + ending)
    if counts != {"recall": 2, "ranking": 2, "latency": 1}:
        raise ValueError(f"Expected exactly four metric lines and one latency line: {counts}")
    return "".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--update-readme", action="store_true")
    parser.add_argument("--bundle-search", type=Path)
    parser.add_argument("--bundle-rec", type=Path)
    args = parser.parse_args(argv)
    preview = bool(args.bundle_search or args.bundle_rec)
    if preview and not (args.bundle_search and args.bundle_rec):
        parser.error("Preview requires both --bundle-search and --bundle-rec")
    output = (args.output or DEFAULT_OUTPUT).resolve()
    if preview and (args.output is None or output == DEFAULT_OUTPUT.resolve() or args.update_readme):
        parser.error("Unpromoted preview requires an explicit noncanonical --output and cannot update README")
    if not output.is_relative_to(ROOT.resolve()):
        parser.error("Output must remain inside the project")
    bundles = {scene: getattr(args, f"bundle_{scene}") if preview else ROOT / f"outputs/deploy/{scene}/hard/active_bundle.json" for scene in EXPECTED}
    for path in bundles.values():
        if not path.is_file():
            parser.error(f"Missing explicitly promoted active bundle (or explicit preview bundle): {path}")
    snapshot = build_snapshot(args.base, bundles, preview=preview)
    docs = {}
    if args.update_readme:
        for name, english in (("README.md", False), ("README_EN.md", True)):
            path = ROOT / name
            docs[path] = update_readme_text(path.read_bytes().decode("utf-8"), snapshot, english=english)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output)
    for path, text in docs.items():
        path.write_bytes(text.encode("utf-8"))
    print(json.dumps({"output": str(output), "status": snapshot["protocol"]["publication_status"],
                      "scenes": {scene: {"model_bundle_id": value["model_bundle_id"], "strategy": value["pipeline"]["serving_strategy"]} for scene, value in snapshot["scenes"].items()}}, indent=2))


if __name__ == "__main__":
    main()
