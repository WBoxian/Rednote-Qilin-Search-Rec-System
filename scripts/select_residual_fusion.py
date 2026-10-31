"""Eight predeclared validation-only Rec fusion trials; test only one frozen winner."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from recall.weighted_context_fusion import fuse_weighted_request_routes

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/evaluation/northstar_fix_20261031"
RESIDUAL = BASE / "rec_residual_20261031_v2"
REC = BASE / "rec_v3"
GRID = tuple({"ann": a, "swing": s, "usercf": 1} for a in (1, 2, 4, 8) for s in (1, 2))
CONTROL = {"ann": 1, "swing": 2, "usercf": 1}
METRICS = ("HitRate@50", "Recall@100", "known_positive_NDCG@10", "MRR@100", "HitRate@500", "Recall@500")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def load_json(path):
    return json.loads(Path(path).read_text())


def verify_hashes(records):
    for name, expected in records.items():
        if digest(name) != expected:
            raise ValueError("Frozen source/input changed: "+name)


def plan(output):
    output.mkdir(parents=True, exist_ok=False)
    inputs = [RESIDUAL/"selection.json", RESIDUAL/"index/build.json", RESIDUAL/"index/validation_ann.parquet",
              REC/"validation/hist025_fold200/swing.parquet", REC/"validation/hist025_fold200/usercf.parquet",
              REC/"requests_validation.parquet", REC/"requests_fit.parquet",
              RESIDUAL/"index/validation_selected.parquet"]
    inputs += sorted((ROOT/"datasets/recommendation_train").glob("*.parquet"))
    inputs += [REC/"cf_graph_0.25"/name for name in ("graph.npz", "swing.npz", "items.npy", "users.npy", "meta.json")]
    code = [Path(__file__), ROOT/"src/recall/weighted_context_fusion.py", ROOT/"src/training/history.py",
            ROOT/"src/recall/context_fusion.py", ROOT/"tests/test_residual_fusion_selection.py"]
    declared = {"scene": "rec", "model": "frozen residual DSSM w.25, unchanged PQ256/probe32",
                "selection_split": "11893 original-train validation requests,40894 unique clicked pairs",
                "grid": list(GRID), "control": CONTROL,
                "fixed": {"rrf_k": 60, "route_topk": 1000, "topk": 500,
                          "context_policy": "filter_history", "recent_history": 20,
                          "tie_order": "descending float64 score, ascending note_idx",
                          "filter": "remove recent20 from each ORIGINAL1000 route, then compact rank; no refill"},
                "selection": "strict Recall100 gain, HR50/KP-NDCG10 point nonregression vs same residual121; max(R100,HR50,KP), then smallest ANN and Swing weights",
                "test": "Parent approval bound to selection SHA required; one winner and121control only, separate fit/allpast graphs; no post-test parameter choice; test previously observed, regression benchmark",
                "input_sha256": {str(p.resolve()): digest(p) for p in inputs},
                "code_sha256": {str(p.resolve()): digest(p) for p in code},
                "frozen_before_validation_metrics": True}
    write_json(output/"plan.json", declared)
    print(json.dumps({"plan": str(output/"plan.json"), "sha256": digest(output/"plan.json"), "trials": len(GRID)}), flush=True)
    return declared


def verify_plan(output):
    declared = load_json(output/"plan.json")
    if declared["grid"] != list(GRID) or declared["control"] != CONTROL:
        raise ValueError("The declared eight-trial grid changed")
    verify_hashes({**declared["input_sha256"], **declared["code_sha256"]})
    return declared


def original_truth(split, expected_ids=None):
    paths = sorted((ROOT/f"datasets/recommendation_{split}").glob("*.parquet"))
    columns = ["request_idx", "user_idx", "recent_clicked_note_idxs", "rec_result_details_with_idx"]
    raw = pd.concat([pd.read_parquet(path, columns=columns) for path in paths], ignore_index=True)
    if raw.request_idx.duplicated().any():
        raise ValueError("Duplicate original request identity")
    if expected_ids is not None:
        raw = raw[raw.request_idx.isin(expected_ids)]
        if set(raw.request_idx) != set(expected_ids):
            raise ValueError("Validation requests are not entirely original-train requests")
    return pd.DataFrame({"gid": raw.request_idx, "user_idx": raw.user_idx,
                         "history": raw.recent_clicked_note_idxs,
                         "positives": [sorted({int(item["note_idx"]) for item in details if float(item["click"]) > 0})
                                       for details in raw.rec_result_details_with_idx]}).sort_values("gid").reset_index(drop=True)


def validation_requests():
    supplied = pd.read_parquet(REC/"requests_validation.parquet").sort_values("gid").reset_index(drop=True)
    original = original_truth("train", set(map(int, supplied.gid)))
    if len(original) != 11893 or sum(map(len, original.positives)) != 40894:
        raise ValueError("Full original validation denominator changed")
    fit = set(pd.read_parquet(REC/"requests_fit.parquet", columns=["gid"]).gid)
    if fit.intersection(original.gid):
        raise ValueError("Original training fit/validation overlap")
    for left, right in zip(original.to_dict("records"), supplied.to_dict("records")):
        if (int(left["gid"]) != int(right["gid"]) or int(left["user_idx"]) != int(right["user_idx"])
                or list(left["history"]) != list(right["history"])
                or left["positives"] != sorted(set(map(int, right["positives"])))):
            raise ValueError("Cached context/truth differs from original raw training split")
    return original


def load_route(path, allowed):
    frame = pd.read_parquet(path, columns=["gid", "note_idx", "rank"])
    if not set(frame.gid).issubset(allowed):
        raise ValueError("Route contains foreign request IDs")
    routes = {}
    for gid, part in frame.groupby("gid", sort=False):
        part = part.sort_values("rank", kind="stable")
        ids = part.note_idx.to_numpy()
        ranks = part["rank"].to_numpy()
        if (not np.issubdtype(ids.dtype, np.integer) or (ids < 0).any()
                or len(part) > 1000 or len(np.unique(ids)) != len(ids)
                or not np.array_equal(ranks, np.arange(1, len(part)+1))):
            raise ValueError("Invalid original route rank/identity/budget")
        routes[int(gid)] = ids.copy()
    return routes


def metric_values(ids, positives):
    positive = set(map(int, positives))
    positions = np.asarray([rank for rank, item in enumerate(ids, 1) if int(item) in positive], dtype=np.int64)
    ideal = float(np.sum(1/np.log2(np.arange(2, min(len(positive), 10)+2))))
    return {"HitRate@50": float(np.any(positions <= 50)),
            "Recall@100": float(np.sum(positions <= 100))/max(len(positive), 1),
            "known_positive_NDCG@10": float(np.sum(1/np.log2(positions[positions <= 10]+1)))/ideal if ideal else 0.,
            "MRR@100": 1/float(positions[0]) if len(positions) and positions[0] <= 100 else 0.,
            "HitRate@500": float(bool(len(positions))),
            "Recall@500": float(len(positions))/max(len(positive), 1)}


def guard(candidate, control):
    return (candidate["Recall@100"] > control["Recall@100"]
            and candidate["HitRate@50"] >= control["HitRate@50"]
            and candidate["known_positive_NDCG@10"] >= control["known_positive_NDCG@10"])


def choose(trials, control):
    eligible = [trial for trial in trials if guard(trial["metrics"], control)]
    return max(eligible, key=lambda trial: (trial["metrics"]["Recall@100"], trial["metrics"]["HitRate@50"],
        trial["metrics"]["known_positive_NDCG@10"], -trial["weights"]["ann"], -trial["weights"]["swing"])) if eligible else None


def route_paths(split="validation", graph="fit"):
    if split == "validation":
        return {"ann": RESIDUAL/"index/validation_ann.parquet",
                **{name: REC/f"validation/hist025_fold200/{name}.parquet" for name in ("swing", "usercf")}}
    cf = REC/("test/cf/hist025_fold200" if graph == "fit" else "refit_cf_pretest")
    return {"ann": RESIDUAL/"test/ann.parquet", **{name: cf/f"{name}.parquet" for name in ("swing", "usercf")}}


def independent_hand_proof(routes, history, weights, actual):
    # Deliberately independent of the production history helper and fusion function.
    recent = []
    for value in history:
        if value is None or not np.isfinite(float(value)) or float(value) < 0:
            continue
        if int(value) != float(value):
            raise ValueError("Nonintegral history in independent proof")
        recent.append(int(value))
        if len(recent) == 20:
            break
    consumed = set(recent)
    expected = {}
    for name in ("ann", "swing", "usercf"):
        kept = [int(value) for value in routes[name] if int(value) not in consumed]
        for position, item in enumerate(kept):
            expected[item] = expected.get(item, 0.) + float(weights[name])/(61+position)
    ordered = sorted(expected, key=lambda item: (-expected[item], item))[:500]
    if ordered != actual.note_idx.tolist() or not np.array_equal(actual.score, [expected[item] for item in ordered]):
        raise ValueError("Independent hand-RRF proof failed")


def replay(requests, paths, grid, output, reference=None, save_winner=False):
    from contextlib import ExitStack
    import pyarrow as pa
    import pyarrow.parquet as pq
    allowed = set(map(int, requests.gid))
    routes = {name: load_route(path, allowed) for name, path in paths.items()}
    records = [[] for _ in grid]
    hand_proofs = 0
    reference_rows = {}
    if reference is not None:
        for gid, part in reference.groupby("gid", sort=False):
            ordered = part.sort_values("rank")
            reference_rows[int(gid)] = ordered.note_idx.to_numpy(), ordered.score.to_numpy()
    with ExitStack() as stack:
        schema = pa.schema([("gid", pa.int64()), ("note_idx", pa.int64()), ("score", pa.float64()), ("rank", pa.int64())])
        writer = stack.enter_context(pq.ParquetWriter(output/"winner_selected.parquet", schema)) if save_winner else None
        pending = []
        for index, request in enumerate(requests.to_dict("records")):
            gid = int(request["gid"])
            current = {name: values.get(gid, np.zeros(0, dtype=np.int64)) for name, values in routes.items()}
            for trial, weights in enumerate(grid):
                result = fuse_weighted_request_routes(current, request["history"], weights)
                records[trial].append({"gid": gid, "user_idx": int(request["user_idx"]),
                    **metric_values(result.note_idx, request["positives"])})
                if index < 64:
                    independent_hand_proof(current, request["history"], weights, result)
                    hand_proofs += 1
                if reference is not None and weights == CONTROL:
                    expected_ids, expected_scores = reference_rows.get(gid, ([], []))
                    if not np.array_equal(result.note_idx, expected_ids) or not np.array_equal(result.score, expected_scores):
                        raise ValueError("Control121 order/scores differ from frozen residual F1 output")
                if writer is not None and weights != CONTROL:
                    pending.append(result.assign(gid=gid))
            if writer is not None and len(pending) >= 128:
                writer.write_table(pa.Table.from_pandas(pd.concat(pending, ignore_index=True)[schema.names], schema=schema, preserve_index=False))
                pending.clear()
            if (index+1) % 2000 == 0:
                print(json.dumps({"requests": index+1, "trials": len(grid)}), flush=True)
        if pending:
            writer.write_table(pa.Table.from_pandas(pd.concat(pending, ignore_index=True)[schema.names], schema=schema, preserve_index=False))
    trials = []
    for index, weights in enumerate(grid):
        frame = pd.DataFrame(records[index])
        name = f"a{weights['ann']}_s{weights['swing']}_u{weights['usercf']}"
        path = output/(name+"_requests.parquet")
        frame.to_parquet(path, index=False)
        trials.append({"weights": weights, "metrics": {metric: float(frame[metric].mean()) for metric in METRICS},
                       "requests_path": str(path.resolve()), "requests_sha256": digest(path)})
    return trials, {"independent_hand_RRF_checks": hand_proofs, "control121_matches_frozen_all_requests": reference is not None}


def select(output):
    declared = verify_plan(output)
    if (output/"selection.json").exists():
        raise FileExistsError("Validation selection is immutable")
    requests = validation_requests()
    reference = pd.read_parquet(RESIDUAL/"index/validation_selected.parquet")
    trials, proof = replay(requests, route_paths(), GRID, output, reference)
    baseline = next(trial["metrics"] for trial in trials if trial["weights"] == CONTROL)
    for trial in trials:
        trial["guard_passed"] = guard(trial["metrics"], baseline)
    selected = choose(trials, baseline)
    result = {"accepted": selected is not None, "selected": selected, "control": baseline, "trials": trials,
              "requests": len(requests), "clicked_pairs": sum(map(len, requests.positives)), "proof": proof,
              "plan_sha256": digest(output/"plan.json"), "code_sha256": declared["code_sha256"],
              "frozen_before_official_test": True, "parameter_choice_after_test": False,
              "limitations": ["Test previously observed by earlier iterations; no blind holdout claim",
                              "Eight validation trials reuse the established original-train validation cohort",
                              "Unobserved catalog items do not become labeled negatives; no AUC claim"]}
    write_json(output/"selection.json", result)
    print(json.dumps(result), flush=True)
    return result


def test(output, approval):
    verify_plan(output)
    selected = load_json(output/"selection.json")
    if not selected["accepted"]:
        raise ValueError("No accepted validation winner; retain121")
    approved = load_json(approval)
    if approved.get("approved") is not True or approved.get("selection_sha256") != digest(output/"selection.json"):
        raise ValueError("Test requires explicit parent approval of the frozen selection SHA")
    destination = output/"test"
    destination.mkdir(exist_ok=False)
    requests = original_truth("test")
    if len(requests) != 11115 or sum(map(len, requests.positives)) != 44683:
        raise ValueError("Original test denominator changed")
    results = {}
    for graph in ("fit", "allpast"):
        paths = route_paths("test", graph)
        folder = destination/graph
        folder.mkdir()
        reference = pd.read_parquet(RESIDUAL/f"test/{graph}_selected.parquet")
        trials, proof = replay(requests, paths, [CONTROL, selected["selected"]["weights"]], folder, reference, save_winner=True)
        baseline, winner = trials
        results[graph] = {"control": baseline, "winner": winner, "guard_passed": guard(winner["metrics"], baseline["metrics"]),
                          "proof": proof, "route_sha256": {str(path.resolve()): digest(path) for path in paths.values()},
                          "winner_candidates": str((folder/"winner_selected.parquet").resolve()),
                          "winner_candidates_sha256": digest(folder/"winner_selected.parquet")}
    result = {"accepted_for_release": results["allpast"]["guard_passed"], "results": results,
              "selected_weights": selected["selected"]["weights"], "selection_sha256": digest(output/"selection.json"),
              "approval_sha256": digest(approval), "parameter_choice_after_test": False,
              "all_requests": len(requests), "clicked_pairs": sum(map(len, requests.positives)),
              "scope": "Previously observed official test regression benchmark; frozen winner/control, no parameter retuning"}
    write_json(destination/"metrics.json", result)
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("plan", "select", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--approval", type=Path)
    args = parser.parse_args()
    if args.phase == "test" and args.approval is None:
        parser.error("Test needs --approval with the reviewed selection SHA")
    {"plan": lambda: plan(args.output), "select": lambda: select(args.output),
     "test": lambda: test(args.output, args.approval)}[args.phase]()


if __name__ == "__main__":
    main()
