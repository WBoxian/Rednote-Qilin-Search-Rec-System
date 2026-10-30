"""Independently verify full-test artifacts, formulas and optional repeat-run equality."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, name):
    if actual is None or expected is None:
        check(actual is expected, name)
    else:
        check(bool(np.isclose(actual, expected, rtol=1e-10, atol=1e-12)), f"{name}: {actual} != {expected}")


def verify(run, verify_inputs):
    metrics = json.loads((run / "metrics.json").read_text())
    manifest = json.loads((run / "manifest.json").read_text())
    status = json.loads((run / "status.json").read_text())
    check(status["execution"] == "COMPLETE", "incomplete run")
    check(bool(metrics["scenes"]), "no evaluated scenes")
    check(status["strict_acceptance"] == metrics["protocol"]["strict_acceptance"], "acceptance status differs")
    check("metrics.json" in manifest["outputs"], "metrics output fingerprint missing")
    for name, entry in manifest["outputs"].items():
        check(digest(run / name) == entry["sha256"], f"output changed: {name}")
    for name, entry in manifest["inputs"].items():
        if verify_inputs or name.startswith("datasets/") or name.startswith("outputs/data/recall_"):
            path = Path(name) if Path(name).is_absolute() else ROOT / name
            check(digest(path) == entry["sha256"], f"input changed: {name}")
    for scene, result in metrics["scenes"].items():
        folder, key, nested = ("search", "search_idx", "search_result_details_with_idx") if scene == "search" else ("recommendation", "request_idx", "rec_result_details_with_idx")
        with duckdb.connect(config={"threads": "2", "memory_limit": "3GB"}) as con:
            con.execute(f"""CREATE TABLE labels AS
                SELECT r.{key} gid,d.note_idx,max(d.click)::INT positive
                FROM read_parquet(?) r,unnest(r.{nested}) u(d) GROUP BY r.{key},d.note_idx""",
                [str(ROOT / f"datasets/{folder}_test/*.parquet")])
            con.execute(f"""CREATE TABLE exposure_labels AS
                SELECT r.{key} gid,d.note_idx,d.click::INT positive,exposure_index
                FROM read_parquet(?) r,unnest(r.{nested}) WITH ORDINALITY u(d,exposure_index)""",
                [str(ROOT / f"datasets/{folder}_test/*.parquet")])
            con.execute(f"""CREATE TABLE raw_requests AS
                SELECT {key} gid FROM read_parquet(?)""", [str(ROOT / f"datasets/{folder}_test/*.parquet")])
            candidates = ROOT / f"outputs/data/recall_{scene}_test_hard_multiroute_top1000.parquet"
            # Independent SQL aggregation over raw labels; no evaluator metric helper imports.
            for k in (100, 500, 1000):
                hr, recall = con.execute(f"""
                    SELECT avg(hit),avg(recall) FROM (
                        SELECT l.gid,max(CASE WHEN c.rank<=? THEN 1.0 ELSE 0.0 END) hit,
                               sum(CASE WHEN c.rank<=? THEN 1.0 ELSE 0.0 END)/count(*) recall
                        FROM labels l LEFT JOIN read_parquet(?) c ON c.{key}=l.gid AND c.note_idx=l.note_idx
                        WHERE l.positive=1 GROUP BY l.gid)
                """, [k, k, str(candidates)]).fetchone()
                close(hr, result["recall"][f"HitRate@{k}"], f"{scene} HR{k}")
                close(recall, result["recall"][f"Recall@{k}"], f"{scene} Recall{k}")
            mrr, median = con.execute(f"""
                SELECT avg(CASE WHEN first_hit<=100 THEN 1.0/first_hit ELSE 0 END),median(first_hit)
                FROM (SELECT l.gid,min(c.rank) first_hit
                      FROM labels l LEFT JOIN read_parquet(?) c ON c.{key}=l.gid AND c.note_idx=l.note_idx
                      WHERE l.positive=1 GROUP BY l.gid)
            """, [str(candidates)]).fetchone()
            close(mrr, result["recall"]["MRR@100"], f"{scene} MRR")
            close(median, result["recall"]["MedianFirstHitRank_among_hits@1000"], f"{scene} conditional median")
            count = con.execute("SELECT count(*) FROM raw_requests").fetchone()[0]
            requests = pd.read_parquet(run / scene / "recall_requests.parquet")
            check(len(requests) == count and requests.gid.nunique() == count, f"{scene}: lost requests")
            con.register("request_results", requests)
            check(con.execute("SELECT count(*) FROM raw_requests FULL JOIN request_results USING(gid) WHERE raw_requests.gid IS NULL OR request_results.gid IS NULL").fetchone()[0] == 0, f"{scene}: request IDs differ")
            con.unregister("request_results")
            check(result["recall"]["all_requests"] == count, f"{scene}: reported request count differs")
            for stage in ("gbdt", "dien"):
                pred = pd.read_parquet(run / scene / f"{stage}_predictions.parquet")
                check(not pred.duplicated(["gid", "exposure_index"]).any(), "duplicate predictions")
                con.register("pred", pred)
                check(con.execute("SELECT count(*) FROM exposure_labels l FULL JOIN pred USING(gid,exposure_index) WHERE l.gid IS NULL OR pred.gid IS NULL OR l.positive<>pred.positive OR l.note_idx<>pred.note_idx").fetchone()[0] == 0,
                      f"{scene}/{stage}: predictions do not exactly match all test labels")
                check(result["ranking"][stage]["rows"] == len(pred), f"{scene}/{stage}: reported exposure count differs")
                scores = pred.score.to_numpy()
                check(np.isfinite(scores).all(), "nonfinite prediction")
                y = pred.positive.to_numpy()
                auc = float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else None
                close(auc, result["ranking"][stage]["AUC"], f"{scene}/{stage}/AUC")
                ndcgs = []
                for _, group in pred.groupby("gid"):
                    if group.positive.sum() == 0:
                        continue
                    ndcgs.append(float(group.positive.iloc[0]) if len(group) == 1 else
                                 float(ndcg_score([group.positive.to_numpy()], [group.score.to_numpy()], k=10)))
                close(float(np.mean(ndcgs)) if ndcgs else None, result["ranking"][stage]["NDCG@10"], f"{scene}/{stage}/NDCG")
                aucs, weights = [], []
                for _, group in pred.groupby("user_idx"):
                    if group.positive.nunique() == 2:
                        aucs.append(roc_auc_score(group.positive, group.score))
                        weights.append(len(group))
                gauc = float(np.average(aucs, weights=weights)) if weights else None
                close(gauc, result["ranking"][stage]["GAUC"], f"{scene}/{stage}/GAUC")
                con.unregister("pred")
    return metrics, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--verify-inputs", action="store_true")
    parser.add_argument("--require-accepted", action="store_true")
    args = parser.parse_args()
    metrics, manifest = verify(args.run, args.verify_inputs)
    summary = {"integrity_and_independent_metrics": "PASS", "repeat_equality": "NOT_RUN",
               "strict_acceptance": metrics["protocol"]["strict_acceptance"]}
    if args.compare:
        other, other_manifest = verify(args.compare, False)
        check(metrics == other, "repeat metrics differ")
        check(manifest["inputs"] == other_manifest["inputs"], "repeat inputs differ")
        check(manifest["outputs"] == other_manifest["outputs"], "repeat prediction/output bytes differ")
        summary["repeat_equality"] = "PASS: all input hashes, output hashes and metrics identical"
    (args.run / ("verification.json" if args.compare else "verification_single.json")).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.require_accepted and summary["strict_acceptance"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
