"""Audit label availability without assigning negatives to unexposed candidates."""
from __future__ import annotations

import duckdb
import pandas as pd


def label_coverage(requests: pd.DataFrame, exposures: pd.DataFrame, candidates: pd.DataFrame) -> dict:
    """Keep every request; exposure diagnostics retain repeated impressions.

    Candidate coverage deduplicates request-item labels using max(click), as the
    known-positive retrieval task does. Coverage never changes metric denominators.
    """
    if requests.gid.duplicated().any() or requests.empty:
        raise ValueError("Coverage requires unique original requests")
    known = set(requests.gid)
    if not set(exposures.gid).issubset(known) or not set(candidates.gid).issubset(known):
        raise ValueError("Foreign request in coverage inputs")
    if candidates.duplicated(["gid", "note_idx"]).any():
        raise ValueError("Duplicate candidate request-item pair")
    if not exposures.click.isin([0, 1]).all():
        raise ValueError("Coverage requires binary click labels")
    with duckdb.connect(config={"threads": 2}) as con:
        con.register("requests", requests[["gid"]])
        con.register("exposures", exposures[["gid", "note_idx", "click"]])
        con.register("candidates", candidates[["gid", "note_idx"]])
        raw = con.execute("""SELECT r.gid,count(e.note_idx) n,
            coalesce(sum(e.click),0) p FROM requests r LEFT JOIN exposures e USING(gid)
            GROUP BY r.gid""").fetchdf()
        con.execute("""CREATE TEMP TABLE labels AS SELECT gid,note_idx,max(click) click
            FROM exposures GROUP BY gid,note_idx""")
        matched = con.execute("""SELECT r.gid,count(c.note_idx) n,count(l.note_idx) labeled,
            coalesce(sum(l.click),0) p FROM requests r LEFT JOIN candidates c USING(gid)
            LEFT JOIN labels l USING(gid,note_idx) GROUP BY r.gid""").fetchdf()
    total = int(matched.n.sum())
    labeled = int(matched.labeled.sum())
    return {
        "scope": "All original test requests; current final candidates joined only to same-request historical exposures",
        "purpose": "Label-coverage diagnostic only; not an unbiased estimate of new-policy CTR or ranking quality",
        "all_requests": len(raw), "exposure_rows": len(exposures),
        "empty_exposure_requests": int((raw.n == 0).sum()),
        "zero_click_requests": int((raw.p == 0).sum()),
        "mean_exposures": float(raw.n.mean()), "median_exposures": float(raw.n.median()),
        "minimum_exposures": int(raw.n.min()), "maximum_exposures": int(raw.n.max()),
        "exposure_requests_with_both_labels": int(((raw.p > 0) & (raw.p < raw.n)).sum()),
        "final_candidate_pairs": total, "labeled_candidate_pairs": labeled,
        "unlabeled_candidate_pairs": total - labeled,
        "labeled_candidate_fraction": labeled / total if total else None,
        "empty_final_candidate_requests": int((matched.n == 0).sum()),
        "requests_without_exposure_overlap": int((matched.labeled == 0).sum()),
        "requests_without_known_click_in_candidates": int((matched.p == 0).sum()),
        "requests_with_both_labels_in_overlap": int(((matched.p > 0) & (matched.p < matched.labeled)).sum()),
        "unexposed_label": "unknown; never converted to an observed negative",
        "traffic_scope": "Published test requests only; zero-click traffic coverage must be checked separately",
    }
