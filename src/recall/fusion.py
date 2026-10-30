"""Shared deterministic rank fusion for DSSM, Swing and UserCF candidates."""
from __future__ import annotations

import math


def weighted_rrf(ann_scores, swing_scores, usercf_scores, *, topk, route_weights=None, rrf_k=60):
    """Score the complete union before truncation; input dictionary order is irrelevant."""
    if topk <= 0:
        return []
    routes = {"ann": ann_scores, "swing": swing_scores, "usercf": usercf_scores}
    weights = {route: 1.0 for route in routes}
    if route_weights:
        weights.update({route: float(value) for route, value in route_weights.items() if route in routes})
    if any(not math.isfinite(weight) or weight < 0 for weight in weights.values()):
        raise ValueError("route weights must be finite and nonnegative")
    if rrf_k < 1:
        raise ValueError("rrf_k must be positive")
    values = {}
    for route, scores in routes.items():
        if weights[route] == 0:
            continue
        ordered = sorted(((int(item), float(score)) for item, score in scores.items()
                          if int(item) >= 0 and math.isfinite(float(score))),
                         key=lambda item: (-item[1], item[0]))
        for rank, (item, _) in enumerate(ordered, 1):
            contribution = weights[route] / (rrf_k + rank)
            record = values.setdefault(item, [0., route, -1.])
            record[0] += contribution
            if contribution > record[2]:
                record[1], record[2] = route, contribution
    result = []
    for item, (score, source, _) in sorted(values.items(), key=lambda entry: (-entry[1][0], entry[0]))[:topk]:
        result.append((item, score, float(ann_scores.get(item, 0)), float(swing_scores.get(item, 0)),
                       float(usercf_scores.get(item, 0)), int(item in ann_scores),
                       int(item in swing_scores), int(item in usercf_scores), source))
    return result
