"""Validation-only ranking selection with explicit discrimination guardrails."""
from __future__ import annotations
import math

RANK_GUARDS = ('NDCG@10', 'AUC', 'GAUC')


def choose_rank_blend(choices):
    baseline = [choice for choice in choices if choice['weight'] == 0]
    if len(baseline) != 1:
        raise ValueError('Exactly one coarse-only baseline is required')
    for choice in choices:
        if any(not isinstance(choice['validation'].get(name), (int, float)) or
               not math.isfinite(choice['validation'][name]) for name in RANK_GUARDS):
            raise ValueError('Every selection candidate requires finite NDCG/AUC/GAUC')
    floor = baseline[0]['validation']
    eligible = [choice for choice in choices if all(
        choice['validation'][name] >= floor[name] - 1e-12 for name in RANK_GUARDS)]
    return max(eligible, key=lambda choice: (
        choice['validation']['NDCG@10'], choice['validation']['GAUC'],
        choice['validation']['AUC'], -choice['weight']))

def choose_rank_model(choices):
    if not choices:
        raise ValueError('At least one ranking model is required')
    floor = choices[0]['validation']
    for choice in choices:
        if any(not math.isfinite(choice['validation'][name]) for name in RANK_GUARDS):
            raise ValueError('Every ranking model requires finite selection metrics')
    eligible = [choice for choice in choices if all(
        choice['validation'][name] >= floor[name] - 1e-12 for name in RANK_GUARDS)]
    return max(eligible, key=lambda choice: tuple(choice['validation'][name]
               for name in ('NDCG@10', 'GAUC', 'AUC')))