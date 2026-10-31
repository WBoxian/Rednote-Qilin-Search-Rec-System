"""Describe same-scope benchmark gains without conflating exposure and catalog tasks."""
from __future__ import annotations
import math

CATALOG_METRICS = ('HitRate@10', 'HitRate@50', 'Recall@100', 'known_positive_NDCG@10', 'MRR@100')


def metric_delta(baseline, current, ci=None):
    baseline, current = float(baseline), float(current)
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in (baseline, current)):
        raise ValueError('Metric values must be finite fractions in [0,1]')
    delta = current - baseline
    result = dict(baseline=baseline, current=current, absolute_delta=delta,
                  absolute_delta_pp=100 * delta,
                  relative_delta_pct=100 * delta / baseline if baseline else None)
    if ci is not None:
        if len(ci) != 2 or not all(math.isfinite(float(x)) for x in ci) or ci[0] > ci[1]:
            raise ValueError('Invalid delta confidence interval')
        result['ci95_delta'] = list(map(float, ci))
    return result


def comparison(baseline, current, *, baseline_name, current_name, scene, paired=None):
    names = ('HitRate@10', 'Recall@100', 'known_positive_NDCG@10', 'MRR@100') if scene == 'search' else ('HitRate@50', 'Recall@100', 'known_positive_NDCG@10', 'MRR@100')
    rows = []
    for name in names:
        if name not in baseline or name not in current:
            raise ValueError(f'Missing comparison metric: {name}')
        evidence = (paired or {}).get(name, {})
        ci = evidence.get('CI95', evidence.get('user_cluster_CI95', evidence.get('user_cluster_CI95_2000')))
        rows.append(dict(name=name, **metric_delta(baseline[name], current[name], ci)))
    return dict(scope='full_catalog_known_click', baseline_name=baseline_name,
                current_name=current_name, metrics=rows)


def exposure_comparison(audit, ranking):
    references = {'NDCG@10': audit['random_expected_NDCG@10'], 'AUC': .5, 'GAUC': .5}
    rows = [dict(name=name, **metric_delta(value, ranking[name])) for name, value in references.items()]
    return dict(scope='logged_exposure_conditional', baseline_name='Random expected ordering',
                current_name='Validation-selected exposure ranker', metrics=rows,
                diagnostics=dict(median_candidates=audit['candidate_count_quantiles']['0.5'],
                                 requests_at_most_10_fraction=audit['requests_with_at_most_10_candidates'] / audit['requests'],
                                 original_position_NDCG_at10=audit['original_position_NDCG@10'],
                                 warning='Position order reflects historical exposure bias; this comparison does not establish online causal gain.'))


def ablation_row(identifier, label, metrics, baseline, *, status, changed, paired=None):
    if status not in {'baseline', 'accepted', 'diagnostic_rejected', 'diagnostic'}:
        raise ValueError('Unknown experiment status')
    values = {key: float(metrics[key]) for key in CATALOG_METRICS if key in metrics}
    differences = {}
    for key, value in values.items():
        evidence = (paired or {}).get(key, {})
        ci = evidence.get('CI95', evidence.get('user_cluster_CI95', evidence.get('user_cluster_CI95_2000')))
        differences[key] = metric_delta(baseline[key], value, ci)
    return dict(id=identifier, label=label, status=status, changed=changed,
                metrics=values, comparison=differences)

def attach_comparisons(root, base, scenes, sources, load_catalog_metrics):
    """Consume independently checked artifacts; never choose recipes from test values."""
    audit = sources.read(root / '.qilin/final_metrics_acceptance_20261031.json')
    for scene, current in scenes.items():
        original = base / ('search_v3/test_selected.parquet' if scene == 'search' else 'rec_v3/refit_cf_pretest/selected.parquet')
        baseline = load_catalog_metrics(scene, sources.add(original))
        gate = sources.read(base / f'{scene}_rank_prior_gate/test/metrics.json')
        numerical = current['pipeline'].get('numerical_replay')
        if numerical:
            evidence = sources.read(root / numerical['path'])
            gate = sources.read(root / evidence['sources']['test_metrics']['path'])
        paired = gate.get('paired_stages', {}).get('final-raw') if scene == 'search' and not numerical else None
        current['comparisons'] = {
            'end_to_end': comparison(baseline, current['pipeline']['test'],
                                    baseline_name='Frozen DSSM v3 + original fusion',
                                    current_name='Current released ordering', scene=scene, paired=paired),
            'logged_exposure': exposure_comparison(audit['scopes'][f'{scene}_test_logged_exposure'],
                                                   current['ranking']['ranking']['test']),
        }
        rows = [ablation_row('frozen_v3', 'Frozen v3 recall order', baseline, baseline,
                            status='baseline', changed='Reference before ranking-prior and subsequent recall iterations')]
        if scene == 'search':
            if numerical:
                rows.append(ablation_row('canonical_raw','Canonical request recall order',gate['stages']['raw'],baseline,status='diagnostic',changed='Same frozen model and fusion; current-request numerical contract replay'))
            for name, label, status in [('coarse','Coarse only','diagnostic_rejected'), ('fine','Coarse + fine','diagnostic_rejected'), ('final','Fine with recall prior','accepted')]:
                rows.append(ablation_row(name, label, gate['stages'][name], baseline, status=status,
                                         changed=f'Fixed input pool, ordering={name}; no model retraining in this contrast',
                                         paired=gate.get('paired_stages', {}).get('final-raw') if name == 'final' and not numerical else None))
        else:
            rows.append(ablation_row('rank_prior_rejected','Ranking-prior experiment',gate['stages']['final'],baseline,
                                     status='diagnostic_rejected',changed='Frozen validation-selected reranker; rejected by full-test release guard'))
            diagnostic = sources.read(base / 'rec_iteration_20261031/diagnostic_test/metrics.json')
            for name, entry in diagnostic['models'].items():
                rows.append(ablation_row(name,name,entry['metrics']['allpast'],baseline,
                                         status='diagnostic_rejected',changed='Three extra train-only epochs; A/B/C duration-matched controls; none passed validation'))
            fusion = sources.read(base / 'rec_iteration_20261031/fusion_gate/test_replay_v2/metrics.json')
            for name, entry in fusion['results']['allpast']['policies'].items():
                rows.append(ablation_row(name,name,entry['metrics'],baseline,
                                         status='accepted' if entry['deployment_eligible'] else 'diagnostic_rejected',
                                         changed='Same v3 weights and all-past graph; history filtering and fixed context fusion controls',
                                         paired=entry['paired_vs_v3']))
            evidence = current['pipeline'].get('release_evidence')
            if evidence:
                release = sources.read(root / evidence['path'])
                residual = sources.read(root / release['sources']['test_metrics']['path'])
                result = residual['results']['allpast']
                rows.append(ablation_row('residual_f1','DSSM residual + history filter',result['metrics'],baseline,
                                         status='accepted',changed='Same learned weights and CF graph; train-only PCA residual representation; fixed 256-dimensional ANN'))
                current['comparisons']['residual_vs_history_filter'] = comparison(
                    result['baseline_v3_F1'], result['metrics'], baseline_name='v3 + accepted history filter',
                    current_name='Residual DSSM + same history filter',scene='rec',paired=result['paired_vs_v3_F1'])
                if 'fusion_selection' in release['sources']:
                    fusion_test = sources.read(root / release['sources']['fusion_test_metrics']['path'])
                    fusion_proof = sources.read(root / release['sources']['fusion_independent_verification']['path'])
                    final_fusion = fusion_test['results']['allpast']
                    rows.append(ablation_row('residual_fusion_selected','Residual DSSM + validation-selected fusion',final_fusion['winner']['metrics'],baseline,
                                             status='accepted',changed='Same residual model and CF graph; validation selected ANN2/Swing1/UserCF1 from eight fixed trials'))
                    current['comparisons']['fusion_vs_residual_control'] = comparison(
                        final_fusion['control']['metrics'], final_fusion['winner']['metrics'],
                        baseline_name='Residual model + 1:2:1',current_name='Same residual model + 2:1:1',scene='rec',
                        paired=fusion_proof['graphs']['allpast']['paired'])
        current['ablations'] = dict(scope='full_catalog_known_click',baseline_name='Frozen DSSM v3 + original fusion',rows=rows,
                                   warning='All rows retain the complete official test cohort. Rejected diagnostics are not served. A/B/C share extra training duration; graph refit data-volume effects are reported separately. Historical test was previously observed. Canonical Search stages share one regenerated input pool; the frozen baseline retains its recorded numerical contract.')