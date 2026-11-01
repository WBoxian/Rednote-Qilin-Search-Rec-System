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

def adapted_release_comparisons(current, evaluated, rows, historical_baseline):
    """Separate index precision, trained representation, and learned merge effects."""
    labels = {
        "original_baseline": "Previous residual DSSM + PQ + 2:1:1 fusion",
        "original_flat211": "Previous residual DSSM + exact FlatIP + 2:1:1 fusion",
        "dual128_rrf": "Trained dual content adapter + exact FlatIP + 2:1:1 fusion",
        "dual128": "Trained dual content adapter + exact FlatIP + learned fusion",
    }
    metrics = {key: evaluated["ablations"][key]["metrics"] for key in labels}
    for key, label in labels.items():
        rows.append(ablation_row("adapter_" + key, label, metrics[key], historical_baseline,
            status="accepted" if key == "dual128" else "diagnostic",
            changed="Frozen full-test regression; same cohort and past-only CF; index precision and learned merge are explicit"))
    for key, control, treatment in (
        ("end_to_end", "original_baseline", "dual128"),
        ("index_precision", "original_baseline", "original_flat211"),
        ("content_adapter", "original_flat211", "dual128_rrf"),
        ("learned_fusion", "dual128_rrf", "dual128"),
    ):
        current["comparisons"][key] = comparison(metrics[control], metrics[treatment],
            baseline_name=labels[control], current_name=labels[treatment], scene="rec",
            paired=evaluated.get("paired", {}).get(treatment + "-" + control))


def multi_interest_refit_comparisons(current, proof):
    """Use the actual immediate predecessor and fixed component contrasts."""
    mapping = {"HitRate@50": "HR50", "Recall@100": "R100", "known_positive_NDCG@10": "KP_NDCG10",
               "MRR@100": "MRR100", "Recall@500": "R500", "HitRate@500": "HR500"}
    keys = ("current_global_LGB", "oldfit_k3_lgb", "oldfit_k3_safehead", "finalrefit_k3_safehead")
    if (proof.get("status") != "PASS" or proof.get("parameter_choice_after_test") is not False
            or (proof.get("all_requests"), proof.get("known_positive_pairs")) != (11115, 44683)):
        raise ValueError("Multi-interest comparisons require independently audited full-cohort results")
    metrics = {}
    for key in keys + (("old_PQ211",) if "old_PQ211" in proof["outputs"] else ()):
        row = proof["outputs"][key]
        if (row.get("all_requests"), row.get("known_positive_pairs")) != (11115, 44683):
            raise ValueError("Comparison arm has a different request/positive denominator")
        metrics[key] = {name: row["metrics"][short] for name, short in mapping.items()}
    selected = "finalrefit_k3_safehead"
    for name, value in metrics[selected].items():
        if name not in current["pipeline"]["test"] or not math.isclose(value, current["pipeline"]["test"][name], rel_tol=0, abs_tol=1e-12):
            raise ValueError("Comparison does not match the current released pipeline: " + name)
    labels = {"current_global_LGB": "Previous global single-interest DSSM + FlatIP + learned fusion",
        "oldfit_k3_lgb": "Same fit heads + K3 + learned fusion",
        "oldfit_k3_safehead": "Same fit K3 + fixed SafeHead fusion postprocessing",
        selected: "All-past content-head refit + K3 + fixed SafeHead"}
    def paired(treatment, control):
        evidence = proof.get("paired_usercluster95_exploratory_CI", {}).get(treatment + "__minus__" + control, {})
        result = {}
        for name, short in mapping.items():
            if short in evidence:
                value = evidence[short]
                if not math.isclose(value["delta"], metrics[treatment][name] - metrics[control][name], rel_tol=0, abs_tol=1e-12):
                    raise ValueError("Audited paired interval belongs to a different contrast")
                result[name] = value
        return result
    comparisons = {}
    for key, control, treatment in (("end_to_end", "current_global_LGB", selected),
        ("interest_queries", "current_global_LGB", "oldfit_k3_lgb"),
        ("head_guard", "oldfit_k3_lgb", "oldfit_k3_safehead"),
        ("content_refit", "oldfit_k3_safehead", selected)):
        comparisons[key] = comparison(metrics[control], metrics[treatment], baseline_name=labels[control],
            current_name=labels[treatment], scene="rec", paired=paired(treatment, control))
    if "old_PQ211" in metrics:
        comparisons["original_PQ_reference"] = comparison(metrics["old_PQ211"], metrics[selected],
            baseline_name="Original residual DSSM + PQ + 2:1:1 fusion", current_name=labels[selected],
            scene="rec", paired=paired(selected, "old_PQ211"))
    current["comparisons"] = comparisons
    baseline = metrics["current_global_LGB"]
    return [ablation_row(key, labels[key], metrics[key], baseline,
        status="baseline" if key == "current_global_LGB" else "accepted" if key == selected else "diagnostic",
        changed=("Immediate previous released global-query pipeline" if key == "current_global_LGB" else
                 "Fixed content weights; interest-query extraction only" if key == "oldfit_k3_lgb" else
                 "Fusion postprocessing only; Top100/Top500 sets preserved; no DIEN gain" if key == "oldfit_k3_safehead" else
                 "Content training data and optimizer steps both change (327 to 408); not an isolated data effect"),
        paired=paired(key, "current_global_LGB") if key == selected else None) for key in keys]


def attach_comparisons(root, base, scenes, sources, load_catalog_metrics):
    """Consume independently checked artifacts; never choose recipes from test values."""
    audit = sources.read(root / '.qilin/final_metrics_acceptance_20261031.json')
    for scene, current in scenes.items():
        evidence = current['pipeline'].get('release_evidence') if scene == 'rec' else None
        release = sources.read(root / evidence['path']) if evidence else None
        if release and release.get('release_schema') == 'multi_interest_refit_v1':
            proof = sources.read(root / release['sources']['independent_verification']['path'])
            rows = multi_interest_refit_comparisons(current, proof)
            current['comparisons']['logged_exposure'] = exposure_comparison(
                audit['scopes']['rec_test_logged_exposure'], current['ranking']['ranking']['test'])
            current['ablations'] = dict(scope='full_catalog_known_click',
                baseline_name=current['comparisons']['end_to_end']['baseline_name'], rows=rows,
                warning='Historical full-test regression. SafeHead is fusion postprocessing; exposure rankers remain diagnostic. Refit changes data and update count together; its same-family paired intervals can cross zero.')
            continue
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
                if release.get("release_schema") == "adapted_content_v1":
                    evaluated = sources.read(root / release["sources"]["test_metrics"]["path"])
                    adapted_release_comparisons(current, evaluated, rows, baseline)
                else:
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