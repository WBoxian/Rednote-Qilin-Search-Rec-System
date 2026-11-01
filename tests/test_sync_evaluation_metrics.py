"""Full-test denominators, frozen selection, and publication safety checks."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

SPEC = importlib.util.spec_from_file_location("sync_evaluation_metrics", Path(__file__).parents[1] / "scripts/sync_evaluation_metrics.py")
sync = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sync)


class TestSyncEvaluationMetrics(unittest.TestCase):
    def test_full_request_denominator_dedup_note_zero_and_actual_budget(self):
        requests = pd.DataFrame({"gid": [1, 2, 3, 4], "positives": [[0, 0, 4], [8], [9], []]})
        candidates = pd.DataFrame({"gid": [1, 1, 2], "note_idx": [0, 3, 7], "rank": [1, 2, 1]})
        result = sync.recall_metrics(requests, candidates)
        self.assertEqual(result["all_requests"], 4)
        self.assertEqual(result["positive_requests"], 3)
        self.assertEqual(result["effective_groups"], 3)
        self.assertEqual(result["positive_pairs"], 4)
        self.assertAlmostEqual(result["HitRate@10"], 1 / 3)
        self.assertAlmostEqual(result["Recall@100"], 1 / 6)
        self.assertEqual(result["MicroRecall@100"], 0.25)
        self.assertAlmostEqual(result["MRR@100"], 1 / 3)
        self.assertEqual(result["MedianFirstHitRank"], 1)
        self.assertEqual(result["missed_requests_at_500"], 2)
        self.assertEqual(result["missed_requests@500"], 2)
        self.assertEqual(result["empty_candidate_requests"], 2)
        self.assertFalse(any("1000" in name for name in result))

    def test_recall_rejects_invalid_candidate_contract(self):
        raw = pd.DataFrame({"gid": [1], "positives": [[2]]})
        cases = [
            [(1, 2, 1), (1, 2, 2)],  # duplicated item
            [(1, 2, 1), (1, 3, 1)],  # duplicated rank
            [(1, 2, 2)],             # missing rank1
            [(1, 2, 501)],           # not actual top500
            [(2, 2, 1)],             # foreign request
            [(1, -1, 1)],
        ]
        for values in cases:
            with self.subTest(values=values), self.assertRaises(ValueError):
                sync.recall_metrics(raw, pd.DataFrame(values, columns=["gid", "note_idx", "rank"]))

    def test_no_hit_median_is_null_and_misses_are_not_removed(self):
        raw = pd.DataFrame({"gid": [1, 2], "positives": [[2], [3]]})
        frame = pd.DataFrame(columns=["gid", "note_idx", "rank"])
        result = sync.recall_metrics(raw, frame)
        self.assertIsNone(result["MedianFirstHitRank"])
        self.assertEqual(result["HitRate@500"], 0)
        self.assertEqual(result["MRR@100"], 0)
        self.assertEqual(result["missed_requests_at_500"], 2)

    def test_sources_detect_post_read_mutation(self):
        with tempfile.TemporaryDirectory(prefix="qilin-metrics-test-") as temporary:
            root = Path(temporary)
            path = root / "fixture.json"
            path.write_text('{"a": 1}')
            sources = sync.Sources(root)
            sources.read(path)
            path.write_text('{"a": 2}')
            with self.assertRaisesRegex(ValueError, "changed"):
                sources.finish()

    def test_bundle_identity_and_release_strategy_are_required(self):
        with tempfile.TemporaryDirectory(prefix="qilin-metrics-test-") as temporary:
            root = Path(temporary)
            path = root / "bundle.json"
            bundle = {"scene": "rec", "tag": "hard", "ranking_strategy": "recall_only"}
            bundle["bundle_id"] = sync.bundle_identity(bundle)
            path.write_text(json.dumps(bundle))
            self.assertEqual(sync.verify_bundle(path, "rec", "recall_only", sync.Sources(root))["bundle_id"], bundle["bundle_id"])
            with self.assertRaisesRegex(ValueError, "strategy"):
                sync.verify_bundle(path, "rec", "learned", sync.Sources(root))
            bundle["ranking_strategy"] = "learned"
            path.write_text(json.dumps(bundle))
            with self.assertRaisesRegex(ValueError, "identity"):
                sync.verify_bundle(path, "rec", "learned", sync.Sources(root))

    def test_unpromoted_preview_cannot_publish(self):
        for extra in ([], ["--update-readme"], ["--output", str(sync.DEFAULT_OUTPUT)]):
            with self.subTest(extra=extra), patch("sys.stderr"), self.assertRaises(SystemExit):
                sync.main(["--bundle-search", "search.json", "--bundle-rec", "rec.json", *extra])

    def test_readme_leads_with_pipeline_and_keeps_exposure_diagnostic_separate(self):
        text = ("- 在线服务\n  - 接口返回总延迟与分阶段延迟\n- Search 离线指标\n"
                "  - 召回（测试集）：old\n  - 排序（DIEN，测试曝光）：old\n- Recommendation 离线指标\n"
                "  - 召回（测试集）：old\n  - 排序（DIEN，测试曝光）：old\n")
        raw = {"HitRate@10": .1, "HitRate@50": .2, "Recall@100": .1, "MRR@100": .1}
        final = {**raw, "Recall@100": .3, "known_positive_NDCG@10": .02}
        rank = {"NDCG@10": .6, "AUC": .61, "GAUC": .62}
        scene = {"recall": {"test": raw}, "pipeline": {"test": final}, "ranking": {"ranking": {"test": rank}}}
        scene["comparisons"] = {"end_to_end": {"metrics": [{"name": "Recall@100", "current": .3, "relative_delta_pct": 200.0}]}}
        snapshot = {"scenes": {"search": scene, "rec": scene}}
        got = sync.update_readme_text(text, snapshot)
        self.assertIn("相对提升 +200.00%", got)
        self.assertIn("端到端（全量测试，已知点击）", got)
        self.assertIn("Recall@100 = 0.3000", got)
        self.assertIn("known_positive_NDCG@10 = 0.0200", got)
        self.assertIn("排序（历史曝光内诊断）", got)
        self.assertEqual(got, sync.update_readme_text(got, snapshot))
        snapshot["scenes"]["rec"] = dict(scene, recall={"test": raw,
            "lineage": {"model_family": "adapted_residual_dssm_v1"}})
        self.assertIn("较上一版（残差DSSM/PQ/211融合）", sync.update_readme_text(text, snapshot))
        self.assertIn("previous residual DSSM/PQ/211 fusion", sync.update_readme_text(text, snapshot, english=True))

    def test_missing_active_bundle_fails_before_snapshot_or_write(self):
        with tempfile.TemporaryDirectory(prefix="qilin-metrics-test-") as temporary:
            root = Path(temporary)
            output = root / "new-snapshot.json"
            with patch.object(sync, "ROOT", root), patch("sys.stderr"), patch.object(sync, "build_snapshot") as build:
                with self.assertRaises(SystemExit):
                    sync.main(["--output", str(output)])
                build.assert_not_called()
                self.assertFalse(output.exists())

    def test_conditional_uses_validation_selected_run_and_frozen_hashes(self):
        with tempfile.TemporaryDirectory(prefix="qilin-metrics-test-") as temporary:
            root = Path(temporary)
            run = root / "selected"
            run.mkdir()
            (run / "manifest.json").write_text('{"scene": "search"}')
            manifest_sha = sync.sha256(run / "manifest.json")
            stage = {"NDCG@10": 0.4, "AUC": 0.6, "GAUC": 0.5, "rows": 5, "all_requests": 2}
            test = {"gbdt": stage, "selected": stage, "test_rows": 5, "test_requests": 2}
            (run / "test_metrics.json").write_text(json.dumps(test))
            test_sha = sync.sha256(run / "test_metrics.json")
            (run / "test_manifest.json").write_text(json.dumps({"frozen_training_manifest_sha256": manifest_sha, "metrics_sha256": test_sha}))
            (run / "independent_ranking_verification.json").write_text(json.dumps({"status": "PASS", "test_metrics_sha256": test_sha, "frozen_training_manifest_sha256": manifest_sha, "test_manifest_sha256": sync.sha256(run / "test_manifest.json")}))
            selection = {"search": {"selection_split": "train_internal_validation", "selected": {
                "run": "selected", "manifest_sha256": manifest_sha, "name": "frozen", "weight": 1},
                "choices": [{"name": "other", "test_NDCG": 0.99}]}}
            result = sync.conditional_ranking(root, "search", selection, sync.Sources(root), (2, 5, 3))
            self.assertEqual(result["ranking"]["test"]["NDCG@10"], 0.4)
            (run / "manifest.json").write_text('{"scene": "changed"}')
            with self.assertRaisesRegex(ValueError, "manifest changed"):
                sync.conditional_ranking(root, "search", selection, sync.Sources(root), (2, 5, 3))

    def test_readme_only_changes_four_metrics_and_latency_preserving_crlf(self):
        text = ("Intro unchanged\r\n- 在线服务\r\n  - 单机串行实测 oldnumbers\r\n- Search 离线指标\r\n"
                "  - 召回（测试集）：old\r\n  - 排序（DIEN，测试曝光）：old\r\n- Recommendation 离线指标\r\n"
                "  - 召回（测试集）：old\r\n  - 排序（DIEN，测试曝光）：old\r\nTail unchanged\r\n")
        recall = {"HitRate@10": 0.55, "HitRate@50": 0.02, "Recall@100": 0.01, "MRR@100": 0.003}
        rank = {"NDCG@10": 0.6, "AUC": 0.61, "GAUC": 0.62}
        scene = {"recall": {"test": recall}, "ranking": {"ranking": {"test": rank}}}
        result = sync.update_readme_text(text, {"scenes": {"search": scene, "rec": scene}})
        before, after = text.splitlines(keepends=True), result.splitlines(keepends=True)
        self.assertEqual(len(before), len(after))
        self.assertEqual(sum(a != b for a, b in zip(before, after)), 5)
        self.assertTrue(all(line.endswith("\r\n") for line in after))
        self.assertEqual(after[0], before[0])
        self.assertEqual(after[-1], before[-1])
        self.assertIn("HitRate@10 = 0.5500", result)
        self.assertIn("HitRate@50 = 0.0200", result)
        self.assertIn("接口返回总延迟与分阶段延迟", result)
        self.assertEqual(sync.update_readme_text(result, {"scenes": {"search": scene, "rec": scene}}), result)


class TestNumericalReplayPublication(unittest.TestCase):
    def prepare(self, root, wrong_truth_count=False):
        def write(name, payload):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload))
            return path
        plan = write('plan.json', {'scene': 'search', 'inference_contract': 'request_fp32_v1',
            'frozen_before_official_test': True, 'input_cap': 160, 'coarse_cap': 160,
            'input_contract': 'LiveContextBuilder_bound_user_profile_v2',
            'profile_categories': ['gender_enc', 'platform_enc', 'age_enc', 'location_enc']})
        candidates = write('raw.json', {'fixture': 'raw candidate bytes'})
        final = write('final.json', {'fixture': 'final candidate bytes'})
        test = write('metrics.json', {'accepted_for_release': True, 'all_test_requests': 6192,
            'parameter_choice_after_test': False, 'selection_sha256': sync.sha256(plan)})
        proof = write('proof.json', {'status': 'PASS', 'selection_sha256': sync.sha256(plan),
            'raw_candidate_sha256': sync.sha256(candidates), 'final_candidate_sha256': sync.sha256(final),
            'original_requests': 6192, 'original_positive_pairs': 1 if wrong_truth_count else 27597})
        context = write('context.json', {'fixture': 'current request features'})
        raw_test = write('raw_test.json', {'fixture': 'original test request source'})
        profile = write('profile.json', {'status': 'PASS', 'original_requests': 6192,
            'profile_mismatches': 0, 'raw_context_mismatches': 0, 'nonzero_profile_requests': 100,
            'selection_sha256': sync.sha256(plan), 'input_contract': 'LiveContextBuilder_bound_user_profile_v2',
            'profile_fields': ['gender_enc', 'platform_enc', 'age_enc', 'location_enc'],
            'user_profile_contract': 'shared_live_context_v1',
            'inputs_sha256': {str(path.resolve()): sync.sha256(path) for path in (plan, test, context, raw_test)}})
        sources = {name: {'path': path.name, 'sha256': sync.sha256(path)} for name, path in {
            'plan': plan, 'test_metrics': test, 'candidates': candidates,
            'final_candidates': final, 'independent_verification': proof, 'profile_input_verification': profile,
            'request_context': context, 'raw_test': raw_test}.items()}
        release = write('release.json', {'sources': sources})
        write('rank/manifest.json', {'dien': {'inference_contract': 'request_fp32_v1'}})
        digest = sync.sha256(release)
        return {'rank_dir': 'rank', 'evaluation': {'numerical_replay': {'path': release.name, 'sha256': digest}},
                'assets': {release.name: digest}}, candidates

    def test_current_proof_passes_but_changed_candidates_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix='qilin-replay-publication-') as directory:
            root = Path(directory)
            bundle, candidate = self.prepare(root)
            self.assertIsNotNone(sync.load_search_numerical_replay(root, bundle, sync.Sources(root)))
            candidate.write_text('changed bytes')
            with self.assertRaisesRegex(ValueError, 'source changed'):
                sync.load_search_numerical_replay(root, bundle, sync.Sources(root))

    def test_pass_badge_cannot_override_wrong_truth_denominator(self):
        with tempfile.TemporaryDirectory(prefix='qilin-replay-publication-') as directory:
            root = Path(directory)
            bundle, _ = self.prepare(root, wrong_truth_count=True)
            with self.assertRaisesRegex(ValueError, 'independent full-test proof'):
                sync.load_search_numerical_replay(root, bundle, sync.Sources(root))

    def test_neutral_only_reference_cannot_pass_publication(self):
        with tempfile.TemporaryDirectory(prefix='qilin-replay-publication-') as directory:
            root = Path(directory)
            bundle, _ = self.prepare(root)
            profile = root / 'profile.json'
            data = json.loads(profile.read_text());data['nonzero_profile_requests'] = 0
            profile.write_text(json.dumps(data))
            release = root / 'release.json'
            evidence = json.loads(release.read_text())
            evidence['sources']['profile_input_verification']['sha256'] = sync.sha256(profile)
            release.write_text(json.dumps(evidence))
            bundle['evaluation']['numerical_replay']['sha256'] = sync.sha256(release)
            bundle['assets']['release.json'] = sync.sha256(release)
            with self.assertRaisesRegex(ValueError, 'neutral-only profile reference'):
                sync.load_search_numerical_replay(root, bundle, sync.Sources(root))


    def test_profile_proof_from_another_request_context_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix='qilin-replay-publication-') as directory:
            root = Path(directory)
            bundle, _ = self.prepare(root)
            profile = root / 'profile.json'
            data = json.loads(profile.read_text())
            data['inputs_sha256'][str((root / 'context.json').resolve())] = '0' * 64
            profile.write_text(json.dumps(data))
            release = root / 'release.json'
            evidence = json.loads(release.read_text())
            evidence['sources']['profile_input_verification']['sha256'] = sync.sha256(profile)
            release.write_text(json.dumps(evidence))
            bundle['evaluation']['numerical_replay']['sha256'] = sync.sha256(release)
            bundle['assets']['release.json'] = sync.sha256(release)
            with self.assertRaisesRegex(ValueError, 'current test phase and request context'):
                sync.load_search_numerical_replay(root, bundle, sync.Sources(root))


if __name__ == "__main__":
    unittest.main()


class TestAdaptedReleasePublication(unittest.TestCase):
    def prepare(self, root):
        def write(name, value):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
            return {"path": name, "sha256": sync.sha256(path)}
        digest = "a" * 64
        config = dict(version=1, feature_schema="rec_context_23_v1", alpha=.5,
                      prior_precision="float32_rrf211_v1")
        fusion = {}
        for key in ("model", "support", "pca_vectors", "pca_ids", "pca_mean", "pca_components"):
            entry = write("fusion/" + key + ".json", {"asset": key})
            config[key] = entry["path"]
            fusion[key] = entry["sha256"]
        policy = dict(strategy="learned_rrf", context_policy="learned_filter_history",
            weights=dict(ann=2, swing=1, usercf=1), rrf_k=60, route_topk=1000, topk=500,
            cf_config=dict(fold_in=True, neighbors=200, history_weight=.25), cf_graph_dir="cf",
            learned_fusion=config)
        recipe = dict(model_checkpoint_sha256=digest, item_vectors_sha256=digest,
            index_sha256=digest, index_type="FlatIP", nprobe=None, fusion_assets_sha256=fusion)
        entries = {}
        entries["selection"] = write("selection.json", dict(scene="rec", selection_split="train_internal_validation",
            frozen_before_official_test=True, inference_contract="request_fp32_v1", recipe=recipe,
            recall_policy=policy, training_cutoff_exclusive=100))
        entries["candidates"] = write("candidates.json", {"fixture": "actual candidates"})
        entries["raw_test"] = write("raw_test.json", {"fixture": "original truth"})
        binding = dict(selection_sha256=entries["selection"]["sha256"],
            candidate_sha256=entries["candidates"]["sha256"], raw_test_sha256=entries["raw_test"]["sha256"])
        entries["test_metrics"] = write("test.json", dict(binding, parameter_choice_after_test=False,
            accepted_for_release=True, all_requests=11115, unique_clicked_pairs=44683))
        entries["independent_verification"] = write("proof.json", dict(binding, status="PASS",
            original_requests=11115, original_positive_pairs=44683))
        entries["model_checkpoint"] = write("checkpoint.fixture", {"fixture": "trained model bytes"})
        entries["item_vectors"] = write("vectors.fixture", {"fixture": "catalog vector bytes"})
        recipe["model_checkpoint_sha256"] = entries["model_checkpoint"]["sha256"]
        recipe["item_vectors_sha256"] = entries["item_vectors"]["sha256"]
        plan_path = root / "selection.json"
        plan_data = json.loads(plan_path.read_text()); plan_data["recipe"] = recipe
        plan_path.write_text(json.dumps(plan_data)); entries["selection"]["sha256"] = sync.sha256(plan_path)
        for filename, logical in (("test.json", "test_metrics"), ("proof.json", "independent_verification")):
            path = root / filename; data = json.loads(path.read_text())
            data["selection_sha256"] = entries["selection"]["sha256"]
            path.write_text(json.dumps(data)); entries[logical]["sha256"] = sync.sha256(path)
        for name in ("plan", "data_contract", "summary"):
            entries["content_training_" + name] = write("training_" + name + ".json", {"fixture": name})
        entries["model_selection"] = write("model.json", dict(
            checkpoint_sha256=recipe["model_checkpoint_sha256"], item_vectors_sha256=recipe["item_vectors_sha256"]))
        entries["model_metadata"] = write("model_metadata.json", dict(model_family="adapted_residual_dssm_v1",
            gradient_update_scope="content_adapter_stage_only", gradient_updates=327,
            packaging_gradient_updates=0, exported_full_checkpoint_sha256=recipe["model_checkpoint_sha256"], residual_weight=.25,
            content_training_steps_by_epoch=[109, 109, 109],
            **{"content_training_" + name + "_sha256": entries["content_training_" + name]["sha256"]
               for name in ("plan", "data_contract", "summary")}))
        index = dict(index_sha256=digest, index_type="FlatIP", selected_checkpoint_sha256=recipe["model_checkpoint_sha256"],
                     selected_item_vectors_sha256=recipe["item_vectors_sha256"])
        entries["index_selection"] = write("index.json", index)
        write("recall/index/dssm_rec_hard_ivfpq_meta.json", index)
        for name in ("graph.npz", "users.npy", "items.npy", "swing.npz"):
            entries["cf_" + name] = write("cf/" + name, {"fixture": name})
        release = dict(release_schema="adapted_content_v1", scene="rec", evaluated_at="2026-11-01", ranking_strategy="recall_only",
                       accepted_for_release=True, sources=entries)
        entry = write("release.json", release)
        bundle = dict(recall_dir="recall", recall_policy=policy, training_cutoff_exclusive=100,
            packaging_provenance=dict(full_checkpoint_source_sha256=recipe["model_checkpoint_sha256"]),
            evaluation=dict(release_evidence=entry),
            assets={entry["path"]: entry["sha256"], **{config[k]: fusion[k] for k in fusion}})
        return bundle

    def test_trained_steps_and_exact_index_lineage_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = self.prepare(root)
            release = sync.load_rec_release(root, bundle, sync.Sources(root))
            self.assertEqual(release["lineage"]["gradient_updates"], 327)
            self.assertEqual(release["lineage"]["packaging_gradient_updates"], 0)
            self.assertEqual(release["lineage"]["index_type"], "FlatIP")
            self.assertIsNone(release["lineage"]["nprobe"])
            self.assertEqual(release["lineage"]["evaluated_at"], "2026-11-01")

    def test_self_consistent_pass_badge_cannot_override_wrong_truth_or_policy(self):
        cases = (("proof.json", "original_positive_pairs", 1),
                 ("test.json", "parameter_choice_after_test", True),
                 ("model_metadata.json", "gradient_updates", 0),
                 ("selection.json", "inference_contract", "batched_fp32"))
        for name, key, value in cases:
            with self.subTest(name=name, key=key), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                bundle = self.prepare(root)
                path = root / name
                data = json.loads(path.read_text()); data[key] = value
                path.write_text(json.dumps(data))
                release_path = root / "release.json"
                release = json.loads(release_path.read_text())
                for entry in release["sources"].values():
                    if entry["path"] == name:
                        entry["sha256"] = sync.sha256(path)
                release_path.write_text(json.dumps(release))
                bundle["evaluation"]["release_evidence"]["sha256"] = sync.sha256(release_path)
                bundle["assets"]["release.json"] = sync.sha256(release_path)
                with self.assertRaises(ValueError):
                    sync.load_rec_release(root, bundle, sync.Sources(root))

    def test_changed_fusion_asset_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = self.prepare(root)
            (root / "fusion/model.json").write_text("different model")
            with self.assertRaisesRegex(ValueError, "fusion asset"):
                sync.load_rec_release(root, bundle, sync.Sources(root))


class TestMultiInterestRefitPublication(unittest.TestCase):
    """Tiny actual-shaped DAG; no fabricated pretest acceptance wrapper."""
    def prepare(self, root):
        bundle = TestAdaptedReleasePublication().prepare(root)
        release = json.loads((root / "release.json").read_text())
        entries = release["sources"]
        def write(name, payload):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload))
            return {"path": name, "sha256": sync.sha256(path)}
        def add(key, name, payload):
            entries[key] = write(name, payload)
            return entries[key]
        add("selection", "selection.json", {"selected_recipe": "A_K3_safehead",
            "no_official_test_feedback_in_this_selection": True,
            "final_training": {"epochs": 3, "planned_updates": 408}})
        for name, payload in {"plan": {"training_cutoff_exclusive": 1732706088},
            "data_contract": {"fit_requests": 69520, "fit_usable_requests": 69313},
            "summary": {"optimizer_updates": 408}}.items():
            add("content_training_" + name, "training_" + name + ".json", payload)
        metadata = json.loads((root / "model_metadata.json").read_text())
        metadata.update(gradient_updates=408, content_training_steps_by_epoch=[136, 136, 136])
        for name in ("plan", "data_contract", "summary"):
            metadata["content_training_" + name + "_sha256"] = entries["content_training_" + name]["sha256"]
        add("model_metadata", "model_metadata.json", metadata)
        full, vectors = (entries[k]["sha256"] for k in ("model_checkpoint", "item_vectors"))
        (root / "recall/index/dssm_rec_hard_item_emb.bin").write_bytes((root / entries["item_vectors"]["path"]).read_bytes())
        index_path = root / "recall/index/dssm_rec_hard_ivfpq.faiss"
        index_path.write_bytes(b"tiny exact index fixture")
        index = {"index_type": "FlatIP", "index_sha256": sync.sha256(index_path),
                 "selected_checkpoint_sha256": full, "selected_item_vectors_sha256": vectors}
        add("index_selection", "index.json", index)
        write("recall/index/dssm_rec_hard_ivfpq_meta.json", index)
        provenance = {"model": {"exported_full_checkpoint_sha256": full},
            "content_training_cutoff_exclusive": 1732706088,
            "original_base_PCA_known_users_cutoff_exclusive": 1732696816,
            "known_users_preserved_count": 8300}
        add("final_refit_provenance", "provenance.json", provenance)
        add("model_selection", "model.json", {"epoch": 3, "checkpoint_sha256": full,
            "item_vectors_sha256": vectors, "selected_recipe_sha256": entries["selection"]["sha256"],
            "final_provenance_sha256": entries["final_refit_provenance"]["sha256"]})
        policy = bundle["recall_policy"]
        policy.update(ann_query={"version": 1, "strategy": "adapted_k3_v1"},
                      head_guard={"version": 1, "strategy": "coarse100_prior10_v1"},
                      ann_engine={"version": 1, "strategy": "cuda_cpu_refine_v1"})
        bundle["training_cutoff_exclusive"] = 1732706088
        config = {"selection_sha256": entries["selection"]["sha256"],
                  "finalrefit_full_checkpoint": str(root / entries["model_checkpoint"]["path"]),
                  "finalrefit_dir": str(root / "recall")}
        frozen_paths = [root / entries["model_metadata"]["path"], root / entries["final_refit_provenance"]["path"], root / entries["model_checkpoint"]["path"], index_path,
            root / "recall/index/dssm_rec_hard_item_emb.bin"]
        frozen_paths += [root / policy["learned_fusion"][k] for k in ("model", "support", "pca_vectors", "pca_ids", "pca_mean", "pca_components")]
        frozen_paths += [root / "cf" / k for k in ("graph.npz", "users.npy", "items.npy", "swing.npz")]
        runtime = {}
        for name in ("learned_context_fusion", "head_guard", "multi_interest", "exact_ip_filter"):
            file = root / "runtime/src/recall" / (name + ".py")
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text("# " + name)
            runtime[str(file)] = sync.sha256(file)
            bundle["assets"]["src/recall/" + name + ".py"] = runtime[str(file)]
        plan = {"config": config, "runtime_sources": runtime,
            "source_snapshot": {str(path.resolve()): {"sha256": sync.sha256(path)} for path in frozen_paths},
            "facade_manifests": {"True": {"recall_policy": policy}}}
        add("generation_plan", "generation_plan.json", plan)
        add("source_inventory", "inventory.json", {"runtime_sources": runtime})
        generation = {"status": "ALL_PREDICTIONS_SEALED_NO_FEEDBACK", "plan_sha256": entries["generation_plan"]["sha256"], "feedback_decoded": False,
            "quality_evaluation": False, "parameter_choice_after_test": False, "requests": 11115,
            "runtime_sources": runtime, "raw_test_sha256_opaque_bytes_only": entries["raw_test"]["sha256"],
            "arms": {"finalrefit_k3_safehead": {"files": {"final": entries["candidates"]}}}}
        add("generation_manifest", "generation.json", generation)
        add("generation_postexit", "postexit.json", {"status": "PASS", "actual_generator_exit_code": 0,
            "generation_manifest_sha256": entries["generation_manifest"]["sha256"],
            "plan_sha256": entries["generation_plan"]["sha256"],
            "source_inventory_sha256": entries["source_inventory"]["sha256"],
            "all_nonlog_artifacts_and_sources_real_SHA_verified": True,
            "sealed_log_prefix_and_exact_append_verified": True, "source_stat_identity_unchanged": True,
            "all5arm_final_ann_stages_pre_guard_SHA_verified": True})
        audited = {path: {"path": path, "sha256": value} for path, value in runtime.items()}
        for name in ("selection", "generation_plan", "generation_manifest", "generation_postexit", "source_inventory", "candidates", "raw_test"):
            entry = entries[name]
            audited[name] = {"path": str((root / entry["path"]).resolve()), "sha256": entry["sha256"]}
        add("independent_plan", "audit_plan.json", {"feedback_decoded": False,
            "historically_seen_regression_not_blind": True, "assets": audited})
        metrics = {"HR50": .1, "R100": .05, "KP_NDCG10": .01, "MRR100": .02, "R500": .15, "HR500": .3}
        add("independent_verification", "proof.json", {"status": "PASS", "all_requests": 11115,
            "known_positive_pairs": 44683, "catalog": 1983938, "parameter_choice_after_test": False,
            "selection_sha256": entries["selection"]["sha256"], "plan_sha256": entries["independent_plan"]["sha256"],
            "raw_test_sha256": entries["raw_test"]["sha256"], "outputs": {"finalrefit_k3_safehead": {
                "all_requests": 11115, "known_positive_pairs": 44683,
                "candidate_sha256": entries["candidates"]["sha256"], "metrics": metrics}}})
        test = {"phase": "post_test_release_acceptance", "selected_recipe": "finalrefit_k3_safehead",
            "accepted_for_release": True, "parameter_choice_after_test": False, "all_requests": 11115,
            "unique_clicked_pairs": 44683, "catalog": 1983938, "metrics": dict(zip(
                ("HitRate@50", "Recall@100", "known_positive_NDCG@10", "MRR@100", "Recall@500", "HitRate@500"), metrics.values()))}
        for field, name in {"selection_sha256": "selection", "generation_plan_sha256": "generation_plan",
            "generation_manifest_sha256": "generation_manifest", "generation_postexit_sha256": "generation_postexit",
            "independent_plan_sha256": "independent_plan", "independent_verification_sha256": "independent_verification",
            "candidate_sha256": "candidates", "raw_test_sha256": "raw_test"}.items():test[field] = entries[name]["sha256"]
        add("test_metrics", "test.json", test)
        release.update(release_schema="multi_interest_refit_v1", evaluated_at="2026-11-02")
        write("release.json", release)
        digest = sync.sha256(root / "release.json")
        bundle["evaluation"]["release_evidence"]["sha256"] = digest
        bundle["assets"]["release.json"] = digest
        bundle["packaging_provenance"]["selected_dssm_sha256"] = entries["model_selection"]["sha256"]
        return bundle

    def resign(self, root, bundle):
        path = root / "release.json"
        release = json.loads(path.read_text())
        for entry in release["sources"].values():entry["sha256"] = sync.sha256(root / entry["path"])
        test_path = root / release["sources"]["test_metrics"]["path"]
        test = json.loads(test_path.read_text())
        for field in ("selection", "generation_plan", "generation_manifest", "generation_postexit", "independent_plan", "independent_verification", "raw_test"):
            test[field + "_sha256"] = release["sources"][field]["sha256"]
        test["candidate_sha256"] = release["sources"]["candidates"]["sha256"]
        test_path.write_text(json.dumps(test))
        release["sources"]["test_metrics"]["sha256"] = sync.sha256(test_path)
        path.write_text(json.dumps(release))
        bundle["evaluation"]["release_evidence"]["sha256"] = sync.sha256(path)
        bundle["assets"]["release.json"] = sync.sha256(path)

    def test_original_selection_and_real_posttest_wrapper_publish_408_steps(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary);bundle = self.prepare(root)
            release = sync.load_rec_release(root, bundle, sync.Sources(root))
            self.assertEqual(release["lineage"]["gradient_updates"], 408)
            self.assertEqual(release["lineage"]["base_training_cutoff_exclusive"], 1732696816)
            self.assertEqual(release["lineage"]["content_training_cutoff_exclusive"], 1732706088)
            self.assertEqual(release["lineage"]["ann_engine"], "cuda_cpu_refine_v1")
            self.assertEqual(release["lineage"]["index_type"], "FlatIP")
            self.assertNotIn("frozen_before_official_test", json.loads((root / "test.json").read_text()))

    def test_resigning_wrappers_cannot_override_denominator_steps_or_metrics(self):
        cases = [("proof.json", lambda d: d.update(known_positive_pairs=1)),
            ("model_metadata.json", lambda d: d.update(gradient_updates=327)),
            ("test.json", lambda d: d["metrics"].update({"Recall@100": .9})),
            ("test.json", lambda d: d.update(frozen_before_official_test=True)),
            ("model.json", lambda d: d.update(epoch=2)),
            ("provenance.json", lambda d: d.update(original_base_PCA_known_users_cutoff_exclusive=1732706088)),
            ("generation.json", lambda d: d.update(feedback_decoded=True)),
            ("postexit.json", lambda d: d.update(actual_generator_exit_code=1))]
        for name, mutate in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary);bundle = self.prepare(root);path = root / name
                data = json.loads(path.read_text());mutate(data);path.write_text(json.dumps(data));self.resign(root, bundle)
                with self.assertRaises(ValueError):sync.load_rec_release(root, bundle, sync.Sources(root))

    def test_policy_and_runtime_tampering_are_rejected(self):
        for key in ("ann_query", "head_guard", "ann_engine"):
            with self.subTest(policy=key), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary);bundle = self.prepare(root);bundle["recall_policy"].pop(key)
                with self.assertRaises(ValueError):sync.load_rec_release(root, bundle, sync.Sources(root))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary);bundle = self.prepare(root)
            bundle["assets"]["src/recall/head_guard.py"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "serving runtime source"):
                sync.load_rec_release(root, bundle, sync.Sources(root))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary);bundle = self.prepare(root)
            (root / "runtime/src/recall/head_guard.py").write_text("changed runtime bytes")
            with self.assertRaisesRegex(ValueError, "runtime source"):sync.load_rec_release(root, bundle, sync.Sources(root))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary);bundle = self.prepare(root)
            (root / "recall/index/dssm_rec_hard_ivfpq.faiss").write_bytes(b"different index")
            with self.assertRaises(ValueError):sync.load_rec_release(root, bundle, sync.Sources(root))


class TestMultiInterestReadme(unittest.TestCase):
    def test_immediate_global_baseline_label_does_not_use_old_pq_label(self):
        text = ("- 在线服务\n  - 接口返回总延迟与分阶段延迟\n- Search 离线指标\n"
                "  - 召回（测试集）：old\n  - 排序（DIEN，测试曝光）：old\n- Recommendation 离线指标\n"
                "  - 召回（测试集）：old\n  - 排序（DIEN，测试曝光）：old\n")
        values = {"HitRate@10": .1, "HitRate@50": .11, "Recall@100": .061,
                  "known_positive_NDCG@10": .015, "MRR@100": .03}
        scene = {"recall": {"test": values}, "pipeline": {"test": values},
            "ranking": {"ranking": {"test": {"NDCG@10": .4, "AUC": .6, "GAUC": .65}}},
            "comparisons": {"end_to_end": {"metrics": [{"name": "Recall@100", "current": .061, "relative_delta_pct": 22.}]}}}
        rec = dict(scene, recall={"test": values, "lineage": {"model_family": "adapted_residual_dssm_v1", "ann_query_strategy": "adapted_k3_v1"}})
        snapshot = {"scenes": {"search": scene, "rec": rec}}
        result = sync.update_readme_text(text, snapshot)
        self.assertIn("全局单兴趣DSSM/FlatIP/学习融合", result)
        self.assertNotIn("残差DSSM/PQ/211", result)
        self.assertNotIn("召回策略对照", result)
        self.assertIn("previous global single-interest DSSM/FlatIP/learned fusion", sync.update_readme_text(text, snapshot, english=True))
