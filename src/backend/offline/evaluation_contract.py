"""评测验收规则：计算成功不等于时间隔离或线上链路验收通过。"""
from __future__ import annotations


def acceptance_checks(scenes: dict) -> dict:
    checks = []
    for scene, result in scenes.items():
        audit = result["data_audit"]
        checks.append({"scene": scene, "check": "request_id_disjoint",
                       "status": "PASS" if audit["train_test_request_overlap"] == 0 else "FAIL"})
        checks.append({"scene": scene, "check": "chronological_split",
                       "status": "PASS" if audit["chronological_split"] else "FAIL"})
        features = result.get("ranking", {}).get("dien", {}).get("feature_audit", {})
        for field, name in (("exposure_position_features", "pre_exposure_features"),
                            ("missing_numeric_features", "required_numeric_features")):
            checks.append({"scene": scene, "check": name,
                           "status": ("FAIL" if features[field] else "PASS") if field in features else "UNKNOWN"})
        # 冻结文件只能复算其输出，不能证明当初模型/索引的训练过程。
        for name in ("model_index_training_lineage", "point_in_time_statistics",
                     "request_time_catalog_availability", "candidate_generation_lineage",
                     "serving_funnel_equivalence"):
            checks.append({"scene": scene, "check": name, "status": "UNKNOWN"})
    status = "PASS" if checks and all(row["status"] == "PASS" for row in checks) else "BLOCKED"
    return {"status": status, "checks": checks}
