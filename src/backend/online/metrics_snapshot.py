"""面向看板与 API 的统一全量测试指标快照。"""
from __future__ import annotations

import json
from pathlib import Path

SNAPSHOT_PATH = Path(__file__).with_name("full_test_metrics.json")
ROOT = Path(__file__).resolve().parents[3]


class MetricsVersionMismatch(RuntimeError):
    pass


def load_metrics_snapshot(scene: str, tag: str) -> dict:
    snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    if scene not in snapshot["scenes"]:
        raise ValueError(f"Unknown metrics scene: {scene}")
    if tag != snapshot["tag"]:
        raise ValueError(f"No full-test metrics for tag={tag}; available tag={snapshot['tag']}")
    model_id = snapshot["scenes"][scene].get("model_bundle_id")
    active = ROOT/"outputs/deploy"/scene/tag/"active_bundle.json"
    if model_id and active.exists():
        from backend.online.bundle import bundle_identity
        try:
            manifest = json.loads(active.read_text())
            valid = isinstance(manifest, dict) and manifest.get("scene") == scene and manifest.get("tag") == tag and manifest.get("bundle_id") == bundle_identity(manifest)
        except (OSError, ValueError, TypeError):
            valid = False
        if not valid or manifest["bundle_id"] != model_id:
            raise MetricsVersionMismatch("Active model and metrics snapshot versions differ; refresh the bound metrics snapshot")
    return {
        "scene": scene, "tag": tag, "protocol": snapshot["protocol"],
        **snapshot["scenes"][scene],
    }
