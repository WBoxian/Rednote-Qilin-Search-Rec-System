"""面向看板与 API 的统一全量测试指标快照。"""
from __future__ import annotations

import json
from pathlib import Path

SNAPSHOT_PATH = Path(__file__).with_name("full_test_metrics.json")


def load_metrics_snapshot(scene: str, tag: str) -> dict:
    snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    if scene not in snapshot["scenes"]:
        raise ValueError(f"Unknown metrics scene: {scene}")
    if tag != snapshot["tag"]:
        raise ValueError(f"No full-test metrics for tag={tag}; available tag={snapshot['tag']}")
    return {
        "scene": scene, "tag": tag, "protocol": snapshot["protocol"],
        **snapshot["scenes"][scene],
    }
