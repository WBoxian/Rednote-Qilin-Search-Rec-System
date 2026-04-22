"""Offline 统一超参数配置。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class OfflineConfig:
    base_dir: Path
    scene: str = "rec"
    modes: str = "easy,hard"

    data_split: str = "all"
    feature_split: str = "all"
    enable_note_text_emb: bool = True
    enable_note_image_emb: bool = True
    enable_query_emb: bool = True

    recall_topk: int = 1000
    preranking_topn: int = 500
    gbdt_train_topn: int = 500
    route_min_quota: int = 100
    route_max_share: float = 0.8
    hard_neg_per_req: int = 100
    dssm_num_neg_per_pos: int = 3
    dien_train_max_neg_per_req: int = 3

    @property
    def outputs_dir(self) -> Path:
        return self.base_dir / "outputs"

    @property
    def deploy_dir(self) -> Path:
        return self.outputs_dir / "deploy"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def load_offline_config(scene: str | None = None) -> OfflineConfig:
    base_dir = Path(__file__).resolve().parents[3]
    return OfflineConfig(
        base_dir=base_dir,
        scene=scene or os.getenv("QILIN_SCENE", "rec"),
        modes=os.getenv("QILIN_MODES", "easy,hard"),
        data_split=os.getenv("QILIN_DATA_SPLIT", "all"),
        feature_split=os.getenv("QILIN_FEATURE_SPLIT", "all"),
        enable_note_text_emb=_env_bool("QILIN_ENABLE_NOTE_TEXT_EMB", True),
        enable_note_image_emb=_env_bool("QILIN_ENABLE_NOTE_IMAGE_EMB", True),
        enable_query_emb=_env_bool("QILIN_ENABLE_QUERY_EMB", True),
        recall_topk=int(os.getenv("QILIN_RECALL_TOPK", "1000")),
        preranking_topn=int(os.getenv("QILIN_PRERANKING_TOPN", "500")),
        gbdt_train_topn=int(os.getenv("QILIN_GBDT_TRAIN_TOPN", "500")),
        route_min_quota=int(os.getenv("QILIN_ROUTE_MIN_QUOTA", "100")),
        route_max_share=float(os.getenv("QILIN_ROUTE_MAX_SHARE", "0.8")),
        hard_neg_per_req=int(os.getenv("QILIN_HARD_NEG_PER_REQ", "100")),
        dssm_num_neg_per_pos=int(os.getenv("QILIN_DSSM_NUM_NEG_PER_POS", "3")),
        dien_train_max_neg_per_req=int(os.getenv("QILIN_DIEN_TRAIN_MAX_NEG_PER_REQ", "3")),
    )


__all__ = ["OfflineConfig", "load_offline_config"]
