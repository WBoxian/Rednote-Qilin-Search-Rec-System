"""
Qilin 离线流水线（Offline Pipeline）
====================================
最简参数（推荐）：
    --only  主阶段控制（逗号分隔）
            可选：data,feature,training,deploy,feature-upload
    --train 训练子阶段控制（仅 training 生效，逗号分隔）
        可选：recall,multiroute,preranking,ranking

常用示例：
    全流程（默认）
        uv run python src/backend/offline/pipeline.py --scene rec

    只跑训练相关主阶段（自动跳过 data/feature）
        uv run python src/backend/offline/pipeline.py --scene rec --train ranking

    只跑 multiroute+粗排+精排
        uv run python src/backend/offline/pipeline.py --scene search --train multiroute,preranking,ranking

    只部署模型（把 outputs/models + index 发布到 outputs/deploy）
        uv run python src/backend/offline/pipeline.py --scene rec --only deploy

规则说明：
    只写 --train 时，主阶段默认是 training,deploy,feature-upload。
    同时写 --only + --train 时，只有 --only 包含 training，--train 才生效。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from backend.offline.config import load_offline_config
from backend.offline.data.build_samples import run_build_samples
from backend.offline.feature.build_features import (
    run_build_features,
    run_note_image_embeddings,
    run_note_text_embeddings,
    run_query_embeddings,
)
from backend.offline.storage.local_deploy import deploy_local_artifacts
from backend.offline.storage.redis_ingest import ingest_user_features_to_redis
from backend.offline.training.run_training import run_training


MAIN_STAGES = ["data", "feature", "training", "deploy", "feature-upload"]


def _parse_csv(raw: str) -> list[str]:
    return [x.strip().lower() for x in str(raw).split(",") if x.strip()]


def _normalize_train_stages(raw: str) -> str:
    out = []
    for token in _parse_csv(raw):
        out.append(token)
    allowed = {"recall", "multiroute", "preranking", "ranking"}
    invalid = set(out) - allowed
    if invalid:
        raise ValueError(f"invalid --train stages: {sorted(invalid)}")
    return ",".join(out)


def _normalize_main_stages(raw: str) -> set[str]:
    selected = set(_parse_csv(raw))
    invalid = selected - set(MAIN_STAGES)
    if invalid:
        raise ValueError(f"invalid --only stages: {sorted(invalid)}")
    return selected


def _resolve_training_modes(base_dir: Path, scene: str, configured_modes: str) -> str:
    normalized = ",".join(_parse_csv(configured_modes))
    if normalized and normalized != "easy,hard":
        return normalized

    hard_neg_file = base_dir / "outputs" / "data" / f"dssm_hard_neg_{scene}.parquet"
    if hard_neg_file.exists():
        print(f"[Backend/Offline] training modes=hard (detected hard neg: {hard_neg_file.name})")
        return "hard"

    print("[Backend/Offline] training modes=easy,hard (first run: hard neg not found)")
    return "easy,hard"

def run_data_stage(base_dir: Path, scene: str) -> None:
    # 数据阶段：按配置构建 train/test 样本
    cfg = load_offline_config(scene=scene)
    splits = ["train", "test"] if cfg.data_split == "all" else [cfg.data_split]
    for split in splits:
        run_build_samples(base_dir=base_dir, scene=scene, split=split)


def run_feature_stage(base_dir: Path, scene: str) -> None:
    # 特征阶段：构建训练特征并按开关产出多模态 embedding
    cfg = load_offline_config(scene=scene)
    splits = ["train", "test"] if cfg.feature_split == "all" else [cfg.feature_split]
    for split in splits:
        run_build_features(base_dir=base_dir, scene=scene, split=split)
    if cfg.enable_note_text_emb:
        run_note_text_embeddings(base_dir=base_dir)
    if cfg.enable_note_image_emb:
        run_note_image_embeddings(base_dir=base_dir)
    if cfg.enable_query_emb and "train" in splits:
        run_query_embeddings(base_dir=base_dir, scene=scene)


def run_training_stage(base_dir: Path, cfg, train_stages: str = "") -> None:
    # 训练阶段：召回、候选、粗排、精排统一编排
    resolved_modes = _resolve_training_modes(base_dir=base_dir, scene=cfg.scene, configured_modes=cfg.modes)
    run_training(
        base_dir=base_dir,
        scene=cfg.scene,
        modes=resolved_modes,
        recall_topk=cfg.recall_topk,
        coarse_topn=cfg.coarse_topn,
        gbdt_train_topn=cfg.gbdt_train_topn,
        route_min_quota=cfg.route_min_quota,
        route_max_share=cfg.route_max_share,
        hard_neg_per_req=cfg.hard_neg_per_req,
        dssm_num_neg_per_pos=cfg.dssm_num_neg_per_pos,
        dien_train_max_neg_per_req=cfg.dien_train_max_neg_per_req,
        train_stages=train_stages,
    )


def run_model_upload_stage(base_dir: Path, cfg) -> None:
    # 模型上传：将离线产物统一发布到 outputs/deploy/{scene}/{tag}
    manifest = deploy_local_artifacts(
        base_dir=base_dir,
        scene=cfg.scene,
    )
    print(f"[Backend/Offline] model upload done: scene={cfg.scene}, manifest={manifest}")


def run_feature_upload_stage(base_dir: Path, cfg, redis_url: str | None) -> None:
    # 特征上传：默认写入 Redis 供在线特征查询
    target_redis_url = (
        (redis_url or "").strip()
        or os.getenv("QILIN_REDIS_URL", "").strip()
        or "redis://127.0.0.1:6379/0"
    )
    ingest_user_features_to_redis(base_dir=base_dir, scene=cfg.scene, redis_url=target_redis_url)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=["search", "rec"], default="rec")
    parser.add_argument("--only", type=str, default="", metavar="STAGES",
                        help="主阶段选择：data,feature,training,deploy,feature-upload（逗号分隔）")
    parser.add_argument("--train", type=str, default="", metavar="STAGES",
                        help="训练子阶段：recall,multiroute,preranking,ranking（逗号分隔）")
    parser.add_argument("--redis-url", type=str, default="")
    args = parser.parse_args()

    cfg = load_offline_config(scene=args.scene)

    train_selector = _normalize_train_stages(args.train.strip())

    if args.only.strip():
        enabled_main = _normalize_main_stages(args.only)
    elif train_selector:
        enabled_main = {"training", "deploy", "feature-upload"}
    else:
        enabled_main = set(MAIN_STAGES)

    skip_data = "data" not in enabled_main
    skip_feature = "feature" not in enabled_main
    skip_training = "training" not in enabled_main
    skip_model_upload = "deploy" not in enabled_main
    skip_feature_upload = "feature-upload" not in enabled_main

    if train_selector:
        skip_data = True
        skip_feature = True

    print(
        "[Backend/Offline] effective stages="
        f"{','.join([s for s in MAIN_STAGES if s in enabled_main])} "
        f"| train={train_selector or 'all'}"
    )

    if not skip_data:
        print(f"[Backend/Offline] >>> start stage=data scene={cfg.scene}")
        run_data_stage(cfg.base_dir, cfg.scene)
        print(f"[Backend/Offline] <<< done stage=data scene={cfg.scene}")
    if not skip_feature:
        print(f"[Backend/Offline] >>> start stage=feature scene={cfg.scene}")
        run_feature_stage(cfg.base_dir, cfg.scene)
        print(f"[Backend/Offline] <<< done stage=feature scene={cfg.scene}")
    if not skip_training:
        print(f"[Backend/Offline] >>> start stage=training scene={cfg.scene}")
        run_training_stage(cfg.base_dir, cfg, train_stages=train_selector)
        print(f"[Backend/Offline] <<< done stage=training scene={cfg.scene}")
    if not skip_model_upload:
        print(f"[Backend/Offline] >>> start stage=deploy scene={cfg.scene}")
        run_model_upload_stage(cfg.base_dir, cfg)
        print(f"[Backend/Offline] <<< done stage=deploy scene={cfg.scene}")
    if not skip_feature_upload:
        print(f"[Backend/Offline] >>> start stage=feature-upload scene={cfg.scene}")
        run_feature_upload_stage(cfg.base_dir, cfg, redis_url=args.redis_url.strip() or None)
        print(f"[Backend/Offline] <<< done stage=feature-upload scene={cfg.scene}")

    print(f"[Backend/Offline] pipeline done. scene={cfg.scene}")


if __name__ == "__main__":
    main()
