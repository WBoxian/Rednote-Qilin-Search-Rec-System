"""离线产物本地部署（backend）。"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_file():
        shutil.copy2(src, dst)
    else:
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    return True


def deploy_local_artifacts(base_dir: Path, scene: str) -> Path:
    deploy_root = base_dir / "outputs" / "deploy" / scene
    models_dir = base_dir / "outputs" / "models"
    index_dir = base_dir / "outputs" / "index"

    summary: dict[str, dict[str, list[str]]] = {}
    for tag in ["easy", "hard"]:
        target = deploy_root / tag
        target.mkdir(parents=True, exist_ok=True)
        copied: list[str] = []

        patterns = [
            models_dir / f"dssm_{scene}_{tag}.pt",
            models_dir / f"dssm_{scene}_{tag}_user_tower.pt",
            models_dir / f"dssm_{scene}_{tag}_item_tower.pt",
            models_dir / f"lgb_{scene}_{tag}.pkl",
            models_dir / f"xgb_{scene}_{tag}.pkl",
            models_dir / f"dien_{scene}_{tag}.pt",
            index_dir / f"dssm_{scene}_{tag}_ivfpq.faiss",
            index_dir / f"dssm_{scene}_{tag}_ivfpq_meta.json",
            index_dir / f"dssm_{scene}_{tag}_row2note.npy",
            index_dir / f"dssm_{scene}_{tag}_item_map.json",
            index_dir / f"dssm_{scene}_{tag}_item_meta.json",
            index_dir / f"dssm_{scene}_{tag}_item_emb.bin",
        ]

        for src in patterns:
            if src.parent == models_dir:
                dst = target / "models" / src.name
            elif src.parent == index_dir:
                dst = target / "index" / src.name
            else:
                continue
            if _copy_if_exists(src, dst):
                copied.append(str(dst.relative_to(target)))

        summary[tag] = {"copied": copied}

    manifest = deploy_root / "deploy_manifest.json"
    manifest.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Storage/LocalDeploy] done: {manifest}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=["search", "rec"], required=True)
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[4]
    deploy_local_artifacts(base_dir=base_dir, scene=args.scene)


if __name__ == "__main__":
    main()
