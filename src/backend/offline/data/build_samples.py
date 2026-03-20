"""离线数据样本构建阶段：构建 Notes 和 Users join 后训练样本（samples）。"""

from __future__ import annotations

import argparse
from pathlib import Path

from backend.offline.config import load_offline_config
from preprocess.build_samples import build_samples


def run_build_samples(base_dir: Path, scene: str, split: str) -> None:
    print(f"[Offline/Data] build_samples(scene={scene}, split={split})")
    build_samples(scene=scene, split=split)


def main() -> None:
    cfg = load_offline_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=["search", "rec"], default=cfg.scene)
    parser.add_argument("--split", choices=["train", "test", "all"], default=cfg.data_split)
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[4]
    splits = ["train", "test"] if args.split == "all" else [args.split]
    for split in splits:
        run_build_samples(base_dir, args.scene, split)


if __name__ == "__main__":
    main()
