"""离线特征构建阶段。"""

from __future__ import annotations

import argparse
from pathlib import Path

from backend.offline.config import load_offline_config

def run_build_features(base_dir: Path, scene: str, split: str) -> None:
    from preprocess.build_features import build_features as build_wide_features

    print(f"[Offline/Feature] build_features(scene={scene}, split={split})")
    build_wide_features(scene=scene, split=split)


def run_query_embeddings(base_dir: Path, scene: str) -> None:
    from preprocess import build_query_text_emb

    print(f"[Offline/Feature] build_query_text_emb(scene={scene})")
    build_query_text_emb.main(scene)


def run_note_text_embeddings(base_dir: Path) -> None:
    from preprocess import build_note_text_emb

    print("[Offline/Feature] build_note_text_emb()")
    build_note_text_emb.main()


def run_note_image_embeddings(base_dir: Path) -> None:
    from preprocess import build_note_image_emb

    print("[Offline/Feature] build_note_image_emb()")
    build_note_image_emb.main()


def main() -> None:
    cfg = load_offline_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=["search", "rec"], default=cfg.scene)
    parser.add_argument("--split", choices=["train", "test", "all"], default=cfg.feature_split)
    parser.add_argument("--skip-note-text-emb", action="store_true")
    parser.add_argument("--skip-note-image-emb", action="store_true")
    parser.add_argument("--skip-query-emb", action="store_true")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[4]
    splits = ["train", "test"] if args.split == "all" else [args.split]
    for split in splits:
        run_build_features(base_dir, args.scene, split)
    if not args.skip_note_text_emb:
        run_note_text_embeddings(base_dir)
    if not args.skip_note_image_emb:
        run_note_image_embeddings(base_dir)
    if ("train" in splits) and (not args.skip_query_emb):
        run_query_embeddings(base_dir, args.scene)


if __name__ == "__main__":
    main()
