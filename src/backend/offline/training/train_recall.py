"""离线训练子阶段：训练召回模型并构建 ANN 索引。"""

from __future__ import annotations

import argparse
from pathlib import Path

from backend.offline.config import load_offline_config

def _parse_modes(modes: str) -> list[str]:
    out = [x.strip().lower() for x in modes.split(",") if x.strip()]
    invalid = [x for x in out if x not in {"easy", "hard"}]
    if invalid:
        raise ValueError(f"invalid modes: {invalid}")
    if not out:
        raise ValueError("modes is empty")
    return out


def run_train_recall(
    base_dir: Path,
    scene: str,
    modes: str,
    num_neg_per_pos: int,
    skip_cf: bool,
    skip_dssm: bool,
    skip_faiss: bool,
) -> None:
    from recall.build_faiss_ivfpq import build_faiss_index  # noqa: PLC0415
    from recall.build_swing_index import build_swing_for_scene
    from recall.build_usercf_index import build_usercf_for_scene
    from recall.cf_shared_index import ensure_user_item_index
    from recall.dssm_trainer import main as train_dssm

    mode_list = _parse_modes(modes)

    if not skip_cf:
        print(f"[Offline/Training/Recall] ensure_user_item_index(scene={scene}, split=train)")
        ensure_user_item_index(scene=scene, split="train", use_cache=True, rebuild=False)
        print(f"[Offline/Training/Recall] build_usercf_for_scene(scene={scene}, split=train)")
        build_usercf_for_scene(
            scene=scene,
            split="train",
            topk=50,
            max_items_per_user=200,
            max_users_per_item=300,
            interest_col="y_multi",
            min_interest=0.0,
            rebuild_ui_index=False,
        )
        print(f"[Offline/Training/Recall] build_swing_for_scene(scene={scene}, split=train)")
        build_swing_for_scene(
            scene=scene,
            split="train",
            topk=50,
            alpha=1.0,
            max_items_per_user=200,
            max_users_per_item=200,
            candidate_topn=200,
            min_common_users=2,
            interest_col="y_multi",
            min_interest=0.0,
            rebuild_ui_index=False,
        )

    hard_neg_path = base_dir / "outputs" / "data" / f"dssm_hard_neg_{scene}.parquet"
    for mode in mode_list:
        if not skip_dssm:
            print(
                f"[Offline/Training/Recall] dssm_trainer.main(scene={scene}, neg_mode={mode}, num_neg_per_pos={num_neg_per_pos})"
            )
            hard_path = None
            if mode == "hard" and hard_neg_path.exists():
                hard_path = hard_neg_path
            train_dssm(
                scene=scene,
                neg_mode=mode,
                hard_neg_path=hard_path,
                num_neg_per_pos=num_neg_per_pos,
            )

        if not skip_faiss:
            print(f"[Offline/Training/Recall] build_faiss_index(scene={scene}, tag={mode})")
            build_faiss_index(scene=scene, tag=mode)


def main() -> None:
    cfg = load_offline_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=["search", "rec"], default=cfg.scene)
    parser.add_argument("--modes", type=str, default=cfg.modes)
    parser.add_argument("--num-neg-per-pos", type=int, default=cfg.dssm_num_neg_per_pos)
    parser.add_argument("--skip-cf", action="store_true")
    parser.add_argument("--skip-dssm", action="store_true")
    parser.add_argument("--skip-faiss", action="store_true")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[4]
    run_train_recall(
        base_dir=base_dir,
        scene=args.scene,
        modes=args.modes,
        num_neg_per_pos=args.num_neg_per_pos,
        skip_cf=args.skip_cf,
        skip_dssm=args.skip_dssm,
        skip_faiss=args.skip_faiss,
    )


if __name__ == "__main__":
    main()
