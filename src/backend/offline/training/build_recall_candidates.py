"""离线训练子阶段：构建多路召回候选（train/test）。"""

from __future__ import annotations

import argparse
from pathlib import Path

from backend.offline.config import load_offline_config
from recall.build_multiroute_recall import build_multiroute_recall


def _parse_modes(modes: str) -> list[str]:
    out = [x.strip().lower() for x in modes.split(",") if x.strip()]
    invalid = [x for x in out if x not in {"easy", "hard"}]
    if invalid:
        raise ValueError(f"invalid modes: {invalid}")
    if not out:
        raise ValueError("modes is empty")
    return out


def run_build_multiroute(
    scene: str,
    modes: str,
    recall_topk: int,
    route_min_quota: int,
    route_max_share: float,
) -> None:
    mode_list = _parse_modes(modes)
    for mode in mode_list:
        for split in ["train", "test"]:
            print(
                f"[Offline/Training/MultiRoute] build_multiroute_recall(scene={scene}, split={split}, tag={mode}, topk={recall_topk})"
            )
            build_multiroute_recall(
                scene=scene,
                split=split,
                tag=mode,
                topk=recall_topk,
                ann_topk=recall_topk,
                swing_topk=recall_topk,
                usercf_topk=recall_topk,
                route_min_quota=route_min_quota,
                route_max_share=route_max_share,
                merge_order=["ann", "swing", "usercf"],
            )


def main() -> None:
    cfg = load_offline_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=["search", "rec"], default=cfg.scene)
    parser.add_argument("--modes", type=str, default=cfg.modes)
    parser.add_argument("--recall-topk", type=int, default=cfg.recall_topk)
    parser.add_argument("--route-min-quota", type=int, default=cfg.route_min_quota)
    parser.add_argument("--route-max-share", type=float, default=cfg.route_max_share)
    args = parser.parse_args()

    run_build_multiroute(
        scene=args.scene,
        modes=args.modes,
        recall_topk=args.recall_topk,
        route_min_quota=args.route_min_quota,
        route_max_share=args.route_max_share,
    )


if __name__ == "__main__":
    main()
