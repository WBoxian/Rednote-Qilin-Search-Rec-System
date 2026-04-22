"""离线训练阶段：按职责拆分编排（召回训练 -> 候选构建 -> 排序训练）。"""

from __future__ import annotations

import argparse
from pathlib import Path

from backend.offline.config import load_offline_config
from backend.offline.training.build_recall_candidates import run_build_multiroute
from backend.offline.training.train_rankers import run_train_rankers
from backend.offline.training.train_recall import run_train_recall


def _parse_modes(modes: str) -> list[str]:
    out = [x.strip().lower() for x in str(modes).split(",") if x.strip()]
    invalid = [x for x in out if x not in {"easy", "hard"}]
    if invalid:
        raise ValueError(f"invalid modes: {invalid}")
    if not out:
        raise ValueError("modes is empty")
    return out


def run_training(
    base_dir: Path,
    scene: str,
    modes: str,
    recall_topk: int,
    preranking_topn: int,
    gbdt_train_topn: int,
    route_min_quota: int,
    route_max_share: float,
    hard_neg_per_req: int,
    dssm_num_neg_per_pos: int,
    dien_train_max_neg_per_req: int,
    train_stages: str = "",
) -> None:
    mode_list = _parse_modes(modes)
    has_easy = "easy" in mode_list
    has_hard = "hard" in mode_list

    selected = {x.strip().lower() for x in str(train_stages).split(",") if x.strip()}
    allowed = {"recall", "multiroute", "preranking", "ranking"}
    invalid = selected - allowed
    if invalid:
        raise ValueError(f"invalid training stages: {sorted(invalid)}")

    print(
        "[Offline/Training] selector="
        f"{','.join(sorted(selected)) if selected else 'all'} "
        f"| scene={scene} | modes={','.join(mode_list)}"
    )

    def enabled(stage: str) -> bool:
        return (not selected) or (stage in selected)

    if has_easy:
        if enabled("recall"):
            print("[Offline/Training] stage=easy/recall")
            run_train_recall(
                base_dir=base_dir,
                scene=scene,
                modes="easy",
                num_neg_per_pos=dssm_num_neg_per_pos,
                skip_cf=False,
                skip_dssm=False,
                skip_faiss=False,
            )

        if enabled("multiroute"):
            print("[Offline/Training] stage=easy/multiroute")
            run_build_multiroute(
                scene=scene,
                modes="easy",
                recall_topk=recall_topk,
                route_min_quota=route_min_quota,
                route_max_share=route_max_share,
            )

        if enabled("preranking") or enabled("ranking"):
            print("[Offline/Training] stage=easy/rankers")
            run_train_rankers(
                base_dir=base_dir,
                scene=scene,
                modes="easy",
                recall_topk=recall_topk,
                gbdt_train_topn=gbdt_train_topn,
                preranking_topn=preranking_topn,
                hard_neg_per_req=hard_neg_per_req,
                dien_train_max_neg_per_req=dien_train_max_neg_per_req,
                stages=",".join([
                    s for s in ["preranking", "ranking"] if enabled(s)
                ]),
            )

    if has_hard:
        if not has_easy:
            print("[Offline/Training][Warn] hard mode without easy stage may miss exported hard negatives.")

        if enabled("recall"):
            print("[Offline/Training] stage=hard/recall")
            run_train_recall(
                base_dir=base_dir,
                scene=scene,
                modes="hard",
                num_neg_per_pos=dssm_num_neg_per_pos,
                skip_cf=has_easy,
                skip_dssm=False,
                skip_faiss=False,
            )

        if enabled("multiroute"):
            print("[Offline/Training] stage=hard/multiroute")
            run_build_multiroute(
                scene=scene,
                modes="hard",
                recall_topk=recall_topk,
                route_min_quota=route_min_quota,
                route_max_share=route_max_share,
            )

        if enabled("preranking") or enabled("ranking"):
            print("[Offline/Training] stage=hard/rankers")
            run_train_rankers(
                base_dir=base_dir,
                scene=scene,
                modes="hard",
                recall_topk=recall_topk,
                gbdt_train_topn=gbdt_train_topn,
                preranking_topn=preranking_topn,
                hard_neg_per_req=hard_neg_per_req,
                dien_train_max_neg_per_req=dien_train_max_neg_per_req,
                stages=",".join([
                    s for s in ["preranking", "ranking"] if enabled(s)
                ]),
            )


def main() -> None:
    cfg = load_offline_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=["search", "rec"], default=cfg.scene)
    parser.add_argument("--modes", type=str, default=cfg.modes)
    parser.add_argument("--recall-topk", type=int, default=cfg.recall_topk)
    parser.add_argument("--preranking-topn", dest="preranking_topn", type=int, default=cfg.preranking_topn)
    parser.add_argument("--gbdt-train-topn", type=int, default=cfg.gbdt_train_topn)
    parser.add_argument("--route-min-quota", type=int, default=cfg.route_min_quota)
    parser.add_argument("--route-max-share", type=float, default=cfg.route_max_share)
    parser.add_argument("--hard-neg-per-req", type=int, default=cfg.hard_neg_per_req)
    parser.add_argument("--dssm-num-neg-per-pos", type=int, default=cfg.dssm_num_neg_per_pos)
    parser.add_argument("--dien-train-max-neg-per-req", type=int, default=cfg.dien_train_max_neg_per_req)
    parser.add_argument("--train", type=str, default="", metavar="STAGES",
                        help="只运行指定训练子阶段，逗号分隔：recall,multiroute,preranking,ranking")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[4]
    run_training(
        base_dir=base_dir,
        scene=args.scene,
        modes=args.modes,
        recall_topk=args.recall_topk,
        preranking_topn=args.preranking_topn,
        gbdt_train_topn=args.gbdt_train_topn,
        route_min_quota=args.route_min_quota,
        route_max_share=args.route_max_share,
        hard_neg_per_req=args.hard_neg_per_req,
        dssm_num_neg_per_pos=args.dssm_num_neg_per_pos,
        dien_train_max_neg_per_req=args.dien_train_max_neg_per_req,
        train_stages=args.train,
    )


if __name__ == "__main__":
    main()
