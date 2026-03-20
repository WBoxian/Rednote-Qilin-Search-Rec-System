"""离线训练子阶段：训练排序模型（GBDT + DIEN）。"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

from backend.offline.config import load_offline_config


def _load_module_from_src(module_name: str, file_path: Path):
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module {module_name} from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

def _parse_modes(modes: str) -> list[str]:
    out = [x.strip().lower() for x in modes.split(",") if x.strip()]
    invalid = [x for x in out if x not in {"easy", "hard"}]
    if invalid:
        raise ValueError(f"invalid modes: {invalid}")
    if not out:
        raise ValueError("modes is empty")
    return out


def run_train_rankers(
    base_dir: Path,
    scene: str,
    modes: str,
    recall_topk: int,
    gbdt_train_topn: int,
    coarse_topn: int,
    hard_neg_per_req: int,
    dien_train_max_neg_per_req: int,
    stages: str = "preranking,ranking",
) -> None:
    src_training_dir = base_dir / "src" / "training"
    _load_module_from_src("utils", src_training_dir / "utils.py")
    dien_ranker = _load_module_from_src("qilin_training_dien_ranker", src_training_dir / "dien_ranker.py")
    gbdt_ranker = _load_module_from_src("qilin_training_gbdt_ranker", src_training_dir / "gbdt_ranker.py")

    mode_list = _parse_modes(modes)
    selected = {x.strip().lower() for x in str(stages).split(",") if x.strip()}
    allowed = {"preranking", "ranking"}
    invalid = selected - allowed
    if invalid:
        raise ValueError(f"invalid ranker stages: {sorted(invalid)}")

    run_preranking = (not selected) or ("preranking" in selected)
    run_ranking = (not selected) or ("ranking" in selected)

    for mode in mode_list:
        if run_preranking:
            print(
                f"[Offline/Training/Rankers] gbdt_ranker.main(scene={scene}, output_tag={mode}, recall_tag={mode})"
            )
            gbdt_ranker.main(
                scene=scene,
                output_tag=mode,
                candidate_path=None,
                recall_tag=mode,
                train_candidate_topn=gbdt_train_topn,
                dien_topn=coarse_topn,
                keep_positive_for_dien=True,
                skip_export_dien=False,
                hard_neg_input_topn=recall_topk,
                hard_neg_per_req=hard_neg_per_req,
                skip_export_hard_neg=False,
                print_every=10,
                train_max_neg_per_req=0,
            )

        if run_ranking:
            print(
                f"[Offline/Training/Rankers] dien_ranker.main(scene={scene}, output_tag={mode}, "
                f"train_max_neg_per_req={dien_train_max_neg_per_req})"
            )
            dien_ranker.main(
                scene=scene,
                train_path=base_dir / "outputs" / "data" / f"dien_{scene}_{mode}_train_from_gbdt_top{coarse_topn}.parquet",
                output_tag=mode,
                coarse_topn=coarse_topn,
                train_max_neg_per_req=dien_train_max_neg_per_req,
                train_head_neg_keep=7,
                aux_loss_weight=0.05,
                max_epochs=10,
                early_stop_rounds=6,
            )


def main() -> None:
    cfg = load_offline_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=["search", "rec"], default=cfg.scene)
    parser.add_argument("--modes", type=str, default=cfg.modes)
    parser.add_argument("--recall-topk", type=int, default=cfg.recall_topk)
    parser.add_argument("--gbdt-train-topn", type=int, default=cfg.gbdt_train_topn)
    parser.add_argument("--coarse-topn", type=int, default=cfg.coarse_topn)
    parser.add_argument("--hard-neg-per-req", type=int, default=cfg.hard_neg_per_req)
    parser.add_argument("--dien-train-max-neg-per-req", type=int, default=cfg.dien_train_max_neg_per_req)
    parser.add_argument("--stages", type=str, default="preranking,ranking", help="可选：preranking,ranking（逗号分隔）")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[4]
    run_train_rankers(
        base_dir=base_dir,
        scene=args.scene,
        modes=args.modes,
        recall_topk=args.recall_topk,
        gbdt_train_topn=args.gbdt_train_topn,
        coarse_topn=args.coarse_topn,
        hard_neg_per_req=args.hard_neg_per_req,
        dien_train_max_neg_per_req=args.dien_train_max_neg_per_req,
        stages=args.stages,
    )


if __name__ == "__main__":
    main()
