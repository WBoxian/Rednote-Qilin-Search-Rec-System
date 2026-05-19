from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from backend.online.api.main import AppContext
from backend.online.config import DEFAULT_GBDT_TOPN, DEFAULT_RECALL_RANK_CAP


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=["search", "rec"], required=True)
    parser.add_argument("--tag", choices=["easy", "hard"], required=True)
    parser.add_argument("--sample-n", type=int, default=1000)
    parser.add_argument("--max-groups", type=int, default=240)
    parser.add_argument("--example-limit", type=int, default=24)
    args = parser.parse_args()

    os.environ["QILIN_ASYNC_PREWARM"] = "0"
    ctx = AppContext(
        tag=args.tag,
        gbdt_topn=DEFAULT_GBDT_TOPN,
        recall_rank_cap=DEFAULT_RECALL_RANK_CAP,
    )
    use_tag = ctx.resolve_scene_tag(args.scene, explicit_tag=args.tag)
    state = ctx.get_runtime(use_tag).get_pipeline(args.scene).state

    metrics = state.compute_validation_metrics(sample_n=max(2, int(args.sample_n)))
    ctx._save_json_snapshot(
        ctx._metrics_snapshot_path(
            scene=args.scene,
            tag=use_tag,
            sample_n=max(2, int(args.sample_n)),
            include_val=True,
        ),
        metrics,
    )

    validation = state.compute_validation_compare(
        max_groups=max(1, int(args.max_groups)),
        example_limit=max(1, int(args.example_limit)),
        context_user_idx=None,
    )
    ctx._save_json_snapshot(
        ctx._validation_snapshot_path(
            scene=args.scene,
            tag=use_tag,
            max_groups=max(1, int(args.max_groups)),
            example_limit=max(1, int(args.example_limit)),
        ),
        validation,
    )
    print(
        f"snapshot_ready scene={args.scene} tag={use_tag} "
        f"sample_n={int(args.sample_n)} groups={int(args.max_groups)} examples={int(args.example_limit)}"
    )


if __name__ == "__main__":
    main()
