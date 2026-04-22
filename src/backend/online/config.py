"""Online 配置定义与加载。"""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_GBDT_TOPN = 500
DEFAULT_RECALL_RANK_CAP = 1000


@dataclass
class OnlineServiceConfig:
    host: str = "0.0.0.0"
    port: int = 18080
    default_tag: str = "hard"
    gbdt_topn: int = DEFAULT_GBDT_TOPN
    recall_rank_cap: int = DEFAULT_RECALL_RANK_CAP


def load_online_config() -> OnlineServiceConfig:
    return OnlineServiceConfig(
        host=os.getenv("QILIN_ONLINE_HOST", "0.0.0.0"),
        port=int(os.getenv("QILIN_ONLINE_PORT", "18080")),
        default_tag=os.getenv("QILIN_DEFAULT_TAG", "hard"),
        gbdt_topn=int(os.getenv("QILIN_GBDT_TOPN", str(DEFAULT_GBDT_TOPN))),
        recall_rank_cap=int(os.getenv("QILIN_RECALL_RANK_CAP", str(DEFAULT_RECALL_RANK_CAP))),
    )


__all__ = [
    "DEFAULT_GBDT_TOPN",
    "DEFAULT_RECALL_RANK_CAP",
    "OnlineServiceConfig",
    "load_online_config",
]
