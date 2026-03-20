"""在线冷启动候选服务。"""

from __future__ import annotations

import pandas as pd

from backend.online.cold_start.popular import build_hot_candidates


def build_cold_start_candidates(feat_req: pd.DataFrame, topk: int) -> pd.DataFrame:
    # 冷启动时直接走热度候选，避免依赖个性化历史
    return build_hot_candidates(feat_req=feat_req, topk=topk)
