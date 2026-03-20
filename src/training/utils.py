import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

# =============================
# Utils
# =============================
def discretize_relevance(y_cont, n_bins=5): # 0:0, 1:0~20%, 2:20~40%, 3:40~60%, 4:60~80%, 5:80~100%
    # 1. 初始化为 0
    res = np.zeros_like(np.asarray(y_cont), dtype=np.int32)
    # 2. 提取正样本
    pos_mask = y_cont > 0
    pos = y_cont[pos_mask]
    if len(pos) == 0:
        return res
    # 3. 计算正样本的分位数
    qs = np.quantile(pos, np.linspace(0, 1, n_bins + 1))
    qs = np.unique(qs)
    if len(qs) < 2:
        res[pos_mask] = 1 # 如果正样本分值都一样，全部设为 1
    else:
        # 4. 将正样本映射到 1 ~ len(qs)-1 之间
        res[pos_mask] = np.digitize(pos, qs[1:-1]) + 1
    return res

def ndcg_at_k(y_true, y_pred, k=10):
    order = np.argsort(-y_pred)
    y = y_true[order][:k]
    k = min(k, len(y))
    if k == 0:
        return 0.0
    y = y[:k]
    gains = (2 ** y) - 1
    discounts = 1 / np.log2(np.arange(2, k + 2))
    dcg = np.sum(gains * discounts)
    ideal = np.sort(y_true)[::-1][:k]
    idcg = np.sum(((2 ** ideal) - 1) * discounts) + 1e-9
    return float(dcg / idcg)

def eval_ndcg_by_group(y, y_pred, g, k):
    vals = []
    for gv in np.unique(g):
        idx = np.where(g == gv)[0]
        if idx.size <= 1:
            continue
        if y[idx].max() == 0:
            continue
        vals.append(ndcg_at_k(y[idx], y_pred[idx], k))
    return float(np.mean(vals)) if vals else 0.0

def sort_by_group(X, y, g):
    order = np.argsort(g, kind="stable")
    if isinstance(X, pd.DataFrame):
        X_sorted = X.iloc[order].reset_index(drop=True)
    else:
        X_sorted = X[order]
    return X_sorted, y[order], g[order], order

def unsort(pred, order):
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    return pred[inv]

def build_group_sizes(g):
    _, cnt = np.unique(g, return_counts=True)
    return cnt.tolist()

def group_norm_score(df, score_col, group_col):
    def safe_norm(x):
        if len(x) <= 1 or x.std() == 0:
            return np.zeros_like(x)
        return (x - x.mean()) / (x.std() + 1e-9)
    df[score_col] = df.groupby(group_col)[score_col].transform(safe_norm)
    return df

def lambda_rank_loss(scores, labels, groups, k=10):
    device = scores.device
    total_loss = []
    eps = 1e-8
    scores = torch.nan_to_num(scores, nan=0.0, posinf=10.0, neginf=-10.0)

    for qid in torch.unique(groups):
        mask = (groups == qid).nonzero(as_tuple=True)[0]
        if mask.numel() <= 1: continue

        s = scores[mask]
        y = labels[mask]
        if torch.max(y) <= 0: continue

        s = torch.nan_to_num(s, nan=0.0)
        order = torch.argsort(s, descending=True)
        if order.max() >= s.size(0) or order.min() < 0: continue
        s, y = s[order], y[order]

        n = s.size(0)
        rank = torch.arange(n, device=device)
        gains = (2.0 ** y.float()) - 1.0
        discounts = 1.0 / torch.log2(rank.float() + 2.0)
        ideal_order = torch.argsort(y, descending=True)
        ideal_dcg = torch.sum(gains[ideal_order][:k] * discounts[:k]) + eps

        s_diff = s.unsqueeze(1) - s.unsqueeze(0)
        y_diff = y.unsqueeze(1) - y.unsqueeze(0)
        pos_mask = (y_diff > 0)
        
        topk_mask = (rank.unsqueeze(1) < k) | (rank.unsqueeze(0) < k)
        pair_mask = pos_mask & topk_mask
        if not pair_mask.any(): continue

        gain_diff = torch.abs(gains.unsqueeze(1) - gains.unsqueeze(0))
        disc_diff = torch.abs(discounts.unsqueeze(1) - discounts.unsqueeze(0))
        
        # 对 delta_ndcg 进行上限限制，防止梯度爆炸
        delta_ndcg = (gain_diff * disc_diff) / ideal_dcg
        delta_ndcg = torch.clamp(delta_ndcg, min=0.0, max=10.0) 

        # 使用逻辑回归损失形式
        loss = F.binary_cross_entropy_with_logits(s_diff, torch.ones_like(s_diff), reduction='none') * delta_ndcg
        total_loss.append(loss[pair_mask])

    return torch.cat(total_loss).mean() if total_loss else torch.tensor(1e-9, device=device, requires_grad=True)
