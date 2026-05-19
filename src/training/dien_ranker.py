"""
Qilin DIEN Ranker (Search & Rec)
- 兴趣抽取层 (GRU) + 兴趣演化层 (AUGRU) + 注意力机制
- 辅助损失 (Auxiliary Loss): 负样本优先来自曝光未点击集合
- 场景适配: 自动识别 search (包含 query_emb 'float[]', seq_embs 'float[][]') 和 rec 场景 (包含 seq_embs 'float[][]')
- 训练指标：LambdaRank Loss
- 评价指标：NDCG@10
- 日志: 集成 TensorBoard 实时查看 Train / Val 曲线

使用示例:
    uv run python src/training/dien_ranker.py --scene search
    uv run python src/training/dien_ranker.py --scene rec
    uv run python src/training/dien_ranker.py --scene search --train-path outputs/data/dien_search_train_from_gbdt_top500.parquet
    nohup uv run tensorboard --logdir=outputs/logs > /dev/null 2>&1 &   # 默认端口 6006
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import json
import argparse
import gc
from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
try:
    from .utils import *
except ImportError:
    from utils import *

# =============================
# Config
# =============================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

BATCH_SIZE = 1024
TOPK = 10
EPOCHS = 10
EARLY_STOP = 4
VALID_RATIO = 0.2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 2e-3
EMB_DIM = 768
SEQ_LEN = 20
AUX_LOSS_WEIGHT = 0.05

BASE_DIR = Path(__file__).resolve().parents[2]  # Qilin/
FEATURE_DIR = BASE_DIR / "features"
EMB_DIR = BASE_DIR / "embeddings"
OUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = OUT_DIR / "models"
RESULT_DIR = OUT_DIR / "results"
LOG_DIR = OUT_DIR / "logs"


def _group_key(scene: str) -> str:
    return "search_idx" if scene == "search" else "request_idx"


def _default_train_path(scene: str, preranking_topn: int, output_tag: str) -> Path | None:
    tag_suffix = f"_{output_tag}" if output_tag else ""
    p = OUT_DIR / "data" / f"dien_{scene}{tag_suffix}_train_from_gbdt_top{preranking_topn}.parquet"
    if p.exists():
        return p
    if not output_tag:
        cands = sorted(
            (OUT_DIR / "data").glob(f"dien_{scene}_*_train_from_gbdt_top{preranking_topn}.parquet"),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )
        if cands:
            return cands[0]
    return None


def _build_exposure_unclicked_map(scene: str) -> dict[int, list[int]]:
    """
    构建 request/search 级的曝光未点击候选表：
    用于 DIEN 辅助损失的负样本抽样。
    """
    group_key = _group_key(scene)
    feat_path = FEATURE_DIR / f"{scene}_train_features.parquet"
    if not feat_path.exists():
        print(f"[Warn] exposure file not found for aux neg: {feat_path}")
        return {}
    exp = pd.read_parquet(feat_path, columns=[group_key, "note_idx", "click"])
    exp_neg = exp[exp["click"] <= 0][[group_key, "note_idx"]].drop_duplicates()
    mp = (
        exp_neg.groupby(group_key)["note_idx"]
        .apply(lambda x: [int(v) for v in x.tolist()])
        .to_dict()
    )
    return {int(k): v for k, v in mp.items()}


def _downsample_negatives_for_dien(
    df: pd.DataFrame,
    group_key: str,
    max_neg_per_req: int,
    head_neg_keep: int,
    mode: str,
) -> pd.DataFrame:
    """
    训练时降采样每请求负样本：
    - 保留全部正样本
    - hard 模式：按 1(hard):2(easy) 的负样本比例采样（总数上限 max_neg_per_req）
    - easy 模式：仅采样 easy negatives（总数上限 max_neg_per_req）
    """
    if max_neg_per_req <= 0:
        return df

    def _sample_one(g: pd.DataFrame, gid) -> pd.DataFrame:
        # 显式回填分组列，避免后续分组/过滤缺失 group_key。
        if group_key not in g.columns:
            g = g.copy()
            g[group_key] = gid
        pos = g[g["y_multi"] > 0]
        neg = g[g["y_multi"] <= 0]
        if len(neg) <= max_neg_per_req:
            return g

        # hard/easy 划分：优先用 preranking_rank，其次 preranking_score/lgb+xgb
        if "preranking_rank" in g.columns:
            hard_pool = neg.nsmallest(min(len(neg), max(1, head_neg_keep)), "preranking_rank")
            easy_pool = neg.drop(index=hard_pool.index)
        elif "preranking_score" in g.columns:
            hard_pool = neg.nlargest(min(len(neg), max(1, head_neg_keep)), "preranking_score")
            easy_pool = neg.drop(index=hard_pool.index)
        elif "lgb_score" in g.columns and "xgb_score" in g.columns:
            tmp = neg.assign(_preranking=(neg["lgb_score"].to_numpy() + neg["xgb_score"].to_numpy()) * 0.5)
            hard_pool = tmp.nlargest(min(len(tmp), max(1, head_neg_keep)), "_preranking").drop(columns=["_preranking"])
            easy_pool = neg.drop(index=hard_pool.index)
        else:
            hard_pool = neg.iloc[:0]
            easy_pool = neg

        if mode == "hard":
            hard_target = max(1, int(round(max_neg_per_req / 3.0)))
            easy_target = max_neg_per_req - hard_target
        else:
            hard_target = 0
            easy_target = max_neg_per_req

        if hard_target > 0 and len(hard_pool) > hard_target:
            head = hard_pool.sample(n=hard_target, random_state=SEED)
        else:
            head = hard_pool if hard_target > 0 else hard_pool.iloc[:0]

        hard_shortage = max(0, hard_target - len(head))
        need = max(0, easy_target + hard_shortage)
        remain = easy_pool
        if need > 0 and len(remain) > need:
            if pd.notna(gid):
                try:
                    gid_int = int(gid)
                except Exception:
                    gid_int = abs(hash(gid))
            else:
                gid_int = 0
            tail = remain.sample(n=need, random_state=SEED + (gid_int % 1000003))
        else:
            tail = remain

        out = pd.concat([pos, head, tail], axis=0)
        if "preranking_rank" in out.columns:
            out = out.sort_values("preranking_rank", ascending=True, kind="mergesort")
        elif "preranking_score" in out.columns:
            out = out.sort_values("preranking_score", ascending=False, kind="mergesort")
        return out

    sampled_parts: list[pd.DataFrame] = []
    for gid, g in df.groupby(group_key, sort=False):
        sampled_parts.append(_sample_one(g, gid))
    if not sampled_parts:
        return df.iloc[:0].copy()
    sampled = pd.concat(sampled_parts, axis=0, ignore_index=True)

    # 某些 pandas 版本可能在 apply 后丢掉分组列，这里兜底恢复并显式校验。
    if group_key not in sampled.columns:
        sampled = sampled.reset_index()
        if group_key not in sampled.columns and "level_0" in sampled.columns:
            sampled = sampled.rename(columns={"level_0": group_key})
        if group_key not in sampled.columns:
            raise KeyError(
                f"Group key '{group_key}' missing after negative sampling. "
                "Please check pandas groupby.apply behavior."
            )

    return sampled.reset_index(drop=True)


def _fit_apply_zscore_by_train(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    num_feats: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, float]]]:
    """
    按训练折统计量做 z-score，避免使用全量数据统计量造成验证集信息泄露。
    同时返回 zscore_stats，使模型自带归一化，线上推理与离线训练输入分布保持一致。

    Returns:
        (normalized_train_df, normalized_val_df, zscore_stats)
        zscore_stats[feat] = {"mean": float, "std": float}
    """
    tr = train_df.copy()
    val = val_df.copy()
    zscore_stats: dict[str, dict[str, float]] = {}
    for c in num_feats:
        tr_v = tr[c].to_numpy(dtype=np.float32, copy=True)
        mu = float(np.mean(tr_v))
        sd = float(np.std(tr_v))
        if not np.isfinite(sd) or sd < 1e-6:
            tr[c] = 0.0
            val[c] = 0.0
            zscore_stats[c] = {"mean": mu, "std": 0.0}  # std=0 → 线上直接置 0
        else:
            tr[c] = (tr[c].to_numpy(dtype=np.float32, copy=False) - mu) / (sd + 1e-6)
            val[c] = (val[c].to_numpy(dtype=np.float32, copy=False) - mu) / (sd + 1e-6)
            zscore_stats[c] = {"mean": mu, "std": sd}
    return tr, val, zscore_stats


def _group_val_split(df: pd.DataFrame, group_key: str, valid_ratio: float = VALID_RATIO) -> tuple[np.ndarray, np.ndarray]:
    groups = df[group_key].drop_duplicates().to_numpy()
    if len(groups) < 2:
        raise ValueError(f"Not enough groups for val split: group_cnt={len(groups)}")

    rng = np.random.RandomState(SEED)
    shuffled = groups.copy()
    rng.shuffle(shuffled)
    val_cnt = int(round(len(shuffled) * float(valid_ratio)))
    val_cnt = max(1, min(val_cnt, len(shuffled) - 1))
    val_groups = set(shuffled[:val_cnt].tolist())

    val_mask = df[group_key].isin(val_groups).to_numpy()
    val_idx = np.where(val_mask)[0]
    tr_idx = np.where(~val_mask)[0]
    if len(tr_idx) == 0 or len(val_idx) == 0:
        raise ValueError("invalid val split: empty train or valid index")
    return tr_idx, val_idx


# =============================
# Embedding Store
# =============================
class EmbeddingStore:
    def __init__(self, emb_dir, key_col, emb_col, is_sequence=False, prefix=None, use_mmap=True, name=None):
        """
        use_mmap: 如果为 True 且存在 .bin/.json 文件，则使用内存映射，否则回退到 Parquet 加载。
        name: mmap 文件的基本名称 (如 'note_text')
        is_sequence: 如果是用户历史行为序列则为 True (得到 [Batch, T, E])，
                     如果是单个 Item (文本/图片) 则为 False (得到 [Batch, E])。
        """
        self.key_col = key_col
        self.emb_col = emb_col
        self.is_sequence = is_sequence
        self.use_mmap = use_mmap
        self.name = name
        self.bin_path = Path(emb_dir) / f"{name}.bin" if name else None
        self.map_path = Path(emb_dir) / f"{name}_map.json" if name else None
        
        if self.bin_path.exists() and self.map_path.exists():
            self._load_mmap()
        else:
            print(f"⚠️ [Store] MMap files for {name} not found or disabled. Falling back to Parquet...")
            self.use_mmap = False
            self.files = sorted(list(Path(emb_dir).glob(f"{prefix or ''}*.parquet")))
            self.global_emb_dict = {}
            self._load_parquet()

    def _load_mmap(self):
        """加载内存映射索引和二进制文件"""
        print(f"[MMap] Linking {self.name}...")
        with open(self.map_path, 'r') as f:
            self.id_to_pos = json.load(f) # Key 是字符串格式
            
        num_keys = len(self.id_to_pos)

        if self.is_sequence:
            self.shape = (num_keys, SEQ_LEN, EMB_DIM)
        else:
            self.shape = (num_keys, EMB_DIM)
        
        # 以只读模式映射数据
        self.data = np.memmap(self.bin_path, dtype='float16', mode='r', shape=self.shape)
        self.zero_fill = np.zeros(self.shape[1:], dtype='float16')
        print(f"[MMap] Linked {self.name} | Keys: {num_keys} | Shape: {self.shape}")

    def _load_parquet(self):
        """ Parquet 加载逻辑"""
        for f in tqdm(self.files, desc="Loading Parquet"):
            df = pd.read_parquet(f)
            for k, emb in zip(df[self.key_col].values, df[self.emb_col].values):
                if k in self.global_emb_dict: continue
                emb_np = np.array(list(emb) if self.is_sequence else emb, dtype=np.float16)
                if self.is_sequence:
                    if emb_np.shape[0] < SEQ_LEN:
                        pad = np.zeros((SEQ_LEN - emb_np.shape[0], EMB_DIM), dtype=np.float16)
                        emb_np = np.vstack([emb_np, pad])
                    else:
                        emb_np = emb_np[:SEQ_LEN]
                self.global_emb_dict[k] = emb_np

    def get_tensor(self, idxs):
        batch_np = np.zeros((len(idxs),) + self.shape[1:], dtype='float16')
        if self.use_mmap:
            for i, k in enumerate(idxs):
                pos = self.id_to_pos.get(str(k)) # JSON load keys are str
                if pos is not None and 0 <= pos < self.shape[0]:
                    batch_np[i] = self.data[pos]
        else:
            for i, k in enumerate(idxs):
                batch_np[i] = self.global_emb_dict.get(int(k), self.zero_fill)
                
        return torch.from_numpy(batch_np).to(device=DEVICE, dtype=torch.float32)

# ======================
# DIEN Ranker
# ======================
class AUGRUCell(nn.Module): # 兴趣演化层 (Interest Evolution Layer)
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        # 门控计算：Update Gate (u), Reset Gate (r)
        self.linear_u = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.linear_r = nn.Linear(input_dim + hidden_dim, hidden_dim)
        # 候选状态计算：Candidate State (h_tilde)
        self.linear_h = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, x, h_prev, attn_score):
        """
        x: [B, H] (当前时刻 GRU 的兴趣状态)
        h_prev: [B, H] (AUGRU 上一时刻隐藏状态)
        attn_score: [B, 1] (当前时刻与 Target Item 的注意力分数)
        """
        # 拼接当前输入和上一时刻状态
        combined = torch.cat([x, h_prev], dim=1)
        
        # 1. 计算标准 GRU 的门控
        u = torch.sigmoid(self.linear_u(combined))  # Update Gate
        r = torch.sigmoid(self.linear_r(combined))  # Reset Gate
        
        # 2. 计算候选隐藏状态
        combined_reset = torch.cat([x, r * h_prev], dim=1)
        h_tilde = torch.tanh(self.linear_h(combined_reset))
        
        # 3. 使用注意力分数缩放更新门
        u_prime = u * attn_score
        
        # 4. 更新隐藏状态
        h_new = (1 - u_prime) * h_prev + u_prime * h_tilde
        return h_new

class DIENRanker(nn.Module):
    def __init__(self, scene, num_feats, embed_dim=768, hidden_dim=256, mlp_dims=[128, 64], dropout=0.2):
        super().__init__()
        self.scene = scene
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # 1. 数值特征投影
        self.num_feats = num_feats
        self.num_proj = nn.ModuleDict({
            f: nn.Linear(1, 32) for f in self.num_feats
        })
        # z-score 归一化 buffer（不参与梯度，随 checkpoint 持久化）
        # 默认 mean=0 / std=1 即 identity，训练完成后由 set_zscore_stats() 注入真实统计量。
        # 线上推理时模型自带归一化，与离线训练完全一致，无需外部文件。
        for f in self.num_feats:
            self.register_buffer(f"zscore_mean_{f}", torch.zeros(1))
            self.register_buffer(f"zscore_std_{f}",  torch.ones(1))
        
        # 2. 兴趣抽取层 (Interest Extractor Layer)
        # 输入用户历史 20 条 note 的 embedding
        self.gru = nn.GRU(input_size=embed_dim, hidden_size=hidden_dim, batch_first=True)
        
        # 3. 辅助网络 (Auxiliary Network)
        # 用于判别 current_interests 和 next_item 的相似性
        self.aux_net = nn.Sequential(
            nn.Linear(hidden_dim + embed_dim, 100),
            nn.Sigmoid(),
            nn.Linear(100, 50),
            nn.Sigmoid(),
            nn.Linear(50, 1)
        )

        # 4. 注意力层 (双线性注意力机制)
        # 公式: score = h_t * W * e_target
        self.bilinear_W = nn.Parameter(torch.randn(hidden_dim, embed_dim))
        nn.init.xavier_uniform_(self.bilinear_W)
        self.target_transform = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU()
        )
        
        # 5. 兴趣演化层 (AUGRU)
        self.augru_cell = AUGRUCell(input_dim=hidden_dim, hidden_dim=hidden_dim)
        self.ln_h = nn.LayerNorm(hidden_dim)
        
        # 6. MLP 预估层
        # 输入拼接: AUGRU 最终隐藏状态 + Target 文本 Emb + Target 图片 Emb + 数值特征
        input_mlp_dim = hidden_dim + (embed_dim * 2) + (len(num_feats) * 32)
        if scene == "search":
            input_mlp_dim += embed_dim # 加上 query_emb 维度
            
        layers = []
        in_d = input_mlp_dim
        for out_d in mlp_dims:
            layers.append(nn.Linear(in_d, out_d))
            layers.append(nn.LayerNorm(out_d))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_d = out_d
        layers.append(nn.Linear(in_d, 1))
        self.mlp = nn.Sequential(*layers)
        
        # 树模型融合参数
        #self.alpha = nn.Parameter(torch.tensor(0.1))
        #self.beta = nn.Parameter(torch.tensor(0.1))

    def set_zscore_stats(self, stats: dict[str, dict[str, float]]) -> None:
        """训练完成后注入 z-score 统计量，使 checkpoint 自带归一化，消除 Training-Serving Skew。"""
        for f, stat in stats.items():
            if not hasattr(self, f"zscore_mean_{f}"):
                continue
            getattr(self, f"zscore_mean_{f}").fill_(float(stat["mean"]))
            sd = float(stat["std"])
            getattr(self, f"zscore_std_{f}").fill_(sd if sd >= 1e-6 else 0.0)

    def forward(self, features, batch_y=None, batch_g=None):
        # A. 准备数据
        seq_embs = features["seq_embs"]         # [B, T, E] (假设已 flip 为时间正序)
        note_txt = features["note_text_emb"]    # [B, E]
        note_img = features["note_img_emb"]     # [B, E]
        # 图像 embedding 缺失时退化为文本 embedding，避免大量全零向量影响学习稳定性。
        img_missing = (note_img.abs().sum(dim=1, keepdim=True) <= 1e-12)
        note_img = torch.where(img_missing, note_txt, note_img)
        batch_size, seq_len, _ = seq_embs.shape

        # B. 兴趣抽取层 (Interest Extractor)
        # interest_states: [B, T, H]
        interest_states, _ = self.gru(seq_embs)
        
        # C. 计算辅助损失 (Auxiliary Loss)
        aux_loss = torch.tensor(0.0, device=DEVICE, dtype=torch.float32)
        if seq_len > 1:
            # 1. 构造正样本对 (Pair)
            current_interests = interest_states[:, :-1, :]   # [B, T-1, H]
            pos_next = seq_embs[:, 1:, :]           # [B, T-1, E]
            # 2. 构造负样本对 (In-group Negative Sampling)
            if "aux_neg_note_text_emb" in features:
                # 使用曝光未点击候选抽样得到的 hard negatives（同请求）
                neg_next = features["aux_neg_note_text_emb"].unsqueeze(1).expand(-1, seq_len - 1, -1)
            elif batch_y is not None and batch_g is not None:
                # 兜底：组内负样本抽样
                group_mask = (batch_g.unsqueeze(0) == batch_g.unsqueeze(1)) # [B, B]
                label_mask = (batch_y <= 0).unsqueeze(0).expand(batch_size, batch_size) # [B, B]
                valid_neg_mask = group_mask & label_mask # [B, B]
                prob_mask = valid_neg_mask.float()
                prob_mask[:, 0] += 1e-9 
                neg_indices = torch.multinomial(prob_mask, 1).squeeze()
                neg_next = features["note_text_emb"][neg_indices].unsqueeze(1).expand(-1, seq_len-1, -1)
            else:
                neg_next = pos_next[torch.randperm(batch_size)]   # [B, T-1, E]
            # 3. Aux Loss 掩码
            valid_mask = (pos_next.abs().sum(dim=-1) > 1e-6).float().unsqueeze(-1)  # [B, T-1, 1]
            # 4. 计算 Logits
            pos_logits = self.aux_net(torch.cat([current_interests.detach(), pos_next], dim=-1))
            neg_logits = self.aux_net(torch.cat([current_interests.detach(), neg_next], dim=-1))
            # 5. Masked BCE Loss
            pos_loss = F.binary_cross_entropy_with_logits(
                pos_logits, torch.ones_like(pos_logits), reduction="none"
            )
            neg_loss = F.binary_cross_entropy_with_logits(
                neg_logits, torch.zeros_like(neg_logits), reduction="none"
            )
            aux_loss = ((pos_loss + neg_loss) * valid_mask).sum() / (valid_mask.sum() * 2 + 1e-9)
        
        # D. 注意力计算 (Bilinear Attention)
        # 1. 计算 h_t * W -> [B, T, E]
        h_W = torch.matmul(interest_states, self.bilinear_W) 
        # 2. MLP 融合 txt_emb 和 img_emb
        target_emb = self.target_transform(torch.cat([note_txt, note_img], dim=-1))
        # 3. 计算 (h_t * W) 和 e_target 的点积 -> [B, T]
        attn_scores = torch.sum(h_W * target_emb.unsqueeze(1), dim=2) / np.sqrt(self.hidden_dim + 1e-9)
        # 4. Softmax 归一化
        seq_mask = (seq_embs.abs().sum(dim=-1) > 0)
        attn_scores = torch.clamp(F.softmax(attn_scores.masked_fill(~seq_mask, -1e9), dim=1), 1e-6, 1.0) # [B, T]

        # E. 兴趣演化层 (Interest Evolution - AUGRU)
        h_t = torch.zeros(batch_size, self.hidden_dim).to(DEVICE)
        for t in range(seq_len):
            current_interests = interest_states[:, t, :] # [B, H]
            current_score = attn_scores[:, t].unsqueeze(1) # [B, 1]
            h_t = self.augru_cell(current_interests, h_t, current_score)
            h_t = self.ln_h(h_t)

        # F. 特征拼接与 MLP
        # num_feats 先经模型内置 z-score 归一化（buffer 来自训练统计量），再投影。
        # 训练期间 buffer 为 (0, 1) → identity；re-save 后线上与离线输入分布一致。
        feat_list = [h_t, note_txt, note_img]
        for f in self.num_feats:
            x = features[f]  # [B, 1]
            sd = getattr(self, f"zscore_std_{f}")
            if sd.item() >= 1e-6:
                x = (x - getattr(self, f"zscore_mean_{f}")) / (sd + 1e-6)
            else:
                x = torch.zeros_like(x)
            feat_list.append(self.num_proj[f](x))
            
        if self.scene == "search":
            feat_list.append(features["query_emb"])
            
        mlp_in = torch.cat(feat_list, dim=1)
        deep_score = self.mlp(mlp_in).squeeze(-1)
        
        # G. 最终融合得分
        #final_score = deep_score + self.alpha * features["lgb_score"] + self.beta * features["xgb_score"]
        return deep_score, aux_loss

# ======================
# Train Fold
# ======================
def batch_generator(df, stores, num_feats, scene, group_key, batch_size, aux_neg_map: dict[int, list[int]] | None = None):
    df = df.reset_index(drop=True)

    def _build_batch(batch_df: pd.DataFrame):
        batch = {"feats": {}, "y": None, "g": None}

        # 数值特征
        for f in num_feats:
            batch["feats"][f] = torch.tensor(batch_df[f].values, dtype=torch.float32, device=DEVICE).unsqueeze(-1)

        # Embedding
        batch["feats"]["seq_embs"] = torch.flip(stores['s'].get_tensor(batch_df[group_key].values), dims=[1])   # flip 成时间正序
        batch["feats"]["note_text_emb"] = stores['t'].get_tensor(batch_df["note_idx"].values)
        batch["feats"]["note_img_emb"] = stores['i'].get_tensor(batch_df["note_idx"].values)

        if scene == "search":
            batch["feats"]["query_emb"] = stores['q'].get_tensor(batch_df[group_key].values)

        # 辅助损失负样本：优先使用“曝光未点击”集合中同请求样本
        if aux_neg_map is not None:
            req_ids = batch_df[group_key].to_numpy()
            batch_note_ids = batch_df["note_idx"].to_numpy(dtype=np.int64, copy=False)
            fallback_neg_ids = batch_df.loc[batch_df["y_multi"] <= 0, "note_idx"].to_numpy(dtype=np.int64, copy=False)
            if fallback_neg_ids.size == 0:
                fallback_neg_ids = batch_note_ids
            aux_neg_note_ids = np.empty(len(batch_df), dtype=np.int64)
            for i, rid in enumerate(req_ids):
                cands = aux_neg_map.get(int(rid), [])
                if cands:
                    aux_neg_note_ids[i] = int(cands[np.random.randint(len(cands))])
                else:
                    aux_neg_note_ids[i] = int(fallback_neg_ids[np.random.randint(len(fallback_neg_ids))])
            batch["feats"]["aux_neg_note_text_emb"] = stores["t"].get_tensor(aux_neg_note_ids)

        # 标签使用连续收益 y_multi，避免离散化带来的信息损失
        batch["y"] = torch.tensor(batch_df["y_multi"].values, dtype=torch.float32, device=DEVICE)
        batch["g"] = torch.tensor(batch_df[group_key].values, dtype=torch.int32, device=DEVICE)

        yield batch

    # 组感知批次：尽量保证同一请求样本不被切碎，提升 LambdaRank 有效梯度
    cache = []
    cache_rows = 0
    for _, g in df.groupby(group_key, sort=False):
        g_rows = len(g)
        if cache_rows > 0 and (cache_rows + g_rows > batch_size):
            batch_df = pd.concat(cache, axis=0, ignore_index=True)
            yield from _build_batch(batch_df)
            cache = []
            cache_rows = 0
        cache.append(g)
        cache_rows += g_rows

    if cache_rows > 0:
        batch_df = pd.concat(cache, axis=0, ignore_index=True)
        yield from _build_batch(batch_df)

def train_fold(
    scene,
    train_df,
    val_df,
    stores,
    num_feats,
    writer,
    aux_loss_weight: float,
    max_epochs: int,
    early_stop_rounds: int,
    output_tag: str,
    aux_neg_map: dict[int, list[int]] | None,
):
    model = DIENRanker(scene=scene, num_feats=num_feats).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", patience=2, factor=0.5)
    group_key = _group_key(scene)

    best_ndcg, best_pred, no_improve = -1, None, 0
    tag_suffix = f"_{output_tag}" if output_tag else ""
    model_ckpt = MODEL_DIR / f"dien_{scene}{tag_suffix}.pt"

    for epoch in range(max_epochs):
        # ====================== TRAIN ======================
        model.train()
        tr_met = {"total_loss": [], "main_loss": [], "aux_loss": [], "ndcg@10": []}
        all_train_y, all_train_p, all_train_g = [], [], []
            
        for batch in tqdm(batch_generator(train_df, stores, num_feats, scene, group_key, BATCH_SIZE, aux_neg_map=aux_neg_map), total=(len(train_df) + BATCH_SIZE - 1) // BATCH_SIZE, desc=f"Epoch {epoch+1} Train"):
            opt.zero_grad()
            # 模型返回的原始 Loss
            pred, aux_loss = model(batch["feats"], batch_y=batch["y"], batch_g=batch["g"])
            main_loss = lambda_rank_loss(pred, batch["y"], batch["g"])
            total_loss = main_loss + (aux_loss_weight * aux_loss)
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            
            tr_met["total_loss"].append(total_loss.item())
            tr_met["main_loss"].append(main_loss.item())
            tr_met["aux_loss"].append(aux_loss.item())
            all_train_y.append(batch["y"].detach().cpu().numpy().reshape(-1))
            all_train_p.append(pred.detach().cpu().numpy().reshape(-1))
            all_train_g.append(batch["g"].detach().cpu().numpy().reshape(-1))
            
        tr_met["ndcg@10"].append(eval_ndcg_by_group(np.concatenate(all_train_y), np.concatenate(all_train_p), np.concatenate(all_train_g), TOPK))

        # ====================== VALIDATION ======================
        model.eval()
        val_met = {"total_loss": [], "main_loss": [], "aux_loss": []}
        all_val_y, all_val_p, all_val_g = [], [], []
        
        with torch.no_grad():
            for batch in tqdm(batch_generator(val_df, stores, num_feats, scene, group_key, BATCH_SIZE, aux_neg_map=aux_neg_map), total=(len(val_df) + BATCH_SIZE - 1) // BATCH_SIZE, desc=f"Epoch {epoch+1} Val"):
                pred, aux_loss = model(batch["feats"], batch_y=batch["y"], batch_g=batch["g"])
                main_loss = lambda_rank_loss(pred, batch["y"], batch["g"])
                total_loss = main_loss + (aux_loss_weight * aux_loss)
                
                val_met["total_loss"].append(total_loss.item())
                val_met["main_loss"].append(main_loss.item())
                val_met["aux_loss"].append(aux_loss.item())
                all_val_y.append(batch["y"].detach().cpu().numpy().reshape(-1))
                all_val_p.append(pred.detach().cpu().numpy().reshape(-1))
                all_val_g.append(batch["g"].detach().cpu().numpy().reshape(-1))

            cur_val_p = np.concatenate(all_val_p)
            val_y = np.concatenate(all_val_y)
            val_g = np.concatenate(all_val_g)
            val_ndcg = eval_ndcg_by_group(val_y, cur_val_p, val_g, TOPK)
            sch.step(val_ndcg)
        current_lr = opt.param_groups[0]['lr']
        
        print(f"\n[Epoch {epoch+1}]")
        print(f"  Train -> TotalLoss: {np.mean(tr_met['total_loss']):.4f} | MainLoss: {np.mean(tr_met['main_loss']):.4f} | NDCG@10: {np.mean(tr_met['ndcg@10']):.4f}")
        print(
            f"  Val   -> TotalLoss: {np.mean(val_met['total_loss']):.4f} | MainLoss: {np.mean(val_met['main_loss']):.4f} "
            f"| NDCG@10: {val_ndcg:.4f}"
        )
        print(
            f"  LR    -> {current_lr:.4f} | AuxLoss: {np.mean(tr_met['aux_loss']):.4f}/{np.mean(val_met['aux_loss']):.4f}"
        )

        # ====================== TB LOGGING ======================
        # 在同一个 Chart 中记录 Train/Val 指标
        writer.add_scalars("Total_Loss", {
            "train": np.mean(tr_met["total_loss"]),
            "valid": np.mean(val_met["total_loss"])
        }, epoch)
        
        writer.add_scalars("Main_Loss", {
            "train": np.mean(tr_met["main_loss"]),
            "valid": np.mean(val_met["main_loss"])
        }, epoch)
        
        writer.add_scalars("NDCG10", {
            "train": np.mean(tr_met["ndcg@10"]),
            "valid": val_ndcg,
        }, epoch)

        if val_ndcg > best_ndcg:
            best_ndcg = val_ndcg
            best_pred = cur_val_p
            torch.save(model.state_dict(), model_ckpt)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= early_stop_rounds:
                break
    final_summary = (
        f"### 🏆 Final Best Results\n"
        f"- **Best NDCG@10**: `{best_ndcg:.4f}`\n"
        f"- **Aux Weight**: `{aux_loss_weight:.4f}`"
    )
    writer.add_text("Best_Results", final_summary, 0)

    del model, opt, sch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return best_ndcg, best_pred


def _score_frame(
    model: DIENRanker,
    df: pd.DataFrame,
    stores,
    num_feats: list[str],
    scene: str,
    group_key: str,
) -> np.ndarray:
    model.eval()
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for batch in tqdm(
            batch_generator(df, stores, num_feats, scene, group_key, BATCH_SIZE, aux_neg_map=None),
            total=(len(df) + BATCH_SIZE - 1) // BATCH_SIZE,
            desc="Score Full Train Candidates",
        ):
            pred, _aux = model(batch["feats"])
            preds.append(pred.detach().cpu().numpy().reshape(-1).astype(np.float32, copy=False))
    if not preds:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(preds, axis=0)


def _groupwise_minmax_score(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    out = np.zeros_like(arr, dtype=np.float32)
    if arr.size == 0:
        return out
    g = np.asarray(groups)
    for gid in pd.unique(g):
        mask = g == gid
        vals = arr[mask]
        if vals.size == 0:
            continue
        vmin = float(np.min(vals))
        vmax = float(np.max(vals))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-6:
            out[mask] = 0.0
        else:
            out[mask] = (vals - vmin) / (vmax - vmin + 1e-6)
    return out


def _binary_auc_local(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.int8)
    s = np.asarray(y_score, dtype=np.float64)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if pos == 0 or neg == 0:
        return 0.5
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    pos_rank_sum = float(ranks[y == 1].sum())
    auc = (pos_rank_sum - pos * (pos + 1) / 2.0) / float(pos * neg)
    return float(max(0.0, min(1.0, auc)))


def _choose_fused_scores(
    val_df: pd.DataFrame,
    raw_dien_scores: np.ndarray,
    group_key: str,
) -> tuple[np.ndarray, float, float]:
    if "preranking_score" not in val_df.columns or len(val_df) != len(raw_dien_scores):
        return np.asarray(raw_dien_scores, dtype=np.float32), 1.0, 0.0
    groups = val_df[group_key].to_numpy()
    y = pd.to_numeric(val_df["y_multi"], errors="coerce").fillna(0.0).to_numpy(np.float32)
    y_disc = discretize_relevance(y)
    pre = pd.to_numeric(val_df["preranking_score"], errors="coerce").fillna(0.0).to_numpy(np.float32)
    pre_norm = _groupwise_minmax_score(pre, groups)
    dien_norm = _groupwise_minmax_score(np.asarray(raw_dien_scores, dtype=np.float32), groups)

    best_w = 1.0
    best_score = -1.0
    best_auc = -1.0
    best_fused = dien_norm.copy()
    for w in np.linspace(0.0, 1.0, 21, dtype=np.float32):
        fused = (1.0 - float(w)) * pre_norm + float(w) * dien_norm
        ndcg = float(eval_ndcg_by_group(y_disc, fused, groups, TOPK))
        auc = float(_binary_auc_local((y > 0).astype(np.int8), fused)) if len(np.unique((y > 0).astype(np.int8))) > 1 else 0.5
        if (ndcg > best_score + 1e-9) or (abs(ndcg - best_score) <= 1e-9 and auc > best_auc):
            best_w = float(w)
            best_score = ndcg
            best_auc = auc
            best_fused = fused.astype(np.float32, copy=False)
    return best_fused, best_w, best_score


def _export_scored_full_frame(
    scene: str,
    output_tag: str,
    full_df: pd.DataFrame,
    dien_scores: np.ndarray,
    raw_dien_scores: np.ndarray | None,
    group_key: str,
) -> Path:
    out = full_df.copy()
    out["dien_score"] = np.nan_to_num(dien_scores.astype(np.float32, copy=False), nan=0.0, posinf=1e6, neginf=-1e6)
    if raw_dien_scores is not None:
        out["raw_dien_score"] = np.nan_to_num(np.asarray(raw_dien_scores, dtype=np.float32), nan=0.0, posinf=1e6, neginf=-1e6)
    out = out.sort_values([group_key, "dien_score", "note_idx"], ascending=[True, False, True], kind="mergesort").copy()
    out["ranking_rank"] = out.groupby(group_key).cumcount().astype(np.int32) + 1

    keep_cols = [
        group_key,
        "user_idx",
        "note_idx",
        "click",
        "y_multi",
        "preranking_score",
        "preranking_rank",
        "lgb_score",
        "xgb_score",
        "dien_score",
        "raw_dien_score",
        "ranking_rank",
    ]
    keep_cols = [c for c in keep_cols if c in out.columns]
    tag_suffix = f"_{output_tag}" if output_tag else ""
    out_path = OUT_DIR / "data" / f"dien_{scene}{tag_suffix}_train_scored_full.parquet"
    out[keep_cols].to_parquet(out_path, index=False)
    print(f"[Export] DIEN full scored set: {out_path}, shape={out[keep_cols].shape}")
    return out_path

def main(
    scene,
    train_path: Path | None,
    output_tag: str,
    preranking_topn: int,
    train_max_neg_per_req: int,
    train_head_neg_keep: int,
    aux_loss_weight: float,
    max_epochs: int,
    early_stop_rounds: int,
    export_only: bool = False,
):
    group_key = _group_key(scene)
    if train_path is None:
        train_path = _default_train_path(scene, preranking_topn, output_tag)
    if train_path is None or not train_path.exists():
        raise FileNotFoundError(
            f"Missing DIEN training set for scene={scene}, tag={output_tag}. "
            "Please run GBDT stage first to export dien_*_train_from_gbdt_topN.parquet."
        )
    print(f"[Load] serial offline train set from preranking rank: {train_path}")

    if not output_tag:
        p = str(train_path).lower()
        if "_easy_" in p:
            output_tag = "easy"
        elif "_hard_" in p:
            output_tag = "hard"

    raw_df = pd.read_parquet(train_path)
    df = raw_df.copy()
    drop_feats = ["click", "y_multi", "note_idx", group_key, "recent_clicked_note_idxs", "first_route"]
    candidate_num_feats = [c for c in raw_df.columns if c not in drop_feats]
    num_feats = [c for c in candidate_num_feats if pd.api.types.is_numeric_dtype(raw_df[c])]
    raw_df[num_feats] = raw_df[num_feats].fillna(0.0).replace([np.inf, -np.inf], 0.0)
    print(f"原始样本数: {len(df)}")

    train_mode = output_tag if output_tag in {"easy", "hard"} else "easy"
    if train_max_neg_per_req > 0:
        before_rows = len(df)
        df = _downsample_negatives_for_dien(
            df=df,
            group_key=group_key,
            max_neg_per_req=train_max_neg_per_req,
            head_neg_keep=train_head_neg_keep,
            mode=train_mode,
        )
        print(
            f"[Train Sampling] max_neg_per_req={train_max_neg_per_req}, "
            f"head_neg_keep={train_head_neg_keep}, mode={train_mode}, rows: {before_rows} -> {len(df)}"
        )
    
    # 基础特征处理
    drop_feats = ["click", "y_multi", "note_idx", group_key, "recent_clicked_note_idxs", "first_route"]
    candidate_num_feats = [c for c in df.columns if c not in drop_feats]
    num_feats = [c for c in candidate_num_feats if pd.api.types.is_numeric_dtype(df[c])]
    df[num_feats] = df[num_feats].fillna(0.0).replace([np.inf, -np.inf], 0.0)

    before_groups = int(df[group_key].nunique())
    # 仅保留有效训练组：组内至少2条样本且存在正反馈
    df = df.groupby(group_key).filter(lambda x: len(x) > 1 and float(x["y_multi"].max()) > 0.0).reset_index(drop=True)
    after_groups = int(df[group_key].nunique())
    print(f"[Group Filter] groups: {before_groups} -> {after_groups}, rows={len(df)}")
    if len(df) == 0:
        raise ValueError("No training rows left after group filtering.")
    eval_df = raw_df.groupby(group_key).filter(lambda x: len(x) > 1 and float(x["y_multi"].max()) > 0.0).reset_index(drop=True)
    if len(eval_df) == 0:
        raise ValueError("No evaluation rows left after group filtering.")
    oof_dien = np.zeros(len(df), dtype=np.float32)

    # Embedding Stores
    stores = {
        's': EmbeddingStore(EMB_DIR / "query_text_emb", key_col=group_key, emb_col="seq_embs", prefix=scene, is_sequence=True, name=f"{scene}_seq"),
        't': EmbeddingStore(EMB_DIR / "note_text_emb", key_col="note_idx", emb_col="note_text_emb", name="note_text"),
        'i': EmbeddingStore(EMB_DIR / "note_img_emb", key_col="note_idx", emb_col="note_img_emb", name="note_img")
    }
    if scene == "search":
        stores['q'] = EmbeddingStore(EMB_DIR / "query_text_emb", key_col=group_key, emb_col="query_emb", prefix="search", name="search_query")

    aux_neg_map = _build_exposure_unclicked_map(scene)
    req_cov = float(np.mean([int(r) in aux_neg_map for r in df[group_key].drop_duplicates().tolist()])) if len(df) > 0 else 0.0
    print(f"[Aux Neg] exposure-unclicked map ready, req_coverage={req_cov:.4f}")

    tag_suffix = f"_{output_tag}" if output_tag else ""
    ckpt_path = MODEL_DIR / f"dien_{scene}{tag_suffix}.pt"
    if export_only:
        if not ckpt_path.exists():
            raise FileNotFoundError(f"checkpoint not found for export-only mode: {ckpt_path}")
        _state = torch.load(ckpt_path, map_location=DEVICE)
        _tmp_model = DIENRanker(scene=scene, num_feats=num_feats).to(DEVICE)
        _tmp_model.load_state_dict(_state)
        full_raw_scores = _score_frame(_tmp_model, eval_df, stores, num_feats, scene, group_key)
        val_groups = set(df.iloc[_group_val_split(df=df, group_key=group_key, valid_ratio=VALID_RATIO)[1]][group_key].astype(int).tolist())
        val_eval = eval_df[eval_df[group_key].isin(val_groups)].copy()
        val_raw = full_raw_scores[eval_df[group_key].isin(val_groups).to_numpy()]
        _fused_val, best_w, best_ndcg = _choose_fused_scores(val_eval, val_raw, group_key)
        pre_norm = _groupwise_minmax_score(pd.to_numeric(eval_df["preranking_score"], errors="coerce").fillna(0.0).to_numpy(np.float32), eval_df[group_key].to_numpy())
        dien_norm = _groupwise_minmax_score(full_raw_scores, eval_df[group_key].to_numpy())
        full_scores = ((1.0 - best_w) * pre_norm + best_w * dien_norm).astype(np.float32, copy=False)
        print(f"[Blend] export_only best_w={best_w:.2f}, val_ndcg10={best_ndcg:.6f}")
        _export_scored_full_frame(scene, output_tag, eval_df, full_scores, full_raw_scores, group_key)
        np.save(RESULT_DIR / f"dien_{scene}{tag_suffix}.npy", full_scores)
        del _tmp_model, _state, full_raw_scores, full_scores, val_eval, val_raw, _fused_val, pre_norm, dien_norm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return

    # TB Writer
    log_dir = LOG_DIR / f"dien_{scene}{tag_suffix}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(log_dir=str(log_dir))

    tr_idx, val_idx = _group_val_split(df=df, group_key=group_key, valid_ratio=VALID_RATIO)
    print(f"[Split] val by group: train={len(tr_idx)}, valid={len(val_idx)}, valid_ratio={VALID_RATIO}")

    tr_fold_df = df.iloc[tr_idx].copy()
    val_fold_df = df.iloc[val_idx].copy()
    tr_fold_df, val_fold_df, zscore_stats = _fit_apply_zscore_by_train(tr_fold_df, val_fold_df, num_feats)

    score, val_pred = train_fold(
        scene,
        tr_fold_df,
        val_fold_df,
        stores,
        num_feats,
        writer,
        aux_loss_weight,
        max_epochs,
        early_stop_rounds,
        output_tag,
        aux_neg_map,
    )
    oof_dien[val_idx] = val_pred
    # 将 z-score 统计量注入已保存的 checkpoint，模型自带归一化
    ckpt_path = MODEL_DIR / f"dien_{scene}{tag_suffix}.pt"
    if ckpt_path.exists():
        _state = torch.load(ckpt_path, map_location="cpu")
        _tmp_model = DIENRanker(scene=scene, num_feats=num_feats)
        _tmp_model.load_state_dict(_state)
        _tmp_model.set_zscore_stats(zscore_stats)
        torch.save(_tmp_model.state_dict(), ckpt_path)
        del _tmp_model, _state
        print(f"[ZScore] 统计量已注入 checkpoint ({len(zscore_stats)} 个特征) → {ckpt_path}")

    print(f"[DIEN Val] Best NDCG: {score:.6f}\n")
    del tr_fold_df, val_fold_df, val_pred
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    blend_weight = 1.0
    blend_ndcg = float(score)
    if ckpt_path.exists():
        _state = torch.load(ckpt_path, map_location=DEVICE)
        _tmp_model = DIENRanker(scene=scene, num_feats=num_feats).to(DEVICE)
        _tmp_model.load_state_dict(_state)
        val_groups = set(val_fold_df[group_key].astype(int).tolist())
        val_eval = eval_df[eval_df[group_key].isin(val_groups)].copy()
        val_raw_scores = _score_frame(_tmp_model, val_eval, stores, num_feats, scene, group_key)
        _fused_val, blend_weight, blend_ndcg = _choose_fused_scores(val_eval, val_raw_scores, group_key)
        full_raw_scores = _score_frame(_tmp_model, eval_df, stores, num_feats, scene, group_key)
        pre_norm = _groupwise_minmax_score(
            pd.to_numeric(eval_df["preranking_score"], errors="coerce").fillna(0.0).to_numpy(np.float32),
            eval_df[group_key].to_numpy(),
        )
        dien_norm = _groupwise_minmax_score(full_raw_scores, eval_df[group_key].to_numpy())
        full_scores = ((1.0 - blend_weight) * pre_norm + blend_weight * dien_norm).astype(np.float32, copy=False)
        print(f"[Blend] best_w={blend_weight:.2f}, val_ndcg10={blend_ndcg:.6f}")
        _export_scored_full_frame(scene, output_tag, eval_df, full_scores, full_raw_scores, group_key)
        np.save(RESULT_DIR / f"dien_{scene}{tag_suffix}.npy", full_scores)
        del _tmp_model, _state, val_eval, val_raw_scores, _fused_val, full_raw_scores, full_scores, pre_norm, dien_norm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    else:
        np.save(RESULT_DIR / f"dien_{scene}{tag_suffix}.npy", oof_dien)
    best_score = float(score)

    writer.add_text(
        "Final/Mean_NDCG10",
        f"{best_score:.6f}",
    )
    writer.add_text(
        "Final/Best_Result",
        f"val_ndcg10={best_score:.6f}",
    )
    writer.add_text(
        "Final/Blend",
        f"best_w={blend_weight:.2f}, fused_val_ndcg10={blend_ndcg:.6f}",
    )
    writer.close()


# =============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True, choices=["search", "rec"])
    parser.add_argument("--output-tag", type=str, default="", help="输出文件标签（例如 easy/hard），为空则不加后缀")
    parser.add_argument("--train-path", type=Path, default=None, help="精排训练集路径（通常来自 gbdt 导出的 topN）")
    parser.add_argument("--preranking-topn", dest="preranking_topn", type=int, default=500, help="当未指定 train-path 时，读取默认 preranking topN 文件")
    parser.add_argument(
        "--train-max-neg-per-req",
        type=int,
        default=0,
        help="DIEN 训练时每请求最多保留负样本数（默认0，表示不做下采样，保持与线上候选分布一致）",
    )
    parser.add_argument("--train-head-neg-keep", type=int, default=7, help="DIEN hard 模式下 hard negative 候选池大小")
    parser.add_argument("--aux-loss-weight", type=float, default=AUX_LOSS_WEIGHT, help="辅助损失权重")
    parser.add_argument("--max-epochs", type=int, default=EPOCHS, help="最大训练轮数")
    parser.add_argument("--early-stop-rounds", type=int, default=EARLY_STOP, help="早停轮数")
    parser.add_argument("--export-only", action="store_true", help="涓嶉噸鏂拌缁冿紝浠呯敤宸叉湁 checkpoint 鍥炲埛鍏ㄩ噺 DIEN 鎵撳垎浜х墿")
    args = parser.parse_args()
    main(
        args.scene,
        args.train_path,
        args.output_tag,
        args.preranking_topn,
        args.train_max_neg_per_req,
        args.train_head_neg_keep,
        args.aux_loss_weight,
        args.max_epochs,
        args.early_stop_rounds,
        args.export_only,
    )
