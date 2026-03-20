import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import json
from pathlib import Path
from datetime import datetime
from collections import OrderedDict
from sklearn.model_selection import GroupKFold
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
try:
    from .utils import *
except ImportError:
    from utils import *

# =============================
# 配置
# =============================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

BATCH_SIZE = 512
N_SPLITS = 5
TOPK = 10
EPOCHS = 1000
EARLY_STOP = 10
PLOT = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 2e-3
SHARD_SIZE = 10000
EMB_DIM = 768

BASE_DIR = Path(__file__).resolve().parents[2]
FEATURE_DIR = BASE_DIR / "features"
EMB_DIR = BASE_DIR / "embeddings"
OUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = OUT_DIR / "models"
RESULT_DIR = OUT_DIR / "results"
PLOT_DIR = OUT_DIR / "plots"

MODEL_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

# =============================
# Shard 缓存（LRU）
# =============================
class ShardCache:
    def __init__(self, max_shards=3):
        self.cache = OrderedDict()
        self.max_shards = max_shards

    def get(self, path):
        if path in self.cache:
            self.cache.move_to_end(path)
            return self.cache[path]
        df = pd.read_parquet(path)
        self.cache[path] = df
        if len(self.cache) > self.max_shards:
            self.cache.popitem(last=False)
        return df

# =============================
# Embedding Store with Shard
# =============================
class EmbeddingStore:
    def __init__(self, emb_dir, key_col, emb_col, prefix=None, shard_size=SHARD_SIZE):
        files = list(emb_dir.glob("*.parquet"))
        if prefix:
            files = [p for p in files if p.name.startswith(prefix)]
        self.files = sorted(files)
        self.key_col = key_col
        self.emb_col = emb_col
        self.cache = ShardCache()
        self.shard_size = shard_size
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.future_next_shard = None
        self.global_emb_dict = {}   # 全局 key -> embedding dict（O(1) lookup）
        self._build_index()

    def _build_index(self):
        """遍历所有 shard 构建 key -> embedding 映射"""
        for f in self.files:
            df = pd.read_parquet(f)
            for k, emb in zip(df[self.key_col].values, df[self.emb_col].values):
                if k in self.global_emb_dict:
                    # 相同 key 在不同 shard 出现，保持第一次出现，保证 group consistency
                    continue
                self.global_emb_dict[k] = emb.astype(np.float16)

    def _load_shard(self, shard_id):
        """从 cache 或磁盘加载 shard"""
        if shard_id >= len(self.files):
            return None
        return self.cache.get(self.files[shard_id])

    def prefetch_shard_by_idx(self, idxs):
        """异步预取可能需要的 shard"""
        if len(idxs) == 0:
            return
        shard_ids = set([i // self.shard_size for i in idxs])
        for shard_id in shard_ids:
            if self.future_next_shard is None:
                self.future_next_shard = self.executor.submit(self._load_shard, shard_id)

    def get_tensor_by_indices(self, idxs):
        """
        batch lookup：
        - idxs: list of key_col
        - 返回 torch.Tensor(batch_size, EMB_DIM) 到 DEVICE
        """
        batch = np.zeros((len(idxs), EMB_DIM), dtype=np.float16)
        shard_map = {}
        for i, idx in enumerate(idxs):
            emb = self.global_emb_dict.get(idx)
            if emb is not None:
                batch[i] = emb
        return torch.tensor(batch, dtype=torch.float16, device=DEVICE)

# ======================
# Dataset
# ======================
class SearchDataset:
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def get_shard_indices(self, shard_size=SHARD_SIZE):
        total = len(self.df)
        num_shards = (total + shard_size - 1) // shard_size
        shards = []
        for shard_id in range(num_shards):
            start = shard_id * shard_size
            end = min(start + shard_size, total)
            shards.append((shard_id, self.df.iloc[start:end]))
        return shards

# ======================
# DCNv2 Layer
# ======================
class DCNv2Layer(nn.Module):
    def __init__(self, input_dim, low_rank):
        super().__init__()
        # 低秩 factorization 权重
        self.w1 = nn.Parameter(torch.randn(input_dim, low_rank) * 0.02)
        self.w2 = nn.Parameter(torch.randn(low_rank, input_dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x0, xl):
        """
        x0: 原始输入 (batch, dim)
        xl: 上一层输出 (batch, dim)
        """
        # DCNv2 低秩交叉: xl+1 = xl + (x0 * (xl @ w1) @ w2 + b)
        cross = torch.mul(x0, xl @ self.w1 @ self.w2) + self.bias
        return xl + cross

# ======================
# DCNv2 Ranker (Stacked)
# ======================
class DCNv2Ranker(nn.Module):
    def __init__(self, num_feats, embed_dim=64, cross_layers=3, deep_hidden=[128, 64], low_rank=32, dropout=0.2):
        super().__init__()
        self.num_feats = num_feats

        # Embedding 投影
        self.query_proj = nn.Linear(EMB_DIM, embed_dim)
        self.note_text_proj = nn.Linear(EMB_DIM, embed_dim)
        self.note_img_proj = nn.Linear(EMB_DIM, embed_dim)

        # 数值特征线性投影
        self.num_proj = nn.ModuleDict({
            f: nn.Linear(1, embed_dim) for f in self.num_feats
        })

        # DCNv2 多层交叉
        input_dim = embed_dim * (3 + len(num_feats))
        self.cross_layers = nn.ModuleList([DCNv2Layer(input_dim, low_rank) for _ in range(cross_layers)])

        # Deep network
        deep_layers = []
        for h in deep_hidden:
            deep_layers.append(nn.Linear(input_dim, h))
            deep_layers.append(nn.ReLU())
            deep_layers.append(nn.Dropout(dropout))
            input_dim = h
        self.deep = nn.Sequential(*deep_layers)
        self.deep_out = nn.Linear(input_dim, 1)

        # 树模型残差融合
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.1))

    def forward(self, features):
        # ====== Embedding 投影 ======
        x_list = [
            self.query_proj(features["query_emb"]),
            self.note_text_proj(features["note_text_emb"]),
            self.note_img_proj(features["note_emb"])
        ]
        for f in self.num_feats:
            val = features[f].unsqueeze(-1)
            x_list.append(self.num_proj[f](val))
        x = torch.cat(x_list, dim=-1)  # (batch, field_dim*embed)

        # ====== DCNv2 Cross ======
        xl = x
        x0 = x
        for layer in self.cross_layers:
            xl = layer(x0, xl)

        # ====== Deep Network ======
        deep_score = self.deep_out(self.deep(x))

        # ====== 融合树模型 ======
        final_score = deep_score.squeeze(-1) + self.alpha * features["lgb_score"] + self.beta * features["xgb_score"]
        return final_score

# ======================
# Train Fold
# ======================
def train_fold(train_df, val_df, fold, q_store, t_store, i_store, num_feats):
    train_dataset = SearchDataset(train_df)
    val_dataset = SearchDataset(val_df)
    model = DCNv2Ranker(num_feats=num_feats).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=LR)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", patience=5)

    best_ndcg = -1
    no_improve = 0

    if PLOT:
        plt.ion()
        fig, ax = plt.subplots()
        ax.set_title(f"Fold {fold}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss/NDCG@10")
        lines = {k: ax.plot([], [], label=k)[0] for k in ["tr_loss","va_loss","tr_ndcg","va_ndcg"]}
        center_window()
        ax.legend()

    history = {k: [] for k in ["tr_loss","va_loss","tr_ndcg","va_ndcg"]}

    for epoch in range(EPOCHS):
        # ========= Train =========
        model.train()
        tr_losses, tr_y, tr_p, tr_g = [], [], [], []

        train_shards = train_dataset.get_shard_indices()
        for shard_idx, shard_df in tqdm(train_shards, desc=f"Train Fold {fold} | Epoch {epoch+1}"):
            idxs_q = shard_df["search_idx"].tolist()
            idxs_t = shard_df["note_idx"].tolist()
            idxs_i = shard_df["note_idx"].tolist()

            q_batch = q_store.get_tensor_by_indices(idxs_q)
            t_batch = t_store.get_tensor_by_indices(idxs_t)
            i_batch = i_store.get_tensor_by_indices(idxs_i)

            q_store.prefetch_shard_by_idx(idxs_q)
            t_store.prefetch_shard_by_idx(idxs_t)
            i_store.prefetch_shard_by_idx(idxs_i)

            batch_features = {
                "query_emb": q_batch,
                "note_text_emb": t_batch,
                "note_emb": i_batch,
                "lgb_score": torch.tensor(shard_df["lgb_score"].values, device=DEVICE, dtype=torch.float16),
                "xgb_score": torch.tensor(shard_df["xgb_score"].values, device=DEVICE, dtype=torch.float16)
            }
            batch_y = torch.tensor(shard_df["y_disc"].values, device=DEVICE, dtype=torch.float16)
            batch_g = torch.tensor(shard_df["search_idx"].values, device=DEVICE, dtype=torch.int32)
            for f in num_feats:
                batch_features[f] = torch.tensor(shard_df[f].values, device=DEVICE, dtype=torch.float16)

            opt.zero_grad()
            pred = model(batch_features)
            loss = lambda_rank_loss(pred, batch_y, batch_g)
            loss.backward()
            opt.step()

            tr_losses.append(loss.item())
            tr_y.append(batch_y.cpu().numpy())
            tr_p.append(pred.detach().cpu().numpy())
            tr_g.append(batch_g.cpu().numpy())

        tr_loss = np.mean(tr_losses)
        tr_ndcg = eval_ndcg_by_group(np.concatenate(tr_y), np.concatenate(tr_p), np.concatenate(tr_g), TOPK)

        # ========= Validation =========
        model.eval()
        va_losses, va_y, va_p, va_g = [], [], [], []
        val_shards = val_dataset.get_shard_indices()
        with torch.no_grad():
            for shard_idx, shard_df in tqdm(val_shards, desc=f"Val Fold {fold} | Epoch {epoch+1}"):
                idxs_q = shard_df["search_idx"].tolist()
                idxs_t = shard_df["note_idx"].tolist()
                idxs_i = shard_df["note_idx"].tolist()

                q_batch = q_store.get_tensor_by_indices(idxs_q)
                t_batch = t_store.get_tensor_by_indices(idxs_t)
                i_batch = i_store.get_tensor_by_indices(idxs_i)

                q_store.prefetch_shard_by_idx(idxs_q)
                t_store.prefetch_shard_by_idx(idxs_t)
                i_store.prefetch_shard_by_idx(idxs_i)

                batch_features = {
                    "query_emb": q_batch,
                    "note_text_emb": t_batch,
                    "note_emb": i_batch,
                    "lgb_score": torch.tensor(shard_df["lgb_score"].values, device=DEVICE, dtype=torch.float16),
                    "xgb_score": torch.tensor(shard_df["xgb_score"].values, device=DEVICE, dtype=torch.float16)
                }
                batch_y = torch.tensor(shard_df["y_disc"].values, device=DEVICE, dtype=torch.float16)
                batch_g = torch.tensor(shard_df["search_idx"].values, device=DEVICE, dtype=torch.int32)
                for f in num_feats:
                    batch_features[f] = torch.tensor(shard_df[f].values, device=DEVICE, dtype=torch.float16)

                pred = model(batch_features)
                loss = lambda_rank_loss(pred, batch_y, batch_g)
                va_losses.append(loss.item())
                va_y.append(batch_y.cpu().numpy())
                va_p.append(pred.cpu().numpy())
                va_g.append(batch_g.cpu().numpy())

        va_loss = np.mean(va_losses)
        va_ndcg = eval_ndcg_by_group(np.concatenate(va_y), np.concatenate(va_p), np.concatenate(va_g), TOPK)

        sch.step(va_ndcg)

        for k, v in zip(["tr_loss","va_loss","tr_ndcg","va_ndcg"], [tr_loss, va_loss, tr_ndcg, va_ndcg]):
            history[k].append(v)
            lines[k].set_data(range(len(history[k])), history[k])
        ax.relim()
        ax.autoscale_view()
        plt.pause(0.01)

        print(f"Epoch {epoch+1} | Train Loss={tr_loss:.4f}, Val Loss={va_loss:.4f}, Train NDCG@10={tr_ndcg:.4f}, Val NDCG@10={va_ndcg:.4f}, LR={opt.param_groups[0]['lr']:.4f}")

        if va_ndcg > best_ndcg + 1e-6:
            best_ndcg = va_ndcg
            torch.save(model.state_dict(), MODEL_DIR / f"dcnv2_fold{fold}.pt")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP:
                print(f"[Fold {fold}] Early stop at epoch {epoch+1}")
                break

    if PLOT:
        plt.ioff()
        plt.savefig(PLOT_DIR / f"dcnv2_fold{fold}_curve.png", dpi=160)
        plt.close()
    return best_ndcg

# ======================
# Main
# ======================
def main():
    df = pd.read_parquet(FEATURE_DIR / "search_train_tree.parquet")
    drop_feats = ["click", "y_multi", "note_idx", "search_idx"]
    sparse_feats = ["gender", "platform", "age", "location", "note_type", "taxonomy1_id", "taxonomy2_id", "taxonomy3_id"]
    num_feats = [c for c in df.columns if c not in drop_feats]
    dense_feats = [c for c in num_feats if c not in sparse_feats]

    # dense_feats 归一化
    for c in dense_feats:
        mean = df[c].mean()
        std = df[c].std()
        df[c] = 0.0 if std < 1e-6 else (df[c]-mean)/std

    df["y_disc"] = discretize_relevance(df["y_multi"].values)
    df["lgb_score"] = np.load(RESULT_DIR / "lgb_search.npy")
    df["xgb_score"] = np.load(RESULT_DIR / "xgb_search.npy")
    df = group_norm_score(df, "lgb_score", "search_idx")
    df = group_norm_score(df, "xgb_score", "search_idx")

    q_store = EmbeddingStore(EMB_DIR / "query_text_emb", key_col="search_idx", emb_col="query_emb", prefix="search")
    t_store = EmbeddingStore(EMB_DIR / "note_text_emb", key_col="note_idx", emb_col="note_text_emb")
    i_store = EmbeddingStore(EMB_DIR / "note_img_emb", key_col="note_idx", emb_col="note_img_emb")

    gkf = GroupKFold(N_SPLITS, shuffle=True, random_state=SEED)
    best_global = -1
    scores_dcnv2 = []
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(df, groups=df["search_idx"]), 1):
        train_df = df.iloc[tr_idx]
        val_df = df.iloc[va_idx]
        best_ndcg = train_fold(train_df, val_df, fold, q_store, t_store, i_store, num_feats)
        best_global = max(best_global, best_ndcg)
        scores_dcnv2.append(best_ndcg)

    print(f"Training finished, best global NDCG@10={best_global:.4f}")

    summary = {
        "scene": "search",
        "model": "dcnv2",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "embed_dim": 32,
            "cross_layers": 3,
            "deep_hidden": [128, 64],
            "batch_size": BATCH_SIZE,
            "topk": TOPK
        },
        "ndcg@10_mean": float(np.mean(scores_dcnv2)),
        "folds": scores_dcnv2
    }

    out_path = RESULT_DIR / "dcnv2_summary_search.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] summary saved to {out_path}")

if __name__ == "__main__":
    main()
