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
# Config
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
# Parquet Shard Cache (LRU)
# =============================
class ShardCache:
    def __init__(self, max_shards=3):
        self.cache = OrderedDict()
        self.max_shards = max_shards

    def get(self, path):
        if path in self.cache:
            self.cache.move_to_end(path)  # 最近使用移到末尾（LRU）
            return self.cache[path]
        df = pd.read_parquet(path)        # 未命中则从磁盘加载
        self.cache[path] = df
        if len(self.cache) > self.max_shards:
            self.cache.popitem(last=False)  # 移除最久未使用的（头部）
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
class SearchDataset:    # 将大的查询 DataFrame 按 shard_size 切分成多个小批次（shards）
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
# AutoInt Ranker
# ======================
class MultiHeadAutoInt(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = embed_dim // num_heads
        self.qw = nn.ParameterList([nn.Parameter(torch.randn(embed_dim, self.d_head) * 0.02) for _ in range(num_heads)])
        self.kw = nn.ParameterList([nn.Parameter(torch.randn(embed_dim, self.d_head) * 0.02) for _ in range(num_heads)])
        self.vw = nn.ParameterList([nn.Parameter(torch.randn(embed_dim, self.d_head) * 0.02) for _ in range(num_heads)])
        self.residual = nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.02)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        head_outputs = []
        for i in range(self.num_heads):
            Q = torch.einsum('bnd,de->bne', x, self.qw[i])
            K = torch.einsum('bnd,de->bne', x, self.kw[i])
            V = torch.einsum('bnd,de->bne', x, self.vw[i])
            attention_score = torch.matmul(Q, K.transpose(-1, -2)) / (self.d_head ** 0.5)
            attention_weights = F.softmax(attention_score, dim=-1)
            head_output = torch.matmul(attention_weights, V)
            head_outputs.append(head_output)
        multi_head_output = torch.cat(head_outputs, dim=-1)
        res = torch.einsum('bnd,de->bne', x, self.residual)
        x = self.norm1(multi_head_output + res)
        x = self.norm2(x + self.ffn(x))
        return x

class AutoIntRanker(nn.Module):
    def __init__(self, num_feats, embed_dim=32, num_heads=2, num_layers=3, ff_hidden=128, dropout=0.2):
        super().__init__()
        # Embedding 投影
        self.query_proj = nn.Linear(EMB_DIM, embed_dim)
        self.note_text_proj = nn.Linear(EMB_DIM, embed_dim)
        self.note_img_proj = nn.Linear(EMB_DIM, embed_dim)
        self.num_feats = num_feats
        self.num_proj = nn.ModuleDict({
            f: nn.Linear(1, embed_dim) for f in self.num_feats
        })

        # AutoInt layers
        self.layers = nn.ModuleList([MultiHeadAutoInt(embed_dim, num_heads, ff_hidden, dropout) for _ in range(num_layers)])

        # 输出 head
        self.head = nn.Linear(embed_dim, 1)

        # 可训练残差权重
        self.alpha = nn.Parameter(torch.tensor(0.1))  # LGB 权重
        self.beta = nn.Parameter(torch.tensor(0.1))   # XGB 权重

    def forward(self, features):
        x_list = [
            self.query_proj(features["query_emb"]),
            self.note_text_proj(features["note_text_emb"]),
            self.note_img_proj(features["note_img_emb"])
        ]
        for f in self.num_feats:
            val = features[f].unsqueeze(-1)  # (batch_size, 1)
            x_list.append(self.num_proj[f](val))
        x = torch.stack(x_list, dim=1)  # (batch_size, num_fields, embed_dim)
        
        for layer in self.layers:
            x = layer(x)
        deep_score = self.head(x.mean(dim=1)).squeeze(-1)  # batch_size
        
        # Residual 融合树模型
        final_score = deep_score + self.alpha * features["lgb_score"] + self.beta * features["xgb_score"]
        return final_score

# ======================
# Train Fold with Shard + Async IO
# ======================
def train_fold(train_df, val_df, fold, q_store, t_store, i_store, num_feats):
    train_dataset = SearchDataset(train_df)
    val_dataset = SearchDataset(val_df)
    model = AutoIntRanker(num_feats=num_feats).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
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

            # 实际取 embedding tensor
            q_batch = q_store.get_tensor_by_indices(idxs_q)
            t_batch = t_store.get_tensor_by_indices(idxs_t)
            i_batch = i_store.get_tensor_by_indices(idxs_i)
            
            # 异步预取下一批可能需要的 shard
            q_store.prefetch_shard_by_idx(idxs_q)
            t_store.prefetch_shard_by_idx(idxs_t)
            i_store.prefetch_shard_by_idx(idxs_i)

            batch_features = {
                "query_emb": q_batch,
                "note_text_emb": t_batch,
                "note_img_emb": i_batch,
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

                # 实际取 embedding tensor
                q_batch = q_store.get_tensor_by_indices(idxs_q)
                t_batch = t_store.get_tensor_by_indices(idxs_t)
                i_batch = i_store.get_tensor_by_indices(idxs_i)
                
                # 异步预取下一批可能需要的 shard
                q_store.prefetch_shard_by_idx(idxs_q)
                t_store.prefetch_shard_by_idx(idxs_t)
                i_store.prefetch_shard_by_idx(idxs_i)

                batch_features = {
                    "query_emb": q_batch,
                    "note_text_emb": t_batch,
                    "note_img_emb": i_batch,
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

        # ========= Scheduler =========
        sch.step(va_ndcg)

        # ========= Record History =========
        for k, v in zip(["tr_loss","va_loss","tr_ndcg","va_ndcg"], [tr_loss, va_loss, tr_ndcg, va_ndcg]):
            history[k].append(v)
            lines[k].set_data(range(len(history[k])), history[k])
        ax.relim()
        ax.autoscale_view()
        plt.pause(0.01)

        print(f"Epoch {epoch+1} | Train Loss={tr_loss:.4f}, Val Loss={va_loss:.4f},  Train NDCG@10={tr_ndcg:.4f}, Val NDCG@10={va_ndcg:.4f}, LR={opt.param_groups[0]['lr']:.4f}")

        # ========= Early Stop =========
        if va_ndcg > best_ndcg + 1e-6:
            best_ndcg = va_ndcg
            torch.save(model.state_dict(), MODEL_DIR / f"autoint_fold{fold}.pt")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP:
                print(f"[Fold {fold}] Early stop at epoch {epoch+1}")
                break

    if PLOT:
        plt.ioff()
        plt.savefig(PLOT_DIR / f"autoint_fold{fold}_curve.png", dpi=160)
        plt.close()
    return best_ndcg

# ======================
# Main
# ======================
def main():
    # 1. Load Data
    df = pd.read_parquet(FEATURE_DIR / "search_train_features.parquet")
    drop_feats = ["click", "y_multi", "note_idx", "search_idx"]
    sparse_feats = ["gender", "platform", "age", "location", "note_type", "taxonomy1_id", "taxonomy2_id", "taxonomy3_id"]
    num_feats = [c for c in df.columns if c not in drop_feats]
    dense_feats = [c for c in num_feats if c not in sparse_feats]
    
    # 2. Preprocessing: dense_feats normalization (z-score)
    for c in dense_feats:
        mean = df[c].mean()
        std = df[c].std()
        if std < 1e-6:
            df[c] = 0.0
        else:
            df[c] = (df[c] - mean) / std
    
    df["y_disc"] = discretize_relevance(df["y_multi"].values)
    df["lgb_score"] = np.load(RESULT_DIR / "lgb_search.npy")
    df["xgb_score"] = np.load(RESULT_DIR / "xgb_search.npy")
    df = group_norm_score(df, "lgb_score", "search_idx")
    df = group_norm_score(df, "xgb_score", "search_idx")

    # 3. Setup Embedding Stores
    q_store = EmbeddingStore(EMB_DIR / "query_text_emb", key_col="search_idx", emb_col="query_emb", prefix="search")
    t_store = EmbeddingStore(EMB_DIR / "note_text_emb", key_col="note_idx", emb_col="note_text_emb")
    i_store = EmbeddingStore(EMB_DIR / "note_img_emb", key_col="note_idx", emb_col="note_img_emb")

    # 4. Cross Validation
    gkf = GroupKFold(N_SPLITS, shuffle=True, random_state=SEED)
    best_global = -1
    scores_autoint = []
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(df, groups=df["search_idx"]), 1):
        train_df = df.iloc[tr_idx]
        val_df = df.iloc[va_idx]
        best_ndcg = train_fold(train_df, val_df, fold, q_store, t_store, i_store, num_feats)
        best_global = max(best_global, best_ndcg)
        scores_autoint.append(best_ndcg)

    print(f"Training finished, best global NDCG@10={best_global:.4f}")

    summary = {
        "scene": "search",
        "model": "autoint",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "embed_dim": 32,
            "num_layers": 3,
            "num_heads": 2,
            "batch_size": BATCH_SIZE,
            "topk": TOPK
        },
        "ndcg@10_mean": float(np.mean(scores_autoint)),
        "folds": scores_autoint
    }

    out_path = RESULT_DIR / "autoint_summary_search.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] summary saved to {out_path}")

if __name__ == "__main__":
    main()