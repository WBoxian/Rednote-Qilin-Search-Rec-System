"""
Qilin DSSM 双塔召回训练脚本 (Search & Rec)
- 特征处理: 离散特征 Embedding，计数特征 Log1p + z-score，比率/稠密特征 z-score，行为序列向量 Mean Pooling
 - Search 场景: user 塔额外接入 query_text_emb
 - Easy Neg: 基于曝光 imp_num^0.75 的非均匀采样 (q)
 - Hard Neg: 召回但被排序模型淘汰的样本
- 训练指标: Pairwise Triplet Loss
- 评价指标: AUC (基于 user/item 余弦相似度)
- 日志: TensorBoard 记录训练/验证 Loss, AUC

使用示例:
    uv run python src/recall/dssm_trainer.py --scene search --neg-mode auto
    uv run python src/recall/dssm_trainer.py --scene rec --neg-mode auto
    nohup uv run tensorboard --logdir=outputs/logs > /dev/null 2>&1 &
"""

import os
import json
import argparse
import logging
import random
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
from pathlib import Path
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# =============================
# 1. 全局配置
# =============================
SEED = 42
BATCH_SIZE = 1024           # in-batch neg 模式下每批即有 1023 个负样本，可用较大 batch
EPOCHS = 3                  # in-batch neg 收敛更快，3 轮通常足够
EARLY_STOP = 2
LR = 1e-3
EMB_DIM = 768              # 文本/图片向量维度
HIDDEN_DIM = 128           # 双塔输出维度
MARGIN = 0.5               # Triplet 阈值（供旧版 loss 兼容）
INBATCH_TEMPERATURE = 0.07 # in-batch softmax 温度，越小对比越尖锐
SEQ_MAX_LEN = 20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

BASE_DIR = Path(__file__).resolve().parents[2]  # Qilin/
EMB_DIR = BASE_DIR / "embeddings"
FEAT_DIR = BASE_DIR / "features"
CAT_MAP_DIR = FEAT_DIR / "vocab_dict"
OUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = OUT_DIR / "models"
LOG_DIR = OUT_DIR / "logs"
RESULT_DIR = BASE_DIR / "outputs" / "index"

for d in [MODEL_DIR, LOG_DIR, RESULT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("dssm")
HARD_NEG_CANDIDATE_COLS = [
    "hard_neg_note_idxs",
    "hard_neg_note_idx",
    "hard_neg_note_ids",
    "hard_neg_items",
]
HARD_NEG_RATIO = 0.5  # hard:ease=1:1
NUM_NEG_PER_POS = 3
HARD_EASY_NEG_RATIO = (1, 2)  # hard mode: 1 hard : 2 easy
HARD_NEG_KEEP_TOPK = 10


# =============================
# 2. 工具函数
# =============================
def set_seed(seed: int = SEED):
    """保证实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_category_sizes():
    """
    加载类别映射文件（gender/platform/age/location/taxonomy）
    """
    sizes = {}
    maps = {
        "gender": "gender.pkl",
        "platform": "platform.pkl",
        "age": "age.pkl",
        "location": "location.pkl",
        "tax1": "taxonomy1_id.pkl",
        "tax2": "taxonomy2_id.pkl",
        "tax3": "taxonomy3_id.pkl",
    }
    for k, fname in maps.items():
        path = CAT_MAP_DIR / fname
        if path.exists():
            with open(path, "rb") as f:
                mp = pickle.load(f)
            sizes[k] = max(mp.values()) + 1
    return sizes

def log1p_and_zscore(df: pd.DataFrame, cols_log, cols_norm):
    """
    - log1p + zscore：先 log1p(x) 压缩长尾，再做 z-score（用于计数类）
    - zscore：直接 (x-μ)/σ（用于比率、dense）
    """
    stats = {}
    for c in cols_log:
        df[c] = np.log1p(df[c].astype(float))
    for c in cols_norm:
        mu = df[c].mean()
        std = df[c].std()
        std = std if std > 1e-6 else 1.0
        df[c] = (df[c] - mu) / std
        stats[c] = (mu, std)
    return df, stats


def split_by_session_idx(df: pd.DataFrame, val_ratio: float = 0.2, seed: int = SEED):
    """按 session_idx 分组切分，避免数据泄露。"""
    sessions = df["session_idx"].drop_duplicates().to_numpy()
    if len(sessions) < 2:
        raise ValueError("session 数量不足，无法切分 train/val")
    rng = np.random.default_rng(seed)
    rng.shuffle(sessions)
    cut = max(1, int(len(sessions) * (1.0 - val_ratio)))
    cut = min(cut, len(sessions) - 1)
    train_sessions = set(sessions[:cut].tolist())
    val_sessions = set(sessions[cut:].tolist())
    train_df = df[df["session_idx"].isin(train_sessions)].copy()
    val_df = df[df["session_idx"].isin(val_sessions)].copy()
    return train_df, val_df


def resolve_neg_mode(df: pd.DataFrame, requested_mode: str):
    """解析负样本模式：auto/easy/hard，并返回 (resolved_mode, hard_col)。"""
    hard_col = next((c for c in HARD_NEG_CANDIDATE_COLS if c in df.columns), None)
    has_hard = False
    if hard_col is not None:
        hard_series = df[hard_col].apply(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0)
        has_hard = bool((hard_series > 0).any())

    if requested_mode == "easy":
        return "easy", hard_col
    if requested_mode == "hard":
        if has_hard:
            return "hard", hard_col
        logger.warning("Requested hard neg, but no valid hard neg samples found. Fallback to easy neg.")
        return "easy", hard_col
    # auto
    return ("hard", hard_col) if has_hard else ("easy", hard_col)


def attach_external_hard_neg(
    df_all: pd.DataFrame,
    scene: str,
    req_col: str,
    hard_neg_path: Path | None,
) -> pd.DataFrame:
    """将外部 hard neg 文件合并到训练特征，优先使用外部 hard neg。"""
    if hard_neg_path is None:
        return df_all
    if not hard_neg_path.exists():
        logger.warning(f"External hard neg file not found: {hard_neg_path}")
        return df_all

    hard_df = pd.read_parquet(hard_neg_path)
    if "hard_neg_note_idxs" not in hard_df.columns:
        logger.warning(f"External hard neg file missing hard_neg_note_idxs: {hard_neg_path}")
        return df_all

    merge_cols = [req_col]
    if "note_idx" in hard_df.columns:
        merge_cols.append("note_idx")
    hard_df = hard_df[merge_cols + ["hard_neg_note_idxs"]].copy()
    hard_df["hard_neg_note_idxs"] = hard_df["hard_neg_note_idxs"].apply(
        lambda x: x.tolist() if isinstance(x, np.ndarray) else (x if isinstance(x, list) else [])
    )
    hard_df = hard_df.rename(columns={"hard_neg_note_idxs": "hard_neg_note_idxs_ext"})
    hard_df = hard_df.drop_duplicates(subset=merge_cols, keep="first")

    if "note_idx" in merge_cols:
        merged = df_all.merge(
            hard_df,
            on=[req_col, "note_idx"],
            how="left",
            suffixes=("", "_ext"),
        )
    else:
        merged = df_all.merge(
            hard_df,
            on=[req_col],
            how="left",
            suffixes=("", "_ext"),
        )

    base_col = next((c for c in HARD_NEG_CANDIDATE_COLS if c in merged.columns), None)
    if base_col is not None:
        base_list = merged[base_col].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else (x if isinstance(x, list) else []))
    else:
        base_list = pd.Series([[] for _ in range(len(merged))], index=merged.index)
    ext_col = "hard_neg_note_idxs_ext"
    ext_list = merged[ext_col].apply(lambda x: x if isinstance(x, list) else [])
    merged["hard_neg_note_idxs"] = [
        e if len(e) > 0 else b
        for b, e in zip(base_list.tolist(), ext_list.tolist())
    ]
    merged = merged.drop(columns=[ext_col])
    if base_col is not None and base_col != "hard_neg_note_idxs":
        merged = merged.drop(columns=[base_col])
    cov = float((merged["hard_neg_note_idxs"].apply(len) > 0).mean()) if len(merged) > 0 else 0.0
    logger.info(f"Attached external hard neg from {hard_neg_path}, coverage={cov:.4f}")
    return merged


# =============================
# 3. 向量内存索引 (MMap)
# =============================
class MMapStore:
    """加载二进制向量文件，支持 O(1) 随机访问"""

    def __init__(self, folder_name: str, file_prefix: str, dim: int = EMB_DIM, is_seq=False):
        path = EMB_DIR / folder_name
        self.dim = dim
        self.is_seq = is_seq
        with open(path / f"{file_prefix}_map.json", "r") as f:
            self.id2pos = json.load(f)
        num_items = len(self.id2pos)
        shape = (num_items, dim) if not is_seq else (num_items, SEQ_MAX_LEN, dim)
        self.data = np.memmap(path / f"{file_prefix}.bin", dtype="float16", mode="r", shape=shape)

    def get_vec(self, idx):
        pos = self.id2pos.get(str(idx))
        if pos is None:
            return torch.zeros(self.dim)
        return torch.from_numpy(self.data[pos].astype("float32"))

    def get_batch_vecs(self, idxs):
        if len(idxs) == 0:
            return torch.zeros(0, self.dim, dtype=torch.float32)
        batch = np.zeros((len(idxs), self.dim), dtype=np.float32)
        positions = [self.id2pos.get(str(i), -1) for i in idxs]
        valid = np.array([p >= 0 for p in positions], dtype=bool)
        if valid.any():
            pos_arr = np.array([positions[i] for i in range(len(positions)) if valid[i]], dtype=np.int64)
            batch[valid] = self.data[pos_arr].astype(np.float32)
        return torch.from_numpy(batch)


# =============================
# 4. 数据集
# =============================
class DSSMDataset(Dataset):
    def __init__(
        self,
        scene,
        req_col,
        df,
        item_meta,
        item_pool,
        neg_sampling_probs,
        use_hard_neg=False,
        hard_neg_col=None,
        hard_neg_ratio=HARD_NEG_RATIO,
        num_neg_per_pos=NUM_NEG_PER_POS,
        use_inbatch_neg: bool = False,
    ):
        """
        df: 当前 split 的样本
        item_meta: note_idx -> item 特征行（确保覆盖负样本）
        item_pool / neg_sampling_probs: 用于负采样的 item 池与概率
        use_inbatch_neg: 启用 in-batch 负样本模式（batch 内其他正样本充当负样本）
        """
        self.df = df.reset_index(drop=True)
        self.scene = scene
        self.req_col = req_col
        self.item_meta = item_meta
        self.item_pool = item_pool
        self.neg_sampling_probs = neg_sampling_probs
        self.neg_sampling_probs_np = neg_sampling_probs.numpy()
        self.use_hard_neg = use_hard_neg and hard_neg_col is not None and hard_neg_col in df.columns
        self.hard_neg_ratio = hard_neg_ratio
        self.num_neg_per_pos = max(1, int(num_neg_per_pos))
        self.use_inbatch_neg = bool(use_inbatch_neg)

        # 将高频访问列预先转成 numpy，降低 __getitem__ 的 pandas 开销
        self.user_idx = self.df["user_idx"].to_numpy(np.int64)
        self.gender = self.df["gender_enc"].to_numpy(np.int64)
        self.platform = self.df["platform_enc"].to_numpy(np.int64)
        self.age = self.df["age_enc"].to_numpy(np.int64)
        self.location = self.df["location_enc"].to_numpy(np.int64)
        self.fans_num = self.df["fans_num"].to_numpy(np.float32)
        self.follows_num = self.df["follows_num"].to_numpy(np.float32)
        self.history = self.df["recent_clicked_note_idxs"].tolist()
        self.note_idx = self.df["note_idx"].to_numpy(np.int64)
        self.req_ids = (
            self.df[req_col].to_numpy(np.int64)
            if req_col in self.df.columns
            else np.zeros(len(self.df), dtype=np.int64)
        )
        self.dense_feats = self.df[[f"dense_feat{i}" for i in range(1, 41)]].to_numpy(np.float32)
        self.labels = self.df["click"].to_numpy(np.int64) if "click" in self.df.columns else np.ones(len(self.df), dtype=np.int64)
        self.hard_neg_lists = self.df[hard_neg_col].tolist() if self.use_hard_neg else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 用户侧输入
        history_ids = self.history[idx] or []
        history_ids = history_ids[-SEQ_MAX_LEN:]
        user_input = {
            "user_idx": self.user_idx[idx],
            "req_id": self.req_ids[idx],
            "gender_enc": self.gender[idx],
            "platform_enc": self.platform[idx],
            "age_enc": self.age[idx],
            "location_enc": self.location[idx],
            "fans_num": self.fans_num[idx],
            "follows_num": self.follows_num[idx],
            "dense_feats": self.dense_feats[idx],
            "history_ids": history_ids,
        }

        # 正样本 item
        pos_item = self._fetch_item(int(self.note_idx[idx]))

        # 负采样：每个正样本采样 num_neg_per_pos 个负样本
        # hard 模式采用 hard:easy=1:2；easy 模式全部 easy
        neg_items = []
        neg_types = []  # 1: hard, 0: easy

        hard_ids = []
        if self.use_hard_neg:
            raw_hard_ids = self.hard_neg_lists[idx]
            if isinstance(raw_hard_ids, np.ndarray):
                hard_ids = [int(x) for x in raw_hard_ids.tolist()[:HARD_NEG_KEEP_TOPK] if int(x) in self.item_meta]
            elif isinstance(raw_hard_ids, list):
                hard_ids = [int(x) for x in raw_hard_ids[:HARD_NEG_KEEP_TOPK] if int(x) in self.item_meta]

        if self.use_inbatch_neg:
            # In-batch neg 模式：每样本至多 1 条 hard neg，不做随机 easy 采样
            # batch 内其他正样本 item 在 compute_inbatch_loss 中自动充当负样本
            if self.use_hard_neg and hard_ids:
                candidate = int(hard_ids[np.random.randint(len(hard_ids))])
                neg_items.append(self._fetch_item(candidate))
                neg_types.append(1)   # 真实 hard neg，追加进 item pool
            else:
                neg_items.append(pos_item)   # placeholder，neg_type=0 在 loss 中不追加
                neg_types.append(0)
        else:
            if self.use_hard_neg and len(hard_ids) > 0:
                hard_num = max(1, int(round(self.num_neg_per_pos * HARD_EASY_NEG_RATIO[0] / sum(HARD_EASY_NEG_RATIO))))
            else:
                hard_num = 0
            easy_num = self.num_neg_per_pos - hard_num

            for _ in range(hard_num):
                candidate = int(hard_ids[np.random.randint(len(hard_ids))])
                neg_items.append(self._fetch_item(candidate))
                neg_types.append(1)

            for _ in range(easy_num):
                neg_idx_in_pool = np.random.choice(len(self.item_pool), p=self.neg_sampling_probs_np)
                neg_note_idx = int(self.item_pool[neg_idx_in_pool])
                neg_items.append(self._fetch_item(neg_note_idx))
                neg_types.append(0)

        label = self.labels[idx]
        return user_input, pos_item, neg_items, np.asarray(neg_types, dtype=np.int64), label

    def _fetch_item(self, note_idx: int):
        """根据 note_idx 获取 item 侧所有特征（含稠密统计）"""
        m = self.item_meta.get(note_idx)
        if m is None:
            # 极端情况兜底
            m = self.item_meta[next(iter(self.item_meta))]
        return m


# =============================
# 5. 模型
# =============================
class DSSMModel(nn.Module):
    def __init__(self, scene, cat_vocabs, item_dense_dim, user_dense_dim):
        super().__init__()
        self.scene = scene

        # --- User Tower ---
        self.u_emb_user = nn.Embedding(cat_vocabs["user_idx"], 32)
        self.u_emb_gender = nn.Embedding(cat_vocabs["gender"], 4)
        self.u_emb_platform = nn.Embedding(cat_vocabs["platform"], 4)
        self.u_emb_age = nn.Embedding(cat_vocabs["age"], 8)
        self.u_emb_loc = nn.Embedding(cat_vocabs["location"], 16)

        u_cat_dim = 32 + 4 + 4 + 8 + 16
        self.user_mlp = nn.Sequential(
            nn.Linear(user_dense_dim + u_cat_dim, 512),
            nn.ReLU(),
            nn.Linear(512, HIDDEN_DIM),
        )

        # --- Item Tower ---
        self.i_emb_item = nn.Embedding(cat_vocabs["note_idx"], 32)
        # note_type 原始值通常是 1图片/2视频，预留 0 作为兜底
        self.i_emb_type = nn.Embedding(3, 4)
        self.i_emb_taxo = nn.ModuleList(
            [
                nn.Embedding(cat_vocabs["tax1"], 16),
                nn.Embedding(cat_vocabs["tax2"], 16),
                nn.Embedding(cat_vocabs["tax3"], 16),
            ]
        )

        i_cat_dim = 32 + 4 + 16 * 3
        self.item_mlp = nn.Sequential(
            nn.Linear(item_dense_dim + i_cat_dim + EMB_DIM * 2, 512),
            nn.ReLU(),
            nn.Linear(512, HIDDEN_DIM),
        )

    @staticmethod
    def _safe_index(x: torch.Tensor, vocab_size: int):
        # 将非法索引裁剪到合法区间，避免 CUDA embedding 越界
        return torch.clamp(x, min=0, max=vocab_size - 1)

    def forward_user(self, batch_data, history_vecs):
        user_idx = self._safe_index(batch_data["user_idx"], self.u_emb_user.num_embeddings)
        gender = self._safe_index(batch_data["gender_enc"], self.u_emb_gender.num_embeddings)
        platform = self._safe_index(batch_data["platform_enc"], self.u_emb_platform.num_embeddings)
        age = self._safe_index(batch_data["age_enc"], self.u_emb_age.num_embeddings)
        location = self._safe_index(batch_data["location_enc"], self.u_emb_loc.num_embeddings)

        e1 = self.u_emb_user(user_idx)
        e2 = self.u_emb_gender(gender)
        e3 = self.u_emb_platform(platform)
        e4 = self.u_emb_age(age)
        e5 = self.u_emb_loc(location)

        u_dense = torch.cat(
            [
                batch_data["fans_num"].unsqueeze(1),
                batch_data["follows_num"].unsqueeze(1),
                batch_data["dense_feats"],
            ],
            dim=1,
        )

        user_parts = [e1, e2, e3, e4, e5, u_dense, history_vecs]
        if self.scene == "search":
            user_parts.append(batch_data["query_vec"])
        combined = torch.cat(user_parts, dim=1)
        return F.normalize(self.user_mlp(combined), p=2, dim=1)

    def forward_item(self, batch_data, text_vecs, img_vecs):
        note_idx = self._safe_index(batch_data["note_idx"], self.i_emb_item.num_embeddings)
        # note_type: 1/2 -> 0/1，异常值会被裁剪
        note_type = self._safe_index(batch_data["note_type"].long() - 1, self.i_emb_type.num_embeddings)
        taxo = batch_data["taxonomy"].long()

        e1 = self.i_emb_item(note_idx)
        e2 = self.i_emb_type(note_type)
        e3 = torch.cat(
            [
                emb(self._safe_index(taxo[:, i], emb.num_embeddings))
                for i, emb in enumerate(self.i_emb_taxo)
            ],
            dim=1,
        )

        combined = torch.cat([e1, e2, e3, batch_data["dense_stats"], text_vecs, img_vecs], dim=1)
        return F.normalize(self.item_mlp(combined), p=2, dim=1)

    @staticmethod
    def cosine_score(user_vec: torch.Tensor, item_vec: torch.Tensor):
        """双塔最终输出：user/item 余弦相似度"""
        return F.cosine_similarity(user_vec, item_vec, dim=1)

    def forward_pair(
        self,
        user_batch,
        history_vecs,
        item_batch,
        item_text_vecs,
        item_img_vecs,
    ):
        """端到端前向，直接输出一个 user-item pair 的余弦相似度"""
        user_vec = self.forward_user(user_batch, history_vecs)
        item_vec = self.forward_item(item_batch, item_text_vecs, item_img_vecs)
        return self.cosine_score(user_vec, item_vec)


# =============================
# 6. 训练 & 验证
# =============================
def build_item_meta(
    df: pd.DataFrame,
    scene: str,
    item_dense_cols,
    full_note_ids: list[int] | None = None,
):
    """从样本构建 note_idx -> 特征字典，供正/负样本使用；可扩展到全量 note 集合。"""
    cols = ["note_idx", "note_type", "taxonomy1_id_enc", "taxonomy2_id_enc", "taxonomy3_id_enc"] + item_dense_cols
    meta_df = df[cols].drop_duplicates(subset=["note_idx"]).set_index("note_idx")
    meta = {}
    for nid, row in meta_df.iterrows():
        note_type = 0 if pd.isna(row["note_type"]) else int(row["note_type"])
        meta[int(nid)] = {
            "note_idx": int(nid),
            "note_type": note_type,
            "taxonomy": np.array(
                [
                    0 if pd.isna(row["taxonomy1_id_enc"]) else int(row["taxonomy1_id_enc"]),
                    0 if pd.isna(row["taxonomy2_id_enc"]) else int(row["taxonomy2_id_enc"]),
                    0 if pd.isna(row["taxonomy3_id_enc"]) else int(row["taxonomy3_id_enc"]),
                ],
                dtype=np.int64,
            ),
            "dense_stats": row[item_dense_cols].astype(np.float32).values,
        }

    if full_note_ids:
        default_dense = np.zeros(len(item_dense_cols), dtype=np.float32)
        for nid in full_note_ids:
            key = int(nid)
            if key in meta:
                continue
            meta[key] = {
                "note_idx": key,
                "note_type": 0,
                "taxonomy": np.array([0, 0, 0], dtype=np.int64),
                "dense_stats": default_dense.copy(),
            }
    return meta


def collate_fn_builder(stores, scene: str):
    """生成按场景的 collate_fn，方便 DataLoader 使用"""

    def collate_fn(batch):
        users, pos_items, neg_items_list, neg_types_list, labels = zip(*batch)
        batch_size = len(users)
        num_neg = len(neg_items_list[0]) if batch_size > 0 else 0

        # 历史 mean pooling
        hist_vecs = []
        for u in users:
            ids = u["history_ids"]
            if len(ids) > 0:
                vecs = stores["note_text"].get_batch_vecs(ids)
                hist_vecs.append(vecs.mean(dim=0))
            else:
                hist_vecs.append(torch.zeros(EMB_DIM))

        # 正/负 item 文本与图片向量
        pos_ids = [i["note_idx"] for i in pos_items]
        pos_txt = stores["note_text"].get_batch_vecs(pos_ids)
        pos_img = stores["note_img"].get_batch_vecs(pos_ids)

        flat_neg_items = [ni for row in neg_items_list for ni in row]
        neg_ids = [i["note_idx"] for i in flat_neg_items]
        neg_txt = stores["note_text"].get_batch_vecs(neg_ids).view(batch_size, num_neg, EMB_DIM)
        neg_img = stores["note_img"].get_batch_vecs(neg_ids).view(batch_size, num_neg, EMB_DIM)

        # 组装张量
        user_batch = {
            k: torch.tensor([u[k] for u in users], dtype=torch.long)
            for k in ["user_idx", "gender_enc", "platform_enc", "age_enc", "location_enc"]
        }
        user_batch["fans_num"] = torch.tensor([u["fans_num"] for u in users], dtype=torch.float32)
        user_batch["follows_num"] = torch.tensor([u["follows_num"] for u in users], dtype=torch.float32)
        user_batch["dense_feats"] = torch.tensor(np.stack([u["dense_feats"] for u in users]), dtype=torch.float32)
        if scene == "search" and "search_query" in stores:
            req_ids = [u["req_id"] for u in users]
            user_batch["query_vec"] = stores["search_query"].get_batch_vecs(req_ids)

        def item_to_tensor(items):
            return {
                "note_idx": torch.tensor([i["note_idx"] for i in items], dtype=torch.long),
                "note_type": torch.tensor([i["note_type"] for i in items], dtype=torch.long),
                "taxonomy": torch.tensor(np.stack([i["taxonomy"] for i in items]), dtype=torch.long),
                "dense_stats": torch.tensor(np.stack([i["dense_stats"] for i in items]), dtype=torch.float32),
            }

        pos_batch = item_to_tensor(pos_items)
        neg_batch_flat = item_to_tensor(flat_neg_items)
        neg_batch = {
            "note_idx": neg_batch_flat["note_idx"].view(batch_size, num_neg),
            "note_type": neg_batch_flat["note_type"].view(batch_size, num_neg),
            "taxonomy": neg_batch_flat["taxonomy"].view(batch_size, num_neg, -1),
            "dense_stats": neg_batch_flat["dense_stats"].view(batch_size, num_neg, -1),
        }

        return (
            user_batch,
            torch.stack(hist_vecs),
            pos_batch,
            pos_txt,
            pos_img,
            neg_batch,
            neg_txt,
            neg_img,
            torch.tensor(np.stack(neg_types_list), dtype=torch.long),
            torch.tensor(labels, dtype=torch.float32),
        )

    return collate_fn


def move_batch_to_device(batch, device):
    """将一个 batch 的所有 Tensor 递归移动到目标设备"""
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: move_batch_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, tuple):
        return tuple(move_batch_to_device(v, device) for v in batch)
    if isinstance(batch, list):
        return [move_batch_to_device(v, device) for v in batch]
    return batch

def compute_multineg_loss(
    pos_sim: torch.Tensor,
    neg_sim: torch.Tensor,
    neg_types: torch.Tensor,
    use_hard_neg: bool,
) -> torch.Tensor:
    """
    损失权重策略：
    - hard 模式：1 pos : 1 hard neg : 2 easy neg（loss = hard_loss + 2 * easy_loss）
    - easy 模式：1 pos : 3 easy neg（loss = 3 * easy_loss）
    """
    margin_term = torch.clamp(neg_sim - pos_sim.unsqueeze(1) + MARGIN, min=0.0)
    if use_hard_neg:
        hard_mask = neg_types > 0
        easy_mask = ~hard_mask
        hard_loss = margin_term[hard_mask].mean() if hard_mask.any() else torch.tensor(0.0, device=margin_term.device)
        easy_loss = margin_term[easy_mask].mean() if easy_mask.any() else torch.tensor(0.0, device=margin_term.device)
        return hard_loss + 2.0 * easy_loss
    easy_loss = margin_term.mean() if margin_term.numel() > 0 else torch.tensor(0.0, device=margin_term.device)
    return 3.0 * easy_loss


def compute_inbatch_loss(
    user_vecs: torch.Tensor,
    pos_vecs: torch.Tensor,
    hard_neg_vecs: torch.Tensor,
    neg_types: torch.Tensor,
    temperature: float = INBATCH_TEMPERATURE,
) -> tuple:
    """
    In-batch softmax 损失（双塔召回业界标准做法）：
    - batch 内 B 个正样本 item 互为彼此的负样本，无需额外随机 IO
    - neg_type==1 的 hard neg 额外追加进 item pool，强化对比
    返回: (loss, pos_sim, neg_sim_mean_scalar)
    """
    B = user_vecs.shape[0]
    item_pool = pos_vecs  # (B, D)

    # 仅追加真实 hard neg（neg_type==1）
    has_hard = (neg_types.reshape(B) == 1)
    if has_hard.any():
        hard_flat = hard_neg_vecs.reshape(B, -1, user_vecs.shape[1])[:, 0, :]  # (B, D)
        real_hard = hard_flat[has_hard]  # (M, D)
        item_pool = torch.cat([item_pool, real_hard], dim=0)  # (B+M, D)

    # 温度缩放的 score matrix：(B, B+M)
    scores = (user_vecs @ item_pool.T) / temperature
    labels = torch.arange(B, device=user_vecs.device)
    loss = F.cross_entropy(scores, labels)

    # 用于日志的 pos/neg sim（未缩放 cosine，与旧版指标一致）
    with torch.no_grad():
        raw = user_vecs.detach().float() @ pos_vecs.detach().T.float()  # (B, B)
        pos_sim = raw.diagonal()  # (B,)
        off_diag = raw[~torch.eye(B, dtype=torch.bool, device=raw.device)]
        neg_sim_mean = off_diag.mean().item()

    return loss, pos_sim, neg_sim_mean


def evaluate(model, loader):
    """
    计算平均 loss 与 AUC（正样本=click=1 对应 pos_sim，负样本=采样 neg）
    """
    model.eval()
    all_scores, all_labels = [], []
    total_loss, total_step = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, DEVICE)
            (
                u_data,
                u_hist,
                i_pos_data,
                i_pos_txt,
                i_pos_img,
                i_neg_data,
                i_neg_txt,
                i_neg_img,
                neg_types,
                labels,
            ) = batch

            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                user_vec = model.forward_user(u_data, u_hist)
                pos_vec = model.forward_item(i_pos_data, i_pos_txt, i_pos_img)
                flat_neg_data = {
                    "note_idx": i_neg_data["note_idx"].reshape(-1),
                    "note_type": i_neg_data["note_type"].reshape(-1),
                    "taxonomy": i_neg_data["taxonomy"].reshape(-1, i_neg_data["taxonomy"].shape[-1]),
                    "dense_stats": i_neg_data["dense_stats"].reshape(-1, i_neg_data["dense_stats"].shape[-1]),
                }
                flat_neg_txt = i_neg_txt.reshape(-1, EMB_DIM)
                flat_neg_img = i_neg_img.reshape(-1, EMB_DIM)
                neg_vec = model.forward_item(flat_neg_data, flat_neg_txt, flat_neg_img).reshape(user_vec.shape[0], -1, HIDDEN_DIM)
                if getattr(loader.dataset, "use_inbatch_neg", False):
                    loss, pos_sim, _ = compute_inbatch_loss(
                        user_vecs=user_vec,
                        pos_vecs=pos_vec,
                        hard_neg_vecs=neg_vec,
                        neg_types=neg_types,
                    )
                    B = user_vec.shape[0]
                    scores = (user_vec.float() @ pos_vec.T.float()).reshape(-1).cpu().numpy()
                    lbls = torch.eye(B, device=user_vec.device).reshape(-1).cpu().numpy()
                else:
                    pos_sim = model.cosine_score(user_vec, pos_vec)
                    neg_sim = F.cosine_similarity(user_vec.unsqueeze(1), neg_vec, dim=2)
                    loss = compute_multineg_loss(
                        pos_sim=pos_sim,
                        neg_sim=neg_sim,
                        neg_types=neg_types,
                        use_hard_neg=getattr(loader.dataset, "use_hard_neg", False),
                    )
                    scores = torch.cat([pos_sim, neg_sim.reshape(-1)]).cpu().numpy()
                    lbls = torch.cat(
                        [
                            torch.ones_like(labels),
                            torch.zeros(neg_sim.numel(), device=labels.device, dtype=labels.dtype),
                        ]
                    ).cpu().numpy()
            total_loss += loss.item()
            total_step += 1
            all_scores.append(scores)
            all_labels.append(lbls)

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    auc = roc_auc_score(labels, scores) if len(labels) > 0 else 0.0
    avg_loss = total_loss / max(total_step, 1)
    return avg_loss, auc


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    scaler,
    writer,
    global_step,
    epoch,
    total_epochs,
):
    """
    封装训练函数，返回本轮 train 指标以及更新后的 global_step。
    """
    model.train()
    total_loss = 0.0
    train_scores, train_labels = [], []
    pos_mean_sum, neg_mean_sum = 0.0, 0.0
    step_cnt = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs} Train"):
        batch = move_batch_to_device(batch, DEVICE)
        (
            u_data,
            u_hist,
            i_pos_data,
            i_pos_txt,
            i_pos_img,
            i_neg_data,
            i_neg_txt,
            i_neg_img,
            neg_types,
            labels,
        ) = batch

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
            user_vec = model.forward_user(u_data, u_hist)
            pos_vec = model.forward_item(i_pos_data, i_pos_txt, i_pos_img)
            flat_neg_data = {
                "note_idx": i_neg_data["note_idx"].reshape(-1),
                "note_type": i_neg_data["note_type"].reshape(-1),
                "taxonomy": i_neg_data["taxonomy"].reshape(-1, i_neg_data["taxonomy"].shape[-1]),
                "dense_stats": i_neg_data["dense_stats"].reshape(-1, i_neg_data["dense_stats"].shape[-1]),
            }
            flat_neg_txt = i_neg_txt.reshape(-1, EMB_DIM)
            flat_neg_img = i_neg_img.reshape(-1, EMB_DIM)
            neg_vec = model.forward_item(flat_neg_data, flat_neg_txt, flat_neg_img).reshape(user_vec.shape[0], -1, HIDDEN_DIM)

            if getattr(train_loader.dataset, "use_inbatch_neg", False):
                loss, pos_sim, neg_sim_mean = compute_inbatch_loss(
                    user_vecs=user_vec,
                    pos_vecs=pos_vec,
                    hard_neg_vecs=neg_vec,
                    neg_types=neg_types,
                )
            else:
                pos_sim = model.cosine_score(user_vec, pos_vec)
                neg_sim = F.cosine_similarity(user_vec.unsqueeze(1), neg_vec, dim=2)
                loss = compute_multineg_loss(
                    pos_sim=pos_sim,
                    neg_sim=neg_sim,
                    neg_types=neg_types,
                    use_hard_neg=getattr(train_loader.dataset, "use_hard_neg", False),
                )
                neg_sim_mean = neg_sim.detach().float().mean().item()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        with torch.no_grad():
            if getattr(train_loader.dataset, "use_inbatch_neg", False):
                B = user_vec.shape[0]
                raw = user_vec.detach().float() @ pos_vec.detach().T.float()  # (B, B)
                train_scores.append(raw.reshape(-1).cpu().numpy())
                train_labels.append(torch.eye(B, device=user_vec.device).reshape(-1).cpu().numpy())
            else:
                step_scores = torch.cat([pos_sim, neg_sim.reshape(-1)]).detach().float().cpu().numpy()
                step_labels = torch.cat(
                    [
                        torch.ones_like(labels),
                        torch.zeros(neg_sim.numel(), device=labels.device, dtype=labels.dtype),
                    ]
                ).detach().cpu().numpy()
                train_scores.append(step_scores)
                train_labels.append(step_labels)
            pos_mean_sum += pos_sim.detach().float().mean().item()
            neg_mean_sum += neg_sim_mean if isinstance(neg_sim_mean, float) else float(neg_sim_mean)
            step_cnt += 1

        writer.add_scalar("train/loss", loss.item(), global_step)
        global_step += 1

    avg_loss = total_loss / max(len(train_loader), 1)
    train_auc = roc_auc_score(np.concatenate(train_labels), np.concatenate(train_scores))
    mean_pos_sim = pos_mean_sum / max(step_cnt, 1)
    mean_neg_sim = neg_mean_sum / max(step_cnt, 1)
    return (
        avg_loss,
        train_auc,
        mean_pos_sim,
        mean_neg_sim,
        global_step,
    )


def export_item_embeddings(model, item_meta, stores, scene, model_tag, batch_size=2048):
    """
    导出 item tower 向量与映射文件，供 Faiss ANN 建索引：
    - dssm_{scene}_{tag}_item_emb.bin  (float32, [N, HIDDEN_DIM])
    - dssm_{scene}_{tag}_item_map.json (note_idx -> row_index)
    """
    note_ids = sorted(item_meta.keys())
    n_items = len(note_ids)
    if n_items == 0:
        logger.warning("No item_meta found, skip item embedding export.")
        return

    emb_path = RESULT_DIR / f"dssm_{scene}_{model_tag}_item_emb.bin"
    map_path = RESULT_DIR / f"dssm_{scene}_{model_tag}_item_map.json"
    meta_path = RESULT_DIR / f"dssm_{scene}_{model_tag}_item_meta.json"
    mmap_arr = np.memmap(emb_path, dtype="float32", mode="w+", shape=(n_items, HIDDEN_DIM))

    model.eval()
    with torch.no_grad():
        for start in tqdm(range(0, n_items, batch_size), desc=f"Export {scene}_{model_tag} item emb"):
            end = min(start + batch_size, n_items)
            batch_ids = note_ids[start:end]
            batch_items = [item_meta[i] for i in batch_ids]
            item_batch = {
                "note_idx": torch.tensor([x["note_idx"] for x in batch_items], dtype=torch.long, device=DEVICE),
                "note_type": torch.tensor([x["note_type"] for x in batch_items], dtype=torch.long, device=DEVICE),
                "taxonomy": torch.tensor(np.stack([x["taxonomy"] for x in batch_items]), dtype=torch.long, device=DEVICE),
                "dense_stats": torch.tensor(np.stack([x["dense_stats"] for x in batch_items]), dtype=torch.float32, device=DEVICE),
            }
            txt_vec = stores["note_text"].get_batch_vecs(batch_ids).to(DEVICE, non_blocking=True)
            img_vec = stores["note_img"].get_batch_vecs(batch_ids).to(DEVICE, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=False):
                item_vec = model.forward_item(item_batch, txt_vec, img_vec)
            mmap_arr[start:end] = item_vec.detach().float().cpu().numpy()

    mmap_arr.flush()
    with open(map_path, "w") as f:
        json.dump({str(nid): i for i, nid in enumerate(note_ids)}, f)
    with open(meta_path, "w") as f:
        json.dump({"num_items": n_items, "dim": HIDDEN_DIM, "dtype": "float32"}, f)
    logger.info(f"Item embeddings exported: {emb_path} / {map_path}")


def export_search_request_embeddings(model, stores, model_tag, batch_size=2048):
    """
    仅 search 场景使用：
    导出 request/user tower 向量（供 ANN 查询）到 train/test 两个 split。
    - dssm_search_{tag}_{split}_query_emb.bin
    - dssm_search_{tag}_{split}_query_map.json
    - dssm_search_{tag}_{split}_query_meta.json
    """
    req_col = "search_idx"
    use_cols = [
        req_col,
        "user_idx",
        "gender_enc",
        "platform_enc",
        "age_enc",
        "location_enc",
        "fans_num",
        "follows_num",
        "recent_clicked_note_idxs",
    ] + [f"dense_feat{i}" for i in range(1, 41)]

    for split in ["train", "test"]:
        feat_path = FEAT_DIR / f"search_{split}_features.parquet"
        if not feat_path.exists():
            logger.warning(f"Missing {feat_path}, skip export search request emb for split={split}.")
            continue

        req_df = pd.read_parquet(feat_path, columns=use_cols)
        req_df["recent_clicked_note_idxs"] = req_df["recent_clicked_note_idxs"].apply(
            lambda x: x.tolist() if isinstance(x, np.ndarray) else (x if isinstance(x, list) else [])
        )
        req_df = req_df.drop_duplicates(subset=[req_col]).reset_index(drop=True)
        n_req = len(req_df)
        if n_req == 0:
            logger.warning(f"No requests in {feat_path}, skip export.")
            continue

        emb_path = RESULT_DIR / f"dssm_search_{model_tag}_{split}_query_emb.bin"
        map_path = RESULT_DIR / f"dssm_search_{model_tag}_{split}_query_map.json"
        meta_path = RESULT_DIR / f"dssm_search_{model_tag}_{split}_query_meta.json"
        mmap_arr = np.memmap(emb_path, dtype="float32", mode="w+", shape=(n_req, HIDDEN_DIM))

        model.eval()
        with torch.no_grad():
            for start in tqdm(range(0, n_req, batch_size), desc=f"Export search_{split}_{model_tag} req emb"):
                end = min(start + batch_size, n_req)
                batch_df = req_df.iloc[start:end]
                req_ids = batch_df[req_col].to_numpy(np.int64)

                hist_vecs = []
                for ids in batch_df["recent_clicked_note_idxs"].tolist():
                    if len(ids) > 0:
                        vecs = stores["note_text"].get_batch_vecs(ids)
                        hist_vecs.append(vecs.mean(dim=0))
                    else:
                        hist_vecs.append(torch.zeros(EMB_DIM))
                hist_vecs = torch.stack(hist_vecs).to(DEVICE, non_blocking=True)

                user_batch = {
                    "user_idx": torch.tensor(batch_df["user_idx"].to_numpy(np.int64), dtype=torch.long, device=DEVICE),
                    "gender_enc": torch.tensor(batch_df["gender_enc"].to_numpy(np.int64), dtype=torch.long, device=DEVICE),
                    "platform_enc": torch.tensor(batch_df["platform_enc"].to_numpy(np.int64), dtype=torch.long, device=DEVICE),
                    "age_enc": torch.tensor(batch_df["age_enc"].to_numpy(np.int64), dtype=torch.long, device=DEVICE),
                    "location_enc": torch.tensor(batch_df["location_enc"].to_numpy(np.int64), dtype=torch.long, device=DEVICE),
                    "fans_num": torch.tensor(batch_df["fans_num"].to_numpy(np.float32), dtype=torch.float32, device=DEVICE),
                    "follows_num": torch.tensor(batch_df["follows_num"].to_numpy(np.float32), dtype=torch.float32, device=DEVICE),
                    "dense_feats": torch.tensor(
                        batch_df[[f"dense_feat{i}" for i in range(1, 41)]].to_numpy(np.float32),
                        dtype=torch.float32,
                        device=DEVICE,
                    ),
                    "query_vec": stores["search_query"].get_batch_vecs(req_ids.tolist()).to(DEVICE, non_blocking=True),
                }
                with torch.amp.autocast("cuda", enabled=False):
                    req_vec = model.forward_user(user_batch, hist_vecs)
                mmap_arr[start:end] = req_vec.detach().float().cpu().numpy()

        mmap_arr.flush()
        with open(map_path, "w") as f:
            json.dump({str(int(rid)): i for i, rid in enumerate(req_df[req_col].tolist())}, f)
        with open(meta_path, "w") as f:
            json.dump(
                {
                    "scene": "search",
                    "split": split,
                    "num_queries": n_req,
                    "dim": HIDDEN_DIM,
                    "dtype": "float32",
                    "req_col": req_col,
                },
                f,
                indent=2,
            )
        logger.info(f"Search request embeddings exported: {emb_path} / {map_path}")


def save_tower_checkpoints(
    state_dict: dict[str, torch.Tensor],
    scene: str,
    model_tag: str,
    cat_vocabs: dict[str, int],
    item_dense_dim: int,
    user_dense_dim: int,
) -> None:
    full_path = MODEL_DIR / f"dssm_{scene}_{model_tag}.pt"
    user_path = MODEL_DIR / f"dssm_{scene}_{model_tag}_user_tower.pt"
    item_path = MODEL_DIR / f"dssm_{scene}_{model_tag}_item_tower.pt"

    torch.save(state_dict, full_path)

    user_state = {
        k: v
        for k, v in state_dict.items()
        if k.startswith("u_emb_") or k.startswith("user_mlp")
    }
    item_state = {
        k: v
        for k, v in state_dict.items()
        if k.startswith("i_emb_") or k.startswith("item_mlp")
    }

    common_meta = {
        "scene": scene,
        "tag": model_tag,
        "hidden_dim": HIDDEN_DIM,
        "emb_dim": EMB_DIM,
        "cat_vocabs": cat_vocabs,
        "user_dense_dim": user_dense_dim,
        "item_dense_dim": item_dense_dim,
    }
    torch.save({"state_dict": user_state, "meta": common_meta}, user_path)
    torch.save({"state_dict": item_state, "meta": common_meta}, item_path)
    logger.info(f"Tower checkpoints saved: {full_path}, {user_path}, {item_path}")


def main(
    scene: str,
    neg_mode: str = "auto",
    hard_neg_path: Path | None = None,
    num_neg_per_pos: int = NUM_NEG_PER_POS,
):
    set_seed()
    assert scene in {"search", "rec"}, "scene 只能是 search 或 rec"
    assert neg_mode in {"auto", "easy", "hard"}, "neg_mode 只能是 auto/easy/hard"

    req_col = "search_idx" if scene == "search" else "request_idx"

    # 读取数据
    file_name = f"{scene}_train_features.parquet"
    df_all = pd.read_parquet(FEAT_DIR / file_name)
    logger.info(f"Loaded {file_name}, shape={df_all.shape}")

    # 正样本（click=1）用于三元组中的 anchor-positive，负样本来自全局物料采样
    df = df_all[df_all["click"] == 1].copy()
    logger.info(f"Positive samples after filter: {df.shape}")

    # item 列定义
    count_cols = [
        "imp_num",
        "click_num",
        "like_num",
        "collect_num",
        "comment_num",
        "share_num",
        "accum_like_num",
        "accum_collect_num",
        "accum_comment_num",
        "view_time",
        "valid_view_times",
        "full_view_times",
        f"imp_{scene}_num",
        f"click_{scene}_num",
        f"{scene}_like_num",
        f"{scene}_collect_num",
        f"{scene}_comment_num",
        f"{scene}_share_num",
        f"{scene}_follow_num",
        f"{scene}_view_time",
    ]

    rate_cols = [
        "ctr",
        "like_rate",
        "collect_rate",
        "comment_rate",
        "share_rate",
        "valid_view_time_rate",
        "full_view_time_rate",
        f"{scene}_ctr",
        f"{scene}_like_rate",
        f"{scene}_collect_rate",
        f"{scene}_share_rate",
        f"{scene}_follow_rate",
        f"{scene}_view_time_rate",
    ]

    # 基础计时/长度等也是计数类，按 log1p + z-score
    item_basic_cols = ["video_duration", "video_height", "video_width", "image_num", "content_length", "commercial_flag"]

    dense_feat_cols = [f"dense_feat{i}" for i in range(1, 41)]
    item_dense_cols = item_basic_cols + count_cols + rate_cols
    user_norm_cols = ["fans_num", "follows_num"] + dense_feat_cols

    # 需要 log1p 的列：计数类 + 用户计数 + item_basic
    log_cols = list(dict.fromkeys(count_cols + item_basic_cols + ["fans_num", "follows_num"]))
    # 需要 z-score 的列：计数（含 log 后）、比率、dense_feat
    norm_cols = list(dict.fromkeys(item_dense_cols + user_norm_cols + rate_cols))

    # 填充缺失
    df_all["recent_clicked_note_idxs"] = df_all["recent_clicked_note_idxs"].apply(
        lambda x: x.tolist() if isinstance(x, np.ndarray) else (x if isinstance(x, list) else [])
    )
    df_all[dense_feat_cols] = df_all[dense_feat_cols].fillna(0)
    df_all[item_dense_cols] = df_all[item_dense_cols].fillna(0)
    df_all[user_norm_cols] = df_all[user_norm_cols].fillna(0)
    for c in HARD_NEG_CANDIDATE_COLS:
        if c in df_all.columns:
            df_all[c] = df_all[c].apply(
                lambda x: x.tolist() if isinstance(x, np.ndarray) else (x if isinstance(x, list) else [])
            )
    df_all = attach_external_hard_neg(
        df_all=df_all,
        scene=scene,
        req_col=req_col,
        hard_neg_path=hard_neg_path,
    )
    resolved_neg_mode, hard_neg_col = resolve_neg_mode(df_all, neg_mode)
    model_tag = "hard" if resolved_neg_mode == "hard" else "easy"
    logger.info(f"Negative mode: requested={neg_mode}, resolved={resolved_neg_mode}, hard_col={hard_neg_col}")

    # 负样本采样分布（Easy Neg）：基于原始 imp_num，优先使用 click=0
    neg_df_raw = df_all[df_all["click"] == 0]
    if len(neg_df_raw) == 0:
        neg_df_raw = df_all  # 兜底
    neg_item_counts = neg_df_raw.groupby("note_idx")["imp_num"].sum().fillna(0) + 1
    item_pool = neg_item_counts.index.tolist()
    neg_sampling_probs = torch.tensor(np.power(neg_item_counts.values, 0.75), dtype=torch.float32)
    neg_sampling_probs = torch.clamp(neg_sampling_probs, min=1e-8)
    neg_sampling_probs /= neg_sampling_probs.sum()

    # 归一化 (用全量数据估计均值方差，再切分正样本子集)
    df_all, _ = log1p_and_zscore(df_all, cols_log=log_cols, cols_norm=norm_cols)
    df = df_all[df_all["click"] == 1].copy()

    # 训练/验证划分：按 session_idx 分组切分
    train_df, val_df = split_by_session_idx(df, val_ratio=0.2, seed=SEED)
    logger.info(f"Train {train_df.shape}, Val {val_df.shape}")

    # 向量内存索引
    stores = {
        "note_text": MMapStore("note_text_emb", "note_text"),
        "note_img": MMapStore("note_img_emb", "note_img"),
    }
    if scene == "search":
        stores["search_query"] = MMapStore("query_text_emb", "search_query")

    # item_meta 构建：用特征表填充已知统计特征，并扩展到全量 note 向量集合
    full_note_ids = sorted({
        int(k)
        for k in set(stores["note_text"].id2pos.keys()) | set(stores["note_img"].id2pos.keys())
    })
    item_meta = build_item_meta(df_all, scene, item_dense_cols, full_note_ids=full_note_ids)
    logger.info(
        "Item meta coverage: feature_items=%d, full_items=%d",
        int(df_all["note_idx"].nunique()),
        int(len(item_meta)),
    )

    # 将全量 note 并入负采样池：未出现在训练数据中的 note 以 count=1 入库，
    # 使它们在训练中也能获得梯度更新（note_idx embedding 不再是随机初始化值）
    known_neg_ids: set[int] = {int(x) for x in item_pool}
    extra_ids = [nid for nid in full_note_ids if nid not in known_neg_ids]
    if extra_ids:
        extra_series = pd.Series(1.0, index=extra_ids, dtype=np.float32)
        neg_item_counts_ext = pd.concat([neg_item_counts, extra_series])
        item_pool = neg_item_counts_ext.index.tolist()
        neg_sampling_probs = torch.tensor(
            np.power(neg_item_counts_ext.values, 0.75), dtype=torch.float32
        )
        neg_sampling_probs = torch.clamp(neg_sampling_probs, min=1e-8)
        neg_sampling_probs /= neg_sampling_probs.sum()
        logger.info(
            "Extended neg pool: from_train=%d → full_corpus=%d (+%d from full_note_ids)",
            len(known_neg_ids),
            len(item_pool),
            len(extra_ids),
        )

    # 词表尺寸
    cat_sizes = load_category_sizes()
    cat_vocabs = {
        "user_idx": int(train_df["user_idx"].max()) + 1,
        "note_idx": int(max(train_df["note_idx"].max() + 1, (max(full_note_ids) + 1) if full_note_ids else 0, len(stores["note_text"].id2pos))),
        "gender": max(cat_sizes.get("gender", 0), int(train_df["gender_enc"].max()) + 1),
        "platform": max(cat_sizes.get("platform", 0), int(train_df["platform_enc"].max()) + 1),
        "age": max(cat_sizes.get("age", 0), int(train_df["age_enc"].max()) + 1),
        "location": max(cat_sizes.get("location", 0), int(train_df["location_enc"].max()) + 1),
        "tax1": max(cat_sizes.get("tax1", 0), int(train_df["taxonomy1_id_enc"].max()) + 1),
        "tax2": max(cat_sizes.get("tax2", 0), int(train_df["taxonomy2_id_enc"].max()) + 1),
        "tax3": max(cat_sizes.get("tax3", 0), int(train_df["taxonomy3_id_enc"].max()) + 1),
    }
    # DataLoader
    collate_fn = collate_fn_builder(stores, scene=scene)
    train_loader = DataLoader(
        DSSMDataset(
            scene=scene,
            req_col=req_col,
            df=train_df,
            item_meta=item_meta,
            item_pool=item_pool,
            neg_sampling_probs=neg_sampling_probs,
            use_hard_neg=(resolved_neg_mode == "hard"),
            hard_neg_col=hard_neg_col,
            hard_neg_ratio=HARD_NEG_RATIO,
            num_neg_per_pos=num_neg_per_pos,
            use_inbatch_neg=True,
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
        prefetch_factor=4,
    )
    val_loader = DataLoader(
        DSSMDataset(
            scene=scene,
            req_col=req_col,
            df=val_df,
            item_meta=item_meta,
            item_pool=item_pool,
            neg_sampling_probs=neg_sampling_probs,
            use_hard_neg=(resolved_neg_mode == "hard"),
            hard_neg_col=hard_neg_col,
            hard_neg_ratio=HARD_NEG_RATIO,
            num_neg_per_pos=num_neg_per_pos,
            use_inbatch_neg=True,
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
        prefetch_factor=2,
    )

    user_dense_dim = 2 + len(dense_feat_cols) + EMB_DIM  # fans/follows + 40 + hist
    if scene == "search":
        user_dense_dim += EMB_DIM  # + query_text_emb
    item_dense_dim = len(item_dense_cols)

    model = DSSMModel(scene, cat_vocabs, item_dense_dim, user_dense_dim).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")

    # TensorBoard
    log_dir = LOG_DIR / f"dssm_{scene}_{model_tag}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    best_auc = 0.0
    best_epoch = -1
    best_state_dict = None
    patience = EARLY_STOP
    global_step = 0

    for epoch in range(EPOCHS):
        (
            avg_loss,
            train_auc,
            mean_pos_sim,
            mean_neg_sim,
            global_step,
        ) = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            writer,
            global_step,
            epoch + 1,
            EPOCHS,
        )

        writer.add_scalar("train/epoch_loss", avg_loss, epoch)

        # 验证
        val_loss, val_auc = evaluate(model, val_loader)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        # 在同一张图中记录 train/val 曲线
        writer.add_scalars("loss", {"train": avg_loss, "val": val_loss}, epoch)
        writer.add_scalars("auc", {"train": train_auc, "val": val_auc}, epoch)
        writer.add_scalar("val/auc", val_auc, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("train/auc", train_auc, epoch)
        writer.add_scalar("train/mean_pos_sim", mean_pos_sim, epoch)
        writer.add_scalar("train/mean_neg_sim", mean_neg_sim, epoch)
        writer.add_scalar("train/lr", current_lr, epoch)
        msg = (
            f"Epoch {epoch+1}: "
            f"train_loss={avg_loss:.4f} val_loss={val_loss:.4f}\n "
            f"train_auc={train_auc:.4f} val_auc={val_auc:.4f}\n "
            f"mean_pos_sim={mean_pos_sim:.4f} mean_neg_sim={mean_neg_sim:.4f} lr={current_lr:.6g}"
        )
        logger.info(msg)

        # 早停
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch + 1
            # 在内存中更新最佳权重，训练结束后统一保存
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = EARLY_STOP
        else:
            patience -= 1
            if patience <= 0:
                logger.info("Early stopping triggered.")
                break

    if best_state_dict is not None:
        save_tower_checkpoints(
            state_dict=best_state_dict,
            scene=scene,
            model_tag=model_tag,
            cat_vocabs=cat_vocabs,
            item_dense_dim=item_dense_dim,
            user_dense_dim=user_dense_dim,
        )
        logger.info(
            f"Best model saved after training: dssm_{scene}_{model_tag}.pt "
            f"(best_epoch={best_epoch}, best_auc={best_auc:.4f})"
        )
        model.load_state_dict(best_state_dict)
        model.to(DEVICE)
        export_item_embeddings(model, item_meta, stores, scene, model_tag)  # 导出 item embedding 供 Faiss 使用
        if scene == "search" and "search_query" in stores:
            export_search_request_embeddings(model, stores, model_tag=model_tag)

    writer.close()
    logger.info(f"Training done. Best AUC={best_auc:.4f}")


# =============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=["search", "rec"], required=True)
    parser.add_argument("--neg-mode", choices=["auto", "easy", "hard"], default="auto")
    parser.add_argument("--hard-neg-path", type=Path, default=None, help="外部 hard neg 文件路径（通常来自 easy 阶段排序淘汰导出）")
    parser.add_argument("--num-neg-per-pos", type=int, default=NUM_NEG_PER_POS, help="每个正样本采样的负样本数量")
    args = parser.parse_args()
    main(args.scene, args.neg_mode, args.hard_neg_path, args.num_neg_per_pos)
