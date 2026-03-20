"""
本脚本用于生成搜索 (Search) 和推荐 (Rec) 场景下的 Query Embedding 特征。
    1. Search 场景: 同时生成 Query 的文本向量 (query_emb) 和用户近期点击笔记的序列向量列表 (seq_embs)。
    2. Rec 场景: 仅生成用户近期点击笔记的序列向量列表 (seq_embs)。

数据流向:
    - 输入: DATA_DIR 下的 {scene}_train_samples.parquet (包含 query 或 recent_clicked_note_idxs)。
    - 依赖: NOTE_TEXT_EMB_DIR 下的笔记 Embedding 库（用于匹配点击序列）。
    - 输出: OUT_DIR 下的分片 Parquet 文件。
    
使用示例:
    uv run python build_query_text_emb.py --scene search    # 43,752 unique search_idx
    uv run python build_query_text_emb.py --scene rec       # 83,437 unique request_idx
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# ======================
# Config
# ======================
BASE_DIR = Path(__file__).resolve().parents[2]  # Qilin/
DATA_DIR = BASE_DIR / "data"
NOTE_TEXT_EMB_DIR = BASE_DIR / "embeddings" / "note_text_emb"
OUT_DIR = BASE_DIR / "embeddings" / "query_text_emb"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "BAAI/bge-base-zh"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 256
MAX_LEN = 512
SHARD_SIZE = 10000
HIDDEN_DIM = 768
SEQ_LEN = 20

# ======================
# Load model
# ======================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

# ======================
# Load note_text_emb
# ======================
def load_note_text_emb():
    print("[Load] note_text_emb")
    dfs = []
    for p in NOTE_TEXT_EMB_DIR.glob("*.parquet"):
        dfs.append(pd.read_parquet(p))
    df = pd.concat(dfs, ignore_index=True)

    emb_map = {
        int(r.note_idx): np.asarray(r.note_text_emb, dtype="float16")
        for r in df.itertuples()
    }
    print(f"[OK] loaded {len(emb_map)} note embeddings")
    return emb_map

NOTE_EMB_MAP = None


def _resolve_train_sample_path(scene: str) -> Path:
    new_path = DATA_DIR / f"{scene}_train_samples.parquet"
    if not new_path.exists():
        raise FileNotFoundError(f"Missing sample file: {new_path}")
    return new_path

# ======================
# Text builder
# ======================
def build_search_text(query: str) -> str:
    if not isinstance(query, str):
        return ""
    return query.strip()

# ======================
# Encoding (search)
# ======================
@torch.no_grad()
def encode(texts):
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    ).to(DEVICE)

    out = model(**inputs).last_hidden_state[:, 0]
    out = out / out.norm(dim=-1, keepdim=True)
    return out.cpu().numpy().astype("float16")

# ======================
# Main
# ======================
def main(scene: str):
    assert scene in ["search", "rec"]
    group_key = "search_idx" if scene == "search" else "request_idx"

    global NOTE_EMB_MAP
    NOTE_EMB_MAP = load_note_text_emb()
    
    IN_FILE = _resolve_train_sample_path(scene)
    print(f"[Load] {IN_FILE}")
    
    if scene == "search":
        df = pd.read_parquet(IN_FILE, columns=[group_key, "query", "recent_clicked_note_idxs"])
    else:
        df = pd.read_parquet(IN_FILE, columns=[group_key, "recent_clicked_note_idxs"])

    total = len(df)
    num_shards = (total + SHARD_SIZE - 1) // SHARD_SIZE
    print(f"[Info] {total} rows → {num_shards} shards")

    for shard_id in range(num_shards):
        out_file = OUT_DIR / f"{scene}-part-{shard_id:05d}.parquet"
        tmp_file = OUT_DIR / f"{scene}-part-{shard_id:05d}.parquet.tmp"

        if out_file.exists():
            continue

        start = shard_id * SHARD_SIZE
        end = min(start + SHARD_SIZE, total)
        shard = df.iloc[start:end]

        # 去重
        initial_len = len(shard)
        shard = shard.drop_duplicates(subset=[group_key], keep='first')
        if len(shard) < initial_len:
            print(f"  [Pre-process] Shard {shard_id}: Skipped {initial_len - len(shard)} redundant rows.")

        # output lists
        keys = []
        seq_embs_list = []
        query_embs_list = []  # search only

        # ======================
        # 处理逻辑
        # ======================
        for _, row in tqdm(shard.iterrows(), total=len(shard), desc=f"{scene} shard {shard_id}"):
            
            # 1. 处理序列部分 (共同逻辑)
            note_idxs = row["recent_clicked_note_idxs"]
            current_seq_embs = []
            for nid in note_idxs:
                nid = int(nid)
                if nid in NOTE_EMB_MAP:
                    current_seq_embs.append(NOTE_EMB_MAP[nid])
            
            if len(current_seq_embs) == 0:
                fixed_seq_embs = np.zeros((SEQ_LEN, HIDDEN_DIM), dtype=np.float16)
            else:
                tmp_arr = np.array(current_seq_embs, dtype=np.float16)
                curr_len = tmp_arr.shape[0]
                
                if curr_len >= SEQ_LEN:
                    # 截取最新的 SEQ_LEN 个行为
                    fixed_seq_embs = tmp_arr[:SEQ_LEN]
                else:
                    # Zero Padding (补零)
                    pad_width = SEQ_LEN - curr_len
                    padding = np.zeros((pad_width, HIDDEN_DIM), dtype=np.float16)
                    fixed_seq_embs = np.vstack([tmp_arr, padding])

            # 2. 处理 Search 独有的 Query Embedding
            if scene == "search":
                txt = build_search_text(row["query"])
                if not txt: # 如果 query 为空则跳过
                    continue
                query_embs_list.append(txt) 

            keys.append(int(row[group_key]))
            seq_embs_list.append(fixed_seq_embs.tolist())

        # ======================
        # 批量计算并保存
        # ======================
        if scene == "search":
            # 批量编码 query 文本
            all_query_vecs = []
            for i in range(0, len(query_embs_list), BATCH_SIZE):
                batch_txt = query_embs_list[i : i + BATCH_SIZE]
                all_query_vecs.extend(list(encode(batch_txt)))
            
            out_df = pd.DataFrame({
                group_key: keys,
                "query_emb": all_query_vecs,
                "seq_embs": seq_embs_list
            })
        else:
            out_df = pd.DataFrame({
                group_key: keys,
                "seq_embs": seq_embs_list
            })

        out_df.to_parquet(tmp_file, index=False, engine='pyarrow')
        tmp_file.rename(out_file)
        print(f"✅ {scene} committed {out_file}")

    print("🎉 All query embeddings done!")

# ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True, choices=["search", "rec"])
    args = parser.parse_args()

    main(args.scene)
