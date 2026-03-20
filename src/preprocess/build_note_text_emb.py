"""
本脚本用于生成笔记 (Note) 的文本 Embedding 特征。
    - 整合笔记的标题 (title) 和正文 (content) 形成完整文本。
    - 使用预训练的 BGE 模型将文本编码为 768 维向量。
    - 采用 float16 精度存储以节省显存和磁盘空间。

数据流向:
    - 输入: DATA_DIR / notes/*.parquet (包含 note_idx, note_title, note_content)。
    - 输出: OUT_DIR / note_text_emb/ 下的分片 Parquet 文件。

使用示例:
    uv run python build_note_text_emb.py    # 1,983,938 note_idx
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# ======================
# Config
# ======================
BASE_DIR = Path(__file__).resolve().parents[2]  # Qilin/
DATA_DIR = BASE_DIR / "datasets"
OUT_DIR = BASE_DIR / "embeddings" / "note_text_emb"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "BAAI/bge-base-zh" # 768 维
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 1024   # ~13h
MAX_LEN = 512
SHARD_SIZE = 10000

# ======================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

# ======================
def build_text(title, content):
    title = (title or "").strip()
    content = (content or "").strip()
    if title and content:
        return f"title：{title}\ncontent：{content}"
    elif title:
        return f"title：{title}"
    else:
        return f"content：{content}"

def encode(texts):
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    ).to(DEVICE)
    with torch.no_grad():
        out = model(**inputs).last_hidden_state[:, 0]
        out = out / out.norm(dim=-1, keepdim=True)
    return out.cpu().numpy().astype("float16")

# ======================
def main():
    print("Loading notes metadata...")
    notes = pd.concat(
        [pd.read_parquet(p, columns=["note_idx", "note_title", "note_content"])
         for p in DATA_DIR.glob("notes/*.parquet")],
        ignore_index=True,
    )

    total = len(notes)
    num_shards = (total + SHARD_SIZE - 1) // SHARD_SIZE

    for shard_id in range(num_shards):
        out_file = OUT_DIR / f"part-{shard_id:05d}.parquet"
        tmp_file = OUT_DIR / f"part-{shard_id:05d}.parquet.tmp"

        if out_file.exists():
            continue

        start = shard_id * SHARD_SIZE
        end = min(start + SHARD_SIZE, total)
        shard = notes.iloc[start:end]

        rows = []
        texts = [
            build_text(t, c)
            for t, c in zip(shard.note_title, shard.note_content)
        ]

        embs = []
        for i in tqdm(range(0, len(texts), BATCH_SIZE),
                      desc=f"Shard {shard_id}"):
            embs.append(encode(texts[i:i + BATCH_SIZE]))

        emb = np.vstack(embs)

        df = pd.DataFrame({
            "note_idx": shard.note_idx.values,
            "note_text_emb": list(emb),
        })

        df.to_parquet(tmp_file, index=False)
        tmp_file.rename(out_file)
        print(f"✅ Committed {out_file}")

    print("🎉 All text embeddings done")

# ======================
if __name__ == "__main__":
    main()
