"""
本脚本用于生成笔记 (Note) 的图像 Embedding 特征。
    - 遍历笔记关联的所有图片，使用 SigLIP 模型提取图像特征。
    - 对多张图片特征进行 Mean Pooling（均值池化），得到笔记级别的单一图像向量。
    - 向量维度为 768，并以 float16 精度保存。

数据流向:
    - 输入: DATA_DIR / notes/*.parquet (获取 note_idx 及 image_path 列表)。
    - 依赖: 磁盘存储的原始图片文件（假设图片已经下载并解压到 “Qilin/image” 路径下）。
        图片下载地址：https://cloud.tsinghua.edu.cn/d/af72ab5dbba1460da6c0/
    - 输出: OUT_DIR / note_img_emb/ 下的分片 Parquet 文件。

使用示例:
    uv run python build_note_image_emb.py   # 1,071,532 note_idx
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel

# ======================
# Config
# ======================
BASE_DIR = Path(__file__).resolve().parents[2]  # Qilin/
DATA_DIR = BASE_DIR / "datasets"
IMG_ROOT = BASE_DIR
OUT_DIR = BASE_DIR / "embeddings" / "note_img_emb"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "google/siglip-base-patch16-224"   # 768 维
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_BATCH_SIZE = 512    # ~40h
NOTE_SHARD_SIZE = 10000
EMB_DIM = 768
DTYPE = np.float16

# ======================
# Load model
# ======================
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model.eval()

# ======================
def load_image(path: Path):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None

def encode_images(imgs):
    inputs = processor(images=imgs, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().astype(DTYPE)

# ======================
def find_image_paths(note_idx: int) -> list[Path]:
    """根据note_idx找到对应的图片文件路径"""
    # 图片按note_idx分组：part_XX/YYZZZZ/note_idx.jpg
    # 其中XX = note_idx // 10000, YYZZZZ = note_idx // 1000
    part = note_idx // 10000
    subdir = note_idx // 1000
    img_dir = IMG_ROOT / "image" / f"part_{part}" / str(subdir)
    
    if not img_dir.exists():
        return []
    
    img_file = img_dir / f"{note_idx}.jpg"
    if img_file.exists():
        return [img_file]
    
    # 有些笔记可能有多张图片，检查是否有其他文件
    # 但通常每篇笔记只有一张图片，所以这里简化处理
    return []

# ======================
def main():
    print("📥 Loading notes metadata...")
    notes = pd.concat(
        [
            pd.read_parquet(p, columns=["note_idx", "image_path", "image_num"])
            for p in DATA_DIR.glob("notes/*.parquet")
        ],
        ignore_index=True,
    )

    notes = notes.reset_index(drop=True)
    n_notes = len(notes)
    n_shards = (n_notes + NOTE_SHARD_SIZE - 1) // NOTE_SHARD_SIZE

    print(f"🧱 Total notes: {n_notes}")
    print(f"🧩 Shards: {n_shards}, shard_size={NOTE_SHARD_SIZE}")

    # 统计有多少笔记有图片
    has_images = notes['image_num'] > 0
    print(f"📸 Notes with images: {has_images.sum()} ({100*has_images.sum()/n_notes:.1f}%)")

    for shard_id in range(n_shards):
        out_file = OUT_DIR / f"part-{shard_id:05d}.parquet"
        tmp_file = OUT_DIR / f"part-{shard_id:05d}.parquet.tmp"
        if out_file.exists():
            continue

        start = shard_id * NOTE_SHARD_SIZE
        end = min((shard_id + 1) * NOTE_SHARD_SIZE, n_notes)
        shard_notes = notes.iloc[start:end]

        rows = []

        for _, row in tqdm(
            shard_notes.iterrows(),
            total=len(shard_notes),
            desc=f"Shard {shard_id}",
        ):
            note_idx = int(row.note_idx)
            imgs = []
            
            # 优先使用image_path字段中的路径
            if len(row.image_path) > 0:
                for p in row.image_path:
                    img = load_image(IMG_ROOT / p)
                    if img:
                        imgs.append(img)
            # 如果没有image_path但image_num > 0，尝试从文件系统查找
            elif row.image_num > 0:
                img_paths = find_image_paths(note_idx)
                for p in img_paths:
                    img = load_image(p)
                    if img:
                        imgs.append(img)

            # 如果没有找到图片，使用零向量
            if not imgs:
                note_emb = np.zeros(EMB_DIM, dtype=DTYPE)
            else:
                embs = []
                for i in range(0, len(imgs), IMG_BATCH_SIZE):
                    embs.append(encode_images(imgs[i:i + IMG_BATCH_SIZE]))
                note_emb = np.vstack(embs).mean(axis=0) # mean pooling

            rows.append({
                "note_idx": note_idx,
                "note_img_emb": note_emb,
            })

        if rows:
            df = pd.DataFrame(rows)
            df.to_parquet(tmp_file, index=False)
            tmp_file.rename(out_file)
            print(f"✅ Written {out_file}")
        else:
            print(f"⚠️ Empty shard {shard_id}, skipped")

    print("🎉 All done.")

# ======================
if __name__ == "__main__":
    main()
