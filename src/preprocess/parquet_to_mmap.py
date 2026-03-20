"""
本脚本用于将离线生成的 Parquet 格式 Embedding 转换为高性能的二进制内存映射文件 (np.memmap)。

主要功能：
    1. 全局扫描：遍历所有分片 Parquet 文件，提取 Key 并进行全局去重。
    2. 维度推断：支持单向量 (如 note_emb) 和序列向量 (如 seq_embs) 的维度识别与对齐。
    3. 精度转换：将数据统一转换为 float16 存储，以平衡磁盘空间和读取性能。
    4. 索引构建：生成 ID 到文件偏移量的映射表 (JSON)，实现 O(1) 级别的特征检索。

数据流向：
    - 输入：embeddings/ 目录下各场景的 *.parquet 分片。
    - 输出：
        - {name}.bin: 原始二进制特征矩阵，支持 OS 级别的 Page Cache。
        - {name}_map.json: 存储 ID -> Matrix Row Index 的映射关系。

使用价值：
    - 解决大规模 Embedding 载入内存慢（几分钟变几秒）和内存占用高（按需读取）的问题。
    - uv run python parquet_to_mmap.py
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm

def convert_to_mmap(input_dir, key_col, emb_col, output_name, prefix=None, output_dir=None):
    """
    prefix: 用于筛选文件名，如 'search' 或 'rec'
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir) if output_dir else input_path
    output_path.mkdir(parents=True, exist_ok=True)
    
    pattern = f"{prefix}*.parquet" if prefix else "*.parquet"
    files = sorted(list(input_path.glob(pattern)))
    
    if not files:
        print(f"❌ No parquet files matching '{pattern}' found in {input_dir}")
        return

    # 1. 第一次遍历：全局扫描 Keys 并去重
    print(f"🔍 [Scanning] {output_name} (prefix: {prefix or 'All'})")
    global_id_to_pos = {}
    curr_unique_idx = 0
    
    # 获取维度
    first_df = pd.read_parquet(files[0])
    sample_emb = np.array(first_df[emb_col].iloc[0].tolist(), dtype='float16')
    emb_shape = sample_emb.shape 
    
    for f in tqdm(files, desc="Scanning Keys"):
        keys = pd.read_parquet(f, columns=[key_col])[key_col].values
        for k in keys:
            k_int = int(k)
            if k_int not in global_id_to_pos:
                global_id_to_pos[k_int] = curr_unique_idx
                curr_unique_idx += 1
    
    total_unique = len(global_id_to_pos)
    final_shape = (total_unique, *emb_shape)
    print(f"✅ Found {total_unique} unique keys. Shape: {final_shape}")

    # 2. 创建内存映射文件
    bin_file = output_path / f"{output_name}.bin"
    mmap_array = np.memmap(bin_file, dtype='float16', mode='w+', shape=final_shape)

    # 3. 第二次遍历：填充数据
    written_keys = set()
    for f in tqdm(files, desc="Filling Data"):
        df = pd.read_parquet(f, columns=[key_col, emb_col])
        keys = df[key_col].values
        embs = df[emb_col].values
        
        for k, emb in zip(keys, embs):
            k_int = int(k)
            if k_int not in written_keys:
                pos = global_id_to_pos[k_int]
                data_to_write = np.array(emb.tolist(), dtype='float16')
                
                if data_to_write.shape != emb_shape:
                    new_data = np.zeros(emb_shape, dtype='float16')
                    copy_len = min(data_to_write.shape[0], emb_shape[0])
                    new_data[:copy_len] = data_to_write[:copy_len]
                    mmap_array[pos] = new_data
                else:
                    mmap_array[pos] = data_to_write
                    
                written_keys.add(k_int)
        del df

    mmap_array.flush()
    
    # 保存索引
    map_file = output_path / f"{output_name}_map.json"
    with open(map_file, 'w') as f:
        json.dump(global_id_to_pos, f)
        
    print(f"✨ Saved: {output_name}\n")


# =============================
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[2]  # Qilin/
    QUERY_DIR = BASE_DIR / "embeddings/query_text_emb"
    
    # --- 1. 处理 Search 场景 ---
    # A. Search 场景的用户历史序列 (seq_embs)
    convert_to_mmap(
        input_dir=QUERY_DIR,
        prefix="search",
        key_col="search_idx",
        emb_col="seq_embs",
        output_name="search_seq"
    )
    # B. Search 场景的 Query 文本向量 (query_emb)
    convert_to_mmap(
        input_dir=QUERY_DIR,
        prefix="search",
        key_col="search_idx",
        emb_col="query_emb",
        output_name="search_query"
    )

    # --- 2. 处理 Rec 场景 ---
    # Rec 场景只有用户历史序列 (seq_embs)
    convert_to_mmap(
        input_dir=QUERY_DIR,
        prefix="rec",
        key_col="request_idx",
        emb_col="seq_embs",
        output_name="rec_seq"
    )

    # --- 3. 处理笔记向量 ---
    convert_to_mmap(
        input_dir=BASE_DIR / "embeddings/note_text_emb",
        key_col="note_idx",
        emb_col="note_text_emb",
        output_name="note_text"
    )
    convert_to_mmap(
        input_dir=BASE_DIR / "embeddings/note_img_emb",
        key_col="note_idx",
        emb_col="note_img_emb",
        output_name="note_img"
    )