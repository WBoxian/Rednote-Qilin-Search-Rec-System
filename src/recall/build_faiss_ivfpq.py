"""
使用 DSSM 导出的 item embedding 构建 Faiss IVF-PQ ANN 索引。

默认读取:
    outputs/index/dssm_{scene}_{tag}_item_emb.bin
    outputs/index/dssm_{scene}_{tag}_item_map.json
    outputs/index/dssm_{scene}_{tag}_item_meta.json

输出:
    outputs/index/dssm_{scene}_{tag}_ivfpq.faiss
    outputs/index/dssm_{scene}_{tag}_row2note.npy
    outputs/index/dssm_{scene}_{tag}_ivfpq_meta.json

示例:
    uv run python src/recall/build_faiss_ivfpq.py --scene search --tag easy
    uv run python src/recall/build_faiss_ivfpq.py --scene rec --tag hard --metric ip --nlist 4096 --m 32 --pq-bits 8 --use-gpu
"""

from __future__ import annotations

import argparse
import json
import logging
import numpy as np
from pathlib import Path

try:
    import faiss
except ImportError as e:
    raise SystemExit(
        "faiss 未安装。请安装 faiss-cpu 或 faiss-gpu，例如: uv add faiss-cpu==1.7.4。注意faiss-gpu仅支持cuda 11.4 & 12.1"
    ) from e


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("build_faiss_ivfpq")

BASE_DIR = Path(__file__).resolve().parents[2]  # Qilin/
RESULT_DIR = BASE_DIR / "outputs" / "index"
if not RESULT_DIR.exists():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

def _default_paths(scene: str, tag: str):
    prefix = f"dssm_{scene}_{tag}"
    emb = RESULT_DIR / f"{prefix}_item_emb.bin"
    mp = RESULT_DIR / f"{prefix}_item_map.json"
    meta = RESULT_DIR / f"{prefix}_item_meta.json"
    return emb, mp, meta, prefix


def _load_inputs(
    scene: str,
    tag: str,
    emb_path: Path | None,
    map_path: Path | None,
    emb_meta_path: Path | None,
):
    d_emb, d_map, d_meta, prefix = _default_paths(scene, tag)
    emb_path = emb_path or d_emb
    map_path = map_path or d_map
    emb_meta_path = emb_meta_path or d_meta

    for p in [emb_path, map_path, emb_meta_path]:
        if not p.exists():
            raise FileNotFoundError(f"输入文件不存在: {p}")

    with open(emb_meta_path, "r") as f:
        emb_meta = json.load(f)
    n_items = int(emb_meta["num_items"])
    dim = int(emb_meta["dim"])
    dtype = emb_meta.get("dtype", "float32")
    if dtype != "float32":
        raise ValueError(f"仅支持 float32 embedding，收到: {dtype}")

    emb = np.memmap(emb_path, dtype="float32", mode="r", shape=(n_items, dim))
    with open(map_path, "r") as f:
        note2row = json.load(f)

    row2note = np.empty(n_items, dtype=np.int64)
    for note_id, row_idx in note2row.items():
        row2note[int(row_idx)] = int(note_id)

    return np.asarray(emb), row2note, prefix, emb_meta


def _build_index(
    xb: np.ndarray,
    metric: str,
    m: int,
    nlist: int,
    pq_bits: int,
    train_size: int,
    use_gpu: bool,
):
    if xb.dtype != np.float32:
        xb = xb.astype(np.float32, copy=False)
    n, d = xb.shape
    if n == 0:
        raise ValueError("embedding 为空，无法建索引")

    metric = metric.lower()
    if metric not in {"ip", "l2"}:
        raise ValueError("metric 只能是 ip 或 l2")

    # 兼容不同 faiss 版本：部分版本不暴露 METRIC_* 常量
    metric_ip = getattr(faiss, "METRIC_INNER_PRODUCT", 0)
    metric_l2 = getattr(faiss, "METRIC_L2", 1)
    if metric == "ip":
        faiss_metric = metric_ip
    else:
        faiss_metric = metric_l2

    nlist = max(1, min(nlist, n))
    # 兼容不同 faiss Python 绑定：有些版本不暴露 IndexFlatIP/IndexFlatL2
    if hasattr(faiss, "IndexFlatIP") and hasattr(faiss, "IndexFlatL2"):
        quantizer = (
            faiss.IndexFlatIP(d) if faiss_metric == metric_ip else faiss.IndexFlatL2(d)
        )
    elif hasattr(faiss, "index_factory"):
        q_desc = "Flat"
        try:
            quantizer = faiss.index_factory(d, q_desc, faiss_metric)
        except TypeError:
            quantizer = faiss.index_factory(d, q_desc)
    else:
        raise RuntimeError(
            "当前 faiss 版本缺少 IndexFlat* 与 index_factory，无法构建 IVF-PQ。"
        )
    # 注意参数顺序: (quantizer, d, nlist, m, nbits, metric)
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, pq_bits, faiss_metric)

    # 训练样本子集：优先覆盖 nlist，避免训练不足
    min_train = min(n, max(nlist * 40, 10000))
    train_size = min(n, max(min_train, train_size))
    if train_size < n:
        rng = np.random.default_rng(42)
        train_idx = rng.choice(n, size=train_size, replace=False)
        xt = xb[train_idx]
    else:
        xt = xb

    logger.info(
        f"Start train IVF-PQ: n={n}, d={d}, metric={metric}, nlist={nlist}, m={m}, pq_bits={pq_bits}, train_size={train_size}"
    )

    gpu_resources = None
    if use_gpu:
        gpu_resources = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(gpu_resources, 0, index)
        gpu_index.train(xt)
        gpu_index.add(xb)
        index = faiss.index_gpu_to_cpu(gpu_index)
    else:
        index.train(xt)
        index.add(xb)

    logger.info(f"Index built: ntotal={index.ntotal}")
    return index


def build_faiss_index(
    scene: str,
    tag: str = "easy",
    emb_path: Path | None = None,
    map_path: Path | None = None,
    emb_meta_path: Path | None = None,
    metric: str = "ip",
    m: int = 32,
    nlist: int = 2048,
    pq_bits: int = 8,
    train_size: int = 200000,
    use_gpu: bool = False,
    nprobe: int = 32,
) -> tuple[Path, Path, Path]:
    xb, row2note, prefix, emb_meta = _load_inputs(
        scene=scene,
        tag=tag,
        emb_path=emb_path,
        map_path=map_path,
        emb_meta_path=emb_meta_path,
    )

    index = _build_index(
        xb=xb,
        metric=metric,
        m=m,
        nlist=nlist,
        pq_bits=pq_bits,
        train_size=train_size,
        use_gpu=use_gpu,
    )
    index.nprobe = nprobe

    index_path = RESULT_DIR / f"{prefix}_ivfpq.faiss"
    row2note_path = RESULT_DIR / f"{prefix}_row2note.npy"
    meta_path = RESULT_DIR / f"{prefix}_ivfpq_meta.json"

    faiss.write_index(index, str(index_path))
    np.save(row2note_path, row2note)

    out_meta = {
        "scene": scene,
        "tag": tag,
        "num_items": int(xb.shape[0]),
        "dim": int(xb.shape[1]),
        "metric": metric,
        "index_type": "IVFPQ",
        "m": int(m),
        "nlist": int(nlist),
        "pq_bits": int(pq_bits),
        "nprobe": int(nprobe),
        "embedding_meta": emb_meta,
        "index_file": index_path.name,
        "row2note_file": row2note_path.name,
    }
    with open(meta_path, "w") as f:
        json.dump(out_meta, f, indent=2)

    logger.info(f"Saved index: {index_path}")
    logger.info(f"Saved row2note: {row2note_path}")
    logger.info(f"Saved meta: {meta_path}")
    return index_path, row2note_path, meta_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=["search", "rec"], required=True)
    parser.add_argument("--tag", default="easy", help="DSSM 版本标签，如 easy/hard")
    parser.add_argument("--emb-path", type=Path, default=None)
    parser.add_argument("--map-path", type=Path, default=None)
    parser.add_argument("--emb-meta-path", type=Path, default=None)
    parser.add_argument("--metric", choices=["ip", "l2"], default="ip") # 相似性度量，IP：内积，L2：欧氏距离
    parser.add_argument("--m", type=int, default=32)    # 子向量数量，默认 32，需能被 dim 整除
    parser.add_argument("--nlist", type=int, default=2048)  # IVF 的聚类中心数量，默认 2048，需远小于样本数量
    parser.add_argument("--pq-bits", type=int, default=8)
    parser.add_argument("--train-size", type=int, default=200000)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--nprobe", type=int, default=32, help="写入索引默认查询 nprobe")   # 查询时访问的倒排列表数量，默认 32，需远小于 nlist
    args = parser.parse_args()

    build_faiss_index(
        scene=args.scene,
        tag=args.tag,
        emb_path=args.emb_path,
        map_path=args.map_path,
        emb_meta_path=args.emb_meta_path,
        metric=args.metric,
        m=args.m,
        nlist=args.nlist,
        pq_bits=args.pq_bits,
        train_size=args.train_size,
        use_gpu=args.use_gpu,
        nprobe=args.nprobe,
    )


# =============================
if __name__ == "__main__":
    main()
