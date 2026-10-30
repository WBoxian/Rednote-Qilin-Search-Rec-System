"""离线/在线共用的静态物品特征；仅使用显式提供的训练词表。"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import pickle
import threading

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ITEM_NUMERIC_FEATURES = (
    "note_type", "video_duration", "video_height", "video_width", "image_num",
    "content_length", "commercial_flag",
)
ITEM_CATEGORY_FEATURES = ("taxonomy1_id", "taxonomy2_id", "taxonomy3_id")
STATIC_ITEM_FEATURES = ITEM_NUMERIC_FEATURES + tuple(name + "_enc" for name in ITEM_CATEGORY_FEATURES)


def encode_static_items(frame: pd.DataFrame, vocabularies: dict) -> pd.DataFrame:
    """不拟合词表、不读取标签/位置/行为统计；未知类别与原训练预处理一样编码为 0。"""
    required = {"note_idx", *ITEM_NUMERIC_FEATURES, *ITEM_CATEGORY_FEATURES}
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError(f"Missing static item columns: {missing}")
    out = pd.DataFrame({"note_idx": frame["note_idx"].to_numpy(dtype=np.int64)})
    for name in ITEM_NUMERIC_FEATURES:
        out[name] = pd.to_numeric(frame[name], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float32)
    for name in ITEM_CATEGORY_FEATURES:
        raw = frame[name].astype("string")
        out[name + "_enc"] = raw.map(vocabularies[name]).fillna(0).to_numpy(np.float32)
    return out


class StaticItemFeatureStore:
    """约 40 字节特征+8 字节 ID/物品；不驻留标题、正文或原始类别字符串。"""

    def __init__(self, notes_dir: Path, vocab_dir: Path):
        self.input_files = tuple(sorted(Path(notes_dir).glob("*.parquet")))
        if not self.input_files:
            raise FileNotFoundError(f"No item parquet files: {notes_dir}")
        vocab_files = tuple(Path(vocab_dir) / (name + ".pkl") for name in ITEM_CATEGORY_FEATURES)
        vocabularies = {}
        for name, path in zip(ITEM_CATEGORY_FEATURES, vocab_files):
            # 仅加载本机受信任训练产物，缺失时禁止从全目录重建词表。
            with path.open("rb") as stream:
                vocabularies[name] = pickle.load(stream)
        self.input_files += vocab_files
        columns = ["note_idx", *ITEM_NUMERIC_FEATURES, *ITEM_CATEGORY_FEATURES]
        chunks = []
        for path in self.input_files[:-len(vocab_files)]:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=65536, columns=columns):
                chunks.append(encode_static_items(batch.to_pandas(), vocabularies))
        encoded = pd.concat(chunks, ignore_index=True)
        if encoded.empty or encoded["note_idx"].isna().any() or (encoded["note_idx"] < 0).any() or encoded["note_idx"].duplicated().any():
            raise ValueError("Item catalog requires unique nonnegative note_idx")
        order = np.argsort(encoded["note_idx"].to_numpy(), kind="stable")
        self.note_ids = encoded["note_idx"].to_numpy(np.int64)[order]
        self.values = encoded[list(STATIC_ITEM_FEATURES)].to_numpy(np.float32)[order]
        self.note_ids.flags.writeable = False
        self.values.flags.writeable = False

    @property
    def nbytes(self) -> int:
        return self.note_ids.nbytes + self.values.nbytes

    def get(self, note_ids) -> pd.DataFrame:
        raw_ids = np.asarray(note_ids)
        ids = np.asarray(note_ids, dtype=np.int64)
        if not np.array_equal(raw_ids, ids):
            raise ValueError("note_ids must be integral IDs")
        if ids.ndim != 1:
            raise ValueError("note_ids must be one-dimensional")
        positions = np.searchsorted(self.note_ids, ids)
        inside = positions < len(self.note_ids)
        found = np.zeros(len(ids), dtype=bool)
        found[inside] = self.note_ids[positions[inside]] == ids[inside]
        values = np.zeros((len(ids), len(STATIC_ITEM_FEATURES)), dtype=np.float32)
        values[found] = self.values[positions[found]]
        out = pd.DataFrame(values, columns=STATIC_ITEM_FEATURES)
        out.insert(0, "note_idx", ids)
        out["item_features_found"] = found
        return out

    def attach(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        features = self.get(out["note_idx"])
        for name in (*STATIC_ITEM_FEATURES, "item_features_found"):
            out[name] = features[name].to_numpy()
        return out


_STORE_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def _load_store(notes_dir: Path, vocab_dir: Path) -> StaticItemFeatureStore:
    return StaticItemFeatureStore(notes_dir, vocab_dir)


def get_item_feature_store(root: Path, vocab_dir: Path | None = None) -> StaticItemFeatureStore:
    root = Path(root).resolve()
    vocab_dir = Path(vocab_dir).resolve() if vocab_dir is not None else root / "features/vocab_dict"
    with _STORE_LOCK:
        return _load_store(root / "datasets/notes", vocab_dir)
