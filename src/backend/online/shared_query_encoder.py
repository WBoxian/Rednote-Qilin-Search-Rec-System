"""Share local-only query encoders without conflating pinned model revisions."""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np
import torch

try:
    from transformers import AutoModel, AutoTokenizer
except Exception:
    AutoModel = AutoTokenizer = None

_ENCODERS: dict[tuple[str, str | None], tuple[Any, Any]] = {}
_FAILED: set[tuple[str, str | None]] = set()
_LOCK = RLock()
_QUERY_VECTORS = OrderedDict()
_QUERY_CACHE_SIZE = 1024


def get_shared_query_encoder(model_dir: Path | str | None = None, revision: str | None = None) -> tuple[Any, Any]:
    source = str(Path(model_dir).resolve()) if model_dir is not None else "BAAI/bge-base-zh"
    key = source, revision
    with _LOCK:
        if key in _ENCODERS:
            return _ENCODERS[key]
        if key in _FAILED or AutoTokenizer is None or AutoModel is None:
            return None, None
        try:
            # Explicit serving bundles point to copied, hash-bound project-local snapshots.
            kwargs = {"local_files_only": True}
            if revision is not None:
                kwargs["revision"] = revision
            tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
            model = AutoModel.from_pretrained(source, **kwargs).to("cpu")
            model.eval()
            _ENCODERS[key] = tokenizer, model
            return _ENCODERS[key]
        except (OSError, ValueError, RuntimeError):
            _FAILED.add(key)
            return None, None


def encode_current_query(text: str, model_dir: Path | str, revision: str | None) -> np.ndarray:
    """Identical pinned BGE CLS inference for recall and rank; never cache failures."""
    query = str(text or "").strip()
    if not query:
        raise ValueError("Current-query encoding requires nonempty text")
    key = str(Path(model_dir).resolve()), revision, query
    with _LOCK:
        if key in _QUERY_VECTORS:
            _QUERY_VECTORS.move_to_end(key)
            return _QUERY_VECTORS[key].copy()
        tokenizer, model = get_shared_query_encoder(model_dir, revision)
        if tokenizer is None or model is None:
            raise RuntimeError("Pinned current-query encoder is unavailable")
        inputs = tokenizer([query], padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.inference_mode():
            vector = model(**inputs).last_hidden_state[:, 0].float()
            vector = vector/vector.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        value = vector.cpu().numpy().reshape(-1).astype(np.float32)
        if value.shape != (768,) or not np.isfinite(value).all() or np.linalg.norm(value) < 1e-6:
            raise RuntimeError("Pinned current-query encoder produced an invalid vector")
        value.flags.writeable = False
        _QUERY_VECTORS[key] = value
        if len(_QUERY_VECTORS) > _QUERY_CACHE_SIZE:
            _QUERY_VECTORS.popitem(last=False)
        return value.copy()


__all__ = ["get_shared_query_encoder", "encode_current_query"]
