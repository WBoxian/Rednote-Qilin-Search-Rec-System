"""Share local-only query encoders without conflating pinned model revisions."""
from __future__ import annotations

from pathlib import Path
from threading import RLock
from typing import Any

try:
    from transformers import AutoModel, AutoTokenizer
except Exception:
    AutoModel = AutoTokenizer = None

_ENCODERS: dict[tuple[str, str | None], tuple[Any, Any]] = {}
_FAILED: set[tuple[str, str | None]] = set()
_LOCK = RLock()


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


__all__ = ["get_shared_query_encoder"]