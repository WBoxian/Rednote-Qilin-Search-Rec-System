from __future__ import annotations

from typing import Any

try:
    from transformers import AutoModel, AutoTokenizer
except Exception:
    AutoModel = None
    AutoTokenizer = None


_TOKENIZER: Any = None
_MODEL: Any = None
_FAILED = False


def get_shared_query_encoder() -> tuple[Any, Any]:
    global _TOKENIZER, _MODEL, _FAILED
    if _FAILED:
        return None, None
    if _TOKENIZER is not None and _MODEL is not None:
        return _TOKENIZER, _MODEL
    if AutoTokenizer is None or AutoModel is None:
        _FAILED = True
        return None, None
    try:
        _TOKENIZER = AutoTokenizer.from_pretrained("BAAI/bge-base-zh", local_files_only=True)
        _MODEL = AutoModel.from_pretrained("BAAI/bge-base-zh", local_files_only=True).to("cpu")
        _MODEL.eval()
    except Exception:
        _FAILED = True
        _TOKENIZER = None
        _MODEL = None
    return _TOKENIZER, _MODEL


__all__ = ["get_shared_query_encoder"]
