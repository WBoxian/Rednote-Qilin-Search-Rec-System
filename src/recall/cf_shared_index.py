"""
Swing / UserCF 共享的 user-item 索引构建与缓存工具。

输出:
- outputs/index/cf_{scene}_{split}_user_item_index.pkl
- outputs/index/cf_{scene}_{split}_user_item_index.meta.json

示例:
    uv run src/recall/cf_shared_index.py --scene search
    uv run src/recall/cf_shared_index.py --scene rec
"""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]  # Qilin/
FEAT_DIR = BASE_DIR / "features"
RESULT_DIR = BASE_DIR / "outputs" / "index"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def _cache_paths(scene: str, split: str):
    prefix = f"cf_{scene}_{split}_user_item_index"
    return (
        RESULT_DIR / f"{prefix}.pkl",
        RESULT_DIR / f"{prefix}.meta.json",
    )


def _load_scene_df(scene: str, split: str, cols: list[str]) -> pd.DataFrame:
    path = FEAT_DIR / f"{scene}_{split}_features.parquet"
    return pd.read_parquet(path, columns=cols)


CACHE_VERSION = "cf-user-item-v2"


def _source_fingerprint(path: Path) -> dict:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("CF source changed while hashing")
    return {"path": str(path.resolve()), "bytes": after.st_size, "sha256": digest.hexdigest()}


def _official_cutoff(scene: str, split: str) -> float | None:
    """Read only the first official test event time, never its click labels."""
    if split != "train":
        return None
    folder = "search" if scene == "search" else "recommendation"
    path = FEAT_DIR.parent / "datasets" / f"{folder}_test" / "train-00000-of-00001.parquet"
    if not path.exists():
        return None  # Custom fixtures must provide an explicit cutoff to claim temporal isolation.
    import duckdb
    details = "search_result_details_with_idx" if scene == "search" else "rec_result_details_with_idx"
    field = "search_timestamp" if scene == "search" else "request_timestamp"
    with duckdb.connect(config={"threads": "2", "memory_limit": "1GB"}) as con:
        value = con.execute(f"SELECT min(d.{field}) FROM read_parquet(?) r,unnest(r.{details}) u(d)",
                            [str(path)]).fetchone()[0]
    if value is None or not np.isfinite(value):
        raise ValueError("Official test cutoff is missing or nonfinite")
    return float(value)


def _atomic_write(path: Path, write) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=path.name+".", delete=False) as handle:
            temporary = Path(handle.name)
            write(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def ensure_user_item_index(scene: str, split: str = "train", interest_col: str = "y_multi",
        min_interest: float = 0.0, use_cache: bool = True, rebuild: bool = False,
        cutoff_exclusive: float | None = None):
    """Build a versioned source-bound CF graph; silently stale cache reuse is forbidden."""
    if scene not in {"search", "rec"} or split not in {"train", "test"}:
        raise ValueError("Unsupported CF scene/split")
    if not np.isfinite(min_interest):
        raise ValueError("CF minimum interest must be finite")
    if cutoff_exclusive is None:
        cutoff_exclusive = _official_cutoff(scene, split)
    if cutoff_exclusive is not None and not np.isfinite(cutoff_exclusive):
        raise ValueError("CF cutoff must be finite")
    source = FEAT_DIR / f"{scene}_{split}_features.parquet"
    contract = {"version": CACHE_VERSION, "scene": scene, "split": split,
        "interest_col": interest_col, "min_interest": float(min_interest),
        "cutoff_exclusive": cutoff_exclusive, "source": _source_fingerprint(source)}
    cache_pkl, cache_meta = _cache_paths(scene, split)
    if use_cache and not rebuild and cache_pkl.exists():
        with cache_pkl.open("rb") as handle:
            cached = pickle.load(handle)
        if cached.get("cache_contract") == contract:
            return cached

    import pyarrow.parquet as pq
    available = set(pq.ParquetFile(source).schema_arrow.names)
    score_col = interest_col if interest_col in available else "click"
    cols = list(dict.fromkeys(["user_idx", "note_idx", "click", score_col]))
    if cutoff_exclusive is not None:
        if "event_timestamp" not in available:
            raise ValueError("Temporal CF build requires event_timestamp")
        cols.append("event_timestamp")
    df = _load_scene_df(scene, split, cols)
    if cutoff_exclusive is not None:
        times = pd.to_numeric(df["event_timestamp"], errors="coerce")
        if not np.isfinite(times).all():
            raise ValueError("CF event times must be finite")
        df = df.loc[times < cutoff_exclusive].copy()
    eligible_rows = len(df)
    score = pd.to_numeric(df[score_col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.)
    df = df.assign(score=score.clip(lower=0.))
    df = df.loc[df.score > min_interest].copy()
    # A zero-interest objective yields an empty graph rather than silently changing its objective.
    grp = df.groupby(["user_idx", "note_idx"], as_index=False)["score"].max()
    user_items, item_users = defaultdict(dict), defaultdict(dict)
    for user, item, value in grp.itertuples(index=False, name=None):
        user, item, value = int(user), int(item), float(value)
        if user < 0 or item < 0:
            raise ValueError("CF IDs must be nonnegative; valid note0 is retained")
        user_items[user][item] = value
        item_users[item][user] = value
    if _source_fingerprint(source) != contract["source"]:
        raise RuntimeError("CF source changed during build")
    result = {"scene": scene, "split": split, "score_col": score_col,
        "min_interest": float(min_interest), "user_items": dict(user_items),
        "item_users": dict(item_users), "cache_contract": contract}
    if use_cache:
        _atomic_write(cache_pkl, lambda handle: pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL))
        meta = {"cache_contract": contract, "score_col": score_col,
            "eligible_source_rows": eligible_rows, "num_users": len(user_items),
            "num_items": len(item_users), "num_edges": len(grp)}
        _atomic_write(cache_meta, lambda handle: handle.write(json.dumps(meta, indent=2).encode()))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=["search", "rec"], required=True)
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--interest-col", type=str, default="y_multi")
    parser.add_argument("--min-interest", type=float, default=0.0)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--cutoff-exclusive", type=float, default=None)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    index = ensure_user_item_index(
        scene=args.scene,
        split=args.split,
        interest_col=args.interest_col,
        min_interest=args.min_interest,
        use_cache=not args.no_cache,
        rebuild=args.rebuild,
        cutoff_exclusive=args.cutoff_exclusive,
    )
    cache_pkl, cache_meta = _cache_paths(args.scene, args.split)
    print(
        f"[OK] scene={args.scene} split={args.split} "
        f"users={len(index['user_items'])} items={len(index['item_users'])} "
        f"score_col={index['score_col']}\n"
        f"pkl={cache_pkl}\nmeta={cache_meta}"
    )


# =============================
if __name__ == "__main__":
    main()
