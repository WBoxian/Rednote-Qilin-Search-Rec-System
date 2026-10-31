"""Local, corpus-built BM25 retrieval with a versioned Chinese tokenizer.

Index construction never consumes query/click labels. Catalog availability at a
historical request time is not established by the provided static snapshot.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import threading
import time
import unicodedata

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

TOKENIZER_VERSION = "qilin_cjk_unigram_bigram_v1"
SCHEMA_VERSION = 1
_RUNS = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+|[a-z0-9]+")
_CJK = re.compile(r"^[\u3400-\u4dbf\u4e00-\u9fff]+$")
DEFAULT_INDEX = Path("outputs/evaluation/northstar_fix_20261031/search/bm25.sqlite")


def tokenize(text, *, query=False):
    """Index CJK characters+bigrams; use bigrams for multi-character queries."""
    text = text if isinstance(text, str) else ""
    runs = _RUNS.findall(unicodedata.normalize("NFKC", text).lower())
    tokens = []
    for run in runs:
        if _CJK.fullmatch(run):
            if not query or len(run) == 1:
                tokens.extend(run)
            if len(run) > 1:
                tokens.extend(run[i:i+2] for i in range(len(run)-1))
        else:
            tokens.append(run)
    # Repeated query terms should not make the MATCH expression arbitrarily large.
    return list(dict.fromkeys(tokens)) if query else tokens


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tokenizer_contract():
    implementation = inspect.getsource(tokenize) + _RUNS.pattern + _CJK.pattern
    return {"version": TOKENIZER_VERSION, "implementation_sha256": hashlib.sha256(implementation.encode()).hexdigest(),
            "unicode_version": unicodedata.unidata_version, "normalization": "NFKC+lower", "query_logic": "quoted_token_OR"}


def _recipe(paths, title_chars, body_chars, title_weight, body_weight, query_chars):
    for value in [title_chars, body_chars, query_chars]:
        if not isinstance(value, int) or value <= 0:
            raise ValueError("Text character budgets must be positive integers")
    if not all(math.isfinite(weight) and weight > 0 for weight in [title_weight, body_weight]):
        raise ValueError("BM25 column weights must be finite and positive")
    sources = []
    for path in paths:
        schema = set(pq.read_schema(path).names)
        missing = {"note_idx", "note_title", "note_content"} - schema
        if missing:
            raise ValueError(f"Missing required catalog fields in {path.name}: {sorted(missing)}")
        sources.append({"name": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path),
                        "rows": pq.ParquetFile(path).metadata.num_rows})
    return {"schema_version": SCHEMA_VERSION, "tokenizer": _tokenizer_contract(), "corpus": sources,
            "title_chars": title_chars, "body_chars": body_chars, "query_chars": query_chars,
            "title_weight": float(title_weight), "body_weight": float(body_weight),
            "sqlite_version": sqlite3.sqlite_version, "fts_detail": "full", "fts_content": "contentless",
            "rowid": "note_idx+1", "labels_used": False,
            "point_in_time_catalog_availability": "UNKNOWN"}


def build_index(notes_dir, output=DEFAULT_INDEX, *, title_chars=128, body_chars=512,
                title_weight=2., body_weight=1., query_chars=128, batch_size=8192,
                cache_mib=256, memory_limit_mib=2048):
    """Stream only static catalog fields into an atomic, fingerprint-bound index."""
    notes_dir, output = Path(notes_dir), Path(output)
    paths = sorted(notes_dir.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No catalog parquet files: {notes_dir}")
    if batch_size <= 0 or memory_limit_mib < 256 or not 16 <= cache_mib <= memory_limit_mib // 2:
        raise ValueError("Invalid batch/cache memory budget")
    recipe = _recipe(paths, title_chars, body_chars, title_weight, body_weight, query_chars)
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    if output.exists() or manifest_path.exists():
        if not (output.is_file() and manifest_path.is_file()):
            raise FileExistsError("Incomplete BM25 index exists; choose a new output path")
        old = json.loads(manifest_path.read_text())
        if old.get("recipe") != recipe or old.get("status") != "ready":
            raise FileExistsError("BM25 corpus/tokenizer/recipe changed; choose a new output path")
        if output.stat().st_size != old.get("index_bytes") or _sha256(output) != old.get("index_sha256"):
            raise ValueError("Existing BM25 index content differs from its frozen manifest")
        print(json.dumps({"status": "reused", "documents": old["documents"], "path": str(output)}), flush=True)
        return old
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.with_suffix(output.suffix + ".lock")
    lock = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    temporary = output.with_suffix(output.suffix + ".building")
    temporary_manifest = manifest_path.with_suffix(manifest_path.suffix + ".building")
    if temporary.exists():
        os.close(lock); lock_path.unlink()
        raise FileExistsError("A prior partial BM25 build exists; choose a new output path")
    con = None
    started = time.monotonic()
    total = 0
    try:
        con = sqlite3.connect(temporary)
        con.execute("PRAGMA journal_mode=OFF")
        con.execute("PRAGMA synchronous=OFF")
        con.execute("PRAGMA temp_store=FILE")
        con.execute(f"PRAGMA cache_size={-int(cache_mib)*1024}")
        # SQLite heap is bounded separately from Python/Arrow batch storage.
        con.execute(f"PRAGMA hard_heap_limit={min(1024, memory_limit_mib//2)*1024*1024}")
        con.execute("CREATE TABLE item_ids(note_idx INTEGER PRIMARY KEY CHECK(note_idx>=0)) WITHOUT ROWID")
        con.execute("CREATE VIRTUAL TABLE documents USING fts5(title,body,content='',detail=full,tokenize='unicode61 remove_diacritics 0')")
        con.execute("INSERT INTO documents(documents,rank) VALUES('automerge',0)")
        con.execute(f"INSERT INTO documents(documents,rank) VALUES('rank','bm25({float(title_weight)},{float(body_weight)})')")
        con.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL) WITHOUT ROWID")
        con.execute("INSERT INTO metadata VALUES('recipe',?)", (json.dumps(recipe, sort_keys=True),))
        con.commit()
        for path in paths:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=["note_idx", "note_title", "note_content"]):
                data = batch.to_pydict()
                rows = []
                for note, title, body in zip(data["note_idx"], data["note_title"], data["note_content"]):
                    if note is None or isinstance(note, bool) or int(note) != note or note < 0 or note >= 2**63-1:
                        raise ValueError("Catalog note_idx must be a nonnegative integral row identifier")
                    title = title[:title_chars] if isinstance(title, str) else ""
                    body = body[:body_chars] if isinstance(body, str) else ""
                    rows.append((int(note)+1, " ".join(tokenize(title)), " ".join(tokenize(body))))
                with con:
                    con.executemany("INSERT INTO item_ids(note_idx) VALUES(?)", ((row[0]-1,) for row in rows))
                    con.executemany("INSERT INTO documents(rowid,title,body) VALUES(?,?,?)", rows)
                total += len(rows)
                elapsed = time.monotonic()-started
                print(json.dumps({"stage": "index", "documents": total, "elapsed_seconds": round(elapsed,2), "documents_per_second": round(total/max(elapsed,1e-9),2)}), flush=True)
        expected = sum(source["rows"] for source in recipe["corpus"])
        catalog_count = con.execute("SELECT count(*) FROM item_ids").fetchone()[0]
        fts_count = con.execute("SELECT count(*) FROM documents").fetchone()[0]
        if total != expected or catalog_count != total or fts_count != total:
            raise ValueError("BM25 construction did not preserve every original catalog item")
        print(json.dumps({"stage": "optimize", "documents": total}), flush=True)
        con.execute("INSERT INTO documents(documents) VALUES('optimize')")
        con.execute("INSERT INTO documents(documents) VALUES('integrity-check')")
        con.execute("INSERT INTO metadata VALUES('documents',?)", (str(total),))
        con.commit()
        check = con.execute("PRAGMA quick_check").fetchone()[0]
        if check != "ok":
            raise ValueError(f"SQLite index integrity check failed: {check}")
        con.close(); con = None
        manifest = {"status": "ready", "recipe": recipe, "documents": total,
                    "index_bytes": temporary.stat().st_size, "index_sha256": _sha256(temporary),
                    "seconds": time.monotonic()-started, "build_cache_mib": cache_mib,
                    "build_memory_limit_mib": memory_limit_mib,
                    "memory_scope": "SQLite heap <= half the declared memory budget; bounded Arrow batches; temp files on disk"}
        temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
        temporary.replace(output)
        temporary_manifest.replace(manifest_path)
        print(json.dumps({"status": "ready", "documents": total, "seconds": manifest["seconds"], "index_bytes": manifest["index_bytes"], "path": str(output)}), flush=True)
        return manifest
    finally:
        if con is not None:
            con.close()
        os.close(lock)
        lock_path.unlink(missing_ok=True)
        # Only this build's temporary files are removed, never an existing index.
        temporary.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)


class BM25Index:
    """Read-only, per-thread connections; close after outstanding searches finish."""
    def __init__(self, path, *, verify_hash=True):
        self.path = Path(path).resolve()
        manifest_path = self.path.with_suffix(self.path.suffix + ".manifest.json")
        self.manifest = json.loads(manifest_path.read_text())
        recipe = self.manifest.get("recipe", {})
        if self.manifest.get("status") != "ready" or recipe.get("schema_version") != SCHEMA_VERSION or recipe.get("tokenizer") != _tokenizer_contract():
            raise ValueError("BM25 tokenizer/schema manifest is incompatible")
        if self.path.stat().st_size != self.manifest.get("index_bytes") or (verify_hash and _sha256(self.path) != self.manifest.get("index_sha256")):
            raise ValueError("BM25 file content does not match the frozen manifest")
        self._local = threading.local()
        self._lock = threading.Lock()
        self._connections = []
        self._closed = False
        connection = self._connection()
        stored = json.loads(connection.execute("SELECT value FROM metadata WHERE key='recipe'").fetchone()[0])
        documents = int(connection.execute("SELECT value FROM metadata WHERE key='documents'").fetchone()[0])
        if stored != recipe or documents != self.manifest.get("documents"):
            self.close()
            raise ValueError("SQLite metadata differs from the sidecar manifest")

    def _connection(self):
        if self._closed:
            raise RuntimeError("BM25 index is closed")
        connection = getattr(self._local, "connection", None)
        if connection is None:
            from urllib.parse import quote
            with self._lock:
                connection = sqlite3.connect("file:"+quote(str(self.path))+"?mode=ro&immutable=1", uri=True, check_same_thread=False)
                connection.execute("PRAGMA query_only=ON")
                connection.execute("PRAGMA cache_size=-8192")
                self._connections.append(connection)
                self._local.connection = connection
        return connection

    def search(self, query, topk=100):
        if not isinstance(topk, (int, np.integer)) or topk <= 0:
            raise ValueError("topk must be a positive integer")
        empty = lambda: pd.DataFrame({"note_idx": pd.Series(dtype="int64"), "score": pd.Series(dtype="float64")})
        if not isinstance(query, str):
            return empty()
        terms = tokenize(query[:self.manifest["recipe"]["query_chars"]], query=True)
        if not terms:
            return empty()
        expression = " OR ".join('"'+term.replace('"', '""')+'"' for term in terms)
        rows = self._connection().execute(
            "SELECT rowid-1 AS note_idx, -rank AS score FROM documents WHERE documents MATCH ? ORDER BY rank ASC,rowid ASC LIMIT ?",
            (expression, int(topk)),
        ).fetchall()
        return pd.DataFrame.from_records(rows, columns=["note_idx", "score"]).astype({"note_idx": "int64", "score": "float64"})

    def close(self):
        with self._lock:
            self._closed = True
            for connection in self._connections:
                connection.close()
            self._connections.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()