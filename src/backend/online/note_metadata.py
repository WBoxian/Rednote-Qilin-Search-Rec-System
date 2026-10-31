"""Immutable indexed catalog metadata; built exclusively from static note files."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import threading
import zlib
from urllib.parse import quote

import pandas as pd
import pyarrow.parquet as pq

COLUMNS = ("note_idx", "note_title", "note_content", "image_path", "accum_like_num",
           "accum_collect_num", "accum_comment_num", "taxonomy1_id", "taxonomy2_id", "taxonomy3_id")
VERSION = 1


def _sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8*1024*1024), b''):
            value.update(part)
    return value.hexdigest()


def build_note_metadata_index(notes_dir, output, batch_size=8192):
    """Create a new atomic SQLite snapshot without changing existing catalog data."""
    output = Path(output)
    sidecar = output.with_suffix(output.suffix+'.manifest.json')
    temporary = output.with_suffix(output.suffix+'.building')
    if output.exists() or sidecar.exists() or temporary.exists():
        raise FileExistsError('Choose a new path for the immutable metadata index')
    paths = sorted(Path(notes_dir).glob('*.parquet'))
    if not paths:
        raise FileNotFoundError('Static notes parquet files are absent')
    sources = [{"name": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size} for path in paths]
    output.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(temporary)
    count = 0
    try:
        con.execute('PRAGMA journal_mode=OFF')
        con.execute('PRAGMA synchronous=OFF')
        con.execute('PRAGMA cache_size=-65536')
        con.execute('CREATE TABLE notes(note_idx INTEGER PRIMARY KEY CHECK(note_idx>=0), payload BLOB NOT NULL)')
        con.execute('CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL) WITHOUT ROWID')
        for path in paths:
            for batch in pq.ParquetFile(path).iter_batches(columns=list(COLUMNS), batch_size=batch_size):
                rows = batch.to_pylist()
                values = [(int(row['note_idx']), zlib.compress(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode(), level=1)) for row in rows]
                con.executemany('INSERT INTO notes VALUES(?,?)', values)
                count += len(values)
        recipe = {"version": VERSION, "columns": list(COLUMNS), "rows": count, "sources": sources, "labels_used": False,
                  "payload_encoding": "zlib_utf8_json_v1"}
        con.execute('INSERT INTO metadata VALUES(?,?)', ('recipe', json.dumps(recipe, sort_keys=True)))
        con.commit()
    finally:
        con.close()
    temporary.rename(output)
    manifest = {**recipe, "index_sha256": _sha256(output), "index_bytes": output.stat().st_size}
    sidecar.write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


class NoteMetadataIndex:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.manifest = json.loads(self.path.with_suffix(self.path.suffix+'.manifest.json').read_text())
        self.encoding = self.manifest.get('payload_encoding', 'json_utf8_v1')
        if self.encoding not in {'json_utf8_v1', 'zlib_utf8_json_v1'}:
            raise ValueError('Unsupported note metadata encoding')
        if self.manifest.get('version') != VERSION or self.manifest.get('columns') != list(COLUMNS) or self.manifest.get('labels_used') is not False or self.path.stat().st_size != self.manifest.get('index_bytes'):
            raise ValueError('Incompatible note metadata snapshot')
        self._local, self._lock, self._connections = threading.local(), threading.Lock(), []
        con = self._connection()
        recipe = json.loads(con.execute("SELECT value FROM metadata WHERE key='recipe'").fetchone()[0])
        if recipe != {key: value for key, value in self.manifest.items() if key not in {'index_sha256', 'index_bytes'}}:
            self.close()
            raise ValueError('Note metadata DB differs from its sidecar')

    def _connection(self):
        value = getattr(self._local, 'connection', None)
        if value is None:
            with self._lock:
                value = sqlite3.connect('file:'+quote(str(self.path))+'?mode=ro&immutable=1', uri=True, check_same_thread=False)
                value.execute('PRAGMA query_only=ON')
                self._local.connection = value
                self._connections.append(value)
        return value

    def fetch(self, note_ids):
        ids = sorted(set(int(note) for note in note_ids))
        rows = []
        for start in range(0, len(ids), 500):
            part = ids[start:start+500]
            rows.extend(json.loads(zlib.decompress(row[0]) if self.encoding == 'zlib_utf8_json_v1' else row[0]) for row in self._connection().execute(
                'SELECT payload FROM notes WHERE note_idx IN ('+','.join('?' for _ in part)+') ORDER BY note_idx', part))
        return pd.DataFrame(rows, columns=list(COLUMNS))

    def close(self):
        for connection in self._connections:
            connection.close()
        self._connections.clear()
