"""Read-only validation of catalog embedding geometry and missing modalities."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate embedding map key: {key}")
        result[key] = value
    return result


def _identity(path):
    value = path.stat()
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def audit_note_vectors(folder, prefix, *, catalog_size, dimension=768, batch_rows=8192):
    """Return diagnostics and zero-vector mask indexed by note ID, not file row.

    Zero vectors are reported as missing modalities, never automatically labeled
    corruption. The caller must compare their mask against source availability.
    """
    for value in (catalog_size, dimension, batch_rows):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("Catalog size, dimension and batch rows must be positive integers")
    folder = Path(folder)
    binary, mapping_file = folder / f"{prefix}.bin", folder / f"{prefix}_map.json"
    before = [_identity(path) for path in (binary, mapping_file)]
    raw_mapping = mapping_file.read_bytes()
    mapping = json.loads(raw_mapping, object_pairs_hook=_unique_keys)
    if not isinstance(mapping, dict) or len(mapping) != catalog_size:
        raise ValueError("Embedding map does not cover the catalog")
    if any(not isinstance(key, str) or not key.isdecimal() or str(int(key)) != key
           or isinstance(row, bool) or not isinstance(row, int)
           for key, row in mapping.items()):
        raise ValueError("Embedding map requires canonical nonnegative IDs and integer offsets")
    ids = np.fromiter((int(key) for key in mapping), np.int64, count=catalog_size)
    rows = np.fromiter(mapping.values(), np.int64, count=catalog_size)
    expected = np.arange(catalog_size)
    if not np.array_equal(np.sort(ids), expected) or not np.array_equal(np.sort(rows), expected):
        raise ValueError("Embedding IDs and file offsets must each cover the catalog exactly once")
    if binary.stat().st_size != catalog_size * dimension * np.dtype(np.float16).itemsize:
        raise ValueError("Embedding binary size differs from the declared geometry")
    vectors = np.memmap(binary, np.float16, "r", shape=(catalog_size, dimension))
    zeros = np.zeros(catalog_size, dtype=bool)
    nonfinite = 0
    digest = hashlib.sha256()
    for first in range(0, catalog_size, batch_rows):
        block = vectors[first:first + batch_rows]
        digest.update(block.tobytes())
        nonfinite += int((~np.isfinite(block)).sum())
        zeros[first:first + len(block)] = np.all(block == 0, axis=1)
    by_id = np.empty_like(zeros)
    by_id[ids] = zeros[rows]
    if before != [_identity(path) for path in (binary, mapping_file)]:
        raise ValueError("Embedding source changed during its audit")
    return {
        "catalog": catalog_size, "dimension": dimension,
        "map_bijection": True, "nonfinite_components": nonfinite,
        "zero_vectors": int(zeros.sum()), "zero_fraction": float(zeros.mean()),
        "binary_sha256": digest.hexdigest(),
        "map_sha256": hashlib.sha256(raw_mapping).hexdigest(),
        "source_identity_unchanged": True,
        "status": "PASS" if nonfinite == 0 else "FAIL_NONFINITE",
    }, by_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--catalog-size", type=int, required=True)
    parser.add_argument("--dimension", type=int, default=768)
    args = parser.parse_args()
    report, _ = audit_note_vectors(args.folder, args.prefix,
                                  catalog_size=args.catalog_size, dimension=args.dimension)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
