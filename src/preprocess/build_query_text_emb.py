"""Build versioned query/chronological-history embeddings from staged samples."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from training.history import HISTORY_ORDER, SEQUENCE_ORDER, history_ids_oldest_first

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "samples"
NOTE_TEXT_EMB_DIR = BASE_DIR / "embeddings" / "note_text_emb"
OUT_DIR = BASE_DIR / "embeddings" / "query_text_emb"
MODEL_NAME = "BAAI/bge-base-zh"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 256
MAX_LEN = 512
SHARD_SIZE = 10000
HIDDEN_DIM = 768
SEQ_LEN = 20
model = tokenizer = None


def _load_encoder():
    global model, tokenizer
    if model is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
        model.eval()
    return tokenizer, model


def _resolve_train_sample_path(scene, sample_dir=None, split="train"):
    path = (Path(sample_dir) if sample_dir is not None else DATA_DIR) / f"{scene}_{split}_samples.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing sample file: {path}")
    return path


def build_search_text(query):
    return query.strip() if isinstance(query, str) else ""


@torch.no_grad()
def encode(texts):
    encoder_tokenizer, encoder_model = _load_encoder()
    inputs = encoder_tokenizer(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
    output = encoder_model(**inputs).last_hidden_state[:, 0]
    output = output / output.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return output.cpu().numpy().astype(np.float16)


def load_note_text_emb(folder=None):
    folder = Path(folder) if folder is not None else NOTE_TEXT_EMB_DIR
    if (folder / "note_text.bin").exists() and (folder / "note_text_map.json").exists():
        from training.rank_stack import MappedVectors
        return MappedVectors(folder, "note_text", HIDDEN_DIM)
    files = sorted(folder.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No note text embeddings in {folder}")
    mapping = {}
    for path in files:
        frame = pd.read_parquet(path, columns=["note_idx", "note_text_emb"])
        for row in frame.itertuples(index=False):
            mapping[int(row.note_idx)] = np.asarray(row.note_text_emb, dtype=np.float16)
    return mapping


def build_sequence_embeddings(raw_ids, note_store, seq_len=SEQ_LEN):
    """Select recent prefix, reverse once, and pad to oldest-first RNN input."""
    ids = history_ids_oldest_first(raw_ids, seq_len)
    output = np.zeros((seq_len, HIDDEN_DIM), dtype=np.float16)
    if ids:
        if isinstance(note_store, dict):
            missing = [note for note in ids if note not in note_store]
            if missing:
                raise ValueError(f"Missing required history embeddings: {len(missing)}")
            values = np.asarray([note_store[note] for note in ids], dtype=np.float16)
        else:
            values = note_store.get(ids, required=True)
        if values.shape != (len(ids), HIDDEN_DIM):
            raise ValueError("History embedding dimensions do not match the model contract")
        output[:len(ids)] = values
    return output


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(scene, *, split="train", sample_dir=None, output_dir=None, note_emb_dir=None):
    if scene not in {"search", "rec"} or split not in {"train", "test"}:
        raise ValueError("Expected scene search/rec and split train/test")
    source = _resolve_train_sample_path(scene, sample_dir, split)
    destination = Path(output_dir) if output_dir is not None else OUT_DIR
    destination.mkdir(parents=True, exist_ok=True)
    key = "search_idx" if scene == "search" else "request_idx"
    columns = [key, "recent_clicked_note_idxs"] + (["query"] if scene == "search" else [])
    frame = pd.read_parquet(source, columns=columns).drop_duplicates(subset=[key], keep="first").reset_index(drop=True)
    if scene == "search":
        frame["query"] = frame["query"].map(build_search_text)
        if (frame["query"] == "").any():
            raise ValueError("Empty query contexts require an explicit policy; refusing to drop requests")
    embedding_folder = Path(note_emb_dir) if note_emb_dir is not None else NOTE_TEXT_EMB_DIR
    files = [embedding_folder / "note_text_map.json", embedding_folder / "note_text.bin"]
    if not all(path.exists() for path in files):
        files = sorted(embedding_folder.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No note text embeddings in {embedding_folder}")
    embedding_inputs = {str(path): {"bytes": path.stat().st_size, "sha256": _sha256(path)} for path in files}
    manifest = {"feature_contract": "query_history_v2", "scene": scene, "split": split,
        "history_order": HISTORY_ORDER, "sequence_order": SEQUENCE_ORDER, "source": str(source),
        "source_sha256": _sha256(source), "requests": len(frame), "dimension": HIDDEN_DIM,
        "sequence_length": SEQ_LEN, "encoder": MODEL_NAME,
        "note_embeddings": str(embedding_folder), "embedding_inputs": embedding_inputs,
        "limitations": ["Source history event timestamps are unavailable; order is a versioned input assumption."]}
    manifest_path = destination / f"{scene}_{split}_manifest.json"
    existing = list(destination.glob(f"{scene}_{split}-part-*.parquet"))
    legacy = list(destination.glob(f"{scene}-part-*.parquet"))
    if legacy or (existing and not manifest_path.exists()):
        raise ValueError("Unversioned embedding outputs exist; choose a separate --output-dir for the new history contract")
    if manifest_path.exists():
        recorded = json.loads(manifest_path.read_text())
        immutable = {name: value for name, value in recorded.items() if name not in {"completed", "outputs_sha256"}}
        if immutable != manifest:
            raise ValueError("Embedding manifest changed; use a new output directory")
    manifest_path.write_text(json.dumps({**manifest, "completed": False}, ensure_ascii=False, indent=2) + "\n")
    note_store = load_note_text_emb(note_emb_dir)
    for start in range(0, len(frame), SHARD_SIZE):
        path = destination / f"{scene}_{split}-part-{start // SHARD_SIZE:05d}.parquet"
        shard = frame.iloc[start:start + SHARD_SIZE]
        if path.exists():
            if pd.read_parquet(path, columns=[key])[key].tolist() != shard[key].tolist():
                raise ValueError("Cached embedding shard request IDs changed")
            continue
        data = {key: shard[key].to_numpy(), "seq_embs": [build_sequence_embeddings(ids, note_store).tolist() for ids in shard.recent_clicked_note_idxs]}
        if scene == "search":
            values = [encode(shard["query"].iloc[index:index+BATCH_SIZE].tolist()) for index in range(0, len(shard), BATCH_SIZE)]
            data["query_emb"] = list(np.vstack(values))
        temporary = path.with_suffix(".tmp.parquet")
        pd.DataFrame(data).to_parquet(temporary, index=False)
        temporary.replace(path)
    manifest.update({"completed": True, "outputs_sha256": {str(path): _sha256(path) for path in sorted(destination.glob(f"{scene}_{split}-part-*.parquet"))}})
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"[Query embeddings] {scene}/{split}: {len(frame)} requests -> {destination}")
    return manifest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, choices=["search", "rec"])
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--sample-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--note-emb-dir", type=Path, default=None)
    args = parser.parse_args()
    main(args.scene, split=args.split, sample_dir=args.sample_dir, output_dir=args.output_dir, note_emb_dir=args.note_emb_dir)