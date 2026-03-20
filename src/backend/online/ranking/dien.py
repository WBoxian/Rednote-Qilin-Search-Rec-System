from __future__ import annotations

import importlib.util
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    from transformers import AutoModel, AutoTokenizer
except Exception:
    AutoModel = None
    AutoTokenizer = None

BASE_DIR = Path(__file__).resolve().parents[4]


def _load_module_from_src(module_name: str, rel_path: str):
    module_path = BASE_DIR / "src" / rel_path
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _ensure_training_utils_loaded() -> None:
    _load_module_from_src("utils", "training/utils.py")


class DIENPredictor:
    _shared_query_tokenizer = None
    _shared_query_model = None

    def __init__(self, scene: str, tag: str, deploy_tag_dir: Path, group_key: str):
        self.scene = scene
        self.tag = tag
        self.deploy_tag_dir = deploy_tag_dir
        self.group_key = group_key
        self.model = None
        self.num_feats: list[str] = []
        self.stores: dict[str, object] = {}
        self.query_tokenizer = None
        self.query_model = None
        self._query_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._query_cache_max = 2048

        _ensure_training_utils_loaded()
        self.dien_train = _load_module_from_src("online_ranking_dien_ranker", "training/dien_ranker.py")
        self.device = self.dien_train.DEVICE
        self._load_model()

    def _artifact_path(self, rel_path: str) -> Path:
        return self.deploy_tag_dir / rel_path

    def _load_model(self) -> None:
        ckpt = self._artifact_path(f"models/dien_{self.scene}_{self.tag}.pt")
        if not ckpt.exists():
            return

        state = torch.load(ckpt, map_location=self.device)
        seen = set()
        num_feats = []
        for k in state.keys():
            if k.startswith("num_proj.") and k.endswith(".weight"):
                feat = k.split(".")[1]
                if feat not in seen:
                    seen.add(feat)
                    num_feats.append(feat)
        if not num_feats:
            return

        model = self.dien_train.DIENRanker(scene=self.scene, num_feats=num_feats).to(self.device)
        model.load_state_dict(state, strict=True)
        model.eval()

        self.num_feats = num_feats
        self.model = model

        seq_name = f"{self.scene}_seq"
        self.stores = {
            "s": self.dien_train.EmbeddingStore(
                BASE_DIR / "embeddings" / "query_text_emb",
                key_col=self.group_key,
                emb_col="seq_embs",
                prefix=self.scene,
                is_sequence=True,
                name=seq_name,
            ),
            "t": self.dien_train.EmbeddingStore(
                BASE_DIR / "embeddings" / "note_text_emb",
                key_col="note_idx",
                emb_col="note_text_emb",
                name="note_text",
            ),
            "i": self.dien_train.EmbeddingStore(
                BASE_DIR / "embeddings" / "note_img_emb",
                key_col="note_idx",
                emb_col="note_img_emb",
                name="note_img",
            ),
        }
        if self.scene == "search":
            self.stores["q"] = self.dien_train.EmbeddingStore(
                BASE_DIR / "embeddings" / "query_text_emb",
                key_col=self.group_key,
                emb_col="query_emb",
                prefix="search",
                name="search_query",
            )
            self._load_query_encoder()

    def _load_query_encoder(self) -> None:
        if AutoTokenizer is None or AutoModel is None:
            return
        if DIENPredictor._shared_query_tokenizer is None or DIENPredictor._shared_query_model is None:
            try:
                DIENPredictor._shared_query_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-zh")
                DIENPredictor._shared_query_model = AutoModel.from_pretrained("BAAI/bge-base-zh").to("cpu")
                DIENPredictor._shared_query_model.eval()
            except Exception:
                DIENPredictor._shared_query_tokenizer = None
                DIENPredictor._shared_query_model = None
        self.query_tokenizer = DIENPredictor._shared_query_tokenizer
        self.query_model = DIENPredictor._shared_query_model

    def _normalize_search_query(self, text: str) -> str:
        return str(text or "").strip()

    def _encode_query_vec(self, text: str) -> np.ndarray | None:
        q = self._normalize_search_query(text)
        if not q or self.query_tokenizer is None or self.query_model is None:
            return None

        cached = self._query_cache.get(q)
        if cached is not None:
            self._query_cache.move_to_end(q)
            return cached

        try:
            inputs = self.query_tokenizer(
                [q],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            with torch.no_grad():
                out = self.query_model(**inputs).last_hidden_state[:, 0]
                out = out / out.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            vec = out.detach().float().cpu().numpy().astype(np.float32)
            self._query_cache[q] = vec
            if len(self._query_cache) > self._query_cache_max:
                self._query_cache.popitem(last=False)
            return vec
        except Exception:
            return None

    def _get_runtime_query_tensor(self, batch_df: pd.DataFrame, gids: np.ndarray) -> torch.Tensor:
        texts = [self._normalize_search_query(x) for x in batch_df.get("query", pd.Series([""] * len(batch_df))).tolist()]
        if not any(texts):
            return self.stores["q"].get_tensor(gids)

        emb_dim = int(self.dien_train.EMB_DIM)
        out = np.zeros((len(texts), emb_dim), dtype=np.float32)
        cache_by_text: dict[str, np.ndarray] = {}
        for i, q in enumerate(texts):
            if not q:
                continue
            vec = cache_by_text.get(q)
            if vec is None:
                vec = self._encode_query_vec(q)
                if vec is not None:
                    cache_by_text[q] = vec
            if vec is not None:
                out[i] = vec.reshape(-1)
        zero_rows = np.where(np.abs(out).sum(axis=1) <= 1e-12)[0]
        if len(zero_rows) > 0:
            fallback = self.stores["q"].get_tensor(gids[zero_rows]).detach().cpu().numpy().astype(np.float32)
            out[zero_rows] = fallback
        return torch.tensor(out, dtype=torch.float32, device=self.device)

    def _build_seq_tensor_from_history(self, history_note_ids: list[int], repeats: int) -> torch.Tensor:
        seq_len = int(self.dien_train.SEQ_LEN)
        emb_dim = int(self.dien_train.EMB_DIM)
        note_ids = [int(x) for x in history_note_ids[:seq_len] if int(x) >= 0]
        if note_ids:
            seq_tensor = self.stores["t"].get_tensor(note_ids)
            seq_np = seq_tensor.detach().cpu().numpy().astype(np.float32)
        else:
            seq_np = np.zeros((0, emb_dim), dtype=np.float32)

        padded = np.zeros((seq_len, emb_dim), dtype=np.float32)
        if len(seq_np) > 0:
            take = min(seq_len, len(seq_np))
            padded[:take] = seq_np[:take]
        padded = np.flip(padded, axis=0).copy()
        batch_np = np.repeat(padded[None, :, :], repeats=int(repeats), axis=0)
        return torch.tensor(batch_np, dtype=torch.float32, device=self.device)

    def predict(
        self,
        cand: pd.DataFrame,
        batch_size: int = 512,
        history_note_ids: list[int] | None = None,
    ) -> np.ndarray:
        if self.model is None or cand.empty:
            return np.zeros(len(cand), dtype=np.float32)

        df = cand.reset_index(drop=True)
        preds = np.zeros(len(df), dtype=np.float32)
        history_seq_tensor: torch.Tensor | None = None
        if history_note_ids:
            history_seq_tensor = self._build_seq_tensor_from_history(
                history_note_ids=history_note_ids,
                repeats=min(int(batch_size), len(df)),
            )
        with torch.no_grad():
            for i in range(0, len(df), batch_size):
                b = df.iloc[i : i + batch_size]
                feats: dict[str, torch.Tensor] = {}
                for f in self.num_feats:
                    if f in b.columns:
                        vals = pd.to_numeric(b[f], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float32)
                    else:
                        vals = np.zeros(len(b), dtype=np.float32)
                    feats[f] = torch.tensor(vals, dtype=torch.float32, device=self.device).unsqueeze(-1)

                gids = b[self.group_key].to_numpy(np.int64)
                nids = b["note_idx"].to_numpy(np.int64)
                if history_seq_tensor is not None:
                    feats["seq_embs"] = history_seq_tensor[: len(b)]
                else:
                    feats["seq_embs"] = torch.flip(self.stores["s"].get_tensor(gids), dims=[1])
                feats["note_text_emb"] = self.stores["t"].get_tensor(nids)
                feats["note_img_emb"] = self.stores["i"].get_tensor(nids)
                if self.scene == "search" and "q" in self.stores:
                    feats["query_emb"] = self._get_runtime_query_tensor(b, gids)

                p, _ = self.model(feats)
                cur = p.detach().float().cpu().numpy().reshape(-1)
                preds[i : i + len(b)] = np.nan_to_num(cur, nan=0.0, posinf=1e6, neginf=-1e6)

        return preds.astype(np.float32)


def apply_dien_scores(
    cand: pd.DataFrame,
    page_size: int,
    predict_dien,
    history_note_ids: list[int] | None = None,
) -> pd.DataFrame:
    if cand.empty:
        return cand
    out = cand.copy()
    dien_topn = len(out)
    dien_pred = np.zeros(len(out), dtype=np.float32)
    if dien_topn > 0:
        dien_pred[:dien_topn] = predict_dien(out.iloc[:dien_topn], history_note_ids=history_note_ids)
    out["dien_score"] = np.nan_to_num(dien_pred, nan=0.0, posinf=1e6, neginf=-1e6)
    return out
