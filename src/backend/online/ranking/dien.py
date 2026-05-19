from __future__ import annotations

import importlib.util
import os
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

from backend.online.shared_query_encoder import get_shared_query_encoder

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
        self._history_seq_cache: OrderedDict[tuple[int, ...], torch.Tensor] = OrderedDict()
        self._history_seq_cache_max = 256

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

        # z-score 归一化 buffer 已烘焙进 checkpoint（训练后由 set_zscore_stats() 注入），
        # load_state_dict 会自动恢复，线上无需额外文件或手动归一化。
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
        self.query_tokenizer, self.query_model = get_shared_query_encoder()

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
        emb_dim = int(self.dien_train.EMB_DIM)
        out = np.zeros((len(texts), emb_dim), dtype=np.float32)
        if "q" in self.stores:
            try:
                ref_ids: list[int] = []
                if "query_ref_request_id" in batch_df.columns:
                    for raw_ref, gid in zip(batch_df["query_ref_request_id"].tolist(), gids.tolist()):
                        try:
                            ref_ids.append(int(pd.to_numeric(raw_ref, errors="coerce")))
                        except Exception:
                            ref_ids.append(int(gid))
                else:
                    ref_ids = [int(g) for g in gids.tolist()]
                precomp = self.stores["q"].get_tensor(ref_ids).detach().cpu().numpy().astype(np.float32)
                valid = np.linalg.norm(precomp, axis=1) > 1e-12
                if valid.any():
                    out[valid] = precomp[valid]
            except Exception:
                pass
        cache_by_text: dict[str, np.ndarray] = {}
        for i, q in enumerate(texts):
            if float(np.linalg.norm(out[i])) > 1e-12:
                continue
            if not q:
                continue
            vec = cache_by_text.get(q)
            if vec is None:
                vec = self._encode_query_vec(q)
                if vec is not None:
                    cache_by_text[q] = vec
            if vec is not None:
                out[i] = vec.reshape(-1)
        return torch.tensor(out, dtype=torch.float32, device=self.device)

    def _get_history_seq_base_tensor(self, history_note_ids: list[int]) -> torch.Tensor:
        seq_len = int(self.dien_train.SEQ_LEN)
        emb_dim = int(self.dien_train.EMB_DIM)
        note_ids = tuple(int(x) for x in history_note_ids[:seq_len] if int(x) >= 0)
        cached = self._history_seq_cache.get(note_ids)
        if cached is not None:
            self._history_seq_cache.move_to_end(note_ids)
            return cached

        if note_ids:
            seq_tensor = self.stores["t"].get_tensor(list(note_ids))
            seq_np = seq_tensor.detach().cpu().numpy().astype(np.float32)
        else:
            seq_np = np.zeros((0, emb_dim), dtype=np.float32)

        padded = np.zeros((seq_len, emb_dim), dtype=np.float32)
        if len(seq_np) > 0:
            take = min(seq_len, len(seq_np))
            padded[:take] = seq_np[:take]
        padded = np.flip(padded, axis=0).copy()
        base = torch.tensor(padded, dtype=torch.float32, device=self.device)
        self._history_seq_cache[note_ids] = base
        if len(self._history_seq_cache) > self._history_seq_cache_max:
            self._history_seq_cache.popitem(last=False)
        return base

    def _build_seq_tensor_from_history(self, history_note_ids: list[int], repeats: int) -> torch.Tensor:
        base = self._get_history_seq_base_tensor(history_note_ids)
        return base.unsqueeze(0).expand(int(repeats), -1, -1)

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
        with torch.no_grad():
            for i in range(0, len(df), batch_size):
                b = df.iloc[i : i + batch_size]
                batch_len = len(b)
                history_seq_tensor: torch.Tensor | None = None
                if history_note_ids:
                    history_seq_tensor = self._build_seq_tensor_from_history(
                        history_note_ids=history_note_ids,
                        repeats=batch_len,
                    )
                feats: dict[str, torch.Tensor] = {}
                for f in self.num_feats:
                    if f in b.columns:
                        col = b[f]
                        if pd.api.types.is_numeric_dtype(col):
                            vals = col.to_numpy(dtype=np.float32, copy=True)
                            vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
                        else:
                            vals = pd.to_numeric(col, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float32)
                    else:
                        vals = np.zeros(len(b), dtype=np.float32)
                    # 传入原始值；模型 forward() 已内置 z-score 归一化（via register_buffer）
                    feats[f] = torch.tensor(vals, dtype=torch.float32, device=self.device).unsqueeze(-1)

                gids = b[self.group_key].to_numpy(np.int64)
                nids = b["note_idx"].to_numpy(np.int64)
                if history_seq_tensor is not None:
                    feats["seq_embs"] = history_seq_tensor[:batch_len]
                else:
                    feats["seq_embs"] = torch.flip(self.stores["s"].get_tensor(gids), dims=[1])
                feats["note_text_emb"] = self.stores["t"].get_tensor(nids)
                feats["note_img_emb"] = self.stores["i"].get_tensor(nids)
                if self.scene == "search" and "q" in self.stores:
                    feats["query_emb"] = self._get_runtime_query_tensor(b, gids)

                p, _ = self.model(feats)
                cur = p.detach().float().cpu().numpy().reshape(-1)
                if len(cur) == batch_len:
                    preds[i : i + batch_len] = np.nan_to_num(cur, nan=0.0, posinf=1e6, neginf=-1e6)
                else:
                    preds[i : i + min(len(cur), batch_len)] = np.nan_to_num(cur[:batch_len], nan=0.0, posinf=1e6, neginf=-1e6)

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
    online_cap = int(os.environ.get("QILIN_ONLINE_DIEN_TOPN", "240"))
    dien_topn = min(len(out), max(0, online_cap))
    dien_pred = pd.to_numeric(out.get("gbdt_score", 0.0), errors="coerce").fillna(0.0).to_numpy(np.float32)
    if dien_topn > 0:
        dien_pred[:dien_topn] = predict_dien(out.iloc[:dien_topn], history_note_ids=history_note_ids)
    out["dien_score"] = np.nan_to_num(dien_pred, nan=0.0, posinf=1e6, neginf=-1e6)
    return out
