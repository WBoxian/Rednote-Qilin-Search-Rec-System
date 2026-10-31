"""在线 DSSM 召回模型加载与 ANN 召回执行。"""

from __future__ import annotations

import json
from collections import OrderedDict
import importlib.util
from pathlib import Path
from typing import Callable
import numpy as np
import pandas as pd
import torch

try:
    from transformers import AutoModel, AutoTokenizer
except Exception:
    AutoModel = None
    AutoTokenizer = None

try:
    import faiss
except Exception:
    faiss = None

from backend.online.shared_query_encoder import get_shared_query_encoder, encode_current_query

BASE_DIR = Path(__file__).resolve().parents[4]


def _load_train_module():
    try:
        from recall import dssm_trainer as train_mod_cls
        return train_mod_cls
    except Exception:
        trainer_path = BASE_DIR / "src" / "recall" / "dssm_trainer.py"
        spec = importlib.util.spec_from_file_location("qilin_recall_dssm_trainer", trainer_path)
        if spec is None or spec.loader is None:
            raise
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


class _NoteRows:
    """Compact immutable reverse lookup for the already bound ANN row array."""
    def __init__(self, row2note):
        values = np.asarray(row2note)
        if values.ndim != 1 or values.dtype.kind not in "iu" or (values < 0).any():
            raise ValueError("ANN row mapping requires nonnegative integer note IDs")
        self.order = None if np.all(values[1:] > values[:-1]) else np.argsort(values, kind="stable")
        self.ids = values if self.order is None else values[self.order]
        if (self.ids[1:] <= self.ids[:-1]).any():
            raise ValueError("ANN note IDs must be unique")

    def get(self, note_id, default=None):
        index = int(np.searchsorted(self.ids, int(note_id)))
        if index >= len(self.ids) or self.ids[index] != int(note_id):
            return default
        return index if self.order is None else int(self.order[index])


class _SharedNoteStore:
    def __init__(self, modality):
        from backend.online.ranking.stack import shared_mapped_vectors
        self.vectors = shared_mapped_vectors(BASE_DIR/"embeddings"/f"note_{modality}_emb", f"note_{modality}")
        self.dim = self.vectors.dimension

    def get_batch_vecs(self, ids):
        return torch.from_numpy(self.vectors.get(ids))

    def get_vec(self, note_id):
        return self.get_batch_vecs([note_id])[0]


class DSSMRecaller:
    def __init__(self, scene: str, tag: str, deploy_tag_dir: Path, group_key: str, *, query_encoder_dir=None, query_encoder_revision=None, use_shared_note_vectors=False):
        self.scene = scene
        self.use_shared_note_vectors = bool(use_shared_note_vectors)
        self.query_encoder_dir = query_encoder_dir
        self.query_encoder_revision = query_encoder_revision
        self.tag = tag
        self.deploy_tag_dir = deploy_tag_dir
        self.group_key = group_key
        self.model = None
        self.train_mod = None
        self.ann_index = None
        self.row2note = np.zeros(0, dtype=np.int64)
        self.note2row: dict[int, int] = {}
        self.item_emb = None
        self.user_store: dict[str, object] = {}
        self.query_tokenizer = None
        self.query_model = None
        self._query_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._query_cache_max = 2048
        self.enabled = False
        self._load_assets()

    def _artifact_path(self, rel_path: str) -> Path:
        return self.deploy_tag_dir / rel_path

    def _load_assets(self) -> None:
        if faiss is None:
            return
        train_mod = _load_train_module()

        user_tower_path = self._artifact_path(f"models/dssm_{self.scene}_{self.tag}_user_tower.pt")
        item_map_path = self._artifact_path(f"index/dssm_{self.scene}_{self.tag}_item_map.json")
        item_meta_path = self._artifact_path(f"index/dssm_{self.scene}_{self.tag}_item_meta.json")
        item_emb_path = self._artifact_path(f"index/dssm_{self.scene}_{self.tag}_item_emb.bin")
        ann_index_path = self._artifact_path(f"index/dssm_{self.scene}_{self.tag}_ivfpq.faiss")
        ann_meta_path = self._artifact_path(f"index/dssm_{self.scene}_{self.tag}_ivfpq_meta.json")
        if not (user_tower_path.exists() and item_map_path.exists() and ann_index_path.exists()):
            return

        from recall.residual_dssm import model_from_checkpoint
        model = model_from_checkpoint(user_tower_path, device="cpu", allow_partial=True)
        meta = model.checkpoint_meta
        if meta.get("scene", self.scene) != self.scene:
            raise ValueError("DSSM checkpoint scene differs from the serving scene")
        self.checkpoint_meta = meta
        self.model_family = meta.get("model_family", "dssm_v1")
        self.retrieval_dim = int(getattr(model, "retrieval_dim", 128))
        if self.model_family == "residual_dssm_v1":
            from training.history import HISTORY_ORDER
            if self.scene != "rec" or meta.get("history_contract") != HISTORY_ORDER or not bool(model.seq_mask_padding) or not bool(model.user_mask_profile):
                raise ValueError("Residual DSSM requires the canonical history and persistent sequence/profile masks")
            known, unknown = meta.get("known_user_ids"), meta.get("unknown_user_id")
            if not isinstance(known, list) or isinstance(unknown, bool) or not isinstance(unknown, int) or unknown != int(meta["cat_vocabs"]["user_idx"])-1:
                raise ValueError("Residual DSSM requires an explicit reserved unknown-user contract")
            self.known_user_ids = frozenset(map(int, known))
        self.history_contract = meta.get("history_contract", "legacy_raw_order_v0")
        self.masked_user_dense = bool(meta.get("masked_user_dense", False))

        row_path = self._artifact_path(f"index/dssm_{self.scene}_{self.tag}_row2note.npy")
        if row_path.exists():
            row2note = np.load(row_path, mmap_mode="r")
            note2row_map = _NoteRows(row2note)
        else:
            with open(item_map_path, "r", encoding="utf-8") as stream:
                note2row = json.load(stream)
            row2note = np.zeros(len(note2row), dtype=np.int64)
            for note_id, row_idx in note2row.items():
                row2note[int(row_idx)] = int(note_id)
            note2row_map = {int(note): int(row) for note, row in note2row.items()}

        index = faiss.read_index(str(ann_index_path))
        if index.ntotal != len(row2note):
            raise ValueError("ANN index and row mapping sizes differ")
        if index.d != self.retrieval_dim or index.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise ValueError("DSSM model and inner-product ANN dimensions/metric differ")
        nprobe = 32
        if ann_meta_path.exists():
            try:
                meta_obj = json.loads(ann_meta_path.read_text(encoding="utf-8"))
                nprobe = int(meta_obj.get("nprobe", nprobe))
            except Exception:
                pass
        if hasattr(index, "nprobe"):
            index.nprobe = nprobe

        self.train_mod = train_mod
        self.model = model
        self.row2note = row2note
        self.note2row = note2row_map
        self.ann_index = index
        if item_meta_path.exists() and item_emb_path.exists():
            try:
                with open(item_meta_path, "r", encoding="utf-8") as f:
                    item_meta = json.load(f)
                n_items = int(item_meta["num_items"])
                dim = int(item_meta["dim"])
                if dim != self.retrieval_dim or n_items != len(row2note):
                    raise ValueError("DSSM item vectors differ from the model/index shape")
                self.item_emb = np.memmap(str(item_emb_path), dtype="float32", mode="r", shape=(n_items, dim))
            except (OSError, KeyError, TypeError, ValueError) as error:
                raise ValueError("Invalid bound DSSM item vectors") from error
        if self.use_shared_note_vectors:
            self.user_store = {"note_text": _SharedNoteStore("text"), "note_img": _SharedNoteStore("img")}
        else:
            self.user_store = {"note_text": train_mod.MMapStore("note_text_emb", "note_text")}
            try:
                self.user_store["note_img"] = train_mod.MMapStore("note_img_emb", "note_img")
            except Exception:
                pass
        if self.scene == "search" and self.query_encoder_dir is None:
            try:
                self.user_store["search_query"] = train_mod.MMapStore("query_text_emb", "search_query")
            except Exception:
                pass

        # Rebuild towers from current request inputs; historical vectors may use another query/history.

        self.enabled = True

    def _load_query_encoder(self) -> None:
        directory = getattr(self, "query_encoder_dir", None)
        self.query_tokenizer, self.query_model = (get_shared_query_encoder(directory, getattr(self, "query_encoder_revision", None))
            if directory is not None else get_shared_query_encoder())

    def _encode_query_vec(self, text: str) -> np.ndarray | None:
        q = str(text or "").strip()
        if not q:
            return None
        directory = getattr(self, "query_encoder_dir", None)
        if directory is not None:
            try:
                return encode_current_query(q, directory, getattr(self, "query_encoder_revision", None)).reshape(1, -1)
            except (OSError, ValueError, RuntimeError):
                return None
        if self.query_tokenizer is None or self.query_model is None:
            self._load_query_encoder()
        if self.query_tokenizer is None or self.query_model is None:
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

    def score_note_text_candidates(
        self,
        query_text: str,
        note_ids: list[int],
        topk: int = 200,
    ) -> tuple[list[int], list[float]]:
        qvec = self._encode_query_vec(query_text)
        if qvec is None or not note_ids or "note_text" not in self.user_store:
            return [], []
        note_ids = [int(x) for x in note_ids if int(x) >= 0]
        if not note_ids:
            return [], []
        try:
            note_vecs = self.user_store["note_text"].get_batch_vecs(note_ids).detach().cpu().numpy().astype(np.float32)
            qv = qvec.reshape(-1).astype(np.float32)
            qn = float(np.linalg.norm(qv))
            if qn <= 1e-12:
                return [], []
            vn = np.linalg.norm(note_vecs, axis=1)
            sim = np.divide(
                note_vecs @ qv,
                vn * qn,
                out=np.zeros(len(note_ids), dtype=np.float32),
                where=vn > 1e-12,
            ).astype(np.float32)
            sim = np.where(np.isfinite(sim), sim, 0.0).astype(np.float32)
            order = np.argsort(-sim, kind="mergesort")[: max(1, int(topk))]
            out_ids = [int(note_ids[i]) for i in order.tolist() if float(sim[i]) > 0.0]
            out_scores = [float(sim[i]) for i in order.tolist() if float(sim[i]) > 0.0]
            return out_ids, out_scores
        except Exception:
            return [], []

    def _get_request_query_vec(self, request_id: int, query_text: str, ref_request_id: int | None = None) -> np.ndarray | None:
        if self.scene != "search":
            return None
        # A typed query may differ from the reference request. Never reuse another query's vector.
        q = str(query_text or "").strip()
        if q:
            return self._encode_query_vec(q)
        ref_id = int(ref_request_id) if ref_request_id is not None else int(request_id)
        sq_store = self.user_store.get("search_query")
        if sq_store is not None:
            try:
                qv = sq_store.get_vec(int(ref_id)).detach().cpu().numpy().astype(np.float32).reshape(1, -1)
                if qv.size > 0 and np.isfinite(qv).all() and float(np.linalg.norm(qv)) > 1e-12:
                    return qv
            except Exception:
                pass
        return None

    def _build_user_vec(
        self,
        feat_req: pd.DataFrame,
        request_id: int,
        user_idx: int,
        ref_request_id: int | None = None,
    ) -> np.ndarray | None:
        if self.model is None or self.train_mod is None or feat_req.empty:
            return None
        row = feat_req.iloc[0]

        def _iget(name: str, default: int = 0) -> int:
            v = row.get(name, default)
            if pd.isna(v):
                return int(default)
            return int(v)

        def _fget(name: str, default: float = 0.0) -> float:
            v = row.get(name, default)
            if pd.isna(v):
                return float(default)
            return float(v)

        masked = bool(getattr(self, "masked_user_dense", False))
        dense_vals = [0.0] * 40 if masked else [_fget(f"dense_feat{i}", 0.0) for i in range(1, 41)]
        hist_raw = row.get("recent_clicked_note_idxs", [])
        if isinstance(hist_raw, np.ndarray):
            hist_ids = [int(x) for x in hist_raw.tolist()]
        elif isinstance(hist_raw, list):
            hist_ids = [int(x) for x in hist_raw]
        else:
            hist_ids = []

        hist = self._build_history_summary_tensor(hist_ids)
        hist_seq, hist_mask = self._build_history_sequence_tensor(hist_ids)

        user_batch = {
            "user_idx": torch.tensor([int(user_idx)], dtype=torch.long),
            "gender_enc": torch.tensor([_iget("gender_enc", 0)], dtype=torch.long),
            "platform_enc": torch.tensor([_iget("platform_enc", 0)], dtype=torch.long),
            "age_enc": torch.tensor([_iget("age_enc", 0)], dtype=torch.long),
            "location_enc": torch.tensor([_iget("location_enc", 0)], dtype=torch.long),
            "fans_num": torch.tensor([0.0 if masked else _fget("fans_num", 0.0)], dtype=torch.float32),
            "follows_num": torch.tensor([0.0 if masked else _fget("follows_num", 0.0)], dtype=torch.float32),
            "dense_feats": torch.tensor([dense_vals], dtype=torch.float32),
        }
        if getattr(self, "model_family", "dssm_v1") == "residual_dssm_v1":
            from recall.residual_dssm import raw_history_text_mean
            meta = self.checkpoint_meta
            if int(user_idx) not in self.known_user_ids:
                user_batch["user_idx"] = torch.tensor([int(meta["unknown_user_id"])], dtype=torch.long)
            content = raw_history_text_mean(hist_ids, self.user_store["note_text"])
            user_batch["residual_history_text"] = torch.from_numpy(content.reshape(1, 768))
        if self.scene == "search":
            query_text = str(row.get("query", "") or "").strip()
            qvec = self._get_request_query_vec(
                request_id=int(request_id),
                query_text=query_text,
                ref_request_id=ref_request_id,
            )
            if qvec is not None:
                user_batch["query_vec"] = torch.from_numpy(qvec)
            else:
                return None

        with torch.no_grad():
            user_vec = self.model.forward_user(user_batch, hist, seq_tokens=hist_seq, seq_mask=hist_mask)
        return user_vec.detach().float().cpu().numpy().astype(np.float32)

    def _build_query_only_vec(self, query_text: str) -> np.ndarray | None:
        if self.scene != "search" or self.model is None or self.train_mod is None:
            return None
        q = str(query_text or "").strip()
        if not q:
            return None
        qvec = self._encode_query_vec(q)
        if qvec is None:
            return None
        zeros_dense = [0.0] * 40
        user_batch = {
            "user_idx": torch.tensor([0], dtype=torch.long),
            "gender_enc": torch.tensor([0], dtype=torch.long),
            "platform_enc": torch.tensor([0], dtype=torch.long),
            "age_enc": torch.tensor([0], dtype=torch.long),
            "location_enc": torch.tensor([0], dtype=torch.long),
            "fans_num": torch.tensor([0.0], dtype=torch.float32),
            "follows_num": torch.tensor([0.0], dtype=torch.float32),
            "dense_feats": torch.tensor([zeros_dense], dtype=torch.float32),
            "query_vec": torch.from_numpy(qvec),
        }
        hist = torch.zeros(1, self.train_mod.EMB_DIM, dtype=torch.float32)
        with torch.no_grad():
            user_vec = self.model.forward_user(user_batch, hist)
        return user_vec.detach().float().cpu().numpy().astype(np.float32)

    def _build_history_only_vec(self, feat_req: pd.DataFrame) -> np.ndarray | None:
        if self.model is None or self.train_mod is None or feat_req.empty:
            return None
        row = feat_req.iloc[0]
        hist_raw = row.get("recent_clicked_note_idxs", [])
        if isinstance(hist_raw, np.ndarray):
            hist_ids = [int(x) for x in hist_raw.tolist()]
        elif isinstance(hist_raw, list):
            hist_ids = [int(x) for x in hist_raw]
        else:
            hist_ids = []
        if not hist_ids:
            return None

        hist = self._build_history_summary_tensor(hist_ids)
        hist_seq, hist_mask = self._build_history_sequence_tensor(hist_ids)
        zeros_dense = [0.0] * 40
        user_batch = {
            "user_idx": torch.tensor([0], dtype=torch.long),
            "gender_enc": torch.tensor([0], dtype=torch.long),
            "platform_enc": torch.tensor([0], dtype=torch.long),
            "age_enc": torch.tensor([0], dtype=torch.long),
            "location_enc": torch.tensor([0], dtype=torch.long),
            "fans_num": torch.tensor([0.0], dtype=torch.float32),
            "follows_num": torch.tensor([0.0], dtype=torch.float32),
            "dense_feats": torch.tensor([zeros_dense], dtype=torch.float32),
        }
        if self.scene == "search":
            user_batch["query_vec"] = torch.zeros(1, self.train_mod.EMB_DIM, dtype=torch.float32)
        with torch.no_grad():
            user_vec = self.model.forward_user(user_batch, hist, seq_tokens=hist_seq, seq_mask=hist_mask)
        return user_vec.detach().float().cpu().numpy().astype(np.float32)

    def _build_history_sequence_tensor(self, hist_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        sequence, mask = self.train_mod.build_history_sequence_from_ids(hist_ids, self.user_store,
            history_contract=getattr(self, "history_contract", "legacy_raw_order_v0"))
        return sequence.unsqueeze(0), mask.unsqueeze(0)

    def _build_history_summary_tensor(self, hist_ids: list[int]) -> torch.Tensor:
        return self.train_mod.build_history_summary_from_ids(hist_ids, self.user_store,
            history_contract=getattr(self, "history_contract", "legacy_raw_order_v0")).reshape(1, -1).float()

    def _search_ann(self, user_vec: np.ndarray, max_rank: int) -> tuple[list[int], list[float]]:
        values = np.asarray(user_vec, dtype=np.float32)
        if values.shape != (1, int(self.ann_index.d)) or not np.isfinite(values).all():
            raise ValueError("DSSM query vector differs from its bound ANN dimensions")
        # Residual towers deliberately retain sqrt(weight) norm for empty content; never renormalize.
        dists, idxs = self.ann_index.search(np.ascontiguousarray(values), int(max_rank))
        note_ids: list[int] = []
        scores: list[float] = []
        for row_idx, score in zip(idxs[0].tolist(), dists[0].tolist()):
            if row_idx < 0 or row_idx >= len(self.row2note):
                continue
            note_ids.append(int(self.row2note[row_idx]))
            scores.append(float(score))
        return note_ids, scores

    def _search_ann_by_history_items(self, hist_ids: list[int], max_rank: int, per_item_topk: int = 180) -> tuple[list[int], list[float]]:
        if getattr(self, "model_family", "dssm_v1") == "residual_dssm_v1":
            raise ValueError("Residual DSSM supports current-request IP retrieval without legacy normalized expansion")
        if self.ann_index is None or self.item_emb is None or not hist_ids:
            return [], []
        scores: dict[int, float] = {}
        hit_counts: dict[int, int] = {}
        hist_set = {int(x) for x in hist_ids}
        recent_hist = hist_ids[-20:]
        for idx, note_idx in enumerate(reversed(recent_hist)):
            row = self.note2row.get(int(note_idx))
            if row is None:
                continue
            q = np.asarray(self.item_emb[int(row)], dtype=np.float32).reshape(1, -1)
            q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
            dists, idxs = self.ann_index.search(q.astype(np.float32), int(max(20, per_item_topk)))
            decay = 1.0 / (1.0 + 0.35 * idx)
            for row_idx, score in zip(idxs[0].tolist(), dists[0].tolist()):
                if row_idx < 0 or row_idx >= len(self.row2note):
                    continue
                cand = int(self.row2note[row_idx])
                if cand in hist_set:
                    continue
                scores[cand] = scores.get(cand, 0.0) + decay * float(score)
                hit_counts[cand] = hit_counts.get(cand, 0) + 1
        if not scores:
            return [], []
        for cand, cnt in hit_counts.items():
            if cnt > 1:
                scores[cand] += 0.035 * float(cnt - 1)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[: max(1, int(max_rank))]
        return [int(k) for k, _ in ranked], [float(v) for _, v in ranked]

    def _fuse_dual_ann(
        self,
        main_note_ids: list[int],
        main_scores: list[float],
        query_note_ids: list[int],
        query_scores: list[float],
    ) -> tuple[list[int], list[float], list[float]]:
        if not main_note_ids:
            return [], [], []

        rank_main = {int(n): i + 1 for i, n in enumerate(main_note_ids)}
        rank_query = {int(n): i + 1 for i, n in enumerate(query_note_ids)}
        score_main = {int(n): float(s) for n, s in zip(main_note_ids, main_scores)}
        score_query = {int(n): float(s) for n, s in zip(query_note_ids, query_scores)}

        union = []
        seen = set()
        for n in main_note_ids + query_note_ids:
            ni = int(n)
            if ni in seen:
                continue
            seen.add(ni)
            union.append(ni)

        if not union:
            return [], [], []

        k = 60.0
        fused_vals: list[float] = []
        ann_vals: list[float] = []
        for n in union:
            rm = float(rank_main.get(n, 10_000))
            rq = float(rank_query.get(n, 10_000))
            rrf_main = 1.0 / (k + rm)
            rrf_query = 1.0 / (k + rq)
            fused = 0.7 * rrf_main + 0.3 * rrf_query
            fused_vals.append(float(fused))
            ann_vals.append(float(max(score_main.get(n, 0.0), score_query.get(n, 0.0))))

        order = np.argsort(-np.asarray(fused_vals, dtype=np.float32), kind="mergesort")
        out_ids = [int(union[i]) for i in order.tolist()]
        out_fused = [float(fused_vals[i]) for i in order.tolist()]
        out_ann = [float(ann_vals[i]) for i in order.tolist()]
        return out_ids, out_ann, out_fused

    def _fuse_search_query_rank(
        self,
        request_id: int,
        note_ids: list[int],
        ann_scores: np.ndarray,
        query_text: str,
        ref_request_id: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if self.scene != "search":
            return None
        q = str(query_text or "").strip()
        if not q or len(note_ids) <= 1:
            return None
        qvec = self._get_request_query_vec(
            request_id=int(request_id),
            query_text=q,
            ref_request_id=ref_request_id,
        )
        if qvec is None:
            return None
        try:
            note_vecs = self.user_store["note_text"].get_batch_vecs(note_ids).detach().cpu().numpy().astype(np.float32)
            qv = qvec.reshape(-1).astype(np.float32)
            qn = float(np.linalg.norm(qv))
            if qn <= 1e-12:
                return None
            vn = np.linalg.norm(note_vecs, axis=1)
            sim = np.divide(
                note_vecs @ qv,
                vn * qn,
                out=np.zeros(len(note_ids), dtype=np.float32),
                where=vn > 1e-12,
            ).astype(np.float32)
            sim = np.where(np.isfinite(sim), sim, 0.0).astype(np.float32)

            n = len(note_ids)
            ann_rank_pos = np.arange(1, n + 1, dtype=np.float32)
            txt_order = np.argsort(-sim, kind="mergesort")
            txt_rank_pos = np.empty(n, dtype=np.float32)
            txt_rank_pos[txt_order] = np.arange(1, n + 1, dtype=np.float32)
            rrf_k = np.float32(60.0)
            ann_rrf = 1.0 / (rrf_k + ann_rank_pos)
            txt_rrf = 1.0 / (rrf_k + txt_rank_pos)
            fused = (0.75 * ann_rrf + 0.25 * txt_rrf).astype(np.float32)
            order = np.argsort(-fused, kind="mergesort")
            return order.astype(np.int64), fused
        except Exception:
            return None

    def fetch_candidates(
        self,
        request_id: int,
        max_rank: int,
        req_df: pd.DataFrame,
        get_feat_req: Callable[[int], pd.DataFrame],
        group_key: str,
        pure_request_ann: bool = False,
    ) -> pd.DataFrame:
        if not self.enabled or self.ann_index is None or len(self.row2note) == 0:
            return pd.DataFrame()
        req_cur = req_df[req_df[group_key] == int(request_id)]
        feat_req = get_feat_req(int(request_id))
        query_text = ""
        if self.scene == "search" and len(feat_req) > 0:
            query_text = str(feat_req.iloc[0].get("query") or "")
            if not query_text and len(req_cur) > 0 and "query" in req_cur.columns:
                query_text = str(req_cur.iloc[0].get("query") or "")
            if query_text:
                feat_req = feat_req.copy()
                feat_req["query"] = query_text
        if len(feat_req) > 0 and "user_idx" in feat_req.columns:
            user_idx = int(feat_req.iloc[0]["user_idx"])
        elif len(req_cur) > 0:
            user_idx = int(req_cur.iloc[0]["user_idx"])
        else:
            user_idx = -1
        ref_request_id = int(request_id)
        if self.scene == "search" and len(feat_req) > 0:
            try:
                ref_request_id = int(pd.to_numeric(feat_req.iloc[0].get("query_ref_request_id", request_id), errors="coerce"))
            except Exception:
                ref_request_id = int(request_id)
        user_vec = self._build_user_vec(
            feat_req=feat_req,
            request_id=int(request_id),
            user_idx=user_idx,
            ref_request_id=int(ref_request_id),
        )
        if user_vec is None:
            if pure_request_ann:
                raise RuntimeError("Current request tower input could not be encoded")
            return pd.DataFrame()

        ann_note_ids, ann_scores = self._search_ann(user_vec=user_vec, max_rank=int(max_rank))
        if not ann_note_ids:
            return pd.DataFrame()
        if pure_request_ann:
            return pd.DataFrame({group_key: int(request_id), "user_idx": user_idx,
                "note_idx": ann_note_ids, "rank": np.arange(1, len(ann_note_ids) + 1),
                "score_ann": ann_scores, "recall_score": ann_scores, "first_route": "ann"})

        fused_note_ids = list(ann_note_ids)
        fused_ann_scores = list(ann_scores)
        fused_chain_scores = list(ann_scores)

        if self.scene == "search" and str(query_text or "").strip():
            query_vec = self._build_query_only_vec(query_text=query_text)
            if query_vec is not None:
                q_note_ids, q_scores = self._search_ann(user_vec=query_vec, max_rank=int(max_rank))
                if q_note_ids:
                    fused_note_ids, fused_ann_scores, fused_chain_scores = self._fuse_dual_ann(
                        main_note_ids=ann_note_ids,
                        main_scores=ann_scores,
                        query_note_ids=q_note_ids,
                        query_scores=q_scores,
                    )

        fused_order = np.arange(len(fused_note_ids), dtype=np.int64)
        fused_scores = np.asarray(fused_chain_scores, dtype=np.float32)
        fused_applied = False
        fused_ret = self._fuse_search_query_rank(
            request_id=int(request_id),
            note_ids=fused_note_ids,
            ann_scores=np.asarray(fused_ann_scores, dtype=np.float32),
            query_text=query_text,
            ref_request_id=int(ref_request_id),
        )
        if fused_ret is not None:
            fused_order, fused_scores = fused_ret
            fused_applied = True

        rows = []
        for rk, src_idx in enumerate(fused_order.tolist(), start=1):
            note_idx = int(fused_note_ids[int(src_idx)])
            ann_score = float(fused_ann_scores[int(src_idx)])
            final_score = float(fused_scores[int(src_idx)]) if fused_applied else ann_score
            rows.append(
                {
                    group_key: int(request_id),
                    "user_idx": int(user_idx),
                    "note_idx": note_idx,
                    "rank": int(rk),
                    "recall_score": final_score,
                    "score_ann": final_score if fused_applied else ann_score,
                    "score_swing": 0.0,
                    "score_usercf": 0.0,
                    "from_ann": 1,
                    "from_swing": 0,
                    "from_usercf": 0,
                    "first_route": "ann",
                }
            )
        return pd.DataFrame(rows)


    def expand_seed_candidates(
        self,
        seed_scores: dict[int, float],
        max_rank: int,
        max_seed: int = 80,
    ) -> tuple[list[int], list[float]]:
        if getattr(self, "model_family", "dssm_v1") == "residual_dssm_v1":
            raise ValueError("Residual DSSM supports current-request IP retrieval without legacy normalized expansion")
        if self.ann_index is None or self.item_emb is None or not seed_scores:
            return [], []
        ranked_seed = sorted(seed_scores.items(), key=lambda x: x[1], reverse=True)[: max(1, int(max_seed))]
        rows = []
        weights = []
        seed_set = set()
        for note_idx, score in ranked_seed:
            row = self.note2row.get(int(note_idx))
            if row is None:
                continue
            rows.append(int(row))
            weights.append(float(max(0.0, score)))
            seed_set.add(int(note_idx))
        if not rows:
            return [], []
        vecs = np.asarray(self.item_emb[rows], dtype=np.float32)
        ws = np.asarray(weights, dtype=np.float32)
        if float(ws.sum()) <= 1e-12:
            ws = np.ones_like(ws, dtype=np.float32)
        ws = ws / np.maximum(ws.sum(), 1e-12)
        q = (vecs * ws.reshape(-1, 1)).sum(axis=0, keepdims=True)
        q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
        dists, idxs = self.ann_index.search(q.astype(np.float32), int(max(20, max_rank)))
        out_ids = []
        out_scores = []
        for row_idx, score in zip(idxs[0].tolist(), dists[0].tolist()):
            if row_idx < 0 or row_idx >= len(self.row2note):
                continue
            note_idx = int(self.row2note[row_idx])
            if note_idx in seed_set:
                continue
            out_ids.append(int(note_idx))
            out_scores.append(float(score))
        return out_ids, out_scores

    def fetch_history_candidates(
        self,
        request_id: int,
        max_rank: int,
        req_df: pd.DataFrame,
        get_feat_req: Callable[[int], pd.DataFrame],
        group_key: str,
    ) -> pd.DataFrame:
        if not self.enabled or self.ann_index is None or len(self.row2note) == 0:
            return pd.DataFrame()
        feat_req = get_feat_req(int(request_id))
        if feat_req.empty:
            return pd.DataFrame()
        req_cur = req_df[req_df[group_key] == int(request_id)]
        if len(req_cur) > 0:
            user_idx = int(req_cur.iloc[0].get("user_idx", -1))
        elif "user_idx" in feat_req.columns:
            try:
                user_idx = int(feat_req.iloc[0].get("user_idx", -1))
            except Exception:
                user_idx = -1
        else:
            user_idx = -1

        row = feat_req.iloc[0]
        hist_raw = row.get("recent_clicked_note_idxs", [])
        if isinstance(hist_raw, np.ndarray):
            hist_ids = [int(x) for x in hist_raw.tolist()]
        elif isinstance(hist_raw, list):
            hist_ids = [int(x) for x in hist_raw]
        else:
            hist_ids = []

        item_note_ids, item_scores = self._search_ann_by_history_items(hist_ids=hist_ids, max_rank=int(max_rank))
        mean_note_ids: list[int] = []
        mean_scores: list[float] = []
        user_vec = self._build_history_only_vec(feat_req=feat_req)
        if user_vec is not None:
            mean_note_ids, mean_scores = self._search_ann(user_vec=user_vec, max_rank=int(max_rank))

        merged_scores: dict[int, float] = {}
        for note_idx, score in zip(item_note_ids, item_scores):
            merged_scores[int(note_idx)] = merged_scores.get(int(note_idx), 0.0) + float(score) * 1.0
        for note_idx, score in zip(mean_note_ids, mean_scores):
            merged_scores[int(note_idx)] = merged_scores.get(int(note_idx), 0.0) + float(score) * 0.62
        if not merged_scores:
            return pd.DataFrame()
        ranked = sorted(merged_scores.items(), key=lambda x: x[1], reverse=True)[: max(1, int(max_rank))]
        note_ids = [int(k) for k, _ in ranked]
        scores = [float(v) for _, v in ranked]
        if not note_ids:
            return pd.DataFrame()
        rows = []
        for rk, (note_idx, score) in enumerate(zip(note_ids, scores), start=1):
            rows.append(
                {
                    group_key: int(request_id),
                    "user_idx": int(user_idx),
                    "note_idx": int(note_idx),
                    "rank": int(rk),
                    "recall_score": float(score),
                    "score_ann": 0.0,
                    "score_swing": float(score),
                    "score_usercf": 0.0,
                    "from_ann": 0,
                    "from_swing": 1,
                    "from_usercf": 0,
                    "first_route": "seq",
                }
            )
        return pd.DataFrame(rows)
