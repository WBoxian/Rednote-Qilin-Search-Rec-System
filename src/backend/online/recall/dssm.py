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

from backend.online.shared_query_encoder import get_shared_query_encoder

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


class DSSMRecaller:
    def __init__(self, scene: str, tag: str, deploy_tag_dir: Path, group_key: str):
        self.scene = scene
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
        self._precomp_user_blocks: list[tuple[np.ndarray, dict[str, int]]] = []
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

        ckpt = torch.load(user_tower_path, map_location="cpu")
        user_state = ckpt.get("state_dict", {})
        meta = ckpt.get("meta", {})
        cat_vocabs = meta.get("cat_vocabs", {})
        item_dense_dim = int(meta.get("item_dense_dim", 0))
        user_dense_dim = int(meta.get("user_dense_dim", 0))
        if not cat_vocabs or user_dense_dim <= 0:
            return

        model = train_mod.DSSMModel(
            scene=self.scene,
            cat_vocabs=cat_vocabs,
            item_dense_dim=item_dense_dim,
            user_dense_dim=user_dense_dim,
        ).to("cpu")
        model.load_state_dict(user_state, strict=False)
        model.eval()

        with open(item_map_path, "r", encoding="utf-8") as f:
            note2row = json.load(f)
        row2note = np.zeros(len(note2row), dtype=np.int64)
        for note_id, row_idx in note2row.items():
            idx = int(row_idx)
            if 0 <= idx < len(row2note):
                row2note[idx] = int(note_id)
        note2row_map = {int(note_id): int(row_idx) for note_id, row_idx in note2row.items()}

        index = faiss.read_index(str(ann_index_path))
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
                self.item_emb = np.memmap(str(item_emb_path), dtype="float32", mode="r", shape=(n_items, dim))
            except Exception:
                self.item_emb = None
        self.user_store = {"note_text": train_mod.MMapStore("note_text_emb", "note_text")}
        try:
            self.user_store["note_img"] = train_mod.MMapStore("note_img_emb", "note_img")
        except Exception:
            pass
        if self.scene == "search":
            try:
                self.user_store["search_query"] = train_mod.MMapStore("query_text_emb", "search_query")
            except Exception:
                pass

        self._load_precomputed_request_vecs(train_mod=train_mod)

        self.enabled = True

    def _get_precomp_user_vec(self, request_id: int) -> np.ndarray | None:
        """Return pre-exported request/user tower vector for this request_id."""
        key = str(int(request_id))
        for arr, row_map in self._precomp_user_blocks:
            row = row_map.get(key)
            if row is None:
                continue
            vec = arr[row]
            if not np.isfinite(vec).all():
                return None
            return vec.reshape(1, -1).astype(np.float32)
        return None

    def _load_precomputed_request_vecs(self, train_mod) -> None:
        def _candidate_files(split: str) -> list[tuple[Path, Path]]:
            out = [
                (
                    self.deploy_tag_dir / "index" / f"dssm_{self.scene}_{self.tag}_{split}_request_emb.bin",
                    self.deploy_tag_dir / "index" / f"dssm_{self.scene}_{self.tag}_{split}_request_map.json",
                )
            ]
            if self.scene == "search":
                out.append(
                    (
                        self.deploy_tag_dir / "index" / f"dssm_search_{self.tag}_{split}_query_emb.bin",
                        self.deploy_tag_dir / "index" / f"dssm_search_{self.tag}_{split}_query_map.json",
                    )
                )
            return out

        seen: set[tuple[str, str]] = set()
        for split in ["test", "train"]:
            for emb_p, map_p in _candidate_files(split):
                key = (str(emb_p), str(map_p))
                if key in seen or not (emb_p.exists() and map_p.exists()):
                    continue
                seen.add(key)
                try:
                    with open(map_p, encoding="utf-8") as f:
                        row_map = {str(k): int(v) for k, v in json.load(f).items()}
                    n_rows = len(row_map)
                    if n_rows <= 0:
                        continue
                    arr = np.memmap(str(emb_p), dtype="float32", mode="r", shape=(n_rows, train_mod.HIDDEN_DIM))
                    finite_rate = float(np.isfinite(arr).all(axis=1).mean())
                    if finite_rate < 0.95:
                        continue
                    self._precomp_user_blocks.append((arr, row_map))
                except Exception:
                    continue

    def _load_query_encoder(self) -> None:
        self.query_tokenizer, self.query_model = get_shared_query_encoder()

    def _encode_query_vec(self, text: str) -> np.ndarray | None:
        q = str(text or "").strip()
        if not q:
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
        ref_id = int(ref_request_id) if ref_request_id is not None else int(request_id)
        sq_store = self.user_store.get("search_query")
        if sq_store is not None:
            try:
                qv = sq_store.get_vec(int(ref_id)).detach().cpu().numpy().astype(np.float32).reshape(1, -1)
                if qv.size > 0 and np.isfinite(qv).all() and float(np.linalg.norm(qv)) > 1e-12:
                    return qv
            except Exception:
                pass
        q = str(query_text or "").strip()
        if not q:
            return None
        return self._encode_query_vec(q)

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

        dense_vals = [_fget(f"dense_feat{i}", 0.0) for i in range(1, 41)]
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
            "fans_num": torch.tensor([_fget("fans_num", 0.0)], dtype=torch.float32),
            "follows_num": torch.tensor([_fget("follows_num", 0.0)], dtype=torch.float32),
            "dense_feats": torch.tensor([dense_vals], dtype=torch.float32),
        }
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
                user_batch["query_vec"] = torch.zeros(1, self.train_mod.EMB_DIM, dtype=torch.float32)

        with torch.no_grad():
            user_vec = self.model.forward_user(user_batch, hist)
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
        ids = [int(x) for x in hist_ids if int(x) >= 0]
        if not ids:
            return (
                torch.zeros(1, self.train_mod.SEQ_MAX_LEN, self.train_mod.EMB_DIM, dtype=torch.float32),
                torch.zeros(1, self.train_mod.SEQ_MAX_LEN, dtype=torch.float32),
            )
        ids = ids[-max(1, int(self.train_mod.SEQ_MAX_LEN)) :]
        text_vecs = self.user_store["note_text"].get_batch_vecs(ids)
        img_store = self.user_store.get("note_img")
        img_vecs = img_store.get_batch_vecs(ids) if img_store is not None else None
        if hasattr(self.train_mod, "_fuse_history_modal_vecs"):
            fused = self.train_mod._fuse_history_modal_vecs(text_vecs=text_vecs, img_vecs=img_vecs)
        else:
            fused = text_vecs.float()
        seq = torch.zeros(1, self.train_mod.SEQ_MAX_LEN, self.train_mod.EMB_DIM, dtype=torch.float32)
        mask = torch.zeros(1, self.train_mod.SEQ_MAX_LEN, dtype=torch.float32)
        valid_len = int(fused.shape[0])
        seq[0, :valid_len] = fused[:valid_len].float()
        mask[0, :valid_len] = 1.0
        return seq, mask

    def _build_history_summary_tensor(self, hist_ids: list[int]) -> torch.Tensor:
        ids = [int(x) for x in hist_ids if int(x) >= 0]
        if not ids:
            return torch.zeros(1, self.train_mod.EMB_DIM, dtype=torch.float32)
        text_vecs = self.user_store["note_text"].get_batch_vecs(ids)
        img_store = self.user_store.get("note_img")
        img_vecs = img_store.get_batch_vecs(ids) if img_store is not None else None
        if hasattr(self.train_mod, "_fuse_history_modal_vecs"):
            fused = self.train_mod._fuse_history_modal_vecs(text_vecs=text_vecs, img_vecs=img_vecs)
        else:
            fused = text_vecs.float()
        if hasattr(self.train_mod, "summarize_history_sequence"):
            summary = self.train_mod.summarize_history_sequence(fused)
        else:
            summary = fused.mean(dim=0)
        return summary.reshape(1, -1).float()

    def _search_ann(self, user_vec: np.ndarray, max_rank: int) -> tuple[list[int], list[float]]:
        dists, idxs = self.ann_index.search(user_vec, int(max_rank))
        note_ids: list[int] = []
        scores: list[float] = []
        for row_idx, score in zip(idxs[0].tolist(), dists[0].tolist()):
            if row_idx < 0 or row_idx >= len(self.row2note):
                continue
            note_ids.append(int(self.row2note[row_idx]))
            scores.append(float(score))
        return note_ids, scores

    def _search_ann_by_history_items(self, hist_ids: list[int], max_rank: int, per_item_topk: int = 180) -> tuple[list[int], list[float]]:
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
    ) -> pd.DataFrame:
        if not self.enabled or self.ann_index is None or len(self.row2note) == 0:
            return pd.DataFrame()
        req_cur = req_df[req_df[group_key] == int(request_id)]
        feat_req = get_feat_req(int(request_id))
        query_text = ""
        if self.scene == "search" and len(feat_req) > 0:
            if len(req_cur) > 0 and "query" in req_cur.columns:
                query_text = str(req_cur.iloc[0].get("query") or "")
            if query_text:
                need_fill_query = (
                    "query" not in feat_req.columns
                    or str(feat_req.iloc[0].get("query") or "").strip() == ""
                )
                if need_fill_query:
                    feat_req = feat_req.copy()
                    feat_req["query"] = query_text
        if len(req_cur) > 0:
            user_idx = int(req_cur.iloc[0]["user_idx"])
        elif len(feat_req) > 0 and "user_idx" in feat_req.columns:
            try:
                user_idx = int(feat_req.iloc[0].get("user_idx", -1))
            except Exception:
                user_idx = -1
        else:
            user_idx = -1
        ref_request_id = int(request_id)
        if self.scene == "search" and len(feat_req) > 0:
            try:
                ref_request_id = int(pd.to_numeric(feat_req.iloc[0].get("query_ref_request_id", request_id), errors="coerce"))
            except Exception:
                ref_request_id = int(request_id)
        # Prefer pre-computed user tower vector; fall back to runtime build
        user_vec = self._get_precomp_user_vec(int(ref_request_id))
        if user_vec is None:
            user_vec = self._build_user_vec(
                feat_req=feat_req,
                request_id=int(request_id),
                user_idx=user_idx,
                ref_request_id=int(ref_request_id),
            )
        if user_vec is None:
            return pd.DataFrame()

        ann_note_ids, ann_scores = self._search_ann(user_vec=user_vec, max_rank=int(max_rank))
        if not ann_note_ids:
            return pd.DataFrame()

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
