"""在线 DSSM 召回模型加载与 ANN 召回执行。"""

from __future__ import annotations

import json
from collections import OrderedDict
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

BASE_DIR = Path(__file__).resolve().parents[4]


class DSSMRecaller:
    _shared_query_tokenizer = None
    _shared_query_model = None

    def __init__(self, scene: str, tag: str, deploy_tag_dir: Path, group_key: str):
        self.scene = scene
        self.tag = tag
        self.deploy_tag_dir = deploy_tag_dir
        self.group_key = group_key
        self.model = None
        self.train_mod = None
        self.ann_index = None
        self.row2note = np.zeros(0, dtype=np.int64)
        self.user_store: dict[str, object] = {}
        self.query_tokenizer = None
        self.query_model = None
        self._query_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._query_cache_max = 2048
        self._precomp_user_vecs: np.ndarray | None = None   # (n, HIDDEN_DIM)
        self._precomp_user_map: dict[str, int] = {}         # search_idx_str -> row
        self.enabled = False
        self._load_assets()

    def _artifact_path(self, rel_path: str) -> Path:
        return self.deploy_tag_dir / rel_path

    def _load_assets(self) -> None:
        if faiss is None:
            return
        from recall import dssm_trainer as train_mod_cls
        train_mod = train_mod_cls

        user_tower_path = self._artifact_path(f"models/dssm_{self.scene}_{self.tag}_user_tower.pt")
        item_map_path = self._artifact_path(f"index/dssm_{self.scene}_{self.tag}_item_map.json")
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
        self.ann_index = index
        self.user_store = {"note_text": train_mod.MMapStore("note_text_emb", "note_text")}
        if self.scene == "search":
            try:
                self.user_store["search_query"] = train_mod.MMapStore("query_text_emb", "search_query")
            except Exception:
                pass

        # Load pre-exported user-tower vectors (test split) with quality gate
        if self.scene == "search":
            for split in ["test", "train"]:
                emb_p = self.deploy_tag_dir / "index" / f"dssm_{self.scene}_{self.tag}_{split}_query_emb.bin"
                map_p = self.deploy_tag_dir / "index" / f"dssm_{self.scene}_{self.tag}_{split}_query_map.json"
                if not (emb_p.exists() and map_p.exists()):
                    continue
                try:
                    with open(map_p) as _f:
                        _m = json.load(_f)
                    _n = len(_m)
                    _arr = np.memmap(str(emb_p), dtype="float32", mode="r", shape=(_n, train_mod.HIDDEN_DIM))
                    # Quality gate: reject if more than 5% NaN/inf
                    _finite_rate = float(np.isfinite(_arr).all(axis=1).mean())
                    if _finite_rate < 0.95:
                        continue
                    if self._precomp_user_vecs is None:
                        self._precomp_user_vecs = _arr
                        self._precomp_user_map = {str(k): int(v) for k, v in _m.items()}
                    else:
                        # Merge maps (test overrides train where conflict)
                        merged_map = dict(self._precomp_user_map)
                        old_n = self._precomp_user_vecs.shape[0]
                        merged_map.update({str(k): int(v) + old_n for k, v in _m.items()})
                        merged_arr = np.concatenate([np.array(self._precomp_user_vecs), np.array(_arr)], axis=0)
                        self._precomp_user_vecs = merged_arr
                        self._precomp_user_map = merged_map
                except Exception:
                    pass

        self.enabled = True

    def _get_precomp_user_vec(self, request_id: int) -> np.ndarray | None:
        """Return pre-exported 128-dim user tower vector for this request_id (search_idx)."""
        if self._precomp_user_vecs is None or not self._precomp_user_map:
            return None
        row = self._precomp_user_map.get(str(int(request_id)))
        if row is None:
            return None
        vec = self._precomp_user_vecs[row]
        if not np.isfinite(vec).all():
            return None
        return vec.reshape(1, -1).astype(np.float32)

    def _load_query_encoder(self) -> None:
        if AutoTokenizer is None or AutoModel is None:
            return
        if DSSMRecaller._shared_query_tokenizer is None or DSSMRecaller._shared_query_model is None:
            try:
                DSSMRecaller._shared_query_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-zh", local_files_only=True)
                DSSMRecaller._shared_query_model = AutoModel.from_pretrained("BAAI/bge-base-zh", local_files_only=True).to("cpu")
                DSSMRecaller._shared_query_model.eval()
            except Exception:
                DSSMRecaller._shared_query_tokenizer = None
                DSSMRecaller._shared_query_model = None
        self.query_tokenizer = DSSMRecaller._shared_query_tokenizer
        self.query_model = DSSMRecaller._shared_query_model

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

        if hist_ids:
            hist_vecs = self.user_store["note_text"].get_batch_vecs(hist_ids)
            hist = hist_vecs.mean(dim=0, keepdim=True)
        else:
            hist = torch.zeros(1, self.train_mod.EMB_DIM, dtype=torch.float32)

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
