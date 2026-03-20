"""在线 DSSM 召回模型加载与 ANN 召回执行。"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Callable

import duckdb
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
            self.user_store["search_query"] = train_mod.MMapStore("query_text_emb", "search_query")
            self._load_query_encoder()
        self.enabled = True

    def _load_query_encoder(self) -> None:
        if AutoTokenizer is None or AutoModel is None:
            return
        if DSSMRecaller._shared_query_tokenizer is None or DSSMRecaller._shared_query_model is None:
            try:
                DSSMRecaller._shared_query_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-zh")
                DSSMRecaller._shared_query_model = AutoModel.from_pretrained("BAAI/bge-base-zh").to("cpu")
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

    def _build_user_vec(self, feat_req: pd.DataFrame, request_id: int, user_idx: int) -> np.ndarray | None:
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
        if self.scene == "search" and "search_query" in self.user_store:
            qv = None
            query_text = str(row.get("query", "") or "").strip()
            qvec = self._encode_query_vec(query_text)
            if qvec is not None:
                qv = torch.from_numpy(qvec)
            else:
                ref_req = request_id
                if "query_ref_request_id" in feat_req.columns and len(feat_req) > 0:
                    try:
                        ref_req = int(feat_req.iloc[0].get("query_ref_request_id", request_id))
                    except Exception:
                        ref_req = int(request_id)
                qv = self.user_store["search_query"].get_batch_vecs([int(ref_req)])
            user_batch["query_vec"] = qv

        with torch.no_grad():
            user_vec = self.model.forward_user(user_batch, hist)
        return user_vec.detach().float().cpu().numpy().astype(np.float32)

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
        if len(req_cur) > 0:
            user_idx = int(req_cur.iloc[0]["user_idx"])
        elif len(feat_req) > 0 and "user_idx" in feat_req.columns:
            try:
                user_idx = int(feat_req.iloc[0].get("user_idx", -1))
            except Exception:
                user_idx = -1
        else:
            user_idx = -1
        user_vec = self._build_user_vec(feat_req=feat_req, request_id=int(request_id), user_idx=user_idx)
        if user_vec is None:
            return pd.DataFrame()

        dists, idxs = self.ann_index.search(user_vec, int(max_rank))
        rows = []
        for rk, (row_idx, score) in enumerate(zip(idxs[0].tolist(), dists[0].tolist()), start=1):
            if row_idx < 0 or row_idx >= len(self.row2note):
                continue
            note_idx = int(self.row2note[row_idx])
            rows.append(
                {
                    group_key: int(request_id),
                    "user_idx": int(user_idx),
                    "note_idx": note_idx,
                    "rank": int(rk),
                    "recall_score": float(score),
                    "score_ann": float(score),
                    "score_swing": 0.0,
                    "score_usercf": 0.0,
                    "from_ann": 1,
                    "from_swing": 0,
                    "from_usercf": 0,
                    "first_route": "ann",
                }
            )
        return pd.DataFrame(rows)


def fetch_dssm_candidates(
    request_id: int,
    max_rank: int,
    group_key: str,
    req_df: pd.DataFrame,
    realtime_ann_enabled: bool,
    dssm_ann_index,
    dssm_row2note: np.ndarray,
    build_realtime_user_vec,
    recall_test_path: Path,
) -> pd.DataFrame:
    if realtime_ann_enabled and dssm_ann_index is not None and len(dssm_row2note) > 0:
        req_cur = req_df[req_df[group_key] == int(request_id)]
        user_idx = int(req_cur.iloc[0]["user_idx"]) if len(req_cur) > 0 else -1
        user_vec = build_realtime_user_vec(request_id=int(request_id), user_idx=user_idx)
        if user_vec is not None:
            dists, idxs = dssm_ann_index.search(user_vec, int(max_rank))
            rows = []
            for rk, (row_idx, score) in enumerate(zip(idxs[0].tolist(), dists[0].tolist()), start=1):
                if row_idx < 0 or row_idx >= len(dssm_row2note):
                    continue
                note_idx = int(dssm_row2note[row_idx])
                rows.append(
                    {
                        group_key: int(request_id),
                        "user_idx": int(user_idx),
                        "note_idx": note_idx,
                        "rank": int(rk),
                        "recall_score": float(score),
                        "score_ann": float(score),
                        "score_swing": 0.0,
                        "score_usercf": 0.0,
                        "from_ann": 1,
                        "from_swing": 0,
                        "from_usercf": 0,
                        "first_route": "ann",
                    }
                )
            return pd.DataFrame(rows)

    if not recall_test_path.exists():
        return pd.DataFrame()

    con = duckdb.connect(database=":memory:")
    try:
        sql = f"""
        SELECT {group_key}, user_idx, note_idx, rank, recall_score,
               score_ann, score_swing, score_usercf,
               from_ann, from_swing, from_usercf, first_route
        FROM read_parquet(?)
        WHERE {group_key} = ? AND rank <= ?
        ORDER BY rank ASC
        """
        return con.execute(sql, [str(recall_test_path), int(request_id), int(max_rank)]).df()
    finally:
        con.close()


__all__ = ["fetch_dssm_candidates"]
