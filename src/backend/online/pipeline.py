"""
Qilin Backend Online Pipeline
- 在线职责：维护单场景运行时状态（请求数据、特征缓存、模型对象）
- 推理链路：冷启动识别 -> 多路召回 -> 粗排 -> 精排 -> 结果组装
- 服务职责：作为 FastAPI 启动入口，完成场景状态初始化与运行时注册
- 运行形态：支持 search / rec 双场景，并按 tag 加载 outputs/deploy/{scene}/{tag}
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
import threading
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import torch
import uvicorn

try:
    import faiss
except Exception:
    faiss = None

BASE_DIR = Path(__file__).resolve().parents[3]

from backend.online.config import load_online_config
from backend.online.cold_start.detector import is_cold_start
from backend.online.realtime_cache import build_realtime_cache
from backend.online.preranking.gbdt import GBDTPredictor
from backend.online.preranking.service import run_preranking
from backend.online.ranking.service import run_ranking
from backend.online.ranking.dien import DIENPredictor
from backend.online.recall.dssm import DSSMRecaller
from backend.online.recall.service import fetch_recall_candidates, run_recall

DATASETS_DIR = BASE_DIR / "datasets"
FEATURE_DIR = BASE_DIR / "features"
OUT_DIR = BASE_DIR / "outputs"
OUT_DATA_DIR = OUT_DIR / "data"
DEPLOY_DIR = OUT_DIR / "deploy"
COLD_START_REQ_THRESHOLD = 3
HOT_ROUTE_TOPK = 300


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


def _to_path_list(v: Any) -> list[str]:
    if isinstance(v, np.ndarray):
        return [str(x) for x in v.tolist()]
    if isinstance(v, list):
        return [str(x) for x in v]
    if isinstance(v, str):
        return [v]
    return []


def _normalize_image_rel_path(p: str) -> str:
    s = str(p or "").strip().lstrip("/")
    if s.startswith("image/"):
        s = s[len("image/") :]
    return s


def _image_exists(rel_path: str) -> bool:
    if not rel_path:
        return False
    f = (BASE_DIR / "image" / rel_path).resolve()
    if BASE_DIR.resolve() not in f.parents:
        return False
    return f.exists() and f.is_file()


def _existing_images(paths: list[str]) -> list[str]:
    out: list[str] = []
    for p in paths:
        rel = _normalize_image_rel_path(p)
        if _image_exists(rel):
            out.append(rel)
    return out


def _char_ngrams(s: str, n: int = 2) -> set[str]:
    s = (s or "").strip().lower()
    if len(s) <= n:
        return {s} if s else set()
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def _text_match_score(query: str, title: str, content: str) -> float:
    q = (query or "").strip().lower()
    if not q:
        return 0.0
    t = (title or "").lower()
    c = (content or "").lower()
    score = 0.0
    if q in t:
        score += 3.0
    if q in c:
        score += 1.0
    qg = _char_ngrams(q, n=2)
    tg = _char_ngrams(t, n=2)
    cg = _char_ngrams(c[:500], n=2)
    if qg and tg:
        score += 2.0 * (len(qg & tg) / max(len(qg), 1))
    if qg and cg:
        score += 0.5 * (len(qg & cg) / max(len(qg), 1))
    return float(score)


def _group_key(scene: str) -> str:
    return "search_idx" if scene == "search" else "request_idx"


def _scene_test_dataset_path(scene: str) -> Path:
    if scene == "search":
        return DATASETS_DIR / "search_test" / "train-00000-of-00001.parquet"
    return DATASETS_DIR / "recommendation_test" / "train-00000-of-00001.parquet"


def _scene_train_dataset_path(scene: str) -> Path:
    if scene == "search":
        return DATASETS_DIR / "search_train" / "train-00000-of-00001.parquet"
    return DATASETS_DIR / "recommendation_train" / "train-00000-of-00001.parquet"


class SceneServingState:
    """统一的单场景 serving 状态（search / rec 共享同一套逻辑）。"""

    def __init__(self, scene: str, tag: str = "easy", gbdt_topn: int = 500, recall_rank_cap: int = 800):
        if scene not in {"search", "rec"}:
            raise ValueError(f"invalid scene: {scene}")
        self.scene = scene
        self.group_key = _group_key(scene)
        self.tag = tag
        self.deploy_tag_dir = DEPLOY_DIR / scene / tag
        self.gbdt_topn = int(gbdt_topn)
        self.recall_rank_cap = int(recall_rank_cap)

        self.scene_test_path = _scene_test_dataset_path(scene)
        self.scene_train_request_path = _scene_train_dataset_path(scene)
        self.user_feat_path = DATASETS_DIR / "user_feat" / "train-00000-of-00001.parquet"
        self.scene_feat_path = FEATURE_DIR / f"{scene}_test_features.parquet"
        self.notes_glob = str(DATASETS_DIR / "notes" / "train-*.parquet")
        self.recall_test_path = OUT_DIR / "data" / f"recall_{scene}_test_{tag}_multiroute_topk.parquet"
        self.realtime_cache = build_realtime_cache()
        self._runtime_request_lock = threading.Lock()
        self._runtime_request_counter = 0

        _load_module_from_src("utils", "training/utils.py")
        dien_train = _load_module_from_src("online_dien_ranker", "training/dien_ranker.py")
        utils_mod = _load_module_from_src("online_training_utils", "training/utils.py")

        self.dien_train = dien_train
        self.discretize_relevance = getattr(utils_mod, "discretize_relevance")
        self.eval_ndcg_by_group = getattr(utils_mod, "eval_ndcg_by_group")

        self._predict_lock = threading.Lock()
        self._note_cache_lock = threading.Lock()
        self._note_meta_cache: dict[int, dict[str, Any]] = {}
        self._load_data()
        self._load_models()

    def _artifact_path(self, rel_path: str) -> Path:
        return self.deploy_tag_dir / rel_path

    def _load_data(self) -> None:
        # 请求、特征、用户画像全部常驻内存，降低在线路径 I/O 开销
        req_cols = [self.group_key, "user_idx", "session_idx"]
        if self.scene == "search":
            req_cols += ["query"]
        req_cols = list(dict.fromkeys(req_cols))

        if self.scene_test_path.exists():
            self.req_df = pd.read_parquet(self.scene_test_path, columns=req_cols)
            self.req_df = self.req_df.drop_duplicates(subset=[self.group_key]).reset_index(drop=True)
        else:
            self.req_df = pd.DataFrame(columns=req_cols)

        self.feat_df = pd.read_parquet(self.scene_feat_path)
        self.feat_df[self.group_key] = self.feat_df[self.group_key].astype(np.int64)
        self.feat_df["note_idx"] = self.feat_df["note_idx"].astype(np.int64)
        self.feat_group_indices = {
            int(k): np.asarray(v, dtype=np.int32)
            for k, v in self.feat_df.groupby(self.group_key, sort=False).indices.items()
        }

        self.user_feat_df = pd.read_parquet(self.user_feat_path)
        self.user_feat_df["user_idx"] = self.user_feat_df["user_idx"].astype(np.int64)
        self.user_feat_df = self.user_feat_df.set_index("user_idx", drop=False)

        req_sorted = (
            self.req_df.sort_values(["user_idx", "session_idx", self.group_key])
            if len(self.req_df) > 0
            else self.req_df
        )
        self.user_requests = (
            req_sorted.groupby("user_idx")[self.group_key].apply(lambda s: [int(x) for x in s.tolist()]).to_dict()
            if len(req_sorted) > 0
            else {}
        )

        if self.scene == "search" and "query" in self.req_df.columns:
            self.search_query_map = {
                int(r[self.group_key]): str(r.get("query") or "")
                for r in self.req_df[[self.group_key, "query"]].to_dict("records")
            }
        else:
            self.search_query_map = {}

        req_max = int(pd.to_numeric(self.req_df.get(self.group_key, pd.Series([0])), errors="coerce").fillna(0).max()) if len(self.req_df) > 0 else 0
        train_max = 0
        if self.scene_train_request_path.exists():
            try:
                train_max = int(pd.read_parquet(self.scene_train_request_path, columns=[self.group_key])[self.group_key].max())
            except Exception:
                train_max = 0
        self._runtime_request_counter = max(req_max, train_max)

    def get_feat_req(self, request_id: int) -> pd.DataFrame:
        idx = self.feat_group_indices.get(int(request_id))
        if idx is None or len(idx) == 0:
            return self.feat_df.iloc[0:0].copy()
        return self.feat_df.iloc[idx].reset_index(drop=True)

    def _load_models(self) -> None:
        # 各阶段模型在状态初始化时加载，service 仅负责编排调用
        self.gbdt_predictor = GBDTPredictor(
            scene=self.scene,
            tag=self.tag,
            deploy_tag_dir=self.deploy_tag_dir,
            group_key=self.group_key,
            feat_columns=list(self.feat_df.columns),
        )
        self.dien_predictor = DIENPredictor(
            scene=self.scene,
            tag=self.tag,
            deploy_tag_dir=self.deploy_tag_dir,
            group_key=self.group_key,
        )
        self.dssm_recaller = DSSMRecaller(
            scene=self.scene,
            tag=self.tag,
            deploy_tag_dir=self.deploy_tag_dir,
            group_key=self.group_key,
        )

        self.lgb_model = self.gbdt_predictor.lgb_model
        self.xgb_model = self.gbdt_predictor.xgb_model
        self.gbdt_feature_cols = self.gbdt_predictor.feature_cols
        self.dien_model = self.dien_predictor.model
        self.dssm_model = self.dssm_recaller.model
        self.realtime_ann_enabled = bool(self.dssm_recaller.enabled)


    def readiness(self) -> dict[str, Any]:
        return {
            "scene": self.scene,
            "scene_test": self.scene_test_path.exists(),
            "scene_test_features": self.scene_feat_path.exists(),
            "user_feat": self.user_feat_path.exists(),
            "redis_cache": self.realtime_cache is not None,
            "lgb_loaded": self.lgb_model is not None,
            "xgb_loaded": self.xgb_model is not None,
            "dien_loaded": self.dien_model is not None,
            "dssm_user_tower_loaded": self.dssm_model is not None,
            "realtime_ann_enabled": bool(self.realtime_ann_enabled),
            "rows": {
                "requests": int(len(self.req_df)),
                "feature_rows": int(len(self.feat_df)),
                "users": int(self.user_feat_df["user_idx"].nunique()),
            },
        }

    def resolve_request(self, user_idx: int, query: str | None) -> tuple[int, str]:
        reqs = []
        if self.realtime_cache is not None:
            reqs = self.realtime_cache.get_user_requests(int(user_idx), self.scene) or []
        if not reqs:
            reqs = self.user_requests.get(int(user_idx), [])

        if self.scene == "rec":
            if not reqs:
                if len(self.req_df) == 0:
                    raise KeyError("no requests available for this scene")
                rid = int(self.req_df.iloc[0][self.group_key])
                return rid, ""
            rid = int(reqs[-1])
            return rid, ""

        if not reqs and len(self.search_query_map) == 0:
            if len(self.req_df) == 0:
                raise KeyError("no requests available for this scene")
            rid = int(self.req_df.iloc[0][self.group_key])
            return rid, ""

        all_search_rids = list(self.search_query_map.keys())
        def _stable_pick(text: str, size: int) -> int:
            if size <= 0:
                return 0
            h = hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()
            return int(h, 16) % int(size)

        if not query or not query.strip():
            if reqs:
                rid = int(reqs[-1])
                return rid, self.search_query_map.get(rid, "")
            if all_search_rids:
                rid = int(all_search_rids[_stable_pick(str(user_idx), len(all_search_rids))])
                return rid, self.search_query_map.get(rid, "")
            rid = int(self.req_df.iloc[0][self.group_key])
            return rid, ""

        q = str(query or "").strip()
        if reqs:
            rid = int(reqs[-1])
            return rid, q
        if all_search_rids:
            rid = int(all_search_rids[_stable_pick(str(user_idx), len(all_search_rids))])
            return rid, q
        rid = int(self.req_df.iloc[0][self.group_key])
        return rid, q

    def _runtime_request_id(self, user_idx: int, query: str | None = None) -> int:
        base = int(self._runtime_request_counter)
        if self.realtime_cache is not None:
            rid = self.realtime_cache.next_runtime_request_id(self.scene, start_from=base)
            with self._runtime_request_lock:
                self._runtime_request_counter = max(self._runtime_request_counter, int(rid))
            return int(rid)
        with self._runtime_request_lock:
            self._runtime_request_counter += 1
            return int(self._runtime_request_counter)

    def get_recent_exposed_notes(self, user_idx: int, max_len: int = 200) -> list[int]:
        if self.realtime_cache is None:
            return []
        return self.realtime_cache.get_recent_exposed_notes(int(user_idx), self.scene, max_len=max_len)

    def record_exposed_notes(self, user_idx: int, note_ids: list[int]) -> None:
        if self.realtime_cache is None:
            return
        self.realtime_cache.record_exposed_notes(int(user_idx), self.scene, note_ids)

    def build_runtime_feat_req(self, user_idx: int, request_id: int, query: str | None = None) -> pd.DataFrame:
        def _as_int(v: Any, default: int = 0) -> int:
            if v is None:
                return int(default)
            try:
                if isinstance(v, float) and np.isnan(v):
                    return int(default)
            except Exception:
                pass
            try:
                return int(v)
            except Exception:
                return int(default)

        def _as_float(v: Any, default: float = 0.0) -> float:
            if v is None:
                return float(default)
            try:
                if isinstance(v, float) and np.isnan(v):
                    return float(default)
            except Exception:
                pass
            try:
                return float(v)
            except Exception:
                return float(default)

        uid = int(user_idx)
        profile = None
        if self.realtime_cache is not None:
            profile = self.realtime_cache.get_user_profile(uid)
        if profile is None and uid in self.user_feat_df.index:
            profile = self.user_feat_df.loc[uid].to_dict()
        if profile is None and len(self.user_feat_df) > 0:
            profile = self.user_feat_df.iloc[0].to_dict()
        profile = dict(profile or {})

        history = []
        if self.realtime_cache is not None:
            history = self.realtime_cache.get_user_history_notes(uid, self.scene, max_len=20)
        if not history:
            raw = profile.get("recent_clicked_note_idxs", [])
            if isinstance(raw, np.ndarray):
                history = [int(x) for x in raw.tolist()[:20]]
            elif isinstance(raw, list):
                history = [int(x) for x in raw[:20]]

        row: dict[str, Any] = {
            self.group_key: int(request_id),
            "user_idx": uid,
            "note_idx": -1,
            "click": 0,
            "y_multi": 0.0,
            "recent_clicked_note_idxs": history,
            "gender_enc": _as_int(profile.get("gender_enc", profile.get("gender", 0)), 0),
            "platform_enc": _as_int(profile.get("platform_enc", profile.get("platform", 0)), 0),
            "age_enc": _as_int(profile.get("age_enc", profile.get("age", 0)), 0),
            "location_enc": _as_int(profile.get("location_enc", profile.get("location", 0)), 0),
            "fans_num": _as_float(profile.get("fans_num", 0.0), 0.0),
            "follows_num": _as_float(profile.get("follows_num", 0.0), 0.0),
        }
        for i in range(1, 41):
            row[f"dense_feat{i}"] = _as_float(profile.get(f"dense_feat{i}", 0.0), 0.0)
        if self.scene == "search":
            row["query"] = str(query or "")
        return pd.DataFrame([row])

    def _fetch_recall_candidates(
        self,
        request_id: int,
        max_rank: int,
        runtime_feat_req: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        def _get_feat_req(req_id: int) -> pd.DataFrame:
            if runtime_feat_req is not None and int(req_id) == int(request_id):
                return runtime_feat_req
            return self.get_feat_req(int(req_id))

        return fetch_recall_candidates(
            request_id=int(request_id),
            max_rank=int(max_rank),
            group_key=self.group_key,
            req_df=self.req_df,
            get_feat_req=_get_feat_req,
            dssm_recaller=self.dssm_recaller,
            recall_test_path=self.recall_test_path,
        )

    def _fetch_query_recall_candidates(self, request_id: int, query: str, topk: int = 240) -> pd.DataFrame:
        if self.scene != "search":
            return pd.DataFrame()
        q = str(query or "").strip()
        if not q:
            return pd.DataFrame()

        con = duckdb.connect(database=":memory:")
        try:
            q_sql = """
                SELECT
                    note_idx,
                    row_number() OVER (ORDER BY note_idx) AS rank
                FROM read_parquet(?)
                WHERE lower(COALESCE(note_title, '')) LIKE '%' || lower(?) || '%'
                   OR lower(COALESCE(note_content, '')) LIKE '%' || lower(?) || '%'
                LIMIT ?
            """
            rows = con.execute(q_sql, [self.notes_glob, q, q, int(max(10, topk))]).df()

            if len(rows) == 0:
                tokens: list[str] = []
                qq = q.strip()
                if " " in qq:
                    tokens = [x.strip() for x in qq.split(" ") if x.strip()]
                if not tokens:
                    tokens = [qq[i : i + 1] for i in range(len(qq)) if qq[i : i + 1].strip()]
                uniq_tokens: list[str] = []
                seen_tok: set[str] = set()
                for t in tokens:
                    if t in seen_tok:
                        continue
                    seen_tok.add(t)
                    uniq_tokens.append(t)
                uniq_tokens = uniq_tokens[:6]

                if uniq_tokens:
                    score_parts = []
                    params: list[Any] = [self.notes_glob]
                    for tok in uniq_tokens:
                        score_parts.append("(CASE WHEN lower(COALESCE(note_title, '')) LIKE '%' || lower(?) || '%' THEN 1.5 ELSE 0 END)")
                        params.append(tok)
                        score_parts.append("(CASE WHEN lower(COALESCE(note_content, '')) LIKE '%' || lower(?) || '%' THEN 0.8 ELSE 0 END)")
                        params.append(tok)
                    score_expr = " + ".join(score_parts) if score_parts else "0"
                    fb_sql = f"""
                        SELECT
                            note_idx,
                            ({score_expr}) AS token_score
                        FROM read_parquet(?)
                        WHERE ({score_expr}) > 0
                        ORDER BY token_score DESC, note_idx
                        LIMIT ?
                    """
                    # score_expr 在 SQL 中出现了两次，因此 token 参数也需要重复两份
                    rows = con.execute(
                        fb_sql,
                        params + params[1:] + [int(max(10, topk))],
                    ).df()
                    if len(rows) > 0 and "rank" not in rows.columns:
                        rows = rows.reset_index(drop=True)
                        rows["rank"] = rows.index + 1
        finally:
            con.close()

        if len(rows) == 0:
            return pd.DataFrame()

        out = pd.DataFrame(
            {
                self.group_key: int(request_id),
                "note_idx": pd.to_numeric(rows["note_idx"], errors="coerce").fillna(-1).astype(int),
                "rank": pd.to_numeric(rows["rank"], errors="coerce").fillna(1).astype(int),
                "recall_score": 1.0,
                "score_ann": 0.0,
                "score_swing": 0.0,
                "score_usercf": 0.0,
                "from_ann": 0,
                "from_swing": 0,
                "from_usercf": 0,
                "from_hot": 0,
                "first_route": "query",
            }
        )
        out = out[out["note_idx"] >= 0].copy()
        if out.empty:
            return pd.DataFrame()
        return out.drop_duplicates(subset=[self.group_key, "note_idx"], keep="first").reset_index(drop=True)

    def list_users(self, limit: int = 30, offset: int = 0, random_show: bool = False) -> list[dict[str, Any]]:
        cols = [
            "user_idx",
            "gender", "age", "platform", "location",
            "gender_enc", "age_enc", "platform_enc", "location_enc",
            "fans_num", "follows_num",
        ]
        use_cols = [c for c in cols if c in self.user_feat_df.columns]
        if not use_cols:
            return []
        total = len(self.user_feat_df)
        if total <= 0:
            return []

        lim = int(max(1, limit))
        off = int(max(0, offset))
        if random_show:
            frame = self.user_feat_df[use_cols].sample(n=min(lim, total), replace=False)
        else:
            frame = self.user_feat_df[use_cols].iloc[off : off + lim]

        out: list[dict[str, Any]] = []
        for _, row in frame.iterrows():
            obj = {k: row[k] for k in use_cols}
            obj["user_idx"] = int(obj.get("user_idx", 0) or 0)
            out.append(obj)
        return out

    def _fetch_notes(self, note_ids: list[int]) -> pd.DataFrame:
        note_cols = [
            "note_idx", "note_title", "note_content", "image_path",
            "accum_like_num", "accum_collect_num", "accum_comment_num"
        ]
        if not note_ids:
            return pd.DataFrame(columns=note_cols)
        uniq_ids = sorted(set(int(x) for x in note_ids))
        with self._note_cache_lock:
            missing_ids = [x for x in uniq_ids if x not in self._note_meta_cache]

        if missing_ids:
            ids_df = pd.DataFrame({"note_idx": missing_ids})
            con = duckdb.connect(database=":memory:")
            try:
                con.register("ids", ids_df)
                q = """
                SELECT n.note_idx, n.note_title, n.note_content, n.image_path,
                       n.accum_like_num, n.accum_collect_num, n.accum_comment_num
                FROM read_parquet(?) n
                INNER JOIN ids USING(note_idx)
                """
                df_new = con.execute(q, [self.notes_glob]).df()
            finally:
                con.close()

            if "image_path" in df_new.columns:
                df_new["image_path"] = df_new["image_path"].apply(_to_path_list)
            new_rows = {
                int(r["note_idx"]): r
                for r in df_new.to_dict("records")
            }
            with self._note_cache_lock:
                self._note_meta_cache.update(new_rows)
                if len(self._note_meta_cache) > 200_000:
                    self._note_meta_cache.clear()
                    self._note_meta_cache.update(new_rows)

        with self._note_cache_lock:
            rows = [self._note_meta_cache[x] for x in uniq_ids if x in self._note_meta_cache]
        if not rows:
            return pd.DataFrame(columns=note_cols)
        out = pd.DataFrame(rows)
        for c in note_cols:
            if c not in out.columns:
                out[c] = np.nan
        return out[note_cols]

    def predict_gbdt(self, cand: pd.DataFrame) -> np.ndarray:
        return self.gbdt_predictor.predict(cand)

    def predict_dien(
        self,
        cand: pd.DataFrame,
        batch_size: int = 512,
        history_note_ids: list[int] | None = None,
    ) -> np.ndarray:
        return self.dien_predictor.predict(
            cand,
            batch_size=batch_size,
            history_note_ids=history_note_ids,
        )

    def build_feed(self, user_idx: int, query: str, page: int, page_size: int, include_seen: bool = False) -> dict[str, Any]:
        return OnlineScenePipeline(self).build_feed(
            user_idx=int(user_idx),
            query=query,
            page=int(page),
            page_size=int(page_size),
            include_seen=bool(include_seen),
        )

    def get_note_detail(self, user_idx: int, request_id: int, note_idx: int, query: str = "") -> dict[str, Any]:
        feed = self.build_feed(user_idx=user_idx, query=str(query or ""), page=1, page_size=500, include_seen=True)
        target = next((x for x in feed["items"] if int(x["note_idx"]) == int(note_idx) and int(x["request_id"]) == int(request_id)), None)
        if target is None:
            target = next((x for x in feed["items"] if int(x["note_idx"]) == int(note_idx)), None)

        note_meta = self._fetch_notes([note_idx])
        if len(note_meta) == 0:
            raise KeyError(f"note_idx not found: {note_idx}")
        n = note_meta.iloc[0].to_dict()

        stage_top500_ranks = {"recall": None, "coarse": None, "final": None, "rerank": None}
        if target is not None and isinstance(target, dict):
            stage_top500_ranks = dict(target.get("stage_ranks") or stage_top500_ranks)

        title_raw = n.get("note_title")
        if title_raw is None or (isinstance(title_raw, float) and np.isnan(title_raw)):
            title_val = "(无标题)"
        else:
            title_val = str(title_raw).strip()
            if not title_val or title_val.lower() == "nan":
                title_val = "(无标题)"

        return {
            "scene": self.scene,
            "user_idx": int(user_idx),
            "request_id": int(request_id),
            "note_idx": int(note_idx),
            "title": title_val,
            "content": str(n.get("note_content") or ""),
            "images": _existing_images(_to_path_list(n.get("image_path"))),
            "accum_like_num": int(float(n.get("accum_like_num", 0) or 0)),
            "accum_collect_num": int(float(n.get("accum_collect_num", 0) or 0)),
            "accum_comment_num": int(float(n.get("accum_comment_num", 0) or 0)),
            "stage_rank_note": "召回/粗排/精排为三阶段位次；重排 Rank 为去重后最终列表中的位次。",
            "stage_top500_ranks": stage_top500_ranks,
        }

    def get_user(self, user_idx: int) -> dict[str, Any]:
        uid = int(user_idx)
        info = None
        if self.realtime_cache is not None:
            info = self.realtime_cache.get_user_profile(uid)
        if info is None:
            if uid in self.user_feat_df.index:
                info = self.user_feat_df.loc[uid].to_dict()
        if info is None:
            raise KeyError(f"user_idx not found: {uid}")
        reqs = []
        if self.realtime_cache is not None:
            reqs = self.realtime_cache.get_user_requests(uid, self.scene) or []
        if not reqs:
            reqs = self.user_requests.get(uid, [])
        recent = []
        for rid in reqs[-30:][::-1]:
            row = {"request_id": int(rid)}
            if self.scene == "search":
                row["query"] = self.search_query_map.get(int(rid), "")
            recent.append(row)
        behaviors: list[dict[str, Any]] = []

        baseline_history: list[int] = []
        if reqs:
            try:
                feat_req = self.get_feat_req(int(reqs[-1]))
                if len(feat_req) > 0 and "recent_clicked_note_idxs" in feat_req.columns:
                    raw = feat_req.iloc[0].get("recent_clicked_note_idxs", [])
                    if isinstance(raw, np.ndarray):
                        baseline_history = [int(x) for x in raw.tolist()[:20]]
                    elif isinstance(raw, list):
                        baseline_history = [int(x) for x in raw[:20]]
            except Exception:
                baseline_history = []

        baseline_events: list[dict[str, Any]] = []
        for nid in baseline_history:
            baseline_events.append(
                {
                    "ts": 0,
                    "scene": self.scene,
                    "action": "offline_history",
                    "note_idx": int(nid),
                    "request_id": int(reqs[-1]) if reqs else None,
                    "query": "",
                    "interaction_score": 0.0,
                }
            )

        realtime_events: list[dict[str, Any]] = []
        if self.realtime_cache is not None:
            realtime_events = self.realtime_cache.get_recent_behaviors(uid, self.scene, max_len=20)

        merged = realtime_events + baseline_events
        deduped: list[dict[str, Any]] = []
        seen: set[str] = set()
        for b in merged:
            key = f"{b.get('note_idx')}|{b.get('request_id')}|{b.get('query','')}|{b.get('scene', self.scene)}"
            if key in seen:
                continue
            seen.add(key)
            deduped.append(b)
            if len(deduped) >= 20:
                break
        behaviors = deduped

        note_ids = []
        for b in behaviors:
            try:
                nid = int(b.get("note_idx", -1))
            except Exception:
                nid = -1
            if nid >= 0:
                note_ids.append(nid)
        title_map: dict[int, str] = {}
        if note_ids:
            note_df = self._fetch_notes(sorted(set(note_ids)))
            if len(note_df) > 0:
                title_map = {
                    int(r["note_idx"]): str(r.get("note_title") or "(无标题)")
                    for r in note_df[["note_idx", "note_title"]].to_dict("records")
                }
        for b in behaviors:
            try:
                nid = int(b.get("note_idx", -1))
            except Exception:
                nid = -1
            b["title"] = title_map.get(nid, "(无标题)") if nid >= 0 else "(无标题)"
            if "scene" not in b:
                b["scene"] = self.scene
        return {
            "scene": self.scene,
            "user_idx": uid,
            "features": info,
            "request_count_in_test": int(len(reqs)),
            "recent_behaviors": behaviors,
        }

    def get_user_history_notes(self, user_idx: int, feat_req: pd.DataFrame, max_len: int = 20) -> list[int]:
        if self.realtime_cache is not None:
            history = self.realtime_cache.get_user_history_notes(int(user_idx), self.scene, max_len=max_len)
            if history:
                return history
        if "recent_clicked_note_idxs" in feat_req.columns and len(feat_req) > 0:
            raw = feat_req.iloc[0].get("recent_clicked_note_idxs", [])
            if isinstance(raw, np.ndarray):
                return [int(x) for x in raw.tolist()][:max_len]
            if isinstance(raw, list):
                return [int(x) for x in raw][:max_len]
        return []

    def record_click(
        self,
        user_idx: int,
        note_idx: int,
        request_id: int | None = None,
        query: str = "",
    ) -> bool:
        if self.realtime_cache is not None:
            return bool(self.realtime_cache.record_click(
                int(user_idx),
                self.scene,
                int(note_idx),
                request_id=(int(request_id) if request_id is not None else None),
                query=str(query or ""),
            ))
        return False

    def record_view(
        self,
        user_idx: int,
        note_idx: int,
        request_id: int | None = None,
        query: str = "",
    ) -> bool:
        if self.realtime_cache is not None:
            return bool(self.realtime_cache.record_view(
                int(user_idx),
                self.scene,
                int(note_idx),
                request_id=(int(request_id) if request_id is not None else None),
                query=str(query or ""),
            ))
        return False

    def record_engage(
        self,
        user_idx: int,
        note_idx: int,
        request_id: int | None = None,
        query: str = "",
        like: int = 0,
        collect: int = 0,
        comment: int = 0,
        share: int = 0,
        page_time: float = 0.0,
    ) -> bool:
        if self.realtime_cache is not None:
            return bool(self.realtime_cache.record_engage(
                int(user_idx),
                self.scene,
                int(note_idx),
                request_id=(int(request_id) if request_id is not None else None),
                query=str(query or ""),
                like=max(0, int(like)),
                collect=max(0, int(collect)),
                comment=max(0, int(comment)),
                share=max(0, int(share)),
                page_time=max(0.0, float(page_time)),
            ))
        return False

    def compute_test_metrics(self, dien_max_groups: int = 1200, max_groups: int = 3000) -> dict[str, Any]:
        recall_json = OUT_DATA_DIR / f"recall_eval_{self.scene}_test_{self.tag}.json"
        recall = {}
        if recall_json.exists():
            recall = json.loads(recall_json.read_text())

        eval_df = self.feat_df.copy()
        all_groups = eval_df[self.group_key].drop_duplicates().to_numpy()
        sampled_eval_groups = int(len(all_groups))
        if int(max_groups) > 0 and len(all_groups) > int(max_groups):
            rng = np.random.default_rng(42)
            gids = rng.choice(all_groups, size=int(max_groups), replace=False)
            eval_df = eval_df[eval_df[self.group_key].isin(gids)].copy()
            sampled_eval_groups = int(len(np.unique(gids)))
        eval_df["gbdt_score"] = self.predict_gbdt(eval_df)
        y_disc = self.discretize_relevance(eval_df["y_multi"].to_numpy())
        groups = eval_df[self.group_key].to_numpy()
        gbdt_ndcg10 = self.eval_ndcg_by_group(y_disc, eval_df["gbdt_score"].to_numpy(), groups, 10)
        gbdt_ndcg100 = self.eval_ndcg_by_group(y_disc, eval_df["gbdt_score"].to_numpy(), groups, 100)

        dien_ndcg10 = None
        gbdt_ndcg10_on_dien_sample = None
        dien_sample_groups = 0
        if self.dien_model is not None:
            pos_groups = eval_df.loc[eval_df["y_multi"] > 0, self.group_key].drop_duplicates().to_numpy()
            if len(pos_groups) > dien_max_groups:
                rng = np.random.default_rng(42)
                pos_groups = rng.choice(pos_groups, size=dien_max_groups, replace=False)
            sub = eval_df[eval_df[self.group_key].isin(pos_groups)].copy()
            sub = (
                sub.sort_values([self.group_key, "gbdt_score"], ascending=[True, False], kind="mergesort")
                .groupby(self.group_key, sort=False)
                .head(120)
                .reset_index(drop=True)
            )
            sub["dien_score"] = self.predict_dien(sub)
            sy = self.discretize_relevance(sub["y_multi"].to_numpy())
            sg = sub[self.group_key].to_numpy()
            dien_ndcg10 = self.eval_ndcg_by_group(sy, sub["dien_score"].to_numpy(), sg, 10)
            gbdt_ndcg10_on_dien_sample = self.eval_ndcg_by_group(sy, sub["gbdt_score"].to_numpy(), sg, 10)
            dien_sample_groups = int(len(np.unique(sg)))

        return {
            "scene": self.scene,
            "split": "test",
            "tag": self.tag,
            "readiness": self.readiness(),
            "recall": recall,
            "gbdt": {
                "ndcg@10_exposed_test": float(gbdt_ndcg10),
                "ndcg@100_exposed_test": float(gbdt_ndcg100),
                "eval_rows": int(len(eval_df)),
                "eval_groups": int(eval_df[self.group_key].nunique()),
                "sampled_eval_groups": sampled_eval_groups,
            },
            "dien": {
                "ndcg@10_exposed_test_sampled": float(dien_ndcg10) if dien_ndcg10 is not None else None,
                "gbdt_ndcg@10_on_same_sample": float(gbdt_ndcg10_on_dien_sample) if gbdt_ndcg10_on_dien_sample is not None else None,
                "sampled_groups": dien_sample_groups,
                "max_groups_cfg": int(dien_max_groups),
            },
        }

    def compute_validation_compare(
        self,
        max_groups: int = 800,
        example_limit: int = 5,
        context_user_idx: int | None = None,
    ) -> dict[str, Any]:
        if not self.scene_feat_path.exists():
            raise FileNotFoundError(f"test feature file missing: {self.scene_feat_path}")

        df_all = pd.read_parquet(self.scene_feat_path)
        keep_cols = set(self.feat_df.columns) | {
            self.group_key,
            "note_idx",
            "y_multi",
            "click",
            "user_idx",
            "query",
            "recent_clicked_note_idxs",
        }
        df_all = df_all[[c for c in df_all.columns if c in keep_cols]].copy()
        df_all = df_all.groupby(self.group_key).filter(lambda x: len(x) > 1 and float(x["y_multi"].max()) > 0.0).reset_index(drop=True)
        if len(df_all) == 0:
            return {"scene": self.scene, "error": "no valid test groups"}

        gids = df_all[self.group_key].drop_duplicates().to_numpy()
        if len(gids) > max_groups:
            rng = np.random.default_rng(42)
            if context_user_idx is not None and "user_idx" in df_all.columns:
                user_gid_arr = (
                    df_all[pd.to_numeric(df_all["user_idx"], errors="coerce").fillna(-1).astype(int) == int(context_user_idx)][self.group_key]
                    .drop_duplicates()
                    .to_numpy()
                )
                user_gid_list = [int(x) for x in user_gid_arr.tolist()]
                if len(user_gid_list) >= int(max_groups):
                    gids = np.asarray(user_gid_list[: int(max_groups)], dtype=np.int64)
                else:
                    user_gid_set = set(user_gid_list)
                    rest_pool = np.asarray([int(x) for x in gids.tolist() if int(x) not in user_gid_set], dtype=np.int64)
                    need = int(max_groups) - len(user_gid_list)
                    if need > 0 and len(rest_pool) > 0:
                        rest_pick = rng.choice(rest_pool, size=min(need, len(rest_pool)), replace=False)
                        gids = np.asarray(user_gid_list + [int(x) for x in rest_pick.tolist()], dtype=np.int64)
                    else:
                        gids = np.asarray(user_gid_list, dtype=np.int64)
            else:
                gids = rng.choice(gids, size=max_groups, replace=False)
            df_all = df_all[df_all[self.group_key].isin(gids)].copy()

        df = df_all.copy()

        df["gbdt_score"] = self.predict_gbdt(df)
        val_row_cap = 120
        df_eval = (
            df.sort_values([self.group_key, "gbdt_score"], ascending=[True, False], kind="mergesort")
            .groupby(self.group_key, sort=False)
            .head(int(val_row_cap))
            .reset_index(drop=True)
        )
        if self.dien_model is not None:
            df_eval["dien_score"] = self.predict_dien(df_eval)
        else:
            df_eval["dien_score"] = 0.0

        if not any(col in df_eval.columns for col in ["score_ann", "dssm_score", "recall_score"]) and self.recall_test_path.exists():
            try:
                recall_cols = [c for c in [self.group_key, "note_idx", "score_ann", "recall_score", "score_swing", "score_usercf", "rank"]]
                gids_df = pd.DataFrame({self.group_key: df_eval[self.group_key].drop_duplicates().astype(np.int64).tolist()})
                con = duckdb.connect(database=":memory:")
                con.register("gids", gids_df)
                rec_df = con.execute(
                    f"""
                    SELECT r.*
                    FROM read_parquet(?) r
                    INNER JOIN gids g USING({self.group_key})
                    """,
                    [str(self.recall_test_path)],
                ).df()
                con.close()
                rec_df = rec_df[[c for c in recall_cols if c in rec_df.columns]].copy()
                rec_df[self.group_key] = pd.to_numeric(rec_df[self.group_key], errors="coerce").fillna(-1).astype(np.int64)
                rec_df["note_idx"] = pd.to_numeric(rec_df["note_idx"], errors="coerce").fillna(-1).astype(np.int64)
                if "rank" in rec_df.columns:
                    rec_df = rec_df.sort_values([self.group_key, "rank"], ascending=[True, True], kind="mergesort")
                rec_df = rec_df.drop_duplicates(subset=[self.group_key, "note_idx"], keep="first")
                merge_cols = [c for c in [self.group_key, "note_idx", "score_ann", "recall_score", "score_swing", "score_usercf"] if c in rec_df.columns]
                df_eval = df_eval.merge(rec_df[merge_cols], on=[self.group_key, "note_idx"], how="left")
            except Exception:
                pass

        dssm_source = None
        for col in ["score_ann", "dssm_score", "recall_score"]:
            if col in df_eval.columns:
                dssm_source = col
                break
        has_dssm_signal = dssm_source is not None
        if dssm_source is None:
            df_eval["dssm_score"] = np.nan
        elif dssm_source != "dssm_score":
            df_eval["dssm_score"] = pd.to_numeric(df_eval[dssm_source], errors="coerce").fillna(0.0).astype(np.float32)

        y_disc = self.discretize_relevance(df_eval["y_multi"].to_numpy())
        groups = df_eval[self.group_key].to_numpy()
        gbdt_ndcg10 = self.eval_ndcg_by_group(y_disc, df_eval["gbdt_score"].to_numpy(), groups, 10)
        dien_ndcg10 = self.eval_ndcg_by_group(y_disc, df_eval["dien_score"].to_numpy(), groups, 10) if self.dien_model is not None else None
        dssm_ndcg10 = self.eval_ndcg_by_group(y_disc, df_eval["dssm_score"].to_numpy(), groups, 10) if has_dssm_signal else None

        cmp_cap = max(0, int(example_limit))
        all_cmp_ids = df[self.group_key].drop_duplicates().tolist()
        user_ids: list[int] = []
        has_user_candidates = False
        user_sample_note = ""
        if context_user_idx is not None and "user_idx" in df.columns:
            user_ids = (
                df[pd.to_numeric(df["user_idx"], errors="coerce").fillna(-1).astype(int) == int(context_user_idx)][self.group_key]
                .drop_duplicates()
                .tolist()
            )
            has_user_candidates = len(user_ids) > 0
            if not has_user_candidates:
                user_sample_note = f"user={int(context_user_idx)} 无可用样例，已随机回退。"

        def _overlap_at_k(a: list[int], b: list[int], k: int = 10) -> float:
            aa = set(int(x) for x in a[:k])
            bb = set(int(x) for x in b[:k])
            if not aa and not bb:
                return 1.0
            if not aa:
                return 0.0
            return float(len(aa & bb) / max(len(aa), 1))

        def _hit_at10(a: list[int], b: list[int]) -> float:
            aa = set(int(x) for x in a[:10])
            bb = set(int(x) for x in b[:10])
            return 1.0 if len(aa & bb) > 0 else 0.0

        def _topk_unique(ids: list[int], k: int = 10) -> list[int]:
            out: list[int] = []
            seen: set[int] = set()
            for x in ids:
                nid = int(x)
                if nid in seen:
                    continue
                seen.add(nid)
                out.append(nid)
                if len(out) >= int(k):
                    break
            return out

        recall_map: dict[int, pd.DataFrame] = {}
        if self.recall_test_path.exists() and all_cmp_ids:
            try:
                gids_df = pd.DataFrame({self.group_key: [int(x) for x in all_cmp_ids]})
                con = duckdb.connect(database=":memory:")
                con.register("gids", gids_df)
                rec_df = con.execute(
                    f"""
                    SELECT r.*
                    FROM read_parquet(?) r
                    INNER JOIN gids g USING({self.group_key})
                    """,
                    [str(self.recall_test_path)],
                ).df()
                con.close()
                if len(rec_df) > 0:
                    rec_df[self.group_key] = pd.to_numeric(rec_df[self.group_key], errors="coerce").fillna(-1).astype(np.int64)
                    rec_df["note_idx"] = pd.to_numeric(rec_df["note_idx"], errors="coerce").fillna(-1).astype(np.int64)
                    if "rank" in rec_df.columns:
                        rec_df = rec_df.sort_values([self.group_key, "rank"], ascending=[True, True], kind="mergesort")
                    rec_df = rec_df.drop_duplicates(subset=[self.group_key, "note_idx"], keep="first")
                    for gid, sub in rec_df.groupby(self.group_key, sort=False):
                        recall_map[int(gid)] = sub.copy()
            except Exception:
                recall_map = {}

        def _build_chain_group(gid: int) -> pd.DataFrame:
            sub_feat = df[df[self.group_key] == int(gid)].copy()
            if len(sub_feat) == 0:
                return sub_feat
            sub_feat["note_idx"] = pd.to_numeric(sub_feat["note_idx"], errors="coerce").fillna(-1).astype(np.int64)
            sub_feat = sub_feat.sort_values(["y_multi"], ascending=[False], kind="mergesort")
            sub_feat = sub_feat.drop_duplicates(subset=["note_idx"], keep="first")

            rec = recall_map.get(int(gid))
            if rec is not None and len(rec) > 0:
                rec = rec.copy()
                if "score_ann" in rec.columns and "dssm_score" not in rec.columns:
                    rec["dssm_score"] = pd.to_numeric(rec["score_ann"], errors="coerce").fillna(0.0)
                elif "recall_score" in rec.columns and "dssm_score" not in rec.columns:
                    rec["dssm_score"] = pd.to_numeric(rec["recall_score"], errors="coerce").fillna(0.0)
                rec = rec[[c for c in rec.columns if c in {"note_idx", "rank", "dssm_score", "score_ann", "recall_score"}]].copy()
                chain = rec.merge(sub_feat, on="note_idx", how="inner")
                if len(chain) < 10:
                    chain = sub_feat.copy()
                    chain["rank"] = np.arange(1, len(chain) + 1, dtype=np.int64)
                    if "dssm_score" not in chain.columns:
                        chain["dssm_score"] = 0.0
            else:
                chain = sub_feat.copy()
                chain["rank"] = np.arange(1, len(chain) + 1, dtype=np.int64)
                if "dssm_score" not in chain.columns:
                    chain["dssm_score"] = 0.0

            if len(chain) == 0:
                return chain
            if "rank" in chain.columns:
                chain = chain.sort_values(["rank"], ascending=[True], kind="mergesort")
            chain = chain.head(int(self.recall_rank_cap)).copy()
            chain["gbdt_score"] = self.predict_gbdt(chain)
            chain = chain.sort_values(["gbdt_score"], ascending=[False], kind="mergesort").head(120).copy()
            if self.dien_model is not None and len(chain) > 0:
                chain["dien_score"] = self.predict_dien(chain)
            else:
                chain["dien_score"] = 0.0
            if "dssm_score" not in chain.columns:
                chain["dssm_score"] = 0.0
            return chain

        def _example_valid(true_rank: list[int], gbdt_rank: list[int], dien_rank: list[int], dssm_rank: list[int]) -> bool:
            if len(true_rank) < 10 or len(gbdt_rank) < 10 or len(dien_rank) < 10:
                return False
            if has_dssm_signal and len(dssm_rank) < 10:
                return False
            return True

        def _build_example_row(gid: int) -> dict[str, Any] | None:
            base_true = df[df[self.group_key] == int(gid)][["note_idx", "y_multi"]].copy()
            if len(base_true) == 0:
                return None
            base_true["note_idx"] = pd.to_numeric(base_true["note_idx"], errors="coerce").fillna(-1).astype(np.int64)
            true_full_rank = _topk_unique(base_true.sort_values("y_multi", ascending=False)["note_idx"].astype(int).tolist(), k=max(10_000, len(base_true)))
            true_rank = _topk_unique(base_true.sort_values("y_multi", ascending=False)["note_idx"].astype(int).tolist(), k=10)

            chain_sub = _build_chain_group(int(gid))
            if len(chain_sub) == 0:
                return None
            gbdt_rank = _topk_unique(chain_sub.sort_values("gbdt_score", ascending=False)["note_idx"].astype(int).tolist(), k=10)
            dien_rank = _topk_unique(chain_sub.sort_values("dien_score", ascending=False)["note_idx"].astype(int).tolist(), k=10)
            dssm_rank = _topk_unique(chain_sub.sort_values("dssm_score", ascending=False)["note_idx"].astype(int).tolist(), k=10) if has_dssm_signal else []
            if not _example_valid(true_rank, gbdt_rank, dien_rank, dssm_rank):
                return None
            true_rank_pos = {int(nid): i + 1 for i, nid in enumerate(true_full_rank)}

            gbdt_hits.append(_hit_at10(true_rank, gbdt_rank))
            dien_hits.append(_hit_at10(true_rank, dien_rank))
            if has_dssm_signal:
                dssm_hits.append(_hit_at10(true_rank, dssm_rank))

            return {
                "group_value": int(gid),
                "group_field": self.group_key,
                "true_top10": true_rank,
                "gbdt_top10": gbdt_rank,
                "dien_top10": dien_rank,
                "dssm_top10": dssm_rank,
                "true_rank_pos": true_rank_pos,
                "gbdt_overlap_at10": _overlap_at_k(true_rank, gbdt_rank, k=10),
                "dien_overlap_at10": _overlap_at_k(true_rank, dien_rank, k=10),
                "dssm_overlap_at10": _overlap_at_k(true_rank, dssm_rank, k=10),
            }

        example_rows = []
        gbdt_hits = []
        dien_hits = []
        dssm_hits = []
        cap = int(max(0, cmp_cap))
        rng_ex = np.random.default_rng()
        user_pool = [int(x) for x in user_ids]
        if user_pool:
            user_pool = [int(x) for x in rng_ex.permutation(np.asarray(user_pool, dtype=np.int64)).tolist()]
        other_pool = [int(x) for x in all_cmp_ids if int(x) not in set(user_pool)]
        if other_pool:
            other_pool = [int(x) for x in rng_ex.permutation(np.asarray(other_pool, dtype=np.int64)).tolist()]

        for gid in user_pool:
            if len(example_rows) >= cap:
                break
            row = _build_example_row(int(gid))
            if row is not None:
                example_rows.append(row)

        if context_user_idx is not None:
            if not has_user_candidates and not example_rows:
                user_sample_note = f"user={int(context_user_idx)} 无可用样例，已随机回退。"
            elif len(example_rows) < cap and not user_sample_note:
                user_sample_note = f"user={int(context_user_idx)} 样例不足，已随机回退补齐。"

        for gid in other_pool:
            if len(example_rows) >= cap:
                break
            row = _build_example_row(int(gid))
            if row is not None:
                example_rows.append(row)

        note_ids: set[int] = set()
        context_query_by_gid: dict[int, str] = {}
        context_recent_by_gid: dict[int, list[int]] = {}
        for ex in example_rows:
            gid = int(ex.get("group_value", -1))
            note_ids.update(int(x) for x in ex.get("true_top10", []))
            note_ids.update(int(x) for x in ex.get("gbdt_top10", []))
            note_ids.update(int(x) for x in ex.get("dien_top10", []))
            note_ids.update(int(x) for x in ex.get("dssm_top10", []))
            sub_ctx = df[df[self.group_key] == gid]
            if len(sub_ctx) > 0:
                if self.scene == "search":
                    qtxt = ""
                    if "query" in sub_ctx.columns:
                        qtxt = str(sub_ctx.iloc[0].get("query") or "")
                    if not qtxt:
                        qtxt = str(self.search_query_map.get(gid, ""))
                    context_query_by_gid[gid] = qtxt
                else:
                    hist_ids: list[int] = []
                    if "recent_clicked_note_idxs" in sub_ctx.columns:
                        raw = sub_ctx.iloc[0].get("recent_clicked_note_idxs", [])
                        if isinstance(raw, np.ndarray):
                            hist_ids = [int(x) for x in raw.tolist()[:20]]
                        elif isinstance(raw, list):
                            hist_ids = [int(x) for x in raw[:20]]
                    context_recent_by_gid[gid] = hist_ids
                    note_ids.update(hist_ids)

        title_map: dict[int, str] = {}
        if note_ids:
            note_df = self._fetch_notes(sorted(note_ids))
            if len(note_df) > 0:
                title_map = {
                    int(r["note_idx"]): str(r.get("note_title") or "(无标题)")
                    for r in note_df[["note_idx", "note_title"]].to_dict("records")
                }

        def _with_title(ids: list[int], true_pos: dict[int, int]) -> list[dict[str, Any]]:
            out: list[dict[str, Any]] = []
            for nid in ids:
                ni = int(nid)
                rank_in_true = true_pos.get(ni)
                out.append(
                    {
                        "note_idx": ni,
                        "title": title_map.get(ni, "(无标题)"),
                        "rank_in_true": (int(rank_in_true) if rank_in_true is not None else "-"),
                    }
                )
            return out

        context_user: dict[str, Any] | None = None
        if context_user_idx is not None:
            try:
                u = self.get_user(int(context_user_idx))
                recent_titles = [str(x.get("title") or "(无标题)") for x in (u.get("recent_behaviors") or [])[:20]]
                context_user = {
                    "user_idx": int(context_user_idx),
                    "features": u.get("features") or {},
                    "recent_behavior_titles": recent_titles,
                }
            except Exception:
                context_user = {
                    "user_idx": int(context_user_idx),
                    "features": {},
                    "recent_behavior_titles": [],
                }

        examples = []
        for ex in example_rows:
            gid = int(ex["group_value"])
            query_text = context_query_by_gid.get(gid, "") if self.scene == "search" else ""
            examples.append(
                {
                    "group_field": str(ex.get("group_field") or self.group_key),
                    "group_value": gid,
                    "gbdt_overlap_at10": float(ex["gbdt_overlap_at10"]),
                    "dien_overlap_at10": float(ex["dien_overlap_at10"]),
                    "dssm_overlap_at10": float(ex["dssm_overlap_at10"]),
                    "true_top10": _with_title(ex["true_top10"], ex["true_rank_pos"]),
                    "gbdt_top10": _with_title(ex["gbdt_top10"], ex["true_rank_pos"]),
                    "dien_top10": _with_title(ex["dien_top10"], ex["true_rank_pos"]),
                    "dssm_top10": _with_title(ex["dssm_top10"], ex["true_rank_pos"]),
                    "query": str(query_text or "") if self.scene == "search" else "",
                    "user_recent_titles": [
                        title_map.get(int(nid), "(无标题)")
                        for nid in context_recent_by_gid.get(gid, [])
                    ] if self.scene == "rec" else [],
                }
            )

        return {
            "scene": self.scene,
            "split": "test",
            "sampled_groups": int(df_eval[self.group_key].nunique()),
            "context_user_idx": (int(context_user_idx) if context_user_idx is not None else None),
            "context_user_group_count": int(
                df[pd.to_numeric(df["user_idx"], errors="coerce").fillna(-1).astype(int) == int(context_user_idx)][self.group_key].nunique()
            ) if (context_user_idx is not None and "user_idx" in df.columns) else 0,
            "context_user_sample_note": user_sample_note,
            "dssm_score_source": str(dssm_source or "none"),
            "dssm_validation_note": ("DSSM评分字段缺失，未计算DSSM对比。" if not has_dssm_signal else ""),
            "gbdt": {
                "ndcg@10": float(gbdt_ndcg10),
                "top10_hit_rate_vs_true": float(np.mean(gbdt_hits)) if gbdt_hits else 0.0,
            },
            "dien": {
                "ndcg@10": float(dien_ndcg10) if dien_ndcg10 is not None else None,
                "top10_hit_rate_vs_true": float(np.mean(dien_hits)) if dien_hits else 0.0,
            },
            "dssm": {
                "ndcg@10": float(dssm_ndcg10) if dssm_ndcg10 is not None else None,
                "top10_hit_rate_vs_true": float(np.mean(dssm_hits)) if dssm_hits else None,
            },
            "context_user": context_user,
            "examples": examples,
        }


class ServingAppState:
    """管理多场景状态，按需懒加载。"""

    def __init__(self, tag: str, gbdt_topn: int, recall_rank_cap: int):
        self.tag = tag
        self.gbdt_topn = gbdt_topn
        self.recall_rank_cap = recall_rank_cap
        self._states: dict[str, SceneServingState] = {}
        self._lock = threading.Lock()

    def get(self, scene: str) -> SceneServingState:
        if scene not in {"search", "rec"}:
            raise KeyError(f"invalid scene: {scene}")
        with self._lock:
            st = self._states.get(scene)
            if st is None:
                st = SceneServingState(
                    scene=scene,
                    tag=self.tag,
                    gbdt_topn=self.gbdt_topn,
                    recall_rank_cap=self.recall_rank_cap,
                )
                self._states[scene] = st
            return st

    def readiness(self) -> dict[str, Any]:
        out = {}
        for scene in ["search", "rec"]:
            try:
                out[scene] = self.get(scene).readiness()
            except Exception as e:  # noqa: BLE001
                out[scene] = {"error": f"{type(e).__name__}: {e}"}
        return out


class SearchServingState(SceneServingState):
    def __init__(self, tag: str = "easy", gbdt_topn: int = 500, recall_rank_cap: int = 800):
        super().__init__(scene="search", tag=tag, gbdt_topn=gbdt_topn, recall_rank_cap=recall_rank_cap)

class OnlineScenePipeline:
    def __init__(self, scene_state):
        self.state = scene_state

    def build_feed(self, user_idx: int, query: str, page: int, page_size: int, include_seen: bool = False) -> dict[str, Any]:
        t0 = time.perf_counter()
        req_id = self.state._runtime_request_id(user_idx=int(user_idx), query=query if self.state.scene == "search" else "rec")
        base_req_id, matched_query = self.state.resolve_request(
            user_idx=int(user_idx),
            query=query if self.state.scene == "search" else None,
        )

        runtime_query = (query or "").strip() if self.state.scene == "search" else matched_query
        runtime_feat_req = self.state.build_runtime_feat_req(user_idx=int(user_idx), request_id=int(req_id), query=runtime_query)
        if self.state.scene == "search":
            runtime_feat_req["query_ref_request_id"] = int(base_req_id)
        feat_req = self.state.get_feat_req(int(base_req_id))
        if feat_req.empty:
            feat_req = runtime_feat_req.copy()
        else:
            feat_req = feat_req.copy()
            feat_req[self.state.group_key] = int(req_id)
            feat_req["user_idx"] = int(user_idx)
            if self.state.scene == "search":
                feat_req["query"] = str(runtime_query or "")
            if len(runtime_feat_req) > 0:
                profile_row = runtime_feat_req.iloc[0].to_dict()
                user_cols = [
                    "gender_enc", "platform_enc", "age_enc", "location_enc",
                    "fans_num", "follows_num",
                ] + [f"dense_feat{i}" for i in range(1, 41)]
                for c in user_cols:
                    if c in profile_row:
                        feat_req[c] = profile_row[c]

        cold = is_cold_start(
            scene=self.state.scene,
            user_idx=int(user_idx),
            user_requests=self.state.user_requests,
            request_threshold=COLD_START_REQ_THRESHOLD,
        )
        if bool(self.state.realtime_ann_enabled):
            cold = False
        t_cold = time.perf_counter()
        recall_cand = run_recall(
            request_id=req_id,
            user_idx=int(user_idx),
            feat_req=feat_req,
            is_cold=cold,
            recall_rank_cap=self.state.recall_rank_cap,
            hot_route_topk=HOT_ROUTE_TOPK,
            fetch_recall_candidates=lambda rid, max_rank: self.state._fetch_recall_candidates(
                request_id=int(rid),
                max_rank=int(max_rank),
                runtime_feat_req=runtime_feat_req,
            ),
            group_key=self.state.group_key,
        )
        if self.state.scene == "search" and runtime_query:
            qrec = self.state._fetch_query_recall_candidates(request_id=int(req_id), query=runtime_query, topk=240)
            if not qrec.empty:
                recall_cand = pd.concat([qrec, recall_cand], ignore_index=True)
                if "rank" in recall_cand.columns:
                    recall_cand = recall_cand.sort_values(["rank"], ascending=[True], kind="mergesort")
                recall_cand = recall_cand.drop_duplicates(subset=[self.state.group_key, "note_idx"], keep="first").reset_index(drop=True)
        t_recall = time.perf_counter()
        prerank_cand = run_preranking(
            user_idx=int(user_idx),
            query=runtime_query,
            scene=self.state.scene,
            group_key=self.state.group_key,
            recall_cand=recall_cand,
            feat_req=feat_req,
            gbdt_topn=self.state.gbdt_topn,
            fetch_notes=self.state._fetch_notes,
            predict_gbdt=self.state.predict_gbdt,
        )
        t_prerank = time.perf_counter()
        cand, page_df = run_ranking(
            cand=prerank_cand,
            page=page,
            page_size=page_size,
            predict_dien=self.state.predict_dien,
            history_note_ids=self.state.get_user_history_notes(user_idx=int(user_idx), feat_req=feat_req),
        )
        if self.state.scene == "search" and str(runtime_query or "").strip() and not cand.empty:
            qtxt = str(runtime_query or "").strip()
            cand = cand.copy()
            cand["query_match_score"] = [
                _text_match_score(
                    qtxt,
                    str(r.get("note_title") or ""),
                    str(r.get("note_content") or ""),
                )
                for r in cand.to_dict("records")
            ]
            cand["query_aware_score"] = (
                pd.to_numeric(cand.get("dien_score", 0.0), errors="coerce").fillna(0.0)
                + 0.35 * pd.to_numeric(cand.get("query_match_score", 0.0), errors="coerce").fillna(0.0)
            )
            cand = cand.sort_values(
                ["query_aware_score", "dien_score", "gbdt_score", "rank"],
                ascending=[False, False, False, True],
                kind="mergesort",
            ).reset_index(drop=True)
            start = max(0, (int(page) - 1) * int(page_size))
            end = start + int(page_size)
            page_df = cand.iloc[start:end].copy()
        if self.state.scene == "rec" and not bool(include_seen):
            exposed_note_ids = set(self.state.get_recent_exposed_notes(user_idx=int(user_idx), max_len=400))
            if exposed_note_ids:
                unseen = cand[~pd.to_numeric(cand["note_idx"], errors="coerce").fillna(-1).astype(int).isin(exposed_note_ids)].reset_index(drop=True)
                if len(unseen) > 0:
                    cand = unseen
            page_df = cand.head(int(page_size)).copy()
        t_rank = time.perf_counter()

        recall_rank_map: dict[int, int] = {}
        if "rank" in cand.columns:
            recall_sorted = cand.sort_values(["rank"], ascending=[True], kind="mergesort")
            recall_rank_map = {
                int(v): i + 1
                for i, v in enumerate(pd.to_numeric(recall_sorted["note_idx"], errors="coerce").fillna(-1).astype(int).tolist())
                if int(v) >= 0
            }
        coarse_sorted = cand.sort_values(["gbdt_score"], ascending=[False], kind="mergesort")
        coarse_rank_map = {
            int(v): i + 1
            for i, v in enumerate(pd.to_numeric(coarse_sorted["note_idx"], errors="coerce").fillna(-1).astype(int).tolist())
            if int(v) >= 0
        }
        final_rank_map = {
            int(v): i + 1
            for i, v in enumerate(pd.to_numeric(cand["note_idx"], errors="coerce").fillna(-1).astype(int).tolist())
            if int(v) >= 0
        }

        stage_ms = {
            "coldstart": float((t_cold - t0) * 1000.0),
            "recall": float((t_recall - t_cold) * 1000.0),
            "preranking": float((t_prerank - t_recall) * 1000.0),
            "ranking": float((t_rank - t_prerank) * 1000.0),
        }
        total_ms = float((t_rank - t0) * 1000.0)

        cards = []
        def _safe_int_num(v: Any, default: int = 0) -> int:
            try:
                x = float(v)
                if np.isnan(x):
                    return int(default)
                return int(x)
            except Exception:
                return int(default)

        def _safe_float_num(v: Any, default: float = 0.0) -> float:
            try:
                x = float(v)
                if np.isnan(x) or np.isinf(x):
                    return float(default)
                return float(x)
            except Exception:
                return float(default)

        def _safe_title(v: Any) -> str:
            try:
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    return "(无标题)"
                s = str(v).strip()
                if not s or s.lower() == "nan":
                    return "(无标题)"
                return s
            except Exception:
                return "(无标题)"

        for r in page_df.to_dict("records"):
            try:
                note_idx = int(float(r.get("note_idx", -1)))
            except Exception:
                continue
            if note_idx < 0:
                continue
            imgs = _existing_images(_to_path_list(r.get("image_path")))
            cards.append(
                {
                    "scene": self.state.scene,
                    "request_id": int(req_id),
                    "search_idx": int(req_id) if self.state.scene == "search" else None,
                    "request_idx": int(req_id) if self.state.scene == "rec" else None,
                    "user_idx": int(user_idx),
                    "note_idx": note_idx,
                    "title": _safe_title(r.get("note_title")),
                    "cover_image": imgs[0] if imgs else "",
                    "image_count": len(imgs),
                    "accum_like_num": _safe_int_num(r.get("accum_like_num", 0), 0),
                    "accum_collect_num": _safe_int_num(r.get("accum_collect_num", 0), 0),
                    "accum_comment_num": _safe_int_num(r.get("accum_comment_num", 0), 0),
                    "scores": {
                        "dssm": _safe_float_num(r.get("dssm_score", 0.0), 0.0),
                        "gbdt": _safe_float_num(r.get("gbdt_score", 0.0), 0.0),
                        "dien": _safe_float_num(r.get("dien_score", 0.0), 0.0),
                        "query_match": _safe_float_num(r.get("query_match_score", 0.0), 0.0),
                    },
                    "labels": {
                        "y_multi": _safe_float_num(r.get("y_multi", 0.0), 0.0),
                        "click": _safe_int_num(r.get("click", 0.0), 0),
                    },
                    "stage_ranks": {
                        "recall": recall_rank_map.get(note_idx),
                        "coarse": coarse_rank_map.get(note_idx),
                        "final": final_rank_map.get(note_idx),
                        "rerank": final_rank_map.get(note_idx),
                    },
                }
            )

        if self.state.scene == "rec" and cards and not bool(include_seen):
            self.state.record_exposed_notes(
                user_idx=int(user_idx),
                note_ids=[int(x.get("note_idx", -1)) for x in cards],
            )

        return {
            "scene": self.state.scene,
            "user_idx": int(user_idx),
            "request_id": int(req_id),
            "request_id_note": "request_id 是当前场景下命中的请求样本ID（search 对应 search_idx，rec 对应 request_idx），并非 user_idx。",
            "cold_start": bool(cold),
            "stages": {
                "coldstart": {"enabled": bool(cold)},
                "recall": {"candidates": int(len(recall_cand))},
                "preranking": {"candidates": int(len(prerank_cand)), "topn": int(self.state.gbdt_topn)},
                "ranking": {"candidates": int(len(cand)), "page_items": int(len(page_df))},
            },
            "query_input": query,
            "matched_query": runtime_query,
            "total": int(len(cand)),
            "page": int(page),
            "page_size": int(page_size),
            "stage_ms": stage_ms,
            "latency_ms": total_ms,
            "latency_note": "latency_ms = 冷启动判定 + 召回 + 粗排 + 精排 四阶段耗时之和（单位毫秒）。",
            "items": cards,
        }


class OnlineRuntime:
    def __init__(self, tag: str, gbdt_topn: int, recall_rank_cap: int):
        self._app_state = ServingAppState(tag=tag, gbdt_topn=gbdt_topn, recall_rank_cap=recall_rank_cap)

    def readiness(self) -> dict[str, Any]:
        return self._app_state.readiness()

    def get_pipeline(self, scene: str) -> OnlineScenePipeline:
        return OnlineScenePipeline(self._app_state.get(scene))


class OnlineRuntimeRegistry:
    def __init__(self, default_tag: str, gbdt_topn: int, recall_rank_cap: int):
        self.default_tag = default_tag if default_tag in {"easy", "hard"} else "easy"
        self.gbdt_topn = int(gbdt_topn)
        self.recall_rank_cap = int(recall_rank_cap)
        self._states: dict[str, OnlineRuntime] = {}

    def get_runtime(self, tag: str | None) -> OnlineRuntime:
        use_tag = str(tag or self.default_tag).lower()
        if use_tag not in {"easy", "hard"}:
            raise ValueError(f"invalid tag: {tag}")
        rt = self._states.get(use_tag)
        if rt is None:
            rt = OnlineRuntime(tag=use_tag, gbdt_topn=self.gbdt_topn, recall_rank_cap=self.recall_rank_cap)
            self._states[use_tag] = rt
        return rt


def main() -> None:
    cfg = load_online_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=cfg.host)
    parser.add_argument("--port", type=int, default=cfg.port)
    parser.add_argument("--tag", type=str, default=cfg.default_tag)
    parser.add_argument("--gbdt-topn", type=int, default=cfg.gbdt_topn)
    parser.add_argument("--recall-rank-cap", type=int, default=cfg.recall_rank_cap)
    args = parser.parse_args()

    from backend.online.api.main import create_app

    app = create_app(tag=args.tag, gbdt_topn=args.gbdt_topn, recall_rank_cap=args.recall_rank_cap)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


__all__ = [
    "COLD_START_REQ_THRESHOLD",
    "HOT_ROUTE_TOPK",
    "SceneServingState",
    "ServingAppState",
    "SearchServingState",
    "OnlineScenePipeline",
    "OnlineRuntime",
    "OnlineRuntimeRegistry",
    "main",
]


if __name__ == "__main__":
    main()
