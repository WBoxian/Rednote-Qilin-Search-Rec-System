"""
Qilin Backend Online Pipeline
- 在线职责：维护单场景运行时状态（请求数据、特征缓存、模型对象）
- 推理链路：冷启动识别 -> 多路召回 -> 粗排 -> 精排 -> 结果组装
- 服务职责：作为 FastAPI 启动入口，完成场景状态初始化与运行时注册
- 运行形态：支持 search / rec 双场景，并按 tag 加载 outputs/deploy/{scene}/{tag}
"""

from __future__ import annotations

import argparse
import importlib.util
import json
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
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
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
        self.user_feat_path = DATASETS_DIR / "user_feat" / "train-00000-of-00001.parquet"
        self.scene_feat_path = FEATURE_DIR / f"{scene}_test_features.parquet"
        self.scene_train_feat_path = FEATURE_DIR / f"{scene}_train_features.parquet"
        self.notes_glob = str(DATASETS_DIR / "notes" / "train-*.parquet")
        self.recall_test_path = None
        self.realtime_cache = build_realtime_cache()

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
        self.feat_groups = {
            int(k): v.reset_index(drop=True)
            for k, v in self.feat_df.groupby(self.group_key, sort=False)
        }

        self.user_feat_df = pd.read_parquet(self.user_feat_path)
        self.user_feat_df["user_idx"] = self.user_feat_df["user_idx"].astype(np.int64)
        self.user_feat_map = {
            int(r["user_idx"]): r for r in self.user_feat_df.to_dict("records")
        }

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
        if not reqs:
            if len(self.req_df) == 0:
                raise KeyError("no requests available for this scene")
            rid = int(self.req_df.iloc[0][self.group_key])
            return rid, ""

        if self.scene == "rec":
            rid = int(reqs[-1])
            return rid, ""

        if not query or not query.strip():
            rid = int(reqs[-1])
            return rid, self.search_query_map.get(rid, "")

        q = query.strip().lower()
        best_rid = int(reqs[-1])
        best_score = -1e9
        for rid in reqs:
            sq = self.search_query_map.get(int(rid), "")
            score = _text_match_score(q, sq, sq)
            if score > best_score:
                best_score = score
                best_rid = int(rid)

        if best_score <= 0 and len(self.search_query_map) > 0:
            for rid, sq in self.search_query_map.items():
                score = _text_match_score(q, sq, sq)
                if score > best_score:
                    best_score = score
                    best_rid = int(rid)

        return best_rid, self.search_query_map.get(best_rid, "")

    def _fetch_recall_candidates(self, request_id: int, max_rank: int) -> pd.DataFrame:
        return fetch_recall_candidates(
            request_id=int(request_id),
            max_rank=int(max_rank),
            group_key=self.group_key,
            req_df=self.req_df,
            feat_groups=self.feat_groups,
            dssm_recaller=self.dssm_recaller,
            recall_test_path=self.recall_test_path,
        )

    def _fetch_notes(self, note_ids: list[int]) -> pd.DataFrame:
        if not note_ids:
            return pd.DataFrame(columns=[
                "note_idx", "note_title", "note_content", "image_path",
                "accum_like_num", "accum_collect_num", "accum_comment_num"
            ])
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
        return pd.DataFrame(rows)

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

    def build_feed(self, user_idx: int, query: str, page: int, page_size: int) -> dict[str, Any]:
        return OnlineScenePipeline(self).build_feed(
            user_idx=int(user_idx),
            query=query,
            page=int(page),
            page_size=int(page_size),
        )

    def get_note_detail(self, user_idx: int, request_id: int, note_idx: int) -> dict[str, Any]:
        feed = self.build_feed(user_idx=user_idx, query="", page=1, page_size=200)
        target = next((x for x in feed["items"] if int(x["note_idx"]) == int(note_idx) and int(x["request_id"]) == int(request_id)), None)

        note_meta = self._fetch_notes([note_idx])
        if len(note_meta) == 0:
            raise KeyError(f"note_idx not found: {note_idx}")
        n = note_meta.iloc[0].to_dict()

        feat_req = self.feat_groups.get(int(request_id), pd.DataFrame())
        row = feat_req[feat_req["note_idx"] == int(note_idx)]
        label = {"y_multi": 0.0, "click": 0}
        if len(row) > 0:
            label = {
                "y_multi": float(row.iloc[0].get("y_multi", 0.0)),
                "click": int(float(row.iloc[0].get("click", 0.0))),
            }

        if target is None:
            one = row.copy() if len(row) > 0 else pd.DataFrame([{self.group_key: request_id, "note_idx": note_idx, "user_idx": user_idx}])
            one = one.merge(note_meta, on="note_idx", how="left")
            one["dssm_score"] = 0.0
            one["gbdt_score"] = self.predict_gbdt(one)
            one["dien_score"] = self.predict_dien(one)
            one["dien_score"] = np.nan_to_num(one["dien_score"], nan=0.0, posinf=1e6, neginf=-1e6)
            scores = {
                "dssm": float(one.iloc[0].get("dssm_score", 0.0)),
                "gbdt": float(one.iloc[0].get("gbdt_score", 0.0)),
                "dien": float(one.iloc[0].get("dien_score", 0.0)),
            }
        else:
            scores = target["scores"]

        return {
            "scene": self.scene,
            "user_idx": int(user_idx),
            "request_id": int(request_id),
            "note_idx": int(note_idx),
            "title": str(n.get("note_title") or "(无标题)"),
            "content": str(n.get("note_content") or ""),
            "images": _existing_images(_to_path_list(n.get("image_path"))),
            "accum_like_num": int(float(n.get("accum_like_num", 0) or 0)),
            "accum_collect_num": int(float(n.get("accum_collect_num", 0) or 0)),
            "accum_comment_num": int(float(n.get("accum_comment_num", 0) or 0)),
            "scores": {
                "dssm": float(scores.get("dssm", 0.0)),
                "gbdt": float(scores.get("gbdt", 0.0)),
                "dien": float(scores.get("dien", 0.0)),
            },
            "labels": label,
        }

    def get_user(self, user_idx: int) -> dict[str, Any]:
        uid = int(user_idx)
        info = None
        if self.realtime_cache is not None:
            info = self.realtime_cache.get_user_profile(uid)
        if info is None:
            info = self.user_feat_map.get(uid)
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
        return {
            "scene": self.scene,
            "user_idx": uid,
            "features": info,
            "request_count_in_test": int(len(reqs)),
            "recent_requests": recent,
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

    def record_click(self, user_idx: int, note_idx: int) -> None:
        if self.realtime_cache is not None:
            self.realtime_cache.record_click(int(user_idx), self.scene, int(note_idx))

    def compute_test_metrics(self, dien_max_groups: int = 1200) -> dict[str, Any]:
        recall_json = OUT_DATA_DIR / f"recall_eval_{self.scene}_test_{self.tag}.json"
        recall = {}
        if recall_json.exists():
            recall = json.loads(recall_json.read_text())

        eval_df = self.feat_df.copy()
        eval_df["gbdt_score"] = self.predict_gbdt(eval_df)
        y_disc = self.discretize_relevance(eval_df["y_multi"].to_numpy())
        groups = eval_df[self.group_key].to_numpy()
        gbdt_ndcg10 = self.eval_ndcg_by_group(y_disc, eval_df["gbdt_score"].to_numpy(), groups, 10)
        gbdt_ndcg100 = self.eval_ndcg_by_group(y_disc, eval_df["gbdt_score"].to_numpy(), groups, 100)

        dien_ndcg10 = None
        dien_sample_groups = 0
        if self.dien_model is not None:
            pos_groups = eval_df.loc[eval_df["y_multi"] > 0, self.group_key].drop_duplicates().to_numpy()
            if len(pos_groups) > dien_max_groups:
                rng = np.random.default_rng(42)
                pos_groups = rng.choice(pos_groups, size=dien_max_groups, replace=False)
            sub = eval_df[eval_df[self.group_key].isin(pos_groups)].copy()
            sub["dien_score"] = self.predict_dien(sub)
            sy = self.discretize_relevance(sub["y_multi"].to_numpy())
            sg = sub[self.group_key].to_numpy()
            dien_ndcg10 = self.eval_ndcg_by_group(sy, sub["dien_score"].to_numpy(), sg, 10)
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
            },
            "dien": {
                "ndcg@10_exposed_test_sampled": float(dien_ndcg10) if dien_ndcg10 is not None else None,
                "sampled_groups": dien_sample_groups,
                "max_groups_cfg": int(dien_max_groups),
            },
        }

    def compute_validation_compare(self, max_groups: int = 800) -> dict[str, Any]:
        if not self.scene_train_feat_path.exists():
            raise FileNotFoundError(f"train feature file missing: {self.scene_train_feat_path}")

        df = pd.read_parquet(self.scene_train_feat_path)
        df = df[[c for c in df.columns if c in set(self.feat_df.columns) | {self.group_key, "note_idx", "y_multi", "click"}]].copy()
        df = df.groupby(self.group_key).filter(lambda x: len(x) > 1 and float(x["y_multi"].max()) > 0.0).reset_index(drop=True)
        if len(df) == 0:
            return {"scene": self.scene, "error": "no valid train groups"}

        gids = df[self.group_key].drop_duplicates().to_numpy()
        if len(gids) > max_groups:
            rng = np.random.default_rng(42)
            gids = rng.choice(gids, size=max_groups, replace=False)
            df = df[df[self.group_key].isin(gids)].copy()

        df["gbdt_score"] = self.predict_gbdt(df)
        if self.dien_model is not None:
            df["dien_score"] = self.predict_dien(df)
        else:
            df["dien_score"] = 0.0

        y_disc = self.discretize_relevance(df["y_multi"].to_numpy())
        groups = df[self.group_key].to_numpy()
        gbdt_ndcg10 = self.eval_ndcg_by_group(y_disc, df["gbdt_score"].to_numpy(), groups, 10)
        dien_ndcg10 = self.eval_ndcg_by_group(y_disc, df["dien_score"].to_numpy(), groups, 10) if self.dien_model is not None else None

        top1_true = (
            df.sort_values([self.group_key, "y_multi"], ascending=[True, False], kind="mergesort")
            .groupby(self.group_key, as_index=False)
            .first()[[self.group_key, "note_idx"]]
            .rename(columns={"note_idx": "true_top1_note"})
        )
        top1_gbdt = (
            df.sort_values([self.group_key, "gbdt_score"], ascending=[True, False], kind="mergesort")
            .groupby(self.group_key, as_index=False)
            .first()[[self.group_key, "note_idx"]]
            .rename(columns={"note_idx": "gbdt_top1_note"})
        )
        top1_dien = (
            df.sort_values([self.group_key, "dien_score"], ascending=[True, False], kind="mergesort")
            .groupby(self.group_key, as_index=False)
            .first()[[self.group_key, "note_idx"]]
            .rename(columns={"note_idx": "dien_top1_note"})
        )
        cmp_df = top1_true.merge(top1_gbdt, on=self.group_key, how="inner").merge(top1_dien, on=self.group_key, how="inner")
        gbdt_top1_hit = float((cmp_df["true_top1_note"] == cmp_df["gbdt_top1_note"]).mean()) if len(cmp_df) > 0 else 0.0
        dien_top1_hit = float((cmp_df["true_top1_note"] == cmp_df["dien_top1_note"]).mean()) if len(cmp_df) > 0 else 0.0

        def _overlap_at_k(a: list[int], b: list[int], k: int = 10) -> float:
            aa = set(int(x) for x in a[:k])
            bb = set(int(x) for x in b[:k])
            if not aa and not bb:
                return 1.0
            if not aa:
                return 0.0
            return float(len(aa & bb) / max(len(aa), 1))

        example_rows = []
        for gid in cmp_df[self.group_key].head(20).tolist():
            sub = df[df[self.group_key] == gid][["note_idx", "y_multi", "gbdt_score", "dien_score"]].copy()
            true_rank = sub.sort_values("y_multi", ascending=False)["note_idx"].astype(int).tolist()[:10]
            gbdt_rank = sub.sort_values("gbdt_score", ascending=False)["note_idx"].astype(int).tolist()[:10]
            dien_rank = sub.sort_values("dien_score", ascending=False)["note_idx"].astype(int).tolist()[:10]

            example_rows.append(
                {
                    "request_id": int(gid),
                    "true_top10": true_rank,
                    "gbdt_top10": gbdt_rank,
                    "dien_top10": dien_rank,
                    "gbdt_overlap_at10": _overlap_at_k(true_rank, gbdt_rank, k=10),
                    "dien_overlap_at10": _overlap_at_k(true_rank, dien_rank, k=10),
                }
            )

        note_ids: set[int] = set()
        for ex in example_rows:
            note_ids.update(int(x) for x in ex.get("true_top10", []))
            note_ids.update(int(x) for x in ex.get("gbdt_top10", []))
            note_ids.update(int(x) for x in ex.get("dien_top10", []))

        title_map: dict[int, str] = {}
        if note_ids:
            note_df = self._fetch_notes(sorted(note_ids))
            if len(note_df) > 0:
                title_map = {
                    int(r["note_idx"]): str(r.get("note_title") or "(无标题)")
                    for r in note_df[["note_idx", "note_title"]].to_dict("records")
                }

        def _with_title(ids: list[int]) -> list[dict[str, Any]]:
            return [{"note_idx": int(nid), "title": title_map.get(int(nid), "(无标题)")} for nid in ids]

        examples = []
        for ex in example_rows:
            examples.append(
                {
                    "request_id": int(ex["request_id"]),
                    "gbdt_overlap_at10": float(ex["gbdt_overlap_at10"]),
                    "dien_overlap_at10": float(ex["dien_overlap_at10"]),
                    "true_top10": _with_title(ex["true_top10"]),
                    "gbdt_top10": _with_title(ex["gbdt_top10"]),
                    "dien_top10": _with_title(ex["dien_top10"]),
                }
            )

        return {
            "scene": self.scene,
            "split": "train(validation-proxy)",
            "sampled_groups": int(df[self.group_key].nunique()),
            "gbdt": {
                "ndcg@10": float(gbdt_ndcg10),
                "top1_hit_rate_vs_true": float(gbdt_top1_hit),
            },
            "dien": {
                "ndcg@10": float(dien_ndcg10) if dien_ndcg10 is not None else None,
                "top1_hit_rate_vs_true": float(dien_top1_hit),
            },
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

    def build_feed(self, user_idx: int, query: str, page: int, page_size: int) -> dict[str, Any]:
        t0 = time.perf_counter()
        req_id, matched_query = self.state.resolve_request(user_idx, query)
        feat_req = self.state.feat_groups.get(req_id)
        if feat_req is None:
            feat_req = self.state.feat_df.iloc[0:0].copy()

        cold = is_cold_start(
            scene=self.state.scene,
            user_idx=int(user_idx),
            user_requests=self.state.user_requests,
            request_threshold=COLD_START_REQ_THRESHOLD,
        )
        t_cold = time.perf_counter()
        recall_cand = run_recall(
            request_id=req_id,
            user_idx=int(user_idx),
            feat_req=feat_req,
            is_cold=cold,
            recall_rank_cap=self.state.recall_rank_cap,
            hot_route_topk=HOT_ROUTE_TOPK,
            fetch_recall_candidates=self.state._fetch_recall_candidates,
            group_key=self.state.group_key,
        )
        t_recall = time.perf_counter()
        prerank_cand = run_preranking(
            user_idx=int(user_idx),
            query=query,
            scene=self.state.scene,
            group_key=self.state.group_key,
            recall_cand=recall_cand,
            feat_req=feat_req,
            gbdt_topn=self.state.gbdt_topn,
            fetch_notes=self.state._fetch_notes,
            text_match_score=_text_match_score,
            predict_gbdt=self.state.predict_gbdt,
        )
        t_prerank = time.perf_counter()
        cand, page_df = run_ranking(
            scene=self.state.scene,
            cand=prerank_cand,
            page=page,
            page_size=page_size,
            predict_dien=self.state.predict_dien,
            history_note_ids=self.state.get_user_history_notes(user_idx=int(user_idx), feat_req=feat_req),
        )
        t_rank = time.perf_counter()

        stage_ms = {
            "coldstart": float((t_cold - t0) * 1000.0),
            "recall": float((t_recall - t_cold) * 1000.0),
            "preranking": float((t_prerank - t_recall) * 1000.0),
            "ranking": float((t_rank - t_prerank) * 1000.0),
        }
        total_ms = float((t_rank - t0) * 1000.0)

        cards = []
        for r in page_df.to_dict("records"):
            imgs = _existing_images(_to_path_list(r.get("image_path")))
            cards.append(
                {
                    "scene": self.state.scene,
                    "request_id": int(req_id),
                    "search_idx": int(req_id) if self.state.scene == "search" else None,
                    "request_idx": int(req_id) if self.state.scene == "rec" else None,
                    "user_idx": int(user_idx),
                    "note_idx": int(r["note_idx"]),
                    "title": str(r.get("note_title") or "(无标题)"),
                    "cover_image": imgs[0] if imgs else "",
                    "image_count": len(imgs),
                    "accum_like_num": int(float(r.get("accum_like_num", 0) or 0)),
                    "accum_collect_num": int(float(r.get("accum_collect_num", 0) or 0)),
                    "accum_comment_num": int(float(r.get("accum_comment_num", 0) or 0)),
                    "scores": {
                        "dssm": float(r.get("dssm_score", 0.0)),
                        "gbdt": float(r.get("gbdt_score", 0.0)),
                        "dien": float(r.get("dien_score", 0.0)),
                        "final": float(r.get("final_score", 0.0)),
                    },
                    "labels": {
                        "y_multi": float(r.get("y_multi", 0.0) or 0.0),
                        "click": int(float(r.get("click", 0.0) or 0.0)),
                    },
                }
            )

        return {
            "scene": self.state.scene,
            "user_idx": int(user_idx),
            "request_id": int(req_id),
            "cold_start": bool(cold),
            "stages": {
                "coldstart": {"enabled": bool(cold)},
                "recall": {"candidates": int(len(recall_cand))},
                "preranking": {"candidates": int(len(prerank_cand)), "topn": int(self.state.gbdt_topn)},
                "ranking": {"candidates": int(len(cand)), "page_items": int(len(page_df))},
            },
            "query_input": query,
            "matched_query": matched_query,
            "total": int(len(cand)),
            "page": int(page),
            "page_size": int(page_size),
            "stage_ms": stage_ms,
            "latency_ms": total_ms,
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
