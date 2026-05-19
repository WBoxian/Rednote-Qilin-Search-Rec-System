"""
Qilin Online API（FastAPI）
- 在线入口（src/backend/online/api）
- 支持 easy/hard tag 切换、search/rec 双场景服务
"""

from __future__ import annotations

import argparse
import random
import json
import math
import mimetypes
import os
import threading
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import duckdb
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel

from backend.online.pipeline import OnlineRuntimeRegistry  # noqa: E402
from backend.online.config import DEFAULT_GBDT_TOPN, DEFAULT_RECALL_RANK_CAP

BASE_DIR = Path(__file__).resolve().parents[4]
FRONTEND_DIST_DIR = BASE_DIR / "src" / "frontend" / "dist"


class BehaviorDeleteItem(BaseModel):
	scene: str
	note_idx: int
	ts: int | None = None
	request_id: int | None = None


class BatchBehaviorDeletePayload(BaseModel):
	user_idx: int
	items: list[BehaviorDeleteItem]


class AppContext:
	def __init__(self, tag: str, gbdt_topn: int, recall_rank_cap: int):
		# 统一运行时注册器
		self.registry = OnlineRuntimeRegistry(default_tag=tag, gbdt_topn=gbdt_topn, recall_rank_cap=recall_rank_cap)
		self.scene_default_tags = {
			"search": self._normalize_tag(os.getenv("QILIN_TAG_SEARCH"), fallback=tag),
			"rec": self._normalize_tag(os.getenv("QILIN_TAG_REC"), fallback=tag),
		}
		self._cache_lock = threading.Lock()
		self._metrics_cache: dict[tuple[str, str, int, bool], tuple[float, dict[str, Any]]] = {}
		self._validation_cache: dict[tuple[str, str, int, int], tuple[float, dict[str, Any]]] = {}
		self._user_validation_example_cache_lock = threading.Lock()
		self._user_validation_example_cache: dict[tuple[str, str, int], dict[str, Any] | None] = {}
		self._login_user_cache_lock = threading.Lock()
		self._login_user_catalog: dict[str, list[dict[str, Any]]] = {}
		self.metrics_ttl_sec = 15 * 60.0
		self.validation_ttl_sec = 15 * 60.0
		self._prewarm_enabled = str(os.getenv("QILIN_ASYNC_PREWARM", "1")).strip().lower() not in {"0", "false", "no"}
		self._blocking_prewarm = str(os.getenv("QILIN_BLOCKING_PREWARM", "0")).strip().lower() in {"1", "true", "yes"}
		self._prewarm_feed_enabled = str(os.getenv("QILIN_PREWARM_FEED", "1")).strip().lower() not in {"0", "false", "no"}
		self._prewarm_feed_page_size = max(12, min(60, int(os.getenv("QILIN_PREWARM_FEED_PAGE_SIZE", "30"))))
		self._blocking_feed_scenes = self._parse_scene_list(os.getenv("QILIN_BLOCKING_PREWARM_FEED_SCENES", "rec"))
		self._async_feed_scenes = self._parse_scene_list(os.getenv("QILIN_ASYNC_PREWARM_FEED_SCENES", "search"))
		self._snapshot_root = BASE_DIR / "outputs" / "serving_cache"
		if self._prewarm_enabled:
			if self._blocking_prewarm:
				self._prewarm_homepage_runtime()
				threading.Thread(target=self._async_prewarm_runtime_artifacts, daemon=True).start()
			else:
				threading.Thread(target=self._async_prewarm_default_runtime, daemon=True).start()

	def get_runtime(self, tag: str | None):
		return self.registry.get_runtime(tag)

	@staticmethod
	def _normalize_tag(raw: str | None, fallback: str = "easy") -> str:
		cand = str(raw or "").strip().lower()
		if cand in {"easy", "hard"}:
			return cand
		base = str(fallback or "easy").strip().lower()
		return base if base in {"easy", "hard"} else "easy"

	def resolve_scene_tag(self, scene: str, explicit_tag: str | None = None) -> str:
		cand = str(explicit_tag or "").strip().lower()
		if cand in {"easy", "hard"}:
			return cand
		scene_key = str(scene or "").strip().lower()
		if scene_key in self.scene_default_tags:
			return self.scene_default_tags[scene_key]
		return self._normalize_tag(self.registry.default_tag, fallback="easy")

	@staticmethod
	def _parse_scene_list(raw: str | None) -> list[str]:
		scene_set: list[str] = []
		for token in str(raw or "").split(","):
			scene = str(token).strip().lower()
			if scene in {"search", "rec"} and scene not in scene_set:
				scene_set.append(scene)
		return scene_set

	def _async_prewarm_default_runtime(self) -> None:
		self._prewarm_homepage_runtime(blocking=False)

	def _load_default_pipelines(self, scenes: list[str] | None = None) -> dict[str, Any]:
		pipelines: dict[str, Any] = {}
		scene_list = scenes or ["search", "rec"]
		for scene in scene_list:
			try:
				use_tag = self.resolve_scene_tag(scene)
				pipelines[scene] = self.registry.get_runtime(use_tag).get_pipeline(scene)
			except Exception:
				continue
		return pipelines

	def _prewarm_homepage_runtime(self, blocking: bool = True) -> None:
		try:
			if self._prewarm_feed_enabled:
				target_scenes = self._blocking_feed_scenes if blocking else self._async_feed_scenes
				pipelines = self._load_default_pipelines(target_scenes or None)
				for scene in (target_scenes or list(pipelines.keys())):
					pipe = pipelines.get(scene)
					if pipe is None:
						continue
					try:
						pipe.prewarm_homepage_feed(page_size=self._prewarm_feed_page_size)
					except Exception:
						continue
		except Exception:
			return

	def _async_prewarm_runtime_artifacts(self) -> None:
		return

	def _login_catalog_paths(self, scene: str) -> tuple[Path, Path]:
		scene_key = "search" if str(scene) == "search" else "rec"
		feature_path = BASE_DIR / "datasets" / "user_feat" / "train-00000-of-00001.parquet"
		request_path = (
			BASE_DIR / "datasets" / ("search_test" if scene_key == "search" else "recommendation_test") / "train-00000-of-00001.parquet"
		)
		return feature_path, request_path

	def _load_login_user_catalog(self, scene: str) -> list[dict[str, Any]]:
		scene_key = "search" if str(scene) == "search" else "rec"
		with self._login_user_cache_lock:
			cached = self._login_user_catalog.get(scene_key)
			if cached is not None:
				return cached
		feature_path, request_path = self._login_catalog_paths(scene_key)
		if not feature_path.exists() or not request_path.exists():
			return []
		con = duckdb.connect(database=":memory:")
		try:
			rows = con.execute(
				"""
				WITH profiles AS (
				  SELECT
				    user_idx,
				    any_value(gender) AS gender,
				    any_value(age) AS age,
				    any_value(platform) AS platform,
				    any_value(location) AS location,
				    max(CAST(fans_num AS BIGINT)) AS fans_num,
				    max(CAST(follows_num AS BIGINT)) AS follows_num
				  FROM read_parquet(?)
				  GROUP BY user_idx
				),
				reqs AS (
				  SELECT
				    user_idx,
				    count(*) AS request_count_in_test
				  FROM read_parquet(?)
				  GROUP BY user_idx
				)
				SELECT
				  p.user_idx,
				  p.gender,
				  p.age,
				  p.platform,
				  p.location,
				  coalesce(p.fans_num, 0) AS fans_num,
				  coalesce(p.follows_num, 0) AS follows_num,
				  coalesce(r.request_count_in_test, 0) AS request_count_in_test
				FROM profiles p
				LEFT JOIN reqs r USING(user_idx)
				ORDER BY request_count_in_test DESC, user_idx ASC
				""",
				[str(feature_path), str(request_path)],
			).fetchdf()
		finally:
			con.close()
		catalog = rows.to_dict("records") if len(rows) > 0 else []
		with self._login_user_cache_lock:
			self._login_user_catalog[scene_key] = catalog
		return catalog

	def list_login_users(self, scene: str, limit: int = 20, offset: int = 0, random_show: bool = False) -> list[dict[str, Any]]:
		rows = list(self._load_login_user_catalog(scene))
		if not rows:
			return []
		n = max(1, min(200, int(limit)))
		off = max(0, int(offset))
		if bool(random_show):
			if len(rows) <= n:
				return rows
			return random.sample(rows, n)
		return rows[off : off + n]

	def get_login_user(self, scene: str, user_idx: int) -> dict[str, Any] | None:
		uid = int(user_idx)
		for row in self._load_login_user_catalog(scene):
			if _safe_int(row.get("user_idx"), -1) == uid:
				return dict(row)
		return None

	def async_prewarm_user_homepage(self, user_idx: int) -> None:
		uid = int(user_idx)
		if uid < 0:
			return
		def _run() -> None:
			try:
				scene = "rec"
				use_tag = self.resolve_scene_tag(scene=scene)
				self.get_runtime(use_tag).get_pipeline(scene).build_feed(
					user_idx=uid,
					query="",
					page=1,
					page_size=self._prewarm_feed_page_size,
					refresh_key="",
					exclude_note_ids=None,
				)
			except Exception:
				return
		threading.Thread(target=_run, daemon=True).start()

	def async_prewarm_scene_runtime(self, scenes: list[str] | None = None) -> None:
		scene_list = [s for s in (scenes or ["search", "rec"]) if s in {"search", "rec"}]
		if not scene_list:
			return
		def _run() -> None:
			try:
				self._load_default_pipelines(scene_list)
			except Exception:
				return
		threading.Thread(target=_run, daemon=True).start()

	def async_prewarm_user_validation_examples(self, user_idx: int) -> None:
		uid = int(user_idx)
		if uid < 0:
			return
		def _run() -> None:
			for scene in ["search", "rec"]:
				try:
					use_tag = self.resolve_scene_tag(scene=scene)
					row = self.get_runtime(use_tag).get_pipeline(scene).state.build_user_validation_example(uid)
					self.set_cached_user_validation_example(scene=scene, tag=use_tag, user_idx=uid, example=row)
				except Exception:
					continue
		threading.Thread(target=_run, daemon=True).start()

	def prewarm_user_homepage_sync(self, user_idx: int) -> dict[str, Any] | None:
		uid = int(user_idx)
		if uid < 0:
			return None
		try:
			scene = "rec"
			use_tag = self.resolve_scene_tag(scene=scene)
			return self.get_runtime(use_tag).get_pipeline(scene).build_feed(
				user_idx=uid,
				query="",
				page=1,
				page_size=self._prewarm_feed_page_size,
				refresh_key="",
				exclude_note_ids=None,
			)
		except Exception:
			return None

	def get_cached_user_validation_example(self, scene: str, tag: str, user_idx: int) -> dict[str, Any] | None:
		key = (str(tag), str(scene), int(user_idx))
		with self._user_validation_example_cache_lock:
			row = self._user_validation_example_cache.get(key)
			return deepcopy(row) if row else None

	def set_cached_user_validation_example(self, scene: str, tag: str, user_idx: int, example: dict[str, Any] | None) -> None:
		key = (str(tag), str(scene), int(user_idx))
		with self._user_validation_example_cache_lock:
			self._user_validation_example_cache[key] = deepcopy(example) if example else None

	def _metrics_snapshot_path(
		self,
		scene: str,
		tag: str,
		sample_n: int,
		include_val: bool,
	) -> Path:
		name = f"metrics_n{int(sample_n)}_train{int(bool(include_val))}.json"
		return self._snapshot_root / str(scene) / str(tag) / name

	def _validation_snapshot_path(
		self,
		scene: str,
		tag: str,
		max_groups: int,
		example_limit: int,
	) -> Path:
		name = f"validation_g{int(max_groups)}_e{int(example_limit)}.json"
		return self._snapshot_root / str(scene) / str(tag) / name

	def _load_json_snapshot(self, path: Path) -> dict[str, Any] | None:
		try:
			if not path.exists():
				return None
			return json.loads(path.read_text(encoding="utf-8"))
		except Exception:
			return None

	def _save_json_snapshot(self, path: Path, payload: dict[str, Any]) -> None:
		try:
			path.parent.mkdir(parents=True, exist_ok=True)
			path.write_text(
				json.dumps(_json_safe(payload), ensure_ascii=False),
				encoding="utf-8",
			)
		except Exception:
			return

	def get_metrics_cached(
		self,
		scene: str,
		tag: str,
		sample_n: int,
		include_val: bool = True,
	) -> tuple[dict[str, Any], bool]:
		key = (str(tag), str(scene), int(sample_n), bool(include_val))
		now = time.time()
		with self._cache_lock:
			cached = self._metrics_cache.get(key)
			if cached is not None and (now - float(cached[0])) <= self.metrics_ttl_sec:
				return cached[1], True
		snapshot = self._load_json_snapshot(
			self._metrics_snapshot_path(
				scene=scene,
				tag=tag,
				sample_n=sample_n,
				include_val=include_val,
			)
		)
		if snapshot is None:
			raise FileNotFoundError(
				f"offline metrics snapshot not found: scene={scene}, tag={tag}, sample_n={sample_n}, include_val={int(bool(include_val))}"
			)
		with self._cache_lock:
			self._metrics_cache[key] = (time.time(), snapshot)
		return snapshot, True

	def get_validation_cached(
		self,
		scene: str,
		tag: str,
		max_groups: int,
		example_limit: int,
		user_idx: int | None,
	) -> tuple[dict[str, Any], bool]:
		key = (str(tag), str(scene), int(max_groups), int(example_limit))
		now = time.time()
		with self._cache_lock:
			cached = self._validation_cache.get(key)
			if cached is not None and (now - float(cached[0])) <= self.validation_ttl_sec:
				return cached[1], True
		snapshot = self._load_json_snapshot(
			self._validation_snapshot_path(
				scene=scene,
				tag=tag,
				max_groups=max_groups,
				example_limit=example_limit,
			)
		)
		if snapshot is None:
			raise FileNotFoundError(
				f"offline validation snapshot not found: scene={scene}, tag={tag}, groups={max_groups}, examples={example_limit}"
			)
		with self._cache_lock:
			self._validation_cache[key] = (time.time(), snapshot)
		return snapshot, True


def _safe_int(v: Any, default: int = 0) -> int:
	try:
		return int(v)
	except Exception:
		return default


def _json_safe(value: Any) -> Any:
	if isinstance(value, dict):
		return {str(k): _json_safe(v) for k, v in value.items()}
	if isinstance(value, (list, tuple, set)):
		return [_json_safe(v) for v in value]
	try:
		if hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray)):
			return _json_safe(value.tolist())
	except Exception:
		pass
	if isinstance(value, float):
		if not math.isfinite(value):
			return 0.0
		return float(value)
	try:
		if hasattr(value, "item"):
			return _json_safe(value.item())
	except Exception:
		return value
	return value


def _personalize_validation(validation: dict[str, Any], context_user_idx: int | None) -> dict[str, Any]:
	if context_user_idx is None:
		return validation
	try:
		examples = list(validation.get("examples") or [])
	except Exception:
		return validation
	if not examples:
		return validation
	uid = int(context_user_idx)
	out = deepcopy(validation)
	personalized = []
	for ex in examples:
		row = dict(ex)
		src_uid = _safe_int(row.get("source_user_idx"), -1)
		row["is_current_user"] = bool(src_uid == uid)
		personalized.append(row)
	personalized.sort(
		key=lambda row: (
			0 if bool(row.get("is_current_user")) else 1,
			-int(_safe_int(row.get("result_count"), 0)),
			-int(_safe_int((row.get("overlap") or {}).get("ranking"), 0)),
			-int(_safe_int((row.get("overlap") or {}).get("preranking"), 0)),
			int(_safe_int(row.get("request_id"), 0)),
		)
	)
	out["examples"] = personalized
	return out


def _ensure_current_user_example(
	validation: dict[str, Any],
	app_ctx: AppContext,
	runtime: Any,
	scene: str,
	tag: str,
	context_user_idx: int | None,
	max_groups: int,
	example_limit: int,
) -> dict[str, Any]:
	if context_user_idx is None:
		return validation
	uid = int(context_user_idx)
	try:
		examples = list(validation.get("examples") or [])
	except Exception:
		examples = []
	if any(_safe_int(row.get("source_user_idx"), -1) == uid for row in examples):
		out = deepcopy(validation)
		personalized = []
		for row in examples:
			item = dict(row)
			item["is_current_user"] = bool(_safe_int(item.get("source_user_idx"), -1) == uid)
			personalized.append(item)
		personalized.sort(
			key=lambda row: (
				0 if bool(row.get("is_current_user")) else 1,
				-int(_safe_int(row.get("result_count"), 0)),
				-int(_safe_int((row.get("overlap") or {}).get("ranking"), 0)),
				-int(_safe_int((row.get("overlap") or {}).get("preranking"), 0)),
				int(_safe_int(row.get("request_id"), 0)),
			)
		)
		out["examples"] = personalized
		return out
	try:
		current_row = app_ctx.get_cached_user_validation_example(scene=scene, tag=tag, user_idx=uid)
		if not current_row:
			current_row = runtime.get_pipeline(scene).state.build_user_validation_example(uid)
			app_ctx.set_cached_user_validation_example(scene=scene, tag=tag, user_idx=uid, example=current_row)
		if not current_row or _safe_int(current_row.get("source_user_idx"), -1) != uid:
			return validation
		first = dict(current_row)
		first["is_current_user"] = True
		seen_requests = {int(first.get("request_id", -1))}
		merged = [first]
		for row in examples:
			rid = _safe_int(row.get("request_id"), -1)
			if rid in seen_requests:
				continue
			item = dict(row)
			item["is_current_user"] = False
			merged.append(item)
			if len(merged) >= max(1, int(example_limit)):
				break
		out = deepcopy(validation)
		out["examples"] = merged
		return out
	except Exception:
		return validation


def create_app(
	tag: str = "hard",
	gbdt_topn: int = DEFAULT_GBDT_TOPN,
	recall_rank_cap: int = DEFAULT_RECALL_RANK_CAP,
) -> FastAPI:
	ctx = AppContext(tag=tag, gbdt_topn=gbdt_topn, recall_rank_cap=recall_rank_cap)

	app = FastAPI(title="Qilin Online API", version="0.0.1")
	app.state.ctx = ctx

	def _resolve_tag(scene: str, explicit_tag: str | None = None) -> str:
		return app.state.ctx.resolve_scene_tag(scene=scene, explicit_tag=explicit_tag)

	app.add_middleware(
		CORSMiddleware,
		allow_origins=["*"],
		allow_credentials=True,
		allow_methods=["*"],
		allow_headers=["*"],
	)

	@app.get("/api/health")
	def health(verbose: int = Query(0)):
		if int(verbose) <= 0:
			loaded_tags = []
			try:
				loaded_tags = sorted(list(app.state.ctx.registry._states.keys()))
			except Exception:
				loaded_tags = []
			return {
				"ok": True,
				"status": "alive",
				"loaded_tags": loaded_tags,
				"note": "append ?verbose=1 to run full model readiness",
			}

		out = {}
		for t in ["easy", "hard"]:
			try:
				out[t] = app.state.ctx.get_runtime(t).readiness()
			except Exception as e:  # noqa: BLE001
				out[t] = {"error": str(e)}
		return {"ok": True, "readiness": out}

	@app.get("/")
	def root():
		index = FRONTEND_DIST_DIR / "index.html"
		if index.exists():
			return FileResponse(path=index, media_type="text/html")
		return PlainTextResponse(
			"Qilin Online API is running.\n"
			"Frontend build not found.\n"
			"Health API: /api/health\n"
		)

	@app.get("/api/scenes")
	def scenes():
		return {"ok": True, "scenes": ["search", "rec"]}

	@app.get("/api/login")
	def login(
		scene: str = Query("search"),
		user_idx: int = Query(...),
		tag: str = Query(""),
	):
		try:
			t0 = time.perf_counter()
			use_tag = _resolve_tag(scene=scene, explicit_tag=tag)
			user = app.state.ctx.get_login_user(scene=scene, user_idx=_safe_int(user_idx))
			if user is None:
				raise KeyError(f"user_idx not found: {_safe_int(user_idx)}")
			homepage_feed = app.state.ctx.prewarm_user_homepage_sync(_safe_int(user_idx))
			app.state.ctx.async_prewarm_scene_runtime(["search"])
			app.state.ctx.async_prewarm_user_validation_examples(_safe_int(user_idx))
			return {
				"ok": True,
				"scene": scene,
				"tag": use_tag,
				"user": user,
				"homepage_feed": _json_safe(homepage_feed),
				"latency_ms": float((time.perf_counter() - t0) * 1000.0),
			}
		except ValueError as e:
			raise HTTPException(status_code=400, detail=str(e)) from e
		except KeyError as e:
			raise HTTPException(status_code=404, detail=str(e)) from e

	@app.get("/api/users")
	def users(
		scene: str = Query("search"),
		tag: str = Query(""),
		limit: int = Query(20),
		offset: int = Query(0),
		random_show: int = Query(0),
	):
		try:
			t0 = time.perf_counter()
			use_tag = _resolve_tag(scene=scene, explicit_tag=tag)
			obj = app.state.ctx.list_login_users(
				scene=scene,
				limit=max(1, min(200, _safe_int(limit, 20))),
				offset=max(0, _safe_int(offset, 0)),
				random_show=bool(_safe_int(random_show, 0)),
			)
			return {"ok": True, "scene": scene, "tag": use_tag, "users": obj, "latency_ms": float((time.perf_counter() - t0) * 1000.0)}
		except ValueError as e:
			raise HTTPException(status_code=400, detail=str(e)) from e

	@app.get("/api/user")
	def user(
		scene: str = Query("search"),
		user_idx: int = Query(...),
		tag: str = Query(""),
	):
		try:
			t0 = time.perf_counter()
			use_tag = _resolve_tag(scene=scene, explicit_tag=tag)
			user_info = app.state.ctx.get_runtime(use_tag).get_pipeline(scene).state.get_user(_safe_int(user_idx))
			return {"ok": True, "scene": scene, "tag": use_tag, "user": user_info, "latency_ms": float((time.perf_counter() - t0) * 1000.0)}
		except ValueError as e:
			raise HTTPException(status_code=400, detail=str(e)) from e
		except KeyError as e:
			raise HTTPException(status_code=404, detail=str(e)) from e

	@app.get("/api/feed")
	def feed(
		scene: str = Query("search"),
		tag: str = Query(""),
		user_idx: int = Query(...),
		query: str = Query(""),
		page: int = Query(1),
		page_size: int = Query(20),
		refresh_key: str = Query(""),
		exclude_note_ids: str = Query(""),
	):
		try:
			t0 = time.perf_counter()
			use_tag = _resolve_tag(scene=scene, explicit_tag=tag)
			exclude_ids = []
			if str(exclude_note_ids or "").strip():
				for raw in str(exclude_note_ids).split(","):
					raw = raw.strip()
					if not raw:
						continue
					try:
						nid = int(raw)
					except Exception:
						continue
					if nid >= 0:
						exclude_ids.append(nid)
				exclude_ids = list(dict.fromkeys(exclude_ids))[:200]
			# 在线推荐主链路：冷启动 -> 召回 -> 粗排 -> 精排
			payload = app.state.ctx.get_runtime(use_tag).get_pipeline(scene).build_feed(
				user_idx=_safe_int(user_idx),
				query=query,
				page=max(1, _safe_int(page, 1)),
				page_size=max(20, min(40, _safe_int(page_size, 20))),
				refresh_key=str(refresh_key or ""),
				exclude_note_ids=exclude_ids,
			)
			return {"ok": True, "scene": scene, "tag": use_tag, "feed": payload, "latency_ms": float((time.perf_counter() - t0) * 1000.0)}
		except ValueError as e:
			raise HTTPException(status_code=400, detail=str(e)) from e
		except KeyError as e:
			raise HTTPException(status_code=404, detail=str(e)) from e

	@app.get("/api/note")
	def note(
		scene: str = Query("search"),
		tag: str = Query(""),
		user_idx: int = Query(...),
		request_id: int = Query(...),
		note_idx: int = Query(...),
		query: str = Query(""),
		meta_only: bool = Query(False),
	):
		try:
			t0 = time.perf_counter()
			use_tag = _resolve_tag(scene=scene, explicit_tag=tag)
			detail = app.state.ctx.get_runtime(use_tag).get_pipeline(scene).state.get_note_detail(
				user_idx=_safe_int(user_idx),
				request_id=_safe_int(request_id),
				note_idx=_safe_int(note_idx),
				query=str(query or ""),
				meta_only=bool(meta_only),
			)
			return {"ok": True, "scene": scene, "tag": use_tag, "detail": detail, "latency_ms": float((time.perf_counter() - t0) * 1000.0)}
		except ValueError as e:
			raise HTTPException(status_code=400, detail=str(e)) from e
		except KeyError as e:
			raise HTTPException(status_code=404, detail=str(e)) from e

	@app.post("/api/behavior/click")
	def behavior_click(
		scene: str = Query("search"),
		tag: str = Query(""),
		user_idx: int = Query(...),
		note_idx: int = Query(...),
		request_id: int | None = Query(None),
		query: str = Query(""),
	):
		try:
			t0 = time.perf_counter()
			use_tag = _resolve_tag(scene=scene, explicit_tag=tag)
			accepted = app.state.ctx.get_runtime(use_tag).get_pipeline(scene).state.record_click(
				user_idx=_safe_int(user_idx),
				note_idx=_safe_int(note_idx),
				request_id=(_safe_int(request_id) if request_id is not None else None),
				query=str(query or ""),
			)
			return {
				"ok": True,
				"scene": scene,
				"tag": use_tag,
				"accepted": bool(accepted),
				"latency_ms": float((time.perf_counter() - t0) * 1000.0),
			}
		except ValueError as e:
			raise HTTPException(status_code=400, detail=str(e)) from e

	@app.post("/api/behavior/view")
	def behavior_view(
		scene: str = Query("search"),
		tag: str = Query(""),
		user_idx: int = Query(...),
		note_idx: int = Query(...),
		request_id: int | None = Query(None),
		query: str = Query(""),
	):
		try:
			t0 = time.perf_counter()
			use_tag = _resolve_tag(scene=scene, explicit_tag=tag)
			accepted = app.state.ctx.get_runtime(use_tag).get_pipeline(scene).state.record_view(
				user_idx=_safe_int(user_idx),
				note_idx=_safe_int(note_idx),
				request_id=(_safe_int(request_id) if request_id is not None else None),
				query=str(query or ""),
			)
			return {
				"ok": True,
				"scene": scene,
				"tag": use_tag,
				"accepted": bool(accepted),
				"latency_ms": float((time.perf_counter() - t0) * 1000.0),
			}
		except ValueError as e:
			raise HTTPException(status_code=400, detail=str(e)) from e

	@app.post("/api/behavior/engage")
	def behavior_engage(
		scene: str = Query("search"),
		tag: str = Query(""),
		user_idx: int = Query(...),
		note_idx: int = Query(...),
		request_id: int | None = Query(None),
		query: str = Query(""),
		like: int = Query(0),
		collect: int = Query(0),
		comment: int = Query(0),
		share: int = Query(0),
		page_time: float = Query(0.0),
	):
		try:
			t0 = time.perf_counter()
			use_tag = _resolve_tag(scene=scene, explicit_tag=tag)
			accepted = app.state.ctx.get_runtime(use_tag).get_pipeline(scene).state.record_engage(
				user_idx=_safe_int(user_idx),
				note_idx=_safe_int(note_idx),
				request_id=(_safe_int(request_id) if request_id is not None else None),
				query=str(query or ""),
				like=max(0, _safe_int(like, 0)),
				collect=max(0, _safe_int(collect, 0)),
				comment=max(0, _safe_int(comment, 0)),
				share=max(0, _safe_int(share, 0)),
				page_time=max(0.0, float(page_time or 0.0)),
			)
			return {
				"ok": True,
				"scene": scene,
				"tag": use_tag,
				"accepted": bool(accepted),
				"latency_ms": float((time.perf_counter() - t0) * 1000.0),
			}
		except ValueError as e:
			raise HTTPException(status_code=400, detail=str(e)) from e

	@app.delete("/api/behavior")
	def delete_behavior(
		scene: str = Query("search"),
		tag: str = Query(""),
		user_idx: int = Query(...),
		note_idx: int = Query(...),
		ts: int | None = Query(None),
		request_id: int | None = Query(None),
	):
		try:
			t0 = time.perf_counter()
			use_tag = _resolve_tag(scene=scene, explicit_tag=tag)
			accepted = app.state.ctx.get_runtime(use_tag).get_pipeline(scene).state.delete_behavior(
				user_idx=_safe_int(user_idx),
				note_idx=_safe_int(note_idx),
				ts=(_safe_int(ts) if ts is not None else None),
				request_id=(_safe_int(request_id) if request_id is not None else None),
			)
			return {
				"ok": True,
				"scene": scene,
				"tag": use_tag,
				"accepted": bool(accepted),
				"latency_ms": float((time.perf_counter() - t0) * 1000.0),
			}
		except ValueError as e:
			raise HTTPException(status_code=400, detail=str(e)) from e

	@app.post("/api/behavior/batch_delete")
	def delete_behaviors_batch(
		payload: BatchBehaviorDeletePayload = Body(...),
		tag: str = Query(""),
	):
		try:
			t0 = time.perf_counter()
			uid = _safe_int(payload.user_idx)
			items = payload.items or []
			if not items:
				return {
					"ok": True,
					"user_idx": uid,
					"deleted_count": 0,
					"requested_count": 0,
					"latency_ms": float((time.perf_counter() - t0) * 1000.0),
				}
			grouped: dict[str, list[dict[str, int | None]]] = {}
			for item in items:
				scene = "search" if str(item.scene or "").strip().lower() != "rec" else "rec"
				grouped.setdefault(scene, []).append(
					{
						"note_idx": _safe_int(item.note_idx),
						"ts": (_safe_int(item.ts) if item.ts is not None else None),
						"request_id": (_safe_int(item.request_id) if item.request_id is not None else None),
					}
				)
			deleted_count = 0
			for scene, scene_items in grouped.items():
				use_tag = _resolve_tag(scene=scene, explicit_tag=tag)
				deleted_count += int(
					app.state.ctx.get_runtime(use_tag).get_pipeline(scene).state.delete_behaviors(
						user_idx=uid,
						items=scene_items,
					)
				)
			return {
				"ok": True,
				"user_idx": uid,
				"deleted_count": int(deleted_count),
				"requested_count": int(len(items)),
				"latency_ms": float((time.perf_counter() - t0) * 1000.0),
			}
		except ValueError as e:
			raise HTTPException(status_code=400, detail=str(e)) from e

	@app.get("/api/suggest")
	def suggest(
		scene: str = Query("search"),
		tag: str = Query(""),
		query: str = Query(""),
		limit: int = Query(8),
	):
		try:
			t0 = time.perf_counter()
			use_tag = _resolve_tag(scene=scene, explicit_tag=tag)
			items = app.state.ctx.get_runtime(use_tag).get_pipeline(scene).state.suggest_queries(
				query=str(query or ""),
				limit=max(1, min(12, _safe_int(limit, 8))),
			)
			return {
				"ok": True,
				"scene": scene,
				"tag": use_tag,
				"items": items,
				"latency_ms": float((time.perf_counter() - t0) * 1000.0),
			}
		except ValueError as e:
			raise HTTPException(status_code=400, detail=str(e)) from e

	@app.get("/api/metrics")
	def metrics(
		scene: str = Query("search"),
		tag: str = Query(""),
		sample_n: int = Query(100),
		include_val: int = Query(1),
	):
		try:
			t0 = time.perf_counter()
			use_tag = _resolve_tag(scene=scene, explicit_tag=tag)
			obj, hit = app.state.ctx.get_metrics_cached(
				scene=scene,
				tag=use_tag,
				sample_n=max(2, _safe_int(sample_n, 100)),
				include_val=bool(_safe_int(include_val, 1)),
			)
			return {
				"ok": True,
				"scene": scene,
				"tag": use_tag,
				"metrics": _json_safe(obj),
				"cache_hit": bool(hit),
				"latency_ms": float((time.perf_counter() - t0) * 1000.0),
			}
		except ValueError as e:
			raise HTTPException(status_code=400, detail=str(e)) from e
		except Exception as e:
			raise HTTPException(status_code=500, detail=str(e)) from e

	@app.get("/api/validation")
	def validation(
		scene: str = Query("search"),
		tag: str = Query(""),
		max_groups: int = Query(800),
		example_limit: int = Query(5),
		user_idx: int | None = Query(None),
	):
		try:
			t0 = time.perf_counter()
			use_tag = _resolve_tag(scene=scene, explicit_tag=tag)
			obj, hit = app.state.ctx.get_validation_cached(
				scene=scene,
				tag=use_tag,
				max_groups=_safe_int(max_groups, 800),
				example_limit=max(0, _safe_int(example_limit, 5)),
				user_idx=(_safe_int(user_idx) if user_idx is not None else None),
			)
			obj = _personalize_validation(
				obj,
				context_user_idx=(_safe_int(user_idx) if user_idx is not None else None),
			)
			obj = _ensure_current_user_example(
				obj,
				app_ctx=app.state.ctx,
				runtime=app.state.ctx.get_runtime(use_tag),
				scene=scene,
				tag=use_tag,
				context_user_idx=(_safe_int(user_idx) if user_idx is not None else None),
				max_groups=_safe_int(max_groups, 800),
				example_limit=max(1, _safe_int(example_limit, 5)),
			)
			try:
				obj["examples"] = list(obj.get("examples") or [])[: max(1, min(5, _safe_int(example_limit, 5)))]
			except Exception:
				pass
			return {
				"ok": True,
				"scene": scene,
				"tag": use_tag,
				"validation": _json_safe(obj),
				"cache_hit": bool(hit),
				"latency_ms": float((time.perf_counter() - t0) * 1000.0),
			}
		except ValueError as e:
			raise HTTPException(status_code=400, detail=str(e)) from e
		except Exception as e:
			raise HTTPException(status_code=500, detail=str(e)) from e

	@app.get("/image/{image_path:path}")
	def image(image_path: str):
		normalized = str(image_path or "").lstrip("/")
		if normalized.startswith("image/"):
			normalized = normalized[len("image/") :]
		image_root = (BASE_DIR / "image").resolve()
		fpath = (image_root / normalized).resolve()
		if fpath != image_root and image_root not in fpath.parents:
			raise HTTPException(status_code=400, detail="invalid image path")
		if not fpath.exists() or not fpath.is_file():
			raise HTTPException(status_code=404, detail="image not found")
		ctype = mimetypes.guess_type(str(fpath))[0] or "application/octet-stream"
		return FileResponse(path=fpath, media_type=ctype)

	@app.get("/{file_path:path}")
	def frontend(file_path: str):
		normalized = str(file_path or "").lstrip("/")
		if normalized.startswith("api/") or normalized.startswith("image/"):
			raise HTTPException(status_code=404, detail="not found")
		index = FRONTEND_DIST_DIR / "index.html"
		if not index.exists():
			raise HTTPException(status_code=404, detail="frontend build not found")
		if normalized:
			fpath = (FRONTEND_DIST_DIR / normalized).resolve()
			dist_root = FRONTEND_DIST_DIR.resolve()
			if dist_root == fpath or dist_root in fpath.parents:
				if fpath.exists() and fpath.is_file():
					ctype = mimetypes.guess_type(str(fpath))[0] or "application/octet-stream"
					return FileResponse(path=fpath, media_type=ctype)
		return FileResponse(path=index, media_type="text/html")

	return app


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--host", type=str, default="0.0.0.0")
	parser.add_argument("--port", type=int, default=18080)
	parser.add_argument("--tag", type=str, default="hard")
	parser.add_argument("--gbdt-topn", type=int, default=DEFAULT_GBDT_TOPN)
	parser.add_argument("--recall-rank-cap", type=int, default=DEFAULT_RECALL_RANK_CAP)
	args = parser.parse_args()

	import uvicorn

	uvicorn.run(
		create_app(
			tag=args.tag,
			gbdt_topn=args.gbdt_topn,
			recall_rank_cap=args.recall_rank_cap,
		),
		host=args.host,
		port=args.port,
		log_level="info",
	)


if __name__ == "__main__":
	main()
