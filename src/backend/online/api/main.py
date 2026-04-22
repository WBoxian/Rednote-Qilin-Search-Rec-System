"""
Qilin Online API（FastAPI）
- 在线入口（src/backend/online/api）
- 支持 easy/hard tag 切换、search/rec 双场景服务
"""

from __future__ import annotations

import argparse
import mimetypes
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse

from backend.online.pipeline import OnlineRuntimeRegistry  # noqa: E402
from backend.online.config import DEFAULT_GBDT_TOPN, DEFAULT_RECALL_RANK_CAP

BASE_DIR = Path(__file__).resolve().parents[4]


class AppContext:
	def __init__(self, tag: str, gbdt_topn: int, recall_rank_cap: int):
		# 统一运行时注册器
		self.registry = OnlineRuntimeRegistry(default_tag=tag, gbdt_topn=gbdt_topn, recall_rank_cap=recall_rank_cap)
		self._cache_lock = threading.Lock()
		self._metrics_cache: dict[tuple[str, str, int], tuple[float, dict[str, Any]]] = {}
		self._validation_cache: dict[tuple[str, str, int, int], tuple[float, dict[str, Any]]] = {}
		self.metrics_ttl_sec = 15 * 60.0
		self.validation_ttl_sec = 3 * 60.0
		# 启动阶段不做全量预加载/预热，避免后端启动过慢、前端首屏直接报代理错误。
		# 运行时仍保持懒加载 + 接口级缓存。

	def get_runtime(self, tag: str | None):
		return self.registry.get_runtime(tag)

	def get_metrics_cached(
		self,
		scene: str,
		tag: str,
		sample_n: int,
		include_val: bool = True,
		test_use_online_recall: bool = True,
	) -> tuple[dict[str, Any], bool]:
		key = (str(tag), str(scene), int(sample_n), bool(include_val), bool(test_use_online_recall))
		now = time.time()
		with self._cache_lock:
			cached = self._metrics_cache.get(key)
			if cached is not None and (now - float(cached[0])) <= self.metrics_ttl_sec:
				return cached[1], True

		obj = self.get_runtime(tag).get_pipeline(scene).state.compute_metrics_layered(
			sample_n=int(sample_n),
			include_val=bool(include_val),
			test_use_online_recall=bool(test_use_online_recall),
		)
		with self._cache_lock:
			self._metrics_cache[key] = (time.time(), obj)
		return obj, False

	def get_validation_cached(
		self,
		scene: str,
		tag: str,
		max_groups: int,
		example_limit: int,
		user_idx: int | None,
	) -> tuple[dict[str, Any], bool]:
		# validation compare currently does not depend on user_idx; keep it out of the cache key
		key = (str(tag), str(scene), int(max_groups), int(example_limit))
		now = time.time()
		with self._cache_lock:
			cached = self._validation_cache.get(key)
			if cached is not None and (now - float(cached[0])) <= self.validation_ttl_sec:
				return cached[1], True

		obj = self.get_runtime(tag).get_pipeline(scene).state.compute_validation_compare(
			max_groups=int(max_groups),
			example_limit=int(example_limit),
			context_user_idx=(int(user_idx) if user_idx is not None else None),
		)
		with self._cache_lock:
			self._validation_cache[key] = (time.time(), obj)
		return obj, False


def _safe_int(v: Any, default: int = 0) -> int:
	try:
		return int(v)
	except Exception:
		return default


def create_app(
	tag: str = "hard",
	gbdt_topn: int = DEFAULT_GBDT_TOPN,
	recall_rank_cap: int = DEFAULT_RECALL_RANK_CAP,
) -> FastAPI:
	ctx = AppContext(tag=tag, gbdt_topn=gbdt_topn, recall_rank_cap=recall_rank_cap)

	app = FastAPI(title="Qilin Online API", version="0.0.1")
	app.state.ctx = ctx

	def _resolve_tag(scene: str, explicit_tag: str | None = None) -> str:
		cand = str(explicit_tag or "").strip().lower()
		if cand in {"easy", "hard"}:
			return cand
		default_tag = str(getattr(app.state.ctx.registry, "default_tag", "easy") or "easy").lower()
		return default_tag if default_tag in {"easy", "hard"} else "easy"

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
		return PlainTextResponse(
			"Qilin Online API is running.\n"
			"Frontend: http://127.0.0.1:5173\n"
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
			user = app.state.ctx.get_runtime(use_tag).get_pipeline(scene).state.get_user(_safe_int(user_idx))
			return {"ok": True, "scene": scene, "tag": use_tag, "user": user, "latency_ms": float((time.perf_counter() - t0) * 1000.0)}
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
			obj = app.state.ctx.get_runtime(use_tag).get_pipeline(scene).state.list_users(
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
	):
		try:
			t0 = time.perf_counter()
			use_tag = _resolve_tag(scene=scene, explicit_tag=tag)
			# 在线推荐主链路：冷启动 -> 召回 -> 粗排 -> 精排
			payload = app.state.ctx.get_runtime(use_tag).get_pipeline(scene).build_feed(
				user_idx=_safe_int(user_idx),
				query=query,
				page=max(1, _safe_int(page, 1)),
				page_size=max(20, min(40, _safe_int(page_size, 20))),
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

	@app.get("/api/metrics")
	def metrics(
		scene: str = Query("search"),
		tag: str = Query(""),
		sample_n: int = Query(100),
		include_val: int = Query(1),
		online_recall: int = Query(0),
	):
		try:
			t0 = time.perf_counter()
			use_tag = _resolve_tag(scene=scene, explicit_tag=tag)
			obj, hit = app.state.ctx.get_metrics_cached(
				scene=scene,
				tag=use_tag,
				sample_n=max(10, _safe_int(sample_n, 100)),
				include_val=bool(_safe_int(include_val, 1)),
				test_use_online_recall=bool(_safe_int(online_recall, 1)),
			)
			return {
				"ok": True,
				"scene": scene,
				"tag": use_tag,
				"metrics": obj,
				"cache_hit": bool(hit),
				"latency_ms": float((time.perf_counter() - t0) * 1000.0),
			}
		except ValueError as e:
			raise HTTPException(status_code=400, detail=str(e)) from e

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
			return {
				"ok": True,
				"scene": scene,
				"tag": use_tag,
				"validation": obj,
				"cache_hit": bool(hit),
				"latency_ms": float((time.perf_counter() - t0) * 1000.0),
			}
		except ValueError as e:
			raise HTTPException(status_code=400, detail=str(e)) from e

	@app.get("/image/{image_path:path}")
	def image(image_path: str):
		normalized = str(image_path or "").lstrip("/")
		if normalized.startswith("image/"):
			normalized = normalized[len("image/") :]
		fpath = (BASE_DIR / "image" / normalized).resolve()
		if BASE_DIR.resolve() not in fpath.parents:
			raise HTTPException(status_code=400, detail="invalid image path")
		if not fpath.exists() or not fpath.is_file():
			raise HTTPException(status_code=404, detail="image not found")
		ctype = mimetypes.guess_type(str(fpath))[0] or "application/octet-stream"
		return FileResponse(path=fpath, media_type=ctype)

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
