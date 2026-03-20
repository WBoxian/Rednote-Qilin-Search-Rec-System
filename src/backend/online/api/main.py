"""
Qilin Online API（FastAPI）
- 在线入口（src/backend/online/api）
- 支持 easy/hard tag 切换、search/rec 双场景服务
"""

from __future__ import annotations

import argparse
import mimetypes
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse

from backend.online.pipeline import OnlineRuntimeRegistry  # noqa: E402

BASE_DIR = Path(__file__).resolve().parents[4]


class AppContext:
	def __init__(self, tag: str, gbdt_topn: int, recall_rank_cap: int):
		# 统一运行时注册器：按 easy/hard 标签懒加载对应场景 pipeline
		self.registry = OnlineRuntimeRegistry(default_tag=tag, gbdt_topn=gbdt_topn, recall_rank_cap=recall_rank_cap)

	def get_runtime(self, tag: str | None):
		return self.registry.get_runtime(tag)


def _safe_int(v: Any, default: int = 0) -> int:
	try:
		return int(v)
	except Exception:
		return default


def create_app(
	tag: str = "hard",
	gbdt_topn: int = 500,
	recall_rank_cap: int = 800,
) -> FastAPI:
	ctx = AppContext(tag=tag, gbdt_topn=gbdt_topn, recall_rank_cap=recall_rank_cap)

	app = FastAPI(title="Qilin Online API", version="0.0.1")
	app.state.ctx = ctx

	def _resolve_tag(scene: str, explicit_tag: str | None = None) -> str:
		cand = str(explicit_tag or "").strip().lower()
		need_ann = scene in {"search", "rec"}

		def _ready(use_tag: str) -> dict[str, Any] | None:
			try:
				return app.state.ctx.get_runtime(use_tag).get_pipeline(scene).state.readiness()
			except Exception:
				return None

		if cand in {"easy", "hard"}:
			if need_ann:
				ready = _ready(cand)
				if bool((ready or {}).get("realtime_ann_enabled")):
					return cand
				for t in ["hard", "easy"]:
					ready = _ready(t)
					if bool((ready or {}).get("realtime_ann_enabled")):
						return t
			return cand

		if need_ann:
			for t in ["hard", "easy"]:
				ready = _ready(t)
				if bool((ready or {}).get("realtime_ann_enabled")):
					return t

		for t in ["hard", "easy"]:
			ready = _ready(t)
			if bool((ready or {}).get("lgb_loaded") or (ready or {}).get("xgb_loaded") or (ready or {}).get("dien_loaded") or (ready or {}).get("dssm_user_tower_loaded")):
				return t
		return "easy"

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
	):
		try:
			t0 = time.perf_counter()
			use_tag = _resolve_tag(scene=scene, explicit_tag=tag)
			detail = app.state.ctx.get_runtime(use_tag).get_pipeline(scene).state.get_note_detail(
				user_idx=_safe_int(user_idx),
				request_id=_safe_int(request_id),
				note_idx=_safe_int(note_idx),
				query=str(query or ""),
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
		dien_max_groups: int = Query(1200),
		max_groups: int = Query(3000),
	):
		try:
			t0 = time.perf_counter()
			use_tag = _resolve_tag(scene=scene, explicit_tag=tag)
			obj = app.state.ctx.get_runtime(use_tag).get_pipeline(scene).state.compute_test_metrics(
				dien_max_groups=_safe_int(dien_max_groups, 1200),
				max_groups=max(100, _safe_int(max_groups, 3000)),
			)
			return {"ok": True, "scene": scene, "tag": use_tag, "metrics": obj, "latency_ms": float((time.perf_counter() - t0) * 1000.0)}
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
			obj = app.state.ctx.get_runtime(use_tag).get_pipeline(scene).state.compute_validation_compare(
				max_groups=_safe_int(max_groups, 800),
				example_limit=max(0, _safe_int(example_limit, 5)),
				context_user_idx=(_safe_int(user_idx) if user_idx is not None else None),
			)
			return {"ok": True, "scene": scene, "tag": use_tag, "validation": obj, "latency_ms": float((time.perf_counter() - t0) * 1000.0)}
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
	parser.add_argument("--gbdt-topn", type=int, default=500)
	parser.add_argument("--recall-rank-cap", type=int, default=800)
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
