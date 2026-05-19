"""在线实时特征与行为缓存（Redis）。"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any


def calc_interaction_score(
    click: int = 0,
    like: int = 0,
    collect: int = 0,
    comment: int = 0,
    share: int = 0,
    page_time: float = 0.0,
) -> float:
    c_click = max(0, int(click))
    base_signal = (
        1.0 * c_click
        + 2.0 * max(0, int(like))
        + 3.0 * max(0, int(collect))
        + 3.0 * max(0, int(comment))
        + 3.0 * max(0, int(share))
        + 0.2 * float(math.log1p(max(0.0, float(page_time))))
    )
    return float(base_signal) if float(base_signal) > 0 else 0.0


@dataclass
class RealtimeCache:
    client: Any
    history_max_len: int = 50
    behavior_max_len: int = 60
    exposure_max_len: int = 500
    dedup_window_sec: int = 2

    def _collapse_behaviors(self, rows: list[dict[str, Any]], max_len: int) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for obj in rows:
            key = (
                f"{obj.get('scene')}|{obj.get('request_id')}|{obj.get('note_idx')}|{obj.get('query', '')}"
            )
            prev = merged.get(key)
            if prev is None:
                merged[key] = obj
                continue
            prev_ts = int(prev.get('ts', 0) or 0)
            cur_ts = int(obj.get('ts', 0) or 0)
            if cur_ts >= prev_ts:
                nxt = dict(prev)
                nxt.update(obj)
                merged[key] = nxt
        out = list(merged.values())
        out.sort(key=lambda x: (int(x.get('ts', 0) or 0), float(x.get('interaction_score', 0) or 0)), reverse=True)
        return out[: int(max_len)]

    def get_user_profile(self, user_idx: int) -> dict[str, Any] | None:
        key = f"qilin:user:{int(user_idx)}:profile"
        obj = self.client.hgetall(key)
        if not obj:
            return None
        out: dict[str, Any] = {}
        for k, v in obj.items():
            try:
                out[k] = json.loads(v)
            except Exception:
                out[k] = v
        return out

    def get_user_requests(self, user_idx: int, scene: str) -> list[int] | None:
        key = f"qilin:user:{int(user_idx)}:{scene}:requests"
        raw = self.client.get(key)
        if not raw:
            return None
        try:
            arr = json.loads(raw)
            return [int(x) for x in arr]
        except Exception:
            return None

    def get_user_history_notes(self, user_idx: int, scene: str, max_len: int = 20) -> list[int]:
        key = f"qilin:user:{int(user_idx)}:{scene}:history_notes"
        arr = self.client.lrange(key, 0, max(0, int(max_len) - 1))
        return [int(x) for x in arr if str(x).strip()]

    def get_recent_behaviors(self, user_idx: int, scene: str, max_len: int = 20) -> list[dict[str, Any]]:
        key = f"qilin:user:{int(user_idx)}:{scene}:behaviors"
        arr = self.client.lrange(key, 0, max(0, int(max_len) * 3))
        out: list[dict[str, Any]] = []
        for x in arr:
            try:
                obj = json.loads(x)
            except Exception:
                continue
            if isinstance(obj, dict):
                if not obj.get("scene"):
                    obj["scene"] = scene
                out.append(obj)
        return self._collapse_behaviors(out, max_len=max_len)

    def get_recent_exposed_notes(self, user_idx: int, scene: str, max_len: int = 200) -> list[int]:
        key = f"qilin:user:{int(user_idx)}:{scene}:exposed_notes"
        arr = self.client.lrange(key, 0, max(0, int(max_len) - 1))
        out: list[int] = []
        for x in arr:
            try:
                nid = int(x)
            except Exception:
                continue
            if nid >= 0:
                out.append(nid)
        return out

    def next_runtime_request_id(self, scene: str, start_from: int) -> int:
        key = f"qilin:{scene}:runtime_request_id"
        try:
            if not self.client.exists(key):
                self.client.set(key, int(start_from))
            return int(self.client.incr(key))
        except Exception:
            return int(start_from) + 1

    def record_exposed_notes(self, user_idx: int, scene: str, note_ids: list[int]) -> None:
        uniq_ids: list[int] = []
        seen: set[int] = set()
        for x in note_ids:
            try:
                nid = int(x)
            except Exception:
                continue
            if nid < 0 or nid in seen:
                continue
            seen.add(nid)
            uniq_ids.append(nid)
        if not uniq_ids:
            return
        key = f"qilin:user:{int(user_idx)}:{scene}:exposed_notes"
        p = self.client.pipeline(transaction=False)
        for nid in uniq_ids:
            p.lrem(key, 0, int(nid))
        for nid in uniq_ids:
            p.lpush(key, int(nid))
        p.ltrim(key, 0, max(0, int(self.exposure_max_len) - 1))
        p.execute()

    def get_recent_behaviors_all(self, user_idx: int, max_len: int = 20) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        for scene in ["search", "rec"]:
            key = f"qilin:user:{int(user_idx)}:{scene}:behaviors"
            arr = self.client.lrange(key, 0, max(0, int(max_len) * 4))
            for x in arr:
                try:
                    obj = json.loads(x)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    if not obj.get("scene"):
                        obj["scene"] = scene
                    merged.append(obj)
        return self._collapse_behaviors(merged, max_len=max_len)

    def _match_behavior_event(
        self,
        obj: dict[str, Any],
        scene: str,
        note_idx: int,
        ts: int | None = None,
        request_id: int | None = None,
    ) -> bool:
        if str(obj.get("scene") or scene) != str(scene):
            return False
        if int(obj.get("note_idx", -1) or -1) != int(note_idx):
            return False
        if request_id is not None and int(obj.get("request_id", -1) or -1) != int(request_id):
            return False
        if ts is not None and int(obj.get("ts", -1) or -1) != int(ts):
            return False
        return True

    def append_behavior_event(
        self,
        user_idx: int,
        scene: str,
        action: str,
        note_idx: int,
        request_id: int | None = None,
        query: str = "",
        click: int = 0,
        like: int = 0,
        collect: int = 0,
        comment: int = 0,
        share: int = 0,
        page_time: float = 0.0,
    ) -> bool:
        action_name = str(action or "click").strip().lower() or "click"
        dedup_key = (
            f"qilin:dedup:{int(user_idx)}:{scene}:{action_name}:"
            f"{int(note_idx)}:{int(request_id) if request_id is not None else -1}:{str(query or '')[:80]}:"
            f"{int(click)}:{int(like)}:{int(collect)}:{int(comment)}:{int(share)}:{int(float(page_time) * 10)}"
        )
        accepted = self.client.set(
            dedup_key,
            "1",
            nx=True,
            ex=max(1, int(self.dedup_window_sec)),
        )
        if not accepted:
            return False

        key = f"qilin:user:{int(user_idx)}:{scene}:behaviors"
        event = {
            "ts": int(time.time()),
            "scene": scene,
            "action": action_name,
            "note_idx": int(note_idx),
            "request_id": (int(request_id) if request_id is not None else None),
            "query": str(query or ""),
            "click": int(click),
            "like": int(like),
            "collect": int(collect),
            "comment": int(comment),
            "share": int(share),
            "page_time": float(max(0.0, float(page_time))),
            "interaction_score": calc_interaction_score(
                click=int(click),
                like=int(like),
                collect=int(collect),
                comment=int(comment),
                share=int(share),
                page_time=float(max(0.0, float(page_time))),
            ),
        }
        arr = self.client.lrange(key, 0, max(0, int(self.behavior_max_len) * 4))
        keep: list[str] = []
        for raw in arr:
            try:
                obj = json.loads(raw)
            except Exception:
                keep.append(raw)
                continue
            if not isinstance(obj, dict):
                keep.append(raw)
                continue
            same_scene = str(obj.get("scene") or scene) == str(scene)
            same_note = int(obj.get("note_idx", -1) or -1) == int(note_idx)
            same_request = int(obj.get("request_id", -1) or -1) == int(request_id if request_id is not None else -1)
            same_query = str(obj.get("query") or "") == str(query or "")
            if same_scene and same_note and same_request and same_query:
                continue
            keep.append(json.dumps(obj, ensure_ascii=False))
        p = self.client.pipeline(transaction=False)
        p.delete(key)
        p.rpush(key, json.dumps(event, ensure_ascii=False), *keep[: max(0, int(self.behavior_max_len) - 1)])
        p.ltrim(key, 0, max(0, int(self.behavior_max_len) - 1))
        p.execute()
        return True

    def record_behavior(
        self,
        user_idx: int,
        scene: str,
        action: str,
        note_idx: int,
        request_id: int | None = None,
        query: str = "",
        update_history: bool = False,
        click: int = 0,
        like: int = 0,
        collect: int = 0,
        comment: int = 0,
        share: int = 0,
        page_time: float = 0.0,
    ) -> bool:
        accepted = self.append_behavior_event(
            user_idx=int(user_idx),
            scene=scene,
            action=action,
            note_idx=int(note_idx),
            request_id=request_id,
            query=query,
            click=int(click),
            like=int(like),
            collect=int(collect),
            comment=int(comment),
            share=int(share),
            page_time=float(page_time),
        )
        if not accepted:
            return False
        if update_history:
            key = f"qilin:user:{int(user_idx)}:{scene}:history_notes"
            p = self.client.pipeline(transaction=False)
            p.lrem(key, 0, int(note_idx))
            p.lpush(key, int(note_idx))
            p.ltrim(key, 0, max(0, int(self.history_max_len) - 1))
            p.execute()
        return True

    def delete_behavior(
        self,
        user_idx: int,
        scene: str,
        note_idx: int,
        ts: int | None = None,
        request_id: int | None = None,
    ) -> bool:
        key = f"qilin:user:{int(user_idx)}:{scene}:behaviors"
        arr = self.client.lrange(key, 0, max(0, int(self.behavior_max_len) * 4))
        if not arr:
            return False
        keep: list[str] = []
        removed = False
        remaining_same_note = False
        for raw in arr:
            try:
                obj = json.loads(raw)
            except Exception:
                keep.append(raw)
                continue
            if not isinstance(obj, dict):
                keep.append(raw)
                continue
            if not removed and self._match_behavior_event(obj, scene=scene, note_idx=note_idx, ts=ts, request_id=request_id):
                removed = True
                continue
            if self._match_behavior_event(obj, scene=scene, note_idx=note_idx, request_id=request_id):
                remaining_same_note = True
            keep.append(json.dumps(obj, ensure_ascii=False))
        if not removed:
            return False

        p = self.client.pipeline(transaction=False)
        p.delete(key)
        if keep:
            p.rpush(key, *keep)
            p.ltrim(key, -max(1, int(self.behavior_max_len)), -1)
        if not remaining_same_note:
            hist_key = f"qilin:user:{int(user_idx)}:{scene}:history_notes"
            p.lrem(hist_key, 0, int(note_idx))
        p.execute()
        return True

    def record_click(
        self,
        user_idx: int,
        scene: str,
        note_idx: int,
        request_id: int | None = None,
        query: str = "",
    ) -> bool:
        return self.record_behavior(
            user_idx=int(user_idx),
            scene=scene,
            action="click",
            note_idx=int(note_idx),
            request_id=request_id,
            query=query,
            update_history=True,
            click=1,
        )

    def record_view(
        self,
        user_idx: int,
        scene: str,
        note_idx: int,
        request_id: int | None = None,
        query: str = "",
    ) -> bool:
        return self.record_behavior(
            user_idx=int(user_idx),
            scene=scene,
            action="view",
            note_idx=int(note_idx),
            request_id=request_id,
            query=query,
            update_history=False,
            click=0,
        )

    def record_engage(
        self,
        user_idx: int,
        scene: str,
        note_idx: int,
        request_id: int | None = None,
        query: str = "",
        like: int = 0,
        collect: int = 0,
        comment: int = 0,
        share: int = 0,
        page_time: float = 0.0,
    ) -> bool:
        has_feedback = int(like) > 0 or int(collect) > 0 or int(comment) > 0 or int(share) > 0 or float(page_time) > 0.0
        return self.record_behavior(
            user_idx=int(user_idx),
            scene=scene,
            action="engage",
            note_idx=int(note_idx),
            request_id=request_id,
            query=query,
            update_history=bool(has_feedback),
            click=0,
            like=max(0, int(like)),
            collect=max(0, int(collect)),
            comment=max(0, int(comment)),
            share=max(0, int(share)),
            page_time=max(0.0, float(page_time)),
        )


def build_realtime_cache() -> RealtimeCache | None:
    redis_url = os.getenv("QILIN_REDIS_URL", "").strip()
    if not redis_url:
        return None
    try:
        import redis  # type: ignore

        client = redis.Redis.from_url(redis_url, decode_responses=True)
        client.ping()
        return RealtimeCache(
            client=client,
            history_max_len=int(os.getenv("QILIN_HISTORY_MAX_LEN", "50")),
            behavior_max_len=int(os.getenv("QILIN_BEHAVIOR_MAX_LEN", "60")),
            dedup_window_sec=int(os.getenv("QILIN_BEHAVIOR_DEDUP_SEC", "2")),
        )
    except Exception:
        return None


__all__ = ["RealtimeCache", "build_realtime_cache"]
