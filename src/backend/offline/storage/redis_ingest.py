"""离线特征上传 Redis（backend）。"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import duckdb
import pandas as pd


REDIS_PIPELINE_FLUSH_EVERY = 5000
DUCKDB_FETCH_ROWS = 5000


def _require_redis_client():
    try:
        import redis  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("redis package is required. install with: uv add redis") from e
    return redis


def _normalize_json_value(value):
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # noqa: BLE001
            pass
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray)):
        try:
            value = value.tolist()
        except Exception:  # noqa: BLE001
            pass
    if isinstance(value, tuple):
        return [_normalize_json_value(v) for v in value]
    if isinstance(value, list):
        return [_normalize_json_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _normalize_json_value(v) for k, v in value.items()}
    try:
        if pd.isna(value):
            return None
    except Exception:  # noqa: BLE001
        pass
    return value


def _flush_pipeline(pipe, pending_cmds: int) -> int:
    if pending_cmds <= 0:
        return 0
    pipe.execute()
    return 0


def _iter_query_rows(con: duckdb.DuckDBPyConnection, sql: str, params: list[object] | None = None):
    cur = con.execute(sql, params or [])
    columns = [d[0] for d in (cur.description or [])]
    while True:
        rows = cur.fetchmany(DUCKDB_FETCH_ROWS)
        if not rows:
            break
        yield columns, rows


def ingest_user_features_to_redis(base_dir: Path, scene: str, redis_url: str) -> None:
    redis = _require_redis_client()
    client = redis.Redis.from_url(redis_url, decode_responses=True)

    user_feat_path = base_dir / "datasets" / "user_feat" / "train-00000-of-00001.parquet"
    req_path = base_dir / "datasets" / (
        "search_test" if scene == "search" else "recommendation_test"
    ) / "train-00000-of-00001.parquet"
    group_key = "search_idx" if scene == "search" else "request_idx"

    pipe = client.pipeline(transaction=False)
    pending_cmds = 0
    req_user_count = 0

    con = duckdb.connect(database=":memory:")
    try:
        user_sql = "SELECT * FROM read_parquet(?)"
        profile_rows = 0
        for columns, rows in _iter_query_rows(con, user_sql, [str(user_feat_path)]):
            for values in rows:
                row = dict(zip(columns, values))
                uid = int(row["user_idx"])
                key = f"qilin:user:{uid}:profile"
                payload = {
                    k: json.dumps(_normalize_json_value(v), ensure_ascii=False)
                    for k, v in row.items()
                }
                pipe.hset(key, mapping=payload)
                pending_cmds += 1
                profile_rows += 1
                if pending_cmds >= REDIS_PIPELINE_FLUSH_EVERY:
                    pending_cmds = _flush_pipeline(pipe, pending_cmds)

        req_sql = f"""
        SELECT user_idx, {group_key}
        FROM read_parquet(?)
        ORDER BY user_idx ASC, session_idx ASC, {group_key} ASC
        """
        current_uid: int | None = None
        current_reqs: list[int] = []
        for _, rows in _iter_query_rows(con, req_sql, [str(req_path)]):
            for uid_raw, req_raw in rows:
                uid = int(uid_raw)
                rid = int(req_raw)
                if current_uid is None:
                    current_uid = uid
                if uid != current_uid:
                    key = f"qilin:user:{current_uid}:{scene}:requests"
                    pipe.set(key, json.dumps(current_reqs, ensure_ascii=False))
                    pending_cmds += 1
                    req_user_count += 1
                    if pending_cmds >= REDIS_PIPELINE_FLUSH_EVERY:
                        pending_cmds = _flush_pipeline(pipe, pending_cmds)
                    current_uid = uid
                    current_reqs = []
                current_reqs.append(rid)
        if current_uid is not None:
            key = f"qilin:user:{current_uid}:{scene}:requests"
            pipe.set(key, json.dumps(current_reqs, ensure_ascii=False))
            pending_cmds += 1
            req_user_count += 1
            if pending_cmds >= REDIS_PIPELINE_FLUSH_EVERY:
                pending_cmds = _flush_pipeline(pipe, pending_cmds)

        # 初始化用户历史点击序列（用于在线 DIEN 实时序列特征）
        feat_path = base_dir / "features" / f"{scene}_test_features.parquet"
        if feat_path.exists():
            hist_sql = """
            SELECT user_idx, recent_clicked_note_idxs
            FROM (
                SELECT
                    user_idx,
                    recent_clicked_note_idxs,
                    row_number() OVER (PARTITION BY user_idx ORDER BY session_idx DESC) AS rn
                FROM read_parquet(?)
            ) t
            WHERE rn = 1
            """
            for _, rows in _iter_query_rows(con, hist_sql, [str(feat_path)]):
                for uid_raw, raw in rows:
                    uid = int(uid_raw)
                    norm = _normalize_json_value(raw)
                    if isinstance(norm, list):
                        seq = [int(x) for x in norm[:50]]
                    else:
                        seq = []
                    key = f"qilin:user:{uid}:{scene}:history_notes"
                    pipe.delete(key)
                    pending_cmds += 1
                    if seq:
                        pipe.rpush(key, *seq)
                        pending_cmds += 1
                    if pending_cmds >= REDIS_PIPELINE_FLUSH_EVERY:
                        pending_cmds = _flush_pipeline(pipe, pending_cmds)

        # 编码字典写入 Redis，便于在线服务统一读取/核对
        cat_dir = base_dir / "features" / "vocab_dict"
        if cat_dir.exists():
            for p in sorted(cat_dir.glob("*.pkl")):
                with open(p, "rb") as f:
                    obj = pickle.load(f)
                key = f"qilin:feature:vocab_dict:{p.stem}"
                pipe.set(key, json.dumps(_normalize_json_value(obj), ensure_ascii=False))
                pending_cmds += 1
                if pending_cmds >= REDIS_PIPELINE_FLUSH_EVERY:
                    pending_cmds = _flush_pipeline(pipe, pending_cmds)
    finally:
        con.close()

    pending_cmds = _flush_pipeline(pipe, pending_cmds)
    print(
        f"[Storage/RedisIngest] done. scene={scene}, users={req_user_count}, "
        f"profiles={profile_rows}, redis={redis_url}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=["search", "rec"], required=True)
    parser.add_argument("--redis-url", type=str, required=True, help="e.g. redis://127.0.0.1:6379/0")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[4]
    ingest_user_features_to_redis(base_dir=base_dir, scene=args.scene, redis_url=args.redis_url)


if __name__ == "__main__":
    main()
