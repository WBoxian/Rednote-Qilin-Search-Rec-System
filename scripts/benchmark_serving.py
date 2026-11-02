"""Real HTTP benchmarks with explicit process-cold, cache-hit and inference scopes."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import time
from urllib.error import HTTPError
from urllib.parse import urlencode, urlparse
from urllib.request import urlopen
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REDIS_URL = "redis://127.0.0.1:16379/0"


def request(base, case, timeout):
    started = time.perf_counter()
    row = {"case": case}
    try:
        with urlopen(base.rstrip("/") + "/api/feed?" + urlencode(case), timeout=timeout) as response:
            row["http_status"] = response.status
            body = json.load(response)
        if not body.get("ok"):
            raise ValueError(str(body.get("detail", body)))
        feed = body.get("feed", {})
        row.update(ok=True, http_ok=True, server_ms=body.get("latency_ms"), cache_hit=feed.get("cache_hit"),
                   pipeline_ms=feed.get("pipeline_ms"), stage_ms=feed.get("stage_ms"),
                   stages=feed.get("stages"), items=len(feed.get("items", [])),
                   model_bundle_id=feed.get("model_bundle_id"), context_id=feed.get("context_id"),
                   degraded_reason=feed.get("degraded_reason"), cold_user=feed.get("cold_start"))
    except HTTPError as exc:
        row.update(ok=False, http_ok=False, http_status=exc.code, error=exc.read(1024).decode(errors="replace"))
    except Exception as exc:
        row.update(ok=False, http_ok=False, error=type(exc).__name__ + ": " + str(exc))
    row["client_ms"] = (time.perf_counter() - started) * 1000
    return row


def validate(row, phase, bundle_id):
    if not row["ok"]:
        return row
    errors = []
    if row.get("model_bundle_id") != bundle_id:
        errors.append("Unexpected model bundle")
    if row.get("degraded_reason"):
        errors.append("Degraded inference: " + str(row["degraded_reason"]))
    if not row.get("items"):
        errors.append("Empty response")
    expected_hit = phase == "cache_hit"
    row["phase_mismatch"] = row.get("cache_hit") is not expected_hit
    if row.get("cache_hit") is True:
        if row.get("pipeline_ms") != 0 or row.get("stage_ms") != {}:
            errors.append("Cache response contains current inference timing")
    elif not isinstance(row.get("pipeline_ms"), (int, float)) or row["pipeline_ms"] <= 0:
        errors.append("Fresh inference timing missing")
    if phase in ("cache_hit", "inference") and row.get("cold_user"):
        errors.append("Cold-user fallback is not the selected warm-user workload")
    if errors:
        row.update(ok=False, validation_error="; ".join(errors))
    return row


def summarize(rows, expected_cache_hit=None):
    successful = [row["client_ms"] for row in rows if row["ok"] and
                  (expected_cache_hit is None or row.get("cache_hit") is expected_cache_hit)]
    errors = sum(not row["ok"] for row in rows)
    return {"requests": len(rows), "latency_samples": len(successful), "errors": errors,
            "http_errors": sum(not row.get("http_ok", row["ok"]) for row in rows),
            "phase_mismatches": sum(bool(row.get("phase_mismatch")) for row in rows),
            "error_rate": errors / len(rows) if rows else None,
            "latency_ms": dict(zip(("p50", "p95", "p99", "max"),
                [float(x) for x in np.percentile(successful, [50, 95, 99, 100])])) if successful else None,
            "cache_hits": sum(row.get("cache_hit") is True for row in rows),
            "latency_population": "Successful HTTP responses passing phase/bundle/nondegradation checks"}


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def gpu_status():
    result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu",
                             "--format=csv,noheader"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else result.stderr.strip()


@contextmanager
def benchmark_redis(log_path):
    import redis
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", 16379)) == 0:
            raise ValueError("Refusing to replace an existing benchmark Redis listener")
    with log_path.open("w") as log:
        process = subprocess.Popen(["redis-server", "--bind", "127.0.0.1", "--port", "16379",
            "--save", "", "--appendonly", "no"], stdout=log, stderr=subprocess.STDOUT)
        try:
            client = redis.Redis.from_url(REDIS_URL, socket_connect_timeout=1, socket_timeout=1)
            deadline = time.monotonic() + 10
            while True:
                if process.poll() is not None:
                    raise RuntimeError("Benchmark Redis exited")
                try:
                    if client.ping():
                        break
                except redis.exceptions.ConnectionError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Benchmark Redis did not become ready")
                    time.sleep(.05)
            yield
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)


@contextmanager
def service(base_url, log_path, namespace):
    parsed = urlparse(base_url)
    if parsed.hostname != "127.0.0.1" or parsed.scheme != "http" or not parsed.port:
        raise ValueError("Managed server requires explicit http://127.0.0.1:PORT")
    with socket.socket() as sock:
        if sock.connect_ex((parsed.hostname, parsed.port)) == 0:
            raise ValueError("Refusing to replace an existing listener")
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src") + ":" + str(ROOT),
               PYTHONDONTWRITEBYTECODE="1", QILIN_ASYNC_PREWARM="0",
               QILIN_BLOCKING_PREWARM="0", QILIN_PREWARM_FEED="0",
               QILIN_TAG_SEARCH="hard", QILIN_TAG_REC="hard",
               QILIN_REDIS_URL=REDIS_URL, QILIN_REDIS_NAMESPACE=namespace,
               OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4",
               TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1")
    started = time.perf_counter()
    process = None
    try:
        with log_path.open("w") as log:
            process = subprocess.Popen([str(ROOT / ".venv/bin/python"),
                "src/backend/online/api/main.py", "--host", parsed.hostname,
                "--port", str(parsed.port), "--tag", "hard"], cwd=ROOT, env=env,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + 90
            while True:
                if process.poll() is not None:
                    raise RuntimeError("HTTP service exited; see " + str(log_path))
                try:
                    with urlopen(base_url + "/api/health", timeout=1) as response:
                        health = json.load(response)
                    if health.get("ok"):
                        if health.get("loaded_tags"):
                            raise RuntimeError("Unexpected model prewarm before first request")
                        break
                except (OSError, TimeoutError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("HTTP service did not become alive")
                    time.sleep(.1)
            yield {"pid": process.pid, "api_boot_ms": (time.perf_counter()-started)*1000,
                   "log": str(log_path), "redis_namespace": namespace}
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=15)
        # Only this run's namespaced exposure/request keys are disposable.
        import redis
        client = redis.Redis.from_url(REDIS_URL, socket_connect_timeout=2, socket_timeout=2)
        for keys in client.scan_iter(match=namespace + ":*", count=100):
            if not keys.startswith((namespace + ":").encode()):
                raise ValueError("Redis namespace escaped")
            client.delete(keys)


def run_phases(args, cases, output):
    scene = args.scene
    bundle_path = ROOT / f"outputs/deploy/{scene}/hard/active_bundle.json"
    bundle_bytes = bundle_path.read_bytes()
    bundle = json.loads(bundle_bytes)
    if len(cases) < max(args.samples + args.warmup_samples, args.cold_samples):
        raise ValueError("Need samples+warmup_samples distinct-user cases")
    identities = [(int(c["user_idx"]), str(c.get("query", ""))) for c in cases]
    if len(set(identities)) != len(identities):
        raise ValueError("Inference cases must be unique user/query pairs")
    run_id = uuid.uuid4().hex
    output.update(bundle_id=bundle["bundle_id"], bundle_manifest_sha256=hashlib.sha256(bundle_bytes).hexdigest(),
        gpu_before=gpu_status(), page_size=cases[0]["page_size"], warmup_samples=args.warmup_samples,
        environment={"python": platform.python_version(), "platform": platform.platform(),
                     "visible_cpus": os.cpu_count(), "cpu_threads": 4, "workers": 1,
                     "redis": "temporary local Redis 16379, dedicated per-process namespace, no persistence", "async_prewarm": False,
                     "cache_policy": "Unchanged production feed/stage TTLs"},
        phases={}, limitations=[
            "Local loopback HTTP; includes JSON response reading/decoding, excludes browser/image fetching.",
            "Process-cold only: OS filesystem/page cache is not flushed; API boot is recorded separately.",
            "Closed-loop fixed-concurrency load, not an open-loop production capacity/SLO claim.",
            "GPU/host are shared; utilization is recorded, not assumed exclusive.",
            "Cold P95/P99 use few restarts and are descriptive sample percentiles.",
            "Expired caches are counted as phase mismatches, not HTTP errors; hit percentiles use actual hits only.",
            "Recommendation bundle is recall_only; its latency does not include disabled learned rankers."])
    cold = []
    output["cold_requests"] = cold
    for index in range(args.cold_samples if args.phase == "all" else 0):
        log = args.output.with_name(args.output.stem + f".cold{index}.server.log")
        with service(args.base_url, log, f"qilin:bench:{run_id}:cold{index}") as meta:
            row = validate(request(args.base_url, cases[index], args.timeout), "cold", bundle["bundle_id"])
            row.update(meta)
            row["api_boot_plus_first_http_ms"] = row["api_boot_ms"] + row["client_ms"]
            cold.append(row)
        output["phases"]["cold"] = summarize(cold)
        write(args.output, output)
        print(json.dumps({"scene": scene, "phase": "cold", "completed": len(cold),
                          "ok": row["ok"], "client_ms": row["client_ms"]}), flush=True)
    log = args.output.with_name(args.output.stem + ".warm.server.log")
    with service(args.base_url, log, f"qilin:bench:{run_id}:warm") as meta:
        output["warm_service"] = meta
        warmups = [validate(request(args.base_url, case, args.timeout), "cold", bundle["bundle_id"])
                   for case in cases[:args.warmup_samples]]
        output["warmups_excluded"] = warmups
        write(args.output, output)
        if not all(row["ok"] for row in warmups):
            raise RuntimeError("Warmup failed; measured phases not started")
        for phase, count in (("cache_hit", args.cache_samples), ("inference", args.samples)):
            selected = [cases[i % 8] for i in range(count)] if phase == "cache_hit" else cases[args.warmup_samples:args.warmup_samples+count]
            started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                rows = list(pool.map(lambda case: validate(request(args.base_url, case, args.timeout),
                                                          phase, bundle["bundle_id"]), selected))
            output[phase + "_requests"] = rows
            summary = summarize(rows, expected_cache_hit=phase == "cache_hit")
            summary.update(concurrency=args.concurrency, wall_seconds=time.perf_counter()-started)
            summary["completed_requests_per_second"] = len(rows) / summary["wall_seconds"]
            output["phases"][phase] = summary
            write(args.output, output)
            print(json.dumps({"scene": scene, "phase": phase, **summary}), flush=True)
    if bundle_path.read_bytes() != bundle_bytes:
        raise ValueError("Active model changed during benchmark")
    output["gpu_after"] = gpu_status()
    failed = any(p["errors"] for p in output["phases"].values()) or output["phases"]["inference"]["phase_mismatches"] > 0
    output["status"] = "FAIL" if failed else ("PASS_WITH_CACHE_MISSES" if output["phases"]["cache_hit"]["phase_mismatches"] else "PASS")
    write(args.output, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18082")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--scene", choices=("search", "rec"), required=True)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("mixed", "all", "warm"), default="mixed")
    parser.add_argument("--cold-samples", type=int, default=10)
    parser.add_argument("--cache-samples", type=int, default=200)
    parser.add_argument("--warmup-samples", type=int, default=32)
    args = parser.parse_args()
    if min(args.samples, args.concurrency, args.cold_samples, args.cache_samples, args.warmup_samples) < 1 or args.timeout <= 0:
        parser.error("Sample counts, concurrency and timeout must be positive")
    if args.output.exists():
        raise FileExistsError("Benchmark outputs are immutable: " + str(args.output))
    cases = [case for case in json.loads(args.cases.read_text()) if case.get("scene") == args.scene]
    if not cases:
        parser.error("cases cannot be empty")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = {"scene": args.scene, "git_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "benchmark_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(), "base_url": args.base_url}
    if args.phase in ("all", "warm"):
        try:
            with benchmark_redis(args.output.with_suffix(".redis.log")):
                run_phases(args, cases, output)
        except Exception as exc:
            output.update(status="FAIL", error=type(exc).__name__ + ": " + str(exc))
            write(args.output, output)
            raise
        if output["status"] not in ("PASS", "PASS_WITH_CACHE_MISSES"):
            raise SystemExit(1)
    else:
        first = [request(args.base_url, case, args.timeout) for case in cases]
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            warm = list(pool.map(lambda i: request(args.base_url, cases[i % len(cases)], args.timeout),
                                 range(args.samples)))
        output.update(concurrency=args.concurrency, first_observations=first,
                      warm_summary=summarize(warm), warm_requests=warm,
                      limitations=["Mixed cache workload; first observation is not necessarily process-cold."])
        write(args.output, output)
        print(json.dumps(output["warm_summary"]), flush=True)
        if any(not row["ok"] for row in first+warm):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
