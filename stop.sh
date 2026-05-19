#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
REDIS_CONTAINER_NAME="qilin-redis-local"
RUN_DIR="${PROJECT_ROOT}/.qilin/run"
BACKEND_PID_FILE="${RUN_DIR}/backend.pid"

cd "${PROJECT_ROOT}"

stop_pattern() {
  local pattern="$1"
  local pids
  pids="$(pgrep -f "${pattern}" || true)"
  if [[ -z "${pids}" ]]; then
    return 0
  fi
  echo "[Qilin] Stopping pattern: ${pattern}"
  kill ${pids} 2>/dev/null || true
  sleep 1
  pids="$(pgrep -f "${pattern}" || true)"
  if [[ -n "${pids}" ]]; then
    kill -9 ${pids} 2>/dev/null || true
  fi
}

stop_pidfile() {
  local pidfile="$1"
  if [[ ! -f "${pidfile}" ]]; then
    return 0
  fi
  local pid
  pid="$(cat "${pidfile}" 2>/dev/null || true)"
  if [[ -z "${pid}" ]]; then
    rm -f "${pidfile}"
    return 0
  fi
  echo "[Qilin] Stopping pid from ${pidfile}: ${pid}"
  kill "${pid}" 2>/dev/null || true
  sleep 1
  if kill -0 "${pid}" 2>/dev/null; then
    kill -9 "${pid}" 2>/dev/null || true
  fi
  rm -f "${pidfile}"
}

stop_port() {
  local port="$1"
  if command -v fuser >/dev/null 2>&1; then
    fuser -k "${port}/tcp" >/dev/null 2>&1 || true
  fi
  if command -v lsof >/dev/null 2>&1; then
    local pids
    pids="$(lsof -ti "tcp:${port}" -sTCP:LISTEN 2>/dev/null || true)"
    if [[ -n "${pids}" ]]; then
      echo "[Qilin] Stopping port ${port}: ${pids}"
      kill ${pids} 2>/dev/null || true
      sleep 1
      pids="$(lsof -ti "tcp:${port}" -sTCP:LISTEN 2>/dev/null || true)"
      if [[ -n "${pids}" ]]; then
        kill -9 ${pids} 2>/dev/null || true
      fi
    fi
  fi
}

stop_pidfile "${BACKEND_PID_FILE}"
stop_pattern "uv run python src/backend/online/api/main.py"
stop_pattern "python src/backend/online/api/main.py"
stop_pattern "uvicorn.*18080"
stop_port 18080
stop_port 6379

if command -v docker >/dev/null 2>&1 && docker ps -a --format '{{.Names}}' | grep -qx "${REDIS_CONTAINER_NAME}"; then
  echo "[Qilin] Removing Redis container: ${REDIS_CONTAINER_NAME}"
  docker rm -f "${REDIS_CONTAINER_NAME}" >/dev/null 2>&1 || true
fi

rm -rf "${PROJECT_ROOT}/.qilin/run"

echo "[Qilin] Stopped."
