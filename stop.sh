#!/usr/bin/env bash
set -euo pipefail

# Qilin 停止脚本（Linux/macOS）
# - 停止后端 FastAPI
# - 停止前端 Vite 开发服务
# - 如 Redis 由 start.sh 拉起，则一并停止对应容器

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
RUN_DIR="${PROJECT_ROOT}/.qilin/run"
BACKEND_PID_FILE="${RUN_DIR}/backend.pid"
FRONTEND_PID_FILE="${RUN_DIR}/frontend.pid"
REDIS_MARKER_FILE="${RUN_DIR}/redis_started_by_script"
REDIS_CONTAINER_NAME="qilin-redis-local"

cd "${PROJECT_ROOT}"

is_pid_running() {
  local pid="$1"
  [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null
}

stop_pid_file() {
  local name="$1"
  local pid_file="$2"
  if [[ ! -f "${pid_file}" ]]; then
    return 0
  fi
  local pid
  pid="$(cat "${pid_file}" 2>/dev/null || true)"
  if is_pid_running "${pid}"; then
    echo "[Qilin] Stopping ${name} pid=${pid}"
    kill "${pid}" 2>/dev/null || true
    sleep 1
    if is_pid_running "${pid}"; then
      kill -9 "${pid}" 2>/dev/null || true
    fi
  fi
  rm -f "${pid_file}"
}

stop_port() {
  local port="$1"
  if ! command -v lsof >/dev/null 2>&1; then
    return 0
  fi
  local pids
  pids=$(lsof -ti tcp:"${port}" -sTCP:LISTEN || true)
  if [[ -z "${pids}" ]]; then
    return 0
  fi
  echo "[Qilin] Stopping processes on port ${port}: ${pids}"
  for pid in ${pids}; do
    kill "${pid}" 2>/dev/null || true
  done
  sleep 1
  pids=$(lsof -ti tcp:"${port}" -sTCP:LISTEN || true)
  if [[ -n "${pids}" ]]; then
    for pid in ${pids}; do
      kill -9 "${pid}" 2>/dev/null || true
    done
  fi
}

stop_pid_file "backend" "${BACKEND_PID_FILE}"
stop_pid_file "frontend" "${FRONTEND_PID_FILE}"
stop_port 18080
stop_port 5173

if [[ -f "${REDIS_MARKER_FILE}" ]]; then
  if command -v docker >/dev/null 2>&1; then
    if docker ps --format '{{.Names}}' | grep -qx "${REDIS_CONTAINER_NAME}"; then
      echo "[Qilin] Stopping Redis container: ${REDIS_CONTAINER_NAME}"
      docker stop "${REDIS_CONTAINER_NAME}" >/dev/null 2>&1 || true
    fi
  fi
  rm -f "${REDIS_MARKER_FILE}"
fi

echo "[Qilin] Stopped."
