#!/usr/bin/env bash
set -euo pipefail

# Qilin 后台启动（Linux/macOS）
# - 后台启动后端 FastAPI（默认 hard）
# - 后台启动前端 Vite 开发服务
# - 自动检测/拉起本地 Redis（用于在线实时特征缓存）

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
RUN_DIR="${PROJECT_ROOT}/.qilin/run"
BACKEND_PID_FILE="${RUN_DIR}/backend.pid"
FRONTEND_PID_FILE="${RUN_DIR}/frontend.pid"
REDIS_MARKER_FILE="${RUN_DIR}/redis_started_by_script"

cd "${PROJECT_ROOT}"

mkdir -p "${RUN_DIR}"

REDIS_URL="${QILIN_REDIS_URL:-redis://127.0.0.1:6379/0}"
REDIS_CONTAINER_NAME="qilin-redis-local"
REDIS_STARTED_BY_SCRIPT=0

is_pid_running() {
  local pid="$1"
  [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null
}

kill_pid_file() {
  local pid_file="$1"
  if [[ ! -f "${pid_file}" ]]; then
    return 0
  fi
  local pid
  pid="$(cat "${pid_file}" 2>/dev/null || true)"
  if is_pid_running "${pid}"; then
    kill "${pid}" 2>/dev/null || true
    sleep 1
    if is_pid_running "${pid}"; then
      kill -9 "${pid}" 2>/dev/null || true
    fi
  fi
  rm -f "${pid_file}"
}

free_port_if_needed() {
  local port="$1"
  if ! command -v lsof >/dev/null 2>&1; then
    return 0
  fi
  local pids
  pids=$(lsof -ti tcp:"${port}" -sTCP:LISTEN || true)
  if [[ -z "${pids}" ]]; then
    return 0
  fi
  echo "[Qilin] Port ${port} is in use, stopping process(es): ${pids}"
  for pid in ${pids}; do
    kill "${pid}" 2>/dev/null || true
  done
  sleep 1
}

start_redis_if_needed() {
  if command -v redis-cli >/dev/null 2>&1; then
    if redis-cli -u "${REDIS_URL}" ping >/dev/null 2>&1; then
      echo "[Qilin] Reusing existing Redis at ${REDIS_URL}"
      return 0
    fi
  fi

  if command -v docker >/dev/null 2>&1; then
    if docker ps --format '{{.Names}}' | grep -qx "${REDIS_CONTAINER_NAME}"; then
      echo "[Qilin] Reusing Docker Redis container: ${REDIS_CONTAINER_NAME}"
      return 0
    fi
    if docker ps -a --format '{{.Names}}' | grep -qx "${REDIS_CONTAINER_NAME}"; then
      echo "[Qilin] Starting existing Redis container: ${REDIS_CONTAINER_NAME}"
      docker start "${REDIS_CONTAINER_NAME}" >/dev/null
      REDIS_STARTED_BY_SCRIPT=1
      printf '1\n' > "${REDIS_MARKER_FILE}"
      return 0
    fi

    echo "[Qilin] Starting Redis container: ${REDIS_CONTAINER_NAME}"
    docker run -d --name "${REDIS_CONTAINER_NAME}" -p 6379:6379 redis:7-alpine >/dev/null
    REDIS_STARTED_BY_SCRIPT=1
    printf '1\n' > "${REDIS_MARKER_FILE}"
    return 0
  fi

  echo "[Qilin][Warn] Redis not found. Install redis or docker for realtime cache support."
  return 0
}

rm -f "${REDIS_MARKER_FILE}"

kill_pid_file "${BACKEND_PID_FILE}"
kill_pid_file "${FRONTEND_PID_FILE}"
free_port_if_needed 18080
free_port_if_needed 5173

start_redis_if_needed

echo "[Qilin] Starting backend on http://127.0.0.1:18080 ..."
nohup env QILIN_REDIS_URL="${REDIS_URL}" uv run python src/backend/online/api/main.py --host 0.0.0.0 --port 18080 --tag hard >/dev/null 2>&1 &
BACKEND_PID=$!
printf '%s\n' "${BACKEND_PID}" > "${BACKEND_PID_FILE}"

echo "[Qilin] Starting frontend on http://127.0.0.1:5173 ..."
(
  cd src/frontend
  npm install --silent >/dev/null 2>&1
  exec nohup ./node_modules/.bin/vite --host 0.0.0.0 --port 5173 >/dev/null 2>&1
) &
FRONTEND_PID=$!
printf '%s\n' "${FRONTEND_PID}" > "${FRONTEND_PID_FILE}"

echo ""
echo "[Qilin] Started."
echo "Backend:  http://127.0.0.1:18080/api/health"
echo "Frontend: http://127.0.0.1:5173"
echo "Redis:    ${REDIS_URL}"
echo "PID Dir:  ${RUN_DIR}"
echo ""
