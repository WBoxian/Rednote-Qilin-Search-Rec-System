#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
LOG_DIR="${PROJECT_ROOT}/.qilin/log"
BACKEND_LOG="${LOG_DIR}/backend.log"
FRONTEND_LOG="${LOG_DIR}/frontend.log"
RUN_DIR="${PROJECT_ROOT}/.qilin/run"
BACKEND_PID_FILE="${RUN_DIR}/backend.pid"
REDIS_CONTAINER_NAME="qilin-redis-local"
REDIS_URL="${QILIN_REDIS_URL:-redis://127.0.0.1:6379/0}"
BIND_HOST="0.0.0.0"
BIND_BACKEND_PORT=18080
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/qilin-uv-cache}"
NPM_CONFIG_CACHE="${NPM_CONFIG_CACHE:-/tmp/qilin-npm-cache}"

cd "${PROJECT_ROOT}"
mkdir -p "${LOG_DIR}"
mkdir -p "${RUN_DIR}"

detect_wsl_ip() {
  hostname -I 2>/dev/null | awk '{ for (i = 1; i <= NF; i++) if ($i !~ /^127\./) { print $i; exit } }'
}

log_prefix() {
  while IFS= read -r line || [[ -n "${line}" ]]; do
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${line}"
  done
}

export -f log_prefix

launch_logged() {
  local logfile="$1"
  shift
  local cmd="$*"
  setsid bash -lc "set -o pipefail; ${cmd} 2>&1 | log_prefix >> '${logfile}'" </dev/null >/dev/null 2>&1 &
  echo $!
}

wait_for_redis_ready() {
  if ! command -v redis-cli >/dev/null 2>&1; then
    return 0
  fi
  local max_wait="${1:-8}"
  local waited=0
  while (( waited < max_wait )); do
    if redis-cli -u "${REDIS_URL}" ping >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
    waited=$((waited + 1))
  done
  return 1
}

start_redis_if_needed() {
  if command -v redis-cli >/dev/null 2>&1; then
    if redis-cli -u "${REDIS_URL}" ping >/dev/null 2>&1; then
      echo "[Qilin] Redis already reachable at ${REDIS_URL}; skip Docker container startup"
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
      if docker start "${REDIS_CONTAINER_NAME}" >/dev/null 2>&1; then
        wait_for_redis_ready 8 >/dev/null 2>&1 || true
        return 0
      fi
      echo "[Qilin][Warn] Failed to start Redis container, continuing without Docker Redis."
      return 0
    fi

    echo "[Qilin] Starting Redis container: ${REDIS_CONTAINER_NAME}"
    if docker run -d --name "${REDIS_CONTAINER_NAME}" -p 6379:6379 redis:7-alpine >/dev/null 2>&1; then
      wait_for_redis_ready 8 >/dev/null 2>&1 || true
      return 0
    fi
    echo "[Qilin][Warn] Failed to create Redis container, continuing without Docker Redis."
    return 0
  fi

  echo "[Qilin][Warn] Redis not found. Install redis or docker for realtime cache support."
  return 0
}

start_redis_if_needed

mkdir -p "${UV_CACHE_DIR}" "${NPM_CONFIG_CACHE}"

echo "[Qilin] Building frontend ..."
if [[ ! -x src/frontend/node_modules/.bin/vite ]]; then
  (
    cd src/frontend
    NPM_CONFIG_CACHE="${NPM_CONFIG_CACHE}" npm install --silent
  ) 2>&1 | log_prefix >> "${FRONTEND_LOG}"
fi
(
  cd src/frontend
  NPM_CONFIG_CACHE="${NPM_CONFIG_CACHE}" npm run build
) 2>&1 | log_prefix >> "${FRONTEND_LOG}"

BACKEND_PID="$(launch_logged "${BACKEND_LOG}" "cd '${PROJECT_ROOT}' && env PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false MALLOC_ARENA_MAX=2 UV_CACHE_DIR='${UV_CACHE_DIR}' QILIN_REDIS_URL='${REDIS_URL}' QILIN_ASYNC_PREWARM=1 QILIN_BLOCKING_PREWARM=1 QILIN_PREWARM_FEED=1 QILIN_PREWARM_METRICS=0 QILIN_PREWARM_VALIDATION=0 QILIN_PREWARM_FEED_PAGE_SIZE=20 QILIN_BLOCKING_PREWARM_FEED_SCENES=rec QILIN_ASYNC_PREWARM_FEED_SCENES=search QILIN_PREWARM_METRICS_SAMPLE_N=48 QILIN_PREWARM_VALIDATION_GROUPS=3 QILIN_PREWARM_VALIDATION_EXAMPLES=3 QILIN_TAG_SEARCH=hard QILIN_TAG_REC=hard uv run python src/backend/online/api/main.py --host '${BIND_HOST}' --port '${BIND_BACKEND_PORT}' --tag hard")"
echo "${BACKEND_PID}" > "${BACKEND_PID_FILE}"

echo ""
echo "[Qilin] Started."
echo "Backend:  http://127.0.0.1:${BIND_BACKEND_PORT}/api/health"
echo "Frontend: http://127.0.0.1:${BIND_BACKEND_PORT}/login"
WSL_IP="$(detect_wsl_ip || true)"
if [[ -n "${WSL_IP}" ]]; then
  echo "Backend:  http://${WSL_IP}:${BIND_BACKEND_PORT}/api/health"
  echo "Frontend: http://${WSL_IP}:${BIND_BACKEND_PORT}/login"
fi
echo "Redis:    ${REDIS_URL}"
echo "Logs:     ${LOG_DIR}"
echo ""
