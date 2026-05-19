/*
Qilin Serving 前端 API 服务
- 统一封装 FastAPI 请求（feed / note / user / metrics / validation）
- 维护本地 scene、user_idx 状态
- 模型模式由后端自动选择：优先 hard，不可用时回退 easy
*/

const API_BASE = import.meta.env.VITE_API_BASE || '';
const behaviorInflight = new Map<string, Promise<any>>();
const behaviorLastTs = new Map<string, number>();
const queryInflight = new Map<string, Promise<any>>();
const queryCache = new Map<string, { ts: number; data: any }>();
const METRICS_TTL_MS = 5 * 60_000;
const VALIDATION_TTL_MS = 15 * 60_000;
const NOTE_TTL_MS = 10 * 60_000;
const USER_TTL_MS = 5 * 60_000;

function invalidateCachedPaths(match: (path: string) => boolean) {
  for (const key of Array.from(queryCache.keys())) {
    if (match(key)) queryCache.delete(key);
  }
  for (const key of Array.from(queryInflight.keys())) {
    if (match(key)) queryInflight.delete(key);
  }
}

function invalidateUserBehaviorViews(userIdx: number) {
  const uid = Math.max(0, Number(userIdx) || 0);
  invalidateCachedPaths((path) =>
    (path.startsWith('/api/user?') && path.includes(`user_idx=${uid}`))
    || (path.startsWith('/api/validation?') && path.includes(`user_idx=${uid}`))
  );
}

export function getScene(): 'search' | 'rec' {
  return (localStorage.getItem('qilin_scene') as 'search' | 'rec') || 'search';
}

export function setScene(scene: 'search' | 'rec') {
  localStorage.setItem('qilin_scene', scene);
}

export function getUserId(): number | null {
  const v = localStorage.getItem('qilin_user_idx');
  if (!v) return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

export function setUserId(uid: number) {
  localStorage.setItem('qilin_user_idx', String(uid));
}

export function clearUserId() {
  localStorage.removeItem('qilin_user_idx');
}

function behaviorKey(scene: string, userIdx: number, noteIdx: number, requestId?: number, query: string = '', action: string = 'click') {
  return `${action}|${scene}|${Number(userIdx)}|${Number(noteIdx)}|${Number(requestId ?? -1)}|${String(query || '').slice(0, 64)}`;
}

async function req(path: string, init?: RequestInit) {
  const r = await fetch(`${API_BASE}${path}`, init);
  const text = await r.text();
  let data: any = null;
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = text;
    }
  }
  if (!r.ok || (data && typeof data === 'object' && data.ok === false)) {
    if (typeof data === 'string') {
      throw new Error(data || `HTTP ${r.status}`);
    }
    throw new Error(data?.detail || data?.error || `HTTP ${r.status}`);
  }
  return data;
}

function cachedReq(path: string, ttlMs: number) {
  const cached = queryCache.get(path);
  const now = Date.now();
  if (cached && (now - cached.ts) <= ttlMs) {
    return Promise.resolve(cached.data);
  }
  const inflight = queryInflight.get(path);
  if (inflight) return inflight;
  const promise = req(path)
    .then((data) => {
      queryCache.set(path, { ts: Date.now(), data });
      return data;
    })
    .finally(() => {
      queryInflight.delete(path);
    });
  queryInflight.set(path, promise);
  return promise;
}

export const api = {
  login: (scene: string, userIdx: number) =>
    req(`/api/login?scene=${scene}&user_idx=${userIdx}`),
  users: (scene: string, limit: number = 20, offset: number = 0, randomShow: boolean = false) =>
    req(
      `/api/users?scene=${scene}`
      + `&limit=${Math.max(1, Math.min(200, Number(limit) || 20))}`
      + `&offset=${Math.max(0, Number(offset) || 0)}`
      + `&random_show=${randomShow ? 1 : 0}`
    ),
  user: (scene: string, userIdx: number) =>
    cachedReq(`/api/user?scene=${scene}&user_idx=${userIdx}`, USER_TTL_MS),
  feed: (
    scene: string,
    userIdx: number,
    query: string,
    page: number,
    pageSize: number,
    refreshKey: string = '',
    excludeNoteIds: number[] = [],
  ) => {
    const exclude = Array.from(
      new Set(
        excludeNoteIds
          .map((x) => Number(x))
          .filter((x) => Number.isFinite(x) && x >= 0)
      )
    )
      .slice(0, 200)
      .join(',');
    return req(
      `/api/feed?scene=${scene}&user_idx=${userIdx}&query=${encodeURIComponent(query)}&page=${page}&page_size=${pageSize}`
      + `${refreshKey ? `&refresh_key=${encodeURIComponent(refreshKey)}` : ''}`
      + `${exclude ? `&exclude_note_ids=${encodeURIComponent(exclude)}` : ''}`
    );
  },
  suggest: (scene: string, query: string, limit: number = 8) =>
    req(`/api/suggest?scene=${scene}&query=${encodeURIComponent(query)}&limit=${Math.max(1, Math.min(12, Number(limit) || 8))}`),
  note: (scene: string, userIdx: number, requestId: number, noteIdx: number, query: string = '', metaOnly = false) => {
    const path =
      `/api/note?scene=${scene}&user_idx=${userIdx}&request_id=${requestId}&note_idx=${noteIdx}`
      + `&query=${encodeURIComponent(query)}&meta_only=${metaOnly ? 1 : 0}`;
    return cachedReq(path, NOTE_TTL_MS);
  },
  click: (
    scene: string,
    userIdx: number,
    noteIdx: number,
    requestId?: number,
    query: string = '',
  ) => {
    const key = behaviorKey(scene, userIdx, noteIdx, requestId, query, 'click');
    const now = Date.now();
    const lastTs = behaviorLastTs.get(key) || 0;
    if (now - lastTs < 1200) {
      return behaviorInflight.get(key) || Promise.resolve({ ok: true, accepted: false, skipped: true });
    }
    const rid = Number.isFinite(Number(requestId)) ? `&request_id=${Number(requestId)}` : '';
    const q = query ? `&query=${encodeURIComponent(query)}` : '';
    behaviorLastTs.set(key, now);
    const p = req(`/api/behavior/click?scene=${scene}&user_idx=${userIdx}&note_idx=${noteIdx}${rid}${q}`, { method: 'POST' })
      .then((data) => {
        invalidateUserBehaviorViews(userIdx);
        return data;
      })
      .finally(() => {
        behaviorInflight.delete(key);
      });
    behaviorInflight.set(key, p);
    return p;
  },
  view: (
    scene: string,
    userIdx: number,
    noteIdx: number,
    requestId?: number,
    query: string = '',
  ) => {
    const key = behaviorKey(scene, userIdx, noteIdx, requestId, query, 'view');
    const now = Date.now();
    const lastTs = behaviorLastTs.get(key) || 0;
    if (now - lastTs < 1200) {
      return behaviorInflight.get(key) || Promise.resolve({ ok: true, accepted: false, skipped: true });
    }
    const rid = Number.isFinite(Number(requestId)) ? `&request_id=${Number(requestId)}` : '';
    const q = query ? `&query=${encodeURIComponent(query)}` : '';
    behaviorLastTs.set(key, now);
    const p = req(`/api/behavior/view?scene=${scene}&user_idx=${userIdx}&note_idx=${noteIdx}${rid}${q}`, { method: 'POST' })
      .then((data) => {
        invalidateUserBehaviorViews(userIdx);
        return data;
      })
      .finally(() => {
        behaviorInflight.delete(key);
      });
    behaviorInflight.set(key, p);
    return p;
  },
  engage: (
    scene: string,
    userIdx: number,
    noteIdx: number,
    requestId?: number,
    query: string = '',
    payload?: { like?: number; collect?: number; comment?: number; share?: number; pageTime?: number },
  ) => {
    const like = Math.max(0, Number(payload?.like || 0));
    const collect = Math.max(0, Number(payload?.collect || 0));
    const comment = Math.max(0, Number(payload?.comment || 0));
    const share = Math.max(0, Number(payload?.share || 0));
    const pageTime = Math.max(0, Number(payload?.pageTime || 0));
    const key = behaviorKey(scene, userIdx, noteIdx, requestId, `${query}|${like}|${collect}|${comment}|${share}|${pageTime.toFixed(1)}`, 'engage');
    const now = Date.now();
    const lastTs = behaviorLastTs.get(key) || 0;
    if (now - lastTs < 1200) {
      return behaviorInflight.get(key) || Promise.resolve({ ok: true, accepted: false, skipped: true });
    }
    const rid = Number.isFinite(Number(requestId)) ? `&request_id=${Number(requestId)}` : '';
    const q = query ? `&query=${encodeURIComponent(query)}` : '';
    behaviorLastTs.set(key, now);
    const p = req(
      `/api/behavior/engage?scene=${scene}&user_idx=${userIdx}&note_idx=${noteIdx}${rid}${q}`
      + `&like=${like}&collect=${collect}&comment=${comment}&share=${share}&page_time=${pageTime}`,
      { method: 'POST' }
    ).then((data) => {
      invalidateUserBehaviorViews(userIdx);
      return data;
    }).finally(() => {
      behaviorInflight.delete(key);
    });
    behaviorInflight.set(key, p);
    return p;
  },
  deleteBehavior: (
    scene: string,
    userIdx: number,
    noteIdx: number,
    ts?: number,
    requestId?: number,
  ) => req(
    `/api/behavior?scene=${scene}&user_idx=${userIdx}&note_idx=${noteIdx}`
    + `${ts == null ? '' : `&ts=${Math.max(0, Number(ts) || 0)}`}`
    + `${requestId == null ? '' : `&request_id=${Math.max(0, Number(requestId) || 0)}`}`,
    { method: 'DELETE' }
  ).then((data) => {
    invalidateUserBehaviorViews(userIdx);
    return data;
  }),
  deleteBehaviorsBatch: (
    userIdx: number,
    items: Array<{ scene: string; note_idx: number; ts?: number; request_id?: number }>,
  ) => req(
    '/api/behavior/batch_delete',
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_idx: Math.max(0, Number(userIdx) || 0),
        items: (items || []).map((item) => ({
          scene: String(item?.scene || 'search'),
          note_idx: Math.max(0, Number(item?.note_idx) || 0),
          ts: item?.ts == null ? null : Math.max(0, Number(item.ts) || 0),
          request_id: item?.request_id == null ? null : Math.max(0, Number(item.request_id) || 0),
        })),
      }),
    }
  ).then((data) => {
    invalidateUserBehaviorViews(userIdx);
    return data;
  }),
  metrics: (scene: string, sampleN: number = 1000, includeVal = true) => {
    const path =
      `/api/metrics?scene=${scene}`
      + `&sample_n=${Math.max(2, Math.min(2000, Number(sampleN) || 1000))}`
      + `&include_val=${includeVal ? 1 : 0}`;
    return cachedReq(path, METRICS_TTL_MS);
  },
  validation: (scene: string, maxGroups: number = 240, exampleLimit: number = 24, userIdx?: number | null) => {
    const path =
      `/api/validation?scene=${scene}`
      + `&shape_v=2`
      + `&max_groups=${Math.max(1, Math.min(1000, Number(maxGroups) || 240))}`
      + `&example_limit=${Math.max(0, Math.min(40, Number(exampleLimit) || 24))}`
      + `${userIdx == null ? '' : `&user_idx=${Math.max(0, Number(userIdx) || 0)}`}`;
    return cachedReq(path, VALIDATION_TTL_MS);
  },
  imageUrl: (path: string) => {
    const normalized = path.replace(/^\/+/, '').replace(/^image\//, '');
    return `${API_BASE}/image/${normalized}`;
  },
};
