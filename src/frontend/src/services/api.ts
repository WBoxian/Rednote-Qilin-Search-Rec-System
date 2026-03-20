/*
Qilin Serving 前端 API 服务
- 统一封装 FastAPI 请求（feed / note / user / metrics / validation）
- 维护本地 scene、user_idx 状态
- 模型模式由后端自动选择：优先 hard，不可用时回退 easy
*/

const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:18080';
const behaviorInflight = new Map<string, Promise<any>>();
const behaviorLastTs = new Map<string, number>();

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
  const data = await r.json();
  if (!r.ok || data.ok === false) {
    throw new Error(data.detail || data.error || `HTTP ${r.status}`);
  }
  return data;
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
    req(`/api/user?scene=${scene}&user_idx=${userIdx}`),
  feed: (scene: string, userIdx: number, query: string, page: number, pageSize: number) =>
    req(`/api/feed?scene=${scene}&user_idx=${userIdx}&query=${encodeURIComponent(query)}&page=${page}&page_size=${pageSize}`),
  note: (scene: string, userIdx: number, requestId: number, noteIdx: number, query: string = '') =>
    req(`/api/note?scene=${scene}&user_idx=${userIdx}&request_id=${requestId}&note_idx=${noteIdx}&query=${encodeURIComponent(query)}`),
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
    ).finally(() => {
      behaviorInflight.delete(key);
    });
    behaviorInflight.set(key, p);
    return p;
  },
  metrics: (scene: string, maxGroups: number = 3000, dienMaxGroups: number = 1200) =>
    req(
      `/api/metrics?scene=${scene}`
      + `&max_groups=${Math.max(100, Math.min(10000, Number(maxGroups) || 3000))}`
      + `&dien_max_groups=${Math.max(100, Math.min(5000, Number(dienMaxGroups) || 1200))}`
    ),
  validation: (scene: string, maxGroups: number = 800, exampleLimit: number = 5, userIdx?: number | null) =>
    req(
      `/api/validation?scene=${scene}`
      + `&max_groups=${Math.max(1, Math.min(5000, Number(maxGroups) || 800))}`
      + `&example_limit=${Math.max(0, Math.min(20, Number(exampleLimit) || 5))}`
      + `${userIdx == null ? '' : `&user_idx=${Math.max(0, Number(userIdx) || 0)}`}`
    ),
  imageUrl: (path: string) => {
    const normalized = path.replace(/^\/+/, '').replace(/^image\//, '');
    return `${API_BASE}/image/${normalized}`;
  },
};
