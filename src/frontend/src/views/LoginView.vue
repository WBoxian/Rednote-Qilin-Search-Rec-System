<script setup lang="ts">
import { onMounted, ref } from 'vue';
import { useRouter } from 'vue-router';
import { api, getScene, setScene, setUserId } from '../services/api';
import TopBar from '../components/TopBar.vue';

const router = useRouter();
const userIdx = ref('');
const msg = ref('');
const scene = ref<'search' | 'rec'>(getScene());
const users = ref<any[]>([]);
const loadingUsers = ref(false);
const listOffset = ref(0);
const userPanelRef = ref<HTMLElement | null>(null);
const latencyMs = ref<number | null>(null);
const HOME_CACHE_KEY = 'qilin_home_cache_v2';
const loggingIn = ref(false);
const loginProgress = ref(0);
const loginStage = ref('');

function setLoginProgress(next: number, stage: string) {
  loginProgress.value = Math.max(0, Math.min(100, Math.round(next)));
  loginStage.value = stage;
}

function switchScene(s: 'search' | 'rec') {
  scene.value = s;
  setScene(s);
  refreshUsers();
}

async function login() {
  const rawUid = String(userIdx.value || '').trim() || '15128';
  const uid = Number(rawUid);
  if (!Number.isInteger(uid) || uid < 0) {
    msg.value = 'user_idx 需为非负整数';
    return;
  }
  userIdx.value = String(uid);
  loggingIn.value = true;
  setLoginProgress(12, '校验用户与初始化会话');
  try {
    const r = await api.login(scene.value, uid);
    setLoginProgress(46, '读取用户信息与首屏推荐');
    latencyMs.value = Number(r?.latency_ms ?? 0);
    setUserId(uid);
    try {
      const feed = r?.homepage_feed || {};
      const batch = Array.isArray(feed?.items) ? feed.items : [];
      setLoginProgress(82, '整理首页结果与阶段延迟');
      if (batch.length > 0) {
        sessionStorage.setItem(
          HOME_CACHE_KEY,
          JSON.stringify({
            uid,
            query: '',
            items: batch.slice(0, 15),
            preloadBuffer: batch.slice(15),
            backendPage: 1,
            feedSessionKey: String(feed?.feed_session_key || `login-${uid}`),
            currentScene: 'rec',
            latencyMs: Number(feed?.latency_ms ?? 0),
            stageMs: (feed?.stage_ms && typeof feed.stage_ms === 'object') ? feed.stage_ms : null,
            refreshSeenNoteIds: [],
            savedAt: Date.now(),
          })
        );
      }
    } catch {
    }
    setLoginProgress(100, '准备进入首页');
    msg.value = `登录成功，scene=${scene.value} 请求数=${r.user.request_count_in_test}`;
    await new Promise((resolve) => window.setTimeout(resolve, 220));
    router.push('/');
  } catch (e: any) {
    msg.value = e?.message || '登录失败，请检查后端服务';
  } finally {
    loggingIn.value = false;
  }
}

function mergeUsers(next: any[]) {
  const seen = new Set(users.value.map((x: any) => Number(x.user_idx)));
  for (const u of next) {
    const uid = Number(u?.user_idx);
    if (Number.isFinite(uid) && !seen.has(uid)) {
      users.value.push(u);
      seen.add(uid);
    }
  }
}

async function loadUsers(reset = false) {
  if (loadingUsers.value) return;
  loadingUsers.value = true;
  try {
    if (reset) {
      users.value = [];
      listOffset.value = 0;
    }
    const r = await api.users(scene.value, 30, listOffset.value, true);
    latencyMs.value = Number(r?.latency_ms ?? 0);
    const rows = Array.isArray(r.users) ? r.users : [];
    mergeUsers(rows);
    listOffset.value += rows.length;
    if (reset && rows.length === 0) {
      msg.value = '当前场景暂无可展示用户';
    }
  } catch (e: any) {
    msg.value = e?.message || '加载用户列表失败';
    if (reset) users.value = [];
  } finally {
    loadingUsers.value = false;
  }
}

async function refreshUsers() {
  await loadUsers(true);
}

function onUserScroll(e: Event) {
  const el = e.target as HTMLElement;
  if (!el) return;
  const remain = el.scrollHeight - el.scrollTop - el.clientHeight;
  if (remain < 80) {
    loadUsers(false);
  }
}

onMounted(refreshUsers);

function formatGender(v: any) {
  const s = String(v ?? '').trim();
  if (!s || s === '-' || s === '0') return '-';
  if (s === '1' || s.toLowerCase() === 'male') return '男';
  if (s === '2' || s.toLowerCase() === 'female') return '女';
  return s;
}

function formatAge(v: any) {
  const s = String(v ?? '').trim();
  if (s && s !== '-' && s !== '0' && Number.isNaN(Number(s))) return s;
  const n = Number(v);
  if (!Number.isFinite(n) || n <= 0) return '-';
  return String(Math.round(n));
}

function formatPlain(v: any) {
  const s = String(v ?? '').trim();
  return s && s !== '0' ? s : '-';
}
</script>

<template>
  <div>
    <TopBar title="小红书麒麟搜推系统项目" :latency-ms="latencyMs" />
    <main class="container login-page">
      <section class="panel login-panel">
        <h3>用户登录</h3>
        <div class="meta">输入 user_idx 进入系统</div>
        <form style="display:grid;grid-template-columns:1fr auto;gap:8px;margin-top:12px;" @submit.prevent="login">
          <input v-model="userIdx" placeholder="例如 15128" @keydown.enter.prevent="login" />
          <button type="submit" class="btn-primary" :disabled="loggingIn">{{ loggingIn ? '加载中...' : '登录' }}</button>
        </form>
        <div class="meta" style="margin-top:10px;">{{ msg }}</div>

        <div v-if="loggingIn" class="login-progress-shell">
          <div class="login-progress-head">
            <div class="login-progress-title">
              <span class="login-spinner" aria-hidden="true" />
              <strong>正在进入首页服务</strong>
            </div>
            <span>{{ loginProgress }}%</span>
          </div>
          <div class="login-progress-bar">
            <div class="login-progress-fill" :style="{ width: `${loginProgress}%` }" />
          </div>
          <div class="meta">{{ loginStage }}</div>
        </div>

        <div style="display:flex;align-items:center;justify-content:space-between;margin:14px 0 8px;gap:10px;">
          <div style="display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;">
            <h3 style="margin:0;">随机用户</h3>
            <div class="meta">提示：点击表格行可自动填充 user_idx。</div>
          </div>
          <button class="btn" @click="refreshUsers">换一批</button>
        </div>
        <div
          v-if="users.length"
          class="user-table-wrap"
          ref="userPanelRef"
          @scroll="onUserScroll"
        >
          <table class="table">
            <tbody>
              <tr><td>user_idx</td><td>gender</td><td>age</td><td>platform</td><td>location</td><td>fans_num</td><td>follows_num</td></tr>
              <tr v-for="u in users" :key="u.user_idx" @click="userIdx = String(u.user_idx)" style="cursor:pointer;">
                <td>{{ u.user_idx }}</td>
                <td>{{ formatGender(u.gender ?? u.gender_enc) }}</td>
                <td>{{ formatAge(u.age ?? u.age_enc) }}</td>
                <td>{{ formatPlain(u.platform ?? u.platform_enc) }}</td>
                <td>{{ formatPlain(u.location ?? u.location_enc) }}</td>
                <td>{{ u.fans_num ?? '-' }}</td>
                <td>{{ u.follows_num ?? '-' }}</td>
              </tr>
            </tbody>
          </table>
        </div>
        <div v-else class="meta">暂无用户列表数据。</div>
      </section>
    </main>
  </div>
</template>

<style scoped>
.login-page {
  min-height: calc(100vh - 180px);
}

.login-panel {
  max-width: 1120px;
  margin: 24px auto;
  min-height: calc(100vh - 250px);
  display: flex;
  flex-direction: column;
}

.user-table-wrap {
  width: 100%;
  flex: 1 1 auto;
  min-height: 420px;
  max-height: none;
  overflow: auto;
  border: 1px solid var(--line);
  border-radius: 10px;
}

.table td {
  word-break: break-word;
  white-space: normal;
}

.login-progress-shell {
  margin-top: 14px;
  padding: 14px 16px;
  border-radius: 16px;
  border: 1px solid rgba(37, 48, 61, 0.08);
  background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(243, 238, 230, 0.92));
  box-shadow: 0 16px 28px rgba(28, 36, 46, 0.08);
}

.login-progress-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 10px;
  color: #1f2a36;
}

.login-progress-title {
  display: inline-flex;
  align-items: center;
  gap: 10px;
}

.login-spinner {
  width: 18px;
  height: 18px;
  border-radius: 999px;
  border: 2px solid rgba(84, 114, 150, 0.22);
  border-top-color: #294566;
  border-right-color: #5f85b2;
  animation: login-spin 0.9s linear infinite;
}

.login-progress-bar {
  width: 100%;
  height: 10px;
  overflow: hidden;
  border-radius: 999px;
  background: rgba(95, 108, 123, 0.14);
  margin-bottom: 10px;
}

.login-progress-fill {
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, #243548, #547296 65%, #91b1d7);
  box-shadow: 0 6px 14px rgba(44, 66, 92, 0.22);
  transition: width 0.18s ease;
}

@keyframes login-spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

@media (max-width: 960px) {
  .login-page {
    min-height: auto;
  }

  .login-panel {
    min-height: auto;
  }

  .user-table-wrap {
    min-height: 360px;
  }
}
</style>
