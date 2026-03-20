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

function switchScene(s: 'search' | 'rec') {
  scene.value = s;
  setScene(s);
  refreshUsers();
}

async function login() {
  // 前端做基础校验，避免把无效参数发给后端
  const uid = Number(userIdx.value || '0');
  if (!Number.isInteger(uid) || uid < 0) {
    msg.value = 'user_idx 需为非负整数';
    return;
  }
  const r = await api.login(scene.value, uid);
  latencyMs.value = Number(r?.latency_ms ?? 0);
  setUserId(uid);
  msg.value = `登录成功，scene=${scene.value} 请求数=${r.user.request_count_in_test}`;
  router.push('/');
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
  } catch {
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
</script>

<template>
  <div>
    <TopBar title="小红书麒麟推荐" :latency-ms="latencyMs">
      <template #actions>
        <button class="btn-tab" :class="{ active: scene === 'search' }" @click="switchScene('search')">Search</button>
        <button class="btn-tab" :class="{ active: scene === 'rec' }" @click="switchScene('rec')">Rec</button>
      </template>
    </TopBar>
    <main class="container">
      <section class="panel" style="max-width:1120px;margin:24px auto;">
        <h3>用户登录</h3>
        <div class="meta">输入 user_idx 进入系统</div>
        <div style="display:grid;grid-template-columns:1fr auto;gap:8px;margin-top:12px;">
          <input v-model="userIdx" placeholder="例如 208" />
          <button class="btn-primary" @click="login">登录</button>
        </div>
        <div class="meta" style="margin-top:10px;">{{ msg }}</div>

        <div style="display:flex;align-items:center;justify-content:space-between;margin:14px 0 8px;gap:10px;">
          <h3 style="margin:0;">随机用户（每次30，可下拉加载）</h3>
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
                <td>{{ u.gender ?? u.gender_enc ?? '-' }}</td>
                <td>{{ u.age ?? u.age_enc ?? '-' }}</td>
                <td>{{ u.platform ?? u.platform_enc ?? '-' }}</td>
                <td>{{ u.location ?? u.location_enc ?? '-' }}</td>
                <td>{{ u.fans_num ?? '-' }}</td>
                <td>{{ u.follows_num ?? '-' }}</td>
              </tr>
            </tbody>
          </table>
        </div>
        <div v-else class="meta">暂无用户列表数据。</div>
        <div class="meta" style="margin-top:8px;">
          提示：点击表格行可自动填充 user_idx；下拉到底部会继续加载。
        </div>
      </section>
    </main>
  </div>
</template>

<style scoped>
.user-table-wrap {
  width: 100%;
  max-height: 420px;
  overflow: auto;
  border: 1px solid var(--line);
  border-radius: 10px;
}

.table td {
  word-break: break-word;
  white-space: normal;
}
</style>
