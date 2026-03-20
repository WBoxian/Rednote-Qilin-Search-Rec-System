<script setup lang="ts">
import { nextTick, onMounted, onUnmounted, ref } from 'vue';
import { useRoute, useRouter } from 'vue-router';
import { api, getUserId } from '../services/api';
import TopBar from '../components/TopBar.vue';

const router = useRouter();
const route = useRoute();
const query = ref('');
const items = ref<any[]>([]);
const backendPage = ref(1);
const preloadBuffer = ref<any[]>([]);
const currentScene = ref<'search' | 'rec'>('rec');
const loading = ref(false);
const errorMsg = ref('');
const latencyMs = ref<number | null>(null);
const loadAnchor = ref<HTMLElement | null>(null);
let observer: IntersectionObserver | null = null;
const clickLocks = new Map<string, number>();
const HOME_CACHE_KEY = 'qilin_home_cache';

const uid = getUserId();
if (uid == null) router.replace('/login');

const activeScene = () => (query.value.trim() ? 'search' : 'rec');
const isRecMode = () => !query.value.trim();

function fmtNum(v: number) {
  if (v >= 1e8) return `${(v / 1e8).toFixed(1)}亿`;
  if (v >= 1e4) return `${(v / 1e4).toFixed(1)}万`;
  return String(Math.round(v));
}

function saveHomeCache() {
  try {
    sessionStorage.setItem(
      HOME_CACHE_KEY,
      JSON.stringify({
        uid,
        query: query.value,
        items: items.value,
        preloadBuffer: preloadBuffer.value,
        backendPage: backendPage.value,
        currentScene: currentScene.value,
        latencyMs: latencyMs.value,
        savedAt: Date.now(),
      })
    );
  } catch {
  }
}

function restoreHomeCache(): boolean {
  try {
    const raw = sessionStorage.getItem(HOME_CACHE_KEY);
    if (!raw) return false;
    const obj = JSON.parse(raw);
    if (!obj || Number(obj.uid) !== Number(uid)) return false;
    if (Date.now() - Number(obj.savedAt || 0) > 30 * 60 * 1000) return false;
    query.value = String(obj.query || '');
    items.value = Array.isArray(obj.items) ? obj.items : [];
    preloadBuffer.value = Array.isArray(obj.preloadBuffer) ? obj.preloadBuffer : [];
    backendPage.value = Math.max(1, Number(obj.backendPage || 1));
    currentScene.value = (obj.currentScene === 'search' ? 'search' : 'rec');
    latencyMs.value = Number(obj.latencyMs || 0);
    return items.value.length > 0;
  } catch {
    return false;
  }
}

async function load(reset = false) {
  if (uid == null) return;
  loading.value = true;
  errorMsg.value = '';
  currentScene.value = activeScene();
  if (reset) {
    backendPage.value = 1;
    items.value = [];
    preloadBuffer.value = [];
  }
  try {
    const reqPage = currentScene.value === 'rec' ? 1 : backendPage.value;
    const r = await api.feed(currentScene.value, uid, query.value, reqPage, 40);
    const f = r.feed;
    latencyMs.value = Number(r?.latency_ms ?? f?.latency_ms ?? 0);
    const batch = Array.isArray(f.items) ? f.items : [];
    const seen = new Set(items.value.map((x) => Number(x.note_idx)));
    const unique = batch.filter((x: any) => {
      const nid = Number(x?.note_idx);
      if (!Number.isFinite(nid) || seen.has(nid)) return false;
      seen.add(nid);
      return true;
    });
    const head = unique.slice(0, 20);
    const tail = unique.slice(20, 40);
    items.value.push(...head);
    preloadBuffer.value.push(...tail);
    if (currentScene.value === 'rec') {
      backendPage.value = 1;
    } else {
      backendPage.value += 1;
    }
  } catch (e: any) {
    errorMsg.value = e?.message || '请求失败，请检查后端服务与模型产物是否可用';
  } finally {
    loading.value = false;
    saveHomeCache();
  }
}

async function refreshRec() {
  if (loading.value || !isRecMode()) return;
  await load(true);
}

async function consumeMore() {
  if (loading.value) return;
  if (preloadBuffer.value.length > 0) {
    items.value.push(...preloadBuffer.value.splice(0, 20));
    if (preloadBuffer.value.length < 10) {
      await load(false);
    }
    return;
  }
  await load(false);
}

function onImgError(e: Event) {
  const img = e.target as HTMLImageElement | null;
  if (!img) return;
  img.style.display = 'none';
}

async function openDetail(it: any) {
  const k = `${it.scene}|${it.user_idx}|${it.request_id}|${it.note_idx}`;
  const now = Date.now();
  if ((clickLocks.get(k) || 0) + 1200 > now) return;
  clickLocks.set(k, now);
  try {
    await api.click(it.scene, it.user_idx, it.note_idx, it.request_id, query.value);
  } catch {
  }
  saveHomeCache();
  const homeRank = Number(it?.stage_ranks?.rerank ?? 0);
  const rankArg = Number.isFinite(homeRank) && homeRank > 0 ? `&home_rank=${homeRank}` : '';
  router.push(`/detail?scene=${it.scene}&user_idx=${it.user_idx}&request_id=${it.request_id}&note_idx=${it.note_idx}&query=${encodeURIComponent(query.value || '')}${rankArg}`);
}

onMounted(async () => {
  const preserve = String(route.query.preserve || '') === '1';
  const restored = preserve ? restoreHomeCache() : false;
  if (!restored) {
    await load(true);
  }
  await nextTick();
  if (loadAnchor.value) {
    observer = new IntersectionObserver(async (entries) => {
      const hit = entries.some((x) => x.isIntersecting);
      if (hit) await consumeMore();
    }, { threshold: 0.1 });
    observer.observe(loadAnchor.value);
  }
});

onUnmounted(() => {
  if (observer) {
    observer.disconnect();
    observer = null;
  }
});
</script>

<template>
  <TopBar title="小红书麒麟推荐" :latency-ms="latencyMs">
    <template #actions>
      <input
        v-model="query"
        class="search-input"
        placeholder="搜索你感兴趣的内容..."
        @keyup.enter="load(true)"
      />
      <button v-if="isRecMode()" class="btn" :disabled="loading" @click="refreshRec">换一批</button>
      <button class="btn-primary" :disabled="loading" @click="load(true)">{{ loading ? '加载中...' : '搜索' }}</button>
    </template>
  </TopBar>

  <main class="container">
    <div v-if="errorMsg" class="panel" style="margin-bottom:10px;">{{ errorMsg }}</div>
    <section class="grid">
      <article v-for="it in items" :key="`${it.request_id}-${it.note_idx}`" class="card" @click="openDetail(it)">
        <img v-if="it.cover_image" :src="api.imageUrl(it.cover_image)" alt="cover" loading="lazy" @error="onImgError" />
        <div v-else style="aspect-ratio:3/4;background:#eef2f7;"></div>
        <div class="card-body">
          <div class="card-title">{{ it.title }}</div>
          <div class="stats">
            <span>❤ {{ fmtNum(it.accum_like_num) }}</span>
            <span>⭐ {{ fmtNum(it.accum_collect_num) }}</span>
            <span>💬 {{ fmtNum(it.accum_comment_num) }}</span>
          </div>
        </div>
      </article>
    </section>
    <div ref="loadAnchor" style="height:24px;display:flex;align-items:center;justify-content:center;margin-top:10px;" class="meta">
      {{ loading ? '加载中...' : (isRecMode() ? '下滑继续发现新推荐' : '下滑自动加载更多') }}
    </div>
  </main>
</template>
