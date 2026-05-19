<script setup lang="ts">
import { nextTick, onMounted, onUnmounted, ref, watch } from 'vue';
import { useRoute, useRouter } from 'vue-router';
import { api, getUserId } from '../services/api';
import TopBar from '../components/TopBar.vue';

const router = useRouter();
const route = useRoute();
const query = ref('');
const items = ref<any[]>([]);
const backendPage = ref(1);
const preloadBuffer = ref<any[]>([]);
const feedSessionKey = ref('');
const currentScene = ref<'search' | 'rec'>('rec');
const loading = ref(false);
const errorMsg = ref('');
const latencyMs = ref<number | null>(null);
const stageMs = ref<Record<string, number> | null>(null);
const loadAnchor = ref<HTMLElement | null>(null);
const suggestItems = ref<Array<{ text: string; hint: string }>>([]);
const suggestOpen = ref(false);
const suggestTimer = ref<number | null>(null);
const refreshSeenNoteIds = ref<number[]>([]);
const brokenImages = ref<Record<string, boolean>>({});
let observer: IntersectionObserver | null = null;
const HOME_CACHE_KEY = 'qilin_home_cache_v2';
const DETAIL_SEED_KEY = 'qilin_detail_seed';
const pageTitle = 'Qilin \u9996\u9875\u63A8\u8350';
const copy = {
  searchPlaceholder: '\u641C\u7D22\u4F60\u611F\u5174\u8DA3\u7684\u5185\u5BB9...',
  refreshBatch: '\u6362\u4E00\u6279',
  refresh: '\u5237\u65B0',
  loading: '\u52A0\u8F7D\u4E2D...',
  loadError: '\u52A0\u8F7D\u9996\u9875\u5185\u5BB9\u5931\u8D25\uFF0C\u8BF7\u7A0D\u540E\u91CD\u8BD5',
  loadMoreRec: '\u7EE7\u7EED\u4E0B\u6ED1\u52A0\u8F7D\u66F4\u591A\u63A8\u8350',
  loadMoreSearch: '\u7EE7\u7EED\u4E0B\u6ED1\u52A0\u8F7D\u66F4\u591A\u641C\u7D22\u7ED3\u679C',
  noImage: '\u6682\u65E0\u56FE\u7247',
};

const uid = getUserId();
if (uid == null) router.replace('/login');

const activeScene = () => (query.value.trim() ? 'search' : 'rec');
const isRecMode = () => !query.value.trim();

function fmtNum(v: number) {
  if (v >= 1e8) return `${(v / 1e8).toFixed(1)}\u4EBF`;
  if (v >= 1e4) return `${(v / 1e4).toFixed(1)}\u4E07`;
  return String(Math.round(v));
}

function cardKey(it: any) {
  return `${it?.request_id ?? 0}-${it?.note_idx ?? 0}`;
}

function titleTags(it: any) {
  return Array.isArray(it?.topic_tokens) ? it.topic_tokens.slice(0, 3) : [];
}

function hasCover(it: any) {
  const key = cardKey(it);
  return Boolean(it?.cover_image) && !brokenImages.value[key];
}

function markImageBroken(it: any) {
  brokenImages.value = { ...brokenImages.value, [cardKey(it)]: true };
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
        feedSessionKey: feedSessionKey.value,
        currentScene: currentScene.value,
        latencyMs: latencyMs.value,
        stageMs: stageMs.value,
        refreshSeenNoteIds: refreshSeenNoteIds.value,
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
    feedSessionKey.value = String(obj.feedSessionKey || '');
    currentScene.value = (obj.currentScene === 'search' ? 'search' : 'rec');
    latencyMs.value = Number(obj.latencyMs || 0);
    stageMs.value = (obj.stageMs && typeof obj.stageMs === 'object') ? obj.stageMs : null;
    refreshSeenNoteIds.value = Array.isArray(obj.refreshSeenNoteIds) ? obj.refreshSeenNoteIds : [];
    brokenImages.value = {};
    return items.value.length > 0;
  } catch {
    return false;
  }
}

function collectShownNoteIds(limit = 1200) {
  const seen = new Set<number>();
  const out: number[] = [];
  for (const row of items.value) {
    const nid = Number(row?.note_idx);
    if (!Number.isFinite(nid) || seen.has(nid)) continue;
    seen.add(nid);
    out.push(nid);
    if (out.length >= limit) break;
  }
  return out;
}

function mergeRefreshExclusions(nextIds: number[]) {
  const merged = Array.from(new Set([...refreshSeenNoteIds.value, ...nextIds.filter((x) => Number.isFinite(x) && x >= 0)]));
  refreshSeenNoteIds.value = merged.slice(-1600);
  return refreshSeenNoteIds.value;
}

async function load(reset = false, forceRefresh = false, excludeNoteIds: number[] = []) {
  if (uid == null) return;
  loading.value = true;
  errorMsg.value = '';
  currentScene.value = activeScene();
  if (reset) {
    backendPage.value = 1;
    items.value = [];
    preloadBuffer.value = [];
    brokenImages.value = {};
    feedSessionKey.value = currentScene.value === 'rec' ? `rec-${Date.now()}` : '';
  }
  try {
    if (forceRefresh && currentScene.value === 'rec') {
      feedSessionKey.value = `rec-${Date.now()}`;
    }
    const reqPage = backendPage.value;
    const pageSize = 15;
    const r = await api.feed(currentScene.value, uid, query.value, reqPage, pageSize, feedSessionKey.value, excludeNoteIds);
    const f = r.feed;
    latencyMs.value = Number(f?.latency_ms ?? r?.latency_ms ?? 0);
    stageMs.value = (f?.stage_ms && typeof f.stage_ms === 'object') ? f.stage_ms : null;
    const batch = Array.isArray(f.items) ? f.items : [];
    const seen = new Set(items.value.map((x) => Number(x.note_idx)));
    const unique = batch.filter((x: any) => {
      const nid = Number(x?.note_idx);
      if (!Number.isFinite(nid) || seen.has(nid)) return false;
      seen.add(nid);
      return true;
    });
    const head = unique.slice(0, 15);
    const tail = unique.slice(15, 20);
    items.value.push(...head);
    preloadBuffer.value.push(...tail);
    backendPage.value += 1;
  } catch (e: any) {
    errorMsg.value = e?.message || copy.loadError;
  } finally {
    loading.value = false;
    saveHomeCache();
  }
}

async function refreshRec() {
  if (loading.value || !isRecMode() || uid == null) return;
  const excludeIds = mergeRefreshExclusions(collectShownNoteIds());
  if (uid == null) return;
  loading.value = true;
  errorMsg.value = '';
  currentScene.value = 'rec';
  brokenImages.value = {};
  try {
    const nextRefreshKey = `rec-${Date.now()}`;
    const r = await api.feed('rec', uid, '', 1, 15, nextRefreshKey, excludeIds);
    const f = r.feed || {};
    latencyMs.value = Number(f?.latency_ms ?? r?.latency_ms ?? 0);
    stageMs.value = (f?.stage_ms && typeof f.stage_ms === 'object') ? f.stage_ms : null;
    const batch = Array.isArray(f.items) ? f.items : [];
    items.value = batch.slice(0, 15);
    preloadBuffer.value = batch.slice(15);
    feedSessionKey.value = String(f.feed_session_key || nextRefreshKey);
    backendPage.value = 1;
  } catch (e: any) {
    errorMsg.value = e?.message || copy.loadError;
  } finally {
    loading.value = false;
    saveHomeCache();
  }
}

async function returnToRecommendationHome() {
  query.value = '';
  suggestItems.value = [];
  suggestOpen.value = false;
  refreshSeenNoteIds.value = [];
  brokenImages.value = {};
  await load(true, true);
}

async function consumeMore() {
  if (loading.value) return;
  if (preloadBuffer.value.length > 0) {
    items.value.push(...preloadBuffer.value.splice(0, 15));
    if (preloadBuffer.value.length < 8) {
      await load(false);
    }
    return;
  }
  await load(false);
}

async function loadSuggest() {
  const q = query.value.trim();
  if (!q) {
    suggestItems.value = [];
    suggestOpen.value = false;
    return;
  }
  try {
    const res = await api.suggest('search', q, 8);
    suggestItems.value = Array.isArray(res?.items)
      ? res.items
          .map((item: any) => {
            if (typeof item === 'string') {
              return { text: item, hint: item === q ? '当前输入' : '猜你想搜' };
            }
            return {
              text: String(item?.text || ''),
              hint: String(item?.hint || (item?.source || '猜你想搜')),
            };
          })
          .filter((item: any) => item.text)
      : [];
    suggestOpen.value = suggestItems.value.length > 0;
  } catch {
    suggestItems.value = [];
    suggestOpen.value = false;
  }
}

function pickSuggest(text: string) {
  query.value = text;
  suggestOpen.value = false;
  load(true);
}

function closeSuggest() {
  window.setTimeout(() => {
    suggestOpen.value = false;
  }, 120);
}

async function refreshFromBrand() {
  query.value = '';
  suggestItems.value = [];
  suggestOpen.value = false;
  refreshSeenNoteIds.value = [];
  brokenImages.value = {};
  await load(true, true);
}

watch(query, () => {
  if (suggestTimer.value != null) {
    window.clearTimeout(suggestTimer.value);
  }
  suggestTimer.value = window.setTimeout(() => {
    loadSuggest();
  }, 140);
});

async function openDetail(it: any) {
  saveHomeCache();
  try {
    if (uid != null) {
      void api.click(
        String(it?.scene || currentScene.value || 'rec'),
        Number(uid),
        Number(it?.note_idx || 0),
        Number(it?.request_id || 0),
        String(query.value || ''),
      );
    }
  } catch {
  }
  try {
    sessionStorage.setItem(
      DETAIL_SEED_KEY,
      JSON.stringify({
        scene: it.scene,
        user_idx: it.user_idx,
        request_id: it.request_id,
        note_idx: it.note_idx,
        query: query.value || '',
        detail: {
          title: it.title || '',
          content: it.content || '',
          cover_image: it.cover_image || '',
          scene: it.scene,
          request_id: it.request_id,
          note_idx: it.note_idx,
          accum_like_num: Number(it.accum_like_num || 0),
          accum_collect_num: Number(it.accum_collect_num || 0),
          accum_comment_num: Number(it.accum_comment_num || 0),
          accum_share_num: Number(it.accum_share_num || 0),
          stage_top500_ranks: {
            recall: Number(it?.stage_ranks?.recall || 0) || '-',
            preranking: Number(it?.stage_ranks?.preranking || 0) || '-',
            ranking: Number(it?.stage_ranks?.ranking || 0) || '-',
            rerank: Number((it?.stage_ranks?.rerank ?? it?.stage_ranks?.ranking ?? 0)) || '-',
          },
          topic_tokens: Array.isArray(it?.topic_tokens) ? it.topic_tokens : [],
        },
        savedAt: Date.now(),
      })
    );
  } catch {
  }
  const homeRank = Number(it?.stage_ranks?.rerank ?? it?.stage_ranks?.ranking ?? 0);
  const rankArg = Number.isFinite(homeRank) && homeRank > 0 ? `&home_rank=${homeRank}` : '';
  router.push(`/detail?scene=${it.scene}&user_idx=${it.user_idx}&request_id=${it.request_id}&note_idx=${it.note_idx}&query=${encodeURIComponent(query.value || '')}${rankArg}`);
}

onMounted(async () => {
  const refreshHome = String(route.query.refresh || '') !== '';
  const restored = !refreshHome ? restoreHomeCache() : false;
  if (!restored) {
    if (refreshHome) {
      query.value = '';
      refreshSeenNoteIds.value = [];
      await load(true, true);
    } else {
      await load(true);
    }
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
  if (suggestTimer.value != null) {
    window.clearTimeout(suggestTimer.value);
  }
  if (observer) {
    observer.disconnect();
    observer = null;
  }
});

watch(
  () => route.query.refresh,
  async (val, prev) => {
    if (val && val !== prev) {
      await refreshFromBrand();
    }
  }
);
</script>

<template>
  <TopBar :title="pageTitle" :latency-ms="latencyMs" :stage-ms="stageMs" :scene-mode="currentScene" @home="returnToRecommendationHome">
    <template #actions>
      <div class="home-search-row">
        <div class="search-shell">
          <input
            v-model="query"
            class="search-input"
            :placeholder="copy.searchPlaceholder"
            @focus="suggestOpen = suggestItems.length > 0"
            @blur="closeSuggest"
            @keyup.enter="load(true)"
          />
          <div v-if="suggestOpen && suggestItems.length" class="suggest-panel">
            <button
              v-for="item in suggestItems"
              :key="item.text"
              class="suggest-item"
              @mousedown.prevent="pickSuggest(item.text)"
            >
              <span class="suggest-item-main">{{ item.text }}</span>
              <span class="suggest-item-meta">{{ item.hint }}</span>
            </button>
          </div>
        </div>
        <div class="home-search-actions">
          <button v-if="isRecMode()" class="btn" :disabled="loading" @click="refreshRec">{{ copy.refreshBatch }}</button>
          <button class="btn-primary" :disabled="loading" @click="load(true)">{{ loading ? copy.loading : copy.refresh }}</button>
        </div>
      </div>
    </template>
  </TopBar>

  <main class="container">
    <div v-if="errorMsg" class="panel" style="margin-bottom:10px;">{{ errorMsg }}</div>
    <section class="grid">
      <article v-for="it in items" :key="`${it.request_id}-${it.note_idx}`" class="card" @click="openDetail(it)">
        <img v-if="hasCover(it)" :src="api.imageUrl(it.cover_image)" alt="cover" loading="lazy" @error="markImageBroken(it)" />
        <div v-else class="media-placeholder card-placeholder">
          <div class="media-placeholder-mark">&#x1F5BC;</div>
          <div class="media-placeholder-title">{{ it.title || copy.noImage }}</div>
          <div class="media-placeholder-tags">
            <span v-for="tag in titleTags(it)" :key="`${it.note_idx}-${tag}`">{{ tag }}</span>
          </div>
        </div>
        <div class="card-body">
          <div class="card-title">{{ it.title }}</div>
          <div class="stats">
            <span class="stat-chip stat-like"><i>&#x2665;</i>{{ fmtNum(it.accum_like_num) }}</span>
            <span class="stat-chip stat-collect"><i>&#x2605;</i>{{ fmtNum(it.accum_collect_num) }}</span>
            <span class="stat-chip stat-comment"><i>&#x1F4AC;</i>{{ fmtNum(it.accum_comment_num) }}</span>
          </div>
        </div>
      </article>
    </section>
    <div ref="loadAnchor" style="height:24px;display:flex;align-items:center;justify-content:center;margin-top:10px;" class="meta">
      {{ loading ? copy.loading : (isRecMode() ? copy.loadMoreRec : copy.loadMoreSearch) }}
    </div>
  </main>
</template>

<style scoped>
.home-search-row {
  width: min(920px, 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12px;
}

.search-shell {
  position: relative;
  flex: 1 1 auto;
  min-width: 0;
}

.home-search-actions {
  display: flex;
  align-items: center;
  gap: 12px;
  flex: 0 0 auto;
}

.suggest-panel {
  position: absolute;
  top: calc(100% + 10px);
  left: 0;
  right: 0;
  display: grid;
  gap: 6px;
  padding: 10px;
  border-radius: 18px;
  border: 1px solid rgba(36, 44, 56, 0.10);
  background: linear-gradient(180deg, rgba(255, 252, 246, 0.98), rgba(248, 243, 234, 0.96));
  box-shadow: 0 20px 42px rgba(24, 33, 43, 0.14);
  backdrop-filter: blur(16px);
}

.suggest-item {
  display: grid;
  gap: 3px;
  justify-content: flex-start;
  text-align: left;
  padding: 11px 14px;
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.82);
  border: 1px solid rgba(36, 44, 56, 0.06);
  color: #141a22;
}

.suggest-item:hover {
  background: rgba(255, 255, 255, 0.96);
}

.suggest-item-main {
  font-size: 14px;
  font-weight: 700;
  color: #141a22;
}

.suggest-item-meta {
  font-size: 11px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #7b6d5f;
}

.card-placeholder {
  aspect-ratio: 3 / 4;
}

.media-placeholder {
  position: relative;
  display: grid;
  align-content: end;
  gap: 8px;
  padding: 18px 16px;
  color: #22303c;
  background:
    radial-gradient(circle at top right, rgba(255,255,255,0.72), transparent 26%),
    linear-gradient(160deg, rgba(237, 233, 226, 0.98), rgba(226, 220, 210, 0.92));
}

.media-placeholder::after {
  content: '';
  position: absolute;
  inset: 10px;
  border-radius: 18px;
  border: 1px solid rgba(255,255,255,0.50);
  pointer-events: none;
}

.media-placeholder-mark {
  position: absolute;
  top: 18px;
  right: 18px;
  font-size: 28px;
  opacity: 0.42;
}

.media-placeholder-title {
  position: relative;
  z-index: 1;
  font-size: 14px;
  font-weight: 700;
  line-height: 1.4;
}

.media-placeholder-tags {
  position: relative;
  z-index: 1;
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.media-placeholder-tags span {
  padding: 4px 8px;
  border-radius: 999px;
  background: rgba(255,255,255,0.70);
  border: 1px solid rgba(36, 44, 56, 0.08);
  font-size: 11px;
  color: #6b5e4f;
}

.stat-chip {
  display: inline-flex;
  align-items: center;
  gap: 4px;
}

.stat-chip i {
  font-style: normal;
  font-size: 13px;
}

.stat-like i {
  color: #d92d20;
}

.stat-collect i {
  color: #c28a04;
}

.stat-comment i {
  color: #586474;
}

@media (max-width: 900px) {
  .home-search-row {
    flex-direction: column;
    align-items: stretch;
  }

  .home-search-actions {
    width: 100%;
    justify-content: stretch;
  }

  .home-search-actions > * {
    flex: 1 1 0;
  }
}
</style>
