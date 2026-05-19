<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue';
import { useRoute, useRouter } from 'vue-router';
import TopBar from '../components/TopBar.vue';
import { api } from '../services/api';

const route = useRoute();
const router = useRouter();
const detail = ref<any>(null);
const currentUser = ref<any>(null);
const latencyMs = ref<number | null>(null);
const sceneRef = ref('search');
const userIdxRef = ref(0);
const requestIdRef = ref(0);
const noteIdxRef = ref(0);
const queryRef = ref('');
const enterAtMs = ref(0);
const actionState = ref({ like: false, collect: false, comment: false, share: false });
const imageIdx = ref(0);
const fullImage = ref(false);
const brokenImages = ref<Record<number, boolean>>({});
const imageOrientations = ref<Record<number, 'landscape' | 'portrait'>>({});
const DETAIL_SEED_KEY = 'qilin_detail_seed';
const pageTitle = 'Qilin \u5E16\u5B50\u8BE6\u60C5';
const copy = {
  noTitle: '\uFF08\u65E0\u6807\u9898\uFF09',
  noImage: '\u6682\u65E0\u56FE\u7247',
  noContent: '\u6682\u65E0\u5185\u5BB9',
  backHome: '\u8FD4\u56DE\u9996\u9875',
  close: '\u5173\u95ED',
  rankNote: '\u6700\u7EC8 Rank \u7EE7\u627F\u9996\u9875\u6362\u4E00\u6279\u4E4B\u540E\u7684\u5C55\u793A\u987A\u5E8F',
};

function fmt(v: any) {
  if (typeof v === 'number') return Number.isInteger(v) ? String(v) : v.toFixed(4);
  if (v == null || v === '') return '-';
  return String(v);
}

function fmtTime(ts: any) {
  const n = Number(ts);
  if (!Number.isFinite(n) || n <= 0) return '-';
  const ms = n > 1e12 ? n : n * 1000;
  const d = new Date(ms);
  if (Number.isNaN(d.getTime())) return '-';
  const pad = (x: number) => String(x).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}

function dedupBehaviors(rows: any[]): any[] {
  const merged = new Map<string, any>();
  for (const row of rows || []) {
    const key = String(row?.note_idx ?? '');
    const prev = merged.get(key);
    if (!prev) {
      merged.set(key, row);
      continue;
    }
    const prevTs = Number(prev?.ts || 0);
    const nextTs = Number(row?.ts || 0);
    if (nextTs >= prevTs) {
      merged.set(key, { ...prev, ...row });
    }
  }
  return Array.from(merged.values())
    .sort((a, b) => Number(b?.ts || 0) - Number(a?.ts || 0))
    .slice(0, 40);
}

function goHomeKeepFeed() {
  router.push('/?preserve=1');
}

function imageList() {
  return Array.isArray(detail.value?.images) ? detail.value.images : [];
}

function curImage() {
  const arr = imageList();
  if (!arr.length) return '';
  const idx = Math.max(0, Math.min(imageIdx.value, arr.length - 1));
  return arr[idx] || '';
}

const currentImageFrameClass = computed(() => imageOrientations.value[imageIdx.value] || 'landscape');

function hasCurrentImage() {
  return Boolean(curImage()) && !brokenImages.value[imageIdx.value];
}

function currentPlaceholderTags() {
  return Array.isArray(detail.value?.topic_tokens) ? detail.value.topic_tokens.slice(0, 4) : [];
}

function onImgError() {
  brokenImages.value = { ...brokenImages.value, [imageIdx.value]: true };
}

function onImgLoad(event: Event) {
  const img = event.target as HTMLImageElement | null;
  if (!img) return;
  imageOrientations.value = {
    ...imageOrientations.value,
    [imageIdx.value]: img.naturalWidth >= img.naturalHeight ? 'landscape' : 'portrait',
  };
}

function moveImage(step: number) {
  const arr = imageList();
  if (!arr.length) return;
  const n = arr.length;
  imageIdx.value = (imageIdx.value + step + n) % n;
}

function onImageWheel(e: WheelEvent) {
  if (!fullImage.value || imageList().length <= 1) return;
  if (e.deltaY > 0) moveImage(1);
  else if (e.deltaY < 0) moveImage(-1);
}

function openFullImage() {
  if (!hasCurrentImage()) return;
  fullImage.value = true;
}

function closeFullImage() {
  fullImage.value = false;
}

onMounted(async () => {
  const scene = String(route.query.scene || 'search');
  const userIdx = Number(route.query.user_idx || 0);
  const requestId = Number(route.query.request_id || 0);
  const noteIdx = Number(route.query.note_idx || 0);
  const query = String(route.query.query || '');
  const homeRank = Number(route.query.home_rank || 0);
  sceneRef.value = scene;
  userIdxRef.value = userIdx;
  requestIdRef.value = requestId;
  noteIdxRef.value = noteIdx;
  queryRef.value = query;
  enterAtMs.value = Date.now();
  try {
    const raw = sessionStorage.getItem(DETAIL_SEED_KEY);
    if (raw) {
      const seed = JSON.parse(raw);
      if (
        String(seed?.scene || '') === scene
        && Number(seed?.user_idx || -1) === userIdx
        && Number(seed?.request_id || -1) === requestId
        && Number(seed?.note_idx || -1) === noteIdx
        && Date.now() - Number(seed?.savedAt || 0) <= 10 * 60 * 1000
      ) {
        detail.value = seed.detail || null;
      }
    }
  } catch {
  }
  const [noteRes, userRes] = await Promise.allSettled([
    api.note(scene, userIdx, requestId, noteIdx, query),
    api.user(scene, userIdx),
  ]);
  if (noteRes.status === 'fulfilled') {
    detail.value = noteRes.value.detail;
    imageIdx.value = 0;
    brokenImages.value = {};
    imageOrientations.value = {};
    fullImage.value = false;
    if (detail.value && homeRank > 0) {
      detail.value.stage_top500_ranks = {
        ...(detail.value.stage_top500_ranks || {}),
        rerank: homeRank,
      };
      detail.value.stage_rank_note = copy.rankNote;
    }
    latencyMs.value = Number(noteRes.value?.latency_ms ?? 0);
  }
  if (userRes.status === 'fulfilled') {
    currentUser.value = userRes.value.user;
  }
});

onUnmounted(async () => {
  try {
    const dwellSec = Math.max(0, (Date.now() - Number(enterAtMs.value || Date.now())) / 1000);
    const hasAction = Object.values(actionState.value).some(Boolean);
    if (!hasAction && dwellSec < 1) return;
    await api.engage(
      sceneRef.value,
      Number(userIdxRef.value),
      Number(noteIdxRef.value),
      Number(requestIdRef.value),
      queryRef.value,
      {
        like: actionState.value.like ? 1 : 0,
        collect: actionState.value.collect ? 1 : 0,
        comment: actionState.value.comment ? 1 : 0,
        share: actionState.value.share ? 1 : 0,
        pageTime: dwellSec,
      }
    );
  } catch {
  }
});

async function engage(kind: 'like' | 'collect' | 'comment' | 'share') {
  const prev = { ...actionState.value };
  const nextFlag = !prev[kind];
  actionState.value = { ...prev, [kind]: nextFlag };
  if (detail.value) {
    const delta = nextFlag ? 1 : -1;
    if (kind === 'like') detail.value.accum_like_num = Math.max(0, Number(detail.value.accum_like_num || 0) + delta);
    if (kind === 'collect') detail.value.accum_collect_num = Math.max(0, Number(detail.value.accum_collect_num || 0) + delta);
    if (kind === 'comment') detail.value.accum_comment_num = Math.max(0, Number(detail.value.accum_comment_num || 0) + delta);
    if (kind === 'share') detail.value.accum_share_num = Math.max(0, Number(detail.value.accum_share_num || 0) + delta);
  }
  const payload: any = {
    like: actionState.value.like ? 1 : 0,
    collect: actionState.value.collect ? 1 : 0,
    comment: actionState.value.comment ? 1 : 0,
    share: actionState.value.share ? 1 : 0,
    pageTime: 0,
  };
  try {
    await api.engage(
      sceneRef.value,
      Number(userIdxRef.value),
      Number(noteIdxRef.value),
      Number(requestIdRef.value),
      queryRef.value,
      payload,
    );
    try {
      const ures = await api.user(sceneRef.value, Number(userIdxRef.value));
      currentUser.value = ures.user;
    } catch {
    }
  } catch {
    actionState.value = prev;
    if (detail.value) {
      const delta = nextFlag ? -1 : 1;
      if (kind === 'like') detail.value.accum_like_num = Math.max(0, Number(detail.value.accum_like_num || 0) + delta);
      if (kind === 'collect') detail.value.accum_collect_num = Math.max(0, Number(detail.value.accum_collect_num || 0) + delta);
      if (kind === 'comment') detail.value.accum_comment_num = Math.max(0, Number(detail.value.accum_comment_num || 0) + delta);
      if (kind === 'share') detail.value.accum_share_num = Math.max(0, Number(detail.value.accum_share_num || 0) + delta);
    }
  }
}

function actionCount(kind: 'like' | 'collect' | 'comment' | 'share') {
  if (!detail.value) return 0;
  if (kind === 'like') return Number(detail.value.accum_like_num || 0);
  if (kind === 'collect') return Number(detail.value.accum_collect_num || 0);
  if (kind === 'comment') return Number(detail.value.accum_comment_num || 0);
  return Number(detail.value.accum_share_num || 0);
}
</script>

<template>
  <TopBar :title="pageTitle" :latency-ms="latencyMs" />
  <main class="container" v-if="detail">
    <section class="panel detail-panel detail-panel-hero">
      <div style="display:flex;justify-content:flex-start;margin-bottom:8px;">
        <button class="btn" @click="goHomeKeepFeed">{{ copy.backHome }}</button>
      </div>
      <div class="meta">scene={{ detail.scene }} | request_id={{ detail.request_id }} | note_idx={{ detail.note_idx }}</div>
      <div class="img-pager">
        <button class="btn img-nav-btn" :disabled="imageList().length <= 1" @click="moveImage(-1)"><</button>
        <div class="img-stage-shell">
          <div class="img-stage" :class="currentImageFrameClass" @click="openFullImage">
            <img v-if="hasCurrentImage()" :src="api.imageUrl(curImage())" alt="img" loading="lazy" @error="onImgError" @load="onImgLoad" />
            <div v-else class="media-placeholder detail-placeholder">
              <div class="media-placeholder-mark">&#x1F5BC;</div>
              <div class="media-placeholder-title">{{ detail.title || copy.noImage }}</div>
              <div class="media-placeholder-tags">
                <span v-for="tag in currentPlaceholderTags()" :key="`${detail.note_idx}-${tag}`">{{ tag }}</span>
              </div>
            </div>
          </div>
        </div>
        <button class="btn img-nav-btn" :disabled="imageList().length <= 1" @click="moveImage(1)">></button>
      </div>
      <div class="meta" style="margin-top:8px;">{{ imageList().length ? `${imageIdx + 1}/${imageList().length}` : '0/0' }}</div>
      <div class="thumb-pages" v-if="imageList().length">
        <button
          v-for="(img, idx) in imageList()"
          :key="`thumb-${img}-${idx}`"
          class="thumb-btn"
          :class="{ active: idx === imageIdx }"
          @click="imageIdx = Number(idx)"
        >
          {{ Number(idx) + 1 }}
        </button>
      </div>
      <div class="note-title">{{ detail.title || copy.noTitle }}</div>
      <div class="note-content">{{ detail.content || copy.noContent }}</div>
    </section>
    <section class="panel detail-panel">
      <h3>&#x5404;&#x9636;&#x6BB5; Top500 &#x6392;&#x4F4D;</h3>
      <table class="table">
        <tbody>
          <tr><td>&#x53EC;&#x56DE; Rank</td><td>{{ detail.stage_top500_ranks?.recall ?? '-' }}</td></tr>
          <tr><td>&#x7C97;&#x6392; Rank</td><td>{{ detail.stage_top500_ranks?.preranking ?? '-' }}</td></tr>
          <tr><td>&#x7CBE;&#x6392; Rank</td><td>{{ detail.stage_top500_ranks?.ranking ?? '-' }}</td></tr>
          <tr><td>&#x6700;&#x7EC8; Rank(&#x9996;&#x9875;&#x5C55;&#x793A;)</td><td>{{ detail.stage_top500_ranks?.rerank ?? '-' }}</td></tr>
        </tbody>
      </table>
      <div class="meta" style="margin-top:8px;">{{ detail.stage_rank_note }}</div>
      <div class="action-row">
        <button class="action-btn action-like" :class="{ active: actionState.like }" @click="engage('like')">
          <span class="action-icon">&#x2665;</span>
          <span class="action-text">&#x70B9;&#x8D5E;</span>
          <span class="action-count">{{ actionCount('like') }}</span>
        </button>
        <button class="action-btn action-collect" :class="{ active: actionState.collect }" @click="engage('collect')">
          <span class="action-icon">&#x2605;</span>
          <span class="action-text">&#x6536;&#x85CF;</span>
          <span class="action-count">{{ actionCount('collect') }}</span>
        </button>
        <button class="action-btn action-comment" :class="{ active: actionState.comment }" @click="engage('comment')">
          <span class="action-icon">&#x1F4AC;</span>
          <span class="action-text">&#x8BC4;&#x8BBA;</span>
          <span class="action-count">{{ actionCount('comment') }}</span>
        </button>
        <button class="action-btn action-share" :class="{ active: actionState.share }" @click="engage('share')">
          <span class="action-icon">&#x27A6;</span>
          <span class="action-text">&#x5206;&#x4EAB;</span>
          <span class="action-count">{{ actionCount('share') }}</span>
        </button>
      </div>
    </section>
    <section class="panel detail-panel" v-if="currentUser">
      <h3 style="margin:0 0 10px;">&#x7528;&#x6237;&#x6700;&#x8FD1;&#x53BB;&#x91CD;&#x884C;&#x4E3A; 40 &#x6761;</h3>
      <table class="table" v-if="dedupBehaviors(currentUser.recent_behaviors || []).length">
        <tbody>
          <tr><td style="width:62%;">&#x884C;&#x4E3A;&#x4FE1;&#x606F;</td><td>&#x6807;&#x9898;</td></tr>
          <tr v-for="(row, i) in dedupBehaviors(currentUser.recent_behaviors || [])" :key="`${row.scene}-${row.request_id}-${row.note_idx}-${i}`">
            <td>
              <div>&#x65F6;&#x95F4;&#xFF1A;{{ fmtTime(row.ts) }}</div>
              <div>scene&#xFF1A;{{ fmt(row.scene) }}</div>
              <div>request_id&#xFF1A;{{ fmt(row.request_id) }}</div>
              <div>query&#xFF1A;{{ fmt(row.query) }}</div>
              <div>note_idx&#xFF1A;{{ fmt(row.note_idx) }}</div>
              <div>&#x4E92;&#x52A8;&#x5206;&#xFF1A;{{ fmt(row.interaction_score) }}</div>
            </td>
            <td>{{ fmt(row.title) }}</td>
          </tr>
        </tbody>
      </table>
      <div v-else class="meta">&#x6682;&#x65E0;&#x6700;&#x8FD1;&#x884C;&#x4E3A;&#x6570;&#x636E;</div>
    </section>
  </main>

  <div v-if="fullImage" class="full-mask" @click.self="closeFullImage" @wheel.prevent="onImageWheel">
    <button class="btn" style="position:absolute;top:16px;right:16px;z-index:2;" @click="closeFullImage">{{ copy.close }}</button>
    <button class="btn img-nav-btn" style="position:absolute;left:16px;top:50%;transform:translateY(-50%);z-index:2;" :disabled="imageList().length<=1" @click="moveImage(-1)"><</button>
    <img v-if="hasCurrentImage()" class="full-img" :src="api.imageUrl(curImage())" alt="full" @error="onImgError" />
    <button class="btn img-nav-btn" style="position:absolute;right:16px;top:50%;transform:translateY(-50%);z-index:2;" :disabled="imageList().length<=1" @click="moveImage(1)">></button>
    <div class="meta" style="position:absolute;bottom:16px;left:50%;transform:translateX(-50%);color:#fff;">{{ imageList().length ? `${imageIdx + 1}/${imageList().length}` : '0/0' }}</div>
  </div>
</template>

<style scoped>
.detail-panel {
  background:
    linear-gradient(145deg, rgba(248, 250, 252, 0.98), rgba(234, 239, 244, 0.95) 54%, rgba(242, 245, 248, 0.96)),
    radial-gradient(circle at top left, rgba(255, 255, 255, 0.68), transparent 28%),
    radial-gradient(circle at 100% 0%, rgba(189, 205, 222, 0.20), transparent 26%);
  border: 1px solid rgba(86, 102, 121, 0.11);
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.78),
    0 18px 36px rgba(24, 33, 43, 0.07);
}

.detail-panel-hero {
  background:
    linear-gradient(145deg, rgba(246, 249, 251, 0.99), rgba(232, 237, 242, 0.95) 52%, rgba(244, 247, 249, 0.96)),
    radial-gradient(circle at top left, rgba(255, 255, 255, 0.72), transparent 24%),
    radial-gradient(circle at 100% 0%, rgba(194, 210, 228, 0.16), transparent 24%);
}

.img-pager {
  display: grid;
  grid-template-columns: auto 1fr auto;
  gap: 8px;
  align-items: center;
}

.img-nav-btn {
  min-width: 56px;
  min-height: 56px;
  padding: 12px 14px;
  font-size: 22px;
  font-weight: 700;
  border-radius: 18px;
}

.img-stage-shell {
  display: flex;
  justify-content: center;
}

.img-stage {
  width: min(100%, 760px);
  background: transparent;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  cursor: zoom-in;
}

.img-stage.landscape {
  aspect-ratio: 16 / 9;
}

.img-stage.portrait {
  width: min(100%, 420px);
  aspect-ratio: 9 / 16;
}

.img-stage img {
  width: 100%;
  height: 100%;
  border-radius: 16px;
  object-fit: cover;
  box-shadow: 0 18px 42px rgba(24, 33, 43, 0.14);
}

.detail-placeholder {
  width: 100%;
  height: 100%;
  border-radius: 16px;
}

.thumb-pages {
  margin-top: 8px;
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.thumb-btn {
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  background: #fff;
  padding: 4px 8px;
  font-size: 12px;
}

.thumb-btn.active {
  border-color: #111827;
  background: #111827;
  color: #fff;
}

.full-mask {
  position: fixed;
  inset: 0;
  z-index: 1000;
  background: rgba(0, 0, 0, 0.92);
  display: flex;
  align-items: center;
  justify-content: center;
}

.full-img {
  max-width: 92vw;
  max-height: 92vh;
  object-fit: contain;
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
    linear-gradient(160deg, rgba(238, 242, 246, 0.98), rgba(222, 229, 236, 0.92));
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
  font-size: 15px;
  font-weight: 700;
  line-height: 1.45;
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
  color: #5a6876;
}

.note-title {
  margin-top: 14px;
  font-size: clamp(24px, 2.8vw, 34px);
  font-weight: 700;
  line-height: 1.08;
  letter-spacing: -0.04em;
  color: #16212c;
}

.note-content {
  margin-top: 12px;
  white-space: pre-wrap;
  line-height: 1.75;
  color: #31414f;
}

.action-row {
  margin-top: 16px;
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
}

.action-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 12px 14px;
  border-radius: 16px;
  border: 1px solid rgba(36, 44, 56, 0.10);
  background: linear-gradient(180deg, rgba(27, 35, 45, 0.22), rgba(27, 35, 45, 0.12));
  color: rgba(24, 33, 43, 0.42);
}

.action-btn .action-count,
.action-btn .action-icon,
.action-btn .action-text {
  transition: color 0.18s ease, opacity 0.18s ease;
}

.action-btn.active {
  transform: translateY(-1px);
  box-shadow: 0 14px 30px rgba(24, 33, 43, 0.08);
}

.action-like.active {
  color: #b42318;
  background: linear-gradient(135deg, rgba(244, 190, 188, 0.38), rgba(255,255,255,0.96));
}

.action-collect.active {
  color: #8a5a00;
  background: linear-gradient(135deg, rgba(244, 220, 161, 0.46), rgba(255,255,255,0.96));
}

.action-comment.active {
  color: #175cd3;
  background: linear-gradient(135deg, rgba(181, 205, 248, 0.46), rgba(255,255,255,0.96));
}

.action-share.active {
  color: #027a48;
  background: linear-gradient(135deg, rgba(176, 234, 212, 0.46), rgba(255,255,255,0.96));
}

.action-text {
  font-weight: 700;
}

.action-count {
  color: rgba(24, 33, 43, 0.62);
}

@media (max-width: 900px) {
  .action-row {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}
</style>
