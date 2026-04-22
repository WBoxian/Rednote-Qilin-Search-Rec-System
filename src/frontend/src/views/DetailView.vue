<script setup lang="ts">
import { onMounted, onUnmounted, ref } from 'vue';
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
  const out: any[] = [];
  const seen = new Set<string>();
  for (const row of (rows || [])) {
    const key = `${row?.note_idx}|${row?.request_id}|${row?.query || ''}|${row?.scene || ''}`;
    if (seen.has(key)) continue;
    out.push(row);
    seen.add(key);
    if (out.length >= 20) break;
  }
  return out;
}

function goHomeKeepFeed() {
  router.push('/?preserve=1');
}

function onImgError(e: Event) {
  const img = e.target as HTMLImageElement | null;
  if (!img) return;
  img.style.display = 'none';
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

function moveImage(step: number) {
  const arr = imageList();
  if (!arr.length) return;
  const n = arr.length;
  imageIdx.value = (imageIdx.value + step + n) % n;
}

function onImageWheel(e: WheelEvent) {
  if (!detail.value || imageList().length <= 1) return;
  if (e.deltaY > 0) moveImage(1);
  else if (e.deltaY < 0) moveImage(-1);
}

function openFullImage() {
  if (!curImage()) return;
  fullImage.value = true;
}

function closeFullImage() {
  fullImage.value = false;
}

onMounted(async () => {
  // 从路由 query 还原上下文并请求详情
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
  const [noteRes, userRes] = await Promise.allSettled([
    api.note(scene, userIdx, requestId, noteIdx, query),
    api.user(scene, userIdx),
  ]);
  if (noteRes.status === 'fulfilled') {
    detail.value = noteRes.value.detail;
    imageIdx.value = 0;
    fullImage.value = false;
    if (detail.value && homeRank > 0) {
      detail.value.stage_top500_ranks = {
        ...(detail.value.stage_top500_ranks || {}),
          ranking: homeRank,
      };
      detail.value.stage_rank_note = '重排 Rank 显示的是你在首页点击该内容时看到的位次。';
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
    await api.engage(
      sceneRef.value,
      Number(userIdxRef.value),
      Number(noteIdxRef.value),
      Number(requestIdRef.value),
      queryRef.value,
      { pageTime: dwellSec }
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
  <TopBar title="小红书麒麟搜推系统项目" :latency-ms="latencyMs" />
  <main class="container" v-if="detail">
    <section class="panel">
      <div style="display:flex;justify-content:flex-start;margin-bottom:8px;">
        <button class="btn" @click="goHomeKeepFeed">返回首页</button>
      </div>
      <div class="meta">scene={{ detail.scene }} | request_id={{ detail.request_id }} | note_idx={{ detail.note_idx }}</div>
      <div class="img-pager" @wheel.prevent="onImageWheel">
        <button class="btn" :disabled="imageList().length <= 1" @click="moveImage(-1)"><</button>
        <div class="img-stage" @click="openFullImage">
          <img v-if="curImage()" :src="api.imageUrl(curImage())" alt="img" loading="lazy" @error="onImgError" />
          <div v-else class="meta">暂无图片</div>
        </div>
        <button class="btn" :disabled="imageList().length <= 1" @click="moveImage(1)">></button>
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
      <div class="note-title">{{ detail.title }}</div>
      <div class="note-content">{{ detail.content }}</div>
    </section>
    <section class="panel">
      <h3>三阶段 Top500 位次</h3>
      <table class="table">
        <tbody>
          <tr><td>召回 Rank</td><td>{{ detail.stage_top500_ranks?.recall ?? '-' }}</td></tr>
          <tr><td>粗排 Rank</td><td>{{ detail.stage_top500_ranks?.preranking ?? '-' }}</td></tr>
          <tr><td>精排 Rank</td><td>{{ detail.stage_top500_ranks?.ranking ?? '-' }}</td></tr>
          <tr><td>重排 Rank(去重后)</td><td>{{ detail.stage_top500_ranks?.ranking ?? '-' }}</td></tr>
        </tbody>
      </table>
      <div class="meta" style="margin-top:8px;">{{ detail.stage_rank_note }}</div>
      <div class="action-row">
        <button class="action-btn" :class="{ active: actionState.like }" @click="engage('like')">
          <span class="action-icon">{{ actionState.like ? '♥' : '♡' }}</span>
          <span class="action-text">点赞</span>
          <span class="action-count">{{ actionCount('like') }}</span>
        </button>
        <button class="action-btn collect" :class="{ active: actionState.collect }" @click="engage('collect')">
          <span class="action-icon">{{ actionState.collect ? '★' : '☆' }}</span>
          <span class="action-text">收藏</span>
          <span class="action-count">{{ actionCount('collect') }}</span>
        </button>
        <button class="action-btn comment" :class="{ active: actionState.comment }" @click="engage('comment')">
          <span class="action-icon">{{ actionState.comment ? '●' : '◌' }}</span>
          <span class="action-text">评论</span>
          <span class="action-count">{{ actionCount('comment') }}</span>
        </button>
        <button class="action-btn share" :class="{ active: actionState.share }" @click="engage('share')">
          <span class="action-icon">{{ actionState.share ? '➜' : '➚' }}</span>
          <span class="action-text">分享</span>
          <span class="action-count">{{ actionCount('share') }}</span>
        </button>
      </div>
    </section>
    <section class="panel" v-if="currentUser">
      <h3 style="margin:0 0 10px;">当前用户最近行为（20条）</h3>
      <table class="table" v-if="dedupBehaviors(currentUser.recent_behaviors || []).length">
        <tbody>
          <tr><td style="width:62%;">行为明细</td><td>对应标题</td></tr>
          <tr v-for="(row, i) in dedupBehaviors(currentUser.recent_behaviors || [])" :key="`${row.ts}-${row.note_idx}-${row.action}-${i}`">
            <td>
              <div>时间：{{ fmtTime(row.ts) }}</div>
              <div>scene：{{ fmt(row.scene) }}</div>
              <div>request_id：{{ fmt(row.request_id) }}</div>
              <div>query：{{ fmt(row.query) }}</div>
              <div>note_idx：{{ fmt(row.note_idx) }}</div>
              <div>互动分：{{ fmt(row.interaction_score) }}</div>
            </td>
            <td>{{ fmt(row.title) }}</td>
          </tr>
        </tbody>
      </table>
      <div v-else class="meta">暂无实时行为记录。</div>
    </section>
  </main>

  <div v-if="fullImage" class="full-mask" @click.self="closeFullImage" @wheel.prevent="onImageWheel">
    <button class="btn" style="position:absolute;top:16px;right:16px;z-index:2;" @click="closeFullImage">关闭</button>
    <button class="btn" style="position:absolute;left:16px;top:50%;transform:translateY(-50%);z-index:2;" :disabled="imageList().length<=1" @click="moveImage(-1)"><</button>
    <img v-if="curImage()" class="full-img" :src="api.imageUrl(curImage())" alt="full" @error="onImgError" />
    <button class="btn" style="position:absolute;right:16px;top:50%;transform:translateY(-50%);z-index:2;" :disabled="imageList().length<=1" @click="moveImage(1)">></button>
    <div class="meta" style="position:absolute;bottom:16px;left:50%;transform:translateX(-50%);color:#fff;">{{ imageList().length ? `${imageIdx + 1}/${imageList().length}` : '0/0' }}</div>
  </div>
</template>

<style scoped>
.img-pager {
  display: grid;
  grid-template-columns: auto 1fr auto;
  gap: 10px;
  align-items: center;
}

.img-stage {
  min-height: 420px;
  background: #111;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  cursor: zoom-in;
}

.img-stage img {
  width: 100%;
  max-height: 560px;
  object-fit: contain;
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

.action-row {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  margin-top: 14px;
}

.action-btn {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 6px;
  min-height: 88px;
  border-radius: 18px;
  border: 1px solid #e5e7eb;
  background: #fff;
  color: #374151;
  transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease, background .18s ease;
}

.action-btn:hover {
  transform: translateY(-1px);
  box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
}

.action-btn.active {
  border-color: transparent;
  background: linear-gradient(135deg, #fff1f2, #ffe4e6);
  color: #e11d48;
}

.action-btn.collect.active {
  background: linear-gradient(135deg, #fff7ed, #ffedd5);
  color: #ea580c;
}

.action-btn.comment.active {
  background: linear-gradient(135deg, #eff6ff, #dbeafe);
  color: #2563eb;
}

.action-btn.share.active {
  background: linear-gradient(135deg, #ecfeff, #cffafe);
  color: #0891b2;
}

.action-icon {
  font-size: 28px;
  line-height: 1;
}

.action-text {
  font-size: 14px;
  font-weight: 700;
}

.action-count {
  font-size: 12px;
  color: inherit;
  opacity: 0.9;
}
</style>
