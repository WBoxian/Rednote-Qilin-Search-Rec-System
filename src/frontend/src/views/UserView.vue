<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue';
import TopBar from '../components/TopBar.vue';
import { api, getScene, getUserId } from '../services/api';

const userByScene = ref<Record<string, any>>({});
const loading = ref(false);
const errorMsg = ref('');
const latencyMs = ref<number | null>(null);
const detailModal = ref<any>(null);
const detailLoading = ref(false);
const detailError = ref('');
const deletingBatch = ref(false);
const selectedBehaviorKeys = ref<string[]>([]);
const menuVisible = ref(false);
const menuRow = ref<any>(null);
const menuX = ref(0);
const menuY = ref(0);
const pageTitle = 'Qilin \u6700\u8FD1\u884C\u4E3A';
const copy = {
  noTitle: '\uFF08\u65E0\u6807\u9898\uFF09',
  needLogin: '\u5F53\u524D\u672A\u767B\u5F55\u7528\u6237\uFF0C\u65E0\u6CD5\u67E5\u770B\u6700\u8FD1\u884C\u4E3A',
  noBehavior: '\u672A\u83B7\u53D6\u5230\u8BE5\u7528\u6237\u7684\u6700\u8FD1\u884C\u4E3A',
  loadError: '\u52A0\u8F7D\u6700\u8FD1\u884C\u4E3A\u5931\u8D25',
  detailError: '\u52A0\u8F7D\u5E16\u5B50\u8BE6\u60C5\u5931\u8D25',
  deleteError: '\u5220\u9664\u884C\u4E3A\u5931\u8D25',
  batchDelete: '\u6279\u91CF\u5220\u9664',
  cancelSelect: '\u53D6\u6D88\u52FE\u9009',
  selectAll: '\u5168\u9009',
  selectedCount: '\u5DF2\u9009',
  loading: '\u52A0\u8F7D\u4E2D...',
  profile: '\u7528\u6237\u753B\u50CF',
};

function fmt(v: any) {
  if (typeof v === 'number') return Number.isInteger(v) ? String(v) : v.toFixed(4);
  if (v == null || v === '') return '-';
  return String(v);
}

function fmtTitle(v: any) {
  if (v == null) return copy.noTitle;
  const s = String(v).trim();
  if (!s) return copy.noTitle;
  const sl = s.toLowerCase();
  if (sl === 'nan' || sl === 'none' || sl === 'null' || sl === 'undefined') return copy.noTitle;
  return s;
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

function collapseBehaviorRows(rows: any[]) {
  const merged = new Map<string, any>();
  for (const row of rows || []) {
    const key = [
      String(row?.scene || ''),
      String(row?.request_id || ''),
      String(row?.note_idx || ''),
      String(row?.query || ''),
    ].join('|');
    const prev = merged.get(key);
    if (!prev) {
      merged.set(key, row);
      continue;
    }
    const prevScore = Number(prev?.interaction_score || 0);
    const nextScore = Number(row?.interaction_score || 0);
    const prevTs = Number(prev?.ts || 0);
    const nextTs = Number(row?.ts || 0);
    if (nextScore > prevScore || (nextScore === prevScore && nextTs >= prevTs)) {
      merged.set(key, { ...prev, ...row, interaction_score: Math.max(prevScore, nextScore) });
    }
  }
  return Array.from(merged.values()).sort((a, b) => Number(b?.ts || 0) - Number(a?.ts || 0)).slice(0, 40);
}

function behaviorRowKey(row: any) {
  return [
    String(row?.scene || ''),
    String(row?.request_id || ''),
    String(row?.note_idx || ''),
    String(row?.ts || ''),
    String(row?.query || ''),
  ].join('|');
}

async function load() {
  const uid = getUserId();
  if (uid == null) {
    errorMsg.value = copy.needLogin;
    return;
  }
  loading.value = true;
  errorMsg.value = '';
  try {
    const [resSearch, resRec] = await Promise.allSettled([
      api.user('search', uid),
      api.user('rec', uid),
    ]);
    const nextByScene: Record<string, any> = {};
    let totalLatency = 0;
    if (resSearch.status === 'fulfilled' && resSearch.value?.user) {
      nextByScene.search = resSearch.value.user;
      totalLatency += Number(resSearch.value?.latency_ms || 0);
    }
    if (resRec.status === 'fulfilled' && resRec.value?.user) {
      nextByScene.rec = resRec.value.user;
      totalLatency += Number(resRec.value?.latency_ms || 0);
    }
    userByScene.value = nextByScene;
    latencyMs.value = totalLatency;
    selectedBehaviorKeys.value = [];
    if (!Object.keys(nextByScene).length) {
      errorMsg.value = copy.noBehavior;
    }
  } catch (e: any) {
    userByScene.value = {};
    errorMsg.value = e?.message || copy.loadError;
  } finally {
    loading.value = false;
  }
}

const mergedBehaviorList = computed(() => {
  const s = Array.isArray(userByScene.value.search?.recent_behaviors) ? userByScene.value.search.recent_behaviors : [];
  const r = Array.isArray(userByScene.value.rec?.recent_behaviors) ? userByScene.value.rec.recent_behaviors : [];
  return collapseBehaviorRows([...s, ...r]);
});

const selectedBehaviorSet = computed(() => new Set(selectedBehaviorKeys.value));
const selectedBehaviorRows = computed(() =>
  mergedBehaviorList.value.filter((row) => selectedBehaviorSet.value.has(behaviorRowKey(row)))
);
const allSelected = computed(() =>
  mergedBehaviorList.value.length > 0
  && selectedBehaviorRows.value.length === mergedBehaviorList.value.length
);

async function openBehavior(row: any) {
  const uid = getUserId();
  if (uid == null) return;
  detailModal.value = null;
  detailError.value = '';
  detailLoading.value = true;
  try {
    const res = await api.note(
      String(row?.scene || 'search'),
      uid,
      Number(row?.request_id || 0),
      Number(row?.note_idx || 0),
      String(row?.query || ''),
    );
    detailModal.value = res?.detail || null;
  } catch (e: any) {
    detailError.value = e?.message || copy.detailError;
  } finally {
    detailLoading.value = false;
  }
}

async function removeBehavior(row: any) {
  const uid = getUserId();
  if (uid == null) return;
  try {
    await api.deleteBehavior(
      String(row?.scene || 'search'),
      uid,
      Number(row?.note_idx || 0),
      Number(row?.ts || 0),
      Number(row?.request_id || 0),
    );
    await load();
  } catch (e: any) {
    errorMsg.value = e?.message || copy.deleteError;
  }
}

function isSelected(row: any) {
  return selectedBehaviorSet.value.has(behaviorRowKey(row));
}

function toggleSelect(row: any) {
  const key = behaviorRowKey(row);
  if (selectedBehaviorSet.value.has(key)) {
    selectedBehaviorKeys.value = selectedBehaviorKeys.value.filter((x) => x !== key);
    return;
  }
  selectedBehaviorKeys.value = [...selectedBehaviorKeys.value, key];
}

function toggleSelectAll() {
  if (allSelected.value) {
    selectedBehaviorKeys.value = [];
    return;
  }
  selectedBehaviorKeys.value = mergedBehaviorList.value.map((row) => behaviorRowKey(row));
}

async function removeSelectedBehaviors() {
  const uid = getUserId();
  if (uid == null || !selectedBehaviorRows.value.length || deletingBatch.value) return;
  deletingBatch.value = true;
  errorMsg.value = '';
  try {
    await api.deleteBehaviorsBatch(
      uid,
      selectedBehaviorRows.value.map((row) => ({
        scene: String(row?.scene || 'search'),
        note_idx: Number(row?.note_idx || 0),
        ts: Number(row?.ts || 0),
        request_id: Number(row?.request_id || 0),
      })),
    );
    selectedBehaviorKeys.value = [];
    await load();
  } catch (e: any) {
    errorMsg.value = e?.message || copy.deleteError;
  } finally {
    deletingBatch.value = false;
  }
}

function closeDetail() {
  detailModal.value = null;
  detailError.value = '';
}

function openContextMenu(event: MouseEvent, row: any) {
  event.preventDefault();
  menuVisible.value = true;
  menuRow.value = row;
  menuX.value = event.clientX;
  menuY.value = event.clientY;
}

function closeContextMenu() {
  menuVisible.value = false;
  menuRow.value = null;
}

async function confirmDelete() {
  const row = menuRow.value;
  closeContextMenu();
  if (!row) return;
  await removeBehavior(row);
}

function onWindowClick() {
  closeContextMenu();
}

function onWindowKey(event: KeyboardEvent) {
  if (event.key === 'Escape') closeContextMenu();
}

onMounted(() => {
  load();
  window.addEventListener('click', onWindowClick);
  window.addEventListener('keydown', onWindowKey);
});

onUnmounted(() => {
  window.removeEventListener('click', onWindowClick);
  window.removeEventListener('keydown', onWindowKey);
});
</script>

<template>
  <TopBar :title="pageTitle" :latency-ms="latencyMs" />
  <main class="container">
    <section class="panel shell-panel" style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;">
      <div class="meta">{{ loading ? copy.loading : copy.profile }}</div>
      <div class="meta">scene={{ getScene() }} | user={{ getUserId() ?? '-' }}</div>
    </section>

    <section v-if="errorMsg" class="panel shell-panel">{{ errorMsg }}</section>

    <template v-else-if="Object.keys(userByScene).length">
      <section class="panel shell-panel">
        <div class="meta" style="margin-bottom:10px;">
          request_count(search)={{ userByScene.search?.request_count_in_test ?? '-' }}
          | request_count(rec)={{ userByScene.rec?.request_count_in_test ?? '-' }}
        </div>
        <table class="table">
          <tbody>
            <tr>
              <td style="font-weight:700;">gender</td><td>{{ fmt((userByScene.search || userByScene.rec)?.features?.gender) }}</td>
              <td style="font-weight:700;">age</td><td>{{ fmt((userByScene.search || userByScene.rec)?.features?.age) }}</td>
            </tr>
            <tr>
              <td style="font-weight:700;">platform</td><td>{{ fmt((userByScene.search || userByScene.rec)?.features?.platform) }}</td>
              <td style="font-weight:700;">location</td><td>{{ fmt((userByScene.search || userByScene.rec)?.features?.location) }}</td>
            </tr>
            <tr>
              <td style="font-weight:700;">fans_num</td><td>{{ fmt((userByScene.search || userByScene.rec)?.features?.fans_num) }}</td>
              <td style="font-weight:700;">follows_num</td><td>{{ fmt((userByScene.search || userByScene.rec)?.features?.follows_num) }}</td>
            </tr>
          </tbody>
        </table>
      </section>

      <section class="panel shell-panel">
        <div class="behavior-toolbar">
          <h3 style="margin:0;">&#x6700;&#x8FD1; 40 &#x6761;&#x884C;&#x4E3A;</h3>
          <div class="behavior-actions">
            <span class="meta">{{ copy.selectedCount }} {{ selectedBehaviorRows.length }}</span>
            <button class="btn btn-light" type="button" @click="toggleSelectAll">
              {{ allSelected ? copy.cancelSelect : copy.selectAll }}
            </button>
            <button
              class="btn-danger"
              type="button"
              :disabled="!selectedBehaviorRows.length || deletingBatch"
              @click="removeSelectedBehaviors"
            >
              {{ deletingBatch ? copy.loading : copy.batchDelete }}
            </button>
          </div>
        </div>
        <div class="meta" style="margin-bottom:10px;">&#x5DF2;&#x6309; scene / request / note / query &#x805A;&#x5408;&#x53BB;&#x91CD;&#xFF0C;&#x91CD;&#x590D;&#x66DD;&#x5149;&#x53EA;&#x4FDD;&#x7559;&#x4E92;&#x52A8;&#x66F4;&#x5F3A;&#x6216;&#x65F6;&#x95F4;&#x66F4;&#x8FD1;&#x7684;&#x4E00;&#x6761;&#x3002;</div>
        <table class="table">
          <tbody>
            <tr>
              <td style="width:6%;">&#x9009;</td>
              <td style="width:10%;">&#x573A;&#x666F;</td>
              <td style="width:22%;">&#x65F6;&#x95F4; / rid</td>
              <td style="width:18%;">query / note</td>
              <td style="width:34%;">&#x6807;&#x9898;</td>
              <td style="width:10%;">&#x5206;&#x503C;</td>
            </tr>
            <tr
              v-for="(row, idx) in mergedBehaviorList"
              :key="`${row.scene}-${row.request_id}-${row.note_idx}-${idx}`"
              class="behavior-row"
              :class="{ selected: isSelected(row) }"
              @click="openBehavior(row)"
              @contextmenu="openContextMenu($event, row)"
            >
              <td @click.stop>
                <input
                  class="behavior-checkbox"
                  type="checkbox"
                  :checked="isSelected(row)"
                  @change="toggleSelect(row)"
                />
              </td>
              <td><span class="scene-pill" :class="row.scene === 'search' ? 'scene-search' : 'scene-rec'">{{ String(row.scene || '-').toUpperCase() }}</span></td>
              <td>
                <div>{{ fmtTime(row.ts) }}</div>
                <div class="meta">rid={{ fmt(row.request_id) }}</div>
              </td>
              <td>
                <div v-if="row.query">query={{ fmt(row.query) }}</div>
                <div>note={{ fmt(row.note_idx) }}</div>
              </td>
              <td>{{ fmtTitle(row.title) }}</td>
              <td>{{ fmt(row.interaction_score) }}</td>
            </tr>
          </tbody>
        </table>
      </section>
    </template>

    <section v-else-if="!loading" class="panel shell-panel empty-state">
      <div class="meta">&#x6682;&#x65E0;&#x53EF;&#x5C55;&#x793A;&#x7684;&#x7528;&#x6237;&#x4FE1;&#x606F;</div>
    </section>
  </main>

  <div v-if="detailModal || detailLoading || detailError" class="detail-mask" @click.self="closeDetail">
    <div class="detail-modal">
      <div class="detail-head">
        <div>
          <div class="meta">&#x5E16;&#x5B50;&#x8BE6;&#x60C5;</div>
          <h3>{{ detailModal?.title || copy.noTitle }}</h3>
        </div>
        <button class="btn" @click="closeDetail">&#x5173;&#x95ED;</button>
      </div>
      <div v-if="detailLoading" class="meta">{{ copy.loading }}</div>
      <div v-else-if="detailError" class="meta">{{ detailError }}</div>
      <template v-else-if="detailModal">
        <img
          v-if="Array.isArray(detailModal.images) && detailModal.images.length"
          class="detail-cover"
          :src="api.imageUrl(detailModal.images[0])"
          alt="detail"
        />
        <div class="detail-content">{{ detailModal.content || '暂无内容' }}</div>
        <div class="detail-stats">
          <span>&#x8D5E; {{ fmt(detailModal.accum_like_num) }}</span>
          <span>&#x85CF; {{ fmt(detailModal.accum_collect_num) }}</span>
          <span>&#x8BC4; {{ fmt(detailModal.accum_comment_num) }}</span>
        </div>
      </template>
    </div>
  </div>

  <div
    v-if="menuVisible && menuRow"
    class="context-menu"
    :style="{ left: `${menuX}px`, top: `${menuY}px` }"
    @click.stop
  >
    <div class="context-menu-head">&#x5220;&#x9664;&#x884C;&#x4E3A;</div>
    <div class="context-menu-note">scene={{ String(menuRow.scene || '-').toUpperCase() }} · note={{ fmt(menuRow.note_idx) }}</div>
    <button class="btn-danger" @click="confirmDelete">&#x786E;&#x8BA4;&#x5220;&#x9664;</button>
  </div>
</template>

<style scoped>
.shell-panel {
  background:
    linear-gradient(135deg, rgba(255, 255, 255, 0.34), rgba(255, 255, 255, 0) 24%),
    radial-gradient(circle at 10% 0%, rgba(255, 255, 255, 0.80), transparent 22%),
    radial-gradient(circle at 100% 16%, rgba(207, 220, 231, 0.28), transparent 28%),
    radial-gradient(circle at 80% 100%, rgba(224, 231, 238, 0.34), transparent 26%),
    linear-gradient(180deg, rgba(244, 246, 248, 0.98), rgba(237, 241, 244, 0.95) 42%, rgba(249, 248, 245, 0.96));
  border-color: rgba(212, 219, 227, 0.90);
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.66),
    0 16px 36px rgba(94, 112, 130, 0.08);
}

.empty-state {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 120px;
}

.behavior-row {
  cursor: pointer;
}

.behavior-row.selected {
  background: rgba(44, 73, 122, 0.08);
}

.behavior-row:hover {
  background: rgba(203, 58, 34, 0.04);
}

.behavior-toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 8px;
  flex-wrap: wrap;
}

.behavior-actions {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}

.behavior-checkbox {
  width: 16px;
  height: 16px;
  accent-color: #2f4e85;
  cursor: pointer;
}

.btn-light {
  background: rgba(255, 255, 255, 0.84);
  color: #2a3342;
  border: 1px solid rgba(42, 51, 66, 0.12);
}

.btn-danger:disabled {
  opacity: 0.48;
  cursor: not-allowed;
}

.context-menu {
  position: fixed;
  z-index: 50;
  width: 220px;
  padding: 12px;
  border-radius: 18px;
  border: 1px solid rgba(36, 44, 56, 0.12);
  background: rgba(250, 251, 252, 0.98);
  box-shadow: 0 18px 40px rgba(23, 31, 42, 0.18);
  backdrop-filter: blur(18px);
}

.context-menu-head {
  font-size: 13px;
  font-weight: 700;
  margin-bottom: 6px;
}

.context-menu-note {
  margin-bottom: 10px;
  color: #667180;
  font-size: 12px;
}

.scene-pill {
  display: inline-flex;
  align-items: center;
  padding: 3px 9px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.08em;
}

.scene-search {
  background: rgba(203, 58, 34, 0.10);
  color: #8d2414;
}

.scene-rec {
  background: rgba(32, 99, 196, 0.10);
  color: #1f52a6;
}

.detail-mask {
  position: fixed;
  inset: 0;
  z-index: 60;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 20px;
  background: rgba(17, 23, 31, 0.42);
  backdrop-filter: blur(8px);
}

.detail-modal {
  width: min(760px, 100%);
  max-height: min(88vh, 920px);
  overflow: auto;
  padding: 18px;
  border-radius: 24px;
  background: rgba(249, 250, 252, 0.96);
  box-shadow: 0 24px 54px rgba(17, 23, 31, 0.22);
}

.detail-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 12px;
}

.detail-head h3 {
  margin: 4px 0 0;
  font-size: 26px;
  letter-spacing: -0.03em;
}

.detail-cover {
  width: 100%;
  border-radius: 18px;
  margin-bottom: 12px;
}

.detail-content {
  line-height: 1.75;
  white-space: pre-wrap;
}

.detail-stats {
  margin-top: 14px;
  display: flex;
  gap: 14px;
  color: #667180;
}
</style>
