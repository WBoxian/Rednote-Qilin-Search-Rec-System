<script setup lang="ts">
import { computed, onMounted, ref } from 'vue';
import TopBar from '../components/TopBar.vue';
import { api, getScene, getUserId, setScene } from '../services/api';

const metricsByScene = ref<Record<string, any>>({});
const validationByScene = ref<Record<string, any>>({});
const userByScene = ref<Record<string, any>>({});
const currentScene = ref<'search' | 'rec'>(getScene());
const loading = ref(false);
const errorMsg = ref('');
const latencyMs = ref<number | null>(null);
const examplePage = ref(0);
const exampleCount = 6;

function fmtNum(v: any, digits = 4) {
  const n = Number(v);
  if (!Number.isFinite(n)) return '-';
  return n.toFixed(digits);
}

function fmtPct(v: any) {
  const n = Number(v);
  if (!Number.isFinite(n)) return '-';
  return `${(n * 100).toFixed(2)}%`;
}

function fmtText(v: any, fallback = '（无标题）') {
  if (v === null || v === undefined) return fallback;
  const s = String(v).trim();
  return s ? s : fallback;
}

function testMetric(scene: 'search' | 'rec', stage: 'recall' | 'preranking' | 'ranking') {
  return metricsByScene.value?.[scene]?.[stage]?.test || {};
}

function examples(scene: 'search' | 'rec') {
  return validationByScene.value?.[scene]?.examples || metricsByScene.value?.[scene]?.examples || [];
}

const activeExample = computed(() => {
  const arr = examples(currentScene.value);
  if (!arr.length) return null;
  return arr[Math.max(0, Math.min(examplePage.value, arr.length - 1))];
});

const behaviorItems = computed(() => {
  const arr = userByScene.value?.[currentScene.value]?.recent_behaviors || [];
  return arr.slice(0, 20).map((x: any, i: number) => ({
    idx: i + 1,
    title: fmtText(x?.title),
    scene: String(x?.scene || currentScene.value).toUpperCase(),
    query: String(x?.query || ''),
  }));
});
const behaviorLeft = computed(() => behaviorItems.value.slice(0, 10));
const behaviorRight = computed(() => behaviorItems.value.slice(10, 20));

async function load() {
  loading.value = true;
  errorMsg.value = '';
  examplePage.value = 0;
  const scene = currentScene.value;
  const uid = getUserId();
  const started = performance.now();
  try {
    const metricReq = api.metrics(scene, 40, false).then((r) => {
      metricsByScene.value = { ...metricsByScene.value, [scene]: r.metrics };
    });
    const validationReq = api.validation(scene, 12, exampleCount, uid).then((r) => {
      validationByScene.value = { ...validationByScene.value, [scene]: r.validation };
    });
    const userReq = uid == null
      ? Promise.resolve()
      : api.user(scene, uid).then((r) => {
          userByScene.value = { ...userByScene.value, [scene]: r.user };
        }).catch(() => undefined);
    await Promise.allSettled([metricReq, validationReq, userReq]);
    latencyMs.value = performance.now() - started;
  } catch (e: any) {
    errorMsg.value = e?.message || '指标加载失败';
  } finally {
    loading.value = false;
  }
}

function switchScene(scene: 'search' | 'rec') {
  currentScene.value = scene;
  setScene(scene);
  load();
}

function prevExample() {
  const arr = examples(currentScene.value);
  if (!arr.length) return;
  examplePage.value = (examplePage.value - 1 + arr.length) % arr.length;
}

function nextExample() {
  const arr = examples(currentScene.value);
  if (!arr.length) return;
  examplePage.value = (examplePage.value + 1) % arr.length;
}

onMounted(() => load());
</script>

<template>
  <TopBar title="Qilin · 指标看板" :latency-ms="latencyMs" />
  <main class="container">
    <section class="panel" style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;">
      <button class="btn-tab" :class="{ active: currentScene === 'search' }" :disabled="loading" @click="switchScene('search')">Search</button>
      <button class="btn-tab" :class="{ active: currentScene === 'rec' }" :disabled="loading" @click="switchScene('rec')">Rec</button>
      <button :disabled="loading" @click="load">{{ loading ? '加载中...' : '刷新' }}</button>
      <div class="meta">scene={{ currentScene }} | sample=50 | examples={{ exampleCount }}</div>
    </section>

    <section v-if="errorMsg" class="panel">{{ errorMsg }}</section>

    <section v-if="metricsByScene[currentScene]" class="panel">
      <h3 style="margin:0 0 10px;">召回指标</h3>
      <div class="metric-grid">
        <div class="metric-item"><div class="meta">Recall@10</div><div class="metric-value">{{ fmtPct(testMetric(currentScene, 'recall')?.['recall@10']) }}</div></div>
        <div class="metric-item"><div class="meta">Recall@100</div><div class="metric-value">{{ fmtPct(testMetric(currentScene, 'recall')?.['recall@100']) }}</div></div>
        <div class="metric-item"><div class="meta">NDCG@10</div><div class="metric-value">{{ fmtNum(testMetric(currentScene, 'recall')?.['ndcg@10']) }}</div></div>
        <div class="metric-item"><div class="meta">NDCG@50</div><div class="metric-value">{{ fmtNum(testMetric(currentScene, 'recall')?.['ndcg@50']) }}</div></div>
      </div>
    </section>

    <section v-if="metricsByScene[currentScene]" class="panel">
      <h3 style="margin:0 0 10px;">排序指标</h3>
      <div class="metric-grid">
        <div class="metric-item"><div class="meta">GBDT NDCG@10</div><div class="metric-value">{{ fmtNum(testMetric(currentScene, 'preranking')?.['ndcg@10']) }}</div></div>
        <div class="metric-item"><div class="meta">GBDT NDCG@50</div><div class="metric-value">{{ fmtNum(testMetric(currentScene, 'preranking')?.['ndcg@50']) }}</div></div>
        <div class="metric-item"><div class="meta">DIEN NDCG@10</div><div class="metric-value">{{ fmtNum(testMetric(currentScene, 'ranking')?.['ndcg@10']) }}</div></div>
        <div class="metric-item"><div class="meta">DIEN NDCG@50</div><div class="metric-value">{{ fmtNum(testMetric(currentScene, 'ranking')?.['ndcg@50']) }}</div></div>
        <div class="metric-item"><div class="meta">联动 ΔGBDT@10</div><div class="metric-value">{{ fmtNum(testMetric(currentScene, 'preranking')?.['linkage_delta_ndcg@10']) }}</div></div>
        <div class="metric-item"><div class="meta">联动 ΔDIEN@10</div><div class="metric-value">{{ fmtNum(testMetric(currentScene, 'ranking')?.['linkage_delta_ndcg@10']) }}</div></div>
        <div class="metric-item"><div class="meta">候选 P50</div><div class="metric-value">{{ fmtNum(testMetric(currentScene, 'preranking')?.p50_labeled_candidates_per_group, 1) }}</div></div>
        <div class="metric-item"><div class="meta">评估组数</div><div class="metric-value">{{ testMetric(currentScene, 'preranking')?.evaluated_groups ?? '-' }}</div></div>
      </div>
    </section>

    <section v-if="behaviorItems.length" class="panel">
      <h3 style="margin:0 0 10px;">搜推联动近期行为</h3>
      <div class="behavior-cols">
        <div class="behavior-list">
          <div v-for="b in behaviorLeft" :key="`l-${b.idx}`" class="behavior-row">
            <span class="behavior-index">{{ b.idx }}</span><span class="behavior-scene">{{ b.scene }}</span><span class="behavior-title">{{ b.title }}</span>
          </div>
        </div>
        <div class="behavior-list">
          <div v-for="b in behaviorRight" :key="`r-${b.idx}`" class="behavior-row">
            <span class="behavior-index">{{ b.idx }}</span><span class="behavior-scene">{{ b.scene }}</span><span class="behavior-title">{{ b.title }}</span>
          </div>
        </div>
      </div>
    </section>

    <section v-if="activeExample" class="panel">
      <div style="display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:10px;">
        <h3 style="margin:0;">样例对比</h3>
        <div style="display:flex;gap:8px;align-items:center;">
          <button class="btn" @click="prevExample">上一条</button>
          <span class="meta">{{ examplePage + 1 }}/{{ examples(currentScene).length }}</span>
          <button class="btn" @click="nextExample">下一条</button>
        </div>
      </div>
      <div class="meta" style="margin-bottom:8px;">request_id={{ activeExample.request_id }} | GBDT overlap@10={{ fmtPct(activeExample.gbdt_overlap_at10) }} | DIEN overlap@10={{ fmtPct(activeExample.dien_overlap_at10) }}</div>
      <div class="compare-grid">
        <div class="compare-cell">
          <h4>真实 Top10</h4>
          <table class="table"><tbody>
            <tr><td>真实位次</td><td>note</td><td>标题</td></tr>
            <tr v-for="(item, idx) in activeExample.true_top10" :key="`t-${idx}-${item.note_idx}`"><td>#{{ item.rank_in_true ?? (Number(idx) + 1) }}</td><td>{{ item.note_idx }}</td><td>{{ item.title }}</td></tr>
          </tbody></table>
        </div>
        <div class="compare-cell">
          <h4>GBDT Top10</h4>
          <table class="table"><tbody>
            <tr><td>真实位次</td><td>note</td><td>标题</td></tr>
            <tr v-for="(item, idx) in activeExample.gbdt_top10" :key="`g-${idx}-${item.note_idx}`"><td>{{ item.rank_in_true ? '#' + item.rank_in_true : '-' }}</td><td>{{ item.note_idx }}</td><td>{{ item.title }}</td></tr>
          </tbody></table>
        </div>
        <div class="compare-cell">
          <h4>DIEN Top10</h4>
          <table class="table"><tbody>
            <tr><td>真实位次</td><td>note</td><td>标题</td></tr>
            <tr v-for="(item, idx) in activeExample.dien_top10" :key="`d-${idx}-${item.note_idx}`"><td>{{ item.rank_in_true ? '#' + item.rank_in_true : '-' }}</td><td>{{ item.note_idx }}</td><td>{{ item.title }}</td></tr>
          </tbody></table>
        </div>
      </div>
    </section>
  </main>
</template>

<style scoped>
.compare-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
}
.compare-cell {
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 10px;
  background: #fff;
}
.compare-cell h4 {
  margin: 0 0 8px;
}
.behavior-cols {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}
.behavior-list {
  display: grid;
  gap: 6px;
}
.behavior-row {
  display: grid;
  grid-template-columns: 28px 58px minmax(0, 1fr);
  gap: 8px;
  align-items: center;
  padding: 7px 9px;
  border: 1px solid var(--line);
  border-radius: 10px;
  background: rgba(255,255,255,0.72);
  font-size: 12px;
}
.behavior-index {
  color: #94a3b8;
  font-weight: 700;
}
.behavior-scene {
  justify-self: start;
  padding: 2px 7px;
  border-radius: 999px;
  background: rgba(255, 36, 66, 0.1);
  color: #b91c1c;
  font-weight: 800;
}
.behavior-title {
  min-width: 0;
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}
@media (max-width: 900px) {
  .compare-grid,
  .behavior-cols {
    grid-template-columns: 1fr;
  }
}
</style>
