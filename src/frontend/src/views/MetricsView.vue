<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue';
import TopBar from '../components/TopBar.vue';
import { api, getScene, getUserId, setScene } from '../services/api';

const metricsByScene = ref<Record<string, any>>({});
const validationByScene = ref<Record<string, any>>({});
const examplesByScene = ref<Record<string, any[]>>({});
const currentScene = ref<'search' | 'rec'>(getScene());
const loading = ref(false);
const validationLoading = ref(false);
const validationProgress = ref(0);
const validationStage = ref('');
const errorMsg = ref('');
const latencyMs = ref<number | null>(null);
const examplePage = ref(0);
const metricSampleGroups = 1000;
const compareSampleGroups = 240;
const candidateExampleLimit = 24;
const displayedExampleLimit = 5;
const detailModal = ref<any>(null);
const detailLoading = ref(false);
const detailError = ref('');
const formulaModal = ref<{ title: string; lines: string[]; desc: string[] } | null>(null);
const pageTitle = 'Qilin 指标看板';
const copy = {
  loading: '加载中...',
  refresh: '刷新',
  loadMetricsError: '指标加载失败',
  loadCompareError: '样例对比加载失败',
  loadDetailError: '帖子详情加载失败',
  noQuery: '无搜索词',
  currentUser: '当前用户',
  otherUser: '其他用户',
  noContent: '暂无内容',
};

function startValidationProgress(scene: 'search' | 'rec') {
  validationProgress.value = 12;
  validationStage.value = scene === 'search' ? '正在准备 Search 样例对比' : '正在准备 Rec 样例对比';
}

function markValidationProgress(next: number, stage: string) {
  validationProgress.value = Math.max(0, Math.min(100, Math.round(next)));
  validationStage.value = stage;
}

function stopValidationProgress() {
  return;
}

const unifiedRecallMetricDefs = [
  { key: 'HitRate@100', label: 'HitRate@100' },
  { key: 'HitRate@500', label: 'HitRate@500' },
  { key: 'Recall@100', label: 'Recall@100' },
  { key: 'Recall@500', label: 'Recall@500' },
  { key: 'Recall@1000', label: 'Recall@1000' },
  { key: 'MRR@100', label: 'MRR@100' },
  { key: 'MedianFirstHitRank', label: 'MedianFirstHitRank', digits: 1 },
  { key: 'TailHitRate101_500@Pre100', label: 'TailHitRate101-500→Pre100' },
] as const;

const recallMetricDefsByScene = {
  search: unifiedRecallMetricDefs,
  rec: unifiedRecallMetricDefs,
} as const;

const rankingMetricDefs = [
  { key: 'NDCG@10', label: 'NDCG@10' },
  { key: 'AUC', label: 'AUC' },
  { key: 'GAUC', label: 'GAUC' },
] as const;

const formulaDefs: Record<string, { lines: string[]; desc: string[] }> = {
  'Recall@100': {
    lines: [
      'Recall@100 = <span class="formula-frac"><span>1</span><span>|Q|</span></span> · Σ<sub>q∈Q</sub> <span class="formula-frac"><span>|Top<sub>100</sub>(q) ∩ Rel(q)|</span><span>|Rel(q)|</span></span>',
    ],
    desc: ['衡量前 100 个召回候选覆盖了多少真实相关结果。'],
  },
  'HitRate@100': {
    lines: [
      'HitRate@100 = <span class="formula-frac"><span>1</span><span>|Q|</span></span> · Σ<sub>q∈Q</sub> 𝟙(Top<sub>100</sub>(q) ∩ Rel(q) ≠ ∅)',
    ],
    desc: ['衡量一个请求在前 100 个召回候选里是否至少命中 1 个真实相关结果。'],
  },
  'HitRate@500': {
    lines: [
      'HitRate@500 = <span class="formula-frac"><span>1</span><span>|Q|</span></span> · Σ<sub>q∈Q</sub> 𝟙(Top<sub>500</sub>(q) ∩ Rel(q) ≠ ∅)',
    ],
    desc: ['衡量粗排入口的 500 个候选里，是否至少命中 1 个真实相关结果。'],
  },
  'Recall@500': {
    lines: [
      'Recall@500 = <span class="formula-frac"><span>1</span><span>|Q|</span></span> · Σ<sub>q∈Q</sub> <span class="formula-frac"><span>|Top<sub>500</sub>(q) ∩ Rel(q)|</span><span>|Rel(q)|</span></span>',
    ],
    desc: ['衡量进入粗排的 500 个候选对真实相关结果的覆盖程度。'],
  },
  'Recall@300': {
    lines: [
      'Recall@300 = <span class="formula-frac"><span>1</span><span>|Q|</span></span> · Σ<sub>q∈Q</sub> <span class="formula-frac"><span>|Top<sub>300</sub>(q) ∩ Rel(q)|</span><span>|Rel(q)|</span></span>',
    ],
    desc: ['观察放宽到 300 个候选后，召回覆盖是否明显抬升。'],
  },
  'Recall@1000': {
    lines: [
      'Recall@1000 = <span class="formula-frac"><span>1</span><span>|Q|</span></span> · Σ<sub>q∈Q</sub> <span class="formula-frac"><span>|Top<sub>1000</sub>(q) ∩ Rel(q)|</span><span>|Rel(q)|</span></span>',
    ],
    desc: ['衡量召回池上限的覆盖能力，用于观察候选天花板。'],
  },
  'MRR@100': {
    lines: [
      'MRR@100 = <span class="formula-frac"><span>1</span><span>|Q|</span></span> · Σ<sub>q∈Q</sub> <span class="formula-frac"><span>1</span><span>rank<sub>first-hit</sub>(q)</span></span>',
    ],
    desc: ['衡量第一个真实相关结果进入前 100 候选后的位置，越靠前越好。'],
  },
  MedianFirstHitRank: {
    lines: [
      'MedianFirstHitRank = median({ rank<sub>first-hit</sub>(q) | Top(q) ∩ Rel(q) ≠ ∅ })',
    ],
    desc: ['统计所有命中请求的首个相关结果名次中位数，越小越好。'],
  },
  'TailHitRate101_500@Pre100': {
    lines: [
      'TailHitRate101_500@Pre100 = <span class="formula-frac"><span>1</span><span>|Q|</span></span> · Σ<sub>q∈Q</sub> 𝟙(∃ i, 101 ≤ rank<sub>recall</sub>(i) ≤ 500, rank<sub>pre</sub>(i) ≤ 100, i ∈ Rel(q))',
    ],
    desc: ['衡量召回尾部 101-500 的候选是否真的被粗排救回到了前 100，并命中了真实相关结果。'],
  },
  'Precision@100': {
    lines: [
      'Precision@100 = <span class="formula-frac"><span>1</span><span>|Q|</span></span> · Σ<sub>q∈Q</sub> <span class="formula-frac"><span>|Top<sub>100</sub>(q) ∩ Rel(q)|</span><span>100</span></span>',
    ],
    desc: ['衡量前 100 个召回候选里相关结果的占比。'],
  },
  MAP: {
    lines: [
      'MAP = <span class="formula-frac"><span>1</span><span>|Q|</span></span> · Σ<sub>q∈Q</sub> AP(q)',
      'AP(q) = <span class="formula-frac"><span>1</span><span>|Rel(q)|</span></span> · Σ<sub>k=1..K</sub> Precision@k(q) · 𝟙(item<sub>k</sub> ∈ Rel(q))',
    ],
    desc: ['综合考察相关结果是否持续出现在更靠前的位置。'],
  },
  'NDCG@10': {
    lines: [
      'NDCG@10 = <span class="formula-frac"><span>1</span><span>|Q|</span></span> · Σ<sub>q∈Q</sub> <span class="formula-frac"><span>DCG@10(q)</span><span>IDCG@10(q)</span></span>',
      'DCG@10(q) = Σ<sub>k=1..10</sub> <span class="formula-frac"><span>2<sup>rel(q,k)</sup> - 1</span><span>log<sub>2</sub>(k + 1)</span></span>',
    ],
    desc: ['同时考虑相关性强弱和排序位置，越靠前命中高相关结果越好。'],
  },
  AUC: {
    lines: ['AUC = P(score(正样本) > score(负样本))'],
    desc: ['表示模型把正样本排到负样本前面的概率。'],
  },
  GAUC: {
    lines: [
      'GAUC = Σ<sub>u</sub> w(u) · AUC(u)',
      'w(u) = <span class="formula-frac"><span>|P<sub>u</sub>| · |N<sub>u</sub>|</span><span>Σ<sub>v</sub> |P<sub>v</sub>| · |N<sub>v</sub>|</span></span>',
    ],
    desc: ['按请求规模加权后的 AUC，更接近真实流量分布。'],
  },
};

function fmtNum(v: any, digits = 4) {
  const n = Number(v);
  if (!Number.isFinite(n) || n < 0) return '-';
  return n.toFixed(digits);
}

function metric(scene: 'search' | 'rec', area: 'recall' | 'preranking' | 'ranking') {
  if (area === 'recall') {
    return metricsByScene.value?.[scene]?.recall?.validation || metricsByScene.value?.[scene]?.recall?.val || {};
  }
  return metricsByScene.value?.[scene]?.ranking?.[area]?.validation || metricsByScene.value?.[scene]?.ranking?.[area]?.val || {};
}

const recallMetricRows = computed(() => {
  const defs = recallMetricDefsByScene[currentScene.value] || [];
  return [defs.slice(0, 4), defs.slice(4)];
});

function metricSampleMeta(scene: 'search' | 'rec', area: 'recall' | 'preranking' | 'ranking') {
  const obj = metric(scene, area);
  const sampled = Number(obj?.sampled_groups || 0);
  const effective = Number(obj?.effective_groups || 0);
  if (!Number.isFinite(sampled) || sampled <= 0) return '离线验证快照';
  if (!Number.isFinite(effective) || effective <= 0) return `离线验证快照 · ${sampled}组请求`;
  return `离线验证快照 · ${effective}/${sampled}组有效请求`;
}

function behaviorTime(row: any) {
  const ts = Number(row?.ts);
  if (!Number.isFinite(ts) || ts <= 0) return '-';
  const ms = ts > 1e12 ? ts : ts * 1000;
  const d = new Date(ms);
  if (Number.isNaN(d.getTime())) return '-';
  const pad = (x: number) => String(x).padStart(2, '0');
  return `${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

function collapseBehaviorRows(rows: any[]) {
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

function overlapTotal(example: any) {
  const overlap = example?.overlap || {};
  return Number(overlap?.dssm || 0) + Number(overlap?.preranking || 0) + Number(overlap?.ranking || 0);
}

function normalizeSceneExamples(scene: 'search' | 'rec', rows: any[]) {
  const normalized = (rows || []).map((row: any) => {
    const collapsed = collapseBehaviorRows(row?.source_recent_behaviors || []);
    return {
      ...row,
      scene,
      overlap_total: overlapTotal(row),
      behavior_count: collapsed.length,
      collapsed_behaviors: collapsed,
    };
  });
  normalized.sort((a: any, b: any) => {
    const resultTierGap =
      (Number(b.result_count || 0) >= 20 ? 2 : Number(b.result_count || 0) >= 10 ? 1 : 0) -
      (Number(a.result_count || 0) >= 20 ? 2 : Number(a.result_count || 0) >= 10 ? 1 : 0);
    if (resultTierGap !== 0) return resultTierGap;
    const behaviorTierGap =
      (Number(b.behavior_count || 0) >= 20 ? 2 : Number(b.behavior_count || 0) >= 10 ? 1 : 0) -
      (Number(a.behavior_count || 0) >= 20 ? 2 : Number(a.behavior_count || 0) >= 10 ? 1 : 0);
    if (behaviorTierGap !== 0) return behaviorTierGap;
    const overlapGap = Number(b.overlap_total || 0) - Number(a.overlap_total || 0);
    if (overlapGap !== 0) return overlapGap;
    const resultGap = Number(b.result_count || 0) - Number(a.result_count || 0);
    if (resultGap !== 0) return resultGap;
    return Number(b.behavior_count || 0) - Number(a.behavior_count || 0);
  });
  return normalized;
}

function selectDisplayExamples(rows: any[], uid: number | null) {
  const currentUid = Number(uid);
  const sorted = [...(rows || [])].sort((a: any, b: any) => {
    const selfGap = Number(b.source_user_idx === currentUid) - Number(a.source_user_idx === currentUid);
    if (selfGap !== 0) return selfGap;
    const resultTierGap =
      (Number(b.result_count || 0) >= 20 ? 2 : Number(b.result_count || 0) >= 10 ? 1 : 0) -
      (Number(a.result_count || 0) >= 20 ? 2 : Number(a.result_count || 0) >= 10 ? 1 : 0);
    if (resultTierGap !== 0) return resultTierGap;
    const behaviorTierGap =
      (Number(b.behavior_count || 0) >= 20 ? 2 : Number(b.behavior_count || 0) >= 10 ? 1 : 0) -
      (Number(a.behavior_count || 0) >= 20 ? 2 : Number(a.behavior_count || 0) >= 10 ? 1 : 0);
    if (behaviorTierGap !== 0) return behaviorTierGap;
    const overlapGap = Number(b.overlap_total || 0) - Number(a.overlap_total || 0);
    if (overlapGap !== 0) return overlapGap;
    return Number(b.result_count || 0) - Number(a.result_count || 0);
  });
  const picked: any[] = [];
  const usedUsers = new Set<number>();
  const currentRows = sorted.filter((row: any) => Number(row.source_user_idx) === currentUid);
  const bestCurrent = currentRows.find((row: any) => Number(row.result_count || 0) >= 10 && Number(row.behavior_count || 0) >= 10) || currentRows[0];
  if (bestCurrent) {
    picked.push(bestCurrent);
    usedUsers.add(Number(bestCurrent.source_user_idx));
  }
  for (const row of sorted) {
    if (picked.length >= displayedExampleLimit) break;
    const rowUid = Number(row.source_user_idx);
    if (usedUsers.has(rowUid)) continue;
    picked.push(row);
    usedUsers.add(rowUid);
  }
  return picked.slice(0, displayedExampleLimit);
}

const mergedExamples = computed(() => selectDisplayExamples(examplesByScene.value[currentScene.value] || [], getUserId()));
const activeExample = computed(() => {
  const rows = mergedExamples.value;
  if (!rows.length) return null;
  return rows[Math.max(0, Math.min(examplePage.value, rows.length - 1))];
});
const activeBehaviors = computed(() => activeExample.value?.collapsed_behaviors || []);
const activeOverlap = computed(() => {
  const overlap = activeExample.value?.overlap || {};
  return {
    dssm: Number(overlap?.dssm || 0),
    preranking: Number(overlap?.preranking || 0),
    ranking: Number(overlap?.ranking || 0),
  };
});
const activeQueryText = computed(() => {
  const query = String(activeExample.value?.query || '').trim();
  return query || copy.noQuery;
});
const activeUserProfile = computed(() => {
  const user = activeExample.value?.source_user;
  const features = user?.features || {};
  const tags = Array.isArray(user?.interest_tags) ? user.interest_tags.slice(0, 8) : [];
  const keywords = Array.isArray(user?.recent_search_keywords) ? user.recent_search_keywords.slice(0, 6) : [];
  const lines = [
    `user=${activeExample.value?.source_user_idx ?? '-'}`,
    features?.gender ? `性别 ${features.gender}` : '',
    features?.age ? `年龄 ${features.age}` : '',
    features?.platform ? `平台 ${features.platform}` : '',
    features?.location ? `地区 ${features.location}` : '',
    Number.isFinite(Number(user?.request_count_in_test)) ? `测试请求 ${Number(user.request_count_in_test)}` : '',
  ].filter(Boolean);
  const behaviorSignals = (activeBehaviors.value || [])
    .slice(0, 6)
    .map((row: any) => {
      if (String(row?.scene || '').toLowerCase() === 'search' && row?.query) {
        return String(row.query).trim();
      }
      const title = String(row?.title || '').trim();
      if (title) return title.slice(0, 18);
      const noteId = Number(row?.note_idx);
      return Number.isFinite(noteId) ? `note ${noteId}` : '';
    })
    .filter(Boolean);
  const recentSignals = [
    ...tags.map((tag: any) => `#tag[${tag}]`),
    ...keywords.map((term: any) => String(term)),
    ...behaviorSignals,
  ]
    .filter((val, idx, arr) => arr.indexOf(val) === idx)
    .slice(0, 10);
  return { lines, recentSignals };
});

function formatTrueRank(value: any, fallback = '(-)') {
  const num = Number(value);
  if (Number.isFinite(num) && num > 0) return String(num);
  return fallback;
}

function nextPaint() {
  return new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
}

function whenBrowserIdle() {
  return new Promise<void>((resolve) => {
    const ric = (window as any)?.requestIdleCallback;
    if (typeof ric === 'function') {
      ric(() => resolve(), { timeout: 180 });
      return;
    }
    window.setTimeout(() => resolve(), 80);
  });
}

function behaviorScene(row: any) {
  return String(row?.scene || '').toLowerCase() === 'rec' ? 'REC' : 'SEARCH';
}

function behaviorPrimary(row: any) {
  return row?.title || row?.query || `note=${row?.note_idx ?? '-'}`;
}

function behaviorSecondary(row: any) {
  const parts = [
    `note ${row?.note_idx ?? '-'}`,
    behaviorScene(row),
    row?.query ? `query: ${row.query}` : '',
  ].filter(Boolean);
  return parts.join(' · ');
}

function openMetricFormula(stage: string, name: string, split: string) {
  const formula = formulaDefs[name];
  if (!formula) return;
  formulaModal.value = {
    title: `${stage} · ${split} · ${name}`,
    lines: formula.lines,
    desc: formula.desc,
  };
}

function closeFormula() {
  formulaModal.value = null;
}

async function loadSceneMetrics(scene: 'search' | 'rec', force = false) {
  if (!force && metricsByScene.value[scene]) return;
  const metricRes = await api.metrics(scene, metricSampleGroups, true);
  metricsByScene.value = { ...metricsByScene.value, [scene]: metricRes.metrics };
  if (currentScene.value === scene) {
    latencyMs.value = Number(metricRes?.latency_ms || 0);
  }
}

async function loadSceneValidation(scene: 'search' | 'rec', uid: number | null, force = false) {
  if (!force && validationByScene.value[scene]) return;
  const res = await api.validation(scene, compareSampleGroups, candidateExampleLimit, uid);
  validationByScene.value = { ...validationByScene.value, [scene]: res.validation };
  examplesByScene.value = {
    ...examplesByScene.value,
    [scene]: normalizeSceneExamples(scene, res.validation?.examples || []),
  };
}

async function load() {
  const uid = getUserId();
  loading.value = true;
  validationLoading.value = !validationByScene.value[currentScene.value];
  errorMsg.value = '';
  examplePage.value = 0;
  try {
    await loadSceneMetrics(currentScene.value, true);
  } catch (e: any) {
    errorMsg.value = e?.message || copy.loadMetricsError;
  } finally {
    loading.value = false;
  }
  await nextPaint();
  validationLoading.value = true;
  startValidationProgress(currentScene.value);
  await whenBrowserIdle();
  try {
    markValidationProgress(44, '正在读取离线样例快照');
    await loadSceneValidation(currentScene.value, uid, true);
    markValidationProgress(84, '正在整理当前用户样例与近期行为');
    markValidationProgress(100, '样例对比已准备完成');
  } catch (e: any) {
    if (!examplesByScene.value[currentScene.value]?.length && !errorMsg.value) {
      errorMsg.value = e?.message || copy.loadCompareError;
    }
  } finally {
    stopValidationProgress();
    validationLoading.value = false;
  }
}

async function switchScene(scene: 'search' | 'rec') {
  if (currentScene.value === scene) return;
  currentScene.value = scene;
  setScene(scene);
  examplePage.value = 0;
  const uid = getUserId();
  loading.value = !metricsByScene.value[scene];
  validationLoading.value = !validationByScene.value[scene];
  errorMsg.value = '';
  try {
    await loadSceneMetrics(scene, false);
  } catch (e: any) {
    errorMsg.value = e?.message || copy.loadMetricsError;
  } finally {
    loading.value = false;
  }
  await nextPaint();
  startValidationProgress(scene);
  await whenBrowserIdle();
  try {
    markValidationProgress(44, scene === 'search' ? '正在读取 Search 样例快照' : '正在读取 Rec 样例快照');
    await loadSceneValidation(scene, uid, false);
    markValidationProgress(84, '正在整理当前用户样例与近期行为');
    markValidationProgress(100, '样例对比已准备完成');
  } catch (e: any) {
    if (!examplesByScene.value[scene]?.length && !errorMsg.value) {
      errorMsg.value = e?.message || copy.loadCompareError;
    }
  } finally {
    stopValidationProgress();
    validationLoading.value = false;
  }
}

function prevExample() {
  const rows = mergedExamples.value;
  if (!rows.length) return;
  examplePage.value = (examplePage.value - 1 + rows.length) % rows.length;
}

function nextExample() {
  const rows = mergedExamples.value;
  if (!rows.length) return;
  examplePage.value = (examplePage.value + 1) % rows.length;
}

async function openExampleDetail(item: any) {
  const ex = activeExample.value;
  if (!ex) return;
  detailModal.value = {
    title: item?.title || copy.loading,
    content: '',
    cover_image: '',
    like_count: 0,
    collect_count: 0,
    comment_count: 0,
  };
  detailError.value = '';
  detailLoading.value = true;
  try {
    const res = await api.note(
      ex.scene,
      Number(ex.source_user_idx || 0),
      Number(ex.request_id || 0),
      Number(item?.note_idx || 0),
      String(ex.query || ''),
      true,
    );
    detailModal.value = res?.detail || null;
  } catch (e: any) {
    detailError.value = e?.message || copy.loadDetailError;
  } finally {
    detailLoading.value = false;
  }
}

async function openBehaviorDetail(row: any) {
  const ex = activeExample.value;
  if (!ex) return;
  detailModal.value = {
    title: row?.title || behaviorPrimary(row) || copy.loading,
    content: '',
    cover_image: '',
    like_count: 0,
    collect_count: 0,
    comment_count: 0,
  };
  detailError.value = '';
  detailLoading.value = true;
  try {
    const res = await api.note(
      String(row?.scene || ex.scene || 'search'),
      Number(ex.source_user_idx || 0),
      Number(row?.request_id || 0),
      Number(row?.note_idx || 0),
      String(row?.query || ''),
      true,
    );
    detailModal.value = res?.detail || null;
  } catch (e: any) {
    detailError.value = e?.message || copy.loadDetailError;
  } finally {
    detailLoading.value = false;
  }
}

function closeDetail() {
  detailModal.value = null;
  detailError.value = '';
}

onMounted(load);
onUnmounted(stopValidationProgress);
</script>

<template>
  <TopBar :title="pageTitle" :latency-ms="latencyMs" :scene-mode="currentScene" />
  <main class="container">
    <section class="panel metrics-hero shell-panel">
      <div>
        <div class="meta">召回 / 粗排 / 精排</div>
        <h2>离线验证指标</h2>
      </div>
      <div class="hero-actions">
        <button class="btn-tab" :class="{ active: currentScene === 'search' }" :disabled="loading" @click="switchScene('search')">Search</button>
        <button class="btn-tab" :class="{ active: currentScene === 'rec' }" :disabled="loading" @click="switchScene('rec')">Rec</button>
        <button class="btn-primary" :disabled="loading" @click="load">{{ loading ? copy.loading : copy.refresh }}</button>
      </div>
    </section>

    <section v-if="errorMsg" class="panel error-panel shell-panel">
      <div class="error-kicker">状态异常</div>
      <div class="error-title">{{ errorMsg }}</div>
      <div class="error-desc">已保留当前已加载内容。刷新后会重新拉取指标与样例。</div>
    </section>

    <template v-if="metricsByScene[currentScene]">
      <section class="metrics-grid">
        <div class="panel shell-panel">
          <div class="section-title">召回指标</div>
          <div class="split-title">验证集</div>
          <div class="split-meta">{{ metricSampleMeta(currentScene, 'recall') }}</div>
          <div class="metric-stack">
            <div v-for="(row, rowIdx) in recallMetricRows" :key="`recall-row-${rowIdx}`" class="metric-row recall-row">
              <button
                v-for="item in row"
                :key="item.key"
                class="metric-item metric-button metric-row-item"
                @click="openMetricFormula('召回', item.key, '验证集')"
              >
                <div class="meta">{{ item.label }}</div>
                <div class="metric-value">{{ fmtNum(metric(currentScene, 'recall')?.[item.key], (item as any).digits ?? 4) }}</div>
              </button>
            </div>
          </div>
        </div>

        <div class="panel shell-panel">
          <div class="section-title">排序指标</div>
          <div class="rank-stage-list single-stage-list">
            <div class="rank-stage">
              <div class="rank-stage-title">粗排 · GBDT</div>
              <div class="split-meta">{{ metricSampleMeta(currentScene, 'preranking') }}</div>
              <div class="metric-row compact-row">
                <button
                  v-for="item in rankingMetricDefs"
                  :key="`pre-${item.key}`"
                  class="metric-item metric-button metric-row-item"
                  @click="openMetricFormula('粗排', item.key, '验证集')"
                >
                  <div class="meta">{{ item.label }}</div>
                  <div class="metric-value">{{ fmtNum(metric(currentScene, 'preranking')?.[item.key]) }}</div>
                </button>
              </div>
            </div>

            <div class="rank-stage">
              <div class="rank-stage-title">精排 · DIEN</div>
              <div class="split-meta">{{ metricSampleMeta(currentScene, 'ranking') }}</div>
              <div class="metric-row compact-row">
                <button
                  v-for="item in rankingMetricDefs"
                  :key="`rank-${item.key}`"
                  class="metric-item metric-button metric-row-item"
                  @click="openMetricFormula('精排', item.key, '验证集')"
                >
                  <div class="meta">{{ item.label }}</div>
                  <div class="metric-value">{{ fmtNum(metric(currentScene, 'ranking')?.[item.key]) }}</div>
                </button>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section class="panel example-panel shell-panel">
        <div class="example-head">
          <div>
            <div class="section-title">样例对比</div>
            <div class="meta" v-if="activeExample">
              scene={{ activeExample.scene.toUpperCase() }} · request_id={{ activeExample.request_id }} · user={{ activeExample.source_user_idx }} · results={{ activeExample.result_count ?? '-' }}
              <span class="source-user-badge" :class="activeExample.is_current_user ? 'badge-self' : 'badge-other'">
                {{ activeExample.is_current_user ? copy.currentUser : copy.otherUser }}
              </span>
            </div>
            <div class="meta">固定展示 5 条样例：当前登录用户优先 1 条，其余补其他用户；最近行为跨 Search / Rec 混合，最多 40 条。</div>
          </div>
          <div v-if="activeExample" class="pager">
            <button class="btn" @click="prevExample">上一条</button>
            <span class="meta">{{ examplePage + 1 }}/{{ mergedExamples.length }}</span>
            <button class="btn" @click="nextExample">下一条</button>
          </div>
        </div>

        <div v-if="validationLoading" class="validation-progress-shell">
          <div class="validation-progress-head">
            <div class="validation-progress-title">
              <span class="validation-spinner" aria-hidden="true" />
              <strong>正在加载样例对比</strong>
            </div>
            <span>{{ validationProgress }}%</span>
          </div>
          <div class="validation-progress-bar">
            <div class="validation-progress-fill" :style="{ width: `${validationProgress}%` }" />
          </div>
          <div class="meta">{{ validationStage || copy.loading }}</div>
        </div>
        <template v-else-if="activeExample">
          <div class="profile-card">
            <div class="profile-line"><strong>用户画像</strong> · {{ activeUserProfile.lines.join(' · ') }}</div>
            <div class="profile-line"><strong>搜索词</strong> · {{ activeQueryText }}</div>
            <div class="profile-line profile-signals"><strong>近期行为信号</strong> · {{ activeUserProfile.recentSignals.join(' · ') || '暂无' }}</div>
          </div>

          <div class="overlap-summary">
            <span class="overlap-pill">DSSM 命中 {{ activeOverlap.dssm }}</span>
            <span class="overlap-pill">GBDT 命中 {{ activeOverlap.preranking }}</span>
            <span class="overlap-pill">DIEN 命中 {{ activeOverlap.ranking }}</span>
          </div>

          <div class="source-head">
            <div class="section-title section-title-small">最近行为</div>
            <div class="meta">最多 40 条，Search / Rec 已打通并按 note 去重</div>
          </div>
          <div class="source-behavior-grid">
            <button
              v-for="row in activeBehaviors"
              :key="`${row.scene}-${row.request_id}-${row.note_idx}-${row.ts}`"
              class="source-behavior-row"
              @click="openBehaviorDetail(row)"
            >
              <span class="source-time">{{ behaviorTime(row) }}</span>
              <span class="source-scene" :class="row.scene === 'rec' ? 'scene-rec' : 'scene-search'">{{ behaviorScene(row) }}</span>
              <span class="source-copy">
                <span class="source-primary">{{ behaviorPrimary(row) }}</span>
                <span class="source-secondary">{{ behaviorSecondary(row) }}</span>
              </span>
            </button>
          </div>

          <div class="compare-grid">
            <div class="compare-cell">
              <h4>真实 Top10</h4>
              <table class="table compact-table">
                <tbody>
                  <tr v-for="(item, idx) in activeExample.true_top10" :key="`truth-${idx}`" class="compare-row" @click="openExampleDetail(item)">
                    <td class="rank-col"><strong>{{ formatTrueRank(item.rank_in_true, String(idx + 1)) }}</strong></td>
                    <td>{{ item.note_idx }}</td>
                    <td>{{ item.title }}</td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div class="compare-cell">
              <h4>DSSM Top10</h4>
              <table class="table compact-table">
                <tbody>
                  <tr v-for="(item, idx) in activeExample.dssm_top10" :key="`dssm-${idx}`" class="compare-row" @click="openExampleDetail(item)">
                    <td class="rank-col"><strong>{{ formatTrueRank(item.rank_in_true) }}</strong></td>
                    <td>{{ item.note_idx }}</td>
                    <td>{{ item.title }}</td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div class="compare-cell">
              <h4>GBDT Top10</h4>
              <table class="table compact-table">
                <tbody>
                  <tr v-for="(item, idx) in activeExample.preranking_top10" :key="`pre-${idx}`" class="compare-row" @click="openExampleDetail(item)">
                    <td class="rank-col"><strong>{{ formatTrueRank(item.rank_in_true) }}</strong></td>
                    <td>{{ item.note_idx }}</td>
                    <td>{{ item.title }}</td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div class="compare-cell">
              <h4>DIEN Top10</h4>
              <table class="table compact-table">
                <tbody>
                  <tr v-for="(item, idx) in activeExample.ranking_top10" :key="`rank-${idx}`" class="compare-row" @click="openExampleDetail(item)">
                    <td class="rank-col"><strong>{{ formatTrueRank(item.rank_in_true) }}</strong></td>
                    <td>{{ item.note_idx }}</td>
                    <td>{{ item.title }}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </template>
      </section>
    </template>

    <div v-if="detailModal || detailLoading || detailError" class="detail-mask" @click.self="closeDetail">
      <div class="detail-modal shell-panel">
        <div class="detail-head">
          <div>
            <div class="meta">帖子详情</div>
            <h3>{{ detailModal?.title || (detailLoading ? copy.loading : detailError || copy.noContent) }}</h3>
          </div>
          <button class="btn" @click="closeDetail">关闭</button>
        </div>
        <div v-if="detailLoading" class="meta">{{ copy.loading }}</div>
        <div v-else-if="detailError" class="error-desc">{{ detailError }}</div>
        <template v-else-if="detailModal">
          <img v-if="detailModal.cover_image" class="detail-cover" :src="api.imageUrl(detailModal.cover_image)" alt="" />
          <div class="detail-content">{{ detailModal.content || copy.noContent }}</div>
          <div class="detail-stats">
            <span>点赞 {{ detailModal.like_count ?? 0 }}</span>
            <span>收藏 {{ detailModal.collect_count ?? 0 }}</span>
            <span>评论 {{ detailModal.comment_count ?? 0 }}</span>
          </div>
        </template>
      </div>
    </div>

    <div v-if="formulaModal" class="detail-mask" @click.self="closeFormula">
      <div class="formula-modal shell-panel">
        <div class="detail-head">
          <div>
            <div class="meta">指标公式</div>
            <h3>{{ formulaModal.title }}</h3>
          </div>
          <button class="btn" @click="closeFormula">关闭</button>
        </div>
        <div class="formula-block">
          <div v-for="(line, idx) in formulaModal.lines" :key="idx" class="formula-line" v-html="line" />
          <div class="formula-desc">
            <div v-for="(line, idx) in formulaModal.desc" :key="idx" class="meta">{{ line }}</div>
          </div>
        </div>
      </div>
    </div>
  </main>
</template>

<style scoped>
.metrics-hero,
.example-head,
.source-head,
.profile-card {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.metrics-grid {
  display: grid;
  grid-template-columns: 1.05fr 1fr;
  gap: 14px;
}

.hero-actions,
.pager,
.overlap-summary {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}

.metric-row {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
}

.metric-stack {
  display: grid;
  gap: 10px;
}

.recall-row {
  grid-template-columns: repeat(4, minmax(0, 1fr));
}

.compact-row {
  grid-template-columns: repeat(3, minmax(0, 1fr));
}

.metric-row-item {
  min-width: 0;
}

.rank-stage-list {
  display: grid;
  gap: 12px;
}

.rank-stage {
  padding: 14px;
  border-radius: 20px;
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(236, 240, 244, 0.93));
  border: 1px solid rgba(92, 108, 123, 0.12);
}

.rank-stage-title,
.split-title,
.section-title-small {
  font-size: 16px;
  font-weight: 700;
  color: #1c2834;
}

.section-title {
  font-size: 17px;
  font-weight: 800;
}

.split-meta {
  margin: 6px 0 12px;
  color: #6b7481;
  font-size: 12px;
}

.profile-card {
  align-items: flex-start;
  flex-direction: column;
  padding: 12px 14px;
  border-radius: 18px;
  background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(242, 238, 232, 0.94));
  border: 1px solid rgba(92, 108, 123, 0.12);
  gap: 4px;
}

.profile-line {
  color: #283341;
  font-size: 12px;
  line-height: 1.7;
}

.validation-progress-shell {
  display: grid;
  gap: 10px;
  padding: 14px 16px;
  border-radius: 18px;
  border: 1px solid rgba(37, 48, 61, 0.08);
  background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(243, 238, 230, 0.92));
  box-shadow: 0 16px 28px rgba(28, 36, 46, 0.08);
}

.validation-progress-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  color: #1f2a36;
}

.validation-progress-title {
  display: inline-flex;
  align-items: center;
  gap: 10px;
}

.validation-spinner {
  width: 18px;
  height: 18px;
  border-radius: 999px;
  border: 2px solid rgba(84, 114, 150, 0.22);
  border-top-color: #294566;
  border-right-color: #5f85b2;
  animation: validation-spin 0.9s linear infinite;
}

.validation-progress-bar {
  height: 8px;
  border-radius: 999px;
  overflow: hidden;
  background: rgba(72, 87, 104, 0.12);
}

.validation-progress-fill {
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, #406eb7, #79a6e6);
  transition: width 0.22s ease;
}

.profile-signals {
  color: #5f5144;
}

.source-user-badge,
.overlap-pill {
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 700;
}

.badge-self {
  color: #114a7a;
  background: rgba(72, 156, 255, 0.16);
}

.badge-other {
  color: #7f4b00;
  background: rgba(255, 191, 88, 0.2);
}

.overlap-pill {
  color: #2c3a47;
  background: rgba(255,255,255,0.92);
  border: 1px solid rgba(92, 108, 123, 0.12);
}

.source-behavior-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 8px 10px;
}

.source-behavior-row {
  display: grid;
  grid-template-columns: auto auto minmax(0, 1fr);
  gap: 8px;
  align-items: flex-start;
  padding: 8px 10px;
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.9);
  border: 1px solid rgba(63, 53, 45, 0.08);
  text-align: left;
  width: 100%;
}

.source-time {
  color: #667180;
  font-size: 10px;
}

.source-scene {
  padding: 2px 7px;
  border-radius: 999px;
  font-weight: 700;
  font-size: 10px;
  letter-spacing: 0.06em;
}

.scene-search {
  color: #1d5bd1;
  background: rgba(64, 128, 255, 0.14);
}

.scene-rec {
  color: #925400;
  background: rgba(255, 182, 72, 0.18);
}

.source-copy {
  min-width: 0;
}

.source-primary,
.source-secondary {
  display: block;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.source-primary {
  color: #1c2834;
  font-weight: 700;
  font-size: 11px;
}

.source-secondary {
  margin-top: 2px;
  color: #6e5d4c;
  font-size: 10px;
}

.compare-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
}

.compare-cell {
  padding: 12px;
  border-radius: 18px;
  background:
    radial-gradient(circle at top left, rgba(255, 255, 255, 0.46), transparent 22%),
    linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(236, 240, 244, 0.93));
  border: 1px solid rgba(92, 108, 123, 0.12);
  box-shadow: 0 10px 22px rgba(31, 39, 48, 0.07);
}

.compare-cell h4 {
  margin: 0 0 10px;
  font-size: 16px;
}

.compact-table td {
  padding: 7px 8px;
  font-size: 11px;
  vertical-align: top;
}

.compact-table {
  width: 100%;
  table-layout: fixed;
}

.compact-table td:first-child {
  width: 58px;
  color: #0f1720;
  text-align: center;
  white-space: nowrap;
  font-weight: 800;
}

.compact-table td:nth-child(2) {
  width: 92px;
  color: #495666;
  white-space: nowrap;
}

.compact-table td:nth-child(3) {
  width: auto;
}

.rank-col {
  font-weight: 800;
  color: #0f1720;
}

.compare-row {
  cursor: pointer;
  transition: background 0.18s ease;
}

.compare-row:hover {
  background: rgba(156, 118, 75, 0.08);
}

@keyframes validation-spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

.error-panel {
  display: grid;
  gap: 6px;
  border-color: rgba(161, 125, 117, 0.18);
  background:
    radial-gradient(circle at top left, rgba(255, 255, 255, 0.48), transparent 24%),
    linear-gradient(135deg, rgba(244, 241, 239, 0.98), rgba(233, 228, 224, 0.96));
}

.error-kicker {
  font-size: 12px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: #93614d;
}

.error-title {
  font-size: 18px;
  font-weight: 700;
  color: #6b3a31;
}

.error-desc {
  color: #75685e;
  font-size: 13px;
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

.detail-modal,
.formula-modal {
  width: min(760px, 100%);
  max-height: min(88vh, 920px);
  overflow: auto;
  padding: 18px;
  border-radius: 24px;
  background: rgba(250, 251, 252, 0.97);
  box-shadow: 0 24px 54px rgba(17, 23, 31, 0.22);
}

.formula-modal {
  width: min(860px, 100%);
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
  font-size: 24px;
  letter-spacing: -0.03em;
}

.formula-block {
  padding: 16px 18px;
  border-radius: 18px;
  background: rgba(255, 252, 246, 0.96);
  border: 1px solid rgba(203, 58, 34, 0.10);
}

.formula-line {
  font-family: "Times New Roman", "STIX Two Text", "Noto Serif SC", serif;
  font-size: 24px;
  line-height: 1.7;
  color: #1f2937;
  word-break: break-word;
}

.formula-line :deep(.formula-frac) {
  display: inline-flex;
  flex-direction: column;
  align-items: center;
  vertical-align: middle;
  margin: 0 4px;
}

.formula-line :deep(.formula-frac > span:first-child) {
  padding: 0 4px 2px;
  border-bottom: 1px solid rgba(31, 41, 55, 0.38);
}

.formula-line :deep(.formula-frac > span:last-child) {
  padding-top: 2px;
}

.formula-desc {
  margin-top: 12px;
  display: grid;
  gap: 6px;
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

@media (max-width: 1280px) {
  .metrics-grid,
  .compare-grid {
    grid-template-columns: 1fr;
  }

  .source-behavior-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 960px) {
  .metrics-hero,
  .example-head,
  .source-head {
    align-items: flex-start;
    flex-direction: column;
  }

  .metric-row,
  .recall-row,
  .compact-row {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .source-behavior-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .formula-line {
    font-size: 18px;
  }
}

@media (max-width: 640px) {
  .metric-row,
  .recall-row,
  .compact-row,
  .source-behavior-grid {
    grid-template-columns: 1fr;
  }
}
</style>
