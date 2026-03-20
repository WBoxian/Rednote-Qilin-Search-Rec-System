<script setup lang="ts">
import { onMounted, ref } from 'vue';
import TopBar from '../components/TopBar.vue';
import { api, getScene, getUserId } from '../services/api';

/** 按场景分别存储 api.user() 返回的 user 对象 */
const userByScene = ref<Record<string, any>>({});
const loading = ref(false);
const errorMsg = ref('');
const latencyMs = ref<number | null>(null);

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

async function load() {
  const uid = getUserId();
  if (uid == null) {
    errorMsg.value = '未检测到登录用户，请先登录';
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
      nextByScene['search'] = resSearch.value.user;
      totalLatency += Number(resSearch.value?.latency_ms || 0);
    }
    if (resRec.status === 'fulfilled' && resRec.value?.user) {
      nextByScene['rec'] = resRec.value.user;
      totalLatency += Number(resRec.value?.latency_ms || 0);
    }
    userByScene.value = nextByScene;
    latencyMs.value = totalLatency;
    if (!Object.keys(nextByScene).length) {
      errorMsg.value = '未找到任何场景的用户画像';
    }
  } catch (e: any) {
    userByScene.value = {};
    errorMsg.value = e?.message || '用户画像加载失败';
  } finally {
    loading.value = false;
  }
}

function dedupBehaviors(rows: any[]): any[] {
  const out: any[] = [];
  const seen = new Set<string>();
  for (const row of (rows || [])) {
    const key = `${row?.note_idx}|${row?.request_id}|${row?.query || ''}|${row?.scene || ''}`;
    if (seen.has(key)) continue;
    out.push(row);
    seen.add(key);
  }
  return out;
}

onMounted(load);
</script>

<template>
  <TopBar title="小红书麒麟推荐" :latency-ms="latencyMs" />
  <main class="container">
    <section class="panel" style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;">
      <div class="meta">{{ loading ? '加载中...' : '用户画像' }}</div>
      <div class="meta">scene={{ getScene() }} | user={{ getUserId() ?? '-' }}</div>
    </section>

    <section v-if="errorMsg" class="panel">{{ errorMsg }}</section>

    <template v-else-if="Object.keys(userByScene).length">
      <section class="panel">
        <div class="meta" style="margin-bottom:10px;">
          request_count(search)={{ userByScene['search']?.request_count_in_test ?? '-' }}
          | request_count(rec)={{ userByScene['rec']?.request_count_in_test ?? '-' }}
        </div>
        <table class="table">
          <tbody>
            <tr>
              <td>gender</td><td>{{ fmt((userByScene['search'] || userByScene['rec'])?.features?.gender) }}</td>
              <td>age</td><td>{{ fmt((userByScene['search'] || userByScene['rec'])?.features?.age) }}</td>
            </tr>
            <tr>
              <td>platform</td><td>{{ fmt((userByScene['search'] || userByScene['rec'])?.features?.platform) }}</td>
              <td>location</td><td>{{ fmt((userByScene['search'] || userByScene['rec'])?.features?.location) }}</td>
            </tr>
            <tr>
              <td>fans_num</td><td>{{ fmt((userByScene['search'] || userByScene['rec'])?.features?.fans_num) }}</td>
              <td>follows_num</td><td>{{ fmt((userByScene['search'] || userByScene['rec'])?.features?.follows_num) }}</td>
            </tr>
          </tbody>
        </table>
      </section>

      <section class="panel">
        <h3 style="margin:0 0 8px;">Search 最近20条行为</h3>
        <table class="table" v-if="dedupBehaviors(userByScene['search']?.recent_behaviors || []).length">
          <tbody>
            <tr><td style="width:62%;">行为详情</td><td>对应标题</td></tr>
            <tr v-for="(row, i) in dedupBehaviors(userByScene['search']?.recent_behaviors || [])" :key="`s-${row.ts}-${row.note_idx}-${row.action}-${i}`">
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
        <div v-else class="meta">暂无 Search 行为记录。</div>
      </section>

      <section class="panel">
        <h3 style="margin:0 0 8px;">Rec 最近20条行为</h3>
        <table class="table" v-if="dedupBehaviors(userByScene['rec']?.recent_behaviors || []).length">
          <tbody>
            <tr><td style="width:62%;">行为详情</td><td>对应标题</td></tr>
            <tr v-for="(row, i) in dedupBehaviors(userByScene['rec']?.recent_behaviors || [])" :key="`r-${row.ts}-${row.note_idx}-${row.action}-${i}`">
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
        <div v-else class="meta">暂无 Rec 行为记录。</div>
      </section>
    </template>

    <section v-else class="panel">暂无画像数据。</section>
  </main>
</template>
