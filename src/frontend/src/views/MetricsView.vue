<script setup lang="ts">
import { onMounted, ref } from 'vue';
import TopBar from '../components/TopBar.vue';
import { api, getScene, getUserId, setScene } from '../services/api';

const metricsByScene = ref<Record<string, any>>({});
const validationByScene = ref<Record<string, any>>({});
const userByScene = ref<Record<string, any>>({});
const currentScene = ref<'search' | 'rec'>(getScene());
const loading = ref(false);
const errorMsg = ref('');
const latencyMs = ref<number | null>(null);

function fmtNum(v: any, digits = 4) {
  const n = Number(v);
  if (!Number.isFinite(n)) return '-';
  return n.toFixed(digits);
}

function valExamples(scene: 'search' | 'rec') {
  return validationByScene.value?.[scene]?.examples || [];
}

function userContext(scene: 'search' | 'rec') {
  const fromVal = validationByScene.value?.[scene]?.context_user;
  if (fromVal) return fromVal;
  const u = userByScene.value?.[scene];
  return {
    user_idx: u?.user_idx,
    features: u?.features || {},
    recent_behavior_titles: (u?.recent_behaviors || []).map((x: any) => String(x?.title || '(无标题)')).slice(0, 20),
  };
}

function behaviorTitlePairs(scene: 'search' | 'rec') {
  const arr = (userContext(scene)?.recent_behavior_titles || []).slice(0, 20);
  const left = arr.slice(0, 10);
  const right = arr.slice(10, 20);
  const rows: Array<{ leftIdx: number; leftTitle: string; rightIdx: number | null; rightTitle: string }> = [];
  const n = Math.max(left.length, right.length);
  for (let i = 0; i < n; i++) {
    rows.push({
      leftIdx: i + 1,
      leftTitle: left[i] || '',
      rightIdx: (i < right.length ? i + 11 : null),
      rightTitle: right[i] || '',
    });
  }
  return rows;
}

function switchScene(scene: 'search' | 'rec') {
  currentScene.value = scene;
  setScene(scene);
  load();
}

async function load() {
  loading.value = true;
  errorMsg.value = '';
  try {
    const uid = getUserId();
    const scene = currentScene.value;
    const nextMetrics: Record<string, any> = { ...(metricsByScene.value || {}) };
    const nextValidation: Record<string, any> = { ...(validationByScene.value || {}) };
    const nextUser: Record<string, any> = { ...(userByScene.value || {}) };
    let totalLatency = 0;
    const [valRes, metricRes, userRes, userOtherRes] = await Promise.allSettled([
      api.validation(scene, 80, 3, uid),
      api.metrics(scene, 500, 300),
      uid == null ? Promise.resolve(null) : api.user(scene, uid),
      uid == null ? Promise.resolve(null) : api.user(scene === 'search' ? 'rec' : 'search', uid),
    ]);
    if (valRes.status === 'fulfilled') {
      nextValidation[scene] = valRes.value.validation;
      totalLatency += Number(valRes.value?.latency_ms || 0);
    }
    if (metricRes.status === 'fulfilled') {
      nextMetrics[scene] = metricRes.value.metrics;
      totalLatency += Number(metricRes.value?.latency_ms || 0);
    }
    if (userRes.status === 'fulfilled' && userRes.value && userRes.value.user) {
      nextUser[scene] = userRes.value.user;
      totalLatency += Number(userRes.value?.latency_ms || 0);
    }
    const otherScene = scene === 'search' ? 'rec' : 'search';
    if (userOtherRes.status === 'fulfilled' && userOtherRes.value && userOtherRes.value.user) {
      nextUser[otherScene] = userOtherRes.value.user;
      totalLatency += Number(userOtherRes.value?.latency_ms || 0);
    }
    metricsByScene.value = nextMetrics;
    validationByScene.value = nextValidation;
    userByScene.value = nextUser;
    latencyMs.value = totalLatency;
    if (!Object.keys(nextMetrics).length && !Object.keys(nextValidation).length) {
      errorMsg.value = '指标加载失败';
    }
  } catch (e: any) {
    errorMsg.value = e?.message || '指标加载失败';
  } finally {
    loading.value = false;
  }
}

onMounted(() => load());
</script>

<template>
  <TopBar title="小红书麒麟推荐" :latency-ms="latencyMs" />
  <main class="container">
    <section class="panel" style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;">
      <button class="btn-tab" :class="{ active: currentScene === 'search' }" :disabled="loading" @click="switchScene('search')">Search</button>
      <button class="btn-tab" :class="{ active: currentScene === 'rec' }" :disabled="loading" @click="switchScene('rec')">Rec</button>
      <div class="meta">{{ loading ? '加载中...' : '已加载' }}</div>
    </section>

    <section v-if="errorMsg" class="panel">{{ errorMsg }}</section>

    <template v-for="scene in [currentScene]" :key="`scene-${scene}`">
    <section v-if="metricsByScene[scene]" class="panel">
      <h2 style="margin:0 0 8px;">{{ scene.toUpperCase() }} · Test 指标概览</h2>
      <div class="meta" style="margin-bottom:10px;">tag={{ metricsByScene[scene]?.tag || '-' }}</div>
      <div class="metric-grid">
        <div class="metric-item">
          <div class="meta">GBDT NDCG@10</div>
          <div class="metric-value" style="font-size:30px;">{{ fmtNum(metricsByScene[scene]?.gbdt?.['ndcg@10_exposed_test']) }}</div>
        </div>
        <div class="metric-item">
          <div class="meta">GBDT NDCG@100</div>
          <div class="metric-value" style="font-size:30px;">{{ fmtNum(metricsByScene[scene]?.gbdt?.['ndcg@100_exposed_test']) }}</div>
        </div>
        <div class="metric-item">
          <div class="meta">DIEN NDCG@10</div>
          <div class="metric-value" style="font-size:30px;">{{ fmtNum(metricsByScene[scene]?.dien?.['ndcg@10_exposed_test_sampled']) }}</div>
        </div>
        <div class="metric-item">
          <div class="meta">评估样本数</div>
          <div class="metric-value" style="font-size:30px;">{{ metricsByScene[scene]?.gbdt?.sampled_eval_groups ?? metricsByScene[scene]?.gbdt?.eval_groups ?? '-' }}</div>
        </div>
      </div>
    </section>

    <section v-if="validationByScene[scene]" class="panel">
      <h2 style="margin:0 0 8px;">{{ scene.toUpperCase() }} · Validation 排序对比</h2>
      <h3 style="margin:0 0 10px;">Validation 排序对比总结</h3>
      <div class="meta" style="margin-bottom:10px;">
        当前用户组命中：{{ validationByScene[scene]?.context_user_group_count ?? 0 }}；DSSM分数字段：{{ validationByScene[scene]?.dssm_score_source || 'none' }}
      </div>
      <div v-if="validationByScene[scene]?.dssm_validation_note" class="meta" style="margin-bottom:10px;color:#b45309;">
        {{ validationByScene[scene]?.dssm_validation_note }}
      </div>
      <div class="metric-grid">
        <div class="metric-item">
          <div class="meta">{{ validationByScene[scene]?.dssm_score_source === 'none' ? 'DSSM NDCG@10(不可用)' : 'DSSM NDCG@10' }}</div>
          <div class="metric-value" style="font-size:30px;">{{ validationByScene[scene]?.dssm_score_source === 'none' ? '-' : fmtNum(validationByScene[scene]?.dssm?.['ndcg@10']) }}</div>
        </div>
        <div class="metric-item">
          <div class="meta">GBDT NDCG@10</div>
          <div class="metric-value" style="font-size:30px;">{{ fmtNum(validationByScene[scene]?.gbdt?.['ndcg@10']) }}</div>
        </div>
        <div class="metric-item">
          <div class="meta">DIEN NDCG@10</div>
          <div class="metric-value" style="font-size:30px;">{{ fmtNum(validationByScene[scene]?.dien?.['ndcg@10']) }}</div>
        </div>
        <div class="metric-item">
          <div class="meta">评估样本组数</div>
          <div class="metric-value" style="font-size:30px;">{{ validationByScene[scene]?.sampled_groups ?? '-' }}</div>
        </div>
      </div>

      <h3 style="margin:14px 0 8px;">当前用户画像与最近行为</h3>
      <table class="table">
        <tbody>
          <tr>
            <td>user_idx</td><td>{{ userContext(scene as 'search' | 'rec')?.user_idx ?? '-' }}</td>
            <td>gender</td><td>{{ userContext(scene as 'search' | 'rec')?.features?.gender ?? userContext(scene as 'search' | 'rec')?.features?.gender_enc ?? '-' }}</td>
            <td>age</td><td>{{ userContext(scene as 'search' | 'rec')?.features?.age ?? userContext(scene as 'search' | 'rec')?.features?.age_enc ?? '-' }}</td>
          </tr>
          <tr>
            <td>platform</td><td>{{ userContext(scene as 'search' | 'rec')?.features?.platform ?? userContext(scene as 'search' | 'rec')?.features?.platform_enc ?? '-' }}</td>
            <td>location</td><td>{{ userContext(scene as 'search' | 'rec')?.features?.location ?? userContext(scene as 'search' | 'rec')?.features?.location_enc ?? '-' }}</td>
            <td>fans/follows</td><td>{{ userContext(scene as 'search' | 'rec')?.features?.fans_num ?? '-' }} / {{ userContext(scene as 'search' | 'rec')?.features?.follows_num ?? '-' }}</td>
          </tr>
        </tbody>
      </table>

      <h4 style="margin:12px 0 6px;">Search 最近20条行为</h4>
      <table class="table" style="margin-top:0;">
        <tbody>
          <tr>
            <td style="width:64px;">序号</td><td>标题(1-10)</td>
            <td style="width:64px;">序号</td><td>标题(11-20)</td>
          </tr>
          <tr v-for="row in behaviorTitlePairs('search')" :key="`search-pair-${row.leftIdx}`">
            <td>{{ row.leftIdx }}</td>
            <td>{{ row.leftTitle || '-' }}</td>
            <td>{{ row.rightIdx ?? '-' }}</td>
            <td>{{ row.rightTitle || '-' }}</td>
          </tr>
          <tr v-if="!behaviorTitlePairs('search').length">
            <td colspan="4" class="meta">暂无 Search 行为记录。</td>
          </tr>
        </tbody>
      </table>

      <h4 style="margin:12px 0 6px;">Rec 最近20条行为</h4>
      <table class="table" style="margin-top:0;">
        <tbody>
          <tr>
            <td style="width:64px;">序号</td><td>标题(1-10)</td>
            <td style="width:64px;">序号</td><td>标题(11-20)</td>
          </tr>
          <tr v-for="row in behaviorTitlePairs('rec')" :key="`rec-pair-${row.leftIdx}`">
            <td>{{ row.leftIdx }}</td>
            <td>{{ row.leftTitle || '-' }}</td>
            <td>{{ row.rightIdx ?? '-' }}</td>
            <td>{{ row.rightTitle || '-' }}</td>
          </tr>
          <tr v-if="!behaviorTitlePairs('rec').length">
            <td colspan="4" class="meta">暂无 Rec 行为记录。</td>
          </tr>
        </tbody>
      </table>

      <h3 style="margin:14px 0 8px;">样例对比（含标题）</h3>
      <div v-if="validationByScene[scene]?.context_user_sample_note" class="meta" style="margin-bottom:10px;color:#b45309;">
        {{ validationByScene[scene]?.context_user_sample_note }}
      </div>
      <section v-for="ex in valExamples(scene as 'search' | 'rec').slice(0, 3)" :key="`${ex.group_field}-${ex.group_value}`" class="panel" style="margin:10px 0;">
        <div class="meta" style="margin-bottom:8px;">{{ ex.group_field || 'group' }}={{ ex.group_value ?? '-' }}</div>
        <div style="font-size:18px;font-weight:700;margin:6px 0 10px;line-height:1.5;white-space:pre-wrap;word-break:break-all;">
          Query：{{ ex.query || '-' }}
        </div>
        <table class="table">
          <tbody>
            <tr>
              <td style="width:20%;">真实 Top10</td>
              <td>
                <table class="table" style="margin:0;">
                  <tbody>
                    <tr><td style="width:18%;">真实位次</td><td style="width:20%;">note_idx</td><td>标题</td></tr>
                    <tr v-for="(item, idx) in ex.true_top10" :key="`t-${item.note_idx}`">
                      <td>#{{ Number(idx) + 1 }}</td>
                      <td>{{ item.note_idx }}</td>
                      <td>{{ item.title }}</td>
                    </tr>
                  </tbody>
                </table>
              </td>
            </tr>
            <tr>
              <td>DSSM Top10</td>
              <td>
                <table class="table" style="margin:0;">
                  <tbody>
                    <tr><td style="width:18%;">真实位次</td><td style="width:20%;">note_idx</td><td>标题</td></tr>
                    <tr v-for="item in ex.dssm_top10" :key="`s-${item.note_idx}`">
                      <td>#{{ item.rank_in_true ?? '-' }}</td>
                      <td>{{ item.note_idx }}</td>
                      <td>{{ item.title }}</td>
                    </tr>
                  </tbody>
                </table>
              </td>
            </tr>
            <tr>
              <td>GBDT Top10</td>
              <td>
                <table class="table" style="margin:0;">
                  <tbody>
                    <tr><td style="width:18%;">真实位次</td><td style="width:20%;">note_idx</td><td>标题</td></tr>
                    <tr v-for="item in ex.gbdt_top10" :key="`g-${item.note_idx}`">
                      <td>#{{ item.rank_in_true ?? '-' }}</td>
                      <td>{{ item.note_idx }}</td>
                      <td>{{ item.title }}</td>
                    </tr>
                  </tbody>
                </table>
              </td>
            </tr>
            <tr>
              <td>DIEN Top10</td>
              <td>
                <table class="table" style="margin:0;">
                  <tbody>
                    <tr><td style="width:18%;">真实位次</td><td style="width:20%;">note_idx</td><td>标题</td></tr>
                    <tr v-for="item in ex.dien_top10" :key="`d-${item.note_idx}`">
                      <td>#{{ item.rank_in_true ?? '-' }}</td>
                      <td>{{ item.note_idx }}</td>
                      <td>{{ item.title }}</td>
                    </tr>
                  </tbody>
                </table>
              </td>
            </tr>
          </tbody>
        </table>
      </section>
    </section>
    </template>
  </main>
</template>
