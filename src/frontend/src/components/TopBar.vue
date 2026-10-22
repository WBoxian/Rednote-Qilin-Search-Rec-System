<script setup lang="ts">
import { computed } from 'vue';
import { useRoute, useRouter } from 'vue-router';
import { clearUserId, getUserId } from '../services/api';

const props = defineProps<{
  title: string;
  latencyMs?: number | null;
  stageMs?: Record<string, number> | null;
  sceneMode?: string | null;
}>();

const emit = defineEmits<{
  (e: 'home'): void;
}>();

const router = useRouter();
const route = useRoute();
const userId = getUserId();

const serviceLatency = computed(() => {
  const n = Number(props.latencyMs ?? 0);
  const hasLatency = props.latencyMs != null;
  const src = props.stageMs && typeof props.stageMs === 'object' ? props.stageMs : {};
  const stageTotal = ['recall', 'preranking', 'ranking'].reduce((sum, key) => {
    const value = Number((src as any)?.[key] ?? 0);
    return Number.isFinite(value) && value > 0 ? sum + value : sum;
  }, 0);
  const display = Math.max(
    Number.isFinite(n) && n > 0 ? n : 0,
    Number.isFinite(stageTotal) && stageTotal > 0 ? stageTotal : 0,
  );
  return display > 0 || hasLatency || props.stageMs ? `${display.toFixed(1)} ms` : '--';
});

const stageEntries = computed(() => {
  const src = props.stageMs && typeof props.stageMs === 'object' ? props.stageMs : {};
  const pairs = [
    ['\u53ec\u56de', Number(src?.recall ?? 0)],
    ['\u7c97\u6392', Number(src?.preranking ?? 0)],
    ['\u7cbe\u6392', Number(src?.ranking ?? 0)],
  ] as Array<[string, number]>;
  return pairs.map(([label, value]) => ({
    label,
    value: Number.isFinite(value) ? `${Math.max(0, value).toFixed(1)} ms` : '--',
  }));
});

const navItems = [
  { to: '/', title: '\u9996\u9875', meta: 'Feed / Search' },
  { to: '/user', title: '\u6700\u8fd1\u884c\u4e3a', meta: 'Behavior Stream' },
  { to: '/metrics', title: '\u6307\u6807\u770b\u677f', meta: 'Metrics / Compare' },
];

function logout() {
  clearUserId();
  router.push('/login');
}

function goBrandHome() {
  if (route.path === '/') {
    emit('home');
    return;
  }
  router.push({ path: '/', query: { refresh: String(Date.now()) } });
}

function goHomeNav() {
  if (route.path === '/') {
    emit('home');
    return;
  }
  router.push({ path: '/', query: { refresh: String(Date.now()) } });
}
</script>

<template>
  <header class="topbar-shell topbar-sticky-shell">
    <div class="topbar topbar-premium topbar-sticky">
      <div class="topbar-main">
        <div class="topbar-left topbar-left-premium">
          <button class="topbar-brand brand-button" @click="goBrandHome">
            <span class="brand-symbol" aria-hidden="true">Q</span>
            <div class="topbar-kicker">QILIN WORKSPACE</div>
            <div class="topbar-title">{{ props.title }}</div>
          </button>

          <nav class="topbar-nav topbar-nav-premium">
            <button
              type="button"
              class="nav-tile nav-tile-button"
              :class="{ 'nav-tile-active': route.path === '/' }"
              @click="goHomeNav"
            >
              <span class="nav-tile-title">{{ navItems[0].title }}</span>
              <span class="nav-tile-meta">{{ navItems[0].meta }}</span>
            </button>
            <router-link v-for="item in navItems.slice(1)" :key="item.to" :to="item.to" class="nav-tile">
              <span class="nav-tile-title">{{ item.title }}</span>
              <span class="nav-tile-meta">{{ item.meta }}</span>
            </router-link>
          </nav>
        </div>
      </div>

      <div class="topbar-metric-row">
        <div class="topbar-metrics topbar-metrics-premium">
          <div class="metric-chip metric-chip-strong metric-chip-wide">
            <span class="metric-label">&#x670D;&#x52A1;&#x5EF6;&#x8FDF;</span>
            <span class="metric-value metric-value-topbar">{{ serviceLatency }}</span>
          </div>
          <div v-for="item in stageEntries" :key="item.label" class="metric-chip metric-chip-soft">
            <span class="metric-label">{{ item.label }}</span>
            <span class="metric-value metric-value-topbar">{{ item.value }}</span>
          </div>
        </div>
        <div class="topbar-side topbar-side-premium">
          <div class="user-chip user-chip-strong">
            <span class="user-chip-label">{{ props.sceneMode ? String(props.sceneMode).toUpperCase() : 'ONLINE' }}</span>
            <span class="user-chip-value">UID {{ userId ?? '-' }}</span>
          </div>
          <button class="nav-chip nav-chip-ghost logout-button" @click="logout">&#x9000;&#x51FA;</button>
        </div>
      </div>

      <div v-if="$slots.actions" class="topbar-actions topbar-actions-premium">
        <slot name="actions" />
      </div>
    </div>
  </header>
</template>

<style scoped>
.topbar-shell { position: sticky; top: 0; z-index: 40; background: #ffffffed; backdrop-filter: blur(16px); border-bottom: 1px solid var(--line); }
.topbar { max-width: 1440px; margin: auto; padding: 16px 32px 12px; display: grid; gap: 12px; }
.topbar-main,.topbar-left,.topbar-metric-row,.topbar-metrics,.topbar-side { display: flex; align-items: center; }
.topbar-left { width: 100%; gap: 40px; }
.brand-button { display: grid; grid-template-columns: 36px auto; column-gap: 10px; text-align: left; padding: 0; border: 0; background: transparent; min-width: 180px; }
.brand-button:hover { background: transparent; }
.brand-symbol { grid-row: 1 / 3; display: grid; place-items: center; width: 36px; height: 36px; border-radius: 10px; background: var(--accent); color: white; font-size: 22px; font-weight: 750; }
.topbar-kicker { font-size: 10px; letter-spacing: .12em; color: var(--muted); line-height: 1.4; }
.topbar-title { font-size: 15px; font-weight: 650; }
.topbar-nav { display: flex; gap: 6px; flex: 1; }
.nav-tile { display: flex; align-items: center; gap: 7px; padding: 9px 14px; border-radius: 8px; border: 1px solid transparent; background: transparent; }
.nav-tile:hover { background: var(--bg); }
.nav-tile-title { font-size: 13px; font-weight: 550; white-space: nowrap; }
.nav-tile-meta { display: none; }
.nav-tile-active,.nav-tile.router-link-exact-active { color: var(--accent); background: var(--accent-soft); border-color: #dbe6fd; }
.topbar-metric-row { justify-content: space-between; gap: 12px; border-top: 1px solid var(--line); padding-top: 10px; }
.topbar-metrics { gap: 22px; flex-wrap: wrap; }
.topbar-side { gap: 14px; }
.user-chip { gap: 10px; }
.user-chip-label { color: var(--accent); background: var(--accent-soft); padding: 3px 7px; border-radius: 4px; }
.logout-button { padding: 4px 9px; font-size: 12px; }
.topbar-actions { display: flex; justify-content: center; padding: 6px 0 2px; }
@media(max-width:800px) {
  .topbar { padding: 14px 16px 10px; }
  .topbar-left { flex-wrap: wrap; gap: 14px; }
  .topbar-nav { flex: 1; justify-content: flex-end; }
  .nav-tile { padding: 8px 10px; }
  .topbar-metrics { gap: 12px; }
  .metric-chip-soft { display: none; }
}
@media(max-width:480px) {
  .topbar-left { display: grid; gap: 12px; }
  .topbar-nav { justify-content: stretch; }
  .nav-tile { flex: 1; justify-content: center; }
  .topbar-metric-row { align-items: flex-start; }
}
</style>
