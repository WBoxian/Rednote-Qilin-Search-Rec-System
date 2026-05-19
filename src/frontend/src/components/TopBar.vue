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
            <div class="topbar-kicker">QILIN SEARCH · REC</div>
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
.brand-button {
  padding: 0;
  border: 0;
  background: transparent;
  text-align: left;
  box-shadow: none;
  flex: 0 1 auto;
  min-width: 220px;
}

.brand-button:hover {
  transform: none;
  box-shadow: none;
}

.topbar-sticky-shell {
  position: sticky;
  top: 0;
  z-index: 40;
}

.topbar-sticky {
  position: relative;
}

.topbar-premium {
  padding: 18px 22px;
  background:
    linear-gradient(135deg, rgba(249, 245, 237, 0.94), rgba(242, 236, 226, 0.82)),
    radial-gradient(circle at top left, rgba(193, 65, 42, 0.08), transparent 34%);
}

.topbar-left-premium,
.topbar-side-premium {
  align-items: stretch;
}

.topbar-left-premium {
  gap: 18px;
  flex: 1 1 auto;
  min-width: 0;
}

.topbar-kicker {
  color: #54483c;
  font-weight: 800;
  white-space: nowrap;
  text-shadow: none;
}

.topbar-title {
  color: #1f2731;
  font-weight: 800;
  white-space: nowrap;
  text-shadow: none;
}

.topbar-nav-premium {
  display: grid;
  grid-template-columns: repeat(3, minmax(132px, 1fr));
  gap: 12px;
  flex: 1 1 auto;
  min-width: 0;
}

.nav-tile {
  display: grid;
  gap: 4px;
  min-width: 140px;
  padding: 15px 18px;
  border-radius: 22px;
  border: 1px solid rgba(36, 44, 56, 0.08);
  background: linear-gradient(180deg, rgba(255,255,255,0.92), rgba(245, 239, 230, 0.78));
  box-shadow: 0 14px 28px rgba(24, 33, 43, 0.08);
  transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
}

.nav-tile-button {
  appearance: none;
  text-align: left;
}

.nav-tile:hover {
  transform: translateY(-2px);
  border-color: rgba(24, 33, 43, 0.16);
  box-shadow: 0 18px 30px rgba(24, 33, 43, 0.12);
}

.nav-tile.router-link-active,
.nav-tile.router-link-exact-active {
  background: linear-gradient(135deg, rgba(24, 33, 43, 0.96), rgba(44, 58, 76, 0.92));
  border-color: rgba(24, 33, 43, 0.94);
}

.nav-tile-active {
  background: linear-gradient(135deg, rgba(24, 33, 43, 0.96), rgba(44, 58, 76, 0.92));
  border-color: rgba(24, 33, 43, 0.94);
}

.nav-tile.router-link-active .nav-tile-title,
.nav-tile.router-link-active .nav-tile-meta,
.nav-tile.router-link-exact-active .nav-tile-title,
.nav-tile.router-link-exact-active .nav-tile-meta {
  color: #f7f2eb;
}

.nav-tile-active .nav-tile-title,
.nav-tile-active .nav-tile-meta {
  color: #f7f2eb;
}

.nav-tile-title {
  font-size: 18px;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: #18212b;
  white-space: nowrap;
}

.nav-tile-meta {
  font-size: 11px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: #847565;
  white-space: nowrap;
}

.metric-chip-wide {
  min-width: 128px;
}

.topbar-metric-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 14px;
}

.metric-chip-soft {
  background: rgba(255, 255, 255, 0.78);
}

.metric-value-topbar {
  margin-top: 0;
  font-size: 13px;
  font-weight: 700;
  letter-spacing: 0;
  white-space: nowrap;
}

.topbar-metrics-premium {
  flex: 1 1 auto;
  justify-content: flex-start;
  flex-wrap: nowrap;
  overflow-x: auto;
  scrollbar-width: none;
}

.topbar-metrics-premium::-webkit-scrollbar {
  display: none;
}

.topbar-side-premium {
  flex: 0 0 auto;
  justify-content: flex-end;
  flex-wrap: nowrap;
}

.user-chip-strong {
  background: linear-gradient(135deg, rgba(203, 58, 34, 0.14), rgba(203, 58, 34, 0.06));
}

.logout-button {
  min-height: 44px;
}

.topbar-actions-premium {
  justify-content: center;
  min-width: 0;
}

@media (max-width: 1280px) {
  .topbar-main {
    flex-direction: column;
    align-items: stretch;
  }

  .topbar-nav-premium {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }

  .topbar-side-premium {
    justify-content: flex-start;
  }
}

@media (max-width: 1100px) {
  .topbar-sticky-shell {
    position: static;
  }

  .topbar-metrics-premium,
  .topbar-side-premium,
  .topbar-left-premium {
    flex-wrap: wrap;
  }

  .topbar-metric-row {
    flex-direction: column;
    align-items: stretch;
  }
}

@media (max-width: 960px) {
  .topbar-nav-premium {
    grid-template-columns: 1fr;
  }

  .topbar-left-premium {
    flex-direction: column;
  }

  .topbar-metrics-premium,
  .topbar-side-premium {
    justify-content: flex-start;
  }

  .nav-tile {
    min-width: 0;
  }
}
</style>
