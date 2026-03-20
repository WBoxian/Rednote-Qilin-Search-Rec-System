<!-- 顶部导航：统一页面跳转与退出登录 -->
<script setup lang="ts">
import { useRouter } from 'vue-router';
import { clearUserId, getUserId } from '../services/api';

const props = defineProps<{
  title?: string;
  latencyMs?: number | null;
}>();
const router = useRouter();

const userId = getUserId();

function fmtLatency(v?: number | null) {
  const n = Number(v);
  if (!Number.isFinite(n) || n < 0) return '--';
  return `${n.toFixed(1)}ms`;
}

function logout() {
  // 清理本地登录态并返回登录页
  clearUserId();
  router.push('/login');
}
</script>

<template>
  <header class="topbar">
    <div class="topbar-inner">
      <router-link :to="{ path: '/', query: { preserve: '1' } }" class="brand-link">{{ props.title || '麒麟推荐' }}</router-link>
      <slot name="actions" />
      <router-link to="/user" class="btn">用户画像</router-link>
      <router-link to="/metrics" class="btn">指标</router-link>
      <button class="btn" style="margin-left:auto;" @click="logout">退出</button>
      <span class="meta">延迟={{ fmtLatency(props.latencyMs) }}</span>
      <span class="meta">user={{ userId ?? '-' }}</span>
    </div>
  </header>
</template>
