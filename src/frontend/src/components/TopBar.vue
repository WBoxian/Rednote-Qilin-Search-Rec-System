<!-- 顶部导航：统一页面跳转与退出登录 -->
<script setup lang="ts">
import { useRouter } from 'vue-router';
import { clearUserId, getUserId } from '../services/api';

const props = defineProps<{ title: string }>();
const router = useRouter();

const userId = getUserId();

function logout() {
  // 清理本地登录态并返回登录页
  clearUserId();
  router.push('/login');
}
</script>

<template>
  <header class="topbar">
    <div class="topbar-inner">
      <div class="brand">{{ props.title }}</div>
      <router-link to="/" class="btn">首页</router-link>
      <router-link to="/user" class="btn">用户画像</router-link>
      <router-link to="/metrics" class="btn">指标</router-link>
      <button class="btn" style="margin-left:auto;" @click="logout">退出</button>
      <span class="meta">user={{ userId ?? '-' }}</span>
    </div>
  </header>
</template>
