/*
前端路由表。
- 登录页：/login
- 业务页：首页/详情/用户画像/指标
*/

import { createRouter, createWebHistory } from 'vue-router';
import LoginView from '../views/LoginView.vue';
import HomeView from '../views/HomeView.vue';
import DetailView from '../views/DetailView.vue';
import UserView from '../views/UserView.vue';
import MetricsView from '../views/MetricsView.vue';

export const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/login', component: LoginView },
    { path: '/', component: HomeView },
    { path: '/detail', component: DetailView },
    { path: '/user', component: UserView },
    { path: '/metrics', component: MetricsView },
  ],
});
