/*
前端路由表。
- 登录页：/login
- 业务页：首页/详情/用户画像/指标
*/

import { createRouter, createWebHistory } from 'vue-router';
import { routes } from './routes';

export const router = createRouter({
  history: createWebHistory(),
  routes,
});
