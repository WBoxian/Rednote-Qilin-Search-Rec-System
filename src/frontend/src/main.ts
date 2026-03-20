/*
前端应用入口。
- 挂载 Vue 应用
- 注册 Pinia 状态管理
- 注册路由
*/

import { createApp } from 'vue';
import { createPinia } from 'pinia';
import App from './App.vue';
import { router } from './router';
import './styles.css';

const app = createApp(App);
app.use(createPinia());
app.use(router);
app.mount('#app');
