import { setup, type Preview } from '@storybook/vue3-vite';
import { createPinia } from 'pinia';
import { createMemoryHistory, createRouter } from 'vue-router';
import { mswLoader } from 'msw-storybook-addon/csf3';
import { routes } from '../src/router/routes';
import { mswHandlers } from './msw-handlers';
import '../src/styles.css';
import './preview.css';

const router = createRouter({ history: createMemoryHistory(), routes });
setup(app => {
  app.use(createPinia());
  app.use(router);
});

const preview: Preview = {
  decorators: [() => ({ template: '<div class="storybook-fixture-note" role="note">Storybook 状态演示 · 指标为冻结评测快照，内容和用户为界面测试样例。</div><story />' })],
  loaders: [mswLoader()],
  async beforeEach(context) {
    const { msw, parameters, id } = context as typeof context & { msw: { use: (...handlers: typeof mswHandlers) => void } };
    msw.use(...mswHandlers);
    localStorage.setItem('qilin_scene', parameters.scene || 'search');
    localStorage.setItem('qilin_user_idx', String(20000 + [...id].reduce((sum, char) => sum + char.charCodeAt(0), 0)));
    sessionStorage.removeItem('qilin_home_cache_v2');
    sessionStorage.removeItem('qilin_detail_seed');
    await router.replace(parameters.route || '/metrics');
  },
  parameters: {
    layout: 'fullscreen',
    controls: {
      matchers: {
       color: /(background|color)$/i,
       date: /Date$/i,
      },
    },
  },
};

export default preview;