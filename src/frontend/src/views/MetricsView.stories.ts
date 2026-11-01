import type { Meta, StoryObj } from '@storybook/vue3-vite';
import { expect } from 'storybook/test';
import { http, HttpResponse } from 'msw';
import MetricsView from './MetricsView.vue';

const meta = { component: MetricsView, tags: ['ai-generated'], parameters: { route: '/metrics', scene: 'search' } } satisfies Meta<typeof MetricsView>;
export default meta;
type Story = StoryObj<typeof meta>;

export const SearchMetrics: Story = {
  beforeEach: async context => {
    const currentUid = Number(localStorage.getItem('qilin_user_idx'));
    const clicked = { note_idx: 101, title: '样例目标笔记', rank_in_true: 1 };
    const example = { scope: 'frozen_full_test_examples', scene: 'search', result_count: 12, query: '测试查询', true_top10: [clicked], dssm_top10: [clicked], preranking_top10: [clicked], ranking_top10: [clicked], source_recent_behaviors: [], overlap: { dssm: 1, preranking: 1, ranking: 1 } };
    (context as any).msw.use(http.get('/api/validation', () => HttpResponse.json({ validation: { examples: [{ ...example, request_id: 7, source_user_idx: 8 }, { ...example, request_id: 9, source_user_idx: currentUid }], scope: 'frozen_full_test_examples' } })));
  },
  play: async ({ canvas, userEvent }) => {
    await expect(await canvas.findByRole('heading', { name: '最终结果与基线提升' })).toBeVisible();
    await expect(canvas.getByText('下方数值保留为冻结的日志曝光池诊断，尚未按当前服务口径重新评测，不代表端到端服务效果。')).toBeVisible();
    await expect(await canvas.findByText('当前用户')).toBeVisible();
    await expect(canvas.getByText(/请求编号=9/)).toBeVisible();
    await expect(canvas.queryByText(/✓/)).not.toBeInTheDocument();
    const metric = await canvas.findByRole('button', { name: /命中率（HitRate@10）/ });
    await userEvent.click(metric);
    await expect(await canvas.findByRole('heading', { name: /召回.*HitRate@10/ })).toBeVisible();
    await userEvent.click(canvas.getByRole('button', { name: '关闭' }));
    await expect(canvas.queryByRole('heading', { name: /召回.*HitRate@10/ })).not.toBeInTheDocument();
    await userEvent.click(canvas.getByText('阶段消融与实验结果'));
    await expect(canvas.getByRole('columnheader', { name: '实验' })).toBeVisible();
  },
};
export const RecommendationMetrics: Story = {
  parameters: { scene: 'rec' },
  play: async ({ canvas, userEvent }) => {
    await expect(await canvas.findByRole('button', { name: /命中率（HitRate@50）/ })).toBeVisible();
    await expect(canvas.getByText(/当前服务：仅召回/)).toBeVisible();
    await userEvent.click(canvas.getByText('阶段消融与实验结果'));
    await expect(canvas.getByRole('rowheader', { name: '内容残差与验证集选择的融合' })).toBeVisible();
  },
};
export const MetricsUnavailable: Story = {
  beforeEach: async context => {
    (context as any).msw.use(http.get('/api/metrics', () => HttpResponse.json({ detail: '演示：指标版本未就绪' }, { status: 503 })));
  },
  play: async ({ canvas }) => { await expect(await canvas.findByText('演示：指标版本未就绪')).toBeVisible(); },
};
