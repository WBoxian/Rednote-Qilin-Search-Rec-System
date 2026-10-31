import type { Meta, StoryObj } from '@storybook/vue3-vite';
import { expect } from 'storybook/test';
import { http, HttpResponse } from 'msw';
import MetricsView from './MetricsView.vue';

const meta = { component: MetricsView, tags: ['ai-generated'], parameters: { route: '/metrics', scene: 'search' } } satisfies Meta<typeof MetricsView>;
export default meta;
type Story = StoryObj<typeof meta>;

export const SearchMetrics: Story = {
  play: async ({ canvas, userEvent }) => {
    await expect(await canvas.findByRole('heading', { name: '最终结果与基线提升' })).toBeVisible();
    await expect(canvas.getByText('下方数值保留为冻结的日志曝光池诊断，尚未按当前服务口径重新评测，不代表端到端服务效果。')).toBeVisible();
    const metric = await canvas.findByRole('button', { name: /^HitRate@10 / });
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
    await expect(await canvas.findByRole('button', { name: /^HitRate@50 / })).toBeVisible();
    await expect(canvas.getByText(/当前服务：仅召回/)).toBeVisible();
    await userEvent.click(canvas.getByText('阶段消融与实验结果'));
    await expect(canvas.getByRole('rowheader', { name: 'Residual DSSM + validation-selected fusion' })).toBeVisible();
  },
};
export const MetricsUnavailable: Story = {
  beforeEach: async context => {
    (context as any).msw.use(http.get('/api/metrics', () => HttpResponse.json({ detail: '演示：指标版本未就绪' }, { status: 503 })));
  },
  play: async ({ canvas }) => { await expect(await canvas.findByText('演示：指标版本未就绪')).toBeVisible(); },
};
