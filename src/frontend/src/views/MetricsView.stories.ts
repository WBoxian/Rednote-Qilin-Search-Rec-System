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
    await expect(await canvas.findByRole('heading', { name: '搜索全库链路' })).toBeVisible();
    await expect(canvas.getByText('55.23%')).toBeVisible();
    await expect(canvas.getByText('63.71%')).toBeVisible();
    await expect(canvas.getByText('0.5468')).toBeVisible();
    await expect(canvas.queryByText('最终结果与基线提升')).not.toBeInTheDocument();
    await expect(canvas.queryByText('完整测试请求与曝光，统一指标口径')).not.toBeInTheDocument();
    await expect(canvas.queryByText('阶段消融与实验结果')).not.toBeInTheDocument();
    await expect(canvas.queryByText('排序指标如何理解')).not.toBeInTheDocument();
    await expect(await canvas.findByText('当前用户')).toBeVisible();
    await expect(canvas.getByText(/请求编号=9/)).toBeVisible();
    const metric = await canvas.findByRole('button', { name: /命中率（HitRate@10）/ });
    await userEvent.click(metric);
    await expect(await canvas.findByRole('heading', { name: /全库链路.*HitRate@10/ })).toBeVisible();
    await userEvent.click(canvas.getByRole('button', { name: '关闭' }));
    await userEvent.click(canvas.getByRole('button', { name: '推荐' }));
    await expect(await canvas.findByRole('heading', { name: 'DSSM召回优化' })).toBeVisible();
    await expect(canvas.getByText('+3.94%')).toBeVisible();
    await expect(canvas.getByText('0.6195')).toBeVisible();
    await userEvent.click(canvas.getByRole('button', { name: '搜索' }));
    await expect(await canvas.findByText('55.23%')).toBeVisible();
  },
};
export const RecommendationMetrics: Story = {
  parameters: { scene: 'rec' },
  play: async ({ canvas, userEvent }) => {
    await expect(await canvas.findByRole('heading', { name: 'DSSM召回优化' })).toBeVisible();
    await expect(canvas.getByText('+3.94%')).toBeVisible();
    await expect(canvas.getByText('+4.19%')).toBeVisible();
    await expect(canvas.getByText(/内部验证.*11,893.*DSSM 单路.*未上线/)).toBeVisible();
    await expect(canvas.getByText('0.6195')).toBeVisible();
    await expect(canvas.getByText('0.6143')).toBeVisible();
    await expect(canvas.getByText('0.6211')).toBeVisible();
    await expect(canvas.queryByText('0.0661')).not.toBeInTheDocument();
    await expect(canvas.queryByText('阶段消融与实验结果')).not.toBeInTheDocument();
    await expect(canvas.queryByText('粗排 · GBDT')).not.toBeInTheDocument();
    await userEvent.click(canvas.getByRole('button', { name: /HitRate@50.*相对提升/ }));
    await expect(await canvas.findByRole('heading', { name: /DSSM召回.*内部验证.*HitRate@50/ })).toBeVisible();
    await expect(canvas.getByText(/控制组 0.057681；新模型 0.059951/)).toBeVisible();
    await userEvent.click(canvas.getByRole('button', { name: '关闭' }));
  },
};
export const MetricsUnavailable: Story = {
  beforeEach: async context => {
    (context as any).msw.use(http.get('/api/metrics', () => HttpResponse.json({ detail: '演示：指标版本未就绪' }, { status: 503 })));
  },
  play: async ({ canvas }) => { await expect(await canvas.findByText('演示：指标版本未就绪')).toBeVisible(); },
};

export const CurrentUserTrainingReplay: Story = {
  beforeEach: async context => {
    const uid = Number(localStorage.getItem('qilin_user_idx'));
    const note = { note_idx: 101, title: '训练样例目标', rank_in_true: 1 };
    const example = { scope: 'current_user_request_replay', split: 'train', evaluation_included: false, scene: 'search', request_id: 99, source_user_idx: uid, result_count: 12, query: '测试查询', true_top10: [note, { note_idx: 102, title: '第二个点击目标', rank_in_true: 2 }], dssm_top10: [{ note_idx: 102, title: '第二个点击目标', rank_in_true: 2 }, { note_idx: 999, title: '未命中目标', rank_in_true: null }, note], preranking_top10: [note], ranking_top10: [note], source_recent_behaviors: [], overlap: { dssm: 1, preranking: 1, ranking: 1 }, sample_note: '当前用户训练请求回放，不能作为测试成绩', notes: [{ ...note, content: '此详情来自当前回放的笔记元数据', accum_like_num: 32, accum_collect_num: 8, accum_comment_num: 2 }] };
    (context as any).msw.use(http.get('/api/validation', () => HttpResponse.json({ validation: { examples: [example] } })));
  },
  play: async ({ canvas, canvasElement, userEvent }) => {
    await expect(await canvas.findByText('当前用户训练请求回放，不能作为测试成绩')).toBeVisible();
    await expect(canvas.getByText('当前用户')).toBeVisible();
    const recallRanks = Array.from(canvasElement.querySelectorAll('.compare-cell:nth-child(2) tbody tr')).map(row => row.querySelector('td')?.textContent?.trim());
    await expect(recallRanks).toEqual(['2', '(-)', '1']);
    await userEvent.click(canvas.getAllByText('训练样例目标')[0]);
    await expect(await canvas.findByText('此详情来自当前回放的笔记元数据')).toBeVisible();
    await expect(canvas.getByText('点赞 32')).toBeVisible();
    await userEvent.click(canvas.getByRole('button', { name: '关闭' }));
  },
};

export const UserWithoutRequests: Story = {
  beforeEach: async context => {
    const example = { scope: 'frozen_full_test_examples', scene: 'search', request_id: 1, source_user_idx: 8, result_count: 2, true_top10: [], dssm_top10: [], preranking_top10: [], ranking_top10: [], source_recent_behaviors: [] };
    (context as any).msw.use(http.get('/api/validation', () => HttpResponse.json({ validation: { examples: [example], current_user_example_status: '当前用户没有可回放的原始请求，以下展示其他用户的固定测试样例' } })));
  },
  play: async ({ canvas }) => {
    await expect(await canvas.findByText('当前用户没有可回放的原始请求，以下展示其他用户的固定测试样例')).toBeVisible();
    await expect(canvas.getByText('其他用户')).toBeVisible();
    await expect(canvas.queryByText('当前用户', { exact: true })).not.toBeInTheDocument();
  },
};
