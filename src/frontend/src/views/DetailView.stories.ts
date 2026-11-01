import type { Meta, StoryObj } from '@storybook/vue3-vite';
import { expect, within } from 'storybook/test';
import { delay, http, HttpResponse } from 'msw';
import DetailView from './DetailView.vue';
import HomeView from './HomeView.vue';
import { demoNote } from '../../.storybook/msw-handlers';

function seedStages(noteIdx: number, enabled: boolean, title = demoNote.title) {
  sessionStorage.setItem('qilin_detail_seed', JSON.stringify({
    scene: 'rec', user_idx: 15128, request_id: 4201, note_idx: noteIdx,
    savedAt: Date.now(), detail: { ...demoNote, title, note_idx: noteIdx,
      stage_enabled: { preranking: enabled, ranking: enabled } },
  }));
}

function mockNote(context: any, noteIdx: number, wait = 0) {
  context.msw.use(http.get('/api/note', async () => {
    if (wait) await delay(wait);
    return HttpResponse.json({ detail: { ...demoNote, note_idx: noteIdx }, latency_ms: 3 });
  }));
}

async function expectStage(canvas: any, label: string, value: string) {
  const row = (await canvas.findByText(label)).closest('tr')!;
  await expect(within(row).getAllByRole('cell')[1]).toHaveTextContent(value);
}

const meta = {
  component: DetailView, tags: ['ai-generated'],
  parameters: { route: '/detail?scene=rec&user_idx=15128&request_id=4201&note_idx=101' },
} satisfies Meta<typeof DetailView>;
export default meta;
type Story = StoryObj<typeof meta>;

export const LikeContent: Story = {
  play: async ({ canvas, userEvent }) => {
    const button = await canvas.findByRole('button', { name: /点赞.*32/ });
    await userEvent.click(button);
    await expect(button).toHaveAttribute('aria-pressed', 'true');
    await expect(button).toHaveTextContent('33');
  },
};
export const ContentUnavailable: Story = {
  parameters: { route: "/detail?scene=rec&user_idx=15128&request_id=4201&note_idx=404" },
  beforeEach: async context => {
    (context as any).msw.use(http.get('/api/note', () => HttpResponse.json({ detail: '演示：内容不存在' }, { status: 404 })));
  },
  play: async ({ canvas }) => { await expect(await canvas.findByText('演示：内容不存在')).toBeVisible(); },
};

export const DisabledRankingStages: Story = {
  parameters: { route: '/detail?scene=rec&user_idx=15128&request_id=4201&note_idx=201' },
  beforeEach: async context => { seedStages(201, false); mockNote(context, 201); },
  play: async ({ canvas }) => {
    await expectStage(canvas, '粗排 Rank', '未启用');
    await expectStage(canvas, '精排 Rank', '未启用');
    await expectStage(canvas, '召回 Rank', '12');
  },
};

export const EnabledRankingStages: Story = {
  parameters: { route: '/detail?scene=rec&user_idx=15128&request_id=4201&note_idx=202' },
  beforeEach: async context => { seedStages(202, true); mockNote(context, 202); },
  play: async ({ canvas }) => {
    await expectStage(canvas, '粗排 Rank', '8');
    await expectStage(canvas, '精排 Rank', '3');
    await expect(canvas.queryByText('未启用')).not.toBeInTheDocument();
    await expect(canvas.queryByText(/状态未确认/)).not.toBeInTheDocument();
  },
};

export const DirectLinkUnknownStages: Story = {
  parameters: { route: '/detail?scene=rec&user_idx=15128&request_id=4201&note_idx=203' },
  beforeEach: async context => {
    seedStages(999, false); // A different note cannot supply this request's stage state.
    mockNote(context, 203);
  },
  play: async ({ canvas }) => {
    await expectStage(canvas, '粗排 Rank', '8（状态未确认）');
    await expectStage(canvas, '精排 Rank', '3（状态未确认）');
    await expect(canvas.queryByText('未启用')).not.toBeInTheDocument();
  },
};

export const RefreshedDetailPreservesRequestState: Story = {
  parameters: { route: '/detail?scene=rec&user_idx=15128&request_id=4201&note_idx=204' },
  beforeEach: async context => {
    seedStages(204, false, '首页缓存内容');
    mockNote(context, 204, 250);
  },
  play: async ({ canvas }) => {
    await canvas.findAllByText(demoNote.title);
    await expectStage(canvas, '粗排 Rank', '未启用');
    await expectStage(canvas, '精排 Rank', '未启用');
  },
};

export const RefreshedFeedSuppliesStageState: Story = {
  parameters: { route: '/' },
  render: () => ({ components: { HomeView }, template: '<HomeView />' }),
  beforeEach: async context => {
    let calls = 0;
    (context as any).msw.use(http.get('/api/feed', () => {
      const enabled = calls++ === 0;
      return HttpResponse.json({ feed: {
        items: [{ ...demoNote, user_idx: 15128, title: enabled ? '刷新前内容' : '刷新后内容',
          stage_ranks: { recall: 12, preranking: 8, ranking: 3, rerank: 3 } }],
        stages: { preranking: { enabled }, ranking: { enabled } },
      } });
    }));
  },
  play: async ({ canvas, userEvent }) => {
    await canvas.findAllByText('刷新前内容');
    await userEvent.click(canvas.getByRole('button', { name: '换一批' }));
    await userEvent.click((await canvas.findAllByText('刷新后内容'))[0]);
    const seed = JSON.parse(sessionStorage.getItem('qilin_detail_seed')!);
    await expect(seed.detail.stage_enabled).toEqual({ preranking: false, ranking: false });
    await expect(seed.detail.stage_top500_ranks.ranking).toBe(3);
  },
};
