import type { Meta, StoryObj } from '@storybook/vue3-vite';
import { expect } from 'storybook/test';
import { http, HttpResponse } from 'msw';
import HomeView from './HomeView.vue';

const meta = { component: HomeView, tags: ['ai-generated'], parameters: { route: '/' } } satisfies Meta<typeof HomeView>;
export default meta;
type Story = StoryObj<typeof meta>;

export const Recommendation: Story = {
  play: async ({ canvas }) => {
    await expect((await canvas.findAllByText('城市漫步：发现日常中的小风景')).length).toBeGreaterThan(0);
    await expect((await canvas.findAllByText(/点赞 32/)).length).toBeGreaterThan(0);
    await expect(canvas.getAllByText(/收藏 8/).length).toBeGreaterThan(0);
    await expect(canvas.getAllByText(/评论 2/).length).toBeGreaterThan(0);
  },
};
export const SearchSuggestion: Story = {
  play: async ({ canvas, userEvent }) => {
    await canvas.findAllByText('城市漫步：发现日常中的小风景');
    await userEvent.type(canvas.getByRole('textbox', { name: '搜索内容' }), '城市');
    await userEvent.click(await canvas.findByRole('button', { name: /城市漫步路线/ }));
    await expect(canvas.getByRole('textbox', { name: '搜索内容' })).toHaveValue('城市漫步路线');
    await expect((await canvas.findAllByText('搜索')).length).toBeGreaterThan(0);
  },
};
export const FeedUnavailable: Story = {
  beforeEach: async context => {
    (context as any).msw.use(http.get('/api/feed', () => HttpResponse.json({ detail: '演示：内容服务暂不可用' }, { status: 503 })));
  },
  play: async ({ canvas }) => { await expect(await canvas.findByText('演示：内容服务暂不可用')).toBeVisible(); },
};
