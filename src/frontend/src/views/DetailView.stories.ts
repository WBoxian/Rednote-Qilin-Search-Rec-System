import type { Meta, StoryObj } from '@storybook/vue3-vite';
import { expect } from 'storybook/test';
import { http, HttpResponse } from 'msw';
import DetailView from './DetailView.vue';

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
