import type { Meta, StoryObj } from '@storybook/vue3-vite';
import { expect } from 'storybook/test';
import { http, HttpResponse } from 'msw';
import UserView from './UserView.vue';

const meta = { component: UserView, tags: ['ai-generated'], parameters: { route: '/user' } } satisfies Meta<typeof UserView>;
export default meta;
type Story = StoryObj<typeof meta>;

export const SelectBehaviors: Story = {
  play: async ({ canvas, userEvent }) => {
    await canvas.findByText('城市漫步：发现日常中的小风景');
    await userEvent.click(canvas.getByRole('button', { name: '全选' }));
    for (const input of canvas.getAllByRole('checkbox')) await expect(input).toBeChecked();
    await expect(canvas.getByRole('button', { name: '批量删除' })).toBeEnabled();
    await userEvent.click(canvas.getByRole('button', { name: '取消勾选' }));
    await expect(canvas.getByRole('button', { name: '批量删除' })).toBeDisabled();
  },
};
export const NoHistory: Story = {
  beforeEach: async context => {
    (context as any).msw.use(http.get('/api/user', () => HttpResponse.json({ user: { features: {}, recent_behaviors: [] } })));
  },
};
