import type { Meta, StoryObj } from '@storybook/vue3-vite';
import { expect } from 'storybook/test';
import { http, HttpResponse } from 'msw';
import LoginView from './LoginView.vue';

const meta = { component: LoginView, tags: ['ai-generated'], parameters: { route: '/login' } } satisfies Meta<typeof LoginView>;
export default meta;
type Story = StoryObj<typeof meta>;

export const ChooseUser: Story = {
  play: async ({ canvas, userEvent }) => {
    await userEvent.click(await canvas.findByText('15128'));
    await expect(canvas.getByRole('textbox', { name: '用户 ID' })).toHaveValue('15128');
  },
};
export const InvalidUser: Story = {
  play: async ({ canvas, userEvent }) => {
    await userEvent.type(canvas.getByRole('textbox', { name: '用户 ID' }), '-1');
    await userEvent.click(canvas.getByRole('button', { name: '登录' }));
    await expect(await canvas.findByText('user_idx 需为非负整数')).toBeVisible();
  },
};
export const ServiceUnavailable: Story = {
  beforeEach: async context => {
    (context as any).msw.use(http.get('/api/login', () => HttpResponse.json({ detail: '演示：服务尚未就绪' }, { status: 503 })));
  },
  play: async ({ canvas, userEvent }) => {
    await userEvent.type(canvas.getByRole('textbox', { name: '用户 ID' }), '15128');
    await userEvent.click(canvas.getByRole('button', { name: '登录' }));
    await expect(await canvas.findByText('演示：服务尚未就绪')).toBeVisible();
  },
};
