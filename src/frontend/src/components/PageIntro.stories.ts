import type { Meta, StoryObj } from '@storybook/vue3-vite';
import { expect } from 'storybook/test';
import PageIntro from './PageIntro.vue';

const meta = {
  component: PageIntro,
  tags: ['ai-generated'],
  decorators: [() => ({ template: '<main class="container"><story /></main>' })],
  args: { eyebrow: 'EVALUATION', title: '可复现的离线评测', description: '统一数据划分与评测口径，明确每一项指标的含义。' },
} satisfies Meta<typeof PageIntro>;
export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};
export const LongDescription: Story = { args: { title: '搜索与推荐分开评估', description: '召回覆盖、候选保留、日志曝光池排序与服务最终输出分别核验，明确指标的候选范围、分母与局限。' } };
export const CssCheck: Story = {
  play: async ({ canvas }) => {
    await expect(getComputedStyle(canvas.getByText('EVALUATION')).color).toBe('rgb(36, 89, 211)');
  },
};
