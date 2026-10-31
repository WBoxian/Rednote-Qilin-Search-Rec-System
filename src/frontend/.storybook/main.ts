import type { StorybookConfig } from '@storybook/vue3-vite';

const config: StorybookConfig = {
  staticDirs: ['../public'],
  "stories": [
    "../src/**/*.mdx",
    "../src/**/*.stories.@(js|jsx|mjs|ts|tsx)"
  ],
  "addons": [
    "@storybook/addon-a11y",
    "@storybook/addon-docs",
    "@storybook/addon-mcp",
    "@storybook/addon-vitest",
    "msw-storybook-addon"
  ],
  "framework": "@storybook/vue3-vite"
};
export default config;