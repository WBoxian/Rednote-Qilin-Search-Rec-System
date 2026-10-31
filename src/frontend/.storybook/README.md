# UI development

The app uses Vue 3 and Vite. Storybook and its test tools require Node 22.12 or later.

```sh
npm ci
npm run storybook -- --host 127.0.0.1
npm run typecheck
npm run typecheck-storybook
npx playwright install chromium --only-shell
npm run test-storybook
npm run build
npm run build-storybook
```

Stories use the app's CSS, Pinia, memory router, and MSW. Content and users are deterministic interface fixtures. Evaluation numbers are a frozen real-test snapshot in `fixtures/evaluation.json`; they do not claim that a production model is currently deployed.

Keep search and recommendation scopes separate. Full-catalog known-click metrics, conditional logged-exposure ranking metrics, and their baselines have different denominators. The app displays comparison and ablation values produced by the backend; it does not select models or calculate quality gains in the browser.

When the accepted metrics snapshot changes, refresh the evaluation fixture from the metrics producer and rerun the metric-page stories. Do not replace it with invented numbers or live API requests.
