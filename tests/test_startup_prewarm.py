import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from backend.online.api.main import AppContext

class StartupPrewarmTests(unittest.TestCase):
    def context(self):
        values = {'QILIN_ASYNC_PREWARM': '1', 'QILIN_BLOCKING_PREWARM': '0', 'QILIN_PREWARM_FEED': '1', 'QILIN_ASYNC_PREWARM_FEED_SCENES': 'rec,search'}
        with patch.dict(os.environ, values), patch('backend.online.api.main.OnlineRuntimeRegistry'), patch('backend.online.api.main.threading.Thread') as thread:
            ctx = AppContext('hard', 100, 500)
            thread.return_value.start.assert_called_once()
            return ctx

    def test_warms_recommendation_then_search_with_observable_timings(self):
        ctx = self.context()
        calls = []
        pipes = {s: SimpleNamespace(prewarm_homepage_feed=Mock(side_effect=lambda page_size, s=s: (calls.append(s) or {'degraded_reason': None, 'route_status': {}}))) for s in ('rec', 'search')}
        ctx.registry.get_runtime.return_value.get_pipeline.side_effect = lambda s: pipes[s]
        self.assertEqual(ctx._async_feed_scenes, ['rec', 'search'])
        self.assertEqual(ctx.prewarm_readiness()['rec']['stage'], 'pending')
        ctx.registry.get_runtime.assert_not_called()
        ctx._prewarm_homepage_runtime(blocking=False)
        self.assertEqual(calls, ['rec', 'search'])
        status = ctx.prewarm_readiness()
        for s in ('rec', 'search'):
            self.assertEqual(status[s]['stage'], 'ready')
            for k in ('pipeline_load_ms', 'feed_warm_ms', 'total_ms'):
                self.assertGreaterEqual(status[s][k], 0)
        status['rec']['stage'] = 'changed'
        self.assertEqual(ctx.prewarm_readiness()['rec']['stage'], 'ready')

    def test_failure_is_visible_and_other_scene_still_warms(self):
        ctx = self.context()
        search = SimpleNamespace(prewarm_homepage_feed=Mock(return_value={'degraded_reason': None, 'route_status': {}}))
        def load(s):
            if s == 'rec':
                raise ValueError('fixture unavailable')
            return search
        ctx.registry.get_runtime.return_value.get_pipeline.side_effect = load
        ctx._prewarm_homepage_runtime(blocking=False)
        status = ctx.prewarm_readiness()
        self.assertEqual(status['rec']['stage'], 'failed')
        self.assertIn('fixture unavailable', status['rec']['error'])
        self.assertEqual(status['search']['stage'], 'ready')
        search.prewarm_homepage_feed.assert_called_once()

    def test_disabled_feed_does_not_load_models(self):
        ctx = self.context()
        ctx._prewarm_feed_enabled = False
        ctx._prewarm_homepage_runtime(blocking=False)
        ctx.registry.get_runtime.assert_not_called()

    def test_returned_route_degradation_is_visible(self):
        ctx = self.context()
        pipe = SimpleNamespace(prewarm_homepage_feed=Mock(return_value={
            'degraded_reason': 'recall_route_failure', 'route_status': {'errors': ['ANN failed']}}))
        ctx.registry.get_runtime.return_value.get_pipeline.return_value = pipe
        ctx._prewarm_homepage_runtime(blocking=False)
        status = ctx.prewarm_readiness()
        self.assertEqual(status['rec']['stage'], 'degraded')
        self.assertEqual(status['rec']['route_errors'], ['ANN failed'])

    def test_concurrent_registry_calls_construct_one_runtime(self):
        from concurrent.futures import ThreadPoolExecutor
        import threading
        import time
        from backend.online.pipeline import OnlineRuntimeRegistry
        registry = OnlineRuntimeRegistry('hard', 100, 500)
        barrier = threading.Barrier(8)
        def construct(**kwargs):
            time.sleep(.02)
            return object()
        def get(_):
            barrier.wait(timeout=5)
            return registry.get_runtime('hard')
        with patch('backend.online.pipeline.OnlineRuntime', side_effect=construct) as factory:
            with ThreadPoolExecutor(max_workers=8) as executor:
                objects = list(executor.map(get, range(8)))
            self.assertEqual(factory.call_count, 1)
            self.assertTrue(all(value is objects[0] for value in objects))

    def test_ready_scene_does_not_wait_for_another_scene_loading(self):
        import threading
        from backend.online.pipeline import ServingAppState
        app = ServingAppState('hard', 100, 500)
        ready = object()
        app._states['rec'] = ready
        result = []
        done = threading.Event()
        def get():
            result.append(app.get('rec'))
            done.set()
        with app._lock:
            thread = threading.Thread(target=get)
            thread.start()
            fast = done.wait(timeout=1)
        thread.join(timeout=2)
        self.assertTrue(fast)
        self.assertEqual(result, [ready])
