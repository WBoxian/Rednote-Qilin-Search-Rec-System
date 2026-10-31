import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi import HTTPException
from backend.online.api.main import create_app
from backend.online.bundle import bundle_identity
import backend.online.metrics_snapshot as metrics


class MetricsBundleVersionTests(unittest.TestCase):
    def test_active_identity_must_match_metrics_but_absent_or_legacy_pointer_stays_available(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root/'metrics.json'
            manifest = {'scene': 'search', 'tag': 'hard', 'version': 1}
            manifest['bundle_id'] = bundle_identity(manifest)
            snapshot.write_text(json.dumps({'tag': 'hard', 'protocol': {}, 'scenes': {'search': {'model_bundle_id': manifest['bundle_id']}}}))
            active = root/'outputs/deploy/search/hard/active_bundle.json'
            with patch.object(metrics, 'ROOT', root), patch.object(metrics, 'SNAPSHOT_PATH', snapshot):
                metrics.load_metrics_snapshot('search', 'hard')
                active.parent.mkdir(parents=True);active.write_text(json.dumps(manifest))
                metrics.load_metrics_snapshot('search', 'hard')
                changed = {**manifest, 'version': 2};changed['bundle_id'] = bundle_identity(changed)
                active.write_text(json.dumps(changed))
                with self.assertRaises(metrics.MetricsVersionMismatch):
                    metrics.load_metrics_snapshot('search', 'hard')
                snapshot.write_text(json.dumps({'tag': 'hard', 'protocol': {}, 'scenes': {'search': {}}}))
                metrics.load_metrics_snapshot('search', 'hard')

    def test_actual_metrics_route_returns_503_for_version_mismatch(self):
        with patch.dict(os.environ, {'QILIN_ASYNC_PREWARM': '0'}):
            app = create_app()
        endpoint = next(route.endpoint for route in app.routes if getattr(route, 'path', '') == '/api/metrics')
        with patch('backend.online.api.main.load_metrics_snapshot', side_effect=metrics.MetricsVersionMismatch('mismatch')):
            with self.assertRaises(HTTPException) as error:
                endpoint(scene='search', tag='hard', sample_n=1000, include_val=True)
        self.assertEqual(error.exception.status_code, 503)


if __name__ == '__main__':
    unittest.main()
