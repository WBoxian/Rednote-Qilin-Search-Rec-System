import unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch
import pandas as pd
from backend.online.pipeline import OnlineScenePipeline

class CardMetadataRegressionTests(unittest.TestCase):
    def test_recall_only_cards_recover_display_counts_without_changing_scores(self):
        import pandas as pd
        from types import SimpleNamespace
        from backend.online.pipeline import OnlineScenePipeline
        pipeline = object.__new__(OnlineScenePipeline)
        pipeline.state = SimpleNamespace(scene="rec", _fetch_notes=lambda ids: pd.DataFrame({"note_idx":[9], "accum_like_num":[32],"accum_collect_num":[8],"accum_comment_num":[2]}))
        frame = pd.DataFrame({"note_idx":[9],"recall_score":[.25],"final_score":[.25]})
        cards = pipeline._build_cards(1,7,frame)
        self.assertEqual([cards[0][k] for k in ("accum_like_num","accum_collect_num","accum_comment_num")],[32,8,2])
        self.assertEqual(cards[0]["scores"]["final"],.25)
        self.assertNotIn("accum_like_num",frame)


class LatencyTests(unittest.TestCase):
    def test_cache_hit_reports_current_elapsed_time_not_historical_inference(self):
        pipeline=OnlineScenePipeline.__new__(OnlineScenePipeline)
        pipeline.state=SimpleNamespace(scene="rec",group_key="request_idx",realtime_cache=None,
            build_live_context=lambda *a:pd.DataFrame({"request_idx":[1],"user_idx":[1],
                "note_idx":[-1],"query":[""],"recent_clicked_note_idxs":[[]]}))
        frame=pd.DataFrame({"note_idx":[1]})
        pipeline._get_cached_feed=Mock(return_value=(frame,{"stages":{},"linkage":{},
            "stage_ms":{"recall":500,"ranking":500},"pipeline_ms":1000}))
        pipeline._select_page_rows=Mock(return_value=frame)
        pipeline._build_cards=Mock(return_value=[])
        with patch("backend.online.pipeline.time.perf_counter",side_effect=[10.,10.001]):
            response=pipeline.build_feed(1,"",1,10)
        self.assertAlmostEqual(response["latency_ms"],1.,places=5)
        self.assertEqual(response["pipeline_ms"],0)
        self.assertEqual(response["stage_ms"],{})
        self.assertEqual(response["cached_pipeline_ms"],1000)
        self.assertTrue(response["cache_hit"])

if __name__=="__main__": unittest.main()
