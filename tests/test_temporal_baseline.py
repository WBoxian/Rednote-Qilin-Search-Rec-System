import unittest
import numpy as np
import pandas as pd
from scripts.train_temporal_baseline import temporal_training_indices


class TemporalBaselineTests(unittest.TestCase):
    def test_official_test_boundary_excludes_whole_requests(self):
        train=pd.DataFrame({"request_idx":[1,1,2,2,3,3,4,4,5,5,6,6],
                            "event_timestamp":[1,2,3,4,2,10,11,12,13,20,20,21]})
        test=pd.DataFrame({"request_idx":[8,8],"event_timestamp":[15,16]})
        historical,fit,val,boundary=temporal_training_indices(train,test,"request_idx",.5)
        self.assertEqual(set(historical.request_idx),{1,2,3,4})
        self.assertEqual(set(historical.iloc[fit].request_idx),{1,2})
        self.assertEqual(set(historical.iloc[val].request_idx),{4})
        self.assertLess(historical.iloc[fit].event_timestamp.max(),historical.iloc[val].event_timestamp.min())
        self.assertLess(historical.iloc[val].event_timestamp.max(),boundary)
        self.assertEqual(len(test),2)

    def test_nonfinite_test_times_fail_closed(self):
        train=pd.DataFrame({"request_idx":[1,2],"event_timestamp":[1,2]})
        test=pd.DataFrame({"request_idx":[3,4],"event_timestamp":[3,np.nan]})
        with self.assertRaises(ValueError):
            temporal_training_indices(train,test,"request_idx")
