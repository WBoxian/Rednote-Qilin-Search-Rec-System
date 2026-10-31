import unittest
from unittest.mock import Mock
import numpy as np
import torch
import pandas as pd
from backend.online.recall.dssm import DSSMRecaller

class OnlineQueryContractTests(unittest.TestCase):
    def test_current_query_takes_precedence_over_reference_cache(self):
        recaller=DSSMRecaller.__new__(DSSMRecaller);recaller.scene='search'
        stale=Mock();stale.get_vec.return_value=torch.full((768,),2.)
        recaller.user_store={'search_query':stale}
        expected=np.ones((1,768),np.float32)
        recaller._encode_query_vec=Mock(return_value=expected)
        result=recaller._get_request_query_vec(1,'new query',ref_request_id=99)
        self.assertIs(result,expected)
        recaller._encode_query_vec.assert_called_once_with('new query')
        stale.get_vec.assert_not_called()

    def test_encoder_failure_does_not_substitute_another_query(self):
        recaller=DSSMRecaller.__new__(DSSMRecaller);recaller.scene='search'
        recaller.user_store={'search_query':Mock()}
        recaller._encode_query_vec=Mock(return_value=None)
        self.assertIsNone(recaller._get_request_query_vec(1,'new query',99))
        recaller.user_store['search_query'].get_vec.assert_not_called()

    def test_live_request_rebuilds_tower_with_actual_query(self):
        recaller=DSSMRecaller.__new__(DSSMRecaller);recaller.scene='search'
        recaller.enabled=True;recaller.ann_index=object();recaller.row2note=[7]
        recaller._get_precomp_user_vec=Mock(return_value=np.full((1,128),9.))
        current=np.ones((1,128),np.float32)
        recaller._build_user_vec=Mock(return_value=current)
        recaller._search_ann=Mock(return_value=([],[]))
        request=pd.DataFrame({'search_idx':[1],'query':['reference query'],'user_idx':[2]})
        feature=pd.DataFrame({'search_idx':[1],'query':['actual query'],'user_idx':[2],'query_ref_request_id':[99]})
        recaller.fetch_candidates(1,500,request,lambda gid:feature,'search_idx')
        recaller._get_precomp_user_vec.assert_not_called()
        actual=recaller._build_user_vec.call_args.kwargs['feat_req']
        self.assertEqual(actual['query'].iloc[0],'actual query')
        self.assertEqual(feature['query'].iloc[0],'actual query')
        self.assertIs(recaller._search_ann.call_args.kwargs['user_vec'],current)

    def test_runtime_user_tower_receives_real_sequence_and_mask(self):
        recaller=DSSMRecaller.__new__(DSSMRecaller);recaller.scene='search'
        recaller.train_mod=Mock(EMB_DIM=768);recaller.model=Mock()
        recaller.model.forward_user.return_value=torch.ones(1,128)
        history=torch.ones(1,768);sequence=torch.ones(1,20,768);mask=torch.ones(1,20)
        recaller._build_history_summary_tensor=Mock(return_value=history)
        recaller._build_history_sequence_tensor=Mock(return_value=(sequence,mask))
        recaller._get_request_query_vec=Mock(return_value=np.ones((1,768),np.float32))
        frame=pd.DataFrame({'query':['actual'],'recent_clicked_note_idxs':[[1,2]]})
        result=recaller._build_user_vec(frame,1,2)
        self.assertEqual(result.shape,(1,128))
        kwargs=recaller.model.forward_user.call_args.kwargs
        self.assertIs(kwargs['seq_tokens'],sequence);self.assertIs(kwargs['seq_mask'],mask)
        recaller.model.forward_user.reset_mock();recaller._get_request_query_vec.return_value=None
        self.assertIsNone(recaller._build_user_vec(frame,1,2))
        recaller.model.forward_user.assert_not_called()

    def test_id_lookup_remains_available_without_raw_query(self):
        recaller=DSSMRecaller.__new__(DSSMRecaller);recaller.scene='search'
        store=Mock();store.get_vec.return_value=torch.ones(768)
        recaller.user_store={'search_query':store}
        result=recaller._get_request_query_vec(1,'',99)
        self.assertEqual(result.shape,(1,768));store.get_vec.assert_called_once_with(99)

if __name__=='__main__':unittest.main()