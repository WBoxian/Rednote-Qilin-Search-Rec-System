import unittest
import tempfile,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from training.rank_stack import forward_blocks,group_minmax,MappedVectors,RequestContext
from training.dien_ranker import DIENRanker


class ForwardStackTests(unittest.TestCase):
    def test_forward_scores_never_use_current_or_future_requests(self):
        frame=pd.DataFrame({'request_idx':np.repeat(np.arange(12),2),'event_timestamp':np.repeat(np.arange(12)*10,2)+np.tile([0,1],12)})
        predicted=[]
        for fit,pred in forward_blocks(frame,'request_idx',4):
            self.assertLess(frame.iloc[fit].event_timestamp.max(),frame.iloc[pred].event_timestamp.min())
            self.assertFalse(set(frame.iloc[fit].request_idx)&set(frame.iloc[pred].request_idx))
            self.assertTrue((frame.iloc[pred].groupby('request_idx').size()==2).all())
            predicted.extend(pred)
        self.assertEqual(len(predicted),len(set(predicted)))
        self.assertEqual(len(predicted),18)

    def test_long_spanning_request_does_not_starve_training_prefix(self):
        frame=pd.DataFrame({'request_idx':np.repeat(np.arange(12),2),'event_timestamp':np.repeat(np.arange(12)*10,2)+np.tile([0,1],12)})
        frame.loc[frame.request_idx==4,'event_timestamp']=[0,41]
        fit,pred=next(forward_blocks(frame,'request_idx',4))
        self.assertEqual(len(fit),6)
        self.assertNotIn(4,set(frame.iloc[pred].request_idx))

    def test_missing_embedding_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            (folder/'test_map.json').write_text(json.dumps({'8':0,'2':1}))
            np.array([[1,2],[3,4]],np.float16).tofile(folder/'test.bin')
            store=MappedVectors(folder,'test',2)
            np.testing.assert_array_equal(store.get([2,8]),[[3,4],[1,2]])
            with self.assertRaises(ValueError):store.get([9],required=True)
            np.testing.assert_array_equal(store.get([9]),[[0,0]])

    def test_history_batch_preserves_raw_values_and_chronology(self):
        class Vectors:
            def get(self,ids,required=False):
                return np.repeat(np.asarray(ids,dtype=np.float32)[:,None],768,axis=1)
        requests=pd.DataFrame({'request_idx':[10,20],'recent_clicked_note_idxs':[[1,2],[]]})
        context=RequestContext(requests,'request_idx',Vectors(),Vectors(),seq_len=3)
        frame=pd.DataFrame({'request_idx':[10,20],'note_idx':[7,8],'x':[1.,2.]})
        batch=context.batch(frame,['x'],torch.device('cpu'))
        np.testing.assert_array_equal(batch['seq_embs'][0,:,0].numpy(),[2,1,0])
        self.assertEqual(float(batch['seq_embs'][1].abs().sum()),0.)
        np.testing.assert_array_equal(batch['note_text_emb'][:,0].numpy(),[7,8])

    def test_group_normalization_does_not_mix_requests(self):
        np.testing.assert_array_equal(group_minmax([100,101,-9,-7,5],[0,0,1,1,2]),[0,1,0,1,0])


class MaskedDIENTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.model=DIENRanker('rec',['x'],embed_dim=4,hidden_dim=3,mlp_dims=[8],dropout=0,mask_padded_history=True,auxiliary_gradients=True).eval()
        self.inputs={'note_text_emb':torch.randn(2,4),'note_img_emb':torch.randn(2,4),'x':torch.ones(2,1)}

    def test_padding_location_and_empty_history_do_not_change_score(self):
        seq=torch.tensor([[[1.,2,3,4],[3.,2,1,0],[0,0,0,0]],[[0.,0,0,0],[0,0,0,0],[0,0,0,0]]])
        self.inputs['seq_embs']=seq
        score,_=self.model(self.inputs)
        self.inputs['seq_embs']=seq[:,[2,0,1]]
        other,_=self.model(self.inputs)
        torch.testing.assert_close(score,other)
        self.assertTrue(torch.isfinite(score).all())

    def test_auxiliary_loss_reaches_interest_extractor(self):
        self.inputs['seq_embs']=torch.randn(2,3,4)
        _,loss=self.model(self.inputs)
        loss.backward()
        self.assertGreater(float(self.model.gru.weight_ih_l0.grad.abs().sum()),0)

    def test_all_click_requests_have_no_invented_auxiliary_negative(self):
        self.inputs['seq_embs']=torch.randn(2,3,4)
        _,loss=self.model(self.inputs,batch_y=torch.ones(2),batch_g=torch.tensor([1,2]))
        self.assertEqual(float(loss),0.)

    def test_legacy_aux_contract_keeps_identical_prediction(self):
        self.inputs['seq_embs']=torch.randn(2,3,4)
        new_score,new_loss=self.model(self.inputs,batch_y=torch.ones(2),batch_g=torch.tensor([1,2]))
        self.model.mask_missing_aux_negatives=False
        old_score,old_loss=self.model(self.inputs,batch_y=torch.ones(2),batch_g=torch.tensor([1,2]))
        torch.testing.assert_close(new_score,old_score,rtol=0,atol=0)
        self.assertEqual(float(new_loss),0.)
        self.assertGreater(float(old_loss),0.)

    def test_fp32_inference_contract_disables_tf32(self):
        model=DIENRanker('rec',['x'],embed_dim=4,hidden_dim=3,mlp_dims=[8],inference_precision='fp32')
        self.assertEqual(model.inference_precision,'fp32')
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        self.assertFalse(torch.backends.cudnn.allow_tf32)
        with self.assertRaises(ValueError):
            DIENRanker('rec',['x'],inference_precision='unknown')

    def test_checkpoint_persists_fit_normalization(self):
        self.model.set_zscore_stats({'x':{'mean':17,'std':3}})
        restored=DIENRanker('rec',['x'],embed_dim=4,hidden_dim=3,mlp_dims=[8],dropout=0,mask_padded_history=True,auxiliary_gradients=True).eval()
        restored.load_state_dict(self.model.state_dict())
        self.inputs['seq_embs']=torch.randn(2,3,4)
        torch.testing.assert_close(self.model(self.inputs)[0],restored(self.inputs)[0])

if __name__=='__main__':unittest.main()