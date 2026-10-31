import tempfile,unittest
from pathlib import Path
import torch
from recall.dssm_trainer import DSSMModel,numeric_feature_columns,save_tower_checkpoints

class DSSMPaddingContractTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42);torch.set_num_threads(2)
        self.vocab={key:3 for key in ('user_idx','note_idx','gender','platform','age','location','tax1','tax2','tax3')}
        self.item_dim=len(numeric_feature_columns('rec')[1])
        self.user_dim=42+768+128
        self.batch={key:torch.zeros(2,dtype=torch.long) for key in ('user_idx','gender_enc','platform_enc','age_enc','location_enc')}
        self.batch.update(fans_num=torch.zeros(2),follows_num=torch.zeros(2),dense_feats=torch.zeros(2,40))
        self.history=torch.zeros(2,768)
    def model(self,packed):
        return DSSMModel('rec',self.vocab,self.item_dim,self.user_dim,mask_sequence_padding=packed).eval()
    def test_masked_tokens_cannot_change_user_vector(self):
        model=self.model(True);tokens=torch.randn(2,20,768);mask=torch.zeros(2,20);mask[0,:3]=1;mask[1,:7]=1
        altered=tokens.clone();altered[mask==0]=100.
        with torch.no_grad():
            before=model.forward_user(self.batch,self.history,tokens,mask)
            after=model.forward_user(self.batch,self.history,altered,mask)
        torch.testing.assert_close(before,after,rtol=0,atol=0)
    def test_empty_sequence_branch_is_zero_and_finite(self):
        model=self.model(True);captured={}
        hook=model.user_mlp.register_forward_pre_hook(lambda module,args:captured.update(value=args[0].detach()))
        with torch.no_grad():result=model.forward_user(self.batch,self.history,torch.randn(2,20,768),torch.zeros(2,20))
        hook.remove();self.assertTrue(torch.isfinite(result).all());self.assertEqual(int(torch.count_nonzero(captured['value'][:,-128:])),0)
    def test_full_and_partial_checkpoint_preserve_contract_old_weights_stay_legacy(self):
        model=self.model(True);state=model.state_dict();restored=self.model(False);restored.load_state_dict(state)
        self.assertTrue(bool(restored.seq_mask_padding))
        old={key:value for key,value in state.items() if key!='seq_mask_padding'}
        restored.load_state_dict(old);self.assertFalse(bool(restored.seq_mask_padding))
        with tempfile.TemporaryDirectory() as folder:
            save_tower_checkpoints(state,'rec','fixture',self.vocab,self.item_dim,self.user_dim,output_dir=Path(folder))
            user=torch.load(Path(folder)/'dssm_rec_fixture_user_tower.pt',map_location='cpu',weights_only=True)
            self.assertTrue(bool(user['state_dict']['seq_mask_padding']))
            self.assertEqual(user['meta']['sequence_padding_contract'],'packed_masked_v1')

    def test_persistent_profile_mask_guards_every_inference_caller(self):
        model=DSSMModel('rec',self.vocab,self.item_dim,self.user_dim,mask_user_profile=True).eval()
        other={key:value.clone() for key,value in self.batch.items()}
        other['dense_feats'].fill_(500.);other['fans_num'].fill_(100.);other['follows_num'].fill_(200.)
        with torch.no_grad():
            before=model.forward_user(self.batch,self.history)
            after=model.forward_user(other,self.history)
        torch.testing.assert_close(before,after,rtol=0,atol=0)
        loaded=self.model(False);loaded.load_state_dict(model.state_dict())
        self.assertTrue(bool(loaded.user_mask_profile))
if __name__=='__main__':unittest.main()