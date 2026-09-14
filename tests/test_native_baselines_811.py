import unittest
from unittest.mock import patch
import numpy as np
import torch
from detection.baselines import torch_hallucination_scores, SVARMLP
from scripts import train_native_baselines_811 as study


class Native811Tests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(7)
        x=rng.normal(size=(90,4)).astype(np.float32)
        x[60:]+=5
        y=np.tile([0,1],45)
        x[:,0]+=3*y
        self.data=dict(groups={'svar':x,'metatoken':x},y=y,
            masks={k:np.isin(np.arange(90),idx) for k,idx in
                   [('train',range(60)),('validation',range(60,76)),('test',range(76,90))]})
        torch.set_num_threads(1)

    def test_metatoken_train_only_scaler_and_saved_real_probability(self):
        for head in ('metatoken_lr','metatoken_gb'):
            result,model=study.train_head(self.data,head,43,'cpu')
            x=self.data['groups']['metatoken'];m=self.data['masks']
            np.testing.assert_allclose(model['standardize'].mean_,x[m['train']].mean(0,dtype=np.float64))
            self.assertFalse(np.allclose(model['standardize'].mean_,x.mean(0)))
            real_index=list(model.classes_).index(0)
            np.testing.assert_allclose(result['test_probabilities'],model.predict_proba(x[m['test']])[:,real_index])

    def test_svar_callback_checkpoint_and_real_probability(self):
        calls=[]
        with patch.dict(study.SVAR,epochs=4):
            result,model=study.train_head(self.data,'svar_native',43,'cpu',calls.append)
        self.assertEqual(calls,list(range(1,len(result['history'])+1)))
        best=min(result['history'],key=lambda r:r['val_loss'])
        self.assertEqual(result['best_epoch'],int(best['epoch']))
        restored=SVARMLP(4,248);restored.load_state_dict(result['state_dict']);restored.eval()
        x=self.data['groups']['svar'][self.data['masks']['test']]
        np.testing.assert_allclose(result['test_probabilities'],1-torch_hallucination_scores(restored,x,torch.device('cpu')))


if __name__=='__main__':unittest.main()
