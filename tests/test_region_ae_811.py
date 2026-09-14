import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts.train_region_ae_811 import split_images
from scripts import train_torch_probe_feature_sets as trainer


class SplitAndCheckpointTests(unittest.TestCase):
    def test_image_split_preserves_train_and_is_disjoint(self):
        outer=dict(train=list(range(3200)),test=list(range(3200,4000)))
        split=split_images(outer)
        self.assertEqual(split,split_images(outer))
        self.assertEqual(set(split['train']),set(outer['train']))
        self.assertEqual(set(split['validation'])|set(split['test']),set(outer['test']))
        self.assertFalse(set(split['validation'])&set(split['test']))
        self.assertEqual([len(v) for v in split.values()],[3200,400,400])

    def run_checkpoint(self, validation):
        # Train loss improves while validation loss worsens: the two policies
        # must select different epochs. Test predictions occur after selection.
        x=np.ones((4,2),np.float32);y=np.array([0,1,0,1])
        pair=(.2,np.array([.2,.8]))
        predictions=([(.3,pair[1]),(.5,pair[1]),(.8,pair[1])] if validation else [])
        predictions += [(.2,np.array([.2,.8,.2,.8])),pair]
        if validation: predictions += [(.3,pair[1])]
        cfg=trainer.TorchProbeConfig(hidden_sizes=(4,),num_epochs=3,batch_size=4,
            checkpoint_selection='minimum_val_loss' if validation else 'minimum_train_loss')
        with tempfile.TemporaryDirectory() as folder, patch.object(trainer,'_train_epoch',side_effect=[.6,.4,.2]), \
                patch.object(trainer,'_predict_loss_and_probs',side_effect=predictions) as predictor:
            result=trainer.train_and_evaluate_probe(X_train=x,y_train=y,X_val=x[:2],y_val=y[:2],
                X_test=x[:2],y_test=y[:2],config=cfg,device=torch.device('cpu'),output_dir=folder,
                return_probabilities=True)
            self.assertEqual(predictor.call_count,6 if validation else 2)
        return result

    def test_validation_selects_epoch_one(self):
        result=self.run_checkpoint(True)
        self.assertEqual(result['best_epoch'],1)
        self.assertEqual(result['best_val_loss'],.3)
        self.assertEqual(result['best_train_loss'],.6)
        self.assertIn('validation_probabilities',result)

    def test_legacy_policy_still_selects_train_minimum(self):
        result=self.run_checkpoint(False)
        self.assertEqual(result['best_epoch'],3)
        self.assertEqual(result['best_train_loss'],.2)
        self.assertIsNone(result['best_val_loss'])
        self.assertNotIn('validation_probabilities',result)


if __name__=='__main__': unittest.main()
