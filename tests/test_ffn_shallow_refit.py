import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts import train_ffn_shallow_refit as refit

search = refit.search


class RefitTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        x=np.random.default_rng(3).normal(size=(96,4)).astype(np.float32)
        y=(x[:,0]>0).astype(np.int32)
        self.train=dict(X_train=x[:80],y_train=y[:80])
        self.outer=dict(X_test=x[80:],y_test=y[80:])
        self.scaler=search.fit_scaler(self.train['X_train'])

    def test_median_uses_validation_epochs_only(self):
        records=[dict(seed=s,best_epoch=e,test_auc=a) for s,e,a in ((43,31,1),(44,22,0),(45,39,.5))]
        self.assertEqual(refit.fixed_epochs(records),31)
        records[0]['test_auc']=-999
        self.assertEqual(refit.fixed_epochs(records),31)
        for bad in (records[:2],records+[records[0]],[dict(r,best_epoch=0) for r in records]):
            with self.assertRaises(ValueError):
                refit.fixed_epochs(bad)

    def test_fixed_final_epoch_not_minimum_loss(self):
        counter=[0]
        def train(model,*args):
            counter[0]+=1
            with torch.no_grad():
                next(model.parameters()).fill_(counter[0])
            return float(counter[0])
        with tempfile.TemporaryDirectory() as directory,patch.object(search,'_train_epoch',side_effect=train):
            result=refit.run_head(search.config(()),43,self.train,self.scaler,Path(directory),'fixture','cpu',3)
            ckpt=torch.load(result['checkpoint'],weights_only=False)
            self.assertEqual(counter[0],3)
            self.assertTrue(torch.all(next(iter(ckpt['state_dict'].values()))==3))
            self.assertEqual(result['epochs_run'],3)
            self.assertEqual(len(json.loads((Path(result['checkpoint']).parent/'history.json').read_text())),3)
            with self.assertRaisesRegex(ValueError,'Only training data'):
                refit.run_head(search.config(()),43,{**self.train,**self.outer},self.scaler,Path(directory),'fixture','cpu',3)

    def test_cpu_and_cuda_reload_immutable_resume_and_test_freeze(self):
        devices=['cpu']+[f'cuda:{i}' for i in range(torch.cuda.device_count())]
        with tempfile.TemporaryDirectory() as directory:
            for device in devices:
                root=Path(directory)/device
                for cfg in (search.config(()),search.config((8,4)),search.config((8,),batch_norm=True),
                            search.config((8,),standardize=False)):
                    value=refit.run_head(cfg,43,self.train,self.scaler,root,'fixture',device,2)
                    self.assertEqual(value['recomputation_max_error'],dict(train=0.))
                    before={str(p):(p.stat().st_mtime_ns,search.sha256_file(p)) for p in root.rglob('*') if p.is_file()}
                    with patch.object(search,'_train_epoch',side_effect=AssertionError('retrained')):
                        refit.run_head(cfg,43,self.train,self.scaler,root,'fixture',device,2)
                    self.assertEqual(before,{str(p):(p.stat().st_mtime_ns,search.sha256_file(p)) for p in root.rglob('*') if p.is_file()})
                    for sig,epochs in (('wrong',2),('fixture',3)):
                        with self.assertRaisesRegex(ValueError,'fingerprint'):
                            refit.run_head(cfg,43,self.train,self.scaler,root,sig,device,epochs)
                with self.assertRaises(FileNotFoundError):
                    search.evaluate(value,self.train,self.outer,root,device)
                search.atomic_json_save(dict(status='FROZEN_BEFORE_OUTER_EVALUATION',final_head_ids=[value['head_id']]),root/'selection.json')
                result=search.evaluate(value,self.train,self.outer,root,device)
                self.assertEqual(result['recomputation_max_error'],0.)
                self.assertEqual(result['metrics'],search.evaluate(value,self.train,self.outer,root,device)['metrics'])

    def test_training_and_scaler_ignore_test_values(self):
        original=self.scaler['mean'].copy()
        self.outer['X_test']*=100000
        np.testing.assert_array_equal(original,search.fit_scaler(self.train['X_train'])['mean'])
        self.assertEqual(self.scaler['n_samples'],80)
        self.assertFalse(np.array_equal(original,search.fit_scaler(self.train['X_train'][:64])['mean']))

    def test_orchestration_freezes_before_fitting_and_trains_all_before_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'new';parent=Path(directory)/'old';parent.mkdir()
            groups={'fixed_arch0':dict(config=search.config(()),epochs=2,original_best_epochs=[1,2,3]),
                    'fixed_arch5':dict(config=search.config((8,4,2)),epochs=3,original_best_epochs=[3,3,4])}
            protocol=dict(groups=groups,primary_group='fixed_arch0',feature_dim=4,numerical_exception={'status':'FAIL retained'})
            search.atomic_json_save({'groups':{}},parent/'summary.json')
            calls=[]
            def prepare(model,device):
                search.study.immutable_json(protocol,root/'protocol.json')
                search.study.immutable_json(dict(status='FROZEN_BEFORE_OUTER_EVALUATION',
                    final_head_ids=[search.key(dict(config=g['config'],seed=s))[:20] for g in groups.values() for s in search.SEEDS]),root/'selection.json')
                return {**self.train,**self.outer},self.scaler,protocol
            fit=refit.run_head;evaluate=search.evaluate
            def train(*args):
                self.assertTrue((root/'selection.json').exists())
                calls.append('train')
                return fit(*args)
            def test(*args):
                self.assertEqual(calls.count('train'),6)
                calls.append('test')
                return evaluate(*args)
            with patch.object(refit,'output_root',return_value=root),patch.object(search,'output_root',return_value=parent), \
                 patch.object(refit,'prepare',side_effect=prepare),patch.object(refit,'run_head',side_effect=train), \
                 patch.object(search,'evaluate',side_effect=test):
                refit.run_model('fixture','cpu')
                self.assertEqual(calls,['train']*6+['test']*6)
                summary=json.loads((root/'summary.json').read_text())
                self.assertEqual(summary['trained_unique_heads'],6)
                before={str(p):(p.stat().st_mtime_ns,search.sha256_file(p)) for p in root.rglob('*') if p.is_file() and p.name!='.lock'}
                calls.clear()
                with patch.object(search,'_train_epoch',side_effect=AssertionError('retrained')):
                    refit.run_model('fixture','cpu')
                self.assertEqual(before,{str(p):(p.stat().st_mtime_ns,search.sha256_file(p)) for p in root.rglob('*') if p.is_file() and p.name!='.lock'})


if __name__=='__main__':
    unittest.main()
