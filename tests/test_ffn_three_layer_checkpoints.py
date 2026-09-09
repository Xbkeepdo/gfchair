import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts import train_ffn_three_layer_checkpoints as study

s=study.search


class CheckpointComparisonTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        x=np.random.default_rng(7).normal(size=(96,5)).astype(np.float32)
        y=(x[:,0]>0).astype(np.int32)
        self.train=dict(X_train=x[:64],y_train=y[:64])
        self.val=dict(X_val=x[64:80],y_val=y[64:80])
        self.outer=dict(X_test=x[80:],y_test=y[80:])
        self.scaler=s.fit_scaler(self.train['X_train'])

    def test_legacy_scheduler_and_two_distinct_checkpoints_same_trajectory(self):
        counter=[0]
        def train(net,*args):
            counter[0]+=1
            with torch.no_grad():next(net.parameters()).fill_(counter[0])
            return 1.+.01*counter[0]
        def pred(net,x,device):
            epoch=int(next(net.parameters()).flatten()[0].item())
            labels=self.train['y_train'] if len(x)==64 else self.val['y_val']
            return (labels*.8+.1).astype(np.float32) if epoch==2 else np.full(len(x),.5,np.float32)
        with tempfile.TemporaryDirectory() as directory,patch.object(s,'_train_epoch',side_effect=train),patch.object(s,'probabilities',side_effect=pred):
            r=study.trajectory(43,self.train,self.scaler,Path(directory),'fixture','cpu',True,self.val,max_epochs=20,patience=10)
            self.assertEqual(r['epochs_run'],11)
            self.assertEqual(r['checkpoints']['max_val_auc']['selected_epoch'],2)
            self.assertEqual(r['checkpoints']['min_train_loss']['selected_epoch'],1)
            history=json.loads(Path(r['history']).read_text())
            optimizer=torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))],lr=.001)
            scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=5)
            for row in history:
                self.assertEqual(row['lr_before'],optimizer.param_groups[0]['lr'])
                scheduler.step(row['train_loss'])
                self.assertEqual(row['lr_after'],optimizer.param_groups[0]['lr'])
            self.assertEqual(history[6]['lr_after'],.0005)

    def test_real_cpu_gpu_fits_checkpoint_reload_resume_and_shared_full_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            for device in ['cpu']+[f'cuda:{i}' for i in range(torch.cuda.device_count())]:
                for scheduled in (False,True):
                    root=Path(directory)/device/str(scheduled)
                    r=study.trajectory(43,self.train,self.scaler,root/'inner','fixture',device,scheduled,self.val,max_epochs=3)
                    self.assertEqual(max(r['recomputation_max_error'].values()),0.)
                    before={str(p):(p.stat().st_mtime_ns,s.sha256_file(p)) for p in root.rglob('*') if p.is_file()}
                    with patch.object(s,'_train_epoch',side_effect=AssertionError('retrained')):
                        study.trajectory(43,self.train,self.scaler,root/'inner','fixture',device,scheduled,self.val,max_epochs=3)
                    self.assertEqual(before,{str(p):(p.stat().st_mtime_ns,s.sha256_file(p)) for p in root.rglob('*') if p.is_file()})
                    with self.assertRaisesRegex(ValueError,'fingerprint'):
                        study.trajectory(43,self.train,self.scaler,root/'inner','changed',device,scheduled,self.val,max_epochs=3)
                    fixed=study.trajectory(44,self.train,self.scaler,root/'fixed','fixture',device,scheduled,fixed_epoch=2,max_epochs=3)
                    full=study.trajectory(44,self.train,self.scaler,root/'min','fixture',device,scheduled,max_epochs=3)
                    self.assertEqual(fixed['epochs_run'],2)
                    self.assertEqual(json.loads(Path(fixed['history']).read_text()),json.loads(Path(full['history']).read_text())[:2])
                    with self.assertRaisesRegex(ValueError,'validation-free'):
                        study.trajectory(43,self.train,self.scaler,root/'bad','fixture',device,scheduled,self.val,fixed_epoch=2)
                    with self.assertRaisesRegex(ValueError,'explicit train/validation'):
                        study.trajectory(43,{**self.train,**self.outer},self.scaler,root/'bad','fixture',device,scheduled)
                    value=r['checkpoints']['max_val_auc']
                    with self.assertRaises(FileNotFoundError):s.evaluate(value,self.train,self.outer,root,device)
                    s.atomic_json_save(dict(status='FROZEN_BEFORE_OUTER_EVALUATION',final_head_ids=[value['head_id']]),root/'selection.json')
                    evaluation=s.evaluate(value,self.train,self.outer,root,device)
                    self.assertEqual(evaluation['recomputation_max_error'],0.)

    def test_full_orchestration_freeze_counts_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'new';old=Path(directory)/'old';old.mkdir()
            inner={**self.train,**self.val}
            full=dict(X_train=np.concatenate([self.train['X_train'],self.val['X_val']]),
                      y_train=np.concatenate([self.train['y_train'],self.val['y_val']]))
            scalers=dict(inner=self.scaler,full=s.fit_scaler(full['X_train']))
            self.assertEqual(scalers['full']['n_samples'],80)
            protocol=dict(numerical_exception={'status':'FAIL retained'})
            s.atomic_json_save({'groups':dict(fixed_arch5={},old_three_hidden_direct_reference={})},old/'summary.json')
            def prepare(*args):
                s.study.immutable_json(protocol,root/'protocol.json')
                return inner,full,self.outer,scalers,protocol
            calls=[];original=study.trajectory;evaluate=s.evaluate
            def fit(*args,**kwargs):
                phase='inner' if kwargs.get('validation') is not None else 'full'
                if phase=='full':self.assertTrue((root/'epoch_freeze.json').exists())
                self.assertFalse((root/'selection.json').exists())
                calls.append(phase);kwargs['max_epochs']=3
                return original(*args,**kwargs)
            def test(*args):
                self.assertEqual(calls.count('inner'),6);self.assertEqual(calls.count('full'),12)
                self.assertTrue((root/'selection.json').exists());calls.append('test')
                return evaluate(*args)
            with patch.object(study,'output_root',return_value=root),patch.object(study,'prepare',side_effect=prepare), \
                 patch.object(s,'output_root',return_value=old),patch.object(study.refit,'output_root',return_value=old), \
                 patch.object(study,'trajectory',side_effect=fit),patch.object(s,'evaluate',side_effect=test):
                study.run_model('fixture','cpu')
                summary=json.loads((root/'summary.json').read_text())
                self.assertEqual(summary['trajectories'],18);self.assertEqual(summary['evaluated_checkpoints'],24)
                self.assertEqual(len(summary['groups']),8)
            before={str(p):(p.stat().st_mtime_ns,s.sha256_file(p)) for p in root.rglob('*') if p.is_file() and p.name!='.lock'}
            def resume(*args,**kwargs):
                kwargs['max_epochs']=3
                return original(*args,**kwargs)
            with patch.object(study,'output_root',return_value=root),patch.object(study,'prepare',side_effect=prepare), \
                 patch.object(s,'output_root',return_value=old),patch.object(study.refit,'output_root',return_value=old), \
                 patch.object(study,'trajectory',side_effect=resume),patch.object(s,'_train_epoch',side_effect=AssertionError('retrained')):
                study.run_model('fixture','cpu')
            self.assertEqual(before,{str(p):(p.stat().st_mtime_ns,s.sha256_file(p)) for p in root.rglob('*') if p.is_file() and p.name!='.lock'})


if __name__=='__main__':unittest.main()
