import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn

from scripts import train_ffn_shallow_standardized_search as search


class ShallowSearchTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.x=np.random.default_rng(5).normal(size=(96,5)).astype(np.float32)
        self.x[:,4]=7
        self.y=(self.x[:,0]>0).astype(np.int32)
        self.inner=dict(X_train=self.x[:64],y_train=self.y[:64],X_val=self.x[64:80],y_val=self.y[64:80])
        self.outer=dict(X_test=self.x[80:],y_test=self.y[80:])
        self.scaler=search.fit_scaler(self.inner['X_train'])

    def test_scaler_train_only_constant_feature_and_image_split(self):
        reference=StandardScaler().fit(self.x[:64])
        np.testing.assert_array_equal(search.transform(self.x,self.scaler),reference.transform(self.x))
        self.assertEqual(self.scaler['scale'][-1],1.)
        old=self.scaler['mean'].copy()
        self.x[64:]*=10000
        np.testing.assert_array_equal(search.fit_scaler(self.x[:64])['mean'],old)
        np.testing.assert_array_equal(search.transform(self.x,self.scaler,False),self.x)
        split=search.old.previous.inner_image_split(range(32),range(32,40),validation_count=8)
        self.assertFalse(set(split['train'])&set(split['validation']))
        self.assertFalse(set(split['test'])&(set(split['train'])|set(split['validation'])))
        with self.assertRaisesRegex(ValueError,'overlap'):
            search.old.previous.inner_image_split(range(32),range(31,40),validation_count=8)

    def test_architectures_grid_bn_and_singleton_batch(self):
        self.assertEqual(len(search.ARCHITECTURES),6)
        for arch in search.ARCHITECTURES:
            configs=search.grid(arch)
            self.assertEqual(len(configs),48 if arch else 12)
            self.assertEqual(len({search.key(c) for c in configs}),len(configs))
            self.assertIn(search.config(arch),configs)
            model=search.make_probe(5,search.config(arch))
            self.assertEqual(sum(isinstance(m,nn.Linear) for m in model.modules()),len(arch)+1)
            self.assertFalse(any(isinstance(m,nn.BatchNorm1d) for m in model.modules()))
        model=search.make_probe(96,search.config((128,64,32),batch_norm=True))
        self.assertEqual(sum(isinstance(m,nn.BatchNorm1d) for m in model.modules()),3)
        dataset=search.MatrixDataset(np.zeros((129,2),np.float32),np.arange(129))
        batches=list(search.epoch_loader(dataset,64))
        self.assertEqual([len(v[1]) for v in batches],[64,65])
        self.assertEqual(sorted(torch.cat([v[1] for v in batches]).tolist()),list(range(129)))

    def test_select_only_validation(self):
        rows=[dict(scope='inner_validation',index=i,auc=a,hall_aupr=h,outer_auc=t)
              for i,(a,h,t) in enumerate([(.8,.7,1),(.9,.5,0),(.9,.6,0),(.9,.6,1)])]
        self.assertEqual([v['index'] for v in search.rank(rows)],[2,3,1,0])
        rows[0]['scope']='test'
        with self.assertRaisesRegex(ValueError,'validation-only'):
            search.rank(rows)

    def test_real_fit_reload_resume_and_frozen_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for cfg in (search.config(()),search.config((8,4)),search.config((8,),batch_norm=True)):
                result=search.run_head(cfg,43,self.inner,self.scaler,root,'fixture','cpu',epochs=3,patience=2)
                self.assertEqual(result['recomputation_max_error'],dict(train=0.,val=0.))
                history=json.loads((Path(result['checkpoint']).parent/'history.json').read_text())
                best=max(history,key=lambda r:(r['validation_auc'],r['validation_hall_aupr'],-r['epoch']))
                self.assertEqual(result['best_epoch'],best['epoch'])
                before={str(p):(p.stat().st_mtime_ns,search.sha256_file(p)) for p in root.rglob('*') if p.is_file()}
                with patch.object(search,'_train_epoch',side_effect=AssertionError('retrained')):
                    search.run_head(cfg,43,self.inner,self.scaler,root,'fixture','cpu',epochs=3,patience=2)
                self.assertEqual(before,{str(p):(p.stat().st_mtime_ns,search.sha256_file(p)) for p in root.rglob('*') if p.is_file()})
                with self.assertRaisesRegex(ValueError,'fingerprint'):
                    search.run_head(cfg,43,self.inner,self.scaler,root,'wrong','cpu',epochs=3,patience=2)
            with self.assertRaises(FileNotFoundError):
                search.evaluate(result,self.inner,self.outer,root,'cpu')
            search.atomic_json_save(dict(status='FROZEN_BEFORE_OUTER_EVALUATION',final_head_ids=[result['head_id']]),root/'selection.json')
            value=search.evaluate(result,self.inner,self.outer,root,'cpu')
            self.assertEqual(value['recomputation_max_error'],0.)
            self.assertEqual(value['metrics'],search.evaluate(result,self.inner,self.outer,root,'cpu')['metrics'])

    def test_checkpoint_is_validation_best_not_train_loss_best(self):
        counter=[0]
        def train(model,*args,**kwargs):
            counter[0]+=1
            with torch.no_grad():
                next(model.parameters()).fill_(counter[0])
            return 1/counter[0]
        def pred(model,x,device):
            epoch=int(next(model.parameters()).flatten()[0].item())
            labels=self.inner['y_train'] if len(x)==64 else self.inner['y_val']
            return (labels*.8+.1).astype(np.float32) if epoch==2 else np.full(len(x),.5,np.float32)
        with tempfile.TemporaryDirectory() as directory,patch.object(search,'_train_epoch',side_effect=train),patch.object(search,'probabilities',side_effect=pred):
            value=search.run_head(search.config(()),43,self.inner,self.scaler,Path(directory),'fixture','cpu',epochs=10,patience=2)
            self.assertEqual(value['best_epoch'],2)
            self.assertEqual(value['epochs_run'],4)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA smoke requires GPU')
    def test_cuda_same_device_and_dtype(self):
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        with tempfile.TemporaryDirectory() as directory:
            for i in range(torch.cuda.device_count()):
                value=search.run_head(search.config((8,4)),43,self.inner,self.scaler,Path(directory)/str(i),
                                      'fixture',f'cuda:{i}',epochs=2,patience=2)
                self.assertEqual(value['recomputation_max_error'],dict(train=0.,val=0.))
                self.assertGreater(value['peak_cuda_allocated_bytes'],0)

    def test_all_choices_precede_outer_evaluation(self):
        calls=[]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            protocol=dict(feature_dim=5,numerical_exception={'status':'FAIL retained'})
            search.study.immutable_json(protocol,root/'protocol.json')
            def fit(cfg,seed,inner,scaler,root,signature,device):
                self.assertIs(inner,self.inner)
                self.assertFalse((root/'selection.json').exists())
                calls.append(('fit',cfg,seed))
                hid=search.key(dict(config=cfg,seed=seed))[:20]
                val=.7+.01*len(cfg['hidden_sizes'])
                v=dict(head_id=hid,seed=seed,config=cfg,best_epoch=2,validation_auc=val,validation_hall_aupr=val,
                       elapsed_seconds=1.,peak_cuda_allocated_bytes=0)
                search.atomic_torch_save(v,root/'heads'/hid/'result.pt')
                return v
            def evaluate(v,inner,outer,root,device):
                self.assertTrue((root/'selection.json').exists())
                self.assertIs(outer,self.outer)
                calls.append(('evaluate',v['head_id']))
                p=self.inner['y_train']*.8+.1;q=self.outer['y_test']*.8+.1
                return dict(seed=v['seed'],train_probabilities=p,test_probabilities=q,
                            metrics=search.old.previous.metric_reports(self.inner['y_train'],p,self.outer['y_test'],q))
            with patch.object(search,'output_root',return_value=root),patch.object(search,'prepare',return_value=(self.inner,self.outer,self.scaler,{},protocol)), \
                    patch.object(search,'run_head',side_effect=fit),patch.object(search,'evaluate',side_effect=evaluate):
                search.run_model('fixture','cpu')
            selection=json.loads((root/'selection.json').read_text())
            self.assertEqual(len(selection['group_configs']),10)
            self.assertEqual(len(selection['tuned']),2)
            first_eval=next(i for i,r in enumerate(calls) if r[0]=='evaluate')
            self.assertTrue(all(c[0]=='evaluate' for c in calls[first_eval:]))
            summary=json.loads((root/'summary.json').read_text())
            self.assertEqual(len(summary['groups']),11)


if __name__=='__main__':
    unittest.main()
