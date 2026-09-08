import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts import train_ffn_ae_log1p_search as search


class DirectLogSearchTest(unittest.TestCase):
    def test_direct_log_no_tau_and_inner_image_isolation(self):
        ae = np.arange(10,dtype=np.float32).reshape(5,2)
        strength = np.array([[0,1],[2,9],[99,999],[1e5,1e6],[1e7,1e8]])
        x = search.direct_features(ae,strength)
        np.testing.assert_array_equal(x[:,:2],ae)
        np.testing.assert_array_equal(x[:,2:],np.log1p(strength).astype(np.float32))
        self.assertEqual(x.dtype,np.float32)
        data = dict(X_train=x[:4],y_train=np.array([0,1,0,1]),
                    train_mentions=[dict(image_id=i) for i in [1,1,2,3]])
        split = dict(train=[1,2],validation=[3],test=[4])
        inner = search.inner_inputs(data,split)
        np.testing.assert_array_equal(inner['X_train'],x[:3])
        np.testing.assert_array_equal(inner['X_test'],x[3:4])
        data['X_train'][3]*=100
        np.testing.assert_array_equal(search.inner_inputs(data,split)['X_train'],inner['X_train'])
        with self.assertRaisesRegex(ValueError,'leakage'):
            search.inner_inputs(data,dict(train=[1,2],validation=[2,3],test=[4]))
        with self.assertRaisesRegex(ValueError,'strength'):
            search.direct_features(ae,-np.ones_like(ae))

    def test_search_budget_parameters_and_validation_only_selection(self):
        for family in search.FAMILIES:
            values = search.candidates(family)
            self.assertEqual(len(values),48)
            self.assertEqual(values,search.candidates(family))
            self.assertEqual(len({json.dumps(c,sort_keys=True) for c in values}),48)
            if family!='xgboost':
                self.assertTrue(all(len(c['hidden_sizes'])==(1 if family=='one_hidden' else 3) for c in values))
                self.assertTrue(all(0<=c['dropout']<=.5 and 1e-5<=c['learning_rate']<=1e-2 for c in values))
        self.assertIn(search.fixed_mlp(),search.candidates('three_hidden'))
        rows = [dict(scope='inner_validation_direct_log1p',index=i,auc=a,hall_aupr=p,outer_auc=t)
                for i,(a,p,t) in enumerate([(.8,.7,1.),(.9,.5,.1),(.9,.6,0),(.9,.6,1)])]
        self.assertEqual([r['index'] for r in search.rank_validation(rows)],[2,3,1,0])
        rows[0]['scope']='outer_test'
        with self.assertRaisesRegex(ValueError,'validation-only'):
            search.rank_validation(rows)

    def test_real_three_families_fit_checkpoint_metrics_and_resume(self):
        torch.set_num_threads(1)
        x = np.random.default_rng(4).normal(size=(64,4)).astype(np.float32)
        y = (x[:,0]>0).astype(np.int32)
        inputs = dict(X_train=x[:48],X_test=x[48:],y_train=y[:48],y_test=y[48:])
        configs = dict(one_hidden=dict(hidden_sizes=[8],dropout=.1,learning_rate=.002,weight_decay=.0002),
                       three_hidden=dict(hidden_sizes=[8,4,2],dropout=.1,learning_rate=.002,weight_decay=.0003),
                       xgboost=dict(max_depth=2,n_estimators=3,learning_rate=.1,min_child_weight=1,
                                    reg_alpha=.2,reg_lambda=2,subsample=.7,colsample_bytree=.7))
        with tempfile.TemporaryDirectory() as directory:
            for family,config in configs.items():
                path=Path(directory)/family
                result=search.run_head(family,config,43,inputs,path,'fixture','cpu',epochs=2)
                self.assertEqual(result['recomputation_max_error'],dict(train=0.,test=0.))
                if family!='xgboost':
                    saved=json.loads((Path(result['checkpoint']).parent/'config.json').read_text())
                    self.assertEqual(saved['weight_decay'],config['weight_decay'])
                before={str(p):(search.sha256_file(p),p.stat().st_mtime_ns) for p in path.rglob('*') if p.is_file()}
                with patch.object(search,'train_and_evaluate_probe',side_effect=AssertionError('retrain')), \
                     patch('xgboost.XGBClassifier.fit',side_effect=AssertionError('refit')):
                    search.run_head(family,config,43,inputs,path,'fixture','cpu',epochs=2)
                self.assertEqual(before,{str(p):(search.sha256_file(p),p.stat().st_mtime_ns) for p in path.rglob('*') if p.is_file()})
                with self.assertRaisesRegex(ValueError,'fingerprint'):
                    search.run_head(family,config,43,inputs,path,'changed','cpu',epochs=2)

    def test_all_searches_and_repeats_frozen_before_outer_test(self):
        y=np.array([0,1,0,1],dtype=np.int32)
        direct=dict(X_train=np.ones((4,4),np.float32),X_test=np.ones((4,4),np.float32),y_train=y,y_test=y)
        scaled={**direct,'X_train':direct['X_train']*2}
        inner={**direct,'X_train':direct['X_train']*3}
        p=np.array([.1,.9,.2,.8],dtype=np.float32)
        calls=[]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            protocol={'numerical_exception':{'original_status':'FAIL'},'test':True}
            search.study.immutable_json(protocol,root/'protocol.json')
            def fit(family,config,seed,inputs,path,signature,device):
                calls.append((family,seed,path))
                if 'search' in path.parts:
                    self.assertIs(inputs,inner)
                    self.assertFalse((root/'selection.json').exists())
                else:
                    self.assertTrue((root/'selection.json').exists())
                    self.assertEqual(len([r for r in calls if 'search' in r[2].parts]),162)
                    self.assertTrue(inputs is direct or inputs is scaled)
                result=dict(seed=seed,config=config,train_probabilities=p,test_probabilities=p,
                            metrics=search.previous.metric_reports(y,p,y,p))
                search.atomic_torch_save(result,path/'result.pt')
                return result
            with patch.object(search,'output_root',return_value=root), \
                 patch.object(search,'prepare',return_value=(direct,scaled,inner,{'groups':{'AE+s':{'old':True}}},protocol)), \
                 patch.object(search,'run_head',side_effect=fit):
                search.run_model('synthetic','cpu')
            self.assertEqual(len(calls),183)
            summary=json.loads((root/'summary.json').read_text())
            self.assertEqual(summary['counts'],dict(search=162,final=21))
            self.assertEqual(len(summary['groups']),8)
            self.assertTrue(all(v['winner']['seeds']==[43,44,45] for v in summary['selected'].values()))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA smoke requires a GPU')
    def test_cuda_allocator_initialization_and_same_device_reload(self):
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False
        x=np.random.default_rng(9).normal(size=(32,4)).astype(np.float32)
        y=(x[:,0]>0).astype(np.int32)
        data=dict(X_train=x[:24],X_test=x[24:],y_train=y[:24],y_test=y[24:])
        with tempfile.TemporaryDirectory() as directory:
            for index in range(torch.cuda.device_count()):
                config=dict(hidden_sizes=[8],dropout=.3,learning_rate=.001,weight_decay=.0001)
                result=search.run_head('one_hidden',config,43,data,Path(directory)/str(index),'cuda-smoke',f'cuda:{index}',epochs=2)
                self.assertEqual(result['recomputation_max_error'],dict(train=0.,test=0.))
                self.assertGreater(result['peak_cuda_allocated_bytes'],0)


if __name__=='__main__':
    unittest.main()
