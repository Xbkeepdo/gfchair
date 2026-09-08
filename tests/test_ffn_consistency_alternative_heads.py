import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts import train_ffn_consistency_alternative_heads as heads


class AlternativeHeadsTest(unittest.TestCase):
    def test_candidates_and_image_group_isolation(self):
        self.assertEqual(len(heads.candidates('one_hidden')),12)
        self.assertEqual(len(heads.candidates('xgboost')),12)
        self.assertTrue(all(len(c['hidden_sizes'])==1 for c in heads.candidates('one_hidden')))
        split=heads.inner_image_split(range(3200),range(3200,4000))
        self.assertEqual(split,heads.inner_image_split(reversed(range(3200)),range(3200,4000)))
        self.assertEqual([len(split[k]) for k in ('train','validation','test')],[2560,640,800])
        self.assertFalse(set(split['train'])&set(split['validation']))
        self.assertEqual(set(split['train'])|set(split['validation']),set(range(3200)))
        with self.assertRaisesRegex(ValueError,'overlap'):
            heads.inner_image_split([1,2,3],[3],1)

    def test_inner_scales_exclude_validation_and_test(self):
        ids=np.array([1,1,2,3,4])
        split=dict(train=[1,2],validation=[3],test=[4])
        raw={k:np.ones((5,2)) for k in ('AE','R_cos','D_EW','D_WF','D_EF','kappa_vec','kappa_end')}
        raw.update(S=np.array([[2.,4.],[4.,8.],[6.,12.],[1e5,2e5],[1e6,2e6]]))
        raw.update(I=3*raw['S'],N_vec=.5*raw['S'],N_end=.6*raw['S'])
        train,validation,scales=heads.inner_features(raw,ids,split)
        np.testing.assert_array_equal(scales['S'],[4.,8.])
        self.assertEqual((len(train),len(validation)),(3,1))
        raw['S'][3:]*=100
        again,_,other=heads.inner_features(raw,ids,split)
        np.testing.assert_array_equal(again,train)
        np.testing.assert_array_equal(other['S'],scales['S'])

    def test_selection_uses_validation_only_and_fixed_ties(self):
        def trial(auc,ap,test):
            return dict(scope='inner_validation_U_SN',validation=dict(auc=auc,hallucination_positive=dict(aupr=ap)),outer_test_auc=test)
        values=[trial(.8,.7,.99),trial(.9,.5,.1),trial(.9,.6,.01),trial(.9,.6,1.)]
        self.assertEqual(heads.select_candidate(values),2)
        values[2]['scope']='outer_test'
        with self.assertRaisesRegex(ValueError,'inner-validation'):
            heads.select_candidate(values)

    def test_small_both_heads_checkpoint_metrics_and_resume(self):
        torch.set_num_threads(1)
        rng=np.random.default_rng(17)
        x=rng.normal(size=(64,5)).astype(np.float32)
        y=(x[:,0]+.5*x[:,1]>0).astype(np.int32)
        data=dict(X_train=x[:48],y_train=y[:48],X_test=x[48:],y_test=y[48:])
        configs={'one_hidden':dict(hidden_sizes=[8],dropout=.3,learning_rate=.001),
                 'xgboost':dict(max_depth=2,learning_rate=.1,n_estimators=3)}
        with tempfile.TemporaryDirectory() as directory:
            for family,config in configs.items():
                root=Path(directory)/family
                result=heads.run_head(family,config,43,data,root,'frozen-data','cpu',epochs=2)
                self.assertEqual(result['test_probabilities'].shape,(16,))
                self.assertEqual(result['recomputation_max_error'],dict(train=0.,test=0.))
                before={str(p):(heads.sha256_file(p),p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file()}
                with patch.object(heads,'train_and_evaluate_probe',side_effect=AssertionError('must not retrain')), \
                     patch('xgboost.XGBClassifier.fit',side_effect=AssertionError('must not refit')):
                    resumed=heads.run_head(family,config,43,data,root,'frozen-data','cpu',epochs=2)
                np.testing.assert_array_equal(resumed['test_probabilities'],result['test_probabilities'])
                self.assertEqual(before,{str(p):(heads.sha256_file(p),p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file()})
                with self.assertRaisesRegex(ValueError,'fingerprint'):
                    heads.run_head(family,config,43,data,root,'changed-data','cpu',epochs=2)

    def test_both_searches_are_sealed_before_any_outer_evaluation(self):
        y=np.array([0,1,0,1],dtype=np.int32)
        data=dict(y_train=y,y_test=y,numerical_exception={'status':'ORIGINAL_FAIL_RETAINED'})
        for split in ('train','test'):
            data['X_'+split]={spec:np.ones((4,len(blocks)*2),dtype=np.float32) for spec,blocks in heads.study.FEATURE_GROUPS.items()}
        inner=dict(X_train=np.zeros((4,8),dtype=np.float32),X_test=np.zeros((4,8),dtype=np.float32),y_train=y,y_test=y)
        probabilities=np.array([.2,.8,.3,.7],dtype=np.float32)
        metrics=heads.metric_reports(y,probabilities,y,probabilities)
        result=dict(metrics=metrics,train_probabilities=probabilities,test_probabilities=probabilities)
        old={spec:{str(seed):result for seed in heads.SEEDS} for spec in heads.study.FEATURE_GROUPS}
        baseline={'groups':heads.study.summarize_heads(data,old)}
        calls=[]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            heads.study.immutable_json({'test_protocol':True},root/'protocol.json')
            def fit(family,config,seed,inputs,path,signature,device):
                calls.append((family,str(path)))
                if 'search' in path.parts:
                    self.assertIs(inputs,inner)
                    self.assertFalse((root/'selection.json').exists())
                else:
                    self.assertTrue((root/'selection.json').exists())
                    self.assertEqual(len([c for c in calls if '/search/' in c[1]]),24)
                    self.assertIs(inputs['y_test'],data['y_test'])
                return result
            with patch.object(heads,'output_root',return_value=root), \
                 patch.object(heads,'prepare',return_value=(data,inner,baseline,{'test_protocol':True})), \
                 patch.object(heads,'run_head',side_effect=fit):
                heads.run_model('synthetic','cpu')
            self.assertEqual(len(calls),120)
            summary=json.loads((root/'summary.json').read_text())
            self.assertEqual(set(summary['methods']),{'three_hidden_fixed','one_hidden','xgboost'})
            self.assertTrue(all(len(v)==16 for v in summary['methods'].values()))


if __name__=='__main__':
    unittest.main()
