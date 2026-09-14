import inspect
import json
import numpy as np
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import torch
from scripts import search_single_mlp_811 as search


def test_train_scaling_and_batch_coverage():
    train=np.array([[1.,4.],[3.,4.]])
    mean,scale=search.scale_fit(train,True)
    np.testing.assert_array_equal(search.transform(train,mean,scale),[[-1.,0.],[1.,0.]])
    np.testing.assert_array_equal(search.transform([[101.,4.]],mean,scale),[[99.,0.]])
    ix=torch.randperm(65)
    chunks=search.batches(ix,32)
    assert all(len(c)>1 for c in chunks)
    torch.testing.assert_close(torch.cat(chunks),ix)


def test_checkpoint_reload_and_validation_selection(monitor):
    torch.set_num_threads(1)
    rng=np.random.default_rng(4)
    x=rng.normal(size=(65,4)).astype(np.float32);y=(x[:,0]>0).astype(int)
    v=rng.normal(size=(23,4)).astype(np.float32);vy=(v[:,0]>0).astype(int)
    cfg=dict(search.candidates()[0],width=8,max_epochs=5,batch_size=32,monitor=monitor,standardize=True)
    r=search.fit(x,y,v,vy,cfg,43,'cpu')
    history=r['history']
    best=min(history,key=lambda h:h['val_loss']) if monitor=='val_loss' else max(history,key=lambda h:h['AUROC'])
    assert r['best_epoch']==best['epoch']
    net=search.SingleMLP(4,cfg);net.load_state_dict(r['state_dict'])
    p=search.predict(net,torch.as_tensor(search.transform(v,r['mean'],r['scale'])))
    np.testing.assert_array_equal(p,r['validation_probabilities'])
    assert r['validation']==search.metrics(vy,p)
    assert not any('test' in key for key in r)
    assert not any('test' in key for key in inspect.signature(search.fit).parameters)


def check_global_gate(tmp_path):
    for i,model in enumerate(search.MODELS):
        assert not search.test_gate()
        root=tmp_path/model;root.mkdir()
        (root/'protocol.json').write_text(json.dumps(dict(candidates=search.candidates(),variants=list(search.VARIANTS),fingerprint=str(i))))
        (root/'selection.json').write_text(json.dumps(dict(fingerprint=str(i))))
    assert search.test_gate()
    (root/'selection.json').write_text(json.dumps(dict(fingerprint='wrong')))
    try:
        search.test_gate()
    except AssertionError:
        pass
    else:
        raise AssertionError("Mismatched fingerprint accepted")


def test_fixed_budget_architecture_and_ranking():
    configs=search.candidates()
    assert len(configs)==24 and configs==search.candidates()
    assert len({json.dumps(c,sort_keys=True) for c in configs})==24
    assert len(search.VARIANTS)*(len(configs)+search.TOP_N*2)==240
    for cfg in configs:
        layers=[m for m in search.SingleMLP(16,cfg).modules() if isinstance(m,torch.nn.Linear)]
        assert len(layers)==2 and layers[0].out_features==cfg['width'] and layers[1].out_features==1
    rows=[dict(index=i,validation=dict(AUROC=.8,HALL_AUPR=.6),test_auc=1-i) for i in range(3)]
    assert max(rows,key=search.rank)['index']==0
    rows[2]['validation']['AUROC']=.81
    assert max(rows,key=search.rank)['index']==2


class SingleMLPSearchTests(unittest.TestCase):
    def test_scaling_and_coverage(self):test_train_scaling_and_batch_coverage()
    def test_loss_checkpoint(self):test_checkpoint_reload_and_validation_selection('val_loss')
    def test_auc_checkpoint(self):test_checkpoint_reload_and_validation_selection('val_auroc')
    def test_budget_and_ranking(self):test_fixed_budget_architecture_and_ranking()
    def test_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            with patch.object(search,'OUT',root):check_global_gate(root)

    def test_interrupted_search_resumes_saved_fits(self):
        protocol=dict(fingerprint='resume-test',candidates=search.candidates())
        arrays={v:dict(train_x=np.zeros((2,2)),val_x=np.zeros((2,2))) for v in search.VARIANTS}
        prepared=(arrays,np.array([0,1]),np.array([0,1]),protocol)
        saved=dict(validation=dict(AUROC=.8,HALL_AUPR=.5),seconds=0.)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            with patch.object(search,'OUT',root),patch.object(search,'prepare',return_value=prepared),patch('builtins.print'):
                with patch.object(search,'fit',side_effect=[dict(saved),RuntimeError('simulated interruption')]):
                    with self.assertRaisesRegex(RuntimeError,'simulated interruption'):search.run_search('model','cpu')
                existing=root/'model'/search.VARIANTS[0]/'candidate00/seed43/result.pt'
                stamp=existing.stat().st_mtime_ns
                with patch.object(search,'fit',side_effect=lambda **kw:dict(saved)) as fitting:
                    search.run_search('model','cpu')
                    self.assertEqual(fitting.call_count,239)
                self.assertEqual(existing.stat().st_mtime_ns,stamp)
                selection=json.loads((root/'model/selection.json').read_text())
                self.assertEqual(selection['champion'],search.VARIANTS[0])
                self.assertFalse(selection['test_accessed'])

if __name__=='__main__':unittest.main()
