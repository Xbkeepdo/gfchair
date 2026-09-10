import unittest
import numpy as np
import torch
from scripts.train_q_softmax_js import q_softmax_js,build_groups


class QSoftmaxJSTest(unittest.TestCase):
    def test_signed_softmax_not_cosine_and_shift_invariance(self):
        q=torch.tensor([[2.,0.,-4.]],dtype=torch.float64)
        t=torch.tensor([[.5,.25,.25]],dtype=torch.float64)
        p=q.softmax(-1);m=(p+t)/2
        expected=.5*((p*(p/m).log()).sum()+(t*(t/m).log()).sum())
        actual,_=q_softmax_js(q,t)
        self.assertAlmostEqual(actual[0],expected.item(),places=7)
        np.testing.assert_array_equal(q_softmax_js(q+100,t)[0],actual)
        self.assertGreater(abs(q_softmax_js(q*7,t)[0][0]-actual[0]),.01)
        self.assertGreater(abs(q_softmax_js(q.abs(),t)[0][0]-actual[0]),.01)

    def test_zero_extreme_and_full_support(self):
        q=torch.zeros(1,80,dtype=torch.float64);t=torch.full_like(q,1/80)
        self.assertAlmostEqual(float(q_softmax_js(q,t)[0][0]),0,places=7)
        q[0,0]=1e8;t.zero_();t[0,-1]=1
        value,a=q_softmax_js(q,t)
        self.assertAlmostEqual(float(value[0]),np.log(2),places=7)
        self.assertEqual(a['softmax_zero_entries'],79)
        for bad in (torch.full_like(q,float('nan')),q[:,:32]):
            with self.assertRaises(ValueError):q_softmax_js(bad,t)
        with self.assertRaises(ValueError):q_softmax_js(q,t*2)

    def test_fixed_groups_order(self):
        ae=np.ones((2,3));s=ae*2;k=ae*.5;j=ae*.2;jc=ae*.1
        g=build_groups(ae,s,k,j,jc)
        self.assertEqual(set(g),{'F','F_K','J_QT','J_cosT','F_J_QT','F_K_J_QT'})
        np.testing.assert_allclose(g['F_K_J_QT'],np.concatenate((ae,np.log1p(s),k,j),axis=1))
        self.assertEqual(g['J_QT'].shape,(2,3))
