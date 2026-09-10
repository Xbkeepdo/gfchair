import unittest
import numpy as np
import torch
from scripts.train_q_temperature007 import tempered_js
from scripts.train_q_softmax_js import q_softmax_js


class QTemperatureTest(unittest.TestCase):
    def test_formula_and_concentration(self):
        q=torch.tensor([[.2,-.1,.04]],dtype=torch.float64)
        t=torch.tensor([[.2,.3,.5]],dtype=torch.float64)
        actual,a=tempered_js(q,t)
        p=torch.softmax(q/.07,-1);m=(p+t)/2
        expected=.5*((p*(p/m).log()).sum()+(t*(t/m).log()).sum())
        self.assertAlmostEqual(float(actual[0]),float(expected),places=7)
        self.assertGreater(a['p_max'][0],q_softmax_js(q,t)[1]['p_max'][0])
        np.testing.assert_array_equal(tempered_js(q,t,1)[0],q_softmax_js(q,t)[0])
        for temp in (0,-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):tempered_js(q,t,temp)

    def test_uniform_zero_and_saturated_signed_values(self):
        t=torch.tensor([[0.,1.,0.]],dtype=torch.float64)
        q=torch.tensor([[1000.,-1000.,0.]],dtype=torch.float64)
        js,a=tempered_js(q,t)
        self.assertTrue(np.isfinite(js).all())
        self.assertAlmostEqual(float(js[0]),np.log(2),places=7)
        self.assertEqual(a['softmax_zero_entries'],2)
        uniform=torch.full((1,3),1/3,dtype=torch.float64)
        self.assertAlmostEqual(float(tempered_js(torch.zeros_like(uniform),uniform)[0][0]),0,places=7)
