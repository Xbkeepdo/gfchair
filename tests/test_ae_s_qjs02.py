import unittest
import numpy as np
from scripts.train_ae_s_qjs02 import features


class FeatureTests(unittest.TestCase):
    def test_order_temperature_and_visual_axis(self):
        base,x=features([2,3],[0,3],[[0,.2],[.2,0]],[[.5,.5],[.5,.5]])
        p=1/(1+np.exp(1))
        np.testing.assert_allclose(base,[2,3,0,np.log(4)],rtol=1e-6)
        prob=np.array([p,1-p]);t=np.array([.5,.5]);mid=(prob+t)/2
        js=.5*np.sum(prob*np.log(prob/mid)+t*np.log(t/mid))
        np.testing.assert_allclose(x,[2,3,0,np.log(4),js,js],rtol=1e-6)
        _,shift=features([2,3],[0,3],[[100,100.2],[-99.8,-100]],[[.5,.5],[.5,.5]])
        np.testing.assert_allclose(x,shift,rtol=1e-6)

    def test_invalid_inputs(self):
        for q in ([[float('nan')]],[],[[1],[2]]):
            with self.assertRaises(ValueError):features([1],[1],q,[[1.]])

    def test_cosine_normalization_and_temperature(self):
        # Directions have cosine 0 and .2 although their signed projections differ.
        args=([2],[3],[[0,.4]],[[.5,.5]])
        base,x=features(*args,gross=[[3,2]])
        _,expected=features([2],[3],[[0,.2]],[[.5,.5]])
        np.testing.assert_allclose(x,expected,rtol=1e-6)
        _,scaled=features([2],[3],[[0,4]],[[.5,.5]],gross=[[30,20]])
        np.testing.assert_allclose(x,scaled,rtol=1e-6)
        _,raw=features(*args)
        self.assertGreater(abs(float(x[-1]-raw[-1])),.01)


if __name__=='__main__':unittest.main()
