import unittest
import numpy as np
from scripts.train_raw_attention_strength import signals

class RawAttentionTests(unittest.TestCase):
    def test_visual_sum_without_renormalization(self):
        r,s=signals([[.1,.2],[.03,.07]],[0,3])
        np.testing.assert_allclose(r,[.3,.1]);np.testing.assert_allclose(s,[0,np.log(4)],rtol=1e-6)
        doubled,_=signals([[.2,.4],[.06,.14]],[0,3]);np.testing.assert_allclose(doubled,2*r)
    def test_invalid(self):
        for a,s in [([[.8,.8]],[1]),([[-.1,.2]],[1]),([[.1]],[float('nan')]),([[.1,.2]],[1,2])]:
            with self.assertRaises(ValueError):signals(a,s)

if __name__=='__main__':unittest.main()
