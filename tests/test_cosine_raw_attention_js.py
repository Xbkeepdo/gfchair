import unittest
import numpy as np
from scripts.train_cosine_raw_attention_js import raw_js

class RawJSTests(unittest.TestCase):
    def test_distribution_identity_and_mass_invariance(self):
        row=dict(path_signed_q=[[0,.4]],ffn_path_gross=[[1,2]])
        p=np.exp([0,1]);p/=p.sum()
        j,_=raw_js(row,[p*.2]);self.assertLess(abs(float(j[0])),1e-7)
        a,_=raw_js(row,[[.1,.1]]);b,_=raw_js(row,[[.3,.3]])
        np.testing.assert_array_equal(a,b)
        mid=(p+.5)/2;expected=.5*np.sum(p*np.log(p/mid)+.5*np.log(.5/mid))
        np.testing.assert_allclose(a,[expected],rtol=1e-6)
    def test_invalid_raw_mass(self):
        row=dict(path_signed_q=[[0,.4]],ffn_path_gross=[[1,2]])
        for a in ([[0,0]],[[1,1]],[[-.1,.2]],[[float('nan'),.1]]):
            with self.assertRaises(ValueError):raw_js(row,a)

if __name__=='__main__':unittest.main()
