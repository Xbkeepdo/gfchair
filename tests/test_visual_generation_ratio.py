import unittest
import numpy as np
from scripts.plot_visual_generation_ratio import ratio

class RatioTests(unittest.TestCase):
    def test_ratio_before_aggregation_and_zero(self):
        x=ratio([[1,0,1],[3,1,0]],[[1,0,0],[2,2,1]])
        np.testing.assert_allclose(x[:,0],[1,1.5]);self.assertEqual(x[:,0].mean(),1.25)
        self.assertNotEqual(x[:,0].mean(),4/3)
        self.assertTrue(np.isnan(x[0,1:]).all());self.assertEqual(x[1,2],0)
    def test_invalid(self):
        for v,g in [([[-1]],[[1]]),([[1]],[[np.nan]]),([[1,2]],[[1]])]:
            with self.assertRaises(ValueError):ratio(v,g)

if __name__=='__main__':unittest.main()
