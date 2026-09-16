import unittest
import numpy as np
from scripts.evaluate_generation_per_position_811 import build_features


class GenerationRatioTest(unittest.TestCase):
    def test_divide_before_log_and_single_position_column(self):
        ratio,groups=build_features([[6,12],[4,8]],[3,2])
        np.testing.assert_array_equal(ratio,[[2,4],[2,4]])
        np.testing.assert_allclose(groups['S_G_per_position'],np.log1p(ratio))
        np.testing.assert_allclose(groups['S_G'],np.log1p([[6,12],[4,8]]))
        self.assertEqual(groups['position'].shape,(2,1))

    def test_zero_position_retains_row_and_raw_undefined(self):
        ratio,groups=build_features([[0,0],[0,2]],[0,2])
        self.assertTrue(np.isnan(ratio[0]).all())
        np.testing.assert_array_equal(groups['S_G_per_position'][0],[0,0])
        np.testing.assert_array_equal(ratio[1],[0,1])
        self.assertEqual(groups['position'][0,0],0)

    def test_invalid_inputs_rejected(self):
        for position in ([-1],[.5],[np.nan],[1,2]):
            with self.assertRaises(ValueError):build_features([[2,3]],position)
        with self.assertRaises(ValueError):build_features([[-1,3]],[1])


if __name__=='__main__':unittest.main()
