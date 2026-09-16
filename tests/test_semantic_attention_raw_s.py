import unittest

import numpy as np

from scripts.train_semantic_attention_raw_s_811 import HEADS, matrix


class RawStrengthTest(unittest.TestCase):
    def test_recovered_strength_and_feature_layout(self):
        strength = np.array([[0.2, 4.0], [20.0, 100.0]], dtype=np.float32)
        np.testing.assert_allclose(np.expm1(np.log1p(strength)), strength, rtol=2e-7)
        cube = np.arange(2 * 2 * 6, dtype=np.float32).reshape(2, 2, 6)
        ae = np.ones((2, 2), dtype=np.float32)
        data = {"S_E": strength, "AE_V": ae, "cubes": {"raw": cube}}
        np.testing.assert_array_equal(matrix(data, "reference", "S_only"), strength)
        np.testing.assert_array_equal(matrix(data, "reference", "AE_only"), ae)
        np.testing.assert_array_equal(matrix(data, "reference", "AE_plus_S"),
                                      np.concatenate((ae, strength), axis=1))
        np.testing.assert_allclose(matrix(data, "reference", "logS_only"), np.log1p(strength))
        np.testing.assert_allclose(matrix(data, "reference", "AE_plus_logS"),
                                   np.concatenate((ae, np.log1p(strength)), axis=1))
        np.testing.assert_array_equal(matrix(data, "raw", "attention_only"),
                                      np.concatenate((cube[:, :, 1], strength), axis=1))
        self.assertEqual(len(HEADS), 19)


if __name__ == "__main__":
    unittest.main()
