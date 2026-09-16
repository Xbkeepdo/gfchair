import unittest

import numpy as np

from scripts.optuna_prefix_full_visual_product_811 import visual_product


class PrefixFullVisualProductTest(unittest.TestCase):
    def test_uses_every_visual_token_without_region_normalization(self):
        attention = np.array([[0.1, 0.2, 0.7], [0.4, 0.5, 0.1]], dtype=np.float32)
        probability = np.array([[0.5, 0.3, 0.2], [0.9, 0.1, 0.6]], dtype=np.float32)
        np.testing.assert_allclose(visual_product(attention, probability), [0.25, 0.47], atol=1e-7)


if __name__ == "__main__":
    unittest.main()
