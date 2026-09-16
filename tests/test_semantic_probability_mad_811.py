import unittest

import numpy as np

from scripts.optuna_semantic_probability_mad_811 import probability_mad


class SemanticProbabilityMADTest(unittest.TestCase):
    def test_median_absolute_deviation_over_visual_tokens(self):
        probability = np.array([[0, 1, 2, 3, 4], [0, 0, 0, 0, 10]], dtype=np.float32)
        np.testing.assert_array_equal(probability_mad(probability), [1, 0])

    def test_tiny_positive_probabilities_are_retained(self):
        probability = np.array([[0, 1e-30, 2e-30, 3e-30, 4e-30]], dtype=np.float32)
        np.testing.assert_allclose(probability_mad(probability), [1e-30], rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
