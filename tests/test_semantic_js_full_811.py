import unittest

import numpy as np

from scripts.optuna_semantic_js_full_811 import js_on_region, one_image


class SemanticJSFullTest(unittest.TestCase):
    def test_base2_js_on_a_restricted_region(self):
        attention = np.array([[4, 0, 10]], dtype=np.float32)
        probability = np.array([[0, 7, 10]], dtype=np.float32)
        mask = np.array([[True, True, False]])
        score, empty = js_on_region(attention, probability, mask)
        self.assertAlmostEqual(float(score[0]), 1.0)
        self.assertEqual(empty, 0)

    def test_attention_top32_restricts_both_distributions(self):
        attention = np.concatenate((np.arange(2, 34), np.full(8, 0.01))).astype(np.float32)[None]
        probability = np.concatenate((attention[0, :32] * 0.01, np.full(8, 100))).astype(np.float32)[None]
        attention_js, union_js, empty_a, empty_union = one_image(attention, probability)
        self.assertAlmostEqual(float(attention_js[0]), 0.0, places=6)
        self.assertGreater(float(union_js[0]), 0.0)
        self.assertEqual((empty_a, empty_union), (0, 0))


if __name__ == "__main__":
    unittest.main()
