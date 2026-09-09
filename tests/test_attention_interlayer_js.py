import math
import unittest

import torch

from scripts.analyze_attention_interlayer_js import interlayer_js


class AttentionInterlayerJSTest(unittest.TestCase):
    def test_one_hot_layer_matrix_is_symmetric_and_bounded(self) -> None:
        values = torch.tensor(
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]]
        )
        result = interlayer_js(values)[0]
        torch.testing.assert_close(result, result.T)
        torch.testing.assert_close(torch.diag(result), torch.zeros(3))
        self.assertAlmostEqual(float(result[0, 1]), math.log(2.0), places=6)
        self.assertAlmostEqual(float(result[0, 2]), 0.0, places=7)

    def test_union_topk_uses_each_layers_top_token(self) -> None:
        values = torch.tensor([[[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]]])
        result = interlayer_js(values, top_k=1)[0]
        expected = 0.875 * math.log(1.75) + 0.125 * math.log(0.25)
        self.assertAlmostEqual(float(result[0, 1]), expected, places=6)


if __name__ == "__main__":
    unittest.main()
