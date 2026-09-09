import math
import unittest

import torch

from scripts.analyze_attention_evidence_js import js_divergence, union_topk_js


class AttentionEvidenceJSTest(unittest.TestCase):
    def test_full_js_extremes(self) -> None:
        left = torch.tensor([[1.0, 0.0]])
        right = torch.tensor([[0.0, 1.0]])
        self.assertAlmostEqual(float(js_divergence(left, left)), 0.0, places=7)
        self.assertAlmostEqual(
            float(js_divergence(left, right)), math.log(2.0), places=6
        )

    def test_union_topk_uses_both_sides_and_renormalizes(self) -> None:
        left = torch.tensor([[0.7, 0.2, 0.1, 0.0]])
        right = torch.tensor([[0.0, 0.1, 0.2, 0.7]])
        actual = union_topk_js(left, right, top_k=1)
        expected = js_divergence(
            torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]])
        )
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
