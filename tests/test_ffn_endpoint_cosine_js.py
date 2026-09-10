import numpy as np
import torch
import unittest

from scripts.analyze_ffn_endpoint_cosine_js import endpoint_cosine_js
from scripts.train_endpoint_cosine_js_decomposition import endpoint_js_decomposition


class EndpointCosineJSTest(unittest.TestCase):
    def test_endpoint_cosine_softmax_and_js(self) -> None:
        position = {
            "ffn_path_gross": torch.tensor([[2.0, 3.0, 4.0], [0.0, 0.0, 0.0]]),
            "path_signed_q": torch.tensor([[2.0, 0.0, -4.0], [0.0, 0.0, 0.0]]),
            "attention_evidence": torch.tensor([[0.5, 0.25, 0.25], [0.6, 0.3, 0.1]]),
        }
        actual, audit = endpoint_cosine_js(position)

        cosine = torch.tensor([[1.0, 0.0, -1.0]], dtype=torch.float64)
        source = torch.softmax(cosine, dim=-1)
        target = torch.tensor([[0.5, 0.25, 0.25]], dtype=torch.float64)
        midpoint = 0.5 * (source + target)
        expected = 0.5 * (
            (source * (source.log() - midpoint.log())).sum()
            + (target * (target.log() - midpoint.log())).sum()
        )
        self.assertAlmostEqual(float(actual[0]), float(expected), places=8)
        self.assertTrue(np.isfinite(actual).all() and 0 <= actual[1] <= np.log(2))
        self.assertEqual(audit["zero_gross_entries"], 3)
        self.assertEqual(audit["clipped_cosine_entries"], 0)

        scaled = dict(position)
        scaled["ffn_path_gross"] = position["ffn_path_gross"] * 7
        scaled["path_signed_q"] = position["path_signed_q"] * 7
        np.testing.assert_array_equal(endpoint_cosine_js(scaled)[0], actual)

        invalid = dict(position)
        invalid["path_signed_q"] = torch.tensor([[2.1, 0.0, -4.0], [0.0, 0.0, 0.0]])
        with self.assertRaisesRegex(ValueError, "cosine bound"):
            endpoint_cosine_js(invalid)

    def test_uniform_and_endpoint_residual_are_an_exact_decomposition(self) -> None:
        position = {
            "ffn_path_gross": torch.tensor([[2.0, 3.0, 4.0], [0.0, 0.0, 0.0]]),
            "path_signed_q": torch.tensor([[2.0, 0.0, -4.0], [0.0, 0.0, 0.0]]),
            "attention_evidence": torch.tensor([[0.5, 0.25, 0.25], [0.6, 0.3, 0.1]]),
        }
        values = endpoint_js_decomposition(position)
        np.testing.assert_allclose(
            values["J_T"] + values["J_E"], values["J_PT"], atol=1e-8, rtol=0
        )
        self.assertAlmostEqual(float(values["J_E"][1]), 0.0, places=8)


if __name__ == "__main__":
    unittest.main()
