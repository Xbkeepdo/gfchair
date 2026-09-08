import unittest

import numpy as np
import torch

from scripts.analyze_ffn_write_comparison import _curve_rows, _summaries, distribution_metrics


class FFNWriteComparisonTest(unittest.TestCase):
    def test_r_amp_is_total_variation_and_tracks_redistributed_mass(self):
        p_write = torch.tensor([[0.6, 0.3, 0.1]])
        p_ffn = torch.tensor([[0.5, 0.2, 0.3]])
        metrics, audit = distribution_metrics(p_write, p_ffn, top_k=2)

        np.testing.assert_allclose(metrics["r_amp"], [0.2], atol=1e-7)
        np.testing.assert_allclose(metrics["spearman"], [0.5], atol=1e-7)
        np.testing.assert_allclose(metrics["top1_agreement"], [1.0])
        np.testing.assert_allclose(metrics["top32_overlap"], [0.5])
        self.assertLess(audit["signed_delta_sum_abs_max"], 1e-7)
        self.assertLess(audit["tv_pair_identity_error_max"], 1e-7)
        self.assertEqual(audit["tv_out_of_range_count"], 0)

    def test_invalid_or_unnormalized_distributions_are_rejected(self):
        with self.assertRaisesRegex(AssertionError, "normalization"):
            distribution_metrics(torch.tensor([[0.2, 0.2]]), torch.tensor([[0.5, 0.5]]))
        with self.assertRaisesRegex(ValueError, "non-negative"):
            distribution_metrics(torch.tensor([[1.1, -0.1]]), torch.tensor([[0.5, 0.5]]))

    def test_binary_agreement_summary_uses_mean_not_median(self):
        labels = np.array([1, 1, 0, 0])
        values = {"top1_agreement": np.array([[1.0], [0.0], [1.0], [1.0]])}
        summary = _summaries(_curve_rows("model", labels, values))["top1_agreement"]

        self.assertEqual(summary["summary_statistic"], "mean")
        self.assertEqual(summary["mean_of_layer_statistic"], {"REAL": 0.5, "HALL": 1.0})
        self.assertEqual(summary["hall_above_real_layers"], 1)


if __name__ == "__main__":
    unittest.main()
