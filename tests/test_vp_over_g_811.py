import unittest

import numpy as np

from scripts import evaluate_vp_over_g_811 as ratio


class VPOverGFeatureTest(unittest.TestCase):
    def test_direct_ratio_and_log_ratio(self):
        signals = {
            "ae_visual_prompt_sum": np.array([[2.0, 6.0]]),
            "ae_generation": np.array([[1.0, 3.0]]),
            "gross_visual": np.array([[2.0, 3.0]]),
            "gross_prompt": np.array([[2.0, 1.0]]),
            "gross_generation": np.array([[2.0, 8.0]]),
        }
        raw, detector, diagnostics = ratio.build_ratio_features(signals)
        np.testing.assert_allclose(raw[:, :2], [[2.0, 2.0]])
        np.testing.assert_allclose(raw[:, 2:], np.log([[2.0, 0.5]]))
        np.testing.assert_allclose(detector, raw.astype(np.float32))
        self.assertEqual(diagnostics["detector_nonfinite"], 0)

    def test_undefined_raw_is_nan_and_detector_is_zero(self):
        signals = {
            "ae_visual_prompt_sum": np.array([[2.0, 0.0]]),
            "ae_generation": np.array([[0.0, 1.0]]),
            "gross_visual": np.array([[1.0, 0.0]]),
            "gross_prompt": np.array([[1.0, 0.0]]),
            "gross_generation": np.array([[0.0, 1.0]]),
        }
        raw, detector, diagnostics = ratio.build_ratio_features(signals)
        self.assertTrue(np.isnan(raw[0, 0]))
        self.assertTrue(np.isnan(raw[0, 2]))
        self.assertTrue(np.isnan(raw[0, 3]))
        self.assertEqual(detector[0, 0], 0.0)
        self.assertEqual(detector[0, 2], 0.0)
        self.assertEqual(detector[0, 3], 0.0)
        self.assertEqual(diagnostics["ae_denominator_zero"], 1)
        self.assertEqual(diagnostics["strength_denominator_zero"], 1)

    def test_rejects_misaligned_sources(self):
        signals = {
            "ae_visual_prompt_sum": np.ones((2, 2)),
            "ae_generation": np.ones((2, 2)),
            "gross_visual": np.ones((2, 2)),
            "gross_prompt": np.ones((2, 3)),
            "gross_generation": np.ones((2, 2)),
        }
        with self.assertRaises(ValueError):
            ratio.build_ratio_features(signals)


if __name__ == "__main__":
    unittest.main()
