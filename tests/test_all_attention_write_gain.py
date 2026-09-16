import unittest
import numpy as np
from scripts.analyze_all_attention_write_gain import token_statistics, feature_sets


class WriteGainTest(unittest.TestCase):
    def test_weighted_gain_is_not_token_median(self):
        row = token_statistics([1, 3], [4, 3])
        np.testing.assert_allclose(row[:3], [4, 7, 1.75])
        np.testing.assert_allclose(row[3:7], [1.75, 2.5, 3.25, 1.5])
        self.assertAlmostEqual(row[10], row[9]+row[11])

    def test_zero_and_tiny_sources(self):
        row = token_statistics([0, 1e-20, 2], [0, 2e-20, 2])
        self.assertAlmostEqual(row[4], 1.5)
        empty = token_statistics([], [])
        np.testing.assert_array_equal(empty[:2], [0, 0])
        self.assertTrue(np.isnan(empty[2:]).all())
        with self.assertRaises(ValueError): token_statistics([0], [1])

    def test_fixed_support_and_fusion_layout(self):
        values = np.zeros((2, 2, 4, 12))
        values[:, :, :, 0] = np.arange(1, 5)
        values[:, :, :, 1] = 2*np.arange(1, 5)
        features = feature_sets(values)
        self.assertEqual(features['PVG/I'].shape, (2, 6))
        self.assertEqual(features['ALL/I'].shape, (2, 2))
        np.testing.assert_allclose(features['PVG/I'][0], np.log1p([1, 1, 2, 2, 3, 3]))
        np.testing.assert_array_equal(features['PVG/I+S'], np.concatenate([features['PVG/I'], features['PVG/S']], axis=1))

    def test_uniform_gain_has_no_within_target_dispersion(self):
        row = token_statistics([1, 2, 8], [3, 6, 24])
        np.testing.assert_allclose(row[2:6], [3, 3, 3, 3])
        np.testing.assert_allclose(row[6:9], [0, 0, 1])

    def test_cluster_weights_equal_explicit_image_repetition(self):
        from scripts.summarize_all_attention_write_gain import weighted_scores
        y = np.array([1, 0, 0, 1, 1, 0])
        p = np.array([.8, .4, .4, .4, .9, .2])
        weights = np.array([2, 2, 0, 0, 3, 3])
        indices = np.repeat(np.arange(len(y)), weights)
        np.testing.assert_allclose(weighted_scores(y, p, weights),
                                   weighted_scores(y[indices], p[indices], np.ones(len(indices))))


if __name__ == '__main__': unittest.main()
