import unittest

import numpy as np
import torch

from features.semantic_attention_topk import topk_features
from scripts.train_semantic_attention_all_visual_811 import full_visual_features, matrix


class AllVisualFeatureTest(unittest.TestCase):
    def test_matches_topk_when_k_is_all_visual_tokens(self):
        rng = np.random.default_rng(7)
        probability = rng.random((4, 37), dtype=np.float32)
        attention = rng.random((4, 37), dtype=np.float32)
        cosine = rng.uniform(-1, 1, (4, 37)).astype(np.float32)
        actual = full_visual_features(probability, attention, cosine)
        _, expected = topk_features(
            torch.from_numpy(probability), torch.from_numpy(attention),
            torch.from_numpy(cosine), 37,
        )
        np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-5, atol=1e-6)

    def test_single_and_all_six_feature_layout(self):
        cube = np.arange(2 * 3 * 6, dtype=np.float32).reshape(2, 3, 6)
        log_s = np.ones((2, 3), dtype=np.float32)
        data = {"cubes": {"raw": cube}, "log1p_S": log_s}
        np.testing.assert_array_equal(matrix(data, "standalone", "raw", "attention_only"), cube[:, :, 1])
        np.testing.assert_array_equal(matrix(data, "plus_logS", "raw", "attention_only"),
                                      np.concatenate((cube[:, :, 1], log_s), axis=1))
        np.testing.assert_array_equal(matrix(data, "standalone", "raw", "all_six"), cube.reshape(2, -1))


if __name__ == "__main__":
    unittest.main()
