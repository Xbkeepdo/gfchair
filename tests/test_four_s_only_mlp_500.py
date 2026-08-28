import unittest

import numpy as np

from scripts.train_four_s_only_mlp_500 import build_s_only_features


class FourSOnlyFeatureTest(unittest.TestCase):
    def test_each_probe_receives_only_its_own_32d_s_block(self):
        matrices = {}
        aggregate = {}
        for split, rows in (("train", 3), ("test", 2)):
            matrices[split] = {
                "y": np.arange(rows) % 2,
                "old_aggregate_s": np.full((rows, 32), 1.0),
                "token_all_s": np.full((rows, 32), 2.0),
                "union_topk_s": np.full((rows, 32), 3.0),
            }
            aggregate[split] = {"gain": np.full((rows, 32), 4.0)}
        features = build_s_only_features(matrices, aggregate)
        expected = {
            "all_aggregate_s": 1.0,
            "all_tokenwise_s": 2.0,
            "union_tokenwise_s": 3.0,
            "union_aggregate_s": 4.0,
        }
        for split in ("train", "test"):
            for name, value in expected.items():
                self.assertEqual(features[split][name].shape[1], 32)
                np.testing.assert_array_equal(features[split][name], value)

    def test_wrong_width_is_rejected(self):
        matrices = {
            split: {
                "y": np.asarray([0, 1]),
                "old_aggregate_s": np.ones((2, 31)),
                "token_all_s": np.ones((2, 32)),
                "union_topk_s": np.ones((2, 32)),
            }
            for split in ("train", "test")
        }
        aggregate = {
            split: {"gain": np.ones((2, 32))} for split in ("train", "test")
        }
        with self.assertRaises(AssertionError):
            build_s_only_features(matrices, aggregate)


if __name__ == "__main__":
    unittest.main()
