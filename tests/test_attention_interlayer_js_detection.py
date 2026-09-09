import unittest

import torch

from scripts.train_attention_interlayer_js_probes import FEATURE_SETS, block_features


class AttentionInterlayerJSDetectionTest(unittest.TestCase):
    def test_block_dimensions_and_signal_identity(self) -> None:
        attention = torch.tensor(
            [
                [
                    [0.7, 0.2, 0.1],
                    [0.1, 0.7, 0.2],
                    [0.2, 0.1, 0.7],
                ]
            ]
        )
        blocks = block_features(attention, attention.clone(), top_k=1)
        self.assertEqual(set(blocks), {
            "adj_A_all", "adj_A_top32", "adj_E_all", "adj_E_top32",
        })
        self.assertEqual(tuple(blocks["adj_A_all"].shape), (1, 2))
        torch.testing.assert_close(blocks["adj_A_all"], blocks["adj_E_all"])
        torch.testing.assert_close(blocks["adj_A_top32"], blocks["adj_E_top32"])
        self.assertEqual(set(FEATURE_SETS), set(blocks))


if __name__ == "__main__":
    unittest.main()
