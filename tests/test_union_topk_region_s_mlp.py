import unittest

import numpy as np
import torch

from features.visual_ffn_jacobian import union_topk_aggregate_diagnostics
from scripts.train_union_topk_region_s_mlp import region_ratio, union_topk_mask


class UnionTopKRegionSTest(unittest.TestCase):
    def test_union_support_contains_both_topk_sets(self) -> None:
        source = np.asarray([[0.4, 0.3, 0.2, 0.1]], dtype=np.float32)
        target = np.asarray([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
        mask = union_topk_mask(source, target, side_top_k=1)
        np.testing.assert_array_equal(mask, [[True, False, False, True]])

    def test_region_ratio_is_ratio_of_norm_sums(self) -> None:
        response = np.asarray([[2.0, 20.0, 6.0, 40.0]], dtype=np.float32)
        write = np.asarray([[1.0, 10.0, 2.0, 20.0]], dtype=np.float32)
        mask = np.asarray([[True, False, True, False]])
        ratio, response_sum, write_sum = region_ratio(response, write, mask)
        np.testing.assert_allclose(response_sum, [8.0])
        np.testing.assert_allclose(write_sum, [3.0])
        np.testing.assert_allclose(ratio, [8.0 / 3.0])

    def test_all_token_control_differs_only_by_support(self) -> None:
        response = np.asarray([[2.0, 4.0, 9.0]], dtype=np.float32)
        write = np.asarray([[1.0, 4.0, 3.0]], dtype=np.float32)
        union, _, _ = region_ratio(response, write, [[True, True, False]])
        all_tokens, _, _ = region_ratio(response, write, np.ones_like(response, dtype=bool))
        np.testing.assert_allclose(union, [6.0 / 5.0])
        np.testing.assert_allclose(all_tokens, [15.0 / 8.0])

    def test_support_shape_mismatch_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            union_topk_mask(np.ones((2, 3)), np.ones((2, 4)))

    def test_union_aggregate_preserves_vector_cancellation(self) -> None:
        # Union selects tokens 0 and 2. Their input directions cancel partly,
        # whereas the responses add constructively.
        a_tokens = torch.tensor(
            [[[2.0, 0.0]], [[8.0, 8.0]], [[-1.0, 0.0]], [[9.0, 9.0]]]
        )
        responses = torch.tensor(
            [[[3.0, 0.0]], [[7.0, 7.0]], [[1.0, 0.0]], [[6.0, 6.0]]]
        )
        source = torch.tensor([[0.8, 0.1, 0.05, 0.05]])
        target = torch.tensor([[0.05, 0.05, 0.8, 0.1]])
        result = union_topk_aggregate_diagnostics(
            a_tokens=a_tokens,
            responses=responses,
            source_distribution=source,
            target_distribution=target,
            side_top_k=1,
        )
        self.assertEqual(int(result["union_size"][0]), 2)
        self.assertAlmostEqual(float(result["input_norm"][0]), 1.0)
        self.assertAlmostEqual(float(result["response_norm"][0]), 4.0)
        self.assertAlmostEqual(float(result["gain"][0]), 4.0)
        # The tokenwise norm ratio would be (3+1)/(2+1)=4/3, proving that
        # aggregate S cannot be reconstructed from R_j/I_j alone.
        self.assertNotAlmostEqual(float(result["gain"][0]), 4.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
