import unittest

import torch

from features.ffn_visual_interactions import (
    aggregate_region_writes,
    regular_grid_regions,
    sampled_shapley,
)


class FFNVisualInteractionsTest(unittest.TestCase):
    def test_regular_regions_cover_each_token_exactly_once(self):
        regions = regular_grid_regions(8, 12, 8)
        flattened = [value for region in regions for value in region]
        self.assertEqual(sorted(flattened), list(range(96)))
        self.assertEqual(len(flattened), len(set(flattened)))

    def test_region_write_aggregation_rejects_overlap(self):
        writes = torch.arange(24, dtype=torch.float64).reshape(6, 4)
        result = aggregate_region_writes(writes, [[0, 1], [2, 3, 4]])
        self.assertTrue(torch.equal(result[0], writes[0] + writes[1]))
        with self.assertRaisesRegex(ValueError, "overlap"):
            aggregate_region_writes(writes, [[0, 1], [1, 2]])

    def test_sampled_shapley_is_exact_for_additive_game(self):
        weights = torch.tensor([1.0, -2.0, 3.5, 0.25], dtype=torch.float64)

        def game(mask):
            return weights[mask].sum()

        result = sampled_shapley(
            value_from_active_mask=game,
            region_count=4,
            permutations=128,
            seed=7,
            persist_every=16,
        )
        self.assertTrue(torch.allclose(result.values, weights))
        self.assertLess(result.completeness_absolute_error, 1e-12)
        self.assertEqual(tuple(result.running_estimates.shape), (8, 4))

    def test_sampled_shapley_completeness_for_interaction_game(self):
        def game(mask):
            values = mask.to(torch.float64)
            return values[0] + 2 * values[1] + 4 * values[0] * values[1]

        result = sampled_shapley(
            value_from_active_mask=game,
            region_count=3,
            permutations=512,
            seed=11,
            persist_every=128,
        )
        self.assertLess(result.completeness_absolute_error, 1e-12)
        self.assertAlmostEqual(float(result.values[2]), 0.0, places=12)
        self.assertLess(abs(float(result.values[0] - 3.0)), 0.2)
        self.assertLess(abs(float(result.values[1] - 4.0)), 0.2)


if __name__ == "__main__":
    unittest.main()
