import unittest

import torch
from torch import nn

from features.swiglu_visual_mechanism import (
    finite_swiglu_components,
    local_swiglu_components,
    neuron_target_contributions,
)


class SwiGLUVisualMechanismTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.norm = nn.LayerNorm(5, dtype=torch.float64)
        self.gate = nn.Linear(5, 9, dtype=torch.float64)
        self.up = nn.Linear(5, 9, dtype=torch.float64)
        self.down = nn.Linear(9, 5, dtype=torch.float64)
        self.z = torch.randn(5, dtype=torch.float64)
        self.writes = 0.1 * torch.randn(7, 5, dtype=torch.float64)

    def test_local_product_rule_matches_full_jvp(self):
        result = local_swiglu_components(
            norm=self.norm,
            gate_projection=self.gate,
            up_projection=self.up,
            down_projection=self.down,
            z=self.z,
            writes=self.writes,
        )
        self.assertEqual(tuple(result.gate_path.shape), (7, 9))
        self.assertEqual(tuple(result.output_directions.shape), (7, 5))
        self.assertLess(result.product_rule_relative_error, 2e-13)

    def test_finite_gate_up_interaction_is_complete(self):
        result = finite_swiglu_components(
            norm=self.norm,
            gate_projection=self.gate,
            up_projection=self.up,
            down_projection=self.down,
            z=self.z,
            direction=self.writes.sum(dim=0),
        )
        self.assertLess(result.completeness_relative_error, 2e-13)
        reconstructed = result.gate_path + result.up_path + result.interaction
        self.assertTrue(torch.allclose(reconstructed, result.total_activation_change))

    def test_neuron_contributions_sum_to_output_pairing(self):
        local = local_swiglu_components(
            norm=self.norm,
            gate_projection=self.gate,
            up_projection=self.up,
            down_projection=self.down,
            z=self.z,
            writes=self.writes,
        )
        gradient = torch.randn(5, dtype=torch.float64)
        contributions = neuron_target_contributions(
            local.activation_directions, self.down, gradient
        )
        direct = local.output_directions @ gradient
        self.assertTrue(torch.allclose(contributions.sum(dim=-1), direct, atol=2e-13, rtol=2e-13))


if __name__ == "__main__":
    unittest.main()
