import unittest

import torch
from torch import nn

from features.ffn_visual_path_attribution import target_scalar_from_logits
from scripts.run_tc_fvpa_path_attribution import (
    _choose_target_indices,
    final_block_gradients_at_replacement,
)


class TCFVPAPathRunnerTest(unittest.TestCase):
    def test_conflicting_label_position_is_selected_only_once(self):
        groups = {
            9: [{"label": 1}, {"label": 0}],
            17: [{"label": 1}],
            85: [{"label": 0}],
        }

        selected = _choose_target_indices(groups, limit=2)

        self.assertEqual(selected, [9, 17])
        self.assertEqual(len(selected), len(set(selected)))

    def test_exact_final_block_suffix_matches_direct_autograd(self):
        torch.manual_seed(20260829)
        dtype = torch.float64
        z = torch.randn(5, dtype=dtype)
        replacement = torch.randn(5, dtype=dtype)
        norm = nn.LayerNorm(5, dtype=dtype).eval()
        head = nn.Linear(5, 9, bias=True, dtype=dtype).eval()
        scalars = ("log_probability", "margin", "logit")
        target_id = 3
        competitor_id = 6

        gradients, values, logits = final_block_gradients_at_replacement(
            z=z,
            final_norm=norm,
            output_embedding=head,
            target_token_id=target_id,
            competitor_token_id=competitor_id,
            replacement=replacement,
            target_scalars=scalars,
        )

        leaf = replacement.detach().clone().requires_grad_(True)
        expected_logits = head(norm(z + leaf))
        for offset, scalar in enumerate(scalars):
            expected_scalar = target_scalar_from_logits(
                expected_logits.float(),
                target_token_id=target_id,
                competitor_token_id=competitor_id,
                scalar=scalar,
            )
            expected_gradient = torch.autograd.grad(
                expected_scalar,
                leaf,
                retain_graph=offset + 1 < len(scalars),
            )[0]
            self.assertAlmostEqual(
                values[scalar], float(expected_scalar.detach()), places=12
            )
            self.assertTrue(
                torch.allclose(
                    gradients[scalar], expected_gradient, atol=1e-12, rtol=1e-12
                )
            )
        self.assertTrue(
            torch.allclose(logits, expected_logits.detach().float(), atol=0.0, rtol=0.0)
        )


if __name__ == "__main__":
    unittest.main()
