import math
import unittest

import torch
from torch import nn

from features.ffn_visual_path_attribution import (
    batched_scalar_gradients,
    batched_ffn_jvps,
    direct_unembedding_scores,
    fixed_clean_competitor,
    frozen_write_leave_one_out,
    local_riesz_attribution,
    quadrature_rule,
    scalar_path_attribution,
    signed_mass_statistics,
    source_scaling_autograd_scores,
    source_scaled_residual,
    streaming_vector_path_statistics,
    symmetric_directional_differences,
    target_scalar_from_logits,
    vector_path_components,
)


DTYPE = torch.float64


class _NonlinearFFN(nn.Module):
    def __init__(self, width: int = 5, intermediate: int = 11) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width, dtype=DTYPE)
        self.gate = nn.Linear(width, intermediate, dtype=DTYPE)
        self.up = nn.Linear(width, intermediate, dtype=DTYPE)
        self.down = nn.Linear(intermediate, width, dtype=DTYPE)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(value)
        return self.down(torch.nn.functional.silu(self.gate(normalized)) * self.up(normalized))


def _case():
    torch.manual_seed(20260829)
    ffn = _NonlinearFFN().eval()
    z = torch.randn(5, dtype=DTYPE)
    writes = 0.12 * torch.randn(7, 5, dtype=DTYPE)
    downstream = nn.Sequential(
        nn.Linear(5, 9, dtype=DTYPE),
        nn.Tanh(),
        nn.Linear(9, 4, dtype=DTYPE),
    ).eval()
    target = 2
    with torch.no_grad():
        clean_logits = downstream(ffn(z))
    competitor = fixed_clean_competitor(clean_logits, target)

    def score(output):
        return target_scalar_from_logits(
            downstream(output),
            target_token_id=target,
            competitor_token_id=competitor,
            scalar="log_probability",
        )

    return ffn, score, z, writes, downstream, target, competitor


class FFNVisualPathAttributionTest(unittest.TestCase):
    def test_batched_scalar_gradients_match_serial_vjps(self) -> None:
        torch.manual_seed(19)
        leaf = torch.randn(4, 5, dtype=DTYPE, requires_grad=True)
        projected = torch.sin(leaf @ torch.randn(5, 7, dtype=DTYPE))
        scalars = {
            "sum": projected.sum(dim=-1),
            "square": projected.square().sum(dim=-1),
            "weighted": projected
            @ torch.linspace(-1.0, 1.0, 7, dtype=DTYPE),
        }
        expected = {}
        names = list(scalars)
        for offset, name in enumerate(names):
            expected[name] = torch.autograd.grad(
                scalars[name].sum(),
                leaf,
                retain_graph=True,
            )[0]
        actual = batched_scalar_gradients(scalars, leaf)
        for name in names:
            self.assertTrue(
                torch.allclose(actual[name], expected[name], atol=1e-13, rtol=1e-13)
            )

    def test_target_scalars_and_fixed_competitor(self) -> None:
        logits = torch.tensor([1.0, 4.0, 3.0, -2.0], dtype=DTYPE)
        self.assertEqual(fixed_clean_competitor(logits, 1), 2)
        self.assertEqual(float(target_scalar_from_logits(logits, target_token_id=1, scalar="logit")), 4.0)
        self.assertEqual(
            float(
                target_scalar_from_logits(
                    logits,
                    target_token_id=1,
                    competitor_token_id=2,
                    scalar="margin",
                )
            ),
            1.0,
        )
        expected = float(torch.log_softmax(logits, -1)[1])
        actual = float(
            target_scalar_from_logits(
                logits, target_token_id=1, scalar="log_probability"
            )
        )
        self.assertAlmostEqual(actual, expected, places=13)

    def test_riesz_duality_additivity_and_explicit_source_scaling(self) -> None:
        ffn, score, z, writes, _downstream, _target, _competitor = _case()
        result = local_riesz_attribution(
            ffn_map=ffn,
            score_from_ffn_output=score,
            z=z,
            writes=writes,
            save_response_vectors=True,
        )
        explicit = source_scaling_autograd_scores(
            ffn_map=ffn,
            score_from_ffn_output=score,
            z=z,
            writes=writes,
        )
        self.assertLess(result.duality_max_abs_error, 2e-13)
        self.assertLess(result.additivity_abs_error, 2e-13)
        self.assertTrue(torch.allclose(result.token_scores, explicit, atol=2e-13, rtol=2e-13))
        self.assertTrue(
            torch.allclose(
                result.direct_pairing_scores,
                result.token_scores,
                atol=2e-13,
                rtol=2e-13,
            )
        )
        scaled = source_scaled_residual(z, writes, [1, 3], 0.25)
        self.assertTrue(
            torch.equal(scaled, z - 0.75 * (writes[1] + writes[3]))
        )

    def test_batched_jvp_matches_finite_difference_for_linear_and_nonlinear_ffn(self) -> None:
        torch.manual_seed(4)
        z = torch.randn(5, dtype=DTYPE)
        writes = torch.randn(6, 5, dtype=DTYPE)
        linear = nn.Linear(5, 5, dtype=DTYPE).eval()
        exact = batched_ffn_jvps(linear, z, writes, chunk_size=2)
        expected = writes @ linear.weight.T
        self.assertTrue(torch.allclose(exact, expected, atol=1e-13, rtol=1e-13))

        nonlinear = _NonlinearFFN().eval()
        exact_nonlinear = batched_ffn_jvps(nonlinear, z, writes)
        eta = 1e-5
        finite = torch.stack(
            [
                (nonlinear(z + eta * direction) - nonlinear(z - eta * direction))
                / (2.0 * eta)
                for direction in writes
            ]
        )
        relative = (finite - exact_nonlinear).norm() / exact_nonlinear.norm()
        self.assertLess(float(relative.detach()), 2e-9)

    def test_vector_and_scalar_path_completeness_converge(self) -> None:
        ffn, score, z, writes, _downstream, _target, _competitor = _case()
        vector_errors = []
        scalar_errors = []
        for count in (4, 8, 16, 32):
            vector = vector_path_components(
                ffn_map=ffn,
                z=z,
                writes=writes,
                method="gauss_legendre",
                integration_points=count,
                chunk_size=3,
            )
            scalar = scalar_path_attribution(
                ffn_map=ffn,
                score_from_ffn_output=score,
                z=z,
                writes=writes,
                method="gauss_legendre",
                integration_points=count,
            )
            vector_errors.append(vector.completeness_relative_error)
            scalar_errors.append(scalar.completeness_relative_error)
        self.assertLess(vector_errors[-1], 2e-12)
        self.assertLess(scalar_errors[-1], 2e-12)
        self.assertLess(vector_errors[-1], vector_errors[0])
        self.assertLess(scalar_errors[-1], scalar_errors[0])

    def test_streaming_vector_statistics_match_full_components(self) -> None:
        torch.manual_seed(31)
        ffn = _NonlinearFFN().eval()
        z = torch.randn(2, 5, dtype=DTYPE)
        writes = 0.08 * torch.randn(7, 2, 5, dtype=DTYPE)
        result = streaming_vector_path_statistics(
            ffn_map=ffn,
            z=z,
            writes=writes,
            method="gauss_legendre",
            integration_points=16,
            token_chunk_size=3,
            save_components=True,
        )
        expected = torch.stack(
            [
                vector_path_components(
                    ffn_map=ffn,
                    z=z[target],
                    writes=writes[:, target],
                    method="gauss_legendre",
                    integration_points=16,
                    chunk_size=3,
                ).components
                for target in range(2)
            ],
            dim=1,
        )
        self.assertTrue(torch.allclose(result.components, expected, atol=2e-13, rtol=2e-13))
        self.assertTrue(torch.allclose(result.ffn_path_gross, expected.norm(dim=-1).T))
        self.assertTrue(torch.allclose(result.p_ffn.sum(dim=-1), torch.ones(2, dtype=DTYPE)))
        self.assertLess(float(result.completeness_relative_error.max()), 2e-12)

        unchunked = streaming_vector_path_statistics(
            ffn_map=ffn,
            z=z,
            writes=writes,
            method="gauss_legendre",
            integration_points=16,
        )
        self.assertIsNone(unchunked.components)
        self.assertTrue(torch.allclose(unchunked.ffn_path_gross, result.ffn_path_gross))
        self.assertTrue(torch.allclose(unchunked.path_signed_q, result.path_signed_q))

    def test_streaming_vector_statistics_zero_strength_is_finite(self) -> None:
        ffn = nn.Linear(5, 5, bias=False, dtype=DTYPE).eval()
        z = torch.randn(3, 5, dtype=DTYPE)
        writes = torch.zeros(4, 3, 5, dtype=DTYPE)
        result = streaming_vector_path_statistics(
            ffn_map=ffn,
            z=z,
            writes=writes,
            method="gauss_legendre",
            integration_points=4,
            token_chunk_size=2,
        )
        self.assertTrue(bool(result.gross_degenerate.all()))
        self.assertTrue(bool(result.net_degenerate.all()))
        self.assertTrue(torch.isfinite(result.p_ffn).all())
        self.assertTrue(torch.equal(result.p_ffn, torch.full_like(result.p_ffn, 0.25)))
        self.assertEqual(float(result.path_signed_q.abs().sum()), 0.0)
        self.assertEqual(float(result.kappa.abs().sum()), 0.0)

    def test_trapezoid_and_gauss_legendre_agree_at_convergence(self) -> None:
        ffn, score, z, writes, _downstream, _target, _competitor = _case()
        trapezoid = scalar_path_attribution(
            ffn_map=ffn,
            score_from_ffn_output=score,
            z=z,
            writes=writes,
            method="trapezoid",
            integration_points=32,
        )
        gauss = scalar_path_attribution(
            ffn_map=ffn,
            score_from_ffn_output=score,
            z=z,
            writes=writes,
            method="gauss_legendre",
            integration_points=32,
        )
        self.assertLess(
            float(
                (trapezoid.token_scores - gauss.token_scores)
                .abs()
                .max()
                .detach()
            ),
            2e-5,
        )
        self.assertLess(trapezoid.completeness_relative_error, 2e-4)
        self.assertLess(gauss.completeness_relative_error, 2e-12)

    def test_k1_is_clean_endpoint_local_jacobian(self) -> None:
        ffn, score, z, writes, _downstream, _target, _competitor = _case()
        local = local_riesz_attribution(
            ffn_map=ffn,
            score_from_ffn_output=score,
            z=z,
            writes=writes,
        )
        for method in ("trapezoid", "gauss_legendre"):
            path = scalar_path_attribution(
                ffn_map=ffn,
                score_from_ffn_output=score,
                z=z,
                writes=writes,
                method=method,
                integration_points=1,
            )
            self.assertTrue(path.quadrature.is_local)
            self.assertTrue(torch.allclose(path.token_scores, local.token_scores, atol=2e-13, rtol=2e-13))
        rule = quadrature_rule("gauss_legendre", 1, device=z.device, dtype=z.dtype)
        self.assertEqual(float(rule.nodes[0]), 1.0)

    def test_frozen_write_loo_and_direct_unembedding(self) -> None:
        ffn, score, z, writes, _downstream, _target, _competitor = _case()
        vector, scalar = frozen_write_leave_one_out(
            ffn_map=ffn,
            score_from_ffn_output=score,
            z=z,
            writes=writes,
            regions=[[0], [1, 2]],
        )
        self.assertEqual(tuple(vector.shape), (2, 5))
        self.assertEqual(tuple(scalar.shape), (2,))
        expected = ffn(z) - ffn(z - writes[1:3].sum(0))
        self.assertTrue(torch.allclose(vector[1], expected))
        direction = torch.randn(5, dtype=DTYPE)
        responses = batched_ffn_jvps(ffn, z, writes)
        self.assertTrue(torch.allclose(direct_unembedding_scores(responses, direction), responses @ direction))

    def test_symmetric_difference_converges_and_reports_curvature(self) -> None:
        torch.manual_seed(9)
        center = torch.randn(5, dtype=DTYPE)
        direction = torch.randn(5, dtype=DTYPE)

        def scalar(value):
            return torch.sin(value).sum() + 0.2 * value.square().sum()

        predicted = torch.dot(torch.cos(center) + 0.4 * center, direction)
        rows = symmetric_directional_differences(
            scalar_function=scalar,
            center=center,
            direction=direction,
            predicted_derivative=predicted,
            etas=(0.2, 0.1, 0.05, 0.025),
        )
        self.assertLess(rows[-1].relative_linearization_error, rows[0].relative_linearization_error)
        self.assertTrue(all(row.curvature >= 0.0 for row in rows))
        self.assertTrue(all(not row.zero_observed for row in rows))

    def test_coordinate_reparameterization_preserves_directional_pairing(self) -> None:
        torch.manual_seed(11)
        dimension = 6
        matrix = torch.randn(dimension, dimension, dtype=DTYPE)
        matrix = matrix + 2.0 * torch.eye(dimension, dtype=DTYPE)
        inverse = torch.linalg.inv(matrix)
        response = torch.randn(dimension, dtype=DTYPE)
        covector = torch.randn(dimension, dtype=DTYPE)
        original = torch.dot(covector, response)

        # x' = Mx; tangent transforms by M and the Riesz/covector coordinate
        # representation by M^{-T}.  Their directional pairing is invariant.
        transformed_response = matrix @ response
        transformed_covector = inverse.T @ covector
        transformed = torch.dot(transformed_covector, transformed_response)
        self.assertTrue(torch.allclose(original, transformed, atol=2e-12, rtol=2e-12))

    def test_negative_scores_are_preserved(self) -> None:
        scores = torch.tensor([3.0, -2.0, 1.0, -0.5], dtype=DTYPE)
        stats = signed_mass_statistics(scores)
        self.assertEqual(stats["positive_mass"], 4.0)
        self.assertEqual(stats["negative_mass"], 2.5)
        self.assertEqual(stats["net_effect"], 1.5)
        self.assertEqual(stats["negative_token_fraction"], 0.5)
        self.assertAlmostEqual(stats["signed_cancellation_ratio"], 1.5 / 6.5)


if __name__ == "__main__":
    unittest.main()
