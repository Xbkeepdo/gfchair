import unittest

import torch

from features.representation_geometry_controls import (
    empirical_cdf_scores,
    fit_whitening,
    mahalanobis_cosine,
    residualize_direction,
    row_cosine,
    transform_covector,
)


class RepresentationGeometryControlsTest(unittest.TestCase):
    def test_raw_cosine_is_rotation_and_positive_scale_invariant(self):
        torch.manual_seed(5)
        left = torch.randn(20, 6, dtype=torch.float64)
        right = torch.randn(20, 6, dtype=torch.float64)
        q, _ = torch.linalg.qr(torch.randn(6, 6, dtype=torch.float64))
        original = row_cosine(left, right)
        rotated = row_cosine(left @ q, right @ q)
        scaled = row_cosine(3.0 * left, 0.2 * right)
        self.assertTrue(torch.allclose(original, rotated, atol=2e-13, rtol=2e-13))
        self.assertTrue(torch.allclose(original, scaled, atol=2e-13, rtol=2e-13))

    def test_raw_cosine_is_not_invariant_to_anisotropic_transform(self):
        left = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
        right = torch.tensor([[1.0, -0.5]], dtype=torch.float64)
        transform = torch.diag(torch.tensor([8.0, 0.2], dtype=torch.float64))
        original = row_cosine(left, right)
        changed = row_cosine(left @ transform, right @ transform)
        self.assertGreater(float((original - changed).abs()), 0.1)

    def test_whitened_cosine_restores_consistent_affine_calibration(self):
        torch.manual_seed(7)
        train = torch.randn(500, 5, dtype=torch.float64)
        left = torch.randn(40, 5, dtype=torch.float64)
        right = torch.randn(40, 5, dtype=torch.float64)
        transform = torch.randn(5, 5, dtype=torch.float64) + 3 * torch.eye(5, dtype=torch.float64)
        base = mahalanobis_cosine(left, right, fit_whitening(train, 1e-9))
        transformed = mahalanobis_cosine(
            left @ transform,
            right @ transform,
            fit_whitening(train @ transform, 1e-9),
        )
        self.assertTrue(torch.allclose(base, transformed, atol=2e-8, rtol=2e-8))

    def test_residualization_removes_exactly_one_direction(self):
        torch.manual_seed(11)
        values = torch.randn(30, 4, dtype=torch.float64)
        direction = torch.randn(4, dtype=torch.float64)
        residual = residualize_direction(values, direction)
        self.assertLess(float((residual @ direction).abs().max()), 2e-13)

    def test_directional_pairing_survives_coordinate_change(self):
        torch.manual_seed(13)
        tangent = torch.randn(7, dtype=torch.float64)
        covector = torch.randn(7, dtype=torch.float64)
        transform = torch.randn(7, 7, dtype=torch.float64) + 4 * torch.eye(7, dtype=torch.float64)
        transformed = torch.dot(
            transform_covector(covector, transform), transform @ tangent
        )
        self.assertTrue(torch.allclose(torch.dot(covector, tangent), transformed, atol=2e-12, rtol=2e-12))

    def test_empirical_cdf_uses_midrank_for_ties(self):
        reference = torch.tensor([0.0, 1.0, 1.0, 3.0])
        scores = empirical_cdf_scores(torch.tensor([1.0, 2.0]), reference)
        self.assertTrue(torch.allclose(scores, torch.tensor([0.5, 0.75])))


if __name__ == "__main__":
    unittest.main()
