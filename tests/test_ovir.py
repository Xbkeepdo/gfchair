import unittest

import numpy as np
import torch

from features.ovir import (
    operator_visual_write_innovation,
    response_from_token_matrix,
    visual_write_energy_basis,
)


class OvirMathTest(unittest.TestCase):
    def test_minimal_raw_energy_rank(self):
        a = torch.diag(torch.tensor([4.0, 2.0, 1.0], dtype=torch.float64))
        q, audit, _ = visual_write_energy_basis(a, energy_threshold=0.95)
        self.assertEqual(audit["rank95"], 2)
        self.assertGreaterEqual(audit["retained_energy"], 0.95)
        self.assertLess(audit["previous_retained_energy"], 0.95)
        np.testing.assert_allclose(audit["retained_energy"], 20 / 21)
        np.testing.assert_allclose(q.T @ q, torch.eye(2), atol=1e-12)

    def test_inside_and_outside_operators(self):
        q = torch.eye(4, dtype=torch.float64)[:, :2]
        inside = q @ torch.tensor([[2.0, 0.5], [0.0, -1.0]], dtype=torch.float64)
        outside = torch.eye(4, dtype=torch.float64)[:, 2:]
        np.testing.assert_allclose(operator_visual_write_innovation(q, inside)["ovir"], 0, atol=1e-14)
        np.testing.assert_allclose(operator_visual_write_innovation(q, outside)["ovir"], 1, atol=1e-14)

    def test_orthogonal_basis_rotation_invariance(self):
        torch.manual_seed(7)
        q = torch.linalg.qr(torch.randn(8, 3, dtype=torch.float64)).Q
        y = torch.randn(8, 3, dtype=torch.float64)
        rotation = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64)).Q
        first = operator_visual_write_innovation(q, y)["ovir"]
        second = operator_visual_write_innovation(q @ rotation, y @ rotation)["ovir"]
        np.testing.assert_allclose(first, second, atol=1e-13)

    def test_operator_energy_identity(self):
        torch.manual_seed(9)
        q = torch.linalg.qr(torch.randn(9, 4, dtype=torch.float64)).Q
        response = torch.randn(9, 4, dtype=torch.float64)
        audit = operator_visual_write_innovation(q, response)
        np.testing.assert_allclose(
            audit["operator_total_energy"],
            audit["operator_inside_energy"] + audit["operator_outside_energy"],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            audit["ovir"],
            audit["operator_outside_energy"] / audit["operator_total_energy"],
            atol=1e-14,
        )
        self.assertLess(audit["operator_energy_identity_error"], 1e-14)

    def test_token_formula_matches_direct_operator_response(self):
        torch.manual_seed(11)
        a = torch.randn(7, 5, dtype=torch.float64)
        a[:, 2:] *= 0.1
        operator = torch.randn(7, 7, dtype=torch.float64)
        q, _, factors = visual_write_energy_basis(a)
        direct = operator @ q
        formula = response_from_token_matrix(operator @ a, factors)
        torch.testing.assert_close(formula, direct, atol=1e-11, rtol=1e-11)
        np.testing.assert_allclose(
            operator_visual_write_innovation(q, formula)["ovir"],
            operator_visual_write_innovation(q, direct)["ovir"],
            atol=1e-13,
        )

    def test_zero_input_and_zero_operator_are_undefined(self):
        q, audit, factors = visual_write_energy_basis(torch.zeros(5, 3, dtype=torch.float64))
        self.assertEqual(tuple(q.shape), (5, 0))
        self.assertTrue(audit["zero_input"])
        self.assertEqual(factors["right"].numel(), 0)
        self.assertTrue(np.isnan(operator_visual_write_innovation(q, q)["ovir"]))
        nonempty = torch.eye(5, dtype=torch.float64)[:, :2]
        metric = operator_visual_write_innovation(nonempty, torch.zeros_like(nonempty))
        self.assertTrue(metric["zero_operator"])
        self.assertTrue(np.isnan(metric["ovir"]))

    def test_invalid_inputs_fail(self):
        with self.assertRaises(ValueError):
            visual_write_energy_basis(torch.ones(3))
        with self.assertRaises(ValueError):
            visual_write_energy_basis(torch.ones(3, 2), energy_threshold=0)
        with self.assertRaises(ValueError):
            operator_visual_write_innovation(torch.ones(3, 2), torch.ones(3, 2))

    def test_resume_partitions_are_disjoint(self):
        from scripts.run_ovir import pending_images
        ids = [1, 2, 3, 4, 5, 6]
        completed = {1, 4}
        even = pending_images(ids, completed, 0)
        odd = pending_images(ids, completed, 1)
        self.assertFalse(set(even) & set(odd))
        self.assertEqual(set(even + odd), set(ids) - completed)
        self.assertEqual(pending_images(ids, completed), [2, 3, 5, 6])


if __name__ == "__main__":
    unittest.main()
