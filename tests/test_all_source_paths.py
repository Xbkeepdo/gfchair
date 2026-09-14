import unittest
import tempfile
from pathlib import Path

import torch

from tests.test_ffn_source_composition import RMS, Gated
from features.ffn_all_source_paths import (partition, path_operator, direct_integral,
    adaptive_b1, compute_target, radial_operator)
from features.ffn_visual_path_attribution import quadrature_rule


class AllSourcePathsTest(unittest.TestCase):
    def test_causal_partition_empty_generation(self):
        mask = partition(8, 4, 0, (1, 3))
        self.assertEqual(mask.sum(1).tolist(), [3, 2, 0])
        self.assertEqual(mask[:, 5:].sum(), 0)
        mask = partition(8, 6, 2, (1, 3))
        self.assertEqual(mask.sum(1).tolist(), [3, 2, 2])
        with self.assertRaises(ValueError): partition(8, 2, 1, (1, 3))

    def test_arbitrary_path_bias_and_direct_jvp(self):
        torch.manual_seed(13)
        norm, ffn = RMS(1e-5).requires_grad_(False), Gated().requires_grad_(False)
        start, delta = torch.randn(2, 3, dtype=torch.float64), torch.randn(2, 3, dtype=torch.float64)
        directions = torch.randn(7, 2, 3, dtype=torch.float64)
        rule = quadrature_rule('gauss_legendre', 32, device='cpu', dtype=torch.float64)
        fast = path_operator(norm, ffn, start, delta, rule.nodes, rule.weights)(directions)
        direct = direct_integral(lambda x: ffn(norm(x)), start, delta, directions, rule.nodes, rule.weights)
        torch.testing.assert_close(fast, direct, rtol=1e-10, atol=1e-11)

    def test_true_residual_zero_and_nonzero_bias(self):
        torch.manual_seed(9)
        norm, ffn = RMS(1e-6).requires_grad_(False), Gated().requires_grad_(False)
        src = torch.randn(4, 1, 3, dtype=torch.float64)*.5
        for bias in (torch.zeros(1, 3, dtype=torch.float64), torch.tensor([[.04, -.03, .02]], dtype=torch.float64)):
            values, radial, stats = adaptive_b1(norm, ffn, src, bias)
            expected = ffn(norm(bias+src.sum(0)))-ffn(norm(bias))
            torch.testing.assert_close(values.sum(0), expected, rtol=1e-4, atol=1e-6)
            torch.testing.assert_close(radial, expected, rtol=1e-4, atol=1e-6)
            self.assertTrue(stats['converged'])

    def test_stable_radial_derivative_retains_epsilon(self):
        torch.manual_seed(42)
        norm, ffn = RMS(1e-6).requires_grad_(False), Gated().requires_grad_(False)
        delta = torch.tensor([[30., 20., -10.]], dtype=torch.float64)
        directions = torch.cat((delta[None], torch.randn(4, 1, 3, dtype=torch.float64)))
        rule = quadrature_rule('gauss_legendre', 16, device='cpu', dtype=torch.float64)
        stable = radial_operator(norm, ffn, delta, rule.nodes, rule.weights)(directions)
        direct = direct_integral(lambda x: ffn(norm(x)), torch.zeros_like(delta), delta,
                                 directions, rule.nodes, rule.weights)
        torch.testing.assert_close(stable, direct, rtol=1e-7, atol=1e-10)
        self.assertGreater(float(stable[0].norm()), 0.)

    def test_zero_path_still_has_nonzero_canceling_sources(self):
        norm, ffn = RMS(1e-5).requires_grad_(False), Gated().requires_grad_(False)
        src = torch.zeros(4, 1, 3, dtype=torch.float64)
        src[0] = 1; src[1] = -1
        values, radial, stats = adaptive_b1(norm, ffn, src, torch.ones(1, 3, dtype=torch.float64))
        torch.testing.assert_close(values.sum(0), torch.zeros_like(radial))
        self.assertGreater(float(values.norm()), 0.)

    def test_full_statistics_and_empty_generation_float32(self):
        torch.manual_seed(20)
        norm, ffn = RMS(1e-6).float().requires_grad_(False), Gated().float().requires_grad_(False)
        writes = torch.randn(8, 1, 3)*.2
        writes[5:] = 0
        residual = torch.randn(1, 3)
        z = residual+writes.sum(0)
        row = compute_target(norm, ffn, z, residual, torch.zeros_like(z), writes,
                             partition(8, 4, 0, (1, 3)), dict(all=16, visual=16, b2=16))
        self.assertNotIn('vectors', row)
        self.assertEqual(len(row['effect_norm']), 5)
        self.assertEqual(row['metrics']['generation_gross_norm'], 0)
        self.assertTrue(torch.isnan(torch.tensor(row['metrics']['cos_vg_out'])))
        self.assertLess(row['diagnostics']['b1']['closure_relative'], .01)
        self.assertLess(row['diagnostics']['b2_quadrature_relative'], .01)

    def test_resume_checks_complete_signature_and_layers(self):
        from scripts.run_ffn_all_source_paths import valid
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'image.pt'
            self.assertFalse(valid(path, 'version'))
            row = dict(complete=True, protocol_signature='version', positions=[dict(
                metrics={'x': torch.ones(2)}, diagnostics=[{}, {}], tokens=[{}, {}])])
            torch.save(row, path)
            self.assertTrue(valid(path, 'version', 2))
            self.assertFalse(valid(path, 'changed', 2))
            self.assertFalse(valid(path, 'version', 3))
            row['complete'] = False
            torch.save(row, path)
            self.assertFalse(valid(path, 'version', 2))


if __name__ == '__main__': unittest.main()
