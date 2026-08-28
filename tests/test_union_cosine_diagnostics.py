import unittest

import torch

from features.dgst_t import _topk_union_indices, _union_cosine_geometry_summary


class UnionCosineDiagnosticsTest(unittest.TestCase):
    def test_excludes_diagonal_and_matches_manual_constant_cost(self):
        visual = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
        )
        source = torch.tensor([0.4, 0.3, 0.2, 0.1])
        target = torch.tensor([0.1, 0.2, 0.3, 0.4])
        support = _topk_union_indices(source, target, top_k=8)
        result = _union_cosine_geometry_summary(
            source_dist=source,
            target_dist=target,
            support=support,
            prediction_state=torch.tensor([1.0, 0.0]),
            visual_states=visual,
            transport_top_k=8,
            tau=0.07,
            hist_bins=20,
        )

        self.assertEqual(int(result["support_size"].item()), 4)
        self.assertEqual(int(result["vv_count"].item()), 6)
        self.assertEqual(int(result["qv_count"].item()), 4)
        self.assertAlmostEqual(float(result["vv_mean"].item()), -1.0 / 3.0, places=6)
        self.assertAlmostEqual(float(result["qv_mean"].item()), 0.0, places=6)
        self.assertAlmostEqual(float(result["total_variation"].item()), 0.4, places=6)
        expected_cosine_constant = 0.4 * (4.0 / 3.0)
        self.assertAlmostEqual(
            float(result["constant_risk_cosine"].item()),
            expected_cosine_constant,
            places=6,
        )
        self.assertEqual(int(result["vv_hist"].sum().item()), 6)
        self.assertEqual(int(result["qv_hist"].sum().item()), 4)


if __name__ == "__main__":
    unittest.main()
