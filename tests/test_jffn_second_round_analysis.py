import unittest

import torch

from scripts.analyze_jffn_second_round_token_maps import (
    batch_rank_metrics,
    matched_distribution,
    q_positive_distribution,
    spatial_metrics,
    spatial_metrics_batch,
)


class JffnSecondRoundAnalysisTest(unittest.TestCase):
    def test_batched_spatial_metrics_match_scalar_formula(self) -> None:
        generator = torch.Generator().manual_seed(7)
        distributions = torch.rand((3, 4, 25), generator=generator)
        distributions /= distributions.sum(dim=-1, keepdim=True)
        overlap = torch.zeros(25)
        overlap[3:11] = torch.linspace(0.1, 1.0, 8)
        batched = spatial_metrics_batch(distributions, overlap)
        for method in range(3):
            for layer in range(4):
                scalar = spatial_metrics(distributions[method, layer], overlap)
                for name, value in scalar.items():
                    self.assertAlmostEqual(
                        value, float(batched[name][method, layer]), places=6
                    )

    def test_entropy_matching_preserves_ranking(self) -> None:
        energy = torch.tensor([[0.2, 3.0, 1.0], [5.0, 2.0, 4.0]])
        matched = matched_distribution(energy, [0.5, 2.0])
        self.assertTrue(torch.equal(energy.argsort(-1), matched.argsort(-1)))
        self.assertTrue(torch.allclose(matched.sum(-1), torch.ones(2)))

    def test_positive_q_is_a_distribution(self) -> None:
        q = torch.tensor([[1.0, -2.0, 3.0], [-1.0, -2.0, -3.0]])
        positive = q_positive_distribution(q)
        self.assertTrue(torch.allclose(positive.sum(-1), torch.ones(2)))
        self.assertTrue(torch.allclose(positive[0], torch.tensor([0.25, 0.0, 0.75])))
        self.assertTrue(torch.allclose(positive[1], torch.full((3,), 1.0 / 3.0)))

    def test_rank_metrics_identical_maps(self) -> None:
        values = torch.arange(20, dtype=torch.float32).reshape(2, 10)
        metrics = batch_rank_metrics(values, values)
        for value in metrics.values():
            self.assertTrue(torch.allclose(value, torch.ones_like(value)))


if __name__ == "__main__":
    unittest.main()
