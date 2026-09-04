import unittest

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from scripts.train_torch_probe_feature_sets import (
    WeightedMatrixDataset,
    _train_epoch,
)


class WeightedTorchProbeTest(unittest.TestCase):
    def test_weighted_epoch_uses_normalized_sample_weighted_bce(self):
        x = np.asarray([[0.0], [2.0]], dtype=np.float32)
        y = np.asarray([0, 1], dtype=np.int32)
        weights = np.asarray([3.0, 1.0], dtype=np.float32)
        dataset = WeightedMatrixDataset(x, y, weights)
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        loss = _train_epoch(
            model,
            loader,
            optimizer,
            nn.BCEWithLogitsLoss(reduction="none"),
            torch.device("cpu"),
        )
        expected = (
            3.0 * torch.nn.functional.softplus(torch.tensor(0.0)).item()
            + torch.nn.functional.softplus(torch.tensor(-2.0)).item()
        ) / 4.0
        self.assertAlmostEqual(loss, expected, places=6)

    def test_invalid_sample_weights_are_rejected(self):
        x = np.ones((2, 1), dtype=np.float32)
        y = np.asarray([0, 1], dtype=np.int32)
        with self.assertRaises(ValueError):
            WeightedMatrixDataset(x, y, np.asarray([1.0], dtype=np.float32))
        with self.assertRaises(ValueError):
            WeightedMatrixDataset(x, y, np.asarray([1.0, 0.0], dtype=np.float32))
        with self.assertRaises(ValueError):
            WeightedMatrixDataset(x, y, np.asarray([1.0, np.nan], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
