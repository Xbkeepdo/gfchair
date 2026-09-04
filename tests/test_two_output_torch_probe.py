import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.train_dhcp_sampler_risk_ev import dhcp_inverse_frequency_weights
from scripts.train_torch_probe_feature_sets import (
    DGSTStyleProbe,
    TorchProbeConfig,
    train_and_evaluate_probe,
)


class TwoOutputTorchProbeTest(unittest.TestCase):
    def test_model_constructs_two_output_logits(self):
        model = DGSTStyleProbe(3, (4, 2), 0.0, output_dim=2)
        model.eval()
        self.assertEqual(tuple(model(torch.ones(5, 3)).shape), (5, 2))
        self.assertEqual(tuple(model.net[-1].weight.shape), (2, 2))

    def test_two_output_training_uses_cross_entropy_and_saves_two_rows(self):
        rng = np.random.default_rng(11)
        x_train = rng.normal(size=(12, 3)).astype(np.float32)
        y_train = np.asarray([1] * 9 + [0] * 3, dtype=np.int32)
        x_test = rng.normal(size=(6, 3)).astype(np.float32)
        y_test = np.asarray([1, 1, 1, 0, 0, 0], dtype=np.int32)
        weights, _ = dhcp_inverse_frequency_weights(y_train)
        config = TorchProbeConfig(
            hidden_sizes=(4,),
            dropout=0.0,
            batch_size=4,
            num_epochs=1,
            early_stopping_patience=1,
            seed=43,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "probe"
            metrics = train_and_evaluate_probe(
                X_train=x_train,
                y_train=y_train,
                X_val=np.empty((0, 3), dtype=np.float32),
                y_val=np.empty((0,), dtype=np.int32),
                X_test=x_test,
                y_test=y_test,
                config=config,
                device=torch.device("cpu"),
                output_dir=str(output),
                return_probabilities=True,
                train_sampler_weights=weights,
                output_mode="two_logit_cross_entropy",
            )
            state = torch.load(output / "model.pt", map_location="cpu", weights_only=True)
        self.assertEqual(metrics["output_mode"], "two_logit_cross_entropy")
        self.assertEqual(metrics["loss"], "CrossEntropyLoss")
        self.assertEqual(metrics["train_sampling"], "weighted_random_replacement")
        self.assertEqual(len(metrics["test_probabilities"]), 6)
        self.assertEqual(tuple(state["net.4.weight"].shape), (2, 4))

    def test_invalid_output_mode_is_rejected(self):
        x = np.ones((4, 2), dtype=np.float32)
        y = np.asarray([0, 1, 0, 1], dtype=np.int32)
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            train_and_evaluate_probe(
                X_train=x,
                y_train=y,
                X_val=np.empty((0, 2), dtype=np.float32),
                y_val=np.empty((0,), dtype=np.int32),
                X_test=x,
                y_test=y,
                config=TorchProbeConfig(num_epochs=1),
                device=torch.device("cpu"),
                output_dir=directory,
                output_mode="three_logits",
            )


if __name__ == "__main__":
    unittest.main()
