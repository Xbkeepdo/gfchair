import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.train_dhcp_sampler_risk_ev import dhcp_inverse_frequency_weights
from scripts.train_torch_probe_feature_sets import (
    TorchProbeConfig,
    train_and_evaluate_probe,
)


class DHCPSamplerRiskEVTest(unittest.TestCase):
    def test_inverse_frequency_weights_equalize_total_class_mass(self):
        labels = np.asarray([1, 1, 1, 1, 0], dtype=np.int32)
        weights, audit = dhcp_inverse_frequency_weights(labels)
        np.testing.assert_allclose(weights, [0.25, 0.25, 0.25, 0.25, 1.0])
        self.assertAlmostEqual(weights[labels == 1].sum(), 1.0)
        self.assertAlmostEqual(weights[labels == 0].sum(), 1.0)
        self.assertEqual(audit["hall_to_real_row_weight_ratio"], 4.0)
        self.assertEqual(audit["expected_class_fraction"], {"real": 0.5, "hall": 0.5})
        self.assertTrue(audit["replacement"])

    def test_nonbinary_or_single_class_labels_are_rejected(self):
        for labels in (
            np.asarray([1, 1], dtype=np.int32),
            np.asarray([0, 0], dtype=np.int32),
            np.asarray([0, 1, 2], dtype=np.int32),
        ):
            with self.assertRaises(ValueError):
                dhcp_inverse_frequency_weights(labels)

    def test_probe_records_weighted_replacement_sampling_without_loss_weight(self):
        rng = np.random.default_rng(7)
        x_train = rng.normal(size=(12, 3)).astype(np.float32)
        y_train = np.asarray([1] * 9 + [0] * 3, dtype=np.int32)
        x_test = rng.normal(size=(6, 3)).astype(np.float32)
        y_test = np.asarray([1, 1, 1, 0, 0, 0], dtype=np.int32)
        sampler_weights, _ = dhcp_inverse_frequency_weights(y_train)
        config = TorchProbeConfig(
            hidden_sizes=(4,),
            dropout=0.0,
            drop_last=False,
            batch_size=4,
            num_epochs=1,
            early_stopping_patience=1,
            seed=43,
        )
        with tempfile.TemporaryDirectory() as directory:
            metrics = train_and_evaluate_probe(
                X_train=x_train,
                y_train=y_train,
                X_val=np.empty((0, 3), dtype=np.float32),
                y_val=np.empty((0,), dtype=np.int32),
                X_test=x_test,
                y_test=y_test,
                config=config,
                device=torch.device("cpu"),
                output_dir=str(Path(directory) / "probe"),
                train_sampler_weights=sampler_weights,
            )
        self.assertEqual(metrics["train_sample_weighting"], "none")
        self.assertEqual(metrics["train_sampling"], "weighted_random_replacement")
        self.assertTrue(metrics["train_sampler_weight_summary"]["replacement"])
        self.assertEqual(
            metrics["train_sampler_weight_summary"]["num_samples_per_epoch"], 12
        )

    def test_sampler_and_loss_weighting_cannot_be_combined(self):
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
                train_sample_weights=np.ones(4, dtype=np.float32),
                train_sampler_weights=np.ones(4, dtype=np.float32),
            )


if __name__ == "__main__":
    unittest.main()
