import tempfile
import unittest
from pathlib import Path

import torch

from features.jffn_experiment import (
    atomic_torch_save,
    assert_finite_position_rows,
    build_all_training_matrices_from_shards,
    build_training_matrix,
    feature_specs,
    fit_entropy_betas,
    load_shards,
    mentions_for_image,
    normalized_entropy,
    select_train_calibration_images,
)
from scripts.run_jffn_p_comparison import (
    patch_overlap_fraction,
    rectangle_union_area,
)


class JFFNExperimentTest(unittest.TestCase):
    def test_mentions_keep_official_rows_but_deduplicate_forward_positions(self):
        labeling = {
            "object_token_spans": [
                {"word": "car", "canonical_object": "car", "token_indices": [2], "label": 1},
                {"word": "cars", "canonical_object": "car", "token_indices": [2], "label": 1},
                {"word": "dog", "canonical_object": "dog", "token_indices": [4], "label": 0},
            ]
        }
        mentions, indices, target_ids = mentions_for_image(
            image_id=9,
            labeling_row=labeling,
            response_token_ids=[10, 11, 12, 13, 14],
        )
        self.assertEqual(len(mentions), 3)
        self.assertEqual(indices, [2, 4])
        self.assertEqual(target_ids, [12, 14])
        self.assertEqual(mentions[0]["target_key"], mentions[1]["target_key"])

    def test_entropy_beta_matches_target_and_preserves_train_only_selection(self):
        energies = [[torch.tensor([[1.0, 2.0, 7.0]]), torch.tensor([[1.0, 3.0]])]]
        target = 0.73
        betas, audit = fit_entropy_betas(
            energies_by_layer=energies,
            target_entropy_by_layer=[target],
            beta_min=0.02,
            beta_max=50.0,
            steps=50,
        )
        self.assertGreater(betas[0], 0.0)
        self.assertAlmostEqual(audit[0]["achieved_normalized_entropy"], target, places=5)
        selected = select_train_calibration_images(
            train_image_ids=[1, 2, 3, 4],
            available_image_ids={2, 3, 4, 99},
            count=2,
            seed=7,
        )
        self.assertTrue(set(selected) <= {2, 3, 4})
        self.assertNotIn(99, selected)

    def test_feature_factorial_has_32_sets(self):
        specs = feature_specs()
        self.assertEqual(len(specs), 32)
        self.assertEqual(len({row["name"] for row in specs}), 32)

    def test_atomic_shards_stream_and_training_expands_mentions(self):
        risk = torch.tensor([0.1, 0.2])
        ev = torch.tensor([0.3, 0.4])
        position = {
            "target_key": "1:2",
            "risks": {
                "old_hmid_cos": {
                    "hpre_raw_logit_gauss": {
                        "sqrt_matched_state": risk,
                    }
                }
            },
            "ev": {"hpre_raw_logit_gauss": ev},
        }
        mentions = [
            {"mention_id": "1:0", "target_key": "1:2", "image_id": 1, "label": 1},
            {"mention_id": "1:1", "target_key": "1:2", "image_id": 1, "label": 0},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features_rank00_shard_00000.pt"
            atomic_torch_save(
                {"image_ids": [1], "positions": [position], "sample_table": mentions},
                path,
            )
            atomic_torch_save(
                {"image_ids": [2], "positions": [], "sample_table": []},
                Path(directory) / "features_rank01_shard_00000.pt",
            )
            self.assertEqual(len(list(load_shards(Path(directory)))), 2)
        matrix, labels, image_ids, mention_ids = build_training_matrix(
            positions={"1:2": position},
            mentions=mentions,
            spec={
                "source": "old_hmid_cos",
                "gate": "hpre_raw_logit_gauss",
                "cost": "sqrt_matched_state",
                "feature": "risk_ev",
            },
            image_ids={1},
        )
        self.assertEqual(matrix.shape, (2, 4))
        self.assertEqual(labels.tolist(), [1, 0])
        self.assertEqual(image_ids.tolist(), [1, 1])
        self.assertEqual(mention_ids, ["1:0", "1:1"])

    def test_bbox_union_and_llava_center_crop_mapping(self):
        self.assertAlmostEqual(
            rectangle_union_area([(0.0, 0.0, 0.75, 1.0), (0.5, 0.0, 1.0, 1.0)]),
            1.0,
            places=7,
        )
        # 640x480 -> 448x336, then center-crop x=[56,392].  This original
        # box maps exactly onto the left half of the 24x24 crop.
        overlap = patch_overlap_fraction(
            model="llava_1_5_7b",
            image_size=(640, 480),
            boxes=[(80.0, 0.0, 320.0, 480.0)],
            grid=(24, 24),
        ).reshape(24, 24)
        self.assertTrue(torch.allclose(overlap[:, :12], torch.ones(24, 12)))
        self.assertTrue(torch.allclose(overlap[:, 12:], torch.zeros(24, 12)))

    def test_streaming_training_matrix_reads_two_rank_shards(self):
        spec = {
            "name": "one",
            "source": "old_hmid_cos",
            "gate": "hpre_raw_logit_gauss",
            "cost": "sqrt_matched_state",
            "feature": "risk_ev",
        }

        def payload(image_id, label):
            key = f"{image_id}:2"
            return {
                "image_ids": [image_id],
                "positions": [
                    {
                        "target_key": key,
                        "risks": {
                            "old_hmid_cos": {
                                "hpre_raw_logit_gauss": {
                                    "sqrt_matched_state": torch.tensor([0.1, 0.2])
                                }
                            }
                        },
                        "ev": {
                            "hpre_raw_logit_gauss": torch.tensor([0.3, 0.4])
                        },
                    }
                ],
                "sample_table": [
                    {
                        "mention_id": f"{image_id}:0",
                        "target_key": key,
                        "image_id": image_id,
                        "label": label,
                    }
                ],
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atomic_torch_save(payload(1, 1), root / "features_rank00_shard_00000.pt")
            atomic_torch_save(payload(2, 0), root / "features_rank01_shard_00000.pt")
            matrices = build_all_training_matrices_from_shards(
                shard_dir=root,
                specs=[spec],
                train_image_ids={1},
                test_image_ids={2},
            )
        self.assertEqual(matrices["one"]["train"][0].shape, (1, 4))
        self.assertEqual(matrices["one"]["test"][0].shape, (1, 4))
        self.assertEqual(matrices["one"]["train"][1].tolist(), [1])
        self.assertEqual(matrices["one"]["test"][1].tolist(), [0])

    def test_nonfinite_shard_row_is_rejected(self):
        assert_finite_position_rows([{"value": torch.tensor([1.0, 2.0])}])
        with self.assertRaisesRegex(ValueError, "positions\\[0\\].value"):
            assert_finite_position_rows([{"value": torch.tensor([float("nan")])}])


if __name__ == "__main__":
    unittest.main()
