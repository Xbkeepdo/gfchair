import json
import tempfile
import unittest
from pathlib import Path

import torch

from features.tc_fvpa_artifacts import (
    ExperimentLayout,
    write_output_checksums,
    write_token_map_shard,
)
from scripts.load_tc_fvpa_results import iter_token_maps, select_token_maps


def token_row(image_id: int, layer: int = 8):
    return {
        "model": "llava_1_5_7b",
        "image_id": image_id,
        "case_id": f"case-{image_id}-{layer}",
        "response_index": 4,
        "prediction_position": 22,
        "target_token_id": 17,
        "label_string": "REAL",
        "label_integer": 1,
        "layer": layer,
        "visual_grid": [2, 2],
        "validity_mask": torch.ones(4, dtype=torch.bool),
        "methods": {
            "WRITE": torch.tensor([1.0, 2.0, 3.0, 4.0]),
            "PATH_LOG_PROBABILITY": torch.tensor([-1.0, 2.0, -3.0, 4.0]),
        },
    }


class LoadTCFVPATest(unittest.TestCase):
    def test_loader_verifies_checksum_selects_and_preserves_negative_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            layout = ExperimentLayout.create(directory)
            write_token_map_shard(
                layout=layout,
                rank=0,
                shard_id=0,
                rows=[token_row(1), token_row(2, layer=16)],
                failures=[],
                provenance={},
            )
            write_output_checksums(layout)
            rows = select_token_maps(
                layout.root,
                model="llava_1_5_7b",
                layer=8,
                method="PATH_LOG_PROBABILITY",
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(torch.as_tensor(rows[0]["scores"]).tolist(), [-1.0, 2.0, -3.0, 4.0])

    def test_loader_rejects_checksum_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            layout = ExperimentLayout.create(directory)
            path = write_token_map_shard(
                layout=layout,
                rank=0,
                shard_id=0,
                rows=[token_row(1)],
                failures=[],
                provenance={},
            )
            write_output_checksums(layout)
            path.write_bytes(path.read_bytes() + b"tamper")
            with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                list(select_token_maps(layout.root))

    def test_loader_rejects_label_direction_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            layout = ExperimentLayout.create(directory)
            row = token_row(1)
            row["label_integer"] = 0
            with self.assertRaisesRegex(ValueError, "label direction"):
                write_token_map_shard(
                    layout=layout,
                    rank=0,
                    shard_id=0,
                    rows=[row],
                    failures=[],
                    provenance={},
                )

    def test_path_cohort_supersedes_duplicate_local_case(self):
        with tempfile.TemporaryDirectory() as directory:
            layout = ExperimentLayout.create(directory)
            path_row = token_row(1)
            path_row["methods"]["PATH_LOGIT"] = torch.arange(4, dtype=torch.float32)
            local_duplicate = token_row(1)
            local_duplicate["methods"] = {"LOCAL_LOGIT": torch.ones(4)}
            local_extra = token_row(2)
            local_extra["methods"] = {"LOCAL_LOGIT": 2.0 * torch.ones(4)}
            write_token_map_shard(
                layout=layout,
                rank=0,
                shard_id=0,
                rows=[path_row],
                failures=[],
                provenance={},
                cohort="path",
            )
            write_token_map_shard(
                layout=layout,
                rank=0,
                shard_id=0,
                rows=[local_duplicate, local_extra],
                failures=[],
                provenance={},
                cohort="local",
            )
            rows = list(iter_token_maps(layout.root, verify_checksums=False))
            self.assertEqual([row["image_id"] for row in rows], [1, 2])
            self.assertIn("PATH_LOGIT", rows[0]["methods"])
            self.assertNotIn("LOCAL_LOGIT", rows[0]["methods"])


if __name__ == "__main__":
    unittest.main()
