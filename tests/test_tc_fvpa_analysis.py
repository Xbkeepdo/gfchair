import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from features.tc_fvpa_artifacts import (
    ExperimentLayout,
    write_token_map_shard,
)
from scripts.analyze_tc_fvpa_comprehensive import (
    _binary_aupr,
    _binary_auroc,
    _token_and_spatial_rows,
)


class TCFVPAAnalysisTest(unittest.TestCase):
    def test_binary_metrics_have_expected_direction_and_ties(self):
        labels = np.asarray([0, 0, 1, 1], dtype=bool)
        self.assertEqual(_binary_auroc(labels, np.asarray([0, 1, 2, 3])), 1.0)
        self.assertEqual(_binary_aupr(labels, np.asarray([0, 1, 2, 3])), 1.0)
        self.assertEqual(
            _binary_auroc(labels, np.asarray([1, 1, 1, 1])), 0.5
        )
        self.assertIsNone(
            _binary_auroc(np.ones(4, dtype=bool), np.arange(4))
        )

    def test_spatial_rows_and_flat_sample_preserve_signed_scores(self):
        with tempfile.TemporaryDirectory() as temporary:
            layout = ExperimentLayout.create(Path(temporary))
            row = {
                "model": "llava_1_5_7b",
                "image_id": 7,
                "case_id": "llava_1_5_7b:7:2:32",
                "response_index": 2,
                "prediction_position": 10,
                "target_token_id": 99,
                "label_string": "REAL",
                "label_integer": 1,
                "layer": 32,
                "visual_grid": [2, 2],
                "validity_mask": torch.ones(4, dtype=torch.bool),
                "box_overlap": torch.tensor([0.0, 0.1, 0.8, 0.0]),
                "box_overlap_status": "MEASURED",
                "methods": {"LOCAL_LOGIT": torch.tensor([-2.0, 0.0, 3.0, 1.0])},
            }
            write_token_map_shard(
                layout=layout,
                rank=0,
                shard_id=0,
                rows=[row],
                failures=[],
                provenance={"test": True},
            )
            sample, spatial = _token_and_spatial_rows(
                layout, "llava_1_5_7b"
            )
            self.assertEqual(len(sample), 4)
            self.assertEqual(sample[0]["score"], -2.0)
            self.assertEqual(len(spatial), 1)
            self.assertEqual(spatial[0]["top1_pointing"], 1)
            self.assertGreater(spatial[0]["patch_auroc"], 0.5)


if __name__ == "__main__":
    unittest.main()
