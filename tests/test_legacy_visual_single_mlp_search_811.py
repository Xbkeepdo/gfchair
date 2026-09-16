import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts import search_legacy_visual_single_mlp_811 as legacy
from scripts import search_single_mlp_811 as established


class LegacyVisualSingleMLPSearchTests(unittest.TestCase):
    def test_reuses_established_candidate_budget_and_ranking(self):
        self.assertEqual(legacy.search.candidates(), established.candidates())
        self.assertEqual(len(legacy.search.candidates()), 24)
        self.assertEqual(legacy.TOP_N, 3)
        self.assertEqual(24 + legacy.TOP_N * 2, 30)
        tied = [
            {"index": 2, "validation": {"AUROC": 0.8, "HALL_AUPR": 0.6}},
            {"index": 1, "validation": {"AUROC": 0.8, "HALL_AUPR": 0.6}},
        ]
        self.assertEqual(max(tied, key=legacy.rank)["index"], 1)

    def test_source_validation_rejects_nonexclusive_masks(self):
        fake = {
            "groups": {legacy.GROUP: np.ones((4, 2), dtype=np.float32)},
            "y": np.array([0, 1, 0, 1]),
            "masks": {
                "train": np.array([1, 1, 0, 0], dtype=bool),
                "validation": np.array([0, 1, 1, 0], dtype=bool),
                "test": np.array([0, 0, 0, 1], dtype=bool),
            },
        }
        with patch.object(legacy.search, "read", return_value=fake):
            with self.assertRaisesRegex(ValueError, "Non-exclusive"):
                legacy.load_source("model")

    def test_global_gate_requires_matching_fingerprints(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(legacy, "OUT", root):
                self.assertFalse(legacy.test_gate())
                for index, model in enumerate(legacy.MODELS):
                    target = root / model
                    target.mkdir()
                    fingerprint = str(index)
                    (target / "protocol.json").write_text(
                        json.dumps(
                            {
                                "candidates": legacy.search.candidates(),
                                "fingerprint": fingerprint,
                            }
                        )
                    )
                    (target / "selection.json").write_text(
                        json.dumps({"fingerprint": fingerprint})
                    )
                self.assertTrue(legacy.test_gate())
                (target / "selection.json").write_text(json.dumps({"fingerprint": "wrong"}))
                with self.assertRaisesRegex(ValueError, "Selection mismatch"):
                    legacy.test_gate()


if __name__ == "__main__":
    unittest.main()
