import unittest

import numpy as np

from scripts import analyze_all_source_paths as source_schema
from scripts import evaluate_selected_geometry_scalars_811 as study


class SelectedGeometryScalars811Tests(unittest.TestCase):
    def test_extracts_exact_blocks_and_builds_controls(self):
        layers, mentions = 3, [dict(response_index=value) for value in (0, 1, 4, 9)]
        f2 = np.concatenate([
            np.full((4, layers), index, dtype=np.float32)
            for index in range(len(source_schema.F2_FIELDS))
        ], axis=1)
        f4 = np.concatenate([
            np.full((4, layers), 100 + index, dtype=np.float32)
            for index in range(len(source_schema.F4_FIELDS))
        ], axis=1)
        groups, position = study.build_features(dict(F2=f2, b1_F4=f4), mentions, layers)
        np.testing.assert_array_equal(
            groups["prompt_within_source_kappa"],
            source_schema.F2_FIELDS.index("prompt_within_source_kappa"),
        )
        np.testing.assert_array_equal(
            groups["generation_within_source_kappa"],
            source_schema.F2_FIELDS.index("generation_within_source_kappa"),
        )
        np.testing.assert_array_equal(
            groups["b1_residual_visual_generation_balance"],
            100 + source_schema.F4_FIELDS.index("residual_visual_generation_balance"),
        )
        np.testing.assert_array_equal(
            groups["b1_delta_residual_generation_cos"],
            100 + source_schema.F4_FIELDS.index("delta_residual_generation_cos"),
        )
        np.testing.assert_allclose(position[:, 0], np.log1p([0, 1, 4, 9]))
        self.assertEqual(groups["five_concat"].shape, (4, 5 * layers))
        self.assertEqual(groups["generation_kappa_plus_position"].shape, (4, layers + 1))

    def test_rejects_bad_layout_and_nonfinite_geometry(self):
        layers = 2
        f2 = np.zeros((1, len(source_schema.F2_FIELDS) * layers), dtype=np.float32)
        f4 = np.zeros((1, len(source_schema.F4_FIELDS) * layers), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "Unexpected matrix shape"):
            study.build_features(dict(F2=f2[:, :-1], b1_F4=f4), [dict(response_index=1)], layers)
        selected = source_schema.F2_FIELDS.index("prompt_within_source_kappa") * layers
        f2[0, selected] = np.inf
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            study.build_features(dict(F2=f2, b1_F4=f4), [dict(response_index=1)], layers)


if __name__ == "__main__":
    unittest.main()
