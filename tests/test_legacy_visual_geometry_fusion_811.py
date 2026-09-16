import unittest

import numpy as np

from scripts import evaluate_legacy_visual_geometry_fusion_811 as study


class LegacyVisualGeometryFusion811Tests(unittest.TestCase):
    def test_builds_exact_fusions_in_registered_order(self):
        rows, layers = 3, 2
        legacy = np.arange(rows * 4, dtype=np.float32).reshape(rows, 4)
        geometry = {
            name: np.full((rows, layers), index + 10, dtype=np.float32)
            for index, name in enumerate(study.SCALARS)
        }
        groups = study.build_features(legacy, geometry)
        self.assertEqual(tuple(groups), study.TRAINED_GROUPS)
        np.testing.assert_array_equal(groups["legacy_visual"], legacy)
        for output_name, scalar_name in study.FUSION_GROUPS.items():
            np.testing.assert_array_equal(groups[output_name][:, :4], legacy)
            np.testing.assert_array_equal(groups[output_name][:, 4:], geometry[scalar_name])
        expected = np.concatenate([legacy] + [geometry[name] for name in study.SCALARS], axis=1)
        np.testing.assert_array_equal(groups["legacy_plus_five"], expected)

    def test_rejects_bad_rows_and_nonfinite_values(self):
        legacy = np.zeros((3, 4), dtype=np.float32)
        geometry = {name: np.zeros((3, 2), dtype=np.float32) for name in study.SCALARS}
        geometry[study.SCALARS[0]] = np.zeros((2, 2), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "Invalid geometry"):
            study.build_features(legacy, geometry)
        geometry[study.SCALARS[0]] = np.zeros((3, 2), dtype=np.float32)
        legacy[0, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "Invalid legacy"):
            study.build_features(legacy, geometry)


if __name__ == "__main__":
    unittest.main()
