import unittest

import numpy as np

from scripts.train_s_times_risk_ev_mlp import curve_statistics


class STimesRiskCurveTests(unittest.TestCase):
    def test_curve_statistics_preserve_hall_minus_real_direction(self) -> None:
        curves = {
            "labels": np.asarray([1, 1, 0, 0], dtype=np.int32),
            "s": np.asarray([[1, 2], [2, 3], [4, 5], [5, 7]], dtype=np.float32),
            "s_times_risk": np.asarray(
                [[0.1, 0.2], [0.2, 0.3], [0.8, 1.0], [1.0, 1.4]],
                dtype=np.float32,
            ),
        }
        label_rows, differences = curve_statistics(curves)
        self.assertEqual(len(label_rows), 8)
        self.assertEqual(len(differences), 4)
        for row in differences:
            self.assertGreater(row["hall_minus_real"], 0.0)
            self.assertGreater(row["cohens_d_hall_minus_real"], 0.0)
            self.assertGreater(row["hall_positive_raw_auroc"], 0.5)


if __name__ == "__main__":
    unittest.main()
