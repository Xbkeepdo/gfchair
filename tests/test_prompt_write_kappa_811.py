import unittest

import numpy as np

from scripts import analyze_all_source_paths as source_schema
from scripts import evaluate_prompt_write_kappa_811 as study


class PromptWriteKappa811Tests(unittest.TestCase):
    def test_reconstructs_write_kappa_and_concatenates_f(self):
        n, layers = 3, 2
        gross = np.array([[2., 4.], [5., 2.], [3., 6.]])
        desired = np.array([[.5, .25], [.75, .6], [.4, .9]])
        gain = np.full((n, layers), 2., dtype=np.float32)
        net = (gross * desired * gain).astype(np.float32)
        fields = [np.zeros((n, layers), dtype=np.float32) for _ in source_schema.F1_FIELDS]
        fields[source_schema.F1_FIELDS.index("prompt_net_norm")] = net
        fields[source_schema.F1_FIELDS.index("prompt_group_gain")] = gain
        statistics = np.zeros((n, layers, 4, 12))
        statistics[:, :, 0, 0] = gross
        f = np.arange(n*layers*2, dtype=np.float32).reshape(n, layers*2)
        result = study.build_features(np.concatenate(fields, axis=1), statistics, f)
        np.testing.assert_allclose(result["write_kappa"], desired, rtol=1e-7)
        np.testing.assert_array_equal(result["F_plus_write_kappa"][:, :layers*2], f)
        np.testing.assert_array_equal(result["F_plus_write_kappa"][:, layers*2:], result["write_kappa"])

    def test_rejects_unbounded_or_undefined_kappa(self):
        layers = 2
        fields = [np.zeros((1, layers), dtype=np.float32) for _ in source_schema.F1_FIELDS]
        fields[source_schema.F1_FIELDS.index("prompt_net_norm")] = np.ones((1, layers), dtype=np.float32)
        fields[source_schema.F1_FIELDS.index("prompt_group_gain")] = np.ones((1, layers), dtype=np.float32)
        statistics = np.zeros((1, layers, 4, 12))
        statistics[:, :, 0, 0] = .5
        with self.assertRaisesRegex(ValueError, "bound"):
            study.build_features(np.concatenate(fields, axis=1), statistics, np.zeros((1, layers*2)))
        statistics[:, :, 0, 0] = 0
        with self.assertRaisesRegex(ValueError, "Undefined"):
            study.build_features(np.concatenate(fields, axis=1), statistics, np.zeros((1, layers*2)))


if __name__ == "__main__":
    unittest.main()
