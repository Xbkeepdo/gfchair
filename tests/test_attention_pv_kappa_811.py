import unittest

import numpy as np

from scripts import analyze_all_source_paths as source
from scripts import evaluate_attention_pv_kappa_811 as study


class AttentionPVKappa811Tests(unittest.TestCase):
    def test_same_path_prompt_visual_strength_and_cancellation(self):
        model = "qwen2_5_vl_7b"
        layers = source.LAYERS[model]
        legacy = {"groups": {"legacy_visual": np.zeros((1, 2 * layers), dtype=np.float32)}}
        f1 = np.zeros((1, len(source.F1_FIELDS) * layers), dtype=np.float32)
        f2 = np.zeros((1, len(source.F2_FIELDS) * layers), dtype=np.float32)

        def fill(matrix, fields, field, value):
            index = fields.index(field)
            matrix[:, index * layers:(index + 1) * layers] = value

        fill(f1, source.F1_FIELDS, "prompt_gross_norm", 5)
        fill(f1, source.F1_FIELDS, "visual_gross_norm", 4)
        fill(f1, source.F1_FIELDS, "prompt_net_norm", 3)
        fill(f1, source.F1_FIELDS, "visual_net_norm", 4)
        fill(f2, source.F2_FIELDS, "prompt_within_source_kappa", 3 / 5)
        fill(f2, source.F2_FIELDS, "cos_vp_out", 0)

        groups, signals = study.build_features(model, legacy, {"groups": {"F1_raw": f1, "F2": f2}})
        np.testing.assert_allclose(signals["S_PV"], 9)
        np.testing.assert_allclose(signals["kP"], 3 / 5)
        np.testing.assert_allclose(signals["kVP"], 5 / 9)
        np.testing.assert_allclose(groups["AE_logS_PV"][:, layers:], np.log1p(9))
        np.testing.assert_allclose(groups["AE_logS_PV_kP"][:, 2 * layers:], 3 / 5)
        np.testing.assert_allclose(groups["AE_logS_PV_kVP"][:, 2 * layers:], 5 / 9)

        fill(f2, source.F2_FIELDS, "cos_vp_out", -1)
        _, opposite = study.build_features(model, legacy, {"groups": {"F1_raw": f1, "F2": f2}})
        np.testing.assert_allclose(opposite["kVP"], 1 / 9)


if __name__ == "__main__":
    unittest.main()
