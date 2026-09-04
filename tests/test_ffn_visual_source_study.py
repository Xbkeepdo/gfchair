import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.analyze_ffn_visual_source_study import (
    _js,
    build_feature_sets,
    freeze_numerics,
    gate_from_results,
    select_frozen_k,
    select_linearize_backend,
)
from scripts.analyze_ffn_visual_source_signal_ablation import (
    ae_times_cosine,
    swap_ae_for_old_ev,
)
from scripts.run_ffn_visual_source_attribution import (
    formal_image_ids,
    resolve_frozen_settings,
)
from scripts.run_ffn_visual_source_counterfactuals import (
    _balanced_targets,
    _bootstrap_mean_differences,
    _rectangle,
    _validate_counterfactual_rows,
)


class FFNVisualSourceStudyTest(unittest.TestCase):
    def test_formal_image_manifest_keeps_images_without_targets(self):
        labels = {1: {"object_token_spans": []}, 2: {"object_token_spans": [1]}}
        generations = {1: {"response_token_ids": [3]}, 2: {"response_token_ids": [4]}}
        splits = {"train": [1], "test": [2]}
        self.assertEqual(formal_image_ids(labels, generations, splits), [1, 2])

    def test_k_selection_uses_smallest_candidate_passing_every_row(self):
        good = {
            "median_spearman": 0.995,
            "median_js": 0.001,
            "median_top32_overlap": 0.97,
            "p90_gross_relative_error": 0.005,
            "p90_kappa_difference": 0.004,
        }
        rows = [
            {"model": model, "layer": layer, "k": k, **good}
            for model in ("qwen", "llava")
            for layer in (1, 2)
            for k in (4, 8, 16, 32)
        ]
        rows[0]["median_spearman"] = 0.98
        frozen, decisions = select_frozen_k(rows)
        self.assertEqual(frozen, 8)
        self.assertFalse(decisions["4"]["passed_all_model_layers"])

    def test_extensions_cannot_refreeze_k_without_primary_models(self):
        with self.assertRaisesRegex(ValueError, "both primary models"):
            freeze_numerics(["qwen3_vl_8b"], smoke=True)

    def test_feature_registry_is_exactly_thirteen_groups(self):
        position = {
            "r_cos": np.array([0.1, 0.2]),
            "ae_strength": np.array([0.3, 0.4]),
            "gross_strength": np.array([0.5, 0.6]),
            "kappa": np.array([0.7, 0.8]),
            "js": {name: np.array([0.1, 0.2]) for name in ("D_EW", "D_WF", "D_EF")},
            "ot": {name: np.array([0.2, 0.3]) for name in ("D_EW", "D_WF", "D_EF")},
        }
        features = build_feature_sets(position)
        self.assertEqual(len(features), 13)
        self.assertEqual(features["C_JS"].shape, (4,))
        self.assertEqual(features["G_OT"].shape, (12,))
        self.assertEqual(features["H_OT"].shape, (14,))

    def test_old_ev_swap_changes_only_the_ae_block(self):
        matrices = {
            "A": np.arange(8, dtype=np.float32).reshape(2, 4),
            "B": np.arange(4, dtype=np.float32).reshape(2, 2),
            "F": np.arange(12, dtype=np.float32).reshape(2, 6),
            "H_JS": np.arange(28, dtype=np.float32).reshape(2, 14),
        }
        ev = np.array([[100, 101], [102, 103]], dtype=np.float32)
        swapped = swap_ae_for_old_ev(matrices, ev)
        np.testing.assert_array_equal(swapped["A"][:, :2], matrices["A"][:, :2])
        np.testing.assert_array_equal(swapped["A"][:, 2:4], ev)
        np.testing.assert_array_equal(swapped["B"], ev)
        np.testing.assert_array_equal(swapped["F"][:, :2], ev)
        np.testing.assert_array_equal(swapped["F"][:, 2:], matrices["F"][:, 2:])
        np.testing.assert_array_equal(swapped["H_JS"][:, :2], matrices["H_JS"][:, :2])
        np.testing.assert_array_equal(swapped["H_JS"][:, 2:4], ev)
        np.testing.assert_array_equal(swapped["H_JS"][:, 4:], matrices["H_JS"][:, 4:])

    def test_ae_times_cosine_is_elementwise_and_shape_checked(self):
        ae = np.array([[0.2, 0.5]], dtype=np.float32)
        cosine = np.array([[-0.5, 0.8]], dtype=np.float32)
        np.testing.assert_allclose(ae_times_cosine(ae, cosine), [[-0.1, 0.4]])
        with self.assertRaisesRegex(ValueError, "same"):
            ae_times_cosine(ae, cosine.T)

    def test_gate_is_any_model_or_distance_with_positive_lower_bound(self):
        def model(low):
            return {
                "paired_bootstrap": {
                    f"G_{family}__minus__C_{family}": {
                        "new_minus_baseline_auroc": 0.01,
                        "auroc_ci95": [low if family == "JS" else -0.1, 0.2],
                    }
                    for family in ("JS", "OT")
                }
            }

        self.assertTrue(gate_from_results({"a": model(0.001), "b": model(-0.01)})["passed"])
        self.assertFalse(gate_from_results({"a": model(0.0), "b": model(-0.01)})["passed"])
        primary_only = gate_from_results(
            {
                "qwen2_5_vl_7b": model(-0.01),
                "llava_1_5_7b": model(-0.01),
                "qwen3_vl_8b": model(0.01),
            }
        )
        self.assertFalse(primary_only["passed"])
        self.assertEqual(len(primary_only["comparisons"]), 4)

    def test_linearize_gate_and_frozen_setting_resume_files(self):
        metric = {
            "vmap_jvp": {"status": "PASS", "seconds": 2.0, "increment_bytes": 100},
            "linearize": {"status": "PASS", "seconds": 1.7, "increment_bytes": 90},
            "relative_l2_error": 1e-4,
        }
        rows = [
            {"model": model, "kind": kind, "metrics": metric}
            for model in ("qwen2_5_vl_7b", "llava_1_5_7b")
            for kind in ("synthetic", "real_fp32")
        ]
        self.assertEqual(select_linearize_backend(rows)["selected_backend"], "linearize")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tables").mkdir()
            (root / "tables/frozen_k.json").write_text(
                json.dumps({"frozen_k": 16}), encoding="utf-8"
            )
            (root / "tables/linearize_decision.json").write_text(
                json.dumps({"selected_backend": "vmap_jvp"}), encoding="utf-8"
            )
            self.assertEqual(resolve_frozen_settings(root, 0, "auto"), (16, "vmap_jvp"))

    def test_js_is_symmetric_and_zero_on_identity(self):
        p = np.array([0.2, 0.3, 0.5])
        q = np.array([0.5, 0.4, 0.1])
        self.assertAlmostEqual(_js(p, p), 0.0, places=14)
        self.assertAlmostEqual(_js(p, q), _js(q, p), places=14)

    def test_counterfactual_selection_regions_and_bootstrap(self):
        positions = [
            {"target_key": f"{image}:0"} for image in range(6)
        ]
        mentions = [
            {
                "target_key": f"{image}:0",
                "image_id": image,
                "response_index": 0,
                "label": image % 2,
            }
            for image in range(6)
        ]
        selected = _balanced_targets(positions, mentions, count=6, seed=3)
        self.assertEqual(sum(row[2] == 0 for row in selected), 3)
        self.assertEqual(_rectangle([0, 1, 4, 5], [2, 4], [400, 200]), (0, 0, 200, 200))

        rows = []
        for image in range(4):
            for family in ("fixed_qk", "activation_patching", "pixel_counterfactual"):
                for strategy, value in zip(
                    ("highest_p_ffn_region", "highest_p_write_region", "random_region"),
                    (3.0, 2.0, 1.0),
                ):
                    rows.append(
                        {
                            "image_id": image,
                            "layer": 1,
                            "intervention_family": family,
                            "strategies": [strategy],
                            "target_logit_delta": value,
                            "fixed_clean_competitor_margin_delta": value,
                            "log_probability_delta": value,
                        }
                    )
        result = _bootstrap_mean_differences(rows, 100)
        self.assertEqual(_validate_counterfactual_rows(rows, 4, layers_per_image=1), 4)
        key = (
            "fixed_qk:target_logit_delta:"
            "highest_p_ffn_region__minus__random_region"
        )
        self.assertEqual(result[key]["effective_resamples"], 100)
        self.assertEqual(result[key]["mean_difference"], 2.0)
        with self.assertRaisesRegex(AssertionError, "strategy map"):
            _validate_counterfactual_rows(rows[:-1], 4, layers_per_image=1)


if __name__ == "__main__":
    unittest.main()
