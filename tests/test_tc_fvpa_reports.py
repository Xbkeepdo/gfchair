import tempfile
import unittest
from pathlib import Path

from scripts.build_tc_fvpa_reports import _status_for, _write_formal_scope_report


class TCFVPAReportTest(unittest.TestCase):
    def setUp(self):
        self.summary = {
            "model": "llava_1_5_7b",
            "case_rows": 3656,
            "cohort_image_counts": {"local": 500, "path": 200},
            "cohort_case_counts": {"local": 3656, "path": 1420},
            "layers": [8, 16, 24, 32],
            "path_convergence_rows": 42600,
            "parquet": {"case_layer": {"status": "BLOCKED"}},
        }
        self.fp32 = {"measured_rows": 60144, "blocked_layer_rows": 6}
        self.counter = {
            "rows": 46830,
            "family_status": {
                "frozen_write": "PASS",
                "fixed_qk": "NOT_RUN",
                "activation_patching": "NOT_RUN",
                "pixel_counterfactual": "NOT_RUN",
            },
        }
        self.shapley = {
            "measured_rows": 300,
            "permutations": 128,
            "not_in_scope_layers": [8, 16, 24],
        }

    def test_formal_local_and_path_are_not_relabelled_incomplete(self):
        local = _status_for(
            "05_RIESZ_FUNCTIONAL_VALIDATION.md",
            self.summary,
            self.fp32,
            self.counter,
            self.shapley,
            {},
        )
        path = _status_for(
            "06_PATH_ATTRIBUTION_VALIDATION.md",
            self.summary,
            self.fp32,
            self.counter,
            self.shapley,
            {},
        )
        self.assertEqual(local[0], "PASS")
        self.assertEqual(path[0], "PASS")
        self.assertIn("Formal llava_1_5_7b", local[1])
        self.assertIn("3 scalars x 5 K values x 2 quadratures", path[1])

    def test_scope_report_is_dynamic_and_preserves_blockers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "19_FORMAL_SCOPE_AND_BLOCKER_AUDIT.md"
            _write_formal_scope_report(
                path,
                model="llava_1_5_7b",
                summary=self.summary,
                fp32=self.fp32,
                counter=self.counter,
                shapley=self.shapley,
                run_status={
                    "stages": {
                        "path_attribution:llava_1_5_7b:shard0": {"status": "PASS"},
                        "fp32_causal:llava_1_5_7b:shard0": {"status": "BLOCKED"},
                    }
                },
            )
            text = path.read_text(encoding="utf-8")
        self.assertIn("500 images; 3656 unique target-layer cases", text)
        self.assertIn("200 images; 1420 unique target-layer cases", text)
        self.assertIn("60144 measured rows; final layer 32 included", text)
        self.assertIn("Cross-model generalization and external QA/VQA benchmarks: **NOT_RUN**", text)
        self.assertNotIn("Formally completed here | 0", text)

    def test_qwen_frozen_layers_are_recognized_as_formal(self):
        summary = dict(self.summary)
        summary.update(
            {
                "model": "qwen2_5_vl_7b",
                "layers": [7, 14, 21, 28],
            }
        )

        local = _status_for(
            "05_RIESZ_FUNCTIONAL_VALIDATION.md",
            summary,
            self.fp32,
            self.counter,
            self.shapley,
            {},
        )
        path = _status_for(
            "06_PATH_ATTRIBUTION_VALIDATION.md",
            summary,
            self.fp32,
            self.counter,
            self.shapley,
            {},
        )

        self.assertEqual(local[0], "PASS")
        self.assertEqual(path[0], "PASS")
        self.assertIn("layers [7, 14, 21, 28]", local[1])

    def test_wrong_layer_set_cannot_pass_formal_gate(self):
        summary = dict(self.summary)
        summary["layers"] = [7, 14, 21, 28]

        local = _status_for(
            "05_RIESZ_FUNCTIONAL_VALIDATION.md",
            summary,
            self.fp32,
            self.counter,
            self.shapley,
            {},
        )

        self.assertEqual(local[0], "PARTIAL")
        self.assertIn("frozen layer set", local[1])

    def test_fp32_without_blockers_is_pass(self):
        status = _status_for(
            "07_FP32_CAUSAL_VALIDATION.md",
            self.summary,
            {"measured_rows": 216384, "blocked_layer_rows": 0, "failures": []},
            self.counter,
            self.shapley,
            {},
        )

        self.assertEqual(status[0], "PASS")
        self.assertIn("no blocked layers or failures", status[1])


if __name__ == "__main__":
    unittest.main()
