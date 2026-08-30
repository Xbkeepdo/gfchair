import json
import tempfile
import unittest
from pathlib import Path

import torch

from features.tc_fvpa_artifacts import (
    ExperimentLayout,
    audit_case_shards,
    case_key,
    load_run_status,
    sha256_file,
    update_stage_status,
    validate_measurement_row,
    write_case_shard,
)


def measured_row(image_id=1, value=0.0):
    return {
        "model": "synthetic",
        "image_id": image_id,
        "response_index": 3,
        "layer": 2,
        "measurement_status": "MEASURED",
        "measured_fields": ["score"],
        "score": value,
    }


class TCFVPAArtifactTest(unittest.TestCase):
    def test_measured_zero_is_allowed_only_with_explicit_status_and_field(self):
        validate_measurement_row(measured_row(value=0.0))
        ambiguous = measured_row()
        ambiguous.pop("measured_fields")
        with self.assertRaisesRegex(ValueError, "enumerate measured_fields"):
            validate_measurement_row(ambiguous)
        with self.assertRaisesRegex(ValueError, "measurement_status"):
            validate_measurement_row({"score": 0.0})

    def test_failed_and_blocked_rows_persist_errors(self):
        base = measured_row()
        base.update({"measurement_status": "FAILED", "measured_fields": []})
        with self.assertRaisesRegex(ValueError, "non-empty error"):
            validate_measurement_row(base)
        base["error"] = "RuntimeError('synthetic')"
        validate_measurement_row(base)

    def test_atomic_shards_deduplicate_and_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            layout = ExperimentLayout.create(directory)
            first = write_case_shard(
                layout=layout,
                rank=0,
                shard_id=0,
                rows=[measured_row(1, 0.25)],
                failures=[],
                provenance={"seed": 7},
            )
            self.assertEqual(len(sha256_file(first)), 64)
            write_case_shard(
                layout=layout,
                rank=1,
                shard_id=0,
                rows=[measured_row(2, -0.5)],
                failures=[{"image_id": 9, "error": "synthetic"}],
                provenance={"seed": 7},
            )
            audit = audit_case_shards(layout)
            self.assertEqual(audit["unique_cases"], 2)
            self.assertEqual(len(audit["failures"]), 1)

            duplicate_path = layout.path("shards/case_rank02_shard_00000.pt")
            torch.save(
                {
                    "schema_version": "tc-fvpa-comprehensive-v1",
                    "rows": [measured_row(1, 9.0)],
                    "failures": [],
                },
                duplicate_path,
            )
            with self.assertRaisesRegex(ValueError, "Duplicate case across shards"):
                audit_case_shards(layout)

    def test_checkpoint_resume_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            layout = ExperimentLayout.create(directory)
            kwargs = dict(
                layout=layout,
                rank=0,
                shard_id=0,
                rows=[measured_row()],
                failures=[],
                provenance={},
            )
            write_case_shard(**kwargs)
            with self.assertRaises(FileExistsError):
                write_case_shard(**kwargs)

    def test_local_and_path_cohorts_use_independent_atomic_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            layout = ExperimentLayout.create(directory)
            path = write_case_shard(
                layout=layout,
                rank=0,
                shard_id=0,
                rows=[measured_row(1)],
                failures=[],
                provenance={},
                cohort="path",
            )
            local = write_case_shard(
                layout=layout,
                rank=0,
                shard_id=0,
                rows=[measured_row(1), measured_row(2)],
                failures=[],
                provenance={},
                cohort="local",
            )
            self.assertEqual(path.name, "case_rank00_shard_00000.pt")
            self.assertEqual(local.name, "local_case_rank00_shard_00000.pt")
            self.assertEqual(audit_case_shards(layout)["unique_cases"], 1)
            self.assertEqual(
                audit_case_shards(layout, cohort="local")["unique_cases"], 2
            )

    def test_case_uniqueness_requires_model_image_position_layer(self):
        self.assertEqual(case_key(measured_row()), "synthetic:1:3:2")
        row = measured_row()
        row.pop("layer")
        with self.assertRaises(KeyError):
            case_key(row)

    def test_status_update_is_idempotent_and_retains_started_time(self):
        with tempfile.TemporaryDirectory() as directory:
            layout = ExperimentLayout.create(directory)
            status_path = layout.path("manifests/run_status.json")
            status_path.write_text(
                json.dumps(
                    {
                        "stages": {},
                        "completed_shards": [],
                        "failed_shards": [],
                        "failed_cases": [],
                        "resume_commands": [],
                    }
                ),
                encoding="utf-8",
            )
            update_stage_status(
                layout=layout,
                stage="synthetic",
                status="RUNNING",
                resume_command="python resume.py",
            )
            first = load_run_status(layout)
            started = first["stages"]["synthetic"]["started_unix"]
            update_stage_status(
                layout=layout,
                stage="synthetic",
                status="PASS",
                resume_command="python resume.py",
            )
            second = load_run_status(layout)
            self.assertEqual(second["stages"]["synthetic"]["started_unix"], started)
            self.assertEqual(second["resume_commands"], ["python resume.py"])


if __name__ == "__main__":
    unittest.main()
