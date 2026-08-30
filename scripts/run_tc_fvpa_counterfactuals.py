#!/usr/bin/env python3
"""Consolidate measured frozen-write counterfactuals and audit missing families."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.tc_fvpa_artifacts import (  # noqa: E402
    ExperimentLayout,
    atomic_json_save,
    iter_case_shards,
    update_stage_status,
)
from scripts.tc_fvpa_common import (  # noqa: E402
    add_common_arguments,
    print_dry_run,
    shell_command,
    validate_common_args,
)


def parse_args() -> argparse.Namespace:
    parser = add_common_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument(
        "--families",
        default="frozen_write,fixed_qk,activation_patching,pixel_counterfactual",
    )
    return parser.parse_args()


def _write_csv_gz(path: Path, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    common = validate_common_args(args)
    families = [value for value in args.families.split(",") if value]
    unknown = set(families) - {
        "frozen_write",
        "fixed_qk",
        "activation_patching",
        "pixel_counterfactual",
    }
    if unknown:
        raise ValueError(f"Unknown counterfactual families {sorted(unknown)}")
    if args.dry_run:
        print_dry_run("counterfactuals", common, {"families": families})
        return
    layout = ExperimentLayout.create(common["output_dir"])
    stage = f"counterfactuals:{args.model}"
    rows = []
    for _path, payload in iter_case_shards(layout):
        for case in payload.get("rows", ()):
            if str(case.get("model")) != args.model:
                continue
            for intervention in case.get("frozen_write_interventions", ()):
                base = {
                    "model": args.model,
                    "case_id": case["case_id"],
                    "image_id": case["image_id"],
                    "response_index": case["response_index"],
                    "layer": case["layer"],
                    "label_string": case["label_string"],
                    "label_integer": case["label_integer"],
                    "intervention_family": "frozen_write",
                    "strategy": intervention["strategy"],
                    "region_size": intervention.get("region_size", 0),
                    "token_or_region_ids": json.dumps(
                        intervention.get("token_indices", [])
                    ),
                    "measurement_status": intervention["measurement_status"],
                    "error": intervention.get("error", ""),
                }
                for scalar, observed in intervention.get(
                    "observed_frozen_write_effect", {}
                ).items():
                    rows.append(
                        {
                            **base,
                            "target_scalar": scalar,
                            "predicted_finite_change": intervention[
                                "predicted_path_effect"
                            ][scalar],
                            "observed_finite_effect": observed,
                        }
                    )
                if intervention["measurement_status"] != "MEASURED":
                    rows.append({**base, "target_scalar": ""})
    table_path = layout.path("tables/counterfactuals.csv.gz")
    _write_csv_gz(table_path, rows)
    family_status = {
        "frozen_write": "PASS" if any(row["measurement_status"] == "MEASURED" for row in rows) else "NOT_RUN",
        "fixed_qk": "NOT_RUN",
        "activation_patching": "NOT_RUN",
        "pixel_counterfactual": "NOT_RUN",
    }
    missing = [
        family
        for family in families
        if family_status[family] != "PASS"
    ]
    summary = {
        "model": args.model,
        "rows": len(rows),
        "family_status": family_status,
        "important_estimand_note": (
            "Frozen-write effects condition on the observed clean attention "
            "decomposition and are not complete image-patch effects."
        ),
        "blocked_or_not_run": missing,
        "resume_command": shell_command(),
    }
    atomic_json_save(summary, layout.path("metrics/counterfactual_summary.json"))
    update_stage_status(
        layout=layout,
        stage=stage,
        status="BLOCKED" if missing else "PASS",
        details=summary,
        resume_command=shell_command(),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
