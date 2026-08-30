#!/usr/bin/env python3
"""Generate compact tables and numerical summaries from persisted TC-FVPA shards."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.tc_fvpa_artifacts import (  # noqa: E402
    ExperimentLayout,
    atomic_json_save,
    audit_case_shards,
    iter_case_shards,
    update_stage_status,
    write_output_checksums,
)
from scripts.tc_fvpa_common import (  # noqa: E402
    add_common_arguments,
    print_dry_run,
    shell_command,
    validate_common_args,
)
from scripts.load_tc_fvpa_results import iter_token_maps  # noqa: E402


def parse_args() -> argparse.Namespace:
    return add_common_arguments(argparse.ArgumentParser(description=__doc__)).parse_args()


def _jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _flat_case(row: Mapping[str, Any]) -> dict[str, Any]:
    skip = {
        "path_convergence",
        "frozen_write_interventions",
        "local_signed_statistics",
        "local_duality_max_abs_error",
        "available_experiments",
        "measured_fields",
    }
    result = {
        key: (_jsonable(value) if isinstance(value, (dict, list, tuple)) else value)
        for key, value in row.items()
        if key not in skip
    }
    for scalar, stats in (row.get("local_signed_statistics") or {}).items():
        for name, value in stats.items():
            result[f"local_{scalar}_{name}"] = value
    for scalar, value in (row.get("local_duality_max_abs_error") or {}).items():
        result[f"local_{scalar}_duality_max_abs_error"] = value
    for name, value in (row.get("available_experiments") or {}).items():
        result[f"available_{name}"] = bool(value)
    for method, metrics in (row.get("path_convergence") or {}).items():
        for name in (
            "score_sum",
            "finite_effect",
            "completeness_absolute_error",
            "completeness_relative_error",
        ):
            if name in metrics:
                result[f"{method.lower()}_{name}"] = metrics[name]
    return result


def _write_csv_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, ensure_ascii=False, sort_keys=True)
                        if isinstance(value, (dict, list, tuple))
                        else value
                    )
                    for key, value in row.items()
                }
            )
    temporary.replace(path)


def _try_parquet(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        import pandas as pd

        frame = pd.DataFrame(rows)
        frame.to_parquet(path, index=False)
        return {"status": "PASS", "rows": len(frame)}
    except Exception as exc:
        return {"status": "BLOCKED", "error": repr(exc), "rows": len(rows)}


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    result = np.empty_like(order, dtype=np.float64)
    result[order] = np.arange(values.size, dtype=np.float64)
    return result


def _correlation(left, right, *, rank: bool = False) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.size < 3 or x.std() <= 1e-20 or y.std() <= 1e-20:
        return float("nan")
    if rank:
        x, y = _ranks(x), _ranks(y)
    return float(np.corrcoef(x, y)[0, 1])


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = labels.astype(bool)
    positive = int(labels.sum())
    negative = int((~labels).sum())
    if positive == 0 or negative == 0:
        return None
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(scores.size, dtype=np.float64)
    index = 0
    while index < scores.size:
        end = index + 1
        while end < scores.size and scores[order[end]] == scores[order[index]]:
            end += 1
        ranks[order[index:end]] = 0.5 * (index + 1 + end)
        index = end
    value = (
        ranks[labels].sum() - positive * (positive + 1) / 2.0
    ) / float(positive * negative)
    return float(value)


def _binary_aupr(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = labels.astype(bool)
    positive = int(labels.sum())
    if positive == 0:
        return None
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order].astype(np.float64)
    precision = np.cumsum(ranked) / np.arange(1, ranked.size + 1)
    return float(precision[ranked.astype(bool)].sum() / positive)


def _token_and_spatial_rows(layout: ExperimentLayout, model: str):
    sample_rows: list[dict[str, Any]] = []
    spatial_rows: list[dict[str, Any]] = []
    for case_offset, row in enumerate(
        iter_token_maps(layout.root, verify_checksums=False)
    ):
        if str(row["model"]) != model:
            continue
        overlap_value = row.get("box_overlap")
        overlap = (
            torch.as_tensor(overlap_value).float().reshape(-1)
            if overlap_value is not None
            else None
        )
        labels = (
            overlap.numpy() > 0.0
            if overlap is not None
            else None
        )
        for method, raw_scores in (row.get("methods") or {}).items():
            scores = torch.as_tensor(raw_scores).float().reshape(-1)
            if case_offset < 3:
                for token_index, score in enumerate(scores.tolist()):
                    sample_rows.append(
                        {
                            "model": model,
                            "case_id": row["case_id"],
                            "image_id": row["image_id"],
                            "layer": row["layer"],
                            "method": method,
                            "visual_token_index": token_index,
                            "score": score,
                            "box_overlap": (
                                float(overlap[token_index])
                                if overlap is not None
                                else None
                            ),
                        }
                    )
            if labels is None or int(labels.sum()) == 0:
                continue
            values = scores.numpy().astype(np.float64)
            positive_values = np.maximum(values, 0.0)
            positive_total = float(positive_values.sum())
            probability = (
                positive_values / positive_total
                if positive_total > 0.0
                else None
            )
            top_index = int(np.argmax(values))
            top_k = min(32, values.size)
            top_indices = np.argsort(-values, kind="stable")[:top_k]
            uniform_mass = float(np.asarray(overlap).mean())
            bbox_mass = (
                float((probability * np.asarray(overlap)).sum())
                if probability is not None
                else None
            )
            spatial_rows.append(
                {
                    "model": model,
                    "case_id": row["case_id"],
                    "image_id": row["image_id"],
                    "layer": row["layer"],
                    "method": method,
                    "patch_count": values.size,
                    "positive_patch_count": int(labels.sum()),
                    "top1_pointing": int(bool(labels[top_index])),
                    "patch_auroc": _binary_auroc(labels, values),
                    "patch_aupr": _binary_aupr(labels, values),
                    "bbox_mass": bbox_mass,
                    "uniform_enrichment": (
                        bbox_mass / max(uniform_mass, 1e-12)
                        if bbox_mass is not None
                        else None
                    ),
                    "top32_recall": float(labels[top_indices].sum() / labels.sum()),
                    "normalization_status": (
                        "POSITIVE_MASS" if probability is not None else "BLOCKED_NO_POSITIVE_MASS"
                    ),
                    "measurement_status": "MEASURED",
                }
            )
    return sample_rows, spatial_rows


def main() -> None:
    args = parse_args()
    common = validate_common_args(args)
    if args.dry_run:
        print_dry_run("analyze", common)
        return
    layout = ExperimentLayout.create(common["output_dir"])
    stage = f"analysis:{args.model}"
    path_shard_audit = audit_case_shards(layout, cohort="path")
    local_shard_audit = audit_case_shards(layout, cohort="local")
    cases_by_key: dict[str, Mapping[str, Any]] = {}
    cohort_case_counts: dict[str, int] = {}
    cohort_image_counts: dict[str, int] = {}
    # Path rows have richer methods and intentionally supersede their local-only
    # duplicate.  The remaining local rows extend the analysis cohort to 500
    # images without pretending that all of them have finite-path results.
    for cohort in ("path", "local"):
        cohort_rows = []
        for _path, payload in iter_case_shards(layout, cohort=cohort):
            cohort_rows.extend(
                case
                for case in payload.get("rows", ())
                if str(case.get("model")) == args.model
            )
        cohort_case_counts[cohort] = len(cohort_rows)
        cohort_image_counts[cohort] = len(
            {int(case["image_id"]) for case in cohort_rows}
        )
        for case in cohort_rows:
            key = str(case["case_id"])
            if cohort == "path" or key not in cases_by_key:
                cases_by_key[key] = case
    case_rows = []
    convergence_rows = []
    counterfactual_rows = []
    for case in cases_by_key.values():
        case_rows.append(_flat_case(case))
        for name, metrics in (case.get("path_convergence") or {}).items():
            method_text, k_text = name.rsplit("_K", 1)
            scalar_text = method_text.removeprefix("PATH_")
            quadrature = (
                "gauss_legendre"
                if "_GAUSS_LEGENDRE" in scalar_text
                else "trapezoid"
            )
            scalar = scalar_text.replace("_GAUSS_LEGENDRE", "").replace(
                "_TRAPEZOID", ""
            ).lower()
            convergence_rows.append(
                {
                    "model": case["model"],
                    "case_id": case["case_id"],
                    "image_id": case["image_id"],
                    "layer": case["layer"],
                    "label_string": case["label_string"],
                    "target_scalar": scalar,
                    "quadrature_type": quadrature,
                    "K": int(k_text),
                    "baseline_type": "zero_current_visual_write",
                    **metrics,
                    "runtime": case.get("runtime_seconds"),
                    "sample_count": 1,
                }
            )
        for intervention in case.get("frozen_write_interventions", ()):
            for scalar, observed in intervention.get(
                "observed_frozen_write_effect", {}
            ).items():
                counterfactual_rows.append(
                    {
                        "model": case["model"],
                        "case_id": case["case_id"],
                        "image_id": case["image_id"],
                        "layer": case["layer"],
                        "label_string": case["label_string"],
                        "strategy": intervention["strategy"],
                        "region_size": intervention.get("region_size", 0),
                        "target_scalar": scalar,
                        "predicted_path_effect": intervention[
                            "predicted_path_effect"
                        ][scalar],
                        "observed_frozen_write_effect": observed,
                        "measurement_status": intervention[
                            "measurement_status"
                        ],
                    }
                )
    if not case_rows:
        update_stage_status(
            layout=layout,
            stage=stage,
            status="BLOCKED",
            details={"error": "No measured case shards"},
            resume_command=shell_command(),
        )
        raise RuntimeError("No measured case shards to analyze")

    _write_csv_gz(layout.path("tables/case_layer.csv.gz"), case_rows)
    _write_csv_gz(layout.path("tables/path_convergence.csv.gz"), convergence_rows)
    _write_csv_gz(layout.path("tables/counterfactuals.csv.gz"), counterfactual_rows)
    token_sample_rows, spatial_rows = _token_and_spatial_rows(layout, args.model)
    _write_csv_gz(
        layout.path("tables/token_scores_sample.csv.gz"), token_sample_rows
    )
    _write_csv_gz(layout.path("tables/spatial_metrics.csv.gz"), spatial_rows)
    parquet_status = {
        "case_layer": _try_parquet(layout.path("tables/case_layer.parquet"), case_rows),
        "path_convergence": _try_parquet(
            layout.path("tables/path_convergence.parquet"), convergence_rows
        ),
        "token_scores_sample": _try_parquet(
            layout.path("tables/token_scores_sample.parquet"), token_sample_rows
        ),
        "spatial_metrics": _try_parquet(
            layout.path("tables/spatial_metrics.parquet"), spatial_rows
        ),
    }
    correlations = {}
    grouped = defaultdict(lambda: ([], []))
    for row in counterfactual_rows:
        if row["measurement_status"] != "MEASURED":
            continue
        key = (row["target_scalar"], row["strategy"])
        grouped[key][0].append(row["predicted_path_effect"])
        grouped[key][1].append(row["observed_frozen_write_effect"])
    for (scalar, strategy), (predicted, observed) in grouped.items():
        correlations[f"{scalar}:{strategy}"] = {
            "count": len(predicted),
            "pearson": _finite_or_none(_correlation(predicted, observed)),
            "spearman": _finite_or_none(
                _correlation(predicted, observed, rank=True)
            ),
            "sign_accuracy": float(
                np.mean(np.sign(predicted) == np.sign(observed))
            )
            if predicted
            else None,
            "mae": float(np.mean(np.abs(np.asarray(predicted) - np.asarray(observed))))
            if predicted
            else None,
        }
    summary = {
        "model": args.model,
        "case_rows": len(case_rows),
        "images": len({row["image_id"] for row in case_rows}),
        "layers": sorted({int(row["layer"]) for row in case_rows}),
        "labels": {
            label: sum(row["label_string"] == label for row in case_rows)
            for label in ("REAL", "HALL")
        },
        "path_convergence_rows": len(convergence_rows),
        "counterfactual_rows": len(counterfactual_rows),
        "token_score_sample_rows": len(token_sample_rows),
        "spatial_metric_rows": len(spatial_rows),
        "path_vs_frozen_write": correlations,
        "parquet": parquet_status,
        "cohort_case_counts": cohort_case_counts,
        "cohort_image_counts": cohort_image_counts,
        "shard_audit": {
            "path": path_shard_audit,
            "local": local_shard_audit,
        },
    }
    atomic_json_save(summary, layout.path("metrics/comprehensive_summary.json"))
    atomic_json_save(
        {
            "schema_version": "tc-fvpa-comprehensive-v1",
            "case_uniqueness": ["model", "image_id", "response_index", "layer"],
            "label_direction": {"HALL": 0, "REAL": 1},
            "measurement_status": ["MEASURED", "FAILED", "BLOCKED", "NOT_RUN"],
        },
        layout.path("schemas/case_layer_schema.json"),
    )
    atomic_json_save(
        {
            "schema_version": "tc-fvpa-comprehensive-v1",
            "storage": "sharded torch tensors",
            "required_identifiers": [
                "model",
                "image_id",
                "case_id",
                "response_index",
                "prediction_position",
                "target_token_id",
                "label_string",
                "label_integer",
                "layer",
                "visual_grid",
                "validity_mask",
            ],
            "method_container": "methods",
            "negative_scores_must_be_preserved": True,
        },
        layout.path("schemas/token_maps_schema.json"),
    )
    formal_requirements = {
        "path_case_rows": 800 if args.formal else 1,
        "local_images": 500 if args.formal else 0,
    }
    status = (
        "PASS"
        if cohort_case_counts.get("path", 0) >= formal_requirements["path_case_rows"]
        and cohort_image_counts.get("local", 0) >= formal_requirements["local_images"]
        else "BLOCKED"
    )
    update_stage_status(
        layout=layout,
        stage=stage,
        status=status,
        details={
            "case_rows": len(case_rows),
            "cohort_case_counts": cohort_case_counts,
            "cohort_image_counts": cohort_image_counts,
            "formal_requirements": formal_requirements,
            "parquet": parquet_status,
        },
        resume_command=shell_command(),
    )
    # Stage status mutates run_status.json, so checksums must be the final write.
    write_output_checksums(layout)
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
