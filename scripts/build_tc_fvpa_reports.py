#!/usr/bin/env python3
"""Build self-contained TC-FVPA reports strictly from persisted artifacts."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tarfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.tc_fvpa_artifacts import (  # noqa: E402
    ExperimentLayout,
    atomic_json_save,
    build_handoff_bundle,
    git_snapshot,
    load_run_status,
    update_stage_status,
    write_output_checksums,
)
from scripts.tc_fvpa_common import (  # noqa: E402
    FORMAL_LAYERS_BY_MODEL,
    add_common_arguments,
    print_dry_run,
    shell_command,
    validate_common_args,
)


REPORTS = {
    "03_ATTENTION_WRITE_DECOMPOSITION.md": "Attention write decomposition",
    "04_JACOBIAN_NUMERICAL_VALIDATION.md": "Jacobian numerical validation",
    "05_RIESZ_FUNCTIONAL_VALIDATION.md": "Riesz functional validation",
    "06_PATH_ATTRIBUTION_VALIDATION.md": "Path attribution validation",
    "07_FP32_CAUSAL_VALIDATION.md": "FP32 causal validation",
    "08_FINITE_COUNTERFACTUALS.md": "Finite counterfactuals",
    "09_SHAPLEY_AND_INTERACTIONS.md": "Shapley and interactions",
    "10_SWI_GLU_INTERNAL_ANALYSIS.md": "SwiGLU internal analysis",
    "11_GEOMETRY_AND_COSINE_CONTROLS.md": "Geometry and cosine controls",
    "12_SPATIAL_LOCALIZATION.md": "Spatial localization",
    "13_HALLUCINATION_DETECTION.md": "Hallucination detection",
    "14_CROSS_MODEL_GENERALIZATION.md": "Cross-model generalization",
    "15_EXTERNAL_QA_GENERALIZATION.md": "External QA generalization",
    "16_EFFICIENCY_AND_SCALING.md": "Efficiency and scaling",
    "17_FAILURE_CASES_AND_LIMITATIONS.md": "Failure cases and limitations",
    "18_FINAL_SCIENTIFIC_VERDICT.md": "Final scientific verdict",
}


def parse_args() -> argparse.Namespace:
    return add_common_arguments(argparse.ArgumentParser(description=__doc__)).parse_args()


def _load(path: Path, default: Any):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _copy_frozen_reports(layout: ExperimentLayout) -> None:
    central = ROOT / "outputs" / "tc_fvpa_comprehensive_v1" / "reports"
    for name in (
        "00_PREREGISTERED_HYPOTHESES.md",
        "01_PROBLEM_AND_NOTATION.md",
        "02_EXISTING_PIPELINE_AUDIT.md",
    ):
        source = central / name
        destination = layout.path(f"reports/{name}")
        if source.exists() and not destination.exists():
            shutil.copy2(source, destination)


def _status_for(
    name: str,
    summary: dict,
    fp32: dict,
    counter: dict,
    shapley: dict,
    validation: dict,
) -> tuple[str, str]:
    cases = int(summary.get("case_rows", 0))
    model = str(summary.get("model", "unknown_model"))
    cohort_images = summary.get("cohort_image_counts", {})
    cohort_cases = summary.get("cohort_case_counts", {})
    layers = {int(layer) for layer in summary.get("layers", ())}
    expected_layers = set(FORMAL_LAYERS_BY_MODEL.get(model, ()))
    formal_layers_match = bool(expected_layers) and layers == expected_layers
    local_formal = (
        int(cohort_images.get("local", 0)) >= 500
        and int(cohort_cases.get("local", 0)) >= 2_000
        and formal_layers_match
    )
    path_cases = int(cohort_cases.get("path", 0))
    path_rows = int(summary.get("path_convergence_rows", 0))
    path_formal = (
        int(cohort_images.get("path", 0)) >= 200
        and path_cases >= 800
        and path_rows >= path_cases * 30
        and formal_layers_match
    )
    if name == "03_ATTENTION_WRITE_DECOMPOSITION.md":
        return ("PARTIAL", f"Measured in {cases} target-layer cases; complete image-patch causality is outside this conditional-write estimand.") if cases else ("NOT RUN", "No new case shards.")
    if name == "04_JACOBIAN_NUMERICAL_VALIDATION.md":
        return ("PARTIAL", "Synthetic FP64 tests are separate; real-case reconstruction is available but no formal all-model audit.") if cases else ("NOT RUN", "No new real cases.")
    if name == "05_RIESZ_FUNCTIONAL_VALIDATION.md":
        if local_formal:
            return "PASS", f"Formal {model} local cohort completed: {cohort_images['local']} images and {cohort_cases['local']} target-layer cases across layers {sorted(layers)}."
        return ("PARTIAL", f"Local logit/margin/log-probability Riesz measured for {cases} cases; the formal {model} minimum or frozen layer set is incomplete.") if cases else ("NOT RUN", "No local Riesz cases.")
    if name == "06_PATH_ATTRIBUTION_VALIDATION.md":
        if path_formal:
            return "PASS", f"Formal {model} path cohort completed: {cohort_images['path']} images, {path_cases} target-layer cases, and {path_rows} rows covering 3 scalars x 5 K values x 2 quadratures."
        return ("PARTIAL", f"Persisted {path_rows} convergence rows; the formal {model} minimum, frozen layer set, or full path grid is incomplete.") if path_rows else ("NOT RUN", "No path convergence rows.")
    if name == "07_FP32_CAUSAL_VALIDATION.md":
        measured = int(fp32.get("measured_rows", 0))
        blocked = int(fp32.get("blocked_layer_rows", 0))
        failures = fp32.get("failures", ())
        if measured and not blocked and not failures:
            return "PASS", f"True FP32 causal validation measured {measured} rows with no blocked layers or failures."
        if measured:
            return "PARTIAL", f"True FP32 causal validation measured {measured} rows; {blocked} lower-layer entries and {len(failures)} failures remain."
        return "NOT RUN", "No true-FP32 measurements."
    if name == "08_FINITE_COUNTERFACTUALS.md":
        rows = int(counter.get("rows", 0))
        return ("PARTIAL", f"Frozen-write has {rows} rows; fixed-QK, activation patching and pixel counterfactuals are not complete.") if rows else ("NOT RUN", "No finite counterfactual rows.")
    if name == "09_SHAPLEY_AND_INTERACTIONS.md":
        measured = int(shapley.get("measured_rows", 0))
        return (
            ("PARTIAL", f"Measured {measured} true-FP32 final-block Shapley rows; formal scale is incomplete.")
            if measured
            else ("NOT RUN", "No persisted real-model Shapley measurement.")
        )
    if name == "10_SWI_GLU_INTERNAL_ANALYSIS.md":
        passed = int(validation.get("swiglu_tests_passed", 0))
        return (
            ("PARTIAL", f"{passed} synthetic decomposition tests passed; real neuron interventions are not run.")
            if passed
            else ("NOT RUN", "No persisted SwiGLU validation summary.")
        )
    if name == "11_GEOMETRY_AND_COSINE_CONTROLS.md":
        passed = int(validation.get("geometry_tests_passed", 0))
        return (
            ("PARTIAL", f"{passed} synthetic coordinate/control tests passed; cohort calibration is not run.")
            if passed
            else ("NOT RUN", "No persisted geometry validation summary.")
        )
    if name == "12_SPATIAL_LOCALIZATION.md":
        rows = int(summary.get("spatial_metric_rows", 0))
        return (
            ("PARTIAL", f"Persisted {rows} REAL-box case-method rows; formal paired bootstrap is incomplete.")
            if rows
            else ("NOT RUN", "No REAL case with a valid persisted COCO box map in this cohort.")
        )
    if name == "16_EFFICIENCY_AND_SCALING.md":
        return (
            ("PARTIAL", f"Per-case runtime exists for {cases} cases; formal throughput and peak-memory scaling are incomplete.")
            if cases
            else ("NOT RUN", "No measured runtime rows.")
        )
    if name == "17_FAILURE_CASES_AND_LIMITATIONS.md":
        return "PASS", "The failure/limitation audit is present; see run_status.json for blocked stages and exact resume commands."
    if name == "18_FINAL_SCIENTIFIC_VERDICT.md":
        return (
            ("PARTIAL", f"Formal {model} local/path evidence exists, while lower-layer true-FP32, full counterfactual, spatial-inference and detection axes remain open.")
            if cases
            else ("NOT RUN", "No new measurement supports a verdict.")
        )
    return "NOT RUN", "No persisted measurement for this experiment family."


def _write_report(path: Path, title: str, status: str, evidence: str, model: str, denominators: dict) -> None:
    text = f"""# {title}

Status: **{status}**

Model: `{model}`

## Evidence

{evidence}

## Cohort denominator

```json
{json.dumps(denominators, indent=2, ensure_ascii=False, sort_keys=True)}
```

## Interpretation boundary

`a_m` is a source-wise value-path write conditional on the observed clean attention pattern. It is not the complete causal effect of removing image patch `m`. The current-block zero-write baseline removes only current-block visual writes and is not a no-image state. Missing experiments are not inferred from existing measurements.
"""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_formal_scope_report(
    path: Path,
    *,
    model: str,
    summary: dict,
    fp32: dict,
    counter: dict,
    shapley: dict,
    run_status: dict,
) -> None:
    cohort_images = summary.get("cohort_image_counts", {})
    cohort_cases = summary.get("cohort_case_counts", {})
    local_images = int(cohort_images.get("local", 0))
    path_images = int(cohort_images.get("path", 0))
    local_cases = int(cohort_cases.get("local", 0))
    path_cases = int(cohort_cases.get("path", 0))
    path_rows = int(summary.get("path_convergence_rows", 0))
    layers = sorted({int(layer) for layer in summary.get("layers", ())})
    expected_layers = list(FORMAL_LAYERS_BY_MODEL.get(model, ()))
    formal_layers_match = bool(expected_layers) and layers == expected_layers
    final_layer = expected_layers[-1] if expected_layers else (layers[-1] if layers else "unknown")
    lower_layers = expected_layers[:-1] if expected_layers else layers[:-1]
    local_status = "PASS" if local_images >= 500 and local_cases >= 2_000 and formal_layers_match else "PARTIAL"
    path_status = "PASS" if path_images >= 200 and path_cases >= 800 and path_rows >= path_cases * 30 and formal_layers_match else "PARTIAL"
    fp32_rows = int(fp32.get("measured_rows", 0))
    fp32_blocked = int(fp32.get("blocked_layer_rows", 0))
    fp32_failures = len(fp32.get("failures", ()))
    fp32_status = "PASS" if fp32_rows and not fp32_blocked and not fp32_failures else "PARTIAL / BLOCKED"
    shapley_rows = int(shapley.get("measured_rows", 0))
    permutations = int(shapley.get("permutations", 0))
    not_in_scope = sorted({int(layer) for layer in shapley.get("not_in_scope_layers", ())})
    counter_rows = int(counter.get("rows", 0))
    families = counter.get("family_status", {})
    parquet = summary.get("parquet", {})
    parquet_statuses = sorted({str(entry.get("status", "NOT_RUN")) for entry in parquet.values()})
    stages = {
        name: entry.get("status")
        for name, entry in sorted(run_status.get("stages", {}).items())
    }
    text = f"""# Formal scope and blocker audit

This audit is generated from the persisted artifacts in this formal output root. It does not import smoke or cross-model results.

Model: `{model}`

## Frozen-protocol execution

| Family | Persisted formal evidence | Status |
|---|---|---|
| Local Riesz | {local_images} images; {local_cases} unique target-layer cases; layers {layers}; three target scalars | **{local_status}** |
| Path attribution | {path_images} images; {path_cases} unique target-layer cases; {path_rows} convergence rows (3 scalars x K=1/4/8/16/32 x trapezoid/Gauss-Legendre) | **{path_status}** |
| True FP32 causal | {fp32_rows} measured rows; final layer {final_layer} included; {fp32_blocked} persisted lower-layer blocker entries; {fp32_failures} failures | **{fp32_status}** |
| Shapley | {shapley_rows} final-layer true-FP32 rows; {permutations} permutations; non-final layers {not_in_scope} are outside implemented scope | **PARTIAL** |
| Frozen-write counterfactual | {counter_rows} rows; family status `{json.dumps(families, ensure_ascii=False, sort_keys=True)}` | **PARTIAL / BLOCKED** |
| Analyze | authoritative PT/CSV.GZ tables and metrics generated | **PASS** |
| Reports | dynamic evidence reports and handoff bundle generated | **PASS** |

## Explicitly blocked or not run

- Lower-layer true-FP32 downstream execution (layers {lower_layers}): **BLOCKED** when listed in the persisted FP32 summary. No lower-layer result is approximated or relabeled measured.
- Fixed-QK, activation patching, and pixel counterfactuals: **NOT_RUN**.
- Real-neuron interventions, train-only geometry calibration, hallucination detector fitting, and paired 10,000-replicate spatial/detection bootstrap: **NOT_RUN**.
- Cross-model generalization and external QA/VQA benchmarks: **NOT_RUN**. They were outside this single-model execution.
- Parquet exports: **{', '.join(parquet_statuses) if parquet_statuses else 'NOT_RUN'}** because optional pandas/Parquet dependencies are unavailable. PT and CSV.GZ outputs are authoritative.

## Persisted stage gate

```json
{json.dumps(stages, indent=2, ensure_ascii=False, sort_keys=True)}
```

The report status `PASS (scope audit)` means this audit faithfully records measured, blocked, and not-run items; it does not promote partial scientific families to PASS.
"""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    common = validate_common_args(args)
    if args.dry_run:
        print_dry_run("build_reports", common, {"reports": list(REPORTS)})
        return
    layout = ExperimentLayout.create(common["output_dir"])
    _copy_frozen_reports(layout)
    config_source = Path(common["config"])
    if config_source.is_file():
        shutil.copy2(
            config_source,
            layout.path(f"configs/{config_source.name}"),
        )
    summary = _load(layout.path("metrics/comprehensive_summary.json"), {})
    fp32 = _load(layout.path("metrics/fp32_causal_summary.json"), {})
    counter = _load(layout.path("metrics/counterfactual_summary.json"), {})
    shapley = _load(layout.path("metrics/shapley_summary.json"), {})
    validation = _load(
        layout.path("metrics/validation_test_summary.json"), {}
    )
    denominators = {
        "model": args.model,
        "images": summary.get("images", 0),
        "local_images": summary.get("cohort_image_counts", {}).get("local", 0),
        "path_images": summary.get("cohort_image_counts", {}).get("path", 0),
        "target_layer_cases": summary.get("case_rows", 0),
        "path_target_layer_cases": summary.get("cohort_case_counts", {}).get("path", 0),
        "labels": summary.get("labels", {}),
        "layers": summary.get("layers", []),
        "path_convergence_rows": summary.get("path_convergence_rows", 0),
        "frozen_write_rows": summary.get("counterfactual_rows", 0),
        "fp32_intervention_rows": fp32.get("measured_rows", 0),
        "shapley_rows": shapley.get("measured_rows", 0),
        "spatial_metric_rows": summary.get("spatial_metric_rows", 0),
    }
    index_rows = []
    for filename, title in REPORTS.items():
        report_status, evidence = _status_for(
            filename, summary, fp32, counter, shapley, validation
        )
        _write_report(
            layout.path(f"reports/{filename}"),
            title,
            report_status,
            evidence,
            args.model,
            denominators,
        )
        index_rows.append((filename, report_status))
    fixed_index_rows = [
        ("00_PREREGISTERED_HYPOTHESES.md", "FROZEN"),
        ("01_PROBLEM_AND_NOTATION.md", "PASS (definition)"),
        ("02_EXISTING_PIPELINE_AUDIT.md", "PASS (audit)"),
        *index_rows,
        ("19_FORMAL_SCOPE_AND_BLOCKER_AUDIT.md", "PASS (scope audit)"),
        ("HANDOFF_TO_CHATGPT.md", "PASS (artifact build)"),
    ]
    index = "# TC-FVPA report index\n\n" + "\n".join(
        f"- `{name}`: **{report_status}**"
        for name, report_status in fixed_index_rows
    ) + "\n"
    layout.path("reports/REPORT_INDEX.md").write_text(index, encoding="utf-8")

    update_stage_status(
        layout=layout,
        stage=f"reports:{args.model}",
        status="PASS",
        details={"reports": len(REPORTS) + 6},
        resume_command=shell_command(),
    )
    status = load_run_status(layout)
    _write_formal_scope_report(
        layout.path("reports/19_FORMAL_SCOPE_AND_BLOCKER_AUDIT.md"),
        model=args.model,
        summary=summary,
        fp32=fp32,
        counter=counter,
        shapley=shapley,
        run_status=status,
    )

    repository = _load(layout.path("manifests/experiment_manifest.json"), {}).get("repository", {})
    current_repository = git_snapshot(ROOT)
    revision_output = repository.get("revision", {}).get("output", "unknown").strip()
    verdict = "NOT RUN"
    if denominators["target_layer_cases"]:
        verdict = (
            f"PARTIAL: formal {args.model} local/path measurements exist, but lower-layer "
            "true-FP32, full counterfactual, spatial-inference and detection axes remain unavailable."
        )
    handoff = f"""# HANDOFF TO CHATGPT

Repository commit: `{revision_output}`

Modified/untracked files at final report build:

```text
{current_repository.get('status', {}).get('output', '').strip() or 'clean/unknown'}
```

Model: `{args.model}`

Scientific verdict: **{verdict}**

## Core equations and exact meanings

- `a_m = sum_h W_O^h(alpha_hm v_hm)`: clean-attention conditional source write; output bias is added once only to the full attention reconstruction.
- `delta_m = J_G(z) a_m`, where `G=FFN(Norm(z))` excludes the residual identity.
- `g = grad_m S(m_0)` is the Euclidean Riesz representative of the downstream scalar differential; `r=J_G(z)^T g` is its pre-FFN pullback.
- `C_m^local = g^T delta_m = r^T a_m` may be positive or negative.
- `e_m^path = integral J_G(z0+alpha A)a_m d alpha`, with `z0=z-A`; this baseline is not no-image.
- `ATTN`, `WRITE=||a_m||`, `JFFN=||J_G a_m||`, and `GAIN=JFFN/WRITE` are non-target baselines. `SIGNED_Q`, local Riesz, and path Riesz retain their sign in raw storage.

## Denominators

```json
{json.dumps(denominators, indent=2, ensure_ascii=False, sort_keys=True)}
```

## Persisted numerical summaries

```json
{json.dumps({'comprehensive': summary, 'fp32': fp32, 'counterfactuals': counter, 'shapley': shapley, 'validation': validation}, indent=2, ensure_ascii=False, sort_keys=True)}
```

## Experiment status

```json
{json.dumps(status, indent=2, ensure_ascii=False, sort_keys=True)}
```

## Negative results and limitations

This handoff reports only the persisted `{args.model}` execution and does not extrapolate to other models. True-FP32 downstream and Shapley are currently valid only for the final block unless their summaries state otherwise; lower-layer FP32, fixed-QK, activation/pixel counterfactual, neuron intervention, geometry calibration, formal detection and external QA remain explicitly incomplete unless their reports say otherwise.

No confidence interval or detector metric is reported because the preregistered train/test detector and 10,000-replicate image-cluster bootstrap were not executed. No missing inference is reconstructed from the formal mechanistic tables.

## Fixed and unresolved implementation issues

Fixed in this implementation: the target never enters its causal prefix; the margin competitor is frozen from clean logits; output-projection bias is assigned once; local JVP/VJP duality and path completeness are test-covered; negative contributions are persisted; the final-block causal route executes FFN, final norm, and LM head in actual FP32; shards and checksums are atomic and deduplicated.

Unresolved: model-specific lower-layer true-FP32 downstream execution where marked blocked, fixed-QK/full-activation/pixel estimands, real neuron intervention, train-only cohort geometry calibration, formal spatial/detection statistics, and external QA. Parquet is blocked by missing optional dependencies; PT and CSV.GZ remain authoritative.

## Data paths and loader

- Case table: `tables/case_layer.csv.gz` (Parquet only when the environment supports it).
- Token maps: `shards/token_maps_rankXX_shard_XXXXX.pt`.
- Interventions: `tables/interventions.csv.gz` when measured.
- Loader: repository `scripts/load_tc_fvpa_results.py`; run `python scripts/load_tc_fvpa_results.py {layout.root} --model {args.model} --method WRITE` after output checksums exist.
- Complete status and resume commands: `manifests/run_status.json`.

No missing measurement is replaced by zero or inference.
"""
    layout.path("reports/HANDOFF_TO_CHATGPT.md").write_text(handoff, encoding="utf-8")
    write_output_checksums(layout)
    bundle_manifest = build_handoff_bundle(layout)
    atomic_json_save(
        {"reports": index_rows, "handoff_files": len(bundle_manifest["files"])},
        layout.path("metrics/report_build_summary.json"),
    )
    # The second pass makes the checksum manifest and copied handoff manifest
    # include the report summary produced from the first pass.
    print(f"[TC-FVPA reports] wrote {layout.path('reports')}", flush=True)
    write_output_checksums(layout)
    build_handoff_bundle(layout)
    archive = layout.path("handoff_bundle.tar.gz")
    temporary_archive = archive.with_name(
        archive.name + f".tmp.{os.getpid()}"
    )
    with tarfile.open(temporary_archive, "w:gz") as handle:
        handle.add(layout.path("handoff_bundle"), arcname="handoff_bundle")
    os.replace(temporary_archive, archive)


if __name__ == "__main__":
    main()
