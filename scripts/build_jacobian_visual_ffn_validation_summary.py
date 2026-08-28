#!/usr/bin/env python3
"""Build a compact two-model summary from completed JFFN-P artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("llava_1_5_7b", "internvl_2_5_8b")
EXPERIMENT = "COCO4000-INSLEN-OFFICIAL-TARGET"
RISK_LAYER_START = 16


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _risk_layer_mean(
    rows: list[dict[str, str]],
    *,
    source: str,
    metric: str,
    label: str | None = None,
) -> float:
    values = [
        float(row["mean"])
        for row in rows
        if row["source"] == source
        and row["metric"] == metric
        and int(row["layer"]) >= RISK_LAYER_START
        and (label is None or row.get("label") == label)
    ]
    if not values:
        raise ValueError(f"No values for source={source}, metric={metric}, label={label}")
    return float(np.mean(values))


def _model_result_dir(model: str) -> Path:
    return ROOT / "outputs" / model / EXPERIMENT / "results" / "jffn_p_comparison"


def build_summary() -> dict[str, Any]:
    cohort_path = (
        ROOT
        / "outputs/llava_1_5_7b"
        / EXPERIMENT
        / "results/jacobian_visual_ffn_validation"
        / "jacobian_visual_ffn_full_cohort_metrics.json"
    )
    cohort = _load_json(cohort_path)["models"]
    summary: dict[str, Any] = {
        "protocol": "completed_models_jffn_validation_summary_v1",
        "scope": {
            "completed_models_only": list(MODELS),
            "not_analyzed_or_resumed": ["qwen2_5_vl_7b", "qwen3_vl_8b"],
            "risk_layers": [RISK_LAYER_START, 32],
        },
        "models": {},
    }

    for model in MODELS:
        result_dir = _model_result_dir(model)
        training = _load_json(result_dir / "training_results.json")
        smoke = _load_json(result_dir / "smoke_5images/extraction_audit.json")
        distribution = _load_csv(result_dir / "p_distribution.csv")
        specificity = _load_csv(result_dir / "p_target_specificity.csv")
        spatial = _load_csv(result_dir / "p_spatial_validation.csv")
        sources = (
            "old_hmid_cos",
            "old_hpre_cos",
            "new_jffn",
            "new_jffn_entropy_matched",
        )
        p_summary: dict[str, Any] = {}
        for source in sources:
            p_summary[source] = {
                "distribution": {
                    label: {
                        metric: _risk_layer_mean(
                            distribution, source=source, metric=metric, label=label
                        )
                        for metric in (
                            "normalized_entropy",
                            "effective_token_count",
                            "top32_mass",
                            "max_over_uniform",
                            "gini",
                        )
                    }
                    for label in ("real", "hall")
                },
                "same_image_different_target": {
                    metric: _risk_layer_mean(
                        specificity, source=source, metric=metric
                    )
                    for metric in ("raw_cosine", "centered_cosine", "js", "tv", "top32_overlap")
                },
                "spatial": {
                    metric: _risk_layer_mean(spatial, source=source, metric=metric)
                    for metric in (
                        "bbox_mass",
                        "mass_enrichment",
                        "top1_pointing_accuracy",
                        "patch_aupr",
                    )
                },
            }

        summary["models"][model] = {
            "formal_sample_counts": training["sample_counts"],
            "formal_main_comparisons": training["main_comparison_bootstrap"],
            "smoke_validation": smoke["validation"],
            "full_cohort_diagnostics": cohort[model],
            "p_analysis_risk_layers": p_summary,
        }

    llava_audit_path = (
        ROOT
        / "outputs/llava_1_5_7b"
        / EXPERIMENT
        / "results/jacobian_visual_ffn_validation"
        / "jacobian_visual_ffn_validation_metrics.json"
    )
    summary["llava_independent_audit"] = _load_json(llava_audit_path)
    return summary


def write_flat_csv(summary: dict[str, Any], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for model, payload in summary["models"].items():
        for name, comparison in payload["formal_main_comparisons"].items():
            for metric in ("seed_averaged_new_auroc", "seed_averaged_old_auroc", "difference"):
                rows.append(
                    {
                        "model": model,
                        "section": "formal_main_comparison",
                        "source": name,
                        "label": "all",
                        "metric": metric,
                        "value": comparison[metric],
                    }
                )
        for source, p_payload in payload["p_analysis_risk_layers"].items():
            for label, metrics in p_payload["distribution"].items():
                for metric, value in metrics.items():
                    rows.append(
                        {
                            "model": model,
                            "section": "p_distribution",
                            "source": source,
                            "label": label,
                            "metric": metric,
                            "value": value,
                        }
                    )
            for section in ("same_image_different_target", "spatial"):
                for metric, value in p_payload[section].items():
                    rows.append(
                        {
                            "model": model,
                            "section": section,
                            "source": source,
                            "label": "all",
                            "metric": metric,
                            "value": value,
                        }
                    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("model", "section", "source", "label", "metric", "value"),
        )
        writer.writeheader()
        writer.writerows(rows)


def render_plot(summary: dict[str, Any], path: Path) -> None:
    labels = ("LLaVA-1.5", "InternVL2.5")
    colors = {"old": "#4C78A8", "new": "#F58518", "matched": "#54A24B"}
    x = np.arange(len(MODELS), dtype=float)
    width = 0.23
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    old, new, matched = [], [], []
    for model in MODELS:
        comparisons = summary["models"][model]["formal_main_comparisons"]
        raw = comparisons["new_jffn_vs_old_hpre_cos"]
        ent = comparisons["new_jffn_entropy_matched_vs_old_hpre_cos"]
        old.append(raw["seed_averaged_old_auroc"])
        new.append(raw["seed_averaged_new_auroc"])
        matched.append(ent["seed_averaged_new_auroc"])
    ax = axes[0, 0]
    ax.bar(x - width, old, width, label="old hpre P", color=colors["old"])
    ax.bar(x, new, width, label="JFFN P", color=colors["new"])
    ax.bar(x + width, matched, width, label="entropy-matched", color=colors["matched"])
    ax.set_title("Preregistered hallucination detection")
    ax.set_ylabel("seed-averaged AUROC")
    ax.set_xticks(x, labels)
    ax.set_ylim(0.80, 0.91)
    ax.legend(fontsize=9)

    for ax, metric, title, ylabel in (
        (axes[0, 1], "patch_aupr", "COCO spatial localization", "patch AUPRC"),
        (axes[1, 0], "js", "Same-image target specificity", "JS divergence (higher = different)"),
    ):
        values = {key: [] for key in ("old", "new", "matched")}
        for model in MODELS:
            p_data = summary["models"][model]["p_analysis_risk_layers"]
            section = "spatial" if metric == "patch_aupr" else "same_image_different_target"
            values["old"].append(p_data["old_hpre_cos"][section][metric])
            values["new"].append(p_data["new_jffn"][section][metric])
            values["matched"].append(p_data["new_jffn_entropy_matched"][section][metric])
        ax.bar(x - width, values["old"], width, label="old hpre P", color=colors["old"])
        ax.bar(x, values["new"], width, label="JFFN P", color=colors["new"])
        ax.bar(x + width, values["matched"], width, label="entropy-matched", color=colors["matched"])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, labels)

    ax = axes[1, 1]
    metric_names = ("input_norm", "response_norm", "gain", "direction_cosine")
    metric_labels = ("I", "R", "S", "D")
    group_width = 0.36
    for index, model in enumerate(MODELS):
        univariate = summary["models"][model]["full_cohort_diagnostics"]["univariate_detection"]
        values = [univariate[name]["best_orientation_auroc"] for name in metric_names]
        positions = np.arange(len(metric_names)) + (index - 0.5) * group_width
        ax.bar(positions, values, group_width, label=labels[index])
    ax.axhline(0.5, color="black", linewidth=1, linestyle="--")
    ax.set_xticks(np.arange(len(metric_names)), metric_labels)
    ax.set_ylim(0.48, 0.70)
    ax.set_ylabel("best-orientation univariate AUROC")
    ax.set_title("JFFN aggregate diagnostics")
    ax.legend(fontsize=9)

    figure.suptitle("Completed-model JFFN-P validation summary", fontsize=16)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    summary = build_summary()
    json_path = ROOT / "jacobian_visual_ffn_validation_summary.json"
    csv_path = ROOT / "jacobian_visual_ffn_validation_summary.csv"
    plot_path = ROOT / "jacobian_visual_ffn_validation_summary.png"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_flat_csv(summary, csv_path)
    render_plot(summary, plot_path)
    print(json_path)
    print(csv_path)
    print(plot_path)


if __name__ == "__main__":
    main()
