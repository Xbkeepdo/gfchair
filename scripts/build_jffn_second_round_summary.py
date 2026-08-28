#!/usr/bin/env python3
"""Build cross-model machine summaries and representative heatmaps."""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_jffn_p_comparison import _image_path  # noqa: E402
from utils.config_utils import load_config  # noqa: E402


MODELS = ("llava_1_5_7b", "internvl_2_5_8b")
EXPERIMENT = "COCO4000-INSLEN-OFFICIAL-TARGET"
PRIMARY_LAYERS = set(range(16, 33))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def grouped_primary(rows, group_names: Sequence[str]):
    groups: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for row in rows:
        if "layer" in row and int(row["layer"]) not in PRIMARY_LAYERS:
            continue
        groups[tuple(str(row[name]) for name in group_names)].append(float(row["mean"]))
    return {
        "|".join(key): float(np.mean(values))
        for key, values in sorted(groups.items())
    }


def summarize_model(model: str, metrics_rows: list[dict[str, Any]]):
    result_dir = ROOT / "outputs" / model / EXPERIMENT / "results/jffn_second_round"
    token = json.loads((result_dir / "token_map_analysis_summary.json").read_text())
    scalar = json.loads((result_dir / "scalar_incremental_metrics.json").read_text())
    logit_path = result_dir / "logit_causal/logit_causal_summary.json"
    logit = json.loads(logit_path.read_text()) if logit_path.exists() else {"status": "NOT RUN"}
    conceptual_path = result_dir / "conceptual_stage_summary.json"
    conceptual = (
        json.loads(conceptual_path.read_text())
        if conceptual_path.exists()
        else {"status": "NOT RUN"}
    )
    conceptual_correlation_path = result_dir / "conceptual_stage_correlations.csv"
    if conceptual_correlation_path.exists():
        conceptual["correlations"] = read_csv(conceptual_correlation_path)
    intervention_path = result_dir / "logit_causal/causal_interventions.csv"
    if intervention_path.exists() and "sample" in logit:
        intervention_rows = read_csv(intervention_path)
        resolution = []
        for eta in sorted({float(row["eta"]) for row in intervention_rows}):
            rows = [row for row in intervention_rows if float(row["eta"]) == eta]
            observed = np.asarray(
                [float(row["observed_delta_margin"]) for row in rows], dtype=np.float64
            )
            predicted = np.asarray(
                [float(row["predicted_delta_margin"]) for row in rows], dtype=np.float64
            )
            nonzero = observed != 0
            sign_accuracy = (
                float(np.mean(np.sign(predicted[nonzero]) == np.sign(observed[nonzero])))
                if np.any(nonzero) else float("nan")
            )
            resolution.append({
                "eta": eta,
                "count": len(rows),
                "zero_observed_fraction": float(np.mean(~nonzero)),
                "minimum_nonzero_absolute_observed_change": (
                    float(np.min(np.abs(observed[nonzero])))
                    if np.any(nonzero) else float("nan")
                ),
                "median_absolute_prediction": float(np.median(np.abs(predicted))),
                "nonzero_observed_sign_accuracy": sign_accuracy,
            })
        logit["intervention_resolution_audit"] = resolution
    logit_spatial_path = result_dir / "logit_causal/logit_positive_spatial.csv"
    if logit_spatial_path.exists() and "sample" in logit:
        logit_spatial_rows = read_csv(logit_spatial_path)
        grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row in logit_spatial_rows:
            grouped[(row["method"], row["metric"])].append(float(row["value"]))
        logit["positive_support_spatial"] = {
            f"{method}|{metric}": {
                "count": len(values),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
            for (method, metric), values in sorted(grouped.items())
        }
    spatial_rows = read_csv(result_dir / "spatial_layer_metrics.csv")
    ranking_rows = read_csv(result_dir / "ranking_layer_metrics.csv")
    specificity_rows = read_csv(result_dir / "same_image_target_specificity.csv")
    gain_rows = read_csv(result_dir / "token_gain_layer_metrics.csv")
    cancellation_rows = read_csv(result_dir / "cancellation_layer_metrics.csv")
    paired_rows = read_csv(result_dir / "spatial_jffn_minus_write_bootstrap.csv")

    spatial = grouped_primary(spatial_rows, ("method", "metric"))
    ranking = grouped_primary(ranking_rows, ("metric",))
    specificity = grouped_primary(specificity_rows, ("method", "metric"))
    gain = grouped_primary(gain_rows, ("label", "metric"))
    cancellation = grouped_primary(cancellation_rows, ("label", "metric"))
    payload = {
        "model": model,
        "cohort": token["cohort_audit"],
        "entropy_matching": token["entropy_matching"],
        "spatial_layers_16_32": spatial,
        "spatial_jffn_minus_write_bootstrap": paired_rows,
        "ranking_layers_16_32": ranking,
        "reranking": token["reranking"],
        "same_image_specificity_layers_16_32": specificity,
        "gain_layers_16_32": gain,
        "cancellation_layers_16_32": cancellation,
        "numerical_audit": {
            "max_q_conservation_relative_error": token["max_q_conservation_relative_error"],
            "max_component_sum_relative_error": token["max_component_sum_relative_error"],
        },
        "scalar_incremental": scalar,
        "logit_causal": logit,
        "conceptual_stages": conceptual,
    }
    for name, values in scalar["models"].items():
        metrics_rows.append({
            "model": model, "section": "scalar", "item": name,
            "metric": "auroc", "value": values["auroc"],
        })
        metrics_rows.append({
            "model": model, "section": "scalar", "item": name,
            "metric": "hall_aupr", "value": values["hall_aupr"],
        })
    metrics_rows.extend({
        "model": model, "section": "spatial", "item": key.split("|", 1)[0],
        "metric": key.split("|", 1)[1], "value": value, "layers": "16-32",
    } for key, value in spatial.items())
    metrics_rows.extend({
        "model": model, "section": "ranking", "item": "WRITE_vs_JFFN",
        "metric": key, "value": value, "layers": "16-32",
    } for key, value in ranking.items())
    metrics_rows.extend({
        "model": model, "section": "cancellation", "item": key.split("|", 1)[0],
        "metric": key.split("|", 1)[1], "value": value, "layers": "16-32",
    } for key, value in cancellation.items())
    for row in paired_rows:
        metrics_rows.append({
            "model": model, "section": "spatial_delta", "item": "JFFN-WRITE",
            "metric": row["metric"], "value": row["difference"],
            "ci95_low": row["ci95_low"], "ci95_high": row["ci95_high"],
            "layers": row["layers"],
        })
    if "sample" in logit:
        for metric, values in logit["real_hall"].items():
            for label in ("real", "hall"):
                metrics_rows.append({
                    "model": model, "section": "logit_attribution",
                    "item": label.upper(), "metric": metric,
                    "value": values[f"{label}_mean"],
                })
        for values in logit["intervention"]:
            for metric in ("pearson", "spearman", "sign_accuracy", "regression_slope", "mean_absolute_prediction_error"):
                metrics_rows.append({
                    "model": model, "section": "causal_intervention",
                    "item": f"eta={values['eta']}", "metric": metric,
                    "value": values[metric],
                })
    if "correlations" in conceptual:
        for row in conceptual["correlations"]:
            metrics_rows.append({
                "model": model,
                "section": "conceptual_stage_correlation",
                "item": row["label"],
                "metric": f"{row['left']}|{row['right']}",
                "value": row["pearson"],
            })
    return payload


def representative_heatmaps(config, summaries):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chosen = []
    for model in MODELS:
        path = ROOT / "outputs" / model / EXPERIMENT / "results/jffn_second_round/logit_causal/logit_attribution_token_maps.pt"
        if not path.exists():
            continue
        cases = torch.load(path, map_location="cpu", weights_only=False)
        candidates = [case for case in cases if int(case["label"]) == 1 and float(case["logit_margin_positive"].sum()) > 0]
        if candidates:
            chosen.append((model, candidates[0]))
    if not chosen:
        return None
    fig, axes = plt.subplots(len(chosen), 5, figsize=(15, 3.1 * len(chosen)), squeeze=False)
    names = ("write", "jffn", "q_positive", "logit_margin_positive")
    titles = ("WRITE", "JFFN", "Q+", "LOGIT-margin+")
    for row_index, (model, case) in enumerate(chosen):
        image = Image.open(_image_path(config, int(case["image_id"]))).convert("RGB")
        size = 336 if model == "llava_1_5_7b" else 448
        processed = (
            ImageOps.fit(image, (size, size), method=Image.Resampling.BICUBIC)
            if model == "llava_1_5_7b"
            else image.resize((size, size), Image.Resampling.BICUBIC)
        )
        axes[row_index, 0].imshow(processed); axes[row_index, 0].set_title(f"{model}\nimage {case['image_id']}")
        axes[row_index, 0].axis("off")
        grid_h, grid_w = map(int, case["grid"])
        for column, (name, title) in enumerate(zip(names, titles), 1):
            values = torch.as_tensor(case[name]).float().reshape(grid_h, grid_w).numpy()
            axes[row_index, column].imshow(processed)
            axes[row_index, column].imshow(values, cmap="magma", alpha=.62, extent=(0, size, size, 0), interpolation="nearest")
            axes[row_index, column].set_title(f"{title} (L{case['layer']})")
            axes[row_index, column].axis("off")
    fig.suptitle("Representative model-aligned visual-token maps")
    fig.tight_layout()
    path = ROOT / "jffn_second_round_representative_heatmaps.png"
    fig.savefig(path, dpi=220, bbox_inches="tight"); plt.close(fig)
    return str(path)


def main() -> None:
    metrics_rows: list[dict[str, Any]] = []
    models = {model: summarize_model(model, metrics_rows) for model in MODELS}
    config = load_config("configs/model_configs_inslen_official_target.yaml")
    figure = representative_heatmaps(config, models)
    payload = {
        "protocol": "jffn_second_round_incremental_validation_v1",
        "completed_models_only": list(MODELS),
        "incomplete_models_excluded": ["qwen2_5_vl_7b", "qwen3_vl_8b"],
        "primary_layers_one_based": [16, 32],
        "models": models,
        "representative_heatmap": figure,
    }
    (ROOT / "jffn_second_round_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_csv(ROOT / "jffn_second_round_metrics.csv", metrics_rows)
    print("[summary] wrote JSON/CSV", flush=True)


if __name__ == "__main__":
    main()
