#!/usr/bin/env python3
"""Create honest, high-contrast visualizations for the LLaVA 500-image S study."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.train_old_risk_ev_s_mlp import EXPERIMENT  # noqa: E402
from scripts.train_union_topk_aggregate_s_mlp import (  # noqa: E402
    load_union_aggregate,
)
from scripts.train_union_topk_region_s_mlp import collect_matrices  # noqa: E402


MODEL = "llava_1_5_7b"
AGGREGATE_SHARDS = "results/jffn_union_aggregate_500/shards"
FAIR_RESULTS = "results/jffn_second_round/union_topk_aggregate_s_mlp_500_fair"
COLORS = {"REAL": "#2563eb", "HALL": "#dc2626"}
FEATURES = (
    ("all_aggregate", "All-token aggregate S"),
    ("all_tokenwise", "All-token tokenwise S"),
    ("union_tokenwise", "Union-TopK tokenwise S"),
    ("union_aggregate", "Union-TopK aggregate S"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260830)
    return parser.parse_args()


def _effect(values: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    real = values[labels == 1].astype(np.float64)
    hall = values[labels == 0].astype(np.float64)
    delta = real.mean(axis=0) - hall.mean(axis=0)
    real_var = real.var(axis=0, ddof=1)
    hall_var = hall.var(axis=0, ddof=1)
    pooled = np.sqrt(
        ((len(real) - 1) * real_var + (len(hall) - 1) * hall_var)
        / max(len(real) + len(hall) - 2, 1)
    )
    return delta, delta / np.maximum(pooled, 1.0e-12)


def image_bootstrap_delta(
    values: np.ndarray,
    labels: np.ndarray,
    image_ids: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap REAL-HALL mean differences by resampling whole images."""
    unique = np.unique(image_ids)
    image_lookup = {int(value): index for index, value in enumerate(unique)}
    layers = int(values.shape[1])
    sums = np.zeros((2, len(unique), layers), dtype=np.float64)
    counts = np.zeros((2, len(unique)), dtype=np.float64)
    for row, (label, image_id) in enumerate(zip(labels, image_ids)):
        image_index = image_lookup[int(image_id)]
        sums[int(label), image_index] += values[row]
        counts[int(label), image_index] += 1.0
    rng = np.random.default_rng(int(seed))
    differences: list[np.ndarray] = []
    batch_size = 500
    for start in range(0, int(replicates), batch_size):
        size = min(batch_size, int(replicates) - start)
        weights = rng.multinomial(
            len(unique), np.full(len(unique), 1.0 / len(unique)), size=size
        ).astype(np.float64)
        real_count = weights @ counts[1]
        hall_count = weights @ counts[0]
        valid = (real_count > 0.0) & (hall_count > 0.0)
        real_mean = (weights[valid] @ sums[1]) / real_count[valid, None]
        hall_mean = (weights[valid] @ sums[0]) / hall_count[valid, None]
        differences.append(real_mean - hall_mean)
    samples = np.concatenate(differences, axis=0)
    if len(samples) < int(replicates) * 0.99:
        raise AssertionError("Too many invalid image bootstrap replicates")
    return (
        np.percentile(samples, 2.5, axis=0),
        np.percentile(samples, 97.5, axis=0),
    )


def _mean_ci(values: np.ndarray, labels: np.ndarray, label: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = values[labels == label].astype(np.float64)
    mean = selected.mean(axis=0)
    sem = selected.std(axis=0, ddof=1) / math.sqrt(len(selected))
    return mean, mean - 1.96 * sem, mean + 1.96 * sem


def _save(fig: plt.Figure, output_dir: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_raw_and_zoom(
    feature_values: Mapping[str, np.ndarray],
    labels: np.ndarray,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(15, 15), sharex="col")
    layers = np.arange(1, next(iter(feature_values.values())).shape[1] + 1)
    for row_index, (key, title) in enumerate(FEATURES):
        statistics = {
            name: _mean_ci(feature_values[key], labels, label)
            for label, name in ((1, "REAL"), (0, "HALL"))
        }
        for column, (start, end, subtitle) in enumerate(
            ((1, len(layers), "Full layers"), (4, min(30, len(layers)), "L4-L30 zoom"))
        ):
            axis = axes[row_index, column]
            region = (layers >= start) & (layers <= end)
            local_values: list[float] = []
            for name in ("REAL", "HALL"):
                mean, low, high = statistics[name]
                axis.plot(layers[region], mean[region], color=COLORS[name], linewidth=2.0, label=name)
                axis.fill_between(
                    layers[region], low[region], high[region], color=COLORS[name], alpha=0.16
                )
                local_values.extend(low[region].tolist())
                local_values.extend(high[region].tolist())
            if column == 1:
                low_value, high_value = np.percentile(local_values, [1, 99])
                margin = max((high_value - low_value) * 0.12, 1.0e-3)
                axis.set_ylim(low_value - margin, high_value + margin)
            axis.set_title(f"{title}: {subtitle}")
            axis.grid(alpha=0.25)
            axis.legend()
            axis.set_ylabel("Mean S")
    axes[-1, 0].set_xlabel("Decoder layer")
    axes[-1, 1].set_xlabel("Decoder layer")
    fig.suptitle("LLaVA-1.5-7B, 500 images: raw S curves and disclosed zoom", y=1.002)
    _save(fig, output_dir, "s_raw_curves_full_and_zoom")


def plot_difference_effect_auc(
    feature_values: Mapping[str, np.ndarray],
    labels: np.ndarray,
    image_ids: np.ndarray,
    *,
    replicates: int,
    seed: int,
    output_dir: Path,
) -> list[dict[str, Any]]:
    fig, axes = plt.subplots(4, 3, figsize=(17, 14), sharex=True)
    layers = np.arange(1, next(iter(feature_values.values())).shape[1] + 1)
    rows: list[dict[str, Any]] = []
    for row_index, (key, title) in enumerate(FEATURES):
        values = feature_values[key]
        delta, effect = _effect(values, labels)
        ci_low, ci_high = image_bootstrap_delta(
            values,
            labels,
            image_ids,
            replicates=int(replicates),
            seed=int(seed) + row_index,
        )
        auc = np.asarray(
            [roc_auc_score(labels, values[:, layer]) for layer in range(values.shape[1])]
        )
        axis = axes[row_index, 0]
        axis.axhline(0.0, color="#111827", linewidth=1.0)
        axis.errorbar(
            layers,
            delta,
            yerr=np.stack((delta - ci_low, ci_high - delta)),
            fmt="o-",
            markersize=3.5,
            linewidth=1.5,
            color="#7c3aed",
            ecolor="#a78bfa",
            capsize=2,
        )
        axis.fill_between(layers, 0.0, delta, where=delta >= 0.0, color="#2563eb", alpha=0.12)
        axis.fill_between(layers, 0.0, delta, where=delta < 0.0, color="#dc2626", alpha=0.12)
        axis.set_ylabel(f"{title}\nREAL - HALL")
        axis.grid(alpha=0.25)

        axis = axes[row_index, 1]
        axis.axhline(0.0, color="#111827", linewidth=1.0)
        axis.bar(
            layers,
            effect,
            color=np.where(effect >= 0.0, "#2563eb", "#dc2626"),
            alpha=0.82,
            width=0.8,
        )
        axis.set_ylabel("Cohen's d")
        axis.grid(axis="y", alpha=0.25)

        axis = axes[row_index, 2]
        axis.axhline(0.5, color="#6b7280", linestyle="--", linewidth=1.2, label="chance")
        axis.plot(layers, auc, color="#059669", marker="o", markersize=3.0, linewidth=1.7)
        axis.fill_between(layers, 0.5, auc, color="#10b981", alpha=0.14)
        axis.set_ylabel("Single-layer AUROC\n(REAL positive)")
        axis.set_ylim(min(0.45, float(auc.min()) - 0.02), max(0.75, float(auc.max()) + 0.02))
        axis.grid(alpha=0.25)
        for layer_index in range(values.shape[1]):
            rows.append(
                {
                    "feature": key,
                    "layer": layer_index + 1,
                    "real_minus_hall": float(delta[layer_index]),
                    "image_bootstrap_ci95_low": float(ci_low[layer_index]),
                    "image_bootstrap_ci95_high": float(ci_high[layer_index]),
                    "cohens_d_real_minus_hall": float(effect[layer_index]),
                    "single_layer_real_positive_auroc": float(auc[layer_index]),
                }
            )
    for column, title in enumerate(
        ("Mean difference with image-bootstrap 95% CI", "Standardized effect", "Univariate separability")
    ):
        axes[0, column].set_title(title)
        axes[-1, column].set_xlabel("Decoder layer")
    fig.suptitle("LLaVA-1.5-7B, 500 images: class separation without absolute-scale compression", y=1.002)
    _save(fig, output_dir, "s_difference_effect_and_single_layer_auc")
    return rows


def plot_representative_distributions(
    train_values: Mapping[str, np.ndarray],
    train_labels: np.ndarray,
    test_values: Mapping[str, np.ndarray],
    test_labels: np.ndarray,
    output_dir: Path,
) -> dict[str, list[int]]:
    fig, axes = plt.subplots(4, 3, figsize=(14, 14))
    selected_layers: dict[str, list[int]] = {}
    for row_index, (key, title) in enumerate(FEATURES):
        _delta, train_effect = _effect(train_values[key], train_labels)
        strongest = int(np.argmax(np.abs(train_effect))) + 1
        layers = []
        for layer in (strongest, 16, 24):
            if layer not in layers:
                layers.append(layer)
        while len(layers) < 3:
            candidate = len(layers) + 1
            if candidate not in layers:
                layers.append(candidate)
        selected_layers[key] = layers
        for column, layer in enumerate(layers):
            axis = axes[row_index, column]
            values = test_values[key][:, layer - 1]
            real = values[test_labels == 1]
            hall = values[test_labels == 0]
            violin = axis.violinplot(
                (real, hall), positions=(1, 2), widths=0.82, showmeans=True, showextrema=False
            )
            for body, color in zip(violin["bodies"], (COLORS["REAL"], COLORS["HALL"])):
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_alpha(0.32)
            violin["cmeans"].set_color("#111827")
            box = axis.boxplot(
                (real, hall),
                positions=(1, 2),
                widths=0.24,
                showfliers=False,
                patch_artist=True,
                medianprops={"color": "#111827", "linewidth": 1.5},
            )
            for patch, color in zip(box["boxes"], (COLORS["REAL"], COLORS["HALL"])):
                patch.set_facecolor(color)
                patch.set_alpha(0.55)
            axis.set_xticks((1, 2), ("REAL", "HALL"))
            axis.set_title(
                f"{title}\nL{layer}" + (" (selected on train)" if column == 0 else " (fixed)")
            )
            axis.grid(axis="y", alpha=0.25)
            axis.set_ylabel("S on held-out test")
    fig.suptitle(
        "Held-out distributions; strongest layer selected using train labels only",
        y=1.002,
    )
    _save(fig, output_dir, "s_representative_test_distributions")
    return selected_layers


def plot_classifier_metrics(result_payload: Mapping[str, Any], output_dir: Path) -> None:
    keys = (
        "old_risk_ev",
        "old_risk_ev_old_aggregate_s",
        "old_risk_ev_token_all_s",
        "old_risk_ev_union_tokenwise_s",
        "union_aggregate_s",
    )
    labels = ("risk+EV", "+all agg", "+all token", "+union token", "+union agg")
    metrics = (
        ("auroc", "Mean AUROC"),
        ("hall_aupr", "Mean HALL AUPR"),
        ("hall_f1", "Mean HALL F1"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    colors = ("#6b7280", "#f59e0b", "#10b981", "#2563eb", "#7c3aed")
    for axis, (metric, title) in zip(axes, metrics):
        means = [result_payload["comparisons"][key]["aggregate"][metric]["mean"] for key in keys]
        stds = [result_payload["comparisons"][key]["aggregate"][metric]["std"] for key in keys]
        x = np.arange(len(keys))
        axis.bar(x, means, yerr=stds, color=colors, alpha=0.86, capsize=4)
        axis.set_xticks(x, labels, rotation=24, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        lower = max(0.0, min(means) - max(0.04, 2.0 * max(stds)))
        upper = min(1.0, max(means) + max(0.04, 2.0 * max(stds)))
        axis.set_ylim(lower, upper)
        for index, value in enumerate(means):
            axis.text(index, value + 0.006, f"{value:.3f}", ha="center", fontsize=9)
    fig.suptitle("Same 500-image cohort, same 3-layer MLP, seeds 43/44/45", y=1.01)
    _save(fig, output_dir, "s_classifier_metric_comparison")


def main() -> None:
    args = parse_args()
    model_root = Path(args.outputs_root) / MODEL / EXPERIMENT
    output_dir = model_root / FAIR_RESULTS / "clear_visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    matrices, _curves, _audit = collect_matrices(model_root, MODEL)
    aggregate, aggregate_audit = load_union_aggregate(
        model_root / AGGREGATE_SHARDS, matrices, allow_subset=True
    )
    split_values: dict[str, dict[str, np.ndarray]] = {}
    for split in ("train", "test"):
        split_values[split] = {
            "all_aggregate": matrices[split]["old_aggregate_s"],
            "all_tokenwise": matrices[split]["token_all_s"],
            "union_tokenwise": matrices[split]["union_topk_s"],
            "union_aggregate": aggregate[split]["gain"],
        }
    values = {
        key: np.concatenate((split_values["train"][key], split_values["test"][key]))
        for key, _title in FEATURES
    }
    labels = np.concatenate((matrices["train"]["y"], matrices["test"]["y"]))
    image_ids = np.concatenate(
        (matrices["train"]["image_ids"], matrices["test"]["image_ids"])
    )
    plot_raw_and_zoom(values, labels, output_dir)
    effect_rows = plot_difference_effect_auc(
        values,
        labels,
        image_ids,
        replicates=int(args.bootstrap_replicates),
        seed=int(args.bootstrap_seed),
        output_dir=output_dir,
    )
    selected = plot_representative_distributions(
        split_values["train"],
        matrices["train"]["y"],
        split_values["test"],
        matrices["test"]["y"],
        output_dir,
    )
    results = json.loads((model_root / FAIR_RESULTS / "results.json").read_text())
    plot_classifier_metrics(results, output_dir)
    with (output_dir / "s_layer_effect_statistics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(effect_rows[0]))
        writer.writeheader()
        writer.writerows(effect_rows)
    (output_dir / "visualization_audit.json").write_text(
        json.dumps(
            {
                "model": MODEL,
                "cohort": "global first 500 images; descriptive curves use all mentions",
                "train_images": int(np.unique(matrices["train"]["image_ids"]).size),
                "test_images": int(np.unique(matrices["test"]["image_ids"]).size),
                "train_mentions": int(len(matrices["train"]["y"])),
                "test_mentions": int(len(matrices["test"]["y"])),
                "bootstrap_unit": "image",
                "bootstrap_replicates": int(args.bootstrap_replicates),
                "representative_layer_selection": "maximum abs Cohen d on train only; L16/L24 fixed",
                "selected_layers": selected,
                "aggregate_audit": aggregate_audit,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[S plots] wrote {output_dir}")


if __name__ == "__main__":
    main()
