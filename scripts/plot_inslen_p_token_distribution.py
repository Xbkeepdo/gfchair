#!/usr/bin/env python3
"""Plot the visual-token distribution of the two active DGST source marginals P."""

from __future__ import annotations

import argparse
import csv
import gc
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.io_utils import load_pkl


HALL = 0
REAL = 1
LABEL_NAMES = {HALL: "Hallucination", REAL: "Real / non-hallucination"}
LABEL_SLUGS = {HALL: "hall", REAL: "real"}
LABEL_COLORS = {HALL: "#d55e00", REAL: "#0072b2"}
SOURCE_SPECS = (
    (
        "hmid_cos",
        "P=hmid",
        "dgst_t_source_dist_per_layer",
    ),
    (
        "hpre_cos",
        "P=hpre",
        "dgst_t_source_hpre_cos_dist_per_layer",
    ),
)
TOP_KS = (1, 5, 10, 32)
METRIC_NAMES = (
    "p50",
    "p90",
    "p99",
    "pmax",
    "relative_p50",
    "relative_p90",
    "relative_p99",
    "relative_pmax",
    "top1_mass",
    "top5_mass",
    "top10_mass",
    "top32_mass",
    "normalized_entropy",
    "effective_token_fraction",
    "mass_above_uniform",
    "fraction_above_uniform",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--feature-files",
        nargs="+",
        default=None,
        help=(
            "Feature pickle files to aggregate. By default, features.part*.pkl "
            "are used when present; otherwise features.pkl is used."
        ),
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Defaults to <output-dir>/results/feature_curves.",
    )
    parser.add_argument("--hist-min-log10", type=float, default=-12.0)
    parser.add_argument("--hist-max-log10", type=float, default=6.0)
    parser.add_argument("--hist-bins", type=int, default=360)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.hist_bins < 20:
        raise ValueError("--hist-bins must be at least 20.")
    if args.hist_min_log10 >= args.hist_max_log10:
        raise ValueError("Histogram minimum must be smaller than its maximum.")

    output_dir = Path(args.output_dir)
    feature_files = _resolve_feature_files(output_dir, args.feature_files)
    results_dir = (
        Path(args.results_dir)
        if args.results_dir
        else output_dir / "results" / "feature_curves"
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    hist_edges = np.linspace(
        args.hist_min_log10,
        args.hist_max_log10,
        args.hist_bins + 1,
        dtype=np.float64,
    )

    accum, rows_per_file = _aggregate(feature_files, hist_edges)
    stats = _finalize(accum)
    stem = f"{args.model}_inslen_p_visual_token_distribution"
    metrics_csv = results_dir / f"{stem}_metrics.csv"
    histogram_csv = results_dir / f"{stem}_histogram.csv"
    metrics_png = results_dir / f"{stem}_metrics.png"
    metrics_pdf = results_dir / f"{stem}_metrics.pdf"
    heatmap_png = results_dir / f"{stem}_histogram_heatmap.png"
    heatmap_pdf = results_dir / f"{stem}_histogram_heatmap.pdf"
    markdown_path = results_dir / f"{stem}.md"

    _write_metrics_csv(args.model, stats, metrics_csv)
    _write_histogram_csv(args.model, accum, hist_edges, histogram_csv)
    _plot_metrics(args.model, stats, metrics_png, metrics_pdf)
    _plot_histogram_heatmap(
        args.model,
        accum,
        hist_edges,
        heatmap_png,
        heatmap_pdf,
    )
    _write_markdown(
        args.model,
        feature_files,
        rows_per_file,
        stats,
        metrics_csv,
        histogram_csv,
        metrics_png,
        heatmap_png,
        markdown_path,
    )

    print("markdown", markdown_path.resolve())
    print("metrics_csv", metrics_csv.resolve())
    print("histogram_csv", histogram_csv.resolve())
    print("metrics_png", metrics_png.resolve())
    print("metrics_pdf", metrics_pdf.resolve())
    print("heatmap_png", heatmap_png.resolve())
    print("heatmap_pdf", heatmap_pdf.resolve())
    for source_slug, source_title, _key in SOURCE_SPECS:
        hall = stats[(source_slug, HALL)]
        real = stats[(source_slug, REAL)]
        entropy_gap = (
            hall["metrics"]["normalized_entropy"]["mean"]
            - real["metrics"]["normalized_entropy"]["mean"]
        )
        peak = int(np.argmax(np.abs(entropy_gap)))
        print(
            f"{source_title}: n_hall={hall['n_samples'][0]} "
            f"n_real={real['n_samples'][0]} "
            f"mean_entropy_gap={np.mean(entropy_gap):+.8f} "
            f"peak_layer={peak + 1} peak_gap={entropy_gap[peak]:+.8f}"
        )


def _resolve_feature_files(output_dir: Path, values: list[str] | None) -> list[Path]:
    if values:
        paths = [Path(value) for value in values]
    else:
        parts = sorted(output_dir.glob("features.part*.pkl"))
        paths = parts if parts else [output_dir / "features.pkl"]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing feature file(s): " + ", ".join(missing))
    return paths


def _as_matrix(
    value: object,
    key: str,
    path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(
            f"{key} in {path} must be a non-empty [layers, visual_tokens] matrix; "
            f"got {matrix.shape}."
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"Non-finite P values in {key} from {path}.")
    if float(matrix.min()) < -1.0e-7:
        raise ValueError(f"Negative P values in {key} from {path}.")
    matrix = np.maximum(matrix, 0.0)
    row_sums = matrix.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError(f"Zero-mass P row in {key} from {path}.")
    row_sum_error = np.abs(row_sums[:, 0] - 1.0)
    if float(np.max(row_sum_error)) > 1.0e-4:
        raise ValueError(
            f"P rows in {key} from {path} are not normalized; "
            f"max |sum(P)-1|={float(np.max(row_sum_error)):.6g}."
        )
    return matrix / row_sums, row_sum_error


def _new_accumulator(layers: int, hist_bins: int) -> dict:
    return {
        "n_samples": np.zeros(layers, dtype=np.int64),
        "n_tokens": np.zeros(layers, dtype=np.int64),
        "n_visual_sum": np.zeros(layers, dtype=np.float64),
        "n_visual_min": np.full(layers, np.iinfo(np.int64).max, dtype=np.int64),
        "n_visual_max": np.zeros(layers, dtype=np.int64),
        "token_sum": np.zeros(layers, dtype=np.float64),
        "token_sum_sq": np.zeros(layers, dtype=np.float64),
        "metric_sum": {
            name: np.zeros(layers, dtype=np.float64) for name in METRIC_NAMES
        },
        "metric_sum_sq": {
            name: np.zeros(layers, dtype=np.float64) for name in METRIC_NAMES
        },
        "hist": np.zeros((layers, hist_bins), dtype=np.int64),
        "max_row_sum_error": np.zeros(layers, dtype=np.float64),
    }


def _sample_metrics(p: np.ndarray) -> dict[str, np.ndarray]:
    layers, n_visual = p.shape
    sorted_p = np.sort(p, axis=1)

    def quantile(q: float) -> np.ndarray:
        position = (n_visual - 1) * q
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        fraction = position - lower
        return (
            sorted_p[:, lower] * (1.0 - fraction)
            + sorted_p[:, upper] * fraction
        )

    quantiles = np.stack(
        (quantile(0.50), quantile(0.90), quantile(0.99)),
        axis=1,
    )
    metrics = {
        "p50": quantiles[:, 0],
        "p90": quantiles[:, 1],
        "p99": quantiles[:, 2],
        "pmax": sorted_p[:, -1],
        "relative_p50": quantiles[:, 0] * n_visual,
        "relative_p90": quantiles[:, 1] * n_visual,
        "relative_p99": quantiles[:, 2] * n_visual,
        "relative_pmax": sorted_p[:, -1] * n_visual,
    }
    for top_k in TOP_KS:
        active_k = min(top_k, n_visual)
        mass = sorted_p[:, -active_k:].sum(axis=1)
        metrics[f"top{top_k}_mass"] = mass

    entropy = -np.sum(p * np.log(np.maximum(p, np.finfo(np.float64).tiny)), axis=1)
    if n_visual > 1:
        normalized_entropy = entropy / math.log(n_visual)
    else:
        normalized_entropy = np.zeros(layers, dtype=np.float64)
    uniform = 1.0 / n_visual
    above = p > uniform
    metrics["normalized_entropy"] = normalized_entropy
    metrics["effective_token_fraction"] = np.exp(entropy) / n_visual
    metrics["mass_above_uniform"] = np.sum(np.where(above, p, 0.0), axis=1)
    metrics["fraction_above_uniform"] = above.mean(axis=1)
    return metrics


def _update_accumulator(
    item: dict,
    p: np.ndarray,
    hist_edges: np.ndarray,
    row_sum_error: np.ndarray | None = None,
) -> None:
    layers, n_visual = p.shape
    item["n_samples"] += 1
    item["n_tokens"] += n_visual
    item["n_visual_sum"] += n_visual
    item["n_visual_min"] = np.minimum(item["n_visual_min"], n_visual)
    item["n_visual_max"] = np.maximum(item["n_visual_max"], n_visual)
    item["token_sum"] += p.sum(axis=1)
    item["token_sum_sq"] += np.square(p).sum(axis=1)
    item["max_row_sum_error"] = np.maximum(
        item["max_row_sum_error"],
        (
            np.abs(p.sum(axis=1) - 1.0)
            if row_sum_error is None
            else row_sum_error
        ),
    )

    metrics = _sample_metrics(p)
    for name, values in metrics.items():
        item["metric_sum"][name] += values
        item["metric_sum_sq"][name] += np.square(values)

    relative = p * n_visual
    log_relative = np.log10(np.maximum(relative, 10.0 ** hist_edges[0]))
    bin_indices = np.searchsorted(hist_edges, log_relative, side="right") - 1
    bin_indices = np.clip(bin_indices, 0, len(hist_edges) - 2)
    layer_offsets = (
        np.arange(layers, dtype=np.int64)[:, None] * (len(hist_edges) - 1)
    )
    counts = np.bincount(
        (bin_indices + layer_offsets).reshape(-1),
        minlength=layers * (len(hist_edges) - 1),
    ).reshape(layers, len(hist_edges) - 1)
    item["hist"] += counts


def _aggregate(
    feature_files: list[Path],
    hist_edges: np.ndarray,
) -> tuple[dict, dict[str, int]]:
    accum: dict[tuple[str, int], dict] = {}
    rows_per_file: dict[str, int] = {}
    expected_layers = None
    total_binary_rows = 0
    for path in feature_files:
        rows = load_pkl(str(path))
        if not isinstance(rows, list):
            raise TypeError(f"{path} must contain a list of feature records.")
        rows_per_file[str(path)] = len(rows)
        for row in rows:
            label = row.get("label")
            if label not in (HALL, REAL):
                continue
            total_binary_rows += 1
            for source_slug, _source_title, key in SOURCE_SPECS:
                if key not in row:
                    raise KeyError(f"Missing {key} in a binary-labeled row from {path}.")
                matrix, row_sum_error = _as_matrix(row[key], key, path)
                layers = int(matrix.shape[0])
                if expected_layers is None:
                    expected_layers = layers
                elif layers != expected_layers:
                    raise ValueError(
                        f"Layer count changed from {expected_layers} to {layers} "
                        f"for {key} in {path}."
                    )
                acc_key = (source_slug, int(label))
                if acc_key not in accum:
                    accum[acc_key] = _new_accumulator(
                        layers,
                        len(hist_edges) - 1,
                    )
                _update_accumulator(
                    accum[acc_key],
                    matrix,
                    hist_edges,
                    row_sum_error=row_sum_error,
                )
        print(f"processed {path}: rows={len(rows)}", flush=True)
        del rows
        gc.collect()

    if total_binary_rows == 0:
        raise ValueError("No binary-labeled feature rows were found.")
    missing_groups = [
        (source_slug, label)
        for source_slug, _title, _key in SOURCE_SPECS
        for label in (HALL, REAL)
        if (source_slug, label) not in accum
    ]
    if missing_groups:
        raise ValueError(f"Missing source/label groups: {missing_groups}")
    return accum, rows_per_file


def _finalize(accum: dict) -> dict:
    stats = {}
    for key, item in accum.items():
        n = item["n_samples"].astype(np.float64)
        if np.any(n < 2):
            raise ValueError(f"Need at least two samples per layer for {key}.")
        metric_stats = {}
        for name in METRIC_NAMES:
            mean = item["metric_sum"][name] / n
            variance = np.maximum(
                (item["metric_sum_sq"][name] - n * np.square(mean)) / (n - 1.0),
                0.0,
            )
            metric_stats[name] = {
                "mean": mean,
                "std": np.sqrt(variance),
                "sem": np.sqrt(variance / n),
            }
        token_mean = item["token_sum"] / item["n_tokens"]
        token_variance = np.maximum(
            item["token_sum_sq"] / item["n_tokens"] - np.square(token_mean),
            0.0,
        )
        stats[key] = {
            "n_samples": item["n_samples"],
            "n_tokens": item["n_tokens"],
            "n_visual_mean": item["n_visual_sum"] / n,
            "n_visual_min": item["n_visual_min"],
            "n_visual_max": item["n_visual_max"],
            "token_mean": token_mean,
            "token_std": np.sqrt(token_variance),
            "max_row_sum_error": item["max_row_sum_error"],
            "metrics": metric_stats,
        }
    return stats


def _write_metrics_csv(model: str, stats: dict, path: Path) -> None:
    fieldnames = [
        "model",
        "source_mode",
        "label",
        "layer",
        "n_samples",
        "n_tokens",
        "n_visual_mean",
        "n_visual_min",
        "n_visual_max",
        "token_mean",
        "token_std",
        "max_row_sum_error",
    ]
    for name in METRIC_NAMES:
        fieldnames.extend((f"{name}_mean", f"{name}_std", f"{name}_sem"))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for source_slug, _source_title, _key in SOURCE_SPECS:
            for label in (HALL, REAL):
                item = stats[(source_slug, label)]
                layers = len(item["n_samples"])
                for index in range(layers):
                    row = {
                        "model": model,
                        "source_mode": source_slug,
                        "label": LABEL_SLUGS[label],
                        "layer": index + 1,
                        "n_samples": int(item["n_samples"][index]),
                        "n_tokens": int(item["n_tokens"][index]),
                        "n_visual_mean": f"{item['n_visual_mean'][index]:.10g}",
                        "n_visual_min": int(item["n_visual_min"][index]),
                        "n_visual_max": int(item["n_visual_max"][index]),
                        "token_mean": f"{item['token_mean'][index]:.10g}",
                        "token_std": f"{item['token_std'][index]:.10g}",
                        "max_row_sum_error": (
                            f"{item['max_row_sum_error'][index]:.10g}"
                        ),
                    }
                    for name in METRIC_NAMES:
                        for statistic in ("mean", "std", "sem"):
                            row[f"{name}_{statistic}"] = (
                                f"{item['metrics'][name][statistic][index]:.10g}"
                            )
                    writer.writerow(row)


def _write_histogram_csv(
    model: str,
    accum: dict,
    hist_edges: np.ndarray,
    path: Path,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "model",
                "source_mode",
                "label",
                "layer",
                "log10_np_lower",
                "log10_np_upper",
                "count",
                "probability_mass",
            ]
        )
        for source_slug, _source_title, _key in SOURCE_SPECS:
            for label in (HALL, REAL):
                hist = accum[(source_slug, label)]["hist"]
                totals = hist.sum(axis=1)
                for layer in range(hist.shape[0]):
                    for bin_index, count in enumerate(hist[layer]):
                        writer.writerow(
                            [
                                model,
                                source_slug,
                                LABEL_SLUGS[label],
                                layer + 1,
                                f"{hist_edges[bin_index]:.10g}",
                                f"{hist_edges[bin_index + 1]:.10g}",
                                int(count),
                                f"{count / totals[layer]:.10g}",
                            ]
                        )


def _plot_metrics(
    model: str,
    stats: dict,
    png_path: Path,
    pdf_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(23, 10), sharex=True)
    percentile_styles = {
        "p50": (":", "P50"),
        "p90": ("--", "P90"),
        "p99": ("-.", "P99"),
        "pmax": ("-", "max"),
    }
    topk_styles = {
        "top1_mass": ("-", "Top-1"),
        "top5_mass": ("--", "Top-5"),
        "top32_mass": (":", "Top-32"),
    }
    concentration_styles = {
        "normalized_entropy": ("-", "normalized entropy"),
        "effective_token_fraction": ("--", "effective-token fraction"),
        "mass_above_uniform": ("-.", "mass above uniform"),
    }
    for row_index, (source_slug, source_title, _key) in enumerate(SOURCE_SPECS):
        example = stats[(source_slug, HALL)]
        layers = np.arange(1, len(example["n_samples"]) + 1)
        for label in (HALL, REAL):
            color = LABEL_COLORS[label]
            item = stats[(source_slug, label)]
            for name, (linestyle, short_name) in percentile_styles.items():
                axes[row_index, 0].plot(
                    layers,
                    np.maximum(item["metrics"][name]["mean"], 1.0e-12),
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.7,
                    label=f"{LABEL_SLUGS[label]} {short_name}",
                )
                relative_name = f"relative_{name}"
                axes[row_index, 1].plot(
                    layers,
                    np.maximum(
                        item["metrics"][relative_name]["mean"], 1.0e-12
                    ),
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.7,
                    label=f"{LABEL_SLUGS[label]} {short_name}",
                )
            for name, (linestyle, short_name) in topk_styles.items():
                axes[row_index, 2].plot(
                    layers,
                    item["metrics"][name]["mean"],
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.8,
                    label=f"{LABEL_SLUGS[label]} {short_name}",
                )
            for name, (linestyle, short_name) in concentration_styles.items():
                axes[row_index, 3].plot(
                    layers,
                    item["metrics"][name]["mean"],
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.8,
                    label=f"{LABEL_SLUGS[label]} {short_name}",
                )

        axes[row_index, 0].set_title(f"{source_title}: raw visual-token P")
        axes[row_index, 0].set_yscale("log")
        axes[row_index, 0].set_ylabel("Probability")
        axes[row_index, 1].set_title(f"{source_title}: relative value N_v P")
        axes[row_index, 1].set_yscale("log")
        axes[row_index, 1].axhline(1.0, color="black", alpha=0.25, linewidth=1)
        axes[row_index, 1].set_ylabel("Relative to uniform")
        axes[row_index, 2].set_title(f"{source_title}: cumulative Top-K mass")
        axes[row_index, 2].set_ylabel("Probability mass")
        axes[row_index, 2].set_ylim(0.0, 1.02)
        axes[row_index, 3].set_title(f"{source_title}: concentration")
        axes[row_index, 3].set_ylabel("Value")
        axes[row_index, 3].set_ylim(0.0, 1.02)
        for column in range(4):
            ax = axes[row_index, column]
            ax.set_xlabel("Layer")
            ax.set_xlim(1, len(layers))
            ax.grid(True, alpha=0.22)
            ax.legend(frameon=False, fontsize=7, ncol=2)

    fig.suptitle(
        f"{model}: P values across visual tokens (sample-level means by CHAIR label)",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def _plot_histogram_heatmap(
    model: str,
    accum: dict,
    hist_edges: np.ndarray,
    png_path: Path,
    pdf_path: Path,
) -> None:
    combined = sum(
        (item["hist"].sum(axis=0) for item in accum.values()),
        start=np.zeros(len(hist_edges) - 1, dtype=np.int64),
    )
    occupied = np.flatnonzero(combined > 0)
    lower = max(int(occupied[0]) - 2, 0)
    upper = min(int(occupied[-1]) + 3, len(hist_edges) - 1)
    plot_edges = hist_edges[lower : upper + 1]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True)
    log_probability_by_group = {}
    for source_slug, _source_title, _key in SOURCE_SPECS:
        for label in (HALL, REAL):
            hist = accum[(source_slug, label)]["hist"][:, lower:upper]
            totals = np.maximum(hist.sum(axis=1, keepdims=True), 1)
            log_probability_by_group[(source_slug, label)] = np.log10(
                hist / totals + 1.0e-10
            ).T
    common_vmax = max(
        -1.0,
        max(float(np.nanmax(values)) for values in log_probability_by_group.values()),
    )
    image = None
    for row_index, (source_slug, source_title, _key) in enumerate(SOURCE_SPECS):
        for column, label in enumerate((HALL, REAL)):
            hist = accum[(source_slug, label)]["hist"][:, lower:upper]
            log_probability = log_probability_by_group[(source_slug, label)]
            layers = hist.shape[0]
            image = axes[row_index, column].pcolormesh(
                np.arange(0.5, layers + 1.5),
                plot_edges,
                log_probability,
                shading="auto",
                cmap="viridis",
                vmin=-6.0,
                vmax=common_vmax,
            )
            axes[row_index, column].axhline(
                0.0,
                color="white",
                linestyle="--",
                linewidth=1.0,
                alpha=0.8,
            )
            axes[row_index, column].set_title(
                f"{source_title} · {LABEL_NAMES[label]}"
            )
            axes[row_index, column].set_xlabel("Layer")
            axes[row_index, column].set_ylabel("log10(N_v P_i)")
    fig.suptitle(
        f"{model}: visual-token P distribution by layer (uniform baseline: N_v P_i=1)",
        fontsize=14,
    )
    fig.subplots_adjust(
        left=0.07,
        right=0.88,
        bottom=0.07,
        top=0.91,
        wspace=0.16,
        hspace=0.22,
    )
    if image is not None:
        colorbar_axis = fig.add_axes([0.91, 0.15, 0.015, 0.70])
        colorbar = fig.colorbar(image, cax=colorbar_axis)
        colorbar.set_label("log10(token fraction in bin)")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def _mean_across_layers(item: dict, metric: str) -> float:
    return float(np.mean(item["metrics"][metric]["mean"]))


def _write_markdown(
    model: str,
    feature_files: list[Path],
    rows_per_file: dict[str, int],
    stats: dict,
    metrics_csv: Path,
    histogram_csv: Path,
    metrics_png: Path,
    heatmap_png: Path,
    path: Path,
) -> None:
    lines = [
        f"# {model}：P 在视觉 token 上的分布统计",
        "",
        "- `P=hmid`：`softmax(cos(o_ffn(q_t), hmid(v_i)) / 0.07)`。",
        "- `P=hpre`：`softmax(cos(o_ffn(q_t), hpre(v_i)) / 0.07)`。",
        "- 每个目标对象、每层作为一个样本；`0=Hall`、`1=Real`。",
        "- 原始 P 的 token 均值恒为约 `1/N_v`，因此同时统计 `N_v P_i`；其均匀分布基线为 1。",
        "- 曲线统计对每个正式对象等权；直方图对每个视觉 token 等权。",
        "- raw/softmax target gate 与 sqrt/cosine transport cost 不改变 P，故没有重复绘制。",
        "",
        "## 输入与产物",
        "",
    ]
    for feature_file in feature_files:
        lines.append(
            f"- `{feature_file}`：{rows_per_file[str(feature_file)]:,} 条。"
        )
    lines.extend(
        [
            f"- 逐层指标：`{metrics_csv.name}`。",
            f"- token 直方图：`{histogram_csv.name}`。",
            f"- 指标曲线：`{metrics_png.name}`。",
            f"- 分布热图：`{heatmap_png.name}`。",
            "",
            "## 全层平均",
            "",
            "| P source | 标签 | 样本数 | 平均视觉 token 数 | token 数范围 | Pmax | N_v Pmax | Top-32 mass | 归一化熵 | 有效 token 比例 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for source_slug, source_title, _key in SOURCE_SPECS:
        for label in (HALL, REAL):
            item = stats[(source_slug, label)]
            n_min = int(np.min(item["n_visual_min"]))
            n_max = int(np.max(item["n_visual_max"]))
            lines.append(
                f"| {source_title} | {LABEL_SLUGS[label]} | "
                f"{int(item['n_samples'][0]):,} | "
                f"{float(np.mean(item['n_visual_mean'])):.2f} | {n_min}–{n_max} | "
                f"{_mean_across_layers(item, 'pmax'):.6f} | "
                f"{_mean_across_layers(item, 'relative_pmax'):.6f} | "
                f"{_mean_across_layers(item, 'top32_mass'):.6f} | "
                f"{_mean_across_layers(item, 'normalized_entropy'):.6f} | "
                f"{_mean_across_layers(item, 'effective_token_fraction'):.6f} |"
            )

    lines.extend(
        [
            "",
            "## Hall − Real",
            "",
            "| P source | Pmax | N_v Pmax | Top-32 mass | 归一化熵 | 有效 token 比例 | 最大熵差层 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for source_slug, source_title, _key in SOURCE_SPECS:
        hall = stats[(source_slug, HALL)]
        real = stats[(source_slug, REAL)]
        gaps = {}
        for metric in (
            "pmax",
            "relative_pmax",
            "top1_mass",
            "top32_mass",
            "normalized_entropy",
            "effective_token_fraction",
        ):
            gaps[metric] = (
                hall["metrics"][metric]["mean"]
                - real["metrics"][metric]["mean"]
            )
        peak = int(np.argmax(np.abs(gaps["normalized_entropy"])))
        lines.append(
            f"| {source_title} | {np.mean(gaps['pmax']):+.6f} | "
            f"{np.mean(gaps['relative_pmax']):+.6f} | "
            f"{np.mean(gaps['top32_mass']):+.6f} | "
            f"{np.mean(gaps['normalized_entropy']):+.6f} | "
            f"{np.mean(gaps['effective_token_fraction']):+.6f} | "
            f"L{peak + 1} ({gaps['normalized_entropy'][peak]:+.6f}) |"
        )
    lines.extend(
        [
            "",
            "注：Hall−Real 的熵为负、Top-K mass/Pmax 为正，均表示 Hall 的 P 更集中；反之表示 Real 更集中。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
