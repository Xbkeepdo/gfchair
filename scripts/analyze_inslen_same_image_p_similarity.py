#!/usr/bin/env python3
"""Measure how target-specific P is for different object words in one image."""

from __future__ import annotations

import argparse
import csv
import gc
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.io_utils import load_pkl


SOURCE_SPECS = (
    ("hmid_cos", "P=hmid", "dgst_t_source_dist_per_layer"),
    ("hpre_cos", "P=hpre", "dgst_t_source_hpre_cos_dist_per_layer"),
)
CATEGORIES = ("all", "real_real", "mixed", "hall_hall")
CATEGORY_TITLES = {
    "all": "all pairs",
    "real_real": "Real–Real",
    "mixed": "Real–Hall",
    "hall_hall": "Hall–Hall",
}
CATEGORY_COLORS = {
    "all": "#000000",
    "real_real": "#0072b2",
    "mixed": "#cc79a7",
    "hall_hall": "#d55e00",
}
METRICS = (
    "raw_cosine",
    "centered_cosine",
    "js_normalized",
    "total_variation",
    "topk_overlap_fraction",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--feature-files", nargs="+", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--pair-chunk-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")
    if args.pair_chunk_size <= 0:
        raise ValueError("--pair-chunk-size must be positive.")
    output_dir = Path(args.output_dir)
    feature_files = _resolve_feature_files(output_dir, args.feature_files)
    results_dir = (
        Path(args.results_dir)
        if args.results_dir
        else output_dir / "results" / "feature_curves"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    accum, specificity, audit = _aggregate(
        feature_files,
        top_k=args.top_k,
        pair_chunk_size=args.pair_chunk_size,
    )
    stats = _finalize(accum)
    specificity_stats = _finalize_specificity(specificity)
    stem = f"{args.model}_same_image_different_target_p_similarity"
    csv_path = results_dir / f"{stem}.csv"
    png_path = results_dir / f"{stem}.png"
    pdf_path = results_dir / f"{stem}.pdf"
    markdown_path = results_dir / f"{stem}.md"
    _write_csv(args.model, stats, specificity_stats, csv_path)
    _plot(args.model, stats, specificity_stats, args.top_k, png_path, pdf_path)
    _write_markdown(
        args.model,
        feature_files,
        audit,
        stats,
        specificity_stats,
        args.top_k,
        csv_path,
        png_path,
        markdown_path,
    )

    print("markdown", markdown_path.resolve())
    print("csv", csv_path.resolve())
    print("png", png_path.resolve())
    print("pdf", pdf_path.resolve())
    for source_slug, title, _key in SOURCE_SPECS:
        item = stats[(source_slug, "all")]
        spec = specificity_stats[source_slug]
        print(
            f"{title}: images={item['n_images'][0]} pairs={item['n_pairs'][0]} "
            f"raw_cos={np.mean(item['image']['raw_cosine']['mean']):.8f} "
            f"centered_cos={np.mean(item['image']['centered_cosine']['mean']):.8f} "
            f"js={np.mean(item['image']['js_normalized']['mean']):.8f} "
            f"top{args.top_k}_overlap="
            f"{np.mean(item['image']['topk_overlap_fraction']['mean']):.8f} "
            f"pair_js/uniform_js={np.mean(spec['ratio']['mean']):.8f}"
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


def _matrix(value: object, key: str, path: Path) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or min(result.shape) < 1:
        raise ValueError(f"{key} in {path} has invalid shape {result.shape}.")
    if not np.all(np.isfinite(result)) or float(result.min()) < -1.0e-7:
        raise ValueError(f"{key} in {path} is not a finite nonnegative P matrix.")
    result = np.maximum(result, 0.0)
    row_sums = result.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0) or float(np.max(np.abs(row_sums - 1.0))) > 1.0e-4:
        raise ValueError(f"{key} in {path} is not row-normalized.")
    return result / row_sums


def _new_item(layers: int) -> dict:
    return {
        "n_pairs": 0,
        "n_images": 0,
        "pair_sum": {name: np.zeros(layers) for name in METRICS},
        "pair_sum_sq": {name: np.zeros(layers) for name in METRICS},
        "image_sum": {name: np.zeros(layers) for name in METRICS},
        "image_sum_sq": {name: np.zeros(layers) for name in METRICS},
    }


def _new_specificity(layers: int) -> dict:
    return {
        "n_images": 0,
        "pair_js_sum": np.zeros(layers),
        "pair_js_sum_sq": np.zeros(layers),
        "uniform_js_sum": np.zeros(layers),
        "uniform_js_sum_sq": np.zeros(layers),
        "ratio_sum": np.zeros(layers),
        "ratio_sum_sq": np.zeros(layers),
        "random_overlap_sum": np.zeros(layers),
        "random_overlap_sum_sq": np.zeros(layers),
    }


def _category_masks(labels: np.ndarray, left: np.ndarray, right: np.ndarray) -> dict:
    left_labels = labels[left]
    right_labels = labels[right]
    return {
        "all": np.ones(left.shape[0], dtype=bool),
        "real_real": (left_labels == 1) & (right_labels == 1),
        "mixed": left_labels != right_labels,
        "hall_hall": (left_labels == 0) & (right_labels == 0),
    }


def _pair_metrics(
    distributions: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    top_k: int,
    pair_chunk_size: int,
) -> dict[str, np.ndarray]:
    targets, layers, n_visual = distributions.shape
    active_k = min(top_k, n_visual)
    top_indices = np.argpartition(
        distributions,
        n_visual - active_k,
        axis=2,
    )[:, :, -active_k:]
    top_mask = np.zeros((targets, layers, n_visual), dtype=bool)
    np.put_along_axis(top_mask, top_indices, True, axis=2)
    centered = distributions - (1.0 / n_visual)
    results = {name: [] for name in METRICS}
    tiny = np.finfo(np.float64).tiny
    for start in range(0, len(left), pair_chunk_size):
        stop = min(start + pair_chunk_size, len(left))
        left_index = left[start:stop]
        right_index = right[start:stop]
        first = distributions[left_index]
        second = distributions[right_index]
        denominator = np.linalg.norm(first, axis=2) * np.linalg.norm(second, axis=2)
        results["raw_cosine"].append(
            np.sum(first * second, axis=2) / np.maximum(denominator, tiny)
        )
        first_centered = centered[left_index]
        second_centered = centered[right_index]
        centered_denominator = (
            np.linalg.norm(first_centered, axis=2)
            * np.linalg.norm(second_centered, axis=2)
        )
        centered_cosine = np.sum(first_centered * second_centered, axis=2) / np.maximum(
            centered_denominator,
            tiny,
        )
        centered_cosine[centered_denominator <= tiny] = np.nan
        results["centered_cosine"].append(centered_cosine)
        midpoint = 0.5 * (first + second)
        js = 0.5 * np.sum(
            first * (np.log(np.maximum(first, tiny)) - np.log(np.maximum(midpoint, tiny))),
            axis=2,
        ) + 0.5 * np.sum(
            second
            * (np.log(np.maximum(second, tiny)) - np.log(np.maximum(midpoint, tiny))),
            axis=2,
        )
        results["js_normalized"].append(js / math.log(2.0))
        results["total_variation"].append(
            0.5 * np.sum(np.abs(first - second), axis=2)
        )
        results["topk_overlap_fraction"].append(
            np.sum(top_mask[left_index] & top_mask[right_index], axis=2) / active_k
        )
    return {name: np.concatenate(values, axis=0) for name, values in results.items()}


def _js_to_uniform(distributions: np.ndarray) -> np.ndarray:
    n_visual = distributions.shape[2]
    uniform = np.full_like(distributions, 1.0 / n_visual)
    midpoint = 0.5 * (distributions + uniform)
    tiny = np.finfo(np.float64).tiny
    js = 0.5 * np.sum(
        distributions
        * (
            np.log(np.maximum(distributions, tiny))
            - np.log(np.maximum(midpoint, tiny))
        ),
        axis=2,
    ) + 0.5 * np.sum(
        uniform
        * (np.log(uniform) - np.log(np.maximum(midpoint, tiny))),
        axis=2,
    )
    return js / math.log(2.0)


def _update_item(item: dict, metrics: dict, mask: np.ndarray) -> None:
    if not np.any(mask):
        return
    item["n_pairs"] += int(np.sum(mask))
    item["n_images"] += 1
    for name in METRICS:
        values = metrics[name][mask]
        if name == "centered_cosine":
            valid = np.isfinite(values)
            if not np.all(valid):
                values = np.where(valid, values, 0.0)
        pair_sum = values.sum(axis=0)
        pair_sum_sq = np.square(values).sum(axis=0)
        image_mean = values.mean(axis=0)
        item["pair_sum"][name] += pair_sum
        item["pair_sum_sq"][name] += pair_sum_sq
        item["image_sum"][name] += image_mean
        item["image_sum_sq"][name] += np.square(image_mean)


def _aggregate(
    feature_files: list[Path],
    *,
    top_k: int,
    pair_chunk_size: int,
) -> tuple[dict, dict, dict]:
    accum = {}
    specificity = {}
    seen_images = set()
    audit = {
        "rows_per_file": {},
        "binary_rows": 0,
        "eligible_images": 0,
        "unique_targets": 0,
        "dropped_same_word_rows": 0,
        "pair_count": 0,
    }
    expected_layers = None
    for path in feature_files:
        rows = load_pkl(str(path))
        if not isinstance(rows, list):
            raise TypeError(f"{path} must contain a list of feature records.")
        audit["rows_per_file"][str(path)] = len(rows)
        groups = defaultdict(list)
        for row in rows:
            if row.get("label") not in (0, 1):
                continue
            audit["binary_rows"] += 1
            groups[int(row["image_id"])].append(row)
        overlap = seen_images.intersection(groups)
        if overlap:
            raise ValueError(
                "An image is split across feature parts, which would omit cross-part "
                f"target pairs: {sorted(overlap)[:5]}"
            )
        seen_images.update(groups)

        for image_rows in groups.values():
            unique_by_word = {}
            for row in image_rows:
                word = str(row.get("token_str") or "").strip().lower()
                if not word:
                    raise ValueError("Feature row has an empty token_str.")
                if word in unique_by_word:
                    audit["dropped_same_word_rows"] += 1
                    continue
                unique_by_word[word] = row
            selected = list(unique_by_word.values())
            if len(selected) < 2:
                continue
            audit["eligible_images"] += 1
            audit["unique_targets"] += len(selected)
            left, right = np.triu_indices(len(selected), k=1)
            audit["pair_count"] += len(left)
            labels = np.asarray([int(row["label"]) for row in selected], dtype=np.int8)
            masks = _category_masks(labels, left, right)

            for source_slug, _title, key in SOURCE_SPECS:
                matrices = [_matrix(row[key], key, path) for row in selected]
                shapes = {matrix.shape for matrix in matrices}
                if len(shapes) != 1:
                    raise ValueError(
                        f"Same-image targets have different P shapes for {source_slug}: "
                        f"{sorted(shapes)}"
                    )
                distributions = np.stack(matrices, axis=0)
                layers = distributions.shape[1]
                n_visual = distributions.shape[2]
                if expected_layers is None:
                    expected_layers = layers
                elif layers != expected_layers:
                    raise ValueError(
                        f"Layer count changed from {expected_layers} to {layers}."
                    )
                metrics = _pair_metrics(
                    distributions,
                    left,
                    right,
                    top_k,
                    pair_chunk_size,
                )
                for category, mask in masks.items():
                    acc_key = (source_slug, category)
                    if acc_key not in accum:
                        accum[acc_key] = _new_item(layers)
                    _update_item(accum[acc_key], metrics, mask)

                uniform_js = _js_to_uniform(distributions).mean(axis=0)
                pair_js = metrics["js_normalized"].mean(axis=0)
                ratio = pair_js / np.maximum(uniform_js, 1.0e-12)
                random_overlap = np.full(layers, min(top_k, n_visual) / n_visual)
                if source_slug not in specificity:
                    specificity[source_slug] = _new_specificity(layers)
                spec = specificity[source_slug]
                spec["n_images"] += 1
                for name, values in (
                    ("pair_js", pair_js),
                    ("uniform_js", uniform_js),
                    ("ratio", ratio),
                    ("random_overlap", random_overlap),
                ):
                    spec[f"{name}_sum"] += values
                    spec[f"{name}_sum_sq"] += np.square(values)

        print(f"processed {path}: rows={len(rows)} images={len(groups)}", flush=True)
        del groups, rows
        gc.collect()

    if audit["eligible_images"] == 0:
        raise ValueError("No image has at least two different target words.")
    return accum, specificity, audit


def _moments(sum_: np.ndarray, sum_sq: np.ndarray, n: int) -> dict:
    if n < 1:
        raise ValueError("Cannot finalize an empty statistic.")
    mean = sum_ / n
    if n > 1:
        variance = np.maximum((sum_sq - n * np.square(mean)) / (n - 1), 0.0)
    else:
        variance = np.zeros_like(mean)
    std = np.sqrt(variance)
    return {"mean": mean, "std": std, "sem": std / math.sqrt(n)}


def _finalize(accum: dict) -> dict:
    stats = {}
    for key, item in accum.items():
        if item["n_pairs"] == 0 or item["n_images"] == 0:
            continue
        stats[key] = {
            "n_pairs": np.full_like(
                next(iter(item["pair_sum"].values())),
                item["n_pairs"],
                dtype=np.int64,
            ),
            "n_images": np.full_like(
                next(iter(item["pair_sum"].values())),
                item["n_images"],
                dtype=np.int64,
            ),
            "pair": {
                name: _moments(
                    item["pair_sum"][name],
                    item["pair_sum_sq"][name],
                    item["n_pairs"],
                )
                for name in METRICS
            },
            "image": {
                name: _moments(
                    item["image_sum"][name],
                    item["image_sum_sq"][name],
                    item["n_images"],
                )
                for name in METRICS
            },
        }
    return stats


def _finalize_specificity(accum: dict) -> dict:
    result = {}
    for source_slug, item in accum.items():
        n = item["n_images"]
        result[source_slug] = {"n_images": n}
        for name in ("pair_js", "uniform_js", "ratio", "random_overlap"):
            result[source_slug][name] = _moments(
                item[f"{name}_sum"], item[f"{name}_sum_sq"], n
            )
    return result


def _write_csv(model: str, stats: dict, specificity: dict, path: Path) -> None:
    fieldnames = [
        "model",
        "source_mode",
        "category",
        "layer",
        "n_pairs",
        "n_images",
    ]
    for weighting in ("pair", "image"):
        for metric in METRICS:
            for statistic in ("mean", "std", "sem"):
                fieldnames.append(f"{weighting}_{metric}_{statistic}")
    for name in ("pair_js", "uniform_js", "ratio", "random_overlap"):
        for statistic in ("mean", "std", "sem"):
            fieldnames.append(f"specificity_{name}_{statistic}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for source_slug, _title, _key in SOURCE_SPECS:
            for category in CATEGORIES:
                item = stats.get((source_slug, category))
                if item is None:
                    continue
                layers = len(item["n_pairs"])
                for index in range(layers):
                    row = {
                        "model": model,
                        "source_mode": source_slug,
                        "category": category,
                        "layer": index + 1,
                        "n_pairs": int(item["n_pairs"][index]),
                        "n_images": int(item["n_images"][index]),
                    }
                    for weighting in ("pair", "image"):
                        for metric in METRICS:
                            for statistic in ("mean", "std", "sem"):
                                row[f"{weighting}_{metric}_{statistic}"] = (
                                    f"{item[weighting][metric][statistic][index]:.10g}"
                                )
                    if category == "all":
                        spec = specificity[source_slug]
                        for name in (
                            "pair_js",
                            "uniform_js",
                            "ratio",
                            "random_overlap",
                        ):
                            for statistic in ("mean", "std", "sem"):
                                row[f"specificity_{name}_{statistic}"] = (
                                    f"{spec[name][statistic][index]:.10g}"
                                )
                    writer.writerow(row)


def _plot(
    model: str,
    stats: dict,
    specificity: dict,
    top_k: int,
    png_path: Path,
    pdf_path: Path,
) -> None:
    metric_specs = (
        ("raw_cosine", "cos(P_a, P_b)", (0.0, 1.02)),
        ("centered_cosine", "cos(P_a-u, P_b-u)", (-1.02, 1.02)),
        ("js_normalized", "JS(P_a, P_b) / ln 2", None),
        ("total_variation", "Total variation", (0.0, 1.02)),
        ("topk_overlap_fraction", f"Top-{top_k} overlap fraction", (0.0, 1.02)),
    )
    fig, axes = plt.subplots(2, len(metric_specs), figsize=(24, 9), sharex="row")
    for row_index, (source_slug, title, _key) in enumerate(SOURCE_SPECS):
        for column, (metric, ylabel, ylim) in enumerate(metric_specs):
            ax = axes[row_index, column]
            for category in CATEGORIES:
                item = stats.get((source_slug, category))
                if item is None:
                    continue
                mean = item["image"][metric]["mean"]
                layers = np.arange(1, len(mean) + 1)
                ax.plot(
                    layers,
                    mean,
                    color=CATEGORY_COLORS[category],
                    linewidth=2.1 if category == "all" else 1.5,
                    linestyle="-" if category == "all" else "--",
                    label=(
                        f"{CATEGORY_TITLES[category]} "
                        f"(images={int(item['n_images'][0])})"
                    ),
                )
            if metric == "topk_overlap_fraction":
                baseline = specificity[source_slug]["random_overlap"]["mean"]
                ax.plot(
                    layers,
                    baseline,
                    color="#777777",
                    linestyle=":",
                    linewidth=1.5,
                    label="random Top-K baseline",
                )
            ax.set_title(f"{title}: {ylabel}")
            ax.set_xlabel("Layer")
            ax.set_ylabel(ylabel)
            ax.set_xlim(1, len(layers))
            if ylim is not None:
                ax.set_ylim(*ylim)
            ax.grid(True, alpha=0.22)
            ax.legend(frameon=False, fontsize=7)
    fig.suptitle(
        f"{model}: same-image P similarity between different target words "
        "(image-weighted means)",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def _layer_mean(item: dict, metric: str, weighting: str = "image") -> float:
    return float(np.mean(item[weighting][metric]["mean"]))


def _write_markdown(
    model: str,
    feature_files: list[Path],
    audit: dict,
    stats: dict,
    specificity: dict,
    top_k: int,
    csv_path: Path,
    png_path: Path,
    path: Path,
) -> None:
    lines = [
        f"# {model}：同图不同目标的 P 相似度",
        "",
        "- 只比较同一图片中 `token_str` 不同的正式 InsLen 目标。",
        "- 主结果先在每张图片内部对所有目标对取平均，再对图片等权平均，避免对象较多的 caption 主导结果。CSV 同时保存 pair-weighted 结果。",
        "- 原始 cosine 会受到 P 的均匀公共分量影响，因此同时报告去均匀分量后的 centered cosine、归一化 JS、TV 和 Top-K 重合率。",
        "- `pair JS / uniform JS` 比较目标间变化与 P 偏离均匀分布的幅度；越接近 0，越说明 P 主要是共享模板而非目标特异分布。",
        "",
        "## 数据",
        "",
    ]
    for feature_file in feature_files:
        lines.append(
            f"- `{feature_file}`：{audit['rows_per_file'][str(feature_file)]:,} 条。"
        )
    lines.extend(
        [
            f"- 二元标签记录：{audit['binary_rows']:,}。",
            f"- 至少含两个不同目标词的图片：{audit['eligible_images']:,}。",
            f"- 不同目标词总数：{audit['unique_targets']:,}。",
            f"- 同图不同目标 pair：{audit['pair_count']:,}。",
            f"- 同词重复行排除：{audit['dropped_same_word_rows']:,}。",
            "",
            "## 全层平均（图片等权）",
            "",
            f"| P source | 图片数 | pair 数 | raw cosine | centered cosine | JS/ln2 | TV | Top-{top_k} overlap | 随机 overlap | pair JS / uniform JS |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for source_slug, title, _key in SOURCE_SPECS:
        item = stats[(source_slug, "all")]
        spec = specificity[source_slug]
        lines.append(
            f"| {title} | {int(item['n_images'][0]):,} | "
            f"{int(item['n_pairs'][0]):,} | "
            f"{_layer_mean(item, 'raw_cosine'):.6f} | "
            f"{_layer_mean(item, 'centered_cosine'):.6f} | "
            f"{_layer_mean(item, 'js_normalized'):.6f} | "
            f"{_layer_mean(item, 'total_variation'):.6f} | "
            f"{_layer_mean(item, 'topk_overlap_fraction'):.6f} | "
            f"{float(np.mean(spec['random_overlap']['mean'])):.6f} | "
            f"{float(np.mean(spec['ratio']['mean'])):.6f} |"
        )
    lines.extend(
        [
            "",
            "## 按标签组合的 JS/ln2",
            "",
            "| P source | Real–Real | Real–Hall | Hall–Hall |",
            "|---|---:|---:|---:|",
        ]
    )
    for source_slug, title, _key in SOURCE_SPECS:
        values = []
        for category in ("real_real", "mixed", "hall_hall"):
            item = stats.get((source_slug, category))
            values.append(
                "n/a" if item is None else f"{_layer_mean(item, 'js_normalized'):.6f}"
            )
        lines.append(f"| {title} | " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            f"- 逐层完整统计：`{csv_path.name}`。",
            f"- 曲线图：`{png_path.name}`。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
