#!/usr/bin/env python3
"""Plot active InsLen risk and mass-times-cosine curves by CHAIR label."""

from __future__ import annotations

import argparse
import csv
import gc
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.io_utils import load_pkl


HALL = 0
REAL = 1


@dataclass(frozen=True)
class FeatureSpec:
    slug: str
    title: str
    group: str
    key: str


FEATURE_SPECS = (
    FeatureSpec(
        "raw_p_hmid_sqrt_risk",
        "Raw logit · P=hmid · sqrt cost",
        "raw_risk",
        "dgst_t_hpre_raw_logit_gauss_risk_sqrt_hpre_per_layer",
    ),
    FeatureSpec(
        "raw_p_hmid_cosine_risk",
        "Raw logit · P=hmid · cosine cost",
        "raw_risk",
        "dgst_t_hpre_raw_logit_gauss_risk_cosine_hpre_per_layer",
    ),
    FeatureSpec(
        "raw_p_hpre_sqrt_risk",
        "Raw logit · P=hpre · sqrt cost",
        "raw_risk",
        "dgst_t_hpre_raw_logit_gauss_source_hpre_cos_risk_sqrt_hpre_per_layer",
    ),
    FeatureSpec(
        "raw_p_hpre_cosine_risk",
        "Raw logit · P=hpre · cosine cost",
        "raw_risk",
        "dgst_t_hpre_raw_logit_gauss_source_hpre_cos_risk_cosine_hpre_per_layer",
    ),
    FeatureSpec(
        "softmax_p_hmid_sqrt_risk",
        "Softmax · P=hmid · sqrt cost",
        "softmax_risk",
        "dgst_t_hpre_softmax_prob_gauss_risk_sqrt_hpre_per_layer",
    ),
    FeatureSpec(
        "softmax_p_hmid_cosine_risk",
        "Softmax · P=hmid · cosine cost",
        "softmax_risk",
        "dgst_t_hpre_softmax_prob_gauss_risk_cosine_hpre_per_layer",
    ),
    FeatureSpec(
        "softmax_p_hpre_sqrt_risk",
        "Softmax · P=hpre · sqrt cost",
        "softmax_risk",
        "dgst_t_hpre_softmax_prob_gauss_source_hpre_cos_risk_sqrt_hpre_per_layer",
    ),
    FeatureSpec(
        "softmax_p_hpre_cosine_risk",
        "Softmax · P=hpre · cosine cost",
        "softmax_risk",
        "dgst_t_hpre_softmax_prob_gauss_source_hpre_cos_risk_cosine_hpre_per_layer",
    ),
    FeatureSpec(
        "raw_mass_x_cosine",
        "Raw logit · mass × cosine",
        "mass_x_cosine",
        "dgst_t_hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine_topk32_hpre_per_layer",
    ),
    FeatureSpec(
        "softmax_mass_x_cosine",
        "Softmax · mass × cosine",
        "mass_x_cosine",
        "dgst_t_hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine_topk32_hpre_per_layer",
    ),
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
            "Feature pickle files to aggregate. By default, use all features.part*.pkl "
            "when present, otherwise use features.pkl."
        ),
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Defaults to <output-dir>/results/feature_curves.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    feature_files = _resolve_feature_files(output_dir, args.feature_files)
    results_dir = (
        Path(args.results_dir)
        if args.results_dir
        else output_dir / "results" / "feature_curves"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    stats, rows_per_file = _aggregate(feature_files)
    stem = f"{args.model}_inslen_risk_mass_x_cosine_by_label"
    plot_paths = {}
    for group in ("raw_risk", "softmax_risk", "mass_x_cosine"):
        group_specs = [spec for spec in FEATURE_SPECS if spec.group == group]
        png_path = results_dir / f"{stem}_{group}.png"
        pdf_path = results_dir / f"{stem}_{group}.pdf"
        _plot_group(group_specs, stats, png_path, pdf_path)
        plot_paths[group] = (png_path, pdf_path)

    csv_path = results_dir / f"{stem}.csv"
    md_path = results_dir / f"{stem}.md"
    _write_csv(stats, csv_path)
    _write_markdown(
        args.model,
        feature_files,
        rows_per_file,
        stats,
        plot_paths,
        csv_path,
        md_path,
    )

    for group, (png_path, pdf_path) in plot_paths.items():
        print(group, "png", png_path.resolve())
        print(group, "pdf", pdf_path.resolve())
    print("csv", csv_path.resolve())
    print("markdown", md_path.resolve())
    for spec in FEATURE_SPECS:
        item = stats[spec.slug]
        peak = int(np.argmax(np.abs(item["gap"])))
        print(
            f"{spec.slug}: n_hall={item['hall']['n']} n_real={item['real']['n']} "
            f"mean_gap={np.mean(item['gap']):+.8f} peak_layer={peak} "
            f"peak_gap={item['gap'][peak]:+.8f} significant_layers={item['significant_count']}"
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


def _empty_accumulator() -> dict:
    return {
        spec.slug: {
            HALL: {"n": 0, "sum": None, "sum_sq": None},
            REAL: {"n": 0, "sum": None, "sum_sq": None},
        }
        for spec in FEATURE_SPECS
    }


def _aggregate(feature_files: list[Path]) -> tuple[dict, dict[str, int]]:
    accum = _empty_accumulator()
    rows_per_file = {}
    expected_layers = None
    for path in feature_files:
        rows = load_pkl(str(path))
        rows_per_file[str(path)] = len(rows)
        missing = [spec.key for spec in FEATURE_SPECS if not any(spec.key in row for row in rows)]
        if missing:
            raise KeyError(f"{path} is missing feature key(s): " + ", ".join(missing))
        for row in rows:
            label = row.get("label")
            if label not in (HALL, REAL):
                continue
            for spec in FEATURE_SPECS:
                if spec.key not in row:
                    continue
                values = np.asarray(row[spec.key], dtype=np.float64).reshape(-1)
                if values.size == 0:
                    continue
                if not np.all(np.isfinite(values)):
                    raise ValueError(f"Non-finite values in {spec.key} from {path}.")
                if expected_layers is None:
                    expected_layers = int(values.size)
                elif values.size != expected_layers:
                    raise ValueError(
                        f"{spec.key} has {values.size} layers; expected {expected_layers}."
                    )
                item = accum[spec.slug][int(label)]
                if item["sum"] is None:
                    item["sum"] = np.zeros_like(values)
                    item["sum_sq"] = np.zeros_like(values)
                item["n"] += 1
                item["sum"] += values
                item["sum_sq"] += np.square(values)
        del rows
        gc.collect()

    result = {}
    for spec in FEATURE_SPECS:
        result[spec.slug] = {}
        for label, label_slug in ((HALL, "hall"), (REAL, "real")):
            item = accum[spec.slug][label]
            n = int(item["n"])
            if n < 2:
                raise ValueError(f"{spec.key} needs at least two rows for label={label}.")
            mean = item["sum"] / n
            variance = (item["sum_sq"] - n * np.square(mean)) / (n - 1)
            variance = np.maximum(variance, 0.0)
            std = np.sqrt(variance)
            sem = std / math.sqrt(n)
            result[spec.slug][label_slug] = {
                "n": n,
                "mean": mean,
                "std": std,
                "sem": sem,
                "ci95": 1.96 * sem,
            }
        hall = result[spec.slug]["hall"]
        real = result[spec.slug]["real"]
        gap = hall["mean"] - real["mean"]
        gap_sem = np.sqrt(np.square(hall["sem"]) + np.square(real["sem"]))
        gap_ci95 = 1.96 * gap_sem
        pooled_var = (
            (hall["n"] - 1) * np.square(hall["std"])
            + (real["n"] - 1) * np.square(real["std"])
        ) / (hall["n"] + real["n"] - 2)
        pooled_std = np.sqrt(np.maximum(pooled_var, 0.0))
        cohen_d = np.divide(
            gap,
            pooled_std,
            out=np.zeros_like(gap),
            where=pooled_std > 0,
        )
        significant = (gap - gap_ci95 > 0) | (gap + gap_ci95 < 0)
        result[spec.slug].update(
            {
                "gap": gap,
                "gap_sem": gap_sem,
                "gap_ci95": gap_ci95,
                "cohen_d": cohen_d,
                "significant": significant,
                "significant_count": int(significant.sum()),
            }
        )
    return result, rows_per_file


def _plot_group(
    specs: list[FeatureSpec],
    stats: dict,
    png_path: Path,
    pdf_path: Path,
) -> None:
    columns = len(specs)
    fig, axes = plt.subplots(
        2,
        columns,
        figsize=(max(5.0 * columns, 10.0), 7.4),
        sharex=True,
        squeeze=False,
    )
    colors = {"hall": "#d55e00", "real": "#0072b2"}
    display = {"hall": "Hallucination", "real": "Real / non-hallucination"}
    for column, spec in enumerate(specs):
        item = stats[spec.slug]
        layers = np.arange(item["gap"].size)
        axis = axes[0, column]
        for label_slug in ("hall", "real"):
            group = item[label_slug]
            axis.plot(
                layers,
                group["mean"],
                color=colors[label_slug],
                linewidth=2.0,
                label=f"{display[label_slug]} (n={group['n']})",
            )
            axis.fill_between(
                layers,
                group["mean"] - group["ci95"],
                group["mean"] + group["ci95"],
                color=colors[label_slug],
                alpha=0.16,
                linewidth=0,
            )
        axis.set_title(spec.title, fontsize=11)
        axis.set_ylabel("Mean feature value")
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False, fontsize=8)

        gap_axis = axes[1, column]
        gap_axis.axhline(0.0, color="#555555", linewidth=1.0)
        gap_axis.plot(layers, item["gap"], color="#333333", linewidth=2.0)
        gap_axis.fill_between(
            layers,
            item["gap"] - item["gap_ci95"],
            item["gap"] + item["gap_ci95"],
            color="#777777",
            alpha=0.18,
            linewidth=0,
        )
        peak = int(np.argmax(np.abs(item["gap"])))
        gap_axis.scatter([peak], [item["gap"][peak]], color="#cc79a7", s=35, zorder=3)
        gap_axis.set_xlabel("Decoder layer")
        gap_axis.set_ylabel("Hall − Real")
        gap_axis.grid(True, alpha=0.25)
        gap_axis.set_xlim(0, len(layers) - 1)

    fig.suptitle(
        "LLaVA-1.5-7B InsLen targets: layerwise feature curves (95% CI)",
        fontsize=14,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def _write_csv(stats: dict, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "feature",
                "layer",
                "n_hall",
                "n_real",
                "hall_mean",
                "hall_std",
                "hall_sem",
                "real_mean",
                "real_std",
                "real_sem",
                "hall_minus_real",
                "gap_ci95_half_width",
                "gap_ci95_low",
                "gap_ci95_high",
                "cohen_d",
                "gap_ci95_excludes_zero",
            ]
        )
        for spec in FEATURE_SPECS:
            item = stats[spec.slug]
            for layer in range(item["gap"].size):
                gap = item["gap"][layer]
                ci95 = item["gap_ci95"][layer]
                writer.writerow(
                    [
                        spec.slug,
                        layer,
                        item["hall"]["n"],
                        item["real"]["n"],
                        f"{item['hall']['mean'][layer]:.10g}",
                        f"{item['hall']['std'][layer]:.10g}",
                        f"{item['hall']['sem'][layer]:.10g}",
                        f"{item['real']['mean'][layer]:.10g}",
                        f"{item['real']['std'][layer]:.10g}",
                        f"{item['real']['sem'][layer]:.10g}",
                        f"{gap:.10g}",
                        f"{ci95:.10g}",
                        f"{gap - ci95:.10g}",
                        f"{gap + ci95:.10g}",
                        f"{item['cohen_d'][layer]:.10g}",
                        int(item["significant"][layer]),
                    ]
                )


def _write_markdown(
    model: str,
    feature_files: list[Path],
    rows_per_file: dict[str, int],
    stats: dict,
    plot_paths: dict[str, tuple[Path, Path]],
    csv_path: Path,
    path: Path,
) -> None:
    lines = [
        f"# {model} risk 与 mass × cosine 分标签逐层曲线",
        "",
        "## 口径",
        "",
        "- 标签：`0=hallucination`，`1=real/non-hallucination`。",
        "- 上排曲线：每层类别均值，阴影为类别均值的 95% CI。",
        "- 下排曲线：`Hall−Real`，阴影为两组独立均值差的近似 95% CI。",
        "- `P=hmid/hpre` 只改变 risk 的 source marginal；同一 gate 的 `mass × cosine` 是 target-side EV，因此不随 P-source 或 transport cost 重复。",
        "- 层号采用保存数组的 0-based decoder layer index。",
        "- 输入分片：",
    ]
    for feature_file in feature_files:
        lines.append(f"  - `{feature_file}`：{rows_per_file[str(feature_file)]} rows")
    lines.extend(
        [
            f"- 逐层完整数值：`{csv_path.name}`。",
            "- 图片/PDF：",
        ]
    )
    for group, (png_path, pdf_path) in plot_paths.items():
        lines.append(f"  - `{group}`：`{png_path.name}` / `{pdf_path.name}`")
    lines.extend(
        [
            "",
            "## 汇总",
            "",
            "| 特征 | Hall n | Real n | Hall 全层均值 | Real 全层均值 | 平均 Hall−Real | 最大差层 | 该层差值 | 最大 |d| 层/值 | 95% CI 不跨零层数 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for spec in FEATURE_SPECS:
        item = stats[spec.slug]
        peak_gap = int(np.argmax(np.abs(item["gap"])))
        peak_d = int(np.argmax(np.abs(item["cohen_d"])))
        lines.append(
            f"| {spec.title} | {item['hall']['n']} | {item['real']['n']} | "
            f"{np.mean(item['hall']['mean']):.6f} | {np.mean(item['real']['mean']):.6f} | "
            f"{np.mean(item['gap']):+.6f} | L{peak_gap} | {item['gap'][peak_gap]:+.6f} | "
            f"L{peak_d} / {item['cohen_d'][peak_d]:+.3f} | "
            f"{item['significant_count']}/{item['gap'].size} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
