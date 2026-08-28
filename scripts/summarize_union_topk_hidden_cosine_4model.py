#!/usr/bin/env python3
"""Combine four per-model union-TopK hidden-cosine diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


MODELS = (
    "llava_1_5_7b",
    "internvl_2_5_8b",
    "qwen2_5_vl_7b",
    "qwen3_vl_8b",
)
DISPLAY = {
    "llava_1_5_7b": "LLaVA-1.5-7B",
    "internvl_2_5_8b": "InternVL-2.5-8B",
    "qwen2_5_vl_7b": "Qwen2.5-VL-7B",
    "qwen3_vl_8b": "Qwen3-VL-8B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--name", default="union_topk_hidden_cosine_500")
    parser.add_argument(
        "--experiment", default="COCO4000-INSLEN-OFFICIAL-TARGET"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs_root = Path(args.outputs_root).resolve()
    summaries = {}
    for model in MODELS:
        result_dir = outputs_root / model / args.experiment / "results" / args.name
        path = result_dir / f"{model}_{args.name}_summary.json"
        if not path.exists():
            raise FileNotFoundError(path)
        summaries[model] = json.loads(path.read_text())

    rows = []
    for model, payload in summaries.items():
        source_rows = payload["rows"]
        for label in ("all", "hall", "real"):
            for branch in sorted({row["branch"] for row in source_rows}):
                selected = [
                    row
                    for row in source_rows
                    if row["label"] == label and row["branch"] == branch
                ]
                if not selected:
                    continue
                mean = lambda key: float(
                    np.mean([float(row[key]) for row in selected])
                )
                rows.append(
                    {
                        "model": model,
                        "display_model": DISPLAY[model],
                        "label": label,
                        "branch": branch,
                        "layers": len(selected),
                        "n_images_min": min(int(row["n_images"]) for row in selected),
                        "n_targets_min": min(int(row["n_targets"]) for row in selected),
                        "vv_cosine": mean("vv_pooled_mean"),
                        "vv_p90_minus_p10": mean("vv_pooled_p90")
                        - mean("vv_pooled_p10"),
                        "vv_centered_cosine": mean("vv_centered_mean"),
                        "cosine_cost_mean": mean("cosine_cost_mean"),
                        "cosine_cost_p90_minus_p10": mean("cosine_cost_p90")
                        - mean("cosine_cost_p10"),
                        "sqrt_cost_mean": mean("sqrt_cost_mean"),
                        "sqrt_cost_p90_minus_p10": mean("sqrt_cost_p90")
                        - mean("sqrt_cost_p10"),
                        "qv_cosine": mean("qv_pooled_mean"),
                        "qv_p90_minus_p10": mean("qv_pooled_p90")
                        - mean("qv_pooled_p10"),
                        "qv_centered_cosine": mean("qv_centered_mean"),
                        "qv_entropy": mean("qv_softmax_normalized_entropy"),
                        "qv_max_over_uniform": mean("qv_softmax_max_over_uniform"),
                        "qv_top1_minus_bottom": mean("qv_top1_minus_bottom"),
                        "union_size": mean("support_size"),
                        "intersection_size": mean("intersection_size"),
                    }
                )

    stem = outputs_root / f"{args.name}_4model_summary"
    _write_csv(stem.with_suffix(".csv"), rows)
    stem.with_suffix(".json").write_text(
        json.dumps(
            {
                "name": args.name,
                "models": list(MODELS),
                "sampling": {
                    model: summaries[model]["sampling"] for model in MODELS
                },
                "audit": {model: summaries[model]["audit"] for model in MODELS},
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    stem.with_suffix(".md").write_text(_render_markdown(rows, summaries) + "\n")
    _plot(rows, stem.with_suffix(".png"), stem.with_suffix(".pdf"))
    print(stem.with_suffix(".md"))


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _render_markdown(rows: list[dict], summaries: dict) -> str:
    lines = [
        "# 四模型 Union Top-K hidden cosine 诊断",
        "",
        "支持集为 source Top-32 与 target Top-32 的并集；visual–visual 排除对角线和对称重复。"
        "vv/qv cosine 分布按全部有效 token 对 pooled；entropy 等标量先按图片聚合再等权平均。",
        "",
        "## 覆盖",
        "",
        "| 模型 | 图片 | 正式目标 | 唯一因果位置 | 重复 span | 标签冲突 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        audit = summaries[model]["audit"]
        lines.append(
            f"| {DISPLAY[model]} | {audit['processed_images']} | "
            f"{audit['processed_targets']} | {audit['processed_causal_positions']} | "
            f"{audit['duplicate_spans_collapsed']} | {audit['target_label_conflicts']} |"
        )
    lines.extend(
        [
            "",
            "## 全层平均（all 标签）",
            "",
            "| 模型 | P/Q 分支 | vv cos | vv P90−P10 | centered vv | cosine-cost P90−P10 | qv cos | qv P90−P10 | qv entropy | qv max/uniform |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        if row["label"] != "all":
            continue
        lines.append(
            f"| {row['display_model']} | {row['branch']} | "
            f"{row['vv_cosine']:.4f} | {row['vv_p90_minus_p10']:.4f} | "
            f"{row['vv_centered_cosine']:.4f} | "
            f"{row['cosine_cost_p90_minus_p10']:.4f} | "
            f"{row['qv_cosine']:.4f} | {row['qv_p90_minus_p10']:.4f} | "
            f"{row['qv_entropy']:.4f} | {row['qv_max_over_uniform']:.2f} |"
        )
    return "\n".join(lines)


def _plot(rows: list[dict], png_path: Path, pdf_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_rows = [row for row in rows if row["label"] == "all"]
    branches = sorted({row["branch"] for row in all_rows})
    colors = ["#0072b2", "#d55e00", "#009e73", "#cc79a7"]
    x = np.arange(len(MODELS), dtype=float)
    width = 0.18
    panels = (
        ("vv_cosine", "visual–visual cosine"),
        ("vv_p90_minus_p10", "visual–visual P90−P10"),
        ("vv_centered_cosine", "centered visual–visual cosine"),
        ("qv_cosine", "$q_t$–visual cosine"),
        ("qv_p90_minus_p10", "$q_t$–visual P90−P10"),
        ("qv_entropy", "$q_t$–visual normalized entropy"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    for branch_index, (branch, color) in enumerate(zip(branches, colors)):
        offset = (branch_index - (len(branches) - 1) / 2.0) * width
        branch_rows = {
            row["model"]: row
            for row in all_rows
            if row["branch"] == branch
        }
        for axis, (field, title) in zip(axes.flat, panels):
            axis.bar(
                x + offset,
                [float(branch_rows[model][field]) for model in MODELS],
                width=width,
                color=color,
                label=branch.replace("_cos__hpre_", " / ").replace("_gauss", ""),
            )
            axis.set_title(title)
            axis.set_xticks(x, [DISPLAY[model] for model in MODELS], rotation=15)
            axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Union Top-32(P) ∪ Top-32(Q): hidden cosine geometry")
    fig.savefig(png_path, dpi=180)
    fig.savefig(pdf_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
