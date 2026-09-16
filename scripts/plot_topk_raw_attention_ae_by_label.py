#!/usr/bin/env python3
"""Plot REAL/HALL layer curves for Top-16/32 raw-attention and AE mass."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/ffn_source_composition_v1"
OUT = ROOT / "outputs/topk_raw_attention_ae_curves_v1"
MODELS = (
    "qwen2_5_vl_7b",
    "llava_1_5_7b",
    "qwen3_vl_8b",
    "internvl_2_5_8b",
)
MODEL_NAMES = {
    "qwen2_5_vl_7b": "Qwen2.5-VL-7B",
    "llava_1_5_7b": "LLaVA-1.5-7B",
    "qwen3_vl_8b": "Qwen3-VL-8B",
    "internvl_2_5_8b": "InternVL2.5-8B",
}
SIGNAL_NAMES = {
    "raw_attention": "Normalized raw attention",
    "ae": "AE / T distribution",
}
TOP_KS = (16, 32)
COLORS = {"HALL": "#d62728", "REAL": "#1f77b4"}


def read_torch(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def normalize_rows(values: torch.Tensor) -> torch.Tensor:
    values = torch.nan_to_num(values.float(), nan=0.0, posinf=0.0, neginf=0.0)
    values = values.clamp_min(0.0)
    total = values.sum(dim=-1, keepdim=True)
    fallback = torch.full_like(values, 1.0 / values.shape[-1])
    return torch.where(total > 1e-12, values / total.clamp_min(1e-12), fallback)


def topk_masses(values: torch.Tensor) -> dict[int, np.ndarray]:
    """Return retained probability mass after each row selects its own Top-K."""
    if values.ndim != 2 or values.shape[-1] == 0:
        raise ValueError(f"Expected a non-empty [layers, visual_tokens] matrix, got {values.shape}")
    probability = normalize_rows(values)
    count = min(max(TOP_KS), probability.shape[-1])
    ranked = torch.topk(probability, k=count, dim=-1, largest=True, sorted=True).values
    cumulative = ranked.cumsum(dim=-1)
    result = {
        k: cumulative[:, min(k, probability.shape[-1]) - 1].numpy().astype(np.float32, copy=False)
        for k in TOP_KS
    }
    for k, mass in result.items():
        if not np.isfinite(mass).all() or np.any(mass < -1e-6) or np.any(mass > 1 + 1e-5):
            raise ValueError(f"Invalid Top-{k} probability mass")
    if np.any(result[32] + 2e-6 < result[16]):
        raise AssertionError("Top-32 mass must be at least Top-16 mass")
    return result


def source_paths(model: str) -> list[Path]:
    paths = sorted((SOURCE / model / "shards/k50").glob("image_*.pt"))
    if len(paths) != 4000:
        raise AssertionError(f"{model}: expected 4000 K50 shards, found {len(paths)}")
    return paths


def load_model(model: str) -> tuple[dict[tuple[str, str, int], np.ndarray], list[dict], np.ndarray, dict]:
    matrix_path = SOURCE / model / "matrices.pt"
    matrix = read_torch(matrix_path)
    mentions = matrix["mentions"]
    labels = np.asarray(matrix["y"], dtype=np.int32)
    if len(mentions) != len(labels) or not np.array_equal(labels, [int(x["label"]) for x in mentions]):
        raise AssertionError(f"{model}: mention/label alignment failed")
    if set(labels.tolist()) != {0, 1}:
        raise AssertionError(f"{model}: both HALL=0 and REAL=1 are required")

    wanted = {str(x["target_key"]) for x in mentions}
    scores: dict[tuple[str, str, int], np.ndarray] = {}
    seen_targets = set()
    visual_token_counts = []
    layer_count = None
    paths = source_paths(model)
    for number, path in enumerate(paths, 1):
        shard = read_torch(path)
        for position in shard["positions"]:
            target_key = str(position["target_key"])
            raw = torch.as_tensor(position["raw_attention_mean"])
            ae = torch.as_tensor(position["attention_evidence"])
            if raw.shape != ae.shape:
                raise AssertionError(f"{model}/{target_key}: raw-attention and AE supports differ")
            if layer_count is None:
                layer_count = int(raw.shape[0])
            if raw.shape[0] != layer_count:
                raise AssertionError(f"{model}/{target_key}: inconsistent layer count")
            if target_key in seen_targets:
                raise AssertionError(f"{model}: duplicate target {target_key}")
            seen_targets.add(target_key)
            visual_token_counts.append(int(raw.shape[-1]))
            for signal, values in (("raw_attention", raw), ("ae", ae)):
                for k, mass in topk_masses(values).items():
                    scores[(target_key, signal, k)] = mass
        if number % 250 == 0:
            print(f"[{model}] loaded {number}/4000 shards", flush=True)

    available = seen_targets
    if available != wanted:
        raise AssertionError(
            f"{model}: target mismatch, missing={len(wanted - available)}, extra={len(available - wanted)}"
        )
    metadata = {
        "images": len(paths),
        "mentions": len(mentions),
        "n_hall": int((labels == 0).sum()),
        "n_real": int((labels == 1).sum()),
        "unique_targets": len(wanted),
        "layers": layer_count,
        "visual_tokens_min": min(visual_token_counts),
        "visual_tokens_median": float(np.median(visual_token_counts)),
        "visual_tokens_max": max(visual_token_counts),
        "source_matrix": str(matrix_path.relative_to(ROOT)),
        "source_shards": "outputs/ffn_source_composition_v1/<model>/shards/k50/image_*.pt",
    }
    return scores, mentions, labels, metadata


def summarize_model(model: str) -> tuple[list[dict], list[dict], dict]:
    scores, mentions, labels, metadata = load_model(model)
    curve_rows = []
    gap_rows = []
    for signal in SIGNAL_NAMES:
        for k in TOP_KS:
            values = np.stack([scores[(str(x["target_key"]), signal, k)] for x in mentions])
            if values.shape != (len(mentions), metadata["layers"]):
                raise AssertionError(f"{model}/{signal}/Top-{k}: unexpected matrix shape {values.shape}")
            medians = {}
            for label, label_name in ((0, "HALL"), (1, "REAL")):
                selected = values[labels == label].astype(np.float64)
                medians[label_name] = np.median(selected, axis=0)
                mean = selected.mean(axis=0)
                q25, q75 = np.quantile(selected, (0.25, 0.75), axis=0)
                for layer in range(values.shape[1]):
                    curve_rows.append(
                        {
                            "model": model,
                            "signal": signal,
                            "top_k": k,
                            "label": label_name,
                            "layer": layer + 1,
                            "n_mentions": len(selected),
                            "mean": float(mean[layer]),
                            "q25": float(q25[layer]),
                            "median": float(medians[label_name][layer]),
                            "q75": float(q75[layer]),
                        }
                    )
            gap = medians["HALL"] - medians["REAL"]
            maximum = int(np.argmax(np.abs(gap)))
            gap_rows.append(
                {
                    "model": model,
                    "signal": signal,
                    "top_k": k,
                    "layers": len(gap),
                    "n_hall": metadata["n_hall"],
                    "n_real": metadata["n_real"],
                    "hall_higher_layers": int((gap > 0).sum()),
                    "real_higher_layers": int((gap < 0).sum()),
                    "equal_layers": int((gap == 0).sum()),
                    "median_hall_minus_real": float(np.median(gap)),
                    "mean_hall_minus_real": float(np.mean(gap)),
                    "mean_absolute_gap": float(np.mean(np.abs(gap))),
                    "largest_absolute_gap": float(abs(gap[maximum])),
                    "largest_gap_layer": maximum + 1,
                    "hall_minus_real_at_largest_gap": float(gap[maximum]),
                }
            )
    return curve_rows, gap_rows, metadata


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def panel_limits(rows: list[dict]) -> tuple[float, float]:
    low = min(float(x["q25"]) for x in rows)
    high = max(float(x["q75"]) for x in rows)
    span = max(high - low, 0.01)
    return max(0.0, low - 0.08 * span), min(1.0, high + 0.08 * span)


def draw_panel(axis, rows: list[dict], model: str, signal: str, k: int) -> None:
    subset = [x for x in rows if x["model"] == model and x["signal"] == signal and x["top_k"] == k]
    for label in ("HALL", "REAL"):
        selected = [x for x in subset if x["label"] == label]
        layer = np.asarray([int(x["layer"]) for x in selected])
        median = np.asarray([float(x["median"]) for x in selected])
        q25 = np.asarray([float(x["q25"]) for x in selected])
        q75 = np.asarray([float(x["q75"]) for x in selected])
        axis.plot(layer, median, color=COLORS[label], linewidth=1.8, label=label)
        axis.fill_between(layer, q25, q75, color=COLORS[label], alpha=0.15)
    axis.set_ylim(*panel_limits(subset))
    axis.set_xlim(1, max(int(x["layer"]) for x in subset))
    axis.grid(alpha=0.2)
    axis.set_xlabel("Decoder layer")


def plot_topk(rows: list[dict], k: int) -> list[str]:
    fig, axes = plt.subplots(len(MODELS), len(SIGNAL_NAMES), figsize=(12.5, 13.5))
    for row_index, model in enumerate(MODELS):
        for column, signal in enumerate(SIGNAL_NAMES):
            axis = axes[row_index, column]
            draw_panel(axis, rows, model, signal, k)
            if row_index == 0:
                axis.set_title(SIGNAL_NAMES[signal], fontsize=12, fontweight="bold")
            if column == 0:
                axis.set_ylabel(f"{MODEL_NAMES[model]}\nTop-{k} mass", fontweight="semibold")
            else:
                axis.set_ylabel(f"Top-{k} mass")
            if row_index == 0 and column == 1:
                axis.legend(frameon=False, loc="best")
    fig.suptitle(f"Visual Top-{k} concentration by target label", fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        0.008,
        "Median with IQR bands | all COCO4000 train+test mentions | mention-weighted | panel-specific y-limits",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.975))
    paths = []
    for extension in ("png", "pdf"):
        path = OUT / f"top{k}_raw_attention_ae_real_hall.{extension}"
        fig.savefig(path, dpi=180, facecolor="white")
        paths.append(str(path.relative_to(ROOT)))
    plt.close(fig)
    return paths


def plot_overview(rows: list[dict]) -> list[str]:
    columns = (
        ("raw_attention", 16),
        ("raw_attention", 32),
        ("ae", 16),
        ("ae", 32),
    )
    fig, axes = plt.subplots(len(MODELS), len(columns), figsize=(19, 13.5))
    for row_index, model in enumerate(MODELS):
        for column, (signal, k) in enumerate(columns):
            axis = axes[row_index, column]
            draw_panel(axis, rows, model, signal, k)
            if row_index == 0:
                axis.set_title(f"{SIGNAL_NAMES[signal]}\nTop-{k}", fontsize=11.5, fontweight="bold")
            if column == 0:
                axis.set_ylabel(f"{MODEL_NAMES[model]}\nTop-K mass", fontweight="semibold")
            else:
                axis.set_ylabel("Top-K mass")
            if row_index == 0 and column == len(columns) - 1:
                axis.legend(frameon=False, loc="best")
    fig.suptitle("Visual Top-K concentration: HALL versus REAL", fontsize=16, fontweight="bold")
    fig.text(
        0.5,
        0.008,
        "Median with IQR bands | all COCO4000 train+test mentions | mention-weighted | panel-specific y-limits",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.975))
    paths = []
    for extension in ("png", "pdf"):
        path = OUT / f"top16_top32_raw_attention_ae_real_hall.{extension}"
        fig.savefig(path, dpi=180, facecolor="white")
        paths.append(str(path.relative_to(ROOT)))
    plt.close(fig)
    return paths


def write_summary(gaps: list[dict], metadata: dict[str, dict], figures: dict) -> None:
    lines = [
        "# Raw attention / AE Top-16 与 Top-32 的 REAL/HALL 曲线",
        "",
        "定义：对每个目标、每层分别在视觉 token 内归一化 raw attention 与既有 `attention_evidence`（AE/T）分布；各自选择本分布权重最大的 K 个位置，曲线值为这些位置的概率质量之和。K 为16或32，不足K时保留全部视觉token。",
        "",
        "统计使用四模型 COCO4000 正式缓存中的全部 train+test mentions，mention 等权。蓝色 REAL、红色 HALL；实线为中位数，阴影为25%–75%分位数，不是置信区间。各面板纵轴自适应，跨面板比较绝对高度时应读取刻度。没有重跑VLM、路径积分或检测器。",
        "",
        f"[Top-16图]({Path(figures['top16'][0]).name}) · [Top-32图]({Path(figures['top32'][0]).name}) · [四项总览]({Path(figures['overview'][0]).name}) · [逐层数据](curves.csv) · [差值摘要](gap_summary.csv)",
        "",
        "## 样本",
        "",
        "| 模型 | 图片 | mentions (HALL / REAL) | 唯一目标 | 层 | 视觉token数 min / median / max |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        item = metadata[model]
        lines.append(
            f"| {MODEL_NAMES[model]} | {item['images']} | {item['mentions']} "
            f"({item['n_hall']} / {item['n_real']}) | {item['unique_targets']} | {item['layers']} | "
            f"{item['visual_tokens_min']} / {item['visual_tokens_median']:.0f} / {item['visual_tokens_max']} |"
        )
    lines += [
        "",
        "## 逐层中位数方向",
        "",
        "`H>R层数`只描述逐层中位数方向；`median Δ`是各层 `(HALL−REAL)` 中位数差的中位数，不是分类指标或显著性结论。",
        "",
        "| 模型 | 信号 | K | H>R层数 | R>H层数 | median Δ(H−R) | 最大绝对差@层（带符号） |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    lookup = {(x["model"], x["signal"], x["top_k"]): x for x in gaps}
    for model in MODELS:
        for signal in SIGNAL_NAMES:
            for k in TOP_KS:
                row = lookup[(model, signal, k)]
                lines.append(
                    f"| {MODEL_NAMES[model]} | {SIGNAL_NAMES[signal]} | {k} | "
                    f"{row['hall_higher_layers']}/{row['layers']} | {row['real_higher_layers']}/{row['layers']} | "
                    f"{row['median_hall_minus_real']:+.6f} | "
                    f"{row['hall_minus_real_at_largest_gap']:+.6f}@L{row['largest_gap_layer']} |"
                )
    lines += [
        "",
        "注意：这里的 AE Top-K 是归一化 token 分布 T 的集中度，不是标量 `AE_K=AE×M_K`，也不是 raw attention 与 AE 的 Top-K 重合率。固定K还受视觉token数量与模型分辨率影响，不宜把不同模型的绝对高度直接作机制比较。",
        "",
        "复现：`python scripts/plot_topk_raw_attention_ae_by_label.py`。",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    args = parser.parse_args()
    if tuple(args.models) != MODELS:
        raise ValueError("The cross-model figures require all four models in the canonical order")

    numerics = json.loads((SOURCE / "numerics.json").read_text(encoding="utf-8"))
    if int(numerics["k"]) != 50:
        raise AssertionError("Expected the completed K50 source-composition cache")
    OUT.mkdir(parents=True, exist_ok=True)
    curves, gaps, metadata = [], [], {}
    for model in MODELS:
        model_curves, model_gaps, model_metadata = summarize_model(model)
        curves.extend(model_curves)
        gaps.extend(model_gaps)
        metadata[model] = model_metadata
        print(f"[{model}] summarized {model_metadata['mentions']} mentions", flush=True)

    expected_rows = sum(metadata[x]["layers"] for x in MODELS) * len(SIGNAL_NAMES) * len(TOP_KS) * 2
    if len(curves) != expected_rows or len(gaps) != len(MODELS) * len(SIGNAL_NAMES) * len(TOP_KS):
        raise AssertionError("Incomplete curve summary")
    write_csv(OUT / "curves.csv", curves)
    write_csv(OUT / "gap_summary.csv", gaps)
    figures = {
        "top16": plot_topk(curves, 16),
        "top32": plot_topk(curves, 32),
        "overview": plot_overview(curves),
    }
    validation = {
        "status": "PASS",
        "definition": "sum of each normalized visual distribution's own Top-K mass",
        "signals": {
            "raw_attention": "raw_attention_mean normalized over visual tokens",
            "ae": "attention_evidence (T) renormalized over visual tokens",
        },
        "top_k": list(TOP_KS),
        "cohort": "all COCO4000 train+test mentions; mention-weighted",
        "curve": "median and IQR; descriptive, not CI",
        "source_quadrature_cache": "K50 (only cached distributions read; Top-K is spatial, not quadrature)",
        "models": metadata,
        "rows": {"curves": len(curves), "gap_summary": len(gaps)},
        "figures": figures,
    }
    (OUT / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_summary(gaps, metadata, figures)
    print(f"Wrote curves, summaries and figures to {OUT}", flush=True)


if __name__ == "__main__":
    main()
