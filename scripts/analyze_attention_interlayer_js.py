#!/usr/bin/env python3
"""Analyze layer-to-layer JS for plain attention and gated attention evidence."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.jffn_experiment import load_shards  # noqa: E402
from scripts.analyze_attention_evidence_js import (  # noqa: E402
    EXPERIMENT,
    LABELS,
    MODELS,
    MODEL_NAMES,
    SIGNAL_NAMES,
    SIGNALS,
    label_by_target,
    load_attention_lookup,
    normalize,
)


TOP_K = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--experiment", default=EXPERIMENT)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/attention_js_studies/interlayer_js",
    )
    return parser.parse_args()


def entropy(values: torch.Tensor) -> torch.Tensor:
    return -torch.where(
        values > 0.0,
        values * values.clamp_min(1.0e-12).log(),
        0.0,
    ).sum(dim=-1)


def interlayer_js(values: torch.Tensor, *, top_k: int | None = None) -> torch.Tensor:
    """Return per-token JS matrices with shape ``[tokens,layers,layers]``."""
    probabilities = normalize(values)
    if probabilities.ndim != 3:
        raise ValueError(f"Expected [tokens,layers,visual_tokens], got {probabilities.shape}")
    left = probabilities.unsqueeze(2)
    right = probabilities.unsqueeze(1)
    if top_k is None:
        middle = 0.5 * (left + right)
        base_entropy = entropy(probabilities)
        result = entropy(middle) - 0.5 * (
            base_entropy.unsqueeze(2) + base_entropy.unsqueeze(1)
        )
    else:
        k = min(max(int(top_k), 1), int(probabilities.shape[-1]))
        top_mask = torch.zeros_like(probabilities, dtype=torch.bool)
        top_mask.scatter_(
            -1, torch.topk(probabilities, k=k, dim=-1).indices, True
        )
        union = top_mask.unsqueeze(2) | top_mask.unsqueeze(1)
        left = torch.where(union, left, 0.0)
        right = torch.where(union, right, 0.0)
        left = left / left.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
        right = right / right.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
        result = entropy(0.5 * (left + right)) - 0.5 * (
            entropy(left) + entropy(right)
        )
    result = result.clamp(min=0.0, max=math.log(2.0))
    diagonal = torch.arange(result.shape[-1], device=result.device)
    result[:, diagonal, diagonal] = 0.0
    return result


@dataclass
class RunningMatrix:
    count: int = 0
    total: np.ndarray | None = None
    total_square: np.ndarray | None = None

    def add(self, value: torch.Tensor) -> None:
        array = value.detach().cpu().numpy().astype(np.float64)
        if self.total is None:
            self.total = np.zeros_like(array)
            self.total_square = np.zeros_like(array)
        self.total += array
        self.total_square += array * array
        self.count += 1

    def statistics(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.count == 0 or self.total is None or self.total_square is None:
            raise ValueError("Cannot summarize an empty accumulator")
        mean = self.total / self.count
        variance = np.maximum(self.total_square / self.count - mean * mean, 0.0)
        std = np.sqrt(variance)
        return mean, std, std / math.sqrt(self.count)


def analyze_model(
    *,
    model: str,
    outputs_root: Path,
    experiment: str,
    device: torch.device,
    top_k: int,
    accumulators: dict[tuple[str, str, str, str], RunningMatrix],
) -> dict[str, Any]:
    model_root = outputs_root / model / experiment
    shard_dir = model_root / "results/ffn_visual_source_attribution_v1/shards/full"
    attention_lookup, audit = load_attention_lookup(model_root)
    shard_count = len(list(shard_dir.glob("features*_shard_*.pt")))
    seen_images: set[int] = set()
    seen_targets: set[str] = set()
    label_conflicts = 0
    missing_attention = 0
    shape_mismatches = 0
    label_target_counts = defaultdict(int)
    label_image_counts = defaultdict(int)

    with torch.inference_mode():
        for shard_index, shard in enumerate(load_shards(shard_dir), 1):
            labels, conflicts = label_by_target(shard["sample_table"])
            label_conflicts += conflicts
            positions_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for position in shard["positions"]:
                key = str(position["target_key"])
                if key in seen_targets:
                    raise AssertionError(f"Duplicate attribution target {model}/{key}")
                seen_targets.add(key)
                if key not in labels:
                    continue
                if key not in attention_lookup:
                    missing_attention += 1
                    continue
                attention = torch.from_numpy(attention_lookup[key])
                evidence = torch.as_tensor(position["attention_evidence"]).float()
                if attention.shape != evidence.shape:
                    shape_mismatches += 1
                    continue
                label = int(labels[key])
                positions_by_image[int(position["image_id"])].append(
                    {
                        "label": label,
                        "attention_distribution": attention,
                        "attention_evidence": evidence,
                    }
                )
                label_target_counts[LABELS[label]] += 1

            for image_id, records in positions_by_image.items():
                if image_id in seen_images:
                    raise AssertionError(f"Image {model}/{image_id} spans shards")
                seen_images.add(image_id)
                for label_value, label_name in LABELS.items():
                    selected = [row for row in records if row["label"] == label_value]
                    if not selected:
                        continue
                    label_image_counts[label_name] += 1
                    for signal in SIGNALS:
                        values = torch.stack([row[signal] for row in selected]).to(
                            device=device, non_blocking=True
                        )
                        full = interlayer_js(values).mean(dim=0)
                        union = interlayer_js(values, top_k=top_k).mean(dim=0)
                        accumulators.setdefault(
                            (model, signal, label_name, "all_visual_tokens"),
                            RunningMatrix(),
                        ).add(full)
                        accumulators.setdefault(
                            (model, signal, label_name, f"union_top{top_k}"),
                            RunningMatrix(),
                        ).add(union)
                        del values, full, union
            del shard
            gc.collect()
            if shard_index % 50 == 0 or shard_index == shard_count:
                print(
                    f"[interlayer JS] {model}: {shard_index}/{shard_count} shards",
                    flush=True,
                )

    del attention_lookup
    gc.collect()
    return {
        **audit,
        "attribution_shards": shard_count,
        "seen_images": len(seen_images),
        "seen_targets": len(seen_targets),
        "used_targets": int(sum(label_target_counts.values())),
        "label_target_counts": dict(label_target_counts),
        "label_image_counts": dict(label_image_counts),
        "label_conflict_targets_excluded": label_conflicts,
        "missing_attention": missing_attention,
        "shape_mismatches": shape_mismatches,
    }


def matrix_rows(
    accumulators: dict[tuple[str, str, str, str], RunningMatrix]
) -> list[dict[str, Any]]:
    rows = []
    for (model, signal, label, support), running in sorted(accumulators.items()):
        mean, std, se = running.statistics()
        for left in range(mean.shape[0]):
            for right in range(mean.shape[1]):
                rows.append(
                    {
                        "model": model,
                        "signal": signal,
                        "label": label,
                        "support": support,
                        "layer_left": left + 1,
                        "layer_right": right + 1,
                        "n_images": running.count,
                        "mean": float(mean[left, right]),
                        "std": float(std[left, right]),
                        "se": float(se[left, right]),
                    }
                )
    return rows


def adjacent_rows(
    accumulators: dict[tuple[str, str, str, str], RunningMatrix]
) -> list[dict[str, Any]]:
    rows = []
    for (model, signal, label, support), running in sorted(accumulators.items()):
        mean, std, se = running.statistics()
        for layer in range(mean.shape[0] - 1):
            rows.append(
                {
                    "model": model,
                    "signal": signal,
                    "label": label,
                    "support": support,
                    "layer_from": layer + 1,
                    "layer_to": layer + 2,
                    "n_images": running.count,
                    "mean": float(mean[layer, layer + 1]),
                    "std": float(std[layer, layer + 1]),
                    "se": float(se[layer, layer + 1]),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_adjacent_one(
    path: Path,
    rows: list[dict[str, Any]],
    model: str,
    signal: str,
    support: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    for label, color, linestyle in (
        ("HALL", "#d62728", "-"),
        ("REAL", "#2ca02c", "--"),
    ):
        selected = [
            row
            for row in rows
            if row["model"] == model
            and row["signal"] == signal
            and row["label"] == label
            and row["support"] == support
        ]
        axis.plot(
            [row["layer_from"] for row in selected],
            [row["mean"] for row in selected],
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            label=label,
        )
    support_title = "all visual tokens" if support == "all_visual_tokens" else "Union-Top32"
    axis.set_title(f"{MODEL_NAMES[model]}\n{SIGNAL_NAMES[signal]} — {support_title}")
    axis.set_xlabel("Layer transition l → l+1")
    axis.set_ylabel("Adjacent-layer JS (nats)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path.with_suffix(".png"), dpi=190, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def matrix_from_rows(
    rows: list[dict[str, Any]],
    *,
    model: str,
    signal: str,
    label: str,
    support: str,
) -> np.ndarray:
    selected = [
        row
        for row in rows
        if row["model"] == model
        and row["signal"] == signal
        and row["label"] == label
        and row["support"] == support
    ]
    layer_count = max(int(row["layer_left"]) for row in selected)
    matrix = np.zeros((layer_count, layer_count), dtype=np.float64)
    for row in selected:
        matrix[int(row["layer_left"]) - 1, int(row["layer_right"]) - 1] = float(
            row["mean"]
        )
    return matrix


def plot_heatmap_one(
    path: Path,
    rows: list[dict[str, Any]],
    model: str,
    signal: str,
    support: str,
    label: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    hall = matrix_from_rows(
        rows, model=model, signal=signal, label="HALL", support=support
    )
    real = matrix_from_rows(
        rows, model=model, signal=signal, label="REAL", support=support
    )
    if label == "HALL_MINUS_REAL":
        matrix = hall - real
        limit = max(float(np.abs(matrix).max()), 1.0e-8)
        color_options = {"cmap": "coolwarm", "vmin": -limit, "vmax": limit}
        color_label = "HALL − REAL JS (nats)"
        label_title = "HALL − REAL"
    else:
        matrix = hall if label == "HALL" else real
        color_options = {
            "cmap": "magma",
            "vmin": 0.0,
            "vmax": max(float(hall.max()), float(real.max())),
        }
        color_label = "JS (nats)"
        label_title = label
    figure, axis = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    image = axis.imshow(matrix, origin="lower", **color_options)
    ticks = np.unique(np.linspace(0, matrix.shape[0] - 1, 6).astype(int))
    axis.set_xticks(ticks, ticks + 1)
    axis.set_yticks(ticks, ticks + 1)
    axis.set_xlabel("Layer j")
    axis.set_ylabel("Layer i")
    support_title = "all visual tokens" if support == "all_visual_tokens" else "Union-Top32"
    axis.set_title(
        f"{MODEL_NAMES[model]} — {label_title}\n{SIGNAL_NAMES[signal]} — {support_title}"
    )
    figure.colorbar(image, ax=axis, label=color_label)
    figure.savefig(path.with_suffix(".png"), dpi=190, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def average_adjacent(
    rows: list[dict[str, Any]], *, model: str, signal: str, label: str, support: str
) -> float:
    values = [
        float(row["mean"])
        for row in rows
        if row["model"] == model
        and row["signal"] == signal
        and row["label"] == label
        and row["support"] == support
    ]
    return float(np.mean(values))


def peak_adjacent(
    rows: list[dict[str, Any]], *, model: str, signal: str, label: str, support: str
) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row["model"] == model
        and row["signal"] == signal
        and row["label"] == label
        and row["support"] == support
    ]
    return max(selected, key=lambda row: float(row["mean"]))


def all_pair_summary(
    rows: list[dict[str, Any]], *, model: str, signal: str, support: str
) -> tuple[float, float, float, tuple[int, int, float], tuple[int, int, float]]:
    by_label = {}
    for label in ("HALL", "REAL"):
        by_label[label] = {
            (int(row["layer_left"]), int(row["layer_right"])): float(row["mean"])
            for row in rows
            if row["model"] == model
            and row["signal"] == signal
            and row["label"] == label
            and row["support"] == support
            and int(row["layer_left"]) < int(row["layer_right"])
        }
    keys = sorted(by_label["HALL"])
    hall = np.asarray([by_label["HALL"][key] for key in keys])
    real = np.asarray([by_label["REAL"][key] for key in keys])
    difference = hall - real
    maximum = int(np.argmax(difference))
    minimum = int(np.argmin(difference))
    return (
        float(hall.mean()),
        float(real.mean()),
        float(difference.mean()),
        (*keys[maximum], float(difference[maximum])),
        (*keys[minimum], float(difference[minimum])),
    )


def report(
    *,
    models: list[str],
    audits: dict[str, Any],
    adjacent: list[dict[str, Any]],
    all_pairs: list[dict[str, Any]],
    top_k: int,
) -> str:
    lines = [
        "# Attention distribution / evidence 层间 JS",
        "",
        "原始全视觉 attention mass 未保存；当前 A 在视觉 support 内已归一化、逐层和恒为 1，因此按用户要求不重跑模型并跳过 raw mass。",
        "",
        "层间 JS 在同一个目标 token 内计算；同标签 token 先在图片内平均，再跨图片平均。JS 使用自然对数。Top-32 是待比较两层各自 Top-32 的并集，并在并集内重新归一化。",
        "",
        "## 相邻层汇总",
        "",
        "| Model | Signal | All H/R/H−R | UTop32 H/R/H−R | HALL peak transition (all) |",
        "|---|---|---:|---:|---:|",
    ]
    for model in models:
        for signal in SIGNALS:
            values = {}
            for support in ("all_visual_tokens", f"union_top{top_k}"):
                hall = average_adjacent(
                    adjacent, model=model, signal=signal, label="HALL", support=support
                )
                real = average_adjacent(
                    adjacent, model=model, signal=signal, label="REAL", support=support
                )
                values[support] = f"{hall:.4f}/{real:.4f}/{hall-real:+.4f}"
            peak = peak_adjacent(
                adjacent,
                model=model,
                signal=signal,
                label="HALL",
                support="all_visual_tokens",
            )
            lines.append(
                f"| {MODEL_NAMES[model]} | {'A' if signal == 'attention_distribution' else 'E'} | "
                f"{values['all_visual_tokens']} | {values[f'union_top{top_k}']} | "
                f"L{peak['layer_from']}→L{peak['layer_to']} (`{float(peak['mean']):.4f}`) |"
            )
    lines.extend(
        [
            "",
            "## 全层对汇总",
            "",
            "以下对所有非对角层对等权平均；括号给出全视觉 token 下 HALL−REAL 最大的层对。",
            "",
            "| Model | Signal | All H/R/H−R | UTop32 H/R/H−R | Strongest H−R pair (all) |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for model in models:
        for signal in SIGNALS:
            full = all_pair_summary(
                all_pairs,
                model=model,
                signal=signal,
                support="all_visual_tokens",
            )
            top = all_pair_summary(
                all_pairs,
                model=model,
                signal=signal,
                support=f"union_top{top_k}",
            )
            lines.append(
                f"| {MODEL_NAMES[model]} | {'A' if signal == 'attention_distribution' else 'E'} | "
                f"{full[0]:.4f}/{full[1]:.4f}/{full[2]:+.4f} | "
                f"{top[0]:.4f}/{top[1]:.4f}/{top[2]:+.4f} | "
                f"L{full[3][0]}↔L{full[3][1]} (`{full[3][2]:+.4f}`) |"
            )
    lines.extend(
        [
            "",
            "四模型的 HALL 全层对平均 JS 都高于 REAL，但差值很小（全视觉约 `+0.008–+0.022` nats），且差值热力图包含蓝色层对；因此这是平均层间不稳定性趋势，不是每个层对都成立。四模型最强正差都连接约 L7 与中后层，提示 HALL 的主要额外变化更像早期到中后期的轨迹偏移。E 与 A 的相邻层均值几乎相同，gate 主要改变长程层对差异，没有制造一致的新相邻层突变。",
        ]
    )
    lines.extend(
        [
            "",
            "## 图的含义",
            "",
            "1. `figures/adjacent/{A或E}/{all或top32}/{model}`：每张图只有一个模型、一个信号和一种视觉 support；仅保留 HALL/REAL 两条线。横轴 L 表示 L→L+1，纵轴是同一 token 在相邻两层的 JS；越高表示该层转换处重排越强。",
            "2. `figures/all_pair/{A或E}/{all或top32}/{HALL或REAL或HALL_MINUS_REAL}/{model}`：每张图只有一张热力图。横纵轴是任意两层；HALL/REAL 图颜色越亮表示两层差异越大。差值图中红色表示 HALL 层间变化更大，蓝色表示 REAL 更大。对角线恒为 0，矩阵关于对角线对称。",
            "",
            "这些图衡量的是层间稳定性，不直接衡量空间定位是否正确，也不是幻觉检测 AUROC。",
            "",
            "## 数据审计",
            "",
            "| Model | HALL/REAL targets | HALL/REAL images | Conflicts excluded | Missing/shape mismatch |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for model in models:
        audit = audits[model]
        lines.append(
            f"| {MODEL_NAMES[model]} | "
            f"{audit['label_target_counts'].get('HALL', 0)}/{audit['label_target_counts'].get('REAL', 0)} | "
            f"{audit['label_image_counts'].get('HALL', 0)}/{audit['label_image_counts'].get('REAL', 0)} | "
            f"{audit['label_conflict_targets_excluded']} | "
            f"{audit['missing_attention']}/{audit['shape_mismatches']} |"
        )
    lines.extend(
        [
            "",
            "## 产物",
            "",
            "- `adjacent_layer_js.csv`：相邻层 image-balanced 统计。",
            "- `all_pair_interlayer_js.csv`：全部层对的均值、标准差和标准误。",
            "- `figures/adjacent/`：16 组单模型相邻层曲线；`figures/all_pair/`：48 组单模型、单标签热力图，均提供 PNG/PDF。",
            "- `figures/combined_overview/`：旧合并图，仅作版本追溯。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {device}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    accumulators: dict[tuple[str, str, str, str], RunningMatrix] = {}
    audits = {}
    for model in args.models:
        print(f"[interlayer JS] loading {model} on {device}", flush=True)
        audits[model] = analyze_model(
            model=model,
            outputs_root=args.outputs_root,
            experiment=args.experiment,
            device=device,
            top_k=args.top_k,
            accumulators=accumulators,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    all_rows = matrix_rows(accumulators)
    adjacent = adjacent_rows(accumulators)
    write_csv(args.output_dir / "all_pair_interlayer_js.csv", all_rows)
    write_csv(args.output_dir / "adjacent_layer_js.csv", adjacent)
    for model in args.models:
        for signal in SIGNALS:
            signal_slug = "A" if signal == "attention_distribution" else "E"
            for support, support_slug in (
                ("all_visual_tokens", "all_visual_tokens"),
                (f"union_top{args.top_k}", f"union_top{args.top_k}"),
            ):
                plot_adjacent_one(
                    args.output_dir
                    / "figures/adjacent"
                    / signal_slug
                    / support_slug
                    / model,
                    adjacent,
                    model,
                    signal,
                    support,
                )
                for label in ("HALL", "REAL", "HALL_MINUS_REAL"):
                    plot_heatmap_one(
                        args.output_dir
                        / "figures/all_pair"
                        / signal_slug
                        / support_slug
                        / label
                        / model,
                        all_rows,
                        model,
                        signal,
                        support,
                        label,
                    )
    payload = {
        "schema_version": "attention-interlayer-js-v1",
        "raw_visual_attention_mass": {
            "status": "NOT RUN",
            "reason": "unnormalized total visual-attention mass was not persisted",
        },
        "device": str(device),
        "top_k": args.top_k,
        "js_log_base": "natural",
        "aggregation": "unique targets averaged within image and label, then images averaged",
        "audits": audits,
        "adjacent_layer_js": adjacent,
        "all_pair_interlayer_js": all_rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "summary.md").write_text(
        report(
            models=args.models,
            audits=audits,
            adjacent=adjacent,
            all_pairs=all_rows,
            top_k=args.top_k,
        ),
        encoding="utf-8",
    )
    print(f"[interlayer JS] wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
