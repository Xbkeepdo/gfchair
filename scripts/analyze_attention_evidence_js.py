#!/usr/bin/env python3
"""Compare HALL/REAL attention maps and attention-to-evidence gate effects."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
from collections import defaultdict
from itertools import combinations, product
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.jffn_experiment import load_shards  # noqa: E402
from utils.io_utils import load_pkl  # noqa: E402


EPS = 1.0e-12
TOP_K = 32
EXPERIMENT = "COCO4000-INSLEN-OFFICIAL-TARGET"
MODELS = (
    "llava_1_5_7b",
    "qwen2_5_vl_7b",
    "qwen3_vl_8b",
    "internvl_2_5_8b",
)
MODEL_NAMES = {
    "llava_1_5_7b": "LLaVA-1.5-7B",
    "qwen2_5_vl_7b": "Qwen2.5-VL-7B",
    "qwen3_vl_8b": "Qwen3-VL-8B",
    "internvl_2_5_8b": "InternVL2.5-8B",
}
SIGNALS = ("attention_distribution", "attention_evidence")
SIGNAL_NAMES = {
    "attention_distribution": "A: plain attention distribution",
    "attention_evidence": "E: gate-weighted attention evidence",
}
LABELS = {0: "HALL", 1: "REAL"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--experiment", default=EXPERIMENT)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/attention_js_studies/hall_real_target_js",
    )
    parser.add_argument("--top-k", type=int, default=TOP_K)
    return parser.parse_args()


def normalize(values: torch.Tensor) -> torch.Tensor:
    values = torch.nan_to_num(
        values.float(), nan=0.0, posinf=0.0, neginf=0.0
    ).clamp_min(0.0)
    total = values.sum(dim=-1, keepdim=True)
    uniform = torch.full_like(values, 1.0 / max(int(values.shape[-1]), 1))
    return torch.where(total > 0.0, values / total.clamp_min(EPS), uniform)


def js_divergence(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Natural-log Jensen-Shannon divergence along the final dimension."""
    left, right = normalize(left), normalize(right)
    middle = 0.5 * (left + right)
    left_term = torch.where(
        left > 0.0,
        left * (left.clamp_min(EPS).log() - middle.clamp_min(EPS).log()),
        0.0,
    )
    right_term = torch.where(
        right > 0.0,
        right * (right.clamp_min(EPS).log() - middle.clamp_min(EPS).log()),
        0.0,
    )
    return 0.5 * (left_term + right_term).sum(dim=-1)


def union_topk_js(
    left: torch.Tensor, right: torch.Tensor, *, top_k: int = TOP_K
) -> torch.Tensor:
    """JS after Top-K-per-side union and renormalization inside that union."""
    left, right = normalize(left), normalize(right)
    if left.shape != right.shape or left.ndim < 2:
        raise ValueError(f"Expected matching [...,layers,tokens], got {left.shape}/{right.shape}")
    k = min(max(int(top_k), 1), int(left.shape[-1]))
    mask = torch.zeros_like(left, dtype=torch.bool)
    mask.scatter_(-1, torch.topk(left, k=k, dim=-1).indices, True)
    mask.scatter_(-1, torch.topk(right, k=k, dim=-1).indices, True)
    return js_divergence(
        torch.where(mask, left, 0.0),
        torch.where(mask, right, 0.0),
    )


def normalized_entropy(values: torch.Tensor) -> torch.Tensor:
    values = normalize(values)
    return -(values * values.clamp_min(EPS).log()).sum(dim=-1) / math.log(
        max(int(values.shape[-1]), 2)
    )


def topk_mass(values: torch.Tensor, top_k: int) -> torch.Tensor:
    values = normalize(values)
    return torch.topk(values, k=min(int(top_k), int(values.shape[-1])), dim=-1).values.sum(-1)


def feature_parts(model_root: Path) -> list[Path]:
    parts = sorted(model_root.glob("features.part*.pkl"))
    return parts or [model_root / "features.pkl"]


def load_attention_lookup(model_root: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    lookup: dict[str, np.ndarray] = {}
    duplicate_rows = 0
    maximum_duplicate_error = 0.0
    rows_seen = 0
    for path in feature_parts(model_root):
        rows = load_pkl(str(path))
        for row in rows:
            rows_seen += 1
            key = f"{int(row['image_id'])}:{int(row['response_token_idx'])}"
            value = np.asarray(
                row["dgst_t_attention_support_per_layer"], dtype=np.float32
            )
            if key in lookup:
                duplicate_rows += 1
                error = float(np.max(np.abs(lookup[key] - value)))
                maximum_duplicate_error = max(maximum_duplicate_error, error)
                if error > 1.0e-6:
                    raise AssertionError(f"Conflicting attention for {key}: {error}")
            else:
                lookup[key] = value
        del rows
        gc.collect()
    return lookup, {
        "feature_parts": [str(path) for path in feature_parts(model_root)],
        "rows_seen": rows_seen,
        "unique_positions": len(lookup),
        "duplicate_rows": duplicate_rows,
        "maximum_duplicate_error": maximum_duplicate_error,
    }


class ImageValues:
    """Keep one layerwise mean per image for image-balanced summaries."""

    def __init__(self) -> None:
        self.values: dict[tuple[str, ...], list[np.ndarray]] = defaultdict(list)

    def add(self, key: tuple[str, ...], value: torch.Tensor) -> None:
        self.values[key].append(value.detach().cpu().float().numpy())

    def rows(self, names: tuple[str, ...]) -> list[dict[str, Any]]:
        output = []
        for key, arrays in sorted(self.values.items()):
            matrix = np.stack(arrays).astype(np.float64)
            for layer in range(matrix.shape[1]):
                column = matrix[:, layer]
                output.append(
                    {
                        **dict(zip(names, key)),
                        "layer": layer + 1,
                        "n_images": int(column.size),
                        "mean": float(column.mean()),
                        "std": float(column.std()),
                        "median": float(np.median(column)),
                        "q25": float(np.quantile(column, 0.25)),
                        "q75": float(np.quantile(column, 0.75)),
                    }
                )
        return output


def label_by_target(sample_table: Iterable[dict[str, Any]]) -> tuple[dict[str, int], int]:
    raw: dict[str, set[int]] = defaultdict(set)
    for mention in sample_table:
        raw[str(mention["target_key"])].add(int(mention["label"]))
    conflicts = sum(len(labels) != 1 for labels in raw.values())
    return {
        key: next(iter(labels)) for key, labels in raw.items() if len(labels) == 1
    }, conflicts


def pairs_for(labels: list[int], pair_type: str) -> list[tuple[int, int]]:
    hall = [index for index, label in enumerate(labels) if label == 0]
    real = [index for index, label in enumerate(labels) if label == 1]
    if pair_type == "HALL-REAL":
        return list(product(hall, real))
    indices = hall if pair_type == "HALL-HALL" else real
    return list(combinations(indices, 2))


def analyze_model(
    *,
    model: str,
    outputs_root: Path,
    experiment: str,
    top_k: int,
    pair_values: ImageValues,
    gate_values: ImageValues,
    shape_values: ImageValues,
) -> dict[str, Any]:
    model_root = outputs_root / model / experiment
    shard_dir = model_root / "results/ffn_visual_source_attribution_v1/shards/full"
    attention_lookup, audit = load_attention_lookup(model_root)
    seen_images: set[int] = set()
    seen_targets: set[str] = set()
    used_targets = 0
    label_conflicts = 0
    missing_attention = 0
    shape_mismatches = 0
    label_target_counts = defaultdict(int)
    pair_counts = defaultdict(int)
    pair_image_counts = defaultdict(int)

    for shard in load_shards(shard_dir):
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
            used_targets += 1
            label_target_counts[LABELS[label]] += 1

        for image_id, records in positions_by_image.items():
            if image_id in seen_images:
                raise AssertionError(f"Image {model}/{image_id} spans attribution shards")
            seen_images.add(image_id)
            record_labels = [int(record["label"]) for record in records]
            for label_value, label_name in LABELS.items():
                selected = [record for record in records if record["label"] == label_value]
                if not selected:
                    continue
                attention = torch.stack(
                    [record["attention_distribution"] for record in selected]
                )
                evidence = torch.stack(
                    [record["attention_evidence"] for record in selected]
                )
                gate_values.add(
                    (model, label_name, "all_visual_tokens"),
                    js_divergence(attention, evidence).mean(dim=0),
                )
                gate_values.add(
                    (model, label_name, f"union_top{top_k}"),
                    union_topk_js(attention, evidence, top_k=top_k).mean(dim=0),
                )
                for signal, values in (
                    ("attention_distribution", attention),
                    ("attention_evidence", evidence),
                ):
                    shape_values.add(
                        (model, signal, label_name, "normalized_entropy"),
                        normalized_entropy(values).mean(dim=0),
                    )
                    shape_values.add(
                        (model, signal, label_name, f"top{top_k}_mass"),
                        topk_mass(values, top_k).mean(dim=0),
                    )

            for pair_type in ("HALL-REAL", "HALL-HALL", "REAL-REAL"):
                pairs = pairs_for(record_labels, pair_type)
                if not pairs:
                    continue
                pair_counts[pair_type] += len(pairs)
                pair_image_counts[pair_type] += 1
                left_indices = [pair[0] for pair in pairs]
                right_indices = [pair[1] for pair in pairs]
                for signal in SIGNALS:
                    values = torch.stack([record[signal] for record in records])
                    left = values[left_indices]
                    right = values[right_indices]
                    pair_values.add(
                        (model, signal, pair_type, "all_visual_tokens"),
                        js_divergence(left, right).mean(dim=0),
                    )
                    pair_values.add(
                        (model, signal, pair_type, f"union_top{top_k}"),
                        union_topk_js(left, right, top_k=top_k).mean(dim=0),
                    )
        del shard
        gc.collect()

    del attention_lookup
    gc.collect()
    return {
        **audit,
        "attribution_shards": len(list(shard_dir.glob("features*_shard_*.pt"))),
        "seen_images": len(seen_images),
        "seen_targets": len(seen_targets),
        "used_targets": used_targets,
        "label_target_counts": dict(label_target_counts),
        "label_conflict_targets_excluded": label_conflicts,
        "missing_attention": missing_attention,
        "shape_mismatches": shape_mismatches,
        "same_image_pair_counts": dict(pair_counts),
        "same_image_pair_image_counts": dict(pair_image_counts),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_pair_js(
    path: Path,
    rows: list[dict[str, Any]],
    models: list[str],
    top_k: int,
    signal: str,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False, sharey=False)
    styles = {
        "all_visual_tokens": ("-", "All visual tokens"),
        f"union_top{top_k}": ("--", f"Union-Top{top_k}"),
    }
    for axis, model in zip(axes.flat, models):
        for support, (linestyle, label) in styles.items():
            selected = [
                row for row in rows
                if row["model"] == model
                and row["signal"] == signal
                and row["pair_type"] == "HALL-REAL"
                and row["support"] == support
            ]
            axis.plot(
                [row["layer"] for row in selected],
                [row["mean"] for row in selected],
                color="#1f77b4" if signal == "attention_distribution" else "#d62728",
                linestyle=linestyle,
                linewidth=1.8,
                label=label,
            )
        axis.set_title(MODEL_NAMES[model])
        axis.set_xlabel("Decoder layer")
        axis.set_ylabel("same-image HALL–REAL JS (nats)")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    for axis in list(axes.flat)[len(models):]:
        axis.set_visible(False)
    figure.suptitle(f"{SIGNAL_NAMES[signal]}: HALL vs REAL target-token JS")
    figure.tight_layout()
    figure.savefig(path.with_suffix(".png"), dpi=190, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_gate_js(path: Path, rows: list[dict[str, Any]], models: list[str], top_k: int) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False, sharey=False)
    styles = {
        ("HALL", "all_visual_tokens"): ("#d62728", "-", "HALL / all"),
        ("REAL", "all_visual_tokens"): ("#2ca02c", "-", "REAL / all"),
        ("HALL", f"union_top{top_k}"): ("#d62728", "--", f"HALL / UTop{top_k}"),
        ("REAL", f"union_top{top_k}"): ("#2ca02c", "--", f"REAL / UTop{top_k}"),
    }
    for axis, model in zip(axes.flat, models):
        for (label_name, support), (color, linestyle, label) in styles.items():
            selected = [
                row for row in rows
                if row["model"] == model
                and row["label"] == label_name
                and row["support"] == support
            ]
            axis.plot(
                [row["layer"] for row in selected],
                [row["mean"] for row in selected],
                color=color,
                linestyle=linestyle,
                linewidth=1.8,
                label=label,
            )
        axis.set_title(MODEL_NAMES[model])
        axis.set_xlabel("Decoder layer")
        axis.set_ylabel("JS(A, E) on the same token (nats)")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    for axis in list(axes.flat)[len(models):]:
        axis.set_visible(False)
    figure.suptitle("How much semantic gating changes attention")
    figure.tight_layout()
    figure.savefig(path.with_suffix(".png"), dpi=190, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_shape_metric(
    path: Path,
    rows: list[dict[str, Any]],
    models: list[str],
    metric: str,
    signal: str,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False, sharey=False)
    styles = {
        "HALL": ("#d62728", "-"),
        "REAL": ("#2ca02c", "--"),
    }
    for axis, model in zip(axes.flat, models):
        for label_name, (color, linestyle) in styles.items():
            selected = [
                row for row in rows
                if row["model"] == model
                and row["signal"] == signal
                and row["label"] == label_name
                and row["metric"] == metric
            ]
            axis.plot(
                [row["layer"] for row in selected],
                [row["mean"] for row in selected],
                color=color,
                linestyle=linestyle,
                linewidth=1.6,
                label=label_name,
            )
        axis.set_title(MODEL_NAMES[model])
        axis.set_xlabel("Decoder layer")
        axis.set_ylabel(metric.replace("_", " "))
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    for axis in list(axes.flat)[len(models):]:
        axis.set_visible(False)
    figure.suptitle(f"{SIGNAL_NAMES[signal]}: {metric.replace('_', ' ')}")
    figure.tight_layout()
    figure.savefig(path.with_suffix(".png"), dpi=190, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def mean_over_layers(
    rows: list[dict[str, Any]], **filters: str
) -> float:
    values = [
        float(row["mean"])
        for row in rows
        if all(str(row[key]) == str(value) for key, value in filters.items())
    ]
    return float(np.mean(values)) if values else float("nan")


def build_report(
    *,
    models: list[str],
    top_k: int,
    audits: dict[str, Any],
    pair_rows: list[dict[str, Any]],
    gate_rows: list[dict[str, Any]],
    shape_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Attention distribution / evidence：HALL–REAL 逐层 JS",
        "",
        "主结果在同一图片内对唯一 HALL 与 REAL 目标 token 配对；每张图片先平均，再跨图片平均。JS 使用自然对数。Top-32 使用两侧各自 Top-32 的并集，并在并集内重新归一化。",
        "",
        "| Model | Targets H/R | Images with H–R pairs | A H–R JS all / UTop32 | E H–R JS all / UTop32 | Same-token JS(A,E) H all / UTop32 | Same-token JS(A,E) R all / UTop32 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in models:
        audit = audits[model]
        h = audit["label_target_counts"].get("HALL", 0)
        r = audit["label_target_counts"].get("REAL", 0)
        pair_images = audit["same_image_pair_image_counts"].get("HALL-REAL", 0)
        pair_metrics = []
        for signal in SIGNALS:
            pair_metrics.append(
                f"{mean_over_layers(pair_rows, model=model, signal=signal, pair_type='HALL-REAL', support='all_visual_tokens'):.4f} / "
                f"{mean_over_layers(pair_rows, model=model, signal=signal, pair_type='HALL-REAL', support=f'union_top{top_k}'):.4f}"
            )
        gate_metrics = []
        for label in ("HALL", "REAL"):
            gate_metrics.append(
                f"{mean_over_layers(gate_rows, model=model, label=label, support='all_visual_tokens'):.4f} / "
                f"{mean_over_layers(gate_rows, model=model, label=label, support=f'union_top{top_k}'):.4f}"
            )
        lines.append(
            f"| {MODEL_NAMES[model]} | {h}/{r} | {pair_images} | "
            f"{pair_metrics[0]} | {pair_metrics[1]} | {gate_metrics[0]} | {gate_metrics[1]} |"
        )
    lines.extend(["", "## 主要发现", ""])
    for model in models:
        a_all = mean_over_layers(
            pair_rows,
            model=model,
            signal="attention_distribution",
            pair_type="HALL-REAL",
            support="all_visual_tokens",
        )
        e_all = mean_over_layers(
            pair_rows,
            model=model,
            signal="attention_evidence",
            pair_type="HALL-REAL",
            support="all_visual_tokens",
        )
        a_top = mean_over_layers(
            pair_rows,
            model=model,
            signal="attention_distribution",
            pair_type="HALL-REAL",
            support=f"union_top{top_k}",
        )
        e_top = mean_over_layers(
            pair_rows,
            model=model,
            signal="attention_evidence",
            pair_type="HALL-REAL",
            support=f"union_top{top_k}",
        )
        gate_h = mean_over_layers(
            gate_rows, model=model, label="HALL", support="all_visual_tokens"
        )
        gate_r = mean_over_layers(
            gate_rows, model=model, label="REAL", support="all_visual_tokens"
        )
        entropy_h = mean_over_layers(
            shape_rows,
            model=model,
            signal="attention_distribution",
            label="HALL",
            metric="normalized_entropy",
        )
        entropy_r = mean_over_layers(
            shape_rows,
            model=model,
            signal="attention_distribution",
            label="REAL",
            metric="normalized_entropy",
        )
        mass_h = mean_over_layers(
            shape_rows,
            model=model,
            signal="attention_distribution",
            label="HALL",
            metric=f"top{top_k}_mass",
        )
        mass_r = mean_over_layers(
            shape_rows,
            model=model,
            signal="attention_distribution",
            label="REAL",
            metric=f"top{top_k}_mass",
        )
        peaks = {}
        for signal in SIGNALS:
            candidates = [
                row
                for row in pair_rows
                if row["model"] == model
                and row["signal"] == signal
                and row["pair_type"] == "HALL-REAL"
                and row["support"] == "all_visual_tokens"
            ]
            peaks[signal] = max(candidates, key=lambda row: float(row["mean"]))
        lines.append(
            f"- **{MODEL_NAMES[model]}**：E 相对 A 将 H–R JS 提高 "
            f"`{e_all - a_all:+.4f}`（全视觉）/`{e_top - a_top:+.4f}`（Union-Top{top_k}）；"
            f"A/E 全视觉峰值分别为 L{peaks['attention_distribution']['layer']}="
            f"`{float(peaks['attention_distribution']['mean']):.4f}`、"
            f"L{peaks['attention_evidence']['layer']}="
            f"`{float(peaks['attention_evidence']['mean']):.4f}`。"
            f"同 token gate-effect 的 H–R 差仅 `{gate_h - gate_r:+.4f}`；"
            f"plain attention 的 H–R entropy/Top-{top_k} mass 差为 "
            f"`{entropy_h - entropy_r:+.4f}`/`{mass_h - mass_r:+.4f}`。"
        )
    lines.extend(
        [
            "",
            "同图、同标签目标对照（跨层均值，全视觉 token）：",
            "",
            "| Model | A: H–R / H–H / R–R | E: H–R / H–H / R–R |",
            "|---|---:|---:|",
        ]
    )
    for model in models:
        controls = []
        for signal in SIGNALS:
            controls.append(
                " / ".join(
                    f"{mean_over_layers(pair_rows, model=model, signal=signal, pair_type=pair_type, support='all_visual_tokens'):.4f}"
                    for pair_type in ("HALL-REAL", "HALL-HALL", "REAL-REAL")
                )
            )
        lines.append(f"| {MODEL_NAMES[model]} | {controls[0]} | {controls[1]} |")
    lines.extend(
        [
            "",
            "四模型的 H–R 均略高于 R–R，但 E 同时提高了 H–R、H–H 和 R–R 的不同目标 JS；因此 E 的更高 H–R JS 主要说明 gate 让目标条件分布更分化，不能单独解释为幻觉特异性。",
        ]
    )
    lines.extend(
        [
            "",
            "## 每张图是什么意思",
            "",
            "1. **`same_image_hall_real_js_attention_distribution`**：只画 A。横轴是 decoder layer，纵轴是在同一图片内 HALL token 与 REAL token 的平均 JS。实线使用全部视觉 token，虚线使用双方 Top-32 并集。越高表示两个目标词的 plain attention 空间分布越不同，但不表示哪一个更正确。",
            "2. **`same_image_hall_real_js_attention_evidence`**：只画 E，坐标和配对方式与上一图相同。与 A 图相比整体升高，表示 gate 加权后不同目标词的证据分布更分化；这本身不等于更强的幻觉检测能力。",
            "3. **`same_token_attention_to_evidence_js`**：比较同一个 token 的 A 与 E。红色是 HALL，绿色是 REAL；实线为全部视觉 token，虚线为 Union-Top32。值越高表示 semantic gate 对原 attention 的重排越强；红绿差才是这种重排是否与幻觉标签相关。",
            "4. **`normalized_entropy_attention_distribution` / `normalized_entropy_attention_evidence`**：A、E 分开画。红色 HALL、绿色 REAL。值接近 1 表示在视觉 token 上更均匀，值越低表示越集中。",
            f"5. **`top{top_k}_mass_attention_distribution` / `top{top_k}_mass_attention_evidence`**：A、E 分开画，显示各分布自己的 Top-{top_k} token 承载了多少概率质量；越高表示头部越集中。该指标受视觉 token 总数影响，适合在同一模型内比较层和标签，不宜直接比较模型间绝对值。",
            "",
            "## 口径",
            "",
            "- `A`/`attention_distribution` 使用主特征中的 `dgst_t_attention_support_per_layer`；它与 LLaVA/InternVL 二轮 shard 的同名量抽样误差不超过 FP32 舍入。Qwen 没有二轮 shard，因此使用这一同定义来源。",
            "- `E`/`attention_evidence = normalize(A × hpre_raw_logit_gauss_gate)`。",
            "- `same-image HALL–REAL JS` 衡量同图不同目标词的空间分布差异；它不是 HALL/REAL 分类器 AUROC。",
            "- 归一化 attention 不保留原始 total visual-attention mass；本分析不能恢复该量。",
            "",
            "## 产物",
            "",
            "- `same_image_hall_real_js_layerwise.csv`：H–R/H–H/R–R 同图配对 JS。",
            "- `same_token_attention_to_evidence_js_layerwise.csv`：语义 gate 对同一 token 注意力的改变量。",
            "- `attention_signal_shape_layerwise.csv`：两种分布的 normalized entropy 与 own-Top32 mass。",
            "- 七组分离 PNG/PDF：A/E 主 JS、同 token gate 改变量、A/E entropy、A/E Top-32 mass；旧 combined 图保留作追溯。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pair_values, gate_values, shape_values = ImageValues(), ImageValues(), ImageValues()
    audits = {}
    for model in args.models:
        print(f"[attention JS] loading {model}", flush=True)
        audits[model] = analyze_model(
            model=model,
            outputs_root=args.outputs_root,
            experiment=args.experiment,
            top_k=args.top_k,
            pair_values=pair_values,
            gate_values=gate_values,
            shape_values=shape_values,
        )
        print(
            f"[attention JS] {model}: {audits[model]['used_targets']} targets, "
            f"{audits[model]['same_image_pair_image_counts'].get('HALL-REAL', 0)} H-R images",
            flush=True,
        )

    pair_rows = pair_values.rows(("model", "signal", "pair_type", "support"))
    gate_rows = gate_values.rows(("model", "label", "support"))
    shape_rows = shape_values.rows(("model", "signal", "label", "metric"))
    write_csv(args.output_dir / "same_image_hall_real_js_layerwise.csv", pair_rows)
    write_csv(
        args.output_dir / "same_token_attention_to_evidence_js_layerwise.csv",
        gate_rows,
    )
    write_csv(args.output_dir / "attention_signal_shape_layerwise.csv", shape_rows)
    for signal in SIGNALS:
        plot_pair_js(
            args.output_dir / f"same_image_hall_real_js_{signal}",
            pair_rows,
            args.models,
            args.top_k,
            signal,
        )
    plot_gate_js(
        args.output_dir / "same_token_attention_to_evidence_js",
        gate_rows,
        args.models,
        args.top_k,
    )
    for signal in SIGNALS:
        plot_shape_metric(
            args.output_dir / f"normalized_entropy_{signal}",
            shape_rows,
            args.models,
            "normalized_entropy",
            signal,
        )
        plot_shape_metric(
            args.output_dir / f"top{args.top_k}_mass_{signal}",
            shape_rows,
            args.models,
            f"top{args.top_k}_mass",
            signal,
        )
    payload = {
        "schema_version": "attention-evidence-js-hall-real-v1",
        "top_k": args.top_k,
        "js_log_base": "natural",
        "primary_aggregation": "same-image unique-target pairs; pair mean within image; image mean across cohort",
        "attention_distribution_source": "dgst_t_attention_support_per_layer",
        "attention_evidence_definition": "normalize(attention_distribution * hpre_raw_logit_gauss_gate)",
        "audits": audits,
        "same_image_pair_js": pair_rows,
        "same_token_attention_to_evidence_js": gate_rows,
        "signal_shape": shape_rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "summary.md").write_text(
        build_report(
            models=args.models,
            top_k=args.top_k,
            audits=audits,
            pair_rows=pair_rows,
            gate_rows=gate_rows,
            shape_rows=shape_rows,
        ),
        encoding="utf-8",
    )
    print(f"[attention JS] wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
