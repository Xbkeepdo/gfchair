#!/usr/bin/env python3
"""Compare path FFN attribution with visual WRITE for REAL and HALL mentions."""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from scipy.stats import rankdata

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.tc_fvpa_artifacts import atomic_json_save  # noqa: E402
from scripts.run_ffn_visual_source_attribution import EXPERIMENT, result_root  # noqa: E402
from scripts.tc_fvpa_common import FORMAL_LAYERS_BY_MODEL  # noqa: E402
from utils.io_utils import load_json  # noqa: E402

EPS = 1e-12
MODELS = {
    "qwen2_5_vl_7b": "Qwen2.5-VL-7B",
    "llava_1_5_7b": "LLaVA-1.5-7B",
    "qwen3_vl_8b": "Qwen3-VL-8B",
    "internvl_2_5_8b": "InternVL2.5-8B",
}
LABELS = ((1, "REAL"), (0, "HALL"))
CURVES = {
    "spearman": ("Spearman(P_WRITE, P_FFN)", "median", False),
    "js": ("JS(P_WRITE, P_FFN)", "median", False),
    "top1_agreement": ("Top-1 agreement", "mean", False),
    "top32_overlap": ("Top-32 overlap", "median", False),
    "r_amp": ("R_amp = TV(P_FFN, P_WRITE)", "median", False),
    "s_write": ("S_WRITE = sum ||a_m||", "median", True),
    "s_ffn": ("S_FFN = sum ||e_m||", "median", True),
    "gross_gain": ("S_FFN / S_WRITE", "median", False),
}


def _row_correlation(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left - left.mean(axis=1, keepdims=True)
    right = right - right.mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    output = np.zeros(left.shape[0], dtype=np.float64)
    valid = denominator > EPS
    output[valid] = (left[valid] * right[valid]).sum(axis=1) / denominator[valid]
    output[~valid] = np.all(np.isclose(left[~valid], right[~valid]), axis=1)
    return np.clip(output, -1.0, 1.0)


def distribution_metrics(
    p_write: torch.Tensor, p_ffn: torch.Tensor, *, top_k: int = 32
) -> tuple[dict[str, np.ndarray], dict[str, float | int]]:
    """Return layerwise map metrics and normalization/TV identity checks."""
    p = torch.as_tensor(p_write, dtype=torch.float64)
    q = torch.as_tensor(p_ffn, dtype=torch.float64)
    if p.ndim != 2 or p.shape != q.shape or p.shape[1] == 0:
        raise ValueError("P_WRITE and P_FFN must share a non-empty [layers,tokens] shape")
    if not bool(torch.isfinite(p).all() and torch.isfinite(q).all()):
        raise ValueError("P_WRITE/P_FFN contains non-finite values")
    if float(torch.minimum(p.min(), q.min())) < -1e-7:
        raise ValueError("P_WRITE/P_FFN must be non-negative")
    p = p.clamp_min(0)
    q = q.clamp_min(0)
    p_sum, q_sum = p.sum(-1), q.sum(-1)
    if bool((p_sum <= EPS).any() or (q_sum <= EPS).any()):
        raise ValueError("P_WRITE/P_FFN has zero total mass")
    normalization_error = float(
        torch.maximum((p_sum - 1).abs().max(), (q_sum - 1).abs().max())
    )
    if normalization_error > 2e-4:
        raise AssertionError(f"P_WRITE/P_FFN normalization error {normalization_error}")
    p, q = p / p_sum[:, None], q / q_sum[:, None]

    delta = q - p
    r_amp = 0.5 * delta.abs().sum(-1)
    positive = delta.clamp_min(0).sum(-1)
    negative = (-delta).clamp_min(0).sum(-1)
    midpoint = 0.5 * (p + q)
    js = 0.5 * (
        torch.where(p > 0, p * (p.log() - midpoint.log()), 0).sum(-1)
        + torch.where(q > 0, q * (q.log() - midpoint.log()), 0).sum(-1)
    )

    count = min(int(top_k), int(p.shape[1]))
    p_top = torch.topk(p, k=count, dim=-1).indices
    q_top = torch.topk(q, k=count, dim=-1).indices
    overlap = (
        (p_top.unsqueeze(-1) == q_top.unsqueeze(-2)).any(-1).sum(-1).double()
        / count
    )
    p_np, q_np = p.numpy(), q.numpy()
    spearman = _row_correlation(
        rankdata(p_np, axis=1, method="average"),
        rankdata(q_np, axis=1, method="average"),
    )
    metrics = {
        "spearman": spearman,
        "js": js.numpy(),
        "top1_agreement": (p.argmax(-1) == q.argmax(-1)).double().numpy(),
        "top32_overlap": overlap.numpy(),
        "r_amp": r_amp.numpy(),
    }
    if not all(np.isfinite(value).all() for value in metrics.values()):
        raise ValueError("Map comparison produced non-finite values")
    audit = {
        "normalization_error_max": normalization_error,
        "signed_delta_sum_abs_max": float(delta.sum(-1).abs().max()),
        "tv_positive_mass_error_max": float((r_amp - positive).abs().max()),
        "tv_negative_mass_error_max": float((r_amp - negative).abs().max()),
        "tv_pair_identity_error_max": float(
            torch.maximum((r_amp - positive).abs().max(), (r_amp - negative).abs().max())
        ),
        "tv_out_of_range_count": int(((r_amp < -1e-9) | (r_amp > 1 + 1e-9)).sum()),
    }
    return metrics, audit


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _curve_rows(
    model: str, labels: np.ndarray, values: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    rows = []
    for metric, matrix in values.items():
        for label, label_name in LABELS:
            selected = matrix[labels == label]
            for layer in range(matrix.shape[1]):
                column = selected[:, layer]
                rows.append({
                    "model": model,
                    "metric": metric,
                    "label": label_name,
                    "layer": layer + 1,
                    "n_mentions": len(column),
                    "mean": float(column.mean()),
                    "q25": float(np.quantile(column, 0.25)),
                    "median": float(np.median(column)),
                    "q75": float(np.quantile(column, 0.75)),
                })
    return rows


def _strength_correlation_rows(
    model: str, labels: np.ndarray, s_write: np.ndarray, s_ffn: np.ndarray
) -> list[dict[str, Any]]:
    rows = []
    for label, name in ((None, "ALL"), *LABELS):
        selected = np.ones(len(labels), dtype=bool) if label is None else labels == label
        for layer in range(s_write.shape[1]):
            left, right = s_write[selected, layer], s_ffn[selected, layer]
            rows.append({
                "model": model,
                "label": name,
                "layer": layer + 1,
                "n_mentions": int(selected.sum()),
                "spearman": float(_row_correlation(
                    rankdata(left)[None, :], rankdata(right)[None, :]
                )[0]),
                "pearson_log": float(_row_correlation(
                    np.log(np.maximum(left, EPS))[None, :],
                    np.log(np.maximum(right, EPS))[None, :],
                )[0]),
            })
    return rows


def _summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    metrics = sorted({row["metric"] for row in rows})
    for metric in metrics:
        statistic = CURVES[metric][1]
        by_label = {
            label: [row for row in rows if row["metric"] == metric and row["label"] == label]
            for label in ("REAL", "HALL")
        }
        gaps = np.asarray([
            hall[statistic] - real[statistic]
            for real, hall in zip(by_label["REAL"], by_label["HALL"])
        ])
        result[metric] = {
            "summary_statistic": statistic,
            "mean_of_layer_statistic": {
                label: float(np.mean([row[statistic] for row in selected]))
                for label, selected in by_label.items()
            },
            "mean_layer_hall_minus_real": float(gaps.mean()),
            "hall_above_real_layers": int((gaps > 0).sum()),
            "real_above_hall_layers": int((gaps < 0).sum()),
            "layers": len(gaps),
        }
    return result


def _plot_curves(
    rows: list[dict[str, Any]], metrics: tuple[str, ...], path: Path, *, title: str
) -> None:
    figure, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4.2))
    axes = np.atleast_1d(axes)
    for axis, metric in zip(axes, metrics):
        ylabel, statistic, log_scale = CURVES[metric]
        for label, color in (("REAL", "#2166ac"), ("HALL", "#b2182b")):
            selected = [row for row in rows if row["metric"] == metric and row["label"] == label]
            x = np.asarray([row["layer"] for row in selected])
            y = np.asarray([row[statistic] for row in selected])
            axis.plot(x, y, color=color, linewidth=1.7, label=label)
            if statistic == "median":
                axis.fill_between(
                    x, [row["q25"] for row in selected], [row["q75"] for row in selected],
                    color=color, alpha=0.13,
                )
        if log_scale:
            axis.set_yscale("log")
        if metric in {"top1_agreement", "top32_overlap", "r_amp"}:
            axis.set_ylim(0, 1)
        if metric == "spearman":
            axis.set_ylim(-1, 1)
        axis.set_xlabel("Decoder layer")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    figure.suptitle(title)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        figure.savefig(path.with_suffix(f".{extension}"), dpi=180)
    plt.close(figure)


def _plot_cross_model_curve(
    rows: list[dict[str, Any]], metric: str, path: Path, *, title: str
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    ylabel, statistic, _log_scale = CURVES[metric]
    for axis, (model, model_name) in zip(axes.flat, MODELS.items()):
        for label, color in (("REAL", "#2166ac"), ("HALL", "#b2182b")):
            selected = [
                row for row in rows
                if row["model"] == model and row["metric"] == metric and row["label"] == label
            ]
            x = np.asarray([row["layer"] for row in selected])
            axis.plot(x, [row[statistic] for row in selected], color=color,
                      linewidth=1.8, label=label)
            if statistic == "median":
                axis.fill_between(x, [row["q25"] for row in selected],
                                  [row["q75"] for row in selected], color=color, alpha=0.13)
        axis.set_title(model_name)
        axis.set_xlabel("Decoder layer")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    figure.suptitle(title)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    for extension in ("png", "pdf"):
        figure.savefig(path.with_suffix(f".{extension}"), dpi=180)
    plt.close(figure)


def _plot_strength_relationship(
    model: str,
    labels: np.ndarray,
    s_write: np.ndarray,
    s_ffn: np.ndarray,
    correlations: list[dict[str, Any]],
    root: Path,
) -> str:
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    for axis, (label, name, cmap) in zip(
        axes[0], ((1, "REAL", "Blues"), (0, "HALL", "Reds"))
    ):
        left, right = s_write[labels == label].reshape(-1), s_ffn[labels == label].reshape(-1)
        keep = (left > 0) & (right > 0)
        image = axis.hexbin(
            left[keep], right[keep], gridsize=55, xscale="log", yscale="log",
            bins="log", mincnt=1, cmap=cmap,
        )
        axis.set_title(f"{name}: mention-layer pairs")
        axis.set_xlabel("S_WRITE")
        axis.set_ylabel("S_FFN")
        axis.grid(alpha=0.15)
        figure.colorbar(image, ax=axis, label="log10(count)")
    for axis, metric, title in zip(
        axes[1], ("spearman", "pearson_log"),
        ("Spearman(S_WRITE, S_FFN)", "Pearson(log S_WRITE, log S_FFN)"),
    ):
        for label, color in (("ALL", "#333333"), ("REAL", "#2166ac"), ("HALL", "#b2182b")):
            selected = [row for row in correlations if row["label"] == label]
            axis.plot([row["layer"] for row in selected], [row[metric] for row in selected],
                      color=color, linewidth=1.7, label=label)
        axis.set_ylim(-1, 1)
        axis.set_xlabel("Decoder layer")
        axis.set_ylabel(title)
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    figure.suptitle(f"{MODELS[model]}: WRITE–FFN gross-strength relationship")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    path = root / "figures/write_ffn_strength_relationship.png"
    for extension in ("png", "pdf"):
        figure.savefig(path.with_suffix(f".{extension}"), dpi=180)
    plt.close(figure)
    return str(path.relative_to(ROOT))


def _plot_fixed_maps(model: str, selected: dict[int, dict[str, Any]], root: Path) -> str:
    figure, axes = plt.subplots(2, 3, figsize=(10.5, 6.5))
    layer = FORMAL_LAYERS_BY_MODEL[model][2]
    for row_index, (label, name) in enumerate(((0, "HALL"), (1, "REAL"))):
        item = selected[label]
        height, width = item["visual_grid"]
        p_write = item["p_write"][layer - 1].reshape(height, width)
        p_ffn = item["p_ffn"][layer - 1].reshape(height, width)
        delta = p_ffn - p_write
        vmax = max(float(p_write.max()), float(p_ffn.max()))
        delta_limit = max(float(np.abs(delta).max()), EPS)
        for column, (values, panel, cmap, low, high) in enumerate((
            (p_write, "P_WRITE", "viridis", 0, vmax),
            (p_ffn, "P_FFN", "viridis", 0, vmax),
            (delta, "P_FFN - P_WRITE", "coolwarm", -delta_limit, delta_limit),
        )):
            image = axes[row_index, column].imshow(values, cmap=cmap, vmin=low, vmax=high)
            axes[row_index, column].set_title(f"{name} {panel}")
            axes[row_index, column].axis("off")
            figure.colorbar(image, ax=axes[row_index, column], fraction=0.046)
        axes[row_index, 0].set_ylabel(item["mention_id"])
    figure.suptitle(f"{MODELS[model]} fixed examples, layer {layer}")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    path = root / "figures/write_ffn_fixed_difference_maps.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)
    return str(path.relative_to(ROOT))


def _load_model(model: str) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any], dict[int, dict[str, Any]]]:
    paths = sorted((result_root(model) / "shards/full").glob("features*_shard_*.pt"))
    if not paths:
        raise FileNotFoundError(f"No formal v1 shards for {model}")
    values = {metric: [] for metric in CURVES}
    labels, mention_ids, target_keys, mention_targets, image_ids = [], set(), set(), set(), set()
    selected: dict[int, dict[str, Any]] = {}
    audit: dict[str, float | int] = {
        "normalization_error_max": 0.0,
        "signed_delta_sum_abs_max": 0.0,
        "tv_positive_mass_error_max": 0.0,
        "tv_negative_mass_error_max": 0.0,
        "tv_pair_identity_error_max": 0.0,
        "tv_out_of_range_count": 0,
        "zero_write_strength_count": 0,
    }
    layer_count = None
    for path in paths:
        shard = torch.load(path, map_location="cpu", weights_only=False)
        shard_images = set(map(int, shard["image_ids"]))
        if image_ids & shard_images:
            raise AssertionError("Duplicate images across formal shards")
        image_ids.update(shard_images)
        local = {}
        for position in shard["positions"]:
            target_key = str(position["target_key"])
            if target_key in target_keys:
                raise AssertionError(f"Duplicate formal target {target_key}")
            target_keys.add(target_key)
            metrics, checks = distribution_metrics(position["p_write"], position["p_ffn"])
            write_strength = torch.as_tensor(position["write_mag"]).double().sum(-1).numpy()
            ffn_strength = torch.as_tensor(position["gross_strength"]).double().numpy()
            if layer_count is None:
                layer_count = len(write_strength)
            if len(write_strength) != layer_count or len(ffn_strength) != layer_count:
                raise AssertionError("Inconsistent decoder layer count")
            zero_write = write_strength <= EPS
            gain = np.divide(
                ffn_strength, write_strength, out=np.zeros_like(ffn_strength), where=~zero_write
            )
            metrics.update(s_write=write_strength, s_ffn=ffn_strength, gross_gain=gain)
            if not all(np.isfinite(item).all() for item in metrics.values()):
                raise ValueError(f"Non-finite comparison metric for {target_key}")
            for key in audit:
                if key == "zero_write_strength_count":
                    audit[key] += int(zero_write.sum())
                elif key == "tv_out_of_range_count":
                    audit[key] += int(checks[key])
                else:
                    audit[key] = max(float(audit[key]), float(checks[key]))
            local[target_key] = (metrics, position)
        for mention in shard["sample_table"]:
            mention_id = str(mention["mention_id"])
            target_key = str(mention["target_key"])
            if mention_id in mention_ids or target_key not in local:
                raise AssertionError(f"Invalid formal mention {mention_id}")
            mention_ids.add(mention_id)
            mention_targets.add(target_key)
            label = int(mention["label"])
            labels.append(label)
            metrics, position = local[target_key]
            for metric in CURVES:
                values[metric].append(metrics[metric])
            order = (int(mention["image_id"]), int(mention["response_index"]), mention_id)
            if label not in selected or order < selected[label]["order"]:
                selected[label] = {
                    "order": order,
                    "mention_id": mention_id,
                    "target_key": target_key,
                    "visual_grid": tuple(map(int, position["visual_grid"])),
                    "p_write": torch.as_tensor(position["p_write"]).double().numpy(),
                    "p_ffn": torch.as_tensor(position["p_ffn"]).double().numpy(),
                }
        del shard, local
    if target_keys != mention_targets:
        raise AssertionError("Formal positions and mention targets do not match")
    splits = load_json(str(ROOT / "outputs" / model / EXPERIMENT / "image_splits.json"))
    expected_images = set(map(int, splits["train"])) | set(map(int, splits["test"]))
    if image_ids != expected_images:
        raise AssertionError("Formal shard images do not match the frozen split")
    if set(selected) != {0, 1}:
        raise AssertionError("Fixed examples require one REAL and one HALL mention")
    matrices = {key: np.stack(item) for key, item in values.items()}
    return matrices, np.asarray(labels, dtype=np.int8), {
        **audit,
        "processed_images": len(image_ids),
        "unique_targets": len(target_keys),
        "mentions": len(labels),
        "layers": int(layer_count or 0),
        "real_mentions": int(sum(labels)),
        "hall_mentions": int(len(labels) - sum(labels)),
        "shards": len(paths),
    }, selected


def main() -> None:
    started = time.monotonic()
    torch.set_num_threads(1)
    combined_rows = []
    results = {}
    for model in MODELS:
        model_started = time.monotonic()
        print(f"[{model}] reading formal v1 shards", flush=True)
        matrices, labels, audit, selected = _load_model(model)
        rows = _curve_rows(model, labels, matrices)
        correlations = _strength_correlation_rows(
            model, labels, matrices["s_write"], matrices["s_ffn"]
        )
        root = result_root(model)
        curve_path = root / "tables/write_ffn_comparison_real_hall_curve.csv"
        correlation_path = root / "tables/write_ffn_strength_correlation_by_layer.csv"
        _write_csv(curve_path, rows)
        _write_csv(correlation_path, correlations)
        _plot_curves(
            rows, ("spearman", "js", "top1_agreement", "top32_overlap", "r_amp"),
            root / "figures/write_ffn_distribution_comparison",
            title=f"{MODELS[model]}: P_WRITE versus P_FFN by target label",
        )
        _plot_curves(
            rows, ("s_write", "s_ffn", "gross_gain"),
            root / "figures/write_ffn_strength_comparison",
            title=f"{MODELS[model]}: WRITE and path-FFN gross strength by target label",
        )
        strength_relationship = _plot_strength_relationship(
            model, labels, matrices["s_write"], matrices["s_ffn"], correlations, root
        )
        fixed_path = _plot_fixed_maps(model, selected, root)
        summaries = _summaries(rows)
        result = {
            "model": model,
            "definitions": {
                "P_WRITE": "||a_m||_2 / sum_k ||a_k||_2",
                "P_FFN": "||e_m||_2 / sum_k ||e_k||_2",
                "S_WRITE": "sum_m ||a_m||_2 = sum_m saved write_mag_m",
                "S_FFN": "sum_m ||e_m||_2 = saved gross_strength",
                "gross_gain": "S_FFN / (S_WRITE + epsilon)",
                "delta_ff": "P_FFN - P_WRITE",
                "R_amp": "0.5 * sum_m |P_FFN_m - P_WRITE_m| = total variation",
            },
            "cohort": "all formal v1 COCO4000 train+test mentions; mention-weighted",
            "status": "DESCRIPTIVE_ONLY_NO_TRAINING_NO_BOOTSTRAP",
            "audit": audit,
            "summaries": summaries,
            "strength_correlations": {
                label: {
                    "mean_layer_spearman": float(np.mean([
                        row["spearman"] for row in correlations if row["label"] == label
                    ])),
                    "mean_layer_pearson_log": float(np.mean([
                        row["pearson_log"] for row in correlations if row["label"] == label
                    ])),
                }
                for label in ("ALL", "REAL", "HALL")
            },
            "fixed_examples": {
                name: {
                    "mention_id": selected[label]["mention_id"],
                    "target_key": selected[label]["target_key"],
                }
                for label, name in ((0, "HALL"), (1, "REAL"))
            },
            "artifacts": {
                "curve_csv": str(curve_path.relative_to(ROOT)),
                "strength_correlation_csv": str(correlation_path.relative_to(ROOT)),
                "strength_relationship": strength_relationship,
                "fixed_maps": fixed_path,
            },
            "elapsed_seconds": time.monotonic() - model_started,
        }
        atomic_json_save(result, root / "metrics/write_ffn_comparison_results.json")
        results[model] = result
        combined_rows.extend(rows)
        ramp = summaries["r_amp"]["mean_of_layer_statistic"]
        gain = summaries["gross_gain"]["mean_of_layer_statistic"]
        print(
            f"[{model}] R_amp REAL/HALL={ramp['REAL']:.6f}/{ramp['HALL']:.6f}; "
            f"gain={gain['REAL']:.6f}/{gain['HALL']:.6f}; "
            f"{result['elapsed_seconds']:.1f}s",
            flush=True,
        )
        del matrices, labels

    _plot_cross_model_curve(
        combined_rows, "r_amp", ROOT / "outputs/ffn_write_ffn_r_amp_real_hall",
        title="Path-FFN redistribution relative to WRITE: REAL / HALL",
    )
    _plot_cross_model_curve(
        combined_rows, "gross_gain", ROOT / "outputs/ffn_write_ffn_gross_gain_real_hall",
        title="Path-FFN gross gain over WRITE: REAL / HALL",
    )
    summary_path = ROOT / "outputs/ffn_write_ffn_comparison_summary.json"
    atomic_json_save(
        {
            "models": results,
            "elapsed_seconds": time.monotonic() - started,
            "note": "R_amp is total variation; sum(P_FFN-P_WRITE) is zero by normalization.",
        },
        summary_path,
    )
    print(f"Completed: {summary_path} ({time.monotonic() - started:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
