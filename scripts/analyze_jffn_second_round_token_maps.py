#!/usr/bin/env python3
"""Analyze WRITE/JFFN token maps, gain, signed-Q, and COCO localization.

This script consumes only the isolated second-round shards.  It intentionally
keeps the official InsLen mention cohort and reports paired WRITE-vs-JFFN
statistics instead of reusing the older cosine-source analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.jffn_experiment import fit_entropy_betas, load_shards  # noqa: E402
from scripts.run_jffn_p_comparison import (  # noqa: E402
    binary_average_precision,
    build_spatial_context,
    patch_overlap_fraction,
)
from utils.config_utils import load_config  # noqa: E402
from utils.io_utils import load_json  # noqa: E402


MODELS = ("llava_1_5_7b", "internvl_2_5_8b")
EXPERIMENT = "COCO4000-INSLEN-OFFICIAL-TARGET"
METHODS = ("ATTN", "WRITE", "JFFN", "JFFN_MATCH_WRITE", "Q_POSITIVE")
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument(
        "--config", default="configs/model_configs_inslen_official_target.yaml"
    )
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--shard-dir")
    parser.add_argument("--result-dir")
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--calibration-images", type=int, default=500)
    parser.add_argument("--calibration-seed", type=int, default=20260818)
    return parser.parse_args()


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class Running:
    def __init__(self) -> None:
        self.values: list[float] = []

    def add(self, value: float) -> None:
        if math.isfinite(float(value)):
            self.values.append(float(value))

    def result(self) -> dict[str, float | int]:
        values = np.asarray(self.values, dtype=np.float64)
        if not values.size:
            return {"count": 0, "mean": float("nan"), "std": float("nan")}
        return {
            "count": int(values.size),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "q10": float(np.quantile(values, 0.10)),
            "median": float(np.median(values)),
            "q90": float(np.quantile(values, 0.90)),
        }


def rows_from_running(acc: Mapping[tuple[Any, ...], Running], names: Sequence[str]):
    return [
        {**dict(zip(names, key)), **running.result()}
        for key, running in sorted(acc.items(), key=lambda item: item[0])
    ]


def normalized_entropy(p: torch.Tensor) -> torch.Tensor:
    safe = p.float().clamp_min(1e-30)
    return -(safe * safe.log()).sum(-1) / math.log(max(int(p.shape[-1]), 2))


def matched_distribution(energy: torch.Tensor, betas: Sequence[float]) -> torch.Tensor:
    beta = torch.as_tensor(betas, dtype=torch.float32).reshape(-1, 1)
    return torch.softmax(torch.log(energy.float().clamp_min(EPS)) / beta, dim=-1)


def q_positive_distribution(q: torch.Tensor) -> torch.Tensor:
    positive = q.float().clamp_min(0)
    total = positive.sum(-1, keepdim=True)
    uniform = torch.full_like(positive, 1.0 / max(int(positive.shape[-1]), 1))
    return torch.where(total > EPS, positive / total.clamp_min(EPS), uniform)


def distribution_map(position: Mapping[str, Any], betas: Sequence[float]):
    return {
        "ATTN": position["attention_distribution"].float(),
        "WRITE": position["write_distribution"].float(),
        "JFFN": position["jffn_distribution"].float(),
        "JFFN_MATCH_WRITE": matched_distribution(position["jffn_energy"], betas),
        "Q_POSITIVE": q_positive_distribution(position["signed_q"]),
    }


def pair_metrics(left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
    left = left.float().clamp_min(0)
    right = right.float().clamp_min(0)
    left = left / left.sum(-1, keepdim=True).clamp_min(EPS)
    right = right / right.sum(-1, keepdim=True).clamp_min(EPS)
    cosine = (left * right).sum(-1) / (
        left.norm(dim=-1) * right.norm(dim=-1)
    ).clamp_min(EPS)
    middle = 0.5 * (left + right)
    js = 0.5 * (
        (left * (left.clamp_min(EPS).log() - middle.clamp_min(EPS).log())).sum(-1)
        + (right * (right.clamp_min(EPS).log() - middle.clamp_min(EPS).log())).sum(-1)
    )
    tv = 0.5 * (left - right).abs().sum(-1)
    k = min(32, int(left.shape[-1]))
    li = torch.topk(left, k=k, dim=-1).indices
    ri = torch.topk(right, k=k, dim=-1).indices
    overlap = (li.unsqueeze(-1) == ri.unsqueeze(-2)).any(-1).sum(-1).float() / k
    return {
        "cosine": cosine,
        "js": js,
        "total_variation": tv,
        "top32_overlap": overlap,
        "top1_agreement": (left.argmax(-1) == right.argmax(-1)).float(),
    }


def batch_rank_metrics(write_e: torch.Tensor, jffn_e: torch.Tensor):
    x = write_e.float()
    y = jffn_e.float()
    xc = x - x.mean(-1, keepdim=True)
    yc = y - y.mean(-1, keepdim=True)
    pearson = (xc * yc).sum(-1) / (xc.norm(dim=-1) * yc.norm(dim=-1)).clamp_min(EPS)
    # Values are continuous; inverse argsort is an exact rank transform absent ties.
    xr = torch.argsort(torch.argsort(x, dim=-1), dim=-1).float()
    yr = torch.argsort(torch.argsort(y, dim=-1), dim=-1).float()
    xrc = xr - xr.mean(-1, keepdim=True)
    yrc = yr - yr.mean(-1, keepdim=True)
    spearman = (xrc * yrc).sum(-1) / (
        xrc.norm(dim=-1) * yrc.norm(dim=-1)
    ).clamp_min(EPS)
    output = {"pearson": pearson, "spearman": spearman}
    for k in (1, 5, 16, 32):
        kk = min(k, int(x.shape[-1]))
        xi = torch.topk(x, k=kk, dim=-1).indices
        yi = torch.topk(y, k=kk, dim=-1).indices
        output[f"top{k}_overlap"] = (
            (xi.unsqueeze(-1) == yi.unsqueeze(-2)).any(-1).sum(-1).float() / kk
        )
    return output


def patch_auroc(labels: torch.Tensor, scores: torch.Tensor) -> float:
    y = labels.numpy().astype(np.int64)
    if np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, scores.float().numpy()))


def spatial_metrics(p: torch.Tensor, overlap: torch.Tensor) -> dict[str, float]:
    labels = overlap > 0
    uniform_mass = float(overlap.mean())
    k = min(32, int(p.numel()))
    top = torch.topk(p, k=k).indices
    positive_count = max(int(labels.sum()), 1)
    return {
        "bbox_mass": float((p * overlap).sum()),
        "uniform_enrichment": float((p * overlap).sum()) / max(uniform_mass, EPS),
        "top1_pointing": float(bool(labels[int(torch.argmax(p))])),
        "patch_auroc": patch_auroc(labels, p),
        "patch_aupr": binary_average_precision(labels, p),
        "top32_box_recall": float(labels[top].sum()) / positive_count,
    }


def spatial_metrics_batch(
    distributions: torch.Tensor, overlap: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Vectorized spatial metrics for ``[method, layer, visual_token]``.

    The rank metrics use one stable descending sort for all methods/layers.
    This is the same deterministic tie policy as ``binary_average_precision``
    and avoids millions of tiny sklearn calls on the full cohort.
    """
    p = distributions.float()
    labels = (overlap > 0).to(dtype=torch.float32)
    positives = labels.sum().clamp_min(1.0)
    negatives = (1.0 - labels).sum().clamp_min(1.0)
    uniform_mass = overlap.float().mean().clamp_min(EPS)
    mass = (p * overlap.reshape(1, 1, -1)).sum(-1)
    top1 = p.argmax(-1)
    pointing = labels[top1]
    order = torch.argsort(p, dim=-1, descending=True, stable=True)
    expanded_labels = labels.reshape(1, 1, -1).expand_as(p)
    ranked = torch.gather(expanded_labels, -1, order)
    positions = torch.arange(
        1, int(p.shape[-1]) + 1, dtype=torch.float32
    ).reshape(1, 1, -1)
    precision = ranked.cumsum(-1) / positions
    aupr = (precision * ranked).sum(-1) / positives
    cumulative_negative = (1.0 - ranked).cumsum(-1)
    negative_after = negatives - cumulative_negative
    auroc = (negative_after * ranked).sum(-1) / (positives * negatives)
    k = min(32, int(p.shape[-1]))
    top = order[..., :k]
    top_labels = torch.gather(expanded_labels, -1, top)
    return {
        "bbox_mass": mass,
        "uniform_enrichment": mass / uniform_mass,
        "top1_pointing": pointing,
        "patch_auroc": auroc,
        "patch_aupr": aupr,
        "top32_box_recall": top_labels.sum(-1) / positives,
    }


def fit_betas(
    shard_dir: Path,
    train_ids: set[int],
    count: int,
    seed: int,
) -> tuple[list[float], list[dict[str, float]], list[int]]:
    available: set[int] = set()
    for shard in load_shards(shard_dir):
        available.update(int(value) for value in shard.get("image_ids", ()))
    candidates = sorted(available & train_ids)
    rng = random.Random(int(seed))
    rng.shuffle(candidates)
    selected = set(candidates[: min(count, len(candidates))])
    energies: list[list[torch.Tensor]] | None = None
    entropy_sum: np.ndarray | None = None
    entropy_count = 0
    for shard in load_shards(shard_dir):
        for position in shard["positions"]:
            if int(position["image_id"]) not in selected:
                continue
            energy = position["jffn_energy"].float()
            write = position["write_distribution"].float()
            if energies is None:
                energies = [[] for _ in range(int(energy.shape[0]))]
                entropy_sum = np.zeros(int(energy.shape[0]), dtype=np.float64)
            for layer, row in enumerate(energy):
                energies[layer].append(row)
            entropy_sum += normalized_entropy(write).numpy()
            entropy_count += 1
    if not energies or entropy_sum is None or entropy_count == 0:
        raise RuntimeError("No train-only entropy calibration rows")
    targets = (entropy_sum / entropy_count).tolist()
    betas, audit = fit_entropy_betas(
        energies_by_layer=energies,
        target_entropy_by_layer=targets,
    )
    return betas, audit, sorted(selected)


def paired_mean_bootstrap(
    by_image: Mapping[int, tuple[float, float, int]], replicates: int, seed: int
) -> dict[str, Any]:
    ids = np.asarray(sorted(by_image), dtype=np.int64)
    left_sum = np.asarray([by_image[int(i)][0] for i in ids], dtype=np.float64)
    right_sum = np.asarray([by_image[int(i)][1] for i in ids], dtype=np.float64)
    counts = np.asarray([by_image[int(i)][2] for i in ids], dtype=np.float64)
    observed = left_sum.sum() / counts.sum() - right_sum.sum() / counts.sum()
    rng = np.random.default_rng(int(seed))
    deltas = np.empty(int(replicates), dtype=np.float64)
    for index in range(int(replicates)):
        sample = rng.integers(0, len(ids), size=len(ids))
        denominator = counts[sample].sum()
        deltas[index] = left_sum[sample].sum() / denominator - right_sum[sample].sum() / denominator
    return {
        "difference": float(observed),
        "ci95_low": float(np.quantile(deltas, 0.025)),
        "ci95_high": float(np.quantile(deltas, 0.975)),
        "images": int(len(ids)),
        "replicates": int(replicates),
    }


def add_paired(
    store: dict[tuple[int, str], list[float]], image_id: int, metric: str, left: float, right: float
) -> None:
    key = (int(image_id), metric)
    if key not in store:
        store[key] = [0.0, 0.0, 0.0]
    store[key][0] += float(left)
    store[key][1] += float(right)
    store[key][2] += 1.0


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    model_root = Path(args.outputs_root) / args.model / EXPERIMENT
    result_dir = Path(args.result_dir or model_root / "results/jffn_second_round")
    shard_dir = Path(args.shard_dir or result_dir / "shards")
    if not shard_dir.exists():
        raise FileNotFoundError(shard_dir)
    splits = load_json(str(model_root / "image_splits.json"))
    train_ids = {int(value) for value in splits["train"]}
    betas, beta_audit, calibration_ids = fit_betas(
        shard_dir, train_ids, args.calibration_images, args.calibration_seed
    )
    config = load_config(args.config)
    spatial_context = build_spatial_context(config)

    distribution_acc: dict[tuple[str, str, int, str], Running] = defaultdict(Running)
    rank_acc: dict[tuple[int, str], Running] = defaultdict(Running)
    specificity_acc: dict[tuple[str, int, str], Running] = defaultdict(Running)
    spatial_acc: dict[tuple[str, int, str], Running] = defaultdict(Running)
    gain_acc: dict[tuple[str, int, str], Running] = defaultdict(Running)
    cancellation_acc: dict[tuple[str, int, str], Running] = defaultdict(Running)
    reconstruction_acc: dict[tuple[int, str], Running] = defaultdict(Running)
    paired_spatial: dict[tuple[int, str], list[float]] = {}
    correction_by_image: dict[int, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    audit = {
        "images": set(), "positions": 0, "mentions": 0,
        "real_mentions": 0, "real_with_boxes": 0, "real_without_boxes": 0,
        "grid_mismatches": 0,
    }

    for shard in load_shards(shard_dir):
        positions = {str(row["target_key"]): row for row in shard["positions"]}
        mentions = list(shard["sample_table"])
        audit["images"].update(int(value) for value in shard.get("image_ids", ()))
        audit["positions"] += len(positions)
        audit["mentions"] += len(mentions)

        maps_by_key: dict[str, dict[str, torch.Tensor]] = {}
        rank_by_key: dict[str, Mapping[str, torch.Tensor]] = {}
        for key, position in positions.items():
            maps = distribution_map(position, betas)
            maps_by_key[key] = maps
            write_e = position["write_energy"].float()
            jffn_e = position["jffn_energy"].float()
            rank = batch_rank_metrics(write_e, jffn_e)
            rank_by_key[key] = rank
            for metric, values in rank.items():
                for layer, value in enumerate(values.tolist(), 1):
                    rank_acc[(layer, metric)].add(value)

            input_median = write_e.median(dim=-1).values
            gain = jffn_e / write_e.clamp_min(EPS)
            q = position["signed_q"].float()
            positive = q.clamp_min(0).sum(-1)
            negative = (-q).clamp_min(0).sum(-1)
            diagnostics = position["diagnostics"]
            for layer in range(int(write_e.shape[0])):
                layer_gain = gain[layer]
                median = float(input_median[layer])
                robust = layer_gain[write_e[layer] >= max(median * 1e-3, EPS)]
                for name, value in (
                    ("gain_mean", layer_gain.mean()),
                    ("gain_cv", layer_gain.std(unbiased=False) / layer_gain.mean().clamp_min(EPS)),
                    ("gain_robust_cv", robust.std(unbiased=False) / robust.mean().clamp_min(EPS)),
                    ("input_min", write_e[layer].min()),
                    ("tiny_abs_1e-8", (write_e[layer] < 1e-8).float().mean()),
                    ("tiny_abs_1e-6", (write_e[layer] < 1e-6).float().mean()),
                    ("tiny_abs_1e-4", (write_e[layer] < 1e-4).float().mean()),
                    ("tiny_relative_median_1e-3", (write_e[layer] < median * 1e-3).float().mean()),
                ):
                    gain_acc[("all", layer + 1, name)].add(float(value))
                for name, value in (
                    ("positive_mass", positive[layer]),
                    ("negative_mass", negative[layer]),
                    ("negative_positive_ratio", negative[layer] / positive[layer].clamp_min(EPS)),
                    ("cancellation_ratio", diagnostics["cancellation_ratio"][layer]),
                    ("negative_fraction", (q[layer] < 0).float().mean()),
                    ("max_positive", q[layer].max()),
                    ("max_negative_magnitude", (-q[layer]).max()),
                    ("q_conservation_relative_error", diagnostics["q_conservation_relative_error"][layer]),
                ):
                    cancellation_acc[("all", layer + 1, name)].add(float(value))
                reconstruction_acc[(layer + 1, "component_sum_relative_error")].add(
                    float(diagnostics["component_sum_relative_error"][layer])
                )
                reconstruction_acc[(layer + 1, "attention_reconstruction_relative_error")].add(
                    float(diagnostics["attention_reconstruction_relative_error"][layer])
                )

        # Distribution summaries are mention-weighted, matching downstream cohort.
        for mention in mentions:
            label = "REAL" if int(mention["label"]) == 1 else "HALL"
            key = str(mention["target_key"])
            position = positions[key]
            maps = maps_by_key[key]
            write_e = position["write_energy"].float()
            jffn_e = position["jffn_energy"].float()
            q = position["signed_q"].float()
            positive = q.clamp_min(0).sum(-1)
            negative = (-q).clamp_min(0).sum(-1)
            diagnostics = position["diagnostics"]
            for method, p in maps.items():
                ent = normalized_entropy(p)
                top1 = p.max(-1).values
                top32 = torch.topk(p, k=min(32, int(p.shape[-1])), dim=-1).values.sum(-1)
                for layer in range(int(p.shape[0])):
                    distribution_acc[(method, label, layer + 1, "normalized_entropy")].add(float(ent[layer]))
                    distribution_acc[(method, label, layer + 1, "top1_mass")].add(float(top1[layer]))
                    distribution_acc[(method, label, layer + 1, "top32_mass")].add(float(top32[layer]))
            gain = jffn_e / write_e.clamp_min(EPS)
            for layer in range(int(gain.shape[0])):
                gain_acc[(label, layer + 1, "gain_mean")].add(float(gain[layer].mean()))
                gain_acc[(label, layer + 1, "gain_cv")].add(float(gain[layer].std(unbiased=False) / gain[layer].mean().clamp_min(EPS)))
                for name, value in (
                    ("input_norm", diagnostics["input_norm"][layer]),
                    ("response_norm", diagnostics["response_norm"][layer]),
                    ("sensitivity", diagnostics["gain"][layer]),
                    ("direction_cosine", diagnostics["direction_cosine"][layer]),
                ):
                    gain_acc[(label, layer + 1, name)].add(float(value))
                for name, value in (
                    ("positive_mass", positive[layer]),
                    ("negative_mass", negative[layer]),
                    ("negative_positive_ratio", negative[layer] / positive[layer].clamp_min(EPS)),
                    ("positive_minus_negative", positive[layer] - negative[layer]),
                    ("cancellation_ratio", diagnostics["cancellation_ratio"][layer]),
                    ("negative_fraction", (q[layer] < 0).float().mean()),
                    ("max_positive", q[layer].max()),
                    ("max_negative_magnitude", (-q[layer]).max()),
                ):
                    cancellation_acc[(label, layer + 1, name)].add(float(value))

            if label != "REAL":
                continue
            audit["real_mentions"] += 1
            image_id = int(mention["image_id"])
            category = str(mention.get("canonical_object") or "")
            boxes = spatial_context["boxes"].get((image_id, category), ())
            if not boxes:
                audit["real_without_boxes"] += 1
                continue
            grid = tuple(int(value) for value in position.get("visual_grid") or ())
            if len(grid) != 2 or grid[0] * grid[1] != int(write_e.shape[-1]):
                audit["grid_mismatches"] += 1
                continue
            audit["real_with_boxes"] += 1
            overlap = patch_overlap_fraction(
                model=args.model,
                image_size=spatial_context["image_size"][image_id],
                boxes=boxes,
                grid=grid,
            )
            labels = overlap > 0
            method_names = list(maps)
            method_tensor = torch.stack([maps[name] for name in method_names])
            batched_spatial = spatial_metrics_batch(method_tensor, overlap)
            for method_index, method in enumerate(method_names):
                for layer in range(int(method_tensor.shape[1])):
                    for metric, values in batched_spatial.items():
                        spatial_acc[(method, layer + 1, metric)].add(
                            float(values[method_index, layer])
                        )
            for layer in range(int(write_e.shape[0])):
                write_index = method_names.index("WRITE")
                jffn_index = method_names.index("JFFN")
                write_values = {
                    metric: float(values[write_index, layer])
                    for metric, values in batched_spatial.items()
                }
                jffn_values = {
                    metric: float(values[jffn_index, layer])
                    for metric, values in batched_spatial.items()
                }
                # The preregistered scalar/risk range is layers 16--32.  Use
                # the same fixed range for the primary paired spatial test;
                # per-layer descriptive results remain in the CSV above.
                if layer + 1 >= 16:
                    for metric in write_values:
                        add_paired(
                            paired_spatial,
                            image_id,
                            metric,
                            jffn_values[metric],
                            write_values[metric],
                        )
                wi = int(torch.argmax(write_e[layer]))
                ji = int(torch.argmax(jffn_e[layer]))
                if wi != ji:
                    write_inside = bool(labels[wi])
                    jffn_inside = bool(labels[ji])
                    if (not write_inside) and jffn_inside:
                        correction_by_image[image_id][0] += 1
                    elif write_inside and (not jffn_inside):
                        correction_by_image[image_id][1] += 1
                    else:
                        correction_by_image[image_id][2] += 1
                    correction_by_image[image_id][3] += 1
                gain_layer = (jffn_e[layer] / write_e[layer].clamp_min(EPS))
                gain_acc[("REAL_BOX", layer + 1, "gain_inside_mean")].add(float(gain_layer[labels].mean()))
                gain_acc[("REAL_BOX", layer + 1, "gain_outside_mean")].add(float(gain_layer[~labels].mean()))
                gain_acc[("REAL_BOX", layer + 1, "gain_top1_pointing")].add(float(bool(labels[int(gain_layer.argmax())])))
                negative_q = (-q[layer]).clamp_min(0)
                negative_total = negative_q.sum().clamp_min(EPS)
                cancellation_acc[("REAL_BOX", layer + 1, "negative_bbox_fraction")].add(float((negative_q * overlap).sum() / negative_total))

        # Same-image specificity uses every unique target pair exactly once.
        by_image: dict[int, list[str]] = defaultdict(list)
        for key, position in positions.items():
            by_image[int(position["image_id"])].append(key)
        for keys in by_image.values():
            for left_index in range(len(keys)):
                for right_index in range(left_index + 1, len(keys)):
                    left_key, right_key = keys[left_index], keys[right_index]
                    for method in ("WRITE", "JFFN", "JFFN_MATCH_WRITE"):
                        left = maps_by_key[left_key][method]
                        right = maps_by_key[right_key][method]
                        if left.shape != right.shape:
                            continue
                        metrics = pair_metrics(left, right)
                        for metric, values in metrics.items():
                            for layer, value in enumerate(values.tolist(), 1):
                                specificity_acc[(method, layer, metric)].add(value)

    # Paired spatial bootstrap is computed per layer/metric with image clusters.
    paired_rows = []
    metrics = sorted({key[1] for key in paired_spatial})
    for offset, metric in enumerate(metrics):
        per_image = {
            image_id: tuple(values)
            for (image_id, name), values in paired_spatial.items()
            if name == metric
        }
        result = paired_mean_bootstrap(
            per_image, args.bootstrap_replicates, 20260819 + offset
        )
        paired_rows.append(
            {"layers": "16-32", "metric": metric, **result}
        )

    corrections = int(sum(value[0] for value in correction_by_image.values()))
    regressions = int(sum(value[1] for value in correction_by_image.values()))
    neutral = int(sum(value[2] for value in correction_by_image.values()))
    changed = int(sum(value[3] for value in correction_by_image.values()))
    rng = np.random.default_rng(20260820)
    image_rows = np.asarray(list(correction_by_image.values()), dtype=np.float64)
    net_samples = []
    if image_rows.size:
        for _ in range(args.bootstrap_replicates):
            selected = rng.integers(0, len(image_rows), size=len(image_rows))
            row = image_rows[selected].sum(0)
            net_samples.append((row[0] - row[1]) / max(row[3], 1))

    audit["images"] = len(audit["images"])
    payload = {
        "protocol": "jffn_second_round_token_maps_v1",
        "model": args.model,
        "entropy_matching": {
            "target": "WRITE normalized entropy",
            "ranking_preserved": True,
            "train_only_calibration_images": len(calibration_ids),
            "test_images_used": 0,
            "betas": betas,
            "audit": beta_audit,
        },
        "cohort_audit": audit,
        "reranking": {
            "corrections": corrections,
            "regressions": regressions,
            "neutral_changes": neutral,
            "top1_changed_cases_with_boxes": changed,
            "net_correction": corrections - regressions,
            "net_rate": (corrections - regressions) / max(changed, 1),
            "net_rate_ci95": [
                float(np.quantile(net_samples, 0.025)) if net_samples else float("nan"),
                float(np.quantile(net_samples, 0.975)) if net_samples else float("nan"),
            ],
        },
        "max_q_conservation_relative_error": max(
            (
                value
                for (label, layer, metric), run in cancellation_acc.items()
                if label == "all" and metric == "q_conservation_relative_error"
                for value in run.values
            ),
            default=float("nan"),
        ),
        "max_component_sum_relative_error": max(
            (
                value
                for (layer, metric), run in reconstruction_acc.items()
                if metric == "component_sum_relative_error"
                for value in run.values
            ),
            default=float("nan"),
        ),
    }
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "token_map_analysis_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_csv(result_dir / "distribution_layer_metrics.csv", rows_from_running(distribution_acc, ("method", "label", "layer", "metric")))
    write_csv(result_dir / "ranking_layer_metrics.csv", rows_from_running(rank_acc, ("layer", "metric")))
    write_csv(result_dir / "same_image_target_specificity.csv", rows_from_running(specificity_acc, ("method", "layer", "metric")))
    write_csv(result_dir / "spatial_layer_metrics.csv", rows_from_running(spatial_acc, ("method", "layer", "metric")))
    write_csv(result_dir / "spatial_jffn_minus_write_bootstrap.csv", paired_rows)
    write_csv(result_dir / "token_gain_layer_metrics.csv", rows_from_running(gain_acc, ("label", "layer", "metric")))
    write_csv(result_dir / "cancellation_layer_metrics.csv", rows_from_running(cancellation_acc, ("label", "layer", "metric")))
    write_csv(result_dir / "decomposition_audit_metrics.csv", rows_from_running(reconstruction_acc, ("layer", "metric")))
    plot_outputs(result_dir, args.model)
    print(f"[token maps] wrote {result_dir}", flush=True)
    return payload


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_outputs(result_dir: Path, model: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spatial = read_csv(result_dir / "spatial_layer_metrics.csv")
    ranking = read_csv(result_dir / "ranking_layer_metrics.csv")
    gain = read_csv(result_dir / "token_gain_layer_metrics.csv")
    cancellation = read_csv(result_dir / "cancellation_layer_metrics.csv")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for axis, metric, title in zip(axes, ("top1_pointing", "patch_aupr", "bbox_mass"), ("Top-1 pointing", "Patch AUPRC", "BBox mass")):
        for method in ("ATTN", "WRITE", "JFFN", "JFFN_MATCH_WRITE", "Q_POSITIVE"):
            rows = [r for r in spatial if r["method"] == method and r["metric"] == metric]
            rows.sort(key=lambda row: int(row["layer"]))
            axis.plot([int(r["layer"]) for r in rows], [float(r["mean"]) for r in rows], label=method)
        axis.set_title(title); axis.set_xlabel("Layer"); axis.grid(alpha=.2)
    axes[0].set_ylabel("Metric")
    axes[-1].legend(fontsize=7)
    fig.suptitle(f"{model}: identical-cohort COCO localization")
    fig.tight_layout(); fig.savefig(result_dir / "figure1_spatial_localization.png", dpi=220); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for metric in ("pearson", "spearman"):
        rows = [r for r in ranking if r["metric"] == metric]
        rows.sort(key=lambda row: int(row["layer"]))
        axes[0].plot([int(r["layer"]) for r in rows], [float(r["mean"]) for r in rows], label=metric)
    for metric in ("top1_overlap", "top5_overlap", "top16_overlap", "top32_overlap"):
        rows = [r for r in ranking if r["metric"] == metric]
        rows.sort(key=lambda row: int(row["layer"]))
        axes[1].plot([int(r["layer"]) for r in rows], [float(r["mean"]) for r in rows], label=metric)
    axes[0].set_title("corr($I_j$, $E_j$)"); axes[1].set_title("WRITE/JFFN rank overlap")
    for axis in axes: axis.set_xlabel("Layer"); axis.grid(alpha=.2); axis.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(result_dir / "figure2_jacobian_reranking.png", dpi=220); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for axis, metric, title in zip(axes, ("input_norm", "response_norm", "sensitivity"), ("I", "R", "S=R/I")):
        for label in ("REAL", "HALL"):
            rows = [r for r in gain if r["label"] == label and r["metric"] == metric]
            rows.sort(key=lambda row: int(row["layer"]))
            x = np.asarray([int(r["layer"]) for r in rows])
            y = np.asarray([float(r["mean"]) for r in rows])
            ci = np.asarray([
                1.96 * float(r["std"]) / math.sqrt(max(int(r["count"]), 1))
                for r in rows
            ])
            axis.plot(x, y, label=label)
            axis.fill_between(x, y - ci, y + ci, alpha=.16)
        axis.set_title(title); axis.set_xlabel("Layer"); axis.grid(alpha=.2); axis.legend()
    fig.suptitle(f"{model}: aggregate visual response")
    fig.tight_layout(); fig.savefig(result_dir / "figure3_I_R_S_real_hall.png", dpi=220); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for axis, metric in zip(axes, ("negative_positive_ratio", "cancellation_ratio", "negative_fraction")):
        for label in ("REAL", "HALL"):
            rows = [r for r in cancellation if r["label"] == label and r["metric"] == metric]
            rows.sort(key=lambda row: int(row["layer"]))
            x = np.asarray([int(r["layer"]) for r in rows])
            y = np.asarray([float(r["mean"]) for r in rows])
            ci = np.asarray([
                1.96 * float(r["std"]) / math.sqrt(max(int(r["count"]), 1))
                for r in rows
            ])
            axis.plot(x, y, label=label)
            axis.fill_between(x, y - ci, y + ci, alpha=.16)
        axis.set_title(metric.replace("_", " ")); axis.set_xlabel("Layer"); axis.grid(alpha=.2); axis.legend()
    fig.suptitle(f"{model}: signed-Q cancellation")
    fig.tight_layout(); fig.savefig(result_dir / "figure4_cancellation_real_hall.png", dpi=220); plt.close(fig)


def main() -> None:
    analyze(parse_args())


if __name__ == "__main__":
    main()
