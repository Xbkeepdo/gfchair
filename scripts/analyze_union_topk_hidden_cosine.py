#!/usr/bin/env python3
"""Measure union-TopK visual/visual and causal-token/visual cosine geometry.

The diagnostic reuses saved captions and InsLen target positions, performs one
teacher-forced forward per sampled image, and reduces hidden states on device.
No hidden-state or pairwise-cost matrix is written to disk.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.coco_loader import load_coco_samples  # noqa: E402
from models import build_model  # noqa: E402
from models.base_wrapper import (  # noqa: E402
    AttentionRequirement,
    ExtractionRequirements,
)
from utils.config_utils import (  # noqa: E402
    get_dataset_cfg,
    get_dgst_t_cfg,
    get_extraction_model_cfg,
    load_config,
)
from utils.io_utils import load_json  # noqa: E402


HALL = 0
REAL = 1
LABEL_NAMES = {HALL: "hall", REAL: "real", "all": "all"}
MODELS = (
    "llava_1_5_7b",
    "internvl_2_5_8b",
    "qwen2_5_vl_7b",
    "qwen3_vl_8b",
)
SCALAR_FIELDS = (
    "support_size",
    "intersection_size",
    "total_variation",
    "constant_risk_sqrt",
    "constant_risk_cosine",
    "vv_mean",
    "vv_std",
    "vv_p10",
    "vv_p50",
    "vv_p90",
    "vv_centered_mean",
    "vv_centered_std",
    "vv_centered_p10",
    "vv_centered_p50",
    "vv_centered_p90",
    "qv_mean",
    "qv_std",
    "qv_p10",
    "qv_p50",
    "qv_p90",
    "qv_centered_mean",
    "qv_centered_std",
    "qv_centered_p10",
    "qv_centered_p50",
    "qv_centered_p90",
    "qv_softmax_normalized_entropy",
    "qv_softmax_max_over_uniform",
    "qv_top1_minus_median",
    "qv_top1_minus_bottom",
    "sqrt_cost_mean",
    "sqrt_cost_std",
    "sqrt_cost_p10",
    "sqrt_cost_p90",
    "cosine_cost_mean",
    "cosine_cost_std",
    "cosine_cost_p10",
    "cosine_cost_p90",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument(
        "--config", default="configs/model_configs_inslen_official_target.yaml"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-images", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--name", default="union_topk_hidden_cosine_500")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_images <= 0:
        raise ValueError("--num-images must be positive")
    os.environ["DGST_UNION_COSINE_DIAGNOSTICS"] = "1"
    os.environ["DGST_UNION_COSINE_DIAGNOSTICS_ONLY"] = "1"

    config = load_config(args.config)
    model_cfg = get_extraction_model_cfg(config, args.model)
    dataset_cfg = get_dataset_cfg(config)
    dgst_cfg = _diagnostic_dgst_config(get_dgst_t_cfg(config))
    prompt = str(
        (config.get("run") or {}).get("prompt")
        or model_cfg.get("prompt")
        or "Describe this image."
    )
    source_output = Path(args.output_dir).resolve()
    labels = {
        int(key): value
        for key, value in load_json(str(source_output / "labeling.json")).items()
    }
    generations = {
        int(key): value
        for key, value in load_json(str(source_output / "generations.json")).items()
    }
    images_dir = Path(dataset_cfg["coco_root"]) / "val2014"
    samples = load_coco_samples(
        images_dir=str(images_dir),
        instances_file=dataset_cfg["annotation_file"],
        captions_file=dataset_cfg["captions_file"],
        num_images=None,
        seed=int(args.seed),
    )
    samples = [sample for sample in samples if int(sample["image_id"]) in labels]
    selected, sampling = _select_stratified_samples(
        samples=samples,
        labels=labels,
        num_images=int(args.num_images),
        seed=int(args.seed),
    )

    result_dir = source_output / "results" / args.name
    result_dir.mkdir(parents=True, exist_ok=True)
    selection_path = result_dir / f"{args.model}_{args.name}_selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "model": args.model,
                "seed": int(args.seed),
                "num_images": len(selected),
                "sampling": sampling,
                "image_ids": [int(sample["image_id"]) for sample in selected],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )

    checkpoint_path = result_dir / f"{args.model}_{args.name}_checkpoint.pkl"
    if args.resume and checkpoint_path.exists():
        state = _load_checkpoint(checkpoint_path)
        completed = {int(value) for value in state["completed_image_ids"]}
        aggregates = state["aggregates"]
        audit = state["audit"]
        if int(state.get("schema_version", 1)) < 2:
            # V1 counted only unique causal positions as targets. Restore the
            # formal InsLen sample count from the duplicate-span audit.
            audit["processed_causal_positions"] = int(audit["processed_targets"])
            audit["processed_targets"] = int(audit["processed_targets"]) + int(
                audit.get("duplicate_spans_collapsed", 0)
            )
        audit.setdefault("processed_causal_positions", 0)
        audit.setdefault("resolved_forward_failures", [])
        print(f"[CosDiag] Resume with {len(completed)} completed images.")
    else:
        completed = set()
        aggregates: dict[tuple[str, str, int], dict[str, Any]] = {}
        audit = {
            "processed_images": 0,
            "processed_targets": 0,
            "processed_causal_positions": 0,
            "duplicate_spans_collapsed": 0,
            "target_label_conflicts": 0,
            "forward_failures": [],
            "resolved_forward_failures": [],
        }

    pending = [sample for sample in selected if int(sample["image_id"]) not in completed]
    if pending:
        print(f"[CosDiag] Loading {args.model} on {args.device} for {len(pending)} images.")
        wrapper = build_model(args.model, model_cfg, device=args.device)
        requirements = ExtractionRequirements(
            attention=AttentionRequirement.NONE,
            logits=False,
            token_hidden_states=False,
            patch_hidden_states=False,
            response_hidden_states=False,
            visual_layout=False,
            dgst_capture=True,
        )
        for pending_offset, sample in enumerate(pending, start=1):
            image_id = int(sample["image_id"])
            try:
                (
                    image_accumulator,
                    target_count,
                    causal_position_count,
                    duplicate_count,
                    conflict_count,
                ) = _process_image(
                    wrapper=wrapper,
                    sample=sample,
                    label_row=labels[image_id],
                    generation_row=generations[image_id],
                    dgst_cfg=dgst_cfg,
                    prompt=prompt,
                    requirements=requirements,
                )
                _commit_image_accumulator(aggregates, image_accumulator)
                audit["processed_images"] += 1
                audit["processed_targets"] += int(target_count)
                audit["processed_causal_positions"] += int(causal_position_count)
                audit["duplicate_spans_collapsed"] += int(duplicate_count)
                audit["target_label_conflicts"] += int(conflict_count)
                prior_failures = [
                    item
                    for item in audit["forward_failures"]
                    if int(item["image_id"]) == image_id
                ]
                if prior_failures:
                    audit["resolved_forward_failures"].extend(prior_failures)
                    audit["forward_failures"] = [
                        item
                        for item in audit["forward_failures"]
                        if int(item["image_id"]) != image_id
                    ]
                completed.add(image_id)
            except Exception as exc:
                audit["forward_failures"].append(
                    {"image_id": image_id, "error": repr(exc)}
                )
                _save_checkpoint(
                    checkpoint_path,
                    completed_image_ids=completed,
                    aggregates=aggregates,
                    audit=audit,
                )
                raise
            if (
                pending_offset % max(int(args.checkpoint_every), 1) == 0
                or pending_offset == len(pending)
            ):
                _save_checkpoint(
                    checkpoint_path,
                    completed_image_ids=completed,
                    aggregates=aggregates,
                    audit=audit,
                )
            print(
                f"[CosDiag] {args.model}: {len(completed)}/{len(selected)} images, "
                f"targets={audit['processed_targets']}, "
                f"causal_positions={audit['processed_causal_positions']}"
            )
            if torch.cuda.is_available() and pending_offset % 20 == 0:
                torch.cuda.empty_cache()

    rows = _summarize_aggregates(aggregates)
    csv_path = result_dir / f"{args.model}_{args.name}_summary.csv"
    _write_summary_csv(csv_path, rows)
    json_path = result_dir / f"{args.model}_{args.name}_summary.json"
    json_path.write_text(
        json.dumps(
            {
                "model": args.model,
                "num_selected_images": len(selected),
                "sampling": sampling,
                "audit": audit,
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    png_path = result_dir / f"{args.model}_{args.name}.png"
    pdf_path = result_dir / f"{args.model}_{args.name}.pdf"
    _plot_summary(rows, model=args.model, png_path=png_path, pdf_path=pdf_path)
    md_path = result_dir / f"{args.model}_{args.name}_report.md"
    md_path.write_text(_render_report(args.model, sampling, audit, rows) + "\n")
    print(f"[CosDiag] Wrote {md_path}")


def _diagnostic_dgst_config(config: dict[str, Any]) -> dict[str, Any]:
    result = dict(config)
    result.update(
        {
            "enabled": True,
            "feature_output_profile": "four_gate_vv",
            "target_gate_mode": "four_gate",
            "four_gate_methods": [
                "hpre_raw_logit_gauss",
                "hpre_softmax_prob_gauss",
            ],
            "source_modes": ["hmid_cos", "hpre_cos"],
            "support_modes": ["vv"],
            "transport_top_k": 64,
            "cost_mode": "sqrt_matched_state",
            "cost_modes": ["sqrt_matched_state", "cosine_matched_state"],
            "compute_capped_topmass_085": False,
            "capped_topmass_alphas": None,
            "source_tau_values": None,
            "transport_top_k_values": None,
            "compute_prompt_cafe": False,
            "compute_ffn_injection_features": False,
        }
    )
    return result


def _valid_spans(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        span
        for span in row.get("object_token_spans", [])
        if span.get("token_indices") and int(span.get("label", -1)) in (HALL, REAL)
    ]


def _select_stratified_samples(
    *,
    samples: list[dict[str, Any]],
    labels: dict[int, dict[str, Any]],
    num_images: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    eligible = []
    has_real = []
    hall_only = []
    for sample in samples:
        image_id = int(sample["image_id"])
        spans = _valid_spans(labels[image_id])
        if not spans:
            continue
        eligible.append(sample)
        values = [int(span["label"]) for span in spans]
        (has_real if REAL in values else hall_only).append(sample)
    if num_images > len(eligible):
        raise ValueError(
            f"Requested {num_images} images but only {len(eligible)} are eligible."
        )
    rng = random.Random(int(seed))
    real_quota = min(num_images // 2, len(has_real))
    hall_quota = min(num_images - real_quota, len(hall_only))
    selected = rng.sample(has_real, real_quota) + rng.sample(hall_only, hall_quota)
    if len(selected) < num_images:
        used = {int(sample["image_id"]) for sample in selected}
        remaining = [
            sample for sample in eligible if int(sample["image_id"]) not in used
        ]
        selected.extend(rng.sample(remaining, num_images - len(selected)))
    rng.shuffle(selected)
    return selected, {
        "eligible_images": len(eligible),
        "has_real_pool": len(has_real),
        "hall_only_pool": len(hall_only),
        "selected_has_real": sum(
            any(int(span["label"]) == REAL for span in _valid_spans(labels[int(s["image_id"])]))
            for s in selected
        ),
        "selected_hall_only": sum(
            all(int(span["label"]) == HALL for span in _valid_spans(labels[int(s["image_id"])]))
            for s in selected
        ),
    }


def _process_image(
    *,
    wrapper: Any,
    sample: dict[str, Any],
    label_row: dict[str, Any],
    generation_row: dict[str, Any],
    dgst_cfg: dict[str, Any],
    prompt: str,
    requirements: ExtractionRequirements,
) -> tuple[dict[tuple[str, str, int], dict[str, Any]], int, int, int, int]:
    image_id = int(sample["image_id"])
    generated_text = str(generation_row.get("generated_text", ""))
    if generated_text != str(label_row.get("generated_text", "")):
        raise ValueError(f"Image {image_id}: generation/label caption mismatch")
    response_ids = [int(value) for value in generation_row.get("response_token_ids", [])]
    if not response_ids:
        raise ValueError(f"Image {image_id}: missing response_token_ids")
    by_index: dict[int, list[dict[str, Any]]] = {}
    duplicate_count = 0
    conflict_count = 0
    for span in _valid_spans(label_row):
        response_index = int(span["token_indices"][0])
        if response_index < 0 or response_index >= len(response_ids):
            raise ValueError(f"Image {image_id}: invalid response index {response_index}")
        target_id = int(response_ids[response_index])
        declared_id = span.get("target_token_id")
        if declared_id is not None and int(declared_id) != target_id:
            raise ValueError(f"Image {image_id}: target ID mismatch at {response_index}")
        if response_index in by_index:
            duplicate_count += 1
            if any(
                int(existing["label"]) != int(span["label"])
                for existing in by_index[response_index]
            ):
                conflict_count += 1
            by_index[response_index].append(span)
        else:
            by_index[response_index] = [span]
    response_indices = sorted(by_index)
    target_ids = [int(response_ids[index]) for index in response_indices]
    image = Image.open(sample["image_path"]).convert("RGB")
    outputs = wrapper.extract_token_features_batch(
        image=image,
        response_token_ids=response_ids,
        response_token_indices=response_indices,
        target_token_ids=target_ids,
        cfg_dgst_t=dgst_cfg,
        prompt=prompt,
        requirements=requirements,
    )
    if len(outputs) != len(response_indices):
        raise RuntimeError(
            f"Image {image_id}: received {len(outputs)} outputs for "
            f"{len(response_indices)} targets"
        )
    image_values: dict[tuple[str, str, int], dict[str, Any]] = {}
    for response_index, output in zip(response_indices, outputs):
        result = output.dgst_t_result or {}
        diagnostic = result.get("dgst_t_union_cosine_diagnostics")
        if not isinstance(diagnostic, dict):
            raise RuntimeError(f"Image {image_id}: missing union cosine diagnostics")
        labels_at_position = sorted(
            {int(span["label"]) for span in by_index[response_index]}
        )
        # The primary statistical unit is one causal position. Repeated
        # surface objects resolving to the same first-token position must not
        # overweight the all-target distribution. If InsLen assigns different
        # CHAIR labels to that same position, include it once in each relevant
        # label-specific group while still including it only once in "all".
        label_keys = ["all", *(LABEL_NAMES[label] for label in labels_at_position)]
        for branch, payload in diagnostic["by_branch"].items():
            layer_count = int(payload["vv_mean"].numel())
            for layer in range(layer_count):
                for label_key in label_keys:
                    key = (str(branch), str(label_key), int(layer))
                    block = image_values.setdefault(key, _empty_image_block())
                    block["target_count"] += 1
                    for field in SCALAR_FIELDS:
                        block["field_sum"][field] += float(
                            payload[field][layer].item()
                        )
                    for prefix in ("vv", "qv"):
                        count = float(payload[f"{prefix}_count"][layer].item())
                        block[f"{prefix}_count"] += count
                        block[f"{prefix}_sum"] += float(
                            payload[f"{prefix}_sum"][layer].item()
                        )
                        block[f"{prefix}_sum_sq"] += float(
                            payload[f"{prefix}_sum_sq"][layer].item()
                        )
                        hist = payload[f"{prefix}_hist"][layer].numpy().astype(
                            np.float64, copy=False
                        )
                        if block[f"{prefix}_hist"] is None:
                            block[f"{prefix}_hist"] = hist.copy()
                        else:
                            block[f"{prefix}_hist"] += hist
    del outputs
    formal_target_count = sum(len(spans) for spans in by_index.values())
    return (
        image_values,
        formal_target_count,
        len(response_indices),
        duplicate_count,
        conflict_count,
    )


def _empty_image_block() -> dict[str, Any]:
    return {
        "target_count": 0,
        "field_sum": {field: 0.0 for field in SCALAR_FIELDS},
        "vv_count": 0.0,
        "vv_sum": 0.0,
        "vv_sum_sq": 0.0,
        "qv_count": 0.0,
        "qv_sum": 0.0,
        "qv_sum_sq": 0.0,
        "vv_hist": None,
        "qv_hist": None,
    }


def _empty_global_block(hist_bins: int) -> dict[str, Any]:
    return {
        "image_count": 0,
        "target_count": 0,
        "field_sum": {field: 0.0 for field in SCALAR_FIELDS},
        "field_sum_sq": {field: 0.0 for field in SCALAR_FIELDS},
        "vv_count": 0.0,
        "vv_sum": 0.0,
        "vv_sum_sq": 0.0,
        "qv_count": 0.0,
        "qv_sum": 0.0,
        "qv_sum_sq": 0.0,
        "vv_hist": np.zeros(hist_bins, dtype=np.float64),
        "qv_hist": np.zeros(hist_bins, dtype=np.float64),
    }


def _commit_image_accumulator(
    aggregates: dict[tuple[str, str, int], dict[str, Any]],
    image_values: dict[tuple[str, str, int], dict[str, Any]],
) -> None:
    for key, image_block in image_values.items():
        hist_bins = int(image_block["vv_hist"].size)
        block = aggregates.setdefault(key, _empty_global_block(hist_bins))
        n_targets = int(image_block["target_count"])
        block["image_count"] += 1
        block["target_count"] += n_targets
        for field in SCALAR_FIELDS:
            value = float(image_block["field_sum"][field]) / max(n_targets, 1)
            block["field_sum"][field] += value
            block["field_sum_sq"][field] += value * value
        for prefix in ("vv", "qv"):
            for suffix in ("count", "sum", "sum_sq"):
                block[f"{prefix}_{suffix}"] += float(
                    image_block[f"{prefix}_{suffix}"]
                )
            block[f"{prefix}_hist"] += image_block[f"{prefix}_hist"]


def _summarize_aggregates(
    aggregates: dict[tuple[str, str, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for (branch, label, layer), block in sorted(aggregates.items()):
        n_images = int(block["image_count"])
        row: dict[str, Any] = {
            "branch": branch,
            "source_mode": branch.split("__", 1)[0],
            "method": branch.split("__", 1)[1],
            "label": label,
            "layer": int(layer) + 1,
            "n_images": n_images,
            "n_targets": int(block["target_count"]),
        }
        for field in SCALAR_FIELDS:
            mean = block["field_sum"][field] / max(n_images, 1)
            variance = max(
                block["field_sum_sq"][field] / max(n_images, 1) - mean * mean,
                0.0,
            )
            row[field] = mean
            row[f"{field}_image_sd"] = math.sqrt(variance)
            row[f"{field}_image_ci95"] = (
                1.96 * math.sqrt(variance) / math.sqrt(n_images)
                if n_images > 1
                else 0.0
            )
        for prefix in ("vv", "qv"):
            count = float(block[f"{prefix}_count"])
            mean = float(block[f"{prefix}_sum"]) / max(count, 1.0)
            variance = max(
                float(block[f"{prefix}_sum_sq"]) / max(count, 1.0) - mean * mean,
                0.0,
            )
            row[f"{prefix}_pooled_count"] = int(count)
            row[f"{prefix}_pooled_mean"] = mean
            row[f"{prefix}_pooled_std"] = math.sqrt(variance)
            quantiles = _hist_quantiles(
                block[f"{prefix}_hist"], np.asarray([0.1, 0.5, 0.9])
            )
            row[f"{prefix}_pooled_p10"] = float(quantiles[0])
            row[f"{prefix}_pooled_p50"] = float(quantiles[1])
            row[f"{prefix}_pooled_p90"] = float(quantiles[2])
        rows.append(row)
    return rows


def _hist_quantiles(counts: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    total = float(counts.sum())
    if total <= 0:
        return np.zeros_like(quantiles, dtype=np.float64)
    cumulative = np.cumsum(counts)
    edges = np.linspace(-1.0, 1.0, counts.size + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    indices = np.searchsorted(cumulative, quantiles * total, side="left")
    return centers[np.clip(indices, 0, centers.size - 1)]


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No diagnostic summary rows were produced")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_summary(
    rows: list[dict[str, Any]],
    *,
    model: str,
    png_path: Path,
    pdf_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_rows = [row for row in rows if row["label"] == "all"]
    branches = sorted({row["branch"] for row in all_rows})
    colors = ["#0072b2", "#d55e00", "#009e73", "#cc79a7"]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    panels = (
        ("vv_pooled_mean", "visual–visual pooled cosine"),
        ("vv_pooled_p90", "visual–visual pooled P90"),
        ("vv_centered_mean", "visual–visual centered cosine"),
        ("qv_pooled_p90", "$q_t$–visual pooled P90"),
        ("qv_softmax_normalized_entropy", "$q_t$–visual softmax normalized entropy"),
        ("qv_top1_minus_bottom", "$q_t$–visual top1 − bottom1"),
    )
    for branch, color in zip(branches, colors):
        branch_rows = sorted(
            (row for row in all_rows if row["branch"] == branch),
            key=lambda row: int(row["layer"]),
        )
        layers = [int(row["layer"]) for row in branch_rows]
        for axis, (field, title) in zip(axes.flat, panels):
            axis.plot(
                layers,
                [float(row[field]) for row in branch_rows],
                label=branch.replace("_cos__hpre_", " / ").replace("_gauss", ""),
                color=color,
                linewidth=1.8,
            )
            axis.set_title(title)
            axis.set_xlabel("decoder layer")
            axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"{model}: union Top-32(P) ∪ Top-32(Q) cosine geometry")
    fig.savefig(png_path, dpi=180)
    fig.savefig(pdf_path)
    plt.close(fig)


def _render_report(
    model: str,
    sampling: dict[str, int],
    audit: dict[str, Any],
    rows: list[dict[str, Any]],
) -> str:
    all_rows = [row for row in rows if row["label"] == "all"]
    branches = sorted({row["branch"] for row in all_rows})
    lines = [
        f"# {model}：Union Top-K hidden cosine 诊断",
        "",
        "支持集严格使用当前协议：source Top-32 与 target Top-32 的并集（最多 64 个视觉 token）。",
        "对角线和对称重复项不进入 visual–visual 统计；`q_t` 是 InsLen 目标首 subtoken 的因果预测行。",
        "",
        "## 覆盖",
        "",
        f"- 图片：{audit['processed_images']}；正式 InsLen 目标：{audit['processed_targets']}；唯一因果位置：{audit['processed_causal_positions']}。",
        f"- 分层抽样：has-real={sampling['selected_has_real']}，hall-only={sampling['selected_hall_only']}。",
        f"- 合并同一因果位置的重复 span：{audit['duplicate_spans_collapsed']}。",
        "",
        "## 全层平均",
        "",
        "`vv/qv cosine` 与 P90−P10 来自全部有效 token 对的 pooled 分布；"
        "`centered vv`、entropy 和 top−bottom 先按图片聚合，再对图片等权平均。",
        "",
        "| P source / Q gate | vv cosine | vv P90−P10 | centered vv | qv cosine | qv P90−P10 | qv entropy | qv top−bottom |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for branch in branches:
        items = [row for row in all_rows if row["branch"] == branch]
        mean = lambda field: float(np.mean([float(row[field]) for row in items]))
        lines.append(
            f"| {branch} | {mean('vv_pooled_mean'):.4f} | "
            f"{mean('vv_pooled_p90') - mean('vv_pooled_p10'):.4f} | "
            f"{mean('vv_centered_mean'):.4f} | {mean('qv_pooled_mean'):.4f} | "
            f"{mean('qv_pooled_p90') - mean('qv_pooled_p10'):.4f} | "
            f"{mean('qv_softmax_normalized_entropy'):.4f} | "
            f"{mean('qv_top1_minus_bottom'):.4f} |"
        )
    lines.extend(
        [
            "",
            "`centered vv` 仅用于判断公共各向异性方向，不参与当前 cost。完整逐层、Real/Hall 分组及置信区间见同目录 CSV。",
        ]
    )
    return "\n".join(lines)


def _save_checkpoint(
    path: Path,
    *,
    completed_image_ids: set[int],
    aggregates: dict[tuple[str, str, int], dict[str, Any]],
    audit: dict[str, Any],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(
            {
                "schema_version": 2,
                "completed_image_ids": sorted(completed_image_ids),
                "aggregates": aggregates,
                "audit": audit,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    temporary.replace(path)


def _load_checkpoint(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return pickle.load(handle)


if __name__ == "__main__":
    main()
