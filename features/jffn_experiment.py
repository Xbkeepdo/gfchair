"""Utilities for the sharded JFFN-P comparison experiment."""

from __future__ import annotations

import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


SOURCE_FIELD_BY_NAME = {
    "old_hmid_cos": "dgst_t_source_dist_per_layer",
    "old_hpre_cos": "dgst_t_source_hpre_cos_dist_per_layer",
    "new_jffn": "dgst_t_source_jffn_dist_per_layer",
    "new_jffn_entropy_matched": (
        "dgst_t_source_jffn_entropy_matched_dist_per_layer"
    ),
}
SOURCE_MODE_BY_NAME = {
    "old_hmid_cos": "hmid_cos",
    "old_hpre_cos": "hpre_cos",
    "new_jffn": "jffn",
    "new_jffn_entropy_matched": "jffn_entropy_matched",
}
GATES = ("hpre_raw_logit_gauss", "hpre_softmax_prob_gauss")
COSTS = ("sqrt_matched_state", "cosine_matched_state")
SOURCES = tuple(SOURCE_FIELD_BY_NAME)


def build_jffn_dgst_config(
    base: Mapping[str, Any],
    *,
    entropy_beta_by_layer: Sequence[float] | None,
    calibration_only: bool = False,
    smoke_validation: bool = False,
    force_full_parallel: bool = False,
) -> dict[str, Any]:
    """Return an isolated config override; the caller's config is untouched."""
    cfg = dict(base)
    cfg.update(
        {
            "enabled": True,
            "feature_output_profile": "four_gate_vv",
            "target_gate_mode": "four_gate",
            "four_gate_methods": list(GATES),
            "cost_mode": "sqrt_matched_state",
            "cost_modes": list(COSTS),
            "support_modes": ["vv"],
            "source_modes": [
                "hmid_cos",
                "hpre_cos",
                "jffn",
                *(
                    ["jffn_entropy_matched"]
                    if entropy_beta_by_layer is not None
                    else []
                ),
            ],
            "source_tau_values": None,
            "transport_top_k_values": None,
            "compute_capped_topmass_085": False,
            "compute_ffn_injection_features": False,
            "compute_prompt_cafe": False,
            "transport_top_k": 64,
            "atarget_visual_top_k": 32,
            "jffn_entropy_beta_by_layer": (
                [float(value) for value in entropy_beta_by_layer]
                if entropy_beta_by_layer is not None
                else None
            ),
            "jffn_validate_linearity": bool(smoke_validation),
            "jffn_finite_difference_eta": 0.05 if smoke_validation else None,
            "jffn_parity_chunk_size": 64 if smoke_validation else None,
            "jffn_force_full_parallel": bool(force_full_parallel),
        }
    )
    if calibration_only:
        cfg["source_modes"] = ["hmid_cos", "hpre_cos", "jffn"]
        cfg["cost_modes"] = ["sqrt_matched_state"]
        cfg["four_gate_methods"] = ["hpre_raw_logit_gauss"]
    return cfg


def mentions_for_image(
    *,
    image_id: int,
    labeling_row: Mapping[str, Any],
    response_token_ids: Sequence[int],
) -> tuple[list[dict[str, Any]], list[int], list[int]]:
    """Build mention rows and a deduplicated causal-position forward request."""
    mentions: list[dict[str, Any]] = []
    response_length = len(response_token_ids)
    for mention_index, span in enumerate(labeling_row.get("object_token_spans") or []):
        raw_indices = span.get("token_indices") or []
        if not raw_indices:
            continue
        response_index = int(raw_indices[0])
        if response_index < 0 or response_index >= response_length:
            raise ValueError(
                f"Image {image_id}: response index {response_index} outside "
                f"length {response_length}."
            )
        target_id = int(response_token_ids[response_index])
        declared = span.get("target_token_id")
        if declared is not None and int(declared) != target_id:
            raise ValueError(
                f"Image {image_id}: target ID mismatch at {response_index}."
            )
        target_key = f"{int(image_id)}:{response_index}"
        mentions.append(
            {
                "mention_id": f"{int(image_id)}:{mention_index}",
                "target_key": target_key,
                "image_id": int(image_id),
                "response_index": response_index,
                "target_token_id": target_id,
                "label": int(span["label"]),
                "word": str(span.get("word") or ""),
                "canonical_object": str(span.get("canonical_object") or ""),
                "official_detected_word": str(
                    span.get("official_detected_word") or span.get("word") or ""
                ),
            }
        )
    unique_indices = sorted({int(row["response_index"]) for row in mentions})
    target_ids = [int(response_token_ids[index]) for index in unique_indices]
    return mentions, unique_indices, target_ids


def _cpu_float(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return torch.as_tensor(value, dtype=torch.float32).contiguous()


def _source_prefix(source: str) -> str:
    mode = SOURCE_MODE_BY_NAME[source]
    return "" if mode == "hmid_cos" else f"source_{mode}_"


def _risk_field(gate: str, source: str, cost: str) -> str:
    state = "hpre"
    cost_suffix = "sqrt" if cost == "sqrt_matched_state" else "cosine"
    return f"dgst_t_{gate}_{_source_prefix(source)}risk_{cost_suffix}_{state}_per_layer"


def compact_position_result(
    *,
    image_id: int,
    response_index: int,
    target_token_id: int,
    visual_grid: Sequence[int] | None,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Reduce one wrapper result to the experiment's stable shard schema."""
    sources = {
        source: _cpu_float(result[field])
        for source, field in SOURCE_FIELD_BY_NAME.items()
    }
    risks = {
        source: {
            gate: {
                cost: _cpu_float(result[_risk_field(gate, source, cost)])
                for cost in COSTS
            }
            for gate in GATES
        }
        for source in SOURCES
    }
    ev = {
        gate: _cpu_float(
            result[
                f"dgst_t_{gate}_ev_target_dist_mass_x_cosine_"
                "topk32_hpre_per_layer"
            ]
        )
        for gate in GATES
    }
    diagnostic_fields = (
        "input_norm",
        "response_norm",
        "gain",
        "direction_cosine",
        "response_sum_relative_error",
        "local_fd_cosine",
        "local_fd_relative_error",
        "attention_reconstruction_relative_error",
        "attention_reconstruction_cosine",
        "chunk_size",
        "parallel_chunk_relative_error",
    )
    diagnostics = {
        field: _cpu_float(result[f"dgst_t_jffn_{field}_per_layer"])
        for field in diagnostic_fields
    }
    source_statistics: dict[str, dict[str, torch.Tensor]] = {}
    for source in ("new_jffn", "new_jffn_entropy_matched"):
        mode = SOURCE_MODE_BY_NAME[source]
        source_statistics[source] = {
            statistic: _cpu_float(
                result[f"dgst_t_source_{mode}_{statistic}_per_layer"]
            )
            for statistic in (
                "normalized_entropy",
                "effective_token_count",
                "max_over_uniform",
                "gini",
                "top1_mass",
                "top5_mass",
                "top10_mass",
                "top32_mass",
            )
        }
    return {
        "target_key": f"{int(image_id)}:{int(response_index)}",
        "image_id": int(image_id),
        "response_index": int(response_index),
        "target_token_id": int(target_token_id),
        "visual_grid": [int(value) for value in visual_grid or ()],
        "sources": sources,
        "jffn_energy": _cpu_float(result["dgst_t_jffn_energy_per_layer"]),
        "jffn_diagnostics": diagnostics,
        "jffn_validation_computed": bool(
            result.get("dgst_t_jffn_validation_computed", False)
        ),
        "source_statistics": source_statistics,
        "risks": risks,
        "ev": ev,
    }


def renormalize_rows(values: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    finite = torch.nan_to_num(values.float(), nan=0.0, posinf=0.0, neginf=0.0)
    finite = finite.clamp_min(0.0)
    total = finite.sum(dim=-1, keepdim=True)
    uniform = torch.full_like(finite, 1.0 / max(int(finite.shape[-1]), 1))
    return torch.where(total > eps, finite / total.clamp_min(eps), uniform)


def atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_json_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def assert_finite_position_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    """Reject a shard before commit if any persisted floating tensor is invalid."""

    def visit(value: Any, path: str) -> None:
        if torch.is_tensor(value):
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise ValueError(f"Non-finite tensor in JFFN shard at {path}")
            return
        if isinstance(value, Mapping):
            for key, item in value.items():
                visit(item, f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")

    for index, row in enumerate(rows):
        visit(row, f"positions[{index}]")


def load_shards(shard_dir: Path) -> Iterable[dict[str, Any]]:
    # Single-GPU shards use ``features_shard_*``.  Two-GPU workers use
    # ``features_rankXX_shard_*`` so that concurrent atomic renames never
    # target the same pathname.  This pattern intentionally reads both.
    for path in sorted(shard_dir.glob("features*_shard_*.pt")):
        yield torch.load(path, map_location="cpu", weights_only=False)


def completed_image_ids(shard_dir: Path) -> set[int]:
    completed: set[int] = set()
    for shard in load_shards(shard_dir):
        completed.update(int(value) for value in shard.get("image_ids", ()))
    return completed


def select_train_calibration_images(
    *,
    train_image_ids: Sequence[int],
    available_image_ids: set[int],
    count: int,
    seed: int,
) -> list[int]:
    candidates = sorted(
        int(value) for value in train_image_ids if int(value) in available_image_ids
    )
    generator = random.Random(int(seed))
    generator.shuffle(candidates)
    return candidates[: min(int(count), len(candidates))]


def normalized_entropy(distribution: torch.Tensor) -> torch.Tensor:
    values = renormalize_rows(distribution)
    count = int(values.shape[-1])
    safe = values.clamp_min(1e-30)
    return -(safe * safe.log()).sum(dim=-1) / math.log(max(count, 2))


def fit_entropy_betas(
    *,
    energies_by_layer: Sequence[Sequence[torch.Tensor]],
    target_entropy_by_layer: Sequence[float],
    beta_min: float = 0.02,
    beta_max: float = 50.0,
    steps: int = 60,
) -> tuple[list[float], list[dict[str, float]]]:
    """Fit beta per layer by monotone bisection over mean normalized entropy."""
    if len(energies_by_layer) != len(target_entropy_by_layer):
        raise ValueError("Entropy calibration layer counts differ")
    betas: list[float] = []
    audit: list[dict[str, float]] = []
    for layer_index, (energy_rows, target_entropy) in enumerate(
        zip(energies_by_layer, target_entropy_by_layer)
    ):
        if not energy_rows:
            raise ValueError(f"No JFFN calibration rows for layer {layer_index}")

        # Qwen has dynamic visual-token counts, while LLaVA/InternVL normally
        # have one fixed width.  Grouping by M turns thousands of tiny softmax
        # calls per bisection step into one batched call per distinct width.
        grouped_rows: dict[int, list[torch.Tensor]] = defaultdict(list)
        for row in energy_rows:
            tensor = torch.as_tensor(row, dtype=torch.float32).reshape(
                -1, int(row.shape[-1])
            )
            grouped_rows[int(tensor.shape[-1])].append(tensor)
        grouped_energy = [
            torch.cat(rows, dim=0) for _width, rows in sorted(grouped_rows.items())
        ]

        def mean_entropy(beta: float) -> float:
            total = 0.0
            count = 0
            for energy in grouped_energy:
                matched = torch.softmax(
                    torch.log(energy.float().clamp_min(1e-12)) / float(beta),
                    dim=-1,
                )
                entropies = normalized_entropy(matched)
                total += float(entropies.sum())
                count += int(entropies.numel())
            return total / max(count, 1)

        low = float(beta_min)
        high = float(beta_max)
        low_entropy = mean_entropy(low)
        high_entropy = mean_entropy(high)
        target = float(target_entropy)
        if target <= low_entropy:
            beta = low
        elif target >= high_entropy:
            beta = high
        else:
            for _ in range(int(steps)):
                middle = (low + high) / 2.0
                if mean_entropy(middle) < target:
                    low = middle
                else:
                    high = middle
            beta = (low + high) / 2.0
        achieved = mean_entropy(beta)
        betas.append(float(beta))
        audit.append(
            {
                "layer": int(layer_index + 1),
                "beta": float(beta),
                "target_normalized_entropy": target,
                "achieved_normalized_entropy": float(achieved),
                "absolute_error": abs(float(achieved) - target),
            }
        )
    return betas, audit


def feature_specs() -> list[dict[str, str]]:
    return [
        {
            "name": f"{source}__{gate}__{cost}__{feature}",
            "source": source,
            "gate": gate,
            "cost": cost,
            "feature": feature,
        }
        for source in SOURCES
        for gate in GATES
        for cost in COSTS
        for feature in ("risk", "risk_ev")
    ]


def position_and_mentions_from_shards(
    shard_dir: Path,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    positions: dict[str, dict[str, Any]] = {}
    mentions: list[dict[str, Any]] = []
    for shard in load_shards(shard_dir):
        for row in shard["positions"]:
            key = str(row["target_key"])
            if key in positions:
                raise ValueError(f"Duplicate position row {key}")
            positions[key] = row
        mentions.extend(dict(row) for row in shard["sample_table"])
    missing = sorted(
        {str(row["target_key"]) for row in mentions} - set(positions)
    )
    if missing:
        raise ValueError(f"Sample table refers to missing target rows: {missing[:5]}")
    return positions, mentions


def build_training_matrix(
    *,
    positions: Mapping[str, Mapping[str, Any]],
    mentions: Sequence[Mapping[str, Any]],
    spec: Mapping[str, str],
    image_ids: set[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    rows: list[np.ndarray] = []
    labels: list[int] = []
    sample_images: list[int] = []
    mention_ids: list[str] = []
    for mention in mentions:
        image_id = int(mention["image_id"])
        if image_id not in image_ids:
            continue
        position = positions[str(mention["target_key"])]
        risk = position["risks"][spec["source"]][spec["gate"]][spec["cost"]]
        blocks = [_cpu_float(risk).numpy().reshape(-1)]
        if spec["feature"] == "risk_ev":
            blocks.append(
                _cpu_float(position["ev"][spec["gate"]]).numpy().reshape(-1)
            )
        row = np.concatenate(blocks).astype(np.float32, copy=False)
        if not np.isfinite(row).all():
            raise ValueError(f"Non-finite training row {mention['mention_id']}")
        rows.append(row)
        labels.append(int(mention["label"]))
        sample_images.append(image_id)
        mention_ids.append(str(mention["mention_id"]))
    if not rows:
        width = 0
        return (
            np.empty((0, width), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int64),
            [],
        )
    return (
        np.stack(rows).astype(np.float32, copy=False),
        np.asarray(labels, dtype=np.int32),
        np.asarray(sample_images, dtype=np.int64),
        mention_ids,
    )


def build_all_training_matrices_from_shards(
    *,
    shard_dir: Path,
    specs: Sequence[Mapping[str, str]],
    train_image_ids: set[int],
    test_image_ids: set[int],
) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]]]:
    """Stream full P shards once and retain only the small probe matrices."""
    spec_by_name = {str(spec["name"]): dict(spec) for spec in specs}
    buffers: dict[str, dict[str, dict[str, list[Any]]]] = {
        name: {
            split: {"rows": [], "labels": [], "images": [], "mentions": []}
            for split in ("train", "test")
        }
        for name in spec_by_name
    }
    for shard in load_shards(shard_dir):
        local_positions = {
            str(row["target_key"]): row for row in shard.get("positions", ())
        }
        for mention in shard.get("sample_table", ()):
            image_id = int(mention["image_id"])
            if image_id in train_image_ids:
                split = "train"
            elif image_id in test_image_ids:
                split = "test"
            else:
                continue
            target_key = str(mention["target_key"])
            if target_key not in local_positions:
                raise ValueError(
                    f"Shard mention {mention['mention_id']} has no local position row"
                )
            position = local_positions[target_key]
            for name, spec in spec_by_name.items():
                risk = _cpu_float(
                    position["risks"][spec["source"]][spec["gate"]][spec["cost"]]
                ).numpy().reshape(-1)
                blocks = [risk]
                if spec["feature"] == "risk_ev":
                    blocks.append(
                        _cpu_float(position["ev"][spec["gate"]]).numpy().reshape(-1)
                    )
                row = np.concatenate(blocks).astype(np.float32, copy=False)
                if not np.isfinite(row).all():
                    raise ValueError(f"Non-finite training row {mention['mention_id']}")
                target = buffers[name][split]
                target["rows"].append(row)
                target["labels"].append(int(mention["label"]))
                target["images"].append(image_id)
                target["mentions"].append(str(mention["mention_id"]))
        del shard, local_positions

    matrices = {}
    for name, split_buffers in buffers.items():
        matrices[name] = {}
        for split, values in split_buffers.items():
            if not values["rows"]:
                raise RuntimeError(f"No {split} rows for {name}")
            matrices[name][split] = (
                np.stack(values["rows"]).astype(np.float32, copy=False),
                np.asarray(values["labels"], dtype=np.int32),
                np.asarray(values["images"], dtype=np.int64),
                list(values["mentions"]),
            )
    return matrices


def distribution_pair_metrics(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    p = renormalize_rows(left.reshape(1, -1))[0]
    q = renormalize_rows(right.reshape(1, -1))[0]
    centered_p = p - p.mean()
    centered_q = q - q.mean()
    raw_cosine = float(torch.nn.functional.cosine_similarity(p, q, dim=0))
    centered_cosine = float(
        torch.nn.functional.cosine_similarity(centered_p, centered_q, dim=0)
    )
    midpoint = 0.5 * (p + q)
    js = 0.5 * (
        (p * (p.clamp_min(1e-30).log() - midpoint.clamp_min(1e-30).log())).sum()
        + (q * (q.clamp_min(1e-30).log() - midpoint.clamp_min(1e-30).log())).sum()
    )
    top_k = min(32, int(p.numel()))
    p_top = set(torch.topk(p, top_k).indices.tolist())
    q_top = set(torch.topk(q, top_k).indices.tolist())
    return {
        "raw_cosine": raw_cosine,
        "centered_cosine": centered_cosine,
        "js": float(js),
        "tv": float(0.5 * torch.abs(p - q).sum()),
        "top32_overlap": len(p_top & q_top) / float(top_k),
    }


def grouped_mean_std(rows: Sequence[Mapping[str, float]]) -> dict[str, dict[str, float]]:
    if not rows:
        return {}
    keys = sorted(set().union(*(row.keys() for row in rows)))
    result = {}
    for key in keys:
        values = np.asarray(
            [float(row[key]) for row in rows if key in row], dtype=np.float64
        )
        result[key] = {
            "count": int(values.size),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "median": float(np.median(values)),
        }
    return result
