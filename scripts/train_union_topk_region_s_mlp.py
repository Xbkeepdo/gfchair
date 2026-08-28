#!/usr/bin/env python3
"""Evaluate tokenwise Jacobian gain inside the P_JFFN/target Union Top-K.

No model forward is performed.  The script joins the completed round-one
JFFN shards (risk, EV, P_JFFN and target Q) with the completed round-two
shards (per-token response energy R_j and write energy I_j).
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from features.jffn_experiment import (  # noqa: E402
    atomic_json_save,
    atomic_torch_save,
)
from scripts.train_old_risk_ev_s_mlp import (  # noqa: E402
    EXPERIMENT,
    MODELS,
    aggregate_seed_metrics,
    align_baseline_predictions,
    ensemble_metrics,
    paired_image_bootstrap,
)
from scripts.train_s_times_risk_ev_mlp import (  # noqa: E402
    align_plus_s_predictions,
)
from scripts.train_torch_probe_feature_sets import (  # noqa: E402
    TorchProbeConfig,
    train_and_evaluate_probe,
)
from utils.config_utils import load_config  # noqa: E402
from utils.io_utils import load_json, load_pkl  # noqa: E402


EPS = 1.0e-12
SIDE_TOP_K = 32
SCHEMA_VERSION = "union-topk-region-s-mlp-v1"
OUTPUT_SUBDIR = "results/jffn_second_round/union_topk_region_s_mlp"
SPECS = ("token_all_s", "union_topk_s")
DISPLAY = {
    "token_all_s": "risk+EV+tokenwise all-token S",
    "union_topk_s": "risk+EV+Union-TopK S",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=(*MODELS, "summarize"))
    parser.add_argument(
        "--config", default="configs/model_configs_inslen_official_target.yaml"
    )
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--training-device", default="auto")
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def normalize_rows(values: Any) -> np.ndarray:
    array = np.nan_to_num(
        np.asarray(torch.as_tensor(values).float().cpu(), dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    array = np.maximum(array, 0.0)
    total = array.sum(axis=-1, keepdims=True)
    uniform = np.full_like(array, 1.0 / float(array.shape[-1]))
    return np.divide(array, np.maximum(total, EPS), out=uniform, where=total > 0.0)


def union_topk_mask(source: Any, target: Any, side_top_k: int = SIDE_TOP_K) -> np.ndarray:
    """Return the per-layer union of source and target Top-K supports."""
    p = normalize_rows(source)
    q = normalize_rows(target)
    if p.ndim != 2 or p.shape != q.shape:
        raise ValueError(f"Expected matching [layers,tokens], got {p.shape}/{q.shape}")
    k = min(max(int(side_top_k), 1), int(p.shape[-1]))
    p_index = np.argsort(-p, axis=-1, kind="stable")[:, :k]
    q_index = np.argsort(-q, axis=-1, kind="stable")[:, :k]
    layers = np.arange(p.shape[0])[:, None]
    mask = np.zeros(p.shape, dtype=bool)
    mask[layers, p_index] = True
    mask[layers, q_index] = True
    return mask


def region_ratio(
    response_energy: Any, write_energy: Any, mask: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute sum(R_j)/sum(I_j) over a boolean token support."""
    response = np.asarray(torch.as_tensor(response_energy).float().cpu(), dtype=np.float64)
    write = np.asarray(torch.as_tensor(write_energy).float().cpu(), dtype=np.float64)
    support = np.asarray(mask, dtype=bool)
    if response.shape != write.shape or response.shape != support.shape:
        raise ValueError(
            f"R/I/mask shape mismatch: {response.shape}/{write.shape}/{support.shape}"
        )
    if np.any(response < 0.0) or np.any(write < 0.0):
        raise ValueError("R_j and I_j must be non-negative norms")
    response_sum = np.where(support, response, 0.0).sum(axis=-1)
    write_sum = np.where(support, write, 0.0).sum(axis=-1)
    ratio = response_sum / np.maximum(write_sum, EPS)
    return (
        ratio.astype(np.float32),
        response_sum.astype(np.float32),
        write_sum.astype(np.float32),
    )


def _load_raw_targets(model_root: Path) -> tuple[dict[tuple[int, int], np.ndarray], dict[str, Any]]:
    """Rebuild only Q_raw from existing compact feature parts."""
    lookup: dict[tuple[int, int], np.ndarray] = {}
    rows_seen = 0
    duplicates = 0
    maximum_duplicate_error = 0.0
    parts = sorted(model_root.glob("features.part*.pkl"))
    if not parts:
        raise FileNotFoundError(f"No feature parts under {model_root}")
    for part in parts:
        rows = load_pkl(str(part))
        for row in rows:
            rows_seen += 1
            key = (int(row["image_id"]), int(row["response_token_idx"]))
            attention = np.asarray(
                row["dgst_t_attention_support_per_layer"], dtype=np.float32
            )
            gate = np.asarray(
                row["dgst_t_hpre_raw_logit_gauss_gate_per_layer"], dtype=np.float32
            )
            target = normalize_rows(attention * gate).astype(np.float32)
            previous = lookup.get(key)
            if previous is None:
                lookup[key] = target
            else:
                duplicates += 1
                error = float(np.max(np.abs(previous - target)))
                maximum_duplicate_error = max(maximum_duplicate_error, error)
                if error > 1.0e-6:
                    raise AssertionError(f"Conflicting reconstructed target {key}")
        del rows
        gc.collect()
    return lookup, {
        "origin": "reconstructed_from_existing_feature_parts",
        "formula": "normalize(attention_support * hpre_raw_logit_gauss_gate)",
        "parts": len(parts),
        "rows_seen": rows_seen,
        "unique_positions": len(lookup),
        "duplicate_rows": duplicates,
        "maximum_duplicate_error": maximum_duplicate_error,
    }


def _shard_pairs(model_root: Path) -> list[tuple[Path, Path]]:
    round_one = model_root / "results/jffn_p_comparison/shards"
    round_two = model_root / "results/jffn_second_round/shards"
    second_files = sorted(round_two.glob("features*_shard_*.pt"))
    if not second_files:
        raise FileNotFoundError(round_two)
    pairs = [(round_one / path.name, path) for path in second_files]
    missing = [str(first) for first, _ in pairs if not first.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} paired round-one shards")
    return pairs


def collect_matrices(
    model_root: Path, model: str
) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray], dict[str, Any]]:
    """Join both JFFN rounds and build 96-D all-token/Union-S inputs."""
    split_payload = load_json(str(model_root / "image_splits.json"))
    train_ids = {int(value) for value in split_payload["train"]}
    test_ids = {int(value) for value in split_payload["test"]}
    if train_ids & test_ids:
        raise AssertionError("Image split overlaps")

    # Some round-one shards predate compact target persistence, while later
    # shards contain Q directly.  Reconstruct the lean raw-gate lookup once so
    # both generations of shards are handled without dropping any position.
    target_lookup, target_audit = _load_raw_targets(model_root)
    target_audit["saved_target_maximum_error"] = 0.0

    feature_names = (
        "old_aggregate_s",
        "token_all_s",
        "union_topk_s",
        "union_response_sum",
        "union_write_sum",
    )
    stores: dict[str, dict[str, Any]] = {
        split: {
            "x": {spec: [] for spec in SPECS},
            **{name: [] for name in feature_names},
            "y": [],
            "image_ids": [],
            "mention_ids": [],
        }
        for split in ("train", "test")
    }
    audit: dict[str, Any] = {
        "round_one_shards": 0,
        "round_two_shards": 0,
        "positions": 0,
        "mentions": 0,
        "maximum_jffn_distribution_error": 0.0,
        "maximum_jffn_energy_error": 0.0,
        "union_sizes": [],
        "target": target_audit,
    }
    seen_mentions: set[str] = set()

    for first_path, second_path in _shard_pairs(model_root):
        first = torch.load(first_path, map_location="cpu", weights_only=False)
        second = torch.load(second_path, map_location="cpu", weights_only=False)
        audit["round_one_shards"] += 1
        audit["round_two_shards"] += 1
        if [int(value) for value in first["image_ids"]] != [
            int(value) for value in second["image_ids"]
        ]:
            raise AssertionError(f"Image mismatch: {first_path.name}")
        first_positions = {str(row["target_key"]): row for row in first["positions"]}
        second_positions = {str(row["target_key"]): row for row in second["positions"]}
        if set(first_positions) != set(second_positions):
            raise AssertionError(f"Position mismatch: {first_path.name}")
        first_mentions = [str(row["mention_id"]) for row in first["sample_table"]]
        second_mentions = [str(row["mention_id"]) for row in second["sample_table"]]
        if first_mentions != second_mentions:
            raise AssertionError(f"Mention mismatch: {first_path.name}")

        position_features: dict[str, dict[str, np.ndarray]] = {}
        for target_key, second_position in second_positions.items():
            first_position = first_positions[target_key]
            response = np.asarray(second_position["jffn_energy"], dtype=np.float32)
            write = np.asarray(second_position["write_energy"], dtype=np.float32)
            p_second = normalize_rows(second_position["jffn_distribution"])
            p_first = normalize_rows(first_position["sources"]["new_jffn"])
            audit["maximum_jffn_distribution_error"] = max(
                audit["maximum_jffn_distribution_error"],
                float(np.max(np.abs(p_second - p_first))),
            )
            audit["maximum_jffn_energy_error"] = max(
                audit["maximum_jffn_energy_error"],
                float(
                    np.max(
                        np.abs(
                            response
                            - np.asarray(first_position["jffn_energy"], dtype=np.float32)
                        )
                    )
                ),
            )
            key = (
                int(second_position["image_id"]),
                int(second_position["response_index"]),
            )
            if target_key != f"{key[0]}:{key[1]}":
                raise AssertionError(
                    f"Inconsistent target key {target_key!r} versus {key} "
                    f"in {second_path.name}"
                )
            if "targets" in first_position:
                target = first_position["targets"]["hpre_raw_logit_gauss"]
                reconstructed_target = target_lookup.get(key)
                if reconstructed_target is not None:
                    target_audit["saved_target_maximum_error"] = max(
                        target_audit["saved_target_maximum_error"],
                        float(
                            np.max(
                                np.abs(
                                    normalize_rows(target) - reconstructed_target
                                )
                            )
                        ),
                    )
            else:
                try:
                    target = target_lookup[key]
                except KeyError as exc:
                    same_image = sorted(
                        response_index
                        for image_id, response_index in target_lookup
                        if image_id == key[0]
                    )
                    raise AssertionError(
                        f"Missing target Q for {key} ({target_key}) in "
                        f"{second_path.name}; reconstructed positions for image: "
                        f"{same_image}"
                    ) from exc
            mask = union_topk_mask(p_second, target)
            union_s, union_r, union_i = region_ratio(response, write, mask)
            all_s, _, _ = region_ratio(response, write, np.ones_like(mask))
            old_s = np.asarray(
                second_position["diagnostics"]["gain"], dtype=np.float32
            ).reshape(-1)
            risk = np.asarray(
                first_position["risks"]["old_hpre_cos"]["hpre_raw_logit_gauss"]
                ["sqrt_matched_state"],
                dtype=np.float32,
            ).reshape(-1)
            ev = np.asarray(
                first_position["ev"]["hpre_raw_logit_gauss"], dtype=np.float32
            ).reshape(-1)
            if not (
                risk.shape
                == ev.shape
                == old_s.shape
                == all_s.shape
                == union_s.shape
            ):
                raise AssertionError(f"Layer mismatch for {target_key}")
            item = {
                "old_aggregate_s": old_s,
                "token_all_s": all_s,
                "union_topk_s": union_s,
                "union_response_sum": union_r,
                "union_write_sum": union_i,
                "x": {
                    "token_all_s": np.concatenate((risk, ev, all_s)).astype(
                        np.float32, copy=False
                    ),
                    "union_topk_s": np.concatenate((risk, ev, union_s)).astype(
                        np.float32, copy=False
                    ),
                },
            }
            if not all(np.isfinite(value).all() for name, value in item.items() if name != "x"):
                raise ValueError(f"Non-finite position feature {target_key}")
            if not all(np.isfinite(value).all() for value in item["x"].values()):
                raise ValueError(f"Non-finite training input {target_key}")
            position_features[target_key] = item
            audit["union_sizes"].extend(mask.sum(axis=-1).tolist())

        for mention in second["sample_table"]:
            mention_id = str(mention["mention_id"])
            if mention_id in seen_mentions:
                raise AssertionError(f"Duplicate mention ID {mention_id}")
            seen_mentions.add(mention_id)
            image_id = int(mention["image_id"])
            split = (
                "train"
                if image_id in train_ids
                else "test"
                if image_id in test_ids
                else None
            )
            if split is None:
                raise AssertionError(f"Image {image_id} is outside official split")
            item = position_features[str(mention["target_key"])]
            store = stores[split]
            for spec in SPECS:
                store["x"][spec].append(item["x"][spec])
            for name in feature_names:
                store[name].append(item[name])
            store["y"].append(int(mention["label"]))
            store["image_ids"].append(image_id)
            store["mention_ids"].append(mention_id)

        audit["positions"] += len(second_positions)
        audit["mentions"] += len(second["sample_table"])
        del first, second, first_positions, second_positions, position_features
        gc.collect()

    del target_lookup
    gc.collect()
    matrices: dict[str, dict[str, Any]] = {}
    for split, store in stores.items():
        matrices[split] = {
            "x": {spec: np.stack(store["x"][spec]) for spec in SPECS},
            **{name: np.stack(store[name]) for name in feature_names},
            "y": np.asarray(store["y"], dtype=np.int32),
            "image_ids": np.asarray(store["image_ids"], dtype=np.int64),
            "mention_ids": list(store["mention_ids"]),
        }
        if set(np.unique(matrices[split]["y"])) != {0, 1}:
            raise AssertionError(f"{split} lacks both labels")

    curves = {
        name: np.concatenate((matrices["train"][name], matrices["test"][name]))
        for name in feature_names
    }
    curves["labels"] = np.concatenate((matrices["train"]["y"], matrices["test"]["y"]))
    union_sizes = np.asarray(audit.pop("union_sizes"), dtype=np.int32)
    audit.update(
        {
            "train_mentions": int(len(matrices["train"]["y"])),
            "test_mentions": int(len(matrices["test"]["y"])),
            "train_images": int(np.unique(matrices["train"]["image_ids"]).size),
            "test_images": int(np.unique(matrices["test"]["image_ids"]).size),
            "layers": int(matrices["train"]["union_topk_s"].shape[1]),
            "input_dimensions": int(matrices["train"]["x"]["union_topk_s"].shape[1]),
            "source": "P_JFFN",
            "target_gate": "hpre_raw_logit_gauss",
            "source_top_k": SIDE_TOP_K,
            "target_top_k": SIDE_TOP_K,
            "union_size_minimum": int(union_sizes.min()),
            "union_size_mean": float(union_sizes.mean()),
            "union_size_maximum": int(union_sizes.max()),
            "feature_normalization": "none",
            "all_finite": True,
        }
    )
    return matrices, curves, audit


def curve_statistics(curves: Mapping[str, np.ndarray]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    labels = np.asarray(curves["labels"], dtype=np.int32)
    feature_names = (
        "old_aggregate_s",
        "token_all_s",
        "union_topk_s",
        "union_response_sum",
        "union_write_sum",
    )
    label_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    for feature in feature_names:
        values = np.asarray(curves[feature], dtype=np.float64)
        for layer in range(values.shape[1]):
            selected: dict[int, np.ndarray] = {}
            stats: dict[int, tuple[float, float]] = {}
            for label, name in ((1, "REAL"), (0, "HALL")):
                row = values[labels == label, layer]
                selected[label] = row
                mean = float(row.mean())
                std = float(row.std(ddof=1))
                sem = std / math.sqrt(row.size)
                stats[label] = (mean, std)
                label_rows.append(
                    {
                        "feature": feature,
                        "label": name,
                        "layer": layer + 1,
                        "n": int(row.size),
                        "mean": mean,
                        "std": std,
                        "sem": sem,
                        "ci95_low": mean - 1.96 * sem,
                        "ci95_high": mean + 1.96 * sem,
                    }
                )
            real_mean, real_std = stats[1]
            hall_mean, hall_std = stats[0]
            difference = hall_mean - real_mean
            difference_sem = math.sqrt(
                hall_std**2 / selected[0].size + real_std**2 / selected[1].size
            )
            pooled = math.sqrt(
                (
                    (selected[0].size - 1) * hall_std**2
                    + (selected[1].size - 1) * real_std**2
                )
                / (selected[0].size + selected[1].size - 2)
            )
            auc = float(roc_auc_score(1 - labels, values[:, layer]))
            effect_rows.append(
                {
                    "feature": feature,
                    "layer": layer + 1,
                    "hall_minus_real": difference,
                    "difference_sem": difference_sem,
                    "difference_ci95_low": difference - 1.96 * difference_sem,
                    "difference_ci95_high": difference + 1.96 * difference_sem,
                    "cohens_d_hall_minus_real": difference / max(pooled, EPS),
                    "hall_positive_raw_auroc": auc,
                    "best_orientation_auroc": max(auc, 1.0 - auc),
                }
            )
    return label_rows, effect_rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(model: str, rows: Sequence[Mapping[str, Any]], output_dir: Path) -> None:
    colors = {"REAL": "#2563eb", "HALL": "#dc2626"}
    features = (
        ("old_aggregate_s", "Old aggregate S = ||Σδ_j|| / ||Σa_j||"),
        ("token_all_s", "Tokenwise all-token S = ΣR_j / ΣI_j"),
        ("union_topk_s", "Union-TopK S = ΣU R_j / ΣU I_j"),
    )
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 10.5), sharex=True)
    for axis, (feature, title) in zip(axes, features):
        for label in ("REAL", "HALL"):
            selected = [
                row for row in rows if row["feature"] == feature and row["label"] == label
            ]
            x = np.asarray([int(row["layer"]) for row in selected])
            mean = np.asarray([float(row["mean"]) for row in selected])
            low = np.asarray([float(row["ci95_low"]) for row in selected])
            high = np.asarray([float(row["ci95_high"]) for row in selected])
            axis.plot(x, mean, color=colors[label], label=label, linewidth=2.0)
            axis.fill_between(x, low, high, color=colors[label], alpha=0.18)
        axis.set_title(title)
        axis.set_ylabel("Mean")
        axis.grid(alpha=0.25)
        axis.legend()
    axes[-1].set_xlabel("Decoder layer")
    fig.suptitle(f"{model}: REAL vs HALL Jacobian sensitivity")
    fig.tight_layout()
    stem = output_dir / f"{model}_union_topk_region_s_real_hall_curves"
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.0), sharex=True)
    for axis, feature, title in zip(
        axes,
        ("union_response_sum", "union_write_sum"),
        ("Union response energy ΣU R_j", "Union write energy ΣU I_j"),
    ):
        for label in ("REAL", "HALL"):
            selected = [
                row for row in rows if row["feature"] == feature and row["label"] == label
            ]
            x = np.asarray([int(row["layer"]) for row in selected])
            mean = np.asarray([float(row["mean"]) for row in selected])
            low = np.asarray([float(row["ci95_low"]) for row in selected])
            high = np.asarray([float(row["ci95_high"]) for row in selected])
            axis.plot(x, mean, color=colors[label], label=label, linewidth=2.0)
            axis.fill_between(x, low, high, color=colors[label], alpha=0.18)
        axis.set_title(title)
        axis.set_ylabel("Mean norm sum")
        axis.grid(alpha=0.25)
        axis.legend()
    axes[-1].set_xlabel("Decoder layer")
    fig.suptitle(f"{model}: components of Union-TopK S")
    fig.tight_layout()
    stem = output_dir / f"{model}_union_topk_region_r_i_real_hall_curves"
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _curve_summary(effect_rows: Sequence[Mapping[str, Any]], feature: str) -> dict[str, Any]:
    rows = [row for row in effect_rows if row["feature"] == feature]
    strongest = max(rows, key=lambda row: abs(float(row["cohens_d_hall_minus_real"])))
    return {
        "hall_higher_layers": sum(float(row["hall_minus_real"]) > 0 for row in rows),
        "real_higher_layers": sum(float(row["hall_minus_real"]) < 0 for row in rows),
        "ci_excludes_zero_hall_higher": sum(
            float(row["difference_ci95_low"]) > 0 for row in rows
        ),
        "ci_excludes_zero_real_higher": sum(
            float(row["difference_ci95_high"]) < 0 for row in rows
        ),
        "strongest_layer": int(strongest["layer"]),
        "strongest_hall_minus_real": float(strongest["hall_minus_real"]),
        "strongest_cohens_d": float(strongest["cohens_d_hall_minus_real"]),
        "strongest_best_orientation_auroc": float(strongest["best_orientation_auroc"]),
    }


def _probe_config(config: Mapping[str, Any], seed: int) -> TorchProbeConfig:
    section = config.get("jffn_p_comparison") or {}
    training = section.get("training") or {}
    inherited = (config.get("training") or {}).get("torch_probe") or {}
    return TorchProbeConfig(
        hidden_sizes=tuple(
            int(value)
            for value in training.get(
                "hidden_sizes", inherited.get("hidden_sizes", [128, 64, 32])
            )
        ),
        dropout=float(inherited.get("dropout", 0.3)),
        drop_last=bool(inherited.get("drop_last", True)),
        batch_size=int(training.get("batch_size", 256)),
        num_epochs=int(training.get("max_epochs", 100)),
        learning_rate=float(inherited.get("learning_rate", 1e-3)),
        weight_decay=float(inherited.get("weight_decay", 1e-5)),
        lr_factor=float(inherited.get("lr_factor", 0.5)),
        lr_patience=int(inherited.get("lr_patience", 5)),
        early_stopping_patience=int(inherited.get("early_stopping_patience", 10)),
        seed=int(seed),
        positive_class="real",
        split_protocol="strict_82_no_validation",
        threshold_selection="train_f1",
        fixed_threshold=float(inherited.get("fixed_threshold", 0.5)),
        checkpoint_selection="minimum_train_loss",
    )


def _comparison_result(
    *, name: str, aggregate: Mapping[str, Any], labels: np.ndarray, probabilities: np.ndarray
) -> dict[str, Any]:
    return {
        "name": name,
        "aggregate": aggregate,
        "seed_ensemble": ensemble_metrics(labels, probabilities),
    }


def train_model(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    model_root = Path(args.outputs_root) / args.model / EXPERIMENT
    output_dir = model_root / OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    matrices, curves, audit = collect_matrices(model_root, args.model)
    label_rows, effect_rows = curve_statistics(curves)
    _write_csv(output_dir / "s_region_label_curves.csv", label_rows)
    _write_csv(output_dir / "s_region_hall_minus_real.csv", effect_rows)
    plot_curves(args.model, label_rows, output_dir)

    train = matrices["train"]
    test = matrices["test"]
    baseline_predictions, baseline_result = align_baseline_predictions(model_root, test)
    old_s_predictions, old_s_result = align_plus_s_predictions(model_root, test)
    comparison_training = (config.get("jffn_p_comparison") or {}).get("training") or {}
    seeds = [int(value) for value in comparison_training.get("seeds", [43, 44, 45])]
    progress_path = output_dir / "training_progress.pt"
    completed: dict[str, dict[int, dict[str, Any]]] = {spec: {} for spec in SPECS}
    probabilities: dict[str, dict[int, np.ndarray]] = {spec: {} for spec in SPECS}
    if args.resume and progress_path.exists():
        progress = torch.load(progress_path, map_location="cpu", weights_only=False)
        if progress.get("schema_version") != SCHEMA_VERSION:
            raise AssertionError("Incompatible Union-TopK S resume artifact")
        for spec in SPECS:
            completed[spec] = {
                int(row["seed"]): row
                for row in (progress.get("seed_results") or {}).get(spec, ())
            }
            probabilities[spec] = {
                int(seed): np.asarray(value, dtype=np.float32)
                for seed, value in (progress.get("probabilities") or {}).get(spec, {}).items()
            }

    device = _device(args.training_device)
    for spec in SPECS:
        for seed in seeds:
            if seed in completed[spec] and seed in probabilities[spec]:
                print(f"[Union-S] reuse {args.model}/{spec}/seed={seed}", flush=True)
                continue
            metrics = train_and_evaluate_probe(
                X_train=train["x"][spec],
                y_train=train["y"],
                X_val=np.empty((0, train["x"][spec].shape[1]), dtype=np.float32),
                y_val=np.empty((0,), dtype=np.int32),
                X_test=test["x"][spec],
                y_test=test["y"],
                config=_probe_config(config, seed),
                device=device,
                output_dir=str(output_dir / "training" / spec / f"seed_{seed}"),
                return_probabilities=True,
            )
            test_prob = np.asarray(metrics.pop("test_probabilities"), dtype=np.float32)
            metrics.pop("train_probabilities", None)
            metrics["seed"] = int(seed)
            metrics["num_features"] = int(train["x"][spec].shape[1])
            completed[spec][seed] = metrics
            probabilities[spec][seed] = test_prob
            atomic_torch_save(
                {
                    "schema_version": SCHEMA_VERSION,
                    "model": args.model,
                    "seed_results": {
                        name: [completed[name][value] for value in sorted(completed[name])]
                        for name in SPECS
                    },
                    "probabilities": probabilities,
                    "labels": test["y"],
                    "image_ids": test["image_ids"],
                    "mention_ids": test["mention_ids"],
                },
                progress_path,
            )
            print(
                f"[Union-S] {args.model}/{spec}/seed={seed} "
                f"AUROC={metrics['auc']:.6f} "
                f"HallAUPR={metrics['hallucination_positive']['aupr']:.6f}",
                flush=True,
            )

    baseline_prob = np.mean([baseline_predictions[seed] for seed in seeds], axis=0)
    old_s_prob = np.mean([old_s_predictions[seed] for seed in seeds], axis=0)
    new_prob = {
        spec: np.mean([probabilities[spec][seed] for seed in seeds], axis=0)
        for spec in SPECS
    }
    results = {
        spec: {
            "name": DISPLAY[spec],
            "seeds": [completed[spec][seed] for seed in seeds],
            "aggregate": aggregate_seed_metrics([completed[spec][seed] for seed in seeds]),
            "seed_ensemble": ensemble_metrics(test["y"], new_prob[spec]),
        }
        for spec in SPECS
    }
    comparisons = {
        "old_risk_ev": _comparison_result(
            name="old risk+EV",
            aggregate=baseline_result["aggregate"],
            labels=test["y"],
            probabilities=baseline_prob,
        ),
        "old_risk_ev_old_aggregate_s": _comparison_result(
            name="old risk+EV+old aggregate S",
            aggregate=old_s_result["aggregate"],
            labels=test["y"],
            probabilities=old_s_prob,
        ),
        **results,
    }
    bootstrap: dict[str, Any] = {}
    for spec in SPECS:
        for reference_name, reference_prob in (
            ("old_risk_ev", baseline_prob),
            ("old_aggregate_s", old_s_prob),
        ):
            bootstrap[f"{spec}_versus_{reference_name}"] = paired_image_bootstrap(
                labels=test["y"],
                image_ids=test["image_ids"],
                new_probabilities=new_prob[spec],
                baseline_probabilities=reference_prob,
                replicates=args.bootstrap_replicates,
                seed=20260828 + len(bootstrap),
            )
    bootstrap["union_topk_s_versus_token_all_s"] = paired_image_bootstrap(
        labels=test["y"],
        image_ids=test["image_ids"],
        new_probabilities=new_prob["union_topk_s"],
        baseline_probabilities=new_prob["token_all_s"],
        replicates=args.bootstrap_replicates,
        seed=20260832,
    )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "definition": {
            "union": "Top32(P_JFFN) union Top32(Q_hpre_raw_logit_gauss) per layer",
            "union_topk_s": "sum_{j in U} ||J_f(z)a_j||_2 / (sum_{j in U} ||a_j||_2 + eps)",
            "token_all_s_control": "sum_j ||J_f(z)a_j||_2 / (sum_j ||a_j||_2 + eps)",
            "old_aggregate_s": "||sum_j J_f(z)a_j||_2 / (||sum_j a_j||_2 + eps)",
        },
        "sample_audit": audit,
        "training_protocol": {
            "hidden_sizes": [128, 64, 32],
            "batch_size": int(comparison_training.get("batch_size", 256)),
            "max_epochs": int(comparison_training.get("max_epochs", 100)),
            "seeds": seeds,
            "feature_normalization": "none",
            "checkpoint_selection": "minimum_train_loss",
            "threshold_selection": "train_f1",
            "split": "existing image-level 80/20",
        },
        "curve_summary": {
            feature: _curve_summary(effect_rows, feature)
            for feature in ("old_aggregate_s", "token_all_s", "union_topk_s")
        },
        "comparisons": comparisons,
        "paired_bootstrap": bootstrap,
    }
    atomic_json_save(payload, output_dir / "results.json")
    (output_dir / f"{args.model}_union_topk_region_s_report.md").write_text(
        report_markdown(args.model, payload), encoding="utf-8"
    )
    print(f"[Union-S] wrote {output_dir}", flush=True)


def report_markdown(model: str, payload: Mapping[str, Any]) -> str:
    lines = [
        f"# {model}: Union-TopK 区域新 S",
        "",
        "新 S 在每层 `Top32(P_JFFN) ∪ Top32(Q_raw)` 内计算 "
        "`ΣR_j / ΣI_j`。`tokenwise all-token S` 使用相同标量聚合但不限制区域，"
        "用于单独检验 Union 区域筛选的贡献。",
        "",
        "## 曲线摘要",
        "",
        "| S | Hall>Real layers | Real>Hall layers | Strongest layer | Hall−Real | Cohen's d |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    curve_names = {
        "old_aggregate_s": "old aggregate S",
        "token_all_s": "tokenwise all-token S",
        "union_topk_s": "Union-TopK S",
    }
    for key, name in curve_names.items():
        row = payload["curve_summary"][key]
        lines.append(
            f"| {name} | {row['hall_higher_layers']} | {row['real_higher_layers']} | "
            f"{row['strongest_layer']} | {row['strongest_hall_minus_real']:+.6f} | "
            f"{row['strongest_cohens_d']:+.4f} |"
        )
    lines += [
        "",
        "## 三种子 MLP",
        "",
        "| Feature | Mean AUROC | Mean Hall F1 | Mean Hall AUPR | Ensemble AUROC | Ensemble Hall AUPR |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    order = (
        "old_risk_ev",
        "old_risk_ev_old_aggregate_s",
        "token_all_s",
        "union_topk_s",
    )
    for key in order:
        row = payload["comparisons"][key]
        aggregate = row["aggregate"]
        ensemble = row["seed_ensemble"]
        lines.append(
            f"| {row['name']} | {aggregate['auroc']['mean']:.6f} ± "
            f"{aggregate['auroc']['std']:.6f} | {aggregate['hall_f1']['mean']:.6f} | "
            f"{aggregate['hall_aupr']['mean']:.6f} ± {aggregate['hall_aupr']['std']:.6f} | "
            f"{ensemble['auroc']:.6f} | {ensemble['hall_aupr']:.6f} |"
        )
    lines += ["", "## 图片级 paired bootstrap", ""]
    for key, name in (
        ("union_topk_s_versus_old_risk_ev", "Union S vs risk+EV"),
        ("union_topk_s_versus_old_aggregate_s", "Union S vs old aggregate S"),
        ("union_topk_s_versus_token_all_s", "Union S vs all-token S"),
    ):
        row = payload["paired_bootstrap"][key]
        lines.append(
            f"- {name}：ensemble AUROC Δ={row['new_minus_baseline_auroc']:+.6f}，"
            f"95% CI [{row['auroc_ci95'][0]:+.6f},{row['auroc_ci95'][1]:+.6f}]；"
            f"Hall-AUPR Δ={row['new_minus_baseline_hall_aupr']:+.6f}，"
            f"95% CI [{row['hall_aupr_ci95'][0]:+.6f},{row['hall_aupr_ci95'][1]:+.6f}]。"
        )
    audit = payload["sample_audit"]
    lines += [
        "",
        f"实际 Union 大小：min/mean/max = {audit['union_size_minimum']}/"
        f"{audit['union_size_mean']:.2f}/{audit['union_size_maximum']}；"
        f"train/test mentions={audit['train_mentions']}/{audit['test_mentions']}。",
        "",
    ]
    return "\n".join(lines)


def summarize(outputs_root: Path) -> None:
    payloads: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        path = outputs_root / model / EXPERIMENT / OUTPUT_SUBDIR / "results.json"
        payload = json.loads(path.read_text())
        payloads[model] = payload
        for feature in (
            "old_risk_ev",
            "old_risk_ev_old_aggregate_s",
            "token_all_s",
            "union_topk_s",
        ):
            result = payload["comparisons"][feature]
            rows.append(
                {
                    "model": model,
                    "feature": feature,
                    "mean_auroc": result["aggregate"]["auroc"]["mean"],
                    "std_auroc": result["aggregate"]["auroc"]["std"],
                    "mean_hall_f1": result["aggregate"]["hall_f1"]["mean"],
                    "mean_hall_aupr": result["aggregate"]["hall_aupr"]["mean"],
                    "ensemble_auroc": result["seed_ensemble"]["auroc"],
                    "ensemble_hall_aupr": result["seed_ensemble"]["hall_aupr"],
                }
            )
    atomic_json_save(
        {"schema_version": SCHEMA_VERSION, "models": payloads},
        outputs_root / "union_topk_region_s_mlp_2model_summary.json",
    )
    _write_csv(outputs_root / "union_topk_region_s_mlp_2model_metrics.csv", rows)
    lines = [
        "# Union-TopK 区域新 S：两模型汇总",
        "",
        "新 S 为 `Top32(P_JFFN) ∪ Top32(Q_raw)` 区域内的 `ΣR_j/ΣI_j`。",
        "",
        "| Model | Feature | Mean AUROC | Mean Hall F1 | Mean Hall AUPR | Ensemble AUROC |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['feature']} | {row['mean_auroc']:.6f} ± "
            f"{row['std_auroc']:.6f} | {row['mean_hall_f1']:.6f} | "
            f"{row['mean_hall_aupr']:.6f} | {row['ensemble_auroc']:.6f} |"
        )
    lines += ["", "## Union S 的 paired bootstrap", ""]
    for model, payload in payloads.items():
        for key, name in (
            ("union_topk_s_versus_old_risk_ev", "vs risk+EV"),
            ("union_topk_s_versus_old_aggregate_s", "vs old aggregate S"),
            ("union_topk_s_versus_token_all_s", "vs all-token S"),
        ):
            row = payload["paired_bootstrap"][key]
            lines.append(
                f"- {model} {name}：AUROC Δ={row['new_minus_baseline_auroc']:+.6f}，"
                f"95% CI [{row['auroc_ci95'][0]:+.6f},{row['auroc_ci95'][1]:+.6f}]。"
            )
    (outputs_root / "union_topk_region_s_mlp_2model_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("[Union-S] wrote two-model summary", flush=True)


def main() -> None:
    args = parse_args()
    if args.model == "summarize":
        summarize(Path(args.outputs_root))
    else:
        train_model(args)


if __name__ == "__main__":
    main()
