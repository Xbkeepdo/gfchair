#!/usr/bin/env python3
"""Train risk+EV+exact Union-TopK aggregate Jacobian S.

The exact Union feature is produced while visual directions are still
vectors:

    ||sum_{j in U} J_f(z) a_j|| / (||sum_{j in U} a_j|| + eps)

This script performs no VLM forward. It joins the compact aggregate shards
with the already completed JFFN comparison/region-S artifacts.
"""

from __future__ import annotations

import argparse
import csv
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

from features.jffn_experiment import atomic_json_save, atomic_torch_save  # noqa: E402
from scripts.train_old_risk_ev_s_mlp import (  # noqa: E402
    EXPERIMENT,
    MODELS,
    aggregate_seed_metrics,
    align_baseline_predictions,
    ensemble_metrics,
    paired_image_bootstrap,
)
from scripts.train_s_times_risk_ev_mlp import align_plus_s_predictions  # noqa: E402
from scripts.train_torch_probe_feature_sets import (  # noqa: E402
    train_and_evaluate_probe,
)
from scripts.train_union_topk_region_s_mlp import (  # noqa: E402
    _comparison_result,
    _curve_summary,
    _probe_config,
    collect_matrices,
    curve_statistics,
)
from utils.config_utils import load_config  # noqa: E402


SCHEMA_VERSION = "union-topk-aggregate-s-mlp-v2"
DEFAULT_EXTRACTION_SUBDIR = "results/jffn_union_aggregate/shards"
DEFAULT_OUTPUT_SUBDIR = "results/jffn_second_round/union_topk_aggregate_s_mlp"
REGION_RESULT_SUBDIR = "results/jffn_second_round/union_topk_region_s_mlp"
FIELDS = (
    "gain",
    "input_norm",
    "response_norm",
    "input_cancellation_ratio",
    "response_cancellation_ratio",
    "union_size",
)


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
    parser.add_argument(
        "--aggregate-shards-subdir", default=DEFAULT_EXTRACTION_SUBDIR
    )
    parser.add_argument("--output-subdir", default=DEFAULT_OUTPUT_SUBDIR)
    parser.add_argument(
        "--allow-subset",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow aggregate shards to contain a strict mention subset.",
    )
    return parser.parse_args()


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


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


def load_union_aggregate(
    shard_dir: Path,
    matrices: dict[str, dict[str, Any]],
    *,
    allow_subset: bool = False,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    paths = sorted(shard_dir.glob("features*_shard_*.pt"))
    if not paths:
        raise FileNotFoundError(f"No Union aggregate shards under {shard_dir}")
    mention_values: dict[str, dict[str, np.ndarray]] = {}
    position_count = 0
    image_ids: set[int] = set()
    union_sizes: list[np.ndarray] = []
    maximum_gain_error = 0.0
    for path in paths:
        shard = torch.load(path, map_location="cpu", weights_only=False)
        if shard.get("schema_version") != "jffn-union-aggregate-v1":
            raise AssertionError(f"Unexpected aggregate shard schema: {path}")
        image_ids.update(int(value) for value in shard["image_ids"])
        positions = {str(row["target_key"]): row for row in shard["positions"]}
        position_count += len(positions)
        for row in positions.values():
            if "union_aggregate" not in row:
                raise AssertionError(f"{path.name} has no Union aggregate fields")
            values = {
                field: np.asarray(row["union_aggregate"][field], dtype=np.float32)
                for field in FIELDS
            }
            if not all(np.isfinite(value).all() for value in values.values()):
                raise ValueError(f"Non-finite Union aggregate position {row['target_key']}")
            recomputed = values["response_norm"] / np.maximum(
                values["input_norm"], 1.0e-12
            )
            maximum_gain_error = max(
                maximum_gain_error,
                float(np.max(np.abs(recomputed - values["gain"]))),
            )
            union_sizes.append(values["union_size"])
        for mention in shard["sample_table"]:
            mention_id = str(mention["mention_id"])
            if mention_id in mention_values:
                raise AssertionError(f"Duplicate aggregate mention {mention_id}")
            position = positions[str(mention["target_key"])]
            mention_values[mention_id] = {
                field: np.asarray(
                    position["union_aggregate"][field], dtype=np.float32
                )
                for field in FIELDS
            }

    expected = {
        str(value)
        for split in ("train", "test")
        for value in matrices[split]["mention_ids"]
    }
    available = set(mention_values)
    if not allow_subset and available != expected:
        raise AssertionError(
            f"Aggregate cohort mismatch: missing={len(expected-available)}, "
            f"extra={len(available-expected)}"
        )
    if allow_subset:
        if not available or not available.issubset(expected):
            raise AssertionError(
                f"Invalid aggregate subset: size={len(available)}, "
                f"extra={len(available-expected)}"
            )
        for split in ("train", "test"):
            original_ids = [str(value) for value in matrices[split]["mention_ids"]]
            mask = np.asarray([value in available for value in original_ids], dtype=bool)
            original_size = len(original_ids)
            for key, value in list(matrices[split].items()):
                if key == "x":
                    matrices[split][key] = {
                        name: array[mask] for name, array in value.items()
                    }
                elif key == "mention_ids":
                    matrices[split][key] = [
                        value[index] for index in np.flatnonzero(mask)
                    ]
                elif isinstance(value, np.ndarray) and len(value) == original_size:
                    matrices[split][key] = value[mask]
            if not len(matrices[split]["mention_ids"]):
                raise AssertionError(f"Aggregate subset has no {split} mentions")
    output: dict[str, dict[str, np.ndarray]] = {}
    for split in ("train", "test"):
        ids = [str(value) for value in matrices[split]["mention_ids"]]
        output[split] = {
            field: np.stack([mention_values[value][field] for value in ids])
            for field in FIELDS
        }
    sizes = np.concatenate(union_sizes).astype(np.int32)
    return output, {
        "shards": len(paths),
        "images": len(image_ids),
        "positions": position_count,
        "mentions": len(mention_values),
        "maximum_gain_identity_error": maximum_gain_error,
        "union_size_minimum": int(sizes.min()),
        "union_size_mean": float(sizes.mean()),
        "union_size_maximum": int(sizes.max()),
    }


def single_feature_statistics(
    feature: str, values: np.ndarray, labels: np.ndarray
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    label_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    for layer in range(values.shape[1]):
        selected: dict[int, np.ndarray] = {}
        means: dict[int, float] = {}
        stds: dict[int, float] = {}
        for label, name in ((1, "REAL"), (0, "HALL")):
            row = values[labels == label, layer].astype(np.float64)
            selected[label] = row
            mean = float(row.mean())
            std = float(row.std(ddof=1))
            sem = std / math.sqrt(row.size)
            means[label], stds[label] = mean, std
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
        difference = means[0] - means[1]
        difference_sem = math.sqrt(
            stds[0] ** 2 / selected[0].size + stds[1] ** 2 / selected[1].size
        )
        pooled = math.sqrt(
            (
                (selected[0].size - 1) * stds[0] ** 2
                + (selected[1].size - 1) * stds[1] ** 2
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
                "cohens_d_hall_minus_real": difference / max(pooled, 1.0e-12),
                "hall_positive_raw_auroc": auc,
                "best_orientation_auroc": max(auc, 1.0 - auc),
            }
        )
    return label_rows, effect_rows


def plot_curves(
    model: str,
    label_rows: Sequence[Mapping[str, Any]],
    component_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> None:
    colors = {"REAL": "#2563eb", "HALL": "#dc2626"}

    def draw(features: Sequence[tuple[str, str]], stem: str, ylabel: str) -> None:
        fig, axes = plt.subplots(len(features), 1, figsize=(10.5, 3.25 * len(features)), sharex=True)
        if len(features) == 1:
            axes = [axes]
        for axis, (feature, title) in zip(axes, features):
            for label in ("REAL", "HALL"):
                rows = [
                    row
                    for row in label_rows if row["feature"] == feature and row["label"] == label
                ]
                if not rows:
                    rows = [
                        row
                        for row in component_rows
                        if row["feature"] == feature and row["label"] == label
                    ]
                x = np.asarray([int(row["layer"]) for row in rows])
                mean = np.asarray([float(row["mean"]) for row in rows])
                low = np.asarray([float(row["ci95_low"]) for row in rows])
                high = np.asarray([float(row["ci95_high"]) for row in rows])
                axis.plot(x, mean, color=colors[label], linewidth=2.0, label=label)
                axis.fill_between(x, low, high, color=colors[label], alpha=0.18)
            axis.set_title(title)
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.25)
            axis.legend()
        axes[-1].set_xlabel("Decoder layer")
        fig.suptitle(f"{model}: REAL vs HALL")
        fig.tight_layout()
        path = output_dir / f"{model}_{stem}"
        fig.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)

    draw(
        (
            ("old_aggregate_s", "All-token aggregate S = ||Σδ|| / ||Σa||"),
            ("union_topk_s", "Union tokenwise S = Σ||δ_j|| / Σ||a_j||"),
            ("union_aggregate_s", "Union aggregate S = ||ΣU δ_j|| / ||ΣU a_j||"),
        ),
        "union_aggregate_s_real_hall_curves",
        "Mean S",
    )
    draw(
        (
            ("union_aggregate_input_norm", "||ΣU a_j||"),
            ("union_aggregate_response_norm", "||ΣU J_f(z)a_j||"),
            ("union_input_cancellation", "Input cancellation ratio"),
            ("union_response_cancellation", "Response cancellation ratio"),
        ),
        "union_aggregate_components_real_hall_curves",
        "Mean",
    )


def align_region_predictions(
    model_root: Path, test: Mapping[str, Any]
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    result_dir = model_root / REGION_RESULT_SUBDIR
    result = json.loads((result_dir / "results.json").read_text())
    progress = torch.load(
        result_dir / "training_progress.pt", map_location="cpu", weights_only=False
    )
    stored = [str(value) for value in progress["mention_ids"]]
    lookup = {value: index for index, value in enumerate(stored)}
    if len(lookup) != len(stored):
        raise AssertionError("Union tokenwise progress has duplicate mention IDs")
    indices = np.asarray(
        [lookup[str(value)] for value in test["mention_ids"]], dtype=np.int64
    )
    if not np.array_equal(np.asarray(progress["labels"])[indices], test["y"]):
        raise AssertionError("Union tokenwise labels do not align")
    predictions = {
        seed: np.asarray(progress["probabilities"]["union_topk_s"][seed])[indices]
        for seed in (43, 44, 45)
    }
    return predictions, result["comparisons"]["union_topk_s"]


def train_model(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    model_root = Path(args.outputs_root) / args.model / EXPERIMENT
    output_dir = model_root / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    matrices, _old_curves_full, base_audit = collect_matrices(model_root, args.model)
    aggregate, extraction_audit = load_union_aggregate(
        model_root / args.aggregate_shards_subdir,
        matrices,
        allow_subset=bool(args.allow_subset),
    )
    for split in ("train", "test"):
        base_x = matrices[split]["x"]["union_topk_s"][:, :64]
        matrices[split]["x_union_aggregate"] = np.concatenate(
            (base_x, aggregate[split]["gain"]), axis=1
        ).astype(np.float32, copy=False)
        if matrices[split]["x_union_aggregate"].shape[1] != 96:
            raise AssertionError("Expected 96-D [risk,EV,Union aggregate S]")

    old_curves = {
        name: np.concatenate(
            (matrices["train"][name], matrices["test"][name])
        )
        for name in (
            "old_aggregate_s",
            "token_all_s",
            "union_topk_s",
            "union_response_sum",
            "union_write_sum",
        )
    }
    old_curves["labels"] = np.concatenate(
        (matrices["train"]["y"], matrices["test"]["y"])
    )
    labels = old_curves["labels"]
    union_gain = np.concatenate((aggregate["train"]["gain"], aggregate["test"]["gain"]))
    old_label_rows, old_effect_rows = curve_statistics(old_curves)
    gain_rows, gain_effects = single_feature_statistics(
        "union_aggregate_s", union_gain, labels
    )
    component_specs = {
        "union_aggregate_input_norm": "input_norm",
        "union_aggregate_response_norm": "response_norm",
        "union_input_cancellation": "input_cancellation_ratio",
        "union_response_cancellation": "response_cancellation_ratio",
    }
    component_rows: list[dict[str, Any]] = []
    component_effects: list[dict[str, Any]] = []
    for feature, field in component_specs.items():
        values = np.concatenate((aggregate["train"][field], aggregate["test"][field]))
        rows, effects = single_feature_statistics(feature, values, labels)
        component_rows.extend(rows)
        component_effects.extend(effects)
    label_rows = old_label_rows + gain_rows
    effect_rows = old_effect_rows + gain_effects
    _write_csv(output_dir / "union_aggregate_label_curves.csv", label_rows + component_rows)
    _write_csv(output_dir / "union_aggregate_hall_minus_real.csv", effect_rows + component_effects)
    plot_curves(args.model, label_rows, component_rows, output_dir)

    train, test = matrices["train"], matrices["test"]
    feature_x = {
        split: {
            "old_risk_ev": matrices[split]["x"]["union_topk_s"][:, :64],
            "old_risk_ev_old_aggregate_s": np.concatenate(
                (
                    matrices[split]["x"]["union_topk_s"][:, :64],
                    matrices[split]["old_aggregate_s"],
                ),
                axis=1,
            ).astype(np.float32, copy=False),
            "old_risk_ev_token_all_s": np.concatenate(
                (
                    matrices[split]["x"]["union_topk_s"][:, :64],
                    matrices[split]["token_all_s"],
                ),
                axis=1,
            ).astype(np.float32, copy=False),
            "old_risk_ev_union_tokenwise_s": matrices[split]["x"][
                "union_topk_s"
            ],
            "union_aggregate_s": matrices[split]["x_union_aggregate"],
        }
        for split in ("train", "test")
    }
    spec_names = {
        "old_risk_ev": "risk+EV",
        "old_risk_ev_old_aggregate_s": "risk+EV+all-token aggregate S",
        "old_risk_ev_token_all_s": "risk+EV+all-token tokenwise S",
        "old_risk_ev_union_tokenwise_s": "risk+EV+Union tokenwise S",
        "union_aggregate_s": "risk+EV+Union aggregate S",
    }
    # A subset experiment must retrain every comparator on the same subset.
    # For a full-cohort run we also train all four here so the implementation
    # and randomization remain exactly symmetric.
    specs = tuple(spec_names)
    section = (config.get("jffn_p_comparison") or {}).get("training") or {}
    seeds = [int(value) for value in section.get("seeds", [43, 44, 45])]
    progress_path = output_dir / "training_progress.pt"
    completed: dict[str, dict[int, dict[str, Any]]] = {spec: {} for spec in specs}
    probabilities: dict[str, dict[int, np.ndarray]] = {spec: {} for spec in specs}
    if args.resume and progress_path.exists():
        progress = torch.load(progress_path, map_location="cpu", weights_only=False)
        if progress.get("schema_version") != SCHEMA_VERSION:
            raise AssertionError("Incompatible Union aggregate resume artifact")
        for spec in specs:
            completed[spec] = {
                int(row["seed"]): row
                for row in progress["seed_results"].get(spec, ())
            }
            probabilities[spec] = {
                int(seed): np.asarray(value, dtype=np.float32)
                for seed, value in progress["probabilities"].get(spec, {}).items()
            }

    device = _device(args.training_device)
    for spec in specs:
        for seed in seeds:
            if seed in completed[spec] and seed in probabilities[spec]:
                print(
                    f"[Union-aggregate] reuse {args.model}/{spec}/seed={seed}",
                    flush=True,
                )
                continue
            width = int(feature_x["train"][spec].shape[1])
            metrics = train_and_evaluate_probe(
                X_train=feature_x["train"][spec],
                y_train=train["y"],
                X_val=np.empty((0, width), dtype=np.float32),
                y_val=np.empty((0,), dtype=np.int32),
                X_test=feature_x["test"][spec],
                y_test=test["y"],
                config=_probe_config(config, seed),
                device=device,
                output_dir=str(output_dir / "training" / spec / f"seed_{seed}"),
                return_probabilities=True,
            )
            test_prob = np.asarray(metrics.pop("test_probabilities"), dtype=np.float32)
            metrics.pop("train_probabilities", None)
            metrics["seed"] = seed
            metrics["num_features"] = width
            completed[spec][seed] = metrics
            probabilities[spec][seed] = test_prob
            atomic_torch_save(
                {
                    "schema_version": SCHEMA_VERSION,
                    "model": args.model,
                    "seed_results": {
                        name: [
                            completed[name][value]
                            for value in sorted(completed[name])
                        ]
                        for name in specs
                    },
                    "probabilities": probabilities,
                    "labels": test["y"],
                    "image_ids": test["image_ids"],
                    "mention_ids": test["mention_ids"],
                },
                progress_path,
            )
            print(
                f"[Union-aggregate] {args.model}/{spec}/seed={seed} "
                f"AUROC={metrics['auc']:.6f} "
                f"HallAUPR={metrics['hallucination_positive']['aupr']:.6f}",
                flush=True,
            )

    ensembles = {
        spec: np.mean([probabilities[spec][seed] for seed in seeds], axis=0)
        for spec in specs
    }
    comparisons = {
        spec: _comparison_result(
            name=spec_names[spec],
            aggregate=aggregate_seed_metrics(
                [completed[spec][seed] for seed in seeds]
            ),
            labels=test["y"],
            probabilities=ensembles[spec],
        )
        for spec in specs
    }
    bootstrap = {}
    for index, key in enumerate(specs[:-1]):
        bootstrap[f"union_aggregate_s_versus_{key}"] = paired_image_bootstrap(
            labels=test["y"],
            image_ids=test["image_ids"],
            new_probabilities=ensembles["union_aggregate_s"],
            baseline_probabilities=ensembles[key],
            replicates=args.bootstrap_replicates,
            seed=20260829 + index,
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "definition": {
            "union": "Top32(P_JFFN) union Top32(Q_hpre_raw_logit_gauss)",
            "union_aggregate_s": "||sum_{j in U} J_f(z)a_j|| / (||sum_{j in U} a_j|| + eps)",
            "contrast": "Union tokenwise S sums norms; Union aggregate S sums vectors before norms.",
        },
        "audit": {"base": base_audit, "aggregate_extraction": extraction_audit},
        "curve_summary": {
            feature: _curve_summary(effect_rows, feature)
            for feature in ("old_aggregate_s", "union_topk_s", "union_aggregate_s")
        },
        "training_protocol": {
            "inputs": {
                "old_risk_ev": "risk_32 + EV_32",
                "old_risk_ev_old_aggregate_s": "risk_32 + EV_32 + all-token aggregate S_32",
                "old_risk_ev_token_all_s": "risk_32 + EV_32 + all-token tokenwise S_32",
                "old_risk_ev_union_tokenwise_s": "risk_32 + EV_32 + Union tokenwise S_32",
                "union_aggregate_s": "risk_32 + EV_32 + Union aggregate S_32",
            },
            "hidden_sizes": [128, 64, 32],
            "batch_size": int(section.get("batch_size", 256)),
            "max_epochs": int(section.get("max_epochs", 100)),
            "seeds": seeds,
            "feature_normalization": "none",
            "split": "existing image-level 80/20",
            "all_comparators_retrained_on_same_cohort": True,
        },
        "comparisons": comparisons,
        "paired_bootstrap": bootstrap,
    }
    atomic_json_save(payload, output_dir / "results.json")
    (output_dir / f"{args.model}_union_topk_aggregate_s_report.md").write_text(
        report_markdown(args.model, payload), encoding="utf-8"
    )
    print(f"[Union-aggregate] wrote {output_dir}", flush=True)


def report_markdown(model: str, payload: Mapping[str, Any]) -> str:
    lines = [
        f"# {model}: Union-TopK aggregate S",
        "",
        "本实验在 `Top32(P_JFFN) ∪ Top32(Q_raw)` 内先做向量求和，再计算 "
        "`||ΣU J_f(z)a_j|| / ||ΣU a_j||`，因此保留视觉 token 间的方向抵消。",
        "",
        "## 曲线摘要",
        "",
        "| S | Hall>Real layers | Real>Hall layers | Strongest layer | Hall−Real | Cohen's d |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    names = {
        "old_aggregate_s": "all-token aggregate S",
        "union_topk_s": "Union tokenwise S",
        "union_aggregate_s": "Union aggregate S",
    }
    for key, name in names.items():
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
    for key in (
        "old_risk_ev",
        "old_risk_ev_old_aggregate_s",
        "old_risk_ev_token_all_s",
        "old_risk_ev_union_tokenwise_s",
        "union_aggregate_s",
    ):
        row = payload["comparisons"][key]
        aggregate = row["aggregate"]
        ensemble = row["seed_ensemble"]
        lines.append(
            f"| {row['name']} | {aggregate['auroc']['mean']:.6f} ± "
            f"{aggregate['auroc']['std']:.6f} | {aggregate['hall_f1']['mean']:.6f} | "
            f"{aggregate['hall_aupr']['mean']:.6f} | {ensemble['auroc']:.6f} | "
            f"{ensemble['hall_aupr']:.6f} |"
        )
    lines += ["", "## 图片级 paired bootstrap", ""]
    for key, name in (
        ("union_aggregate_s_versus_old_risk_ev", "vs risk+EV"),
        ("union_aggregate_s_versus_old_risk_ev_old_aggregate_s", "vs all-token aggregate S"),
        ("union_aggregate_s_versus_old_risk_ev_token_all_s", "vs all-token tokenwise S"),
        ("union_aggregate_s_versus_old_risk_ev_union_tokenwise_s", "vs Union tokenwise S"),
    ):
        row = payload["paired_bootstrap"][key]
        lines.append(
            f"- {name}：ensemble AUROC Δ={row['new_minus_baseline_auroc']:+.6f}，"
            f"95% CI [{row['auroc_ci95'][0]:+.6f},{row['auroc_ci95'][1]:+.6f}]；"
            f"Hall-AUPR Δ={row['new_minus_baseline_hall_aupr']:+.6f}，"
            f"95% CI [{row['hall_aupr_ci95'][0]:+.6f},{row['hall_aupr_ci95'][1]:+.6f}]。"
        )
    audit = payload["audit"]["aggregate_extraction"]
    lines += [
        "",
        f"Union 大小 min/mean/max={audit['union_size_minimum']}/"
        f"{audit['union_size_mean']:.2f}/{audit['union_size_maximum']}；"
        f"positions/mentions={audit['positions']}/{audit['mentions']}。",
        "",
    ]
    return "\n".join(lines)


def summarize(outputs_root: Path) -> None:
    payloads: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        path = (
            outputs_root / model / EXPERIMENT / DEFAULT_OUTPUT_SUBDIR / "results.json"
        )
        payload = json.loads(path.read_text())
        payloads[model] = payload
        for feature, result in payload["comparisons"].items():
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
        outputs_root / "union_topk_aggregate_s_mlp_2model_summary.json",
    )
    _write_csv(outputs_root / "union_topk_aggregate_s_mlp_2model_metrics.csv", rows)
    lines = [
        "# Union-TopK aggregate S：两模型汇总",
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
    (outputs_root / "union_topk_aggregate_s_mlp_2model_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("[Union-aggregate] wrote two-model summary", flush=True)


def main() -> None:
    args = parse_args()
    if args.model == "summarize":
        summarize(Path(args.outputs_root))
    else:
        train_model(args)


if __name__ == "__main__":
    main()
