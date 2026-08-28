#!/usr/bin/env python3
"""Plot S/S×risk curves and train [S×old-risk, EV] three-layer probes."""

from __future__ import annotations

import argparse
import csv
import json
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
    load_shards,
)
from scripts.train_old_risk_ev_s_mlp import (  # noqa: E402
    BASELINE_SPEC,
    EXPERIMENT,
    MODELS,
    aggregate_seed_metrics,
    align_baseline_predictions,
    ensemble_metrics,
    paired_image_bootstrap,
)
from scripts.train_torch_probe_feature_sets import (  # noqa: E402
    TorchProbeConfig,
    train_and_evaluate_probe,
)
from utils.config_utils import load_config  # noqa: E402
from utils.io_utils import load_json  # noqa: E402


FEATURE_NAME = "jacobian_S_times_old_hpre_risk_sqrt_plus_ev"
SCHEMA_VERSION = "s-times-risk-ev-mlp-v1"
PLUS_S_DIR = "results/jffn_second_round/old_risk_ev_s_mlp"


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


def _array(value: Any) -> np.ndarray:
    return torch.as_tensor(value).float().cpu().numpy().reshape(-1)


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


def collect_rows(
    model_root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray], dict[str, Any]]:
    """Collect official mentions and build [S*old-risk, EV] without normalization."""
    splits = load_json(str(model_root / "image_splits.json"))
    train_ids = {int(value) for value in splits["train"]}
    test_ids = {int(value) for value in splits["test"]}
    if train_ids & test_ids:
        raise AssertionError("Image split overlaps")

    names = ("x", "risk", "ev", "s", "s_times_risk")
    stores: dict[str, dict[str, list[Any]]] = {
        split: {**{name: [] for name in names}, "y": [], "image_ids": [], "mention_ids": []}
        for split in ("train", "test")
    }
    shard_dir = model_root / "results/jffn_p_comparison/shards"
    for shard in load_shards(shard_dir):
        positions = {str(row["target_key"]): row for row in shard["positions"]}
        for mention in shard["sample_table"]:
            image_id = int(mention["image_id"])
            split = "train" if image_id in train_ids else "test" if image_id in test_ids else None
            if split is None:
                raise AssertionError(f"Image {image_id} is outside the official split")
            position = positions[str(mention["target_key"])]
            risk = _array(
                position["risks"]["old_hpre_cos"]["hpre_raw_logit_gauss"]
                ["sqrt_matched_state"]
            )
            ev = _array(position["ev"]["hpre_raw_logit_gauss"])
            sensitivity = _array(position["jffn_diagnostics"]["gain"])
            if not (risk.shape == ev.shape == sensitivity.shape):
                raise AssertionError(
                    f"Layer mismatch for {mention['mention_id']}: "
                    f"{risk.shape}, {ev.shape}, {sensitivity.shape}"
                )
            interaction = sensitivity * risk
            x = np.concatenate((interaction, ev)).astype(np.float32, copy=False)
            if not all(
                np.isfinite(value).all()
                for value in (risk, ev, sensitivity, interaction, x)
            ):
                raise ValueError(f"Non-finite feature row {mention['mention_id']}")
            store = stores[split]
            store["x"].append(x)
            store["risk"].append(risk)
            store["ev"].append(ev)
            store["s"].append(sensitivity)
            store["s_times_risk"].append(interaction)
            store["y"].append(int(mention["label"]))
            store["image_ids"].append(image_id)
            store["mention_ids"].append(str(mention["mention_id"]))

    matrices: dict[str, dict[str, Any]] = {}
    for split, store in stores.items():
        matrices[split] = {
            name: np.stack(store[name]).astype(np.float32, copy=False)
            for name in names
        }
        matrices[split].update(
            {
                "y": np.asarray(store["y"], dtype=np.int32),
                "image_ids": np.asarray(store["image_ids"], dtype=np.int64),
                "mention_ids": list(store["mention_ids"]),
            }
        )
        if set(np.unique(matrices[split]["y"])) != {0, 1}:
            raise AssertionError(f"{split} does not contain both labels")

    curves = {
        name: np.concatenate((matrices["train"][name], matrices["test"][name]))
        for name in ("s", "s_times_risk")
    }
    curves["labels"] = np.concatenate((matrices["train"]["y"], matrices["test"]["y"]))
    layers = int(matrices["train"]["risk"].shape[1])
    audit = {
        "train_mentions": int(len(matrices["train"]["y"])),
        "test_mentions": int(len(matrices["test"]["y"])),
        "train_images": int(np.unique(matrices["train"]["image_ids"]).size),
        "test_images": int(np.unique(matrices["test"]["image_ids"]).size),
        "layers": layers,
        "input_dimensions": int(matrices["train"]["x"].shape[1]),
        "block_order": ["jacobian_S_times_old_hpre_risk_sqrt", "mass_x_cosine_EV"],
        "feature_normalization": "none",
        "all_finite": True,
    }
    for name in ("risk", "ev", "s", "s_times_risk"):
        values = matrices["train"][name].astype(np.float64)
        audit[f"{name}_train_stats"] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        }
    return matrices, curves, audit


def curve_statistics(
    curves: Mapping[str, np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return mention-level descriptive CIs and Hall-minus-Real effects."""
    labels = np.asarray(curves["labels"], dtype=np.int32)
    label_rows: list[dict[str, Any]] = []
    difference_rows: list[dict[str, Any]] = []
    for feature in ("s", "s_times_risk"):
        values = np.asarray(curves[feature], dtype=np.float64)
        for layer_index in range(values.shape[1]):
            by_label: dict[int, np.ndarray] = {}
            stats: dict[int, dict[str, float]] = {}
            for label, label_name in ((1, "REAL"), (0, "HALL")):
                selected = values[labels == label, layer_index]
                by_label[label] = selected
                std = float(selected.std(ddof=1))
                sem = std / np.sqrt(selected.size)
                stats[label] = {
                    "mean": float(selected.mean()),
                    "std": std,
                    "sem": float(sem),
                }
                label_rows.append(
                    {
                        "feature": feature,
                        "label": label_name,
                        "layer": layer_index + 1,
                        "n": int(selected.size),
                        "mean": stats[label]["mean"],
                        "std": std,
                        "sem": float(sem),
                        "ci95_low": float(stats[label]["mean"] - 1.96 * sem),
                        "ci95_high": float(stats[label]["mean"] + 1.96 * sem),
                    }
                )
            real = stats[1]
            hall = stats[0]
            difference = hall["mean"] - real["mean"]
            difference_sem = np.sqrt(
                hall["std"] ** 2 / by_label[0].size
                + real["std"] ** 2 / by_label[1].size
            )
            pooled_numerator = (
                (by_label[0].size - 1) * hall["std"] ** 2
                + (by_label[1].size - 1) * real["std"] ** 2
            )
            pooled_denominator = by_label[0].size + by_label[1].size - 2
            pooled_std = np.sqrt(pooled_numerator / pooled_denominator)
            auc = float(roc_auc_score(1 - labels, values[:, layer_index]))
            difference_rows.append(
                {
                    "feature": feature,
                    "layer": layer_index + 1,
                    "hall_minus_real": float(difference),
                    "difference_sem": float(difference_sem),
                    "difference_ci95_low": float(difference - 1.96 * difference_sem),
                    "difference_ci95_high": float(difference + 1.96 * difference_sem),
                    "cohens_d_hall_minus_real": float(difference / pooled_std),
                    "hall_positive_raw_auroc": auc,
                    "best_orientation_auroc": max(auc, 1.0 - auc),
                }
            )
    return label_rows, difference_rows


def plot_curves(
    *, model: str, label_rows: Sequence[Mapping[str, Any]], output_dir: Path
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.0), sharex=True)
    display = {"s": "Jacobian sensitivity S = R / I", "s_times_risk": "S × old OT risk"}
    colors = {"REAL": "#2563eb", "HALL": "#dc2626"}
    for axis, feature in zip(axes, ("s", "s_times_risk")):
        for label in ("REAL", "HALL"):
            rows = [
                row
                for row in label_rows
                if row["feature"] == feature and row["label"] == label
            ]
            x = np.asarray([row["layer"] for row in rows])
            mean = np.asarray([row["mean"] for row in rows])
            low = np.asarray([row["ci95_low"] for row in rows])
            high = np.asarray([row["ci95_high"] for row in rows])
            axis.plot(x, mean, label=label, color=colors[label], linewidth=2.0)
            axis.fill_between(x, low, high, color=colors[label], alpha=0.18)
        axis.set_title(display[feature])
        axis.set_ylabel("Mean feature value")
        axis.grid(alpha=0.25)
        axis.legend()
    axes[-1].set_xlabel("Decoder layer")
    fig.suptitle(f"{model}: REAL vs HALL layer curves (mention-level 95% CI)")
    fig.tight_layout()
    stem = output_dir / f"{model}_s_and_s_times_risk_real_hall_curves"
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_effect_curves(
    *, model: str, difference_rows: Sequence[Mapping[str, Any]], output_dir: Path
) -> None:
    """Plot Hall-minus-Real differences and standardized effect sizes."""
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 7.5), sharex=True)
    display = {"s": "Jacobian sensitivity S", "s_times_risk": "S × old OT risk"}
    for row_index, feature in enumerate(("s", "s_times_risk")):
        rows = [row for row in difference_rows if row["feature"] == feature]
        x = np.asarray([int(row["layer"]) for row in rows], dtype=np.int32)
        difference = np.asarray(
            [float(row["hall_minus_real"]) for row in rows], dtype=np.float64
        )
        low = np.asarray(
            [float(row["difference_ci95_low"]) for row in rows], dtype=np.float64
        )
        high = np.asarray(
            [float(row["difference_ci95_high"]) for row in rows], dtype=np.float64
        )
        effect = np.asarray(
            [float(row["cohens_d_hall_minus_real"]) for row in rows], dtype=np.float64
        )
        axes[row_index, 0].plot(x, difference, color="#7c3aed", linewidth=2.0)
        axes[row_index, 0].fill_between(x, low, high, color="#7c3aed", alpha=0.18)
        axes[row_index, 0].axhline(0.0, color="black", linewidth=1.0)
        axes[row_index, 0].set_title(f"{display[feature]}: HALL − REAL")
        axes[row_index, 0].set_ylabel("Mean difference")
        axes[row_index, 1].plot(x, effect, color="#059669", linewidth=2.0)
        axes[row_index, 1].axhline(0.0, color="black", linewidth=1.0)
        axes[row_index, 1].set_title(f"{display[feature]}: Cohen's d")
        axes[row_index, 1].set_ylabel("Standardized effect")
        for axis in axes[row_index]:
            axis.grid(alpha=0.25)
    for axis in axes[-1]:
        axis.set_xlabel("Decoder layer")
    fig.suptitle(f"{model}: layerwise HALL-vs-REAL feature separation")
    fig.tight_layout()
    stem = output_dir / f"{model}_s_and_s_times_risk_hall_minus_real_effects"
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def align_plus_s_predictions(
    model_root: Path, test: Mapping[str, Any]
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    result_dir = model_root / PLUS_S_DIR
    result = json.loads((result_dir / "results.json").read_text())
    progress = torch.load(
        result_dir / "training_progress.pt", map_location="cpu", weights_only=False
    )
    expected = list(test["mention_ids"])
    stored = list(progress["mention_ids"])
    lookup = {str(value): index for index, value in enumerate(stored)}
    if len(lookup) != len(stored):
        raise AssertionError("old risk+EV+S progress has duplicate mention IDs")
    indices = np.asarray([lookup[value] for value in expected], dtype=np.int64)
    labels = np.asarray(progress["labels"], dtype=np.int32)[indices]
    images = np.asarray(progress["image_ids"], dtype=np.int64)[indices]
    if not np.array_equal(labels, test["y"]) or not np.array_equal(
        images, test["image_ids"]
    ):
        raise AssertionError("old risk+EV+S predictions do not align")
    predictions = {
        seed: np.asarray(progress["probabilities"][seed], dtype=np.float32)[indices]
        for seed in (43, 44, 45)
    }
    return predictions, result["new"]


def _feature_curve_summary(
    difference_rows: Sequence[Mapping[str, Any]], feature: str
) -> dict[str, Any]:
    rows = [row for row in difference_rows if row["feature"] == feature]
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


def _report(model: str, payload: Mapping[str, Any]) -> str:
    comparisons = payload["comparisons"]
    new = payload["new"]
    lines = [
        f"# {model}: S、S×risk 曲线与 S×risk+EV 三层 MLP",
        "",
        "S 为逐层 Jacobian sensitivity R/I；S×risk 为与同层 old-hpre raw-logit "
        "Gaussian / sqrt-matched-state Union Top-K OT risk 的逐元素乘积。",
        "",
        "## 曲线摘要",
        "",
        "| Feature | Hall>Real layers | Real>Hall layers | Strongest layer | Hall−Real | Cohen's d |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for feature, display in (("s", "S"), ("s_times_risk", "S×risk")):
        row = payload["curve_summary"][feature]
        lines.append(
            f"| {display} | {row['hall_higher_layers']} | {row['real_higher_layers']} | "
            f"{row['strongest_layer']} | {row['strongest_hall_minus_real']:+.6f} | "
            f"{row['strongest_cohens_d']:+.4f} |"
        )
    lines += [
        "",
        "阴影为 mention-level 均值的近似 95% CI；用于描述曲线，不当作图片级独立显著性检验。",
        "",
        "## 三种子 MLP",
        "",
        "| Seed | AUROC | Real F1 | Hall F1 | Hall AUPR | Best epoch |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in new["seeds"]:
        lines.append(
            f"| {row['seed']} | {row['auc']:.6f} | {row['real_positive']['f1']:.6f} | "
            f"{row['hallucination_positive']['f1']:.6f} | "
            f"{row['hallucination_positive']['aupr']:.6f} | {row['best_epoch']} |"
        )
    lines += [
        "",
        "## 公平对照",
        "",
        "| Feature | Dimensions | Mean AUROC | Mean Hall AUPR | Ensemble AUROC | Ensemble Hall AUPR |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, display, dims in (
        ("old_risk_ev", "old risk+EV", 64),
        ("old_risk_ev_s", "old risk+EV+S", 96),
        ("s_times_risk_ev", "S×risk+EV", 64),
    ):
        row = comparisons[key]
        lines.append(
            f"| {display} | {dims} | {row['aggregate']['auroc']['mean']:.6f} ± "
            f"{row['aggregate']['auroc']['std']:.6f} | "
            f"{row['aggregate']['hall_aupr']['mean']:.6f} ± "
            f"{row['aggregate']['hall_aupr']['std']:.6f} | "
            f"{row['seed_ensemble']['auroc']:.6f} | "
            f"{row['seed_ensemble']['hall_aupr']:.6f} |"
        )
    for key, display in (
        ("versus_old_risk_ev", "S×risk+EV − old risk+EV"),
        ("versus_old_risk_ev_s", "S×risk+EV − old risk+EV+S"),
    ):
        boot = payload["paired_bootstrap"][key]
        lines += [
            "",
            f"{display} 的 seed-ensemble AUROC 差值 "
            f"{boot['new_minus_baseline_auroc']:+.6f}，图片级 95% CI "
            f"[{boot['auroc_ci95'][0]:+.6f},{boot['auroc_ci95'][1]:+.6f}]；"
            f"Hall-AUPR 差值 {boot['new_minus_baseline_hall_aupr']:+.6f}，95% CI "
            f"[{boot['hall_aupr_ci95'][0]:+.6f},{boot['hall_aupr_ci95'][1]:+.6f}]。",
        ]
    lines.append("")
    return "\n".join(lines)


def train_model(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    model_root = Path(args.outputs_root) / args.model / EXPERIMENT
    output_dir = model_root / "results/jffn_second_round/s_times_risk_ev_mlp"
    output_dir.mkdir(parents=True, exist_ok=True)
    matrices, curves, audit = collect_rows(model_root)
    label_rows, difference_rows = curve_statistics(curves)
    _write_csv(output_dir / "s_and_s_times_risk_label_curves.csv", label_rows)
    _write_csv(output_dir / "s_and_s_times_risk_hall_minus_real.csv", difference_rows)
    plot_curves(model=args.model, label_rows=label_rows, output_dir=output_dir)
    plot_effect_curves(
        model=args.model, difference_rows=difference_rows, output_dir=output_dir
    )

    train = matrices["train"]
    test = matrices["test"]
    baseline_predictions, baseline_result = align_baseline_predictions(model_root, test)
    plus_s_predictions, plus_s_result = align_plus_s_predictions(model_root, test)

    section = config.get("jffn_p_comparison") or {}
    comparison_training = section.get("training") or {}
    inherited = (config.get("training") or {}).get("torch_probe") or {}
    seeds = [int(value) for value in comparison_training.get("seeds", [43, 44, 45])]
    progress_path = output_dir / "training_progress.pt"
    completed: dict[int, dict[str, Any]] = {}
    probabilities: dict[int, np.ndarray] = {}
    if args.resume and progress_path.exists():
        progress = torch.load(progress_path, map_location="cpu", weights_only=False)
        if progress.get("schema_version") != SCHEMA_VERSION:
            raise AssertionError("Incompatible S×risk+EV resume artifact")
        completed = {int(row["seed"]): row for row in progress.get("seed_results", ())}
        probabilities = {
            int(seed): np.asarray(value, dtype=np.float32)
            for seed, value in (progress.get("probabilities") or {}).items()
        }

    device = _device(args.training_device)
    for seed in seeds:
        if seed in completed and seed in probabilities:
            print(f"[S*risk+EV] reuse {args.model} seed={seed}", flush=True)
            continue
        probe_cfg = TorchProbeConfig(
            hidden_sizes=tuple(
                int(value)
                for value in comparison_training.get(
                    "hidden_sizes", inherited.get("hidden_sizes", [128, 64, 32])
                )
            ),
            dropout=float(inherited.get("dropout", 0.3)),
            drop_last=bool(inherited.get("drop_last", True)),
            batch_size=int(comparison_training.get("batch_size", 256)),
            num_epochs=int(comparison_training.get("max_epochs", 100)),
            learning_rate=float(inherited.get("learning_rate", 1e-3)),
            weight_decay=float(inherited.get("weight_decay", 1e-5)),
            lr_factor=float(inherited.get("lr_factor", 0.5)),
            lr_patience=int(inherited.get("lr_patience", 5)),
            early_stopping_patience=int(inherited.get("early_stopping_patience", 10)),
            seed=seed,
            positive_class="real",
            split_protocol="strict_82_no_validation",
            threshold_selection="train_f1",
            fixed_threshold=float(inherited.get("fixed_threshold", 0.5)),
            checkpoint_selection="minimum_train_loss",
        )
        metrics = train_and_evaluate_probe(
            X_train=train["x"],
            y_train=train["y"],
            X_val=np.empty((0, train["x"].shape[1]), dtype=np.float32),
            y_val=np.empty((0,), dtype=np.int32),
            X_test=test["x"],
            y_test=test["y"],
            config=probe_cfg,
            device=device,
            output_dir=str(output_dir / "training" / f"seed_{seed}"),
            return_probabilities=True,
        )
        test_probabilities = np.asarray(metrics.pop("test_probabilities"), dtype=np.float32)
        metrics.pop("train_probabilities", None)
        metrics["seed"] = seed
        metrics["num_features"] = int(train["x"].shape[1])
        completed[seed] = metrics
        probabilities[seed] = test_probabilities
        atomic_torch_save(
            {
                "schema_version": SCHEMA_VERSION,
                "model": args.model,
                "seed_results": [completed[value] for value in sorted(completed)],
                "probabilities": probabilities,
                "labels": test["y"],
                "image_ids": test["image_ids"],
                "mention_ids": test["mention_ids"],
            },
            progress_path,
        )
        print(
            f"[S*risk+EV] {args.model} seed={seed} AUROC={metrics['auc']:.6f} "
            f"HallAUPR={metrics['hallucination_positive']['aupr']:.6f}",
            flush=True,
        )

    rows = [completed[seed] for seed in seeds]
    new_prob = np.mean([probabilities[seed] for seed in seeds], axis=0)
    baseline_prob = np.mean([baseline_predictions[seed] for seed in seeds], axis=0)
    plus_s_prob = np.mean([plus_s_predictions[seed] for seed in seeds], axis=0)
    new_result = {
        "name": FEATURE_NAME,
        "seeds": rows,
        "aggregate": aggregate_seed_metrics(rows),
        "seed_ensemble": ensemble_metrics(test["y"], new_prob),
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "feature": {
            "name": FEATURE_NAME,
            "definition": "[Jacobian_S_L * old_hpre_risk_sqrt_L, mass_x_cosine_EV_L]",
            "input_dimensions": int(train["x"].shape[1]),
        },
        "sample_audit": audit,
        "training_protocol": {
            "hidden_sizes": [128, 64, 32],
            "hidden_layer_count": 3,
            "batch_size": int(comparison_training.get("batch_size", 256)),
            "max_epochs": int(comparison_training.get("max_epochs", 100)),
            "seeds": seeds,
            "feature_normalization": "none",
            "checkpoint_selection": "minimum_train_loss",
            "threshold_selection": "train_f1",
            "split": "existing image-level 80/20",
        },
        "curve_summary": {
            feature: _feature_curve_summary(difference_rows, feature)
            for feature in ("s", "s_times_risk")
        },
        "new": new_result,
        "comparisons": {
            "old_risk_ev": {
                "aggregate": baseline_result["aggregate"],
                "seed_ensemble": ensemble_metrics(test["y"], baseline_prob),
            },
            "old_risk_ev_s": {
                "aggregate": plus_s_result["aggregate"],
                "seed_ensemble": ensemble_metrics(test["y"], plus_s_prob),
            },
            "s_times_risk_ev": {
                "aggregate": new_result["aggregate"],
                "seed_ensemble": new_result["seed_ensemble"],
            },
        },
        "paired_bootstrap": {
            "versus_old_risk_ev": paired_image_bootstrap(
                labels=test["y"], image_ids=test["image_ids"],
                new_probabilities=new_prob, baseline_probabilities=baseline_prob,
                replicates=args.bootstrap_replicates, seed=20260821,
            ),
            "versus_old_risk_ev_s": paired_image_bootstrap(
                labels=test["y"], image_ids=test["image_ids"],
                new_probabilities=new_prob, baseline_probabilities=plus_s_prob,
                replicates=args.bootstrap_replicates, seed=20260822,
            ),
        },
    }
    atomic_json_save(payload, output_dir / "results.json")
    (output_dir / f"{args.model}_s_times_risk_ev_mlp_report.md").write_text(
        _report(args.model, payload), encoding="utf-8"
    )
    print(f"[S*risk+EV] wrote {output_dir}", flush=True)


def summarize(outputs_root: Path) -> None:
    payloads: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []
    for model in MODELS:
        path = outputs_root / model / EXPERIMENT / (
            "results/jffn_second_round/s_times_risk_ev_mlp/results.json"
        )
        payloads[model] = json.loads(path.read_text())
        for feature in ("old_risk_ev", "old_risk_ev_s", "s_times_risk_ev"):
            row = payloads[model]["comparisons"][feature]
            csv_rows.append(
                {
                    "model": model,
                    "feature": feature,
                    "mean_auroc": row["aggregate"]["auroc"]["mean"],
                    "std_auroc": row["aggregate"]["auroc"]["std"],
                    "mean_hall_f1": row["aggregate"]["hall_f1"]["mean"],
                    "mean_hall_aupr": row["aggregate"]["hall_aupr"]["mean"],
                    "ensemble_auroc": row["seed_ensemble"]["auroc"],
                    "ensemble_hall_aupr": row["seed_ensemble"]["hall_aupr"],
                }
            )
    atomic_json_save(
        {"schema_version": SCHEMA_VERSION, "models": payloads},
        outputs_root / "s_times_risk_ev_mlp_2model_summary.json",
    )
    _write_csv(outputs_root / "s_times_risk_ev_mlp_2model_metrics.csv", csv_rows)
    lines = [
        "# S、S×risk 曲线与 S×risk+EV：两模型汇总",
        "",
        "S×risk 是 32 层逐元素乘积；与 32 层 EV 拼接后为 64 维三隐藏层 MLP。",
        "",
        "| Model | Feature | Mean AUROC | Mean Hall F1 | Mean Hall AUPR | Ensemble AUROC |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    labels = {
        "old_risk_ev": "old risk+EV",
        "old_risk_ev_s": "old risk+EV+S",
        "s_times_risk_ev": "S×risk+EV",
    }
    for row in csv_rows:
        lines.append(
            f"| {row['model']} | {labels[row['feature']]} | "
            f"{row['mean_auroc']:.6f} ± {row['std_auroc']:.6f} | "
            f"{row['mean_hall_f1']:.6f} | {row['mean_hall_aupr']:.6f} | "
            f"{row['ensemble_auroc']:.6f} |"
        )
    lines += ["", "## 图片级 paired bootstrap", ""]
    for model, payload in payloads.items():
        for key, display in (
            ("versus_old_risk_ev", "vs old risk+EV"),
            ("versus_old_risk_ev_s", "vs old risk+EV+S"),
        ):
            boot = payload["paired_bootstrap"][key]
            lines.append(
                f"- {model} {display}：AUROC Δ={boot['new_minus_baseline_auroc']:+.6f}，"
                f"95% CI [{boot['auroc_ci95'][0]:+.6f},{boot['auroc_ci95'][1]:+.6f}]；"
                f"Hall-AUPR Δ={boot['new_minus_baseline_hall_aupr']:+.6f}，"
                f"95% CI [{boot['hall_aupr_ci95'][0]:+.6f},{boot['hall_aupr_ci95'][1]:+.6f}]。"
            )
    (outputs_root / "s_times_risk_ev_mlp_2model_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("[S*risk+EV] wrote two-model summary", flush=True)


def main() -> None:
    args = parse_args()
    if args.model == "summarize":
        summarize(Path(args.outputs_root))
    else:
        train_model(args)


if __name__ == "__main__":
    main()
