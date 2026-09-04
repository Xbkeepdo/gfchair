#!/usr/bin/env python3
"""Compare DHCP inverse-frequency sampling on the original risk+EV probe.

This is a controlled sampling ablation: features, image split, three-hidden-layer
probe, optimizer, epochs, and seeds remain unchanged.  Only the training loader
switches from ordinary shuffling to DHCP's inverse-class-frequency
WeightedRandomSampler with replacement.  The loss stays unweighted.
"""

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from features.jffn_experiment import atomic_json_save, atomic_torch_save  # noqa: E402
from scripts.train_old_risk_ev_s_mlp import EXPERIMENT, MODELS  # noqa: E402
from scripts.train_sqrt_hall_weight_risk_ev import (  # noqa: E402
    OUTPUT_SUBDIR as SQRT_OUTPUT_SUBDIR,
    SCHEMA_VERSION as SQRT_SCHEMA_VERSION,
    aggregate_fixed_threshold_metrics,
    aggregate_metrics,
    load_inputs,
    load_unweighted_baseline,
)
from scripts.train_torch_probe_feature_sets import train_and_evaluate_probe  # noqa: E402
from scripts.train_union_topk_region_s_mlp import (  # noqa: E402
    _comparison_result,
    _probe_config,
)
from utils.config_utils import load_config  # noqa: E402


SCHEMA_VERSION = "dhcp-sampler-risk-ev-v1"
OUTPUT_SUBDIR = "results/jffn_second_round/dhcp_sampler_risk_ev"
SUMMARY_STEM = "dhcp_sampler_risk_ev_2model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=(*MODELS, "summarize"))
    parser.add_argument(
        "--config", default="configs/model_configs_inslen_official_target.yaml"
    )
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--training-device", default="auto")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def dhcp_inverse_frequency_weights(
    labels: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the exact per-row weights used by DHCP's weighted sampler."""
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    unique = set(np.unique(labels).tolist())
    if unique != {0, 1}:
        raise ValueError(f"Expected binary labels {{0,1}}, got {sorted(unique)}")
    real_count = int(np.count_nonzero(labels == 1))
    hall_count = int(np.count_nonzero(labels == 0))
    class_weights = {1: 1.0 / real_count, 0: 1.0 / hall_count}
    weights = np.asarray([class_weights[int(label)] for label in labels], dtype=np.float64)
    class_mass = {
        "real": float(weights[labels == 1].sum()),
        "hall": float(weights[labels == 0].sum()),
    }
    return weights, {
        "real_label": 1,
        "hall_label": 0,
        "real_count": real_count,
        "hall_count": hall_count,
        "real_to_hall_ratio": float(real_count / hall_count),
        "real_row_sampling_weight": float(class_weights[1]),
        "hall_row_sampling_weight": float(class_weights[0]),
        "hall_to_real_row_weight_ratio": float(class_weights[0] / class_weights[1]),
        "class_sampling_mass": class_mass,
        "expected_class_fraction": {"real": 0.5, "hall": 0.5},
        "num_samples_per_epoch": int(labels.size),
        "replacement": True,
        "formula": "row_weight=1/N_class (training rows only)",
        "loss_weighting": "none",
    }


def load_sqrt_results(
    model_root: Path,
    test: Mapping[str, Any],
    seeds: Sequence[int],
) -> tuple[dict[int, dict[str, Any]], dict[int, np.ndarray]]:
    path = model_root / SQRT_OUTPUT_SUBDIR / "training_progress.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != SQRT_SCHEMA_VERSION:
        raise AssertionError(f"Incompatible sqrt-weight artifact: {path}")
    if list(payload["mention_ids"]) != list(test["mention_ids"]):
        raise AssertionError(f"Mention order mismatch while reusing {path}")
    if not np.array_equal(np.asarray(payload["labels"]), test["y"]):
        raise AssertionError(f"Label mismatch while reusing {path}")
    if not np.array_equal(np.asarray(payload["image_ids"]), test["image_ids"]):
        raise AssertionError(f"Image order mismatch while reusing {path}")
    rows = {int(row["seed"]): row for row in payload.get("seed_results", ())}
    probabilities = {
        int(seed): np.asarray(values, dtype=np.float32)
        for seed, values in (payload.get("probabilities") or {}).items()
    }
    missing = [seed for seed in seeds if seed not in rows or seed not in probabilities]
    if missing:
        raise AssertionError(f"Missing sqrt-weight seeds in {path}: {missing}")
    return rows, probabilities


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(output_dir: Path, model: str, comparisons: Mapping[str, Any]) -> None:
    specs = ("unweighted", "sqrt_hall_weight", "dhcp_sampler")
    display = ("Unweighted", "Sqrt loss weight", "DHCP sampler")
    colors = ("#A0A0A0", "#F2A541", "#4C78A8")
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.0))
    for axis, (metric, title) in zip(
        axes, (("auroc", "AUROC"), ("hall_aupr", "Hall AUPR"), ("hall_f1", "Hall F1"))
    ):
        means = [comparisons[spec]["aggregate"][metric]["mean"] for spec in specs]
        stds = [comparisons[spec]["aggregate"][metric]["std"] for spec in specs]
        axis.barh(np.arange(3), means, xerr=stds, color=colors, capsize=4)
        axis.set_yticks(np.arange(3), display)
        axis.set_xlim(0.0, 1.0)
        axis.invert_yaxis()
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.25)
    fig.suptitle(f"{model}: class-balancing ablation on original risk+EV")
    fig.tight_layout()
    stem = output_dir / f"{model}_dhcp_sampler_risk_ev_metrics"
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _report(model: str, payload: Mapping[str, Any]) -> str:
    lines = [
        f"# {model}: original risk+EV 的 DHCP 反频率采样消融",
        "",
        "这是只替换训练采样器的受控实验：特征、split、三层 MLP、优化器、epoch "
        "和 seeds 均不变。DHCP sampler 使用 `1/N_class`、有放回抽样，每个 epoch "
        "抽取与原训练集相同的行数；loss 不加权。",
        "",
        f"训练 REAL/HALL=`{payload['sampling']['real_count']}/"
        f"{payload['sampling']['hall_count']}`，HALL/REAL 单行抽样权重比=`"
        f"{payload['sampling']['hall_to_real_row_weight_ratio']:.6f}`。",
        "",
        "## 当前 train-REAL-F1 阈值",
        "",
        "| Training | AUROC | Hall AUPR | Hall precision | Hall recall | Hall F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for spec, display in (
        ("unweighted", "Unweighted"),
        ("sqrt_hall_weight", "Sqrt HALL loss weight"),
        ("dhcp_sampler", "DHCP balanced sampler"),
    ):
        agg = payload["comparisons"][spec]["aggregate"]
        lines.append(
            f"| {display} | {agg['auroc']['mean']:.6f} ± {agg['auroc']['std']:.6f} | "
            f"{agg['hall_aupr']['mean']:.6f} | {agg['hall_precision']['mean']:.6f} | "
            f"{agg['hall_recall']['mean']:.6f} | {agg['hall_f1']['mean']:.6f} |"
        )
    lines += [
        "",
        "## 固定 0.5（等价于二分类 argmax）",
        "",
        "| Training | Hall precision | Hall recall | Hall F1 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for spec, display in (
        ("unweighted", "Unweighted"),
        ("sqrt_hall_weight", "Sqrt HALL loss weight"),
        ("dhcp_sampler", "DHCP balanced sampler"),
    ):
        fixed = payload["fixed_threshold_comparisons"][spec]
        lines.append(
            f"| {display} | {fixed['hall_precision']['mean']:.6f} | "
            f"{fixed['hall_recall']['mean']:.6f} | {fixed['hall_f1']['mean']:.6f} |"
        )
    lines += ["", "## DHCP sampler 的增量", ""]
    dhcp = payload["comparisons"]["dhcp_sampler"]["aggregate"]
    for reference, label in (("unweighted", "unweighted"), ("sqrt_hall_weight", "sqrt")):
        base = payload["comparisons"][reference]["aggregate"]
        lines.append(
            f"- 相对 {label}: AUROC `{dhcp['auroc']['mean']-base['auroc']['mean']:+.6f}`，"
            f"Hall AUPR `{dhcp['hall_aupr']['mean']-base['hall_aupr']['mean']:+.6f}`，"
            f"Hall F1 `{dhcp['hall_f1']['mean']-base['hall_f1']['mean']:+.6f}`。"
        )
    return "\n".join(lines) + "\n"


def train_model(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    model_root = Path(args.outputs_root) / args.model / EXPERIMENT
    output_dir = model_root / OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    matrices = load_inputs(model_root, args.model)
    train, test = matrices["train"], matrices["test"]
    sampler_weights, sampling = dhcp_inverse_frequency_weights(train["y"])
    training = (config.get("jffn_p_comparison") or {}).get("training") or {}
    seeds = [int(value) for value in training.get("seeds", [43, 44, 45])]
    baseline_rows, baseline_probs = load_unweighted_baseline(model_root, test, seeds)
    sqrt_rows, sqrt_probs = load_sqrt_results(model_root, test, seeds)

    progress_path = output_dir / "training_progress.pt"
    completed: dict[int, dict[str, Any]] = {}
    probabilities: dict[int, np.ndarray] = {}
    if args.resume and progress_path.exists():
        progress = torch.load(progress_path, map_location="cpu", weights_only=False)
        if progress.get("schema_version") != SCHEMA_VERSION:
            raise AssertionError("Incompatible DHCP-sampler resume artifact")
        completed = {int(row["seed"]): row for row in progress.get("seed_results", ())}
        probabilities = {
            int(seed): np.asarray(value, dtype=np.float32)
            for seed, value in (progress.get("probabilities") or {}).items()
        }

    device = _device(args.training_device)
    for seed in seeds:
        if seed in completed and seed in probabilities:
            print(f"[DHCP-sampler] reuse {args.model}/seed={seed}", flush=True)
            continue
        metrics = train_and_evaluate_probe(
            X_train=train["x"],
            y_train=train["y"],
            X_val=np.empty((0, train["x"].shape[1]), dtype=np.float32),
            y_val=np.empty((0,), dtype=np.int32),
            X_test=test["x"],
            y_test=test["y"],
            config=_probe_config(config, seed),
            device=device,
            output_dir=str(output_dir / "training" / f"seed_{seed}"),
            return_probabilities=True,
            train_sampler_weights=sampler_weights,
        )
        test_prob = np.asarray(metrics.pop("test_probabilities"), dtype=np.float32)
        metrics.pop("train_probabilities", None)
        metrics["seed"] = seed
        metrics["num_features"] = int(train["x"].shape[1])
        metrics["sampling"] = sampling
        completed[seed] = metrics
        probabilities[seed] = test_prob
        atomic_torch_save(
            {
                "schema_version": SCHEMA_VERSION,
                "model": args.model,
                "seed_results": [completed[value] for value in sorted(completed)],
                "probabilities": probabilities,
                "labels": test["y"],
                "image_ids": test["image_ids"],
                "mention_ids": test["mention_ids"],
                "sampling": sampling,
            },
            progress_path,
        )
        hall = metrics["hallucination_positive"]
        print(
            f"[DHCP-sampler] {args.model}/seed={seed} AUROC={metrics['auc']:.6f} "
            f"HallP/R/F1={hall['precision']:.6f}/{hall['recall']:.6f}/{hall['f1']:.6f}",
            flush=True,
        )

    def comparison(
        name: str, rows: Sequence[Mapping[str, Any]], probs: Sequence[np.ndarray]
    ) -> dict[str, Any]:
        return _comparison_result(
            name=name,
            aggregate=aggregate_metrics(rows),
            labels=test["y"],
            probabilities=np.mean(probs, axis=0),
        )

    row_groups = {
        "unweighted": [baseline_rows[seed] for seed in seeds],
        "sqrt_hall_weight": [sqrt_rows[seed] for seed in seeds],
        "dhcp_sampler": [completed[seed] for seed in seeds],
    }
    probability_groups = {
        "unweighted": [baseline_probs[seed] for seed in seeds],
        "sqrt_hall_weight": [sqrt_probs[seed] for seed in seeds],
        "dhcp_sampler": [probabilities[seed] for seed in seeds],
    }
    comparisons = {
        spec: comparison(spec, row_groups[spec], probability_groups[spec])
        for spec in row_groups
    }
    fixed_threshold_comparisons = {
        spec: aggregate_fixed_threshold_metrics(rows)
        for spec, rows in row_groups.items()
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "feature": "old_hpre_cos risk + hpre_raw_logit_gauss mass_x_cosine EV",
        "feature_dimensions": int(train["x"].shape[1]),
        "sampling": sampling,
        "sample_audit": {
            "train_mentions": int(train["y"].shape[0]),
            "test_mentions": int(test["y"].shape[0]),
            "train_images": int(np.unique(train["image_ids"]).size),
            "test_images": int(np.unique(test["image_ids"]).size),
            "test_is_naturally_distributed": True,
            "all_finite": bool(np.isfinite(train["x"]).all() and np.isfinite(test["x"]).all()),
        },
        "training_protocol": {
            "controlled_variable": "training sampler only",
            "head": "current three-hidden-layer [128,64,32] DGST-style probe",
            "loss": "ordinary unweighted BCEWithLogitsLoss",
            "sampler": "DHCP inverse-frequency WeightedRandomSampler",
            "batch_size": int(training.get("batch_size", 256)),
            "max_epochs": int(training.get("max_epochs", 100)),
            "seeds": seeds,
            "feature_normalization": "none",
            "checkpoint_selection": "minimum_sampled_train_loss",
            "threshold_selection": "unweighted full-train REAL-F1",
            "fixed_0.5": "binary equivalent of official two-logit argmax",
            "split": "existing image-level 80/20",
            "bootstrap": "not run",
        },
        "comparisons": comparisons,
        "fixed_threshold_comparisons": fixed_threshold_comparisons,
    }
    atomic_json_save(payload, output_dir / "results.json")
    seed_rows: list[dict[str, Any]] = []
    for training_name, rows in row_groups.items():
        for row in rows:
            hall = row["hallucination_positive"]
            fixed_hall = row["threshold_reports"]["fixed_0.5"]["test_metrics"][
                "hallucination_positive"
            ]
            seed_rows.append(
                {
                    "training": training_name,
                    "seed": row["seed"],
                    "auroc": row["auc"],
                    "hall_aupr": hall["aupr"],
                    "hall_precision": hall["precision"],
                    "hall_recall": hall["recall"],
                    "hall_f1": hall["f1"],
                    "fixed_0_5_hall_precision": fixed_hall["precision"],
                    "fixed_0_5_hall_recall": fixed_hall["recall"],
                    "fixed_0_5_hall_f1": fixed_hall["f1"],
                    "accuracy": row["accuracy"],
                    "decision_threshold": row["decision_threshold"],
                    "best_epoch": row["best_epoch"],
                }
            )
    _write_csv(output_dir / "seed_metrics.csv", seed_rows)
    _plot(output_dir, args.model, comparisons)
    report_path = output_dir / f"{args.model}_dhcp_sampler_risk_ev_report.md"
    report_path.write_text(_report(args.model, payload), encoding="utf-8")
    print(f"[DHCP-sampler] wrote {output_dir}", flush=True)


def summarize(outputs_root: Path) -> None:
    models: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        path = outputs_root / model / EXPERIMENT / OUTPUT_SUBDIR / "results.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        models[model] = payload
        for spec in ("unweighted", "sqrt_hall_weight", "dhcp_sampler"):
            agg = payload["comparisons"][spec]["aggregate"]
            fixed = payload["fixed_threshold_comparisons"][spec]
            rows.append(
                {
                    "model": model,
                    "training": spec,
                    "hall_to_real_row_weight_ratio": payload["sampling"][
                        "hall_to_real_row_weight_ratio"
                    ],
                    "auroc_mean": agg["auroc"]["mean"],
                    "auroc_std": agg["auroc"]["std"],
                    "hall_aupr_mean": agg["hall_aupr"]["mean"],
                    "hall_precision_mean": agg["hall_precision"]["mean"],
                    "hall_recall_mean": agg["hall_recall"]["mean"],
                    "hall_f1_mean": agg["hall_f1"]["mean"],
                    "fixed_0_5_hall_precision_mean": fixed["hall_precision"]["mean"],
                    "fixed_0_5_hall_recall_mean": fixed["hall_recall"]["mean"],
                    "fixed_0_5_hall_f1_mean": fixed["hall_f1"]["mean"],
                }
            )
    atomic_json_save(
        {
            "schema_version": SCHEMA_VERSION,
            "models": {model: models[model]["comparisons"] for model in MODELS},
            "sampling": {model: models[model]["sampling"] for model in MODELS},
            "fixed_threshold_comparisons": {
                model: models[model]["fixed_threshold_comparisons"] for model in MODELS
            },
        },
        outputs_root / f"{SUMMARY_STEM}.json",
    )
    _write_csv(outputs_root / f"{SUMMARY_STEM}_metrics.csv", rows)
    lines = [
        "# original risk+EV：DHCP 反频率 sampler 两模型对照",
        "",
        "只替换训练采样器；相同特征、split、三层 MLP 和三 seeds，不做 bootstrap。",
    ]
    for model in MODELS:
        payload = models[model]
        lines += ["", _report(model, payload).rstrip()]
    (outputs_root / f"{SUMMARY_STEM}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"[DHCP-sampler] wrote {outputs_root / (SUMMARY_STEM + '.md')}", flush=True)


def main() -> None:
    args = parse_args()
    if args.model == "summarize":
        summarize(Path(args.outputs_root))
    else:
        train_model(args)


if __name__ == "__main__":
    main()
