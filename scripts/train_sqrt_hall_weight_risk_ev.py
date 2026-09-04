#!/usr/bin/env python3
"""Compare unweighted and sqrt-HALL-sample-weighted original risk+EV probes."""

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from features.jffn_experiment import atomic_json_save, atomic_torch_save  # noqa: E402
from scripts.train_old_risk_ev_s_mlp import (  # noqa: E402
    EXPERIMENT,
    MODELS,
    aggregate_seed_metrics,
)
from scripts.train_torch_probe_feature_sets import train_and_evaluate_probe  # noqa: E402
from scripts.train_union_topk_irs_ablation import (  # noqa: E402
    CACHE_SCHEMA_VERSION,
    OUTPUT_SUBDIR as CACHE_OUTPUT_SUBDIR,
)
from scripts.train_union_topk_region_s_mlp import (  # noqa: E402
    _comparison_result,
    _probe_config,
)
from utils.config_utils import load_config  # noqa: E402


SCHEMA_VERSION = "sqrt-hall-sample-weight-risk-ev-v1"
OUTPUT_SUBDIR = "results/jffn_second_round/sqrt_hall_weight_risk_ev"
SUMMARY_STEM = "sqrt_hall_weight_risk_ev_2model"
BASELINE_PROGRESS = (
    "results/jffn_second_round/union_topk_irs_ablation/training_progress.pt"
)
BASELINE_SPEC = "old_risk_ev"


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


def sqrt_hall_sample_weights(labels: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    unique = set(np.unique(labels).tolist())
    if unique != {0, 1}:
        raise ValueError(f"Expected binary labels {{0,1}}, got {sorted(unique)}")
    real_count = int(np.count_nonzero(labels == 1))
    hall_count = int(np.count_nonzero(labels == 0))
    hall_weight = math.sqrt(real_count / hall_count)
    weights = np.where(labels == 0, hall_weight, 1.0).astype(np.float32)
    return weights, {
        "real_label": 1,
        "hall_label": 0,
        "real_count": real_count,
        "hall_count": hall_count,
        "real_to_hall_ratio": float(real_count / hall_count),
        "real_sample_weight": 1.0,
        "hall_sample_weight": float(hall_weight),
        "formula": "sqrt(N_REAL/N_HALL), computed from training rows only",
    }


def load_inputs(model_root: Path, model: str) -> dict[str, dict[str, Any]]:
    cache_path = model_root / CACHE_OUTPUT_SUBDIR / "feature_matrix_cache.pt"
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise AssertionError(f"Incompatible cache schema in {cache_path}")
    if payload.get("model") != model:
        raise AssertionError(f"Cache model mismatch in {cache_path}")
    output: dict[str, dict[str, Any]] = {}
    for split in ("train", "test"):
        source = payload["matrices"][split]
        x = np.asarray(source["x"][BASELINE_SPEC], dtype=np.float32)
        if x.ndim != 2 or not np.isfinite(x).all():
            raise ValueError(f"Invalid risk+EV matrix in {cache_path}/{split}")
        output[split] = {
            "x": np.ascontiguousarray(x),
            "y": np.asarray(source["y"], dtype=np.int32),
            "image_ids": np.asarray(source["image_ids"], dtype=np.int64),
            "mention_ids": list(source["mention_ids"]),
        }
    return output


def load_unweighted_baseline(
    model_root: Path,
    test: Mapping[str, Any],
    seeds: Sequence[int],
) -> tuple[dict[int, dict[str, Any]], dict[int, np.ndarray]]:
    path = model_root / BASELINE_PROGRESS
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if list(payload["mention_ids"]) != list(test["mention_ids"]):
        raise AssertionError(f"Mention order mismatch while reusing {path}")
    if not np.array_equal(np.asarray(payload["labels"]), test["y"]):
        raise AssertionError(f"Label mismatch while reusing {path}")
    if not np.array_equal(np.asarray(payload["image_ids"]), test["image_ids"]):
        raise AssertionError(f"Image order mismatch while reusing {path}")
    rows = {
        int(row["seed"]): row
        for row in (payload.get("seed_results") or {}).get(BASELINE_SPEC, ())
    }
    probabilities = {
        int(seed): np.asarray(values, dtype=np.float32)
        for seed, values in (payload.get("probabilities") or {})
        .get(BASELINE_SPEC, {})
        .items()
    }
    missing = [seed for seed in seeds if seed not in rows or seed not in probabilities]
    if missing:
        raise AssertionError(f"Missing unweighted baseline seeds in {path}: {missing}")
    return rows, probabilities


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output = aggregate_seed_metrics(rows)
    for name, key in (
        ("hall_precision", "precision"),
        ("hall_recall", "recall"),
    ):
        values = np.asarray(
            [float(row["hallucination_positive"][key]) for row in rows],
            dtype=np.float64,
        )
        output[name] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "values": values.tolist(),
        }
    return output


def aggregate_fixed_threshold_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, path in (
        ("hall_precision", ("hallucination_positive", "precision")),
        ("hall_recall", ("hallucination_positive", "recall")),
        ("hall_f1", ("hallucination_positive", "f1")),
        ("real_f1", ("real_positive", "f1")),
        ("accuracy", ("accuracy",)),
    ):
        values = []
        for row in rows:
            value: Any = row["threshold_reports"]["fixed_0.5"]["test_metrics"]
            for key in path:
                value = value[key]
            values.append(float(value))
        array = np.asarray(values, dtype=np.float64)
        output[name] = {
            "mean": float(array.mean()),
            "std": float(array.std()),
            "values": array.tolist(),
        }
    return output


def _plot(output_dir: Path, model: str, comparisons: Mapping[str, Any]) -> None:
    specs = ("unweighted", "sqrt_hall_weight")
    display = ("Unweighted risk+EV", "Sqrt-HALL-weighted risk+EV")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.0))
    for axis, (metric, title) in zip(
        axes, (("auroc", "AUROC"), ("hall_aupr", "Hall AUPR"), ("hall_f1", "Hall F1"))
    ):
        means = [comparisons[spec]["aggregate"][metric]["mean"] for spec in specs]
        stds = [comparisons[spec]["aggregate"][metric]["std"] for spec in specs]
        axis.barh(np.arange(2), means, xerr=stds, color=("#A0A0A0", "#E45756"), capsize=4)
        axis.set_yticks(np.arange(2), display)
        axis.set_xlim(0.0, 1.0)
        axis.invert_yaxis()
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.25)
    fig.suptitle(f"{model}: sqrt HALL sample weighting on original risk+EV")
    fig.tight_layout()
    stem = output_dir / f"{model}_sqrt_hall_weight_risk_ev_metrics"
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _report(model: str, payload: Mapping[str, Any]) -> str:
    lines = [
        f"# {model}: original risk+EV 平方根 HALL sample weight",
        "",
        f"训练集 HALL 权重=`{payload['weighting']['hall_sample_weight']:.6f}`，"
        "REAL 权重=`1.0`；测试集不加权。",
        "",
        "| Training | AUROC | Hall AUPR | Hall precision | Hall recall | Hall F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for spec, display in (
        ("unweighted", "Unweighted"),
        ("sqrt_hall_weight", "Sqrt HALL sample weight"),
    ):
        agg = payload["comparisons"][spec]["aggregate"]
        lines.append(
            f"| {display} | {agg['auroc']['mean']:.6f} ± {agg['auroc']['std']:.6f} | "
            f"{agg['hall_aupr']['mean']:.6f} | {agg['hall_precision']['mean']:.6f} | "
            f"{agg['hall_recall']['mean']:.6f} | {agg['hall_f1']['mean']:.6f} |"
        )
    weighted = payload["comparisons"]["sqrt_hall_weight"]["aggregate"]
    baseline = payload["comparisons"]["unweighted"]["aggregate"]
    lines += ["", "## 加权减无权重", ""]
    for metric, label in (
        ("auroc", "AUROC"),
        ("hall_aupr", "Hall AUPR"),
        ("hall_precision", "Hall precision"),
        ("hall_recall", "Hall recall"),
        ("hall_f1", "Hall F1"),
    ):
        lines.append(
            f"- {label}: {weighted[metric]['mean']-baseline[metric]['mean']:+.6f}。"
        )
    lines += [
        "",
        "## 固定 0.5 阈值补充结果",
        "",
        "| Training | Hall precision | Hall recall | Hall F1 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for spec, display in (
        ("unweighted", "Unweighted"),
        ("sqrt_hall_weight", "Sqrt HALL sample weight"),
    ):
        fixed = payload["fixed_threshold_comparisons"][spec]
        lines.append(
            f"| {display} | {fixed['hall_precision']['mean']:.6f} | "
            f"{fixed['hall_recall']['mean']:.6f} | {fixed['hall_f1']['mean']:.6f} |"
        )
    return "\n".join(lines) + "\n"


def train_model(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    model_root = Path(args.outputs_root) / args.model / EXPERIMENT
    output_dir = model_root / OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    matrices = load_inputs(model_root, args.model)
    train, test = matrices["train"], matrices["test"]
    weights, weighting = sqrt_hall_sample_weights(train["y"])
    training = (config.get("jffn_p_comparison") or {}).get("training") or {}
    seeds = [int(value) for value in training.get("seeds", [43, 44, 45])]
    baseline_rows, baseline_probs = load_unweighted_baseline(model_root, test, seeds)

    progress_path = output_dir / "training_progress.pt"
    completed: dict[int, dict[str, Any]] = {}
    probabilities: dict[int, np.ndarray] = {}
    if args.resume and progress_path.exists():
        progress = torch.load(progress_path, map_location="cpu", weights_only=False)
        if progress.get("schema_version") != SCHEMA_VERSION:
            raise AssertionError("Incompatible sqrt-HALL-weight resume artifact")
        completed = {int(row["seed"]): row for row in progress.get("seed_results", ())}
        probabilities = {
            int(seed): np.asarray(value, dtype=np.float32)
            for seed, value in (progress.get("probabilities") or {}).items()
        }

    device = _device(args.training_device)
    for seed in seeds:
        if seed in completed and seed in probabilities:
            print(f"[Sqrt-HALL] reuse {args.model}/seed={seed}", flush=True)
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
            train_sample_weights=weights,
        )
        test_prob = np.asarray(metrics.pop("test_probabilities"), dtype=np.float32)
        metrics.pop("train_probabilities", None)
        metrics["seed"] = seed
        metrics["num_features"] = int(train["x"].shape[1])
        metrics["weighting"] = weighting
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
                "weighting": weighting,
            },
            progress_path,
        )
        print(
            f"[Sqrt-HALL] {args.model}/seed={seed} AUROC={metrics['auc']:.6f} "
            f"HallP/R/F1={metrics['hallucination_positive']['precision']:.6f}/"
            f"{metrics['hallucination_positive']['recall']:.6f}/"
            f"{metrics['hallucination_positive']['f1']:.6f}",
            flush=True,
        )

    def comparison(name: str, rows: Sequence[Mapping[str, Any]], probs: Sequence[np.ndarray]) -> dict[str, Any]:
        return _comparison_result(
            name=name,
            aggregate=aggregate_metrics(rows),
            labels=test["y"],
            probabilities=np.mean(probs, axis=0),
        )

    comparisons = {
        "unweighted": comparison(
            "Unweighted original risk+EV",
            [baseline_rows[seed] for seed in seeds],
            [baseline_probs[seed] for seed in seeds],
        ),
        "sqrt_hall_weight": comparison(
            "Sqrt-HALL-weighted original risk+EV",
            [completed[seed] for seed in seeds],
            [probabilities[seed] for seed in seeds],
        ),
    }
    fixed_threshold_comparisons = {
        "unweighted": aggregate_fixed_threshold_metrics(
            [baseline_rows[seed] for seed in seeds]
        ),
        "sqrt_hall_weight": aggregate_fixed_threshold_metrics(
            [completed[seed] for seed in seeds]
        ),
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "feature": "old_hpre_cos risk + hpre_raw_logit_gauss mass_x_cosine EV",
        "feature_dimensions": int(train["x"].shape[1]),
        "weighting": weighting,
        "sample_audit": {
            "train_mentions": int(train["y"].shape[0]),
            "test_mentions": int(test["y"].shape[0]),
            "train_images": int(np.unique(train["image_ids"]).size),
            "test_images": int(np.unique(test["image_ids"]).size),
            "test_is_unweighted": True,
            "all_finite": bool(np.isfinite(train["x"]).all() and np.isfinite(test["x"]).all()),
        },
        "training_protocol": {
            "hidden_sizes": [128, 64, 32],
            "batch_size": int(training.get("batch_size", 256)),
            "max_epochs": int(training.get("max_epochs", 100)),
            "seeds": seeds,
            "feature_normalization": "none",
            "checkpoint_selection": "minimum_weighted_train_loss",
            "threshold_selection": "unweighted_train_real_f1",
            "split": "existing image-level 80/20",
            "bootstrap": "not run",
        },
        "comparisons": comparisons,
        "fixed_threshold_comparisons": fixed_threshold_comparisons,
    }
    atomic_json_save(payload, output_dir / "results.json")
    seed_rows: list[dict[str, Any]] = []
    for training_name, source in (
        ("unweighted", baseline_rows),
        ("sqrt_hall_weight", completed),
    ):
        for seed in seeds:
            row = source[seed]
            hall = row["hallucination_positive"]
            fixed_hall = row["threshold_reports"]["fixed_0.5"]["test_metrics"][
                "hallucination_positive"
            ]
            seed_rows.append(
                {
                    "training": training_name,
                    "seed": seed,
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
    (output_dir / f"{args.model}_sqrt_hall_weight_risk_ev_report.md").write_text(
        _report(args.model, payload), encoding="utf-8"
    )
    print(f"[Sqrt-HALL] wrote {output_dir}", flush=True)


def summarize(outputs_root: Path) -> None:
    models: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        path = outputs_root / model / EXPERIMENT / OUTPUT_SUBDIR / "results.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        models[model] = payload
        for spec in ("unweighted", "sqrt_hall_weight"):
            agg = payload["comparisons"][spec]["aggregate"]
            rows.append(
                {
                    "model": model,
                    "training": spec,
                    "hall_sample_weight": payload["weighting"]["hall_sample_weight"],
                    "auroc_mean": agg["auroc"]["mean"],
                    "auroc_std": agg["auroc"]["std"],
                    "hall_aupr_mean": agg["hall_aupr"]["mean"],
                    "hall_precision_mean": agg["hall_precision"]["mean"],
                    "hall_recall_mean": agg["hall_recall"]["mean"],
                    "hall_f1_mean": agg["hall_f1"]["mean"],
                }
            )
    atomic_json_save(
        {
            "schema_version": SCHEMA_VERSION,
            "models": {model: models[model]["comparisons"] for model in MODELS},
            "weighting": {model: models[model]["weighting"] for model in MODELS},
            "fixed_threshold_comparisons": {
                model: models[model]["fixed_threshold_comparisons"]
                for model in MODELS
            },
        },
        outputs_root / f"{SUMMARY_STEM}.json",
    )
    _write_csv(outputs_root / f"{SUMMARY_STEM}_metrics.csv", rows)
    lines = [
        "# original risk+EV：平方根 HALL sample weight 两模型对照",
        "",
        "HALL 权重只由训练集计算，测试集不加权；三 seeds、相同 MLP/split，不做 bootstrap。",
    ]
    for model in MODELS:
        lines += [
            "",
            f"## {model}（HALL weight={models[model]['weighting']['hall_sample_weight']:.6f}）",
            "",
            "| Training | AUROC | Hall AUPR | Hall precision | Hall recall | Hall F1 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for spec, display in (
            ("unweighted", "Unweighted"),
            ("sqrt_hall_weight", "Sqrt HALL weight"),
        ):
            agg = models[model]["comparisons"][spec]["aggregate"]
            lines.append(
                f"| {display} | {agg['auroc']['mean']:.6f} ± {agg['auroc']['std']:.6f} | "
                f"{agg['hall_aupr']['mean']:.6f} | {agg['hall_precision']['mean']:.6f} | "
                f"{agg['hall_recall']['mean']:.6f} | {agg['hall_f1']['mean']:.6f} |"
            )
        lines += [
            "",
            "固定 0.5 阈值：",
            "",
            "| Training | Hall precision | Hall recall | Hall F1 |",
            "| --- | ---: | ---: | ---: |",
        ]
        for spec, display in (
            ("unweighted", "Unweighted"),
            ("sqrt_hall_weight", "Sqrt HALL weight"),
        ):
            fixed = models[model]["fixed_threshold_comparisons"][spec]
            lines.append(
                f"| {display} | {fixed['hall_precision']['mean']:.6f} | "
                f"{fixed['hall_recall']['mean']:.6f} | {fixed['hall_f1']['mean']:.6f} |"
            )
    (outputs_root / f"{SUMMARY_STEM}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[Sqrt-HALL] wrote {outputs_root / (SUMMARY_STEM + '.md')}", flush=True)


def main() -> None:
    args = parse_args()
    if args.model == "summarize":
        summarize(Path(args.outputs_root))
    else:
        train_model(args)


if __name__ == "__main__":
    main()
