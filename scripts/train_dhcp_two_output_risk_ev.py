#!/usr/bin/env python3
"""Compare one- versus two-logit heads under DHCP balanced sampling."""

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
from scripts.train_dhcp_sampler_risk_ev import (  # noqa: E402
    OUTPUT_SUBDIR as SINGLE_OUTPUT_SUBDIR,
    SCHEMA_VERSION as SINGLE_OUTPUT_SCHEMA_VERSION,
    dhcp_inverse_frequency_weights,
)
from scripts.train_old_risk_ev_s_mlp import EXPERIMENT, MODELS  # noqa: E402
from scripts.train_sqrt_hall_weight_risk_ev import (  # noqa: E402
    aggregate_fixed_threshold_metrics,
    aggregate_metrics,
    load_inputs,
)
from scripts.train_torch_probe_feature_sets import train_and_evaluate_probe  # noqa: E402
from scripts.train_union_topk_region_s_mlp import (  # noqa: E402
    _comparison_result,
    _probe_config,
)
from utils.config_utils import load_config  # noqa: E402


SCHEMA_VERSION = "dhcp-sampler-two-output-risk-ev-v1"
OUTPUT_SUBDIR = "results/jffn_second_round/dhcp_sampler_two_output_risk_ev"
SUMMARY_STEM = "dhcp_sampler_one_vs_two_output_risk_ev_2model"


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


def load_single_output_results(
    model_root: Path,
    test: Mapping[str, Any],
    seeds: Sequence[int],
) -> tuple[dict[int, dict[str, Any]], dict[int, np.ndarray]]:
    path = model_root / SINGLE_OUTPUT_SUBDIR / "training_progress.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != SINGLE_OUTPUT_SCHEMA_VERSION:
        raise AssertionError(f"Incompatible one-output DHCP artifact: {path}")
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
        raise AssertionError(f"Missing one-output DHCP seeds in {path}: {missing}")
    return rows, probabilities


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(output_dir: Path, model: str, comparisons: Mapping[str, Any]) -> None:
    specs = ("one_output_bce", "two_output_cross_entropy")
    display = ("1 output + BCE", "2 outputs + CE")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.0))
    for axis, (metric, title) in zip(
        axes, (("auroc", "AUROC"), ("hall_aupr", "Hall AUPR"), ("hall_f1", "Hall F1"))
    ):
        means = [comparisons[spec]["aggregate"][metric]["mean"] for spec in specs]
        stds = [comparisons[spec]["aggregate"][metric]["std"] for spec in specs]
        axis.barh(
            np.arange(2), means, xerr=stds, color=("#4C78A8", "#59A14F"), capsize=4
        )
        axis.set_yticks(np.arange(2), display)
        axis.set_xlim(0.0, 1.0)
        axis.invert_yaxis()
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.25)
    fig.suptitle(f"{model}: one vs two output logits under DHCP sampling")
    fig.tight_layout()
    stem = output_dir / f"{model}_dhcp_one_vs_two_output_metrics"
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _report(model: str, payload: Mapping[str, Any]) -> str:
    lines = [
        f"# {model}: DHCP sampler 下单输出与双输出头对照",
        "",
        "唯一训练变量是输出参数化：`Linear(32,1)+BCEWithLogitsLoss` 对比 "
        "`Linear(32,2)+CrossEntropyLoss`。两者共享 risk+EV、DHCP sampler、三层隐藏层、"
        "split、epoch 与 seeds。",
        "",
        "## 当前 train-REAL-F1 阈值",
        "",
        "| Head | AUROC | Hall AUPR | Hall precision | Hall recall | Hall F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for spec, display in (
        ("one_output_bce", "1 output + BCE"),
        ("two_output_cross_entropy", "2 outputs + CE"),
    ):
        agg = payload["comparisons"][spec]["aggregate"]
        lines.append(
            f"| {display} | {agg['auroc']['mean']:.6f} ± {agg['auroc']['std']:.6f} | "
            f"{agg['hall_aupr']['mean']:.6f} | {agg['hall_precision']['mean']:.6f} | "
            f"{agg['hall_recall']['mean']:.6f} | {agg['hall_f1']['mean']:.6f} |"
        )
    lines += [
        "",
        "## 固定 0.5 / 双输出 argmax",
        "",
        "| Head | Hall precision | Hall recall | Hall F1 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for spec, display in (
        ("one_output_bce", "1 output + BCE"),
        ("two_output_cross_entropy", "2 outputs + CE"),
    ):
        fixed = payload["fixed_threshold_comparisons"][spec]
        lines.append(
            f"| {display} | {fixed['hall_precision']['mean']:.6f} | "
            f"{fixed['hall_recall']['mean']:.6f} | {fixed['hall_f1']['mean']:.6f} |"
        )
    one = payload["comparisons"]["one_output_bce"]["aggregate"]
    two = payload["comparisons"]["two_output_cross_entropy"]["aggregate"]
    lines += [
        "",
        "## 双输出减单输出",
        "",
        f"- AUROC: `{two['auroc']['mean']-one['auroc']['mean']:+.6f}`。",
        f"- Hall AUPR: `{two['hall_aupr']['mean']-one['hall_aupr']['mean']:+.6f}`。",
        f"- Hall F1: `{two['hall_f1']['mean']-one['hall_f1']['mean']:+.6f}`。",
    ]
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
    one_rows, one_probs = load_single_output_results(model_root, test, seeds)

    progress_path = output_dir / "training_progress.pt"
    completed: dict[int, dict[str, Any]] = {}
    probabilities: dict[int, np.ndarray] = {}
    if args.resume and progress_path.exists():
        progress = torch.load(progress_path, map_location="cpu", weights_only=False)
        if progress.get("schema_version") != SCHEMA_VERSION:
            raise AssertionError("Incompatible two-output DHCP resume artifact")
        completed = {int(row["seed"]): row for row in progress.get("seed_results", ())}
        probabilities = {
            int(seed): np.asarray(value, dtype=np.float32)
            for seed, value in (progress.get("probabilities") or {}).items()
        }

    device = _device(args.training_device)
    for seed in seeds:
        if seed in completed and seed in probabilities:
            print(f"[DHCP-2out] reuse {args.model}/seed={seed}", flush=True)
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
            output_mode="two_logit_cross_entropy",
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
            f"[DHCP-2out] {args.model}/seed={seed} AUROC={metrics['auc']:.6f} "
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
        "one_output_bce": [one_rows[seed] for seed in seeds],
        "two_output_cross_entropy": [completed[seed] for seed in seeds],
    }
    probability_groups = {
        "one_output_bce": [one_probs[seed] for seed in seeds],
        "two_output_cross_entropy": [probabilities[seed] for seed in seeds],
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
            "all_finite": bool(np.isfinite(train["x"]).all() and np.isfinite(test["x"]).all()),
        },
        "training_protocol": {
            "controlled_variable": "output head and binary loss parameterization only",
            "shared_hidden_sizes": [128, 64, 32],
            "one_output": "Linear(32,1) + BCEWithLogitsLoss",
            "two_output": "Linear(32,2) + CrossEntropyLoss",
            "sampler": "DHCP inverse-frequency WeightedRandomSampler",
            "batch_size": int(training.get("batch_size", 256)),
            "max_epochs": int(training.get("max_epochs", 100)),
            "seeds": seeds,
            "feature_normalization": "none",
            "checkpoint_selection": "minimum_sampled_train_loss",
            "threshold_selection": "unweighted full-train REAL-F1",
            "fixed_0.5": "two-output argmax equivalent",
            "split": "existing image-level 80/20",
            "bootstrap": "not run",
        },
        "comparisons": comparisons,
        "fixed_threshold_comparisons": fixed_threshold_comparisons,
    }
    atomic_json_save(payload, output_dir / "results.json")
    seed_rows: list[dict[str, Any]] = []
    for head, rows in row_groups.items():
        for row in rows:
            hall = row["hallucination_positive"]
            fixed_hall = row["threshold_reports"]["fixed_0.5"]["test_metrics"][
                "hallucination_positive"
            ]
            seed_rows.append(
                {
                    "head": head,
                    "seed": row["seed"],
                    "auroc": row["auc"],
                    "hall_aupr": hall["aupr"],
                    "hall_precision": hall["precision"],
                    "hall_recall": hall["recall"],
                    "hall_f1": hall["f1"],
                    "fixed_0_5_hall_precision": fixed_hall["precision"],
                    "fixed_0_5_hall_recall": fixed_hall["recall"],
                    "fixed_0_5_hall_f1": fixed_hall["f1"],
                    "decision_threshold": row["decision_threshold"],
                    "best_epoch": row["best_epoch"],
                }
            )
    _write_csv(output_dir / "seed_metrics.csv", seed_rows)
    _plot(output_dir, args.model, comparisons)
    report_path = output_dir / f"{args.model}_dhcp_one_vs_two_output_report.md"
    report_path.write_text(_report(args.model, payload), encoding="utf-8")
    print(f"[DHCP-2out] wrote {output_dir}", flush=True)


def summarize(outputs_root: Path) -> None:
    models: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        path = outputs_root / model / EXPERIMENT / OUTPUT_SUBDIR / "results.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        models[model] = payload
        for spec in ("one_output_bce", "two_output_cross_entropy"):
            agg = payload["comparisons"][spec]["aggregate"]
            fixed = payload["fixed_threshold_comparisons"][spec]
            rows.append(
                {
                    "model": model,
                    "head": spec,
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
            "fixed_threshold_comparisons": {
                model: models[model]["fixed_threshold_comparisons"] for model in MODELS
            },
        },
        outputs_root / f"{SUMMARY_STEM}.json",
    )
    _write_csv(outputs_root / f"{SUMMARY_STEM}_metrics.csv", rows)
    lines = [
        "# DHCP sampler：risk+EV 单输出与双输出头对照",
        "",
        "只替换输出头和二分类 loss 参数化；两模型、三 seeds，不做 bootstrap。",
    ]
    for model in MODELS:
        lines += ["", _report(model, models[model]).rstrip()]
    (outputs_root / f"{SUMMARY_STEM}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"[DHCP-2out] wrote {outputs_root / (SUMMARY_STEM + '.md')}", flush=True)


def main() -> None:
    args = parse_args()
    if args.model == "summarize":
        summarize(Path(args.outputs_root))
    else:
        train_model(args)


if __name__ == "__main__":
    main()
