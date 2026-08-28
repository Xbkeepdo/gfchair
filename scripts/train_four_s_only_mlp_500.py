#!/usr/bin/env python3
"""Train four 32-D Jacobian-S variants without risk or EV on one fair cohort."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping

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
    aggregate_seed_metrics,
    ensemble_metrics,
    paired_image_bootstrap,
)
from scripts.train_torch_probe_feature_sets import train_and_evaluate_probe  # noqa: E402
from scripts.train_union_topk_aggregate_s_mlp import (  # noqa: E402
    load_union_aggregate,
)
from scripts.train_union_topk_region_s_mlp import (  # noqa: E402
    _comparison_result,
    _probe_config,
    collect_matrices,
)
from utils.config_utils import load_config  # noqa: E402


SCHEMA_VERSION = "four-s-only-mlp-v2"
LEGACY_SCHEMA_VERSION = "four-s-only-mlp-500-v1"
MODEL = "llava_1_5_7b"
DEFAULT_AGGREGATE_SHARDS = "results/jffn_union_aggregate_500/shards"
DEFAULT_OUTPUT_SUBDIR = "results/jffn_second_round/four_s_only_mlp_500_fair"
DEFAULT_COHORT_LABEL = "500-image fair cohort"
DEFAULT_ARTIFACT_STEM = "llava_1_5_7b_four_s_only_mlp_500"
SPECS = (
    "all_aggregate_s",
    "all_tokenwise_s",
    "union_tokenwise_s",
    "union_aggregate_s",
)
DISPLAY = {
    "all_aggregate_s": "All-token aggregate S",
    "all_tokenwise_s": "All-token tokenwise S",
    "union_tokenwise_s": "Union-TopK tokenwise S",
    "union_aggregate_s": "Union-TopK aggregate S",
}
FORMULAS = {
    "all_aggregate_s": "||sum_j J_f(z)a_j|| / (||sum_j a_j|| + eps)",
    "all_tokenwise_s": "sum_j ||J_f(z)a_j|| / (sum_j ||a_j|| + eps)",
    "union_tokenwise_s": "sum_{j in U} ||J_f(z)a_j|| / (sum_{j in U} ||a_j|| + eps)",
    "union_aggregate_s": "||sum_{j in U} J_f(z)a_j|| / (||sum_{j in U} a_j|| + eps)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL, choices=(MODEL,))
    parser.add_argument(
        "--config", default="configs/model_configs_inslen_official_target.yaml"
    )
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--training-device", default="auto")
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--aggregate-shards-subdir", default=DEFAULT_AGGREGATE_SHARDS
    )
    parser.add_argument("--output-subdir", default=DEFAULT_OUTPUT_SUBDIR)
    parser.add_argument("--cohort-label", default=DEFAULT_COHORT_LABEL)
    parser.add_argument("--artifact-stem", default=DEFAULT_ARTIFACT_STEM)
    parser.add_argument(
        "--allow-subset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow aggregate shards to define a strict subset of the full cohort.",
    )
    return parser.parse_args()


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def build_s_only_features(
    matrices: Mapping[str, Mapping[str, Any]],
    aggregate: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, dict[str, np.ndarray]]:
    output: dict[str, dict[str, np.ndarray]] = {}
    for split in ("train", "test"):
        output[split] = {
            "all_aggregate_s": np.asarray(
                matrices[split]["old_aggregate_s"], dtype=np.float32
            ),
            "all_tokenwise_s": np.asarray(
                matrices[split]["token_all_s"], dtype=np.float32
            ),
            "union_tokenwise_s": np.asarray(
                matrices[split]["union_topk_s"], dtype=np.float32
            ),
            "union_aggregate_s": np.asarray(
                aggregate[split]["gain"], dtype=np.float32
            ),
        }
        rows = len(np.asarray(matrices[split]["y"]))
        for spec, values in output[split].items():
            if values.shape != (rows, 32):
                raise AssertionError(
                    f"{split}/{spec} expected {(rows, 32)}, got {values.shape}"
                )
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite values in {split}/{spec}")
    return output


def _plot_results(
    output_dir: Path,
    comparisons: Mapping[str, Any],
    *,
    cohort_label: str,
    artifact_stem: str,
) -> None:
    metrics = (
        ("auroc", "AUROC"),
        ("hall_aupr", "Hall AUPR"),
        ("hall_f1", "Hall F1"),
    )
    colors = ("#4C78A8", "#F58518", "#54A24B", "#E45756")
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8))
    x = np.arange(len(SPECS))
    for axis, (metric, title) in zip(axes, metrics):
        means = [comparisons[spec]["aggregate"][metric]["mean"] for spec in SPECS]
        stds = [comparisons[spec]["aggregate"][metric]["std"] for spec in SPECS]
        bars = axis.bar(x, means, yerr=stds, color=colors, capsize=4)
        axis.set_xticks(x, [DISPLAY[spec].replace(" ", "\n") for spec in SPECS])
        axis.set_ylim(0.0, 1.0)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, means):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.025,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    fig.suptitle(
        f"LLaVA: four standalone 32-D Jacobian-S probes ({cohort_label})"
    )
    fig.tight_layout()
    stem = output_dir / f"{artifact_stem}_metrics"
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _write_seed_csv(output_dir: Path, completed: Mapping[str, Mapping[int, Any]]) -> None:
    rows = []
    for spec in SPECS:
        for seed in sorted(completed[spec]):
            result = completed[spec][seed]
            rows.append(
                {
                    "feature": spec,
                    "display": DISPLAY[spec],
                    "seed": seed,
                    "input_dimensions": result["num_features"],
                    "auroc": result["auc"],
                    "accuracy": result["accuracy"],
                    "real_f1": result["real_positive"]["f1"],
                    "hall_f1": result["hallucination_positive"]["f1"],
                    "hall_aupr": result["hallucination_positive"]["aupr"],
                    "best_epoch": result["best_epoch"],
                }
            )
    with (output_dir / "seed_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _report(payload: Mapping[str, Any]) -> str:
    cohort_label = str(payload["cohort_label"])
    lines = [
        f"# LLaVA：四种 Jacobian S 单独训练（{cohort_label}）",
        "",
        "四个分类头均只输入 32 层 S，不包含 risk 或 EV；MLP、图片级 split、"
        "seeds 43/44/45 和训练协议完全一致。",
        "",
        "| S-only feature | Mean AUROC | Mean Hall F1 | Mean Hall AUPR | Ensemble AUROC | Ensemble Hall AUPR |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for spec in SPECS:
        row = payload["comparisons"][spec]
        aggregate = row["aggregate"]
        ensemble = row["seed_ensemble"]
        lines.append(
            f"| {DISPLAY[spec]} | {aggregate['auroc']['mean']:.6f} ± "
            f"{aggregate['auroc']['std']:.6f} | {aggregate['hall_f1']['mean']:.6f} | "
            f"{aggregate['hall_aupr']['mean']:.6f} | {ensemble['auroc']:.6f} | "
            f"{ensemble['hall_aupr']:.6f} |"
        )
    lines += ["", "## 两两 seed-ensemble 图片级 bootstrap", ""]
    for key, row in payload["paired_bootstrap"].items():
        left, right = key.split("__versus__")
        lines.append(
            f"- {DISPLAY[left]} − {DISPLAY[right]}：AUROC Δ="
            f"{row['new_minus_baseline_auroc']:+.6f}，95% CI "
            f"[{row['auroc_ci95'][0]:+.6f},{row['auroc_ci95'][1]:+.6f}]；"
            f"Hall-AUPR Δ={row['new_minus_baseline_hall_aupr']:+.6f}，95% CI "
            f"[{row['hall_aupr_ci95'][0]:+.6f},{row['hall_aupr_ci95'][1]:+.6f}]。"
        )
    audit = payload["sample_audit"]
    lines += [
        "",
        "## 协议核验",
        "",
        f"- train/test images：{audit['train_images']}/{audit['test_images']}。",
        f"- train/test mentions：{audit['train_mentions']}/{audit['test_mentions']}。",
        "- 每个输入均为 32 维；无特征标准化。",
        "- checkpoint 按 minimum train loss，阈值只用 train F1 选择。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model_root = Path(args.outputs_root) / args.model / EXPERIMENT
    output_dir = model_root / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    matrices, _curves, base_audit = collect_matrices(model_root, args.model)
    aggregate, aggregate_audit = load_union_aggregate(
        model_root / args.aggregate_shards_subdir,
        matrices,
        allow_subset=args.allow_subset,
    )
    feature_x = build_s_only_features(matrices, aggregate)
    train, test = matrices["train"], matrices["test"]
    section = (config.get("jffn_p_comparison") or {}).get("training") or {}
    seeds = [int(value) for value in section.get("seeds", [43, 44, 45])]
    progress_path = output_dir / "training_progress.pt"
    completed: dict[str, dict[int, dict[str, Any]]] = {spec: {} for spec in SPECS}
    probabilities: dict[str, dict[int, np.ndarray]] = {spec: {} for spec in SPECS}
    if args.resume and progress_path.exists():
        progress = torch.load(progress_path, map_location="cpu", weights_only=False)
        if progress.get("schema_version") not in {
            SCHEMA_VERSION,
            LEGACY_SCHEMA_VERSION,
        }:
            raise AssertionError("Incompatible four-S-only resume artifact")
        for spec in SPECS:
            completed[spec] = {
                int(row["seed"]): row
                for row in progress.get("seed_results", {}).get(spec, ())
            }
            probabilities[spec] = {
                int(seed): np.asarray(value, dtype=np.float32)
                for seed, value in progress.get("probabilities", {}).get(spec, {}).items()
            }

    device = _device(args.training_device)
    for spec in SPECS:
        for seed in seeds:
            if seed in completed[spec] and seed in probabilities[spec]:
                print(f"[S-only] reuse {spec}/seed={seed}", flush=True)
                continue
            metrics = train_and_evaluate_probe(
                X_train=feature_x["train"][spec],
                y_train=train["y"],
                X_val=np.empty((0, 32), dtype=np.float32),
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
            metrics["num_features"] = 32
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
                f"[S-only] {spec}/seed={seed} AUROC={metrics['auc']:.6f} "
                f"HallAUPR={metrics['hallucination_positive']['aupr']:.6f}",
                flush=True,
            )

    ensembles = {
        spec: np.mean([probabilities[spec][seed] for seed in seeds], axis=0)
        for spec in SPECS
    }
    comparisons = {
        spec: _comparison_result(
            name=DISPLAY[spec],
            aggregate=aggregate_seed_metrics([completed[spec][seed] for seed in seeds]),
            labels=test["y"],
            probabilities=ensembles[spec],
        )
        for spec in SPECS
    }
    bootstrap = {}
    for index, (left, right) in enumerate(combinations(SPECS, 2)):
        bootstrap[f"{left}__versus__{right}"] = paired_image_bootstrap(
            labels=test["y"],
            image_ids=test["image_ids"],
            new_probabilities=ensembles[left],
            baseline_probabilities=ensembles[right],
            replicates=args.bootstrap_replicates,
            seed=20260830 + index,
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "cohort_label": args.cohort_label,
        "definitions": FORMULAS,
        "sample_audit": {
            "train_images": int(np.unique(train["image_ids"]).size),
            "test_images": int(np.unique(test["image_ids"]).size),
            "train_mentions": int(len(train["y"])),
            "test_mentions": int(len(test["y"])),
            "test_real": int(np.sum(test["y"] == 1)),
            "test_hall": int(np.sum(test["y"] == 0)),
            "base": base_audit,
            "aggregate_extraction": aggregate_audit,
        },
        "training_protocol": {
            "inputs": {spec: f"S_32 only: {FORMULAS[spec]}" for spec in SPECS},
            "hidden_sizes": [128, 64, 32],
            "batch_size": int(section.get("batch_size", 256)),
            "max_epochs": int(section.get("max_epochs", 100)),
            "seeds": seeds,
            "feature_normalization": "none",
            "checkpoint_selection": "minimum_train_loss",
            "threshold_selection": "train_f1",
            "split": args.cohort_label,
        },
        "comparisons": comparisons,
        "paired_bootstrap": bootstrap,
    }
    atomic_json_save(payload, output_dir / "results.json")
    _write_seed_csv(output_dir, completed)
    _plot_results(
        output_dir,
        comparisons,
        cohort_label=args.cohort_label,
        artifact_stem=args.artifact_stem,
    )
    (output_dir / f"{args.artifact_stem}_report.md").write_text(
        _report(payload), encoding="utf-8"
    )
    print(f"[S-only] wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
