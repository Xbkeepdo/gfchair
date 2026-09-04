#!/usr/bin/env python3
"""Plot aggregate JFFN I/R curves and train controlled I/R feature ablations."""

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

from features.jffn_experiment import (  # noqa: E402
    atomic_json_save,
    atomic_torch_save,
    load_shards,
)
from scripts.train_old_risk_ev_s_mlp import (  # noqa: E402
    EXPERIMENT,
    MODELS,
    aggregate_seed_metrics,
    ensemble_metrics,
    paired_image_bootstrap,
)
from scripts.train_torch_probe_feature_sets import train_and_evaluate_probe  # noqa: E402
from scripts.train_union_topk_region_s_mlp import (  # noqa: E402
    _comparison_result,
    _probe_config,
)
from utils.config_utils import load_config  # noqa: E402
from utils.io_utils import load_json  # noqa: E402


SCHEMA_VERSION = "jffn-i-r-feature-ablation-v1"
OUTPUT_SUBDIR = "results/jffn_second_round/i_r_feature_ablation"
SOURCE_LABELS = {
    "old": "Original old-hpre cosine-P risk",
    "jffn": "P_JFFN risk",
}
SPECS = (
    "i_only",
    "r_only",
    "i_r_only",
    "old_risk_ev",
    "old_risk_ev_i",
    "old_risk_ev_r",
    "old_risk_ev_i_r",
    "jffn_risk_ev",
    "jffn_risk_ev_i",
    "jffn_risk_ev_r",
    "jffn_risk_ev_i_r",
)
DISPLAY = {
    "i_only": "I only",
    "r_only": "R only",
    "i_r_only": "I + R",
    "old_risk_ev": "Original risk + EV",
    "old_risk_ev_i": "Original risk + EV + I",
    "old_risk_ev_r": "Original risk + EV + R",
    "old_risk_ev_i_r": "Original risk + EV + I + R",
    "jffn_risk_ev": "P_JFFN risk + EV",
    "jffn_risk_ev_i": "P_JFFN risk + EV + I",
    "jffn_risk_ev_r": "P_JFFN risk + EV + R",
    "jffn_risk_ev_i_r": "P_JFFN risk + EV + I + R",
}
BOOTSTRAP_PAIRS = (
    ("r_only", "i_only"),
    ("i_r_only", "i_only"),
    ("old_risk_ev_i", "old_risk_ev"),
    ("old_risk_ev_r", "old_risk_ev"),
    ("old_risk_ev_i_r", "old_risk_ev"),
    ("jffn_risk_ev_i", "jffn_risk_ev"),
    ("jffn_risk_ev_r", "jffn_risk_ev"),
    ("jffn_risk_ev_i_r", "jffn_risk_ev"),
    ("jffn_risk_ev", "old_risk_ev"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODELS)
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
    return np.asarray(torch.as_tensor(value).float().cpu(), dtype=np.float32).reshape(-1)


def build_feature_sets(blocks: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Construct the exact standalone and risk/EV concatenation ablations."""
    required = ("i", "r", "old_risk", "jffn_risk", "ev")
    arrays = {name: np.asarray(blocks[name], dtype=np.float32) for name in required}
    shapes = {name: value.shape for name, value in arrays.items()}
    if len(set(shapes.values())) != 1 or arrays["i"].ndim != 2:
        raise ValueError(f"Expected five matching [rows,layers] blocks, got {shapes}")
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("I/R/risk/EV blocks must all be finite")
    i, r = arrays["i"], arrays["r"]
    old, jffn, ev = arrays["old_risk"], arrays["jffn_risk"], arrays["ev"]
    output = {
        "i_only": i,
        "r_only": r,
        "i_r_only": np.concatenate((i, r), axis=1),
        "old_risk_ev": np.concatenate((old, ev), axis=1),
        "old_risk_ev_i": np.concatenate((old, ev, i), axis=1),
        "old_risk_ev_r": np.concatenate((old, ev, r), axis=1),
        "old_risk_ev_i_r": np.concatenate((old, ev, i, r), axis=1),
        "jffn_risk_ev": np.concatenate((jffn, ev), axis=1),
        "jffn_risk_ev_i": np.concatenate((jffn, ev, i), axis=1),
        "jffn_risk_ev_r": np.concatenate((jffn, ev, r), axis=1),
        "jffn_risk_ev_i_r": np.concatenate((jffn, ev, i, r), axis=1),
    }
    return {name: np.ascontiguousarray(output[name]) for name in SPECS}


def collect_matrices(model_root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    split_payload = load_json(str(model_root / "image_splits.json"))
    train_ids = {int(value) for value in split_payload["train"]}
    test_ids = {int(value) for value in split_payload["test"]}
    if train_ids & test_ids:
        raise AssertionError("Official image split overlaps")

    names = ("i", "r", "old_risk", "jffn_risk", "ev")
    stores: dict[str, dict[str, list[Any]]] = {
        split: {
            **{name: [] for name in names},
            "y": [],
            "image_ids": [],
            "mention_ids": [],
        }
        for split in ("train", "test")
    }
    seen_mentions: set[str] = set()
    shard_count = position_count = mention_count = 0
    shard_dir = model_root / "results/jffn_p_comparison/shards"
    for shard in load_shards(shard_dir):
        shard_count += 1
        positions = {str(row["target_key"]): row for row in shard["positions"]}
        position_count += len(positions)
        for mention in shard["sample_table"]:
            mention_id = str(mention["mention_id"])
            if mention_id in seen_mentions:
                raise AssertionError(f"Duplicate mention {mention_id}")
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
            position = positions[str(mention["target_key"])]
            blocks = {
                "i": _array(position["jffn_diagnostics"]["input_norm"]),
                "r": _array(position["jffn_diagnostics"]["response_norm"]),
                "old_risk": _array(
                    position["risks"]["old_hpre_cos"]["hpre_raw_logit_gauss"]
                    ["sqrt_matched_state"]
                ),
                "jffn_risk": _array(
                    position["risks"]["new_jffn"]["hpre_raw_logit_gauss"]
                    ["sqrt_matched_state"]
                ),
                "ev": _array(position["ev"]["hpre_raw_logit_gauss"]),
            }
            if len({value.shape for value in blocks.values()}) != 1:
                raise AssertionError(f"Layer mismatch for {mention_id}")
            if not all(np.isfinite(value).all() for value in blocks.values()):
                raise ValueError(f"Non-finite block for {mention_id}")
            store = stores[split]
            for name, value in blocks.items():
                store[name].append(value)
            store["y"].append(int(mention["label"]))
            store["image_ids"].append(image_id)
            store["mention_ids"].append(mention_id)
            mention_count += 1

    matrices: dict[str, dict[str, Any]] = {}
    for split, store in stores.items():
        blocks = {name: np.stack(store[name]) for name in names}
        feature_sets = build_feature_sets(blocks)
        matrices[split] = {
            "blocks": blocks,
            "x": feature_sets,
            "y": np.asarray(store["y"], dtype=np.int32),
            "image_ids": np.asarray(store["image_ids"], dtype=np.int64),
            "mention_ids": list(store["mention_ids"]),
        }
        if set(np.unique(matrices[split]["y"])) != {0, 1}:
            raise AssertionError(f"{split} lacks both labels")

    layers = int(matrices["train"]["blocks"]["i"].shape[1])
    audit = {
        "shards": shard_count,
        "positions": position_count,
        "mentions": mention_count,
        "train_images": int(np.unique(matrices["train"]["image_ids"]).size),
        "test_images": int(np.unique(matrices["test"]["image_ids"]).size),
        "train_mentions": int(len(matrices["train"]["y"])),
        "test_mentions": int(len(matrices["test"]["y"])),
        "test_real": int(np.sum(matrices["test"]["y"] == 1)),
        "test_hall": int(np.sum(matrices["test"]["y"] == 0)),
        "layers": layers,
        "source_protocols": {
            "old": "old_hpre_cos + hpre_raw_logit_gauss + sqrt_matched_state",
            "jffn": "new_jffn + hpre_raw_logit_gauss + sqrt_matched_state",
        },
        "ev": "hpre_raw_logit_gauss mass_x_cosine",
        "all_finite": True,
    }
    return matrices, audit


def curve_statistics(
    blocks: Mapping[str, np.ndarray], labels: np.ndarray
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    label_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    for feature in ("i", "r"):
        values = np.asarray(blocks[feature], dtype=np.float64)
        for layer in range(values.shape[1]):
            selected: dict[int, np.ndarray] = {}
            stats: dict[int, tuple[float, float]] = {}
            for label, label_name in ((1, "REAL"), (0, "HALL")):
                row = values[labels == label, layer]
                selected[label] = row
                mean = float(row.mean())
                std = float(row.std(ddof=1))
                half = 1.96 * std / math.sqrt(row.size)
                stats[label] = (mean, std)
                label_rows.append(
                    {
                        "feature": feature.upper(),
                        "label": label_name,
                        "layer": layer + 1,
                        "n": int(row.size),
                        "mean": mean,
                        "std": std,
                        "ci95_low": mean - half,
                        "ci95_high": mean + half,
                    }
                )
            real_mean, real_std = stats[1]
            hall_mean, hall_std = stats[0]
            pooled = math.sqrt(
                (
                    (selected[1].size - 1) * real_std**2
                    + (selected[0].size - 1) * hall_std**2
                )
                / max(selected[1].size + selected[0].size - 2, 1)
            )
            auc = float(roc_auc_score(labels, values[:, layer]))
            effect_rows.append(
                {
                    "feature": feature.upper(),
                    "layer": layer + 1,
                    "hall_minus_real": hall_mean - real_mean,
                    "cohens_d_hall_minus_real": (
                        (hall_mean - real_mean) / pooled if pooled > 0.0 else 0.0
                    ),
                    "real_positive_auroc": auc,
                    "best_orientation_auroc": max(auc, 1.0 - auc),
                }
            )
    return label_rows, effect_rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(
    output_dir: Path,
    model: str,
    label_rows: Sequence[Mapping[str, Any]],
    effect_rows: Sequence[Mapping[str, Any]],
) -> None:
    colors = {"REAL": "#4C78A8", "HALL": "#E45756"}
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.2), sharex=True)
    for axis, feature, title in zip(
        axes,
        ("I", "R"),
        (
            r"$I=\|\sum_j a_j\|_2$: aggregate visual write norm",
            r"$R=\|J_f(z)\sum_j a_j\|_2$: aggregate FFN response norm",
        ),
    ):
        for label in ("REAL", "HALL"):
            rows = [
                row
                for row in label_rows
                if row["feature"] == feature and row["label"] == label
            ]
            x = np.asarray([row["layer"] for row in rows], dtype=np.int32)
            mean = np.asarray([row["mean"] for row in rows], dtype=np.float64)
            low = np.asarray([row["ci95_low"] for row in rows], dtype=np.float64)
            high = np.asarray([row["ci95_high"] for row in rows], dtype=np.float64)
            axis.plot(x, mean, color=colors[label], linewidth=2.0, label=label)
            axis.fill_between(x, low, high, color=colors[label], alpha=0.18)
        axis.set_title(title)
        axis.set_ylabel("Mean norm")
        axis.grid(alpha=0.25)
        axis.legend()
    axes[-1].set_xlabel("Decoder layer")
    fig.suptitle(f"{model}: REAL/HALL aggregate JFFN I and R")
    fig.tight_layout()
    stem = output_dir / f"{model}_jffn_i_r_real_hall_curves"
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.2), sharex=True)
    for axis, feature in zip(axes, ("I", "R")):
        rows = [row for row in effect_rows if row["feature"] == feature]
        x = np.asarray([row["layer"] for row in rows], dtype=np.int32)
        difference = np.asarray([row["hall_minus_real"] for row in rows])
        effect = np.asarray([row["cohens_d_hall_minus_real"] for row in rows])
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.plot(x, difference, color="#E45756", linewidth=2.0, label="HALL − REAL")
        twin = axis.twinx()
        twin.plot(x, effect, color="#72B7B2", linestyle="--", label="Cohen's d")
        axis.set_ylabel(f"{feature}: mean difference")
        twin.set_ylabel("Cohen's d")
        axis.grid(alpha=0.25)
        handles1, labels1 = axis.get_legend_handles_labels()
        handles2, labels2 = twin.get_legend_handles_labels()
        axis.legend(handles1 + handles2, labels1 + labels2, loc="best")
    axes[-1].set_xlabel("Decoder layer")
    fig.suptitle(f"{model}: I/R class separation by layer")
    fig.tight_layout()
    stem = output_dir / f"{model}_jffn_i_r_hall_minus_real"
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_metrics(output_dir: Path, model: str, comparisons: Mapping[str, Any]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 7.4))
    metrics = (("auroc", "AUROC"), ("hall_aupr", "Hall AUPR"), ("hall_f1", "Hall F1"))
    y = np.arange(len(SPECS))
    colors = [
        "#9C755F" if spec in {"i_only", "r_only", "i_r_only"}
        else "#4C78A8" if spec.startswith("old_")
        else "#F58518"
        for spec in SPECS
    ]
    for axis, (metric, title) in zip(axes, metrics):
        means = [comparisons[spec]["aggregate"][metric]["mean"] for spec in SPECS]
        stds = [comparisons[spec]["aggregate"][metric]["std"] for spec in SPECS]
        axis.barh(y, means, xerr=stds, color=colors, capsize=3)
        axis.set_yticks(y, [DISPLAY[spec] for spec in SPECS])
        axis.set_xlim(0.0, 1.0)
        axis.invert_yaxis()
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.25)
    fig.suptitle(f"{model}: aggregate I/R feature ablation")
    fig.tight_layout()
    stem = output_dir / f"{model}_jffn_i_r_feature_ablation_metrics"
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _curve_summary(effect_rows: Sequence[Mapping[str, Any]], feature: str) -> dict[str, Any]:
    rows = [row for row in effect_rows if row["feature"] == feature]
    strongest = max(rows, key=lambda row: abs(float(row["cohens_d_hall_minus_real"])))
    return {
        "hall_higher_layers": sum(float(row["hall_minus_real"]) > 0 for row in rows),
        "real_higher_layers": sum(float(row["hall_minus_real"]) < 0 for row in rows),
        "strongest_layer": int(strongest["layer"]),
        "strongest_hall_minus_real": float(strongest["hall_minus_real"]),
        "strongest_cohens_d": float(strongest["cohens_d_hall_minus_real"]),
        "strongest_best_orientation_auroc": float(strongest["best_orientation_auroc"]),
    }


def report_markdown(model: str, payload: Mapping[str, Any]) -> str:
    lines = [
        f"# {model}: aggregate I/R 曲线与幻觉检测消融",
        "",
        "I/R 均为 32 层 aggregate norm。所有分类器使用相同的 `[128,64,32]` "
        "三隐藏层 MLP、图片级 split、seeds 43/44/45、无特征标准化；REAL 为正类。",
        "",
        "## 三 seed 结果",
        "",
        "| Feature | Dim | AUROC | Hall AUPR | Hall F1 | Ensemble AUROC |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for spec in SPECS:
        row = payload["comparisons"][spec]
        aggregate = row["aggregate"]
        lines.append(
            f"| {DISPLAY[spec]} | {payload['feature_dimensions'][spec]} | "
            f"{aggregate['auroc']['mean']:.6f} ± {aggregate['auroc']['std']:.6f} | "
            f"{aggregate['hall_aupr']['mean']:.6f} | "
            f"{aggregate['hall_f1']['mean']:.6f} | "
            f"{row['seed_ensemble']['auroc']:.6f} |"
        )
    if payload["paired_bootstrap"]:
        lines += ["", "## 定向图片级 paired bootstrap", ""]
        for key, row in payload["paired_bootstrap"].items():
            new, baseline = key.split("__versus__")
            lines.append(
                f"- {DISPLAY[new]} − {DISPLAY[baseline]}：AUROC Δ="
                f"{row['new_minus_baseline_auroc']:+.6f}，95% CI "
                f"[{row['auroc_ci95'][0]:+.6f},{row['auroc_ci95'][1]:+.6f}]；"
                f"Hall-AUPR Δ={row['new_minus_baseline_hall_aupr']:+.6f}，95% CI "
                f"[{row['hall_aupr_ci95'][0]:+.6f},{row['hall_aupr_ci95'][1]:+.6f}]。"
            )
    else:
        lines += ["", "## 不确定性估计", "", "- 按本次实验要求未运行 bootstrap；仅报告三个随机种子的均值与标准差。"]
    lines += ["", "## I/R 曲线摘要", ""]
    for feature in ("I", "R"):
        row = payload["curve_summary"][feature]
        lines.append(
            f"- {feature}：HALL>REAL / REAL>HALL 层数="
            f"{row['hall_higher_layers']}/{row['real_higher_layers']}；最强 L"
            f"{row['strongest_layer']}，Hall−Real={row['strongest_hall_minus_real']:+.6f}，"
            f"d={row['strongest_cohens_d']:+.4f}，最佳方向单层 AUROC="
            f"{row['strongest_best_orientation_auroc']:.4f}。"
        )
    audit = payload["sample_audit"]
    lines += [
        "",
        "## 协议核验",
        "",
        f"- train/test images：{audit['train_images']}/{audit['test_images']}。",
        f"- train/test mentions：{audit['train_mentions']}/{audit['test_mentions']}。",
        "- 原始 risk 与 P_JFFN risk 共享同一个 hpre raw-logit Gaussian target、"
        "sqrt-matched-state cost、Union Top-K 和 mass×cosine EV。",
        "- 曲线置信带是 mention-level 描述性 95% CI；分类性能仅报告三个随机种子。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates < 0:
        raise ValueError("--bootstrap-replicates must be non-negative")
    config = load_config(args.config)
    model_root = Path(args.outputs_root) / args.model / EXPERIMENT
    output_dir = model_root / OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    matrices, audit = collect_matrices(model_root)
    train, test = matrices["train"], matrices["test"]

    all_blocks = {
        name: np.concatenate((train["blocks"][name], test["blocks"][name]), axis=0)
        for name in ("i", "r")
    }
    all_labels = np.concatenate((train["y"], test["y"]))
    label_rows, effect_rows = curve_statistics(all_blocks, all_labels)
    _write_csv(output_dir / "i_r_label_curves.csv", label_rows)
    _write_csv(output_dir / "i_r_hall_minus_real.csv", effect_rows)
    plot_curves(output_dir, args.model, label_rows, effect_rows)

    training = (config.get("jffn_p_comparison") or {}).get("training") or {}
    seeds = [int(value) for value in training.get("seeds", [43, 44, 45])]
    progress_path = output_dir / "training_progress.pt"
    completed: dict[str, dict[int, dict[str, Any]]] = {spec: {} for spec in SPECS}
    probabilities: dict[str, dict[int, np.ndarray]] = {spec: {} for spec in SPECS}
    if args.resume and progress_path.exists():
        progress = torch.load(progress_path, map_location="cpu", weights_only=False)
        if progress.get("schema_version") != SCHEMA_VERSION:
            raise AssertionError("Incompatible I/R ablation resume artifact")
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
        x_train = train["x"][spec]
        x_test = test["x"][spec]
        for seed in seeds:
            if seed in completed[spec] and seed in probabilities[spec]:
                print(f"[I/R] reuse {args.model}/{spec}/seed={seed}", flush=True)
                continue
            metrics = train_and_evaluate_probe(
                X_train=x_train,
                y_train=train["y"],
                X_val=np.empty((0, x_train.shape[1]), dtype=np.float32),
                y_val=np.empty((0,), dtype=np.int32),
                X_test=x_test,
                y_test=test["y"],
                config=_probe_config(config, seed),
                device=device,
                output_dir=str(output_dir / "training" / spec / f"seed_{seed}"),
                return_probabilities=True,
            )
            test_prob = np.asarray(metrics.pop("test_probabilities"), dtype=np.float32)
            metrics.pop("train_probabilities", None)
            metrics["seed"] = seed
            metrics["num_features"] = int(x_train.shape[1])
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
                f"[I/R] {args.model}/{spec}/seed={seed} "
                f"AUROC={metrics['auc']:.6f} "
                f"HallAUPR={metrics['hallucination_positive']['aupr']:.6f}",
                flush=True,
            )

    ensemble_prob = {
        spec: np.mean([probabilities[spec][seed] for seed in seeds], axis=0)
        for spec in SPECS
    }
    comparisons = {
        spec: _comparison_result(
            name=DISPLAY[spec],
            aggregate=aggregate_seed_metrics([completed[spec][seed] for seed in seeds]),
            labels=test["y"],
            probabilities=ensemble_prob[spec],
        )
        for spec in SPECS
    }
    bootstrap = (
        {
            f"{new}__versus__{baseline}": paired_image_bootstrap(
                labels=test["y"],
                image_ids=test["image_ids"],
                new_probabilities=ensemble_prob[new],
                baseline_probabilities=ensemble_prob[baseline],
                replicates=args.bootstrap_replicates,
                seed=20260902 + index,
            )
            for index, (new, baseline) in enumerate(BOOTSTRAP_PAIRS)
        }
        if args.bootstrap_replicates > 0
        else {}
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "definitions": {
            "I": "||sum_j a_visual_j||_2",
            "R": "||J_f(z) sum_j a_visual_j||_2",
            "old_risk": audit["source_protocols"]["old"],
            "jffn_risk": audit["source_protocols"]["jffn"],
            "EV": audit["ev"],
        },
        "feature_dimensions": {
            spec: int(train["x"][spec].shape[1]) for spec in SPECS
        },
        "sample_audit": audit,
        "training_protocol": {
            "hidden_sizes": [128, 64, 32],
            "batch_size": int(training.get("batch_size", 256)),
            "max_epochs": int(training.get("max_epochs", 100)),
            "seeds": seeds,
            "feature_normalization": "none",
            "checkpoint_selection": "minimum_train_loss",
            "threshold_selection": "train_f1",
            "split": "existing image-level 80/20",
            "bootstrap_replicates": int(args.bootstrap_replicates),
        },
        "curve_summary": {
            feature: _curve_summary(effect_rows, feature) for feature in ("I", "R")
        },
        "comparisons": comparisons,
        "paired_bootstrap": bootstrap,
    }
    atomic_json_save(payload, output_dir / "results.json")
    seed_rows = []
    for spec in SPECS:
        for seed in seeds:
            row = completed[spec][seed]
            seed_rows.append(
                {
                    "feature": spec,
                    "display": DISPLAY[spec],
                    "seed": seed,
                    "input_dimensions": row["num_features"],
                    "auroc": row["auc"],
                    "accuracy": row["accuracy"],
                    "real_f1": row["real_positive"]["f1"],
                    "hall_f1": row["hallucination_positive"]["f1"],
                    "hall_aupr": row["hallucination_positive"]["aupr"],
                    "best_epoch": row["best_epoch"],
                }
            )
    _write_csv(output_dir / "seed_metrics.csv", seed_rows)
    plot_metrics(output_dir, args.model, comparisons)
    (output_dir / f"{args.model}_jffn_i_r_feature_ablation_report.md").write_text(
        report_markdown(args.model, payload), encoding="utf-8"
    )
    print(f"[I/R] wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
