#!/usr/bin/env python3
"""Compare per-sample L2-normalized I/R sum with aggregate Jacobian S."""

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

from features.jffn_experiment import atomic_json_save, atomic_torch_save, load_shards  # noqa: E402
from scripts.train_old_risk_ev_s_mlp import (  # noqa: E402
    EXPERIMENT,
    MODELS,
    aggregate_seed_metrics,
)
from scripts.train_torch_probe_feature_sets import train_and_evaluate_probe  # noqa: E402
from scripts.train_union_topk_region_s_mlp import (  # noqa: E402
    _comparison_result,
    _probe_config,
)
from utils.config_utils import load_config  # noqa: E402
from utils.io_utils import load_json  # noqa: E402


SCHEMA_VERSION = "jffn-normalized-ir-s-ablation-v1"
OUTPUT_SUBDIR = "results/jffn_second_round/normalized_ir_s_ablation"
SUMMARY_STEM = "normalized_ir_s_ablation_2model"
EPS = 1.0e-12

SPECS = (
    "normalized_ir_sum_only",
    "s_only",
    "normalized_ir_sum_s",
    "old_risk_ev",
    "old_risk_ev_normalized_ir_sum",
    "old_risk_ev_s",
    "old_risk_ev_normalized_ir_sum_s",
    "jffn_risk_ev",
    "jffn_risk_ev_normalized_ir_sum",
    "jffn_risk_ev_s",
    "jffn_risk_ev_normalized_ir_sum_s",
)

DISPLAY = {
    "normalized_ir_sum_only": "N = L2(I) + L2(R)",
    "s_only": "S = R / I",
    "normalized_ir_sum_s": "N + S",
    "old_risk_ev": "Original risk + EV",
    "old_risk_ev_normalized_ir_sum": "Original risk + EV + N",
    "old_risk_ev_s": "Original risk + EV + S",
    "old_risk_ev_normalized_ir_sum_s": "Original risk + EV + N + S",
    "jffn_risk_ev": "P_JFFN risk + EV",
    "jffn_risk_ev_normalized_ir_sum": "P_JFFN risk + EV + N",
    "jffn_risk_ev_s": "P_JFFN risk + EV + S",
    "jffn_risk_ev_normalized_ir_sum_s": "P_JFFN risk + EV + N + S",
}

DELTA_PAIRS = (
    ("normalized_ir_sum_s", "normalized_ir_sum_only"),
    ("normalized_ir_sum_s", "s_only"),
    ("old_risk_ev_normalized_ir_sum", "old_risk_ev"),
    ("old_risk_ev_s", "old_risk_ev"),
    ("old_risk_ev_normalized_ir_sum_s", "old_risk_ev"),
    ("jffn_risk_ev_normalized_ir_sum", "jffn_risk_ev"),
    ("jffn_risk_ev_s", "jffn_risk_ev"),
    ("jffn_risk_ev_normalized_ir_sum_s", "jffn_risk_ev"),
)


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


def _array(value: Any) -> np.ndarray:
    return np.asarray(torch.as_tensor(value).float().cpu(), dtype=np.float32).reshape(-1)


def l2_normalize_rows(values: np.ndarray, eps: float = EPS) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("L2 normalization expects a finite [rows,layers] matrix")
    norms = np.linalg.norm(values.astype(np.float64), axis=1, keepdims=True)
    if np.any(norms <= eps):
        raise ValueError("I/R row has a zero L2 norm")
    return np.ascontiguousarray(values / norms.astype(np.float32))


def build_feature_sets(blocks: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    required = ("i", "r", "s", "old_risk", "jffn_risk", "ev")
    arrays = {name: np.asarray(blocks[name], dtype=np.float32) for name in required}
    shapes = {name: value.shape for name, value in arrays.items()}
    if len(set(shapes.values())) != 1 or arrays["i"].ndim != 2:
        raise ValueError(f"Expected six matching [rows,layers] blocks, got {shapes}")
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("All input blocks must be finite")

    normalized_i = l2_normalize_rows(arrays["i"])
    normalized_r = l2_normalize_rows(arrays["r"])
    normalized_sum = normalized_i + normalized_r
    s = arrays["s"]
    old, jffn, ev = arrays["old_risk"], arrays["jffn_risk"], arrays["ev"]
    output = {
        "normalized_ir_sum_only": normalized_sum,
        "s_only": s,
        "normalized_ir_sum_s": np.concatenate((normalized_sum, s), axis=1),
        "old_risk_ev": np.concatenate((old, ev), axis=1),
        "old_risk_ev_normalized_ir_sum": np.concatenate((old, ev, normalized_sum), axis=1),
        "old_risk_ev_s": np.concatenate((old, ev, s), axis=1),
        "old_risk_ev_normalized_ir_sum_s": np.concatenate(
            (old, ev, normalized_sum, s), axis=1
        ),
        "jffn_risk_ev": np.concatenate((jffn, ev), axis=1),
        "jffn_risk_ev_normalized_ir_sum": np.concatenate(
            (jffn, ev, normalized_sum), axis=1
        ),
        "jffn_risk_ev_s": np.concatenate((jffn, ev, s), axis=1),
        "jffn_risk_ev_normalized_ir_sum_s": np.concatenate(
            (jffn, ev, normalized_sum, s), axis=1
        ),
    }
    return {name: np.ascontiguousarray(output[name]) for name in SPECS}


def collect_matrices(model_root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    split_payload = load_json(str(model_root / "image_splits.json"))
    train_ids = {int(value) for value in split_payload["train"]}
    test_ids = {int(value) for value in split_payload["test"]}
    if train_ids & test_ids:
        raise AssertionError("Official image split overlaps")

    names = ("i", "r", "s", "old_risk", "jffn_risk", "ev")
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
    max_s_parity_error = 0.0
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
                "s": _array(position["jffn_diagnostics"]["gain"]),
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
            recomputed_s = blocks["r"] / np.maximum(blocks["i"], EPS)
            max_s_parity_error = max(
                max_s_parity_error,
                float(np.max(np.abs(recomputed_s - blocks["s"]))),
            )
            store = stores[split]
            for name, value in blocks.items():
                store[name].append(value)
            store["y"].append(int(mention["label"]))
            store["image_ids"].append(image_id)
            store["mention_ids"].append(mention_id)
            mention_count += 1

    matrices: dict[str, dict[str, Any]] = {}
    max_unit_norm_error = 0.0
    for split, store in stores.items():
        blocks = {name: np.stack(store[name]) for name in names}
        feature_sets = build_feature_sets(blocks)
        for name in ("i", "r"):
            normalized = l2_normalize_rows(blocks[name])
            max_unit_norm_error = max(
                max_unit_norm_error,
                float(np.max(np.abs(np.linalg.norm(normalized, axis=1) - 1.0))),
            )
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
        "normalization": "per-sample L2 across decoder layers, separately for I and R",
        "normalized_sum": "I / ||I||_2 + R / ||R||_2; no second normalization",
        "s": "stored aggregate gain R/(I+epsilon)",
        "max_s_recompute_absolute_error": max_s_parity_error,
        "max_normalized_ir_unit_norm_error": max_unit_norm_error,
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
    normalized_sum = l2_normalize_rows(blocks["i"]) + l2_normalize_rows(blocks["r"])
    curves = {"N": normalized_sum, "S": np.asarray(blocks["s"], dtype=np.float32)}
    label_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    for feature, values_raw in curves.items():
        values = np.asarray(values_raw, dtype=np.float64)
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
                        "feature": feature,
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
                    "feature": feature,
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
    titles = {
        "N": r"$N=I/\|I\|_2+R/\|R\|_2$",
        "S": r"$S=R/(I+\epsilon)$",
    }
    colors = {"REAL": "#4C78A8", "HALL": "#E45756"}
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.2), sharex=True)
    for axis, feature in zip(axes, ("N", "S")):
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
            axis.plot(x, mean, color=colors[label], linewidth=2.0, label=label)
            axis.fill_between(x, low, high, color=colors[label], alpha=0.18)
        axis.set_title(titles[feature])
        axis.set_ylabel("Mean feature value")
        axis.grid(alpha=0.25)
        axis.legend()
    axes[-1].set_xlabel("Decoder layer")
    fig.suptitle(f"{model}: normalized I/R sum versus aggregate S")
    fig.tight_layout()
    stem = output_dir / f"{model}_normalized_ir_sum_s_real_hall_curves"
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.2), sharex=True)
    for axis, feature in zip(axes, ("N", "S")):
        rows = [row for row in effect_rows if row["feature"] == feature]
        x = np.asarray([row["layer"] for row in rows])
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
    fig.suptitle(f"{model}: normalized-I/R and S class separation")
    fig.tight_layout()
    stem = output_dir / f"{model}_normalized_ir_sum_s_hall_minus_real"
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_metrics(output_dir: Path, model: str, comparisons: Mapping[str, Any]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 7.4))
    metrics = (("auroc", "AUROC"), ("hall_aupr", "Hall AUPR"), ("hall_f1", "Hall F1"))
    y = np.arange(len(SPECS))
    colors = [
        "#9C755F"
        if spec in {"normalized_ir_sum_only", "s_only", "normalized_ir_sum_s"}
        else "#4C78A8"
        if spec.startswith("old_")
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
    fig.suptitle(f"{model}: normalized I/R sum and S ablation")
    fig.tight_layout()
    stem = output_dir / f"{model}_normalized_ir_s_ablation_metrics"
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
        f"# {model}: normalized I/R sum 与 aggregate S 消融",
        "",
        "`N=I/||I||₂+R/||R||₂`：每个样本分别沿 32 个 decoder 层对 I、R 做 L2 "
        "归一化后逐层相加，不做第二次归一化。S 使用已有 aggregate `R/(I+epsilon)`。",
        "",
        "所有分类头均为 `[128,64,32]` 三隐藏层 MLP、seeds 43/44/45、图片级 split、"
        "无特征标准化；REAL 为 AUROC 正类。未运行 bootstrap。",
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
    lines += ["", "## 描述性均值差", ""]
    for new, baseline in DELTA_PAIRS:
        new_row = payload["comparisons"][new]["aggregate"]
        base_row = payload["comparisons"][baseline]["aggregate"]
        lines.append(
            f"- {DISPLAY[new]} − {DISPLAY[baseline]}：AUROC "
            f"{new_row['auroc']['mean'] - base_row['auroc']['mean']:+.6f}；"
            f"Hall AUPR {new_row['hall_aupr']['mean'] - base_row['hall_aupr']['mean']:+.6f}；"
            f"Hall F1 {new_row['hall_f1']['mean'] - base_row['hall_f1']['mean']:+.6f}。"
        )
    lines += ["", "## N/S 曲线摘要", ""]
    for feature in ("N", "S"):
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
        f"- stored S 与 R/I 最大绝对误差：{audit['max_s_recompute_absolute_error']:.3e}。",
        f"- I/R 单位范数最大误差：{audit['max_normalized_ir_unit_norm_error']:.3e}。",
        "- 两种 risk 只改变 source P；target Q、Union Top-K、cost 与 EV 全部相同。",
        "- 不使用跨样本均值/方差，不读取测试集统计进行归一化。",
        "",
    ]
    return "\n".join(lines)


def train_model(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    model_root = Path(args.outputs_root) / args.model / EXPERIMENT
    output_dir = model_root / OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    matrices, audit = collect_matrices(model_root)
    train, test = matrices["train"], matrices["test"]

    all_blocks = {
        name: np.concatenate((train["blocks"][name], test["blocks"][name]), axis=0)
        for name in ("i", "r", "s")
    }
    all_labels = np.concatenate((train["y"], test["y"]))
    label_rows, effect_rows = curve_statistics(all_blocks, all_labels)
    _write_csv(output_dir / "normalized_ir_s_label_curves.csv", label_rows)
    _write_csv(output_dir / "normalized_ir_s_hall_minus_real.csv", effect_rows)
    plot_curves(output_dir, args.model, label_rows, effect_rows)

    training = (config.get("jffn_p_comparison") or {}).get("training") or {}
    seeds = [int(value) for value in training.get("seeds", [43, 44, 45])]
    progress_path = output_dir / "training_progress.pt"
    completed: dict[str, dict[int, dict[str, Any]]] = {spec: {} for spec in SPECS}
    probabilities: dict[str, dict[int, np.ndarray]] = {spec: {} for spec in SPECS}
    if args.resume and progress_path.exists():
        progress = torch.load(progress_path, map_location="cpu", weights_only=False)
        if progress.get("schema_version") != SCHEMA_VERSION:
            raise AssertionError("Incompatible normalized-I/R resume artifact")
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
                print(f"[NIR-S] reuse {args.model}/{spec}/seed={seed}", flush=True)
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
                f"[NIR-S] {args.model}/{spec}/seed={seed} "
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
    payload = {
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "definitions": {
            "I": "||sum_j a_visual_j||_2",
            "R": "||J_f(z) sum_j a_visual_j||_2",
            "normalized_ir_sum": "I/||I||_2 + R/||R||_2 across decoder layers per sample",
            "S": "aggregate R/(I+epsilon)",
        },
        "feature_dimensions": {spec: int(train["x"][spec].shape[1]) for spec in SPECS},
        "sample_audit": audit,
        "training_protocol": {
            "hidden_sizes": [128, 64, 32],
            "batch_size": int(training.get("batch_size", 256)),
            "max_epochs": int(training.get("max_epochs", 100)),
            "seeds": seeds,
            "feature_normalization": "only I/R per-sample L2; no classifier-wide standardization",
            "checkpoint_selection": "minimum_train_loss",
            "threshold_selection": "train_f1",
            "split": "existing image-level 80/20",
            "bootstrap": "not run",
        },
        "curve_summary": {
            feature: _curve_summary(effect_rows, feature) for feature in ("N", "S")
        },
        "comparisons": comparisons,
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
    (output_dir / f"{args.model}_normalized_ir_s_ablation_report.md").write_text(
        report_markdown(args.model, payload), encoding="utf-8"
    )
    print(f"[NIR-S] wrote {output_dir}", flush=True)


def summarize(outputs_root: Path) -> None:
    models: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        path = outputs_root / model / EXPERIMENT / OUTPUT_SUBDIR / "results.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        models[model] = payload
        for spec in SPECS:
            aggregate = payload["comparisons"][spec]["aggregate"]
            rows.append(
                {
                    "model": model,
                    "feature": spec,
                    "display": DISPLAY[spec],
                    "dimensions": payload["feature_dimensions"][spec],
                    "auroc_mean": aggregate["auroc"]["mean"],
                    "auroc_std": aggregate["auroc"]["std"],
                    "hall_aupr_mean": aggregate["hall_aupr"]["mean"],
                    "hall_f1_mean": aggregate["hall_f1"]["mean"],
                }
            )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "models": {model: models[model]["comparisons"] for model in MODELS},
        "sample_audits": {model: models[model]["sample_audit"] for model in MODELS},
    }
    atomic_json_save(summary, outputs_root / f"{SUMMARY_STEM}.json")
    _write_csv(outputs_root / f"{SUMMARY_STEM}_metrics.csv", rows)
    lines = [
        "# Normalized I/R sum 与 aggregate S：两模型汇总",
        "",
        "`N=I/||I||₂+R/||R||₂` 为逐样本、沿 decoder 层分别归一化 I/R 后求和；"
        "S 为已有 aggregate `R/(I+epsilon)`。三 seed、相同 MLP/split，不做 bootstrap。",
    ]
    for model in MODELS:
        lines += [
            "",
            f"## {model}",
            "",
            "| Feature | AUROC | Hall AUPR | Hall F1 |",
            "| --- | ---: | ---: | ---: |",
        ]
        for spec in SPECS:
            aggregate = models[model]["comparisons"][spec]["aggregate"]
            lines.append(
                f"| {DISPLAY[spec]} | {aggregate['auroc']['mean']:.6f} ± "
                f"{aggregate['auroc']['std']:.6f} | {aggregate['hall_aupr']['mean']:.6f} | "
                f"{aggregate['hall_f1']['mean']:.6f} |"
            )
    (outputs_root / f"{SUMMARY_STEM}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[NIR-S] wrote {outputs_root / (SUMMARY_STEM + '.md')}", flush=True)


def main() -> None:
    args = parse_args()
    if args.model == "summarize":
        summarize(Path(args.outputs_root))
    else:
        train_model(args)


if __name__ == "__main__":
    main()
