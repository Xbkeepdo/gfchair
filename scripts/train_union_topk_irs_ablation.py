#!/usr/bin/env python3
"""Train Union-TopK tokenwise I/R/S hallucination-detection ablations."""

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
from scripts.train_old_risk_ev_s_mlp import (  # noqa: E402
    EXPERIMENT,
    MODELS,
    aggregate_seed_metrics,
)
from scripts.train_torch_probe_feature_sets import train_and_evaluate_probe  # noqa: E402
from scripts.train_union_topk_region_s_mlp import (  # noqa: E402
    _comparison_result,
    _curve_summary,
    _probe_config,
    collect_matrices as collect_union_matrices,
    curve_statistics,
    plot_curves,
)
from utils.config_utils import load_config  # noqa: E402


SCHEMA_VERSION = "union-topk-irs-ablation-v1"
CACHE_SCHEMA_VERSION = "union-topk-irs-input-cache-v1"
OUTPUT_SUBDIR = "results/jffn_second_round/union_topk_irs_ablation"
SUMMARY_STEM = "union_topk_irs_ablation_2model"

SPECS = (
    "union_i_only",
    "union_r_only",
    "union_s_only",
    "union_i_r_s",
    "old_risk_ev",
    "old_risk_ev_union_i",
    "old_risk_ev_union_r",
    "old_risk_ev_union_s",
    "old_risk_ev_union_i_r_s",
    "jffn_risk_ev",
    "jffn_risk_ev_union_i",
    "jffn_risk_ev_union_r",
    "jffn_risk_ev_union_s",
    "jffn_risk_ev_union_i_r_s",
)

DISPLAY = {
    "union_i_only": "Union I only",
    "union_r_only": "Union R only",
    "union_s_only": "Union S only",
    "union_i_r_s": "Union I + R + S",
    "old_risk_ev": "Original risk + EV",
    "old_risk_ev_union_i": "Original risk + EV + Union I",
    "old_risk_ev_union_r": "Original risk + EV + Union R",
    "old_risk_ev_union_s": "Original risk + EV + Union S",
    "old_risk_ev_union_i_r_s": "Original risk + EV + Union I + R + S",
    "jffn_risk_ev": "P_JFFN risk + EV",
    "jffn_risk_ev_union_i": "P_JFFN risk + EV + Union I",
    "jffn_risk_ev_union_r": "P_JFFN risk + EV + Union R",
    "jffn_risk_ev_union_s": "P_JFFN risk + EV + Union S",
    "jffn_risk_ev_union_i_r_s": "P_JFFN risk + EV + Union I + R + S",
}

DELTA_PAIRS = (
    ("union_i_r_s", "union_i_only"),
    ("union_i_r_s", "union_r_only"),
    ("union_i_r_s", "union_s_only"),
    ("old_risk_ev_union_i", "old_risk_ev"),
    ("old_risk_ev_union_r", "old_risk_ev"),
    ("old_risk_ev_union_s", "old_risk_ev"),
    ("old_risk_ev_union_i_r_s", "old_risk_ev"),
    ("jffn_risk_ev_union_i", "jffn_risk_ev"),
    ("jffn_risk_ev_union_r", "jffn_risk_ev"),
    ("jffn_risk_ev_union_s", "jffn_risk_ev"),
    ("jffn_risk_ev_union_i_r_s", "jffn_risk_ev"),
)

REUSE_SOURCES = {
    "old_risk_ev": (
        "normalized_ir_s_ablation/training_progress.pt",
        "old_risk_ev",
    ),
    "jffn_risk_ev": (
        "normalized_ir_s_ablation/training_progress.pt",
        "jffn_risk_ev",
    ),
    "old_risk_ev_union_s": (
        "union_topk_region_s_mlp/training_progress.pt",
        "union_topk_s",
    ),
}


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


def build_feature_sets(blocks: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    required = ("i", "r", "s", "old_risk", "jffn_risk", "ev")
    arrays = {name: np.asarray(blocks[name], dtype=np.float32) for name in required}
    shapes = {name: value.shape for name, value in arrays.items()}
    if len(set(shapes.values())) != 1 or arrays["i"].ndim != 2:
        raise ValueError(f"Expected six matching [rows,layers] blocks, got {shapes}")
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Union I/R/S, risks and EV must all be finite")
    i, r, s = arrays["i"], arrays["r"], arrays["s"]
    old, jffn, ev = arrays["old_risk"], arrays["jffn_risk"], arrays["ev"]
    output = {
        "union_i_only": i,
        "union_r_only": r,
        "union_s_only": s,
        "union_i_r_s": np.concatenate((i, r, s), axis=1),
        "old_risk_ev": np.concatenate((old, ev), axis=1),
        "old_risk_ev_union_i": np.concatenate((old, ev, i), axis=1),
        "old_risk_ev_union_r": np.concatenate((old, ev, r), axis=1),
        "old_risk_ev_union_s": np.concatenate((old, ev, s), axis=1),
        "old_risk_ev_union_i_r_s": np.concatenate((old, ev, i, r, s), axis=1),
        "jffn_risk_ev": np.concatenate((jffn, ev), axis=1),
        "jffn_risk_ev_union_i": np.concatenate((jffn, ev, i), axis=1),
        "jffn_risk_ev_union_r": np.concatenate((jffn, ev, r), axis=1),
        "jffn_risk_ev_union_s": np.concatenate((jffn, ev, s), axis=1),
        "jffn_risk_ev_union_i_r_s": np.concatenate((jffn, ev, i, r, s), axis=1),
    }
    return {name: np.ascontiguousarray(output[name]) for name in SPECS}


def _compact_matrices(matrices: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for split in ("train", "test"):
        row = matrices[split]
        blocks = {
            "i": np.asarray(row["union_write_sum"], dtype=np.float32),
            "r": np.asarray(row["union_response_sum"], dtype=np.float32),
            "s": np.asarray(row["union_topk_s"], dtype=np.float32),
            "old_risk": np.asarray(row["old_risk"], dtype=np.float32),
            "jffn_risk": np.asarray(row["jffn_risk"], dtype=np.float32),
            "ev": np.asarray(row["ev"], dtype=np.float32),
        }
        output[split] = {
            "blocks": blocks,
            "x": build_feature_sets(blocks),
            "y": np.asarray(row["y"], dtype=np.int32),
            "image_ids": np.asarray(row["image_ids"], dtype=np.int64),
            "mention_ids": list(row["mention_ids"]),
        }
    return output


def load_or_build_inputs(
    model_root: Path, model: str, output_dir: Path, resume: bool
) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray], dict[str, Any], bool]:
    cache_path = output_dir / "feature_matrix_cache.pt"
    if resume and cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload.get("schema_version") == CACHE_SCHEMA_VERSION and payload.get("model") == model:
            return payload["matrices"], payload["curves"], payload["audit"], True
    source_matrices, curves, audit = collect_union_matrices(model_root, model)
    matrices = _compact_matrices(source_matrices)
    atomic_torch_save(
        {
            "schema_version": CACHE_SCHEMA_VERSION,
            "model": model,
            "matrices": matrices,
            "curves": curves,
            "audit": audit,
        },
        cache_path,
    )
    return matrices, curves, audit, False


def _preload_existing(
    model_root: Path,
    test: Mapping[str, Any],
    completed: dict[str, dict[int, dict[str, Any]]],
    probabilities: dict[str, dict[int, np.ndarray]],
) -> list[str]:
    reused: list[str] = []
    base = model_root / "results/jffn_second_round"
    for destination, (relative_path, source) in REUSE_SOURCES.items():
        path = base / relative_path
        if not path.exists():
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if list(payload["mention_ids"]) != list(test["mention_ids"]):
            raise AssertionError(f"Mention order mismatch while reusing {path}")
        if not np.array_equal(np.asarray(payload["labels"]), test["y"]):
            raise AssertionError(f"Label mismatch while reusing {path}")
        if not np.array_equal(np.asarray(payload["image_ids"]), test["image_ids"]):
            raise AssertionError(f"Image order mismatch while reusing {path}")
        rows = (payload.get("seed_results") or {}).get(source, ())
        prediction_store = (payload.get("probabilities") or {}).get(source, {})
        completed[destination] = {int(row["seed"]): row for row in rows}
        probabilities[destination] = {
            int(seed): np.asarray(value, dtype=np.float32)
            for seed, value in prediction_store.items()
        }
        if completed[destination] and probabilities[destination]:
            reused.append(destination)
    return reused


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_metrics(output_dir: Path, model: str, comparisons: Mapping[str, Any]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 9.0))
    metrics = (("auroc", "AUROC"), ("hall_aupr", "Hall AUPR"), ("hall_f1", "Hall F1"))
    y = np.arange(len(SPECS))
    colors = [
        "#9C755F"
        if spec.startswith("union_")
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
    fig.suptitle(f"{model}: Union-TopK I/R/S ablation")
    fig.tight_layout()
    stem = output_dir / f"{model}_union_topk_irs_ablation_metrics"
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def report_markdown(model: str, payload: Mapping[str, Any]) -> str:
    lines = [
        f"# {model}: Union-TopK 区域 I/R/S 检测消融",
        "",
        "每层 `U=Top32(P_JFFN)∪Top32(Q_raw)`；`I_U=Σ_{j∈U}||a_j||₂`，"
        "`R_U=Σ_{j∈U}||J_f(z)a_j||₂`，`S_U=R_U/(I_U+epsilon)`。I/R/S 联合项均为特征拼接。",
        "",
        "三隐藏层 `[128,64,32]` MLP，seeds 43/44/45、相同图片级 split、无特征标准化。"
        "未运行 bootstrap。",
        "",
        "## 三 seed 结果",
        "",
        "| Feature | Dim | AUROC | Hall AUPR | Hall F1 | Ensemble AUROC |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for spec in SPECS:
        row = payload["comparisons"][spec]
        agg = row["aggregate"]
        lines.append(
            f"| {DISPLAY[spec]} | {payload['feature_dimensions'][spec]} | "
            f"{agg['auroc']['mean']:.6f} ± {agg['auroc']['std']:.6f} | "
            f"{agg['hall_aupr']['mean']:.6f} | {agg['hall_f1']['mean']:.6f} | "
            f"{row['seed_ensemble']['auroc']:.6f} |"
        )
    lines += ["", "## 描述性均值差", ""]
    for new, baseline in DELTA_PAIRS:
        new_row = payload["comparisons"][new]["aggregate"]
        base_row = payload["comparisons"][baseline]["aggregate"]
        lines.append(
            f"- {DISPLAY[new]} − {DISPLAY[baseline]}：AUROC "
            f"{new_row['auroc']['mean'] - base_row['auroc']['mean']:+.6f}；Hall AUPR "
            f"{new_row['hall_aupr']['mean'] - base_row['hall_aupr']['mean']:+.6f}；Hall F1 "
            f"{new_row['hall_f1']['mean'] - base_row['hall_f1']['mean']:+.6f}。"
        )
    names = {
        "union_write_sum": "I_U",
        "union_response_sum": "R_U",
        "union_topk_s": "S_U",
    }
    lines += ["", "## 曲线摘要", ""]
    for feature, label in names.items():
        row = payload["curve_summary"][feature]
        lines.append(
            f"- {label}：HALL>REAL / REAL>HALL="
            f"{row['hall_higher_layers']}/{row['real_higher_layers']}；最强 L"
            f"{row['strongest_layer']}，Hall−Real={row['strongest_hall_minus_real']:+.6f}，"
            f"d={row['strongest_cohens_d']:+.4f}。"
        )
    audit = payload["sample_audit"]
    lines += [
        "",
        "## 审计",
        "",
        f"- train/test images：{audit['train_images']}/{audit['test_images']}。",
        f"- train/test mentions：{audit['train_mentions']}/{audit['test_mentions']}。",
        f"- Union size min/mean/max：{audit['union_size_minimum']}/"
        f"{audit['union_size_mean']:.2f}/{audit['union_size_maximum']}。",
        f"- 输入缓存：{'复用' if payload['cache_reused'] else '本轮新建'}。",
        f"- 复用的既有训练头：{', '.join(payload['reused_feature_sets']) or '无'}。",
        "",
    ]
    return "\n".join(lines)


def train_model(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    model_root = Path(args.outputs_root) / args.model / EXPERIMENT
    output_dir = model_root / OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    matrices, curves, audit, cache_reused = load_or_build_inputs(
        model_root, args.model, output_dir, args.resume
    )
    train, test = matrices["train"], matrices["test"]

    label_rows, effect_rows = curve_statistics(curves)
    _write_csv(output_dir / "union_topk_irs_label_curves.csv", label_rows)
    _write_csv(output_dir / "union_topk_irs_hall_minus_real.csv", effect_rows)
    plot_curves(args.model, label_rows, output_dir)

    training = (config.get("jffn_p_comparison") or {}).get("training") or {}
    seeds = [int(value) for value in training.get("seeds", [43, 44, 45])]
    progress_path = output_dir / "training_progress.pt"
    completed: dict[str, dict[int, dict[str, Any]]] = {spec: {} for spec in SPECS}
    probabilities: dict[str, dict[int, np.ndarray]] = {spec: {} for spec in SPECS}
    if args.resume and progress_path.exists():
        progress = torch.load(progress_path, map_location="cpu", weights_only=False)
        if progress.get("schema_version") != SCHEMA_VERSION:
            raise AssertionError("Incompatible Union I/R/S resume artifact")
        for spec in SPECS:
            completed[spec] = {
                int(row["seed"]): row
                for row in (progress.get("seed_results") or {}).get(spec, ())
            }
            probabilities[spec] = {
                int(seed): np.asarray(value, dtype=np.float32)
                for seed, value in (progress.get("probabilities") or {}).get(spec, {}).items()
            }
        reused = list(progress.get("reused_feature_sets") or [])
    else:
        reused = _preload_existing(model_root, test, completed, probabilities)

    device = _device(args.training_device)
    for spec in SPECS:
        for seed in seeds:
            if seed in completed[spec] and seed in probabilities[spec]:
                print(f"[Union-IRS] reuse {args.model}/{spec}/seed={seed}", flush=True)
                continue
            x_train, x_test = train["x"][spec], test["x"][spec]
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
                    "reused_feature_sets": reused,
                },
                progress_path,
            )
            print(
                f"[Union-IRS] {args.model}/{spec}/seed={seed} AUROC="
                f"{metrics['auc']:.6f} HallAUPR="
                f"{metrics['hallucination_positive']['aupr']:.6f}",
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
        "definition": {
            "union": "Top32(P_JFFN) union Top32(Q_hpre_raw_logit_gauss) per layer",
            "I": "sum_{j in U} ||a_j||_2",
            "R": "sum_{j in U} ||J_f(z)a_j||_2",
            "S": "R/(I+epsilon)",
            "joint_features": "concatenation",
        },
        "feature_dimensions": {spec: int(train["x"][spec].shape[1]) for spec in SPECS},
        "sample_audit": audit,
        "cache_reused": cache_reused,
        "reused_feature_sets": reused,
        "training_protocol": {
            "hidden_sizes": [128, 64, 32],
            "batch_size": int(training.get("batch_size", 256)),
            "max_epochs": int(training.get("max_epochs", 100)),
            "seeds": seeds,
            "feature_normalization": "none",
            "checkpoint_selection": "minimum_train_loss",
            "threshold_selection": "train_f1",
            "split": "existing image-level 80/20",
            "bootstrap": "not run",
        },
        "curve_summary": {
            feature: _curve_summary(effect_rows, feature)
            for feature in ("union_write_sum", "union_response_sum", "union_topk_s")
        },
        "comparisons": comparisons,
    }
    atomic_json_save(payload, output_dir / "results.json")
    seed_rows: list[dict[str, Any]] = []
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
    (output_dir / f"{args.model}_union_topk_irs_ablation_report.md").write_text(
        report_markdown(args.model, payload), encoding="utf-8"
    )
    print(f"[Union-IRS] wrote {output_dir}", flush=True)


def summarize(outputs_root: Path) -> None:
    models: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        path = outputs_root / model / EXPERIMENT / OUTPUT_SUBDIR / "results.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        models[model] = payload
        for spec in SPECS:
            agg = payload["comparisons"][spec]["aggregate"]
            rows.append(
                {
                    "model": model,
                    "feature": spec,
                    "display": DISPLAY[spec],
                    "dimensions": payload["feature_dimensions"][spec],
                    "auroc_mean": agg["auroc"]["mean"],
                    "auroc_std": agg["auroc"]["std"],
                    "hall_aupr_mean": agg["hall_aupr"]["mean"],
                    "hall_f1_mean": agg["hall_f1"]["mean"],
                }
            )
    atomic_json_save(
        {
            "schema_version": SCHEMA_VERSION,
            "models": {model: models[model]["comparisons"] for model in MODELS},
            "sample_audits": {model: models[model]["sample_audit"] for model in MODELS},
        },
        outputs_root / f"{SUMMARY_STEM}.json",
    )
    _write_csv(outputs_root / f"{SUMMARY_STEM}_metrics.csv", rows)
    lines = [
        "# Union-TopK 区域 I/R/S：两模型汇总",
        "",
        "`I_U=Σ||a_j||`、`R_U=Σ||Ja_j||`、`S_U=R_U/I_U`；三 seed、同一 MLP/split，"
        "无标准化，不做 bootstrap。",
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
            agg = models[model]["comparisons"][spec]["aggregate"]
            lines.append(
                f"| {DISPLAY[spec]} | {agg['auroc']['mean']:.6f} ± "
                f"{agg['auroc']['std']:.6f} | {agg['hall_aupr']['mean']:.6f} | "
                f"{agg['hall_f1']['mean']:.6f} |"
            )
    (outputs_root / f"{SUMMARY_STEM}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[Union-IRS] wrote {outputs_root / (SUMMARY_STEM + '.md')}", flush=True)


def main() -> None:
    args = parse_args()
    if args.model == "summarize":
        summarize(Path(args.outputs_root))
    else:
        train_model(args)


if __name__ == "__main__":
    main()
