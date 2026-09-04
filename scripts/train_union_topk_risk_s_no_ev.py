#!/usr/bin/env python3
"""Train risk-only versus risk + Union-TopK S probes without EV."""

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
from scripts.train_union_topk_irs_ablation import (  # noqa: E402
    CACHE_SCHEMA_VERSION,
    OUTPUT_SUBDIR as CACHE_OUTPUT_SUBDIR,
)
from scripts.train_union_topk_region_s_mlp import (  # noqa: E402
    _comparison_result,
    _probe_config,
)
from utils.config_utils import load_config  # noqa: E402


SCHEMA_VERSION = "union-topk-risk-s-no-ev-v1"
OUTPUT_SUBDIR = "results/jffn_second_round/union_topk_risk_s_no_ev"
SUMMARY_STEM = "union_topk_risk_s_no_ev_2model"

SPECS = (
    "old_risk_only",
    "old_risk_union_s",
    "jffn_risk_only",
    "jffn_risk_union_s",
)

DISPLAY = {
    "old_risk_only": "Original risk only",
    "old_risk_union_s": "Original risk + Union S",
    "jffn_risk_only": "P_JFFN risk only",
    "jffn_risk_union_s": "P_JFFN risk + Union S",
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
    names = ("old_risk", "jffn_risk", "s")
    arrays = {name: np.asarray(blocks[name], dtype=np.float32) for name in names}
    shapes = {name: value.shape for name, value in arrays.items()}
    if len(set(shapes.values())) != 1 or arrays["s"].ndim != 2:
        raise ValueError(f"Expected three matching [rows,layers] blocks, got {shapes}")
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Risk and Union S blocks must all be finite")
    old, jffn, union_s = arrays["old_risk"], arrays["jffn_risk"], arrays["s"]
    return {
        "old_risk_only": np.ascontiguousarray(old),
        "old_risk_union_s": np.ascontiguousarray(
            np.concatenate((old, union_s), axis=1)
        ),
        "jffn_risk_only": np.ascontiguousarray(jffn),
        "jffn_risk_union_s": np.ascontiguousarray(
            np.concatenate((jffn, union_s), axis=1)
        ),
    }


def load_inputs(model_root: Path, model: str) -> dict[str, dict[str, Any]]:
    cache_path = model_root / CACHE_OUTPUT_SUBDIR / "feature_matrix_cache.pt"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Union I/R/S compact cache is required; run "
            f"train_union_topk_irs_ablation.py first: {cache_path}"
        )
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise AssertionError(f"Incompatible cache schema in {cache_path}")
    if payload.get("model") != model:
        raise AssertionError(f"Cache model mismatch in {cache_path}")
    output: dict[str, dict[str, Any]] = {}
    for split in ("train", "test"):
        source = payload["matrices"][split]
        output[split] = {
            "x": build_feature_sets(source["blocks"]),
            "y": np.asarray(source["y"], dtype=np.int32),
            "image_ids": np.asarray(source["image_ids"], dtype=np.int64),
            "mention_ids": list(source["mention_ids"]),
        }
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(output_dir: Path, model: str, comparisons: Mapping[str, Any]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.4))
    metrics = (("auroc", "AUROC"), ("hall_aupr", "Hall AUPR"), ("hall_f1", "Hall F1"))
    y = np.arange(len(SPECS))
    colors = ["#4C78A8", "#72A0CF", "#F58518", "#FFB55A"]
    for axis, (metric, title) in zip(axes, metrics):
        means = [comparisons[spec]["aggregate"][metric]["mean"] for spec in SPECS]
        stds = [comparisons[spec]["aggregate"][metric]["std"] for spec in SPECS]
        axis.barh(y, means, xerr=stds, color=colors, capsize=3)
        axis.set_yticks(y, [DISPLAY[spec] for spec in SPECS])
        axis.set_xlim(0.0, 1.0)
        axis.invert_yaxis()
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.25)
    fig.suptitle(f"{model}: risk + Union-TopK S (no EV)")
    fig.tight_layout()
    stem = output_dir / f"{model}_union_topk_risk_s_no_ev_metrics"
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _report(model: str, payload: Mapping[str, Any]) -> str:
    lines = [
        f"# {model}: risk + Union-TopK S（无 EV）",
        "",
        "固定 `S_U=Σ_{j∈U}||J_f(z)a_j||₂ / (Σ_{j∈U}||a_j||₂+epsilon)`，"
        "`U=Top32(P_JFFN)∪Top32(Q_hpre_raw_logit_gauss)`。",
        "",
        "| Feature | Dim | AUROC | Hall AUPR | Hall F1 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for spec in SPECS:
        row = payload["comparisons"][spec]["aggregate"]
        lines.append(
            f"| {DISPLAY[spec]} | {payload['feature_dimensions'][spec]} | "
            f"{row['auroc']['mean']:.6f} ± {row['auroc']['std']:.6f} | "
            f"{row['hall_aupr']['mean']:.6f} | {row['hall_f1']['mean']:.6f} |"
        )
    lines += ["", "## S 的增量", ""]
    for added, base in (
        ("old_risk_union_s", "old_risk_only"),
        ("jffn_risk_union_s", "jffn_risk_only"),
    ):
        a = payload["comparisons"][added]["aggregate"]
        b = payload["comparisons"][base]["aggregate"]
        lines.append(
            f"- {DISPLAY[added]} − {DISPLAY[base]}：AUROC "
            f"{a['auroc']['mean']-b['auroc']['mean']:+.6f}；Hall AUPR "
            f"{a['hall_aupr']['mean']-b['hall_aupr']['mean']:+.6f}；Hall F1 "
            f"{a['hall_f1']['mean']-b['hall_f1']['mean']:+.6f}。"
        )
    return "\n".join(lines) + "\n"


def train_model(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    model_root = Path(args.outputs_root) / args.model / EXPERIMENT
    output_dir = model_root / OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    matrices = load_inputs(model_root, args.model)
    train, test = matrices["train"], matrices["test"]
    training = (config.get("jffn_p_comparison") or {}).get("training") or {}
    seeds = [int(value) for value in training.get("seeds", [43, 44, 45])]
    progress_path = output_dir / "training_progress.pt"
    completed: dict[str, dict[int, dict[str, Any]]] = {spec: {} for spec in SPECS}
    probabilities: dict[str, dict[int, np.ndarray]] = {spec: {} for spec in SPECS}
    if args.resume and progress_path.exists():
        progress = torch.load(progress_path, map_location="cpu", weights_only=False)
        if progress.get("schema_version") != SCHEMA_VERSION:
            raise AssertionError("Incompatible risk+S no-EV resume artifact")
        if list(progress["mention_ids"]) != list(test["mention_ids"]):
            raise AssertionError("Mention order mismatch in resume artifact")
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
                print(f"[Risk-S-noEV] reuse {args.model}/{spec}/seed={seed}", flush=True)
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
                },
                progress_path,
            )
            print(
                f"[Risk-S-noEV] {args.model}/{spec}/seed={seed} AUROC="
                f"{metrics['auc']:.6f} HallAUPR="
                f"{metrics['hallucination_positive']['aupr']:.6f}",
                flush=True,
            )

    comparisons = {}
    for spec in SPECS:
        ensemble = np.mean([probabilities[spec][seed] for seed in seeds], axis=0)
        comparisons[spec] = _comparison_result(
            name=DISPLAY[spec],
            aggregate=aggregate_seed_metrics([completed[spec][seed] for seed in seeds]),
            labels=test["y"],
            probabilities=ensemble,
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "definition": {
            "union": "Top32(P_JFFN) union Top32(Q_hpre_raw_logit_gauss) per layer",
            "S": "sum_{j in U} ||J_f(z)a_j||_2 / (sum_{j in U} ||a_j||_2 + epsilon)",
            "EV": "excluded",
        },
        "feature_dimensions": {spec: int(train["x"][spec].shape[1]) for spec in SPECS},
        "sample_audit": {
            "train_mentions": int(train["y"].shape[0]),
            "test_mentions": int(test["y"].shape[0]),
            "train_images": int(np.unique(train["image_ids"]).size),
            "test_images": int(np.unique(test["image_ids"]).size),
            "all_finite": bool(
                all(np.isfinite(value).all() for split in matrices.values() for value in split["x"].values())
            ),
        },
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
    _plot(output_dir, args.model, comparisons)
    (output_dir / f"{args.model}_union_topk_risk_s_no_ev_report.md").write_text(
        _report(args.model, payload), encoding="utf-8"
    )
    print(f"[Risk-S-noEV] wrote {output_dir}", flush=True)


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
        },
        outputs_root / f"{SUMMARY_STEM}.json",
    )
    _write_csv(outputs_root / f"{SUMMARY_STEM}_metrics.csv", rows)
    lines = [
        "# risk + Union-TopK S（无 EV）：两模型汇总",
        "",
        "三 seed、相同三层 MLP 与图片级 split、无标准化、不做 bootstrap。",
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
    print(f"[Risk-S-noEV] wrote {outputs_root / (SUMMARY_STEM + '.md')}", flush=True)


def main() -> None:
    args = parse_args()
    if args.model == "summarize":
        summarize(Path(args.outputs_root))
    else:
        train_model(args)


if __name__ == "__main__":
    main()
