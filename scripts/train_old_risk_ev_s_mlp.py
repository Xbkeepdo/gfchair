#!/usr/bin/env python3
"""Train the targeted old-risk + EV + Jacobian-S three-hidden-layer MLP.

The comparison reuses the completed JFFN shards and the exact official-target
image split.  Its baseline is the already-trained, protocol-identical
old-hpre-risk + EV feature set; no model forward or feature extraction is
repeated.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from features.jffn_experiment import (  # noqa: E402
    atomic_json_save,
    atomic_torch_save,
    load_shards,
)
from scripts.train_torch_probe_feature_sets import (  # noqa: E402
    TorchProbeConfig,
    train_and_evaluate_probe,
)
from utils.config_utils import load_config  # noqa: E402
from utils.io_utils import load_json  # noqa: E402


MODELS = ("llava_1_5_7b", "internvl_2_5_8b")
EXPERIMENT = "COCO4000-INSLEN-OFFICIAL-TARGET"
BASELINE_SPEC = (
    "old_hpre_cos__hpre_raw_logit_gauss__sqrt_matched_state__risk_ev"
)
FEATURE_NAME = "old_hpre_risk_sqrt_plus_ev_plus_jacobian_S"
SCHEMA_VERSION = "old-risk-ev-s-mlp-v1"


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


def _float_array(value: Any) -> np.ndarray:
    return torch.as_tensor(value).float().cpu().numpy().reshape(-1)


def collect_feature_rows(
    model_root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Build 96-D rows: 32 old-risk + 32 EV + 32 aggregate gain S."""
    splits = load_json(str(model_root / "image_splits.json"))
    train_ids = {int(value) for value in splits["train"]}
    test_ids = {int(value) for value in splits["test"]}
    if train_ids & test_ids:
        raise AssertionError("Image split overlaps")

    stores: dict[str, dict[str, list[Any]]] = {
        name: {"x": [], "y": [], "image_ids": [], "mention_ids": []}
        for name in ("train", "test")
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
            risk = _float_array(
                position["risks"]["old_hpre_cos"]["hpre_raw_logit_gauss"][
                    "sqrt_matched_state"
                ]
            )
            ev = _float_array(position["ev"]["hpre_raw_logit_gauss"])
            sensitivity = _float_array(position["jffn_diagnostics"]["gain"])
            if not (risk.shape == ev.shape == sensitivity.shape):
                raise AssertionError(
                    f"Layer mismatch for {mention['mention_id']}: "
                    f"{risk.shape}, {ev.shape}, {sensitivity.shape}"
                )
            row = np.concatenate((risk, ev, sensitivity)).astype(np.float32, copy=False)
            if not np.isfinite(row).all():
                raise ValueError(f"Non-finite row {mention['mention_id']}")
            stores[split]["x"].append(row)
            stores[split]["y"].append(int(mention["label"]))
            stores[split]["image_ids"].append(image_id)
            stores[split]["mention_ids"].append(str(mention["mention_id"]))

    result: dict[str, dict[str, Any]] = {}
    for split, store in stores.items():
        x = np.stack(store["x"])
        y = np.asarray(store["y"], dtype=np.int32)
        image_ids = np.asarray(store["image_ids"], dtype=np.int64)
        if set(np.unique(y)) != {0, 1}:
            raise AssertionError(f"{split} does not contain both labels")
        result[split] = {
            "x": x,
            "y": y,
            "image_ids": image_ids,
            "mention_ids": list(store["mention_ids"]),
        }

    train_x = result["train"]["x"]
    layers = train_x.shape[1] // 3
    block_stats = {}
    for index, name in enumerate(("old_risk", "EV", "S")):
        values = train_x[:, index * layers : (index + 1) * layers].astype(np.float64)
        block_stats[name] = {
            "dimensions": int(values.shape[1]),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        }
    audit = {
        "train_mentions": int(len(result["train"]["y"])),
        "test_mentions": int(len(result["test"]["y"])),
        "train_images": int(np.unique(result["train"]["image_ids"]).size),
        "test_images": int(np.unique(result["test"]["image_ids"]).size),
        "input_dimensions": int(train_x.shape[1]),
        "layers": int(layers),
        "block_order": ["old_hpre_risk_sqrt", "mass_x_cosine_EV", "jacobian_S"],
        "block_stats_train_only": block_stats,
        "all_finite": True,
    }
    return result, audit


def aggregate_seed_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    paths = {
        "auroc": lambda row: row["auc"],
        "accuracy": lambda row: row["accuracy"],
        "real_f1": lambda row: row["real_positive"]["f1"],
        "real_aupr": lambda row: row["real_positive"]["aupr"],
        "hall_f1": lambda row: row["hallucination_positive"]["f1"],
        "hall_aupr": lambda row: row["hallucination_positive"]["aupr"],
    }
    output = {}
    for name, getter in paths.items():
        values = np.asarray([float(getter(row)) for row in rows], dtype=np.float64)
        output[name] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "values": values.tolist(),
        }
    return output


def align_baseline_predictions(
    model_root: Path,
    test: Mapping[str, Any],
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    result_dir = model_root / "results/jffn_p_comparison"
    baseline_results = json.loads((result_dir / "training_results.json").read_text())
    baseline = baseline_results["feature_sets"][BASELINE_SPEC]
    progress = torch.load(
        result_dir / "training_progress.pt", map_location="cpu", weights_only=False
    )
    predictions = progress["predictions"][BASELINE_SPEC]
    expected_ids = list(test["mention_ids"])
    aligned: dict[int, np.ndarray] = {}
    for seed in (43, 44, 45):
        row = predictions[seed]
        lookup = {
            str(mention_id): index
            for index, mention_id in enumerate(row["mention_ids"])
        }
        if len(lookup) != len(row["mention_ids"]):
            raise AssertionError(f"Duplicate baseline mention IDs for seed {seed}")
        try:
            indices = np.asarray([lookup[value] for value in expected_ids], dtype=np.int64)
        except KeyError as exc:
            raise AssertionError(f"Baseline prediction misses {exc}") from exc
        labels = np.asarray(row["labels"], dtype=np.int32)[indices]
        images = np.asarray(row["image_ids"], dtype=np.int64)[indices]
        if not np.array_equal(labels, test["y"]):
            raise AssertionError(f"Baseline labels mismatch for seed {seed}")
        if not np.array_equal(images, test["image_ids"]):
            raise AssertionError(f"Baseline image IDs mismatch for seed {seed}")
        aligned[seed] = np.asarray(row["probabilities"], dtype=np.float32)[indices]
    return aligned, baseline


def paired_image_bootstrap(
    *,
    labels: np.ndarray,
    image_ids: np.ndarray,
    new_probabilities: np.ndarray,
    baseline_probabilities: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    unique_images = np.unique(image_ids)
    indices_by_image = {
        int(image_id): np.flatnonzero(image_ids == image_id)
        for image_id in unique_images
    }
    rng = np.random.default_rng(int(seed))
    auc_delta = np.empty(int(replicates), dtype=np.float64)
    hall_aupr_delta = np.empty(int(replicates), dtype=np.float64)
    for index in range(int(replicates)):
        sampled = rng.choice(unique_images, size=len(unique_images), replace=True)
        rows = np.concatenate([indices_by_image[int(value)] for value in sampled])
        y = labels[rows]
        new = new_probabilities[rows]
        old = baseline_probabilities[rows]
        auc_delta[index] = roc_auc_score(y, new) - roc_auc_score(y, old)
        hall_aupr_delta[index] = average_precision_score(
            1 - y, 1.0 - new
        ) - average_precision_score(1 - y, 1.0 - old)
    return {
        "replicates": int(replicates),
        "new_minus_baseline_auroc": float(
            roc_auc_score(labels, new_probabilities)
            - roc_auc_score(labels, baseline_probabilities)
        ),
        "auroc_ci95": [
            float(np.quantile(auc_delta, 0.025)),
            float(np.quantile(auc_delta, 0.975)),
        ],
        "new_minus_baseline_hall_aupr": float(
            average_precision_score(1 - labels, 1.0 - new_probabilities)
            - average_precision_score(1 - labels, 1.0 - baseline_probabilities)
        ),
        "hall_aupr_ci95": [
            float(np.quantile(hall_aupr_delta, 0.025)),
            float(np.quantile(hall_aupr_delta, 0.975)),
        ],
    }


def ensemble_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(labels, probabilities)),
        "hall_aupr": float(average_precision_score(1 - labels, 1.0 - probabilities)),
    }


def _report(model: str, payload: Mapping[str, Any]) -> str:
    old = payload["baseline"]["aggregate"]
    new = payload["new"]["aggregate"]
    old_e = payload["baseline"]["seed_ensemble"]
    new_e = payload["new"]["seed_ensemble"]
    boot = payload["paired_bootstrap"]
    seeds = payload["new"]["seeds"]
    lines = [
        f"# {model}: old risk + EV + S 三层 MLP",
        "",
        "输入为 32 层 old-hpre raw-logit Gaussian / sqrt-matched-state OT risk、"
        "32 层 mass×cosine EV、32 层 Jacobian sensitivity S，合计 96 维。",
        "",
        "MLP 使用三个隐藏层 [128,64,32]；split、seeds 43/44/45、batch 256、"
        "最多 100 epochs、train-loss checkpoint、train-F1 threshold、无特征归一化"
        "均与现有 JFFN 对照实验一致。",
        "",
        "## 三种子结果",
        "",
        "| Seed | AUROC | Real F1 | Hall F1 | Hall AUPR | Best epoch |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in seeds:
        lines.append(
            f"| {row['seed']} | {row['auc']:.6f} | "
            f"{row['real_positive']['f1']:.6f} | "
            f"{row['hallucination_positive']['f1']:.6f} | "
            f"{row['hallucination_positive']['aupr']:.6f} | "
            f"{row['best_epoch']} |"
        )
    lines += [
        "",
        "## 与 old risk + EV 对比",
        "",
        "| Feature | Mean AUROC | Mean Hall AUPR | Ensemble AUROC | Ensemble Hall AUPR |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| old risk + EV | {old['auroc']['mean']:.6f} ± {old['auroc']['std']:.6f} | "
        f"{old['hall_aupr']['mean']:.6f} ± {old['hall_aupr']['std']:.6f} | "
        f"{old_e['auroc']:.6f} | {old_e['hall_aupr']:.6f} |",
        f"| old risk + EV + S | {new['auroc']['mean']:.6f} ± {new['auroc']['std']:.6f} | "
        f"{new['hall_aupr']['mean']:.6f} ± {new['hall_aupr']['std']:.6f} | "
        f"{new_e['auroc']:.6f} | {new_e['hall_aupr']:.6f} |",
        "",
        f"Seed-ensemble paired image bootstrap：AUROC 差值 "
        f"{boot['new_minus_baseline_auroc']:+.6f}，95% CI "
        f"[{boot['auroc_ci95'][0]:+.6f},{boot['auroc_ci95'][1]:+.6f}]；"
        f"Hall AUPR 差值 {boot['new_minus_baseline_hall_aupr']:+.6f}，95% CI "
        f"[{boot['hall_aupr_ci95'][0]:+.6f},{boot['hall_aupr_ci95'][1]:+.6f}]。",
        "",
        "注意：baseline 直接复用协议完全相同的既有训练及预测，不重新随机训练；"
        "因此比较只新增 S 这一组输入。",
        "",
    ]
    return "\n".join(lines)


def train_model(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    model_root = Path(args.outputs_root) / args.model / EXPERIMENT
    output_dir = model_root / "results/jffn_second_round/old_risk_ev_s_mlp"
    output_dir.mkdir(parents=True, exist_ok=True)
    matrices, audit = collect_feature_rows(model_root)
    train = matrices["train"]
    test = matrices["test"]
    baseline_predictions, baseline_result = align_baseline_predictions(
        model_root, test
    )

    section = config.get("jffn_p_comparison") or {}
    comparison_training = section.get("training") or {}
    inherited = (config.get("training") or {}).get("torch_probe") or {}
    seeds = [int(value) for value in comparison_training.get("seeds", [43, 44, 45])]
    progress_path = output_dir / "training_progress.pt"
    completed: dict[int, dict[str, Any]] = {}
    probabilities: dict[int, np.ndarray] = {}
    if args.resume and progress_path.exists():
        progress = torch.load(progress_path, map_location="cpu", weights_only=False)
        completed = {
            int(row["seed"]): row for row in progress.get("seed_results", ())
        }
        probabilities = {
            int(seed): np.asarray(value, dtype=np.float32)
            for seed, value in (progress.get("probabilities") or {}).items()
        }

    device = _device(args.training_device)
    for seed in seeds:
        if seed in completed and seed in probabilities:
            print(f"[old risk+EV+S] reuse {args.model} seed={seed}", flush=True)
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
            early_stopping_patience=int(
                inherited.get("early_stopping_patience", 10)
            ),
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
        test_probabilities = np.asarray(
            metrics.pop("test_probabilities"), dtype=np.float32
        )
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
            f"[old risk+EV+S] {args.model} seed={seed} "
            f"AUROC={metrics['auc']:.6f} "
            f"HallAUPR={metrics['hallucination_positive']['aupr']:.6f}",
            flush=True,
        )

    rows = [completed[seed] for seed in seeds]
    new_ensemble_prob = np.mean([probabilities[seed] for seed in seeds], axis=0)
    baseline_ensemble_prob = np.mean(
        [baseline_predictions[seed] for seed in seeds], axis=0
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "feature": {
            "name": FEATURE_NAME,
            "definition": "[old_hpre_risk_sqrt_L, mass_x_cosine_EV_L, Jacobian_S_L]",
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
        "baseline": {
            "name": BASELINE_SPEC,
            "aggregate": baseline_result["aggregate"],
            "seed_ensemble": ensemble_metrics(
                test["y"], baseline_ensemble_prob
            ),
        },
        "new": {
            "name": FEATURE_NAME,
            "seeds": rows,
            "aggregate": aggregate_seed_metrics(rows),
            "seed_ensemble": ensemble_metrics(test["y"], new_ensemble_prob),
        },
        "paired_bootstrap": paired_image_bootstrap(
            labels=test["y"],
            image_ids=test["image_ids"],
            new_probabilities=new_ensemble_prob,
            baseline_probabilities=baseline_ensemble_prob,
            replicates=args.bootstrap_replicates,
            seed=20260820,
        ),
    }
    atomic_json_save(payload, output_dir / "results.json")
    (output_dir / f"{args.model}_old_risk_ev_s_mlp_report.md").write_text(
        _report(args.model, payload), encoding="utf-8"
    )
    print(f"[old risk+EV+S] wrote {output_dir}", flush=True)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(outputs_root: Path) -> None:
    payloads = {}
    rows = []
    for model in MODELS:
        path = (
            outputs_root
            / model
            / EXPERIMENT
            / "results/jffn_second_round/old_risk_ev_s_mlp/results.json"
        )
        payload = json.loads(path.read_text())
        payloads[model] = payload
        for feature_group in ("baseline", "new"):
            aggregate = payload[feature_group]["aggregate"]
            ensemble = payload[feature_group]["seed_ensemble"]
            rows.append(
                {
                    "model": model,
                    "feature": payload[feature_group]["name"],
                    "mean_auroc": aggregate["auroc"]["mean"],
                    "std_auroc": aggregate["auroc"]["std"],
                    "mean_hall_aupr": aggregate["hall_aupr"]["mean"],
                    "std_hall_aupr": aggregate["hall_aupr"]["std"],
                    "ensemble_auroc": ensemble["auroc"],
                    "ensemble_hall_aupr": ensemble["hall_aupr"],
                }
            )
    root_json = outputs_root / "old_risk_ev_s_mlp_2model_summary.json"
    atomic_json_save(
        {
            "schema_version": SCHEMA_VERSION,
            "completed_models": list(MODELS),
            "models": payloads,
        },
        root_json,
    )
    write_csv(outputs_root / "old_risk_ev_s_mlp_2model_metrics.csv", rows)
    lines = [
        "# old risk + EV + S：两模型三层 MLP 汇总",
        "",
        "输入为逐层 old-hpre risk、mass×cosine EV 和 Jacobian S，合计 96 维；"
        "三个隐藏层为 [128,64,32]。",
        "",
        "| Model | Feature | Mean AUROC | Mean Hall AUPR | Ensemble AUROC | Ensemble Hall AUPR |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        short = "old risk+EV" if row["feature"] == BASELINE_SPEC else "old risk+EV+S"
        lines.append(
            f"| {row['model']} | {short} | "
            f"{row['mean_auroc']:.6f} ± {row['std_auroc']:.6f} | "
            f"{row['mean_hall_aupr']:.6f} ± {row['std_hall_aupr']:.6f} | "
            f"{row['ensemble_auroc']:.6f} | {row['ensemble_hall_aupr']:.6f} |"
        )
    lines += ["", "## Paired image bootstrap", ""]
    for model, payload in payloads.items():
        boot = payload["paired_bootstrap"]
        lines.append(
            f"- {model}：AUROC Δ={boot['new_minus_baseline_auroc']:+.6f}, "
            f"95% CI [{boot['auroc_ci95'][0]:+.6f},{boot['auroc_ci95'][1]:+.6f}]；"
            f"Hall AUPR Δ={boot['new_minus_baseline_hall_aupr']:+.6f}, "
            f"95% CI [{boot['hall_aupr_ci95'][0]:+.6f},"
            f"{boot['hall_aupr_ci95'][1]:+.6f}]。"
        )
    (outputs_root / "old_risk_ev_s_mlp_2model_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("[old risk+EV+S] wrote two-model summary", flush=True)


def main() -> None:
    args = parse_args()
    if args.model == "summarize":
        summarize(Path(args.outputs_root))
    else:
        train_model(args)


if __name__ == "__main__":
    main()
