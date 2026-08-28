#!/usr/bin/env python3
"""Train our JFFN-era feature blocks with the native SVAR/MetaToken heads.

This is a classifier-head swap only.  It reuses the completed JFFN shards,
the official image split, and the exact mention cohort.  Two inputs are used:

* old hpre OT risk + mass×cosine EV (64 dimensions for 32-layer models)
* the same input plus Jacobian sensitivity S (96 dimensions)

Each input is trained with SVAR's one-hidden-layer 248-unit detector and with
MetaToken's standardized LR and 100-estimator GB heads.  No model forward or
feature extraction is performed.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from detection.baselines import (  # noqa: E402
    SVARMLP,
    build_metatoken_classifier,
    evaluate_detection_scores,
    raw_labels_to_hallucination_targets,
    select_detection_threshold,
    sklearn_hallucination_scores,
    torch_hallucination_scores,
    train_torch_detector,
)
from features.jffn_experiment import atomic_json_save, atomic_torch_save  # noqa: E402
from scripts.train_old_risk_ev_s_mlp import (  # noqa: E402
    BASELINE_SPEC,
    EXPERIMENT,
    MODELS,
    align_baseline_predictions,
    collect_feature_rows,
)
from utils.config_utils import load_config  # noqa: E402


SCHEMA_VERSION = "ours-native-baseline-heads-v1"
FEATURE_SPECS = {
    "old_risk_ev": {
        "display": "old risk + EV",
        "blocks": ("old_hpre_risk_sqrt", "mass_x_cosine_EV"),
    },
    "old_risk_ev_s": {
        "display": "old risk + EV + Jacobian S",
        "blocks": (
            "old_hpre_risk_sqrt",
            "mass_x_cosine_EV",
            "jacobian_S",
        ),
    },
}
HEADS = (
    "svar_native",
    "svar_native_h128",
    "metatoken_lr",
    "metatoken_gb",
)
SEEDS = (43, 44, 45)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=(*MODELS, "summarize"))
    parser.add_argument("--config", default="configs/model_configs_unified.yaml")
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--training-device", default="auto")
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def feature_variants(matrix: np.ndarray, layers: int) -> dict[str, np.ndarray]:
    """Slice the canonical [risk, EV, S] matrix into the two experiments."""

    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 3 * int(layers):
        raise ValueError(
            f"Expected [N,{3 * int(layers)}] risk/EV/S matrix, got {values.shape}"
        )
    result = {
        "old_risk_ev": values[:, : 2 * int(layers)],
        "old_risk_ev_s": values,
    }
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError("Native-head input contains NaN/Inf")
    return result


def _threshold_reports(
    *,
    train_labels: np.ndarray,
    train_hall_scores: np.ndarray,
    test_labels: np.ndarray,
    test_hall_scores: np.ndarray,
    selected_threshold: float,
) -> dict[str, Any]:
    reports = {}
    for name, threshold in (
        ("fixed_0.5", 0.5),
        ("train_f1", float(selected_threshold)),
    ):
        reports[name] = {
            "threshold": float(threshold),
            "train_metrics": evaluate_detection_scores(
                train_labels,
                train_hall_scores,
                threshold,
                positive_class="real",
            ),
            "test_metrics": evaluate_detection_scores(
                test_labels,
                test_hall_scores,
                threshold,
                positive_class="real",
            ),
        }
    return reports


def _metric_payload(
    *,
    seed: int,
    train_labels: np.ndarray,
    train_hall_scores: np.ndarray,
    test_labels: np.ndarray,
    test_hall_scores: np.ndarray,
    threshold: float,
    input_dim: int,
    head_config: Mapping[str, Any],
) -> dict[str, Any]:
    train_metrics = evaluate_detection_scores(
        train_labels,
        train_hall_scores,
        threshold,
        positive_class="real",
    )
    test_metrics = evaluate_detection_scores(
        test_labels,
        test_hall_scores,
        threshold,
        positive_class="real",
    )
    return {
        "seed": int(seed),
        "input_dim": int(input_dim),
        "threshold": float(threshold),
        "threshold_score_class": "real",
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "threshold_reports": _threshold_reports(
            train_labels=train_labels,
            train_hall_scores=train_hall_scores,
            test_labels=test_labels,
            test_hall_scores=test_hall_scores,
            selected_threshold=threshold,
        ),
        "head_config": dict(head_config),
    }


def _train_svar(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    cfg: Mapping[str, Any],
    device: torch.device,
    checkpoint_path: Path,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    hidden_dim = int(cfg.get("hidden_dim", 248))
    learning_rate = float(cfg.get("learning_rate", 1e-3))
    batch_size = int(cfg.get("batch_size", 32))
    epochs = int(cfg.get("epochs", 50))
    _seed_everything(seed)
    model = SVARMLP(X_train.shape[1], hidden_dim=hidden_dim)
    trained = train_torch_detector(
        model=model,
        X_train=X_train,
        raw_y_train=y_train,
        # Strict 8:2 has no validation; the native baseline adapter passes the
        # training rows here for train-loss diagnostics and train-F1 threshold.
        X_val=X_train,
        raw_y_val=y_train,
        X_test=X_test,
        raw_y_test=y_test,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        device=str(device),
        weight_decay=0.0,
        weighted_sampler=bool(cfg.get("weighted_sampler", False)),
        standardize=bool(cfg.get("standardize", False)),
        seed=seed,
        positive_class="real",
        strict_82_no_validation=True,
    )
    train_hall = torch_hallucination_scores(model, X_train, device)
    test_hall = torch_hallucination_scores(model, X_test, device)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(
        {
            "schema_version": SCHEMA_VERSION,
            "state_dict": trained.state_dict,
            "input_dim": int(X_train.shape[1]),
            "hidden_dim": hidden_dim,
            "seed": int(seed),
        },
        checkpoint_path,
    )
    metrics = _metric_payload(
        seed=seed,
        train_labels=y_train,
        train_hall_scores=train_hall,
        test_labels=y_test,
        test_hall_scores=test_hall,
        threshold=trained.threshold,
        input_dim=X_train.shape[1],
        head_config={
            "family": "SVARMLP",
            "structure": f"Linear({X_train.shape[1]},{hidden_dim})-ReLU-Linear({hidden_dim},2)",
            "hidden_dim": hidden_dim,
            "optimizer": "Adam",
            "learning_rate": learning_rate,
            "weight_decay": 0.0,
            "batch_size": batch_size,
            "epochs": epochs,
            "loss": "CrossEntropyLoss",
            "standardize": bool(cfg.get("standardize", False)),
            "weighted_sampler": bool(cfg.get("weighted_sampler", False)),
            "checkpoint_selection": "last_epoch_under_strict_82",
        },
    )
    metrics["epochs_completed"] = len(trained.history)
    metrics["history"] = trained.history
    return metrics, 1.0 - train_hall, 1.0 - test_hall


def _train_metatoken(
    *,
    kind: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    checkpoint_path: Path,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    classifier = build_metatoken_classifier(kind, seed=seed)
    classifier.fit(X_train, raw_labels_to_hallucination_targets(y_train))
    train_hall = sklearn_hallucination_scores(classifier, X_train)
    test_hall = sklearn_hallucination_scores(classifier, X_test)
    threshold = select_detection_threshold(
        y_train,
        train_hall,
        positive_class="real",
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(classifier, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(checkpoint_path)
    family = "LogisticRegression(lbfgs)" if kind == "lr" else "GradientBoostingClassifier"
    metrics = _metric_payload(
        seed=seed,
        train_labels=y_train,
        train_hall_scores=train_hall,
        test_labels=y_test,
        test_hall_scores=test_hall,
        threshold=threshold,
        input_dim=X_train.shape[1],
        head_config={
            "family": family,
            "pipeline": "StandardScaler -> classifier",
            "standardize": True,
            "solver": "lbfgs" if kind == "lr" else None,
            "max_iter": 2000 if kind == "lr" else None,
            "n_estimators": 100 if kind == "gb" else None,
            "checkpoint_selection": "single_final_fit",
        },
    )
    return metrics, 1.0 - train_hall, 1.0 - test_hall


def aggregate_seed_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    getters = {
        "auroc": lambda row: row["test_metrics"]["real_positive"]["auc"],
        "accuracy": lambda row: row["test_metrics"]["accuracy"],
        "real_f1": lambda row: row["test_metrics"]["real_positive"]["f1"],
        "real_aupr": lambda row: row["test_metrics"]["real_positive"]["aupr"],
        "hall_f1": lambda row: row["test_metrics"]["hallucination_positive"]["f1"],
        "hall_aupr": lambda row: row["test_metrics"]["hallucination_positive"]["aupr"],
    }
    output = {}
    for name, getter in getters.items():
        values = np.asarray([float(getter(row)) for row in rows], dtype=np.float64)
        output[name] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "values": values.tolist(),
        }
    return output


def ranking_metrics(labels: np.ndarray, real_probabilities: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(labels, real_probabilities)),
        "real_aupr": float(average_precision_score(labels, real_probabilities)),
        "hall_aupr": float(
            average_precision_score(1 - labels, 1.0 - real_probabilities)
        ),
    }


def _align_probabilities(
    *,
    source_ids: Sequence[str],
    source_values: np.ndarray,
    target_ids: Sequence[str],
) -> np.ndarray:
    lookup = {str(value): index for index, value in enumerate(source_ids)}
    if len(lookup) != len(source_ids):
        raise AssertionError("Reference prediction contains duplicate mention IDs")
    try:
        indices = np.asarray([lookup[str(value)] for value in target_ids], dtype=np.int64)
    except KeyError as exc:
        raise AssertionError(f"Reference prediction misses mention {exc}") from exc
    return np.asarray(source_values, dtype=np.float32)[indices]


def load_torch_references(
    model_root: Path,
    test: Mapping[str, Any],
) -> dict[str, Any]:
    risk_ev_predictions, risk_ev_result = align_baseline_predictions(model_root, test)
    s_dir = model_root / "results/jffn_second_round/old_risk_ev_s_mlp"
    s_progress = torch.load(
        s_dir / "training_progress.pt", map_location="cpu", weights_only=False
    )
    s_result = json.loads((s_dir / "results.json").read_text())
    source_ids = [str(value) for value in s_progress["mention_ids"]]
    s_predictions = {
        int(seed): _align_probabilities(
            source_ids=source_ids,
            source_values=np.asarray(values, dtype=np.float32),
            target_ids=test["mention_ids"],
        )
        for seed, values in s_progress["probabilities"].items()
    }
    if not np.array_equal(
        _align_probabilities(
            source_ids=source_ids,
            source_values=np.asarray(s_progress["labels"], dtype=np.float32),
            target_ids=test["mention_ids"],
        ).astype(np.int32),
        test["y"],
    ):
        raise AssertionError("96-D Torch reference labels do not align")
    return {
        "old_risk_ev": {
            "name": BASELINE_SPEC,
            "aggregate": risk_ev_result["aggregate"],
            "probabilities": risk_ev_predictions,
        },
        "old_risk_ev_s": {
            "name": s_result["new"]["name"],
            "aggregate": s_result["new"]["aggregate"],
            "probabilities": s_predictions,
        },
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def _report(model: str, payload: Mapping[str, Any]) -> str:
    lines = [
        f"# {model}: Ours 特征 × S-VAR/MetaToken 原生分类头",
        "",
        "只替换分类头；特征、InsLen cohort、图片级 8:2 split 和 seeds 43/44/45 不变。",
        "S-VAR 保留单隐藏层 248、50 epochs、Adam；MetaToken 保留 StandardScaler+LR/GB-100。",
        "",
        "## 三种子均值",
        "",
        "| Feature | Head | AUROC | Hall F1 | Hall AUPR | Ensemble AUROC |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for feature_name in FEATURE_SPECS:
        feature = payload["feature_sets"][feature_name]
        reference = feature["current_torch_reference"]
        lines.append(
            f"| {FEATURE_SPECS[feature_name]['display']} | current 3-layer Torch MLP | "
            f"{reference['aggregate']['auroc']['mean']:.6f} ± "
            f"{reference['aggregate']['auroc']['std']:.6f} | "
            f"{reference['aggregate']['hall_f1']['mean']:.6f} | "
            f"{reference['aggregate']['hall_aupr']['mean']:.6f} | "
            f"{reference['seed_ensemble']['auroc']:.6f} |"
        )
        for head in HEADS:
            result = feature["heads"][head]
            aggregate = result["aggregate"]
            lines.append(
                f"| {FEATURE_SPECS[feature_name]['display']} | {head} | "
                f"{aggregate['auroc']['mean']:.6f} ± {aggregate['auroc']['std']:.6f} | "
                f"{aggregate['hall_f1']['mean']:.6f} | "
                f"{aggregate['hall_aupr']['mean']:.6f} | "
                f"{result['seed_ensemble']['auroc']:.6f} |"
            )
    lines += ["", "## 相对当前三层 MLP", ""]
    for feature_name in FEATURE_SPECS:
        feature = payload["feature_sets"][feature_name]
        for head in HEADS:
            delta = feature["heads"][head]["delta_vs_current_torch"]
            lines.append(
                f"- {FEATURE_SPECS[feature_name]['display']} / {head}："
                f"mean AUROC {delta['mean_auroc']:+.6f}，"
                f"mean Hall-AUPR {delta['mean_hall_aupr']:+.6f}，"
                f"ensemble AUROC {delta['ensemble_auroc']:+.6f}。"
            )
    lines += ["", "## 加入 S 的头内增量", ""]
    for head, delta in payload["s_increment_by_head"].items():
        lines.append(
            f"- {head}：mean AUROC {delta['mean_auroc']:+.6f}，"
            f"mean Hall-AUPR {delta['mean_hall_aupr']:+.6f}，"
            f"ensemble AUROC {delta['ensemble_auroc']:+.6f}。"
        )
    return "\n".join(lines) + "\n"


def train_model(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    model_root = Path(args.outputs_root) / args.model / EXPERIMENT
    output_dir = (
        model_root / "results/jffn_second_round/ours_with_native_baseline_heads"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    matrices, base_audit = collect_feature_rows(model_root)
    layers = int(base_audit["layers"])
    train_variants = feature_variants(matrices["train"]["x"], layers)
    test_variants = feature_variants(matrices["test"]["x"], layers)
    references = load_torch_references(model_root, matrices["test"])

    baseline_cfg = ((config.get("feature_extraction") or {}).get("baseline") or {})
    svar_cfg = dict(baseline_cfg.get("svar") or {})
    seeds = tuple(
        int(value)
        for value in ((config.get("training") or {}).get("baseline") or {}).get(
            "seeds", SEEDS
        )
    )
    if seeds != SEEDS:
        raise ValueError(f"Expected fair-comparison seeds {SEEDS}, got {seeds}")
    progress_path = output_dir / "training_progress.pt"
    progress: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "seed_results": {},
        "train_real_probabilities": {},
        "test_real_probabilities": {},
        "labels": matrices["test"]["y"],
        "image_ids": matrices["test"]["image_ids"],
        "mention_ids": matrices["test"]["mention_ids"],
    }
    if args.resume and progress_path.exists():
        loaded = torch.load(progress_path, map_location="cpu", weights_only=False)
        if loaded.get("schema_version") == SCHEMA_VERSION and loaded.get("model") == args.model:
            progress = loaded

    device = _device(args.training_device)
    for feature_name in FEATURE_SPECS:
        X_train = train_variants[feature_name]
        X_test = test_variants[feature_name]
        for head in HEADS:
            for seed in seeds:
                key = f"{feature_name}__{head}__seed{seed}"
                if (
                    key in progress["seed_results"]
                    and key in progress["test_real_probabilities"]
                ):
                    print(f"[NativeHead] reuse {args.model} {key}", flush=True)
                    continue
                checkpoint_root = output_dir / "checkpoints" / feature_name / head
                if head.startswith("svar_native"):
                    effective_svar_cfg = dict(svar_cfg)
                    if head == "svar_native_h128":
                        effective_svar_cfg["hidden_dim"] = 128
                    metrics, train_probs, test_probs = _train_svar(
                        X_train=X_train,
                        y_train=matrices["train"]["y"],
                        X_test=X_test,
                        y_test=matrices["test"]["y"],
                        seed=seed,
                        cfg=effective_svar_cfg,
                        device=device,
                        checkpoint_path=checkpoint_root / f"seed{seed}.pt",
                    )
                else:
                    kind = "lr" if head.endswith("lr") else "gb"
                    metrics, train_probs, test_probs = _train_metatoken(
                        kind=kind,
                        X_train=X_train,
                        y_train=matrices["train"]["y"],
                        X_test=X_test,
                        y_test=matrices["test"]["y"],
                        seed=seed,
                        checkpoint_path=checkpoint_root / f"seed{seed}.pkl",
                    )
                progress["seed_results"][key] = metrics
                progress["train_real_probabilities"][key] = train_probs.astype(np.float32)
                progress["test_real_probabilities"][key] = test_probs.astype(np.float32)
                atomic_torch_save(progress, progress_path)
                print(
                    f"[NativeHead] {args.model} {key} "
                    f"AUROC={metrics['test_metrics']['real_positive']['auc']:.6f} "
                    f"HallAUPR={metrics['test_metrics']['hallucination_positive']['aupr']:.6f}",
                    flush=True,
                )

    feature_sets: dict[str, Any] = {}
    for feature_name in FEATURE_SPECS:
        reference = references[feature_name]
        reference_ensemble_prob = np.mean(
            [reference["probabilities"][seed] for seed in seeds], axis=0
        )
        reference_ensemble = ranking_metrics(
            matrices["test"]["y"], reference_ensemble_prob
        )
        head_results = {}
        for head in HEADS:
            rows = [
                progress["seed_results"][f"{feature_name}__{head}__seed{seed}"]
                for seed in seeds
            ]
            ensemble_prob = np.mean(
                [
                    np.asarray(
                        progress["test_real_probabilities"][
                            f"{feature_name}__{head}__seed{seed}"
                        ],
                        dtype=np.float32,
                    )
                    for seed in seeds
                ],
                axis=0,
            )
            aggregate = aggregate_seed_metrics(rows)
            ensemble = ranking_metrics(matrices["test"]["y"], ensemble_prob)
            head_results[head] = {
                "seeds": rows,
                "aggregate": aggregate,
                "seed_ensemble": ensemble,
                "delta_vs_current_torch": {
                    "mean_auroc": float(
                        aggregate["auroc"]["mean"]
                        - reference["aggregate"]["auroc"]["mean"]
                    ),
                    "mean_hall_aupr": float(
                        aggregate["hall_aupr"]["mean"]
                        - reference["aggregate"]["hall_aupr"]["mean"]
                    ),
                    "ensemble_auroc": float(
                        ensemble["auroc"] - reference_ensemble["auroc"]
                    ),
                    "ensemble_hall_aupr": float(
                        ensemble["hall_aupr"] - reference_ensemble["hall_aupr"]
                    ),
                },
            }
        feature_sets[feature_name] = {
            "display_name": FEATURE_SPECS[feature_name]["display"],
            "blocks": list(FEATURE_SPECS[feature_name]["blocks"]),
            "input_dimensions": int(train_variants[feature_name].shape[1]),
            "current_torch_reference": {
                "name": reference["name"],
                "aggregate": reference["aggregate"],
                "seed_ensemble": reference_ensemble,
            },
            "heads": head_results,
        }

    s_increment = {}
    for head in HEADS:
        base = feature_sets["old_risk_ev"]["heads"][head]
        plus = feature_sets["old_risk_ev_s"]["heads"][head]
        s_increment[head] = {
            "mean_auroc": float(
                plus["aggregate"]["auroc"]["mean"]
                - base["aggregate"]["auroc"]["mean"]
            ),
            "mean_hall_aupr": float(
                plus["aggregate"]["hall_aupr"]["mean"]
                - base["aggregate"]["hall_aupr"]["mean"]
            ),
            "ensemble_auroc": float(
                plus["seed_ensemble"]["auroc"] - base["seed_ensemble"]["auroc"]
            ),
            "ensemble_hall_aupr": float(
                plus["seed_ensemble"]["hall_aupr"]
                - base["seed_ensemble"]["hall_aupr"]
            ),
        }

    payload = {
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "experiment": {
            "description": "classifier-head swap only",
            "cohort": "existing InsLen official target mentions",
            "split": "existing image-level strict 80/20",
            "seeds": list(seeds),
            "target_threshold": "train Real-F1; fixed 0.5 also stored",
        },
        "sample_audit": {
            **base_audit,
            "variant_dimensions": {
                name: int(value.shape[1]) for name, value in train_variants.items()
            },
        },
        "feature_sets": feature_sets,
        "s_increment_by_head": s_increment,
    }
    atomic_json_save(_json_ready(payload), output_dir / "results.json")
    report_path = output_dir / f"{args.model}_ours_native_heads_report.md"
    report_path.write_text(_report(args.model, payload), encoding="utf-8")
    print(f"[NativeHead] wrote {output_dir}", flush=True)


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


def summarize(outputs_root: Path) -> None:
    payloads = {}
    rows = []
    for model in MODELS:
        path = (
            outputs_root
            / model
            / EXPERIMENT
            / "results/jffn_second_round/ours_with_native_baseline_heads/results.json"
        )
        payload = json.loads(path.read_text())
        payloads[model] = payload
        for feature_name, feature in payload["feature_sets"].items():
            reference = feature["current_torch_reference"]
            rows.append(
                {
                    "model": model,
                    "feature": feature_name,
                    "head": "current_3layer_torch_mlp",
                    "mean_auroc": reference["aggregate"]["auroc"]["mean"],
                    "std_auroc": reference["aggregate"]["auroc"]["std"],
                    "mean_hall_f1": reference["aggregate"]["hall_f1"]["mean"],
                    "mean_hall_aupr": reference["aggregate"]["hall_aupr"]["mean"],
                    "ensemble_auroc": reference["seed_ensemble"]["auroc"],
                    "ensemble_hall_aupr": reference["seed_ensemble"]["hall_aupr"],
                }
            )
            for head, result in feature["heads"].items():
                aggregate = result["aggregate"]
                rows.append(
                    {
                        "model": model,
                        "feature": feature_name,
                        "head": head,
                        "mean_auroc": aggregate["auroc"]["mean"],
                        "std_auroc": aggregate["auroc"]["std"],
                        "mean_hall_f1": aggregate["hall_f1"]["mean"],
                        "mean_hall_aupr": aggregate["hall_aupr"]["mean"],
                        "ensemble_auroc": result["seed_ensemble"]["auroc"],
                        "ensemble_hall_aupr": result["seed_ensemble"]["hall_aupr"],
                    }
                )
    atomic_json_save(
        {
            "schema_version": SCHEMA_VERSION,
            "completed_models": list(MODELS),
            "models": payloads,
        },
        outputs_root / "ours_native_baseline_heads_2model_summary.json",
    )
    _write_csv(outputs_root / "ours_native_baseline_heads_2model_metrics.csv", rows)
    lines = [
        "# Ours 特征 × S-VAR/MetaToken 原生分类头：两模型汇总",
        "",
        "仅更换分类头；两种特征、样本、图片级 split 和 seeds 均严格对齐。",
        "",
        "| Model | Feature | Head | Mean AUROC | Hall F1 | Hall AUPR | Ensemble AUROC |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['feature']} | {row['head']} | "
            f"{row['mean_auroc']:.6f} ± {row['std_auroc']:.6f} | "
            f"{row['mean_hall_f1']:.6f} | {row['mean_hall_aupr']:.6f} | "
            f"{row['ensemble_auroc']:.6f} |"
        )
    lines += ["", "## 加入 S 的头内增量", ""]
    for model, payload in payloads.items():
        for head, delta in payload["s_increment_by_head"].items():
            lines.append(
                f"- {model} / {head}：mean AUROC {delta['mean_auroc']:+.6f}，"
                f"Hall-AUPR {delta['mean_hall_aupr']:+.6f}，"
                f"ensemble AUROC {delta['ensemble_auroc']:+.6f}。"
            )
    (outputs_root / "ours_native_baseline_heads_2model_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("[NativeHead] wrote two-model summary", flush=True)


def main() -> None:
    args = parse_args()
    if args.model == "summarize":
        summarize(Path(args.outputs_root))
    else:
        train_model(args)


if __name__ == "__main__":
    main()
