#!/usr/bin/env python3
"""Stream completed JFFN shards and summarize full-cohort Jacobian diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.jffn_experiment import load_shards  # noqa: E402
from utils.io_utils import load_json  # noqa: E402


MODELS = ("llava_1_5_7b", "internvl_2_5_8b")
LABEL_NAMES = {0: "hall", 1: "real"}
DIAGNOSTICS = ("input_norm", "response_norm", "gain", "direction_cosine")
SOURCE_STATS = (
    "normalized_entropy",
    "effective_token_count",
    "max_over_uniform",
    "gini",
    "top1_mass",
    "top5_mass",
    "top10_mass",
    "top32_mass",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument(
        "--result-dir",
        default=(
            "outputs/llava_1_5_7b/COCO4000-INSLEN-OFFICIAL-TARGET/"
            "results/jacobian_visual_ffn_validation"
        ),
    )
    parser.add_argument("--risk-start-layer", type=int, default=16)
    return parser.parse_args()


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _metric_summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "q10": float("nan"),
            "q90": float("nan"),
        }
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "median": float(np.median(array)),
        "q10": float(np.quantile(array, 0.1)),
        "q90": float(np.quantile(array, 0.9)),
    }


def _safe_corr(left: Sequence[float], right: Sequence[float]) -> float:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(left_array) & np.isfinite(right_array)
    left_array = left_array[valid]
    right_array = right_array[valid]
    if (
        left_array.size < 3
        or left_array.std() <= 1e-20
        or right_array.std() <= 1e-20
    ):
        return float("nan")
    return float(np.corrcoef(left_array, right_array)[0, 1])


def _evaluate_scores(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "hall_aupr": float(average_precision_score(labels, scores)),
    }


def _fit_logistic(
    *,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
) -> dict[str, float]:
    estimator = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=43,
        ),
    )
    estimator.fit(train_x, train_y)
    score = estimator.predict_proba(test_x)[:, 1]
    return _evaluate_scores(test_y, score)


def _source_risk(row: dict[str, Any], source: str) -> np.ndarray:
    value = row["risks"][source]["hpre_raw_logit_gauss"]["sqrt_matched_state"]
    return value.float().numpy()


def analyze_model(
    *, model: str, output_root: Path, risk_start_layer: int
) -> dict[str, Any]:
    experiment_root = (
        output_root / model / "COCO4000-INSLEN-OFFICIAL-TARGET"
    )
    shard_dir = experiment_root / "results" / "jffn_p_comparison" / "shards"
    splits = load_json(str(experiment_root / "image_splits.json"))
    train_ids = {int(value) for value in splits["train"]}
    test_ids = {int(value) for value in splits["test"]}

    accumulators: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    correlation_values: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    feature_rows: list[dict[str, Any]] = []
    conflict_targets: list[str] = []
    duplicate_mentions = 0
    total_positions = 0
    total_mentions = 0
    reconstruction_errors: list[float] = []
    reconstruction_cosines: list[float] = []
    formal_validation_flags: list[bool] = []

    for shard in load_shards(shard_dir):
        positions = {str(row["target_key"]): row for row in shard["positions"]}
        mention_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for mention in shard["sample_table"]:
            mention_groups[str(mention["target_key"])].append(mention)
        total_positions += len(positions)
        total_mentions += sum(len(values) for values in mention_groups.values())
        for target_key, mentions in mention_groups.items():
            labels = {int(value["label"]) for value in mentions}
            if len(labels) != 1:
                conflict_targets.append(target_key)
                continue
            label = labels.pop()
            label_name = LABEL_NAMES[label]
            row = positions[target_key]
            image_id = int(row["image_id"])
            diagnostic = row["jffn_diagnostics"]
            layers = int(diagnostic["gain"].numel())
            if len(mentions) > 1:
                duplicate_mentions += len(mentions) - 1
            repeats = len(mentions)
            for layer_index in range(layers):
                for metric in DIAGNOSTICS:
                    value = float(diagnostic[metric][layer_index])
                    accumulators[(label_name, layer_index + 1, metric)].extend(
                        [value] * repeats
                    )
                    correlation_values[(label_name, layer_index + 1)][metric].extend(
                        [value] * repeats
                    )
                for metric in SOURCE_STATS:
                    value = float(
                        row["source_statistics"]["new_jffn"][metric][layer_index]
                    )
                    accumulators[
                        (label_name, layer_index + 1, f"p_{metric}")
                    ].extend([value] * repeats)
            reconstruction_errors.extend(
                float(value)
                for value in diagnostic[
                    "attention_reconstruction_relative_error"
                ]
            )
            reconstruction_cosines.extend(
                float(value)
                for value in diagnostic["attention_reconstruction_cosine"]
            )
            formal_validation_flags.append(bool(row["jffn_validation_computed"]))

            start = max(int(risk_start_layer) - 1, 0)
            old_risk = _source_risk(row, "old_hpre_cos")[start:]
            new_risk = _source_risk(row, "new_jffn")[start:]
            diag_matrix = np.stack(
                [diagnostic[metric].float().numpy()[start:] for metric in DIAGNOSTICS],
                axis=0,
            ).reshape(-1)
            for mention in mentions:
                feature_rows.append(
                    {
                        "image_id": image_id,
                        "label_hall": int(label == 0),
                        "old_risk": old_risk,
                        "new_risk": new_risk,
                        "diagnostics": diag_matrix,
                    }
                )
        del shard, positions, mention_groups

    aggregate_rows: list[dict[str, Any]] = []
    for (label, layer, metric), values in sorted(accumulators.items()):
        aggregate_rows.append(
            {
                "model": model,
                "label": label,
                "layer": int(layer),
                "metric": metric,
                **_metric_summary(values),
            }
        )

    correlation_rows: list[dict[str, Any]] = []
    for (label, layer), values in sorted(correlation_values.items()):
        correlation_rows.append(
            {
                "model": model,
                "label": label,
                "layer": int(layer),
                "input_response_pearson": _safe_corr(
                    values["input_norm"], values["response_norm"]
                ),
                "input_gain_pearson": _safe_corr(
                    values["input_norm"], values["gain"]
                ),
                "response_gain_pearson": _safe_corr(
                    values["response_norm"], values["gain"]
                ),
            }
        )

    labels_array = np.asarray(
        [int(row["label_hall"]) for row in feature_rows], dtype=np.int64
    )
    images_array = np.asarray(
        [int(row["image_id"]) for row in feature_rows], dtype=np.int64
    )
    train_mask = np.asarray([image_id in train_ids for image_id in images_array])
    test_mask = np.asarray([image_id in test_ids for image_id in images_array])
    feature_sets = {
        "old_hpre_risk": np.stack([row["old_risk"] for row in feature_rows]),
        "new_jffn_risk": np.stack([row["new_risk"] for row in feature_rows]),
        "old_plus_new_risk": np.stack(
            [
                np.concatenate((row["old_risk"], row["new_risk"]))
                for row in feature_rows
            ]
        ),
        "jffn_diagnostics_I_R_S_D": np.stack(
            [row["diagnostics"] for row in feature_rows]
        ),
        "old_risk_plus_jffn_diagnostics": np.stack(
            [
                np.concatenate((row["old_risk"], row["diagnostics"]))
                for row in feature_rows
            ]
        ),
    }
    auxiliary_detection = {}
    for name, matrix in feature_sets.items():
        auxiliary_detection[name] = _fit_logistic(
            train_x=matrix[train_mask],
            train_y=labels_array[train_mask],
            test_x=matrix[test_mask],
            test_y=labels_array[test_mask],
        )

    # Univariate layer-averaged diagnostics, with Hall as the positive class.
    univariate_detection = {}
    diagnostics_matrix = feature_sets["jffn_diagnostics_I_R_S_D"].reshape(
        len(feature_rows), len(DIAGNOSTICS), -1
    )
    for metric_index, metric in enumerate(DIAGNOSTICS):
        score = diagnostics_matrix[:, metric_index, :].mean(axis=1)[test_mask]
        raw = _evaluate_scores(labels_array[test_mask], score)
        inverted = _evaluate_scores(labels_array[test_mask], -score)
        univariate_detection[metric] = {
            "raw_higher_means_hall": raw,
            "inverted_higher_means_hall": inverted,
            "best_orientation_auroc": max(raw["auroc"], inverted["auroc"]),
        }

    return {
        "model": model,
        "aggregate_rows": aggregate_rows,
        "correlation_rows": correlation_rows,
        "audit": {
            "positions": total_positions,
            "mentions": total_mentions,
            "feature_rows": len(feature_rows),
            "duplicate_mentions": duplicate_mentions,
            "conflict_targets": conflict_targets,
            "all_formal_rows_validation_computed": bool(
                formal_validation_flags and all(formal_validation_flags)
            ),
            "formal_validation_true_count": int(sum(formal_validation_flags)),
            "attention_reconstruction_relative_error": _metric_summary(
                reconstruction_errors
            ),
            "attention_reconstruction_cosine": _metric_summary(
                reconstruction_cosines
            ),
            "train_mentions": int(train_mask.sum()),
            "test_mentions": int(test_mask.sum()),
        },
        "auxiliary_detection": auxiliary_detection,
        "univariate_detection": univariate_detection,
    }


def main() -> None:
    args = parse_args()
    output_root = Path(args.outputs_root).resolve()
    result_dir = Path(args.result_dir).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    results = [
        analyze_model(
            model=model,
            output_root=output_root,
            risk_start_layer=int(args.risk_start_layer),
        )
        for model in MODELS
    ]
    aggregate_rows = [row for result in results for row in result["aggregate_rows"]]
    correlation_rows = [
        row for result in results for row in result["correlation_rows"]
    ]
    payload = {
        "protocol": "completed_models_full_cohort_jffn_diagnostic_v1",
        "models": {
            result["model"]: {
                key: value
                for key, value in result.items()
                if key not in {"model", "aggregate_rows", "correlation_rows"}
            }
            for result in results
        },
    }
    _write_csv(result_dir / "jacobian_visual_ffn_full_cohort.csv", aggregate_rows)
    _write_csv(
        result_dir / "jacobian_visual_ffn_full_cohort_correlations.csv",
        correlation_rows,
    )
    path = result_dir / "jacobian_visual_ffn_full_cohort_metrics.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(f"[cohort] wrote {path}")


if __name__ == "__main__":
    main()
