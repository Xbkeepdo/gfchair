#!/usr/bin/env python3
"""Test Jacobian scalar value conditional on incoming visual-write magnitude."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.jffn_experiment import load_shards  # noqa: E402
from utils.io_utils import load_json  # noqa: E402


MODELS = ("llava_1_5_7b", "internvl_2_5_8b")
EXPERIMENT = "COCO4000-INSLEN-OFFICIAL-TARGET"
PRIMARY_START_LAYER = 16
SEED = 20260819


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    return parser.parse_args()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "hall_aupr": float(average_precision_score(labels, scores)),
    }


def _fit_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
) -> tuple[Any, np.ndarray]:
    estimator = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=3000,
            random_state=43,
            solver="lbfgs",
        ),
    )
    estimator.fit(train_x, train_y)
    return estimator, estimator.predict_proba(test_x)[:, 1]


def _paired_image_bootstrap(
    *,
    labels: np.ndarray,
    images: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    unique_images, inverse = np.unique(images, return_inverse=True)
    rng = np.random.default_rng(seed)
    auroc_delta: list[float] = []
    aupr_delta: list[float] = []
    for _ in range(int(replicates)):
        sampled = rng.integers(0, len(unique_images), size=len(unique_images))
        image_weights = np.bincount(sampled, minlength=len(unique_images)).astype(
            np.float64
        )
        weights = image_weights[inverse]
        if not np.any((labels == 0) & (weights > 0)) or not np.any(
            (labels == 1) & (weights > 0)
        ):
            continue
        auroc_delta.append(
            float(
                roc_auc_score(labels, left, sample_weight=weights)
                - roc_auc_score(labels, right, sample_weight=weights)
            )
        )
        aupr_delta.append(
            float(
                average_precision_score(labels, left, sample_weight=weights)
                - average_precision_score(labels, right, sample_weight=weights)
            )
        )
    return {
        "replicates_requested": int(replicates),
        "replicates_valid": len(auroc_delta),
        "auroc_difference": float(
            roc_auc_score(labels, left) - roc_auc_score(labels, right)
        ),
        "auroc_ci95": [
            float(np.quantile(auroc_delta, 0.025)),
            float(np.quantile(auroc_delta, 0.975)),
        ],
        "hall_aupr_difference": float(
            average_precision_score(labels, left)
            - average_precision_score(labels, right)
        ),
        "hall_aupr_ci95": [
            float(np.quantile(aupr_delta, 0.025)),
            float(np.quantile(aupr_delta, 0.975)),
        ],
    }


def _old_risk(row: Mapping[str, Any], start: int) -> np.ndarray:
    return (
        row["risks"]["old_hpre_cos"]["hpre_raw_logit_gauss"]
        ["sqrt_matched_state"][start:]
        .float()
        .numpy()
    )


def collect_rows(model_root: Path) -> list[dict[str, Any]]:
    shard_dir = model_root / "results/jffn_p_comparison/shards"
    start = PRIMARY_START_LAYER - 1
    rows: list[dict[str, Any]] = []
    for shard in load_shards(shard_dir):
        positions = {str(row["target_key"]): row for row in shard["positions"]}
        for mention in shard["sample_table"]:
            position = positions[str(mention["target_key"])]
            diagnostics = position["jffn_diagnostics"]
            values = {
                name: diagnostics[name][start:].float().numpy()
                for name in ("input_norm", "response_norm", "gain", "direction_cosine")
            }
            rows.append(
                {
                    "mention_id": str(mention["mention_id"]),
                    "target_key": str(mention["target_key"]),
                    "image_id": int(mention["image_id"]),
                    "label_hall": int(int(mention["label"]) == 0),
                    "I": values["input_norm"],
                    "R": values["response_norm"],
                    "S": values["gain"],
                    "D": values["direction_cosine"],
                    "old_risk": _old_risk(position, start),
                }
            )
        del shard, positions
    return rows


def analyze_model(
    *, model: str, outputs_root: Path, bootstrap_replicates: int
) -> dict[str, Any]:
    model_root = outputs_root / model / EXPERIMENT
    result_dir = model_root / "results/jffn_second_round"
    result_dir.mkdir(parents=True, exist_ok=True)
    rows = collect_rows(model_root)
    splits = load_json(str(model_root / "image_splits.json"))
    train_ids = {int(value) for value in splits["train"]}
    test_ids = {int(value) for value in splits["test"]}

    labels = np.asarray([row["label_hall"] for row in rows], dtype=np.int64)
    images = np.asarray([row["image_id"] for row in rows], dtype=np.int64)
    train_mask = np.asarray([value in train_ids for value in images])
    test_mask = np.asarray([value in test_ids for value in images])
    if np.any(train_mask & test_mask) or not np.all(train_mask | test_mask):
        raise AssertionError("Image split is overlapping or incomplete")

    means = {
        name: np.asarray([float(np.mean(row[name])) for row in rows])[:, None]
        for name in ("I", "R", "S", "D")
    }
    old = np.stack([row["old_risk"] for row in rows])
    feature_sets = {
        "C1_I": means["I"],
        "C2_S": means["S"],
        "C3_R": means["R"],
        "C4_I_S": np.concatenate((means["I"], means["S"]), axis=1),
        "C5_I_S_D": np.concatenate(
            (means["I"], means["S"], means["D"]), axis=1
        ),
        "C6_old_risk": old,
        "C7_old_risk_I": np.concatenate((old, means["I"]), axis=1),
        "C8_old_risk_S": np.concatenate((old, means["S"]), axis=1),
        "C9_old_risk_I_S_D": np.concatenate(
            (old, means["I"], means["S"], means["D"]), axis=1
        ),
        "old_risk_I_S": np.concatenate((old, means["I"], means["S"]), axis=1),
    }
    predictions: dict[str, np.ndarray] = {}
    model_metrics: dict[str, Any] = {}
    for name, matrix in feature_sets.items():
        estimator, scores = _fit_predict(
            matrix[train_mask], labels[train_mask], matrix[test_mask]
        )
        predictions[name] = scores
        model_metrics[name] = {
            **_metrics(labels[test_mask], scores),
            "input_dimensions": int(matrix.shape[1]),
            "standardization_fit": "train_only",
            "class_weight": "balanced",
        }
        del estimator

    # Residual R unexplained by I.  Regression and residual standardization are
    # fitted strictly on train images, then a balanced logistic model selects
    # direction using train labels only.
    eps = 1e-12
    log_i = np.log(means["I"] + eps)
    log_r = np.log(means["R"] + eps)
    residual_regression = LinearRegression().fit(log_i[train_mask], log_r[train_mask])
    residual = log_r - residual_regression.predict(log_i)
    _res_estimator, residual_scores = _fit_predict(
        residual[train_mask], labels[train_mask], residual[test_mask]
    )
    predictions["residualized_log_R_given_log_I"] = residual_scores
    residual_metrics = {
        **_metrics(labels[test_mask], residual_scores),
        "beta0": float(residual_regression.intercept_.reshape(-1)[0]),
        "beta1": float(residual_regression.coef_.reshape(-1)[0]),
        "train_residual_mean": float(residual[train_mask].mean()),
        "test_residual_mean_real": float(
            residual[test_mask & (labels == 0)].mean()
        ),
        "test_residual_mean_hall": float(
            residual[test_mask & (labels == 1)].mean()
        ),
    }

    test_labels = labels[test_mask]
    test_images = images[test_mask]
    paired = {
        "C4_I_S_minus_C1_I": _paired_image_bootstrap(
            labels=test_labels,
            images=test_images,
            left=predictions["C4_I_S"],
            right=predictions["C1_I"],
            replicates=bootstrap_replicates,
            seed=SEED,
        ),
        "old_I_S_minus_old_I": _paired_image_bootstrap(
            labels=test_labels,
            images=test_images,
            left=predictions["old_risk_I_S"],
            right=predictions["C7_old_risk_I"],
            replicates=bootstrap_replicates,
            seed=SEED + 1,
        ),
        "C9_old_I_S_D_minus_C7_old_I": _paired_image_bootstrap(
            labels=test_labels,
            images=test_images,
            left=predictions["C9_old_risk_I_S_D"],
            right=predictions["C7_old_risk_I"],
            replicates=bootstrap_replicates,
            seed=SEED + 2,
        ),
    }

    prediction_rows = []
    test_indices = np.flatnonzero(test_mask)
    for local_index, global_index in enumerate(test_indices):
        row = rows[int(global_index)]
        prediction_rows.append(
            {
                "model": model,
                "mention_id": row["mention_id"],
                "target_key": row["target_key"],
                "image_id": row["image_id"],
                "label_hall": row["label_hall"],
                "I_mean": float(means["I"][global_index, 0]),
                "R_mean": float(means["R"][global_index, 0]),
                "S_mean": float(means["S"][global_index, 0]),
                "D_mean": float(means["D"][global_index, 0]),
                "R_residual": float(residual[global_index, 0]),
                **{
                    f"score_{name}": float(scores[local_index])
                    for name, scores in predictions.items()
                },
            }
        )
    _write_csv(result_dir / "scalar_incremental_predictions.csv", prediction_rows)
    payload = {
        "protocol": "jffn_second_round_scalar_incremental_v1",
        "model": model,
        "primary_layers_one_based": [PRIMARY_START_LAYER, 32],
        "sample_counts": {
            "all_mentions": len(rows),
            "train_mentions": int(train_mask.sum()),
            "test_mentions": int(test_mask.sum()),
            "train_images": len({int(v) for v in images[train_mask]}),
            "test_images": len({int(v) for v in images[test_mask]}),
            "test_real": int(np.sum(test_labels == 0)),
            "test_hall": int(np.sum(test_labels == 1)),
        },
        "models": model_metrics,
        "residualized_R": residual_metrics,
        "paired_bootstrap": paired,
        "leakage_audit": {
            "image_overlap": len(train_ids & test_ids),
            "standardization": "train_only",
            "regression": "train_only",
            "labels_used_for_layer_selection": False,
        },
    }
    path = result_dir / "scalar_incremental_metrics.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    metric_rows = []
    for name, values in model_metrics.items():
        metric_rows.append({"model": model, "feature_set": name, **values})
    metric_rows.append(
        {
            "model": model,
            "feature_set": "residualized_log_R_given_log_I",
            **residual_metrics,
        }
    )
    _write_csv(result_dir / "scalar_incremental_metrics.csv", metric_rows)
    print(f"[scalar] wrote {path}", flush=True)
    return payload


def main() -> None:
    args = parse_args()
    outputs_root = Path(args.outputs_root).resolve()
    payload = {
        "protocol": "jffn_second_round_scalar_incremental_v1",
        "models": {
            model: analyze_model(
                model=model,
                outputs_root=outputs_root,
                bootstrap_replicates=int(args.bootstrap_replicates),
            )
            for model in MODELS
        },
    }
    path = ROOT / "outputs/jffn_second_round_scalar_summary.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"[scalar] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
