#!/usr/bin/env python3
"""Train the missing all-layer AE + R_cos + S ablation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.tc_fvpa_artifacts import atomic_json_save, atomic_torch_save
from scripts.analyze_ffn_visual_source_signal_ablation import train_matrices
from scripts.analyze_ffn_visual_source_study import (
    EXTENSION_MODELS,
    PRIMARY_MODELS,
    SEEDS,
    _detector_matrices,
    _json_ready,
)
from scripts.train_ffn_consistency_alternative_heads import predict_checkpoint
from scripts.run_ffn_visual_source_attribution import result_root


SPEC = "AE+R_cos+S"


def ae_r_s_matrices(data: dict) -> dict[str, dict[str, np.ndarray]]:
    """Return U with only its final kappa block removed: [R_cos, AE, S]."""
    result = {}
    for split in ("train", "test"):
        formal = data[f"X_{split}"]
        r_cos, ae = np.split(formal["A"], 2, axis=1)
        ae_from_f, strength, _kappa = np.split(formal["F"], 3, axis=1)
        np.testing.assert_array_equal(ae, ae_from_f)
        values = np.concatenate([r_cos, ae, strength], axis=1)
        if not np.isfinite(values).all():
            raise ValueError(f"{SPEC} contains non-finite values")
        result[split] = {SPEC: values}
    return result


def _ensemble_summary(progress: dict, spec: str, labels: np.ndarray) -> dict:
    probabilities = np.mean(
        [np.asarray(progress["predictions"][spec][seed]) for seed in SEEDS], axis=0
    )
    return {
        "ensemble_auroc": float(roc_auc_score(labels, probabilities)),
        "ensemble_real_aupr": float(average_precision_score(labels, probabilities)),
        "ensemble_hall_aupr": float(average_precision_score(1 - labels, 1 - probabilities)),
    }


def run(model: str, device: torch.device) -> dict:
    torch.set_num_threads(1)
    data = _detector_matrices(model)
    matrices = ae_r_s_matrices(data)
    root = result_root(model)
    progress_path = root / "metrics/ae_r_s_feature_training_progress.pt"

    digest = hashlib.sha256()
    for split in ("train", "test"):
        digest.update(data[f"y_{split}"].tobytes())
        digest.update(json.dumps(data[f"{split}_target_keys"]).encode())
        digest.update(matrices[split][SPEC].tobytes())
    signature = digest.hexdigest()
    if progress_path.exists():
        saved = torch.load(progress_path, map_location="cpu", weights_only=False)
        if saved.get("cohort_feature_sha256") != signature:
            raise AssertionError("Saved AE+R_cos+S cohort/features differ")
    else:
        atomic_torch_save(
            {"metrics": {}, "predictions": {}, "cohort_feature_sha256": signature},
            progress_path,
        )

    progress, ensembles, summaries = train_matrices(
        model,
        data,
        matrices,
        (SPEC,),
        device=device,
        resume=True,
        progress_name=progress_path.name,
        probe_directory="metrics/probes_ae_r_s",
    )
    summary = summaries[SPEC]
    seed_aurocs = list(summary["per_seed_auroc"].values())
    summary.update(
        mean_seed_auroc=float(np.mean(seed_aurocs)),
        std_seed_auroc=float(np.std(seed_aurocs)),
        ensemble_real_aupr=float(average_precision_score(data["y_test"], ensembles[SPEC])),
        ensemble_hall_aupr=float(average_precision_score(1 - data["y_test"], 1 - ensembles[SPEC])),
    )
    checkpoint_replay, probe_configs = {}, []
    for seed in SEEDS:
        path = root / f"metrics/probes_ae_r_s/{SPEC}/seed{seed}/model.pt"
        config = json.loads(path.with_name("config.json").read_text())
        probe_configs.append(config)
        replayed = predict_checkpoint(
            "torch", path, config, matrices["test"][SPEC], device
        )
        difference = float(np.max(np.abs(replayed - progress["predictions"][SPEC][seed])))
        if difference > 1e-7:
            raise AssertionError(f"Checkpoint replay differs for seed {seed}: {difference}")
        checkpoint_replay[str(seed)] = {"maximum_probability_difference": difference}

    formal = torch.load(root / "metrics/detector_training_progress.pt", map_location="cpu", weights_only=False)
    core = torch.load(root / "metrics/core_signal_feature_training_progress.pt", map_location="cpu", weights_only=False)
    baselines = {
        "AE+R_cos": _ensemble_summary(formal, "A", data["y_test"]),
        "AE+S": _ensemble_summary(core, "AE+S", data["y_test"]),
        "AE+R_cos+S+kappa": _ensemble_summary(core, "U", data["y_test"]),
    }
    comparisons = {
        f"{SPEC}__minus__{name}": {
            "ensemble_auroc_delta": summary["ensemble_auroc"] - values["ensemble_auroc"],
            "ensemble_hall_aupr_delta": summary["ensemble_hall_aupr"] - values["ensemble_hall_aupr"],
        }
        for name, values in baselines.items()
    }
    result = {
        "model": model,
        "status": "EXPLORATORY_AE_R_S_COMPLETE",
        "feature_definition": "concatenate [R_cos, AE, raw S]; exactly U with kappa removed",
        "cohort_feature_sha256": signature,
        "counts": data["counts"],
        "protocol": {
            "split": "same fixed image 8:2 split",
            "seeds": list(SEEDS),
            "device": str(device),
            "bootstrap": "NOT_RUN",
            "probe_configs": probe_configs,
        },
        "summary": summary,
        "checkpoint_replay": checkpoint_replay,
        "baselines": baselines,
        "comparisons": comparisons,
    }
    atomic_json_save(_json_ready(result), root / "metrics/ae_r_s_feature_results.json")

    rows = []
    for seed in SEEDS:
        for rule, report in progress["metrics"][SPEC][seed]["threshold_reports"].items():
            metrics = report["test_metrics"]
            rows.append({
                "seed": seed,
                "threshold_rule": rule,
                "decision_threshold": report["threshold"],
                "auroc": metrics["auc"],
                **{
                    f"{label}_{metric}": metrics[key][metric]
                    for label, key in (("real", "real_positive"), ("hall", "hallucination_positive"))
                    for metric in ("aupr", "precision", "recall", "f1")
                },
            })
    with (root / "tables/ae_r_s_feature_seed_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return result


def _self_check() -> None:
    a = np.array([[1, 2, 3, 4]], dtype=np.float32)
    f = np.array([[3, 4, 5, 6, 7, 8]], dtype=np.float32)
    data = {f"X_{split}": {"A": a, "F": f} for split in ("train", "test")}
    for values in ae_r_s_matrices(data).values():
        np.testing.assert_array_equal(values[SPEC], [[1, 2, 3, 4, 5, 6]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=",".join((*PRIMARY_MODELS, *EXTENSION_MODELS)))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        _self_check()
        print("self-check PASS")
        return
    for model in filter(None, map(str.strip, args.models.split(","))):
        result = run(model, torch.device(args.device))
        print(model, json.dumps(_json_ready(result["summary"]), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
