#!/usr/bin/env python3
"""Test visual-token MAD of target-word vocabulary-softmax probabilities."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from features.tc_fvpa_artifacts import atomic_json_save, atomic_torch_save
from scripts import optuna_four_ae_semantic_single_mlp_811 as search
from utils.io_utils import load_pkl


OUT = ROOT / "outputs/coco4000_512_endac_semantic_probability_mad_optuna_811"
SOURCE = ROOT / "outputs/coco4000_512_endac_semantic_attention_topk"
REFERENCE = ROOT / "outputs/coco4000_512_endac_semantic_attention_all_visual_811"
VARIANTS = ("raw_mad", "norm_mad", "raw_mad_logs", "norm_mad_logs")


def probability_mad(probability: np.ndarray) -> np.ndarray:
    """Median_v |p_v(y) - median_u p_u(y)| for every decoder layer."""
    values = np.asarray(probability, dtype=np.float64)
    center = np.median(values, axis=1, keepdims=True)
    return np.median(np.abs(values - center), axis=1).astype(np.float32)


def prepare(model: str) -> dict:
    target = OUT / model / "features.pt"
    if target.exists():
        return torch.load(target, map_location="cpu", weights_only=False, mmap=True)
    reference = torch.load(REFERENCE / model / "features.pt", map_location="cpu", weights_only=False, mmap=True)
    rows = load_pkl(SOURCE / model / "semantic_attention.pkl")
    if [row["sample_id"] for row in rows] != list(reference["sample_id"]):
        raise ValueError(f"{model}: sample order differs")
    log_s = np.asarray(reference["log1p_S"], dtype=np.float32)
    matrices = {}
    zeros = {}
    for kind in ("raw", "norm"):
        mad = np.stack([probability_mad(row["matrices"][f"object_probability_{kind}"])
                        for row in rows])
        zeros[kind] = int(np.count_nonzero(mad == 0))
        matrices[f"{kind}_mad"] = mad
        matrices[f"{kind}_mad_logs"] = np.concatenate((mad, log_s), axis=1)
    if any(not np.isfinite(value).all() for value in matrices.values()):
        raise ValueError(f"{model}: non-finite MAD")
    result = {
        "matrices": matrices,
        "labels": np.asarray(reference["labels"]),
        "masks": reference["masks"],
        "compact": {"sample_id": reference["sample_id"], "image_id": reference["image_id"]},
        "zero_mad_layer_rows": zeros,
    }
    atomic_torch_save(result, target)
    print(f"PREPARED {model}: {len(rows)} mentions; zero MAD layer-rows {zeros}", flush=True)
    return result


def protocol(model: str) -> dict:
    return {
        "schema": "semantic-probability-mad-optuna-811-v1",
        "model": model,
        "source": str(SOURCE / model / "semantic_attention.pkl"),
        "definition": "median over all visual tokens of abs(p_v(y) - median_visual(p_u(y))), per decoder layer",
        "probability": "first object token; raw or final-Norm vocabulary softmax; no visual-region normalization",
        "variants": list(VARIANTS),
        "inputs": "standalone per-layer MAD or [per-layer MAD, per-layer log1p(S_E)]",
        "training": "train-only z-score; no BN; single-hidden MLP; 24 Optuna TPE trials, validation top3 rerun seeds44/45",
        "selection": "hyperparameters/checkpoints on validation; feature comparison on test (exploratory)",
        "split": "existing ENDAC 3200/400/400 images; seeds43/44/45",
    }


def summarize() -> None:
    search.summarize()
    with (OUT / "summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    lines = [
        "# Four-model ENDAC-811 target-probability MAD (Optuna)", "",
        "Per layer: `median_v |p_v(y) - median_u p_u(y)|` over all visual tokens. "
        "The probability is the original vocabulary softmax, not spatially normalized. "
        "Standalone and +`log1p(S_E)` each use validation-selected parameters and checkpoints.", "",
        "Test AUROC and HALL-AUPR are three-seed mean ± population SD, in percent. "
        "Test-based feature ranking is exploratory.", "",
        "| Model | Feature | Val AUROC | Test AUROC | Test HALL-AUPR |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['variant']} | {100*float(row['validation_AUROC']):.2f} | "
            f"{100*float(row['AUROC_mean']):.2f} ± {100*float(row['AUROC_std']):.2f} | "
            f"{100*float(row['HALL_AUPR_mean']):.2f} ± {100*float(row['HALL_AUPR_std']):.2f} |"
        )
    lines.extend(["", "## Numerical notes", "",
                  "A zero MAD can be genuine flatness or information lost when the saved FP32 probability underflowed. "
                  "The existing trainer sets the z-score scale to 1 for training columns with SD below `1e-12`; "
                  "these columns are effectively not standardized.", "",
                  "| Model | raw zero layer-rows | norm zero layer-rows | raw affected mentions | total layer-rows | raw SD<1e-12 columns | norm SD<1e-12 columns |",
                  "|---|---:|---:|---:|---:|---:|---:|"])
    for model in search.MODELS:
        data = search.read(OUT / model / "features.pt")
        count = data["zero_mad_layer_rows"]
        raw = data["matrices"]["raw_mad"]
        norm = data["matrices"]["norm_mad"]
        train = data["masks"]["train"]
        low_raw = int((raw[train].astype(np.float64).std(axis=0) < 1e-12).sum())
        low_norm = int((norm[train].astype(np.float64).std(axis=0) < 1e-12).sum())
        lines.append(f"| {model} | {count['raw']} | {count['norm']} | "
                     f"{int((raw == 0).any(axis=1).sum())} | {raw.size} | {low_raw} | {low_norm} |")
    lines.extend(["", "Full per-seed metrics and predictions are in each model directory. "
                  "Selected hyperparameters are in `selected_configs.csv`; all trials are in `studies/*.db`. "
                  "For comparison, see `../coco4000_512_endac_semantic_js_full_optuna_811/summary.md` and "
                  "`../coco4000_512_endac_four_ae_semantic_optuna_811/summary.md`."])
    validation_path = OUT / "validation.json"
    if validation_path.exists():
        validation = json.loads(validation_path.read_text())
        lines.extend(["", f"Checkpoint reload: {validation['checked_heads']}/{validation['expected_heads']} passed; "
                      f"maximum CPU/GPU probability difference {validation['max_cpu_reload_probability_error']:.2g} "
                      f"({validation['worst_head']}), tolerance {validation['probability_tolerance']:.0g}; "
                      f"maximum metric difference {validation['max_cpu_reload_metric_error']:.2g}."])
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate() -> None:
    """Reload every selected head on CPU; allow 1e-5 GPU/CPU probability drift."""
    checked = 0
    max_probability_error = 0.0
    max_metric_error = 0.0
    worst_head = ""
    for model in search.MODELS:
        data = prepare(model)
        selection = json.loads((OUT / model / "selection.json").read_text())
        for variant in VARIANTS:
            matrix = data["matrices"][variant]
            trial = selection["variants"][variant]["number"]
            for seed in search.SEEDS:
                trained = search.selected_result(model, variant, trial, seed)
                final = search.read(OUT / model / "final" / variant / f"seed{seed}.pt")
                mean, scale = search.mlp.scale_fit(matrix[data["masks"]["train"]], True)
                np.testing.assert_array_equal(trained["mean"], mean)
                np.testing.assert_array_equal(trained["scale"], scale)
                network = search.mlp.SingleMLP(matrix.shape[1], trained["config"])
                network.load_state_dict(trained["state_dict"])
                test_x = torch.as_tensor(search.mlp.transform(matrix[data["masks"]["test"]], mean, scale))
                recomputed = search.mlp.predict(network, test_x)
                difference = float(np.max(np.abs(recomputed - final["test_probabilities"])))
                if difference > max_probability_error:
                    max_probability_error = difference
                    worst_head = f"{model}/{variant}/seed{seed}"
                labels = data["labels"][data["masks"]["test"]]
                for metric, value in search.mlp.metrics(labels, recomputed).items():
                    max_metric_error = max(max_metric_error, abs(value - final["test_metrics"][metric]))
                checked += 1
    if max_probability_error > 1e-5 or max_metric_error > 1e-8:
        raise ValueError(f"CPU reload differs: probability={max_probability_error}, metric={max_metric_error}")
    report = {
        "status": "pass", "checked_heads": checked, "expected_heads": len(search.MODELS) * len(VARIANTS) * len(search.SEEDS),
        "max_cpu_reload_probability_error": max_probability_error,
        "max_cpu_reload_metric_error": max_metric_error,
        "worst_head": worst_head, "probability_tolerance": 1e-5,
    }
    atomic_json_save(report, OUT / "validation.json")
    print(report, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "train", "test", "summary", "validate"), required=True)
    parser.add_argument("--model", choices=(*search.MODELS, "all"), required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(1)
    search.OUT = OUT
    search.VARIANTS = VARIANTS
    search.source = prepare
    search.protocol = protocol
    if args.stage == "summary":
        summarize()
    elif args.stage == "validate":
        validate()
    elif args.model == "all":
        parser.error("--model all is only for summary/validate")
    elif args.stage == "prepare":
        prepare(args.model)
    elif args.stage == "train":
        search.train_model(args.model, args.device)
    else:
        search.test_model(args.model, args.device)


if __name__ == "__main__":
    main()
