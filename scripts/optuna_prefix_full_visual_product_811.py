#!/usr/bin/env python3
"""Shikra and MiniGPT-4 ENDAC-811 full-visual attention×probability probes."""

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

from features.tc_fvpa_artifacts import atomic_torch_save
from scripts import optuna_four_ae_semantic_single_mlp_811 as search
from utils.io_utils import load_pkl


MODELS = ("minigpt4_7b", "shikra_7b")
VARIANTS = ("raw_full_product", "norm_full_product")
OUT = ROOT / "outputs/coco4000_512_endac_prefix_full_product_optuna_811"
SOURCE = ROOT / "outputs/coco4000_512_endac_811"
COMPACT = ROOT / "outputs/coco4000_512_endac_prefix_semantic_attention_topk_detection_811"


def visual_product(attention: np.ndarray, probability: np.ndarray) -> np.ndarray:
    """Unnormalized sum_v a[l,v] * p[l,v,y] for each decoder layer."""
    return (attention * probability).sum(axis=1).astype(np.float32, copy=False)


def prepare(model: str) -> dict:
    target = OUT / model / "features.pt"
    if target.exists():
        return torch.load(target, map_location="cpu", weights_only=False, mmap=True)
    compact = torch.load(COMPACT / model / "features.pt", map_location="cpu", weights_only=False, mmap=True)
    rows = load_pkl(SOURCE / model / "features.pkl")
    if [row["sample_id"] for row in rows] != list(compact["sample_id"]):
        raise ValueError(f"{model}: sample order differs")
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    if not np.array_equal(labels, compact["labels"]):
        raise ValueError(f"{model}: labels differ")
    log_s = np.asarray(compact["logS"], dtype=np.float32)
    matrices = {}
    products = {}
    zero_layer_rows = {}
    for kind in ("raw", "norm"):
        product = np.stack([
            visual_product(row["semantic_attention"]["matrices"]["attention"],
                           row["semantic_attention"]["matrices"][f"object_probability_{kind}"])
            for row in rows
        ])
        products[kind] = product
        zero_layer_rows[kind] = int(np.count_nonzero(product == 0))
        matrices[f"{kind}_full_product"] = np.concatenate((product, log_s), axis=1)
    if any(not np.isfinite(value).all() for value in matrices.values()):
        raise ValueError(f"{model}: non-finite feature")
    result = {
        "matrices": matrices, "products": products, "labels": labels,
        "masks": compact["masks"],
        "compact": {"sample_id": compact["sample_id"], "image_id": compact["image_id"]},
        "zero_product_layer_rows": zero_layer_rows,
    }
    atomic_torch_save(result, target)
    print(f"PREPARED {model}: {len(rows)} mentions; zero product rows {zero_layer_rows}", flush=True)
    return result


def protocol(model: str) -> dict:
    return {
        "schema": "prefix-full-product-optuna-811-v1", "model": model,
        "source": str(SOURCE / model / "features.pkl"),
        "definition": "sum over every visual token of head-mean attention * target-first-token vocabulary-softmax probability",
        "probability": "raw or final-Norm reverse embedding",
        "inputs": "[per-layer full-visual product, per-layer log1p(S_E)]",
        "variants": list(VARIANTS), "seeds": list(search.SEEDS),
        "training": "train-only z-score, no BN, single-hidden MLP, 24 Optuna TPE trials per variant",
        "selection": "hyperparameters/checkpoints on validation; feature comparison on test (exploratory)",
        "split": "existing ENDAC 3200/400/400 image split",
    }


def summarize() -> None:
    search.summarize()
    with (OUT / "summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with (OUT / "selected_configs.csv").open(newline="", encoding="utf-8") as handle:
        configs = list(csv.DictReader(handle))
    lines = [
        "# Shikra / MiniGPT-4 ENDAC-811 full-visual attention×probability (Optuna)", "",
        "Per layer: `sum_{v in all visual tokens} a_v p_v(y)`; raw and final-Norm vocabulary-softmax variants. "
        "Each is concatenated with `log1p(S_E)` and uses validation-selected hyperparameters/checkpoints.", "",
        "AUROC and HALL-AUPR are three-seed test mean ± population SD, in percent. "
        "Feature comparison on this previously examined test split is exploratory.", "",
        "| Model | Feature | Val AUROC | Test AUROC | Test HALL-AUPR |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['variant']} | {100*float(row['validation_AUROC']):.2f} | "
            f"{100*float(row['AUROC_mean']):.2f} ± {100*float(row['AUROC_std']):.2f} | "
            f"{100*float(row['HALL_AUPR_mean']):.2f} ± {100*float(row['HALL_AUPR_std']):.2f} |"
        )
    lines.extend(["", "## Validation-selected hyperparameters", "",
                  "| Model | Feature | Width | Dropout | Activation | LR | WD | Batch | Monitor | Epochs 43/44/45 |",
                  "|---|---|---:|---:|---|---:|---:|---:|---|---|"])
    for row in configs:
        lines.append(f"| {row['model']} | {row['variant']} | {row['width']} | {row['dropout']} | "
                     f"{row['activation']} | {float(row['learning_rate']):.3g} | "
                     f"{float(row['weight_decay']):.1g} | {row['batch_size']} | "
                     f"{row['monitor']} | {row['best_epochs']} |")
    lines.extend(["", "## Input and checkpoint checks", ""])
    for model in MODELS:
        counts = search.read(OUT / model / "features.pt")["zero_product_layer_rows"]
        lines.append(f"- {model}: zero full-product layer-rows raw={counts['raw']}, norm={counts['norm']}.")
    validation_path = OUT / "validation.json"
    if validation_path.exists():
        validation = json.loads(validation_path.read_text())
        lines.append(f"- CPU reload: {validation['checked_heads']}/{validation['expected_heads']} heads passed; "
                     f"maximum probability error {validation['max_cpu_reload_probability_error']:.3g}; "
                     f"metric error {validation['max_cpu_reload_metric_error']:.3g}.")
    lines.extend(["", "Per-seed metrics and predictions are in each model directory. "
                  "Previously tested prefix AE/Top-K features: "
                  "`../coco4000_512_endac_prefix_ae_semantic_single_mlp_811/summary.md` "
                  "(same 24-run budget but frozen candidate grid rather than Optuna)."])
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "train", "test", "summary", "validate"), required=True)
    parser.add_argument("--model", choices=(*MODELS, "all"), required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(1)
    search.MODELS = MODELS
    search.VARIANTS = VARIANTS
    search.OUT = OUT
    search.source = prepare
    search.protocol = protocol
    if args.stage == "summary":
        summarize()
    elif args.stage == "validate":
        search.validate()
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
