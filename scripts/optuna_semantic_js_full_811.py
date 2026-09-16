#!/usr/bin/env python3
"""Four-model ENDAC JS alignment and full-visual semantic-attention probes."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from features.tc_fvpa_artifacts import atomic_torch_save
from scripts import optuna_four_ae_semantic_single_mlp_811 as search
from utils.io_utils import load_pkl


OUT = ROOT / "outputs/coco4000_512_endac_semantic_js_full_optuna_811"
MATRICES = ROOT / "outputs/coco4000_512_endac_semantic_attention_topk"
FULL = ROOT / "outputs/coco4000_512_endac_semantic_attention_all_visual_811"
VARIANTS = (
    "raw_js_attention32", "norm_js_attention32",
    "raw_js_union32", "norm_js_union32",
    "raw_full_product", "norm_full_product",
)


def js_on_region(attention: np.ndarray, probability: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Base-2 JS after separately normalizing both vectors in each region."""
    aa = np.where(mask, attention, 0).astype(np.float64)
    pp = np.where(mask, probability, 0).astype(np.float64)
    aa_sum = aa.sum(axis=1, keepdims=True)
    pp_sum = pp.sum(axis=1, keepdims=True)
    aa = np.divide(aa, aa_sum, out=np.zeros_like(aa), where=aa_sum > 0)
    pp = np.divide(pp, pp_sum, out=np.zeros_like(pp), where=pp_sum > 0)
    empty_a = aa_sum[:, 0] == 0
    empty_p = pp_sum[:, 0] == 0
    if empty_a.any():
        aa[empty_a] = mask[empty_a] / mask[empty_a].sum(axis=1, keepdims=True)
    if empty_p.any():
        pp[empty_p] = mask[empty_p] / mask[empty_p].sum(axis=1, keepdims=True)
    middle = (aa + pp) / 2
    score = np.zeros(len(aa), dtype=np.float64)
    for values in (aa, pp):
        positive = values > 0
        term = np.zeros_like(values)
        term[positive] = values[positive] * np.log2(values[positive] / middle[positive])
        score += term.sum(axis=1) / 2
    return score.astype(np.float32), int(empty_a.sum() + empty_p.sum())


def one_image(attention: np.ndarray, probability: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
    layer_count, visual_count = attention.shape
    ranks_a = np.argpartition(attention, visual_count - 32, axis=1)[:, -32:]
    ranks_p = np.argpartition(probability, visual_count - 32, axis=1)[:, -32:]
    layer_index = np.arange(layer_count)[:, None]
    mask_a = np.zeros((layer_count, visual_count), dtype=bool)
    mask_p = np.zeros_like(mask_a)
    mask_a[layer_index, ranks_a] = True
    mask_p[layer_index, ranks_p] = True
    attention_js, empty_a = js_on_region(attention, probability, mask_a)
    union_js, empty_union = js_on_region(attention, probability, mask_a | mask_p)
    return attention_js, union_js, empty_a, empty_union


def prepare(model: str) -> dict:
    target = OUT / model / "features.pt"
    if target.exists():
        return torch.load(target, map_location="cpu", weights_only=False, mmap=True)
    reference = torch.load(FULL / model / "features.pt", map_location="cpu", weights_only=False, mmap=True)
    rows = load_pkl(MATRICES / model / "semantic_attention.pkl")
    if [row["sample_id"] for row in rows] != list(reference["sample_id"]):
        raise ValueError(f"{model}: sample order differs")
    values = {variant: [] for variant in VARIANTS[:4]}
    empty = {variant: 0 for variant in VARIANTS[:4]}
    for row in rows:
        matrices = row["matrices"]
        attention = matrices["attention"]
        for kind in ("raw", "norm"):
            probability = matrices[f"object_probability_{kind}"]
            js_attention, js_union, empty_a, empty_union = one_image(attention, probability)
            values[f"{kind}_js_attention32"].append(js_attention)
            values[f"{kind}_js_union32"].append(js_union)
            empty[f"{kind}_js_attention32"] += empty_a
            empty[f"{kind}_js_union32"] += empty_union
    log_s = np.asarray(reference["log1p_S"], dtype=np.float32)
    matrices = {
        variant: np.concatenate((np.stack(rows), log_s), axis=1)
        for variant, rows in values.items()
    }
    for kind in ("raw", "norm"):
        full_product = np.asarray(reference["cubes"][kind][:, :, 3], dtype=np.float32)
        matrices[f"{kind}_full_product"] = np.concatenate((full_product, log_s), axis=1)
    if any(not np.isfinite(value).all() for value in matrices.values()):
        raise ValueError(f"{model}: non-finite features")
    result = {
        "matrices": matrices, "labels": np.asarray(reference["labels"]),
        "masks": reference["masks"],
        "compact": {"sample_id": reference["sample_id"], "image_id": reference["image_id"]},
        "zero_probability_regions": empty,
    }
    atomic_torch_save(result, target)
    print(f"PREPARED {model}: {len(rows)} mentions, zero-p regions {empty}", flush=True)
    return result


def protocol(model: str) -> dict:
    return {
        "schema": "semantic-js-full-optuna-811-v1", "model": model,
        "source": str(MATRICES / model / "semantic_attention.pkl"),
        "full_product_source": str(FULL / model / "features.pt"),
        "variants": list(VARIANTS), "seeds": list(search.SEEDS),
        "js": "base-2 JS, attention and target-token probability separately normalized within selected visual positions",
        "attention32": "Top32 by original head-mean attention; both distributions restricted to these 32 positions",
        "union32": "union of attention Top32 and target-probability Top32; 32-64 positions",
        "full_product": "sum over every visual token of unnormalized attention * target-token vocabulary-softmax probability",
        "inputs": "[one scalar per decoder layer, log1p(S_E) per decoder layer]",
        "training": "train-only z-score, no BN, single-hidden MLP; 24 TPE trials then 3-seed validation selection",
        "selection": "parameters and checkpoints on validation; feature comparison on test",
        "split": "existing ENDAC 3200/400/400 images; seeds 43/44/45",
        "caveat": "Test-based feature ranking is exploratory, not an independent test estimate.",
    }


def summarize() -> None:
    search.summarize()
    with (OUT / "summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with (OUT / "selected_configs.csv").open(newline="", encoding="utf-8") as handle:
        configs = list(csv.DictReader(handle))
    lines = [
        "# Four-model ENDAC-811 JS and full-visual product (Optuna)", "",
        "Each variant uses its own validation-selected parameters/checkpoint; the table reports three-seed test mean ± population SD. All inputs are [new per-layer feature, log1p(S_E)].", "",
        "JS is base-2 after separately renormalizing attention and object-probability spatial distributions within the selected region. `attention32` uses attention Top32; `union32` uses the union of attention Top32 and semantic Top32. `full_product` is the original unnormalized sum over all visual tokens.", "",
        "| Model | Feature | Val AUROC (%) | Test AUROC (%) | Test HALL-AUPR (%) |",
        "|---|---|---:|---:|---:|",
    ]
    for model in search.MODELS:
        selected = [row for row in rows if row["model"] == model]
        best_auc = max(selected, key=lambda row: (float(row["AUROC_mean"]), float(row["HALL_AUPR_mean"])))
        best_ap = max(selected, key=lambda row: (float(row["HALL_AUPR_mean"]), float(row["AUROC_mean"])))
        for row in selected:
            mark = " (test AUROC best)" if row is best_auc else ""
            mark += " (test AP best)" if row is best_ap else ""
            lines.append(
                f"| {model} | {row['variant']}{mark} | {100*float(row['validation_AUROC']):.2f} | "
                f"{100*float(row['AUROC_mean']):.2f} ± {100*float(row['AUROC_std']):.2f} | "
                f"{100*float(row['HALL_AUPR_mean']):.2f} ± {100*float(row['HALL_AUPR_std']):.2f} |"
            )
    lines.extend(["", "## Validation-selected hyperparameters", "",
                  "| Model | Feature | Trial | Width | Dropout | Activation | LR | WD | Batch | Monitor | Best epochs 43/44/45 |",
                  "|---|---|---:|---:|---:|---|---:|---:|---:|---|---|"])
    for row in configs:
        lines.append(
            f"| {row['model']} | {row['variant']} | {row['trial']} | {row['width']} | "
            f"{row['dropout']} | {row['activation']} | {float(row['learning_rate']):.3g} | "
            f"{float(row['weight_decay']):.1g} | {row['batch_size']} | {row['monitor']} | {row['best_epochs']} |"
        )
    lines.extend(["", "## Numerical note", "",
                  "A spatial distribution with zero mass inside the selected region cannot be renormalized. "
                  "For such mention-by-layer rows, this experiment uses a uniform distribution over that region and records the count below. "
                  "Qwen2.5 raw attention32 has 7,543/221,872 affected rows (3.40%); its attention mass itself is never zero.", "",
                  "| Model | raw attention32 | norm attention32 | raw union32 | norm union32 |",
                  "|---|---:|---:|---:|---:|"])
    for model in search.MODELS:
        counts = search.read(OUT / model / "features.pt")["zero_probability_regions"]
        lines.append(f"| {model} | {counts['raw_js_attention32']} | {counts['norm_js_attention32']} | "
                     f"{counts['raw_js_union32']} | {counts['norm_js_union32']} |")
    lines.extend(["", "The full 24-trial Optuna histories are in each model's `studies/*.db`. "
                  "Existing AE and Top-K baselines are in `../coco4000_512_endac_four_ae_semantic_optuna_811/summary.md`. "
                  "Test-based feature ranking is exploratory, not an independent test estimate."])
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
