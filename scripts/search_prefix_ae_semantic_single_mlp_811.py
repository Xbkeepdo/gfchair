#!/usr/bin/env python3
"""No-BN, train-standardized single-MLP search for prefix AE and semantic attention."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.semantic_attention_topk import FEATURE_NAMES
from features.tc_fvpa_artifacts import atomic_json_save, atomic_torch_save
from scripts import search_single_mlp_811 as mlp
from utils.io_utils import load_pkl


MODELS = ("minigpt4_7b", "shikra_7b")
VARIANTS = ("ae_logs", "raw_top16", "raw_top32", "norm_top16", "norm_top32")
SEEDS = (43, 44, 45)
TOP_N = 3
SOURCE = ROOT / "outputs/coco4000_512_endac_811"
SEMANTIC = ROOT / "outputs/coco4000_512_endac_prefix_semantic_attention_topk_detection_811"
OUT = ROOT / "outputs/coco4000_512_endac_prefix_ae_semantic_single_mlp_811"


def candidates() -> list[dict]:
    result = []
    for value in mlp.candidates():
        value = dict(value, standardize=True, batch_norm=False)
        if value not in result:
            result.append(value)
    for changes in (
        dict(width=128, dropout=.3, activation="relu", learning_rate=.001,
             weight_decay=1e-5, batch_size=128, monitor="val_loss"),
        dict(width=128, dropout=.3, activation="relu", learning_rate=.003,
             weight_decay=1e-4, batch_size=128, monitor="val_loss"),
        dict(width=256, dropout=.3, activation="relu", learning_rate=.001,
             weight_decay=1e-5, batch_size=128, monitor="val_loss"),
        dict(width=512, dropout=.1, activation="relu", learning_rate=.001,
             weight_decay=1e-5, batch_size=256, monitor="val_loss"),
        dict(width=128, dropout=0., activation="relu", learning_rate=.001,
             weight_decay=0., batch_size=256, monitor="val_loss"),
        dict(width=256, dropout=.1, activation="gelu", learning_rate=.0003,
             weight_decay=1e-5, batch_size=256, monitor="val_loss"),
    ):
        value = dict(result[0], **changes)
        if value not in result:
            result.append(value)
    if len(result) != 24:
        raise ValueError(f"Expected 24 distinct no-BN standardized candidates, got {len(result)}")
    return result


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source(model: str) -> dict:
    compact = read(SEMANTIC / model / "features.pt")
    rows = load_pkl(SOURCE / model / "features.pkl")
    if [row["sample_id"] for row in rows] != compact["sample_id"]:
        raise ValueError(f"{model}: AE and semantic sample order differ")
    labels = np.asarray(compact["labels"], dtype=np.int64)
    masks = compact["masks"]
    ae = np.stack([row["features"]["visual_only"] for row in rows]).astype(np.float32)
    log_s = np.asarray(compact["logS"], dtype=np.float32)
    if not np.array_equal(ae[:, 1::2], log_s):
        raise ValueError(f"{model}: visual-only log1p(S_E) source differs")
    result = {"ae_logs": ae}
    index = FEATURE_NAMES.index("semantic_attention")
    for variant in VARIANTS[1:]:
        semantic_attention = np.asarray(compact["cubes"][variant][:, :, index], dtype=np.float32)
        result[variant] = np.concatenate((semantic_attention, log_s), axis=1)
    if any(not np.isfinite(value).all() for value in result.values()):
        raise ValueError(f"{model}: non-finite feature")
    return dict(matrices=result, labels=labels, masks=masks, compact=compact)


def protocol(model: str) -> dict:
    return dict(
        schema="prefix-ae-semantic-single-mlp-811-v1",
        model=model,
        features={"ae_logs": "interleaved [AE_V,log1p(S_E)]",
                  "semantic_variants": "[semantic_attention=a*p at each layer,log1p(S_E)]"},
        variants=list(VARIANTS), candidates=candidates(), seeds=list(SEEDS),
        shortlist_seed=43, shortlist_size=TOP_N,
        selection="seed43 top3 by validation AUROC/HALL-AUPR, then three-seed validation mean; test after both models freeze",
        split="existing ENDAC 3200/400/400 image split",
        source=str(SOURCE / model / "features.pkl"),
        semantic_source=str(SEMANTIC / model / "features.pt"),
        note="Test split has been examined by prior experiments; this is exploratory.",
    )


def progress(model: str, stage: str, completed: int, total: int) -> None:
    atomic_json_save(dict(model=model, stage=stage, completed=completed, total=total,
                          updated_at=now()), OUT / model / "progress.json")


def result_path(model: str, variant: str, index: int, seed: int) -> Path:
    return OUT / model / "search" / variant / f"candidate{index:02d}_seed{seed}.pt"


def rank(row: dict) -> tuple[float, float, int]:
    return (float(row["validation"]["AUROC"]), float(row["validation"]["HALL_AUPR"]),
            -int(row["index"]))


def train_model(model: str, device: str) -> None:
    data = source(model)
    root = OUT / model
    setup = protocol(model)
    path = root / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != setup:
        raise ValueError(f"Changed protocol: {path}")
    atomic_json_save(setup, path)
    configs = setup["candidates"]
    labels, masks = data["labels"], data["masks"]
    total = len(VARIANTS) * (len(configs) + TOP_N * 2)
    completed = 0
    choices = {}
    for variant in VARIANTS:
        matrix = data["matrices"][variant]

        def one(index: int, seed: int) -> dict:
            nonlocal completed
            saved = result_path(model, variant, index, seed)
            if saved.exists():
                result = read(saved)
            else:
                progress(model, f"{variant}/candidate{index:02d}/seed{seed}", completed, total)
                result = mlp.fit(matrix[masks["train"]], labels[masks["train"]],
                                 matrix[masks["val"]], labels[masks["val"]],
                                 configs[index], seed, device)
                result.update(model=model, variant=variant, index=index)
                atomic_torch_save(result, saved)
            completed += 1
            progress(model, f"{variant}/candidate{index:02d}/seed{seed}", completed, total)
            print("FIT", model, variant, index, seed, result["validation"], flush=True)
            return dict(index=index, seed=seed, validation=result["validation"])

        first = [one(index, 43) for index in range(len(configs))]
        shortlist = sorted(first, key=rank, reverse=True)[:TOP_N]
        finalists = []
        for entry in shortlist:
            runs = [entry, one(entry["index"], 44), one(entry["index"], 45)]
            finalists.append(dict(
                index=entry["index"],
                validation={metric: float(np.mean([run["validation"][metric] for run in runs]))
                            for metric in ("AUROC", "HALL_AUPR")},
            ))
        choices[variant] = max(finalists, key=rank)
    champion = max(enumerate(VARIANTS), key=lambda item:
                   (*rank(choices[item[1]])[:2], -item[0]))[1]
    selection = dict(model=model, variants=choices, champion=champion,
                     frozen_at=now(), ranking="validation AUROC then HALL-AUPR")
    atomic_json_save(selection, root / "selection.json")
    progress(model, "validation_complete", total, total)


def test_model(model: str, device: str) -> None:
    if not all((OUT / name / "selection.json").exists() for name in MODELS):
        raise FileNotFoundError("Freeze both model selections before test")
    data = source(model)
    root = OUT / model
    selection = json.loads((root / "selection.json").read_text())
    labels, masks = data["labels"], data["masks"]
    rows = []
    for variant in VARIANTS:
        matrix = data["matrices"][variant]
        index = selection["variants"][variant]["index"]
        for seed in SEEDS:
            trained = read(result_path(model, variant, index, seed))
            network = mlp.SingleMLP(matrix.shape[1], trained["config"]).to(device)
            network.load_state_dict(trained["state_dict"])
            test_x = torch.as_tensor(mlp.transform(matrix[masks["test"]],
                                                   trained["mean"], trained["scale"]), device=device)
            probability = mlp.predict(network, test_x)
            metrics = mlp.metrics(labels[masks["test"]], probability)
            final = dict(model=model, variant=variant, index=index, seed=seed,
                         validation=trained["validation"], test_metrics=metrics,
                         test_probabilities=probability)
            path = root / "final" / variant / f"seed{seed}.pt"
            atomic_torch_save(final, path)
            full = np.empty(len(labels), dtype=np.float32)
            for split, values in (("train", trained["train_probabilities"]),
                                  ("val", trained["validation_probabilities"]),
                                  ("test", probability)):
                full[masks[split]] = values
            np.savez_compressed(path.with_name(f"seed{seed}_predictions.npz"),
                                sample_id=np.asarray(data["compact"]["sample_id"]),
                                image_id=data["compact"]["image_id"], label=labels,
                                split=np.asarray(["train" if masks["train"][i] else
                                                  "val" if masks["val"][i] else "test"
                                                  for i in range(len(labels))]),
                                real_probability=full)
            rows.append(dict(model=model, variant=variant, seed=seed, **metrics))
            print("TEST", model, variant, seed, metrics, flush=True)
    write_csv(root / "seed_metrics.csv", rows)
    summary = []
    for variant in VARIANTS:
        selected = [row for row in rows if row["variant"] == variant]
        summary.append(dict(model=model, variant=variant,
                            selected=(variant == selection["champion"]),
                            validation_AUROC=selection["variants"][variant]["validation"]["AUROC"],
                            validation_HALL_AUPR=selection["variants"][variant]["validation"]["HALL_AUPR"],
                            **{f"{metric}_{stat}": float(function([row[metric] for row in selected]))
                               for metric in ("AUROC", "HALL_AUPR")
                               for stat, function in (("mean", np.mean), ("std", np.std))}))
    write_csv(root / "summary.csv", summary)
    progress(model, "complete", len(VARIANTS) * len(SEEDS), len(VARIANTS) * len(SEEDS))


def summarize() -> None:
    rows = []
    for model in MODELS:
        with (OUT / model / "summary.csv").open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    write_csv(OUT / "summary.csv", rows)
    lines = ["# Prefix AE vs semantic×attention + log1p(S_E), Torch single MLP",
             "", "Train-only standardization, no BN; 24 frozen candidates per feature; ",
             "seed43 shortlist top3, seeds44/45 validation selection, then test.",
             "3200/400/400 image split; AUROC/HALL-AUPR are three-seed test mean ± population std.",
             "", "| Model | Feature | Val AUROC (%) | Test AUROC (%) | Test HALL-AUPR (%) |",
             "|---|---|---:|---:|---:|"]
    for row in rows:
        marker = " (selected)" if row["selected"] == "True" else ""
        lines.append(f"| {row['model']} | {row['variant']}{marker} | "
                     f"{100*float(row['validation_AUROC']):.2f} | "
                     f"{100*float(row['AUROC_mean']):.2f} ± {100*float(row['AUROC_std']):.2f} | "
                     f"{100*float(row['HALL_AUPR_mean']):.2f} ± "
                     f"{100*float(row['HALL_AUPR_std']):.2f} |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate() -> None:
    checked = 0
    error = 0.
    for model in MODELS:
        data = source(model)
        selection = json.loads((OUT / model / "selection.json").read_text())
        for variant in VARIANTS:
            matrix = data["matrices"][variant]
            index = selection["variants"][variant]["index"]
            for seed in SEEDS:
                trained = read(result_path(model, variant, index, seed))
                final = read(OUT / model / "final" / variant / f"seed{seed}.pt")
                mean, scale = mlp.scale_fit(matrix[data["masks"]["train"]], True)
                np.testing.assert_array_equal(trained["mean"], mean)
                np.testing.assert_array_equal(trained["scale"], scale)
                network = mlp.SingleMLP(matrix.shape[1], trained["config"])
                network.load_state_dict(trained["state_dict"])
                test_x = torch.as_tensor(mlp.transform(matrix[data["masks"]["test"]], mean, scale))
                recomputed = mlp.predict(network, test_x)
                error = max(error, float(np.max(np.abs(recomputed - final["test_probabilities"]))))
                for key, value in mlp.metrics(data["labels"][data["masks"]["test"]], recomputed).items():
                    if abs(value - final["test_metrics"][key]) > 1e-8:
                        raise ValueError(f"{model}/{variant}/seed{seed}: metric differs")
                checked += 1
    report = dict(status="pass", checked_heads=checked, expected_heads=len(MODELS)*len(VARIANTS)*len(SEEDS),
                  max_cpu_reload_probability_error=error, validated_at=now())
    atomic_json_save(report, OUT / "validation.json")
    print(json.dumps(report, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=(*MODELS, "all"), required=True)
    parser.add_argument("--stage", choices=("train", "test", "summary", "validate"), required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.stage == "summary":
        summarize()
    elif args.stage == "validate":
        validate()
    elif args.model == "all":
        parser.error("--model all only works for summary/validate")
    elif args.stage == "train":
        train_model(args.model, args.device)
    else:
        test_model(args.model, args.device)


if __name__ == "__main__":
    main()
