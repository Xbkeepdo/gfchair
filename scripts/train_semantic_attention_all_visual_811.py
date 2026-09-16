#!/usr/bin/env python3
"""Evaluate semantic-attention features over every visual token, with/without logS."""

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

from detection.baselines import evaluate_detection_scores, select_detection_threshold
from features.tc_fvpa_artifacts import atomic_json_save, atomic_torch_save
from scripts import train_semantic_attention_logs_fusion_811 as topk_fusion
from scripts import train_semantic_attention_topk_811 as topk
from utils.io_utils import load_pkl

SOURCE = ROOT / "outputs/coco4000_512_endac_semantic_attention_topk"
OUT = ROOT / "outputs/coco4000_512_endac_semantic_attention_all_visual_811"
MODELS = topk.MODELS
SEEDS = topk.SEEDS
FEATURE_NAMES = topk.FEATURE_NAMES
GROUPS = FEATURE_NAMES + ("all_six",)
VARIANTS = ("raw", "norm")
MODES = ("standalone", "plus_logS")
HEADS = tuple((mode, variant, group) for mode in MODES for variant in VARIANTS for group in GROUPS)
CONFIG = topk.CONFIG


def read(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def full_visual_features(probability: np.ndarray, attention: np.ndarray, cosine: np.ndarray) -> np.ndarray:
    """Apply the original six reductions after replacing Top-K by all visual tokens."""
    semantic_mean = probability.mean(axis=1)
    attention_sum = attention.sum(axis=1)
    cosine_mean = cosine.mean(axis=1)
    semantic_attention = (probability * attention).sum(axis=1)
    cosine_attention = cosine_mean * attention_sum
    joint = (probability * attention * np.maximum(cosine, 0)).sum(axis=1)
    return np.stack(
        (semantic_mean, attention_sum, cosine_mean, semantic_attention, cosine_attention, joint),
        axis=1,
    ).astype(np.float32, copy=False)


def prepare(model: str) -> dict:
    root = OUT / model
    target = root / "features.pt"
    protocol = {
        "schema": "semantic-attention-all-visual-811-v1",
        "model": model,
        "source": str(SOURCE / model / "semantic_attention.pkl"),
        "region": "all visual tokens; no probability ranking or Top-K selection",
        "semantic_only": "mean_v p(v,y)",
        "attention_only": "sum_v a(v)",
        "cosine_only": "mean_v cos(v)",
        "semantic_attention": "sum_v p(v,y)*a(v)",
        "cosine_attention": "mean_v cos(v) * sum_v a(v)",
        "semantic_attention_positive_cosine": "sum_v p(v,y)*a(v)*max(cos(v),0)",
        "variants": list(VARIANTS),
        "modes": list(MODES),
        "groups": list(GROUPS),
        "seeds": list(SEEDS),
        "classifier": CONFIG,
        "selection": "validation AUROC then HALL-AUPR; all four selections frozen before test",
    }
    protocol_path = root / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError(f"Changed protocol: {protocol_path}")
    else:
        atomic_json_save(protocol, protocol_path)
    if target.exists():
        return read(target)

    reference = topk_fusion.prepare(model)
    rows = load_pkl(SOURCE / model / "semantic_attention.pkl")
    if [row["sample_id"] for row in rows] != list(reference["sample_id"]):
        raise ValueError(f"{model}: mention order differs from ENDAC-811")
    cubes = {variant: [] for variant in VARIANTS}
    for row in rows:
        matrices = row["matrices"]
        attention = matrices["attention"]
        cosine = matrices["cosine_similarity"]
        for variant in VARIANTS:
            probability = matrices[f"object_probability_{variant}"]
            cubes[variant].append(full_visual_features(probability, attention, cosine))
    cubes = {variant: np.stack(values) for variant, values in cubes.items()}
    if any(not np.isfinite(cube).all() for cube in cubes.values()):
        raise ValueError(f"{model}: non-finite all-visual feature")
    data = {
        "schema": protocol["schema"],
        "model": model,
        "cubes": cubes,
        "log1p_S": reference["log1p_S"],
        "labels": reference["labels"],
        "masks": reference["masks"],
        "sample_id": reference["sample_id"],
        "image_id": reference["image_id"],
    }
    atomic_torch_save(data, target)
    print(f"PREPARED {model}: {len(rows)} mentions, {cubes['raw'].shape}", flush=True)
    return data


def matrix(data: dict, mode: str, variant: str, group: str) -> np.ndarray:
    cube = data["cubes"][variant]
    feature = topk.feature_matrix(cube, group)
    if mode == "plus_logS":
        return np.concatenate((feature, data["log1p_S"]), axis=1)
    return feature


def progress(model: str, stage: str, completed: int, **extra) -> None:
    atomic_json_save(
        {
            "model": model,
            "stage": stage,
            "completed": completed,
            "total": len(HEADS) * len(SEEDS),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            **extra,
        },
        OUT / model / "progress.json",
    )


def trained_path(model: str, mode: str, variant: str, group: str, seed: int) -> Path:
    return OUT / model / "validation" / mode / variant / group / f"seed{seed}.pt"


def summarize_seed_rows(rows: list[dict]) -> list[dict]:
    output = []
    for mode, variant, group in HEADS:
        selected = [
            row for row in rows
            if (row["mode"], row["variant"], row["group"]) == (mode, variant, group)
        ]
        if len(selected) != len(SEEDS):
            raise ValueError(f"Incomplete head: {mode}/{variant}/{group}")
        output.append(
            {
                "model": selected[0]["model"],
                "mode": mode,
                "variant": variant,
                "group": group,
                **{
                    f"{metric}_{stat}": float(fn([float(row[metric]) for row in selected]))
                    for metric in ("AUROC", "HALL_AUPR")
                    for stat, fn in (("mean", np.mean), ("std", np.std))
                },
            }
        )
    return output


def rank(row: dict) -> tuple[float, float, int, int, int]:
    return (
        float(row["AUROC_mean"]),
        float(row["HALL_AUPR_mean"]),
        -MODES.index(row["mode"]),
        -VARIANTS.index(row["variant"]),
        -GROUPS.index(row["group"]),
    )


def train(model: str, device: str) -> None:
    data = prepare(model)
    labels, masks = data["labels"], data["masks"]
    rows = []
    completed = 0
    for mode, variant, group in HEADS:
        features = matrix(data, mode, variant, group)
        for seed in SEEDS:
            path = trained_path(model, mode, variant, group, seed)
            if path.exists():
                trained = read(path)
            else:
                progress(model, f"{mode}/{variant}/{group}/seed{seed}", completed)
                trained = topk.trainer.fit(
                    train_x=features[masks["train"]],
                    train_y=labels[masks["train"]],
                    val_x=features[masks["val"]],
                    val_y=labels[masks["val"]],
                    cfg=CONFIG,
                    seed=seed,
                    device=device,
                )
                trained.update(model=model, mode=mode, variant=variant, group=group)
                atomic_torch_save(trained, path)
            completed += 1
            rows.append(
                {
                    "model": model,
                    "mode": mode,
                    "variant": variant,
                    "group": group,
                    "seed": seed,
                    **trained["validation"],
                    "best_epoch": trained["best_epoch"],
                }
            )
            progress(model, f"{mode}/{variant}/{group}/seed{seed}", completed)
            print("VAL", model, mode, variant, group, seed, trained["validation"], flush=True)
    write_csv(OUT / model / "validation_seed_metrics.csv", rows)
    write_csv(OUT / model / "validation_summary.csv", summarize_seed_rows(rows))
    progress(model, "validation_complete", completed, status="complete")


def select() -> None:
    summaries = {model: read_csv(OUT / model / "validation_summary.csv") for model in MODELS}
    per_model = {model: max(rows, key=rank) for model, rows in summaries.items()}
    macro = []
    for mode, variant, group in HEADS:
        selected = [
            next(row for row in summaries[model]
                 if (row["mode"], row["variant"], row["group"]) == (mode, variant, group))
            for model in MODELS
        ]
        macro.append(
            {
                "mode": mode,
                "variant": variant,
                "group": group,
                "AUROC_mean": float(np.mean([float(row["AUROC_mean"]) for row in selected])),
                "HALL_AUPR_mean": float(np.mean([float(row["HALL_AUPR_mean"]) for row in selected])),
            }
        )
    global_champion = max(macro, key=rank)
    def identity(row):
        return {key: row[key] for key in ("mode", "variant", "group")}
    selection = {
        "schema": "semantic-attention-all-visual-selection-811-v1",
        "ranking": "three-seed validation AUROC then HALL-AUPR; no test used",
        "per_model": {model: identity(row) for model, row in per_model.items()},
        "global": identity(global_champion),
        "frozen_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json_save(selection, OUT / "selection.json")
    write_csv(OUT / "validation_macro_summary.csv", sorted(macro, key=rank, reverse=True))
    print(json.dumps(selection, ensure_ascii=False, indent=2), flush=True)


def evaluate(model: str, device: str) -> None:
    selection = json.loads((OUT / "selection.json").read_text())
    data = prepare(model)
    labels, masks = data["labels"], data["masks"]
    split_names = np.where(masks["train"], "train", np.where(masks["val"], "val", "test"))
    rows = []
    completed = 0
    for mode, variant, group in HEADS:
        features = matrix(data, mode, variant, group)
        for seed in SEEDS:
            trained = read(trained_path(model, mode, variant, group, seed))
            network = topk.trainer.SingleMLP(features.shape[1], CONFIG).to(device)
            network.load_state_dict(trained["state_dict"])
            test_x = torch.as_tensor(
                topk.trainer.transform(features[masks["test"]], trained["mean"], trained["scale"]),
                device=device,
            )
            test_probability = topk.trainer.predict(network, test_x)
            metrics = topk.trainer.metrics(labels[masks["test"]], test_probability)
            threshold = select_detection_threshold(
                labels[masks["train"]], 1.0 - trained["train_probabilities"],
                positive_class="real",
            )
            final = {
                "model": model,
                "mode": mode,
                "variant": variant,
                "group": group,
                "seed": seed,
                "validation": trained["validation"],
                "test_metrics": metrics,
                "threshold": threshold,
                "threshold_reports": {
                    name: evaluate_detection_scores(
                        labels[masks["test"]], 1.0 - test_probability, value,
                        positive_class="real",
                    )
                    for name, value in (("train_f1", threshold), ("fixed_0.5", 0.5))
                },
                "test_probabilities": test_probability,
            }
            final_path = OUT / model / "final" / mode / variant / group / f"seed{seed}.pt"
            atomic_torch_save(final, final_path)
            full = np.empty(len(labels), dtype=np.float32)
            full[masks["train"]] = trained["train_probabilities"]
            full[masks["val"]] = trained["validation_probabilities"]
            full[masks["test"]] = test_probability
            np.savez_compressed(
                final_path.with_name(f"seed{seed}_predictions.npz"),
                sample_id=np.asarray(data["sample_id"]), image_id=data["image_id"],
                label=labels, split=split_names, real_probability=full,
            )
            rows.append({"model": model, "mode": mode, "variant": variant,
                         "group": group, "seed": seed, **metrics})
            completed += 1
            progress(model, "test_evaluation", completed)
    write_csv(OUT / model / "test_seed_metrics.csv", rows)
    summary = summarize_seed_rows(rows)
    chosen = selection["per_model"][model]
    for row in summary:
        row["per_model_champion"] = all(row[key] == chosen[key] for key in chosen)
        row["global_champion"] = all(row[key] == selection["global"][key]
                                     for key in selection["global"])
    write_csv(OUT / model / "test_summary.csv", summary)
    progress(model, "complete", completed, status="complete")


def summarize() -> None:
    selection = json.loads((OUT / "selection.json").read_text())
    all_rows = []
    lines = [
        "# All-visual-token semantic-attention ENDAC-811",
        "",
        "T_k is replaced by every visual token. Original mean and sum reductions are retained.",
        "Fixed 3200/400/400 image split; seeds 43/44/45; fixed single-hidden 128 MLP.",
        "All validation choices were frozen before test evaluation.",
    ]
    for model in MODELS:
        rows = read_csv(OUT / model / "test_summary.csv")
        all_rows.extend(rows)
        chosen = selection["per_model"][model]
        lines.extend(("", f"## {model}", "", "| feature | raw | norm | raw + logS | norm + logS |",
                      "|---|---:|---:|---:|---:|"))
        for group in GROUPS:
            cells = []
            for mode, variant in (("standalone", "raw"), ("standalone", "norm"),
                                  ("plus_logS", "raw"), ("plus_logS", "norm")):
                row = next(row for row in rows if (row["mode"], row["variant"], row["group"])
                           == (mode, variant, group))
                cells.append(f"{100*float(row['AUROC_mean']):.2f} / "
                             f"{100*float(row['HALL_AUPR_mean']):.2f}")
            lines.append(f"| {group} | " + " | ".join(cells) + " |")
        lines.append(f"\nValidation champion: {chosen['mode']}/{chosen['variant']}/{chosen['group']}.")
    write_csv(OUT / "test_summary.csv", all_rows)
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate() -> None:
    selection = json.loads((OUT / "selection.json").read_text())
    checked = 0
    metric_error = 0.0
    prediction_error = 0.0
    reload_error = 0.0
    for model in MODELS:
        data = prepare(model)
        labels, masks = data["labels"], data["masks"]
        chosen = selection["per_model"][model]
        for mode, variant, group in HEADS:
            features = matrix(data, mode, variant, group)
            for seed in SEEDS:
                trained = read(trained_path(model, mode, variant, group, seed))
                mean, scale = topk.trainer.scale_fit(features[masks["train"]], True)
                np.testing.assert_array_equal(trained["mean"], mean)
                np.testing.assert_array_equal(trained["scale"], scale)
                expected_epoch = 1 + int(np.argmin([r["val_loss"] for r in trained["history"]]))
                if trained["best_epoch"] != expected_epoch:
                    raise ValueError("Checkpoint is not minimum validation loss")
                validation = topk.trainer.metrics(
                    labels[masks["val"]], trained["validation_probabilities"])
                for key in validation:
                    metric_error = max(metric_error,
                                       abs(validation[key] - trained["validation"][key]))
                final_path = OUT / model / "final" / mode / variant / group / f"seed{seed}.pt"
                final = read(final_path)
                test = topk.trainer.metrics(labels[masks["test"]], final["test_probabilities"])
                for key in test:
                    metric_error = max(metric_error, abs(test[key] - final["test_metrics"][key]))
                with np.load(final_path.with_name(f"seed{seed}_predictions.npz")) as saved:
                    np.testing.assert_array_equal(saved["sample_id"], data["sample_id"])
                    np.testing.assert_array_equal(saved["label"], labels)
                    for split, probs in (("train", trained["train_probabilities"]),
                                         ("val", trained["validation_probabilities"]),
                                         ("test", final["test_probabilities"])):
                        prediction_error = max(prediction_error, float(np.max(np.abs(
                            saved["real_probability"][masks[split]] - probs))))
                if (mode, variant, group) == (chosen["mode"], chosen["variant"], chosen["group"]):
                    network = topk.trainer.SingleMLP(features.shape[1], CONFIG)
                    network.load_state_dict(trained["state_dict"])
                    for split, saved in (("train", trained["train_probabilities"]),
                                         ("val", trained["validation_probabilities"]),
                                         ("test", final["test_probabilities"])):
                        transformed = torch.as_tensor(topk.trainer.transform(
                            features[masks[split]], trained["mean"], trained["scale"]))
                        recomputed = topk.trainer.predict(network, transformed)
                        reload_error = max(reload_error, float(np.max(np.abs(recomputed - saved))))
                checked += 1
    report = {
        "schema": "semantic-attention-all-visual-validation-811-v1",
        "checked_heads": checked,
        "expected_heads": len(MODELS) * len(HEADS) * len(SEEDS),
        "metric_recompute_max_abs_error": metric_error,
        "prediction_file_max_abs_error": prediction_error,
        "champion_cpu_reload_max_abs_error": reload_error,
        "status": "pass",
        "validated_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json_save(report, OUT / "validation.json")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=(*MODELS, "all"), required=True)
    parser.add_argument("--stage", choices=("prepare", "train", "select", "test", "summary", "validate"), required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.stage == "select":
        select()
    elif args.stage == "summary":
        summarize()
    elif args.stage == "validate":
        validate()
    elif args.model == "all":
        parser.error("--model all is only valid for select/summary/validate")
    elif args.stage == "prepare":
        prepare(args.model)
    elif args.stage == "train":
        train(args.model, args.device)
    else:
        evaluate(args.model, args.device)


if __name__ == "__main__":
    main()
