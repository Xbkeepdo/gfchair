#!/usr/bin/env python3
"""Compare unlogged visual-only S_E with log1p(S_E) on the fixed ENDAC-811 split."""

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

from features.tc_fvpa_artifacts import atomic_json_save, atomic_torch_save
from scripts import train_semantic_attention_all_visual_811 as previous
from utils.io_utils import load_pkl

OUT = ROOT / "outputs/coco4000_512_endac_semantic_attention_raw_s_811"
ENDAC = ROOT / "outputs/coco4000_512_endac_811"
MODELS = previous.MODELS
SEEDS = previous.SEEDS
GROUPS = previous.GROUPS
CONFIG = previous.CONFIG
HEADS = (("reference", "S_only"), ("reference", "AE_only"),
         ("reference", "AE_plus_S")) + tuple(
    (variant, group) for variant in previous.VARIANTS for group in GROUPS
) + (("reference", "logS_only"), ("reference", "AE_plus_logS"))


def read(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare(model: str) -> dict:
    root = OUT / model
    target = root / "features.pt"
    protocol = {
        "schema": "semantic-attention-raw-S-811-v1",
        "model": model,
        "semantic_source": str(previous.OUT / model / "features.pt"),
        "AE_source": str(ENDAC / model / "features.pkl"),
        "strength": "raw-S heads use expm1(saved float32 log1p(S_E)); matched logged references use log1p(S_E)",
        "heads": [list(head) for head in HEADS],
        "split": "existing ENDAC image-level 3200/400/400",
        "seeds": list(SEEDS),
        "classifier": CONFIG,
        "selection": "validation AUROC then HALL-AUPR before test access",
    }
    protocol_path = root / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError(f"Changed protocol: {protocol_path}")
    else:
        atomic_json_save(protocol, protocol_path)
    if target.exists():
        return read(target)

    prior = previous.prepare(model)
    rows = load_pkl(ENDAC / model / "features.pkl")
    if [row["sample_id"] for row in rows] != list(prior["sample_id"]):
        raise ValueError(f"{model}: ENDAC mention order differs")
    np.testing.assert_array_equal([row["label"] for row in rows], prior["labels"])
    np.testing.assert_array_equal([row["image_id"] for row in rows], prior["image_id"])
    layers = prior["log1p_S"].shape[1]
    visual = np.stack([row["features"]["visual_only"] for row in rows]).reshape(-1, layers, 2)
    np.testing.assert_array_equal(visual[:, :, 1], prior["log1p_S"])
    strength = np.expm1(prior["log1p_S"]).astype(np.float32, copy=False)
    if not np.isfinite(strength).all() or np.any(strength < 0):
        raise ValueError(f"{model}: invalid raw S_E")
    data = {
        "schema": protocol["schema"], "model": model,
        "cubes": prior["cubes"],
        "AE_V": visual[:, :, 0].astype(np.float32, copy=False),
        "S_E": strength,
        "labels": prior["labels"], "masks": prior["masks"],
        "sample_id": prior["sample_id"], "image_id": prior["image_id"],
    }
    atomic_torch_save(data, target)
    print(f"PREPARED {model}: {len(rows)} mentions, {layers} layers", flush=True)
    return data


def matrix(data: dict, variant: str, group: str) -> np.ndarray:
    if variant == "reference":
        if group == "S_only":
            return data["S_E"]
        if group == "AE_only":
            return data["AE_V"]
        if group == "AE_plus_S":
            return np.concatenate((data["AE_V"], data["S_E"]), axis=1)
        logged = np.log1p(data["S_E"])
        if group == "logS_only":
            return logged
        if group == "AE_plus_logS":
            return np.concatenate((data["AE_V"], logged), axis=1)
        raise ValueError(f"Unknown reference: {group}")
    semantic = previous.topk.feature_matrix(data["cubes"][variant], group)
    return np.concatenate((semantic, data["S_E"]), axis=1)


def progress(model: str, stage: str, completed: int) -> None:
    atomic_json_save({
        "model": model, "stage": stage, "completed": completed,
        "total": len(HEADS) * len(SEEDS),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }, OUT / model / "progress.json")


def checkpoint(model: str, variant: str, group: str, seed: int) -> Path:
    return OUT / model / "validation" / variant / group / f"seed{seed}.pt"


def aggregate(rows: list[dict]) -> list[dict]:
    result = []
    for variant, group in HEADS:
        selected = [row for row in rows if (row["variant"], row["group"]) == (variant, group)]
        if len(selected) != len(SEEDS):
            raise ValueError(f"Incomplete {variant}/{group}")
        result.append({
            "model": selected[0]["model"], "variant": variant, "group": group,
            **{f"{metric}_{stat}": float(fn([float(row[metric]) for row in selected]))
               for metric in ("AUROC", "HALL_AUPR")
               for stat, fn in (("mean", np.mean), ("std", np.std))},
        })
    return result


def rank(row: dict) -> tuple[float, float, int]:
    return (float(row["AUROC_mean"]), float(row["HALL_AUPR_mean"]),
            -HEADS.index((row["variant"], row["group"])))


def train(model: str, device: str) -> None:
    data = prepare(model)
    labels, masks = data["labels"], data["masks"]
    rows = []
    completed = 0
    for variant, group in HEADS:
        features = matrix(data, variant, group)
        for seed in SEEDS:
            path = checkpoint(model, variant, group, seed)
            if path.exists():
                fitted = read(path)
            else:
                progress(model, f"{variant}/{group}/seed{seed}", completed)
                fitted = previous.topk.trainer.fit(
                    train_x=features[masks["train"]], train_y=labels[masks["train"]],
                    val_x=features[masks["val"]], val_y=labels[masks["val"]],
                    cfg=CONFIG, seed=seed, device=device,
                )
                fitted.update(model=model, variant=variant, group=group)
                atomic_torch_save(fitted, path)
            rows.append({"model": model, "variant": variant, "group": group,
                         "seed": seed, **fitted["validation"],
                         "best_epoch": fitted["best_epoch"]})
            completed += 1
            progress(model, f"{variant}/{group}/seed{seed}", completed)
            print("VAL", model, variant, group, seed, fitted["validation"], flush=True)
    write_csv(OUT / model / "validation_seed_metrics.csv", rows)
    write_csv(OUT / model / "validation_summary.csv", aggregate(rows))
    progress(model, "validation_complete", completed)


def select() -> None:
    summaries = {model: read_csv(OUT / model / "validation_summary.csv") for model in MODELS}
    selected = {
        model: max((row for row in rows if row["variant"] != "reference"), key=rank)
        for model, rows in summaries.items()
    }
    choice = {
        "schema": "semantic-attention-raw-S-selection-811-v1",
        "ranking": "three-seed validation AUROC then HALL-AUPR; no test used",
        "per_model": {model: {key: row[key] for key in ("variant", "group")}
                      for model, row in selected.items()},
        "frozen_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json_save(choice, OUT / "selection.json")
    print(json.dumps(choice, ensure_ascii=False, indent=2), flush=True)


def evaluate(model: str, device: str) -> None:
    choice = json.loads((OUT / "selection.json").read_text())
    data = prepare(model)
    labels, masks = data["labels"], data["masks"]
    splits = np.where(masks["train"], "train", np.where(masks["val"], "val", "test"))
    rows = []
    completed = 0
    for variant, group in HEADS:
        features = matrix(data, variant, group)
        for seed in SEEDS:
            fitted = read(checkpoint(model, variant, group, seed))
            network = previous.topk.trainer.SingleMLP(features.shape[1], CONFIG).to(device)
            network.load_state_dict(fitted["state_dict"])
            x = torch.as_tensor(previous.topk.trainer.transform(
                features[masks["test"]], fitted["mean"], fitted["scale"]), device=device)
            probabilities = previous.topk.trainer.predict(network, x)
            metrics = previous.topk.trainer.metrics(labels[masks["test"]], probabilities)
            path = OUT / model / "final" / variant / group / f"seed{seed}.pt"
            atomic_torch_save({"model": model, "variant": variant, "group": group,
                               "seed": seed, "test_metrics": metrics,
                               "test_probabilities": probabilities}, path)
            full = np.empty(len(labels), dtype=np.float32)
            for split, values in (("train", fitted["train_probabilities"]),
                                  ("val", fitted["validation_probabilities"]),
                                  ("test", probabilities)):
                full[masks[split]] = values
            np.savez_compressed(path.with_name(f"seed{seed}_predictions.npz"),
                                sample_id=np.asarray(data["sample_id"]),
                                image_id=data["image_id"], label=labels,
                                split=splits, real_probability=full)
            rows.append({"model": model, "variant": variant, "group": group,
                         "seed": seed, **metrics})
            completed += 1
            progress(model, "test_evaluation", completed)
    write_csv(OUT / model / "test_seed_metrics.csv", rows)
    summary = aggregate(rows)
    for row in summary:
        row["validation_champion"] = all(row[key] == choice["per_model"][model][key]
                                         for key in ("variant", "group"))
    write_csv(OUT / model / "test_summary.csv", summary)
    progress(model, "complete", completed)


def summarize() -> None:
    lines = [
        "# Raw S_E vs log1p(S_E), ENDAC-811",
        "",
        "Same full-visual features, image split, three seeds, and fixed 128/no-BN MLP.",
        "Raw S_E = expm1(saved float32 log1p(S_E)); validation selection was frozen before test.",
        "All entries are test AUROC / HALL-AUPR (%), three-seed means.",
    ]
    all_rows = []
    single_lines = [
        "# Each all-visual semantic-attention feature: raw S_E vs log1p(S_E)",
        "",
        "For each feature, raw/norm is fixed by the earlier logS three-seed validation"
        " AUROC, then HALL-AUPR; the same variant is used for S and logS.",
        "Test scores are three-seed means (%); delta is S minus logS in percentage points.",
    ]
    for model in MODELS:
        rows = read_csv(OUT / model / "test_summary.csv")
        old = read_csv(previous.OUT / model / "test_summary.csv")
        old_validation = read_csv(previous.OUT / model / "validation_summary.csv")
        all_rows.extend(rows)
        lines += ["", f"## {model}", "",
                  "| feature | raw + S | raw + logS | norm + S | norm + logS |",
                  "|---|---:|---:|---:|---:|"]
        for group in GROUPS:
            cells = []
            for variant in previous.VARIANTS:
                current = next(row for row in rows if (row["variant"], row["group"]) == (variant, group))
                logged = next(row for row in old if (row["mode"], row["variant"], row["group"])
                              == ("plus_logS", variant, group))
                for row in (current, logged):
                    cells.append(f"{100*float(row['AUROC_mean']):.2f} / "
                                 f"{100*float(row['HALL_AUPR_mean']):.2f}")
            lines.append(f"| {group} | " + " | ".join(cells) + " |")
        refs = {row["group"]: row for row in rows if row["variant"] == "reference"}
        lines += ["", "| reference | AUROC / HALL-AUPR |", "|---|---:|"]
        for group in ("S_only", "logS_only", "AE_only", "AE_plus_S", "AE_plus_logS"):
            row = refs[group]
            lines.append(f"| {group} | {100*float(row['AUROC_mean']):.2f} / "
                         f"{100*float(row['HALL_AUPR_mean']):.2f} |")
        single_lines += ["", f"## {model}", "",
                         "| feature | fixed variant | +S AUROC/AP | +logS AUROC/AP | delta pp |",
                         "|---|---|---:|---:|---:|"]
        for group in previous.FEATURE_NAMES:
            choice_log = max((row for row in old_validation
                              if row["mode"] == "plus_logS" and row["group"] == group),
                             key=lambda row: (float(row["AUROC_mean"]),
                                              float(row["HALL_AUPR_mean"])))
            result_s = next(row for row in rows
                            if (row["variant"], row["group"]) == (choice_log["variant"], group))
            result_log = next(row for row in old
                              if (row["mode"], row["variant"], row["group"])
                              == ("plus_logS", choice_log["variant"], group))
            score_s = [100 * float(result_s[key]) for key in ("AUROC_mean", "HALL_AUPR_mean")]
            score_log = [100 * float(result_log[key]) for key in ("AUROC_mean", "HALL_AUPR_mean")]
            single_lines.append(
                f"| {group} | {choice_log['variant']} | {score_s[0]:.2f} / {score_s[1]:.2f} | "
                f"{score_log[0]:.2f} / {score_log[1]:.2f} | "
                f"{score_s[0]-score_log[0]:+.2f} / {score_s[1]-score_log[1]:+.2f} |"
            )
    write_csv(OUT / "test_summary.csv", all_rows)
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "single_feature_summary.md").write_text(
        "\n".join(single_lines) + "\n", encoding="utf-8")


def validate() -> None:
    choice = json.loads((OUT / "selection.json").read_text())
    checked = 0
    error = 0.0
    reload_error = 0.0
    for model in MODELS:
        data = prepare(model)
        labels, masks = data["labels"], data["masks"]
        for variant, group in HEADS:
            features = matrix(data, variant, group)
            for seed in SEEDS:
                fitted = read(checkpoint(model, variant, group, seed))
                mean, scale = previous.topk.trainer.scale_fit(features[masks["train"]], True)
                np.testing.assert_array_equal(fitted["mean"], mean)
                np.testing.assert_array_equal(fitted["scale"], scale)
                expected_epoch = 1 + int(np.argmin([row["val_loss"] for row in fitted["history"]]))
                if fitted["best_epoch"] != expected_epoch:
                    raise ValueError("Checkpoint not minimum validation loss")
                final_path = OUT / model / "final" / variant / group / f"seed{seed}.pt"
                final = read(final_path)
                for split, probabilities, metrics in (
                    ("val", fitted["validation_probabilities"], fitted["validation"]),
                    ("test", final["test_probabilities"], final["test_metrics"]),
                ):
                    recomputed = previous.topk.trainer.metrics(labels[masks[split]], probabilities)
                    for key in metrics:
                        error = max(error, abs(recomputed[key] - metrics[key]))
                with np.load(final_path.with_name(f"seed{seed}_predictions.npz")) as saved:
                    np.testing.assert_array_equal(saved["sample_id"], data["sample_id"])
                    np.testing.assert_array_equal(saved["label"], labels)
                    for split, probabilities in (("train", fitted["train_probabilities"]),
                                                 ("val", fitted["validation_probabilities"]),
                                                 ("test", final["test_probabilities"])):
                        error = max(error, float(np.max(np.abs(
                            saved["real_probability"][masks[split]] - probabilities))))
                chosen = choice["per_model"][model]
                if (variant, group) in (("reference", "AE_plus_S"),
                                        ("reference", "AE_plus_logS"),
                                        (chosen["variant"], chosen["group"])):
                    network = previous.topk.trainer.SingleMLP(features.shape[1], CONFIG)
                    network.load_state_dict(fitted["state_dict"])
                    for split, probabilities in (("train", fitted["train_probabilities"]),
                                                 ("val", fitted["validation_probabilities"]),
                                                 ("test", final["test_probabilities"])):
                        x = torch.as_tensor(previous.topk.trainer.transform(
                            features[masks[split]], fitted["mean"], fitted["scale"]))
                        recomputed = previous.topk.trainer.predict(network, x)
                        reload_error = max(reload_error, float(np.max(np.abs(recomputed - probabilities))))
                checked += 1
    report = {"schema": "semantic-attention-raw-S-validation-811-v1",
              "checked_heads": checked, "expected_heads": len(MODELS)*len(HEADS)*len(SEEDS),
              "metric_prediction_max_abs_error": error,
              "champion_and_AE_cpu_reload_max_abs_error": reload_error,
              "status": "pass", "validated_at": datetime.now(timezone.utc).isoformat()}
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
