#!/usr/bin/env python3
"""Evaluate semantic-attention Top-K features with the fixed 811 probe."""

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
from features.semantic_attention_topk import FEATURE_NAMES
from features.tc_fvpa_artifacts import atomic_json_save, atomic_torch_save
from scripts import search_single_mlp_811 as trainer
from utils.io_utils import load_pkl


SOURCE = ROOT / "outputs/coco4000_512_endac_semantic_attention_topk"
OUT = ROOT / "outputs/coco4000_512_endac_semantic_attention_topk_detection_811"
COHORT = "four"
SCHEMA = "semantic-attention-topk-detection-811-v1"
MODELS = (
    "qwen2_5_vl_7b",
    "llava_1_5_7b",
    "qwen3_vl_8b",
    "internvl_2_5_8b",
)
SEEDS = (43, 44, 45)
VARIANTS = ("raw_top16", "raw_top32", "norm_top16", "norm_top32")
GROUPS = FEATURE_NAMES + ("all_six",)
CONFIG = {
    "width": 128,
    "standardize": True,
    "batch_norm": False,
    "dropout": 0.3,
    "activation": "relu",
    "learning_rate": 0.001,
    "weight_decay": 1.0e-5,
    "batch_size": 128,
    "max_epochs": 150,
    "patience": 20,
    "lr_patience": 6,
    "monitor": "val_loss",
    "early_stopping": True,
}


def configure(cohort: str) -> None:
    global SOURCE, OUT, COHORT, SCHEMA, MODELS, GROUPS
    if cohort == "prefix":
        COHORT = cohort
        SCHEMA = "semantic-attention-topk-prefix-811-v1"
        SOURCE = ROOT / "outputs/coco4000_512_endac_811"
        OUT = ROOT / "outputs/coco4000_512_endac_prefix_semantic_attention_topk_detection_811"
        MODELS = ("minigpt4_7b", "shikra_7b")
        base = FEATURE_NAMES + ("all_six",)
        GROUPS = base + tuple(f"{name}+logS" for name in base) + tuple(
            f"{name}+S" for name in base
        )


def read_torch(path: Path):
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


def feature_matrix(
    cube: np.ndarray, group: str, log_s: np.ndarray | None = None,
    raw_s: np.ndarray | None = None,
) -> np.ndarray:
    extra = None
    if group.endswith("+logS"):
        group, extra = group[:-5], log_s
    elif group.endswith("+S"):
        group, extra = group[:-2], raw_s
    matrix = cube.reshape(len(cube), -1) if group == "all_six" else cube[:, :, FEATURE_NAMES.index(group)]
    if extra is not None:
        matrix = np.concatenate((matrix, extra), axis=1)
    return matrix


def prepare(model: str) -> dict:
    source_path = SOURCE / model / (
        "features.pkl" if COHORT == "prefix" else "semantic_attention.pkl"
    )
    protocol = {
        "schema": SCHEMA,
        "model": model,
        "source": str(source_path),
        "variants": list(VARIANTS),
        "groups": list(GROUPS),
        "seeds": list(SEEDS),
        "config": CONFIG,
        "split": "existing ENDAC 3200/400/400 image split",
        "selection": "validation AUROC, then HALL-AUPR; frozen before test",
        "features": "full-layer vector for each scalar; all_six concatenates six full-layer vectors",
        "labels": "REAL=1; HALL-AUPR uses 1-p_REAL",
    }
    protocol_path = OUT / model / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError(f"Changed protocol: {protocol_path}")
    else:
        atomic_json_save(protocol, protocol_path)
    target = OUT / model / "features.pt"
    if target.exists():
        data = read_torch(target)
        if data.get("schema") != SCHEMA:
            raise ValueError(f"Unexpected cached feature schema: {target}")
        return data

    rows = load_pkl(source_path)
    semantic_rows = (
        [row["semantic_attention"] for row in rows]
        if COHORT == "prefix" else rows
    )
    cubes = {
        variant: np.stack([row["features"][variant] for row in semantic_rows]).astype(
            np.float32, copy=False
        )
        for variant in VARIANTS
    }
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    masks = {
        part: np.asarray([row["split"] == part for row in rows], dtype=bool)
        for part in ("train", "val", "test")
    }
    if any(not np.isfinite(value).all() for value in cubes.values()):
        raise ValueError(f"{model}: non-finite compact feature")
    if np.any(sum(mask.astype(np.int8) for mask in masks.values()) != 1):
        raise ValueError(f"{model}: invalid 811 split masks")
    data = {
        "schema": SCHEMA,
        "model": model,
        "feature_names": list(FEATURE_NAMES),
        "variants": list(VARIANTS),
        "groups": list(GROUPS),
        "cubes": cubes,
        "labels": labels,
        "masks": masks,
        "sample_id": [row["sample_id"] for row in rows],
        "image_id": np.asarray([row["image_id"] for row in rows], dtype=np.int64),
    }
    if COHORT == "prefix":
        data["logS"] = np.stack(
            [row["features"]["visual_only"][1::2] for row in rows]
        ).astype(np.float32)
        data["rawS"] = np.stack(
            [row["features"]["visual_only_gross_raw"] for row in rows]
        ).astype(np.float32)
    atomic_torch_save(data, target)
    return data


def progress(model: str, stage: str, completed: int, total: int, **extra) -> None:
    atomic_json_save(
        {
            "model": model,
            "stage": stage,
            "completed": completed,
            "total": total,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            **extra,
        },
        OUT / model / "progress.json",
    )


def result_path(model: str, variant: str, group: str, seed: int) -> Path:
    return OUT / model / "validation" / variant / group / f"seed{seed}.pt"


def train_model(model: str, device: str) -> None:
    data = prepare(model)
    labels, masks = data["labels"], data["masks"]
    total = len(VARIANTS) * len(GROUPS) * len(SEEDS)
    completed = 0
    rows = []
    for variant in VARIANTS:
        cube = data["cubes"][variant]
        for group in GROUPS:
            matrix = feature_matrix(cube, group, data.get("logS"), data.get("rawS"))
            for seed in SEEDS:
                path = result_path(model, variant, group, seed)
                if path.exists():
                    result = read_torch(path)
                else:
                    progress(
                        model,
                        f"{variant}/{group}/seed{seed}",
                        completed,
                        total,
                    )
                    result = trainer.fit(
                        train_x=matrix[masks["train"]],
                        train_y=labels[masks["train"]],
                        val_x=matrix[masks["val"]],
                        val_y=labels[masks["val"]],
                        cfg=CONFIG,
                        seed=seed,
                        device=device,
                    )
                    result.update(model=model, variant=variant, group=group)
                    atomic_torch_save(result, path)
                completed += 1
                rows.append(
                    {
                        "model": model,
                        "variant": variant,
                        "group": group,
                        "seed": seed,
                        **result["validation"],
                        "best_epoch": result["best_epoch"],
                    }
                )
                progress(
                    model,
                    f"{variant}/{group}/seed{seed}",
                    completed,
                    total,
                )
                print(
                    "VAL",
                    model,
                    variant,
                    group,
                    seed,
                    result["validation"],
                    flush=True,
                )
    write_csv(OUT / model / "validation_seed_metrics.csv", rows)
    summary = summarize_seed_rows(rows, metric_prefix="validation")
    write_csv(OUT / model / "validation_summary.csv", summary)
    progress(model, "validation_complete", total, total, status="complete")


def summarize_seed_rows(rows: list[dict], metric_prefix: str) -> list[dict]:
    output = []
    for variant in VARIANTS:
        for group in GROUPS:
            selected = [
                row
                for row in rows
                if row["variant"] == variant and row["group"] == group
            ]
            if len(selected) != len(SEEDS):
                raise ValueError(f"Incomplete {metric_prefix}: {variant}/{group}")
            output.append(
                {
                    "model": selected[0]["model"],
                    "variant": variant,
                    "group": group,
                    **{
                        f"{metric}_{stat}": float(
                            function([float(row[metric]) for row in selected])
                        )
                        for metric in ("AUROC", "HALL_AUPR")
                        for stat, function in (("mean", np.mean), ("std", np.std))
                    },
                }
            )
    return output


def rank(row: dict) -> tuple[float, float, int, int]:
    return (
        float(row["AUROC_mean"]),
        float(row["HALL_AUPR_mean"]),
        -VARIANTS.index(row["variant"]),
        -GROUPS.index(row["group"]),
    )


def freeze_selection() -> dict:
    by_model = {}
    all_rows = []
    for model in MODELS:
        path = OUT / model / "validation_summary.csv"
        if not path.exists():
            raise FileNotFoundError(f"Validation is incomplete: {path}")
        rows = read_csv(path)
        for row in rows:
            for key in ("AUROC_mean", "HALL_AUPR_mean"):
                row[key] = float(row[key])
        by_model[model] = max(rows, key=rank)
        all_rows.extend(rows)

    macro = []
    for variant in VARIANTS:
        for group in GROUPS:
            selected = [
                row
                for row in all_rows
                if row["variant"] == variant and row["group"] == group
            ]
            macro.append(
                {
                    "variant": variant,
                    "group": group,
                    "AUROC_mean": float(np.mean([row["AUROC_mean"] for row in selected])),
                    "HALL_AUPR_mean": float(
                        np.mean([row["HALL_AUPR_mean"] for row in selected])
                    ),
                }
            )
    global_champion = max(macro, key=rank)
    selection = {
        "schema": "semantic-attention-topk-selection-811-v1",
        "ranking": "three-seed validation AUROC, then HALL-AUPR; test unseen",
        "per_model": {
            model: {"variant": row["variant"], "group": row["group"]}
            for model, row in by_model.items()
        },
        "global": {
            "variant": global_champion["variant"],
            "group": global_champion["group"],
        },
        "frozen_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json_save(selection, OUT / "selection.json")
    write_csv(OUT / "validation_macro_summary.csv", sorted(macro, key=rank, reverse=True))
    print(json.dumps(selection, ensure_ascii=False, indent=2), flush=True)
    return selection


def evaluate_model(model: str, device: str) -> None:
    selection_path = OUT / "selection.json"
    if not selection_path.exists():
        raise FileNotFoundError("Freeze validation selection before test evaluation")
    selection = json.loads(selection_path.read_text())
    data = prepare(model)
    labels, masks = data["labels"], data["masks"]
    rows = []
    total = len(VARIANTS) * len(GROUPS) * len(SEEDS)
    completed = 0
    for variant in VARIANTS:
        cube = data["cubes"][variant]
        for group in GROUPS:
            matrix = feature_matrix(cube, group, data.get("logS"), data.get("rawS"))
            for seed in SEEDS:
                trained = read_torch(result_path(model, variant, group, seed))
                network = trainer.SingleMLP(matrix.shape[1], CONFIG).to(device)
                network.load_state_dict(trained["state_dict"])
                transformed = {
                    part: torch.as_tensor(
                        trainer.transform(matrix[mask], trained["mean"], trained["scale"]),
                        device=device,
                    )
                    for part, mask in masks.items()
                }
                probabilities = {
                    "train": trained["train_probabilities"],
                    "val": trained["validation_probabilities"],
                    "test": trainer.predict(network, transformed["test"]),
                }
                metrics = trainer.metrics(labels[masks["test"]], probabilities["test"])
                threshold = select_detection_threshold(
                    labels[masks["train"]],
                    1.0 - probabilities["train"],
                    positive_class="real",
                )
                final = {
                    "model": model,
                    "variant": variant,
                    "group": group,
                    "seed": seed,
                    "validation": trained["validation"],
                    "test_metrics": metrics,
                    "threshold": threshold,
                    "threshold_reports": {
                        name: evaluate_detection_scores(
                            labels[masks["test"]],
                            1.0 - probabilities["test"],
                            value,
                            positive_class="real",
                        )
                        for name, value in (("train_f1", threshold), ("fixed_0.5", 0.5))
                    },
                    "test_probabilities": probabilities["test"],
                }
                final_path = OUT / model / "final" / variant / group / f"seed{seed}.pt"
                atomic_torch_save(final, final_path)
                full = np.empty(len(labels), dtype=np.float32)
                for part, values in probabilities.items():
                    full[masks[part]] = np.asarray(values, dtype=np.float32)
                np.savez_compressed(
                    final_path.with_name(f"seed{seed}_predictions.npz"),
                    sample_id=np.asarray(data["sample_id"]),
                    image_id=data["image_id"],
                    label=labels,
                    split=np.asarray(
                        [
                            "train" if masks["train"][index] else "val" if masks["val"][index] else "test"
                            for index in range(len(labels))
                        ]
                    ),
                    real_probability=full,
                )
                rows.append(
                    {
                        "model": model,
                        "variant": variant,
                        "group": group,
                        "seed": seed,
                        **metrics,
                    }
                )
                completed += 1
                progress(model, "test_evaluation", completed, total)
                del network, transformed
    write_csv(OUT / model / "test_seed_metrics.csv", rows)
    summary = summarize_seed_rows(rows, metric_prefix="test")
    for row in summary:
        row["per_model_champion"] = (
            row["variant"] == selection["per_model"][model]["variant"]
            and row["group"] == selection["per_model"][model]["group"]
        )
        row["global_champion"] = (
            row["variant"] == selection["global"]["variant"]
            and row["group"] == selection["global"]["group"]
        )
    write_csv(OUT / model / "test_summary.csv", summary)
    progress(model, "complete", total, total, status="complete")


def summarize_all() -> None:
    selection = json.loads((OUT / "selection.json").read_text())
    rows = []
    for model in MODELS:
        rows.extend(read_csv(OUT / model / "test_summary.csv"))
    write_csv(OUT / "test_summary.csv", rows)

    if COHORT == "prefix":
        for model in MODELS:
            validation = {
                (row["variant"], row["group"]): row
                for row in read_csv(OUT / model / "validation_summary.csv")
            }
            chosen = selection["per_model"][model]
            lines = [
                f"# {model} semantic-attention Top-K ENDAC-811",
                "",
                "3200/400/400 image split; seeds 43/44/45. Mean ± population std.",
                "Selection uses validation AUROC, then HALL-AUPR; test is held out.",
                "",
                "| variant | feature | validation AUROC (%) | test AUROC (%) | test HALL-AUPR (%) |",
                "|---|---|---:|---:|---:|",
            ]
            for row in (row for row in rows if row["model"] == model):
                key = (row["variant"], row["group"])
                marker = " (selected)" if key == (chosen["variant"], chosen["group"]) else ""
                lines.append(
                    f"| {row['variant']} | {row['group']}{marker} | "
                    f"{100 * float(validation[key]['AUROC_mean']):.2f} | "
                    f"{100 * float(row['AUROC_mean']):.2f} ± {100 * float(row['AUROC_std']):.2f} | "
                    f"{100 * float(row['HALL_AUPR_mean']):.2f} ± "
                    f"{100 * float(row['HALL_AUPR_std']):.2f} |"
                )
            (OUT / model / "summary.md").write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )

    macro = []
    for variant in VARIANTS:
        for group in GROUPS:
            selected = [
                row
                for row in rows
                if row["variant"] == variant and row["group"] == group
            ]
            macro.append(
                {
                    "variant": variant,
                    "group": group,
                    "AUROC_mean": float(
                        np.mean([float(row["AUROC_mean"]) for row in selected])
                    ),
                    "HALL_AUPR_mean": float(
                        np.mean([float(row["HALL_AUPR_mean"]) for row in selected])
                    ),
                    "global_champion": (
                        variant == selection["global"]["variant"]
                        and group == selection["global"]["group"]
                    ),
                }
            )
    write_csv(OUT / "test_macro_summary.csv", sorted(macro, key=rank, reverse=True))

    lines = [
        "# Semantic-attention Top-K ENDAC-811 detection",
        "",
        "固定 3200/400/400 图片划分；seeds 43/44/45；train-only z-score；",
        "单隐藏 128/ReLU/dropout .3、无 BN；最低 validation BCE loss checkpoint。",
        f"所有 {len(VARIANTS) * len(GROUPS)} 个预注册特征组在访问 test 前完成 validation 选择。",
        "",
        "## Validation-selected champions",
        "",
        "| model | variant/group | test AUROC (%) | test HALL-AUPR (%) |",
        "|---|---|---:|---:|",
    ]
    for model in MODELS:
        chosen = selection["per_model"][model]
        row = next(
            row
            for row in rows
            if row["model"] == model
            and row["variant"] == chosen["variant"]
            and row["group"] == chosen["group"]
        )
        lines.append(
            f"| {model} | {chosen['variant']}/{chosen['group']} | "
            f"{100 * float(row['AUROC_mean']):.2f} ± {100 * float(row['AUROC_std']):.2f} | "
            f"{100 * float(row['HALL_AUPR_mean']):.2f} ± "
            f"{100 * float(row['HALL_AUPR_std']):.2f} |"
        )
    lines.extend(
        [
            "",
            f"## {len(MODELS)}-model test macro average (supplementary)",
            "",
            "| variant/group | AUROC (%) | HALL-AUPR (%) |",
            "|---|---:|---:|",
        ]
    )
    for row in sorted(macro, key=rank, reverse=True):
        marker = " **(global validation champion)**" if row["global_champion"] else ""
        lines.append(
            f"| {row['variant']}/{row['group']}{marker} | "
            f"{100 * row['AUROC_mean']:.2f} | {100 * row['HALL_AUPR_mean']:.2f} |"
        )
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_all() -> None:
    selection = json.loads((OUT / "selection.json").read_text())
    checked = 0
    metric_error = 0.0
    prediction_error = 0.0
    reload_error = 0.0
    for model in MODELS:
        data = prepare(model)
        labels, masks = data["labels"], data["masks"]
        chosen = selection["per_model"][model]
        for variant in VARIANTS:
            cube = data["cubes"][variant]
            for group in GROUPS:
                matrix = feature_matrix(cube, group, data.get("logS"), data.get("rawS"))
                for seed in SEEDS:
                    trained = read_torch(result_path(model, variant, group, seed))
                    expected_mean, expected_scale = trainer.scale_fit(
                        matrix[masks["train"]], True
                    )
                    np.testing.assert_array_equal(trained["mean"], expected_mean)
                    np.testing.assert_array_equal(trained["scale"], expected_scale)
                    expected_epoch = 1 + int(
                        np.argmin([row["val_loss"] for row in trained["history"]])
                    )
                    if int(trained["best_epoch"]) != expected_epoch:
                        raise ValueError("Checkpoint is not minimum validation loss")
                    validation = trainer.metrics(
                        labels[masks["val"]], trained["validation_probabilities"]
                    )
                    for key, value in validation.items():
                        metric_error = max(
                            metric_error, abs(value - trained["validation"][key])
                        )

                    final_path = (
                        OUT / model / "final" / variant / group / f"seed{seed}.pt"
                    )
                    final = read_torch(final_path)
                    test = trainer.metrics(
                        labels[masks["test"]], final["test_probabilities"]
                    )
                    for key, value in test.items():
                        metric_error = max(
                            metric_error, abs(value - final["test_metrics"][key])
                        )
                    with np.load(
                        final_path.with_name(f"seed{seed}_predictions.npz")
                    ) as saved:
                        np.testing.assert_array_equal(saved["label"], labels)
                        np.testing.assert_array_equal(saved["image_id"], data["image_id"])
                        np.testing.assert_array_equal(saved["sample_id"], data["sample_id"])
                        full = saved["real_probability"]
                        for part, values in (
                            ("train", trained["train_probabilities"]),
                            ("val", trained["validation_probabilities"]),
                            ("test", final["test_probabilities"]),
                        ):
                            prediction_error = max(
                                prediction_error,
                                float(
                                    np.max(
                                        np.abs(
                                            full[masks[part]]
                                            - np.asarray(values, dtype=np.float32)
                                        )
                                    )
                                ),
                            )
                    if variant == chosen["variant"] and group == chosen["group"]:
                        network = trainer.SingleMLP(matrix.shape[1], CONFIG)
                        network.load_state_dict(trained["state_dict"])
                        for part, saved in (
                            ("train", trained["train_probabilities"]),
                            ("val", trained["validation_probabilities"]),
                            ("test", final["test_probabilities"]),
                        ):
                            transformed = torch.as_tensor(
                                trainer.transform(
                                    matrix[masks[part]],
                                    trained["mean"],
                                    trained["scale"],
                                )
                            )
                            recomputed = trainer.predict(network, transformed)
                            reload_error = max(
                                reload_error,
                                float(np.max(np.abs(recomputed - saved))),
                            )
                    checked += 1
    report = {
        "schema": "semantic-attention-topk-detection-811-validation-v1",
        "checked_heads": checked,
        "expected_heads": len(MODELS) * len(VARIANTS) * len(GROUPS) * len(SEEDS),
        "metric_recompute_max_abs_error": metric_error,
        "prediction_file_max_abs_error": prediction_error,
        "champion_cpu_reload_max_abs_error": reload_error,
        "status": "pass",
        "validated_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json_save(report, OUT / "validation.json")
    print(json.dumps(report, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cohort", choices=("four", "prefix"), default="four")
    parser.add_argument(
        "--stage",
        choices=("prepare", "train", "select", "test", "summary", "validate"),
        required=True,
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    configure(args.cohort)
    if args.model not in (*MODELS, "all"):
        parser.error(f"unknown model {args.model!r}; choose from {MODELS}")
    torch.set_num_threads(1)
    if args.stage == "select":
        freeze_selection()
    elif args.stage == "summary":
        summarize_all()
    elif args.stage == "validate":
        validate_all()
    elif args.model == "all":
        parser.error("--model all is only valid with --stage select/summary/validate")
    elif args.stage == "prepare":
        prepare(args.model)
    elif args.stage == "train":
        train_model(args.model, args.device)
    else:
        evaluate_model(args.model, args.device)


if __name__ == "__main__":
    main()
