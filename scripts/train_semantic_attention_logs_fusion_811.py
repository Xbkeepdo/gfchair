#!/usr/bin/env python3
"""Fuse semantic-attention Top-K features with visual-only log1p(S_E)."""

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
from scripts import train_semantic_attention_topk_811 as base
from utils.io_utils import load_pkl


ENDAC_SOURCE = ROOT / "outputs/coco4000_512_endac_811"
OUT = ROOT / "outputs/coco4000_512_endac_semantic_attention_logs_fusion_811"
MODELS = base.MODELS
SEEDS = base.SEEDS
VARIANTS = base.VARIANTS
GROUPS = base.GROUPS
CONFIG = base.CONFIG
REFERENCE = ("reference", "log1p_S")
FUSIONS = tuple((variant, group) for variant in VARIANTS for group in GROUPS)
HEADS = (REFERENCE,) + FUSIONS


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


def prepare(model: str) -> dict:
    semantic = base.prepare(model)
    target = OUT / model / "features.pt"
    protocol = {
        "schema": "semantic-attention-logS-fusion-811-v1",
        "model": model,
        "semantic_source": str(base.OUT / model / "features.pt"),
        "strength_source": str(ENDAC_SOURCE / model / "features.pkl"),
        "strength": "per-layer visual-only conditional-path log1p(S_E)",
        "fusions": [list(value) for value in FUSIONS],
        "reference": list(REFERENCE),
        "seeds": list(SEEDS),
        "config": CONFIG,
        "split": "existing ENDAC 3200/400/400 image split",
        "selection": "validation AUROC, then HALL-AUPR; frozen before test",
    }
    protocol_path = OUT / model / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError(f"Changed protocol: {protocol_path}")
    else:
        atomic_json_save(protocol, protocol_path)
    if target.exists():
        strength = read_torch(target)
    else:
        rows = load_pkl(ENDAC_SOURCE / model / "features.pkl")
        if [row["sample_id"] for row in rows] != list(semantic["sample_id"]):
            raise ValueError(f"{model}: semantic and ENDAC mention order differ")
        np.testing.assert_array_equal(
            [row["label"] for row in rows], semantic["labels"]
        )
        np.testing.assert_array_equal(
            [row["image_id"] for row in rows], semantic["image_id"]
        )
        expected_split = [
            "train"
            if semantic["masks"]["train"][index]
            else "val"
            if semantic["masks"]["val"][index]
            else "test"
            for index in range(len(rows))
        ]
        np.testing.assert_array_equal([row["split"] for row in rows], expected_split)
        layers = semantic["cubes"][VARIANTS[0]].shape[1]
        visual = np.stack([row["features"]["visual_only"] for row in rows])
        visual = visual.reshape(len(rows), layers, 2)
        log_s = visual[:, :, 1].astype(np.float32, copy=False)
        if not np.isfinite(log_s).all() or np.any(log_s < 0):
            raise ValueError(f"{model}: invalid visual-only log1p(S_E)")
        strength = {
            "schema": "semantic-attention-logS-fusion-811-v1",
            "model": model,
            "log1p_S": log_s,
        }
        atomic_torch_save(strength, target)
    if strength.get("schema") != "semantic-attention-logS-fusion-811-v1":
        raise ValueError(f"Unexpected fusion cache: {target}")
    if strength["log1p_S"].shape != semantic["cubes"][VARIANTS[0]].shape[:2]:
        raise ValueError(f"{model}: log1p(S_E) shape mismatch")
    return {**semantic, "log1p_S": strength["log1p_S"]}


def matrix(data: dict, variant: str, group: str) -> np.ndarray:
    if (variant, group) == REFERENCE:
        return data["log1p_S"]
    semantic = base.feature_matrix(data["cubes"][variant], group)
    return np.concatenate((semantic, data["log1p_S"]), axis=1)


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
    total = len(HEADS) * len(SEEDS)
    completed = 0
    rows = []
    for variant, group in HEADS:
        features = matrix(data, variant, group)
        for seed in SEEDS:
            path = result_path(model, variant, group, seed)
            if path.exists():
                result = read_torch(path)
            else:
                progress(model, f"{variant}/{group}/seed{seed}", completed, total)
                result = base.trainer.fit(
                    train_x=features[masks["train"]],
                    train_y=labels[masks["train"]],
                    val_x=features[masks["val"]],
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
            progress(model, f"{variant}/{group}/seed{seed}", completed, total)
            print("VAL", model, variant, group, seed, result["validation"], flush=True)
    write_csv(OUT / model / "validation_seed_metrics.csv", rows)
    write_csv(OUT / model / "validation_summary.csv", summarize_rows(rows))
    progress(model, "validation_complete", total, total, status="complete")


def summarize_rows(rows: list[dict]) -> list[dict]:
    output = []
    for variant, group in HEADS:
        selected = [
            row
            for row in rows
            if row["variant"] == variant and row["group"] == group
        ]
        if len(selected) != len(SEEDS):
            raise ValueError(f"Incomplete head: {variant}/{group}")
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


def freeze_selection() -> None:
    summaries = {}
    for model in MODELS:
        rows = read_csv(OUT / model / "validation_summary.csv")
        fusions = [row for row in rows if (row["variant"], row["group"]) != REFERENCE]
        summaries[model] = fusions
    per_model = {model: max(rows, key=rank) for model, rows in summaries.items()}
    macro = []
    for variant, group in FUSIONS:
        selected = [
            next(
                row
                for row in summaries[model]
                if row["variant"] == variant and row["group"] == group
            )
            for model in MODELS
        ]
        macro.append(
            {
                "variant": variant,
                "group": group,
                "AUROC_mean": float(np.mean([float(row["AUROC_mean"]) for row in selected])),
                "HALL_AUPR_mean": float(
                    np.mean([float(row["HALL_AUPR_mean"]) for row in selected])
                ),
            }
        )
    global_champion = max(macro, key=rank)
    selection = {
        "schema": "semantic-attention-logS-fusion-selection-811-v1",
        "ranking": "three-seed validation AUROC, then HALL-AUPR; test unseen",
        "per_model": {
            model: {"variant": row["variant"], "group": row["group"]}
            for model, row in per_model.items()
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


def evaluate_model(model: str, device: str) -> None:
    selection = json.loads((OUT / "selection.json").read_text())
    data = prepare(model)
    labels, masks = data["labels"], data["masks"]
    total = len(HEADS) * len(SEEDS)
    completed = 0
    rows = []
    split_names = np.asarray(
        [
            "train" if masks["train"][index] else "val" if masks["val"][index] else "test"
            for index in range(len(labels))
        ]
    )
    for variant, group in HEADS:
        features = matrix(data, variant, group)
        for seed in SEEDS:
            trained = read_torch(result_path(model, variant, group, seed))
            network = base.trainer.SingleMLP(features.shape[1], CONFIG).to(device)
            network.load_state_dict(trained["state_dict"])
            test_x = torch.as_tensor(
                base.trainer.transform(
                    features[masks["test"]], trained["mean"], trained["scale"]
                ),
                device=device,
            )
            test_probability = base.trainer.predict(network, test_x)
            metrics = base.trainer.metrics(labels[masks["test"]], test_probability)
            threshold = select_detection_threshold(
                labels[masks["train"]],
                1.0 - trained["train_probabilities"],
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
                        1.0 - test_probability,
                        value,
                        positive_class="real",
                    )
                    for name, value in (("train_f1", threshold), ("fixed_0.5", 0.5))
                },
                "test_probabilities": test_probability,
            }
            final_path = OUT / model / "final" / variant / group / f"seed{seed}.pt"
            atomic_torch_save(final, final_path)
            full = np.empty(len(labels), dtype=np.float32)
            full[masks["train"]] = trained["train_probabilities"]
            full[masks["val"]] = trained["validation_probabilities"]
            full[masks["test"]] = test_probability
            np.savez_compressed(
                final_path.with_name(f"seed{seed}_predictions.npz"),
                sample_id=np.asarray(data["sample_id"]),
                image_id=data["image_id"],
                label=labels,
                split=split_names,
                real_probability=full,
            )
            rows.append(
                {"model": model, "variant": variant, "group": group, "seed": seed, **metrics}
            )
            completed += 1
            progress(model, "test_evaluation", completed, total)
    write_csv(OUT / model / "test_seed_metrics.csv", rows)
    summary = summarize_rows(rows)
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
    semantic_rows = base.read_csv(base.OUT / "test_summary.csv")
    comparison = []
    for model in MODELS:
        chosen = selection["per_model"][model]
        fusion = next(
            row
            for row in rows
            if row["model"] == model
            and row["variant"] == chosen["variant"]
            and row["group"] == chosen["group"]
        )
        standalone = next(
            row
            for row in semantic_rows
            if row["model"] == model
            and row["variant"] == chosen["variant"]
            and row["group"] == chosen["group"]
        )
        strength = next(
            row
            for row in rows
            if row["model"] == model and (row["variant"], row["group"]) == REFERENCE
        )
        comparison.append(
            {
                "model": model,
                "variant": chosen["variant"],
                "group": chosen["group"],
                **{
                    f"fusion_{metric}": float(fusion[f"{metric}_mean"])
                    for metric in ("AUROC", "HALL_AUPR")
                },
                **{
                    f"fusion_{metric}_std": float(fusion[f"{metric}_std"])
                    for metric in ("AUROC", "HALL_AUPR")
                },
                **{
                    f"standalone_{metric}": float(standalone[f"{metric}_mean"])
                    for metric in ("AUROC", "HALL_AUPR")
                },
                **{
                    f"logS_{metric}": float(strength[f"{metric}_mean"])
                    for metric in ("AUROC", "HALL_AUPR")
                },
            }
        )
    write_csv(OUT / "comparison.csv", comparison)
    lines = [
        "# Semantic-attention + log1p(S_E) ENDAC-811",
        "",
        "固定 3200/400/400；seeds 43/44/45；train-only z-score；",
        "单隐藏 128/ReLU/dropout .3、无 BN；validation 选择在 test 前冻结。",
        "",
        "| model | validation champion | fusion AUROC (%) | fusion AP (%) |",
        "|---|---|---:|---:|",
    ]
    for row in comparison:
        lines.append(
            f"| {row['model']} | {row['variant']}/{row['group']}+logS | "
            f"{100*row['fusion_AUROC']:.2f} ± {100*row['fusion_AUROC_std']:.2f} | "
            f"{100*row['fusion_HALL_AUPR']:.2f} ± "
            f"{100*row['fusion_HALL_AUPR_std']:.2f} |"
        )
    lines.extend(
        [
            "",
            "| model | standalone AUROC / AP | logS-only AUROC / AP | fusion−standalone (pp) | fusion−logS (pp) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in comparison:
        lines.append(
            f"| {row['model']} | {100*row['standalone_AUROC']:.2f} / "
            f"{100*row['standalone_HALL_AUPR']:.2f} | {100*row['logS_AUROC']:.2f} / "
            f"{100*row['logS_HALL_AUPR']:.2f} | "
            f"{100*(row['fusion_AUROC']-row['standalone_AUROC']):+.2f} / "
            f"{100*(row['fusion_HALL_AUPR']-row['standalone_HALL_AUPR']):+.2f} | "
            f"{100*(row['fusion_AUROC']-row['logS_AUROC']):+.2f} / "
            f"{100*(row['fusion_HALL_AUPR']-row['logS_HALL_AUPR']):+.2f} |"
        )
    macro = {
        key: float(np.mean([row[key] for row in comparison]))
        for key in (
            "fusion_AUROC",
            "fusion_HALL_AUPR",
            "standalone_AUROC",
            "standalone_HALL_AUPR",
            "logS_AUROC",
            "logS_HALL_AUPR",
        )
    }
    lines.extend(
        [
            "",
            f"逐模型 champion 宏平均：fusion "
            f"{100*macro['fusion_AUROC']:.2f}/{100*macro['fusion_HALL_AUPR']:.2f}；"
            f"standalone {100*macro['standalone_AUROC']:.2f}/"
            f"{100*macro['standalone_HALL_AUPR']:.2f}；logS-only "
            f"{100*macro['logS_AUROC']:.2f}/{100*macro['logS_HALL_AUPR']:.2f}。",
        ]
    )
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summarize_single_fusions()


def summarize_single_fusions() -> None:
    semantic_rows = base.read_csv(base.OUT / "test_summary.csv")
    rows = []
    for model in MODELS:
        validation = read_csv(OUT / model / "validation_summary.csv")
        fusion_test = read_csv(OUT / model / "test_summary.csv")
        log_s = next(row for row in fusion_test if (row["variant"], row["group"]) == REFERENCE)
        for group in base.FEATURE_NAMES:
            selected = max(
                (row for row in validation if row["group"] == group),
                key=rank,
            )
            variant = selected["variant"]
            fusion = next(
                row
                for row in fusion_test
                if row["variant"] == variant and row["group"] == group
            )
            standalone = next(
                row
                for row in semantic_rows
                if row["model"] == model
                and row["variant"] == variant
                and row["group"] == group
            )
            record = {"model": model, "group": group, "variant": variant}
            for metric in ("AUROC", "HALL_AUPR"):
                fusion_value = float(fusion[f"{metric}_mean"])
                standalone_value = float(standalone[f"{metric}_mean"])
                log_s_value = float(log_s[f"{metric}_mean"])
                record.update(
                    {
                        f"fusion_{metric}_mean": fusion_value,
                        f"fusion_{metric}_std": float(fusion[f"{metric}_std"]),
                        f"standalone_{metric}_mean": standalone_value,
                        f"logS_{metric}_mean": log_s_value,
                        f"vs_standalone_{metric}_pp": 100.0
                        * (fusion_value - standalone_value),
                        f"vs_logS_{metric}_pp": 100.0 * (fusion_value - log_s_value),
                    }
                )
            rows.append(record)
    write_csv(OUT / "single_feature_comparison.csv", rows)

    macro = []
    for group in base.FEATURE_NAMES:
        selected = [row for row in rows if row["group"] == group]
        macro.append(
            {
                "group": group,
                **{
                    key: float(np.mean([row[key] for row in selected]))
                    for key in (
                        "fusion_AUROC_mean",
                        "fusion_HALL_AUPR_mean",
                        "vs_standalone_AUROC_pp",
                        "vs_standalone_HALL_AUPR_pp",
                        "vs_logS_AUROC_pp",
                        "vs_logS_HALL_AUPR_pp",
                    )
                },
            }
        )
    write_csv(OUT / "single_feature_macro_summary.csv", macro)

    lines = [
        "# Single semantic-attention feature + log1p(S_E)",
        "",
        "每个模型×单特征仅用三 seed validation AUROC/HALL-AUPR "
        "在 raw/norm×top16/32 中选择变体，然后报告对应 test。"
        "standalone 增量始终与该选中变体的未拼接结果比较。",
        "",
        "## Four-model macro average",
        "",
        "| single feature + logS | AUROC / AP (%) | vs standalone (pp) | vs logS-only (pp) |",
        "|---|---:|---:|---:|",
    ]
    for row in macro:
        lines.append(
            f"| {row['group']} + logS | {100*row['fusion_AUROC_mean']:.2f} / "
            f"{100*row['fusion_HALL_AUPR_mean']:.2f} | "
            f"{row['vs_standalone_AUROC_pp']:+.2f} / "
            f"{row['vs_standalone_HALL_AUPR_pp']:+.2f} | "
            f"{row['vs_logS_AUROC_pp']:+.2f} / "
            f"{row['vs_logS_HALL_AUPR_pp']:+.2f} |"
        )
    lines.extend(
        [
            "",
            "## Per-model results",
            "",
            "| model | feature + logS | validation-selected variant | AUROC / AP (%) | vs standalone (pp) | vs logS-only (pp) |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['group']} + logS | {row['variant']} | "
            f"{100*row['fusion_AUROC_mean']:.2f} / "
            f"{100*row['fusion_HALL_AUPR_mean']:.2f} | "
            f"{row['vs_standalone_AUROC_pp']:+.2f} / "
            f"{row['vs_standalone_HALL_AUPR_pp']:+.2f} | "
            f"{row['vs_logS_AUROC_pp']:+.2f} / "
            f"{row['vs_logS_HALL_AUPR_pp']:+.2f} |"
        )
    (OUT / "single_feature_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


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
        for variant, group in HEADS:
            features = matrix(data, variant, group)
            for seed in SEEDS:
                trained = read_torch(result_path(model, variant, group, seed))
                mean, scale = base.trainer.scale_fit(features[masks["train"]], True)
                np.testing.assert_array_equal(trained["mean"], mean)
                np.testing.assert_array_equal(trained["scale"], scale)
                expected_epoch = 1 + int(
                    np.argmin([row["val_loss"] for row in trained["history"]])
                )
                if trained["best_epoch"] != expected_epoch:
                    raise ValueError("Checkpoint is not minimum validation loss")
                validation = base.trainer.metrics(
                    labels[masks["val"]], trained["validation_probabilities"]
                )
                for key in validation:
                    metric_error = max(
                        metric_error, abs(validation[key] - trained["validation"][key])
                    )
                final_path = OUT / model / "final" / variant / group / f"seed{seed}.pt"
                final = read_torch(final_path)
                test = base.trainer.metrics(
                    labels[masks["test"]], final["test_probabilities"]
                )
                for key in test:
                    metric_error = max(
                        metric_error, abs(test[key] - final["test_metrics"][key])
                    )
                with np.load(final_path.with_name(f"seed{seed}_predictions.npz")) as saved:
                    full = saved["real_probability"]
                    np.testing.assert_array_equal(saved["sample_id"], data["sample_id"])
                    np.testing.assert_array_equal(saved["label"], labels)
                    for part, values in (
                        ("train", trained["train_probabilities"]),
                        ("val", trained["validation_probabilities"]),
                        ("test", final["test_probabilities"]),
                    ):
                        prediction_error = max(
                            prediction_error,
                            float(np.max(np.abs(full[masks[part]] - values))),
                        )
                if (variant, group) == (chosen["variant"], chosen["group"]):
                    network = base.trainer.SingleMLP(features.shape[1], CONFIG)
                    network.load_state_dict(trained["state_dict"])
                    for part, saved in (
                        ("train", trained["train_probabilities"]),
                        ("val", trained["validation_probabilities"]),
                        ("test", final["test_probabilities"]),
                    ):
                        transformed = torch.as_tensor(
                            base.trainer.transform(
                                features[masks[part]], trained["mean"], trained["scale"]
                            )
                        )
                        recomputed = base.trainer.predict(network, transformed)
                        reload_error = max(
                            reload_error, float(np.max(np.abs(recomputed - saved)))
                        )
                checked += 1
    report = {
        "schema": "semantic-attention-logS-fusion-validation-811-v1",
        "checked_heads": checked,
        "expected_heads": len(MODELS) * len(HEADS) * len(SEEDS),
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
    parser.add_argument("--model", choices=(*MODELS, "all"), required=True)
    parser.add_argument(
        "--stage",
        choices=("prepare", "train", "select", "test", "summary", "validate"),
        required=True,
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.stage == "select":
        freeze_selection()
    elif args.stage == "summary":
        summarize_all()
    elif args.stage == "validate":
        validate_all()
    elif args.model == "all":
        parser.error("--model all is only valid for select/summary/validate")
    elif args.stage == "prepare":
        prepare(args.model)
    elif args.stage == "train":
        train_model(args.model, args.device)
    else:
        evaluate_model(args.model, args.device)


if __name__ == "__main__":
    main()
