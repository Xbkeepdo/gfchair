#!/usr/bin/env python3
"""Evaluate adjacent-layer JS features as hallucination detectors."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.jffn_experiment import atomic_json_save, atomic_torch_save, load_shards  # noqa: E402
from scripts.analyze_attention_evidence_js import (  # noqa: E402
    EXPERIMENT,
    LABELS,
    MODELS,
    MODEL_NAMES,
    label_by_target,
    load_attention_lookup,
    js_divergence,
    union_topk_js,
)
from scripts.train_old_risk_ev_s_mlp import (  # noqa: E402
    aggregate_seed_metrics,
    ensemble_metrics,
)
from scripts.train_torch_probe_feature_sets import (  # noqa: E402
    TorchProbeConfig,
    train_and_evaluate_probe,
)
from utils.config_utils import load_config  # noqa: E402
from utils.io_utils import load_json  # noqa: E402


TOP_K = 32
SEEDS = (43, 44, 45)
SCHEMA_VERSION = "attention-interlayer-js-detection-v1"
BLOCKS = (
    "adj_A_all",
    "adj_A_top32",
    "adj_E_all",
    "adj_E_top32",
)
FEATURE_SETS = {
    "adj_A_all": ("adj_A_all",),
    "adj_A_top32": ("adj_A_top32",),
    "adj_E_all": ("adj_E_all",),
    "adj_E_top32": ("adj_E_top32",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=(*MODELS, "summarize"))
    parser.add_argument("--outputs-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--experiment", default=EXPERIMENT)
    parser.add_argument(
        "--config", default=ROOT / "configs/model_configs_inslen_official_target.yaml"
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def block_features(
    attention: torch.Tensor, evidence: torch.Tensor, *, top_k: int
) -> dict[str, torch.Tensor]:
    """Build the four adjacent-layer JS blocks."""
    if attention.shape != evidence.shape or attention.ndim != 3:
        raise ValueError(
            f"Expected matching [targets,layers,tokens], got {attention.shape}/{evidence.shape}"
        )
    output = {}
    for slug, values in (("A", attention), ("E", evidence)):
        output[f"adj_{slug}_all"] = js_divergence(
            values[:, :-1, :], values[:, 1:, :]
        )
        output[f"adj_{slug}_top32"] = union_topk_js(
            values[:, :-1, :], values[:, 1:, :], top_k=top_k
        )
    return output


def feature_cache_path(output_dir: Path) -> Path:
    return output_dir / "token_feature_cache.pt"


def build_cache(
    *, model_root: Path, output_dir: Path, device: torch.device, top_k: int
) -> dict[str, Any]:
    cache_path = feature_cache_path(output_dir)
    attention_lookup, attention_audit = load_attention_lookup(model_root)
    shard_dir = model_root / "results/ffn_visual_source_attribution_v1/shards/full"
    values_by_block: dict[str, list[torch.Tensor]] = defaultdict(list)
    target_keys: list[str] = []
    labels_out: list[int] = []
    images_out: list[int] = []
    conflicts = 0
    missing_attention = 0
    shape_mismatches = 0
    seen_targets: set[str] = set()
    shard_count = len(list(shard_dir.glob("features*_shard_*.pt")))

    with torch.inference_mode():
        for shard_index, shard in enumerate(load_shards(shard_dir), 1):
            labels, shard_conflicts = label_by_target(shard["sample_table"])
            conflicts += shard_conflicts
            grouped: dict[tuple[int, int], list[tuple[str, int, int, torch.Tensor]]] = defaultdict(list)
            for position in shard["positions"]:
                key = str(position["target_key"])
                if key in seen_targets:
                    raise AssertionError(f"Duplicate target {key}")
                seen_targets.add(key)
                if key not in labels:
                    continue
                if key not in attention_lookup:
                    missing_attention += 1
                    continue
                attention = torch.from_numpy(attention_lookup[key])
                evidence = torch.as_tensor(position["attention_evidence"]).float()
                if attention.shape != evidence.shape:
                    shape_mismatches += 1
                    continue
                grouped[(int(attention.shape[0]), int(attention.shape[1]))].append(
                    (
                        key,
                        int(position["image_id"]),
                        int(labels[key]),
                        evidence,
                    )
                )
            for (_layers, _tokens), records in grouped.items():
                keys = [row[0] for row in records]
                attention = torch.stack(
                    [torch.from_numpy(attention_lookup[key]) for key in keys]
                ).to(device)
                evidence = torch.stack([row[3] for row in records]).to(device)
                blocks = block_features(attention, evidence, top_k=top_k)
                for name, values in blocks.items():
                    values_by_block[name].append(values.detach().cpu().float())
                target_keys.extend(keys)
                images_out.extend(row[1] for row in records)
                labels_out.extend(row[2] for row in records)
                del attention, evidence, blocks
            del shard
            gc.collect()
            if shard_index % 50 == 0 or shard_index == shard_count:
                print(
                    f"[interlayer detection] cache {model_root.parent.name}: "
                    f"{shard_index}/{shard_count} shards",
                    flush=True,
                )

    del attention_lookup
    gc.collect()
    blocks = {name: torch.cat(values, dim=0) for name, values in values_by_block.items()}
    row_count = len(target_keys)
    if set(blocks) != set(BLOCKS) or any(int(value.shape[0]) != row_count for value in blocks.values()):
        raise AssertionError("Incomplete or misaligned feature blocks")
    labels_tensor = torch.as_tensor(labels_out, dtype=torch.int8)
    images_tensor = torch.as_tensor(images_out, dtype=torch.int64)
    if set(labels_tensor.tolist()) != {0, 1}:
        raise AssertionError("Feature cache requires both labels")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "top_k": int(top_k),
        "target_keys": target_keys,
        "labels": labels_tensor,
        "image_ids": images_tensor,
        "blocks": blocks,
        "audit": {
            **attention_audit,
            "attribution_shards": shard_count,
            "seen_targets": len(seen_targets),
            "used_targets": row_count,
            "label_counts": {
                LABELS[label]: int((labels_tensor == label).sum()) for label in LABELS
            },
            "label_conflict_targets_excluded": conflicts,
            "missing_attention": missing_attention,
            "shape_mismatches": shape_mismatches,
            "block_dimensions": {
                name: int(value.shape[1]) for name, value in blocks.items()
            },
        },
    }
    atomic_torch_save(payload, cache_path)
    return payload


def load_or_build_cache(
    *,
    model_root: Path,
    output_dir: Path,
    device: torch.device,
    top_k: int,
    resume: bool,
) -> dict[str, Any]:
    path = feature_cache_path(output_dir)
    if resume and path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != SCHEMA_VERSION or int(payload["top_k"]) != int(top_k):
            raise ValueError(f"Incompatible feature cache: {path}")
        print(f"[interlayer detection] reuse {path}", flush=True)
        return payload
    return build_cache(
        model_root=model_root, output_dir=output_dir, device=device, top_k=top_k
    )


def feature_matrix(cache: Mapping[str, Any], feature_set: str) -> np.ndarray:
    arrays = [
        torch.as_tensor(cache["blocks"][name]).float().numpy()
        for name in FEATURE_SETS[feature_set]
    ]
    return np.concatenate(arrays, axis=1).astype(np.float32, copy=False)


def split_cache(cache: Mapping[str, Any], splits: Mapping[str, list[int]], feature_set: str):
    matrix = feature_matrix(cache, feature_set)
    labels = torch.as_tensor(cache["labels"]).numpy().astype(np.int32)
    image_ids = torch.as_tensor(cache["image_ids"]).numpy().astype(np.int64)
    train_mask = np.isin(image_ids, np.asarray(splits["train"], dtype=np.int64))
    test_mask = np.isin(image_ids, np.asarray(splits["test"], dtype=np.int64))
    if np.any(train_mask & test_mask) or not np.all(train_mask | test_mask):
        raise AssertionError("Feature rows do not map uniquely to the official split")
    return {
        "train": (matrix[train_mask], labels[train_mask], image_ids[train_mask]),
        "test": (matrix[test_mask], labels[test_mask], image_ids[test_mask]),
    }


def scalar_metrics(cache: Mapping[str, Any], splits: Mapping[str, list[int]]) -> list[dict[str, Any]]:
    labels = torch.as_tensor(cache["labels"]).numpy().astype(np.int32)
    image_ids = torch.as_tensor(cache["image_ids"]).numpy().astype(np.int64)
    test_mask = np.isin(image_ids, np.asarray(splits["test"], dtype=np.int64))
    hall = 1 - labels[test_mask]
    rows = []
    for block in BLOCKS:
        score = torch.as_tensor(cache["blocks"][block]).float().mean(dim=1).numpy()[test_mask]
        rows.append(
            {
                "block": block,
                "dimensions": int(torch.as_tensor(cache["blocks"][block]).shape[1]),
                "hall_auroc": float(roc_auc_score(hall, score)),
                "hall_aupr": float(average_precision_score(hall, score)),
            }
        )
    return rows


def probe_config(config: Mapping[str, Any], seed: int) -> TorchProbeConfig:
    values = config["training"]["torch_probe"]
    return TorchProbeConfig(
        hidden_sizes=tuple(int(value) for value in values["hidden_sizes"]),
        dropout=float(values["dropout"]),
        drop_last=bool(values["drop_last"]),
        batch_size=int(values["batch_size"]),
        num_epochs=int(values["max_epochs"]),
        learning_rate=float(values["learning_rate"]),
        weight_decay=float(values["weight_decay"]),
        lr_factor=float(values["lr_factor"]),
        lr_patience=int(values["lr_patience"]),
        early_stopping_patience=int(values["early_stopping_patience"]),
        seed=int(seed),
        positive_class="real",
        split_protocol="strict_82_no_validation",
        threshold_selection="train_f1",
        fixed_threshold=float(values["fixed_threshold"]),
        checkpoint_selection="minimum_train_loss",
    )


def train_model(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    model_root = args.outputs_root / args.model / args.experiment
    output_dir = args.outputs_root / "attention_js_studies/interlayer_js/detection" / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(str(args.config))
    splits = load_json(str(model_root / "image_splits.json"))
    if (len(splits["train"]), len(splits["test"]), len(splits.get("val", []))) != (3200, 800, 0):
        raise ValueError("Expected strict 3200/800 image split without validation")
    cache = load_or_build_cache(
        model_root=model_root,
        output_dir=output_dir,
        device=device,
        top_k=args.top_k,
        resume=args.resume,
    )
    progress_path = output_dir / "training_progress.pt"
    completed: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    probabilities: dict[str, dict[int, np.ndarray]] = defaultdict(dict)
    if args.resume and progress_path.exists():
        progress = torch.load(progress_path, map_location="cpu", weights_only=False)
        for feature_set, rows in progress.get("metrics", {}).items():
            completed[feature_set] = {int(seed): row for seed, row in rows.items()}
        for feature_set, rows in progress.get("probabilities", {}).items():
            probabilities[feature_set] = {
                int(seed): np.asarray(value, dtype=np.float32) for seed, value in rows.items()
            }

    split_data = {name: split_cache(cache, splits, name) for name in FEATURE_SETS}
    for feature_set, data in split_data.items():
        train_x, train_y, _train_images = data["train"]
        test_x, test_y, _test_images = data["test"]
        for seed in SEEDS:
            if seed in completed[feature_set] and seed in probabilities[feature_set]:
                print(f"[interlayer detection] reuse {args.model}/{feature_set}/seed{seed}", flush=True)
                continue
            metrics = train_and_evaluate_probe(
                X_train=train_x,
                y_train=train_y,
                X_val=np.empty((0, train_x.shape[1]), dtype=np.float32),
                y_val=np.empty((0,), dtype=np.int32),
                X_test=test_x,
                y_test=test_y,
                config=probe_config(config, seed),
                device=device,
                output_dir=str(output_dir / "training" / feature_set / f"seed{seed}"),
                return_probabilities=True,
            )
            test_probabilities = np.asarray(metrics.pop("test_probabilities"), dtype=np.float32)
            metrics.pop("train_probabilities", None)
            metrics["seed"] = int(seed)
            metrics["dimensions"] = int(train_x.shape[1])
            completed[feature_set][seed] = metrics
            probabilities[feature_set][seed] = test_probabilities
            atomic_torch_save(
                {
                    "schema_version": SCHEMA_VERSION,
                    "model": args.model,
                    "metrics": dict(completed),
                    "probabilities": dict(probabilities),
                },
                progress_path,
            )
            print(
                f"[interlayer detection] {args.model}/{feature_set}/seed{seed}: "
                f"AUROC={metrics['auc']:.6f}, "
                f"HallAUPR={metrics['hallucination_positive']['aupr']:.6f}",
                flush=True,
            )

    feature_results = {}
    for feature_set, data in split_data.items():
        test_y = data["test"][1]
        seed_rows = [completed[feature_set][seed] for seed in SEEDS]
        ensemble_probability = np.mean(
            [probabilities[feature_set][seed] for seed in SEEDS], axis=0
        )
        feature_results[feature_set] = {
            "blocks": list(FEATURE_SETS[feature_set]),
            "dimensions": int(data["train"][0].shape[1]),
            "seed_metrics": seed_rows,
            "aggregate": aggregate_seed_metrics(seed_rows),
            "ensemble": ensemble_metrics(test_y, ensemble_probability),
        }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "sample_audit": {
            **cache["audit"],
            "train_rows": int(split_data["adj_A_all"]["train"][0].shape[0]),
            "test_rows": int(split_data["adj_A_all"]["test"][0].shape[0]),
            "train_images": 3200,
            "test_images": 800,
            "test_label_counts": {
                LABELS[label]: int(
                    (split_data["adj_A_all"]["test"][1] == label).sum()
                )
                for label in LABELS
            },
            "test_hall_prior": float(
                (split_data["adj_A_all"]["test"][1] == 0).mean()
            ),
        },
        "training_protocol": {
            "classifier": "Linear-BatchNorm-ReLU-Dropout MLP",
            "hidden_sizes": [128, 64, 32],
            "seeds": list(SEEDS),
            "feature_normalization": "none",
            "checkpoint_selection": "minimum_train_loss",
            "threshold_selection": "train_f1",
            "positive_class": "real; Hall metrics use complementary score",
            "split": "existing strict image-level 3200/800",
        },
        "scalar_mean_js": scalar_metrics(cache, splits),
        "feature_sets": feature_results,
    }
    atomic_json_save(payload, output_dir / "results.json")
    write_model_outputs(output_dir, payload)
    return payload


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_model_outputs(output_dir: Path, payload: Mapping[str, Any]) -> None:
    metric_rows = []
    for feature_set, result in payload["feature_sets"].items():
        metric_rows.append(
            {
                "model": payload["model"],
                "feature_set": feature_set,
                "dimensions": result["dimensions"],
                "seed_mean_auroc": result["aggregate"]["auroc"]["mean"],
                "seed_std_auroc": result["aggregate"]["auroc"]["std"],
                "ensemble_auroc": result["ensemble"]["auroc"],
                "seed_mean_hall_aupr": result["aggregate"]["hall_aupr"]["mean"],
                "ensemble_hall_aupr": result["ensemble"]["hall_aupr"],
                "seed_mean_hall_f1": result["aggregate"]["hall_f1"]["mean"],
            }
        )
    write_csv(output_dir / "metrics.csv", metric_rows)
    write_csv(output_dir / "scalar_mean_js_metrics.csv", list(payload["scalar_mean_js"]))
    best = max(metric_rows, key=lambda row: float(row["ensemble_auroc"]))
    lines = [
        f"# {MODEL_NAMES[payload['model']]} 层间 JS 幻觉检测",
        "",
        "严格图片级 3200/800 split；唯一无冲突 target token；三层 MLP [128,64,32]；seeds 43/44/45。主排序看 seed-ensemble AUROC。",
        "",
        f"最佳层间 JS 特征为 `{best['feature_set']}`：AUROC `{float(best['ensemble_auroc']):.4f}`，Hall AUPR `{float(best['ensemble_hall_aupr']):.4f}`。",
        "",
        "| Feature set | Dim | AUROC mean±std | Ensemble AUROC | Ensemble Hall AUPR | Hall F1 mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(metric_rows, key=lambda item: float(item["ensemble_auroc"]), reverse=True):
        lines.append(
            f"| `{row['feature_set']}` | {row['dimensions']} | "
            f"{float(row['seed_mean_auroc']):.4f}±{float(row['seed_std_auroc']):.4f} | "
            f"{float(row['ensemble_auroc']):.4f} | {float(row['ensemble_hall_aupr']):.4f} | "
            f"{float(row['seed_mean_hall_f1']):.4f} |"
        )
    lines.extend(
        [
            "",
            "四组特征均独立训练，只使用相邻层 JS；A/E 为 plain attention/evidence。没有融合，且全层两两 JS 不进入检测器。",
            "",
            "## 不训练：相邻层 JS 均值",
            "",
            "这里直接把每个 token 的全部相邻层 JS 取均值作为 hall score；用于判断信号来自总体幅度还是层位模式。",
            "",
            "| Block | Dim | Hall AUROC | Hall AUPR |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in payload["scalar_mean_js"]:
        lines.append(
            f"| `{row['block']}` | {row['dimensions']} | "
            f"{float(row['hall_auroc']):.4f} | {float(row['hall_aupr']):.4f} |"
        )
    lines.extend(
        [
            "",
            f"测试集 HALL 比例为 `{float(payload['sample_audit']['test_hall_prior']):.4f}`；"
            "因此 Hall AUPR 应同时与该先验比较。均值分数明显弱于层向量 MLP，说明主要信息来自哪几个层转换发生变化，而非总体 JS 单调升高。",
            "",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def summarize(outputs_root: Path) -> None:
    root = outputs_root / "attention_js_studies/interlayer_js/detection"
    payloads = {
        model: json.loads((root / model / "results.json").read_text()) for model in MODELS
    }
    rows = []
    lines = [
        "# 四模型层间 JS 幻觉检测汇总",
        "",
        "严格图片级 3200/800 split；唯一无冲突 target token；三层 MLP [128,64,32]；seeds 43/44/45；无特征归一化。",
        "",
        "| Model | Best feature | Dim | AUROC mean±std | Ensemble AUROC | Hall AUPR | Hall prior | Hall F1 mean |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        best_name, best = max(
            payloads[model]["feature_sets"].items(),
            key=lambda item: float(item[1]["ensemble"]["auroc"]),
        )
        row = {
            "model": model,
            "best_feature_set": best_name,
            "dimensions": best["dimensions"],
            "seed_mean_auroc": best["aggregate"]["auroc"]["mean"],
            "seed_std_auroc": best["aggregate"]["auroc"]["std"],
            "ensemble_auroc": best["ensemble"]["auroc"],
            "ensemble_hall_aupr": best["ensemble"]["hall_aupr"],
            "test_hall_prior": payloads[model]["sample_audit"]["test_hall_prior"],
            "seed_mean_hall_f1": best["aggregate"]["hall_f1"]["mean"],
        }
        rows.append(row)
        lines.append(
            f"| {MODEL_NAMES[model]} | `{best_name}` | {row['dimensions']} | "
            f"{float(row['seed_mean_auroc']):.4f}±{float(row['seed_std_auroc']):.4f} | "
            f"{float(row['ensemble_auroc']):.4f} | {float(row['ensemble_hall_aupr']):.4f} | "
            f"{float(row['test_hall_prior']):.4f} | "
            f"{float(row['seed_mean_hall_f1']):.4f} |"
        )
    lines.extend(
        [
            "",
            "四组 A/E × all/Top32 特征完全独立训练；没有特征融合。单块通常是 all-token 优于 Top32。",
            "",
            "该实验只评估层间 JS standalone detector；没有把 JS 与现有 risk+EV 拼接，因此不能据此判断增量价值。",
        ]
    )
    write_csv(root / "summary.csv", rows)
    atomic_json_save(
        {"schema_version": SCHEMA_VERSION, "models": payloads, "best_by_model": rows},
        root / "summary.json",
    )
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.model == "summarize":
        summarize(args.outputs_root)
    else:
        train_model(args)


if __name__ == "__main__":
    main()
