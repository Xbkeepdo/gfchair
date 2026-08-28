#!/usr/bin/env python3
"""Train union-Top32 JS(P_JFFN,Q)+EV probes on completed JFFN cohorts."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from features.jffn_experiment import (  # noqa: E402
    GATES,
    atomic_json_save,
    atomic_torch_save,
    load_shards,
)
from scripts.train_old_risk_ev_s_mlp import (  # noqa: E402
    EXPERIMENT,
    MODELS,
    aggregate_seed_metrics,
    ensemble_metrics,
    paired_image_bootstrap,
)
from scripts.train_torch_probe_feature_sets import (  # noqa: E402
    TorchProbeConfig,
    train_and_evaluate_probe,
)
from utils.config_utils import load_config  # noqa: E402
from utils.io_utils import load_json, load_pkl  # noqa: E402


EPS = 1.0e-12
SIDE_TOP_K = 32
SCHEMA_VERSION = "jffn-union-topk-js-ev-mlp-v1"
SOURCES = ("old_hpre_cos", "new_jffn")
OUTPUT_SUBDIR = "results/jffn_second_round/jffn_union_topk_js_ev_mlp"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=(*MODELS, "summarize"))
    parser.add_argument(
        "--config", default="configs/model_configs_inslen_official_target.yaml"
    )
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--training-device", default="auto")
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _normalize_rows(values: Any) -> np.ndarray:
    array = np.nan_to_num(
        np.asarray(torch.as_tensor(values).float().cpu(), dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    array = np.maximum(array, 0.0)
    totals = array.sum(axis=-1, keepdims=True)
    uniform = np.full_like(array, 1.0 / float(array.shape[-1]))
    return np.divide(array, np.maximum(totals, EPS), out=uniform, where=totals > 0.0)


def union_topk_js(
    source: Any, target: Any, *, side_top_k: int = SIDE_TOP_K
) -> np.ndarray:
    """Natural-log JS after Top-K-per-side union and within-union renormalization."""
    source_prob = _normalize_rows(source)
    target_prob = _normalize_rows(target)
    if source_prob.ndim != 2 or source_prob.shape != target_prob.shape:
        raise ValueError(
            f"Expected matching [layers,tokens], got {source_prob.shape}/{target_prob.shape}"
        )
    k = min(max(int(side_top_k), 1), int(source_prob.shape[-1]))
    source_indices = np.argsort(-source_prob, axis=-1, kind="stable")[:, :k]
    target_indices = np.argsort(-target_prob, axis=-1, kind="stable")[:, :k]
    layer_indices = np.arange(source_prob.shape[0])[:, None]
    mask = np.zeros(source_prob.shape, dtype=bool)
    mask[layer_indices, source_indices] = True
    mask[layer_indices, target_indices] = True

    def masked(values: np.ndarray) -> np.ndarray:
        region = np.where(mask, values, 0.0)
        totals = region.sum(axis=-1, keepdims=True)
        if np.any(totals <= 0.0):
            raise ValueError("Union support has zero probability mass")
        region = region / totals
        region = np.where(mask, np.maximum(region, EPS), 0.0)
        return region / region.sum(axis=-1, keepdims=True)

    p = masked(source_prob)
    q = masked(target_prob)
    midpoint = 0.5 * (p + q)
    p_safe = np.maximum(p, EPS)
    q_safe = np.maximum(q, EPS)
    midpoint_safe = np.maximum(midpoint, EPS)
    js = 0.5 * np.sum(
        np.where(mask, p * (np.log(p_safe) - np.log(midpoint_safe)), 0.0), axis=-1
    )
    js += 0.5 * np.sum(
        np.where(mask, q * (np.log(q_safe) - np.log(midpoint_safe)), 0.0), axis=-1
    )
    return js.astype(np.float32)


def _spec(source: str, gate: str) -> str:
    return f"{source}__{gate}__union_top32_js_ev"


def _required_position_keys(shard_dir: Path) -> tuple[set[tuple[int, int]], bool]:
    required: set[tuple[int, int]] = set()
    targets_complete = True
    for shard in load_shards(shard_dir):
        for position in shard["positions"]:
            required.add((int(position["image_id"]), int(position["response_index"])))
            targets_complete &= bool(
                isinstance(position.get("targets"), Mapping)
                and all(gate in position["targets"] for gate in GATES)
            )
    return required, targets_complete


def _load_reconstructed_targets(
    model_root: Path, required: set[tuple[int, int]]
) -> tuple[dict[tuple[int, int], dict[str, np.ndarray]], dict[str, Any]]:
    """Reconstruct Q=normalize(attention*gate) from existing feature parts."""
    lookup: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    duplicate_rows = 0
    maximum_duplicate_error = 0.0
    for part_path in sorted(model_root.glob("features.part*.pkl")):
        rows = load_pkl(str(part_path))
        for row in rows:
            key = (int(row["image_id"]), int(row["response_token_idx"]))
            if key not in required:
                continue
            attention = np.asarray(
                row["dgst_t_attention_support_per_layer"], dtype=np.float32
            )
            item: dict[str, np.ndarray] = {}
            for gate in GATES:
                gate_values = np.asarray(
                    row[f"dgst_t_{gate}_gate_per_layer"], dtype=np.float32
                )
                item[gate] = _normalize_rows(attention * gate_values).astype(np.float32)
                item[f"ev_{gate}"] = np.asarray(
                    row[
                        f"dgst_t_{gate}_ev_target_dist_mass_x_cosine_"
                        "topk32_hpre_per_layer"
                    ],
                    dtype=np.float32,
                ).reshape(-1)
            if key in lookup:
                duplicate_rows += 1
                for name, value in item.items():
                    error = float(np.max(np.abs(lookup[key][name] - value)))
                    maximum_duplicate_error = max(maximum_duplicate_error, error)
                    if error > 1.0e-6:
                        raise AssertionError(f"Conflicting duplicate target {key}/{name}")
            else:
                lookup[key] = item
        del rows
        gc.collect()
    missing = required - set(lookup)
    if missing:
        raise AssertionError(f"Target reconstruction misses {len(missing)} positions")
    return lookup, {
        "origin": "reconstructed_from_existing_feature_parts",
        "required_positions": len(required),
        "resolved_positions": len(lookup),
        "duplicate_feature_rows": duplicate_rows,
        "maximum_duplicate_error": maximum_duplicate_error,
        "formula": "normalize(dgst_t_attention_support_per_layer * gate_per_layer)",
    }


def collect_matrices(
    model_root: Path,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    splits = load_json(str(model_root / "image_splits.json"))
    train_ids = {int(value) for value in splits["train"]}
    test_ids = {int(value) for value in splits["test"]}
    if train_ids & test_ids:
        raise AssertionError("Image split overlaps")
    shard_dir = model_root / "results/jffn_p_comparison/shards"
    required, shard_targets_complete = _required_position_keys(shard_dir)
    reconstructed: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    if shard_targets_complete:
        target_audit = {
            "origin": "saved_jffn_shard_targets",
            "required_positions": len(required),
            "resolved_positions": len(required),
            "formula": "saved normalize(attention * gate)",
        }
    else:
        reconstructed, target_audit = _load_reconstructed_targets(model_root, required)

    specs = [_spec(source, gate) for source in SOURCES for gate in GATES]
    stores: dict[str, dict[str, Any]] = {
        split: {
            "x": {spec: [] for spec in specs},
            "js": {_spec("new_jffn", gate): [] for gate in GATES},
            "y": [],
            "image_ids": [],
            "mention_ids": [],
        }
        for split in ("train", "test")
    }
    target_ev_max_error = 0.0
    for shard in load_shards(shard_dir):
        positions = {str(row["target_key"]): row for row in shard["positions"]}
        position_features: dict[str, dict[str, np.ndarray]] = {}
        for target_key, position in positions.items():
            key = (int(position["image_id"]), int(position["response_index"]))
            if shard_targets_complete:
                targets = {
                    gate: _normalize_rows(position["targets"][gate]).astype(np.float32)
                    for gate in GATES
                }
            else:
                targets = {gate: reconstructed[key][gate] for gate in GATES}
                for gate in GATES:
                    expected_ev = reconstructed[key][f"ev_{gate}"]
                    actual_ev = np.asarray(position["ev"][gate], dtype=np.float32)
                    target_ev_max_error = max(
                        target_ev_max_error,
                        float(np.max(np.abs(expected_ev - actual_ev))),
                    )
            item = {}
            for gate in GATES:
                ev = np.asarray(position["ev"][gate], dtype=np.float32).reshape(-1)
                for source in SOURCES:
                    js = union_topk_js(position["sources"][source], targets[gate])
                    if js.shape != ev.shape:
                        raise AssertionError(f"JS/EV layer mismatch for {target_key}")
                    item[_spec(source, gate)] = np.concatenate((js, ev)).astype(
                        np.float32, copy=False
                    )
                    if source == "new_jffn":
                        item[f"js::{_spec(source, gate)}"] = js
            position_features[target_key] = item
        for mention in shard["sample_table"]:
            image_id = int(mention["image_id"])
            split = "train" if image_id in train_ids else "test" if image_id in test_ids else None
            if split is None:
                raise AssertionError(f"Image {image_id} is outside official split")
            item = position_features[str(mention["target_key"])]
            store = stores[split]
            for spec in specs:
                row = item[spec]
                if not np.isfinite(row).all():
                    raise ValueError(f"Non-finite {spec}/{mention['mention_id']}")
                store["x"][spec].append(row)
            for gate in GATES:
                spec = _spec("new_jffn", gate)
                store["js"][spec].append(item[f"js::{spec}"])
            store["y"].append(int(mention["label"]))
            store["image_ids"].append(image_id)
            store["mention_ids"].append(str(mention["mention_id"]))

    if not shard_targets_complete and target_ev_max_error > 1.0e-6:
        raise AssertionError(
            f"Reconstructed target alignment failed: EV error {target_ev_max_error}"
        )
    matrices: dict[str, dict[str, Any]] = {}
    for split, store in stores.items():
        matrices[split] = {
            "x": {spec: np.stack(store["x"][spec]) for spec in specs},
            "js": {spec: np.stack(store["js"][spec]) for spec in store["js"]},
            "y": np.asarray(store["y"], dtype=np.int32),
            "image_ids": np.asarray(store["image_ids"], dtype=np.int64),
            "mention_ids": list(store["mention_ids"]),
        }
        if set(np.unique(matrices[split]["y"])) != {0, 1}:
            raise AssertionError(f"{split} lacks a binary label")
    del reconstructed
    gc.collect()

    curve_rows = _curve_rows(matrices)
    audit = {
        "train_mentions": int(len(matrices["train"]["y"])),
        "test_mentions": int(len(matrices["test"]["y"])),
        "train_images": int(np.unique(matrices["train"]["image_ids"]).size),
        "test_images": int(np.unique(matrices["test"]["image_ids"]).size),
        "layers": int(next(iter(matrices["train"]["js"].values())).shape[1]),
        "input_dimensions": int(next(iter(matrices["train"]["x"].values())).shape[1]),
        "source_top_k": SIDE_TOP_K,
        "target_top_k": SIDE_TOP_K,
        "union_max_size": 2 * SIDE_TOP_K,
        "within_union_renormalization": True,
        "js_log_base": "natural",
        "feature_normalization": "none",
        "target_ev_max_absolute_error": target_ev_max_error,
        "target_audit": target_audit,
        "all_finite": True,
    }
    return matrices, curve_rows, audit


def _curve_rows(matrices: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    labels = np.concatenate((matrices["train"]["y"], matrices["test"]["y"]))
    rows: list[dict[str, Any]] = []
    for gate in GATES:
        spec = _spec("new_jffn", gate)
        values = np.concatenate(
            (matrices["train"]["js"][spec], matrices["test"]["js"][spec])
        ).astype(np.float64)
        for label, name in ((1, "REAL"), (0, "HALL")):
            selected = values[labels == label]
            mean = selected.mean(axis=0)
            std = selected.std(axis=0, ddof=1)
            sem = std / np.sqrt(selected.shape[0])
            for layer in range(values.shape[1]):
                rows.append(
                    {
                        "gate": gate,
                        "label": name,
                        "layer": layer + 1,
                        "n": int(selected.shape[0]),
                        "mean": float(mean[layer]),
                        "std": float(std[layer]),
                        "sem": float(sem[layer]),
                        "ci95_low": float(mean[layer] - 1.96 * sem[layer]),
                        "ci95_high": float(mean[layer] + 1.96 * sem[layer]),
                    }
                )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot_curves(model: str, rows: Sequence[Mapping[str, Any]], output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.0), sharex=True)
    colors = {"REAL": "#2563eb", "HALL": "#dc2626"}
    for axis, gate in zip(axes, GATES):
        for label in ("REAL", "HALL"):
            selected = [row for row in rows if row["gate"] == gate and row["label"] == label]
            x = np.asarray([int(row["layer"]) for row in selected])
            mean = np.asarray([float(row["mean"]) for row in selected])
            low = np.asarray([float(row["ci95_low"]) for row in selected])
            high = np.asarray([float(row["ci95_high"]) for row in selected])
            axis.plot(x, mean, color=colors[label], label=label, linewidth=2.0)
            axis.fill_between(x, low, high, color=colors[label], alpha=0.18)
        axis.set_title(f"P_JFFN vs {gate} target: union-Top32 JS")
        axis.set_ylabel("Natural-log JS")
        axis.grid(alpha=0.25)
        axis.legend()
    axes[-1].set_xlabel("Decoder layer")
    fig.suptitle(f"{model}: REAL vs HALL JFFN source-target divergence")
    fig.tight_layout()
    stem = output_dir / f"{model}_jffn_union_topk_js_real_hall_curves"
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _align_jffn_predictions(
    model_root: Path, spec: str, test: Mapping[str, Any]
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    result_dir = model_root / "results/jffn_p_comparison"
    results = json.loads((result_dir / "training_results.json").read_text())
    aggregate = results["feature_sets"][spec]
    progress = torch.load(
        result_dir / "training_progress.pt", map_location="cpu", weights_only=False
    )
    expected = list(test["mention_ids"])
    predictions: dict[int, np.ndarray] = {}
    for seed in (43, 44, 45):
        row = progress["predictions"][spec][seed]
        lookup = {str(value): index for index, value in enumerate(row["mention_ids"])}
        indices = np.asarray([lookup[value] for value in expected], dtype=np.int64)
        if not np.array_equal(np.asarray(row["labels"])[indices], test["y"]):
            raise AssertionError(f"JFFN reference label mismatch: {spec}/{seed}")
        if not np.array_equal(np.asarray(row["image_ids"])[indices], test["image_ids"]):
            raise AssertionError(f"JFFN reference image mismatch: {spec}/{seed}")
        predictions[seed] = np.asarray(row["probabilities"], dtype=np.float32)[indices]
    return predictions, aggregate


def _curve_summary(rows: Sequence[Mapping[str, Any]], gate: str) -> dict[str, Any]:
    real = {int(row["layer"]): float(row["mean"]) for row in rows if row["gate"] == gate and row["label"] == "REAL"}
    hall = {int(row["layer"]): float(row["mean"]) for row in rows if row["gate"] == gate and row["label"] == "HALL"}
    differences = {layer: hall[layer] - real[layer] for layer in real}
    peak = max(differences, key=lambda layer: abs(differences[layer]))
    return {
        "hall_higher_layers": sum(value > 0 for value in differences.values()),
        "real_higher_layers": sum(value < 0 for value in differences.values()),
        "full_layer_mean_real": float(np.mean(list(real.values()))),
        "full_layer_mean_hall": float(np.mean(list(hall.values()))),
        "peak_absolute_difference_layer": int(peak),
        "peak_hall_minus_real": float(differences[peak]),
    }


def _report(model: str, payload: Mapping[str, Any]) -> str:
    lines = [
        f"# {model}: P_JFFN-target Union-Top32 JS + EV",
        "",
        "每层分别取 source/target Top-32 的并集，在并集内各自重新归一化后计算自然对数 JS；"
        "32层 JS 与同 gate 的32层 mass×cosine EV 拼接为64维。",
        "",
        "## 三种子结果",
        "",
        "| Source | Target gate | Mean AUROC | Mean Hall F1 | Mean Hall AUPR | Ensemble AUROC |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for source in SOURCES:
        for gate in GATES:
            result = payload["feature_sets"][_spec(source, gate)]
            agg = result["aggregate"]
            lines.append(
                f"| {source} | {gate} | {agg['auroc']['mean']:.6f} ± "
                f"{agg['auroc']['std']:.6f} | {agg['hall_f1']['mean']:.6f} | "
                f"{agg['hall_aupr']['mean']:.6f} | {result['seed_ensemble']['auroc']:.6f} |"
            )
    lines += ["", "## 配对对照", ""]
    for gate in GATES:
        for comparison, display in (
            ("versus_old_hpre_js_ev", "P_JFFN JS+EV − old-hpre-P JS+EV"),
            ("versus_jffn_risk_ev", "P_JFFN JS+EV − P_JFFN OT-risk+EV"),
        ):
            boot = payload["paired_bootstrap"][gate][comparison]
            lines.append(
                f"- {gate}，{display}：ensemble AUROC Δ="
                f"{boot['new_minus_baseline_auroc']:+.6f}，95% CI "
                f"[{boot['auroc_ci95'][0]:+.6f},{boot['auroc_ci95'][1]:+.6f}]；"
                f"Hall-AUPR Δ={boot['new_minus_baseline_hall_aupr']:+.6f}，95% CI "
                f"[{boot['hall_aupr_ci95'][0]:+.6f},{boot['hall_aupr_ci95'][1]:+.6f}]。"
            )
    lines += ["", "## JS 曲线摘要", ""]
    for gate in GATES:
        row = payload["curve_summary"][gate]
        lines.append(
            f"- {gate}：Hall>Real {row['hall_higher_layers']}/32 层；全层均值 "
            f"Real={row['full_layer_mean_real']:.6f}、Hall={row['full_layer_mean_hall']:.6f}；"
            f"最大绝对差 L{row['peak_absolute_difference_layer']}，"
            f"Hall−Real={row['peak_hall_minus_real']:+.6f}。"
        )
    lines.append("")
    return "\n".join(lines)


def train_model(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    model_root = Path(args.outputs_root) / args.model / EXPERIMENT
    output_dir = model_root / OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    matrices, curve_rows, audit = collect_matrices(model_root)
    _write_csv(output_dir / "jffn_union_topk_js_label_curves.csv", curve_rows)
    _plot_curves(args.model, curve_rows, output_dir)
    train, test = matrices["train"], matrices["test"]

    section = config.get("jffn_p_comparison") or {}
    comparison_training = section.get("training") or {}
    inherited = (config.get("training") or {}).get("torch_probe") or {}
    seeds = [int(value) for value in comparison_training.get("seeds", [43, 44, 45])]
    specs = [_spec(source, gate) for source in SOURCES for gate in GATES]
    progress_path = output_dir / "training_progress.pt"
    completed: dict[str, dict[int, dict[str, Any]]] = {spec: {} for spec in specs}
    probabilities: dict[str, dict[int, np.ndarray]] = {spec: {} for spec in specs}
    if args.resume and progress_path.exists():
        progress = torch.load(progress_path, map_location="cpu", weights_only=False)
        if progress.get("schema_version") != SCHEMA_VERSION:
            raise AssertionError("Incompatible union-JS resume artifact")
        completed = {
            spec: {int(row["seed"]): row for row in rows}
            for spec, rows in progress.get("seed_results", {}).items()
        }
        probabilities = {
            spec: {int(seed): np.asarray(value, dtype=np.float32) for seed, value in rows.items()}
            for spec, rows in progress.get("probabilities", {}).items()
        }
        for spec in specs:
            completed.setdefault(spec, {})
            probabilities.setdefault(spec, {})

    device = _device(args.training_device)
    for spec in specs:
        for seed in seeds:
            if seed in completed[spec] and seed in probabilities[spec]:
                print(f"[JFFN JS+EV] reuse {args.model} {spec} seed={seed}", flush=True)
                continue
            probe_cfg = TorchProbeConfig(
                hidden_sizes=tuple(
                    int(value)
                    for value in comparison_training.get(
                        "hidden_sizes", inherited.get("hidden_sizes", [128, 64, 32])
                    )
                ),
                dropout=float(inherited.get("dropout", 0.3)),
                drop_last=bool(inherited.get("drop_last", True)),
                batch_size=int(comparison_training.get("batch_size", 256)),
                num_epochs=int(comparison_training.get("max_epochs", 100)),
                learning_rate=float(inherited.get("learning_rate", 1e-3)),
                weight_decay=float(inherited.get("weight_decay", 1e-5)),
                lr_factor=float(inherited.get("lr_factor", 0.5)),
                lr_patience=int(inherited.get("lr_patience", 5)),
                early_stopping_patience=int(inherited.get("early_stopping_patience", 10)),
                seed=seed,
                positive_class="real",
                split_protocol="strict_82_no_validation",
                threshold_selection="train_f1",
                fixed_threshold=float(inherited.get("fixed_threshold", 0.5)),
                checkpoint_selection="minimum_train_loss",
            )
            metrics = train_and_evaluate_probe(
                X_train=train["x"][spec], y_train=train["y"],
                X_val=np.empty((0, 64), dtype=np.float32),
                y_val=np.empty((0,), dtype=np.int32),
                X_test=test["x"][spec], y_test=test["y"],
                config=probe_cfg, device=device,
                output_dir=str(output_dir / "training" / spec / f"seed_{seed}"),
                return_probabilities=True,
            )
            test_prob = np.asarray(metrics.pop("test_probabilities"), dtype=np.float32)
            metrics.pop("train_probabilities", None)
            metrics["seed"] = seed
            metrics["num_features"] = 64
            completed[spec][seed] = metrics
            probabilities[spec][seed] = test_prob
            atomic_torch_save(
                {
                    "schema_version": SCHEMA_VERSION,
                    "model": args.model,
                    "seed_results": {
                        name: [completed[name][value] for value in sorted(completed[name])]
                        for name in specs
                    },
                    "probabilities": probabilities,
                    "labels": test["y"], "image_ids": test["image_ids"],
                    "mention_ids": test["mention_ids"],
                },
                progress_path,
            )
            print(
                f"[JFFN JS+EV] {args.model} {spec} seed={seed} "
                f"AUROC={metrics['auc']:.6f} HallAUPR="
                f"{metrics['hallucination_positive']['aupr']:.6f}", flush=True
            )

    results: dict[str, Any] = {}
    ensembles: dict[str, np.ndarray] = {}
    for spec in specs:
        rows = [completed[spec][seed] for seed in seeds]
        ensembles[spec] = np.mean([probabilities[spec][seed] for seed in seeds], axis=0)
        results[spec] = {
            "seeds": rows,
            "aggregate": aggregate_seed_metrics(rows),
            "seed_ensemble": ensemble_metrics(test["y"], ensembles[spec]),
        }

    reference_results: dict[str, Any] = {}
    reference_ensembles: dict[str, np.ndarray] = {}
    for gate in GATES:
        ref_spec = f"new_jffn__{gate}__sqrt_matched_state__risk_ev"
        ref_predictions, ref_result = _align_jffn_predictions(model_root, ref_spec, test)
        reference_ensembles[gate] = np.mean(
            [ref_predictions[seed] for seed in seeds], axis=0
        )
        reference_results[gate] = {
            "name": ref_spec,
            "aggregate": ref_result["aggregate"],
            "seed_ensemble": ensemble_metrics(test["y"], reference_ensembles[gate]),
        }

    paired: dict[str, Any] = {}
    for gate in GATES:
        new_spec = _spec("new_jffn", gate)
        old_spec = _spec("old_hpre_cos", gate)
        paired[gate] = {
            "versus_old_hpre_js_ev": paired_image_bootstrap(
                labels=test["y"], image_ids=test["image_ids"],
                new_probabilities=ensembles[new_spec], baseline_probabilities=ensembles[old_spec],
                replicates=args.bootstrap_replicates, seed=20260823,
            ),
            "versus_jffn_risk_ev": paired_image_bootstrap(
                labels=test["y"], image_ids=test["image_ids"],
                new_probabilities=ensembles[new_spec], baseline_probabilities=reference_ensembles[gate],
                replicates=args.bootstrap_replicates, seed=20260824,
            ),
        }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "feature_definition": {
            "source": "new_jffn P (with old_hpre_cos P as controlled source baseline)",
            "target_gates": list(GATES),
            "support": "union(source Top-32, target Top-32), max 64",
            "divergence": "natural-log Jensen-Shannon after within-union renormalization",
            "input": "[32-layer JS, corresponding 32-layer mass_x_cosine EV]",
        },
        "sample_audit": audit,
        "training_protocol": {
            "hidden_sizes": [128, 64, 32], "hidden_layer_count": 3,
            "batch_size": int(comparison_training.get("batch_size", 256)),
            "max_epochs": int(comparison_training.get("max_epochs", 100)),
            "seeds": seeds, "feature_normalization": "none",
            "checkpoint_selection": "minimum_train_loss",
            "threshold_selection": "train_f1", "split": "existing image-level 80/20",
        },
        "curve_summary": {gate: _curve_summary(curve_rows, gate) for gate in GATES},
        "feature_sets": results,
        "jffn_risk_ev_references": reference_results,
        "paired_bootstrap": paired,
    }
    atomic_json_save(payload, output_dir / "results.json")
    (output_dir / f"{args.model}_jffn_union_topk_js_ev_report.md").write_text(
        _report(args.model, payload), encoding="utf-8"
    )
    print(f"[JFFN JS+EV] wrote {output_dir}", flush=True)


def summarize(outputs_root: Path) -> None:
    payloads = {}
    csv_rows = []
    for model in MODELS:
        path = outputs_root / model / EXPERIMENT / OUTPUT_SUBDIR / "results.json"
        payload = json.loads(path.read_text())
        payloads[model] = payload
        for source in SOURCES:
            for gate in GATES:
                result = payload["feature_sets"][_spec(source, gate)]
                csv_rows.append(
                    {
                        "model": model, "source": source, "gate": gate,
                        "mean_auroc": result["aggregate"]["auroc"]["mean"],
                        "std_auroc": result["aggregate"]["auroc"]["std"],
                        "mean_hall_f1": result["aggregate"]["hall_f1"]["mean"],
                        "mean_hall_aupr": result["aggregate"]["hall_aupr"]["mean"],
                        "ensemble_auroc": result["seed_ensemble"]["auroc"],
                        "ensemble_hall_aupr": result["seed_ensemble"]["hall_aupr"],
                    }
                )
    atomic_json_save(
        {"schema_version": SCHEMA_VERSION, "models": payloads},
        outputs_root / "jffn_union_topk_js_ev_2model_summary.json",
    )
    _write_csv(outputs_root / "jffn_union_topk_js_ev_2model_metrics.csv", csv_rows)
    lines = [
        "# P_JFFN-target Union-Top32 JS + EV：两模型汇总", "",
        "| Model | Source P | Target gate | Mean AUROC | Hall F1 | Hall AUPR | Ensemble AUROC |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in csv_rows:
        lines.append(
            f"| {row['model']} | {row['source']} | {row['gate']} | "
            f"{row['mean_auroc']:.6f} ± {row['std_auroc']:.6f} | "
            f"{row['mean_hall_f1']:.6f} | {row['mean_hall_aupr']:.6f} | "
            f"{row['ensemble_auroc']:.6f} |"
        )
    lines += ["", "## Paired image bootstrap", ""]
    for model, payload in payloads.items():
        for gate in GATES:
            for key, display in (
                ("versus_old_hpre_js_ev", "vs old-hpre-P JS+EV"),
                ("versus_jffn_risk_ev", "vs P_JFFN OT-risk+EV"),
            ):
                boot = payload["paired_bootstrap"][gate][key]
                lines.append(
                    f"- {model} / {gate} {display}：AUROC Δ="
                    f"{boot['new_minus_baseline_auroc']:+.6f}，95% CI "
                    f"[{boot['auroc_ci95'][0]:+.6f},{boot['auroc_ci95'][1]:+.6f}]；"
                    f"Hall-AUPR Δ={boot['new_minus_baseline_hall_aupr']:+.6f}，95% CI "
                    f"[{boot['hall_aupr_ci95'][0]:+.6f},{boot['hall_aupr_ci95'][1]:+.6f}]。"
                )
    (outputs_root / "jffn_union_topk_js_ev_2model_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("[JFFN JS+EV] wrote two-model summary", flush=True)


def main() -> None:
    args = parse_args()
    if args.model == "summarize":
        summarize(Path(args.outputs_root))
    else:
        train_model(args)


if __name__ == "__main__":
    main()
