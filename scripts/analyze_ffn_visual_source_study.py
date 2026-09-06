#!/usr/bin/env python3
"""Freeze K, train the 13 detector blocks, bootstrap, plot, and apply the gate."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.tc_fvpa_artifacts import (  # noqa: E402
    ExperimentLayout,
    atomic_json_save,
    atomic_torch_save,
    initialize_manifests,
    update_stage_status,
)
from scripts.run_ffn_visual_source_attribution import (  # noqa: E402
    EXPERIMENT,
    result_root,
)
from scripts.tc_fvpa_common import FORMAL_LAYERS_BY_MODEL  # noqa: E402
from scripts.train_old_risk_ev_s_mlp import paired_image_bootstrap  # noqa: E402
from scripts.train_torch_probe_feature_sets import (  # noqa: E402
    TorchProbeConfig,
    train_and_evaluate_probe,
)
from utils.io_utils import load_json  # noqa: E402


PRIMARY_MODELS = ("qwen2_5_vl_7b", "llava_1_5_7b")
EXTENSION_MODELS = ("qwen3_vl_8b", "internvl_2_5_8b")
SEEDS = (43, 44, 45)
DISTANCES = ("JS", "OT")
PAIR_NAMES = ("D_EW", "D_WF", "D_EF")
SHARED_SPECS = ("A", "B", "F")
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", default=",".join(PRIMARY_MODELS), help="comma-separated model names"
    )
    parser.add_argument(
        "--stage", choices=("freeze", "detectors", "all"), default="all"
    )
    parser.add_argument("--training-device", default="cuda:0")
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def _audit_paths(model: str, smoke: bool) -> list[Path]:
    name = "audit_smoke" if smoke else "audit"
    return sorted((result_root(model) / f"shards/{name}").glob("audit_rank*_image_*.pt"))


def _js(left: np.ndarray, right: np.ndarray) -> float:
    p = np.maximum(np.asarray(left, dtype=np.float64), 0.0)
    q = np.maximum(np.asarray(right, dtype=np.float64), 0.0)
    p = np.maximum(p, EPS); p /= p.sum()
    q = np.maximum(q, EPS); q /= q.sum()
    midpoint = 0.5 * (p + q)
    return float(
        0.5 * np.sum(p * np.log(p / midpoint))
        + 0.5 * np.sum(q * np.log(q / midpoint))
    )


def _top_overlap(left: np.ndarray, right: np.ndarray, k: int = 32) -> float:
    count = min(int(k), int(left.size))
    first = set(np.argsort(left, kind="stable")[-count:].tolist())
    second = set(np.argsort(right, kind="stable")[-count:].tolist())
    return len(first & second) / float(count)


def summarize_k_convergence(models: Sequence[str], *, smoke: bool) -> dict[str, Any]:
    records: dict[tuple[str, int, int], list[dict[str, float]]] = {}
    benchmark_rows = []
    control_rows = []
    scaling_rows = []
    shapley_rows = []
    processed = {}
    detailed = {}
    for model in models:
        paths = _audit_paths(model, smoke)
        if not paths:
            raise FileNotFoundError(f"No audit shards for {model}")
        image_ids = set()
        detailed_count = 0
        for path in paths:
            shard = torch.load(path, map_location="cpu", weights_only=False)
            image_ids.add(int(shard["image_id"]))
            if shard.get("synthetic_linearize_benchmark"):
                benchmark_rows.append(
                    {
                        "model": model,
                        "kind": "synthetic",
                        "metrics": shard["synthetic_linearize_benchmark"],
                    }
                )
            for case in shard["cases"]:
                reference = case["k_sweep"]["64"]
                ref_p = reference["p_ffn"].numpy()
                for raw_k, candidate in case["k_sweep"].items():
                    k = int(raw_k)
                    if k == 64:
                        continue
                    p = candidate["p_ffn"].numpy()
                    correlation = float(spearmanr(p, ref_p).statistic)
                    if not np.isfinite(correlation):
                        correlation = 1.0 if np.array_equal(p, ref_p) else 0.0
                    records.setdefault((model, int(case["layer"]), k), []).append(
                        {
                            "spearman": correlation,
                            "js": _js(p, ref_p),
                            "top32_overlap": _top_overlap(p, ref_p),
                            "gross_relative_error": abs(
                                float(candidate["gross_strength"])
                                - float(reference["gross_strength"])
                            ) / max(abs(float(reference["gross_strength"])), EPS),
                            "kappa_difference": abs(
                                float(candidate["kappa"])
                                - float(reference["kappa"])
                            ),
                        }
                    )
                fp32 = case.get("fp32_audit")
                if fp32:
                    detailed_count += 1
                    native = reference
                    gauss = fp32["gauss_legendre_k64"]
                    for control, candidate in (
                        ("native_vs_fp32_gauss_k64", gauss),
                        ("fp32_trapezoid_vs_gauss_k64", fp32["trapezoid_k64"]),
                    ):
                        left = candidate["p_ffn"].numpy()
                        right = (native if control.startswith("native") else gauss)[
                            "p_ffn"
                        ].numpy()
                        correlation = float(spearmanr(left, right).statistic)
                        if not np.isfinite(correlation):
                            correlation = 1.0 if np.array_equal(left, right) else 0.0
                        denominator = float(
                            (native if control.startswith("native") else gauss)[
                                "gross_strength"
                            ]
                        )
                        control_rows.append(
                            {
                                "model": model,
                                "case_id": case["case_id"],
                                "layer": int(case["layer"]),
                                "control": control,
                                "spearman": correlation,
                                "js": _js(left, right),
                                "top32_overlap": _top_overlap(left, right),
                                "gross_relative_error": abs(
                                    float(candidate["gross_strength"]) - denominator
                                ) / max(abs(denominator), EPS),
                                "kappa_difference": abs(
                                    float(candidate["kappa"])
                                    - float(
                                        (native if control.startswith("native") else gauss)[
                                            "kappa"
                                        ]
                                    )
                                ),
                                "candidate_completeness_relative_error": float(
                                    candidate["completeness_relative_error"]
                                ),
                            }
                        )
                    for strategy, values in fp32["source_scaling"].items():
                        for value in values:
                            scaling_rows.append(
                                {
                                    "model": model,
                                    "case_id": case["case_id"],
                                    "layer": int(case["layer"]),
                                    "strategy": strategy,
                                    "lambda": float(value["lambda"]),
                                    "finite_effect_norm": float(
                                        value["finite_effect_norm"]
                                    ),
                                }
                            )
                    if set(fp32["shapley"]) != {"8", "16"}:
                        raise AssertionError("FP32 audit must contain 8/16-region Shapley")
                    expected_permutations = 2 if smoke else 128
                    if any(
                        int(
                            value.get(
                                "permutation_count",
                                expected_permutations,
                            )
                        )
                        != expected_permutations
                        or len(value["running_estimates"])
                        != int(np.ceil(expected_permutations / 16))
                        or not np.isfinite(
                            float(value["vector_completeness_relative_error"])
                        )
                        for value in fp32["shapley"].values()
                    ):
                        raise AssertionError("Shapley permutations/completeness failed")
                    for region_count, value in fp32["shapley"].items():
                        cosines = value["path_shapley_cosine"].numpy()
                        errors = value["path_shapley_norm_relative_error"].numpy()
                        shapley_rows.append(
                            {
                                "model": model,
                                "case_id": case["case_id"],
                                "layer": int(case["layer"]),
                                "regions": int(region_count),
                                "permutations": int(
                                    value.get(
                                        "permutation_count",
                                        expected_permutations,
                                    )
                                ),
                                "median_path_shapley_cosine": float(
                                    np.median(cosines)
                                ),
                                "median_path_shapley_norm_relative_error": float(
                                    np.median(errors)
                                ),
                                "vector_completeness_relative_error": float(
                                    value["vector_completeness_relative_error"]
                                ),
                            }
                        )
                    observed_lambdas = {
                        float(row["lambda"])
                        for rows in fp32["source_scaling"].values()
                        for row in rows
                    }
                    if observed_lambdas != {0.0, 0.25, 0.5, 0.75, 1.0, 1.25}:
                        raise AssertionError("Source-scaling lambda grid changed")
                    if fp32.get("linearize_benchmark"):
                        benchmark_rows.append(
                            {
                                "model": model,
                                "kind": "real_fp32",
                                "case_id": case["case_id"],
                                "metrics": fp32["linearize_benchmark"],
                            }
                        )
        processed[model] = len(image_ids)
        detailed[model] = detailed_count
    summaries = []
    for (model, layer, k), values in sorted(records.items()):
        summaries.append(
            {
                "model": model,
                "layer": layer,
                "k": k,
                "cases": len(values),
                "median_spearman": float(np.median([v["spearman"] for v in values])),
                "median_js": float(np.median([v["js"] for v in values])),
                "median_top32_overlap": float(
                    np.median([v["top32_overlap"] for v in values])
                ),
                "p90_gross_relative_error": float(
                    np.quantile([v["gross_relative_error"] for v in values], 0.9)
                ),
                "p90_kappa_difference": float(
                    np.quantile([v["kappa_difference"] for v in values], 0.9)
                ),
            }
        )
    if not smoke:
        for model in models:
            if processed[model] != 200 or detailed[model] != 50:
                raise AssertionError(
                    f"{model} audit acceptance requires 200 images and 50 "
                    f"FP32/Shapley cases, got {processed[model]}/{detailed[model]}"
                )
    return {
        "processed_images": processed,
        "fp32_shapley_cases": detailed,
        "summaries": summaries,
        "linearize_benchmarks": benchmark_rows,
        "audit_control_rows": control_rows,
        "source_scaling_rows": scaling_rows,
        "shapley_rows": shapley_rows,
    }


def select_frozen_k(summaries: Sequence[Mapping[str, Any]]) -> tuple[int, dict[str, Any]]:
    decisions = {}
    for k in (4, 8, 16, 32):
        rows = [row for row in summaries if int(row["k"]) == k]
        passed = bool(rows) and all(
            float(row["median_spearman"]) >= 0.99
            and float(row["median_js"]) <= 0.005
            and float(row["median_top32_overlap"]) >= 0.95
            and float(row["p90_gross_relative_error"]) <= 0.01
            and float(row["p90_kappa_difference"]) <= 0.01
            for row in rows
        )
        decisions[str(k)] = {"passed_all_model_layers": passed, "rows": len(rows)}
        if passed:
            return k, decisions
    return 64, decisions


def select_linearize_backend(benchmarks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    checks = []
    for row in benchmarks:
        metrics = row["metrics"]
        vmap = metrics.get("vmap_jvp", {})
        linear = metrics.get("linearize", {})
        passed = (
            vmap.get("status") == "PASS"
            and linear.get("status") == "PASS"
            and float(metrics.get("relative_l2_error", float("inf"))) <= 5e-3
            and int(linear.get("increment_bytes", 1))
            <= int(vmap.get("increment_bytes", 0))
            and float(linear.get("seconds", float("inf")))
            <= 0.9 * float(vmap.get("seconds", 0.0))
        )
        checks.append({**row, "passed": passed})
    required_kinds = {"synthetic", "real_fp32"}
    kinds_by_model = {
        model: {row["kind"] for row in checks if row["model"] == model}
        for model in PRIMARY_MODELS
    }
    accepted = (
        bool(checks)
        and all(row["passed"] for row in checks)
        and all(required_kinds.issubset(kinds_by_model[model]) for model in PRIMARY_MODELS)
    )
    return {
        "selected_backend": "linearize" if accepted else "vmap_jvp",
        "accepted": accepted,
        "criteria": {
            "relative_l2_max": 5e-3,
            "peak_memory_must_not_increase": True,
            "minimum_runtime_reduction": 0.10,
            "required": "FP32 synthetic and real cases for both primary models",
        },
        "checks": checks,
    }


def freeze_numerics(models: Sequence[str], *, smoke: bool) -> dict[str, Any]:
    missing_primary = set(PRIMARY_MODELS) - set(models)
    if missing_primary:
        raise ValueError(
            "K/backend may only be frozen with both primary models present: "
            f"missing {sorted(missing_primary)}"
        )
    audit = summarize_k_convergence(models, smoke=smoke)
    primary_summaries = [
        row for row in audit["summaries"] if row["model"] in PRIMARY_MODELS
    ]
    frozen_k, decisions = select_frozen_k(primary_summaries)
    backend = select_linearize_backend(audit["linearize_benchmarks"])
    payload = {
        "reference": "Gauss-Legendre K64 numerical reference; not ground truth",
        "selection_models": list(PRIMARY_MODELS),
        "frozen_k": frozen_k,
        "candidate_decisions": decisions,
        **audit,
    }
    for model in models:
        root = result_root(model)
        suffix = "_smoke" if smoke else ""
        atomic_json_save(payload, root / f"tables/frozen_k{suffix}.json")
        atomic_json_save(backend, root / f"tables/linearize_decision{suffix}.json")
        with (root / f"tables/k_convergence{suffix}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(audit["summaries"][0]))
            writer.writeheader(); writer.writerows(audit["summaries"])
        for name, key in (
            ("audit_controls", "audit_control_rows"),
            ("source_scaling", "source_scaling_rows"),
            ("vector_shapley", "shapley_rows"),
        ):
            rows = [row for row in audit[key] if row["model"] == model]
            with (root / f"tables/{name}{suffix}.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
    return {"frozen_k": frozen_k, "backend": backend, "audit": audit}


def _load_full_model(
    model: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[int]]:
    paths = sorted((result_root(model) / "shards/full").glob("features*_shard_*.pt"))
    if not paths:
        raise FileNotFoundError(f"No full source-attribution shards for {model}")
    positions = []
    mentions = []
    image_ids = set()
    target_keys = set()
    for path in paths:
        shard = torch.load(path, map_location="cpu", weights_only=False)
        overlap = image_ids & set(map(int, shard["image_ids"]))
        if overlap:
            raise AssertionError(f"Duplicate images across shards: {sorted(overlap)[:5]}")
        image_ids.update(map(int, shard["image_ids"]))
        for row in shard["positions"]:
            if row["target_key"] in target_keys:
                raise AssertionError(f"Duplicate target {row['target_key']}")
            target_keys.add(row["target_key"])
            positions.append(row)
        mentions.extend(shard["sample_table"])
    return positions, mentions, image_ids


def build_feature_sets(position: Mapping[str, Any]) -> dict[str, np.ndarray]:
    blocks = {
        "R": np.asarray(position["r_cos"], dtype=np.float32).reshape(-1),
        "AE": np.asarray(position["ae_strength"], dtype=np.float32).reshape(-1),
        "S": np.asarray(position["gross_strength"], dtype=np.float32).reshape(-1),
        "K": np.asarray(position["kappa"], dtype=np.float32).reshape(-1),
    }
    result = {
        "A": np.concatenate([blocks["R"], blocks["AE"]]),
        "B": blocks["AE"],
        "F": np.concatenate([blocks["AE"], blocks["S"], blocks["K"]]),
    }
    for family in DISTANCES:
        values = {
            name: np.asarray(position[family.lower()][name], dtype=np.float32).reshape(-1)
            for name in PAIR_NAMES
        }
        result[f"C_{family}"] = np.concatenate([blocks["AE"], values["D_EW"]])
        result[f"D_{family}"] = np.concatenate([blocks["AE"], values["D_WF"]])
        result[f"E_{family}"] = np.concatenate([blocks["AE"], values["D_EF"]])
        result[f"G_{family}"] = np.concatenate(
            [blocks["AE"], *(values[name] for name in PAIR_NAMES), blocks["S"], blocks["K"]]
        )
        result[f"H_{family}"] = np.concatenate([blocks["R"], result[f"G_{family}"]])
    if set(result) != {
        "A", "B", "F", "C_JS", "D_JS", "E_JS", "G_JS", "H_JS",
        "C_OT", "D_OT", "E_OT", "G_OT", "H_OT",
    }:
        raise AssertionError("Detector feature-set registry is not exactly 13 groups")
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError("Detector feature blocks contain non-finite values")
    return result


def _detector_matrices(model: str) -> dict[str, Any]:
    positions, mentions, processed_image_ids = _load_full_model(model)
    by_target = {row["target_key"]: row for row in positions}
    rows = []
    seen_mentions = set()
    for mention in mentions:
        mention_id = str(mention["mention_id"])
        if mention_id in seen_mentions:
            raise AssertionError(f"Duplicate mention {mention_id}")
        seen_mentions.add(mention_id)
        target_key = str(mention["target_key"])
        if target_key not in by_target:
            raise KeyError(f"Mention {mention_id} misses target {target_key}")
        rows.append(
            {
                "mention_id": mention_id,
                "target_key": target_key,
                "image_id": int(mention["image_id"]),
                "label": int(mention["label"]),
                "net_strength": np.asarray(
                    by_target[target_key]["net_strength"], dtype=np.float32
                ).reshape(-1),
                "write_strength": np.asarray(
                    by_target[target_key]["write_mag"], dtype=np.float32
                ).sum(axis=-1),
                "features": build_feature_sets(by_target[target_key]),
            }
        )
    splits = load_json(str(ROOT / "outputs" / model / EXPERIMENT / "image_splits.json"))
    train_ids = set(map(int, splits["train"]))
    test_ids = set(map(int, splits["test"]))
    expected_images = train_ids | test_ids
    if processed_image_ids != expected_images:
        raise AssertionError(
            f"Formal extraction requires every split image: missing="
            f"{len(expected_images - processed_image_ids)}, extra="
            f"{len(processed_image_ids - expected_images)}"
        )
    mentioned_targets = {str(row["target_key"]) for row in mentions}
    if set(by_target) != mentioned_targets:
        raise AssertionError(
            "Formal extraction positions do not exactly match unique mention targets"
        )
    layer_count = len(next(iter(by_target.values()))["r_cos"])
    for target_key, position in by_target.items():
        for field in (
            "attention_evidence", "write_mag", "p_write", "p_ffn", "path_signed_q"
        ):
            values = torch.as_tensor(position[field])
            if int(values.shape[0]) != layer_count or not bool(torch.isfinite(values).all()):
                raise AssertionError(f"Invalid all-layer {field} for {target_key}")
        for field in ("attention_evidence", "p_write", "p_ffn"):
            sums = torch.as_tensor(position[field]).float().sum(dim=-1)
            if not torch.allclose(sums, torch.ones_like(sums), atol=2e-4, rtol=0.0):
                raise AssertionError(f"Non-normalized {field} for {target_key}")
    train = [row for row in rows if row["image_id"] in train_ids]
    test = [row for row in rows if row["image_id"] in test_ids]
    if not train or not test or set(row["image_id"] for row in rows) - train_ids - test_ids:
        raise AssertionError("Rows do not align to the fixed 8:2 image split")
    specs = list(rows[0]["features"])
    return {
        "specs": specs,
        "X_train": {
            spec: np.stack([row["features"][spec] for row in train]) for spec in specs
        },
        "X_test": {
            spec: np.stack([row["features"][spec] for row in test]) for spec in specs
        },
        "y_train": np.asarray([row["label"] for row in train], dtype=np.int32),
        "y_test": np.asarray([row["label"] for row in test], dtype=np.int32),
        "net_strength_train": np.stack([row["net_strength"] for row in train]),
        "net_strength_test": np.stack([row["net_strength"] for row in test]),
        "write_strength_train": np.stack([row["write_strength"] for row in train]),
        "write_strength_test": np.stack([row["write_strength"] for row in test]),
        "train_target_keys": [row["target_key"] for row in train],
        "test_target_keys": [row["target_key"] for row in test],
        "test_image_ids": np.asarray([row["image_id"] for row in test], dtype=np.int64),
        "test_mention_ids": [row["mention_id"] for row in test],
        "counts": {
            "processed_images": len(processed_image_ids),
            "unique_targets": len(positions),
            "mentions": len(rows),
            "train_mentions": len(train),
            "test_mentions": len(test),
        },
    }


def _train_model(
    model: str, *, device: torch.device, resume: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    data = _detector_matrices(model)
    root = result_root(model)
    progress_path = root / "metrics/detector_training_progress.pt"
    progress = (
        torch.load(progress_path, map_location="cpu", weights_only=False)
        if resume and progress_path.exists()
        else {"metrics": {}, "predictions": {}}
    )
    for spec in data["specs"]:
        progress["metrics"].setdefault(spec, {})
        progress["predictions"].setdefault(spec, {})
        for seed in SEEDS:
            if seed in progress["predictions"][spec]:
                continue
            config = TorchProbeConfig(
                hidden_sizes=(128, 64, 32),
                dropout=0.3,
                batch_size=256,
                num_epochs=100,
                seed=seed,
                positive_class="real",
                split_protocol="strict_82_no_validation",
                threshold_selection="train_f1",
                checkpoint_selection="minimum_train_loss",
            )
            metrics = train_and_evaluate_probe(
                X_train=data["X_train"][spec],
                y_train=data["y_train"],
                X_val=np.empty((0, data["X_train"][spec].shape[1]), dtype=np.float32),
                y_val=np.empty(0, dtype=np.int32),
                X_test=data["X_test"][spec],
                y_test=data["y_test"],
                config=config,
                device=device,
                output_dir=str(root / f"metrics/probes/{spec}/seed{seed}"),
                return_probabilities=True,
            )
            probabilities = np.asarray(metrics.pop("test_probabilities"), dtype=np.float32)
            metrics.pop("train_probabilities", None)
            progress["metrics"][spec][seed] = metrics
            progress["predictions"][spec][seed] = probabilities
            atomic_torch_save(progress, progress_path)
            print(
                f"[detector] {model} {spec} seed={seed} "
                f"AUROC={metrics['auc']:.4f}",
                flush=True,
            )
    return progress, data


def _bootstrap_model(
    model: str,
    progress: Mapping[str, Any],
    data: Mapping[str, Any],
    replicates: int,
) -> dict[str, Any]:
    ensembles = {
        spec: np.mean(
            [np.asarray(progress["predictions"][spec][seed]) for seed in SEEDS],
            axis=0,
        )
        for spec in data["specs"]
    }
    comparisons = []
    for family in DISTANCES:
        comparisons.extend(
            [
                (f"G_{family}", f"C_{family}", "preregistered_gate"),
                (f"H_{family}", "A", "h_minus_a"),
                (f"G_{family}", f"D_{family}", "single_block_ablation"),
                (f"G_{family}", f"E_{family}", "single_block_ablation"),
                (f"G_{family}", "F", "single_block_ablation"),
            ]
        )
    bootstrap = {}
    for offset, (new, baseline, kind) in enumerate(comparisons):
        row = paired_image_bootstrap(
            labels=data["y_test"],
            image_ids=data["test_image_ids"],
            new_probabilities=ensembles[new],
            baseline_probabilities=ensembles[baseline],
            replicates=replicates,
            seed=20260829 + offset,
        )
        if int(row["replicates"]) != int(replicates):
            raise AssertionError("Bootstrap did not produce the requested resamples")
        bootstrap[f"{new}__minus__{baseline}"] = {**row, "kind": kind}
    ensemble_metrics = {
        spec: {
            "auroc": float(roc_auc_score(data["y_test"], probabilities)),
        }
        for spec, probabilities in ensembles.items()
    }
    return {
        "model": model,
        "counts": data["counts"],
        "seeds": list(SEEDS),
        "feature_sets": data["specs"],
        "per_seed_metrics": progress["metrics"],
        "seed_ensemble_metrics": ensemble_metrics,
        "paired_bootstrap": bootstrap,
        "bootstrap_effective_resamples": replicates,
    }


def gate_from_results(results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    comparisons = []
    gate_models = [model for model in PRIMARY_MODELS if model in results] or list(results)
    for model in gate_models:
        result = results[model]
        for family in DISTANCES:
            key = f"G_{family}__minus__C_{family}"
            row = result["paired_bootstrap"][key]
            comparisons.append(
                {
                    "model": model,
                    "distance": family,
                    "auroc_delta": row["new_minus_baseline_auroc"],
                    "ci95": row["auroc_ci95"],
                    "passed": float(row["auroc_ci95"][0]) > 0.0,
                }
            )
    passed = any(row["passed"] for row in comparisons)
    return {
        "gate": "PASS" if passed else "NOT_PASS",
        "rule": (
            "Run extension if any model x {JS,OT} G-C seed-ensemble AUROC "
            "10,000-image-bootstrap 95% CI lower bound is > 0"
        ),
        "passed": passed,
        "comparisons": comparisons,
        "conditional_stage_status": (
            "RUN_REQUIRED" if passed else "NOT_RUN_BY_PREREGISTERED_GATE"
        ),
    }


def _plot_model(model: str, payload: Mapping[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = payload["seed_ensemble_metrics"]
    names = list(metrics)
    values = [metrics[name]["auroc"] for name in names]
    figure, axis = plt.subplots(figsize=(12, 4.5))
    axis.bar(range(len(names)), values)
    axis.set_xticks(range(len(names)), names, rotation=55, ha="right")
    axis.set_ylabel("Test AUROC")
    axis.set_title(f"{model}: Vector FFN source detector blocks")
    axis.set_ylim(0.0, 1.0)
    figure.tight_layout()
    figure.savefig(result_root(model) / "figures/detector_auroc.png", dpi=180)
    figure.savefig(result_root(model) / "figures/detector_auroc.pdf")
    plt.close(figure)


def _plot_fixed_examples(model: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    positions, mentions, _images = _load_full_model(model)
    by_target = {row["target_key"]: row for row in positions}
    selected = {}
    for mention in sorted(
        mentions, key=lambda row: (int(row["image_id"]), int(row["response_index"]))
    ):
        label = int(mention["label"])
        if label not in selected and mention["target_key"] in by_target:
            selected[label] = mention["target_key"]
    if set(selected) != {0, 1}:
        raise AssertionError("Fixed examples require one HALL and one REAL target")
    layer_number = FORMAL_LAYERS_BY_MODEL[model][2]
    layer_index = layer_number - 1
    figure, axes = plt.subplots(2, 4, figsize=(13, 6))
    fields = (
        ("attention_evidence", "T"),
        ("p_write", "P_WRITE"),
        ("p_ffn", "P_FFN"),
        ("path_signed_q", "signed-Q"),
    )
    for row_index, label in enumerate((0, 1)):
        position = by_target[selected[label]]
        height, width = map(int, position["visual_grid"])
        for column, (field, title) in enumerate(fields):
            values = torch.as_tensor(position[field])[layer_index].reshape(height, width)
            cmap = "coolwarm" if field == "path_signed_q" else "viridis"
            image = axes[row_index, column].imshow(values.numpy(), cmap=cmap)
            axes[row_index, column].set_title(
                f"{'HALL' if label == 0 else 'REAL'} {title}"
            )
            axes[row_index, column].axis("off")
            figure.colorbar(image, ax=axes[row_index, column], fraction=0.045)
    figure.suptitle(
        f"{model} fixed preregistered examples, layer {layer_number} (bbox auxiliary only)"
    )
    figure.tight_layout()
    figure.savefig(result_root(model) / "figures/fixed_real_hall_maps.png", dpi=180)
    figure.savefig(result_root(model) / "figures/fixed_real_hall_maps.pdf")
    plt.close(figure)
    atomic_json_save(
        {
            "selection_rule": "lowest (image_id,response_index) per label; no outcome filtering",
            "layer": layer_number,
            "HALL": selected[0],
            "REAL": selected[1],
            "bbox_role": "auxiliary_sanity_check_only",
        },
        result_root(model) / "tables/fixed_example_selection.json",
    )


def _write_detector_table(model: str, payload: Mapping[str, Any]) -> None:
    path = result_root(model) / "tables/detector_metrics.csv"
    fields = (
        "feature_set", "seed", "threshold_rule", "auroc", "real_aupr",
        "hall_aupr", "precision", "recall", "f1", "decision_threshold",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for spec, seeds in payload["per_seed_metrics"].items():
            for seed, metrics in seeds.items():
                for rule in ("fixed_0.5", "train_f1"):
                    report = metrics["threshold_reports"][rule]
                    test = report["test_metrics"]
                    writer.writerow(
                        {
                            "feature_set": spec,
                            "seed": seed,
                            "threshold_rule": rule,
                            "auroc": test["auc"],
                            "real_aupr": test["real_positive"]["aupr"],
                            "hall_aupr": test["hallucination_positive"]["aupr"],
                            "precision": test["precision"],
                            "recall": test["recall"],
                            "f1": test["f1"],
                            "decision_threshold": report["threshold"],
                        }
                    )


def _mark_gate_status(gate: Mapping[str, Any]) -> None:
    source_root = result_root(PRIMARY_MODELS[0])
    frozen = json.loads(
        (source_root / "tables/frozen_k.json").read_text(encoding="utf-8")
    )
    backend = json.loads(
        (source_root / "tables/linearize_decision.json").read_text(encoding="utf-8")
    )
    for model in (*PRIMARY_MODELS, *EXTENSION_MODELS):
        root = result_root(model)
        layout = ExperimentLayout.create(root)
        if not (root / "manifests/run_status.json").exists():
            initialize_manifests(
                layout=layout,
                repo_root=ROOT,
                experiment_config={"conditional_gate": gate["rule"]},
                input_paths=[
                    ROOT / "outputs" / model / EXPERIMENT / "generations.json",
                    ROOT / "outputs" / model / EXPERIMENT / "labeling.json",
                    ROOT / "outputs" / model / EXPERIMENT / "image_splits.json",
                ],
                exact_command=[sys.executable, *sys.argv],
            )
        atomic_json_save(gate, root / "tables/preregistered_gate.json")
        if gate["passed"] and model in EXTENSION_MODELS:
            atomic_json_save(frozen, root / "tables/frozen_k.json")
            atomic_json_save(backend, root / "tables/linearize_decision.json")
        if not gate["passed"]:
            if model in EXTENSION_MODELS:
                update_stage_status(
                    layout=layout,
                    stage=f"extension:{model}",
                    status="NOT_RUN",
                    details={"reason": "NOT_RUN_BY_PREREGISTERED_GATE"},
                )
            update_stage_status(
                layout=layout,
                stage=f"counterfactuals:{model}",
                status="NOT_RUN",
                details={"reason": "NOT_RUN_BY_PREREGISTERED_GATE"},
            )


def run_detectors(
    models: Sequence[str], *, device: torch.device, replicates: int, resume: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    results = {}
    for model in models:
        progress, data = _train_model(model, device=device, resume=resume)
        results[model] = _bootstrap_model(model, progress, data, replicates)
        atomic_json_save(
            _json_ready(results[model]), result_root(model) / "metrics/detector_results.json"
        )
        _plot_model(model, results[model])
        _plot_fixed_examples(model)
        _write_detector_table(model, results[model])
    gate_results = dict(results)
    for primary in PRIMARY_MODELS:
        if primary in gate_results:
            continue
        path = result_root(primary) / "metrics/detector_results.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Cannot evaluate the preregistered gate without {primary}: {path}"
            )
        gate_results[primary] = json.loads(path.read_text(encoding="utf-8"))
    gate = gate_from_results(gate_results)
    for model in models:
        atomic_json_save(gate, result_root(model) / "tables/preregistered_gate.json")
    _mark_gate_status(gate)
    return results, gate


def main() -> None:
    args = parse_args()
    models = tuple(item.strip() for item in args.models.split(",") if item.strip())
    if not models:
        raise ValueError("--models must not be empty")
    if args.bootstrap_resamples <= 0:
        raise ValueError("--bootstrap-resamples must be positive")
    if args.stage in {"freeze", "all"}:
        frozen = freeze_numerics(models, smoke=args.smoke)
        print(
            f"[freeze] K={frozen['frozen_k']} "
            f"backend={frozen['backend']['selected_backend']}",
            flush=True,
        )
    if args.stage in {"detectors", "all"}:
        results, gate = run_detectors(
            models,
            device=torch.device(args.training_device),
            replicates=args.bootstrap_resamples,
            resume=args.resume,
        )
        for model in models:
            layout = ExperimentLayout.create(result_root(model))
            update_stage_status(
                layout=layout,
                stage=f"analysis_and_detectors:{model}",
                status="PASS",
                details={
                    "feature_sets": 13,
                    "seeds": list(SEEDS),
                    "bootstrap_resamples": args.bootstrap_resamples,
                    "gate": gate["gate"],
                },
            )
        print(json.dumps(gate, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
