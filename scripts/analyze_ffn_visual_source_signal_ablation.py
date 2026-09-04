#!/usr/bin/env python3
"""Train exploratory source-signal detector ablations and plot strength curves."""

from __future__ import annotations

import argparse
import csv
import gc
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.tc_fvpa_artifacts import atomic_json_save, atomic_torch_save  # noqa: E402
from scripts.analyze_ffn_visual_source_study import (  # noqa: E402
    EXTENSION_MODELS,
    PRIMARY_MODELS,
    SEEDS,
    _detector_matrices,
    _json_ready,
)
from scripts.run_ffn_visual_source_attribution import EXPERIMENT, result_root  # noqa: E402
from scripts.train_old_risk_ev_s_mlp import paired_image_bootstrap  # noqa: E402
from scripts.train_torch_probe_feature_sets import (  # noqa: E402
    TorchProbeConfig,
    train_and_evaluate_probe,
)
from utils.io_utils import load_pkl  # noqa: E402

SPECS = (
    "strength",
    "kappa",
    "strength+kappa",
    "R_cos",
    "strength+kappa+R_cos",
)
OLD_EV_KEY = (
    "dgst_t_hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine_"
    "topk32_hpre_per_layer"
)
OLD_COSINE_KEY = (
    "dgst_t_hpre_raw_logit_gauss_target_cosine_topk32_hpre_per_layer"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", default=",".join((*PRIMARY_MODELS, *EXTENSION_MODELS))
    )
    parser.add_argument("--training-device", default="cuda:0")
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--study", choices=("signals", "old_ev", "ae_cosine"), default="signals"
    )
    return parser.parse_args()


def signal_matrices(data: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    output: dict[str, dict[str, np.ndarray]] = {}
    for split in ("train", "test"):
        a = data[f"X_{split}"]["A"]
        f = data[f"X_{split}"]["F"]
        if a.shape[1] % 2 or f.shape[1] % 3 or a.shape[1] // 2 != f.shape[1] // 3:
            raise AssertionError("A/F dimensions do not encode the same layer count")
        layers = a.shape[1] // 2
        r_cos, ae_from_a = a[:, :layers], a[:, layers:]
        ae_from_f, strength, kappa = np.split(f, 3, axis=1)
        if not np.array_equal(ae_from_a, ae_from_f):
            raise AssertionError("A/F attention-evidence blocks differ")
        output[split] = {
            "strength": strength,
            "kappa": kappa,
            "strength+kappa": np.concatenate([strength, kappa], axis=1),
            "R_cos": r_cos,
            "strength+kappa+R_cos": np.concatenate(
                [strength, kappa, r_cos], axis=1
            ),
        }
        if set(output[split]) != set(SPECS) or not all(
            np.isfinite(values).all() for values in output[split].values()
        ):
            raise AssertionError("Signal ablation registry or values are invalid")
    return output


def swap_ae_for_old_ev(
    matrices: dict[str, np.ndarray], ev: np.ndarray
) -> dict[str, np.ndarray]:
    """Replace only the AE block in each of the formal 13 feature groups."""
    ev = np.asarray(ev, dtype=np.float32)
    if ev.ndim != 2 or not np.isfinite(ev).all():
        raise ValueError("old EV must be a finite [mentions,layers] matrix")
    output = {}
    for spec, values in matrices.items():
        values = np.asarray(values, dtype=np.float32)
        if values.shape[0] != ev.shape[0]:
            raise ValueError(f"{spec} rows do not align with old EV")
        start = ev.shape[1] if spec == "A" or spec.startswith("H_") else 0
        if values.shape[1] < start + ev.shape[1]:
            raise ValueError(f"{spec} has no complete AE block")
        replaced = values.copy()
        replaced[:, start : start + ev.shape[1]] = ev
        output[spec] = replaced
    return output


def ae_times_cosine(ae: np.ndarray, cosine: np.ndarray) -> np.ndarray:
    ae = np.asarray(ae, dtype=np.float32)
    cosine = np.asarray(cosine, dtype=np.float32)
    if ae.shape != cosine.shape or ae.ndim != 2:
        raise ValueError("AE and top-32 cosine must have the same [mentions,layers] shape")
    result = ae * cosine
    if not np.isfinite(result).all():
        raise ValueError("AE times cosine contains non-finite values")
    return result


def old_ev_matrices(
    model: str, data: dict[str, Any], *, replacement: str = "old_ev"
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    """Align an existing top-32 cosine signal to the formal mention rows."""
    if replacement not in {"old_ev", "ae_cosine"}:
        raise ValueError(f"Unknown AE replacement: {replacement}")
    feature_key = OLD_EV_KEY if replacement == "old_ev" else OLD_COSINE_KEY
    split_keys = {
        split: list(data[f"{split}_target_keys"]) for split in ("train", "test")
    }
    wanted = set(split_keys["train"]) | set(split_keys["test"])
    lookup: dict[str, np.ndarray] = {}
    duplicates = 0
    maximum_duplicate_error = 0.0
    rows = load_pkl(str(ROOT / "outputs" / model / EXPERIMENT / "features.pkl"))
    for row in rows:
        target_key = f"{int(row['image_id'])}:{int(row['response_token_idx'])}"
        if target_key not in wanted:
            continue
        value = np.asarray(row[feature_key], dtype=np.float32).reshape(-1)
        if not np.isfinite(value).all():
            raise ValueError(f"Non-finite old EV for {target_key}")
        previous = lookup.get(target_key)
        if previous is None:
            lookup[target_key] = value
        else:
            duplicates += 1
            error = float(np.max(np.abs(previous - value)))
            maximum_duplicate_error = max(maximum_duplicate_error, error)
            if error > 1e-6:
                raise AssertionError(f"Conflicting old EV for {target_key}")
    del rows
    gc.collect()
    missing = wanted - set(lookup)
    if missing:
        raise AssertionError(f"Old EV misses {len(missing)} formal targets")
    ev = {
        split: np.stack([lookup[key] for key in keys]).astype(np.float32, copy=False)
        for split, keys in split_keys.items()
    }
    if replacement == "ae_cosine":
        for split in ("train", "test"):
            ev[split] = ae_times_cosine(data[f"X_{split}"]["B"], ev[split])
    matrices = {
        split: swap_ae_for_old_ev(data[f"X_{split}"], ev[split])
        for split in ("train", "test")
    }
    return matrices, {
        "feature_key": feature_key,
        "replacement": replacement,
        "unique_targets": len(lookup),
        "duplicate_feature_rows": duplicates,
        "maximum_duplicate_error": maximum_duplicate_error,
        "layers": int(ev["train"].shape[1]),
    }


def write_strength_curve(
    model: str,
    data: dict[str, Any],
    matrices: dict[str, dict[str, np.ndarray]],
    root: Path,
) -> dict[str, Any]:
    """Write all-cohort mention-weighted median/IQR layer curves."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    strength = np.concatenate(
        [matrices["train"]["strength"], matrices["test"]["strength"]]
    )
    labels = np.concatenate([data["y_train"], data["y_test"]])
    rows = []
    for label, name in ((0, "HALL"), (1, "REAL")):
        selected = strength[labels == label]
        for layer in range(strength.shape[1]):
            values = selected[:, layer]
            rows.append(
                {
                    "model": model,
                    "label": name,
                    "layer": layer + 1,
                    "n_mentions": len(values),
                    "mean": float(values.mean()),
                    "q25": float(np.quantile(values, 0.25)),
                    "median": float(np.median(values)),
                    "q75": float(np.quantile(values, 0.75)),
                    "raw_strength_real_auroc": float(
                        roc_auc_score(labels, strength[:, layer])
                    ),
                }
            )

    table_path = root / "tables/strength_real_hall_curve.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    figure_path = root / "figures/strength_real_hall_curve.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, color in (("REAL", "#2166ac"), ("HALL", "#b2182b")):
        selected = [row for row in rows if row["label"] == name]
        x = np.asarray([row["layer"] for row in selected])
        median = np.asarray([row["median"] for row in selected])
        q25 = np.asarray([row["q25"] for row in selected])
        q75 = np.asarray([row["q75"] for row in selected])
        ax.plot(x, median, color=color, linewidth=1.8, label=name)
        ax.fill_between(x, q25, q75, color=color, alpha=0.14)
    ax.set_yscale("log")
    ax.set_xlabel("Decoder layer")
    ax.set_ylabel("gross strength (median; IQR, log scale)")
    ax.set_title(f"{model}: FFN path strength by label")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    return {
        "cohort": "all formal mentions (train + test), descriptive only",
        "n_hall": int((labels == 0).sum()),
        "n_real": int((labels == 1).sum()),
        "layers": int(strength.shape[1]),
        "table": str(table_path.relative_to(ROOT)),
        "figure": str(figure_path.relative_to(ROOT)),
    }


def train_matrices(
    model: str,
    data: dict[str, Any],
    matrices: dict[str, dict[str, np.ndarray]],
    specs: tuple[str, ...],
    *,
    device: torch.device,
    resume: bool,
    progress_name: str,
    probe_directory: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    root = result_root(model)
    progress_path = root / f"metrics/{progress_name}"
    progress = (
        torch.load(progress_path, map_location="cpu", weights_only=False)
        if resume and progress_path.exists()
        else {"metrics": {}, "predictions": {}}
    )
    for spec in specs:
        progress["metrics"].setdefault(spec, {})
        progress["predictions"].setdefault(spec, {})
        for seed in SEEDS:
            if seed in progress["predictions"][spec]:
                continue
            metrics = train_and_evaluate_probe(
                X_train=matrices["train"][spec],
                y_train=data["y_train"],
                X_val=np.empty(
                    (0, matrices["train"][spec].shape[1]), dtype=np.float32
                ),
                y_val=np.empty(0, dtype=np.int32),
                X_test=matrices["test"][spec],
                y_test=data["y_test"],
                config=TorchProbeConfig(
                    hidden_sizes=(128, 64, 32),
                    dropout=0.3,
                    batch_size=256,
                    num_epochs=100,
                    seed=seed,
                    positive_class="real",
                    split_protocol="strict_82_no_validation",
                    threshold_selection="train_f1",
                    checkpoint_selection="minimum_train_loss",
                ),
                device=device,
                output_dir=str(root / probe_directory / spec / f"seed{seed}"),
                return_probabilities=True,
            )
            probabilities = np.asarray(metrics.pop("test_probabilities"), dtype=np.float32)
            metrics.pop("train_probabilities", None)
            progress["metrics"][spec][seed] = metrics
            progress["predictions"][spec][seed] = probabilities
            atomic_torch_save(progress, progress_path)
            print(f"[{model}] {spec} seed={seed} AUROC={metrics['auc']:.6f}", flush=True)

    ensembles = {
        spec: np.mean(
            [np.asarray(progress["predictions"][spec][seed]) for seed in SEEDS],
            axis=0,
        )
        for spec in specs
    }
    summaries = {}
    for spec in specs:
        reports = [
            progress["metrics"][spec][seed]["threshold_reports"]["train_f1"][
                "test_metrics"
            ]
            for seed in SEEDS
        ]
        summaries[spec] = {
            "dimensions": int(matrices["train"][spec].shape[1]),
            "ensemble_auroc": float(roc_auc_score(data["y_test"], ensembles[spec])),
            "mean_real_aupr": float(
                np.mean([row["real_positive"]["aupr"] for row in reports])
            ),
            "mean_hall_aupr": float(
                np.mean([row["hallucination_positive"]["aupr"] for row in reports])
            ),
            "mean_hall_precision": float(
                np.mean([row["hallucination_positive"]["precision"] for row in reports])
            ),
            "mean_hall_recall": float(
                np.mean([row["hallucination_positive"]["recall"] for row in reports])
            ),
            "mean_hall_f1": float(
                np.mean([row["hallucination_positive"]["f1"] for row in reports])
            ),
            "per_seed_auroc": {
                str(seed): float(progress["metrics"][spec][seed]["auc"])
                for seed in SEEDS
            },
        }
    return progress, ensembles, summaries


def run_model(
    model: str, *, device: torch.device, bootstrap_resamples: int, resume: bool
) -> dict[str, Any]:
    data = _detector_matrices(model)
    matrices = signal_matrices(data)
    root = result_root(model)
    strength_curve = write_strength_curve(model, data, matrices, root)
    progress, ensembles, summaries = train_matrices(
        model,
        data,
        matrices,
        SPECS,
        device=device,
        resume=resume,
        progress_name="signal_ablation_training_progress.pt",
        probe_directory="metrics/probes_signal_ablation",
    )
    bootstrap = {}
    comparisons = (
        ("strength+kappa", "strength"),
        ("strength+kappa", "kappa"),
        ("strength+kappa+R_cos", "strength+kappa"),
        ("strength+kappa+R_cos", "strength"),
        ("strength+kappa+R_cos", "kappa"),
        ("strength+kappa+R_cos", "R_cos"),
    )
    for offset, (combined, baseline) in enumerate(comparisons):
        bootstrap[f"{combined}__minus__{baseline}"] = paired_image_bootstrap(
            labels=data["y_test"],
            image_ids=data["test_image_ids"],
            new_probabilities=ensembles[combined],
            baseline_probabilities=ensembles[baseline],
            replicates=bootstrap_resamples,
            seed=20260904 + offset,
        )
    result = {
        "model": model,
        "status": "EXPLORATORY_NOT_PREREGISTERED",
        "feature_definitions": {
            "strength": "all-layer gross_strength",
            "kappa": "all-layer net_strength/gross_strength",
            "R_cos": "all-layer old_hpre_cos/hpre_raw_logit_gauss/sqrt_matched_state OT risk",
            "strength+kappa": "concatenate strength and kappa; no other blocks",
            combined: "concatenate strength, kappa, and R_cos; no AE or JS/OT path-distance blocks",
        },
        "protocol": {
            "split": "same fixed image 8:2 split as formal detector",
            "seeds": list(SEEDS),
            "hidden_sizes": [128, 64, 32],
            "dropout": 0.3,
            "drop_last": False,
            "batch_size": 256,
            "epochs_max": 100,
            "standardization": "none",
            "checkpoint": "minimum_train_loss",
            "threshold": "train_REAL_F1",
            "bootstrap_resamples": bootstrap_resamples,
        },
        "counts": data["counts"],
        "strength_curve": strength_curve,
        "summaries": summaries,
        "paired_image_bootstrap": bootstrap,
    }
    atomic_json_save(_json_ready(result), root / "metrics/signal_ablation_results.json")
    table_path = root / "tables/signal_ablation_metrics.csv"
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        rows = [{"feature_set": spec, **summaries[spec]} for spec in SPECS]
        fields = [key for key in rows[0] if key != "per_seed_auroc"] + [
            f"seed{seed}_auroc" for seed in SEEDS
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            seeds = row.pop("per_seed_auroc")
            writer.writerow(
                {**row, **{f"seed{seed}_auroc": seeds[str(seed)] for seed in SEEDS}}
            )
    return result


def run_old_ev_model(
    model: str,
    *,
    device: torch.device,
    bootstrap_resamples: int,
    resume: bool,
    replacement: str = "old_ev",
) -> dict[str, Any]:
    """Retrain all 13 groups after replacing only their AE block."""
    data = _detector_matrices(model)
    matrices, alignment = old_ev_matrices(model, data, replacement=replacement)
    specs = tuple(data["specs"])
    root = result_root(model)
    slug = "old_ev" if replacement == "old_ev" else "ae_cosine"
    _progress, ensembles, summaries = train_matrices(
        model,
        data,
        matrices,
        specs,
        device=device,
        resume=resume,
        progress_name=f"{slug}_feature_group_training_progress.pt",
        probe_directory=f"metrics/probes_{slug}_feature_groups",
    )

    formal_progress = torch.load(
        root / "metrics/detector_training_progress.pt",
        map_location="cpu",
        weights_only=False,
    )
    formal_ensembles = {
        spec: np.mean(
            [np.asarray(formal_progress["predictions"][spec][seed]) for seed in SEEDS],
            axis=0,
        )
        for spec in specs
    }
    bootstrap: dict[str, Any] = {}
    delta_name = f"{slug}_minus_ae_auroc"
    for offset, spec in enumerate(specs):
        summaries[spec]["formal_ae_ensemble_auroc"] = float(
            roc_auc_score(data["y_test"], formal_ensembles[spec])
        )
        summaries[spec][delta_name] = float(
            summaries[spec]["ensemble_auroc"]
            - summaries[spec]["formal_ae_ensemble_auroc"]
        )
        if replacement == "old_ev":
            bootstrap[f"{spec}_old_EV__minus__{spec}_AE"] = paired_image_bootstrap(
                labels=data["y_test"],
                image_ids=data["test_image_ids"],
                new_probabilities=ensembles[spec],
                baseline_probabilities=formal_ensembles[spec],
                replicates=bootstrap_resamples,
                seed=20260920 + offset,
            )

    result = {
        "model": model,
        "status": (
            "EXPLORATORY_AE_TO_OLD_EV_SWAP"
            if replacement == "old_ev"
            else "EXPLORATORY_AE_TIMES_COSINE_SWAP"
        ),
        "replacement": {
            "old": "AE_l=sum_j max(attention_lj * hpre_raw_logit_gauss_gate_lj, 0)",
            "new": (
                "old top32 target-distribution mass * mean hpre target-state cosine"
                if replacement == "old_ev"
                else "AE_l * mean hpre target-state cosine over the same target-distribution top32 region"
            ),
            "scope": "replace only the AE block in all 13 formal groups",
            "unchanged": ["R_cos", "S", "kappa", "JS distances", "OT distances"],
        },
        "alignment": alignment,
        "protocol": {
            "split": "same fixed image 8:2 split as formal detector",
            "seeds": list(SEEDS),
            "hidden_sizes": [128, 64, 32],
            "dropout": 0.3,
            "drop_last": False,
            "batch_size": 256,
            "epochs_max": 100,
            "standardization": "none",
            "checkpoint": "minimum_train_loss",
            "threshold": "train_REAL_F1",
            "bootstrap_resamples_per_direct_AE_comparison": (
                bootstrap_resamples
                if replacement == "old_ev"
                else "NOT_RUN_BY_USER_REQUEST"
            ),
        },
        "counts": data["counts"],
        "summaries": summaries,
        f"{slug}_minus_ae_paired_image_bootstrap": (
            bootstrap if replacement == "old_ev" else "NOT_RUN_BY_USER_REQUEST"
        ),
    }
    atomic_json_save(
        _json_ready(result), root / f"metrics/{slug}_feature_group_results.json"
    )

    table_path = root / f"tables/{slug}_feature_group_metrics.csv"
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "feature_set",
            "dimensions",
            "formal_ae_ensemble_auroc",
            f"{slug}_ensemble_auroc",
            delta_name,
            *(("auroc_ci95_low", "auroc_ci95_high") if bootstrap else ()),
            "mean_hall_aupr",
            "mean_hall_precision",
            "mean_hall_recall",
            "mean_hall_f1",
            *(f"seed{seed}_auroc" for seed in SEEDS),
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for spec in specs:
            summary = summaries[spec]
            row = {
                "feature_set": spec,
                "dimensions": summary["dimensions"],
                "formal_ae_ensemble_auroc": summary["formal_ae_ensemble_auroc"],
                f"{slug}_ensemble_auroc": summary["ensemble_auroc"],
                delta_name: summary[delta_name],
                "mean_hall_aupr": summary["mean_hall_aupr"],
                "mean_hall_precision": summary["mean_hall_precision"],
                "mean_hall_recall": summary["mean_hall_recall"],
                "mean_hall_f1": summary["mean_hall_f1"],
                **{
                    f"seed{seed}_auroc": summary["per_seed_auroc"][str(seed)]
                    for seed in SEEDS
                },
            }
            if bootstrap:
                comparison = bootstrap[f"{spec}_old_EV__minus__{spec}_AE"]
                row.update(
                    auroc_ci95_low=comparison["auroc_ci95"][0],
                    auroc_ci95_high=comparison["auroc_ci95"][1],
                )
            writer.writerow(row)
    return result


def main() -> None:
    args = parse_args()
    if args.study != "ae_cosine" and args.bootstrap_resamples <= 0:
        raise ValueError("--bootstrap-resamples must be positive")
    models = tuple(value.strip() for value in args.models.split(",") if value.strip())
    for model in models:
        if args.study == "signals":
            result = run_model(
                model,
                device=torch.device(args.training_device),
                bootstrap_resamples=args.bootstrap_resamples,
                resume=args.resume,
            )
        else:
            result = run_old_ev_model(
                model,
                device=torch.device(args.training_device),
                bootstrap_resamples=args.bootstrap_resamples,
                resume=args.resume,
                replacement=args.study,
            )
        print(_json_ready(result["summaries"]), flush=True)


if __name__ == "__main__":
    main()
