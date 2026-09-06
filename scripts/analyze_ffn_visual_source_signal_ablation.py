#!/usr/bin/env python3
"""Train exploratory source-signal detector ablations and plot strength curves."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.tc_fvpa_artifacts import atomic_json_save, atomic_torch_save  # noqa: E402
from scripts.analyze_ffn_visual_source_study import (  # noqa: E402
    EXTENSION_MODELS,
    PRIMARY_MODELS,
    SEEDS,
    _detector_matrices,
    _json_ready,
    _load_full_model,
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
UNION_JS_SPECS = ("C_JS", "D_JS", "E_JS", "G_JS", "H_JS")
JS_PAIRS = ("D_EW", "D_WF", "D_EF")
NET_SPECS = ("net_strength", "D_OT+strength")
BOUNDED_STRENGTH_SPECS = ("bounded_strength", "D_OT+bounded_strength")
RELATIVE_STRENGTH_SPECS = ("relative_strength", "D_OT+relative_strength")
LOG1P_STRENGTH_SPECS = ("log1p_strength", "D_OT+log1p_strength")
CORE_SPECS = (
    "AE", "AE+I", "AE+S", "AE+I+S", "AE+N", "AE+S+kappa",
    "AE+S+N", "U", "H_OT", "H_JS",
)
CORE_REFERENCES = {"AE": "B", "AE+S+kappa": "F", "H_OT": "H_OT", "H_JS": "H_JS"}
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
        "--study",
        choices=(
            "signals",
            "old_ev",
            "ae_cosine",
            "union_topk_js",
            "net_strength",
            "bounded_strength",
            "relative_strength",
            "log1p_strength",
            "core_signals",
        ),
        default="signals",
    )
    return parser.parse_args()


def union_topk_js(
    left: np.ndarray | torch.Tensor,
    right: np.ndarray | torch.Tensor,
    *,
    side_top_k: int = 32,
) -> np.ndarray:
    """Compute layerwise JS after restricting both marginals to their Top-K union."""
    p = torch.as_tensor(left, dtype=torch.float64)
    q = torch.as_tensor(right, dtype=torch.float64)
    if p.ndim != 2 or p.shape != q.shape or p.shape[1] == 0:
        raise ValueError("union Top-K JS expects matching non-empty [layers,tokens]")
    if not bool(torch.isfinite(p).all() and torch.isfinite(q).all()):
        raise ValueError("union Top-K JS distributions must be finite")
    p, q = p.clamp_min(0.0), q.clamp_min(0.0)
    k = min(max(int(side_top_k), 1), int(p.shape[1]))
    mask = torch.zeros_like(p, dtype=torch.bool)
    mask.scatter_(1, torch.topk(p, k=k, dim=1).indices, True)
    mask.scatter_(1, torch.topk(q, k=k, dim=1).indices, True)
    p = torch.where(mask, p.clamp_min(1e-12), 0.0)
    q = torch.where(mask, q.clamp_min(1e-12), 0.0)
    p /= p.sum(dim=1, keepdim=True)
    q /= q.sum(dim=1, keepdim=True)
    midpoint = 0.5 * (p + q)
    midpoint = midpoint.clamp_min(1e-12)
    js = 0.5 * (p * torch.log(p.clamp_min(1e-12) / midpoint)).sum(dim=1)
    js += 0.5 * (q * torch.log(q.clamp_min(1e-12) / midpoint)).sum(dim=1)
    return js.float().numpy()


def build_union_topk_js_feature_sets(
    formal: dict[str, np.ndarray], distances: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Reuse the formal blocks, changing only the three JS trajectories."""
    ae = np.asarray(formal["B"], dtype=np.float32)
    layers = int(ae.shape[1])
    r_cos = np.asarray(formal["A"], dtype=np.float32)[:, :layers]
    _ae, strength, kappa = np.split(
        np.asarray(formal["F"], dtype=np.float32), 3, axis=1
    )
    if not np.array_equal(ae, _ae) or set(distances) != set(JS_PAIRS):
        raise AssertionError("Formal blocks or union JS distance registry are invalid")
    values = {
        name: np.asarray(distances[name], dtype=np.float32) for name in JS_PAIRS
    }
    if any(
        value.shape != ae.shape or not np.isfinite(value).all()
        for value in values.values()
    ):
        raise ValueError("Union JS trajectories must be finite [mentions,layers] matrices")
    result = {
        "C_JS": np.concatenate([ae, values["D_EW"]], axis=1),
        "D_JS": np.concatenate([ae, values["D_WF"]], axis=1),
        "E_JS": np.concatenate([ae, values["D_EF"]], axis=1),
        "G_JS": np.concatenate(
            [ae, *(values[name] for name in JS_PAIRS), strength, kappa], axis=1
        ),
    }
    result["H_JS"] = np.concatenate([r_cos, result["G_JS"]], axis=1)
    return result


def union_topk_js_matrices(
    model: str, data: dict[str, Any], *, side_top_k: int = 32
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    """Recompute pairwise JS from saved full visual distributions."""
    positions, _mentions, _image_ids = _load_full_model(model)
    by_target = {}
    for position in positions:
        distributions = {
            "E": position["attention_evidence"],
            "W": position["p_write"],
            "F": position["p_ffn"],
        }
        by_target[str(position["target_key"])] = {
            name: union_topk_js(
                distributions[left], distributions[right], side_top_k=side_top_k
            )
            for name, left, right in (
                ("D_EW", "E", "W"),
                ("D_WF", "W", "F"),
                ("D_EF", "E", "F"),
            )
        }
    matrices = {}
    for split in ("train", "test"):
        distances = {
            name: np.stack(
                [by_target[key][name] for key in data[f"{split}_target_keys"]]
            )
            for name in JS_PAIRS
        }
        matrices[split] = build_union_topk_js_feature_sets(
            data[f"X_{split}"], distances
        )
    return matrices, {
        "unique_targets": len(by_target),
        "side_top_k": int(side_top_k),
        "maximum_possible_union_size": int(2 * side_top_k),
    }


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


def net_strength_matrices(data: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    """Build the requested net-strength and D_OT+gross-strength probes."""
    output = {}
    for split in ("train", "test"):
        _ae, gross, _kappa = np.split(data[f"X_{split}"]["F"], 3, axis=1)
        net = np.asarray(data[f"net_strength_{split}"], dtype=np.float32)
        output[split] = {
            "net_strength": net,
            "D_OT+strength": np.concatenate(
                [data[f"X_{split}"]["D_OT"], gross], axis=1
            ),
        }
        if set(output[split]) != set(NET_SPECS) or not all(
            np.isfinite(values).all() for values in output[split].values()
        ):
            raise AssertionError("Net-strength feature registry or values are invalid")
    return output


def bounded_strength_matrices(
    data: dict[str, Any],
) -> tuple[dict[str, dict[str, np.ndarray]], np.ndarray]:
    """Bound gross strength with train-only per-layer median calibration."""
    raw = {
        split: np.split(data[f"X_{split}"]["F"], 3, axis=1)[1]
        for split in ("train", "test")
    }
    if any(not np.isfinite(values).all() or np.any(values < 0) for values in raw.values()):
        raise ValueError("Gross strength must be finite and non-negative")
    tau = np.maximum(
        np.median(raw["train"], axis=0), np.finfo(np.float32).eps
    ).astype(np.float32)
    upper = np.nextafter(np.float32(1.0), np.float32(0.0))
    output = {}
    for split in ("train", "test"):
        bounded = np.clip(raw[split] / (raw[split] + tau), 0.0, upper).astype(
            np.float32, copy=False
        )
        output[split] = {
            "bounded_strength": bounded,
            "D_OT+bounded_strength": np.concatenate(
                [data[f"X_{split}"]["D_OT"], bounded], axis=1
            ),
        }
        if set(output[split]) != set(BOUNDED_STRENGTH_SPECS) or not all(
            np.isfinite(values).all() for values in output[split].values()
        ):
            raise AssertionError("Bounded-strength feature registry or values are invalid")
    return output, tau


def relative_strength_matrices(
    data: dict[str, Any],
) -> dict[str, dict[str, np.ndarray]]:
    """Compute R/(R+W) from gross FFN response R and visual write energy W."""
    output = {}
    upper = np.nextafter(np.float32(1.0), np.float32(0.0))
    for split in ("train", "test"):
        gross = np.split(data[f"X_{split}"]["F"], 3, axis=1)[1]
        write = np.asarray(data[f"write_strength_{split}"], dtype=np.float32)
        if (
            gross.shape != write.shape
            or not np.isfinite(gross).all()
            or not np.isfinite(write).all()
            or np.any(gross < 0)
            or np.any(write < 0)
        ):
            raise ValueError("Gross response and write strength must be aligned and non-negative")
        denominator = gross + write
        relative = np.divide(
            gross,
            denominator,
            out=np.zeros_like(gross),
            where=denominator > np.finfo(np.float32).eps,
        )
        relative = np.clip(relative, 0.0, upper).astype(np.float32, copy=False)
        output[split] = {
            "relative_strength": relative,
            "D_OT+relative_strength": np.concatenate(
                [data[f"X_{split}"]["D_OT"], relative], axis=1
            ),
        }
        if set(output[split]) != set(RELATIVE_STRENGTH_SPECS) or not all(
            np.isfinite(values).all() for values in output[split].values()
        ):
            raise AssertionError("Relative-strength feature registry or values are invalid")
    return output


def log1p_strength_matrices(
    data: dict[str, Any],
) -> dict[str, dict[str, np.ndarray]]:
    """Apply elementwise ln(1+S) to the saved non-negative gross strength."""
    output = {}
    for split in ("train", "test"):
        gross = np.split(data[f"X_{split}"]["F"], 3, axis=1)[1]
        if not np.isfinite(gross).all() or np.any(gross < 0):
            raise ValueError("Gross strength must be finite and non-negative")
        transformed = np.log1p(gross).astype(np.float32, copy=False)
        output[split] = {
            "log1p_strength": transformed,
            "D_OT+log1p_strength": np.concatenate(
                [data[f"X_{split}"]["D_OT"], transformed], axis=1
            ),
        }
        if set(output[split]) != set(LOG1P_STRENGTH_SPECS) or not all(
            np.isfinite(values).all() for values in output[split].values()
        ):
            raise AssertionError("Log1p-strength feature registry or values are invalid")
    return output


def core_signal_matrices(data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Raw WRITE/gross/net ablations; U preserves H's non-distance block order."""
    matrices, audit = {}, {}
    for split in ("train", "test"):
        formal = data[f"X_{split}"]
        ae, gross, kappa = np.split(formal["F"], 3, axis=1)
        r_cos, ae_from_a = np.split(formal["A"], 2, axis=1)
        np.testing.assert_array_equal(ae, formal["B"])
        np.testing.assert_array_equal(ae, ae_from_a)
        write = np.asarray(data[f"write_strength_{split}"], dtype=np.float32)
        net = np.asarray(data[f"net_strength_{split}"], dtype=np.float32)
        for values in (write, gross, net, kappa):
            if values.shape != ae.shape or not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError("Core strengths must be finite, non-negative and layer-aligned")
        product = gross.astype(np.float64) * kappa.astype(np.float64)
        np.testing.assert_allclose(net, product, rtol=2e-6, atol=1e-7)
        error = np.abs(net - product)
        matrices[split] = {
            "AE": ae,
            "AE+I": np.concatenate([ae, write], axis=1),
            "AE+S": np.concatenate([ae, gross], axis=1),
            "AE+I+S": np.concatenate([ae, write, gross], axis=1),
            "AE+N": np.concatenate([ae, net], axis=1),
            "AE+S+kappa": formal["F"],
            "AE+S+N": np.concatenate([ae, gross, net], axis=1),
            "U": np.concatenate([r_cos, ae, gross, kappa], axis=1),
            "H_OT": formal["H_OT"],
            "H_JS": formal["H_JS"],
        }
        layers = ae.shape[1]
        for family in ("JS", "OT"):
            np.testing.assert_array_equal(
                matrices[split]["U"],
                np.concatenate([formal[f"H_{family}"][:, :2 * layers],
                                formal[f"H_{family}"][:, -2 * layers:]], axis=1),
            )
        if set(matrices[split]) != set(CORE_SPECS) or not all(
            np.isfinite(values).all() for values in matrices[split].values()
        ):
            raise AssertionError("Core feature registry or finite check failed")
        audit[split] = {
            "n_equals_s_kappa_max_absolute_error": float(error.max()),
            "n_equals_s_kappa_max_relative_error": float(
                (error / np.maximum(np.abs(net), 1e-12)).max()
            ),
            "zero_gross_entries": int((gross == 0).sum()),
            "ranges": {
                name: {"min": float(values.min()), "max": float(values.max())}
                for name, values in (("I", write), ("S", gross), ("N", net), ("kappa", kappa))
            },
        }
    return matrices, audit


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


def write_label_curve(
    model: str,
    data: dict[str, Any],
    train_values: np.ndarray,
    test_values: np.ndarray,
    root: Path,
    *,
    slug: str,
    ylabel: str,
    log_scale: bool,
) -> dict[str, Any]:
    """Write all-cohort mention-weighted median/IQR layer curves."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values_by_layer = np.concatenate([train_values, test_values])
    labels = np.concatenate([data["y_train"], data["y_test"]])
    if values_by_layer.ndim != 2 or len(values_by_layer) != len(labels):
        raise ValueError("Curve values must be a mention-aligned [mentions,layers] matrix")
    rows = []
    for label, name in ((0, "HALL"), (1, "REAL")):
        selected = values_by_layer[labels == label]
        for layer in range(values_by_layer.shape[1]):
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
                    f"raw_{slug}_real_auroc": float(
                        roc_auc_score(labels, values_by_layer[:, layer])
                    ),
                }
            )

    table_path = root / f"tables/{slug}_real_hall_curve.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    figure_path = root / f"figures/{slug}_real_hall_curve.png"
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
    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("Decoder layer")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{model}: {slug.replace('_', ' ')} by label")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    return {
        "cohort": "all formal mentions (train + test), descriptive only",
        "n_hall": int((labels == 0).sum()),
        "n_real": int((labels == 1).sum()),
        "layers": int(values_by_layer.shape[1]),
        "table": str(table_path.relative_to(ROOT)),
        "figure": str(figure_path.relative_to(ROOT)),
    }


def write_strength_curve(
    model: str,
    data: dict[str, Any],
    matrices: dict[str, dict[str, np.ndarray]],
    root: Path,
) -> dict[str, Any]:
    return write_label_curve(
        model,
        data,
        matrices["train"]["strength"],
        matrices["test"]["strength"],
        root,
        slug="strength",
        ylabel="gross strength (median; IQR, log scale)",
        log_scale=True,
    )


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


def run_union_topk_js_model(
    model: str, *, device: torch.device, resume: bool
) -> dict[str, Any]:
    """Retrain only the five JS groups with pairwise Top-32 union support."""
    data = _detector_matrices(model)
    gc.collect()
    matrices, alignment = union_topk_js_matrices(model, data)
    root = result_root(model)
    _progress, ensembles, summaries = train_matrices(
        model,
        data,
        matrices,
        UNION_JS_SPECS,
        device=device,
        resume=resume,
        progress_name="union_topk_js_feature_group_training_progress.pt",
        probe_directory="metrics/probes_union_topk_js_feature_groups",
    )
    formal_progress = torch.load(
        root / "metrics/detector_training_progress.pt",
        map_location="cpu",
        weights_only=False,
    )
    for spec in UNION_JS_SPECS:
        formal_probabilities = np.mean(
            [
                np.asarray(formal_progress["predictions"][spec][seed])
                for seed in SEEDS
            ],
            axis=0,
        )
        summaries[spec]["full_support_js_ensemble_auroc"] = float(
            roc_auc_score(data["y_test"], formal_probabilities)
        )
        summaries[spec]["union_topk_js_minus_full_js_auroc"] = float(
            summaries[spec]["ensemble_auroc"]
            - summaries[spec]["full_support_js_ensemble_auroc"]
        )

    result = {
        "model": model,
        "status": "EXPLORATORY_UNION_TOP32_JS",
        "replacement": {
            "old": "JS on every visual token",
            "new": (
                "pairwise JS after selecting Top-32 from each marginal, taking "
                "their union, and renormalizing both marginals on that union"
            ),
            "scope": "replace only D_EW/D_WF/D_EF in C_JS/D_JS/E_JS/G_JS/H_JS",
            "unchanged": ["AE", "R_cos", "S", "kappa", "OT groups"],
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
            "bootstrap": "NOT_RUN_BY_USER_REQUEST",
        },
        "counts": data["counts"],
        "summaries": summaries,
    }
    atomic_json_save(
        _json_ready(result),
        root / "metrics/union_topk_js_feature_group_results.json",
    )
    table_path = root / "tables/union_topk_js_feature_group_metrics.csv"
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "feature_set",
            "dimensions",
            "full_support_js_ensemble_auroc",
            "union_topk_js_ensemble_auroc",
            "union_topk_js_minus_full_js_auroc",
            "mean_hall_aupr",
            "mean_hall_precision",
            "mean_hall_recall",
            "mean_hall_f1",
            *(f"seed{seed}_auroc" for seed in SEEDS),
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for spec in UNION_JS_SPECS:
            summary = summaries[spec]
            writer.writerow(
                {
                    "feature_set": spec,
                    "dimensions": summary["dimensions"],
                    "full_support_js_ensemble_auroc": summary[
                        "full_support_js_ensemble_auroc"
                    ],
                    "union_topk_js_ensemble_auroc": summary["ensemble_auroc"],
                    "union_topk_js_minus_full_js_auroc": summary[
                        "union_topk_js_minus_full_js_auroc"
                    ],
                    "mean_hall_aupr": summary["mean_hall_aupr"],
                    "mean_hall_precision": summary["mean_hall_precision"],
                    "mean_hall_recall": summary["mean_hall_recall"],
                    "mean_hall_f1": summary["mean_hall_f1"],
                    **{
                        f"seed{seed}_auroc": summary["per_seed_auroc"][str(seed)]
                        for seed in SEEDS
                    },
                }
            )
    return result


def run_net_strength_model(
    model: str, *, device: torch.device, resume: bool
) -> dict[str, Any]:
    """Train S_net and D_OT+gross-strength, and write S_net/kappa curves."""
    data = _detector_matrices(model)
    matrices = net_strength_matrices(data)
    signals = signal_matrices(data)
    root = result_root(model)
    curves = {
        "net_strength": write_label_curve(
            model,
            data,
            matrices["train"]["net_strength"],
            matrices["test"]["net_strength"],
            root,
            slug="net_strength",
            ylabel="net strength (median; IQR, log scale)",
            log_scale=True,
        ),
        "kappa": write_label_curve(
            model,
            data,
            signals["train"]["kappa"],
            signals["test"]["kappa"],
            root,
            slug="kappa",
            ylabel="kappa (median; IQR)",
            log_scale=False,
        ),
    }
    _progress, _ensembles, summaries = train_matrices(
        model,
        data,
        matrices,
        NET_SPECS,
        device=device,
        resume=resume,
        progress_name="net_strength_feature_training_progress.pt",
        probe_directory="metrics/probes_net_strength_features",
    )
    formal_progress = torch.load(
        root / "metrics/detector_training_progress.pt",
        map_location="cpu",
        weights_only=False,
    )
    signal_progress = torch.load(
        root / "metrics/signal_ablation_training_progress.pt",
        map_location="cpu",
        weights_only=False,
    )
    references = {
        "net_strength": (signal_progress, "strength"),
        "D_OT+strength": (formal_progress, "D_OT"),
    }
    for spec, (progress, reference) in references.items():
        probabilities = np.mean(
            [np.asarray(progress["predictions"][reference][seed]) for seed in SEEDS],
            axis=0,
        )
        reference_auroc = float(roc_auc_score(data["y_test"], probabilities))
        summaries[spec].update(
            reference_feature=reference,
            reference_ensemble_auroc=reference_auroc,
            ensemble_auroc_minus_reference=float(
                summaries[spec]["ensemble_auroc"] - reference_auroc
            ),
        )

    result = {
        "model": model,
        "status": "EXPLORATORY_NET_STRENGTH_AND_D_OT_PLUS_STRENGTH",
        "feature_definitions": {
            "net_strength": "all-layer ||sum_m e_m||_2",
            "D_OT+strength": "AE + D_WF^OT + gross_strength",
            "kappa": "net_strength / gross_strength; curve only",
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
            "bootstrap": "NOT_RUN_BY_USER_REQUEST",
        },
        "counts": data["counts"],
        "curves": curves,
        "summaries": summaries,
    }
    atomic_json_save(
        _json_ready(result), root / "metrics/net_strength_feature_results.json"
    )
    table_path = root / "tables/net_strength_feature_metrics.csv"
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "feature_set",
            "dimensions",
            "ensemble_auroc",
            "reference_feature",
            "reference_ensemble_auroc",
            "ensemble_auroc_minus_reference",
            "mean_hall_aupr",
            "mean_hall_precision",
            "mean_hall_recall",
            "mean_hall_f1",
            *(f"seed{seed}_auroc" for seed in SEEDS),
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for spec in NET_SPECS:
            summary = summaries[spec]
            writer.writerow(
                {
                    "feature_set": spec,
                    **{name: summary[name] for name in fields[1:-3]},
                    **{
                        f"seed{seed}_auroc": summary["per_seed_auroc"][str(seed)]
                        for seed in SEEDS
                    },
                }
            )
    return result


def run_bounded_strength_model(
    model: str, *, device: torch.device, resume: bool
) -> dict[str, Any]:
    """Train median-calibrated bounded strength alone and with formal D_OT."""
    data = _detector_matrices(model)
    matrices, tau = bounded_strength_matrices(data)
    root = result_root(model)
    curve = write_label_curve(
        model,
        data,
        matrices["train"]["bounded_strength"],
        matrices["test"]["bounded_strength"],
        root,
        slug="bounded_strength",
        ylabel="bounded gross strength (median; IQR)",
        log_scale=False,
    )
    _progress, _ensembles, summaries = train_matrices(
        model,
        data,
        matrices,
        BOUNDED_STRENGTH_SPECS,
        device=device,
        resume=resume,
        progress_name="bounded_strength_feature_training_progress.pt",
        probe_directory="metrics/probes_bounded_strength_features",
    )
    references = {
        "bounded_strength": (
            "signal_ablation_training_progress.pt",
            "strength",
        ),
        "D_OT+bounded_strength": (
            "net_strength_feature_training_progress.pt",
            "D_OT+strength",
        ),
    }
    for spec, (progress_name, reference) in references.items():
        progress = torch.load(
            root / f"metrics/{progress_name}", map_location="cpu", weights_only=False
        )
        probabilities = np.mean(
            [np.asarray(progress["predictions"][reference][seed]) for seed in SEEDS],
            axis=0,
        )
        reference_auroc = float(roc_auc_score(data["y_test"], probabilities))
        summaries[spec].update(
            raw_reference=reference,
            raw_reference_ensemble_auroc=reference_auroc,
            bounded_minus_raw_auroc=float(
                summaries[spec]["ensemble_auroc"] - reference_auroc
            ),
        )

    formal_progress = torch.load(
        root / "metrics/detector_training_progress.pt",
        map_location="cpu",
        weights_only=False,
    )
    d_ot_probabilities = np.mean(
        [np.asarray(formal_progress["predictions"]["D_OT"][seed]) for seed in SEEDS],
        axis=0,
    )
    summaries["D_OT+bounded_strength"].update(
        d_ot_ensemble_auroc=float(roc_auc_score(data["y_test"], d_ot_probabilities))
    )
    summaries["D_OT+bounded_strength"]["bounded_combo_minus_d_ot_auroc"] = float(
        summaries["D_OT+bounded_strength"]["ensemble_auroc"]
        - summaries["D_OT+bounded_strength"]["d_ot_ensemble_auroc"]
    )

    calibration_path = root / "tables/bounded_strength_calibration.csv"
    raw_train = np.split(data["X_train"]["F"], 3, axis=1)[1]
    raw_test = np.split(data["X_test"]["F"], 3, axis=1)[1]
    with calibration_path.open("w", newline="", encoding="utf-8") as handle:
        fields = (
            "layer", "tau_train_median", "train_raw_min", "train_raw_max",
            "test_raw_min", "test_raw_max", "train_bounded_min",
            "train_bounded_max", "test_bounded_min", "test_bounded_max",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for layer in range(len(tau)):
            writer.writerow({
                "layer": layer + 1,
                "tau_train_median": float(tau[layer]),
                "train_raw_min": float(raw_train[:, layer].min()),
                "train_raw_max": float(raw_train[:, layer].max()),
                "test_raw_min": float(raw_test[:, layer].min()),
                "test_raw_max": float(raw_test[:, layer].max()),
                "train_bounded_min": float(
                    matrices["train"]["bounded_strength"][:, layer].min()
                ),
                "train_bounded_max": float(
                    matrices["train"]["bounded_strength"][:, layer].max()
                ),
                "test_bounded_min": float(
                    matrices["test"]["bounded_strength"][:, layer].min()
                ),
                "test_bounded_max": float(
                    matrices["test"]["bounded_strength"][:, layer].max()
                ),
            })

    result = {
        "model": model,
        "status": "EXPLORATORY_BOUNDED_STRENGTH",
        "feature_definitions": {
            "bounded_strength": "S_raw / (S_raw + train-layer-median(S_raw) + eps)",
            "D_OT+bounded_strength": "AE + D_WF^OT + bounded_strength",
        },
        "protocol": {
            "split": "same fixed image 8:2 split as formal detector",
            "calibration": "per-model, per-layer median from training mentions only",
            "seeds": list(SEEDS),
            "hidden_sizes": [128, 64, 32],
            "dropout": 0.3,
            "drop_last": False,
            "batch_size": 256,
            "epochs_max": 100,
            "standardization": "none beyond bounded-strength transform",
            "checkpoint": "minimum_train_loss",
            "threshold": "train_REAL_F1",
            "bootstrap": "NOT_RUN_BY_USER_REQUEST",
        },
        "counts": data["counts"],
        "tau_train_median": tau.tolist(),
        "curve": curve,
        "calibration_table": str(calibration_path.relative_to(ROOT)),
        "summaries": summaries,
    }
    atomic_json_save(
        _json_ready(result), root / "metrics/bounded_strength_feature_results.json"
    )
    table_path = root / "tables/bounded_strength_feature_metrics.csv"
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "feature_set", "dimensions", "ensemble_auroc", "raw_reference",
            "raw_reference_ensemble_auroc", "bounded_minus_raw_auroc",
            "d_ot_ensemble_auroc", "bounded_combo_minus_d_ot_auroc",
            "mean_hall_aupr", "mean_hall_precision", "mean_hall_recall",
            "mean_hall_f1", *(f"seed{seed}_auroc" for seed in SEEDS),
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for spec in BOUNDED_STRENGTH_SPECS:
            summary = summaries[spec]
            writer.writerow({
                "feature_set": spec,
                **{name: summary.get(name, "") for name in fields[1:-3]},
                **{
                    f"seed{seed}_auroc": summary["per_seed_auroc"][str(seed)]
                    for seed in SEEDS
                },
            })
    return result


def run_relative_strength_model(
    model: str, *, device: torch.device, resume: bool
) -> dict[str, Any]:
    """Train extraction-native R/(R+W) alone and with formal D_OT."""
    data = _detector_matrices(model)
    matrices = relative_strength_matrices(data)
    root = result_root(model)
    curve = write_label_curve(
        model,
        data,
        matrices["train"]["relative_strength"],
        matrices["test"]["relative_strength"],
        root,
        slug="relative_strength",
        ylabel="relative FFN response R/(R+W) (median; IQR)",
        log_scale=False,
    )
    _progress, _ensembles, summaries = train_matrices(
        model,
        data,
        matrices,
        RELATIVE_STRENGTH_SPECS,
        device=device,
        resume=resume,
        progress_name="relative_strength_feature_training_progress.pt",
        probe_directory="metrics/probes_relative_strength_features",
    )
    references = {
        "relative_strength": ("signal_ablation_training_progress.pt", "strength"),
        "D_OT+relative_strength": (
            "net_strength_feature_training_progress.pt",
            "D_OT+strength",
        ),
    }
    for spec, (progress_name, reference) in references.items():
        progress = torch.load(
            root / f"metrics/{progress_name}", map_location="cpu", weights_only=False
        )
        probabilities = np.mean(
            [np.asarray(progress["predictions"][reference][seed]) for seed in SEEDS],
            axis=0,
        )
        reference_auroc = float(roc_auc_score(data["y_test"], probabilities))
        summaries[spec].update(
            raw_reference=reference,
            raw_reference_ensemble_auroc=reference_auroc,
            relative_minus_raw_auroc=float(
                summaries[spec]["ensemble_auroc"] - reference_auroc
            ),
        )

    formal_progress = torch.load(
        root / "metrics/detector_training_progress.pt",
        map_location="cpu",
        weights_only=False,
    )
    d_ot_probabilities = np.mean(
        [np.asarray(formal_progress["predictions"]["D_OT"][seed]) for seed in SEEDS],
        axis=0,
    )
    summaries["D_OT+relative_strength"].update(
        d_ot_ensemble_auroc=float(roc_auc_score(data["y_test"], d_ot_probabilities))
    )
    summaries["D_OT+relative_strength"]["relative_combo_minus_d_ot_auroc"] = float(
        summaries["D_OT+relative_strength"]["ensemble_auroc"]
        - summaries["D_OT+relative_strength"]["d_ot_ensemble_auroc"]
    )

    ranges = {}
    for split in ("train", "test"):
        values = matrices[split]["relative_strength"]
        ranges[split] = {"min": float(values.min()), "max": float(values.max())}
    result = {
        "model": model,
        "status": "EXPLORATORY_EXTRACTION_NATIVE_RELATIVE_STRENGTH",
        "feature_definitions": {
            "R": "sum_m ||e_m||_2 (gross FFN path response)",
            "W": "sum_m ||a_m||_2 (gross visual write energy)",
            "relative_strength": "R / (R + W), zero when R+W is degenerate",
            "D_OT+relative_strength": "AE + D_WF^OT + relative_strength",
        },
        "protocol": {
            "split": "same fixed image 8:2 split as formal detector",
            "calibration": "none; sample-local extraction-time R and W only",
            "seeds": list(SEEDS),
            "hidden_sizes": [128, 64, 32],
            "dropout": 0.3,
            "drop_last": False,
            "batch_size": 256,
            "epochs_max": 100,
            "standardization": "none",
            "checkpoint": "minimum_train_loss",
            "threshold": "train_REAL_F1",
            "bootstrap": "NOT_RUN_BY_USER_REQUEST",
        },
        "counts": data["counts"],
        "ranges": ranges,
        "curve": curve,
        "summaries": summaries,
    }
    atomic_json_save(
        _json_ready(result), root / "metrics/relative_strength_feature_results.json"
    )
    table_path = root / "tables/relative_strength_feature_metrics.csv"
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "feature_set", "dimensions", "ensemble_auroc", "raw_reference",
            "raw_reference_ensemble_auroc", "relative_minus_raw_auroc",
            "d_ot_ensemble_auroc", "relative_combo_minus_d_ot_auroc",
            "mean_hall_aupr", "mean_hall_precision", "mean_hall_recall",
            "mean_hall_f1", *(f"seed{seed}_auroc" for seed in SEEDS),
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for spec in RELATIVE_STRENGTH_SPECS:
            summary = summaries[spec]
            writer.writerow({
                "feature_set": spec,
                **{name: summary.get(name, "") for name in fields[1:-3]},
                **{
                    f"seed{seed}_auroc": summary["per_seed_auroc"][str(seed)]
                    for seed in SEEDS
                },
            })
    return result


def run_log1p_strength_model(
    model: str, *, device: torch.device, resume: bool
) -> dict[str, Any]:
    """Train plain ln(1+S) alone and with formal D_OT."""
    data = _detector_matrices(model)
    matrices = log1p_strength_matrices(data)
    root = result_root(model)
    curve = write_label_curve(
        model,
        data,
        matrices["train"]["log1p_strength"],
        matrices["test"]["log1p_strength"],
        root,
        slug="log1p_strength",
        ylabel="log1p gross strength (median; IQR)",
        log_scale=False,
    )
    _progress, _ensembles, summaries = train_matrices(
        model,
        data,
        matrices,
        LOG1P_STRENGTH_SPECS,
        device=device,
        resume=resume,
        progress_name="log1p_strength_feature_training_progress.pt",
        probe_directory="metrics/probes_log1p_strength_features",
    )
    references = {
        "log1p_strength": ("strength", "bounded_strength"),
        "D_OT+log1p_strength": ("D_OT+strength", "D_OT+bounded_strength"),
    }
    raw_progress = torch.load(
        root / "metrics/signal_ablation_training_progress.pt",
        map_location="cpu",
        weights_only=False,
    )
    raw_combo_progress = torch.load(
        root / "metrics/net_strength_feature_training_progress.pt",
        map_location="cpu",
        weights_only=False,
    )
    bounded_progress = torch.load(
        root / "metrics/bounded_strength_feature_training_progress.pt",
        map_location="cpu",
        weights_only=False,
    )
    for spec, (raw_reference, bounded_reference) in references.items():
        source = raw_progress if spec == "log1p_strength" else raw_combo_progress
        raw_probabilities = np.mean(
            [np.asarray(source["predictions"][raw_reference][seed]) for seed in SEEDS],
            axis=0,
        )
        bounded_probabilities = np.mean(
            [
                np.asarray(bounded_progress["predictions"][bounded_reference][seed])
                for seed in SEEDS
            ],
            axis=0,
        )
        raw_auroc = float(roc_auc_score(data["y_test"], raw_probabilities))
        bounded_auroc = float(roc_auc_score(data["y_test"], bounded_probabilities))
        summaries[spec].update(
            raw_reference=raw_reference,
            raw_reference_ensemble_auroc=raw_auroc,
            log1p_minus_raw_auroc=float(summaries[spec]["ensemble_auroc"] - raw_auroc),
            bounded_reference=bounded_reference,
            bounded_reference_ensemble_auroc=bounded_auroc,
            log1p_minus_bounded_auroc=float(
                summaries[spec]["ensemble_auroc"] - bounded_auroc
            ),
        )

    formal_progress = torch.load(
        root / "metrics/detector_training_progress.pt",
        map_location="cpu",
        weights_only=False,
    )
    d_ot_probabilities = np.mean(
        [np.asarray(formal_progress["predictions"]["D_OT"][seed]) for seed in SEEDS],
        axis=0,
    )
    summaries["D_OT+log1p_strength"].update(
        d_ot_ensemble_auroc=float(roc_auc_score(data["y_test"], d_ot_probabilities))
    )
    summaries["D_OT+log1p_strength"]["log1p_combo_minus_d_ot_auroc"] = float(
        summaries["D_OT+log1p_strength"]["ensemble_auroc"]
        - summaries["D_OT+log1p_strength"]["d_ot_ensemble_auroc"]
    )

    ranges = {
        split: {
            "min": float(matrices[split]["log1p_strength"].min()),
            "max": float(matrices[split]["log1p_strength"].max()),
        }
        for split in ("train", "test")
    }
    result = {
        "model": model,
        "status": "EXPLORATORY_PLAIN_LOG1P_STRENGTH",
        "feature_definitions": {
            "log1p_strength": "natural-log ln(1 + gross_strength), no tau or standardization",
            "D_OT+log1p_strength": "AE + D_WF^OT + log1p_strength",
        },
        "protocol": {
            "split": "same fixed image 8:2 split as formal detector",
            "calibration": "none; elementwise natural log1p only",
            "seeds": list(SEEDS),
            "hidden_sizes": [128, 64, 32],
            "dropout": 0.3,
            "drop_last": False,
            "batch_size": 256,
            "epochs_max": 100,
            "standardization": "none",
            "checkpoint": "minimum_train_loss",
            "threshold": "train_REAL_F1",
            "bootstrap": "NOT_RUN_BY_USER_REQUEST",
        },
        "counts": data["counts"],
        "ranges": ranges,
        "curve": curve,
        "summaries": summaries,
    }
    atomic_json_save(
        _json_ready(result), root / "metrics/log1p_strength_feature_results.json"
    )
    table_path = root / "tables/log1p_strength_feature_metrics.csv"
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "feature_set", "dimensions", "ensemble_auroc", "raw_reference",
            "raw_reference_ensemble_auroc", "log1p_minus_raw_auroc",
            "bounded_reference", "bounded_reference_ensemble_auroc",
            "log1p_minus_bounded_auroc", "d_ot_ensemble_auroc",
            "log1p_combo_minus_d_ot_auroc", "mean_hall_aupr",
            "mean_hall_precision", "mean_hall_recall", "mean_hall_f1",
            *(f"seed{seed}_auroc" for seed in SEEDS),
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for spec in LOG1P_STRENGTH_SPECS:
            summary = summaries[spec]
            writer.writerow({
                "feature_set": spec,
                **{name: summary.get(name, "") for name in fields[1:-3]},
                **{
                    f"seed{seed}_auroc": summary["per_seed_auroc"][str(seed)]
                    for seed in SEEDS
                },
            })
    return result


def run_core_signal_model(model: str, *, device: torch.device, resume: bool) -> dict[str, Any]:
    """Train six missing groups and reuse four exact formal three-seed references."""
    started = time.monotonic()
    torch.set_num_threads(1)
    data = _detector_matrices(model)
    matrices, identity = core_signal_matrices(data)
    root = result_root(model)
    split_path = ROOT / "outputs" / model / EXPERIMENT / "image_splits.json"
    split_ids = json.loads(split_path.read_text(encoding="utf-8"))
    assert not (set(split_ids["train"]) & set(split_ids["test"]))
    digest = hashlib.sha256(split_path.read_bytes())
    for split in ("train", "test"):
        digest.update(json.dumps(data[f"{split}_target_keys"]).encode())
        digest.update(data[f"y_{split}"].tobytes())
        for spec in CORE_SPECS:
            values = matrices[split][spec]
            digest.update(f"{split}:{spec}:{values.shape}:{values.dtype}".encode())
            digest.update(values.tobytes())
    signature = digest.hexdigest()
    progress_name = "core_signal_feature_training_progress.pt"
    progress_path = root / "metrics" / progress_name
    if progress_path.exists():
        if not resume:
            raise FileExistsError("Core ablation already exists; use --resume to preserve it")
        progress = torch.load(progress_path, map_location="cpu", weights_only=False)
        if progress.get("cohort_feature_sha256") != signature:
            raise AssertionError("Resume cohort/features differ from the saved core ablation")
    else:
        progress = {"metrics": {}, "predictions": {}, "cohort_feature_sha256": signature}
    formal = torch.load(root / "metrics/detector_training_progress.pt",
                        map_location="cpu", weights_only=False)
    sources = {}
    for spec in CORE_SPECS:
        reference = CORE_REFERENCES.get(spec)
        sources[spec] = {
            "kind": "REUSED_FORMAL" if reference else "NEW_TRAINING",
            "probe_directory": f"metrics/probes/{reference}" if reference else
                               f"metrics/probes_core_signal_features/{spec}",
            "source_feature": reference or spec,
        }
        if reference is None:
            continue
        for seed in SEEDS:
            config_path = root / f"metrics/probes/{reference}/seed{seed}/config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            expected = _json_ready(asdict(TorchProbeConfig(seed=seed)))
            assert all(config[key] == value for key, value in expected.items()), config_path
            probabilities = np.asarray(formal["predictions"][reference][seed])
            assert probabilities.shape == data["y_test"].shape and np.isfinite(probabilities).all()
            assert abs(roc_auc_score(data["y_test"], probabilities)
                       - formal["metrics"][reference][seed]["auc"]) < 1e-10
            if spec in progress["predictions"]:
                np.testing.assert_array_equal(progress["predictions"][spec][seed], probabilities)
        progress["metrics"][spec] = formal["metrics"][reference]
        progress["predictions"][spec] = formal["predictions"][reference]
    if not progress_path.exists():
        atomic_torch_save(progress, progress_path)
    completed_path = root / "metrics/core_signal_feature_results.json"
    if completed_path.exists():
        completed = json.loads(completed_path.read_text(encoding="utf-8"))
        assert completed["cohort_feature_sha256"] == signature
        for spec in CORE_SPECS:
            assert set(progress["predictions"][spec]) == set(SEEDS)
            for seed in SEEDS:
                assert (root / sources[spec]["probe_directory"] / f"seed{seed}/model.pt").is_file()
        assert all((root / f"tables/core_signal_feature_{name}.csv").is_file()
                   for name in ("metrics", "seed_metrics", "comparisons"))
        print(f"[{model}] COMPLETE: verified cohort; no training or artifact rewrite", flush=True)
        return completed
    print(f"[{model}] N=S*kappa PASS; cohort={signature}; training six new groups", flush=True)
    progress, ensembles, summaries = train_matrices(
        model, data, matrices, CORE_SPECS, device=device, resume=True,
        progress_name=progress_name, probe_directory="metrics/probes_core_signal_features",
    )
    for spec, summary in summaries.items():
        seed_aurocs = list(summary["per_seed_auroc"].values())
        summary.update(
            mean_seed_auroc=float(np.mean(seed_aurocs)),
            std_seed_auroc=float(np.std(seed_aurocs)),
            ensemble_real_aupr=float(average_precision_score(data["y_test"], ensembles[spec])),
            ensemble_hall_aupr=float(average_precision_score(1 - data["y_test"], 1 - ensembles[spec])),
        )
    comparisons = {}
    for new, base in (
        ("AE+I", "AE"), ("AE+S", "AE"), ("AE+S", "AE+I"),
        ("AE+I+S", "AE+I"), ("AE+I+S", "AE+S"), ("AE+N", "AE+S"),
        ("AE+N", "AE+S+kappa"), ("AE+S+kappa", "AE+S"),
        ("AE+S+N", "AE+S"), ("AE+S+kappa", "AE+S+N"),
        ("U", "AE+S+kappa"), ("H_OT", "U"), ("H_JS", "U"),
    ):
        comparisons[f"{new}__minus__{base}"] = {
            "new": new, "baseline": base,
            "ensemble_auroc_delta": summaries[new]["ensemble_auroc"] - summaries[base]["ensemble_auroc"],
            "ensemble_hall_aupr_delta": summaries[new]["ensemble_hall_aupr"] - summaries[base]["ensemble_hall_aupr"],
            "per_seed_auroc_delta": {
                str(seed): summaries[new]["per_seed_auroc"][str(seed)]
                           - summaries[base]["per_seed_auroc"][str(seed)] for seed in SEEDS
            },
        }
    result = {
        "model": model, "status": "EXPLORATORY_CORE_SIGNAL_ABLATION_COMPLETE",
        "feature_definitions": {
            "AE": "sum_m attention_m * hpre_raw_logit_gauss_gate_m",
            "I": "sum_m ||a_m||_2 = sum_m saved write_mag_m (gross WRITE, not ||sum_m a_m||)",
            "S": "sum_m ||e_m||_2 = saved gross_strength",
            "N": "saved net_strength = ||G(z)-G(z0)||_2; finite quadrature need not sum exactly to this",
            "kappa": "saved N/S (zero on degenerate gross); no clipping/recalibration",
            "R_cos": "formal old_hpre_cos source / hpre_raw_logit_gauss target / sqrt_matched_state OT",
            "U": "concatenate [R_cos, AE, S, kappa]; exactly H with all three distances removed",
            "H_JS": "formal full-visual-support JS, not Union-Top32 JS",
        },
        "protocol": {**_json_ready(asdict(TorchProbeConfig())), "seeds": list(SEEDS),
                     "standardization": "none", "strength_transform": "none; raw I/S/N",
                     "class_weighting_or_resampling": "none", "bootstrap": "NOT_RUN_BY_USER_REQUEST",
                     "inference_boundary": "No VLM forward; MLP training only; same test cohort, exploratory"},
        "counts": data["counts"], "identity_audit": identity,
        "cohort_feature_sha256": signature, "sources": sources,
        "summaries": summaries, "comparisons": comparisons,
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_json_save(_json_ready(result), root / "metrics/core_signal_feature_results.json")
    rows = [{"feature_set": spec, **{k: v for k, v in summaries[spec].items() if k != "per_seed_auroc"},
             **{f"seed{seed}_auroc": summaries[spec]["per_seed_auroc"][str(seed)] for seed in SEEDS}}
            for spec in CORE_SPECS]
    seed_rows = []
    for spec in CORE_SPECS:
        for seed in SEEDS:
            for rule, report in progress["metrics"][spec][seed]["threshold_reports"].items():
                metrics = report["test_metrics"]
                seed_rows.append({
                    "feature_set": spec, "seed": seed, "threshold_rule": rule,
                    "decision_threshold": report["threshold"], "auroc": metrics["auc"],
                    **{f"{label}_{name}": metrics[key][name]
                       for label, key in (("real", "real_positive"), ("hall", "hallucination_positive"))
                       for name in ("aupr", "precision", "recall", "f1")},
                })
    comparison_rows = [{k: v for k, v in row.items() if k != "per_seed_auroc_delta"}
                       for row in comparisons.values()]
    for name, records in (("metrics", rows), ("seed_metrics", seed_rows), ("comparisons", comparison_rows)):
        with (root / f"tables/core_signal_feature_{name}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
    return result


def main() -> None:
    args = parse_args()
    if args.study in {"signals", "old_ev"} and args.bootstrap_resamples <= 0:
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
        elif args.study in {"old_ev", "ae_cosine"}:
            result = run_old_ev_model(
                model,
                device=torch.device(args.training_device),
                bootstrap_resamples=args.bootstrap_resamples,
                resume=args.resume,
                replacement=args.study,
            )
        elif args.study == "union_topk_js":
            result = run_union_topk_js_model(
                model,
                device=torch.device(args.training_device),
                resume=args.resume,
            )
        elif args.study == "net_strength":
            result = run_net_strength_model(
                model,
                device=torch.device(args.training_device),
                resume=args.resume,
            )
        elif args.study == "bounded_strength":
            result = run_bounded_strength_model(
                model,
                device=torch.device(args.training_device),
                resume=args.resume,
            )
        elif args.study == "relative_strength":
            result = run_relative_strength_model(
                model,
                device=torch.device(args.training_device),
                resume=args.resume,
            )
        elif args.study == "core_signals":
            result = run_core_signal_model(
                model, device=torch.device(args.training_device), resume=args.resume,
            )
        else:
            result = run_log1p_strength_model(
                model,
                device=torch.device(args.training_device),
                resume=args.resume,
            )
        print(_json_ready(result["summaries"]), flush=True)


if __name__ == "__main__":
    main()
