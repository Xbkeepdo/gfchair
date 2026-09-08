#!/usr/bin/env python3
"""Compare full AE with Top32(T) AE using saved features and the existing MLP."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.tc_fvpa_artifacts import atomic_json_save, atomic_torch_save, sha256_file
from scripts.analyze_ffn_visual_source_signal_ablation import (
    core_signal_matrices, net_strength_matrices, swap_ae_for_old_ev,
    train_matrices, write_label_curve,
)
from scripts.analyze_ffn_visual_source_study import (
    SEEDS, _detector_matrices, _json_ready, _load_full_model,
)
from scripts.run_ffn_visual_source_attribution import EXPERIMENT, result_root
from scripts.train_torch_probe_feature_sets import TorchProbeConfig

MODELS = ("qwen2_5_vl_7b", "llava_1_5_7b", "qwen3_vl_8b", "internvl_2_5_8b")
EXTRA = ("AE+I", "AE+S", "AE+I+S", "AE+N", "AE+S+N", "U")
SLUG = "top32_ae"


def topk_ae(strength, distribution, k: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """AE_k = AE * sum_{TopK(T)} T, without renormalizing the selected support."""
    ae = np.asarray(strength, dtype=np.float64)
    p = np.asarray(distribution, dtype=np.float64)
    if (k < 1 or p.ndim != 2 or p.shape[1] == 0 or ae.shape != p.shape[:1]
            or not np.isfinite(p).all() or not np.isfinite(ae).all()
            or np.any(p < 0) or np.any(ae < 0)):
        raise ValueError("AE and T must be finite, non-negative and layer-aligned; k>0")
    np.testing.assert_allclose(p.sum(axis=1), 1, atol=2e-4, rtol=0)
    count = min(k, p.shape[1])
    # Retain saved T, including its FP32 normalization error; no new calibration.
    mass = np.partition(p, p.shape[1] - count, axis=1)[:, -count:].sum(axis=1)
    return (ae * mass).astype(np.float32), mass.astype(np.float32)


def ae_replacement_matrices(data, ae32):
    """Replace AE only: 13 formal groups plus six core groups and D_OT+S."""
    core, _ = core_signal_matrices(data)
    net = net_strength_matrices(data)
    originals, replaced = {}, {}
    for split in ("train", "test"):
        originals[split] = {
            **data[f"X_{split}"], **{key: core[split][key] for key in EXTRA},
            "D_OT+strength": net[split]["D_OT+strength"],
        }
        assert len(originals[split]) == 20
        x = np.asarray(ae32[split], dtype=np.float32)
        assert x.shape == data[f"X_{split}"]["B"].shape
        assert np.isfinite(x).all() and np.all(x >= 0)
        assert np.all(x <= data[f"X_{split}"]["B"] + 2e-4)
        replaced[split] = swap_ae_for_old_ev(
            {key: value for key, value in originals[split].items() if key != "U"}, x
        )
        replaced[split]["U"] = swap_ae_for_old_ev({"A": originals[split]["U"]}, x)["A"]
        layers = x.shape[1]
        for key, value in originals[split].items():
            start = layers if key in {"A", "U"} or key.startswith("H_") else 0
            np.testing.assert_array_equal(value[:, start:start + layers], data[f"X_{split}"]["B"])
            mask = np.ones(value.shape[1], dtype=bool)
            mask[start:start + layers] = False
            np.testing.assert_array_equal(replaced[split][key][:, mask], value[:, mask])
    return originals, replaced


def checked_progress(path: Path, signature: str):
    if path.exists():
        progress = torch.load(path, map_location="cpu", weights_only=False)
        if progress.get("cohort_feature_sha256") != signature:
            raise AssertionError("Resume cohort/features do not match Top32 AE experiment")
        return progress
    return {"metrics": {}, "predictions": {}, "cohort_feature_sha256": signature}


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_model(model: str, device: torch.device):
    started = time.monotonic()
    torch.set_num_threads(1)
    data = _detector_matrices(model)
    positions, _, _ = _load_full_model(model)
    lookup = {str(p["target_key"]): topk_ae(p["ae_strength"], p["attention_evidence"])
              for p in positions}
    assert set(lookup) == set(data["train_target_keys"]) | set(data["test_target_keys"])
    del positions
    ae32 = {split: np.stack([lookup[key][0] for key in data[f"{split}_target_keys"]])
            for split in ("train", "test")}
    mass = {split: np.stack([lookup[key][1] for key in data[f"{split}_target_keys"]])
            for split in ("train", "test")}
    originals, matrices = ae_replacement_matrices(data, ae32)
    specs = tuple(originals["train"])
    root = result_root(model)
    split_path = ROOT / "outputs" / model / EXPERIMENT / "image_splits.json"
    split_ids = json.loads(split_path.read_text())
    assert len(split_ids["train"]) == 3200 and len(split_ids["test"]) == 800
    assert not (set(split_ids["train"]) & set(split_ids["test"]))
    digest = hashlib.sha256(split_path.read_bytes())
    for split in ("train", "test"):
        digest.update(json.dumps(data[f"{split}_target_keys"]).encode())
        digest.update(data[f"y_{split}"].tobytes())
        for spec in specs:
            value = matrices[split][spec]
            digest.update(f"{split}:{spec}:{value.shape}:{value.dtype}".encode())
            digest.update(value.tobytes())
    signature = digest.hexdigest()
    progress_path = root / f"metrics/{SLUG}_feature_training_progress.pt"
    progress = checked_progress(progress_path, signature)
    result_path = root / f"metrics/{SLUG}_feature_results.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        assert result["cohort_feature_sha256"] == signature
        for spec in specs:
            assert set(progress["predictions"][spec]) == set(SEEDS)
        for path, checksum in result["artifact_checksums"].items():
            assert sha256_file(ROOT / path) == checksum, path
        print(f"[{model}] COMPLETE: resume verified; no writes/training", flush=True)
        return result
    if not progress_path.exists():
        atomic_torch_save(progress, progress_path)

    reference_sources = {key: ("detector_training_progress.pt", key, f"metrics/probes/{key}")
                         for key in data["specs"]}
    reference_sources.update({key: ("core_signal_feature_training_progress.pt", key,
                                   f"metrics/probes_core_signal_features/{key}") for key in EXTRA})
    reference_sources["D_OT+strength"] = ("net_strength_feature_training_progress.pt",
                                           "D_OT+strength", "metrics/probes_net_strength_features/D_OT+strength")
    references = {name: torch.load(root / "metrics" / name, map_location="cpu", weights_only=False)
                  for name in {value[0] for value in reference_sources.values()}}
    for spec, (filename, key, directory) in reference_sources.items():
        for seed in SEEDS:
            config = json.loads((root / directory / f"seed{seed}/config.json").read_text())
            expected = _json_ready(asdict(TorchProbeConfig(seed=seed)))
            assert all(config[k] == v for k, v in expected.items())
            p = np.asarray(references[filename]["predictions"][key][seed])
            assert p.shape == data["y_test"].shape and np.isfinite(p).all()
            assert abs(roc_auc_score(data["y_test"], p) - references[filename]["metrics"][key][seed]["auc"]) < 1e-10

    curves = {}
    for slug, values in (("ae", {s: data[f"X_{s}"]["B"] for s in ("train", "test")}),
                         ("ae_top32", ae32)):
        curves[slug] = write_label_curve(model, data, values["train"], values["test"], root,
                                        slug=slug, ylabel=f"{slug} (median; IQR)", log_scale=False)
    print(f"[{model}] Top32 AE ready; 20 groups x 3 seeds; no VLM or bootstrap", flush=True)
    progress, ensembles, summaries = train_matrices(
        model, data, matrices, specs, device=device, resume=True,
        progress_name=progress_path.name, probe_directory=f"metrics/probes_{SLUG}_features",
    )
    rows, seed_rows = [], []
    for spec in specs:
        filename, key, directory = reference_sources[spec]
        reference = references[filename]
        p_full = np.mean([reference["predictions"][key][seed] for seed in SEEDS], axis=0)
        full_auc = float(roc_auc_score(data["y_test"], p_full))
        summary = summaries[spec]
        summary.update(
            full_ae_ensemble_auroc=full_auc,
            top32_minus_full_auroc=summary["ensemble_auroc"] - full_auc,
            ensemble_hall_aupr=float(average_precision_score(1-data["y_test"], 1-ensembles[spec])),
            full_ae_ensemble_hall_aupr=float(average_precision_score(1-data["y_test"], 1-p_full)),
            mean_seed_auroc=float(np.mean(list(summary["per_seed_auroc"].values()))),
            std_seed_auroc=float(np.std(list(summary["per_seed_auroc"].values()))),
            per_seed_auroc_delta={str(seed): summary["per_seed_auroc"][str(seed)] - reference["metrics"][key][seed]["auc"] for seed in SEEDS},
        )
        rows.append({"feature_set": spec, **{k: v for k, v in summary.items() if not isinstance(v, dict)}})
        for variant, source, source_key in (("top32", progress, spec), ("full", reference, key)):
            for seed in SEEDS:
                for rule, report in source["metrics"][source_key][seed]["threshold_reports"].items():
                    metric = report["test_metrics"]
                    seed_rows.append({
                        "feature_set": spec, "ae_variant": variant, "seed": seed,
                        "threshold_rule": rule, "threshold": report["threshold"], "auroc": metric["auc"],
                        **{f"{label}_{name}": metric[key][name]
                           for label, key in (("real", "real_positive"), ("hall", "hallucination_positive"))
                           for name in ("aupr", "precision", "recall", "f1")},
                    })
    for name, records in (("metrics", rows), ("seed_metrics", seed_rows)):
        write_csv(root / f"tables/{SLUG}_feature_{name}.csv", records)
    artifact_paths = [progress_path, split_path, Path(__file__),
                      ROOT / "scripts/analyze_ffn_visual_source_signal_ablation.py"]
    artifact_paths += list((root / f"metrics/probes_{SLUG}_features").glob("*/seed*/*"))
    artifact_paths += [root / f"tables/{SLUG}_feature_{name}.csv" for name in ("metrics", "seed_metrics")]
    artifact_paths += [ROOT / curve[key] for curve in curves.values() for key in ("figure", "table")]
    result = {
        "model": model, "status": "EXPLORATORY_TOP32_AE_COMPLETE",
        "definition": "AE32=sum_{j in Top32(T)} attention_j*gate_j = AE*sum_{Top32(T)}T_j; no selected-support renormalization",
        "unchanged": "S/I/N/kappa/R_cos and all three JS/OT distances unchanged; JS full support; OT pairwise Top32 union",
        "protocol": {**_json_ready(asdict(TorchProbeConfig())), "seeds": list(SEEDS),
                     "standardization": "none", "strength_transform": "none; raw S/I/N",
                     "class_weighting_or_resampling": "none", "bootstrap": "NOT_RUN_BY_USER_REQUEST",
                     "inference_boundary": "saved features only; no VLM; previously viewed test cohort, exploratory"},
        "counts": data["counts"], "cohort_feature_sha256": signature,
        "reference_sources": reference_sources, "curves": curves, "summaries": summaries,
        "ranges": {split: {name: {"min": float(x.min()), "median": float(np.median(x)), "max": float(x.max())}
                            for name, x in (("AE", data[f"X_{split}"]["B"]), ("AE32", ae32[split]), ("mass32", mass[split]))}
                   for split in ("train", "test")},
        "artifact_checksums": {str(p.relative_to(ROOT)): sha256_file(p) for p in artifact_paths},
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_json_save(_json_ready(result), result_path)
    print(f"[{model}] COMPLETE: {result['elapsed_seconds']:.1f}s", flush=True)
    return result


def summarize(models):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = {model: json.loads((result_root(model) / f"metrics/{SLUG}_feature_results.json").read_text()) for model in models}
    for slug in ("ae", "ae_top32"):
        fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharey=True)
        for ax, (model, result) in zip(axes.flat, results.items()):
            with (ROOT / result["curves"][slug]["table"]).open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            for label, color in (("REAL", "#2166ac"), ("HALL", "#b2182b")):
                selected = [r for r in rows if r["label"] == label]
                x = [int(r["layer"]) for r in selected]
                ax.plot(x, [float(r["median"]) for r in selected], color=color, label=f"{label} (n={selected[0]['n_mentions']})")
                ax.fill_between(x, [float(r["q25"]) for r in selected], [float(r["q75"]) for r in selected], color=color, alpha=.14)
            ax.set(title=model, xlabel="Decoder layer", ylabel=slug, ylim=(0, 1))
            ax.grid(alpha=.2)
            ax.legend(frameon=False, fontsize=9)
        for ax in list(axes.flat)[len(results):]:
            ax.set_visible(False)
        fig.suptitle(f"{slug}: REAL / HALL", fontsize=15)
        fig.text(.5, .014, "Medians and IQR (not CI) | All train + test mentions | No support renormalization", ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .04, 1, .96))
        for extension in ("png", "pdf"):
            fig.savefig(ROOT / f"outputs/ffn_visual_source_{slug}_real_hall.{extension}", dpi=180)
        plt.close(fig)
    atomic_json_save({"models": results}, ROOT / "outputs/ffn_visual_source_top32_ae_summary.json")
    print("Wrote AE/AE32 overview figures and cross-model summary", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--training-device", default="cuda:0")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    models = tuple(args.models.split(","))
    if not models or len(set(models)) != len(models) or set(models) - set(MODELS):
        raise ValueError("Choose unique supported model names")
    if args.summarize_only:
        summarize(models)
    else:
        for model in models:
            run_model(model, torch.device(args.training_device))


if __name__ == "__main__":
    main()
