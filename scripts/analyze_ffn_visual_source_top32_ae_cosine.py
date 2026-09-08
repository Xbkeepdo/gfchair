#!/usr/bin/env python3
"""Replace AE32 by AE32 times the unweighted hpre cosine on the same Top32."""
from __future__ import annotations

import argparse
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
    OLD_COSINE_KEY, OLD_EV_KEY, ae_times_cosine, swap_ae_for_old_ev, train_matrices,
)
from scripts.analyze_ffn_visual_source_study import SEEDS, _detector_matrices, _json_ready, _load_full_model
from scripts.analyze_ffn_visual_source_top32_ae import MODELS, ae_replacement_matrices, checked_progress, topk_ae, write_csv
from scripts.run_ffn_visual_source_attribution import EXPERIMENT, result_root
from scripts.train_torch_probe_feature_sets import TorchProbeConfig
from utils.io_utils import load_pkl

SLUG = "top32_ae_cosine"


def cosine_replacement_matrices(ae32_groups, cosine):
    """Multiply only AE32 by signed cosine; all other 20-group columns stay fixed."""
    output = {}
    for split in ("train", "test"):
        groups = ae32_groups[split]
        c = np.asarray(cosine[split], dtype=np.float32)
        if not np.isfinite(c).all() or np.any(np.abs(c) > 1 + 2e-6):
            raise ValueError("Mean cosine must be finite and in [-1,1]")
        product = ae_times_cosine(groups["B"], c)
        output[split] = swap_ae_for_old_ev({k: v for k, v in groups.items() if k != "U"}, product)
        output[split]["U"] = swap_ae_for_old_ev({"A": groups["U"]}, product)["A"]
        layers = product.shape[1]
        for key, value in groups.items():
            start = layers if key in {"A", "U"} or key.startswith("H_") else 0
            np.testing.assert_array_equal(value[:, start:start+layers], groups["B"])
            expected = value.copy()
            expected[:, start:start+layers] = product
            np.testing.assert_array_equal(output[split][key], expected)
    return output


def load_inputs(model, data):
    positions, _, _ = _load_full_model(model)
    ae_lookup = {str(p["target_key"]): topk_ae(p["ae_strength"], p["attention_evidence"])[0] for p in positions}
    wanted = set(data["train_target_keys"]) | set(data["test_target_keys"])
    assert set(ae_lookup) == wanted
    del positions
    rows = load_pkl(str(ROOT / "outputs" / model / EXPERIMENT / "features.pkl"))
    lookup, duplicates, max_duplicate_error = {}, 0, 0.
    for row in rows:
        key = f"{int(row['image_id'])}:{int(row['response_token_idx'])}"
        if key not in wanted:
            continue
        cosine = np.asarray(row[OLD_COSINE_KEY], dtype=np.float32).reshape(-1)
        old_ev = np.asarray(row[OLD_EV_KEY], dtype=np.float32).reshape(-1)
        attention = np.asarray(row["dgst_t_attention_support_per_layer"], dtype=np.float32)
        gate = np.asarray(row["dgst_t_hpre_raw_logit_gauss_gate_per_layer"], dtype=np.float32)
        raw = attention * gate
        assert raw.ndim == 2 and raw.shape[0] == len(cosine)
        assert np.isfinite(raw).all() and np.all(raw >= 0)
        k = min(32, raw.shape[1])
        direct = np.partition(raw, raw.shape[1]-k, axis=1)[:, -k:].sum(axis=1, dtype=np.float64).astype(np.float32)
        value = np.stack([cosine, old_ev, direct, raw.sum(axis=1)])
        assert np.isfinite(value).all()
        if key in lookup:
            duplicates += 1
            error = float(np.abs(value - lookup[key]).max())
            max_duplicate_error = max(max_duplicate_error, error)
            assert error <= 1e-6, f"Conflicting source rows: {key}"
        else:
            lookup[key] = value
    del rows
    assert set(lookup) == wanted
    ae32, cosine, checks = {}, {}, {}
    for split in ("train", "test"):
        keys = data[f"{split}_target_keys"]
        ae32[split] = np.stack([ae_lookup[key] for key in keys])
        aligned = np.stack([lookup[key] for key in keys])
        cosine[split], old_ev, direct, raw_ae = (aligned[:, i, :] for i in range(4))
        np.testing.assert_allclose(raw_ae, data[f"X_{split}"]["B"], atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(ae32[split], direct, atol=1e-6, rtol=1e-6)
        product = ae_times_cosine(ae32[split], cosine[split])
        identity = data[f"X_{split}"]["B"] * old_ev
        np.testing.assert_allclose(product, identity, atol=1e-6, rtol=2e-5)
        checks[split] = {
            "direct_raw_top32_ae_max_absolute_error": float(np.abs(ae32[split]-direct).max()),
            "ae32_cos_equals_ae_times_old_ev_max_absolute_error": float(np.abs(product-identity).max()),
            "ranges": {name: {"min": float(x.min()), "max": float(x.max()), "negative_entries": int((x<0).sum())}
                       for name, x in (("AE32", ae32[split]), ("cosine", cosine[split]), ("product", product))},
        }
    return ae32, cosine, {"splits": checks, "duplicate_rows": duplicates, "max_duplicate_error": max_duplicate_error,
                          "cosine_source_key": OLD_COSINE_KEY, "unique_targets": len(lookup)}


def run_model(model, device):
    started = time.monotonic()
    torch.set_num_threads(1)
    print(f"[{model}] Loading saved AE32 and matched Top32 hpre cosine", flush=True)
    data = _detector_matrices(model)
    ae32, cosine, input_audit = load_inputs(model, data)
    originals, ae32_groups = ae_replacement_matrices(data, ae32)
    matrices = cosine_replacement_matrices(ae32_groups, cosine)
    specs = tuple(originals["train"])
    root = result_root(model)
    split_path = ROOT / "outputs" / model / EXPERIMENT / "image_splits.json"
    split_ids = json.loads(split_path.read_text())
    assert len(split_ids["train"]) == 3200 and len(split_ids["test"]) == 800
    assert not (set(split_ids["train"]) & set(split_ids["test"]))
    base_path = root / "metrics/top32_ae_feature_results.json"
    base = json.loads(base_path.read_text())
    digest = hashlib.sha256(split_path.read_bytes())
    for split in ("train", "test"):
        digest.update(json.dumps(data[f"{split}_target_keys"]).encode())
        digest.update(data[f"y_{split}"].tobytes())
        for spec in specs:
            x = ae32_groups[split][spec]
            digest.update(f"{split}:{spec}:{x.shape}:{x.dtype}".encode())
            digest.update(x.tobytes())
    assert digest.hexdigest() == base["cohort_feature_sha256"], "AE32 reference cohort/features changed"
    for split in ("train", "test"):
        digest.update(cosine[split].tobytes())
        for spec in specs:
            digest.update(matrices[split][spec].tobytes())
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
            assert sha256_file(ROOT/path) == checksum, path
        print(f"[{model}] COMPLETE: verified resume; no training or writes", flush=True)
        return result
    if not progress_path.exists():
        atomic_torch_save(progress, progress_path)
    cache_path = root / f"metrics/{SLUG}_inputs.pt"
    if not cache_path.exists():
        atomic_torch_save({"ae32": ae32, "cosine": cosine, "input_audit": input_audit,
                           "cohort_feature_sha256": signature}, cache_path)
    reference_path = root / "metrics/top32_ae_feature_training_progress.pt"
    reference = torch.load(reference_path, map_location="cpu", weights_only=False)
    ae_cos_path = root / "metrics/ae_cosine_feature_group_training_progress.pt"
    ae_cos = torch.load(ae_cos_path, map_location="cpu", weights_only=False)
    for spec in specs:
        for seed in SEEDS:
            config_path = root / f"metrics/probes_top32_ae_features/{spec}/seed{seed}/config.json"
            config = json.loads(config_path.read_text())
            assert all(config[k] == v for k, v in _json_ready(asdict(TorchProbeConfig(seed=seed))).items())
            p = np.asarray(reference["predictions"][spec][seed])
            assert p.shape == data["y_test"].shape and np.isfinite(p).all()
            assert abs(roc_auc_score(data["y_test"], p)-reference["metrics"][spec][seed]["auc"]) < 1e-10
    print(f"[{model}] Input identities PASS; training 20 groups x 3 seeds", flush=True)
    progress, ensembles, summaries = train_matrices(
        model, data, matrices, specs, device=device, resume=True,
        progress_name=progress_path.name, probe_directory=f"metrics/probes_{SLUG}_features",
    )
    rows, seed_rows = [], []
    for spec in specs:
        summary = summaries[spec]
        p32 = np.mean([reference["predictions"][spec][seed] for seed in SEEDS], axis=0)
        base_auc = float(roc_auc_score(data["y_test"], p32))
        assert abs(base_auc-base["summaries"][spec]["ensemble_auroc"]) < 1e-10
        full_auc = base["summaries"][spec]["full_ae_ensemble_auroc"]
        summary.update(
            ae32_ensemble_auroc=base_auc, product_minus_ae32_auroc=summary["ensemble_auroc"]-base_auc,
            full_ae_ensemble_auroc=full_auc, product_minus_full_ae_auroc=summary["ensemble_auroc"]-full_auc,
            ensemble_hall_aupr=float(average_precision_score(1-data["y_test"], 1-ensembles[spec])),
            mean_seed_auroc=float(np.mean(list(summary["per_seed_auroc"].values()))),
            std_seed_auroc=float(np.std(list(summary["per_seed_auroc"].values()))),
            per_seed_auroc_delta={str(seed): summary["per_seed_auroc"][str(seed)]-reference["metrics"][spec][seed]["auc"] for seed in SEEDS},
        )
        if spec in data["specs"]:
            pac = np.mean([ae_cos["predictions"][spec][seed] for seed in SEEDS], axis=0)
            summary["full_ae_cosine_ensemble_auroc"] = float(roc_auc_score(data["y_test"], pac))
            summary["product_minus_full_ae_cosine_auroc"] = summary["ensemble_auroc"]-summary["full_ae_cosine_ensemble_auroc"]
        rows.append({"feature_set": spec, **{k: v for k, v in summary.items() if not isinstance(v, dict)}})
        for variant, source in (("AE32_cosine", progress), ("AE32", reference)):
            for seed in SEEDS:
                for rule, report in source["metrics"][spec][seed]["threshold_reports"].items():
                    metric = report["test_metrics"]
                    seed_rows.append({"feature_set": spec, "variant": variant, "seed": seed,
                                      "threshold_rule": rule, "threshold": report["threshold"], "auroc": metric["auc"],
                                      **{f"{label}_{name}": metric[key][name]
                                         for label, key in (("real", "real_positive"), ("hall", "hallucination_positive"))
                                         for name in ("aupr", "precision", "recall", "f1")}})
    for name, records in (("metrics", rows), ("seed_metrics", seed_rows)):
        write_csv(root/f"tables/{SLUG}_feature_{name}.csv", records)
    paths = [progress_path, cache_path, split_path, base_path, reference_path, ae_cos_path, Path(__file__),
             ROOT/"scripts/analyze_ffn_visual_source_top32_ae.py", ROOT/"scripts/analyze_ffn_visual_source_signal_ablation.py"]
    paths += list((root/f"metrics/probes_{SLUG}_features").glob("*/seed*/*"))
    paths += [root/f"tables/{SLUG}_feature_{name}.csv" for name in ("metrics", "seed_metrics")]
    result = {
        "model": model, "status": "EXPLORATORY_AE32_COSINE_COMPLETE",
        "definition": "X=AE32*mean_{j in Top32(AE)}cos(hpre_prediction,hpre_visual_j); arithmetic, unweighted, signed cosine; not sum(AE_j*cos_j)",
        "identity": "X=full_AE*old_EV when old_EV=Top32_mass*mean_cosine; not old_EV alone or full_AE*mean_cosine",
        "unchanged": "all S/I/N/kappa/R_cos and distance blocks; full-support JS and pairwise Top32-union OT",
        "protocol": {**_json_ready(asdict(TorchProbeConfig())), "seeds": list(SEEDS), "standardization": "none",
                     "class_weighting_or_resampling": "none", "bootstrap": "NOT_RUN_BY_USER_REQUEST",
                     "inference_boundary": "no VLM forward; fixed previously viewed test cohort; exploratory"},
        "counts": data["counts"], "input_audit": input_audit, "cohort_feature_sha256": signature,
        "summaries": summaries, "elapsed_seconds": time.monotonic()-started,
        "artifact_checksums": {str(p.relative_to(ROOT)): sha256_file(p) for p in paths},
    }
    atomic_json_save(_json_ready(result), result_path)
    print(f"[{model}] COMPLETE: {result['elapsed_seconds']:.1f}s", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--training-device", default="cuda:0")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    models = tuple(args.models.split(","))
    if len(models) != len(set(models)) or set(models)-set(MODELS):
        raise ValueError("Choose unique supported models")
    if args.summarize_only:
        results = {m: json.loads((result_root(m)/f"metrics/{SLUG}_feature_results.json").read_text()) for m in models}
        atomic_json_save({"models": results}, ROOT/f"outputs/ffn_visual_source_{SLUG}_summary.json")
    else:
        for model in models:
            run_model(model, torch.device(args.training_device))


if __name__ == "__main__":
    main()
