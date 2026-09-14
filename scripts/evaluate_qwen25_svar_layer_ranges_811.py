"""Evaluate Qwen2.5 native SVAR on decoder layer slices [5,16) and [5,17)."""

import csv
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from detection.baselines import (
    SVARMLP,
    _seed_everything,
    evaluate_detection_scores,
    select_detection_threshold,
    torch_hallucination_scores,
    train_torch_detector,
)
from features.baseline import get_baseline_payload, svar_training_vector
from features.tc_fvpa_artifacts import atomic_json_save, atomic_torch_save
from scripts import train_baselines_811 as shared
from scripts import train_native_baselines_811 as native


MODEL = "qwen2_5_vl_7b"
SEEDS = (43, 44, 45)
RANGES = ((5, 16), (5, 17))
OUT = ROOT / "outputs/qwen25_svar_layer_ranges_811_v1"
SCHEMA = "qwen25-native-svar-layer-ranges-811-v1"


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main():
    torch.set_num_threads(1)
    device = "cuda:0"
    cache_root = native.OUT / "feature_cache" / MODEL
    cached = native.read(cache_root / "matrices.pt")
    cached_protocol = json.loads((cache_root / "protocol.json").read_text())
    raw_path = ROOT / cached_protocol["source"]
    with raw_path.open("rb") as handle:
        records = pickle.load(handle)
    records = shared.align_records(records, cached["mentions"])
    matrices = {
        f"{start}_{end}": np.stack([
            svar_training_vector(get_baseline_payload(record, "svar"), layer_start=start, layer_end=end)
            for record in records
        ])
        for start, end in RANGES
    }
    protocol = {
        "schema": SCHEMA,
        "model": MODEL,
        "ranges_zero_based_end_exclusive": [list(value) for value in RANGES],
        "input_dimensions": {key: int(value.shape[1]) for key, value in matrices.items()},
        "source": cached_protocol["source"],
        "source_fingerprint": cached["fingerprint"],
        "split_seed": 20260912,
        "split": "3200 train / 400 validation / 400 test images; all mentions",
        "seeds": list(SEEDS),
        "classifier": "Linear(D,248)-ReLU-Linear(248,2)",
        "training": dict(native.SVAR),
        "selection": "Minimum validation cross-entropy; early-stop patience5; validation REAL-F1 threshold.",
        "status": "Exploratory layer-range follow-up after test-set access; both endpoint interpretations reported.",
    }
    digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode())
    for key in sorted(matrices):
        digest.update(key.encode())
        digest.update(matrices[key].tobytes())
    protocol["fingerprint"] = digest.hexdigest()
    atomic_json_save(protocol, OUT / "protocol.json")

    rows = []
    for start, end in RANGES:
        key = f"{start}_{end}"
        x = matrices[key]
        train, validation, test = (cached["masks"][part] for part in ("train", "validation", "test"))
        for seed in SEEDS:
            result_path = OUT / f"layers_{key}" / f"seed{seed}" / "result.pt"
            if result_path.exists():
                result = native.read(result_path)
                assert result["fingerprint"] == protocol["fingerprint"]
            else:
                _seed_everything(seed)
                model = SVARMLP(x.shape[1], hidden_dim=native.SVAR["hidden_dim"])
                trained = train_torch_detector(
                    model=model,
                    X_train=x[train],
                    raw_y_train=cached["y"][train],
                    X_val=x[validation],
                    raw_y_val=cached["y"][validation],
                    X_test=x[test],
                    raw_y_test=cached["y"][test],
                    device=device,
                    seed=seed,
                    positive_class="real",
                    strict_82_no_validation=False,
                    **{name: value for name, value in native.SVAR.items() if name != "hidden_dim"},
                )
                model.load_state_dict(trained.state_dict)
                hall = {
                    part: torch_hallucination_scores(model, x[mask], torch.device(device))
                    for part, mask in cached["masks"].items()
                }
                real = {part: 1.0 - value for part, value in hall.items()}
                threshold = select_detection_threshold(
                    cached["y"][validation], hall["validation"], positive_class="real"
                )
                result = {
                    "fingerprint": protocol["fingerprint"],
                    "seed": seed,
                    "layer_start": start,
                    "layer_end_exclusive": end,
                    "input_dim": int(x.shape[1]),
                    "state_dict": trained.state_dict,
                    "history": trained.history,
                    "best_epoch": int(min(trained.history, key=lambda row: row["val_loss"])["epoch"]),
                    "threshold": threshold,
                    "threshold_report": evaluate_detection_scores(
                        cached["y"][test], hall["test"], threshold, positive_class="real"
                    ),
                    "test_metrics": native.shared.study.scores(cached["y"][test], real["test"]),
                    "test_probabilities": real["test"],
                }
                atomic_torch_save(result, result_path)
            rows.append({
                "layer_start": start,
                "layer_end_exclusive": end,
                "layers": end - start,
                "input_dim": result["input_dim"],
                "seed": seed,
                "best_epoch": result["best_epoch"],
                "AUROC": result["test_metrics"]["AUROC"],
                "HALL_AUPR": result["test_metrics"]["HALL_AUPR"],
            })

    summaries = []
    for start, end in RANGES:
        selected = [row for row in rows if row["layer_start"] == start and row["layer_end_exclusive"] == end]
        summaries.append({
            "layer_start": start,
            "layer_end_exclusive": end,
            "layers": end - start,
            "input_dim": selected[0]["input_dim"],
            "best_epochs": [row["best_epoch"] for row in selected],
            "AUROC_mean": float(np.mean([row["AUROC"] for row in selected])),
            "AUROC_std": float(np.std([row["AUROC"] for row in selected])),
            "HALL_AUPR_mean": float(np.mean([row["HALL_AUPR"] for row in selected])),
            "HALL_AUPR_std": float(np.std([row["HALL_AUPR"] for row in selected])),
        })
    write_csv(rows, OUT / "seed_metrics.csv")
    atomic_json_save({"schema": protocol["schema"], "summaries": summaries}, OUT / "summary.json")
    print(json.dumps(summaries, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
