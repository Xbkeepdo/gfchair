"""SUPERSEDED: erroneous literal [5%,55%) interpretation; do not report."""

import argparse
import csv
import hashlib
import json
import math
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


MODELS = tuple(native.shared.study.original.MODELS)
SEEDS = (43, 44, 45)
START_FRACTION = 0.05
END_FRACTION = 0.55
OUT = ROOT / "outputs/svar_fraction_5_55_811_v1"


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def prepare(model_name):
    cache_root = native.OUT / "feature_cache" / model_name
    cached = native.read(cache_root / "matrices.pt")
    cached_protocol = json.loads((cache_root / "protocol.json").read_text())
    raw_path = ROOT / cached_protocol["source"]
    with raw_path.open("rb") as handle:
        records = shared.align_records(pickle.load(handle), cached["mentions"])
    payload = get_baseline_payload(records[0], "svar")
    matrix = np.asarray(payload["visual_attention_ratio"])
    stored_start = int(payload.get("layer_start", 0))
    stored_end = int(payload.get("layer_end_exclusive", stored_start + matrix.shape[0]))
    if stored_start != 0 or stored_end != matrix.shape[0]:
        raise ValueError(f"Expected all-layer SVAR artifact, got [{stored_start},{stored_end})/{matrix.shape}")
    total_layers = int(matrix.shape[0])
    start = int(math.floor(total_layers * START_FRACTION))
    end = int(math.ceil(total_layers * END_FRACTION))
    features = np.stack([
        svar_training_vector(get_baseline_payload(record, "svar"), layer_start=start, layer_end=end)
        for record in records
    ])
    return cached, cached_protocol, features, total_layers, start, end


def run_model(model_name, device):
    cached, cached_protocol, x, total_layers, start, end = prepare(model_name)
    protocol = {
        "schema": "native-svar-fraction-5-55-811-v1",
        "model": model_name,
        "total_decoder_layers": total_layers,
        "start_fraction": START_FRACTION,
        "end_fraction": END_FRACTION,
        "rounding": "start=floor(L*0.05), end=ceil(L*0.55), end-exclusive",
        "layer_start": start,
        "layer_end_exclusive": end,
        "selected_layers": end - start,
        "input_dim": int(x.shape[1]),
        "source": cached_protocol["source"],
        "source_fingerprint": cached["fingerprint"],
        "split_seed": 20260912,
        "split": "3200 train / 400 validation / 400 test images; all mentions",
        "seeds": list(SEEDS),
        "classifier": "Linear(D,248)-ReLU-Linear(248,2)",
        "training": dict(native.SVAR),
        "selection": "Minimum validation cross-entropy; early-stop patience5; validation REAL-F1 threshold.",
        "status": "Exploratory fraction follow-up after test-set access.",
    }
    digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode())
    digest.update(x.tobytes())
    protocol["fingerprint"] = digest.hexdigest()
    model_root = OUT / model_name
    atomic_json_save(protocol, model_root / "protocol.json")
    train, validation, test = (cached["masks"][part] for part in ("train", "validation", "test"))
    rows = []
    probabilities = []
    for seed in SEEDS:
        result_path = model_root / f"seed{seed}" / "result.pt"
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
                **{key: value for key, value in native.SVAR.items() if key != "hidden_dim"},
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
        probabilities.append(result["test_probabilities"])
        rows.append({
            "model": model_name,
            "seed": seed,
            "total_layers": total_layers,
            "layer_start": start,
            "layer_end_exclusive": end,
            "selected_layers": end - start,
            "input_dim": int(x.shape[1]),
            "best_epoch": result["best_epoch"],
            "AUROC": result["test_metrics"]["AUROC"],
            "HALL_AUPR": result["test_metrics"]["HALL_AUPR"],
        })
    summary = {
        "model": model_name,
        "total_layers": total_layers,
        "layer_start": start,
        "layer_end_exclusive": end,
        "selected_layers": end - start,
        "input_dim": int(x.shape[1]),
        "best_epochs": [row["best_epoch"] for row in rows],
        "AUROC_mean": float(np.mean([row["AUROC"] for row in rows])),
        "AUROC_std": float(np.std([row["AUROC"] for row in rows])),
        "HALL_AUPR_mean": float(np.mean([row["HALL_AUPR"] for row in rows])),
        "HALL_AUPR_std": float(np.std([row["HALL_AUPR"] for row in rows])),
        "ensemble": native.shared.study.scores(cached["y"][test], np.mean(probabilities, axis=0)),
    }
    write_csv(rows, model_root / "seed_metrics.csv")
    atomic_json_save(summary, model_root / "summary.json")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def summarize_all():
    summaries = []
    rows = []
    for model_name in MODELS:
        summaries.append(json.loads((OUT / model_name / "summary.json").read_text()))
        with (OUT / model_name / "seed_metrics.csv").open() as handle:
            rows.extend(csv.DictReader(handle))
    atomic_json_save({"schema": "native-svar-fraction-5-55-811-v1", "summaries": summaries}, OUT / "summary.json")
    write_csv(rows, OUT / "seed_metrics.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.summarize:
        summarize_all()
    elif args.models:
        for model_name in args.models:
            run_model(model_name, args.device)
    else:
        parser.error("Provide --models or --summarize")


if __name__ == "__main__":
    main()
