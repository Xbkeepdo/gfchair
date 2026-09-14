"""Sweep batch size for True-RMS V/VP+G and compare with proportional SVAR."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.tc_fvpa_artifacts import atomic_json_save, atomic_torch_save
from scripts import search_single_mlp_811 as search


MODELS = tuple(search.MODELS)
GROUPS = ("visual", "vp_generation")
BATCH_SIZES = (32, 64, 128, 256, 512)
SEEDS = (43, 44, 45)
OUT = ROOT / "outputs/ours_batchsize_sweep_vs_svar_811_v1"
CONTROLLED = ROOT / "outputs/v_vs_vpg_standardized_no_bn_811_v1/summary.json"
QWEN_VPG = ROOT / "outputs/qwen25_vpg_standardized_no_bn_search_811_v1/summary.json"
SVAR = ROOT / "outputs/svar_llava_5_18_proportional_811_v1/summary.json"


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def base_configs():
    rows = json.loads(CONTROLLED.read_text())["summaries"]
    configs = {(row["model"], row["group"]): dict(row["config"]) for row in rows}
    qwen = json.loads(QWEN_VPG.read_text())
    configs[("qwen2_5_vl_7b", "vp_generation")] = dict(qwen["winner"]["config"])
    for config in configs.values():
        config.update(
            standardize=True,
            batch_norm=False,
            early_stopping=False,
            monitor="val_loss",
            max_epochs=150,
        )
    return configs


def protocol_for(model_name, source, configs):
    protocol = {
        "schema": "ours-batchsize-sweep-vs-svar-811-v1",
        "model": model_name,
        "source": "True-RMS all-attention z-A_all -> z; local FP32 Gauss-Legendre K32",
        "source_fingerprint": source["fingerprint"],
        "features": {
            "visual": "[AE_V,log1p(S_V_all)]",
            "vp_generation": "[AE_VP,log1p(S_V_all+S_P_all),AE_G,log1p(S_G_all)]",
        },
        "base_configs": {group: configs[(model_name, group)] for group in GROUPS},
        "control": "Within model/group, change batch_size only.",
        "batch_sizes": list(BATCH_SIZES),
        "seeds": list(SEEDS),
        "split_seed": 20260912,
        "split": "3200 train / 400 validation / 400 test images; all mentions",
        "normalization": "Per-column train-only population Z-score",
        "training": "No BatchNorm; all 150 epochs; minimum validation BCE checkpoint",
        "filter": "Report candidates whose three-seed mean test AUROC exceeds corresponding proportional SVAR mean AUROC",
        "status": "Exploratory batch-size follow-up after test-set access; all candidates retained.",
    }
    protocol["fingerprint"] = hashlib.sha256(
        json.dumps(protocol, sort_keys=True).encode()
    ).hexdigest()
    return protocol


def run_model(model_name, device):
    configs = base_configs()
    source = search.read(search.SOURCES["true_rms"] / model_name / "matrices.pt")
    protocol = protocol_for(model_name, source, configs)
    model_root = OUT / model_name
    atomic_json_save(protocol, model_root / "protocol.json")
    train, validation, test = (source["masks"][part] for part in ("train", "validation", "test"))
    rows = []
    for group in GROUPS:
        for batch_size in BATCH_SIZES:
            config = dict(configs[(model_name, group)], batch_size=batch_size)
            for seed in SEEDS:
                path = model_root / group / f"batch{batch_size}" / f"seed{seed}" / "result.pt"
                if path.exists():
                    result = search.read(path)
                    assert result["protocol_fingerprint"] == protocol["fingerprint"]
                    assert result["config"] == config
                else:
                    result = search.fit(
                        train_x=source["groups"][group][train],
                        train_y=source["y"][train],
                        val_x=source["groups"][group][validation],
                        val_y=source["y"][validation],
                        cfg=config,
                        seed=seed,
                        device=device,
                    )
                    network = search.SingleMLP(result["input_dim"], config).to(device)
                    network.load_state_dict(result["state_dict"])
                    test_x = torch.as_tensor(
                        search.transform(
                            source["groups"][group][test], result["mean"], result["scale"]
                        ),
                        device=device,
                    )
                    probabilities = search.predict(network, test_x)
                    result.update(
                        protocol_fingerprint=protocol["fingerprint"],
                        model=model_name,
                        group=group,
                        batch_size=batch_size,
                        test=search.metrics(source["y"][test], probabilities),
                        test_probabilities=probabilities,
                    )
                    atomic_torch_save(result, path)
                rows.append({
                    "model": model_name,
                    "group": group,
                    "batch_size": batch_size,
                    "seed": seed,
                    "best_epoch": result["best_epoch"],
                    "validation_AUROC": result["validation"]["AUROC"],
                    "validation_HALL_AUPR": result["validation"]["HALL_AUPR"],
                    "test_AUROC": result["test"]["AUROC"],
                    "test_HALL_AUPR": result["test"]["HALL_AUPR"],
                })
                print(model_name, group, batch_size, seed, result["test"], flush=True)
    summaries = []
    for group in GROUPS:
        for batch_size in BATCH_SIZES:
            selected = [
                row for row in rows
                if row["group"] == group and row["batch_size"] == batch_size
            ]
            summaries.append({
                "model": model_name,
                "group": group,
                "batch_size": batch_size,
                "AUROC_mean": float(np.mean([row["test_AUROC"] for row in selected])),
                "AUROC_std": float(np.std([row["test_AUROC"] for row in selected])),
                "HALL_AUPR_mean": float(np.mean([row["test_HALL_AUPR"] for row in selected])),
                "HALL_AUPR_std": float(np.std([row["test_HALL_AUPR"] for row in selected])),
                "best_epochs": [row["best_epoch"] for row in selected],
            })
    write_csv(rows, model_root / "seed_metrics.csv")
    atomic_json_save({"model": model_name, "summaries": summaries}, model_root / "summary.json")


def summarize_all():
    summaries = []
    rows = []
    svar_rows = json.loads(SVAR.read_text())["summaries"]
    svar = {row["model"]: row for row in svar_rows}
    for model_name in MODELS:
        summaries.extend(json.loads((OUT / model_name / "summary.json").read_text())["summaries"])
        with (OUT / model_name / "seed_metrics.csv").open() as handle:
            rows.extend(csv.DictReader(handle))
    passing = []
    for row in summaries:
        baseline = svar[row["model"]]
        enriched = dict(
            row,
            SVAR_AUROC=baseline["AUROC_mean"],
            delta_AUROC=row["AUROC_mean"] - baseline["AUROC_mean"],
            SVAR_HALL_AUPR=baseline["HALL_AUPR_mean"],
            delta_HALL_AUPR=row["HALL_AUPR_mean"] - baseline["HALL_AUPR_mean"],
        )
        if enriched["delta_AUROC"] > 0:
            passing.append(enriched)
    passing.sort(key=lambda row: (MODELS.index(row["model"]), GROUPS.index(row["group"]), row["batch_size"]))
    atomic_json_save(
        {
            "schema": "ours-batchsize-sweep-vs-svar-811-v1",
            "criterion": "three-seed mean test AUROC > proportional SVAR mean test AUROC",
            "summaries": summaries,
            "passing": passing,
        },
        OUT / "summary.json",
    )
    write_csv(rows, OUT / "seed_metrics.csv")
    if passing:
        write_csv(passing, OUT / "passing_auroc.csv")
    print(json.dumps(passing, ensure_ascii=False), flush=True)


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
