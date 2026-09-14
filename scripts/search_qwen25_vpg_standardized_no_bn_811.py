"""Validation-only HPO for Qwen2.5 True-RMS VP+G under the standardized no-BN protocol."""

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.tc_fvpa_artifacts import atomic_json_save, atomic_torch_save
from scripts import search_single_mlp_811 as search


MODEL = "qwen2_5_vl_7b"
GROUP = "vp_generation"
OUT = ROOT / "outputs/qwen25_vpg_standardized_no_bn_search_811_v1"
SEEDS = (43, 44, 45)


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def candidate_configs():
    values = []
    for template in search.candidates():
        config = dict(template)
        config.update(standardize=True, batch_norm=False, monitor="val_loss", early_stopping=False)
        if config not in values:
            values.append(config)
    return values


def rank(row):
    return row["validation"]["AUROC"], row["validation"]["HALL_AUPR"], -row["index"]


def main():
    torch.set_num_threads(1)
    device = "cuda:0"
    source = search.read(search.SOURCES["true_rms"] / MODEL / "matrices.pt")
    train = source["masks"]["train"]
    validation = source["masks"]["validation"]
    test = source["masks"]["test"]
    configs = candidate_configs()
    protocol = {
        "schema": "qwen25-vpg-standardized-no-bn-search-811-v1",
        "model": MODEL,
        "group": GROUP,
        "source": "True-RMS All-attention K32",
        "source_fingerprint": source["fingerprint"],
        "split_seed": 20260912,
        "split": "3200 train / 400 validation / 400 test images; all mentions",
        "candidates": configs,
        "candidate_construction": "Deduplicate the prior 24 fixed templates after forcing StandardScaler=True, BN=False, val_loss, no early stopping.",
        "search_seed": 20260914,
        "shortlist": "Run all candidates with seed43; rank validation AUROC, HALL-AUPR, then lower index; top3 add seeds44/45.",
        "selection": "Rank top3 by three-seed validation AUROC mean, HALL-AUPR mean, then lower index; access test only after freezing.",
        "checkpoint": "Train all 150 epochs and retain minimum validation BCE loss.",
        "status": "Exploratory follow-up because the fixed test set was previously accessed.",
    }
    protocol["fingerprint"] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    atomic_json_save(protocol, OUT / "protocol.json")

    def one(index, seed):
        result_path = OUT / f"candidate{index:02d}" / f"seed{seed}" / "result.pt"
        if result_path.exists():
            result = search.read(result_path)
            assert result["fingerprint"] == protocol["fingerprint"]
        else:
            result = search.fit(
                train_x=source["groups"][GROUP][train],
                train_y=source["y"][train],
                val_x=source["groups"][GROUP][validation],
                val_y=source["y"][validation],
                cfg=configs[index],
                seed=seed,
                device=device,
            )
            result.update(index=index, fingerprint=protocol["fingerprint"])
            atomic_torch_save(result, result_path)
        return result

    trials = []
    for index in range(len(configs)):
        result = one(index, 43)
        trials.append({"index": index, "seed": 43, "validation": result["validation"]})
        print("seed43", index, result["validation"], flush=True)
    shortlist = sorted(trials, key=rank, reverse=True)[:3]
    comparisons = []
    for trial in shortlist:
        runs = [one(trial["index"], seed) for seed in SEEDS]
        comparison = {
            "index": trial["index"],
            "config": configs[trial["index"]],
            "seeds": list(SEEDS),
            "best_epochs": [run["best_epoch"] for run in runs],
            "validation": {
                key: float(np.mean([run["validation"][key] for run in runs]))
                for key in runs[0]["validation"]
            },
        }
        comparisons.append(comparison)
        print("shortlist", json.dumps(comparison, ensure_ascii=False), flush=True)
    winner = max(comparisons, key=rank)
    selection = {
        "fingerprint": protocol["fingerprint"],
        "winner": winner,
        "shortlist": comparisons,
        "seed43_trials": trials,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "test_accessed": False,
    }
    atomic_json_save(selection, OUT / "selection.json")

    test_y = source["y"][test]
    seed_rows = []
    probabilities = []
    for seed in SEEDS:
        trained = one(winner["index"], seed)
        net = search.SingleMLP(trained["input_dim"], trained["config"]).to(device)
        net.load_state_dict(trained["state_dict"])
        test_x = torch.as_tensor(
            search.transform(source["groups"][GROUP][test], trained["mean"], trained["scale"]),
            device=device,
        )
        probability = search.predict(net, test_x)
        value = search.metrics(test_y, probability)
        probabilities.append(probability)
        seed_rows.append({
            "model": MODEL,
            "group": GROUP,
            "candidate": winner["index"],
            "seed": seed,
            "best_epoch": trained["best_epoch"],
            "AUROC": value["AUROC"],
            "HALL_AUPR": value["HALL_AUPR"],
        })
        atomic_torch_save(
            {"seed": seed, "candidate": winner["index"], "test": value, "test_probabilities": probability},
            OUT / "final" / f"seed{seed}" / "result.pt",
        )
    summary = {
        "schema": protocol["schema"],
        "model": MODEL,
        "group": GROUP,
        "winner": winner,
        "test_AUROC_mean": float(np.mean([row["AUROC"] for row in seed_rows])),
        "test_AUROC_std": float(np.std([row["AUROC"] for row in seed_rows])),
        "test_HALL_AUPR_mean": float(np.mean([row["HALL_AUPR"] for row in seed_rows])),
        "test_HALL_AUPR_std": float(np.std([row["HALL_AUPR"] for row in seed_rows])),
        "ensemble": search.metrics(test_y, np.mean(probabilities, axis=0)),
    }
    write_csv(seed_rows, OUT / "seed_metrics.csv")
    atomic_json_save(summary, OUT / "summary.json")
    selection["test_accessed"] = True
    atomic_json_save(selection, OUT / "selection.json")
    print("FINAL", json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
