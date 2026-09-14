"""Evaluate train-only column StandardScaler, no BN, and val-loss patience 5/10/20."""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.tc_fvpa_artifacts import atomic_json_save, atomic_torch_save
from scripts import search_single_mlp_811 as search


OUT = ROOT / "outputs/standardized_no_bn_patience_811_v1"
PATIENCES = (5, 10, 20)


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def evaluate_model(model_name, device):
    selection = json.loads((search.OUT / model_name / "selection.json").read_text())
    variant = selection["champion"]
    path_name, group_name = variant.split("/")
    source = search.read(search.SOURCES[path_name] / model_name / "matrices.pt")
    train_mask = source["masks"]["train"]
    validation_mask = source["masks"]["validation"]
    test_mask = source["masks"]["test"]
    test_y = source["y"][test_mask]
    original = selection["variants"][variant]
    model_root = OUT / model_name
    protocol = {
        "schema": "standardized-no-bn-patience-811-v1",
        "model": model_name,
        "variant": variant,
        "source_fingerprint": source["fingerprint"],
        "parent_candidate": original["index"],
        "parent_config": original["config"],
        "patiences": list(PATIENCES),
        "seeds": list(search.SEEDS),
        "split_seed": 20260912,
        "normalization": "Per-column Z-score; fit on training mentions only with population standard deviation.",
        "fixed_changes": {"standardize": True, "batch_norm": False, "monitor": "val_loss"},
        "scheduler": {"factor": 0.5, "patience": 6, "min_lr": 1e-6},
        "status": "Exploratory follow-up after test-set access; all patience values are reported.",
    }
    atomic_json_save(protocol, model_root / "protocol.json")
    seed_rows = []
    summaries = []
    for patience in PATIENCES:
        config = dict(original["config"])
        config.update(standardize=True, batch_norm=False, monitor="val_loss", patience=patience)
        values = []
        probabilities = []
        for seed in search.SEEDS:
            result_path = model_root / f"patience{patience}" / f"seed{seed}" / "result.pt"
            if result_path.exists():
                result = search.read(result_path)
                assert result["config"] == config
                assert result["variant"] == variant
                assert result["source_fingerprint"] == source["fingerprint"]
            else:
                result = search.fit(
                    train_x=source["groups"][group_name][train_mask],
                    train_y=source["y"][train_mask],
                    val_x=source["groups"][group_name][validation_mask],
                    val_y=source["y"][validation_mask],
                    cfg=config,
                    seed=seed,
                    device=device,
                )
                net = search.SingleMLP(result["input_dim"], config).to(device)
                net.load_state_dict(result["state_dict"])
                test_x = torch.as_tensor(
                    search.transform(source["groups"][group_name][test_mask], result["mean"], result["scale"]),
                    device=device,
                )
                test_probabilities = search.predict(net, test_x)
                result.update(
                    variant=variant,
                    source_fingerprint=source["fingerprint"],
                    test=search.metrics(test_y, test_probabilities),
                    test_probabilities=test_probabilities,
                )
                atomic_torch_save(result, result_path)
            values.append(result["test"])
            probabilities.append(result["test_probabilities"])
            seed_rows.append({
                "model": model_name,
                "variant": variant,
                "patience": patience,
                "seed": seed,
                "best_epoch": result["best_epoch"],
                "epochs": result["epochs"],
                "validation_loss": result["history"][result["best_epoch"] - 1]["val_loss"],
                "validation_AUROC": result["validation"]["AUROC"],
                "validation_HALL_AUPR": result["validation"]["HALL_AUPR"],
                "test_AUROC": result["test"]["AUROC"],
                "test_HALL_AUPR": result["test"]["HALL_AUPR"],
            })
        subset = [row for row in seed_rows if row["patience"] == patience]
        summaries.append({
            "model": model_name,
            "variant": variant,
            "patience": patience,
            "config": config,
            "best_epochs": [row["best_epoch"] for row in subset],
            "stopped_epochs": [row["epochs"] for row in subset],
            "validation_loss_mean": float(np.mean([row["validation_loss"] for row in subset])),
            "validation_AUROC_mean": float(np.mean([row["validation_AUROC"] for row in subset])),
            "validation_HALL_AUPR_mean": float(np.mean([row["validation_HALL_AUPR"] for row in subset])),
            "test_AUROC_mean": float(np.mean([value["AUROC"] for value in values])),
            "test_AUROC_std": float(np.std([value["AUROC"] for value in values])),
            "test_HALL_AUPR_mean": float(np.mean([value["HALL_AUPR"] for value in values])),
            "test_HALL_AUPR_std": float(np.std([value["HALL_AUPR"] for value in values])),
            "ensemble": search.metrics(test_y, np.mean(probabilities, axis=0)),
        })
    write_csv(seed_rows, model_root / "seed_metrics.csv")
    atomic_json_save({"model": model_name, "variant": variant, "summaries": summaries}, model_root / "summary.json")
    for summary in summaries:
        print(json.dumps(summary, ensure_ascii=False), flush=True)


def summarize_all():
    all_summaries = []
    all_seed_rows = []
    for model_name in search.MODELS:
        model_root = OUT / model_name
        value = json.loads((model_root / "summary.json").read_text())
        all_summaries.extend(value["summaries"])
        with (model_root / "seed_metrics.csv").open() as handle:
            all_seed_rows.extend(csv.DictReader(handle))
    atomic_json_save({"schema": "standardized-no-bn-patience-811-v1", "summaries": all_summaries}, OUT / "summary.json")
    write_csv(all_seed_rows, OUT / "seed_metrics.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=search.MODELS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.summarize:
        summarize_all()
    elif args.models:
        for model_name in args.models:
            evaluate_model(model_name, args.device)
    else:
        parser.error("Provide --models or --summarize")


if __name__ == "__main__":
    main()
