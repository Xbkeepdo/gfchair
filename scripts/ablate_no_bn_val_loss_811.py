"""Controlled 811 ablation: champion settings with BN disabled and val-loss checkpoints."""

import argparse
import csv
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


OUT = ROOT / "outputs/no_bn_val_loss_811_v1"


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def run_model(model_name, device, *, output_root=OUT, force_standardize=False, blockwise_standardize=False,
              disable_early_stopping=False):
    selection_path = search.OUT / model_name / "selection.json"
    selection = json.loads(selection_path.read_text())
    variant = selection["champion"]
    path_name, group_name = variant.split("/")
    source = search.read(search.SOURCES[path_name] / model_name / "matrices.pt")
    train_mask = source["masks"]["train"]
    validation_mask = source["masks"]["validation"]
    test_mask = source["masks"]["test"]

    original = selection["variants"][variant]
    config = dict(original["config"])
    config["batch_norm"] = False
    config["monitor"] = "val_loss"
    if force_standardize or blockwise_standardize:
        config["standardize"] = True
    if blockwise_standardize:
        input_dim = int(source["groups"][group_name].shape[1])
        block_count = 4 if group_name == "vp_generation" else 2
        if group_name not in {"visual", "visual_prompt_sum", "generation", "vp_generation"} or input_dim % block_count:
            raise ValueError(f"Unsupported block-wise group/dimension {group_name}/{input_dim}")
        config["block_sizes"] = [input_dim // block_count] * block_count
        config["normalization"] = "blockwise_train_mentions_x_layers"
    if disable_early_stopping:
        config["early_stopping"] = False

    folder = output_root / model_name
    protocol = {
        "schema": (
            "standardized-no-bn-no-early-stop-811-v1" if disable_early_stopping else
            "no-bn-blockwise-val-loss-811-v1" if blockwise_standardize else
            "no-bn-standardized-val-loss-811-v1" if force_standardize else
            "no-bn-val-loss-811-v1"
        ),
        "model": model_name,
        "variant": variant,
        "source_fingerprint": source["fingerprint"],
        "parent_candidate": original["index"],
        "parent_config": original["config"],
        "ablation_config": config,
        "seeds": list(search.SEEDS),
        "split_seed": 20260912,
        "change": (
            "Set column-wise standardize=True, batch_norm=False, monitor=val_loss, and train all max_epochs without early stopping."
            if disable_early_stopping else
            "Apply one train-only mean/std per VP+G feature block across mentions and layers; set batch_norm=False and monitor=val_loss."
            if blockwise_standardize else
            "Set column-wise standardize=True, batch_norm=False and monitor=val_loss; preserve all other champion hyperparameters."
            if force_standardize else
            "Set batch_norm=False and monitor=val_loss; preserve all other champion hyperparameters."
        ),
        "status": "Exploratory follow-up after test-set access; not a newly sealed model selection.",
    }
    atomic_json_save(protocol, folder / "protocol.json")

    test_y = source["y"][test_mask]
    rows = []
    probabilities = []
    for seed in search.SEEDS:
        result_path = folder / f"seed{seed}" / "result.pt"
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
        probabilities.append(result["test_probabilities"])
        rows.append(
            {
                "model": model_name,
                "variant": variant,
                "seed": seed,
                "best_epoch": result["best_epoch"],
                "epochs": result["epochs"],
                "validation_loss": result["history"][result["best_epoch"] - 1]["val_loss"],
                "validation_AUROC": result["validation"]["AUROC"],
                "validation_HALL_AUPR": result["validation"]["HALL_AUPR"],
                "test_AUROC": result["test"]["AUROC"],
                "test_HALL_AUPR": result["test"]["HALL_AUPR"],
            }
        )
    summary = {
        "model": model_name,
        "variant": variant,
        "config": config,
        "best_epochs": [row["best_epoch"] for row in rows],
        "trained_epochs": [row["epochs"] for row in rows],
        "validation_AUROC_mean": float(np.mean([row["validation_AUROC"] for row in rows])),
        "validation_HALL_AUPR_mean": float(np.mean([row["validation_HALL_AUPR"] for row in rows])),
        "test_AUROC_mean": float(np.mean([row["test_AUROC"] for row in rows])),
        "test_AUROC_std": float(np.std([row["test_AUROC"] for row in rows])),
        "test_HALL_AUPR_mean": float(np.mean([row["test_HALL_AUPR"] for row in rows])),
        "test_HALL_AUPR_std": float(np.std([row["test_HALL_AUPR"] for row in rows])),
        "ensemble": search.metrics(test_y, np.mean(probabilities, axis=0)),
    }
    atomic_json_save(summary, folder / "summary.json")
    write_csv(rows, folder / "seed_metrics.csv")
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=search.MODELS, default=list(search.MODELS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force-standardize", action="store_true")
    parser.add_argument("--blockwise-standardize", action="store_true")
    parser.add_argument("--disable-early-stopping", action="store_true")
    parser.add_argument("--output-name", default=OUT.name)
    args = parser.parse_args()
    if args.force_standardize and args.blockwise_standardize:
        parser.error("Choose either --force-standardize or --blockwise-standardize")
    if args.disable_early_stopping and not args.force_standardize:
        parser.error("--disable-early-stopping currently requires --force-standardize")
    output_root = ROOT / "outputs" / args.output_name
    torch.set_num_threads(1)
    summaries = []
    rows = []
    for model_name in args.models:
        summary, model_rows = run_model(
            model_name,
            args.device,
            output_root=output_root,
            force_standardize=args.force_standardize,
            blockwise_standardize=args.blockwise_standardize,
            disable_early_stopping=args.disable_early_stopping,
        )
        summaries.append(summary)
        rows.extend(model_rows)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    atomic_json_save(
        {
            "schema": (
                "standardized-no-bn-no-early-stop-811-v1" if args.disable_early_stopping else
                "no-bn-blockwise-val-loss-811-v1" if args.blockwise_standardize else
                "no-bn-standardized-val-loss-811-v1" if args.force_standardize else
                "no-bn-val-loss-811-v1"
            ),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "models": args.models,
            "summaries": summaries,
        },
        output_root / "summary.json",
    )
    write_csv(rows, output_root / "seed_metrics.csv")


if __name__ == "__main__":
    main()
