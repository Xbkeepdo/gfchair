#!/usr/bin/env python3
"""Train native baselines plus fixed and searched shared MLPs on ENDAC-811 features."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from detection.baselines import (
    SVARMLP,
    build_metatoken_classifier,
    evaluate_detection_scores,
    raw_labels_to_hallucination_targets,
    select_detection_threshold,
    sklearn_hallucination_scores,
    torch_hallucination_scores,
    train_torch_detector,
)
from scripts.search_single_mlp_811 import SingleMLP, candidates, fit, predict, transform
from scripts.train_torch_probe_feature_sets import TorchProbeConfig, train_and_evaluate_probe
from utils.config_utils import load_config
from utils.io_utils import load_pkl

MODELS = ("qwen2_5_vl_7b", "llava_1_5_7b", "qwen3_vl_8b", "internvl_2_5_8b")
METHODS = ("svar", "metatoken", "visual_only", "all_attention")
SEEDS = (43, 44, 45)
DEFAULT_CONFIG = ROOT / "configs/model_configs_endac_811.yaml"


def _save_torch(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _output_root(config: dict) -> Path:
    path = Path(config["endac_811"]["output_root"])
    return path if path.is_absolute() else ROOT / path


def scores(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    return {
        "AUROC": float(roc_auc_score(labels, probabilities)),
        "HALL_AUPR": float(average_precision_score(1 - labels, 1 - probabilities)),
    }


def load_data(model: str, config: dict) -> dict:
    rows = load_pkl(_output_root(config) / model / "features.pkl")
    selected_all = str(config["endac_811"]["models"][model]["all_attention_group"])
    all_key = "all_attention_v" if selected_all == "v" else "all_attention_vp_generation"
    groups = {
        "svar": np.stack([row["features"]["svar"] for row in rows]).astype(np.float32),
        "metatoken": np.stack([row["features"]["metatoken"] for row in rows]).astype(np.float32),
        "visual_only": np.stack([row["features"]["visual_only"] for row in rows]).astype(np.float32),
        "all_attention": np.stack([row["features"][all_key] for row in rows]).astype(np.float32),
    }
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    masks = {
        name: np.asarray([row["split"] == name for row in rows], dtype=bool)
        for name in ("train", "val", "test")
    }
    if any(not np.isfinite(matrix).all() for matrix in groups.values()):
        raise ValueError(f"{model}: non-finite training features")
    if np.any(sum(mask.astype(np.int8) for mask in masks.values()) != 1):
        raise ValueError(f"{model}: feature rows do not have exactly one split")
    for name, mask in masks.items():
        if set(labels[mask]) != {0, 1}:
            raise ValueError(f"{model}: {name} does not contain both labels")
    return {"rows": rows, "groups": groups, "labels": labels, "masks": masks, "all_key": all_key}


def _probability_payload(data: dict, probabilities: dict[str, np.ndarray]) -> dict:
    return {
        f"{name}_probabilities": np.asarray(value, dtype=np.float32)
        for name, value in probabilities.items()
    }


def _save_predictions(path: Path, data: dict, probabilities: dict[str, np.ndarray]) -> None:
    rows = data["rows"]
    masks = data["masks"]
    selected = []
    scores_by_row = []
    for split in ("train", "val", "test"):
        indices = np.flatnonzero(masks[split])
        values = np.asarray(probabilities[split], dtype=np.float32)
        selected.extend(int(index) for index in indices)
        scores_by_row.extend(float(value) for value in values)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        sample_id=np.asarray([rows[index]["sample_id"] for index in selected]),
        image_id=np.asarray([rows[index]["image_id"] for index in selected], dtype=np.int64),
        label=np.asarray([rows[index]["label"] for index in selected], dtype=np.int64),
        split=np.asarray([rows[index]["split"] for index in selected]),
        real_probability=np.asarray(scores_by_row, dtype=np.float32),
    )


def train_native(model: str, data: dict, output: Path, device: str) -> None:
    labels, masks = data["labels"], data["masks"]
    train, val, test = (masks[name] for name in ("train", "val", "test"))
    for head, method in (
        ("svar_native", "svar"),
        ("metatoken_lr", "metatoken"),
        ("metatoken_gb", "metatoken"),
    ):
        matrix = data["groups"][method]
        for seed in SEEDS:
            result_path = output / "results/native" / head / f"seed{seed}" / "result.pt"
            if result_path.exists():
                continue
            if head == "svar_native":
                estimator = SVARMLP(matrix.shape[1], hidden_dim=248)
                trained = train_torch_detector(
                    model=estimator,
                    X_train=matrix[train],
                    raw_y_train=labels[train],
                    X_val=matrix[val],
                    raw_y_val=labels[val],
                    X_test=matrix[test],
                    raw_y_test=labels[test],
                    epochs=50,
                    learning_rate=0.001,
                    batch_size=32,
                    device=device,
                    weight_decay=0.0,
                    weighted_sampler=False,
                    standardize=False,
                    early_stopping_patience=5,
                    seed=seed,
                    positive_class="real",
                )
                hall = {
                    name: torch_hallucination_scores(estimator, matrix[mask], torch.device(device))
                    for name, mask in masks.items()
                }
                payload = {
                    "state_dict": trained.state_dict,
                    "history": trained.history,
                    "threshold": trained.threshold,
                }
            else:
                estimator = build_metatoken_classifier(head.rsplit("_", 1)[1], seed=seed)
                estimator.fit(matrix[train], raw_labels_to_hallucination_targets(labels[train]))
                hall = {
                    name: sklearn_hallucination_scores(estimator, matrix[mask])
                    for name, mask in masks.items()
                }
                threshold = select_detection_threshold(
                    labels[val], hall["val"], positive_class="real"
                )
                payload = {"threshold": threshold}
                model_path = result_path.with_name("model.pkl")
                model_path.parent.mkdir(parents=True, exist_ok=True)
                with model_path.open("wb") as handle:
                    pickle.dump(estimator, handle, protocol=pickle.HIGHEST_PROTOCOL)
            real = {name: 1.0 - value for name, value in hall.items()}
            payload.update(
                seed=seed,
                head=head,
                method=method,
                test_metrics=scores(labels[test], real["test"]),
                threshold_reports={
                    name: evaluate_detection_scores(
                        labels[test], hall["test"], threshold, positive_class="real"
                    )
                    for name, threshold in (
                        ("validation_f1", float(payload["threshold"])),
                        ("fixed_0.5", 0.5),
                    )
                },
                **_probability_payload(data, real),
            )
            _save_torch(result_path, payload)
            _save_predictions(result_path.with_name("predictions.npz"), data, real)
            print(f"DONE {model} native {head} seed{seed} {payload['test_metrics']}", flush=True)


def train_fixed(model: str, data: dict, output: Path, device: str, config: dict) -> None:
    labels, masks = data["labels"], data["masks"]
    settings = config["endac_811"]["fixed_mlp"]
    for method in METHODS:
        matrix = data["groups"][method]
        for seed in SEEDS:
            folder = output / "results/shared_mlp_fixed" / method / f"seed{seed}"
            result_path = folder / "result.pt"
            if result_path.exists():
                continue
            probe_config = TorchProbeConfig(
                hidden_sizes=tuple(map(int, settings["hidden_sizes"])),
                dropout=float(settings["dropout"]),
                drop_last=False,
                batch_size=int(settings["batch_size"]),
                num_epochs=int(settings["max_epochs"]),
                learning_rate=float(settings["learning_rate"]),
                weight_decay=float(settings["weight_decay"]),
                early_stopping_patience=int(settings["early_stopping_patience"]),
                seed=seed,
                positive_class="real",
                split_protocol="image_811_validation",
                checkpoint_selection="minimum_val_loss",
            )
            result = train_and_evaluate_probe(
                X_train=matrix[masks["train"]],
                y_train=labels[masks["train"]],
                X_val=matrix[masks["val"]],
                y_val=labels[masks["val"]],
                X_test=matrix[masks["test"]],
                y_test=labels[masks["test"]],
                config=probe_config,
                device=torch.device(device),
                output_dir=str(folder),
                return_probabilities=True,
            )
            payload = {
                "seed": seed,
                "method": method,
                "test_metrics": scores(labels[masks["test"]], result["test_probabilities"]),
                **result,
            }
            _save_torch(result_path, payload)
            _save_predictions(
                result_path.with_name("predictions.npz"),
                data,
                {
                    "train": result["train_probabilities"],
                    "val": result["validation_probabilities"],
                    "test": result["test_probabilities"],
                },
            )
            print(f"DONE {model} fixed {method} seed{seed} {payload['test_metrics']}", flush=True)


def _rank(value: dict, index: int) -> tuple[float, float, int]:
    return (float(value["AUROC"]), float(value["HALL_AUPR"]), -int(index))


def train_single(model: str, data: dict, output: Path, device: str, config: dict) -> None:
    labels, masks = data["labels"], data["masks"]
    configured = config["endac_811"]["single_mlp"]
    candidate_list = candidates()[: int(configured["candidates"])]
    shortlist_size = int(configured["shortlist"])
    for method in METHODS:
        matrix = data["groups"][method]
        root = output / "results/shared_mlp_single" / method
        seed43 = []
        for index, candidate in enumerate(candidate_list):
            path = root / "search" / f"candidate{index:02d}_seed43.pt"
            if path.exists():
                result = torch.load(path, map_location="cpu", weights_only=False)
            else:
                result = fit(
                    train_x=matrix[masks["train"]],
                    train_y=labels[masks["train"]],
                    val_x=matrix[masks["val"]],
                    val_y=labels[masks["val"]],
                    cfg=candidate,
                    seed=43,
                    device=device,
                )
                result.update(index=index, method=method)
                _save_torch(path, result)
            seed43.append(result)
        shortlist = sorted(
            seed43,
            key=lambda row: _rank(row["validation"], row["index"]),
            reverse=True,
        )[:shortlist_size]
        finalists = []
        for first in shortlist:
            runs = [first]
            index = int(first["index"])
            for seed in (44, 45):
                path = root / "search" / f"candidate{index:02d}_seed{seed}.pt"
                if path.exists():
                    current = torch.load(path, map_location="cpu", weights_only=False)
                else:
                    current = fit(
                        train_x=matrix[masks["train"]],
                        train_y=labels[masks["train"]],
                        val_x=matrix[masks["val"]],
                        val_y=labels[masks["val"]],
                        cfg=candidate_list[index],
                        seed=seed,
                        device=device,
                    )
                    current.update(index=index, method=method)
                    _save_torch(path, current)
                runs.append(current)
            validation = {
                name: float(np.mean([run["validation"][name] for run in runs]))
                for name in ("AUROC", "HALL_AUPR")
            }
            finalists.append({"index": index, "validation": validation, "runs": runs})
        selected = max(finalists, key=lambda row: _rank(row["validation"], row["index"]))
        _save_json(
            root / "selection.json",
            {
                "selected_index": selected["index"],
                "selected_config": candidate_list[selected["index"]],
                "validation": selected["validation"],
                "shortlist": [
                    {"index": row["index"], "validation": row["validation"]}
                    for row in finalists
                ],
            },
        )
        for trained in selected["runs"]:
            seed = int(trained["seed"])
            final_path = root / f"seed{seed}" / "result.pt"
            if final_path.exists():
                continue
            network = SingleMLP(trained["input_dim"], trained["config"]).to(device)
            network.load_state_dict(trained["state_dict"])
            probabilities = {
                name: predict(
                    network,
                    torch.as_tensor(
                        transform(matrix[mask], trained["mean"], trained["scale"]),
                        device=device,
                    ),
                )
                for name, mask in masks.items()
            }
            threshold = select_detection_threshold(
                labels[masks["val"]], 1.0 - probabilities["val"], positive_class="real"
            )
            payload = {
                "seed": seed,
                "method": method,
                "candidate": selected["index"],
                "config": trained["config"],
                "validation": trained["validation"],
                "test_metrics": scores(labels[masks["test"]], probabilities["test"]),
                "threshold": threshold,
                "threshold_reports": {
                    name: evaluate_detection_scores(
                        labels[masks["test"]],
                        1.0 - probabilities["test"],
                        value,
                        positive_class="real",
                    )
                    for name, value in (("validation_f1", threshold), ("fixed_0.5", 0.5))
                },
                **_probability_payload(data, probabilities),
            }
            _save_torch(final_path, payload)
            _save_predictions(final_path.with_name("predictions.npz"), data, probabilities)
            print(f"DONE {model} single {method} seed{seed} {payload['test_metrics']}", flush=True)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(model: str, data: dict, output: Path) -> list[dict]:
    seed_rows, summary_rows = [], []
    heads = [
        ("native", "svar_native", "svar"),
        ("native", "metatoken_lr", "metatoken"),
        ("native", "metatoken_gb", "metatoken"),
        *(('shared_mlp_fixed', method, method) for method in METHODS),
        *(('shared_mlp_single', method, method) for method in METHODS),
    ]
    for family, head, method in heads:
        folder = output / "results" / family / head
        runs = [
            torch.load(folder / f"seed{seed}" / "result.pt", map_location="cpu", weights_only=False)
            for seed in SEEDS
        ]
        for seed, run in zip(SEEDS, runs):
            seed_rows.append(
                {"model": model, "family": family, "head": head, "method": method, "seed": seed, **run["test_metrics"]}
            )
        summary_rows.append(
            {
                "model": model,
                "family": family,
                "head": head,
                "method": method,
                **{
                    f"{metric}_{stat}": float(function([run["test_metrics"][metric] for run in runs]))
                    for metric in ("AUROC", "HALL_AUPR")
                    for stat, function in (("mean", np.mean), ("std", np.std))
                },
            }
        )
    _write_csv(output / "results/seed_metrics.csv", seed_rows)
    _write_csv(output / "results/summary.csv", summary_rows)
    lines = [
        f"# {model} ENDAC-811 results",
        "",
        "3200/400/400 image split; seeds 43/44/45. Scores are mean ± population std.",
        "",
        "| family | head | AUROC (%) | HALL-AUPR (%) |",
        "|---|---|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['family']} | {row['head']} | "
            f"{100 * row['AUROC_mean']:.2f} ± {100 * row['AUROC_std']:.2f} | "
            f"{100 * row['HALL_AUPR_mean']:.2f} ± {100 * row['HALL_AUPR_std']:.2f} |"
        )
    (output / "results/summary.md").write_text("\n".join(lines) + "\n")
    return summary_rows


def summarize_all(config: dict) -> None:
    output_root = _output_root(config)
    rows = []
    for model in MODELS:
        path = output_root / model / "results/summary.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    _write_csv(output_root / "results/summary.csv", rows)
    lines = [
        "# Four-model ENDAC-811 results",
        "",
        "Exact first-canonical mentions; 3200/400/400 image split; seeds 43/44/45.",
        "",
        "| model | family | head | AUROC (%) | HALL-AUPR (%) |",
        "|---|---|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['family']} | {row['head']} | "
            f"{100 * float(row['AUROC_mean']):.2f} ± {100 * float(row['AUROC_std']):.2f} | "
            f"{100 * float(row['HALL_AUPR_mean']):.2f} ± {100 * float(row['HALL_AUPR_std']):.2f} |"
        )
    (output_root / "results/summary.md").parent.mkdir(parents=True, exist_ok=True)
    (output_root / "results/summary.md").write_text("\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> None:
    config = load_config(str(args.config))
    output = _output_root(config) / args.model
    data = load_data(args.model, config)
    if args.stage in {"native", "all"}:
        train_native(args.model, data, output, args.device)
    if args.stage in {"fixed", "all"}:
        train_fixed(args.model, data, output, args.device, config)
    if args.stage in {"single", "all"}:
        train_single(args.model, data, output, args.device, config)
    if args.stage in {"summary", "all"}:
        summarize(args.model, data, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=(*MODELS, "minigpt4_7b", "shikra_7b", "all"), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stage", choices=("native", "fixed", "single", "summary", "all"), default="all")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.model == "all":
        if args.stage != "summary":
            parser.error("--model all is only valid with --stage summary")
        summarize_all(load_config(str(args.config)))
    else:
        run(args)


if __name__ == "__main__":
    main()
