#!/usr/bin/env python3
"""Optuna-search a standardized, no-BN single MLP on four ENDAC models."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import optuna
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.semantic_attention_topk import FEATURE_NAMES
from features.tc_fvpa_artifacts import atomic_json_save, atomic_torch_save
from scripts import search_single_mlp_811 as mlp
from utils.io_utils import load_pkl


MODELS = ("qwen2_5_vl_7b", "llava_1_5_7b", "qwen3_vl_8b", "internvl_2_5_8b")
VARIANTS = ("ae_logs", "raw_top16", "raw_top32", "norm_top16", "norm_top32")
SEEDS = (43, 44, 45)
TRIALS = 24
TOP_N = 3
SOURCE = ROOT / "outputs/coco4000_512_endac_811"
SEMANTIC = ROOT / "outputs/coco4000_512_endac_semantic_attention_topk_detection_811"
OUT = ROOT / "outputs/coco4000_512_endac_four_ae_semantic_optuna_811"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source(model: str) -> dict:
    compact = read(SEMANTIC / model / "features.pt")
    rows = load_pkl(SOURCE / model / "features.pkl")
    if [row["sample_id"] for row in rows] != compact["sample_id"]:
        raise ValueError(f"{model}: feature sample order differs")
    labels = np.asarray(compact["labels"], dtype=np.int64)
    if not np.array_equal(labels, [row["label"] for row in rows]):
        raise ValueError(f"{model}: feature labels differ")
    masks = compact["masks"]
    ae = np.stack([row["features"]["visual_only"] for row in rows]).astype(np.float32)
    log_s = ae[:, 1::2].copy()
    result = {"ae_logs": ae}
    index = FEATURE_NAMES.index("semantic_attention")
    for variant in VARIANTS[1:]:
        semantic_attention = np.asarray(compact["cubes"][variant][:, :, index], dtype=np.float32)
        result[variant] = np.concatenate((semantic_attention, log_s), axis=1)
    if any(not np.isfinite(value).all() for value in result.values()):
        raise ValueError(f"{model}: non-finite feature")
    return dict(matrices=result, labels=labels, masks=masks, compact=compact)


def protocol(model: str) -> dict:
    return dict(
        schema="four-ae-semantic-optuna-811-v1", model=model,
        features={"ae_logs": "interleaved [AE_V,log1p(S_E)]",
                  "semantic_variants": "[sum_Tk probability(object)*attention,log1p(S_E)]"},
        variants=list(VARIANTS), seeds=list(SEEDS), trials_per_variant=TRIALS,
        top_n=TOP_N, seed43_sampler="TPESampler(seed=20260916, n_startup_trials=8)",
        training="train-only z-score; no BatchNorm; single-hidden ReLU/GELU, dropout, Adam, BCE REAL=1",
        ranking="seed43 top3 validation AUROC then HALL-AUPR; seeds44/45; three-seed validation mean",
        split="existing ENDAC 3200/400/400 image split",
        search_space=dict(width=[64, 128, 248, 256, 512, 1024], dropout=[0., .1, .3, .5],
                          activation=["relu", "gelu"], learning_rate=[1e-4, 1e-2],
                          weight_decay=[0., 1e-6, 1e-5, 1e-4, 1e-3],
                          batch_size=[64, 128, 256, 512], monitor=["val_loss", "val_auroc"]),
        optuna_version=optuna.__version__,
        source=str(SOURCE / model / "features.pkl"),
        semantic_source=str(SEMANTIC / model / "features.pt"),
        caveat="Prior experiments have examined this held-out test split; exploratory comparison only.",
    )


def progress(model: str, stage: str, completed: int, total: int) -> None:
    atomic_json_save(dict(model=model, stage=stage, completed=completed, total=total,
                          updated_at=now()), OUT / model / "progress.json")


def config(trial: optuna.Trial) -> dict:
    return dict(width=trial.suggest_categorical("width", [64, 128, 248, 256, 512, 1024]),
                standardize=True, batch_norm=False,
                dropout=trial.suggest_categorical("dropout", [0., .1, .3, .5]),
                activation=trial.suggest_categorical("activation", ["relu", "gelu"]),
                learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
                weight_decay=trial.suggest_categorical("weight_decay", [0., 1e-6, 1e-5, 1e-4, 1e-3]),
                batch_size=trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
                max_epochs=150, patience=20, lr_patience=6,
                monitor=trial.suggest_categorical("monitor", ["val_loss", "val_auroc"]))


def trial_path(model: str, variant: str, number: int) -> Path:
    return OUT / model / "search" / variant / f"trial{number:02d}_seed43.pt"


def finalist_path(model: str, variant: str, number: int, seed: int) -> Path:
    return OUT / model / "finalists" / variant / f"trial{number:02d}_seed{seed}.pt"


def rank(row: dict) -> tuple[float, float, int]:
    return (float(row["validation"]["AUROC"]), float(row["validation"]["HALL_AUPR"]),
            -int(row["number"]))


def train_model(model: str, device: str) -> None:
    data = source(model)
    root = OUT / model
    setup = protocol(model)
    setup_path = root / "protocol.json"
    if setup_path.exists() and json.loads(setup_path.read_text()) != setup:
        raise ValueError(f"Changed protocol: {setup_path}")
    atomic_json_save(setup, setup_path)
    labels, masks = data["labels"], data["masks"]
    total = len(VARIANTS) * (TRIALS + TOP_N * 2)
    completed = 0
    choices = {}
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for variant in VARIANTS:
        matrix = data["matrices"][variant]
        database = root / "studies" / f"{variant}.db"
        database.parent.mkdir(parents=True, exist_ok=True)
        study = optuna.create_study(
            study_name=f"{model}_{variant}", storage=f"sqlite:///{database}",
            load_if_exists=True, direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=20260916, n_startup_trials=8),
            pruner=optuna.pruners.NopPruner(),
        )
        if not study.trials:
            study.enqueue_trial(dict(width=128, dropout=.3, activation="relu",
                                     learning_rate=.001, weight_decay=1e-5,
                                     batch_size=128, monitor="val_loss"))
        completed += len([trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE])

        def objective(trial: optuna.Trial) -> float:
            nonlocal completed
            number = trial.number
            progress(model, f"{variant}/trial{number:02d}", completed, total)
            trained = mlp.fit(matrix[masks["train"]], labels[masks["train"]],
                              matrix[masks["val"]], labels[masks["val"]],
                              config(trial), 43, device)
            trained.update(model=model, variant=variant, number=number)
            atomic_torch_save(trained, trial_path(model, variant, number))
            trial.set_user_attr("HALL_AUPR", trained["validation"]["HALL_AUPR"])
            completed += 1
            progress(model, f"{variant}/trial{number:02d}", completed, total)
            print("TRIAL", model, variant, number, trained["validation"], flush=True)
            return float(trained["validation"]["AUROC"])

        remaining = TRIALS - len([trial for trial in study.trials
                                  if trial.state == optuna.trial.TrialState.COMPLETE])
        if remaining > 0:
            study.optimize(objective, n_trials=remaining, n_jobs=1)
        completed_trials = [trial for trial in study.trials
                            if trial.state == optuna.trial.TrialState.COMPLETE]
        if len(completed_trials) != TRIALS:
            raise ValueError(f"{model}/{variant}: {len(completed_trials)} completed trials")
        first = [dict(number=trial.number, validation=read(trial_path(model, variant, trial.number))["validation"])
                 for trial in completed_trials]
        shortlist = sorted(first, key=rank, reverse=True)[:TOP_N]
        finalists = []
        for entry in shortlist:
            number = entry["number"]
            seed43 = read(trial_path(model, variant, number))
            values = [entry["validation"]]
            for seed in (44, 45):
                saved = finalist_path(model, variant, number, seed)
                if saved.exists():
                    trained = read(saved)
                else:
                    progress(model, f"{variant}/trial{number:02d}/seed{seed}", completed, total)
                    trained = mlp.fit(matrix[masks["train"]], labels[masks["train"]],
                                      matrix[masks["val"]], labels[masks["val"]],
                                      seed43["config"], seed, device)
                    trained.update(model=model, variant=variant, number=number)
                    atomic_torch_save(trained, saved)
                values.append(trained["validation"])
                completed += 1
                progress(model, f"{variant}/trial{number:02d}/seed{seed}", completed, total)
                print("FINALIST", model, variant, number, seed, trained["validation"], flush=True)
            finalists.append(dict(number=number,
                                  validation={metric: float(np.mean([value[metric] for value in values]))
                                              for metric in ("AUROC", "HALL_AUPR")}))
        choices[variant] = max(finalists, key=rank)
    champion = max(enumerate(VARIANTS), key=lambda item:
                   (*rank(choices[item[1]])[:2], -item[0]))[1]
    atomic_json_save(dict(model=model, variants=choices, champion=champion,
                          frozen_at=now(), ranking="validation AUROC then HALL-AUPR"),
                     root / "selection.json")
    progress(model, "validation_complete", total, total)


def selected_result(model: str, variant: str, number: int, seed: int):
    return read(trial_path(model, variant, number) if seed == 43 else
                finalist_path(model, variant, number, seed))


def test_model(model: str, device: str) -> None:
    if not all((OUT / name / "selection.json").exists() for name in MODELS):
        raise FileNotFoundError("Freeze all four validation selections before test")
    data = source(model)
    root = OUT / model
    selection = json.loads((root / "selection.json").read_text())
    labels, masks = data["labels"], data["masks"]
    rows = []
    for variant in VARIANTS:
        matrix = data["matrices"][variant]
        number = selection["variants"][variant]["number"]
        for seed in SEEDS:
            trained = selected_result(model, variant, number, seed)
            network = mlp.SingleMLP(matrix.shape[1], trained["config"]).to(device)
            network.load_state_dict(trained["state_dict"])
            test_x = torch.as_tensor(mlp.transform(matrix[masks["test"]],
                                                   trained["mean"], trained["scale"]), device=device)
            probability = mlp.predict(network, test_x)
            metrics = mlp.metrics(labels[masks["test"]], probability)
            path = root / "final" / variant / f"seed{seed}.pt"
            atomic_torch_save(dict(model=model, variant=variant, number=number, seed=seed,
                                   validation=trained["validation"], test_metrics=metrics,
                                   test_probabilities=probability), path)
            full = np.empty(len(labels), dtype=np.float32)
            for split, values in (("train", trained["train_probabilities"]),
                                  ("val", trained["validation_probabilities"]),
                                  ("test", probability)):
                full[masks[split]] = values
            np.savez_compressed(path.with_name(f"seed{seed}_predictions.npz"),
                                sample_id=np.asarray(data["compact"]["sample_id"]),
                                image_id=data["compact"]["image_id"], label=labels,
                                split=np.asarray(["train" if masks["train"][i] else
                                                  "val" if masks["val"][i] else "test"
                                                  for i in range(len(labels))]),
                                real_probability=full)
            rows.append(dict(model=model, variant=variant, seed=seed, **metrics))
            print("TEST", model, variant, seed, metrics, flush=True)
    write_csv(root / "seed_metrics.csv", rows)
    summary = []
    for variant in VARIANTS:
        selected = [row for row in rows if row["variant"] == variant]
        summary.append(dict(model=model, variant=variant,
                            selected=(variant == selection["champion"]),
                            trial=selection["variants"][variant]["number"],
                            validation_AUROC=selection["variants"][variant]["validation"]["AUROC"],
                            validation_HALL_AUPR=selection["variants"][variant]["validation"]["HALL_AUPR"],
                            **{f"{metric}_{stat}": float(function([row[metric] for row in selected]))
                               for metric in ("AUROC", "HALL_AUPR")
                               for stat, function in (("mean", np.mean), ("std", np.std))}))
    write_csv(root / "summary.csv", summary)
    progress(model, "complete", len(VARIANTS) * len(SEEDS), len(VARIANTS) * len(SEEDS))


def summarize() -> None:
    rows = []
    for model in MODELS:
        with (OUT / model / "summary.csv").open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    write_csv(OUT / "summary.csv", rows)
    selected_configs = []
    for row in rows:
        model, variant, number = row["model"], row["variant"], int(row["trial"])
        trained = [selected_result(model, variant, number, seed) for seed in SEEDS]
        cfg = trained[0]["config"]
        selected_configs.append(dict(
            model=model, variant=variant, model_champion=row["selected"], trial=number,
            width=cfg["width"], dropout=cfg["dropout"], activation=cfg["activation"],
            learning_rate=cfg["learning_rate"], weight_decay=cfg["weight_decay"],
            batch_size=cfg["batch_size"], monitor=cfg["monitor"],
            best_epochs="/".join(str(run["best_epoch"]) for run in trained),
        ))
    write_csv(OUT / "selected_configs.csv", selected_configs)
    lines = ["# Four-model AE vs semantic×attention + log1p(S_E): Optuna single MLP",
             "", "Train-only standardization, no BN; TPE 24 trials per feature;",
             "seed43 shortlist top3, seeds44/45 validation selection, then test.",
             "3200/400/400 image split; AUROC/HALL-AUPR are three-seed test mean ± population std.",
             "", "| Model | Feature | Trial | Val AUROC (%) | Test AUROC (%) | Test HALL-AUPR (%) |",
             "|---|---|---:|---:|---:|---:|"]
    for row in rows:
        marker = " (selected)" if row["selected"] == "True" else ""
        lines.append(f"| {row['model']} | {row['variant']}{marker} | {row['trial']} | "
                     f"{100*float(row['validation_AUROC']):.2f} | "
                     f"{100*float(row['AUROC_mean']):.2f} ± {100*float(row['AUROC_std']):.2f} | "
                     f"{100*float(row['HALL_AUPR_mean']):.2f} ± "
                     f"{100*float(row['HALL_AUPR_std']):.2f} |")
    lines.extend(["", "## Validation-selected hyperparameters", "",
                  "All rows use train-only standardization and no BatchNorm.", "",
                  "| Model | Feature | Width | Dropout | Activation | LR | WD | Batch | Monitor | Best epochs 43/44/45 |",
                  "|---|---|---:|---:|---|---:|---:|---:|---|---|"])
    for row in selected_configs:
        marker = " (selected)" if row["model_champion"] == "True" else ""
        lines.append(f"| {row['model']} | {row['variant']}{marker} | {row['width']} | "
                     f"{row['dropout']:.1f} | {row['activation']} | "
                     f"{row['learning_rate']:.3g} | {row['weight_decay']:.1g} | "
                     f"{row['batch_size']} | {row['monitor']} | {row['best_epochs']} |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate() -> None:
    checked = 0
    error = 0.
    cpu_metric_error = 0.
    for model in MODELS:
        data = source(model)
        selection = json.loads((OUT / model / "selection.json").read_text())
        for variant in VARIANTS:
            matrix = data["matrices"][variant]
            number = selection["variants"][variant]["number"]
            for seed in SEEDS:
                trained = selected_result(model, variant, number, seed)
                final = read(OUT / model / "final" / variant / f"seed{seed}.pt")
                mean, scale = mlp.scale_fit(matrix[data["masks"]["train"]], True)
                np.testing.assert_array_equal(trained["mean"], mean)
                np.testing.assert_array_equal(trained["scale"], scale)
                network = mlp.SingleMLP(matrix.shape[1], trained["config"])
                network.load_state_dict(trained["state_dict"])
                test_x = torch.as_tensor(mlp.transform(matrix[data["masks"]["test"]], mean, scale))
                recomputed = mlp.predict(network, test_x)
                error = max(error, float(np.max(np.abs(recomputed - final["test_probabilities"]))))
                test_labels = data["labels"][data["masks"]["test"]]
                for key, value in mlp.metrics(test_labels, final["test_probabilities"]).items():
                    if abs(value - final["test_metrics"][key]) > 1e-8:
                        raise ValueError(f"{model}/{variant}/seed{seed}: metric differs")
                for key, value in mlp.metrics(test_labels, recomputed).items():
                    cpu_metric_error = max(cpu_metric_error, abs(value - final["test_metrics"][key]))
                checked += 1
    if error > 1e-6 or cpu_metric_error > 1e-3:
        raise ValueError(f"CPU reload differs: probability={error}, metric={cpu_metric_error}")
    report = dict(status="pass", checked_heads=checked,
                  expected_heads=len(MODELS)*len(VARIANTS)*len(SEEDS),
                  max_cpu_reload_probability_error=error,
                  max_cpu_reload_metric_error=cpu_metric_error,
                  saved_probability_metric_error=0., validated_at=now())
    atomic_json_save(report, OUT / "validation.json")
    print(json.dumps(report, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=(*MODELS, "all"), required=True)
    parser.add_argument("--stage", choices=("train", "test", "summary", "validate"), required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.stage == "summary":
        summarize()
    elif args.stage == "validate":
        validate()
    elif args.model == "all":
        parser.error("--model all only works for summary/validate")
    elif args.stage == "train":
        train_model(args.model, args.device)
    else:
        test_model(args.model, args.device)


if __name__ == "__main__":
    main()
