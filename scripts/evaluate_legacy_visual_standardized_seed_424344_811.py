"""Force train-only standardization on frozen Visual-only MLPs for seeds 42/43/44."""

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.tc_fvpa_artifacts import atomic_json_save, atomic_torch_save
from scripts import evaluate_legacy_visual_seed_424344_811 as frozen
from scripts import search_legacy_visual_single_mlp_811 as legacy
from scripts import search_single_mlp_811 as search


OUT = ROOT / "outputs/legacy_visual_standardized_seed424344_811_v1"
BASELINE_OUT = frozen.OUT
STANDARDIZED_SVAR_OUT = ROOT / "outputs/svar_proportional_standardized_seed424344_811_v1"
SOURCE_OUT = legacy.OUT
MODELS = tuple(legacy.MODELS)
SEEDS = (42, 43, 44)


def now():
    return datetime.now(timezone.utc).isoformat()


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def standardized_config(selection):
    return {**selection["config"], "standardize": True}


def candidate_index(config):
    return search.candidates().index(config)


def existing_training_path(model, selected_candidate, candidate, seed):
    if candidate == selected_candidate:
        path = frozen.training_path(model, candidate, seed)
        if path.exists():
            return path
    path = SOURCE_OUT / model / "search" / f"candidate{candidate:02d}" / f"seed{seed}" / "result.pt"
    return path if path.exists() else None


def make_protocol(model, source, selection, config, candidate):
    already_standardized = bool(selection["config"]["standardize"])
    protocol = {
        "schema": "legacy-visual-forced-standardized-seed424344-811-v1",
        "model": model,
        "feature": "legacy_visual=[AE_V, log1p(S_E)]",
        "seeds": list(SEEDS),
        "changed_factor": (
            "none; selected config already standardized"
            if already_standardized
            else "selected config standardize false -> true"
        ),
        "source_selected_candidate": selection["index"],
        "standardized_candidate": candidate,
        "source_config": selection["config"],
        "standardized_config": config,
        "source_selection_fingerprint": selection["fingerprint"],
        "source_matrix_fingerprint": source["fingerprint"],
        "split": "unchanged image-level 3200/400/400 split; no train+validation refit",
        "selection": "no hyperparameter reselection and no test-based choice",
        "standardization": "column-wise mean/std fit on train mentions only",
        "caveat": "Exploratory ablation after fixed test access.",
    }
    protocol["fingerprint"] = hashlib.sha256(
        json.dumps(protocol, sort_keys=True).encode()
    ).hexdigest()
    return protocol


def train_or_reuse(model, selection, config, candidate, seed, train_x, train_y, val_x, val_y, device, protocol):
    existing = existing_training_path(model, selection["index"], candidate, seed)
    if existing is not None:
        trained = search.read(existing)
        reused = True
        path = existing
    else:
        path = OUT / model / "training" / f"seed{seed}" / "result.pt"
        if path.exists():
            trained = search.read(path)
            if trained["fingerprint"] != protocol["fingerprint"]:
                raise ValueError(f"Training fingerprint mismatch for {model} seed{seed}")
        else:
            trained = search.fit(train_x, train_y, val_x, val_y, config, seed, device)
            trained.update(
                fingerprint=protocol["fingerprint"],
                source_selection_fingerprint=selection["fingerprint"],
                index=candidate,
                feature=legacy.GROUP,
            )
            atomic_torch_save(trained, path)
        reused = False
    if trained["config"] != config or trained["input_dim"] != train_x.shape[1]:
        raise ValueError(f"Config/input mismatch for {model} seed{seed}")
    return trained, path, reused


def run_model(model, device):
    source, matrix, y, masks = legacy.load_source(model)
    selection = json.loads((SOURCE_OUT / model / "selection.json").read_text())
    config = standardized_config(selection)
    candidate = candidate_index(config)
    protocol = make_protocol(model, source, selection, config, candidate)
    root = OUT / model
    protocol_path = root / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError(f"Protocol changed for {model}")
    else:
        atomic_json_save(protocol, protocol_path)

    train_x, train_y = matrix[masks["train"]], y[masks["train"]]
    val_x, val_y = matrix[masks["validation"]], y[masks["validation"]]
    test_x, test_y = matrix[masks["test"]], y[masks["test"]]
    rows = []
    probabilities = []
    for seed in SEEDS:
        trained, path, reused = train_or_reuse(
            model,
            selection,
            config,
            candidate,
            seed,
            train_x,
            train_y,
            val_x,
            val_y,
            device,
            protocol,
        )
        network = search.SingleMLP(trained["input_dim"], config).to(device)
        network.load_state_dict(trained["state_dict"])
        probability = search.predict(
            network,
            torch.as_tensor(
                search.transform(test_x, trained["mean"], trained["scale"]),
                device=device,
            ),
        )
        metrics = search.metrics(test_y, probability)
        final = {
            "schema": "legacy-visual-forced-standardized-final-v1",
            "model": model,
            "seed": seed,
            "candidate": candidate,
            "config": config,
            "protocol_fingerprint": protocol["fingerprint"],
            "training_source": str(path.relative_to(ROOT)),
            "reused_training": reused,
            "best_epoch": trained["best_epoch"],
            "validation": trained["validation"],
            "test_metrics": metrics,
            "test_probabilities": probability,
            "test_labels": test_y,
        }
        atomic_torch_save(final, root / "final" / f"seed{seed}" / "result.pt")
        rows.append(
            {
                "model": model,
                "seed": seed,
                "standardization_was_already_enabled": selection["config"]["standardize"],
                "reused_training": reused,
                "best_epoch": trained["best_epoch"],
                **metrics,
            }
        )
        probabilities.append(probability)

    summary = {
        "schema": "legacy-visual-forced-standardized-model-summary-v1",
        "model": model,
        "source_candidate": selection["index"],
        "standardized_candidate": candidate,
        "source_config": selection["config"],
        "config": config,
        "standardization_was_already_enabled": selection["config"]["standardize"],
        "seeds": list(SEEDS),
        **{
            f"{key}_{stat}": float(fn([row[key] for row in rows]))
            for key in ("AUROC", "HALL_AUPR")
            for stat, fn in (("mean", np.mean), ("std", np.std))
        },
        "ensemble": search.metrics(test_y, np.mean(probabilities, axis=0)),
    }
    write_csv(rows, root / "seed_metrics.csv")
    atomic_json_save(summary, root / "summary.json")
    atomic_json_save(
        {"stage": "complete", "status": "completed", "completed": 3, "total": 3, "heartbeat": now()},
        root / "progress.json",
    )


def summarize():
    rows = []
    summaries = []
    for model in MODELS:
        standardized = json.loads((OUT / model / "summary.json").read_text())
        baseline = json.loads((BASELINE_OUT / model / "summary.json").read_text())
        standardized_svar = json.loads(
            (STANDARDIZED_SVAR_OUT / model / "summary.json").read_text()
        )
        row = {
            "model": model,
            "already_standardized": standardized["standardization_was_already_enabled"],
            "standardized_AUROC_mean": standardized["AUROC_mean"],
            "standardized_AUROC_std": standardized["AUROC_std"],
            "standardized_HALL_AUPR_mean": standardized["HALL_AUPR_mean"],
            "standardized_HALL_AUPR_std": standardized["HALL_AUPR_std"],
            "baseline_AUROC_mean": baseline["AUROC_mean"],
            "baseline_AUROC_std": baseline["AUROC_std"],
            "baseline_HALL_AUPR_mean": baseline["HALL_AUPR_mean"],
            "baseline_HALL_AUPR_std": baseline["HALL_AUPR_std"],
            "AUROC_delta_pp": 100 * (standardized["AUROC_mean"] - baseline["AUROC_mean"]),
            "HALL_AUPR_delta_pp": 100
            * (standardized["HALL_AUPR_mean"] - baseline["HALL_AUPR_mean"]),
            "standardized_svar_AUROC_mean": standardized_svar["AUROC_mean"],
            "standardized_svar_HALL_AUPR_mean": standardized_svar["HALL_AUPR_mean"],
            "vs_standardized_svar_AUROC_pp": 100
            * (standardized["AUROC_mean"] - standardized_svar["AUROC_mean"]),
            "vs_standardized_svar_HALL_AUPR_pp": 100
            * (standardized["HALL_AUPR_mean"] - standardized_svar["HALL_AUPR_mean"]),
        }
        rows.append(row)
        summaries.append(standardized)
    macro = {
        key: float(np.mean([row[key] for row in rows]))
        for key in (
            "standardized_AUROC_mean",
            "standardized_HALL_AUPR_mean",
            "baseline_AUROC_mean",
            "baseline_HALL_AUPR_mean",
            "AUROC_delta_pp",
            "HALL_AUPR_delta_pp",
            "standardized_svar_AUROC_mean",
            "standardized_svar_HALL_AUPR_mean",
            "vs_standardized_svar_AUROC_pp",
            "vs_standardized_svar_HALL_AUPR_pp",
        )
    }
    write_csv(rows, OUT / "comparison.csv")
    atomic_json_save(
        {
            "schema": "legacy-visual-forced-standardized-seed424344-summary-v1",
            "summaries": summaries,
            "comparison": rows,
            "macro_average": macro,
        },
        OUT / "summary.json",
    )
    titles = {
        "qwen2_5_vl_7b": "Qwen2.5-VL-7B",
        "llava_1_5_7b": "LLaVA-1.5-7B",
        "qwen3_vl_8b": "Qwen3-VL-8B",
        "internvl_2_5_8b": "InternVL2.5-8B",
    }
    lines = [
        "# Visual-only冻结单层MLP统一标准化（固定811，seeds42/43/44）",
        "",
        "Qwen2.5/LLaVA仅把冻结配置的standardize由false改为true；Qwen3/InternVL原配置已为true，直接复用。",
        "",
        "| 模型 | 统一标准化 AUROC/AP | 原冻结配置 AUROC/AP | 变化 AUROC/AP (pp) |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {titles[row['model']]} | "
            f"{100*row['standardized_AUROC_mean']:.2f}±{100*row['standardized_AUROC_std']:.2f} / "
            f"{100*row['standardized_HALL_AUPR_mean']:.2f}±{100*row['standardized_HALL_AUPR_std']:.2f} | "
            f"{100*row['baseline_AUROC_mean']:.2f}±{100*row['baseline_AUROC_std']:.2f} / "
            f"{100*row['baseline_HALL_AUPR_mean']:.2f}±{100*row['baseline_HALL_AUPR_std']:.2f} | "
            f"{row['AUROC_delta_pp']:+.2f} / {row['HALL_AUPR_delta_pp']:+.2f} |"
        )
    lines.append(
        f"| 四模型宏平均 | {100*macro['standardized_AUROC_mean']:.2f} / "
        f"{100*macro['standardized_HALL_AUPR_mean']:.2f} | "
        f"{100*macro['baseline_AUROC_mean']:.2f} / {100*macro['baseline_HALL_AUPR_mean']:.2f} | "
        f"{macro['AUROC_delta_pp']:+.2f} / {macro['HALL_AUPR_delta_pp']:+.2f} |"
    )
    lines.extend(
        [
            "",
            "| 模型 | Visual-only统一标准化 | SVAR标准化 | Visual-only−SVAR (AUROC/AP pp) |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {titles[row['model']]} | {100*row['standardized_AUROC_mean']:.2f} / "
            f"{100*row['standardized_HALL_AUPR_mean']:.2f} | "
            f"{100*row['standardized_svar_AUROC_mean']:.2f} / "
            f"{100*row['standardized_svar_HALL_AUPR_mean']:.2f} | "
            f"{row['vs_standardized_svar_AUROC_pp']:+.2f} / "
            f"{row['vs_standardized_svar_HALL_AUPR_pp']:+.2f} |"
        )
    lines.append(
        f"| 四模型宏平均 | {100*macro['standardized_AUROC_mean']:.2f} / "
        f"{100*macro['standardized_HALL_AUPR_mean']:.2f} | "
        f"{100*macro['standardized_svar_AUROC_mean']:.2f} / "
        f"{100*macro['standardized_svar_HALL_AUPR_mean']:.2f} | "
        f"{macro['vs_standardized_svar_AUROC_pp']:+.2f} / "
        f"{macro['vs_standardized_svar_HALL_AUPR_pp']:+.2f} |"
    )
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")


def validate():
    checked = 0
    max_probability_error = 0.0
    max_mean_error = 0.0
    max_scale_error = 0.0
    new_training_fits = 0
    reused_training_fits = 0
    for model in MODELS:
        _, matrix, y, masks = legacy.load_source(model)
        protocol = json.loads((OUT / model / "protocol.json").read_text())
        expected_mean, expected_scale = search.scale_fit(matrix[masks["train"]], True)
        for seed in SEEDS:
            final = search.read(OUT / model / "final" / f"seed{seed}" / "result.pt")
            trained = search.read(ROOT / final["training_source"])
            if final["protocol_fingerprint"] != protocol["fingerprint"]:
                raise AssertionError((model, seed, "fingerprint"))
            if not trained["config"]["standardize"]:
                raise AssertionError((model, seed, "standardize"))
            max_mean_error = max(
                max_mean_error, float(np.max(np.abs(trained["mean"] - expected_mean)))
            )
            max_scale_error = max(
                max_scale_error, float(np.max(np.abs(trained["scale"] - expected_scale)))
            )
            network = search.SingleMLP(trained["input_dim"], trained["config"])
            network.load_state_dict(trained["state_dict"])
            probability = search.predict(
                network,
                torch.as_tensor(
                    search.transform(matrix[masks["test"]], trained["mean"], trained["scale"])
                ),
            )
            max_probability_error = max(
                max_probability_error,
                float(np.max(np.abs(probability - final["test_probabilities"]))),
            )
            metrics = search.metrics(y[masks["test"]], probability)
            for key, value in metrics.items():
                if abs(value - final["test_metrics"][key]) > 1e-12:
                    raise AssertionError((model, seed, key))
            reused_training_fits += int(final["reused_training"])
            new_training_fits += int(not final["reused_training"])
            checked += 1
    report = {
        "schema": "legacy-visual-forced-standardized-seed424344-validation-v1",
        "status": "PASS",
        "checked_heads": checked,
        "new_training_fits": new_training_fits,
        "reused_training_fits": reused_training_fits,
        "max_probability_abs_error": max_probability_error,
        "max_train_mean_abs_error": max_mean_error,
        "max_train_scale_abs_error": max_scale_error,
        "validated_at": now(),
    }
    atomic_json_save(report, OUT / "validation.json")
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("run", "summarize", "validate"), required=True)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.stage == "run":
        if args.model is None:
            parser.error("--model is required for --stage run")
        root = OUT / args.model
        root.mkdir(parents=True, exist_ok=True)
        with (root / ".lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run_model(args.model, args.device)
    elif args.stage == "summarize":
        summarize()
    else:
        validate()


if __name__ == "__main__":
    main()
