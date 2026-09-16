"""Evaluate standardized Visual-only frozen MLP configs after removing BatchNorm."""

import argparse
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
from scripts import evaluate_legacy_visual_standardized_seed_424344_811 as standardized


OUT = ROOT / "outputs/legacy_visual_standardized_no_bn_seed424344_811_v1"
BASELINE_OUT = standardized.OUT
STANDARDIZED_SVAR_OUT = standardized.STANDARDIZED_SVAR_OUT
SOURCE_OUT = standardized.SOURCE_OUT
MODELS = standardized.MODELS
SEEDS = standardized.SEEDS
legacy = standardized.legacy
search = standardized.search
frozen = standardized.frozen


def target_config(selection):
    return {**standardized.standardized_config(selection), "batch_norm": False}


def make_protocol(model, source, selection, baseline_config, config):
    already_no_bn = not baseline_config["batch_norm"]
    protocol = {
        "schema": "legacy-visual-standardized-no-bn-seed424344-811-v1",
        "model": model,
        "feature": "legacy_visual=[AE_V, log1p(S_E)]",
        "seeds": list(SEEDS),
        "changed_factor": (
            "none; standardized baseline already has batch_norm=false"
            if already_no_bn
            else "relative to standardized baseline: batch_norm true -> false"
        ),
        "source_selected_candidate": selection["index"],
        "source_config": selection["config"],
        "comparison_baseline_config": baseline_config,
        "target_config": config,
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


def training_result(model, selection, config, seed, train_x, train_y, val_x, val_y, device, protocol):
    already_no_bn = bool(
        selection["config"]["standardize"] and not selection["config"]["batch_norm"]
    )
    if already_no_bn:
        path = frozen.training_path(model, selection["index"], seed)
        trained = search.read(path)
        reused = True
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
    baseline_config = json.loads((BASELINE_OUT / model / "summary.json").read_text())["config"]
    config = target_config(selection)
    if {**baseline_config, "batch_norm": False} != config:
        raise ValueError(f"Unexpected non-BN config difference for {model}")
    protocol = make_protocol(model, source, selection, baseline_config, config)
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
        trained, path, reused = training_result(
            model,
            selection,
            config,
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
            "schema": "legacy-visual-standardized-no-bn-final-v1",
            "model": model,
            "seed": seed,
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
                "bn_was_already_disabled": not selection["config"]["batch_norm"],
                "reused_training": reused,
                "best_epoch": trained["best_epoch"],
                **metrics,
            }
        )
        probabilities.append(probability)

    summary = {
        "schema": "legacy-visual-standardized-no-bn-model-summary-v1",
        "model": model,
        "source_config": selection["config"],
        "config": config,
        "bn_was_already_disabled": not selection["config"]["batch_norm"],
        "seeds": list(SEEDS),
        **{
            f"{key}_{stat}": float(fn([row[key] for row in rows]))
            for key in ("AUROC", "HALL_AUPR")
            for stat, fn in (("mean", np.mean), ("std", np.std))
        },
        "ensemble": search.metrics(test_y, np.mean(probabilities, axis=0)),
    }
    standardized.write_csv(rows, root / "seed_metrics.csv")
    atomic_json_save(summary, root / "summary.json")
    atomic_json_save(
        {
            "stage": "complete",
            "status": "completed",
            "completed": 3,
            "total": 3,
            "heartbeat": standardized.now(),
        },
        root / "progress.json",
    )


def summarize():
    rows = []
    summaries = []
    for model in MODELS:
        no_bn = json.loads((OUT / model / "summary.json").read_text())
        with_bn = json.loads((BASELINE_OUT / model / "summary.json").read_text())
        svar = json.loads((STANDARDIZED_SVAR_OUT / model / "summary.json").read_text())
        row = {
            "model": model,
            "bn_was_already_disabled": no_bn["bn_was_already_disabled"],
            "no_bn_AUROC_mean": no_bn["AUROC_mean"],
            "no_bn_AUROC_std": no_bn["AUROC_std"],
            "no_bn_HALL_AUPR_mean": no_bn["HALL_AUPR_mean"],
            "no_bn_HALL_AUPR_std": no_bn["HALL_AUPR_std"],
            "with_bn_AUROC_mean": with_bn["AUROC_mean"],
            "with_bn_HALL_AUPR_mean": with_bn["HALL_AUPR_mean"],
            "AUROC_delta_pp": 100 * (no_bn["AUROC_mean"] - with_bn["AUROC_mean"]),
            "HALL_AUPR_delta_pp": 100
            * (no_bn["HALL_AUPR_mean"] - with_bn["HALL_AUPR_mean"]),
            "standardized_svar_AUROC_mean": svar["AUROC_mean"],
            "standardized_svar_HALL_AUPR_mean": svar["HALL_AUPR_mean"],
            "vs_standardized_svar_AUROC_pp": 100
            * (no_bn["AUROC_mean"] - svar["AUROC_mean"]),
            "vs_standardized_svar_HALL_AUPR_pp": 100
            * (no_bn["HALL_AUPR_mean"] - svar["HALL_AUPR_mean"]),
        }
        rows.append(row)
        summaries.append(no_bn)
    macro = {
        key: float(np.mean([row[key] for row in rows]))
        for key in (
            "no_bn_AUROC_mean",
            "no_bn_HALL_AUPR_mean",
            "with_bn_AUROC_mean",
            "with_bn_HALL_AUPR_mean",
            "AUROC_delta_pp",
            "HALL_AUPR_delta_pp",
            "standardized_svar_AUROC_mean",
            "standardized_svar_HALL_AUPR_mean",
            "vs_standardized_svar_AUROC_pp",
            "vs_standardized_svar_HALL_AUPR_pp",
        )
    }
    standardized.write_csv(rows, OUT / "comparison.csv")
    atomic_json_save(
        {
            "schema": "legacy-visual-standardized-no-bn-seed424344-summary-v1",
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
        "# Visual-only统一标准化后去掉BN（固定811，seeds42/43/44）",
        "",
        "保持统一train-only z-score及各模型其余冻结超参数，只把BatchNorm关闭；InternVL原本无BN，直接复用。",
        "",
        "| 模型 | 标准化+无BN AUROC/AP | 标准化原配置 AUROC/AP | 去BN变化 AUROC/AP (pp) |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {titles[row['model']]} | "
            f"{100*row['no_bn_AUROC_mean']:.2f}±{100*row['no_bn_AUROC_std']:.2f} / "
            f"{100*row['no_bn_HALL_AUPR_mean']:.2f}±{100*row['no_bn_HALL_AUPR_std']:.2f} | "
            f"{100*row['with_bn_AUROC_mean']:.2f} / {100*row['with_bn_HALL_AUPR_mean']:.2f} | "
            f"{row['AUROC_delta_pp']:+.2f} / {row['HALL_AUPR_delta_pp']:+.2f} |"
        )
    lines.append(
        f"| 四模型宏平均 | {100*macro['no_bn_AUROC_mean']:.2f} / "
        f"{100*macro['no_bn_HALL_AUPR_mean']:.2f} | "
        f"{100*macro['with_bn_AUROC_mean']:.2f} / {100*macro['with_bn_HALL_AUPR_mean']:.2f} | "
        f"{macro['AUROC_delta_pp']:+.2f} / {macro['HALL_AUPR_delta_pp']:+.2f} |"
    )
    lines.extend(
        [
            "",
            "| 模型 | Visual-only标准化+无BN | SVAR标准化 | Visual-only−SVAR (AUROC/AP pp) |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {titles[row['model']]} | {100*row['no_bn_AUROC_mean']:.2f} / "
            f"{100*row['no_bn_HALL_AUPR_mean']:.2f} | "
            f"{100*row['standardized_svar_AUROC_mean']:.2f} / "
            f"{100*row['standardized_svar_HALL_AUPR_mean']:.2f} | "
            f"{row['vs_standardized_svar_AUROC_pp']:+.2f} / "
            f"{row['vs_standardized_svar_HALL_AUPR_pp']:+.2f} |"
        )
    lines.append(
        f"| 四模型宏平均 | {100*macro['no_bn_AUROC_mean']:.2f} / "
        f"{100*macro['no_bn_HALL_AUPR_mean']:.2f} | "
        f"{100*macro['standardized_svar_AUROC_mean']:.2f} / "
        f"{100*macro['standardized_svar_HALL_AUPR_mean']:.2f} | "
        f"{macro['vs_standardized_svar_AUROC_pp']:+.2f} / "
        f"{macro['vs_standardized_svar_HALL_AUPR_pp']:+.2f} |"
    )
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")


def validate():
    checked = 0
    new_training_fits = 0
    reused_training_fits = 0
    max_probability_error = 0.0
    max_mean_error = 0.0
    max_scale_error = 0.0
    for model in MODELS:
        _, matrix, y, masks = legacy.load_source(model)
        protocol = json.loads((OUT / model / "protocol.json").read_text())
        expected_mean, expected_scale = search.scale_fit(matrix[masks["train"]], True)
        for seed in SEEDS:
            final = search.read(OUT / model / "final" / f"seed{seed}" / "result.pt")
            trained = search.read(ROOT / final["training_source"])
            if final["protocol_fingerprint"] != protocol["fingerprint"]:
                raise AssertionError((model, seed, "fingerprint"))
            if not trained["config"]["standardize"] or trained["config"]["batch_norm"]:
                raise AssertionError((model, seed, "config"))
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
        "schema": "legacy-visual-standardized-no-bn-seed424344-validation-v1",
        "status": "PASS",
        "checked_heads": checked,
        "new_training_fits": new_training_fits,
        "reused_training_fits": reused_training_fits,
        "max_probability_abs_error": max_probability_error,
        "max_train_mean_abs_error": max_mean_error,
        "max_train_scale_abs_error": max_scale_error,
        "validated_at": standardized.now(),
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
