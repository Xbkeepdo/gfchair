"""Evaluate train-only standardized proportional-layer SVAR with seeds 42/43/44."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
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
from features.tc_fvpa_artifacts import atomic_json_save, atomic_torch_save
from scripts import evaluate_svar_llava_5_18_proportional_811 as base


OUT = ROOT / "outputs/svar_proportional_standardized_seed424344_811_v1"
RAW_OUT = ROOT / "outputs/svar_proportional_seed424344_811_v1"
SEEDS = (42, 43, 44)
MODELS = tuple(base.MODELS)
TRAINING = {**base.native.SVAR, "standardize": True}


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def fit_standardizer(x, train_mask):
    train = np.asarray(x[train_mask], dtype=np.float32)
    mean = train.mean(axis=0, keepdims=True)
    scale = train.std(axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    return mean, scale


def transform(x, mean, scale):
    value = (np.asarray(x, dtype=np.float32) - mean) / scale
    if not np.isfinite(value).all():
        raise ValueError("Non-finite standardized SVAR feature")
    return value.astype(np.float32, copy=False)


def protocol_for(model_name, cached, cached_protocol, x, total_layers, start, end):
    protocol = {
        "schema": "native-svar-proportional-standardized-seed424344-811-v1",
        "model": model_name,
        "total_decoder_layers": total_layers,
        "reference": "LLaVA zero-based layers 5 through 18 inclusive, [5,19) of 32 layers",
        "mapping": "start=round(L*5/32), end_exclusive=round(L*19/32)",
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
        "training": dict(TRAINING),
        "standardization": {
            "type": "column-wise z-score",
            "fit": "training mentions only",
            "application": "same saved mean/scale applied to train, validation, and test",
            "small_scale_rule": "replace scale < 1e-6 with 1.0",
        },
        "selection": "Minimum validation cross-entropy; early-stop patience5; validation REAL-F1 threshold.",
        "status": "Exploratory standardization ablation after test-set access.",
    }
    digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode())
    digest.update(np.asarray(x, dtype=np.float32).tobytes())
    protocol["fingerprint"] = digest.hexdigest()
    return protocol


def run_model(model_name, device):
    cached, cached_protocol, x, total_layers, start, end = base.prepare(model_name)
    protocol = protocol_for(
        model_name, cached, cached_protocol, x, total_layers, start, end
    )
    model_root = OUT / model_name
    atomic_json_save(protocol, model_root / "protocol.json")
    masks = cached["masks"]
    mean, scale = fit_standardizer(x, masks["train"])
    rows = []
    probabilities = []
    for seed in SEEDS:
        result_path = model_root / f"seed{seed}" / "result.pt"
        if result_path.exists():
            result = base.native.read(result_path)
            if result["fingerprint"] != protocol["fingerprint"]:
                raise AssertionError((model_name, seed, "fingerprint"))
        else:
            _seed_everything(seed)
            model = SVARMLP(x.shape[1], hidden_dim=TRAINING["hidden_dim"])
            trained = train_torch_detector(
                model=model,
                X_train=x[masks["train"]],
                raw_y_train=cached["y"][masks["train"]],
                X_val=x[masks["validation"]],
                raw_y_val=cached["y"][masks["validation"]],
                X_test=x[masks["test"]],
                raw_y_test=cached["y"][masks["test"]],
                device=device,
                seed=seed,
                positive_class="real",
                strict_82_no_validation=False,
                **{key: value for key, value in TRAINING.items() if key != "hidden_dim"},
            )
            model.load_state_dict(trained.state_dict)
            hall = {
                part: torch_hallucination_scores(
                    model,
                    transform(x[mask], mean, scale),
                    torch.device(device),
                )
                for part, mask in masks.items()
            }
            real = {part: 1.0 - value for part, value in hall.items()}
            threshold = select_detection_threshold(
                cached["y"][masks["validation"]],
                hall["validation"],
                positive_class="real",
            )
            result = {
                "fingerprint": protocol["fingerprint"],
                "seed": seed,
                "state_dict": trained.state_dict,
                "history": trained.history,
                "best_epoch": int(
                    min(trained.history, key=lambda row: row["val_loss"])["epoch"]
                ),
                "mean": mean,
                "scale": scale,
                "threshold": threshold,
                "threshold_report": evaluate_detection_scores(
                    cached["y"][masks["test"]],
                    hall["test"],
                    threshold,
                    positive_class="real",
                ),
                "test_metrics": base.native.shared.study.scores(
                    cached["y"][masks["test"]], real["test"]
                ),
                "test_probabilities": real["test"],
            }
            atomic_torch_save(result, result_path)
        probabilities.append(result["test_probabilities"])
        rows.append(
            {
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
            }
        )
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
        "ensemble": base.native.shared.study.scores(
            cached["y"][masks["test"]], np.mean(probabilities, axis=0)
        ),
    }
    write_csv(rows, model_root / "seed_metrics.csv")
    atomic_json_save(summary, model_root / "summary.json")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def summarize():
    summaries = []
    seed_rows = []
    comparisons = []
    for model_name in MODELS:
        standardized = json.loads((OUT / model_name / "summary.json").read_text())
        raw = json.loads((RAW_OUT / model_name / "summary.json").read_text())
        summaries.append(standardized)
        with (OUT / model_name / "seed_metrics.csv").open() as handle:
            seed_rows.extend(csv.DictReader(handle))
        comparisons.append(
            {
                "model": model_name,
                "standardized_AUROC_mean": standardized["AUROC_mean"],
                "standardized_AUROC_std": standardized["AUROC_std"],
                "standardized_HALL_AUPR_mean": standardized["HALL_AUPR_mean"],
                "standardized_HALL_AUPR_std": standardized["HALL_AUPR_std"],
                "raw_AUROC_mean": raw["AUROC_mean"],
                "raw_AUROC_std": raw["AUROC_std"],
                "raw_HALL_AUPR_mean": raw["HALL_AUPR_mean"],
                "raw_HALL_AUPR_std": raw["HALL_AUPR_std"],
                "AUROC_delta_pp": 100 * (standardized["AUROC_mean"] - raw["AUROC_mean"]),
                "HALL_AUPR_delta_pp": 100
                * (standardized["HALL_AUPR_mean"] - raw["HALL_AUPR_mean"]),
            }
        )
    macro = {
        key: float(np.mean([row[key] for row in comparisons]))
        for key in (
            "standardized_AUROC_mean",
            "standardized_HALL_AUPR_mean",
            "raw_AUROC_mean",
            "raw_HALL_AUPR_mean",
            "AUROC_delta_pp",
            "HALL_AUPR_delta_pp",
        )
    }
    write_csv(seed_rows, OUT / "seed_metrics.csv")
    write_csv(comparisons, OUT / "comparison.csv")
    atomic_json_save(
        {
            "schema": "native-svar-proportional-standardized-seed424344-811-v1",
            "summaries": summaries,
            "comparison_to_unstandardized": comparisons,
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
        "# 比例层SVAR训练集标准化：训练种子42/43/44（固定811）",
        "",
        "仅增加逐特征train-only z-score；层范围、图片划分、网络、优化器、早停和训练种子均与未标准化对照一致。",
        "",
        "| 模型 | 标准化 AUROC/AP | 未标准化 AUROC/AP | 标准化变化 AUROC/AP (pp) |",
        "|---|---:|---:|---:|",
    ]
    for row in comparisons:
        lines.append(
            f"| {titles[row['model']]} | "
            f"{100*row['standardized_AUROC_mean']:.2f}±{100*row['standardized_AUROC_std']:.2f} / "
            f"{100*row['standardized_HALL_AUPR_mean']:.2f}±{100*row['standardized_HALL_AUPR_std']:.2f} | "
            f"{100*row['raw_AUROC_mean']:.2f}±{100*row['raw_AUROC_std']:.2f} / "
            f"{100*row['raw_HALL_AUPR_mean']:.2f}±{100*row['raw_HALL_AUPR_std']:.2f} | "
            f"{row['AUROC_delta_pp']:+.2f} / {row['HALL_AUPR_delta_pp']:+.2f} |"
        )
    lines.append(
        f"| 四模型宏平均 | {100*macro['standardized_AUROC_mean']:.2f} / "
        f"{100*macro['standardized_HALL_AUPR_mean']:.2f} | "
        f"{100*macro['raw_AUROC_mean']:.2f} / {100*macro['raw_HALL_AUPR_mean']:.2f} | "
        f"{macro['AUROC_delta_pp']:+.2f} / {macro['HALL_AUPR_delta_pp']:+.2f} |"
    )
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")


def validate():
    checked = 0
    max_probability_error = 0.0
    max_mean_error = 0.0
    max_scale_error = 0.0
    for model_name in MODELS:
        cached, _, x, _, _, _ = base.prepare(model_name)
        masks = cached["masks"]
        expected_mean, expected_scale = fit_standardizer(x, masks["train"])
        protocol = json.loads((OUT / model_name / "protocol.json").read_text())
        if protocol["seeds"] != list(SEEDS) or not protocol["training"]["standardize"]:
            raise AssertionError((model_name, "protocol"))
        for seed in SEEDS:
            result = base.native.read(OUT / model_name / f"seed{seed}" / "result.pt")
            if result["fingerprint"] != protocol["fingerprint"]:
                raise AssertionError((model_name, seed, "fingerprint"))
            max_mean_error = max(
                max_mean_error, float(np.max(np.abs(result["mean"] - expected_mean)))
            )
            max_scale_error = max(
                max_scale_error, float(np.max(np.abs(result["scale"] - expected_scale)))
            )
            network = SVARMLP(x.shape[1], hidden_dim=TRAINING["hidden_dim"])
            network.load_state_dict(result["state_dict"])
            hall = torch_hallucination_scores(
                network,
                transform(x[masks["test"]], result["mean"], result["scale"]),
                torch.device("cpu"),
            )
            probability = 1.0 - hall
            max_probability_error = max(
                max_probability_error,
                float(np.max(np.abs(probability - result["test_probabilities"]))),
            )
            metrics = base.native.shared.study.scores(
                cached["y"][masks["test"]], probability
            )
            for key, value in metrics.items():
                if abs(value - result["test_metrics"][key]) > 1e-12:
                    raise AssertionError((model_name, seed, key))
            checked += 1
    report = {
        "schema": "native-svar-proportional-standardized-seed424344-811-validation-v1",
        "status": "PASS",
        "checked_heads": checked,
        "max_probability_abs_error": max_probability_error,
        "max_train_mean_abs_error": max_mean_error,
        "max_train_scale_abs_error": max_scale_error,
        "validated_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json_save(report, OUT / "validation.json")
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.summarize:
        summarize()
    elif args.validate:
        validate()
    elif args.models:
        for model_name in args.models:
            run_model(model_name, args.device)
    else:
        parser.error("Provide --models, --summarize, or --validate")


if __name__ == "__main__":
    main()
