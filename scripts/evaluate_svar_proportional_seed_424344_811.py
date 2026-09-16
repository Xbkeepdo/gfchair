"""Run proportional-layer native SVAR on the fixed 811 split with seeds 42/43/44."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.tc_fvpa_artifacts import atomic_json_save
from scripts import evaluate_svar_llava_5_18_proportional_811 as base


OUT = ROOT / "outputs/svar_proportional_seed424344_811_v1"
OLD_OUT = ROOT / "outputs/svar_llava_5_18_proportional_811_v1"
SEEDS = (42, 43, 44)
MODELS = tuple(base.MODELS)


def configure_base():
    base.OUT = OUT
    base.SEEDS = SEEDS


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize():
    configure_base()
    base.summarize_all()
    rows = []
    for model in MODELS:
        new = json.loads((OUT / model / "summary.json").read_text())
        old = json.loads((OLD_OUT / model / "summary.json").read_text())
        rows.append(
            {
                "model": model,
                "AUROC_mean_424344": new["AUROC_mean"],
                "AUROC_std_424344": new["AUROC_std"],
                "HALL_AUPR_mean_424344": new["HALL_AUPR_mean"],
                "HALL_AUPR_std_424344": new["HALL_AUPR_std"],
                "AUROC_mean_434445": old["AUROC_mean"],
                "AUROC_std_434445": old["AUROC_std"],
                "HALL_AUPR_mean_434445": old["HALL_AUPR_mean"],
                "HALL_AUPR_std_434445": old["HALL_AUPR_std"],
                "AUROC_delta_pp": 100 * (new["AUROC_mean"] - old["AUROC_mean"]),
                "HALL_AUPR_delta_pp": 100 * (
                    new["HALL_AUPR_mean"] - old["HALL_AUPR_mean"]
                ),
            }
        )
    write_csv(rows, OUT / "comparison.csv")
    combined = json.loads((OUT / "summary.json").read_text())
    combined["comparison_to_seeds434445"] = rows
    atomic_json_save(combined, OUT / "summary.json")

    titles = {
        "qwen2_5_vl_7b": "Qwen2.5-VL-7B",
        "llava_1_5_7b": "LLaVA-1.5-7B",
        "qwen3_vl_8b": "Qwen3-VL-8B",
        "internvl_2_5_8b": "InternVL2.5-8B",
    }
    lines = [
        "# 比例层原生SVAR：训练种子42/43/44（固定811）",
        "",
        "LLaVA使用零基层5–18；其他模型按decoder深度同比例映射。图片划分固定3200/400/400，所有设置仅将训练种子改为42/43/44。",
        "",
        "| 模型 | seeds42/43/44 AUROC/AP | seeds43/44/45 AUROC/AP | 变化 AUROC/AP (pp) |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {titles[row['model']]} | "
            f"{100*row['AUROC_mean_424344']:.2f}±{100*row['AUROC_std_424344']:.2f} / "
            f"{100*row['HALL_AUPR_mean_424344']:.2f}±{100*row['HALL_AUPR_std_424344']:.2f} | "
            f"{100*row['AUROC_mean_434445']:.2f}±{100*row['AUROC_std_434445']:.2f} / "
            f"{100*row['HALL_AUPR_mean_434445']:.2f}±{100*row['HALL_AUPR_std_434445']:.2f} | "
            f"{row['AUROC_delta_pp']:+.2f} / {row['HALL_AUPR_delta_pp']:+.2f} |"
        )
    lines.extend(
        [
            "",
            "两组三seed共享43/44；变化表示以seed42替换seed45后的均值变化。",
        ]
    )
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")


def validate():
    configure_base()
    checked = 0
    max_error = 0.0
    for model_name in MODELS:
        cached, cached_protocol, x, total_layers, start, end = base.prepare(model_name)
        protocol = json.loads((OUT / model_name / "protocol.json").read_text())
        if protocol["seeds"] != list(SEEDS):
            raise AssertionError((model_name, protocol["seeds"]))
        for seed in SEEDS:
            result = base.native.read(OUT / model_name / f"seed{seed}" / "result.pt")
            if result["fingerprint"] != protocol["fingerprint"]:
                raise AssertionError((model_name, seed, "fingerprint"))
            network = base.SVARMLP(x.shape[1], hidden_dim=base.native.SVAR["hidden_dim"])
            network.load_state_dict(result["state_dict"])
            hall = base.torch_hallucination_scores(
                network, x[cached["masks"]["test"]], torch.device("cpu")
            )
            probability = 1.0 - hall
            max_error = max(
                max_error,
                float(np.max(np.abs(probability - result["test_probabilities"]))),
            )
            metrics = base.native.shared.study.scores(
                cached["y"][cached["masks"]["test"]], probability
            )
            for key, value in metrics.items():
                if abs(value - result["test_metrics"][key]) > 1e-12:
                    raise AssertionError((model_name, seed, key))
            checked += 1
    report = {
        "schema": "native-svar-proportional-seed424344-811-validation-v1",
        "status": "PASS",
        "checked_heads": checked,
        "max_probability_abs_error": max_error,
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
    configure_base()
    if args.summarize:
        summarize()
    elif args.validate:
        validate()
    elif args.models:
        for model in args.models:
            base.run_model(model, args.device)
    else:
        parser.error("Provide --models, --summarize, or --validate")


if __name__ == "__main__":
    main()
