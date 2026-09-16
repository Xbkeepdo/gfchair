"""Re-evaluate frozen legacy-visual MLP configs with training seeds 42/43/44."""

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
from scripts import search_legacy_visual_single_mlp_811 as legacy
from scripts import search_single_mlp_811 as search


OUT = ROOT / "outputs/legacy_visual_single_mlp_seed424344_811_v1"
SOURCE_OUT = legacy.OUT
MODELS = tuple(legacy.MODELS)
SEEDS = (42, 43, 44)


def now():
    return datetime.now(timezone.utc).isoformat()


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_protocol(model, source, selection):
    protocol = {
        "schema": "legacy-visual-frozen-config-seed424344-811-v1",
        "model": model,
        "feature": "legacy_visual=[AE_V, log1p(S_E)]",
        "seeds": list(SEEDS),
        "changed_factor": "training random seed only",
        "frozen_candidate": selection["index"],
        "frozen_config": selection["config"],
        "source_selection_fingerprint": selection["fingerprint"],
        "source_matrix_fingerprint": source["fingerprint"],
        "reuse": {
            "seed42": "new fit with the frozen config",
            "seed43": "reuse original saved fit",
            "seed44": "reuse original saved fit",
        },
        "split": "unchanged image-level 3200/400/400 split; no train+validation refit",
        "selection": "no hyperparameter reselection and no test-based choice",
    }
    protocol["fingerprint"] = hashlib.sha256(
        json.dumps(protocol, sort_keys=True).encode()
    ).hexdigest()
    return protocol


def training_path(model, candidate, seed):
    if seed == 42:
        return OUT / model / "training" / "seed42" / "result.pt"
    return (
        SOURCE_OUT
        / model
        / "search"
        / f"candidate{candidate:02d}"
        / f"seed{seed}"
        / "result.pt"
    )


def run_model(model, device):
    source, matrix, y, masks = legacy.load_source(model)
    source_selection = json.loads((SOURCE_OUT / model / "selection.json").read_text())
    protocol = make_protocol(model, source, source_selection)
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
    candidate = source_selection["index"]
    rows = []
    probabilities = []
    for seed in SEEDS:
        path = training_path(model, candidate, seed)
        if path.exists():
            trained = search.read(path)
        elif seed == 42:
            trained = search.fit(
                train_x,
                train_y,
                val_x,
                val_y,
                source_selection["config"],
                seed,
                device,
            )
            trained.update(
                fingerprint=protocol["fingerprint"],
                source_selection_fingerprint=source_selection["fingerprint"],
                index=candidate,
                feature=legacy.GROUP,
            )
            atomic_torch_save(trained, path)
        else:
            raise FileNotFoundError(path)

        if trained["config"] != source_selection["config"] or trained["input_dim"] != matrix.shape[1]:
            raise ValueError(f"Frozen config/input mismatch for {model} seed{seed}")
        if seed == 42:
            if trained["fingerprint"] != protocol["fingerprint"]:
                raise ValueError(f"Seed42 fingerprint mismatch for {model}")
        elif trained["fingerprint"] != source_selection["fingerprint"]:
            raise ValueError(f"Reused fit fingerprint mismatch for {model} seed{seed}")

        network = search.SingleMLP(trained["input_dim"], trained["config"]).to(device)
        network.load_state_dict(trained["state_dict"])
        transformed = torch.as_tensor(
            search.transform(test_x, trained["mean"], trained["scale"]), device=device
        )
        probability = search.predict(network, transformed)
        metrics = search.metrics(test_y, probability)
        final = {
            "schema": "legacy-visual-frozen-config-seed424344-final-v1",
            "model": model,
            "seed": seed,
            "candidate": candidate,
            "config": trained["config"],
            "protocol_fingerprint": protocol["fingerprint"],
            "training_source": str(path.relative_to(ROOT)),
            "reused_training": seed in (43, 44),
            "best_epoch": trained["best_epoch"],
            "validation": trained["validation"],
            "test_metrics": metrics,
            "test_probabilities": probability,
            "test_labels": test_y,
        }
        atomic_torch_save(final, root / "final" / f"seed{seed}" / "result.pt")
        rows.append({"model": model, "seed": seed, **metrics})
        probabilities.append(probability)

    summary = {
        "schema": "legacy-visual-frozen-config-seed424344-model-summary-v1",
        "model": model,
        "candidate": candidate,
        "config": source_selection["config"],
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
        {
            "stage": "complete",
            "status": "completed",
            "completed": 3,
            "total": 3,
            "heartbeat": now(),
        },
        root / "progress.json",
    )


def summarize():
    rows = []
    summaries = []
    for model in MODELS:
        new = json.loads((OUT / model / "summary.json").read_text())
        old = json.loads((SOURCE_OUT / model / "summary.json").read_text())
        row = {
            "model": model,
            "new_seeds": "42,43,44",
            "old_seeds": "43,44,45",
            "AUROC_mean_424344": new["AUROC_mean"],
            "AUROC_std_424344": new["AUROC_std"],
            "HALL_AUPR_mean_424344": new["HALL_AUPR_mean"],
            "HALL_AUPR_std_424344": new["HALL_AUPR_std"],
            "AUROC_mean_434445": old["AUROC_mean"],
            "AUROC_std_434445": old["AUROC_std"],
            "HALL_AUPR_mean_434445": old["HALL_AUPR_mean"],
            "HALL_AUPR_std_434445": old["HALL_AUPR_std"],
            "AUROC_delta_pp": 100 * (new["AUROC_mean"] - old["AUROC_mean"]),
            "HALL_AUPR_delta_pp": 100 * (new["HALL_AUPR_mean"] - old["HALL_AUPR_mean"]),
        }
        rows.append(row)
        summaries.append(new)
    write_csv(rows, OUT / "comparison.csv")
    atomic_json_save(
        {
            "schema": "legacy-visual-frozen-config-seed424344-summary-v1",
            "summaries": summaries,
            "comparison": rows,
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
        "# Visual-only冻结单层参数：训练种子42/43/44",
        "",
        "保持上一轮各模型验证集选出的候选和所有训练设置不变；新增seed42，复用seed43/44保存权重。表中为固定811测试集三seed均值±总体标准差，单位%。",
        "",
        "| 模型 | seeds42/43/44 AUROC/AP | seeds43/44/45 AUROC/AP | 均值变化 AUROC/AP (pp) |",
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
            "两组三seed共享seed43/44，因此差值只反映以seed42替换seed45后的均值变化；不是三组独立重复实验。",
        ]
    )
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")


def validate():
    checked = 0
    max_error = 0.0
    for model in MODELS:
        source, matrix, y, masks = legacy.load_source(model)
        selection = json.loads((SOURCE_OUT / model / "selection.json").read_text())
        for seed in SEEDS:
            trained = search.read(training_path(model, selection["index"], seed))
            final = search.read(OUT / model / "final" / f"seed{seed}" / "result.pt")
            network = search.SingleMLP(trained["input_dim"], trained["config"])
            network.load_state_dict(trained["state_dict"])
            transformed = torch.as_tensor(
                search.transform(
                    matrix[masks["test"]], trained["mean"], trained["scale"]
                )
            )
            probability = search.predict(network, transformed)
            max_error = max(
                max_error,
                float(np.max(np.abs(probability - final["test_probabilities"]))),
            )
            recomputed = search.metrics(y[masks["test"]], probability)
            for key, value in recomputed.items():
                if abs(value - final["test_metrics"][key]) > 1e-12:
                    raise AssertionError(
                        (model, seed, key, value, final["test_metrics"][key])
                    )
            checked += 1
    report = {
        "schema": "legacy-visual-frozen-config-seed424344-validation-v1",
        "status": "PASS",
        "checked_heads": checked,
        "new_training_fits": len(MODELS),
        "reused_training_fits": len(MODELS) * 2,
        "max_probability_abs_error": max_error,
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
