"""Evaluate five preselected all-source/B1 geometry scalars with the fixed 811 probe."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.tc_fvpa_artifacts import atomic_json_save, atomic_torch_save
from scripts import analyze_all_source_paths as source_schema
from scripts import search_single_mlp_811 as trainer
from scripts.summarize_all_attention_write_gain import weighted_scores


SOURCE = ROOT / "outputs/ffn_all_source_paths_v1"
SPLIT_SOURCE = ROOT / "outputs/all_attention_ae_811_v1"
POSITION_SOURCE = ROOT / "outputs/generation_per_position_811_v1"
OUT = ROOT / "outputs/selected_geometry_scalars_811_v1"
MODELS = tuple(source_schema.MODELS)
SEEDS = (43, 44, 45)
SCALARS = (
    "prompt_within_source_kappa",
    "generation_within_source_kappa",
    "b1_residual_visual_generation_balance",
    "b1_delta_residual_generation_cos",
    "visual_context_relative_change",
)
TRAINED_GROUPS = SCALARS + ("five_concat", "generation_kappa_plus_position")
ALL_GROUPS = TRAINED_GROUPS + ("position",)
CONFIG = dict(
    width=128,
    standardize=True,
    batch_norm=False,
    dropout=.3,
    activation="relu",
    learning_rate=.001,
    weight_decay=1e-5,
    batch_size=128,
    max_epochs=150,
    patience=20,
    lr_patience=6,
    monitor="val_loss",
    early_stopping=True,
)
PAIRS = (
    tuple((name, "position") for name in SCALARS)
    + (("generation_kappa_plus_position", "generation_within_source_kappa"),
       ("generation_kappa_plus_position", "position"))
    + tuple(("five_concat", name) for name in SCALARS)
    + (("five_concat", "position"),)
)


def read(path):
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def block(matrix, fields, name, layers):
    value = np.asarray(matrix, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != len(fields) * layers:
        raise ValueError(f"Unexpected matrix shape for {name}: {value.shape}")
    index = fields.index(name)
    return value[:, index * layers:(index + 1) * layers].copy()


def build_features(source_groups, mentions, layers):
    groups = {
        "prompt_within_source_kappa": block(
            source_groups["F2"], source_schema.F2_FIELDS,
            "prompt_within_source_kappa", layers),
        "generation_within_source_kappa": block(
            source_groups["F2"], source_schema.F2_FIELDS,
            "generation_within_source_kappa", layers),
        "b1_residual_visual_generation_balance": block(
            source_groups["b1_F4"], source_schema.F4_FIELDS,
            "residual_visual_generation_balance", layers),
        "b1_delta_residual_generation_cos": block(
            source_groups["b1_F4"], source_schema.F4_FIELDS,
            "delta_residual_generation_cos", layers),
        "visual_context_relative_change": block(
            source_groups["F2"], source_schema.F2_FIELDS,
            "visual_context_relative_change", layers),
    }
    if any(not np.isfinite(value).all() for value in groups.values()):
        raise ValueError("Nonfinite selected geometry feature")
    position = np.log1p(np.asarray([m["response_index"] for m in mentions], dtype=np.float64))[:, None]
    if not np.isfinite(position).all() or (position < 0).any():
        raise ValueError("Invalid response position")
    groups["five_concat"] = np.concatenate([groups[name] for name in SCALARS], axis=1)
    groups["generation_kappa_plus_position"] = np.concatenate(
        [groups["generation_within_source_kappa"], position.astype(np.float32)], axis=1)
    if tuple(groups) != TRAINED_GROUPS:
        raise AssertionError("Unexpected feature order")
    return groups, position.astype(np.float32)


def progress(model, stage, completed, **extra):
    atomic_json_save(dict(
        stage=stage,
        completed=completed,
        total=len(TRAINED_GROUPS) * len(SEEDS),
        heartbeat=datetime.now(timezone.utc).isoformat(),
        **extra,
    ), OUT / model / "progress.json")


def prepare(model):
    source = read(SOURCE / model / "matrices.pt")
    split = read(SPLIT_SOURCE / model / "matrices.pt")
    if source["mentions"] != split["mentions"]:
        raise ValueError("Source and 811 split mention order differ")
    np.testing.assert_array_equal(source["y"], split["y"])
    masks = {name: np.asarray(mask, dtype=bool) for name, mask in split["masks"].items()}
    if tuple(masks) != ("train", "validation", "test"):
        raise ValueError("Unexpected split masks")
    if any(mask.shape != (len(source["y"]),) for mask in masks.values()):
        raise ValueError("Invalid split mask shape")
    if np.any(sum(mask.astype(np.uint8) for mask in masks.values()) != 1):
        raise ValueError("811 masks do not partition mentions")
    layers = source_schema.LAYERS[model]
    groups, position = build_features(source["groups"], source["mentions"], layers)
    split_protocol = json.loads((SPLIT_SOURCE / model / "protocol.json").read_text())
    image_split = split_protocol["split"]
    if [len(image_split[name]) for name in ("train", "validation", "test")] != [3200, 400, 400]:
        raise ValueError("Expected fixed 3200/400/400 image split")
    protocol = dict(
        schema="selected-geometry-scalars-811-v1",
        model=model,
        source_fingerprint=source["fingerprint"],
        split_source_fingerprint=split["fingerprint"],
        split=image_split,
        seeds=list(SEEDS),
        config=CONFIG,
        scalars=list(SCALARS),
        trained_groups=list(TRAINED_GROUPS),
        dimensions={name: value.shape[1] for name, value in groups.items()},
        definitions={
            "prompt_within_source_kappa": "||E_P|| / sum_prompt ||e_m||",
            "generation_within_source_kappa": "||E_G|| / sum_generation ||e_m||",
            "b1_residual_visual_generation_balance": "cos(E_R,E_G)-cos(E_R,E_V), stable true-Norm B1",
            "b1_delta_residual_generation_cos": "cos(E_R,E_G)-cos(R,A_G), stable true-Norm B1",
            "visual_context_relative_change": "||E_V_all-E_V_visual_only||/||E_V_visual_only||",
            "five_concat": "Five full-layer scalar blocks concatenated in listed order",
            "generation_kappa_plus_position": "generation kappa full layers plus log1p(response_index)",
        },
        transform="Raw geometry scalars; position uses log1p; every column train-only population Z-score",
        training="Single hidden 128/ReLU/dropout .3, no BN, Adam; early stop and checkpoint by validation BCE loss",
        labels="REAL=1 model probability; HALL-AUPR computed from 1-p_REAL",
        position_reference="Reuse same-protocol position-only heads from generation_per_position_811_v1",
        bootstrap=dict(replicates=2000, seed=20260916, unit="400 test images including empty images", pairs=PAIRS),
        caveat="Previously inspected test split; exploratory. Nominal intervals have no multiple-comparison correction.",
    )
    digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode())
    digest.update(np.asarray(source["y"]).tobytes())
    for name, value in groups.items():
        digest.update(name.encode())
        digest.update(value.tobytes())
    protocol["fingerprint"] = digest.hexdigest()
    path = OUT / model / "protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise ValueError("Changed protocol; refusing stale outputs")
    else:
        atomic_json_save(protocol, path)
    feature_path = OUT / model / "features.pt"
    payload = dict(
        groups=groups,
        position=position,
        mentions=source["mentions"],
        y=np.asarray(source["y"]),
        masks=masks,
        fingerprint=protocol["fingerprint"],
    )
    if feature_path.exists():
        saved = read(feature_path)
        if saved["fingerprint"] != protocol["fingerprint"]:
            raise ValueError("Stale feature cache")
    else:
        atomic_torch_save(payload, feature_path)
    return payload, protocol


def validate_result(result, group, x, y, masks, protocol):
    if result["fingerprint"] != protocol["fingerprint"] or result["group"] != group:
        raise ValueError("Stale trained result")
    if result["config"] != CONFIG:
        raise ValueError("Changed classifier config")
    mean, scale = trainer.scale_fit(x[masks["train"]], True)
    np.testing.assert_array_equal(mean, result["mean"])
    np.testing.assert_array_equal(scale, result["scale"])
    if result["best_epoch"] != 1 + int(np.argmin([row["val_loss"] for row in result["history"]])):
        raise ValueError("Saved checkpoint is not minimum validation loss")
    net = trainer.SingleMLP(x.shape[1], CONFIG)
    net.load_state_dict(result["state_dict"])
    maximum = 0.0
    for part in ("train", "validation", "test"):
        probabilities = trainer.predict(
            net,
            torch.as_tensor(trainer.transform(x[masks[part]], mean, scale)),
        )
        saved = result[part + "_probabilities"]
        np.testing.assert_allclose(probabilities, saved, rtol=1e-5, atol=1e-6)
        maximum = max(maximum, float(np.max(np.abs(probabilities - saved))))
        if part != "train":
            expected = trainer.metrics(y[masks[part]], saved)
            for metric, value in expected.items():
                np.testing.assert_allclose(value, result[part][metric], atol=1e-12)
    return maximum


def position_results(model, data):
    source_features = read(POSITION_SOURCE / model / "features.pt")
    if source_features["mentions"] != data["mentions"]:
        raise ValueError("Position reference mentions differ")
    np.testing.assert_array_equal(source_features["y"], data["y"])
    for part in data["masks"]:
        np.testing.assert_array_equal(source_features["masks"][part], data["masks"][part])
    results = {}
    for seed in SEEDS:
        result = read(POSITION_SOURCE / model / "position" / f"seed{seed}.pt")
        if result["config"] != CONFIG:
            raise ValueError("Position reference classifier differs")
        results[seed] = result
    return results


def run_model(model, device):
    torch.set_num_threads(1)
    data, protocol = prepare(model)
    y, masks = data["y"], data["masks"]
    rows, checks, results = [], [], {}
    completed = 0
    for group, x in data["groups"].items():
        for seed in SEEDS:
            path = OUT / model / "heads" / group / f"seed{seed}.pt"
            if path.exists():
                result = read(path)
            else:
                result = trainer.fit(
                    x[masks["train"]], y[masks["train"]],
                    x[masks["validation"]], y[masks["validation"]],
                    CONFIG, seed, device,
                    callback=lambda epoch, g=group, s=seed: progress(
                        model, f"{g} seed{s}", completed, epoch=epoch),
                )
                net = trainer.SingleMLP(x.shape[1], CONFIG).to(device)
                net.load_state_dict(result["state_dict"])
                test_probability = trainer.predict(
                    net,
                    torch.as_tensor(
                        trainer.transform(x[masks["test"]], result["mean"], result["scale"]),
                        device=device,
                    ),
                )
                result.update(
                    fingerprint=protocol["fingerprint"],
                    group=group,
                    test_probabilities=test_probability,
                    test=trainer.metrics(y[masks["test"]], test_probability),
                )
                atomic_torch_save(result, path)
            error = validate_result(result, group, x, y, masks, protocol)
            completed += 1
            rows.append(dict(
                model=model,
                feature=group,
                seed=seed,
                input_dim=x.shape[1],
                best_epoch=result["best_epoch"],
                epochs=result["epochs"],
                validation_AUROC=result["validation"]["AUROC"],
                validation_HALL_AUPR=result["validation"]["HALL_AUPR"],
                **result["test"],
            ))
            checks.append(dict(feature=group, seed=seed, status="PASS", max_probability_error=error))
            results[group, seed] = result
            write_csv(OUT / model / "seed_metrics.csv", rows)
            atomic_json_save(checks, OUT / model / "validation.json")
            progress(model, f"{group} seed{seed}", completed)
            print(model, group, seed, result["test"], flush=True)
    positions = position_results(model, data)
    for seed, result in positions.items():
        results["position", seed] = result
    run_bootstrap(model, data, results, protocol)
    progress(model, "完成", completed, status="completed")


def run_bootstrap(model, data, results, protocol):
    destination = OUT / model / "bootstrap.json"
    if destination.exists():
        saved = json.loads(destination.read_text())
        if saved["fingerprint"] != protocol["fingerprint"]:
            raise ValueError("Stale bootstrap")
        return saved["rows"]
    mask = data["masks"]["test"]
    y = data["y"][mask]
    test_ids = np.asarray([m["image_id"] for m in data["mentions"]])[mask]
    images = np.asarray(sorted(protocol["split"]["test"]))
    mapping = {image: index for index, image in enumerate(images)}
    cluster = np.asarray([mapping[image] for image in test_ids])
    probabilities = np.stack([
        np.stack([results[group, seed]["test_probabilities"] for seed in SEEDS])
        for group in ALL_GROUPS
    ])
    base = np.asarray([
        [weighted_scores(y, prediction, np.ones(len(y))) for prediction in group]
        for group in probabilities
    ]).mean(1)
    rng = np.random.default_rng(20260916)
    samples = []
    started = time.monotonic()
    for replicate in range(2000):
        counts = np.bincount(rng.integers(len(images), size=len(images)), minlength=len(images))
        weights = counts[cluster]
        if not all(weights[y == label].sum() > 0 for label in (0, 1)):
            continue
        samples.append(np.asarray([
            [weighted_scores(y, prediction, weights) for prediction in group]
            for group in probabilities
        ]).mean(1))
        if (replicate + 1) % 250 == 0:
            print(model, "bootstrap", replicate + 1, "/2000", flush=True)
    samples = np.asarray(samples)
    rows = []
    for left, right in PAIRS:
        li, ri = ALL_GROUPS.index(left), ALL_GROUPS.index(right)
        for metric_index, metric in enumerate(("AUROC", "HALL_AUPR")):
            delta = samples[:, li, metric_index] - samples[:, ri, metric_index]
            low, high = np.quantile(delta, [.025, .975])
            rows.append(dict(
                model=model,
                left=left,
                right=right,
                metric=metric,
                difference=base[li, metric_index] - base[ri, metric_index],
                ci95_low=low,
                ci95_high=high,
                valid_replicates=len(delta),
            ))
    payload = dict(
        fingerprint=protocol["fingerprint"],
        replicates=2000,
        seed=20260916,
        seconds=time.monotonic() - started,
        rows=rows,
    )
    atomic_json_save(payload, destination)
    write_csv(OUT / model / "bootstrap.csv", rows)
    return rows


def summarize():
    detection, seed_rows, intervals, validation = [], [], [], {}
    for model in MODELS:
        with (OUT / model / "seed_metrics.csv").open() as stream:
            local = list(csv.DictReader(stream))
        if len(local) != len(TRAINED_GROUPS) * len(SEEDS):
            raise ValueError(f"Incomplete heads for {model}")
        seed_rows.extend(local)
        checks = json.loads((OUT / model / "validation.json").read_text())
        if len(checks) != len(local) or not all(row["status"] == "PASS" for row in checks):
            raise ValueError(f"Failed reload validation for {model}")
        validation[model] = dict(
            heads=len(checks),
            max_probability_error=max(row["max_probability_error"] for row in checks),
            status="PASS",
        )
        position = position_results(model, read(OUT / model / "features.pt"))
        for group in ALL_GROUPS:
            if group == "position":
                values = [position[seed]["test"] for seed in SEEDS]
                input_dim = 1
            else:
                selected = [row for row in local if row["feature"] == group]
                if sorted(int(row["seed"]) for row in selected) != list(SEEDS):
                    raise ValueError(f"Incomplete {model} {group}")
                values = [{metric: float(row[metric]) for metric in ("AUROC", "HALL_AUPR")} for row in selected]
                input_dim = int(selected[0]["input_dim"])
            row = dict(model=model, feature=group, input_dim=input_dim)
            for metric in ("AUROC", "HALL_AUPR"):
                scores = [value[metric] for value in values]
                row[metric + "_mean"] = float(np.mean(scores))
                row[metric + "_std"] = float(np.std(scores))
            detection.append(row)
        intervals.extend(json.loads((OUT / model / "bootstrap.json").read_text())["rows"])
    write_csv(OUT / "detection_summary.csv", detection)
    write_csv(OUT / "seed_metrics.csv", seed_rows)
    write_csv(OUT / "bootstrap.csv", intervals)
    atomic_json_save(validation, OUT / "validation.json")
    atomic_json_save(dict(detection=detection, bootstrap=intervals), OUT / "summary.json")
    plot_detection(detection)
    write_report(detection, intervals, validation)


def plot_detection(rows):
    labels = ["P-kappa", "G-kappa", "R balance", "Delta cos(R,G)", "V context", "Five concat", "G-kappa+pos", "Position"]
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    x = np.arange(len(ALL_GROUPS))
    width = .19
    for model_index, model in enumerate(MODELS):
        selected = [next(row for row in rows if row["model"] == model and row["feature"] == group) for group in ALL_GROUPS]
        for metric_index, metric in enumerate(("AUROC", "HALL_AUPR")):
            axes[metric_index].bar(
                x + (model_index - 1.5) * width,
                [100 * row[metric + "_mean"] for row in selected],
                width=width,
                yerr=[100 * row[metric + "_std"] for row in selected],
                capsize=2,
                label=("Qwen2.5", "LLaVA", "Qwen3", "InternVL")[model_index],
            )
            axes[metric_index].set_ylabel(metric + " (%)")
            axes[metric_index].grid(axis="y", alpha=.2)
    axes[0].legend(ncol=4)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    fig.suptitle("Selected geometry scalars | fixed 811 | 3-seed mean +/- population std")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(OUT / f"detection.{extension}", dpi=160)
    plt.close(fig)


def write_report(rows, intervals, validation):
    names = {
        "prompt_within_source_kappa": "prompt kappa",
        "generation_within_source_kappa": "generation kappa",
        "b1_residual_visual_generation_balance": "B1 residual G-V balance",
        "b1_delta_residual_generation_cos": "B1 delta cos(R,G)",
        "visual_context_relative_change": "visual context relative change",
        "five_concat": "五量拼接",
        "generation_kappa_plus_position": "generation kappa + position",
        "position": "position-only（复用）",
    }
    lines = [
        "# 五个来源/残差几何标量：固定811幻觉检测结果",
        "",
        "复用四模型4000图的真实RMS All-attention K32与数值通过的B1缓存；没有重跑VLM。B2 K4未使用。",
        "固定3200训练/400验证/400测试图片及全部mentions；seeds43/44/45。原几何量不取log，position取log1p；逐列StandardScaler只拟合训练集。统一单隐藏128/ReLU/dropout.3、无BN、Adam lr.001/wd1e-5/batch128、最多150epoch、早停20，最低validation BCE loss checkpoint；无HPO。",
        "",
        "## 检测结果（%，三seed均值±总体std）",
        "",
        "| 特征 | Qwen2.5 | LLaVA | Qwen3 | InternVL |",
        "|---|---:|---:|---:|---:|",
    ]
    for group in ALL_GROUPS:
        cells = []
        for model in MODELS:
            row = next(row for row in rows if row["model"] == model and row["feature"] == group)
            cells.append(" / ".join(
                f"{100 * row[metric + '_mean']:.2f}±{100 * row[metric + '_std']:.2f}"
                for metric in ("AUROC", "HALL_AUPR")
            ))
        lines.append("| " + names[group] + " | " + " | ".join(cells) + " |")
    lines += [
        "",
        "每格为AUROC / HALL-AUPR；HALL为AP正类。position-only直接复用完全相同配置与811划分的既有三头，其余共84个新头。",
        "",
        "## 配对图片bootstrap",
        "",
        "| 模型 | 比较 | 指标 | 差值pp [名义95% CI] |",
        "|---|---|---|---:|",
    ]
    for row in intervals:
        lines.append(
            f"| {row['model']} | {names[row['left']]} − {names[row['right']]} | {row['metric']} | "
            f"{100 * row['difference']:+.2f} [{100 * row['ci95_low']:+.2f},{100 * row['ci95_high']:+.2f}] |"
        )
    lines += [
        "",
        "每模型2000次测试图片簇配对bootstrap，抽样框架包含无mention测试图片；区间未作多重比较校正。测试集已在此前实验中查看，本结果属于探索性复核。",
        "",
        "## 核验",
        "",
        "84个新头均已CPU重载；train-only scaler、最低val-loss checkpoint、train/validation/test概率和指标全部复算。最大概率误差："
        + "；".join(f"{model}={value['max_probability_error']:.3g}" for model, value in validation.items()) + "。",
        "",
        "[检测图](../outputs/selected_geometry_scalars_811_v1/detection.png) · [逐seed](../outputs/selected_geometry_scalars_811_v1/seed_metrics.csv) · [bootstrap](../outputs/selected_geometry_scalars_811_v1/bootstrap.csv) · [完整JSON](../outputs/selected_geometry_scalars_811_v1/summary.json)",
    ]
    report = ROOT / "docs/SELECTED_GEOMETRY_SCALARS_811_RESULTS.md"
    report.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize()
    elif args.models:
        for model in args.models:
            run_model(model, args.device)
    else:
        parser.error("Provide --models or --summarize")


if __name__ == "__main__":
    main()
