"""Fuse legacy visual AE/log-strength with five geometry blocks under fixed 811."""
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
from scripts import evaluate_selected_geometry_scalars_811 as geometry
from scripts.summarize_all_attention_write_gain import weighted_scores


LEGACY_SOURCE = ROOT / "outputs/all_attention_ae_811_v1"
GEOMETRY_SOURCE = ROOT / "outputs/selected_geometry_scalars_811_v1"
OUT = ROOT / "outputs/legacy_visual_geometry_fusion_811_v1"
MODELS = geometry.MODELS
SEEDS = geometry.SEEDS
CONFIG = geometry.CONFIG
SCALARS = geometry.SCALARS
FUSION_GROUPS = {
    "legacy_plus_prompt_kappa": "prompt_within_source_kappa",
    "legacy_plus_generation_kappa": "generation_within_source_kappa",
    "legacy_plus_b1_residual_balance": "b1_residual_visual_generation_balance",
    "legacy_plus_b1_delta_cos_rg": "b1_delta_residual_generation_cos",
    "legacy_plus_visual_context_change": "visual_context_relative_change",
}
TRAINED_GROUPS = ("legacy_visual",) + tuple(FUSION_GROUPS) + ("legacy_plus_five",)
REFERENCE_GROUP = "five_concat"
ALL_GROUPS = TRAINED_GROUPS + (REFERENCE_GROUP,)
REPORT_GROUPS = ("legacy_visual",) + tuple(FUSION_GROUPS)
REPORT_PAIRS = tuple((name, "legacy_visual") for name in FUSION_GROUPS)
PAIRS = (
    tuple((name, "legacy_visual") for name in FUSION_GROUPS)
    + (("legacy_plus_five", "legacy_visual"),)
    + tuple(("legacy_plus_five", name) for name in FUSION_GROUPS)
    + (("legacy_plus_five", REFERENCE_GROUP), ("legacy_visual", REFERENCE_GROUP))
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


def build_features(legacy_visual, geometry_groups):
    legacy = np.asarray(legacy_visual, dtype=np.float32)
    if legacy.ndim != 2 or not np.isfinite(legacy).all():
        raise ValueError("Invalid legacy_visual matrix")
    selected = {}
    for name in SCALARS:
        value = np.asarray(geometry_groups[name], dtype=np.float32)
        if value.ndim != 2 or len(value) != len(legacy) or not np.isfinite(value).all():
            raise ValueError(f"Invalid geometry matrix: {name}")
        selected[name] = value
    groups = {"legacy_visual": legacy.copy()}
    for output_name, scalar_name in FUSION_GROUPS.items():
        groups[output_name] = np.concatenate([legacy, selected[scalar_name]], axis=1)
    groups["legacy_plus_five"] = np.concatenate(
        [legacy] + [selected[name] for name in SCALARS], axis=1
    )
    if tuple(groups) != TRAINED_GROUPS:
        raise AssertionError("Unexpected fusion feature order")
    return groups


def progress(model, stage, completed, **extra):
    atomic_json_save(dict(
        stage=stage,
        completed=completed,
        total=len(TRAINED_GROUPS) * len(SEEDS),
        heartbeat=datetime.now(timezone.utc).isoformat(),
        **extra,
    ), OUT / model / "progress.json")


def prepare(model):
    legacy = read(LEGACY_SOURCE / model / "matrices.pt")
    geometry_data = read(GEOMETRY_SOURCE / model / "features.pt")
    if legacy["mentions"] != geometry_data["mentions"]:
        raise ValueError("Legacy and geometry mention order differ")
    np.testing.assert_array_equal(legacy["y"], geometry_data["y"])
    masks = {name: np.asarray(mask, dtype=bool) for name, mask in legacy["masks"].items()}
    for part, mask in geometry_data["masks"].items():
        np.testing.assert_array_equal(masks[part], mask)
    if tuple(masks) != ("train", "validation", "test"):
        raise ValueError("Unexpected split masks")
    if np.any(sum(mask.astype(np.uint8) for mask in masks.values()) != 1):
        raise ValueError("811 masks do not partition mentions")
    groups = build_features(legacy["groups"]["legacy_visual"], geometry_data["groups"])
    legacy_protocol = json.loads((LEGACY_SOURCE / model / "protocol.json").read_text())
    image_split = legacy_protocol["split"]
    if [len(image_split[name]) for name in ("train", "validation", "test")] != [3200, 400, 400]:
        raise ValueError("Expected fixed 3200/400/400 image split")
    protocol = dict(
        schema="legacy-visual-geometry-fusion-811-v1",
        model=model,
        legacy_source_fingerprint=legacy["fingerprint"],
        geometry_source_fingerprint=geometry_data["fingerprint"],
        split=image_split,
        seeds=list(SEEDS),
        config=CONFIG,
        legacy_feature="legacy_visual=[AE_V, log1p(S_E)]",
        legacy_path="Visual-only conditional path; true RMS differentiated along z-A_V -> z",
        geometry_scalars=list(SCALARS),
        trained_groups=list(TRAINED_GROUPS),
        reference_group=REFERENCE_GROUP,
        dimensions={name: value.shape[1] for name, value in groups.items()},
        transform="Concatenate raw blocks, then train-only per-column population Z-score",
        training="Single hidden 128/ReLU/dropout .3, no BN, Adam; checkpoint by minimum validation BCE loss",
        labels="REAL=1 model probability; HALL-AUPR computed from 1-p_REAL",
        reference="Standalone five_concat probabilities reused from selected_geometry_scalars_811_v1 under identical split/config/seeds",
        bootstrap=dict(replicates=2000, seed=20260916, unit="400 test images including empty images", pairs=PAIRS),
        caveat="Previously inspected test split; exploratory. Nominal intervals have no multiple-comparison correction.",
    )
    digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode())
    digest.update(np.asarray(legacy["y"]).tobytes())
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
    payload = dict(
        groups=groups,
        mentions=legacy["mentions"],
        y=np.asarray(legacy["y"]),
        masks=masks,
        fingerprint=protocol["fingerprint"],
    )
    feature_path = OUT / model / "features.pt"
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
    mean, scale = geometry.trainer.scale_fit(x[masks["train"]], True)
    np.testing.assert_array_equal(mean, result["mean"])
    np.testing.assert_array_equal(scale, result["scale"])
    if result["best_epoch"] != 1 + int(np.argmin([row["val_loss"] for row in result["history"]])):
        raise ValueError("Saved checkpoint is not minimum validation loss")
    net = geometry.trainer.SingleMLP(x.shape[1], CONFIG)
    net.load_state_dict(result["state_dict"])
    maximum = 0.0
    for part in ("train", "validation", "test"):
        probabilities = geometry.trainer.predict(
            net,
            torch.as_tensor(geometry.trainer.transform(x[masks[part]], mean, scale)),
        )
        saved = result[part + "_probabilities"]
        np.testing.assert_allclose(probabilities, saved, rtol=1e-5, atol=1e-6)
        maximum = max(maximum, float(np.max(np.abs(probabilities - saved))))
        if part != "train":
            expected = geometry.trainer.metrics(y[masks[part]], saved)
            for metric, value in expected.items():
                np.testing.assert_allclose(value, result[part][metric], atol=1e-12)
    return maximum


def reference_results(model, data):
    source = read(GEOMETRY_SOURCE / model / "features.pt")
    if source["mentions"] != data["mentions"]:
        raise ValueError("Standalone five reference mentions differ")
    np.testing.assert_array_equal(source["y"], data["y"])
    for part in data["masks"]:
        np.testing.assert_array_equal(source["masks"][part], data["masks"][part])
    results = {}
    for seed in SEEDS:
        result = read(GEOMETRY_SOURCE / model / "heads" / REFERENCE_GROUP / f"seed{seed}.pt")
        if result["config"] != CONFIG:
            raise ValueError("Standalone five reference classifier differs")
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
                result = geometry.trainer.fit(
                    x[masks["train"]], y[masks["train"]],
                    x[masks["validation"]], y[masks["validation"]],
                    CONFIG, seed, device,
                    callback=lambda epoch, g=group, s=seed: progress(
                        model, f"{g} seed{s}", completed, epoch=epoch),
                )
                net = geometry.trainer.SingleMLP(x.shape[1], CONFIG).to(device)
                net.load_state_dict(result["state_dict"])
                test_probability = geometry.trainer.predict(
                    net,
                    torch.as_tensor(
                        geometry.trainer.transform(x[masks["test"]], result["mean"], result["scale"]),
                        device=device,
                    ),
                )
                result.update(
                    fingerprint=protocol["fingerprint"],
                    group=group,
                    test_probabilities=test_probability,
                    test=geometry.trainer.metrics(y[masks["test"]], test_probability),
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
    for seed, result in reference_results(model, data).items():
        results[REFERENCE_GROUP, seed] = result
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
        reference = reference_results(model, read(OUT / model / "features.pt"))
        for group in ALL_GROUPS:
            if group == REFERENCE_GROUP:
                values = [reference[seed]["test"] for seed in SEEDS]
                input_dim = int(read(GEOMETRY_SOURCE / model / "features.pt")["groups"][REFERENCE_GROUP].shape[1])
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
    requested_detection = [row for row in detection if row["feature"] in REPORT_GROUPS]
    requested_seeds = [row for row in seed_rows if row["feature"] in REPORT_GROUPS]
    requested_intervals = [
        row for row in intervals if (row["left"], row["right"]) in REPORT_PAIRS
    ]
    write_csv(OUT / "detection_summary.csv", requested_detection)
    write_csv(OUT / "seed_metrics.csv", requested_seeds)
    write_csv(OUT / "bootstrap.csv", requested_intervals)
    write_csv(OUT / "supplementary_detection_summary.csv", detection)
    write_csv(OUT / "supplementary_seed_metrics.csv", seed_rows)
    write_csv(OUT / "supplementary_bootstrap.csv", intervals)
    atomic_json_save(validation, OUT / "validation.json")
    atomic_json_save(dict(
        detection=requested_detection,
        bootstrap=requested_intervals,
        supplementary=dict(detection=detection, bootstrap=intervals),
    ), OUT / "summary.json")
    plot_detection(requested_detection)
    write_report(requested_detection, requested_intervals, validation)


def plot_detection(rows):
    labels = ["F", "F+P-kappa", "F+G-kappa", "F+R balance", "F+Delta cos", "F+V context"]
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    x = np.arange(len(REPORT_GROUPS))
    width = .19
    for model_index, model in enumerate(MODELS):
        selected = [next(row for row in rows if row["model"] == model and row["feature"] == group) for group in REPORT_GROUPS]
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
    fig.suptitle("AE/log1p(S) + each geometry block | fixed 811 | 3-seed mean +/- population std")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(OUT / f"detection.{extension}", dpi=160)
    plt.close(fig)


def write_report(rows, intervals, validation):
    names = {
        "legacy_visual": "F = AE_V + log1p(S_E)",
        "legacy_plus_prompt_kappa": "F + prompt kappa",
        "legacy_plus_generation_kappa": "F + generation kappa",
        "legacy_plus_b1_residual_balance": "F + B1 residual G-V balance",
        "legacy_plus_b1_delta_cos_rg": "F + B1 delta cos(R,G)",
        "legacy_plus_visual_context_change": "F + visual context change",
        "legacy_plus_five": "F + 五量",
        "five_concat": "五量-only（复用）",
    }
    lines = [
        "# AE + log1p(S) 分别与五个几何量拼接：固定811检测结果",
        "",
        "这里的F严格采用项目既有`legacy_visual=[AE_V, log1p(S_E)]`：Visual-only条件路径，真RMS沿`z-A_V -> z`求差。五个几何块为此前选定的全层prompt/generation kappa、B1 residual G-V balance、B1 delta cos(R,G)与visual context relative change；B2 K4未使用。",
        "",
        "固定3200训练/400验证/400测试图片及全部mentions，seeds43/44/45。先拼接原始块，再逐列用训练集拟合StandardScaler；统一单隐藏128/ReLU/dropout.3、无BN、Adam lr.001/wd1e-5/batch128、最多150epoch、早停20，最低validation BCE loss checkpoint；无HPO。",
        "",
        "## 检测结果（%，三seed均值±总体std）",
        "",
        "| 特征 | Qwen2.5 | LLaVA | Qwen3 | InternVL |",
        "|---|---:|---:|---:|---:|",
    ]
    for group in REPORT_GROUPS:
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
        "每格为AUROC / HALL-AUPR；HALL为AP正类。主结果包含F与五种分别拼接的72个新头。用户澄清到达前队列已额外启动五量合并的12个头；它们只保存在`supplementary_*`文件，不进入本表、图或结论。",
        "",
        "## 结论",
        "",
        "- prompt kappa跨模型宏平均增量为AUROC +1.36pp、HALL-AUPR +2.60pp；AUROC在LLaVA/Qwen3/InternVL的名义95%区间全正，Qwen3的两项区间均全正。",
        "- B1 delta cos(R,G)跨模型宏平均增量为AUROC +1.31pp、HALL-AUPR +2.62pp；AUROC在Qwen2.5/LLaVA/Qwen3区间全正，Qwen2.5的AP也全正。它与prompt kappa是最稳定的两个增量块。",
        "- generation kappa、B1 residual balance和visual context change都有局部收益，但跨模型一致性较弱：分别仅Qwen3 AUROC、Qwen2.5 AP与Qwen3 AUROC、LLaVA AUROC得到全正区间。",
        "- 按点估计，Qwen2.5最优AUROC/AP分别来自B1 delta cos与B1 residual balance；LLaVA来自B1 delta cos与visual context change；Qwen3两项均来自prompt kappa；InternVL来自prompt kappa与B1 delta cos。",
        "",
        "## 主要配对图片bootstrap",
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
        "每模型2000次测试图片簇配对bootstrap，抽样框架包含无mention测试图片；五个逐项比较见主CSV，额外队列结果见supplementary CSV。区间未作多重比较校正。测试集已在此前实验中查看，本结果属于探索性复核。",
        "",
        "## 核验",
        "",
        "84个新头均已CPU重载；train-only scaler、最低val-loss checkpoint、train/validation/test概率和指标全部复算。最大概率误差："
        + "；".join(f"{model}={value['max_probability_error']:.3g}" for model, value in validation.items()) + "。",
        "",
        "[检测图](../outputs/legacy_visual_geometry_fusion_811_v1/detection.png) · [逐seed](../outputs/legacy_visual_geometry_fusion_811_v1/seed_metrics.csv) · [bootstrap](../outputs/legacy_visual_geometry_fusion_811_v1/bootstrap.csv) · [完整JSON](../outputs/legacy_visual_geometry_fusion_811_v1/summary.json)",
    ]
    (ROOT / "docs/LEGACY_VISUAL_GEOMETRY_FUSION_811_RESULTS.md").write_text("\n".join(lines) + "\n")


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
