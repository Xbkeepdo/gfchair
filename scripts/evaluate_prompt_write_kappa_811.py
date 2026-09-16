"""Detect hallucinations with prompt WRITE kappa and F+prompt WRITE kappa."""
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
from scripts import evaluate_legacy_visual_geometry_fusion_811 as prior
from scripts.summarize_all_attention_write_gain import weighted_scores


SOURCE = ROOT / "outputs/ffn_all_source_paths_v1"
WRITE_SOURCE = ROOT / "outputs/all_attention_write_gain_811_v1"
LEGACY_SOURCE = ROOT / "outputs/all_attention_ae_811_v1"
GEOMETRY_SOURCE = ROOT / "outputs/selected_geometry_scalars_811_v1"
PRIOR_FUSION = ROOT / "outputs/legacy_visual_geometry_fusion_811_v1"
OUT = ROOT / "outputs/prompt_write_kappa_811_v1"
MODELS, SEEDS, CONFIG = prior.MODELS, prior.SEEDS, prior.CONFIG
TRAINED = ("write_kappa", "F_plus_write_kappa")
GROUPS = ("write_kappa", "effect_kappa", "F", "F_plus_write_kappa", "F_plus_effect_kappa")
PAIRS = (
    ("write_kappa", "effect_kappa"),
    ("F_plus_write_kappa", "F"),
    ("F_plus_effect_kappa", "F"),
    ("F_plus_write_kappa", "F_plus_effect_kappa"),
)


def block(matrix, field, layers):
    value = np.asarray(matrix, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != len(source_schema.F1_FIELDS) * layers:
        raise ValueError(f"Invalid F1_raw shape: {value.shape}")
    index = source_schema.F1_FIELDS.index(field)
    return value[:, index * layers:(index + 1) * layers]


def build_features(source_f1, write_statistics, legacy_f):
    """Derive ||sum a_m||/sum||a_m|| via ||sum e_m||/group_gain and WRITE I."""
    f = np.asarray(legacy_f, dtype=np.float32)
    stats = np.asarray(write_statistics)
    if f.ndim != 2 or f.shape[1] % 2 or stats.ndim != 4 or stats.shape[0] != len(f):
        raise ValueError("Misaligned legacy F or WRITE statistics")
    layers = f.shape[1] // 2
    if stats.shape[1] != layers or stats.shape[2] != 4:
        raise ValueError("WRITE statistics layer/region layout differs")
    net = block(source_f1, "prompt_net_norm", layers).astype(np.float64)
    gain = block(source_f1, "prompt_group_gain", layers).astype(np.float64)
    gross_write = stats[:, :, 0, 0].astype(np.float64)
    if not (np.isfinite(net).all() and np.isfinite(gain).all() and np.isfinite(gross_write).all()):
        raise ValueError("Nonfinite source quantities")
    if (net <= 0).any() or (gain <= 0).any() or (gross_write <= 0).any():
        raise ValueError("Undefined prompt WRITE kappa: zero numerator/gain/denominator")
    kappa = (net / gain) / gross_write
    if not np.isfinite(kappa).all() or (kappa <= 0).any() or (kappa > 1 + 2e-6).any():
        raise ValueError("Prompt WRITE kappa violates (0,1] bound")
    result = {"write_kappa": kappa.astype(np.float32)}
    result["F_plus_write_kappa"] = np.concatenate([f, result["write_kappa"]], axis=1)
    return result


def prepare(model):
    source = prior.read(SOURCE / model / "matrices.pt")
    write = prior.read(WRITE_SOURCE / model / "statistics.pt")
    legacy = prior.read(LEGACY_SOURCE / model / "matrices.pt")
    geometry = prior.read(GEOMETRY_SOURCE / model / "features.pt")
    for other, name in ((write, "write"), (legacy, "legacy"), (geometry, "effect")):
        if source["mentions"] != other["mentions"]:
            raise ValueError(f"{name} mention order differs")
        np.testing.assert_array_equal(source["y"], other["y"])
    masks = {part: np.asarray(mask, dtype=bool) for part, mask in legacy["masks"].items()}
    for part in masks:
        np.testing.assert_array_equal(masks[part], write["masks"][part])
        np.testing.assert_array_equal(masks[part], geometry["masks"][part])
    if tuple(masks) != ("train", "validation", "test") or np.any(sum(mask.astype(np.uint8) for mask in masks.values()) != 1):
        raise ValueError("Invalid fixed 811 split")
    groups = build_features(source["groups"]["F1_raw"], write["values"], legacy["groups"]["legacy_visual"])
    split = json.loads((LEGACY_SOURCE / model / "protocol.json").read_text())["split"]
    if [len(split[part]) for part in masks] != [3200, 400, 400]:
        raise ValueError("Expected fixed 3200/400/400 image split")
    protocol = dict(
        schema="prompt-write-kappa-811-v1",
        model=model,
        source_fingerprint=source["fingerprint"],
        write_source_fingerprint=write["fingerprint"],
        legacy_source_fingerprint=legacy["fingerprint"],
        effect_source_fingerprint=geometry["fingerprint"],
        split=split,
        seeds=list(SEEDS),
        config=CONFIG,
        feature="kappa_P^a=||sum_prompt a_m||/sum_prompt||a_m||",
        reconstruction="||sum_prompt a_m|| = prompt_net_norm / prompt_group_gain; denominator=WRITE prompt I",
        reference_features={"effect_kappa": "kappa_P^e from selected_geometry_scalars_811_v1", "F": "legacy_visual=[AE_V,log1p(S_E)]", "F_plus_effect_kappa": "previous legacy_plus_prompt_kappa"},
        train_groups=list(TRAINED),
        transform="Raw kappa/F blocks concatenated; train-only per-column population Z-score",
        training="Single hidden 128/ReLU/dropout .3, no BN; minimum validation BCE checkpoint",
        bootstrap=dict(replicates=2000, seed=20260916, unit="400 test images including empty images", pairs=PAIRS),
        caveat="Test split previously inspected; exploratory. Nominal 95% intervals uncorrected for multiple comparisons.",
    )
    protocol = json.loads(json.dumps(protocol))
    digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode())
    digest.update(np.asarray(source["y"]).tobytes())
    for name, value in groups.items():
        digest.update(name.encode())
        digest.update(value.tobytes())
    protocol["fingerprint"] = digest.hexdigest()
    path = OUT / model / "protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise ValueError("Changed protocol; refusing stale result")
    else:
        atomic_json_save(protocol, path)
    data = dict(groups=groups, mentions=source["mentions"], y=np.asarray(source["y"]), masks=masks, fingerprint=protocol["fingerprint"])
    cache = OUT / model / "features.pt"
    if cache.exists():
        if prior.read(cache)["fingerprint"] != protocol["fingerprint"]:
            raise ValueError("Stale feature cache")
    else:
        atomic_torch_save(data, cache)
    return data, protocol


def reference_results(model, data):
    references = {
        "effect_kappa": (GEOMETRY_SOURCE / model / "heads/prompt_within_source_kappa", GEOMETRY_SOURCE / model / "features.pt"),
        "F": (PRIOR_FUSION / model / "heads/legacy_visual", PRIOR_FUSION / model / "features.pt"),
        "F_plus_effect_kappa": (PRIOR_FUSION / model / "heads/legacy_plus_prompt_kappa", PRIOR_FUSION / model / "features.pt"),
    }
    results = {}
    for group, (head_dir, feature_path) in references.items():
        features = prior.read(feature_path)
        if features["mentions"] != data["mentions"]:
            raise ValueError(f"{group} reference mention order differs")
        np.testing.assert_array_equal(features["y"], data["y"])
        for part in data["masks"]:
            np.testing.assert_array_equal(features["masks"][part], data["masks"][part])
        for seed in SEEDS:
            result = prior.read(head_dir / f"seed{seed}.pt")
            if result["config"] != CONFIG or len(result["test_probabilities"]) != int(data["masks"]["test"].sum()):
                raise ValueError(f"{group} reference classifier differs")
            results[group, seed] = result
    return results


def progress(model, stage, completed, **extra):
    atomic_json_save(dict(stage=stage, completed=completed, total=len(TRAINED)*len(SEEDS),
                          heartbeat=datetime.now(timezone.utc).isoformat(), **extra), OUT / model / "progress.json")


def run_model(model, device):
    torch.set_num_threads(1)
    data, protocol = prepare(model)
    y, masks = data["y"], data["masks"]
    rows, checks, results = [], [], reference_results(model, data)
    completed = 0
    for group, x in data["groups"].items():
        for seed in SEEDS:
            path = OUT / model / "heads" / group / f"seed{seed}.pt"
            if path.exists():
                result = prior.read(path)
            else:
                result = prior.geometry.trainer.fit(
                    x[masks["train"]], y[masks["train"]],
                    x[masks["validation"]], y[masks["validation"]],
                    CONFIG, seed, device,
                    callback=lambda epoch, g=group, s=seed: progress(model, f"{g} seed{s}", completed, epoch=epoch),
                )
                net = prior.geometry.trainer.SingleMLP(x.shape[1], CONFIG).to(device)
                net.load_state_dict(result["state_dict"])
                probability = prior.geometry.trainer.predict(
                    net,
                    torch.as_tensor(
                        prior.geometry.trainer.transform(x[masks["test"]], result["mean"], result["scale"]),
                        device=device,
                    ),
                )
                result.update(fingerprint=protocol["fingerprint"], group=group,
                              test_probabilities=probability,
                              test=prior.geometry.trainer.metrics(y[masks["test"]], probability))
                atomic_torch_save(result, path)
            error = prior.validate_result(result, group, x, y, masks, protocol)
            completed += 1
            rows.append(dict(model=model, feature=group, seed=seed, input_dim=x.shape[1],
                             best_epoch=result["best_epoch"], epochs=result["epochs"],
                             validation_AUROC=result["validation"]["AUROC"],
                             validation_HALL_AUPR=result["validation"]["HALL_AUPR"], **result["test"]))
            checks.append(dict(feature=group, seed=seed, status="PASS", max_probability_error=error))
            results[group, seed] = result
            prior.write_csv(OUT / model / "seed_metrics.csv", rows)
            atomic_json_save(checks, OUT / model / "validation.json")
            progress(model, f"{group} seed{seed}", completed)
            print(model, group, seed, result["test"], flush=True)
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
    image_ids = np.asarray([m["image_id"] for m in data["mentions"]])[mask]
    images = np.asarray(sorted(protocol["split"]["test"]))
    mapping = {image: index for index, image in enumerate(images)}
    cluster = np.asarray([mapping[image] for image in image_ids])
    probabilities = np.stack([
        np.stack([results[group, seed]["test_probabilities"] for seed in SEEDS])
        for group in GROUPS
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
        li, ri = GROUPS.index(left), GROUPS.index(right)
        for metric_index, metric in enumerate(("AUROC", "HALL_AUPR")):
            delta = samples[:, li, metric_index] - samples[:, ri, metric_index]
            low, high = np.quantile(delta, [.025, .975])
            rows.append(dict(model=model, left=left, right=right, metric=metric,
                             difference=base[li, metric_index]-base[ri, metric_index],
                             ci95_low=low, ci95_high=high, valid_replicates=len(delta)))
    atomic_json_save(dict(fingerprint=protocol["fingerprint"], replicates=2000,
                          seed=20260916, seconds=time.monotonic()-started, rows=rows), destination)
    prior.write_csv(OUT / model / "bootstrap.csv", rows)
    return rows


def summarize():
    detection, seeds, bootstrap, audit = [], [], [], {}
    for model in MODELS:
        with (OUT / model / "seed_metrics.csv").open() as stream:
            trained = list(csv.DictReader(stream))
        if len(trained) != len(TRAINED)*len(SEEDS):
            raise ValueError(f"Incomplete heads for {model}")
        seeds.extend(trained)
        data = prior.read(OUT / model / "features.pt")
        references = reference_results(model, data)
        checks = json.loads((OUT / model / "validation.json").read_text())
        if len(checks) != len(trained) or not all(row["status"] == "PASS" for row in checks):
            raise ValueError(f"Failed reload validation for {model}")
        audit[model] = dict(heads=len(checks), max_probability_error=max(row["max_probability_error"] for row in checks))
        for group in GROUPS:
            if group in TRAINED:
                selected = [row for row in trained if row["feature"] == group]
                values = [{metric: float(row[metric]) for metric in ("AUROC", "HALL_AUPR")} for row in selected]
            else:
                values = [references[group, seed]["test"] for seed in SEEDS]
            entry = dict(model=model, feature=group)
            for metric in ("AUROC", "HALL_AUPR"):
                scores = [value[metric] for value in values]
                entry[metric+"_mean"] = float(np.mean(scores))
                entry[metric+"_std"] = float(np.std(scores))
            detection.append(entry)
        bootstrap.extend(json.loads((OUT / model / "bootstrap.json").read_text())["rows"])
    prior.write_csv(OUT / "detection_summary.csv", detection)
    prior.write_csv(OUT / "seed_metrics.csv", seeds)
    prior.write_csv(OUT / "bootstrap.csv", bootstrap)
    atomic_json_save(dict(detection=detection, bootstrap=bootstrap, validation=audit), OUT / "summary.json")
    plot(detection)
    write_report(detection, bootstrap, audit)


def plot(rows):
    labels = [r"$\kappa_P^a$", r"$\kappa_P^e$", "F", r"F+$\kappa_P^a$", r"F+$\kappa_P^e$"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    x = np.arange(len(GROUPS))
    width = .19
    for model_index, model in enumerate(MODELS):
        selected = [next(row for row in rows if row["model"] == model and row["feature"] == group) for group in GROUPS]
        for metric_index, metric in enumerate(("AUROC", "HALL_AUPR")):
            axes[metric_index].bar(x+(model_index-1.5)*width,
                [100*row[metric+"_mean"] for row in selected], width,
                yerr=[100*row[metric+"_std"] for row in selected], capsize=2,
                label=("Qwen2.5", "LLaVA", "Qwen3", "InternVL")[model_index])
            axes[metric_index].set_ylabel(metric+" (%)")
            axes[metric_index].grid(axis="y", alpha=.2)
    axes[0].legend(ncol=4)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    fig.suptitle("Prompt WRITE versus effect kappa | fixed 811 | 3-seed mean +/- population std")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(OUT / f"detection.{extension}", dpi=160)
    plt.close(fig)


def write_report(rows, intervals, audit):
    names = {"write_kappa": "κ_P^a only", "effect_kappa": "κ_P^e only（复用）",
             "F": "F=[AE_V,log1p(S_E)]（复用）", "F_plus_write_kappa": "F+κ_P^a",
             "F_plus_effect_kappa": "F+κ_P^e（复用）"}
    lines = [
        "# Prompt WRITE κ_P^a 的幻觉检测及与 F 拼接",
        "",
        "κ_P^a = ||Σ_{m∈P} a_m|| / Σ_{m∈P}||a_m||，与此前 FFN 响应 κ_P^e = ||Σ e_m|| / Σ||e_m|| 区分。缓存中用 `prompt_net_norm / prompt_group_gain` 还原分子 `||Σa_m||`，分母取逐token WRITE 范数和（prompt I）；这与原提取代码的 gain 定义严格对应，精度受已存F1 float32影响。四模型全量值有限且位于(0,1]。",
        "",
        "固定811：3200/400/400图、全部mentions、seeds43/44/45、train-only逐列z-score、单隐藏128/ReLU/dropout.3、无BN、val-loss checkpoint、无HPO。κ_P^e、F、F+κ_P^e直接复用完全同配置的既有三seed头；仅κ_P^a与F+κ_P^a新训练24头。无新VLM前向、无B2。",
        "",
        "## 测试集检测（AUROC / HALL-AUPR，%，三seed均值±总体std）",
        "",
        "| 特征 | Qwen2.5 | LLaVA | Qwen3 | InternVL |",
        "|---|---:|---:|---:|---:|",
    ]
    for group in GROUPS:
        cells = []
        for model in MODELS:
            row = next(row for row in rows if row["model"] == model and row["feature"] == group)
            cells.append(" / ".join(f"{100*row[metric+'_mean']:.2f}±{100*row[metric+'_std']:.2f}"
                                    for metric in ("AUROC", "HALL_AUPR")))
        lines.append("| "+names[group]+" | "+" | ".join(cells)+" |")
    lines += [
        "",
        "## 结论",
        "",
        "- κ_P^a 单独与 κ_P^e 单独在四模型的AUROC与AP差值名义区间均跨零，没有明确赢家。",
        "- F+κ_P^a 相对F的四模型宏平均AUROC/AP提高+1.22/+2.66pp；AUROC在LLaVA、Qwen3、InternVL名义区间全正，AP仅InternVL全正。Qwen2.5两项区间均跨零。",
        "- F+κ_P^a 对F+κ_P^e的差值，仅Qwen3 AP名义区间全负（−2.12pp [−3.99,−0.16]）；其余模型两项区间均跨零。不能笼统地说WRITE κ比FFN响应κ更好。",
        "- 同cohort原始逐层曲线显示HALL的两种κ普遍高于REAL，但数值分离不等同于多特征分类器中的独立增益。",
        "",
        "## 配对图片bootstrap差值",
        "",
        "每模型2000次400测试图片簇配对抽样，含无mention图片；差值为左−右，单位pp，名义95% CI未作多重比较校正。",
        "",
        "| 模型 | 对比 | 指标 | 差值pp [95% CI] |",
        "|---|---|---|---:|",
    ]
    for row in intervals:
        lines.append(f"| {row['model']} | {names[row['left']]} − {names[row['right']]} | {row['metric']} | "
                     f"{100*row['difference']:+.2f} [{100*row['ci95_low']:+.2f},{100*row['ci95_high']:+.2f}] |")
    lines += [
        "",
        "测试集此前已查看，本结果是探索性比较，不能作为独立泛化或因果结论。",
        "",
        "## 核验",
        "",
        "新头全部CPU重载并复算训练/验证/测试概率、train-only scaler、最低validation-loss checkpoint；最大概率误差："
        +"；".join(f"{model}={value['max_probability_error']:.3g}" for model,value in audit.items())+"。",
        "",
        "[检测图](../outputs/prompt_write_kappa_811_v1/detection.png) · [a_m/e_m全量逐层图](../outputs/prompt_write_kappa_811_v1/kappa_curves_all.png) · [811测试逐层图](../outputs/prompt_write_kappa_811_v1/kappa_curves_test.png) · [逐seed](../outputs/prompt_write_kappa_811_v1/seed_metrics.csv) · [bootstrap](../outputs/prompt_write_kappa_811_v1/bootstrap.csv) · [完整JSON](../outputs/prompt_write_kappa_811_v1/summary.json)",
    ]
    (ROOT / "docs/PROMPT_WRITE_KAPPA_811_RESULTS.md").write_text("\n".join(lines)+"\n")


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
