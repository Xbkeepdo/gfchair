"""Fixed-811 probe for matched all-attention P+V strength and cancellation."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.tc_fvpa_artifacts import atomic_json_save, atomic_torch_save
from scripts import analyze_all_source_paths as source
from scripts import evaluate_legacy_visual_geometry_fusion_811 as prior

OUT = ROOT / "outputs/decomposition_pv_kappa_811_v1/all_attention"
SOURCE = ROOT / "outputs/ffn_all_source_paths_v1"
LEGACY = ROOT / "outputs/all_attention_ae_811_v1"
GEOMETRY = ROOT / "outputs/selected_geometry_scalars_811_v1"
MODELS, SEEDS, CONFIG = prior.MODELS, prior.SEEDS, prior.CONFIG
TRAINED = ("AE_logS_PV", "AE_logS_PV_kP", "AE_logS_PV_kVP", "kVP")
GROUPS = TRAINED + ("kP",)
PAIRS = (("AE_logS_PV_kP", "AE_logS_PV"),
         ("AE_logS_PV_kVP", "AE_logS_PV"),
         ("AE_logS_PV_kVP", "AE_logS_PV_kP"),
         ("kVP", "kP"))


def block(matrix, fields, name, layers):
    matrix = np.asarray(matrix, dtype=np.float64)
    return matrix[:, fields.index(name) * layers:(fields.index(name) + 1) * layers]


def build_features(model, legacy, all_source):
    layers = source.LAYERS[model]
    f = np.asarray(legacy["groups"]["legacy_visual"], dtype=np.float32)
    if f.ndim != 2 or f.shape[1] != 2 * layers:
        raise ValueError(f"{model}: unexpected legacy F dimensions")
    f1, f2 = all_source["groups"]["F1_raw"], all_source["groups"]["F2"]
    gross_p = block(f1, source.F1_FIELDS, "prompt_gross_norm", layers)
    gross_v = block(f1, source.F1_FIELDS, "visual_gross_norm", layers)
    net_p = block(f1, source.F1_FIELDS, "prompt_net_norm", layers)
    net_v = block(f1, source.F1_FIELDS, "visual_net_norm", layers)
    cos_pv = block(f2, source.F2_FIELDS, "cos_vp_out", layers)
    k_p = block(f2, source.F2_FIELDS, "prompt_within_source_kappa", layers)
    s_pv = gross_p + gross_v
    if np.any(s_pv <= 0) or not np.isfinite(s_pv).all():
        raise ValueError(f"{model}: undefined P+V strength")
    np.testing.assert_allclose(net_p / gross_p, k_p, rtol=2e-6, atol=2e-6)
    numerator_sq = net_p**2 + net_v**2 + 2 * net_p * net_v * cos_pv
    if np.min(numerator_sq) < -1e-7:
        raise ValueError(f"{model}: negative P+V net norm square")
    k_pv = np.sqrt(np.maximum(numerator_sq, 0)) / s_pv
    if not all(np.isfinite(x).all() and (x >= 0).all() and (x <= 1 + 2e-6).all()
               for x in (k_p, k_pv)):
        raise ValueError(f"{model}: invalid cancellation ratio")
    ae = f[:, :layers]
    base = np.concatenate((ae, np.log1p(s_pv).astype(np.float32)), axis=1)
    groups = {
        "AE_logS_PV": base,
        "AE_logS_PV_kP": np.concatenate((base, k_p.astype(np.float32)), axis=1),
        "AE_logS_PV_kVP": np.concatenate((base, k_pv.astype(np.float32)), axis=1),
        "kVP": k_pv.astype(np.float32),
    }
    return groups, dict(S_PV=s_pv.astype(np.float32), kP=k_p.astype(np.float32),
                        kVP=k_pv.astype(np.float32))


def prepare(model):
    legacy = prior.read(LEGACY / model / "matrices.pt")
    all_source = prior.read(SOURCE / model / "matrices.pt")
    geometry = prior.read(GEOMETRY / model / "features.pt")
    for other in (all_source, geometry):
        if legacy["mentions"] != other["mentions"]:
            raise ValueError(f"{model}: mention order mismatch")
        np.testing.assert_array_equal(legacy["y"], other["y"])
        if "masks" in other:
            for part in ("train", "validation", "test"):
                np.testing.assert_array_equal(legacy["masks"][part], other["masks"][part])
    groups, signals = build_features(model, legacy, all_source)
    np.testing.assert_allclose(signals["kP"],
        geometry["groups"]["prompt_within_source_kappa"], rtol=0, atol=0)
    masks = {part: np.asarray(legacy["masks"][part], dtype=bool)
             for part in ("train", "validation", "test")}
    split = json.loads((LEGACY / model / "protocol.json").read_text())["split"]
    if [len(split[part]) for part in masks] != [3200, 400, 400]:
        raise ValueError(f"{model}: unexpected fixed 811 split")
    protocol = dict(schema="all-attention-pv-kappa-811-v1", model=model,
                    source_fingerprint=all_source["fingerprint"],
                    legacy_fingerprint=legacy["fingerprint"],
                    geometry_fingerprint=geometry["fingerprint"], split=split,
                    seeds=list(SEEDS), config=CONFIG, trained=list(TRAINED),
                    definition="true-RMS all-attention K32; S_PV=sum_P||e_m||+sum_V||e_m||; kP and kVP use token-level gross denominators",
                    transform="raw AE and kappa, log1p(S_PV), then train-only columnwise z-score",
                    bootstrap=dict(replicates=2000, seed=20260916, unit="400 test images", pairs=PAIRS),
                    caveat="Test split previously viewed; exploratory and uncorrected intervals")
    protocol = json.loads(json.dumps(protocol))
    digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode())
    digest.update(np.asarray(legacy["y"]).tobytes())
    for name, x in groups.items():
        digest.update(name.encode()); digest.update(x.tobytes())
    protocol["fingerprint"] = digest.hexdigest()
    path = OUT / model / "protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise ValueError(f"{model}: stale protocol")
    else:
        atomic_json_save(protocol, path)
    data = dict(groups=groups, signals=signals, mentions=legacy["mentions"],
                y=np.asarray(legacy["y"]), masks=masks, fingerprint=protocol["fingerprint"])
    path = OUT / model / "features.pt"
    if path.exists():
        if prior.read(path)["fingerprint"] != protocol["fingerprint"]:
            raise ValueError(f"{model}: stale feature cache")
    else:
        atomic_torch_save(data, path)
    return data, protocol


def reference(model, data):
    heads = {}
    for seed in SEEDS:
        head = prior.read(GEOMETRY / model / "heads/prompt_within_source_kappa" / f"seed{seed}.pt")
        if head["config"] != CONFIG or len(head["test_probabilities"]) != int(data["masks"]["test"].sum()):
            raise ValueError(f"{model}: incompatible kP reference")
        heads["kP", seed] = head
    return heads


def run_model(model, device):
    torch.set_num_threads(1)
    data, protocol = prepare(model)
    y, masks = data["y"], data["masks"]
    rows, checks = [], []
    for group, x in data["groups"].items():
        for seed in SEEDS:
            path = OUT / model / "heads" / group / f"seed{seed}.pt"
            if path.exists():
                result = prior.read(path)
            else:
                result = prior.geometry.trainer.fit(
                    x[masks["train"]], y[masks["train"]],
                    x[masks["validation"]], y[masks["validation"]], CONFIG, seed, device)
                net = prior.geometry.trainer.SingleMLP(x.shape[1], CONFIG).to(device)
                net.load_state_dict(result["state_dict"])
                p = prior.geometry.trainer.predict(net, torch.as_tensor(
                    prior.geometry.trainer.transform(x[masks["test"]], result["mean"], result["scale"]), device=device))
                result.update(fingerprint=protocol["fingerprint"], group=group,
                              test_probabilities=p,
                              test=prior.geometry.trainer.metrics(y[masks["test"]], p))
                atomic_torch_save(result, path)
                del net
            error = prior.validate_result(result, group, x, y, masks, protocol)
            rows.append(dict(model=model, feature=group, seed=seed, input_dim=x.shape[1],
                             best_epoch=result["best_epoch"],
                             validation_AUROC=result["validation"]["AUROC"],
                             validation_HALL_AUPR=result["validation"]["HALL_AUPR"], **result["test"]))
            checks.append(dict(feature=group, seed=seed, max_probability_error=error, status="PASS"))
            prior.write_csv(OUT / model / "seed_metrics.csv", rows)
            atomic_json_save(checks, OUT / model / "validation.json")
            print(model, group, seed, result["test"], flush=True)


def summarize():
    rows, intervals, mechanisms, validation = [], [], [], {}
    for model in MODELS:
        data, protocol = prepare(model)
        checks = json.loads((OUT / model / "validation.json").read_text())
        if len(checks) != len(TRAINED) * len(SEEDS) or any(r["status"] != "PASS" for r in checks):
            raise ValueError(f"{model}: incomplete validation")
        validation[model] = max(r["max_probability_error"] for r in checks)
        heads = reference(model, data)
        for group, x in data["groups"].items():
            for seed in SEEDS:
                head = prior.read(OUT / model / "heads" / group / f"seed{seed}.pt")
                prior.validate_result(head, group, x, data["y"], data["masks"], protocol)
                heads[group, seed] = head
        for group in GROUPS:
            scores = [heads[group, seed]["test"] for seed in SEEDS]
            row = dict(model=model, feature=group)
            for metric in ("AUROC", "HALL_AUPR"):
                values = [score[metric] for score in scores]
                row[metric + "_mean"] = float(np.mean(values))
                row[metric + "_std"] = float(np.std(values))
            rows.append(row)
        for scope, mask in (("all", np.ones(len(data["y"]), dtype=bool)),
                            ("test", data["masks"]["test"])):
            for name in ("kP", "kVP"):
                x = data["signals"][name][mask]
                labels = data["y"][mask]
                for layer in range(x.shape[1]):
                    real, hall = x[labels == 1, layer], x[labels == 0, layer]
                    mechanisms.append(dict(model=model, scope=scope, signal=name, layer=layer + 1,
                                           real_mean=float(np.mean(real)), hall_mean=float(np.mean(hall)),
                                           hall_minus_real=float(np.mean(hall)-np.mean(real)),
                                           real_median=float(np.median(real)), hall_median=float(np.median(hall)),
                                           real_count=len(real), hall_count=len(hall)))
        mask = data["masks"]["test"]
        y = data["y"][mask]
        ids = np.asarray([m["image_id"] for m in data["mentions"]])[mask]
        images = np.asarray(sorted(protocol["split"]["test"]))
        mapping = {image: index for index, image in enumerate(images)}
        cluster = np.asarray([mapping[image] for image in ids])
        p = np.stack([np.stack([heads[group, seed]["test_probabilities"] for seed in SEEDS])
                      for group in GROUPS])
        base = np.asarray([[prior.weighted_scores(y, prediction, np.ones(len(y)))
                            for prediction in group] for group in p]).mean(1)
        rng = np.random.default_rng(20260916)
        samples = []
        for _ in range(2000):
            counts = np.bincount(rng.integers(len(images), size=len(images)), minlength=len(images))
            weights = counts[cluster]
            if not all(weights[y == label].sum() > 0 for label in (0, 1)):
                continue
            samples.append(np.asarray([[prior.weighted_scores(y, prediction, weights)
                                        for prediction in group] for group in p]).mean(1))
        samples = np.asarray(samples)
        for left, right in PAIRS:
            li, ri = GROUPS.index(left), GROUPS.index(right)
            for index, metric in enumerate(("AUROC", "HALL_AUPR")):
                delta = samples[:, li, index] - samples[:, ri, index]
                low, high = np.quantile(delta, [.025, .975])
                intervals.append(dict(model=model, left=left, right=right, metric=metric,
                                      difference=base[li, index]-base[ri, index],
                                      ci95_low=low, ci95_high=high,
                                      valid_replicates=len(delta)))
        print(model, "bootstrap complete", len(samples), flush=True)
    prior.write_csv(OUT / "detection_summary.csv", rows)
    prior.write_csv(OUT / "bootstrap.csv", intervals)
    prior.write_csv(OUT / "kappa_layers.csv", mechanisms)
    atomic_json_save(dict(detection=rows, bootstrap=intervals, validation=validation), OUT / "summary.json")


def mechanism():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = list(csv.DictReader((OUT / "kappa_layers.csv").open()))
    paired = []
    for model in MODELS:
        data = prior.read(OUT / model / "features.pt")
        mentions, labels = data["mentions"], np.asarray(data["y"])
        target_labels = defaultdict(set)
        for mention, label in zip(mentions, labels):
            target_labels[mention["target_key"]].add(int(label))
        valid = np.asarray([len(target_labels[m["target_key"]]) == 1 for m in mentions])
        image_ids = np.asarray([m["image_id"] for m in mentions])
        for scope, selection in (("all", np.ones(len(labels), dtype=bool)),
                                 ("test", data["masks"]["test"])):
            selection = selection & valid
            for signal in ("kP", "kVP"):
                values = data["signals"][signal]
                differences = []
                for image_id in np.unique(image_ids[selection]):
                    current = selection & (image_ids == image_id)
                    real, hall = current & (labels == 1), current & (labels == 0)
                    if real.any() and hall.any():
                        differences.append(values[hall].mean(0) - values[real].mean(0))
                differences = np.asarray(differences)
                if not len(differences):
                    raise ValueError(f"{model}: no paired {scope} images")
                rng = np.random.default_rng(20260916)
                boot = differences[rng.integers(len(differences), size=(2000, len(differences)))].mean(1)
                for li in range(values.shape[1]):
                    low, high = np.quantile(boot[:, li], [.025, .975])
                    paired.append(dict(model=model, scope=scope, signal=signal, layer=li + 1,
                                       paired_images=len(differences),
                                       hall_minus_real=float(differences[:, li].mean()),
                                       ci95_low=float(low), ci95_high=float(high)))
    prior.write_csv(OUT / "kappa_paired_images.csv", paired)
    fig, axes = plt.subplots(2, 4, figsize=(18, 7), squeeze=False)
    for col, model in enumerate(MODELS):
        for row, signal in enumerate(("kP", "kVP")):
            subset = sorted((r for r in curves if r["model"] == model and r["scope"] == "test"
                             and r["signal"] == signal), key=lambda r: int(r["layer"]))
            layer = [int(r["layer"]) for r in subset]
            ax = axes[row, col]
            ax.plot(layer, [float(r["real_mean"]) for r in subset], label="REAL")
            ax.plot(layer, [float(r["hall_mean"]) for r in subset], label="HALL")
            ax.set_title(f"{model} | {signal}")
            ax.set_xlabel("Layer")
            ax.set_ylabel("mean κ")
            ax.grid(alpha=.2)
    axes[0, 0].legend()
    fig.suptitle("All-attention true RMS | test mentions | P vs P+V token cancellation")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(OUT / f"kappa_test.{extension}", dpi=160)
    plt.close(fig)


def report():
    result = json.loads((OUT / "summary.json").read_text())
    paired = list(csv.DictReader((OUT / "kappa_paired_images.csv").open()))
    names = {"AE_logS_PV": "AE_V+log1p(S_PV)",
             "AE_logS_PV_kP": "AE_V+log1p(S_PV)+κ_P",
             "AE_logS_PV_kVP": "AE_V+log1p(S_PV)+κ_V∪P",
             "kP": "κ_P-only", "kVP": "κ_V∪P-only"}
    lines = ["# 同一 All-attention 分解的 S_PV、κ_P 与 κ_V∪P：固定811检测", "",
             "本表只使用真RMS All-attention K32路径 `z−A_all→z` 的逐token FFN响应 `e_m`：`S_PV=Σ_{m∈P∪V}||e_m||`；两个κ的分母均为各自token集合的范数和，κ_V∪P不是两个组向量的κ。AE_V是所有实验共用的视觉语义注意力块，不混入旧Visual-only路径的S。",
             "", "固定4000图及3200/400/400图片级811划分，全部mentions、seeds43/44/45、训练集逐列z-score、128/ReLU/dropout0.3/无BN单隐藏层、最低validation BCE-loss checkpoint，无HPO及新VLM前向。κ_P-only复用同设置旧头，其余四组共48个新头。测试集此前已查看，属于探索性比较。",
             "", "## 测试集 AUROC / HALL-AUPR（%，三seed均值±总体std）", "",
             "| 特征 | Qwen2.5 | LLaVA | Qwen3 | InternVL |", "|---|---:|---:|---:|---:|"]
    for group in GROUPS:
        cells = []
        for model in MODELS:
            row = next(r for r in result["detection"] if r["model"] == model and r["feature"] == group)
            cells.append(" / ".join(f"{100*row[m+'_mean']:.2f}±{100*row[m+'_std']:.2f}"
                                    for m in ("AUROC", "HALL_AUPR")))
        lines.append("| " + names[group] + " | " + " | ".join(cells) + " |")
    lines += ["", "## 测试图片内 REAL/HALL 的κ差异", "",
              "先排除同一target标签冲突，再对每张同时有两类mention的图片算HALL−REAL均值，最后跨图片等权平均；下表为正差层数，不是独立图片数。逐层的2000次配对图片bootstrap区间见CSV。",
              "", "| 模型 | 同图图片数 | κ_P正差层数 | κ_V∪P正差层数 | 层均κ_P差 | 层均κ_V∪P差 |",
              "|---|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        blocks = {name: [r for r in paired if r["model"] == model and r["scope"] == "test"
                         and r["signal"] == name] for name in ("kP", "kVP")}
        p, vp = blocks["kP"], blocks["kVP"]
        lines.append(f"| {model} | {p[0]['paired_images']} | "
                     f"{sum(float(r['hall_minus_real'])>0 for r in p)}/{len(p)} | "
                     f"{sum(float(r['hall_minus_real'])>0 for r in vp)}/{len(vp)} | "
                     f"{np.mean([float(r['hall_minus_real']) for r in p]):+.4f} | "
                     f"{np.mean([float(r['hall_minus_real']) for r in vp]):+.4f} |")
    lines += ["", "## 配对测试图片bootstrap差值（pp，名义95%区间）", "",
              "| 模型 | 比较 | 指标 | 差值 [95% CI] |", "|---|---|---|---:|"]
    for row in result["bootstrap"]:
        lines.append(f"| {row['model']} | {names[row['left']]} − {names[row['right']]} | "
                     f"{row['metric']} | {100*row['difference']:+.2f} "
                     f"[{100*row['ci95_low']:+.2f}, {100*row['ci95_high']:+.2f}] |")
    lines += ["", "每模型2000次以全部400张测试图片为抽样框架的配对bootstrap，含无mention图片；未作多重比较校正。48个新头均CPU重载复算，最大三划分概率误差 "
              + "；".join(f"{model}={error:.3g}" for model, error in result["validation"].items()) + "。",
              "", "[逐层曲线](kappa_test.png) · [同图差值](kappa_paired_images.csv) · [检测CSV](detection_summary.csv) · [bootstrap CSV](bootstrap.csv) · [完整JSON](summary.json)"]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--mechanism", action="store_true")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    if args.report:
        report()
    elif args.mechanism:
        mechanism()
    elif args.summarize:
        summarize()
    elif args.models:
        for model in args.models:
            if args.prepare_only:
                prepare(model)
            else:
                run_model(model, args.device)
    else:
        parser.error("Provide --models or --summarize")


if __name__ == "__main__":
    main()
