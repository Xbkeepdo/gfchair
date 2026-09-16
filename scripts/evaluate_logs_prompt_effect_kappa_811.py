"""Evaluate log1p(S_E) + prompt FFN-effect kappa on the fixed 811 split."""
import argparse
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
from scripts import evaluate_legacy_visual_geometry_fusion_811 as prior

LEGACY = ROOT / "outputs/all_attention_ae_811_v1"
GEOMETRY = ROOT / "outputs/selected_geometry_scalars_811_v1"
FUSION = ROOT / "outputs/legacy_visual_geometry_fusion_811_v1"
OUT = ROOT / "outputs/logs_prompt_effect_kappa_811_v1"
MODELS, SEEDS, CONFIG = prior.MODELS, prior.SEEDS, prior.CONFIG
TRAINED = ("logS", "logS_plus_kappa")
GROUPS = ("logS", "kappa", "logS_plus_kappa", "F_plus_kappa")
PAIRS = (
    ("logS_plus_kappa", "logS"),
    ("logS_plus_kappa", "kappa"),
    ("F_plus_kappa", "logS_plus_kappa"),
)


def prepare(model):
    legacy = prior.read(LEGACY / model / "matrices.pt")
    geometry = prior.read(GEOMETRY / model / "features.pt")
    fusion = prior.read(FUSION / model / "features.pt")
    for other in (geometry, fusion):
        if legacy["mentions"] != other["mentions"]:
            raise ValueError(f"{model}: mention order differs")
        np.testing.assert_array_equal(legacy["y"], other["y"])
        for part in ("train", "validation", "test"):
            np.testing.assert_array_equal(legacy["masks"][part], other["masks"][part])
    f = np.asarray(legacy["groups"]["legacy_visual"], dtype=np.float32)
    kappa = np.asarray(geometry["groups"]["prompt_within_source_kappa"], dtype=np.float32)
    if f.ndim != 2 or f.shape[1] != 2 * kappa.shape[1] or f.shape[0] != kappa.shape[0]:
        raise ValueError(f"{model}: unexpected AE/logS/kappa dimensions")
    groups = {
        "logS": f[:, kappa.shape[1]:].copy(),
        "logS_plus_kappa": np.concatenate((f[:, kappa.shape[1]:], kappa), axis=1),
    }
    if not all(np.isfinite(x).all() for x in groups.values()):
        raise ValueError(f"{model}: non-finite features")
    np.testing.assert_array_equal(
        np.concatenate((f, kappa), axis=1),
        fusion["groups"]["legacy_plus_prompt_kappa"],
    )
    split = json.loads((LEGACY / model / "protocol.json").read_text())["split"]
    if [len(split[part]) for part in ("train", "validation", "test")] != [3200, 400, 400]:
        raise ValueError(f"{model}: expected fixed 811 split")
    masks = {part: np.asarray(legacy["masks"][part], dtype=bool)
             for part in ("train", "validation", "test")}
    if np.any(sum(mask.astype(np.uint8) for mask in masks.values()) != 1):
        raise ValueError(f"{model}: masks do not partition mentions")
    protocol = dict(
        schema="logs-prompt-effect-kappa-811-v1", model=model,
        legacy_fingerprint=legacy["fingerprint"],
        geometry_fingerprint=geometry["fingerprint"],
        fusion_fingerprint=fusion["fingerprint"],
        split=split, seeds=list(SEEDS), config=CONFIG,
        features={"logS": "visual-only true-RMS log1p(S_E), all layers",
                  "kappa": "||sum_prompt e_m|| / sum_prompt ||e_m||, all layers"},
        trained=list(TRAINED), references=["kappa", "F_plus_kappa"],
        transform="concatenate blocks, then train-only per-column population z-score",
        bootstrap=dict(replicates=2000, seed=20260916, unit="400 test images", pairs=PAIRS),
        caveat="Previously viewed test split; exploratory, uncorrected nominal intervals",
    )
    protocol = json.loads(json.dumps(protocol))
    digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode())
    digest.update(np.asarray(legacy["y"]).tobytes())
    for name, x in groups.items():
        digest.update(name.encode())
        digest.update(x.tobytes())
    protocol["fingerprint"] = digest.hexdigest()
    path = OUT / model / "protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise ValueError(f"{model}: stale protocol")
    else:
        atomic_json_save(protocol, path)
    data = dict(groups=groups, mentions=legacy["mentions"], y=np.asarray(legacy["y"]),
                masks=masks, fingerprint=protocol["fingerprint"])
    feature_path = OUT / model / "features.pt"
    if feature_path.exists():
        if prior.read(feature_path)["fingerprint"] != protocol["fingerprint"]:
            raise ValueError(f"{model}: stale feature cache")
    else:
        atomic_torch_save(data, feature_path)
    return data, protocol


def references(model, data):
    locations = {
        "kappa": GEOMETRY / model / "heads/prompt_within_source_kappa",
        "F_plus_kappa": FUSION / model / "heads/legacy_plus_prompt_kappa",
    }
    result = {}
    for name, folder in locations.items():
        for seed in SEEDS:
            head = prior.read(folder / f"seed{seed}.pt")
            if head["config"] != CONFIG or len(head["test_probabilities"]) != int(data["masks"]["test"].sum()):
                raise ValueError(f"{model}: invalid reference {name} seed{seed}")
            result[name, seed] = head
    return result


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
                    x[masks["validation"]], y[masks["validation"]],
                    CONFIG, seed, device,
                )
                net = prior.geometry.trainer.SingleMLP(x.shape[1], CONFIG).to(device)
                net.load_state_dict(result["state_dict"])
                test_probability = prior.geometry.trainer.predict(
                    net, torch.as_tensor(prior.geometry.trainer.transform(
                        x[masks["test"]], result["mean"], result["scale"]), device=device),
                )
                result.update(fingerprint=protocol["fingerprint"], group=group,
                              test_probabilities=test_probability,
                              test=prior.geometry.trainer.metrics(y[masks["test"]], test_probability))
                atomic_torch_save(result, path)
                del net
            error = prior.validate_result(result, group, x, y, masks, protocol)
            rows.append(dict(model=model, feature=group, seed=seed, input_dim=x.shape[1],
                             best_epoch=result["best_epoch"],
                             validation_AUROC=result["validation"]["AUROC"],
                             validation_HALL_AUPR=result["validation"]["HALL_AUPR"],
                             **result["test"]))
            checks.append(dict(feature=group, seed=seed, status="PASS", max_probability_error=error))
            prior.write_csv(OUT / model / "seed_metrics.csv", rows)
            atomic_json_save(checks, OUT / model / "validation.json")
            print(model, group, seed, result["test"], flush=True)
    atomic_json_save(dict(status="complete", heads=len(rows)), OUT / model / "progress.json")


def summarize():
    summary, seeds, intervals, checks = [], [], [], {}
    for model in MODELS:
        data, protocol = prepare(model)
        y = data["y"][data["masks"]["test"]]
        saved = list(csv.DictReader((OUT / model / "seed_metrics.csv").open()))
        if len(saved) != len(TRAINED) * len(SEEDS):
            raise ValueError(f"{model}: incomplete heads")
        seeds.extend(saved)
        validation = json.loads((OUT / model / "validation.json").read_text())
        if len(validation) != len(saved) or any(row["status"] != "PASS" for row in validation):
            raise ValueError(f"{model}: failed head validation")
        checks[model] = max(row["max_probability_error"] for row in validation)
        heads = references(model, data)
        for group, x in data["groups"].items():
            for seed in SEEDS:
                heads[group, seed] = prior.read(OUT / model / "heads" / group / f"seed{seed}.pt")
                prior.validate_result(heads[group, seed], group, x, data["y"], data["masks"], protocol)
        for group in GROUPS:
            values = [heads[group, seed]["test"] for seed in SEEDS]
            row = dict(model=model, feature=group)
            for metric in ("AUROC", "HALL_AUPR"):
                scores = [value[metric] for value in values]
                row[metric + "_mean"] = float(np.mean(scores))
                row[metric + "_std"] = float(np.std(scores))
            summary.append(row)
        mask = data["masks"]["test"]
        test_ids = np.asarray([m["image_id"] for m in data["mentions"]])[mask]
        images = np.asarray(sorted(protocol["split"]["test"]))
        mapping = {image: index for index, image in enumerate(images)}
        cluster = np.asarray([mapping[image] for image in test_ids])
        probabilities = np.stack([
            np.stack([heads[group, seed]["test_probabilities"] for seed in SEEDS])
            for group in GROUPS
        ])
        base = np.asarray([
            [prior.weighted_scores(y, p, np.ones(len(y))) for p in group]
            for group in probabilities
        ]).mean(1)
        rng = np.random.default_rng(20260916)
        samples = []
        for _ in range(2000):
            counts = np.bincount(rng.integers(len(images), size=len(images)), minlength=len(images))
            weights = counts[cluster]
            if not all(weights[y == label].sum() > 0 for label in (0, 1)):
                continue
            samples.append(np.asarray([
                [prior.weighted_scores(y, p, weights) for p in group]
                for group in probabilities
            ]).mean(1))
        samples = np.asarray(samples)
        for left, right in PAIRS:
            li, ri = GROUPS.index(left), GROUPS.index(right)
            for index, metric in enumerate(("AUROC", "HALL_AUPR")):
                delta = samples[:, li, index] - samples[:, ri, index]
                low, high = np.quantile(delta, [.025, .975])
                intervals.append(dict(model=model, left=left, right=right, metric=metric,
                                      difference=base[li, index] - base[ri, index],
                                      ci95_low=low, ci95_high=high,
                                      valid_replicates=len(delta)))
        print(model, "bootstrap complete", len(samples), flush=True)
    prior.write_csv(OUT / "detection_summary.csv", summary)
    prior.write_csv(OUT / "seed_metrics.csv", seeds)
    prior.write_csv(OUT / "bootstrap.csv", intervals)
    atomic_json_save(dict(detection=summary, bootstrap=intervals, validation=checks), OUT / "summary.json")
    names = {"logS": "log1p(S_E)", "kappa": "κ_P^e",
             "logS_plus_kappa": "log1p(S_E)+κ_P^e",
             "F_plus_kappa": "AE_V+log1p(S_E)+κ_P^e"}
    lines = ["# log1p(S_E) + κ_P^e：固定811幻觉检测", "",
             "S_E为Visual-only真实RMS条件路径的逐视觉token FFN响应范数和；κ_P^e=||Σ_P e_m||/Σ_P||e_m||，来自All-attention K32。两块各取全层向量，直接拼接后仅用训练集拟合逐列z-score；不包含AE_V。",
             "", "图片级3200/400/400，保留全部mentions；seeds43/44/45，单隐藏128/ReLU/dropout0.3、无BN、Adam，最低validation BCE-loss checkpoint，无HPO、无新VLM前向。κ-only与AE+logS+κ直接复用同协议已有头。测试集已多次查看，以下为探索性结果。",
             "", "## 测试集 AUROC / HALL-AUPR（%，三seed均值±总体std）", "",
             "| 特征 | Qwen2.5 | LLaVA | Qwen3 | InternVL |", "|---|---:|---:|---:|---:|"]
    for group in GROUPS:
        cells = []
        for model in MODELS:
            row = next(r for r in summary if r["model"] == model and r["feature"] == group)
            cells.append(" / ".join(f"{100*row[m+'_mean']:.2f}±{100*row[m+'_std']:.2f}"
                                    for m in ("AUROC", "HALL_AUPR")))
        lines.append("| " + names[group] + " | " + " | ".join(cells) + " |")
    lines += ["", "## 配对测试图片bootstrap差值（pp，名义95%区间）", "",
              "| 模型 | 比较 | 指标 | 差值 [95% CI] |", "|---|---|---|---:|"]
    for row in intervals:
        lines.append(f"| {row['model']} | {names[row['left']]} − {names[row['right']]} | {row['metric']} | "
                     f"{100*row['difference']:+.2f} [{100*row['ci95_low']:+.2f}, {100*row['ci95_high']:+.2f}] |")
    lines += ["", "每模型2000次以400张测试图片为簇的配对bootstrap（包括无mention图片）；区间未作多重比较校正。24个新头均CPU重载、train-only scaler、最低val-loss checkpoint和三划分概率复算通过；最大概率误差 "
              + "；".join(f"{model}={value:.3g}" for model, value in checks.items()) + "。",
              "", "[逐seed](seed_metrics.csv) · [bootstrap](bootstrap.csv) · [完整JSON](summary.json)"]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
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
