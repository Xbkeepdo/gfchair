"""Evaluate a direct VP/G ratio feature on the fixed 8:1:1 detector split."""

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
from scripts import evaluate_ours_batchsize_vs_svar_811 as batch_sweep
from scripts import search_single_mlp_811 as search


MODELS = tuple(search.MODELS)
SEEDS = (43, 44, 45)
BATCH_SIZE = 128
OUT = ROOT / "outputs/vp_over_g_standardized_no_bn_batch128_811_v1"
SIGNALS = ROOT / "outputs/all_attention_ae_log_strength_v1"
CONTROLLED = ROOT / "outputs/ours_batchsize_sweep_vs_svar_811_v1/summary.json"
SVAR = ROOT / "outputs/svar_llava_5_18_proportional_811_v1/summary.json"


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def build_ratio_features(signals):
    """Return raw [AE_VP/AE_G, log(S_VP/S_G)] and zero-filled detector input."""
    required = (
        "ae_visual_prompt_sum",
        "ae_generation",
        "gross_visual",
        "gross_prompt",
        "gross_generation",
    )
    arrays = {name: np.asarray(signals[name], dtype=np.float64) for name in required}
    shapes = {value.shape for value in arrays.values()}
    if len(shapes) != 1 or len(next(iter(shapes))) != 2:
        raise ValueError(f"Ratio sources must have one shared [mentions,layers] shape: {shapes}")

    ae_numerator = arrays["ae_visual_prompt_sum"]
    ae_denominator = arrays["ae_generation"]
    strength_numerator = arrays["gross_visual"] + arrays["gross_prompt"]
    strength_denominator = arrays["gross_generation"]

    ae_ratio = np.full_like(ae_numerator, np.nan)
    ae_valid = np.isfinite(ae_numerator) & np.isfinite(ae_denominator) & (ae_denominator != 0)
    np.divide(ae_numerator, ae_denominator, out=ae_ratio, where=ae_valid)

    strength_ratio = np.full_like(strength_numerator, np.nan)
    strength_valid = (
        np.isfinite(strength_numerator)
        & np.isfinite(strength_denominator)
        & (strength_numerator > 0)
        & (strength_denominator > 0)
    )
    np.divide(strength_numerator, strength_denominator, out=strength_ratio, where=strength_valid)
    log_strength_ratio = np.full_like(strength_ratio, np.nan)
    positive_ratio = np.isfinite(strength_ratio) & (strength_ratio > 0)
    np.log(strength_ratio, out=log_strength_ratio, where=positive_ratio)

    raw = np.concatenate((ae_ratio, log_strength_ratio), axis=1)
    detector = np.where(np.isfinite(raw), raw, 0.0).astype(np.float32)
    diagnostics = {
        "mentions": int(raw.shape[0]),
        "layers": int(ae_ratio.shape[1]),
        "input_dim": int(raw.shape[1]),
        "ae_denominator_zero": int(np.count_nonzero(ae_denominator == 0)),
        "strength_denominator_zero": int(np.count_nonzero(strength_denominator == 0)),
        "ae_nonfinite_raw": int(np.count_nonzero(~np.isfinite(ae_ratio))),
        "log_strength_nonfinite_raw": int(np.count_nonzero(~np.isfinite(log_strength_ratio))),
        "detector_nonfinite": int(np.count_nonzero(~np.isfinite(detector))),
    }
    return raw, detector, diagnostics


def save_ratio_features(path, raw, detector, layers):
    temporary = path.with_name(path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            ae_vp_over_ae_g=raw[:, :layers],
            log_strength_vp_over_g=raw[:, layers:],
            detector_matrix=detector,
        )
    os.replace(temporary, path)


def model_config(model_name):
    config = dict(batch_sweep.base_configs()[(model_name, "vp_generation")])
    config.update(
        batch_size=BATCH_SIZE,
        standardize=True,
        batch_norm=False,
        early_stopping=False,
        monitor="val_loss",
        max_epochs=150,
    )
    return config


def progress(model_name, completed, stage, status="running", epoch=None, error=None):
    value = {
        "stage": stage,
        "completed": completed,
        "total": len(SEEDS),
        "status": status,
        "heartbeat": datetime.now(timezone.utc).isoformat(),
    }
    if epoch is not None:
        value["epoch"] = epoch
    if error:
        value["error"] = str(error)[:180]
    atomic_json_save(value, OUT / model_name / "progress.json")


def prepare(model_name):
    source = search.read(search.SOURCES["true_rms"] / model_name / "matrices.pt")
    signal_path = SIGNALS / model_name / "signals.npz"
    with np.load(signal_path) as loaded:
        raw, detector, diagnostics = build_ratio_features(loaded)
    if len(detector) != len(source["mentions"]):
        raise ValueError(f"Signal/matrix mention mismatch: {len(detector)} != {len(source['mentions'])}")
    layers = diagnostics["layers"]
    vp_matrix = np.asarray(source["groups"]["visual_prompt_sum"], dtype=np.float64)
    generation_matrix = np.asarray(source["groups"]["generation"], dtype=np.float64)
    np.testing.assert_allclose(
        raw[:, :layers],
        vp_matrix[:, :layers] / generation_matrix[:, :layers],
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        raw[:, layers:],
        np.log(np.expm1(vp_matrix[:, layers:]) / np.expm1(generation_matrix[:, layers:])),
        rtol=2e-5,
        atol=2e-6,
    )
    config = model_config(model_name)
    protocol = {
        "schema": "vp-over-g-standardized-no-bn-batch128-811-v1",
        "model": model_name,
        "source": "True-RMS all-attention z-A_all -> z; local FP32 Gauss-Legendre K32",
        "source_fingerprint": source["fingerprint"],
        "raw_signal_file": str(signal_path.relative_to(ROOT)),
        "features": ["AE_VP / AE_G", "log((S_V + S_P) / S_G)"],
        "column_layout": "All L AE-ratio columns, followed by all L log-strength-ratio columns",
        "division": "Direct division without additive denominator offset",
        "undefined_policy": "Store undefined raw values as NaN; replace nonfinite detector inputs with zero; retain every mention; no validity indicator",
        "diagnostics": diagnostics,
        "config_source": "The model's fixed VP+G classifier configuration; only the input feature changes",
        "config": config,
        "batch_size": BATCH_SIZE,
        "seeds": list(SEEDS),
        "split_seed": 20260912,
        "split": "3200 train / 400 validation / 400 test images; all mentions",
        "normalization": "Per-column train-only population Z-score after nonfinite-to-zero mapping",
        "training": "No BatchNorm; all 150 epochs; retain minimum validation BCE loss checkpoint",
        "status": "Exploratory follow-up after test-set access; compare against fixed V, VP+G and proportional SVAR",
    }
    digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode())
    digest.update(np.ascontiguousarray(raw).tobytes())
    protocol["fingerprint"] = digest.hexdigest()
    model_root = OUT / model_name
    protocol_path = model_root / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("Protocol changed; refusing to reuse stale ratio results")
    atomic_json_save(protocol, protocol_path)
    save_ratio_features(model_root / "ratio_features.npz", raw, detector, layers)
    return source, detector, protocol


def run_model(model_name, device):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source, matrix, protocol = prepare(model_name)
    config = protocol["config"]
    train, validation, test = (source["masks"][part] for part in ("train", "validation", "test"))
    rows = []
    probabilities = []
    completed = 0
    progress(model_name, completed, "VP/G比值检测准备")
    try:
        for seed in SEEDS:
            result_path = OUT / model_name / f"seed{seed}" / "result.pt"
            if result_path.exists():
                result = search.read(result_path)
                if result["protocol_fingerprint"] != protocol["fingerprint"] or result["config"] != config:
                    raise ValueError("Stale VP/G result does not match the frozen protocol")
            else:
                stage = f"VP/G比值检测 seed{seed}"
                progress(model_name, completed, stage, epoch=0)
                last_heartbeat = [time.monotonic()]

                def epoch_callback(epoch):
                    now = time.monotonic()
                    if now - last_heartbeat[0] >= 9.0:
                        progress(model_name, completed, stage, epoch=epoch)
                        last_heartbeat[0] = now

                result = search.fit(
                    train_x=matrix[train],
                    train_y=source["y"][train],
                    val_x=matrix[validation],
                    val_y=source["y"][validation],
                    cfg=config,
                    seed=seed,
                    device=device,
                    callback=epoch_callback,
                )
                network = search.SingleMLP(result["input_dim"], config).to(device)
                network.load_state_dict(result["state_dict"])
                test_x = torch.as_tensor(
                    search.transform(matrix[test], result["mean"], result["scale"]), device=device
                )
                test_probabilities = search.predict(network, test_x)
                result.update(
                    protocol_fingerprint=protocol["fingerprint"],
                    model=model_name,
                    feature="vp_over_g",
                    test=search.metrics(source["y"][test], test_probabilities),
                    test_probabilities=test_probabilities,
                )
                atomic_torch_save(result, result_path)
            completed += 1
            progress(model_name, completed, f"VP/G比值检测 seed{seed}")
            probabilities.append(result["test_probabilities"])
            rows.append({
                "model": model_name,
                "seed": seed,
                "input_dim": result["input_dim"],
                "best_epoch": result["best_epoch"],
                "trained_epochs": result["epochs"],
                "validation_loss": result["history"][result["best_epoch"] - 1]["val_loss"],
                "validation_AUROC": result["validation"]["AUROC"],
                "validation_HALL_AUPR": result["validation"]["HALL_AUPR"],
                "test_AUROC": result["test"]["AUROC"],
                "test_HALL_AUPR": result["test"]["HALL_AUPR"],
            })
        summary = {
            "model": model_name,
            "feature": "vp_over_g",
            "input_dim": rows[0]["input_dim"],
            "AUROC_mean": float(np.mean([row["test_AUROC"] for row in rows])),
            "AUROC_std": float(np.std([row["test_AUROC"] for row in rows])),
            "HALL_AUPR_mean": float(np.mean([row["test_HALL_AUPR"] for row in rows])),
            "HALL_AUPR_std": float(np.std([row["test_HALL_AUPR"] for row in rows])),
            "best_epochs": [row["best_epoch"] for row in rows],
            "ensemble": search.metrics(source["y"][test], np.mean(probabilities, axis=0)),
            "diagnostics": protocol["diagnostics"],
        }
        write_csv(rows, OUT / model_name / "seed_metrics.csv")
        atomic_json_save(summary, OUT / model_name / "summary.json")
        progress(model_name, len(SEEDS), "VP/G比值检测完成", status="completed")
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    except BaseException as exc:
        progress(model_name, completed, "VP/G比值检测失败", status="failed", error=exc)
        raise


def comparison_rows():
    controlled_rows = json.loads(CONTROLLED.read_text())["summaries"]
    controlled = {
        (row["model"], row["group"], row["batch_size"]): row for row in controlled_rows
    }
    svar = {row["model"]: row for row in json.loads(SVAR.read_text())["summaries"]}
    rows = []
    for model_name in MODELS:
        ratio = json.loads((OUT / model_name / "summary.json").read_text())
        methods = {
            "V": controlled[(model_name, "visual", BATCH_SIZE)],
            "VP+G": controlled[(model_name, "vp_generation", BATCH_SIZE)],
            "VP/G": ratio,
            "SVAR": svar[model_name],
        }
        for method, value in methods.items():
            rows.append({
                "model": model_name,
                "method": method,
                "AUROC_mean": value["AUROC_mean"],
                "AUROC_std": value["AUROC_std"],
                "HALL_AUPR_mean": value["HALL_AUPR_mean"],
                "HALL_AUPR_std": value["HALL_AUPR_std"],
                "delta_AUROC_vs_SVAR": value["AUROC_mean"] - methods["SVAR"]["AUROC_mean"],
                "delta_HALL_AUPR_vs_SVAR": value["HALL_AUPR_mean"] - methods["SVAR"]["HALL_AUPR_mean"],
            })
    return rows


def plot_comparison(rows):
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    methods = ("V", "VP+G", "VP/G", "SVAR")
    x = np.arange(len(MODELS))
    width = 0.19
    for axis, metric, title in (
        (axes[0], "AUROC_mean", "Test AUROC"),
        (axes[1], "HALL_AUPR_mean", "Test HALL-AUPR"),
    ):
        for index, method in enumerate(methods):
            values = [next(row[metric] for row in rows if row["model"] == model and row["method"] == method) for model in MODELS]
            axis.bar(x + (index - 1.5) * width, values, width=width, label=method)
        axis.set_xticks(x, ("Qwen2.5", "LLaVA", "Qwen3", "InternVL"), rotation=15)
        axis.set_ylim(0, 1)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    axes[1].legend(ncol=2)
    figure.tight_layout()
    figure.savefig(OUT / "comparison.png", dpi=180)
    figure.savefig(OUT / "comparison.pdf")
    plt.close(figure)


def write_report(rows):
    lines = [
        "# VP/G 直接比值特征 811 检测结果",
        "",
        "## 实验设置",
        "",
        "- 四模型固定3200/400/400图片划分，全部mentions；seeds 43/44/45分别训练并报告均值与总体标准差。",
        "- 每层特征为 `AE_VP/AE_G` 与 `log((S_V+S_P)/S_G)`；直接相除，不添加分母偏移。",
        "- 未定义原始值保存NaN，检测输入固定置零，不删除mention，不加入有效性标记。实际四模型本轮均无零分母或非有限比值。",
        "- 单隐藏层MLP沿用各模型VP+G配置；StandardScaler只拟合train，无BatchNorm，batch128，完整150 epochs，最低validation BCE loss checkpoint。",
        "- V和VP+G取既有batch128受控实验，各自沿用已固定的组配置；VP/G与VP+G使用相同配置，因此二者是直接特征对照。",
        "- SVAR为原生248维单隐藏层分类器，LLaVA取第5–18层；按层数比例映射为Qwen2.5第4–16层、Qwen3第6–20层、InternVL第5–18层。",
        "- 这是查看过同一test split后的探索性补充，不作独立盲测或显著性结论。",
        "",
        "## 三seed测试结果",
        "",
        "| 模型 | 方法 | AUROC (%) | HALL-AUPR (%) | ΔAUROC vs SVAR (pp) | ΔAP vs SVAR (pp) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    names = {
        "qwen2_5_vl_7b": "Qwen2.5-VL-7B",
        "llava_1_5_7b": "LLaVA-1.5-7B",
        "qwen3_vl_8b": "Qwen3-VL-8B",
        "internvl_2_5_8b": "InternVL2.5-8B",
    }
    for row in rows:
        lines.append(
            f"| {names[row['model']]} | {row['method']} | "
            f"{100*row['AUROC_mean']:.2f} ± {100*row['AUROC_std']:.2f} | "
            f"{100*row['HALL_AUPR_mean']:.2f} ± {100*row['HALL_AUPR_std']:.2f} | "
            f"{100*row['delta_AUROC_vs_SVAR']:+.2f} | "
            f"{100*row['delta_HALL_AUPR_vs_SVAR']:+.2f} |"
        )
    lines.extend([
        "",
        "## VP/G 相对原特征",
        "",
        "| 模型 | ΔAUROC vs V | ΔAP vs V | ΔAUROC vs VP+G | ΔAP vs VP+G |",
        "|---|---:|---:|---:|---:|",
    ])
    for model_name in MODELS:
        by_method = {row["method"]: row for row in rows if row["model"] == model_name}
        ratio = by_method["VP/G"]
        visual = by_method["V"]
        concatenated = by_method["VP+G"]
        lines.append(
            f"| {names[model_name]} | "
            f"{100*(ratio['AUROC_mean']-visual['AUROC_mean']):+.2f} | "
            f"{100*(ratio['HALL_AUPR_mean']-visual['HALL_AUPR_mean']):+.2f} | "
            f"{100*(ratio['AUROC_mean']-concatenated['AUROC_mean']):+.2f} | "
            f"{100*(ratio['HALL_AUPR_mean']-concatenated['HALL_AUPR_mean']):+.2f} |"
        )
    lines.extend([
        "",
        "VP/G 的 AUROC 在四模型上都低于 V 和 VP+G。比值保留相对平衡，却丢掉 VP、G 各自的绝对强度；StandardScaler只能调整剩余比值列的尺度，不能恢复被除法消去的信息。除法还把分母变化耦合进全部比值，因此即使本轮没有零分母，也会放大低 G 区域的波动。结果不支持用 VP/G 替代原来的 VP+G 拼接。",
        "",
        "![四模型VP/G与对照](../outputs/vp_over_g_standardized_no_bn_batch128_811_v1/comparison.png)",
        "",
        "完整逐seed结果见 `outputs/vp_over_g_standardized_no_bn_batch128_811_v1/seed_metrics.csv`。",
    ])
    (ROOT / "docs/VP_OVER_G_811_RESULTS.md").write_text("\n".join(lines) + "\n")


def summarize_all():
    rows = comparison_rows()
    ratio_seed_rows = []
    for model_name in MODELS:
        with (OUT / model_name / "seed_metrics.csv").open() as handle:
            ratio_seed_rows.extend(csv.DictReader(handle))
    write_csv(rows, OUT / "comparison.csv")
    write_csv(ratio_seed_rows, OUT / "seed_metrics.csv")
    atomic_json_save(
        {"schema": "vp-over-g-standardized-no-bn-batch128-811-v1", "comparison": rows},
        OUT / "summary.json",
    )
    plot_comparison(rows)
    write_report(rows)
    print(json.dumps(rows, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize_all()
    elif args.models:
        for model_name in args.models:
            run_model(model_name, args.device)
    else:
        parser.error("Provide --models or --summarize")


if __name__ == "__main__":
    main()
