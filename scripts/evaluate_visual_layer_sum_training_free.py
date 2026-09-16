"""Evaluate fixed visual AE/strength layer-sum scores without a classifier."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.tc_fvpa_artifacts import atomic_json_save


SOURCE = ROOT / "outputs/all_attention_ae_log_strength_v1"
OUT = ROOT / "outputs/visual_layer_sum_training_free_4000_v1"
MODELS = (
    "qwen2_5_vl_7b",
    "llava_1_5_7b",
    "qwen3_vl_8b",
    "internvl_2_5_8b",
)
WEIGHTS = (0.0, 0.2, 0.5, 0.8, 1.0)


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def build_scores(ae_visual: np.ndarray, gross_visual: np.ndarray) -> dict[str, np.ndarray]:
    """Return layer sums and fixed evidence scores; no scaling or fitted parameters."""
    ae = np.asarray(ae_visual, dtype=np.float64)
    gross = np.asarray(gross_visual, dtype=np.float64)
    if ae.shape != gross.shape or ae.ndim != 2:
        raise ValueError(f"Expected matching [mentions,layers] arrays, got {ae.shape}, {gross.shape}")
    if not np.isfinite(ae).all() or not np.isfinite(gross).all():
        raise ValueError("Visual source arrays contain nonfinite values")
    if (ae < 0).any() or (gross < 0).any():
        raise ValueError("Visual AE and gross strengths must be nonnegative")

    ae_sum = ae.sum(axis=1, dtype=np.float64)
    strength_free = np.log1p(gross.sum(axis=1, dtype=np.float64))
    result = {"ae_sum": ae_sum, "strength_free": strength_free}
    for weight in WEIGHTS:
        result[f"evidence_w{weight:g}"] = (1.0 - weight) * strength_free + weight * ae_sum
    return result


def hall_metrics(y_real: np.ndarray, evidence: np.ndarray) -> dict[str, float]:
    y_hall = 1 - np.asarray(y_real, dtype=np.int64)
    risk = -np.asarray(evidence, dtype=np.float64)
    return {
        "AUROC": float(roc_auc_score(y_hall, risk)),
        "HALL_AUPR": float(average_precision_score(y_hall, risk)),
    }


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def load_model(model: str) -> tuple[dict, dict[str, np.ndarray], dict]:
    folder = SOURCE / model
    source = torch.load(folder / "matrices.pt", map_location="cpu", weights_only=False)
    with np.load(folder / "signals.npz") as loaded:
        ae = np.asarray(loaded["ae_visual"], dtype=np.float64)
        gross = np.asarray(loaded["gross_visual"], dtype=np.float64)
    scores = build_scores(ae, gross)

    y = np.asarray(source["y"], dtype=np.int32)
    mentions = source["mentions"]
    if len(y) != len(mentions) or len(y) != len(ae):
        raise ValueError(f"Mention alignment failed for {model}")
    np.testing.assert_array_equal(y, [mention["label"] for mention in mentions])
    layers = ae.shape[1]
    old_visual = np.asarray(source["groups"]["visual"], dtype=np.float64)
    np.testing.assert_allclose(ae, old_visual[:, :layers], rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(gross, np.expm1(old_visual[:, layers:]), rtol=2e-5, atol=2e-6)

    source_protocol = json.loads((folder / "protocol.json").read_text())
    split = source_protocol["image_split"]
    cohort_ids = set().union(*(set(split[part]) for part in ("train", "validation", "test")))
    if len(cohort_ids) != 4000:
        raise ValueError(f"Expected the 4000-image cohort, found {len(cohort_ids)}")
    mention_image_ids = np.asarray([mention["image_id"] for mention in mentions], dtype=np.int64)
    if not set(mention_image_ids).issubset(cohort_ids):
        raise ValueError("A mention image is outside the frozen 4000-image cohort")

    diagnostics = {
        "model": model,
        "cohort_images": len(cohort_ids),
        "images_with_mentions": int(len(np.unique(mention_image_ids))),
        "images_without_mentions": int(len(cohort_ids - set(mention_image_ids))),
        "mentions": int(len(y)),
        "real_mentions": int(np.count_nonzero(y == 1)),
        "hall_mentions": int(np.count_nonzero(y == 0)),
        "hall_prevalence": float(np.mean(y == 0)),
        "layers": int(layers),
        "ae_sum_mean": float(scores["ae_sum"].mean()),
        "ae_sum_std": float(scores["ae_sum"].std()),
        "strength_free_mean": float(scores["strength_free"].mean()),
        "strength_free_std": float(scores["strength_free"].std()),
    }
    return {"y": y, "image_ids": mention_image_ids}, scores, diagnostics


def make_plot(rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    for model in MODELS:
        selected = [row for row in rows if row["model"] == model]
        weights = [row["weight"] for row in selected]
        axes[0].plot(weights, [100 * row["AUROC"] for row in selected], marker="o", label=model)
        axes[1].plot(weights, [100 * row["HALL_AUPR"] for row in selected], marker="o", label=model)
    axes[0].set_ylabel("AUROC (%)")
    axes[1].set_ylabel("HALL-AUPR (%)")
    for axis in axes:
        axis.set_xlabel(r"$w$ in $(1-w)S_{FREE}+wA_V$")
        axis.set_xticks(WEIGHTS)
        axis.grid(alpha=0.25)
    axes[1].legend(fontsize=8, loc="best")
    fig.suptitle("Visual layer-sum training-free detector | full 4000-image cohort")
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"performance.{suffix}", dpi=220)
    plt.close(fig)


def make_report(rows: list[dict], diagnostics: list[dict]) -> None:
    lines = [
        "# 视觉AE与gross全层聚合：4000图training-free结果",
        "",
        "## 设置",
        "",
        "- 四模型原4000图cohort、全部mentions；无训练/验证/测试划分用于打分，无分类器、无标准化、无参数拟合。没有mention的图片不产生样本。",
        "- 信号复用真实RMS All-attention路径 `z-A_all -> z`、局部FP32、Gauss-Legendre K32缓存。",
        "- `A_V=sum_l AE_V(l)`；`S_FREE=log(1+sum_l S_V(l))`，先跨层合计原始visual gross，再做一次log1p。",
        "- 证据分数 `E_w=(1-w)S_FREE+wA_V`，w固定为0、0.2、0.5、0.8、1；HALL风险为`-E_w`。AUROC以HALL为正类，AUPR为HALL-AUPR。",
        "- 权重由用户预先指定；本表无随机种子方差。4000图cohort此前已反复查看，属于探索性结果。",
        "",
        "## AUROC / HALL-AUPR（%）",
        "",
        "| 模型 | w=0（S only） | w=0.2 | w=0.5 | w=0.8 | w=1（AE only） |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        selected = [row for row in rows if row["model"] == model]
        cells = [f"{100*row['AUROC']:.2f} / {100*row['HALL_AUPR']:.2f}" for row in selected]
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## 样本和尺度",
        "",
        "| 模型 | 层 | cohort图/有mention图 | mentions（REAL/HALL） | mean A_V | mean S_FREE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in diagnostics:
        lines.append(
            f"| {item['model']} | {item['layers']} | {item['cohort_images']}/{item['images_with_mentions']} | "
            f"{item['mentions']}（{item['real_mentions']}/{item['hall_mentions']}） | "
            f"{item['ae_sum_mean']:.3f} | {item['strength_free_mean']:.3f} |"
        )
    lines += [
        "",
        "AE的均值约为S_FREE的3到4倍，因此未经标准化时w=0.5及w=0.8主要由AE控制；w=0.2更接近数值尺度上的均衡。该实验检验用户指定的原始代数权重，不把w解释为实际贡献比例。",
        "",
        "逐mention分数见各模型的`scores.npz`；精确指标见`metrics.csv`，协议和完整性检查见`protocol.json`与`validation.json`。",
    ]
    temporary = OUT / "summary.md.tmp"
    temporary.write_text("\n".join(lines) + "\n")
    os.replace(temporary, OUT / "summary.md")
    docs = ROOT / "docs/AE_LOGS_LAYER_SUM_TRAINING_FREE_4000_RESULTS.md"
    temporary = docs.with_suffix(docs.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n")
    os.replace(temporary, docs)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {
        "schema": "visual-layer-sum-training-free-4000-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "models": list(MODELS),
        "cohort": "Original fixed COCO 4000-image cohort; all mentions across the full cohort",
        "source": "True-RMS all-attention z-A_all -> z; local FP32 Gauss-Legendre K32",
        "ae": "A_V = sum over all decoder layers of raw visual AE",
        "strength": "S_FREE = log(1 + sum over all decoder layers of raw visual gross S)",
        "evidence": "E_w = (1-w) * S_FREE + w * A_V",
        "weights": list(WEIGHTS),
        "hall_risk": "-E_w; lower visual evidence means higher hallucination risk",
        "normalization": "None",
        "fitting": "None; no classifier, label-dependent sign selection, or threshold selection",
        "metrics": "Mention-level AUROC with HALL positive and HALL-AUPR",
        "status": "Exploratory: the full cohort and labels have been inspected in earlier experiments",
    }
    atomic_json_save(protocol, OUT / "protocol.json")

    rows: list[dict] = []
    diagnostics: list[dict] = []
    for model in MODELS:
        identity, scores, audit = load_model(model)
        diagnostics.append(audit)
        arrays = {**identity, **scores}
        atomic_npz(OUT / model / "scores.npz", **arrays)
        for weight in WEIGHTS:
            metric = hall_metrics(identity["y"], scores[f"evidence_w{weight:g}"])
            rows.append({
                "model": model,
                "weight": weight,
                "feature": "S_FREE" if weight == 0 else "AE_V" if weight == 1 else "fixed_fusion",
                **metric,
                "mentions": audit["mentions"],
                "hall_mentions": audit["hall_mentions"],
                "hall_prevalence": audit["hall_prevalence"],
            })
    write_csv(rows, OUT / "metrics.csv")
    atomic_json_save({"models": diagnostics}, OUT / "validation.json")
    atomic_json_save({"protocol": protocol, "metrics": rows, "diagnostics": diagnostics}, OUT / "summary.json")
    make_plot(rows)
    make_report(rows, diagnostics)
    print((OUT / "summary.md").read_text())


if __name__ == "__main__":
    main()
