"""Plot per-layer prompt within-source kappa from the fixed-811 feature cache."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/selected_geometry_scalars_811_v1"
OUT = ROOT / "outputs/legacy_visual_geometry_fusion_811_v1/prompt_kappa"
MODELS = (
    ("qwen2_5_vl_7b", "Qwen2.5-VL"),
    ("llava_1_5_7b", "LLaVA-1.5"),
    ("qwen3_vl_8b", "Qwen3-VL"),
    ("internvl_2_5_8b", "InternVL2.5"),
)
SCOPES = ("all", "test")


def summarize(x, labels, mask, model, scope):
    """Use mention-weighted raw kappa; REAL=1, HALL=0."""
    rows = []
    for label_name, label_value in (("REAL", 1), ("HALL", 0)):
        values = x[mask & (labels == label_value)]
        if not len(values):
            raise ValueError(f"Empty {model} {scope} {label_name} class")
        for layer in range(values.shape[1]):
            column = values[:, layer].astype(np.float64)
            rows.append(dict(
                model=model,
                scope=scope,
                label=label_name,
                layer=layer + 1,
                n_mentions=len(values),
                mean=float(np.mean(column)),
                median=float(np.median(column)),
                q25=float(np.quantile(column, .25)),
                q75=float(np.quantile(column, .75)),
            ))
    return rows


def plot(rows, scope):
    fig, axes = plt.subplots(2, 4, figsize=(19, 8), sharex="col")
    for index, (model, title) in enumerate(MODELS):
        ax, difference_ax = axes[:, index]
        selected = [row for row in rows if row["model"] == model and row["scope"] == scope]
        by_label = {}
        for label, color in (("REAL", "#2563eb"), ("HALL", "#dc2626")):
            subset = sorted((row for row in selected if row["label"] == label), key=lambda row: row["layer"])
            layers = np.array([row["layer"] for row in subset])
            means = np.array([row["mean"] for row in subset])
            ax.plot(layers, means, color=color, lw=2.2, label=f"{label} (n={subset[0]['n_mentions']:,})")
            ax.fill_between(
                layers,
                [row["q25"] for row in subset],
                [row["q75"] for row in subset],
                color=color, alpha=.11,
            )
            by_label[label] = means
        difference = by_label["HALL"] - by_label["REAL"]
        difference_ax.plot(layers, difference, color="#7c3aed", lw=2)
        difference_ax.axhline(0, color="#4b5563", lw=.8)
        difference_ax.fill_between(layers, 0, difference, color="#7c3aed", alpha=.13)
        ax.set_title(title)
        ax.set_ylim(.25, 1.02)
        ax.grid(alpha=.2)
        ax.legend(loc="lower right", fontsize=8)
        difference_ax.set_xlabel("Decoder layer")
        difference_ax.grid(alpha=.2)
        difference_ax.text(
            .02, .97,
            f"HALL > REAL: {(difference > 0).sum()}/{len(difference)} layers",
            transform=difference_ax.transAxes, ha="left", va="top", fontsize=9,
        )
    axes[0, 0].set_ylabel(r"$\kappa_P$ (mean; shaded IQR)")
    axes[1, 0].set_ylabel(r"HALL - REAL mean $\kappa_P$")
    fig.suptitle(
        rf"Prompt within-source $\kappa_P=\|\sum_{{m\in P}}e_m\|/\sum_{{m\in P}}\|e_m\|$ | {scope.upper()} | mention-weighted",
        fontsize=13,
    )
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(OUT / f"prompt_kappa_{scope}.{extension}", dpi=170)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, checks = [], []
    for model, _ in MODELS:
        source = torch.load(SOURCE / model / "features.pt", map_location="cpu", weights_only=False, mmap=True)
        x = np.asarray(source["groups"]["prompt_within_source_kappa"], dtype=np.float32)
        y = np.asarray(source["y"])
        if x.ndim != 2 or x.shape[0] != len(y) or not np.isfinite(x).all() or np.any((x <= 0) | (x > 1)):
            raise ValueError(f"Invalid prompt kappa for {model}")
        masks = {"all": np.ones(len(y), dtype=bool), "test": np.asarray(source["masks"]["test"], dtype=bool)}
        for scope, mask in masks.items():
            local = summarize(x, y, mask, model, scope)
            rows.extend(local)
            real = [row for row in local if row["label"] == "REAL"]
            hall = [row for row in local if row["label"] == "HALL"]
            difference = np.array([h["mean"] - r["mean"] for h, r in zip(hall, real)])
            checks.append(dict(
                model=model, scope=scope,
                images_with_mentions=len({m["image_id"] for m, keep in zip(source["mentions"], mask) if keep}),
                real_mentions=int((mask & (y == 1)).sum()),
                hall_mentions=int((mask & (y == 0)).sum()),
                positive_layers=int((difference > 0).sum()),
                layers=x.shape[1],
                mean_layer_difference=float(difference.mean()),
                real_layer_mean=float(np.mean([row["mean"] for row in real])),
                hall_layer_mean=float(np.mean([row["mean"] for row in hall])),
            ))
    for name, values in (("curves.csv", rows), ("summary.csv", checks)):
        with (OUT / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    for scope in SCOPES:
        plot(rows, scope)
    lines = [
        "# Prompt 组内 κ_P：REAL / HALL 逐层曲线",
        "",
        "复用固定811检测所用四模型4000图缓存（测试集400图），不重跑VLM、不训练检测器。横轴decoder layer；蓝色REAL，红色HALL；实线为mention加权均值，浅色带为各类mention的IQR；下排为HALL−REAL均值。缓存中κ_P均在(0,1]且无零值。",
        "",
        "| 模型 | all HALL−REAL层均值 | all正差层数 | test HALL−REAL层均值 | test正差层数 |",
        "|---|---:|---:|---:|---:|",
    ]
    for model, title in MODELS:
        all_row = next(row for row in checks if row["model"] == model and row["scope"] == "all")
        test_row = next(row for row in checks if row["model"] == model and row["scope"] == "test")
        lines.append(
            f"| {title} | {all_row['mean_layer_difference']:+.4f} | "
            f"{all_row['positive_layers']}/{all_row['layers']} | "
            f"{test_row['mean_layer_difference']:+.4f} | "
            f"{test_row['positive_layers']}/{test_row['layers']} |"
        )
    lines += [
        "",
        "κ_P越大，prompt来源内的token响应方向越一致、相互抵消越少。图显示HALL总体更高，但这是相关性描述，不证明prompt导致幻觉；两类mention数量、图片/文本长度等可能混杂。test集此前已查看，此图仅作探索性展示。",
        "",
        "[全部mention曲线](prompt_kappa_all.png) · [811测试曲线](prompt_kappa_test.png) · [逐层数据](curves.csv) · [样本数与汇总](summary.csv)",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    for row in checks:
        print(row)


if __name__ == "__main__":
    main()
