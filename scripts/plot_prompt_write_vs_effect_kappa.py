"""Plot prompt WRITE and FFN-effect within-source kappas on matching mentions."""
import csv
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.plot_prompt_kappa_811 import summarize


WRITE = ROOT / "outputs/prompt_write_kappa_811_v1"
EFFECT = ROOT / "outputs/selected_geometry_scalars_811_v1"
MODELS = (
    ("qwen2_5_vl_7b", "Qwen2.5-VL"),
    ("llava_1_5_7b", "LLaVA-1.5"),
    ("qwen3_vl_8b", "Qwen3-VL"),
    ("internvl_2_5_8b", "InternVL2.5"),
)


def read(path):
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def plot(rows, scope):
    fig, axes = plt.subplots(3, 4, figsize=(19, 11), sharex="col")
    for index, (model, title) in enumerate(MODELS):
        for source, row_index, math_title in (("a_m", 0, r"WRITE $\kappa_P^a$"),
                                               ("e_m", 1, r"Effect $\kappa_P^e$")):
            ax = axes[row_index, index]
            selected = [row for row in rows if row["model"] == model and row["scope"] == scope and row["source"] == source]
            by_label = {}
            for label, color in (("REAL", "#2563eb"), ("HALL", "#dc2626")):
                subset = sorted((row for row in selected if row["label"] == label), key=lambda row: row["layer"])
                layers = np.array([row["layer"] for row in subset])
                means = np.array([row["mean"] for row in subset])
                ax.plot(layers, means, lw=2, color=color, label=f"{label} n={subset[0]['n_mentions']:,}")
                ax.fill_between(layers, [row["q25"] for row in subset],
                                [row["q75"] for row in subset], color=color, alpha=.1)
                by_label[label] = means
            if row_index == 0:
                ax.set_title(title)
                ax.legend(fontsize=7.5, loc="lower right")
            if index == 0:
                ax.set_ylabel(math_title+" (mean; IQR)")
            ax.set_ylim(.25, 1.02)
            ax.grid(alpha=.2)
            delta = by_label["HALL"]-by_label["REAL"]
            ax = axes[2, index]
            ax.plot(layers, delta, lw=2, label=math_title)
            ax.grid(alpha=.2)
            ax.axhline(0, color="#4b5563", lw=.7)
            if row_index == 1:
                ax.set_xlabel("Decoder layer")
                ax.legend(fontsize=8)
        if index == 0:
            axes[2, index].set_ylabel("HALL−REAL mean κ")
    fig.suptitle(f"Prompt within-source kappa: attention WRITE a_m versus FFN response e_m | {scope.upper()} | mention-weighted")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(WRITE / f"kappa_curves_{scope}.{extension}", dpi=160)
    plt.close(fig)


def main():
    rows, summary = [], []
    for model, _ in MODELS:
        write = read(WRITE / model / "features.pt")
        effect = read(EFFECT / model / "features.pt")
        if write["mentions"] != effect["mentions"]:
            raise ValueError(f"{model} mention order differs")
        np.testing.assert_array_equal(write["y"], effect["y"])
        for part in write["masks"]:
            np.testing.assert_array_equal(write["masks"][part], effect["masks"][part])
        masks = dict(all=np.ones(len(write["y"]), dtype=bool), test=np.asarray(write["masks"]["test"], dtype=bool))
        for scope, mask in masks.items():
            for source, matrix in (("a_m", write["groups"]["write_kappa"]),
                                   ("e_m", effect["groups"]["prompt_within_source_kappa"])):
                selected = summarize(np.asarray(matrix), np.asarray(write["y"]), mask, model, scope)
                rows.extend(dict(row, source=source) for row in selected)
                real = [row for row in selected if row["label"] == "REAL"]
                hall = [row for row in selected if row["label"] == "HALL"]
                delta = np.asarray([h["mean"]-r["mean"] for h, r in zip(hall, real)])
                summary.append(dict(model=model, scope=scope, source=source,
                                    real_mentions=real[0]["n_mentions"], hall_mentions=hall[0]["n_mentions"],
                                    positive_layers=int((delta > 0).sum()), layers=len(delta),
                                    mean_layer_difference=float(delta.mean())))
    for filename, values in (("kappa_curves.csv", rows), ("kappa_curves_summary.csv", summary)):
        with (WRITE / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    for scope in ("all", "test"):
        plot(rows, scope)
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
