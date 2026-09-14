"""Plot model-specific curve-segment detection results."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/ours_curve_layer_ranges_batch128_811_v1/summary.json"
SVAR = ROOT / "outputs/svar_llava_5_18_proportional_811_v1/summary.json"
OUT = ROOT / "outputs/ours_curve_layer_ranges_batch128_811_v1"
MODELS = ("qwen2_5_vl_7b", "llava_1_5_7b", "qwen3_vl_8b", "internvl_2_5_8b")
TITLES = ("Qwen2.5-VL-7B", "LLaVA-1.5-7B", "Qwen3-VL-8B", "InternVL2.5-8B")
GROUPS = (("visual", "V", "#2678b2"), ("vp_generation", "VP+G", "#d95f02"))


def main():
    rows = json.loads(SOURCE.read_text())["summaries"]
    baselines = {row["model"]: row for row in json.loads(SVAR.read_text())["summaries"]}
    figure, axes = plt.subplots(4, 2, figsize=(13, 16), constrained_layout=True)
    for model_index, (model, title) in enumerate(zip(MODELS, TITLES)):
        model_rows = [row for row in rows if row["model"] == model]
        order = ["segment_1", "segment_2", "segment_3", "segment_4", "all"]
        template = {row["range"]: row for row in model_rows if row["group"] == "visual"}
        labels = [
            "All" if name == "all" else f"{template[name]['layer_start']}-{template[name]['layer_end_exclusive']-1}"
            for name in order
        ]
        x = np.arange(len(order))
        for metric_index, (metric, std, baseline_key, ylabel) in enumerate((
            ("AUROC_mean", "AUROC_std", "AUROC_mean", "AUROC"),
            ("HALL_AUPR_mean", "HALL_AUPR_std", "HALL_AUPR_mean", "HALL-AUPR"),
        )):
            axis = axes[model_index, metric_index]
            for group, label, color in GROUPS:
                lookup = {row["range"]: row for row in model_rows if row["group"] == group}
                values = [lookup[name][metric] for name in order]
                errors = [lookup[name][std] for name in order]
                axis.errorbar(x, values, yerr=errors, marker="o", linewidth=2, capsize=3, label=label, color=color)
            axis.axhline(
                baselines[model][baseline_key], color="0.35", linestyle="--", linewidth=1.3,
                label="SVAR" if metric_index == 0 else None,
            )
            axis.set_xticks(x, labels)
            axis.set_ylabel(ylabel)
            axis.grid(axis="y", alpha=0.25)
            axis.set_title(f"{title}: {ylabel}")
            if metric_index == 0:
                axis.legend(fontsize=8)
    figure.savefig(OUT / "layer_range_detection.png", dpi=180)
    figure.savefig(OUT / "layer_range_detection.pdf")
    plt.close(figure)


if __name__ == "__main__":
    main()
