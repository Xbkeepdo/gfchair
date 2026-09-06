#!/usr/bin/env python3
"""Plot WRITE-to-FFN or target-to-FFN OT/JS trajectories by REAL/HALL label."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.tc_fvpa_artifacts import atomic_json_save
from scripts.analyze_ffn_visual_source_signal_ablation import (
    union_topk_js_matrices,
    write_label_curve,
)
from scripts.analyze_ffn_visual_source_study import _detector_matrices
from scripts.run_ffn_visual_source_attribution import result_root

MODELS = {
    "qwen2_5_vl_7b": "Qwen2.5-VL-7B",
    "llava_1_5_7b": "LLaVA-1.5-7B",
    "qwen3_vl_8b": "Qwen3-VL-8B",
    "internvl_2_5_8b": "InternVL2.5-8B",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance", choices=("ot", "js", "js_union_topk"), default="ot")
    parser.add_argument("--pair", choices=("wf", "ef"), default="wf",
                        help="wf: P_WRITE/P_FFN; ef: T/P_FFN (T=ATTENTION_EVIDENCE)")
    args = parser.parse_args()
    kind, pair = args.distance, args.pair
    family = "OT" if kind == "ot" else "JS"
    group = "D" if pair == "wf" else "E"
    slug = f"d_{kind}" if pair == "wf" else f"d_ef_{kind}"
    pair_name = f"D_{pair.upper()}"
    arguments = "P_WRITE, P_FFN" if pair == "wf" else "P_FFN, T"
    plot_title = "WRITE vs FFN" if pair == "wf" else "FFN vs target T"
    upper_bound = 1.0 if kind == "ot" else float(np.log(2))
    support = (
        "all visual tokens; natural-log JS divergence"
        if kind == "js" else "pairwise Top32 union, renormalized; " + (
            "sqrt_matched_state cost" if kind == "ot" else "natural-log JS divergence"
        )
    )
    plot_label = rf"$D_{{{pair.upper()}}}^{{{family}}}$"
    plot_support = "all visual tokens" if kind == "js" else "Top32 union"
    started = time.monotonic()
    torch.set_num_threads(1)
    results, curves = {}, {}
    for model in MODELS:
        model_started = time.monotonic()
        print(f"[{model}] Loading formal cohort", flush=True)
        data = _detector_matrices(model)
        matrices = {split: data[f"X_{split}"] for split in ("train", "test")}
        if kind == "js_union_topk":
            print(f"[{model}] Computing Union-Top32 JS from saved distributions", flush=True)
            matrices, alignment = union_topk_js_matrices(model, data)
            assert alignment["unique_targets"] == data["counts"]["unique_targets"]
        values = {}
        for split in ("train", "test"):
            ae, distance = np.split(matrices[split][f"{group}_{family}"], 2, axis=1)
            np.testing.assert_array_equal(ae, data[f"X_{split}"]["B"])
            assert distance.shape == ae.shape
            offset = (2 if pair == "wf" else 3) * ae.shape[1]
            np.testing.assert_array_equal(
                distance, matrices[split][f"G_{family}"][:, offset:offset + ae.shape[1]]
            )
            assert np.isfinite(distance).all()
            assert distance.min() >= -1e-7 and distance.max() <= upper_bound + 1e-6
            values[split] = distance
        curve = write_label_curve(
            model, data, values["train"], values["test"], result_root(model),
            slug=slug, ylabel=f"{plot_label} (median; IQR)", log_scale=False,
        )
        with (ROOT / curve["table"]).open(newline="", encoding="utf-8") as handle:
            curves[model] = list(csv.DictReader(handle))
        by_label = {
            label: [row for row in curves[model] if row["label"] == label]
            for label in ("REAL", "HALL")
        }
        gap = np.asarray([
            float(hall["median"]) - float(real["median"])
            for real, hall in zip(by_label["REAL"], by_label["HALL"])
        ])
        assert len(gap) == curve["layers"]
        result = {
            "model": model,
            "definition": (
                f"{pair_name}^{family} = {family}({arguments}); "
                f"detector {group}_{family} = AE + {pair_name}^{family}"
            ),
            "target_definition": "T = ATTENTION_EVIDENCE = normalize(attention * hpre_raw_logit_gauss_gate)",
            "support": support,
            "theoretical_range": [0.0, upper_bound],
            "cohort": curve["cohort"],
            "weighting": "one observation per formal mention, matching detector cohort",
            "bootstrap": "NOT_RUN_BY_USER_REQUEST",
            "counts": data["counts"],
            "curve": curve,
            "ranges": {
                split: {"min": float(x.min()), "max": float(x.max())}
                for split, x in values.items()
            },
            "hall_median_above_real_layers": int((gap > 0).sum()),
            "real_median_above_hall_layers": int((gap < 0).sum()),
            "mean_layer_median_hall_minus_real": float(gap.mean()),
            "mean_of_layer_medians": {
                label: float(np.mean([float(row["median"]) for row in rows]))
                for label, rows in by_label.items()
            },
            "hall_minus_real_median_by_layer": gap.tolist(),
            "elapsed_seconds": time.monotonic() - model_started,
        }
        results[model] = result
        atomic_json_save(result, result_root(model) / f"metrics/{slug}_curve_results.json")
        print(
            f"[{model}] HALL median > REAL: {(gap > 0).sum()}/{len(gap)}, "
            f"mean median gap={gap.mean():+.6f}; {result['elapsed_seconds']:.1f}s",
            flush=True,
        )
        del data, values, matrices

    prefix = ROOT / f"outputs/ffn_visual_source_{slug}_real_hall"
    for full_range in (False, True):
        fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharey=full_range)
        for ax, (model, title) in zip(axes.flat, MODELS.items()):
            for label, color in (("REAL", "#2166ac"), ("HALL", "#b2182b")):
                rows = [row for row in curves[model] if row["label"] == label]
                x = [int(row["layer"]) for row in rows]
                ax.plot(x, [float(row["median"]) for row in rows], color=color,
                        linewidth=1.8, label=f"{label} (n={rows[0]['n_mentions']})")
                ax.fill_between(x, [float(row["q25"]) for row in rows],
                                [float(row["q75"]) for row in rows], color=color, alpha=.14)
            if full_range:
                ax.set_ylim(0, upper_bound)
            else:
                ax.set_ylim(bottom=0)
            ax.set_title(title)
            ax.set_xlabel("Decoder layer")
            ax.set_ylabel(plot_label)
            ax.grid(alpha=.2)
            ax.legend(frameon=False, fontsize=9)
        fig.suptitle(f"{plot_title} {family}: REAL / HALL ({plot_support})", fontsize=15)
        fig.text(.5, .014,
                 "Lines: medians | Shading: 25th-75th percentiles, not confidence intervals | "
                 "All train + test mentions",
                 ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .04, 1, .96))
        suffix = ("_full_01" if kind == "ot" else "_full_ln2") if full_range else ""
        for extension in ("png", "pdf"):
            fig.savefig(f"{prefix}{suffix}.{extension}", dpi=180)
        plt.close(fig)
    atomic_json_save(
        {"models": results, "elapsed_seconds": time.monotonic() - started},
        Path(f"{prefix}.json"),
    )
    print(f"Completed: {prefix}.png ({time.monotonic() - started:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
