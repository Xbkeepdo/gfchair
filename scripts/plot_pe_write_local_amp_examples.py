#!/usr/bin/env python3
"""Plot ten examples of the unsummed local |P_E - P_WRITE| contribution."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import plot_sadt_style_write_effect_gallery as gallery

OUT = gallery.OUT / "pe_write_local_amp_examples"
MODELS = list(gallery.MODELS)


def peak_score(row: dict) -> float:
    relation = row["attention_pe"]
    return float(row.get(
        "divergence_peak_score",
        max(relation["js"])
        + 0.25 * (1 - min(relation["overlap16"]))
        + 0.15 * (1 - min(relation["overlap32"])),
    ))


def choose_ten() -> list[dict]:
    """One REAL and HALL per model, then the best unused REAL and HALL."""
    pools, chosen, used_images = [], [], set()
    for model in MODELS:
        rows = json.loads((gallery.OUT / model / "candidates.json").read_text())
        for row in rows:
            row["selection_score"] = peak_score(row)
        pools.extend(rows)
        for label in (0, 1):
            candidates = sorted(
                (row for row in rows if row["label"] == label),
                key=lambda row: row["selection_score"], reverse=True,
            )
            selected = next(row for row in candidates if row["image_id"] not in used_images)
            chosen.append(selected)
            used_images.add(selected["image_id"])
    for label in (0, 1):
        candidates = sorted(
            (row for row in pools if row["label"] == label),
            key=lambda row: row["selection_score"], reverse=True,
        )
        selected = next(row for row in candidates if row["image_id"] not in used_images)
        chosen.append(selected)
        used_images.add(selected["image_id"])
    return chosen


def load_case(row: dict) -> dict:
    old, matrices, _ = gallery.load_arrays(
        row["model"], row["image_id"], row["target_key"], row["layers"]
    )
    # Deliberately omit 1/2. Summing this map therefore gives 2*TV(P_E, P_WRITE).
    local_amp = np.abs(matrices["pe"] - matrices["attention"]).astype(np.float32)
    return dict(row=row, old=old, local_amp=local_amp)


def overlay(image: Image.Image, values: np.ndarray, grid: tuple[int, int], vmax: float):
    scaled = np.clip(values.reshape(grid) / vmax, 0.0, 1.0).astype(np.float32)
    heat = Image.fromarray(scaled).resize(image.size, Image.Resampling.BILINEAR)
    color = plt.get_cmap("jet")(np.asarray(heat, dtype=np.float32))[..., :3]
    base = np.asarray(image, dtype=np.float32) / 255.0
    return np.clip(0.45 * color + 0.55 * base, 0.0, 1.0)


def plot_case(case: dict, index: int, vmax: float) -> dict:
    row, old, local_amp = case["row"], case["old"], case["local_amp"]
    model, image_id, layers = row["model"], int(row["image_id"]), row["layers"]
    source = Image.open(gallery.COCO / "val2014" / f"COCO_val2014_{image_id:012d}.jpg")
    image, _ = gallery.model_view(source, model)
    source.close()
    label_name = "HALL" if row["label"] == 0 else "REAL"
    card = gallery.response_card(
        row["generated_text"], row["surface"], row["occurrence_index"], label_name
    )
    columns = len(layers)
    image_columns = 2
    header_ratio = max(2.7, min(7.5, 2.5 * (columns - image_columns) * card.height / card.width))
    fig = plt.figure(figsize=(2.7 * columns, header_ratio + 4.2), constrained_layout=True)
    grid_spec = fig.add_gridspec(2, columns, height_ratios=[header_ratio, 3.2])
    axis = fig.add_subplot(grid_spec[0, :image_columns])
    axis.imshow(image)
    axis.set_axis_off()
    axis.set_anchor("N")
    axis.set_title("Original image (model input view)", fontsize=10.5, fontweight="bold")
    axis = fig.add_subplot(grid_spec[0, image_columns:])
    axis.imshow(card)
    axis.set_axis_off()
    visual_grid = tuple(int(x) for x in old["visual_grid"])
    layer_rows = []
    for column, (layer, values) in enumerate(zip(layers, local_amp)):
        axis = fig.add_subplot(grid_spec[1, column])
        axis.imshow(overlay(image, values, visual_grid, vmax), interpolation="nearest")
        total, maximum = float(values.sum()), float(values.max())
        axis.set_title(f"Layer {layer}", fontsize=10.5, fontweight="bold")
        axis.text(
            0.018, 0.025, f"sum={total:.3f}  max={maximum:.4f}",
            transform=axis.transAxes, color="white", fontsize=7.2,
            bbox={"boxstyle": "square,pad=.15", "facecolor": "black", "edgecolor": "none", "alpha": .62},
        )
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color("#333333")
            spine.set_linewidth(.55)
        layer_rows.append(dict(layer=int(layer), local_amp_sum=total, local_amp_max=maximum))
    colorbar = fig.colorbar(
        ScalarMappable(norm=Normalize(0, vmax), cmap="jet"),
        ax=fig.axes, location="bottom", shrink=.48, pad=.018, aspect=45,
    )
    colorbar.set_label(
        rf"local difference $d_m=|P_{{E,m}}-P_{{W,m}}|$ (common scale; clipped at {vmax:.4f})",
        fontsize=9,
    )
    fig.suptitle(
        f"{gallery.MODEL_NAMES[model]} | {label_name} target '{row['surface']}' "
        f"({row['canonical_object']}) | COCO {image_id}\n"
        r"Unsummed local PE-WRITE difference: $d_m=|P_{E,m}-P_{W,m}|$;  "
        r"$\sum_m d_m=2\,TV(P_E,P_W)$",
        fontsize=12.5, fontweight="bold",
    )
    slug = gallery.safe_slug(row["canonical_object"])
    filename = f"{index:02d}_{model}_{label_name.lower()}_{image_id}_{slug}.jpg"
    path = OUT / filename
    fig.savefig(
        path, dpi=155, bbox_inches="tight", facecolor="white",
        pil_kwargs={"quality": 94, "subsampling": 0, "optimize": True},
    )
    plt.close(fig)
    return dict(
        index=index, model=model, model_name=gallery.MODEL_NAMES[model],
        label=row["label"], label_name=label_name, image_id=image_id,
        target_key=row["target_key"], surface=row["surface"],
        canonical_object=row["canonical_object"], layers=layers,
        selection_score=row["selection_score"], display_vmax=vmax,
        jpg=filename, layer_statistics=layer_rows,
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cases = [load_case(row) for row in choose_ten()]
    all_values = np.concatenate([case["local_amp"].reshape(-1) for case in cases])
    # A fixed robust scale keeps all ten figures comparable without letting one token
    # make the other maps visually blank.
    vmax = float(np.quantile(all_values, .995))
    manifest = [plot_case(case, index, vmax) for index, case in enumerate(cases, 1)]
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    lines = [
        "# 未求和的 PE–WRITE local amp：10个案例", "",
        r"每个视觉token绘制 $d_m=|P_{E,m}-P_{W,m}|$，不乘 $1/2$、不求和、不做TopK。",
        r"图中 `sum` 是该层 $\sum_m d_m=2TV(P_E,P_W)$，`max` 是最大单token差值。",
        f"10图统一使用候选局部差值的99.5%分位数 `{vmax:.6f}` 作为显示上限；颜色可跨图比较，超过上限的极少数位置仅在显示时饱和，保存的统计不截断。",
        "选择为每模型各一个高差异HALL和REAL，再补全局最高的未重复HALL和REAL，共HALL/REAL各5。", "",
        "| # | 模型 | 标签 | 目标 | 图片 | 最大层sum | 最大单token差 |", "|---:|---|---|---|---|---:|---:|",
    ]
    for row in manifest:
        max_sum = max(x["local_amp_sum"] for x in row["layer_statistics"])
        max_token = max(x["local_amp_max"] for x in row["layer_statistics"])
        lines.append(
            f"| {row['index']} | {row['model_name']} | {row['label_name']} | "
            f"{row['surface']} ({row['canonical_object']}) | [{row['image_id']}]({row['jpg']}) | "
            f"{max_sum:.3f} | {max_token:.4f} |"
        )
    lines += ["", "运行：", "", "```bash", "/opt/conda/private/envs/vicr/bin/python scripts/plot_pe_write_local_amp_examples.py", "```", "",
              "只读取已有K50 attention缓存和all-attention K32路径结果，不运行模型或路径积分。"]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
