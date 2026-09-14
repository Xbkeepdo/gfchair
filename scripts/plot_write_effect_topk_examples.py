#!/usr/bin/env python3
"""Ten cached spatial examples for attention, AE, all-path P_E and cos(a_m,e_m)."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.plot_spatial_attention_gate_source_heatmaps import (
    load_annotations, model_view, projected_boxes, thermal_overlay,
)

OUT = ROOT/'outputs/ffn_write_effect_topk_examples_v1'
COCO = Path('/home/apulis-dev/userdata/DGST/token-grounding-detector/data/coco')
MODEL_NAMES = {
    'qwen2_5_vl_7b': 'Qwen2.5-VL-7B',
    'llava_1_5_7b': 'LLaVA-1.5-7B',
    'qwen3_vl_8b': 'Qwen3-VL-8B',
    'internvl_2_5_8b': 'InternVL2.5-8B',
}

# Ordered deliberately: eight small-object words, then two larger-object words;
# five HALL and five REAL, with ten different COCO images.
CASES = (
    ('small', 'qwen2_5_vl_7b', 7355, '7355:25', 0, 'cell phone'),
    ('small', 'qwen2_5_vl_7b', 1852, '1852:17', 1, 'scissors'),
    ('small', 'llava_1_5_7b', 13220, '13220:62', 0, 'knife'),
    ('small', 'llava_1_5_7b', 283, '283:9', 1, 'bottle'),
    ('small', 'qwen3_vl_8b', 17313, '17313:17', 0, 'wine glass'),
    ('small', 'qwen3_vl_8b', 2149, '2149:85', 1, 'apple'),
    ('small', 'internvl_2_5_8b', 8418, '8418:164', 0, 'bottle'),
    ('small', 'internvl_2_5_8b', 3580, '3580:35', 1, 'remote'),
    ('large', 'qwen2_5_vl_7b', 15978, '15978:34', 0, 'chair'),
    ('large', 'qwen3_vl_8b', 1818, '1818:18', 1, 'zebra'),
)
COLORS = ('#2878b5', '#8f5aae', '#16a085')
SELECTORS = ('Attention', 'AE / T', 'Joint')


def read(path: Path):
    return torch.load(path, map_location='cpu', weights_only=False)


def normalize_rows(x: torch.Tensor) -> torch.Tensor:
    x = torch.nan_to_num(x.float(), nan=0., posinf=0., neginf=0.).clamp_min(0.)
    return x/x.sum(-1, keepdim=True).clamp_min(1e-12)


def load_case(model: str, image_id: int, target_key: str, label: int):
    all_path = read(ROOT/'outputs/ffn_all_source_paths_v1'/model/'shards'/f'image_{image_id:012d}.pt')
    compact = read(ROOT/'outputs/ffn_source_composition_v1'/model/'shards/k50'/f'image_{image_id:012d}.pt')
    row = next(r for r in all_path['positions'] if r['target_key'] == target_key)
    old = next(r for r in compact['positions'] if r['target_key'] == target_key)
    mention = next(m for m in all_path['sample_table']
                   if m['target_key'] == target_key and int(m['label']) == label)
    attention = normalize_rows(old['raw_attention_mean'])
    evidence = normalize_rows(old['attention_evidence'])
    effects, cosines = [], []
    expected_positions = torch.arange(*old['visual_range'])
    for token in row['tokens']:
        visual = token['source_type'] == 1
        if not torch.equal(token['source_position'][visual], expected_positions):
            raise ValueError(f'Visual ordering mismatch for {model} {target_key}')
        effects.append(token['effect_norm'][visual])
        cosines.append(token['direction_cosine'][visual])
    effect = torch.stack(effects)
    cosine = torch.stack(cosines)
    if effect.shape != attention.shape or cosine.shape != attention.shape:
        raise ValueError(f'Shape mismatch for {model} {target_key}')
    pe = normalize_rows(effect)
    joint = normalize_rows(pe*evidence)
    return mention, old, attention.numpy(), evidence.numpy(), pe.numpy(), joint.numpy(), cosine.numpy()


def topk(values: np.ndarray, k: int) -> np.ndarray:
    k = min(k, values.size)
    return np.argsort(-values, kind='stable')[:k]


def add_boxes(axis, boxes):
    for x, y, width, height in boxes:
        axis.add_patch(Rectangle((x, y), width, height, fill=False,
                                 edgecolor='#ff3030', linewidth=1.25))


def show_overlay(axis, image: Image.Image, values: np.ndarray, grid, title: str, boxes, indices=None):
    selected = np.asarray(values, dtype=np.float32).copy()
    if indices is not None:
        mask = np.zeros(selected.size, dtype=bool); mask[indices] = True
        selected[~mask] = 0.
    patch = selected.reshape(grid)
    overlay, _ = thermal_overlay(image, patch, float(patch.max(initial=0.)))
    axis.imshow(overlay, interpolation='nearest')
    add_boxes(axis, boxes)
    if indices is not None:
        height, width = grid
        cell_w, cell_h = image.width/width, image.height/height
        for index in indices:
            row, col = divmod(int(index), width)
            axis.add_patch(Rectangle((col*cell_w, row*cell_h), cell_w, cell_h,
                                     fill=False, edgecolor='white', linewidth=.35, alpha=.75))
    axis.set_title(title, fontsize=9.2, fontweight='semibold')
    axis.set_xticks([]); axis.set_yticks([])


def show_cosine(axis, image: Image.Image, cosine: np.ndarray, grid, boxes):
    values = np.nan_to_num(cosine, nan=0.).reshape(grid).clip(-1., 1.)
    heat = Image.fromarray(values.astype(np.float32)).resize(image.size, Image.Resampling.BICUBIC)
    heat = np.asarray(heat, dtype=np.float32).clip(-1., 1.)
    base = np.asarray(image, dtype=np.float32)/255.
    color = plt.get_cmap('coolwarm')((heat+1.)/2.)[..., :3]
    alpha = (.24+.55*np.abs(heat))[..., None]
    axis.imshow(np.clip(base*(1.-alpha)+color*alpha, 0., 1.), interpolation='nearest')
    add_boxes(axis, boxes)
    axis.set_title(r'Mean cosine $\cos(a_m,e_m)$', fontsize=9.2, fontweight='semibold')
    axis.set_xticks([]); axis.set_yticks([])


def violin(axis, arrays, title, ylabel, *, zero=False):
    arrays = [np.asarray(x)[np.isfinite(x)] for x in arrays]
    parts = axis.violinplot(arrays, positions=np.arange(3), showmeans=False,
                            showmedians=True, showextrema=True, widths=.78)
    for body, color in zip(parts['bodies'], COLORS):
        body.set_facecolor(color); body.set_edgecolor('#202020'); body.set_alpha(.78)
    for key in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        parts[key].set_color('#252525'); parts[key].set_linewidth(.8)
    axis.set_xticks(range(3), SELECTORS, fontsize=8)
    axis.set_title(title, fontsize=9.2, fontweight='semibold')
    axis.set_ylabel(ylabel, fontsize=8)
    axis.grid(axis='y', alpha=.2)
    if zero: axis.axhline(0, color='#444444', linewidth=.8, linestyle='--')


def plot_case(number, size_class, model, image_id, target_key, label, expected_word,
              category_ids, annotations_by_image, dpi):
    mention, old, attention, evidence, pe, joint, cosine = load_case(model, image_id, target_key, label)
    canonical = str(mention['canonical_object'])
    surface = str(mention['word'])
    if canonical != expected_word:
        raise ValueError(f'Unexpected target: {canonical} != {expected_word}')
    original = Image.open(COCO/'val2014'/f'COCO_val2014_{image_id:012d}.jpg')
    image, crop_offset = model_view(original, model); original.close()
    grid = tuple(int(x) for x in old['visual_grid'])
    if grid[0]*grid[1] != attention.shape[1]:
        raise ValueError(f'Invalid grid {grid} for {attention.shape[1]} tokens')
    boxes = projected_boxes(image_id=image_id, canonical=canonical, category_ids=category_ids,
                            annotations_by_image=annotations_by_image,
                            crop_offset=crop_offset, view_size=image.size)

    means = tuple(x.mean(0) for x in (attention, evidence, pe, joint))
    cos_mean = np.nanmean(cosine, axis=0)
    fig = plt.figure(figsize=(23, 11.2), constrained_layout=True)
    gs = fig.add_gridspec(3, 6, height_ratios=(1., 1., 1.))
    top_axes = [fig.add_subplot(gs[0, i]) for i in range(6)]
    top_axes[0].imshow(image); add_boxes(top_axes[0], boxes)
    top_axes[0].set_title('Model input image', fontsize=9.2, fontweight='semibold')
    top_axes[0].set_xticks([]); top_axes[0].set_yticks([])
    for axis, values, title in zip(top_axes[1:5], means,
            ('Attention (layer mean)', 'AE / T (layer mean)',
             r'$P_E^{all}$ from $\|e_m\|$', r'Joint $\propto T\,P_E^{all}$')):
        show_overlay(axis, image, values, grid, title, boxes)
    show_cosine(top_axes[5], image, cos_mean, grid, boxes)

    statistic_rows = []
    selections = (means[0], means[1], means[3])
    for plot_row, k in enumerate((16, 32), start=1):
        indices = [topk(values, k) for values in selections]
        for column, (axis_name, values, chosen) in enumerate(zip(SELECTORS, selections, indices)):
            pe_mass = float(means[2][chosen].sum())
            cos_value = float(np.nanmean(cosine[:, chosen]))
            selector_mass = float(values[chosen].sum())
            axis = fig.add_subplot(gs[plot_row, column])
            show_overlay(axis, image, values, grid, f'{axis_name} Top-{k}', boxes, chosen)
            axis.text(.02, .025, f'selector mass: {selector_mass:.3f}\n'
                      f'P_E mass: {pe_mass:.3f}\nmean cos: {cos_value:+.3f}',
                      transform=axis.transAxes, ha='left', va='bottom', color='white', fontsize=7.2,
                      bbox={'boxstyle': 'square,pad=.2', 'facecolor': 'black',
                            'edgecolor': 'none', 'alpha': .62})
            selected_cos = cosine[:, chosen].reshape(-1)
            selected_pe = pe[:, chosen].reshape(-1)
            finite_cos = selected_cos[np.isfinite(selected_cos)]
            statistic_rows.append(dict(case=number, size_priority=size_class, model=model,
                label=label, label_name='HALL' if label == 0 else 'REAL', image_id=image_id,
                target_key=target_key, surface=surface, canonical_object=canonical,
                layers=attention.shape[0], visual_tokens=attention.shape[1], k=k, selector=axis_name,
                selector_mass=selector_mass, pe_mass=pe_mass,
                cosine_mean=float(np.mean(finite_cos)), cosine_median=float(np.median(finite_cos)),
                cosine_q25=float(np.quantile(finite_cos, .25)), cosine_q75=float(np.quantile(finite_cos, .75)),
                pe_mean=float(np.mean(selected_pe)), pe_median=float(np.median(selected_pe)),
                pe_q25=float(np.quantile(selected_pe, .25)), pe_q75=float(np.quantile(selected_pe, .75))))
        cos_axis = fig.add_subplot(gs[plot_row, 3:5])
        violin(cos_axis, [cosine[:, chosen].reshape(-1) for chosen in indices],
               f'Top-{k}: cosine distribution over layers × selected patches',
               r'$\cos(a_m,e_m)$', zero=True)
        cos_axis.set_ylim(-1.03, 1.03)
        pe_axis = fig.add_subplot(gs[plot_row, 5])
        violin(pe_axis, [np.log10(pe[:, chosen].reshape(-1).clip(1e-9)) for chosen in indices],
               f'Top-{k}: $P_E$ distribution', r'$\log_{10} P_E$')

    label_name = 'HALL' if label == 0 else 'REAL'
    fig.suptitle(f"{number:02d} | {size_class}-object priority | {MODEL_NAMES[model]} | "
                 f"target '{surface}' ({canonical}) | {label_name} | COCO {image_id}\n"
                 r"All decoder layers are equally averaged; $P_E^{all}$ and cosine use the same all-attention K32 $e_m$.",
                 fontsize=13, fontweight='bold')
    intensity = ScalarMappable(norm=Normalize(0., 1.), cmap='turbo')
    colorbar = fig.colorbar(intensity, ax=top_axes[1:5], location='right', shrink=.75, pad=.012)
    colorbar.set_label('Relative intensity within each panel', fontsize=8)
    cosine_bar = ScalarMappable(norm=Normalize(-1., 1.), cmap='coolwarm')
    cbar = fig.colorbar(cosine_bar, ax=top_axes[5], location='right', shrink=.75, pad=.012)
    cbar.set_label(r'$\cos(a_m,e_m)$', fontsize=8)
    slug = f'{number:02d}_{size_class}_{label_name.lower()}_{model}_{image_id}_{canonical.replace(" ","-")}'
    destination = OUT/(slug+'.png')
    fig.savefig(destination, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    manifest = dict(case=number, size_priority=size_class, model=model, model_name=MODEL_NAMES[model],
        label=label, label_name=label_name, image_id=image_id, target_key=target_key,
        response_index=int(mention['response_index']), surface=surface, canonical_object=canonical,
        layers=attention.shape[0], visual_tokens=attention.shape[1], visual_grid=list(grid),
        coco_boxes=len(boxes), image=str(destination.relative_to(ROOT)))
    return manifest, statistic_rows


def write_outputs(manifest, statistics):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    with (OUT/'region_statistics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(statistics[0]))
        writer.writeheader(); writer.writerows(statistics)
    lines = ['# Attention、AE、P_E 与 write–effect cosine：10个空间案例', '',
        '共10张图：前8张为小物体词，后2张为较大物体词；5个HALL、5个REAL，图片ID不重复。',
        '每张图使用全部decoder层等权平均。Attention先在视觉范围内逐层归一化；AE图为既有证据分布T；'
        'P_E^all按同一all-attention K32路径的||e_m||逐层归一化；联合分布逐层按T*P_E^all归一化。',
        'Top16/32从各自层均值图选择。Top-K标题同时报告选择器自身质量、该区域承载的P_E质量和区域内全部层×patch的平均cos(a_m,e_m)。',
        '右侧violin展示Top-K区域内全部层×patch的cos和log10(P_E)分布。热图沿用SVAR空间图的turbo冷色低值背景与热色高值叠加；cos采用零中心coolwarm。红框为可用的目标COCO标注框。',
        '这里的P_E^all与此前visual-only conditional P_E不是同一路径；本图为了让P_E和cos严格来自同一e_m而采用all-attention版本。联合分布是可视化约定，不是新的归因闭合项。', '',
        '| # | 优先级 | 标签 | 模型 | 目标词 | COCO类别 | 图片 | 网格 | 图 |',
        '|---:|---|---|---|---|---|---:|---:|---|']
    for row in manifest:
        lines.append(f"| {row['case']} | {row['size_priority']} | {row['label_name']} | {row['model_name']} | "
                     f"{row['surface']} | {row['canonical_object']} | {row['image_id']} | "
                     f"{row['visual_grid'][0]}×{row['visual_grid'][1]} | "
                     f"[PNG]({Path(row['image']).name}) |")
    lines += ['', '逐区域数值见 `region_statistics.csv`；身份、网格和输出路径见 `manifest.json`。',
              '运行：`python scripts/plot_write_effect_topk_examples.py`。只读取缓存和COCO图像，不运行模型或积分。']
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dpi', type=int, default=180)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    category_ids, annotations = load_annotations(COCO/'annotations/instances_val2014.json')
    manifest, statistics = [], []
    for number, case in enumerate(CASES, 1):
        row, values = plot_case(number, *case, category_ids, annotations, args.dpi)
        manifest.append(row); statistics.extend(values)
        print('PLOT', number, row['label_name'], row['canonical_object'], row['image'], flush=True)
    write_outputs(manifest, statistics)


if __name__ == '__main__':
    main()
