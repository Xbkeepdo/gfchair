#!/usr/bin/env python3
"""Select and plot paper-style per-layer attention/AE/P_E/cosine examples."""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics
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
    model_view, projected_boxes,
)

MODELS = ('qwen2_5_vl_7b', 'llava_1_5_7b', 'qwen3_vl_8b', 'internvl_2_5_8b')
MODEL_NAMES = dict(qwen2_5_vl_7b='Qwen2.5-VL-7B', llava_1_5_7b='LLaVA-1.5-7B',
                   qwen3_vl_8b='Qwen3-VL-8B', internvl_2_5_8b='InternVL2.5-8B')
LAYER_COUNTS = dict(zip(MODELS, (28, 32, 36, 32)))
OUT = ROOT/'outputs/ffn_write_effect_multilayer_examples_v1'
COCO = Path('/home/apulis-dev/userdata/DGST/token-grounding-detector/data/coco')
COHORT = ROOT/'outputs/ffn_output_cosine_20260909/cohort500.json'
SMALL_MAX_FRACTION = .03
MIN_REAL_FRACTION = .001
SIGNALS = (('attention', 'Attention'), ('evidence', 'AE / T'),
           ('pe', r'$P_E^{all}$'), ('joint', r'Joint $\propto T\,P_E^{all}$'))


def read(path: Path):
    return torch.load(path, map_location='cpu', weights_only=False)


def selected_layers(model: str) -> list[int]:
    count = LAYER_COUNTS[model]
    return list(dict.fromkeys(layer for layer in (10, 15, 20, 25, 30, count) if layer <= count))


def normalize_rows(x) -> np.ndarray:
    x = np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0., posinf=0., neginf=0.)
    x = np.maximum(x, 0.)
    total = x.sum(-1, keepdims=True)
    return np.divide(x, np.maximum(total, 1e-12), out=np.zeros_like(x), where=total > 0)


def top_mass(x: np.ndarray, k=16) -> float:
    k = min(k, x.size)
    return float(np.partition(x, -k)[-k:].sum())


def top_indices(x: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(-x, kind='stable')[:min(k, x.size)]


def coco_metadata():
    data = json.loads((COCO/'annotations/instances_val2014.json').read_text())
    category_ids = {str(x['name']).lower(): int(x['id']) for x in data['categories']}
    category_names = {int(x['id']): str(x['name']).lower() for x in data['categories']}
    image_sizes = {int(x['id']): (int(x['width']), int(x['height'])) for x in data['images']}
    by_image, fractions = defaultdict(list), defaultdict(list)
    for annotation in data['annotations']:
        image_id = int(annotation['image_id']); by_image[image_id].append(annotation)
        width, height = image_sizes[image_id]
        area = float(annotation.get('area', annotation['bbox'][2]*annotation['bbox'][3]))
        fractions[category_names[int(annotation['category_id'])]].append(area/max(width*height, 1))
    medians = {name: float(statistics.median(values)) for name, values in fractions.items()}
    return category_ids, by_image, image_sizes, medians


def actual_fraction(image_id, canonical, category_ids, annotations, image_sizes):
    category = category_ids.get(canonical)
    if category is None: return None
    width, height = image_sizes[image_id]
    values = [float(a.get('area', a['bbox'][2]*a['bbox'][3]))/max(width*height, 1)
              for a in annotations.get(image_id, []) if int(a['category_id']) == category]
    return min(values) if values else None


def effect_and_cosine(all_row: dict, old_row: dict, layers: list[int]):
    expected = torch.arange(*old_row['visual_range'])
    effects, cosines = [], []
    for layer in layers:
        token = all_row['tokens'][layer-1]; visual = token['source_type'] == 1
        if not torch.equal(token['source_position'][visual], expected):
            raise ValueError(f"Visual order mismatch: {all_row['target_key']} L{layer}")
        effects.append(token['effect_norm'][visual].numpy())
        cosines.append(token['direction_cosine'][visual].numpy())
    return normalize_rows(np.stack(effects)), np.stack(cosines).astype(np.float32)


def rank_model(model: str):
    layers = selected_layers(model)
    category_ids, annotations, image_sizes, medians = coco_metadata()
    matrix = read(ROOT/'outputs/ffn_source_composition_v1'/model/'matrices.pt')
    cohort = json.loads(COHORT.read_text())['split']; image_ids = set(cohort['train']+cohort['test'])
    labels = defaultdict(set)
    for mention in matrix['mentions']: labels[mention['target_key']].add(int(mention['label']))
    by_image = defaultdict(list)
    seen = set()
    for mention in matrix['mentions']:
        image_id = int(mention['image_id']); label = int(mention['label'])
        canonical = str(mention.get('canonical_object') or '').strip().lower()
        key = (mention['target_key'], label, canonical)
        if image_id not in image_ids or labels[mention['target_key']] != {label} or key in seen:
            continue
        seen.add(key)
        observed = actual_fraction(image_id, canonical, category_ids, annotations, image_sizes)
        if label == 0:
            if observed is not None or canonical not in medians: continue
            fraction, basis = medians[canonical], 'COCO category median'
        else:
            if observed is None or observed < MIN_REAL_FRACTION: continue
            fraction, basis = observed, 'smallest matching GT box'
        by_image[image_id].append(dict(target_key=mention['target_key'], label=label,
            response_index=int(mention['response_index']), surface=str(mention.get('word') or canonical),
            canonical_object=canonical, object_fraction=float(fraction), size_basis=basis,
            size_priority='small' if fraction <= SMALL_MAX_FRACTION else 'large'))

    candidates = []
    image_list = sorted(by_image)
    for image_number, image_id in enumerate(image_list, 1):
        old_path = ROOT/'outputs/ffn_source_composition_v1'/model/'shards/k50'/f'image_{image_id:012d}.pt'
        all_path = ROOT/'outputs/ffn_all_source_paths_v1'/model/'shards'/f'image_{image_id:012d}.pt'
        if not old_path.exists() or not all_path.exists(): continue
        old = read(old_path); all_source = read(all_path)
        old_rows = {x['target_key']: x for x in old['positions']}
        all_rows = {x['target_key']: x for x in all_source['positions']}
        for meta in by_image[image_id]:
            key = meta['target_key']
            if key not in old_rows or key not in all_rows: continue
            old_row = old_rows[key]
            attention = normalize_rows(old_row['raw_attention_mean'][[x-1 for x in layers]].numpy())
            evidence = normalize_rows(old_row['attention_evidence'][[x-1 for x in layers]].numpy())
            pe, cosine = effect_and_cosine(all_rows[key], old_row, layers)
            if attention.shape != evidence.shape or attention.shape != pe.shape:
                raise ValueError(f'Shape mismatch: {model} {key}')
            a16 = np.asarray([top_mass(x) for x in attention])
            t16 = np.asarray([top_mass(x) for x in evidence])
            e16 = np.asarray([top_mass(x) for x in pe])
            gate_delta, pe_delta = t16-a16, t16-e16
            candidates.append(dict(model=model, image_id=image_id, layers=layers,
                visual_tokens=int(attention.shape[1]), visual_grid=list(old_row['visual_grid']), **meta,
                attention_top16_mean=float(a16.mean()), ae_top16_mean=float(t16.mean()),
                pe_top16_mean=float(e16.mean()), gate_focus_score=float(gate_delta.mean()),
                gate_focus_positive_layers=int((gate_delta > 0).sum()),
                ae_pe_contrast_score=float(pe_delta.mean()),
                ae_pe_positive_layers=int((pe_delta > 0).sum()),
                cosine_mean=float(np.nanmean(cosine)), cosine_negative_fraction=float(np.nanmean(cosine < 0))))
        if image_number % 50 == 0:
            print('RANK', model, image_number, '/', len(image_list), 'images', len(candidates), 'targets', flush=True)
    folder = OUT/model; folder.mkdir(parents=True, exist_ok=True)
    (folder/'candidates.json').write_text(json.dumps(candidates, ensure_ascii=False, indent=2)+'\n')
    selection = select_candidates(model, candidates)
    (folder/'selection.json').write_text(json.dumps(selection, ensure_ascii=False, indent=2)+'\n')
    print('SELECT', model, [(x['label_name'], x['size_priority'], x['canonical_object'], x['pattern']) for x in selection], flush=True)


def take(candidates, metric, count, used_images, used_words, pattern):
    positive_layers = {'gate_focus_score': 'gate_focus_positive_layers',
                       'ae_pe_contrast_score': 'ae_pe_positive_layers'}[metric]
    ranked = sorted(candidates, key=lambda x:(x[metric], x[positive_layers]), reverse=True)
    selected = []
    for require_new_word in (True, False):
        for row in ranked:
            if len(selected) == count: break
            if row['image_id'] in used_images or row['image_id'] in {x['image_id'] for x in selected}: continue
            if require_new_word and row['canonical_object'] in used_words: continue
            if row in selected: continue
            value = dict(row, pattern=pattern, selection_metric=metric, selection_score=row[metric])
            selected.append(value); used_words.add(row['canonical_object'])
        if len(selected) == count: break
    used_images.update(x['image_id'] for x in selected)
    if len(selected) != count: raise RuntimeError(f'Only selected {len(selected)}/{count} for {pattern}')
    return selected


def select_candidates(model: str, candidates: list[dict]):
    selected_by_group = {}
    used_images, used_words = set(), set()
    for label in (0, 1):
        small = [x for x in candidates if x['label'] == label and x['size_priority'] == 'small']
        large = [x for x in candidates if x['label'] == label and x['size_priority'] == 'large']
        chosen = take(small, 'gate_focus_score', 2, used_images, used_words, 'AE more concentrated than attention')
        chosen += take(small, 'ae_pe_contrast_score', 2, used_images, used_words, 'AE concentrated while P_E dispersed')
        large_metric = 'gate_focus_score'
        if max((x['ae_pe_contrast_score'] for x in large), default=-math.inf) > max((x['gate_focus_score'] for x in large), default=-math.inf):
            large_metric = 'ae_pe_contrast_score'
        chosen += take(large, large_metric, 1, used_images, used_words,
                       'large object: '+('AE more concentrated than attention' if large_metric == 'gate_focus_score'
                                         else 'AE concentrated while P_E dispersed'))
        selected_by_group[label] = chosen
    ordered = selected_by_group[0][:4]+selected_by_group[1][:4]+selected_by_group[0][4:]+selected_by_group[1][4:]
    for number, row in enumerate(ordered, 1):
        row['case'] = number; row['label_name'] = 'HALL' if row['label'] == 0 else 'REAL'
    if len(ordered) != 10 or Counter(x['label'] for x in ordered) != Counter({0:5, 1:5}):
        raise ValueError(f'Invalid selection for {model}')
    return ordered


def load_plot_arrays(spec):
    model, image_id, key = spec['model'], spec['image_id'], spec['target_key']
    old = read(ROOT/'outputs/ffn_source_composition_v1'/model/'shards/k50'/f'image_{image_id:012d}.pt')
    all_source = read(ROOT/'outputs/ffn_all_source_paths_v1'/model/'shards'/f'image_{image_id:012d}.pt')
    old_row = next(x for x in old['positions'] if x['target_key'] == key)
    all_row = next(x for x in all_source['positions'] if x['target_key'] == key)
    indices = [x-1 for x in spec['layers']]
    attention = normalize_rows(old_row['raw_attention_mean'][indices].numpy())
    evidence = normalize_rows(old_row['attention_evidence'][indices].numpy())
    pe, cosine = effect_and_cosine(all_row, old_row, spec['layers'])
    joint = normalize_rows(evidence*pe)
    return old_row, dict(attention=attention, evidence=evidence, pe=pe, joint=joint), cosine


def add_boxes(axis, boxes):
    for x, y, width, height in boxes:
        axis.add_patch(Rectangle((x, y), width, height, fill=False,
                                 edgecolor='#ff2525', linewidth=1.15))


def cold_thermal_overlay(image, values, grid, scale_max):
    relative = np.clip(np.asarray(values, dtype=np.float32).reshape(grid)/max(scale_max, 1e-12), 0., 1.)
    heat = Image.fromarray(relative).resize(image.size, Image.Resampling.BICUBIC)
    heat = np.asarray(heat, dtype=np.float32).clip(0., 1.)
    base = np.asarray(image, dtype=np.float32)/255.
    cold = .42*base + .58*np.asarray([.015, .055, .25], dtype=np.float32)
    color = plt.get_cmap('turbo')(heat)[..., :3]
    alpha = (.92*np.power(heat, .62))[..., None]
    return np.clip(cold*(1.-alpha)+color*alpha, 0., 1.)


def cosine_overlay(image, values, grid):
    patch = np.nan_to_num(values, nan=0.).reshape(grid).clip(-1., 1.)
    heat = Image.fromarray(patch.astype(np.float32)).resize(image.size, Image.Resampling.BICUBIC)
    heat = np.asarray(heat, dtype=np.float32).clip(-1., 1.)
    base = np.asarray(image, dtype=np.float32)/255.
    cold = .48*base + .52*np.asarray([.015, .055, .25], dtype=np.float32)
    color = plt.get_cmap('coolwarm')((heat+1.)/2.)[..., :3]
    alpha = (.88*np.power(np.abs(heat), .62))[..., None]
    return np.clip(cold*(1.-alpha)+color*alpha, 0., 1.)


def plot_one(spec, category_ids, annotations, dpi):
    model, image_id = spec['model'], spec['image_id']; layers = spec['layers']
    old, matrices, cosine = load_plot_arrays(spec)
    original = Image.open(COCO/'val2014'/f'COCO_val2014_{image_id:012d}.jpg')
    image, crop_offset = model_view(original, model); original.close()
    grid = tuple(int(x) for x in old['visual_grid'])
    boxes = projected_boxes(image_id=image_id, canonical=spec['canonical_object'],
        category_ids=category_ids, annotations_by_image=annotations,
        crop_offset=crop_offset, view_size=image.size)
    row_scales = {name:max(float(matrices[name].max()), 1e-12) for name, _ in SIGNALS}
    fig, axes = plt.subplots(5, len(layers), figsize=(3.45*len(layers), 14.8), constrained_layout=True)
    statistic_rows = []
    for column, layer in enumerate(layers):
        for row, (name, title) in enumerate(SIGNALS):
            values = matrices[name][column]
            overlay = cold_thermal_overlay(image, values, grid, row_scales[name])
            axis = axes[row, column]; axis.imshow(overlay, interpolation='nearest')
            add_boxes(axis, boxes)
            if row == 0: axis.set_title(f'Layer {layer}', fontsize=11.5, fontweight='bold')
            if column == 0:
                axis.set_ylabel(title, fontsize=10.5, fontweight='semibold', labelpad=7)
            mass16, mass32 = top_mass(values, 16), top_mass(values, 32)
            axis.text(.02, .025, f'T16={mass16:.3f}  T32={mass32:.3f}', transform=axis.transAxes,
                      color='white', fontsize=7.2,
                      bbox={'boxstyle':'square,pad=.18','facecolor':'black','edgecolor':'none','alpha':.6})
            axis.set_xticks([]); axis.set_yticks([])
            statistic_rows.append(dict(model=model, case=spec['case'], label=spec['label'],
                label_name=spec['label_name'], image_id=image_id, target_key=spec['target_key'],
                canonical_object=spec['canonical_object'], surface=spec['surface'], layer=layer,
                signal=name, top16_mass=mass16, top32_mass=mass32))
        axis = axes[4, column]; values = cosine[column]
        axis.imshow(cosine_overlay(image, values, grid), interpolation='nearest')
        add_boxes(axis, boxes)
        finite = values[np.isfinite(values)]
        mean, median = float(finite.mean()), float(np.median(finite))
        neg, pos = float((finite < 0).mean()), float((finite > 0).mean())
        if column == 0:
            axis.set_ylabel(r'$\cos(a_m,e_m)$', fontsize=10.5,
                            fontweight='semibold', labelpad=7)
        axis.text(.02, .025, f'mean={mean:+.3f}  median={median:+.3f}\nneg={100*neg:.1f}%  pos={100*pos:.1f}%',
                  transform=axis.transAxes, color='white', fontsize=7.2,
                  bbox={'boxstyle':'square,pad=.18','facecolor':'black','edgecolor':'none','alpha':.6})
        axis.set_xticks([]); axis.set_yticks([])
        statistic_rows.append(dict(model=model, case=spec['case'], label=spec['label'],
            label_name=spec['label_name'], image_id=image_id, target_key=spec['target_key'],
            canonical_object=spec['canonical_object'], surface=spec['surface'], layer=layer,
            signal='cosine', cosine_mean=mean, cosine_median=median,
            cosine_negative_fraction=neg, cosine_positive_fraction=pos,
            joint_top16_cosine_mean=float(np.nanmean(values[top_indices(matrices['joint'][column],16)])),
            joint_top32_cosine_mean=float(np.nanmean(values[top_indices(matrices['joint'][column],32)]))))

    for row, (name, title) in enumerate(SIGNALS):
        colorbar = fig.colorbar(ScalarMappable(norm=Normalize(0., row_scales[name]), cmap='turbo'),
                               ax=axes[row, :], location='right', shrink=.82, pad=.008)
        colorbar.set_label(f'{title} mass', fontsize=8); colorbar.ax.tick_params(labelsize=7)
    cosine_bar = fig.colorbar(ScalarMappable(norm=Normalize(-1.,1.), cmap='coolwarm'),
                              ax=axes[4,:], location='right', shrink=.82, pad=.008)
    cosine_bar.set_label('negative ← cosine → positive', fontsize=8); cosine_bar.ax.tick_params(labelsize=7)
    fig.suptitle(f"{MODEL_NAMES[model]} | case {spec['case']:02d} | target '{spec['surface']}' "
                 f"({spec['canonical_object']}) | {spec['label_name']} | COCO {image_id}\n"
                 f"Selected for: {spec['pattern']} | object size proxy={100*spec['object_fraction']:.2f}% "
                 f"({spec['size_basis']})",
                 fontsize=13, fontweight='bold')
    slug = f"{spec['case']:02d}_{spec['size_priority']}_{spec['label_name'].lower()}_{image_id}_{spec['canonical_object'].replace(' ','-')}"
    folder = OUT/model; png, pdf = folder/(slug+'.png'), folder/(slug+'.pdf')
    fig.savefig(png, dpi=dpi, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return dict(spec, png=str(png.relative_to(ROOT)), pdf=str(pdf.relative_to(ROOT)), coco_boxes=len(boxes)), statistic_rows


def plot_model(model, dpi):
    category_ids, annotations, _, _ = coco_metadata()
    selection = json.loads((OUT/model/'selection.json').read_text())
    manifest, statistics = [], []
    for spec in selection:
        row, stats = plot_one(spec, category_ids, annotations, dpi)
        manifest.append(row); statistics.extend(stats)
        print('PLOT', model, spec['case'], spec['label_name'], spec['canonical_object'], flush=True)
    (OUT/model/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    with (OUT/model/'layer_statistics.csv').open('w',newline='') as handle:
        fields=[]
        for row in statistics:
            for key in row:
                if key not in fields: fields.append(key)
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(statistics)
    lines=[f'# {MODEL_NAMES[model]}：10个逐层空间案例','',
        '前8张为小物体，后2张为较大物体；HALL/REAL各5张。每张图按列展示指定层，未做跨层mean。', '',
        '| # | 大小 | 标签 | 目标词 | 选择模式 | 图片 | PNG | PDF |','|---:|---|---|---|---|---:|---|---|']
    for x in manifest:
        lines.append(f"| {x['case']} | {x['size_priority']} | {x['label_name']} | {x['surface']} "
                     f"({x['canonical_object']}) | {x['pattern']} | {x['image_id']} | "
                     f"[PNG]({Path(x['png']).name}) | [PDF]({Path(x['pdf']).name}) |")
    (OUT/model/'summary.md').write_text('\n'.join(lines)+'\n')


def summarize():
    manifests=[]; selections={}; model_cosine={}
    for model in MODELS:
        manifests.extend(json.loads((OUT/model/'manifest.json').read_text()))
        selections[model]=json.loads((OUT/model/'selection.json').read_text())
    cosine_rows=[]
    for model in MODELS:
        with (OUT/model/'layer_statistics.csv').open() as f:
            rows=[x for x in csv.DictReader(f) if x['signal']=='cosine']
        cosine_rows += rows
        mean=np.asarray([float(x['cosine_mean']) for x in rows])
        neg=np.asarray([float(x['cosine_negative_fraction']) for x in rows])
        pos=np.asarray([float(x['cosine_positive_fraction']) for x in rows])
        model_cosine[model]=(float(mean.mean()),float(np.median(neg)),int((pos>0).sum()),len(pos))
    negative=np.asarray([float(x['cosine_negative_fraction']) for x in cosine_rows])
    positive=np.asarray([float(x['cosine_positive_fraction']) for x in cosine_rows])
    lines=['# 四模型逐层 Attention、AE、P_E 与 cosine 论文候选图','',
        '每模型10张，共40张；每模型HALL/REAL各5张，前8张小物体、后2张较大物体。候选来自固定共享500图。',
        '层使用10/15/20/25/30/最后一层；Qwen2.5只有28层，因此使用10/15/20/25/28。所有图为逐层值，没有跨层mean。',
        '案例按AE相对Attention的Top16集中度增量，或AE相对P_E的Top16集中度增量排序选择；每类各保留两张小物体，并补一张大物体。选择分数、正向层数和大小依据保存在selection.json。',
        'Attention、AE/T、P_E及Joint均画完整分布：原图先覆盖统一深蓝冷色，再叠加连续热力图，不画patch框。每格左下角保留Top16/Top32质量；Joint=normalize(T*P_E)。红框为REAL目标的COCO标注。',
        'P_E使用all-attention K32的逐token||e_m||，与cos(a_m,e_m)严格同源；不是旧visual-only conditional P_E。',
        'Cosine用固定[-1,1]蓝白红图：蓝色负、冷色背景接近0、红色正。每格标注mean/median及正负patch比例。',
        f'40个案例×所示层中，单层负cos patch比例的中位数为{100*np.median(negative):.1f}%，范围{100*negative.min():.1f}%–{100*negative.max():.1f}%；'
        f'正cos比例中位数为{100*np.median(positive):.1f}%。因此cosine总体偏负，但并非全部为负。',
        '这些是按可视化对比强度挑选的论文候选案例，适合展示机制，不用于估计总体效应或检测性能。', '',
        '## Cosine正负概况', '',
        '| 模型 | 所示层平均cos | 单层负patch比例中位数 | 含正cos patch的层 |','|---|---:|---:|---:|']
    for model in MODELS:
        mean, neg, positive_layers, total_layers=model_cosine[model]
        lines.append(f'| {MODEL_NAMES[model]} | {mean:+.3f} | {100*neg:.1f}% | {positive_layers}/{total_layers} |')
    lines += ['', '## 首选查看的HALL/REAL案例', '',
        '下表优先小物体，再按各自选择指标的分数列出每个模型的首选案例；其余案例见各模型图册。', '',
        '| 模型 | HALL | REAL |','|---|---|---|']
    for model in MODELS:
        chosen=[]
        for label in ('HALL','REAL'):
            row=max((x for x in selections[model]
                     if x['label_name']==label and x['size_priority']=='small'),
                    key=lambda x:x['selection_score'])
            png=next(Path(x['png']).name for x in manifests
                     if x['model']==model and x['case']==row['case'])
            chosen.append(f"[case {row['case']:02d}: {row['surface']}]({model}/{png})")
        lines.append(f'| {MODEL_NAMES[model]} | {chosen[0]} | {chosen[1]} |')
    lines += ['', '## 全部图册', '',
        '| 模型 | 图数 | 层 | 图册 | 候选与分数 | 逐层数值 |','|---|---:|---|---|---|---|']
    for model in MODELS:
        layers='/'.join(map(str,selected_layers(model)))
        lines.append(f"| {MODEL_NAMES[model]} | 10 | {layers} | [{model}/summary.md]({model}/summary.md) | "
                     f"[{model}/selection.json]({model}/selection.json) | "
                     f"[{model}/layer_statistics.csv]({model}/layer_statistics.csv) |")
    lines += ['', '运行方式：', '',
        '```bash',
        'python scripts/plot_write_effect_multilayer_examples.py --stage rank --models <model>',
        'python scripts/plot_write_effect_multilayer_examples.py --stage plot --models <model>',
        'python scripts/plot_write_effect_multilayer_examples.py --stage summarize',
        '```', '', '只读取既有K32/K50紧凑缓存和COCO图片，不运行模型或积分。']
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=('rank','plot','summarize'),required=True)
    parser.add_argument('--models',nargs='+',choices=MODELS,default=MODELS)
    parser.add_argument('--dpi',type=int,default=180)
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if args.stage=='rank':
        for model in args.models:rank_model(model)
    elif args.stage=='plot':
        for model in args.models:plot_model(model,args.dpi)
    else:summarize()


if __name__=='__main__':main()
