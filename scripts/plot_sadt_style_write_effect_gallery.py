#!/usr/bin/env python3
"""Build SADT-style Top-16/Top-32 Attention, AE, P_E and cosine galleries."""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
import statistics
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.plot_spatial_attention_gate_source_heatmaps import model_view

MODELS = ('qwen2_5_vl_7b', 'llava_1_5_7b', 'qwen3_vl_8b', 'internvl_2_5_8b')
MODEL_NAMES = dict(qwen2_5_vl_7b='Qwen2.5-VL-7B', llava_1_5_7b='LLaVA-1.5-7B',
                   qwen3_vl_8b='Qwen3-VL-8B', internvl_2_5_8b='InternVL2.5-8B')
LAYER_COUNTS = dict(zip(MODELS, (28, 32, 36, 32)))
SIGNALS = (('attention', 'Attention'), ('ae', 'AE / T'), ('pe', r'$P_E^{all}$'),
           ('cosine_softmax', r'softmax(cos / 0.2)'))
OUT = ROOT/'outputs/ffn_write_effect_sadt_gallery_v1'
COCO = Path('/home/apulis-dev/userdata/DGST/token-grounding-detector/data/coco')
COHORT = ROOT/'outputs/ffn_output_cosine_20260909/cohort500.json'
OFFICIAL = 'COCO4000-INSLEN-OFFICIAL-TARGET'
TAU = .2
SMALL_MAX_FRACTION = .03
MIN_REAL_FRACTION = .001


def read(path: Path):
    return torch.load(path, map_location='cpu', weights_only=False)


def selected_layers(model: str) -> list[int]:
    count = LAYER_COUNTS[model]
    return list(dict.fromkeys(x for x in (10, 15, 20, 25, 30, count) if x <= count))


def normalize_rows(values) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0., posinf=0., neginf=0.)
    values = np.maximum(values, 0.)
    total = values.sum(-1, keepdims=True)
    return np.divide(values, np.maximum(total, 1e-12), out=np.zeros_like(values), where=total > 0)


def softmax_rows(values, tau=TAU) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0., posinf=1., neginf=-1.)/tau
    values -= values.max(-1, keepdims=True)
    exp = np.exp(values)
    return exp/exp.sum(-1, keepdims=True)


def top_indices(values: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(-values, kind='stable')[:min(k, values.size)]


def top_mass(values: np.ndarray, k: int) -> float:
    return float(values[top_indices(values, k)].sum())


def js_distance(p: np.ndarray, q: np.ndarray) -> float:
    midpoint = .5*(p+q)
    left = np.where(p > 0, p*np.log(np.maximum(p, 1e-12)/np.maximum(midpoint, 1e-12)), 0.)
    right = np.where(q > 0, q*np.log(np.maximum(q, 1e-12)/np.maximum(midpoint, 1e-12)), 0.)
    return float(.5*(left.sum()+right.sum()))


def rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a, kind='stable'), kind='stable').astype(np.float64)
    rb = np.argsort(np.argsort(b, kind='stable'), kind='stable').astype(np.float64)
    if ra.std() == 0 or rb.std() == 0:
        return 0.
    return float(np.corrcoef(ra, rb)[0, 1])


def overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    ia, ib = set(top_indices(a, k).tolist()), set(top_indices(b, k).tolist())
    return len(ia & ib)/max(min(k, a.size, b.size), 1)


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
    values = [float(x.get('area', x['bbox'][2]*x['bbox'][3]))/max(width*height, 1)
              for x in annotations.get(image_id, []) if int(x['category_id']) == category]
    return min(values) if values else None


def official_data(model: str):
    folder = ROOT/'outputs'/model/OFFICIAL
    return (json.loads((folder/'generations.json').read_text()),
            json.loads((folder/'labeling.json').read_text()))


def matching_span(labeling_row: dict, response_index: int, label: int, canonical: str):
    matches = []
    for span in labeling_row.get('object_token_spans', []):
        positions = [int(x) for x in span.get('token_indices', [])]
        if positions and positions[0] == response_index and int(span.get('label', -1)) == label:
            matches.append(span)
    exact = [x for x in matches if str(x.get('canonical_object') or x.get('word') or '').lower() == canonical]
    span = (exact or matches or [{}])[0]
    return dict(surface=str(span.get('surface') or span.get('surface_word') or canonical),
                detected_word=str(span.get('official_detected_word') or span.get('surface') or canonical),
                occurrence_index=int(span.get('occurrence_index') or 1))


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


def load_arrays(model: str, image_id: int, target_key: str, layers: list[int]):
    old = read(ROOT/'outputs/ffn_source_composition_v1'/model/'shards/k50'/f'image_{image_id:012d}.pt')
    all_source = read(ROOT/'outputs/ffn_all_source_paths_v1'/model/'shards'/f'image_{image_id:012d}.pt')
    old_row = next(x for x in old['positions'] if x['target_key'] == target_key)
    all_row = next(x for x in all_source['positions'] if x['target_key'] == target_key)
    indices = [x-1 for x in layers]
    attention = normalize_rows(old_row['raw_attention_mean'][indices].numpy())
    ae = normalize_rows(old_row['attention_evidence'][indices].numpy())
    pe, cosine = effect_and_cosine(all_row, old_row, layers)
    return old_row, dict(attention=attention, ae=ae, pe=pe,
                         cosine_softmax=softmax_rows(cosine)), cosine


def relationship(attention: np.ndarray, pe: np.ndarray):
    result = dict(js=[], tv=[], spearman=[], overlap16=[], overlap32=[])
    for a, p in zip(attention, pe):
        result['js'].append(js_distance(a, p))
        result['tv'].append(float(.5*np.abs(a-p).sum()))
        result['spearman'].append(rank_correlation(a, p))
        result['overlap16'].append(overlap(a, p, 16))
        result['overlap32'].append(overlap(a, p, 32))
    return result


def rank_model(model: str):
    layers = selected_layers(model)
    category_ids, annotations, image_sizes, medians = coco_metadata()
    generations, labeling = official_data(model)
    matrix = read(ROOT/'outputs/ffn_source_composition_v1'/model/'matrices.pt')
    cohort = json.loads(COHORT.read_text())['split']; image_ids = set(cohort['train']+cohort['test'])
    labels = defaultdict(set)
    for mention in matrix['mentions']: labels[mention['target_key']].add(int(mention['label']))
    by_image = defaultdict(list); seen = set()
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
        label_row = labeling.get(str(image_id), {})
        span = matching_span(label_row, int(mention['response_index']), label, canonical)
        generated_text = str(generations[str(image_id)]['generated_text'])
        by_image[image_id].append(dict(target_key=mention['target_key'], label=label,
            response_index=int(mention['response_index']), canonical_object=canonical,
            object_fraction=float(fraction), size_basis=basis,
            size_priority='small' if fraction <= SMALL_MAX_FRACTION else 'large',
            generated_text=generated_text, response_chars=len(generated_text), **span))

    candidates=[]
    for number, image_id in enumerate(sorted(by_image), 1):
        old_path=ROOT/'outputs/ffn_source_composition_v1'/model/'shards/k50'/f'image_{image_id:012d}.pt'
        all_path=ROOT/'outputs/ffn_all_source_paths_v1'/model/'shards'/f'image_{image_id:012d}.pt'
        if not old_path.exists() or not all_path.exists(): continue
        old=read(old_path); all_source=read(all_path)
        old_rows={x['target_key']:x for x in old['positions']}; all_rows={x['target_key']:x for x in all_source['positions']}
        for meta in by_image[image_id]:
            key=meta['target_key']
            if key not in old_rows or key not in all_rows: continue
            old_row=old_rows[key]; indices=[x-1 for x in layers]
            attention=normalize_rows(old_row['raw_attention_mean'][indices].numpy())
            ae=normalize_rows(old_row['attention_evidence'][indices].numpy())
            pe, cosine=effect_and_cosine(all_rows[key],old_row,layers)
            rel=relationship(attention,pe)
            divergence=float(np.mean(rel['js'])+.25*(1-np.mean(rel['overlap16']))+
                             .15*(1-np.mean(rel['overlap32'])))
            divergence_peak=float(max(rel['js'])+.25*(1-min(rel['overlap16']))+
                                  .15*(1-min(rel['overlap32'])))
            focus=float(np.mean([[top_mass(x,16) for x in attention],
                                 [top_mass(x,16) for x in ae],
                                 [top_mass(x,16) for x in pe]]))
            candidates.append(dict(model=model,image_id=image_id,layers=layers,
                visual_grid=list(old_row['visual_grid']),visual_tokens=int(attention.shape[1]),
                divergence_score=divergence,divergence_peak_score=divergence_peak,focus_score=focus,
                attention_pe=rel,js_mean=float(np.mean(rel['js'])),tv_mean=float(np.mean(rel['tv'])),
                spearman_mean=float(np.mean(rel['spearman'])),
                overlap16_mean=float(np.mean(rel['overlap16'])),
                overlap32_mean=float(np.mean(rel['overlap32'])),
                cosine_mean=float(np.nanmean(cosine)),**meta))
        if number%100==0: print('RANK',model,number,'/',len(by_image),len(candidates),flush=True)
    folder=OUT/model;folder.mkdir(parents=True,exist_ok=True)
    (folder/'candidates.json').write_text(json.dumps(candidates,ensure_ascii=False,indent=2)+'\n')
    selection=select_candidates(candidates)
    (folder/'selection.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2)+'\n')
    print('SELECT',model,[(x['gallery'],x['case'],x['label_name'],x['canonical_object'],round(x['js_mean'],3),round(x['overlap16_mean'],2)) for x in selection],flush=True)


def take(pool, metric, count, used_images, used_words, gallery, size, reverse=True):
    pool=[x for x in pool if x['size_priority']==size]
    ranked=sorted(pool,key=lambda x:(x[metric],-x['response_chars']),reverse=reverse)
    chosen=[]
    for new_word in (True,False):
        for row in ranked:
            if len(chosen)==count: break
            if row['image_id'] in used_images or row['image_id'] in {x['image_id'] for x in chosen}: continue
            if new_word and row['canonical_object'] in used_words: continue
            chosen.append(dict(row,gallery=gallery,selection_metric=metric,
                               selection_score=row[metric]))
            used_words.add(row['canonical_object'])
        if len(chosen)==count: break
    if len(chosen)!=count: raise RuntimeError(f'{gallery} {size}: {len(chosen)}/{count}')
    used_images.update(x['image_id'] for x in chosen)
    return chosen


def select_candidates(candidates):
    used_images,used_words=set(),set(); main=[]
    hall=[x for x in candidates if x['label']==0]; real=[x for x in candidates if x['label']==1]
    for row in hall:
        rel=row['attention_pe']
        row.setdefault('divergence_peak_score',float(max(rel['js'])+.25*(1-min(rel['overlap16']))+
                                                     .15*(1-min(rel['overlap32']))))
    extra=take(hall,'divergence_peak_score',3,used_images,used_words,'divergence_hall','small')
    main += take(hall,'divergence_score',4,used_images,used_words,'main','small')
    main += take(real,'focus_score',4,used_images,used_words,'main','small')
    main += take(hall,'divergence_score',1,used_images,used_words,'main','large')
    main += take(real,'focus_score',1,used_images,used_words,'main','large')
    for i,row in enumerate(main,1): row['case']=i;row['label_name']='HALL' if row['label']==0 else 'REAL'
    for i,row in enumerate(extra,1): row['case']=i;row['label_name']='HALL'
    result=main+extra
    if Counter(x['label'] for x in main)!=Counter({0:5,1:5}): raise ValueError('Bad main balance')
    return result


def response_segments(text: str, target: str, occurrence: int):
    matches=list(re.finditer(re.escape(target),text,flags=re.IGNORECASE)) if target else []
    chosen=matches[min(max(occurrence-1,0),len(matches)-1)] if matches else None
    pieces=[];position=0
    for match in re.finditer(r'\n|[^\s\n]+|[ \t]+',text.replace('\r','')):
        start,end=match.span(); token=match.group()
        highlight=bool(chosen and start<chosen.end() and end>chosen.start())
        pieces.append((token,highlight));position=end
    return pieces


def response_card(text: str, target: str, occurrence: int, label_name: str, width=1900):
    regular_path=font_manager.findfont('DejaVu Sans')
    bold_path=str(Path(regular_path).with_name('DejaVuSans-Bold.ttf'))
    body=ImageFont.truetype(regular_path,25);bold=ImageFont.truetype(bold_path,26)
    title=ImageFont.truetype(bold_path,32);badge=ImageFont.truetype(bold_path,28)
    margin=34;line_height=35;usable=width-2*margin
    draw=ImageDraw.Draw(Image.new('RGB',(1,1),'white'))
    lines=[];line=[];line_width=0.
    for token,highlight in response_segments(text,target,occurrence):
        if token=='\n':
            lines.append(line);line=[];line_width=0.;continue
        token=' ' if token.isspace() else token
        font=bold if highlight else body
        token_width=draw.textlength(token,font=font)
        if line and line_width+token_width>usable:
            lines.append(line);line=[];line_width=0.
            if token==' ': continue
        line.append((token,highlight,font,token_width));line_width+=token_width
    if line or not lines: lines.append(line)
    height=126+line_height*len(lines)+margin
    card=Image.new('RGB',(width,height),(248,248,248));d=ImageDraw.Draw(card)
    d.rectangle((1,1,width-2,height-2),outline=(175,175,175),width=2)
    d.text((margin,22),'Model-generated response',font=title,fill=(20,20,20))
    badge_text=f'DETECTED {label_name}: {target}'
    badge_width=d.textlength(badge_text,font=badge)+24
    d.rounded_rectangle((margin,68,margin+badge_width,108),radius=6,fill=(190,22,38))
    d.text((margin+12,73),badge_text,font=badge,fill='white')
    y=124
    for line in lines:
        x=margin
        for token,highlight,font,token_width in line:
            if highlight:
                d.rectangle((x-2,y,x+token_width+2,y+line_height-3),fill=(255,221,225))
            d.text((x,y),token,font=font,fill=(190,22,38) if highlight else (25,25,25))
            x+=token_width
        y+=line_height
    return card


def sadt_overlay(image: Image.Image, values: np.ndarray, grid: tuple[int,int], scale: float):
    relative=np.clip(values.reshape(grid)/max(scale,1e-12)*1.5,0.,1.).astype(np.float32)
    heat=Image.fromarray(relative).resize(image.size,Image.Resampling.BILINEAR)
    heat=np.asarray(heat,dtype=np.float32).clip(0.,1.)
    color=plt.get_cmap('jet')(heat)[...,:3]
    base=np.asarray(image,dtype=np.float32)/255.
    return np.clip(.45*color+.55*base,0.,1.)


def retained(values: np.ndarray, k: int):
    result=np.zeros_like(values)
    indices=top_indices(values,k);result[indices]=values[indices]
    return result,indices


def safe_slug(text: str):
    return re.sub(r'[^a-z0-9]+','-',text.lower()).strip('-') or 'target'


def plot_one(spec: dict, dpi: int):
    model=spec['model'];layers=spec['layers'];image_id=spec['image_id']
    old,matrices,raw_cosine=load_arrays(model,image_id,spec['target_key'],layers)
    original=Image.open(COCO/'val2014'/f'COCO_val2014_{image_id:012d}.jpg')
    image,_=model_view(original,model);original.close()
    grid=tuple(int(x) for x in old['visual_grid'])
    scales={name:max(float(values.max()),1e-12) for name,values in matrices.items()}
    card=response_card(spec['generated_text'],spec['surface'],spec['occurrence_index'],spec['label_name'])
    columns=len(layers);image_columns=2 if columns>=5 else 1
    header_ratio=max(2.8,min(10.,2.65*(columns-image_columns)*card.height/card.width))
    fig=plt.figure(figsize=(2.65*columns,header_ratio+15.2),constrained_layout=True)
    gs=fig.add_gridspec(9,columns,height_ratios=[header_ratio]+[1]*8)
    ax=fig.add_subplot(gs[0,:image_columns]);ax.imshow(image);ax.set_axis_off();ax.set_anchor('N')
    ax.set_title('Original image (model input view)',fontsize=10.5,fontweight='bold')
    ax=fig.add_subplot(gs[0,image_columns:]);ax.imshow(card);ax.set_axis_off()
    statistic_rows=[]
    row_index=1
    for k in (16,32):
        for signal,title in SIGNALS:
            for column,layer in enumerate(layers):
                values=matrices[signal][column]
                kept,indices=retained(values,k)
                axis=fig.add_subplot(gs[row_index,column])
                axis.imshow(sadt_overlay(image,kept,grid,scales[signal]),interpolation='nearest')
                if row_index==1: axis.set_title(f'Layer {layer}',fontsize=10.5,fontweight='bold')
                if column==0: axis.set_ylabel(f'{title}\nTop-{k}',fontsize=9.2,fontweight='semibold',labelpad=6)
                axis.text(.018,.025,f'mass={values[indices].sum():.3f}',transform=axis.transAxes,
                          color='white',fontsize=6.8,
                          bbox={'boxstyle':'square,pad=.15','facecolor':'black','edgecolor':'none','alpha':.58})
                axis.set_xticks([]);axis.set_yticks([])
                for spine in axis.spines.values(): spine.set_color('#333333');spine.set_linewidth(.55)
                statistic_rows.append(dict(model=model,gallery=spec['gallery'],case=spec['case'],
                    label=spec['label'],label_name=spec['label_name'],image_id=image_id,
                    target_key=spec['target_key'],surface=spec['surface'],canonical_object=spec['canonical_object'],
                    layer=layer,signal=signal,top_k=k,topk_mass=float(values[indices].sum()),
                    topk_indices=','.join(map(str,indices.tolist()))))
            row_index+=1
    peak_offset=int(np.argmax(spec['attention_pe']['js']));peak_layer=layers[peak_offset]
    fig.suptitle(f"{MODEL_NAMES[model]} | {spec['label_name']} target '{spec['surface']}' "
                 f"({spec['canonical_object']}) | COCO {image_id}\n"
                 f"Attention vs $P_E$: mean JS={spec['js_mean']:.3f}, max JS={spec['attention_pe']['js'][peak_offset]:.3f} "
                 f"at L{peak_layer}, min overlap@16={min(spec['attention_pe']['overlap16']):.3f}, "
                 f"min overlap@32={min(spec['attention_pe']['overlap32']):.3f}",fontsize=12.5,fontweight='bold')
    prefix='main' if spec['gallery']=='main' else 'divergence_hall'
    stem=f"{prefix}_{spec['case']:02d}_{spec['label_name'].lower()}_{image_id}_{safe_slug(spec['canonical_object'])}"
    path=OUT/model/(stem+'.jpg')
    fig.savefig(path,dpi=dpi,bbox_inches='tight',facecolor='white',
                pil_kwargs={'quality':94,'subsampling':0,'optimize':True})
    plt.close(fig)
    return dict(spec,jpg=str(path.relative_to(ROOT))),statistic_rows


def plot_model(model: str,dpi: int):
    selection=json.loads((OUT/model/'selection.json').read_text())
    manifest=[];statistics_rows=[]
    for spec in selection:
        row,stats=plot_one(spec,dpi);manifest.append(row);statistics_rows+=stats
        print('PLOT',model,spec['gallery'],spec['case'],spec['label_name'],spec['canonical_object'],flush=True)
    (OUT/model/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    with (OUT/model/'topk_statistics.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(statistics_rows[0]));writer.writeheader();writer.writerows(statistics_rows)
    lines=[f'# {MODEL_NAMES[model]} SADT风格逐层案例','',
        '每图顶部为模型实际输入视图与完整生成原文，红色突出本次检测词；下方依次为四种信号的Top16与Top32。',
        '热图采用SADT作者代码的视觉原则：未保留位置置零，JET以0.45权重覆盖0.55原图，不画patch框。',
        'main共10例（HALL/REAL各5）；divergence_hall另含3个Attention与P_E差异大的HALL案例。','',
        '| 组 | # | 标签 | 目标 | 图片 | mean JS | max JS@层 | min overlap@16 | min overlap@32 |',
        '|---|---:|---|---|---:|---:|---:|---:|---:|']
    for row in manifest:
        peak=int(np.argmax(row['attention_pe']['js']))
        lines.append(f"| {row['gallery']} | {row['case']} | {row['label_name']} | {row['surface']} ({row['canonical_object']}) | "
                     f"[{row['image_id']}]({Path(row['jpg']).name}) | {row['js_mean']:.3f} | "
                     f"{row['attention_pe']['js'][peak]:.3f}@L{row['layers'][peak]} | "
                     f"{min(row['attention_pe']['overlap16']):.3f} | {min(row['attention_pe']['overlap32']):.3f} |")
    (OUT/model/'summary.md').write_text('\n'.join(lines)+'\n')


def relationship_outputs():
    rows=[]
    for model in MODELS:
        candidates=json.loads((OUT/model/'candidates.json').read_text())
        for item in candidates:
            for offset,layer in enumerate(item['layers']):
                rows.append(dict(model=model,label=item['label'],label_name='HALL' if item['label']==0 else 'REAL',
                    image_id=item['image_id'],target_key=item['target_key'],surface=item['surface'],
                    canonical_object=item['canonical_object'],layer=layer,
                    js=item['attention_pe']['js'][offset],tv=item['attention_pe']['tv'][offset],
                    spearman=item['attention_pe']['spearman'][offset],
                    overlap16=item['attention_pe']['overlap16'][offset],
                    overlap32=item['attention_pe']['overlap32'][offset]))
    with (OUT/'attention_pe_relationship.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    fig,axes=plt.subplots(len(MODELS),3,figsize=(13,11),constrained_layout=True)
    metrics=(('js','JS (natural log)'),('overlap16','Top-16 overlap'),('spearman','rank correlation'))
    for r,model in enumerate(MODELS):
        model_rows=[x for x in rows if x['model']==model];layers=selected_layers(model)
        for c,(metric,title) in enumerate(metrics):
            axis=axes[r,c]
            for label,color,name in ((0,'#d62728','HALL'),(1,'#1f77b4','REAL')):
                groups=[[float(x[metric]) for x in model_rows if x['label']==label and x['layer']==layer] for layer in layers]
                median=np.asarray([np.median(x) for x in groups]);lo=np.asarray([np.quantile(x,.25) for x in groups]);hi=np.asarray([np.quantile(x,.75) for x in groups])
                axis.plot(layers,median,color=color,marker='o',label=name);axis.fill_between(layers,lo,hi,color=color,alpha=.16)
            axis.grid(alpha=.2);axis.set_xticks(layers)
            if r==0:axis.set_title(title,fontweight='bold')
            if c==0:axis.set_ylabel(MODEL_NAMES[model],fontweight='semibold')
            if r==0 and c==2:axis.legend(frameon=False)
    path=OUT/'attention_pe_relationship.png';fig.savefig(path,dpi=180,bbox_inches='tight',facecolor='white');plt.close(fig)
    return rows


def summarize():
    relationship_rows=relationship_outputs();all_manifest=[]
    for model in MODELS:all_manifest+=json.loads((OUT/model/'manifest.json').read_text())
    lines=['# SADT风格 Attention、AE、P_E 与 cosine Top16/Top32 图册','',
        '每模型10个main案例（HALL/REAL各5）和3个额外高差异HALL案例，共52张大图。所有图保留指定层，不作跨层mean。',
        '每张图先给模型实际输入视图和完整生成原文，并用红色标出本次检测词；随后分别给Top16和Top32下的Attention、AE、P_E与softmax(cos/0.2)。',
        '空间图仿照SADT作者实现：TopK之外置零，JET热图0.45+原图0.55；因此背景呈冷蓝色且没有patch框。',
        'P_E来自all-attention K32逐token ||e_m||并在视觉token上归一化；cosine为cos(a_m,e_m)，softmax在每层全部视觉token上计算。softmax会抹去cosine正负号，本图只表达相对空间排序。',
        '额外HALL案例按mean JS、Top16/32低重合共同排序；完整候选逐层关系保存在CSV，避免只用挑图论证P_E不同于Attention。','',
        '[Attention–P_E全候选关系曲线](attention_pe_relationship.png)；[逐目标逐层CSV](attention_pe_relationship.csv)。','',
        '| 模型 | main | 额外差异HALL | 层 | 图册 |','|---|---:|---:|---|---|']
    for model in MODELS:
        layers='/'.join(map(str,selected_layers(model)))
        lines.append(f'| {MODEL_NAMES[model]} | 10 | 3 | {layers} | [{model}/summary.md]({model}/summary.md) |')
    lines+=['','## 全候选Attention–P_E关系','',
        '| 模型 | target-layer数 | JS中位数 | rank correlation中位数 | Top16 overlap中位数 | Top32 overlap中位数 |','|---|---:|---:|---:|---:|---:|']
    for model in MODELS:
        subset=[x for x in relationship_rows if x['model']==model]
        med=lambda key:float(np.median([float(x[key]) for x in subset]))
        lines.append(f"| {MODEL_NAMES[model]} | {len(subset)} | {med('js'):.3f} | {med('spearman'):.3f} | {med('overlap16'):.3f} | {med('overlap32'):.3f} |")
    lines+=['','运行：','', '```bash',
        'python scripts/plot_sadt_style_write_effect_gallery.py --stage rank --models <model>',
        'python scripts/plot_sadt_style_write_effect_gallery.py --stage plot --models <model>',
        'python scripts/plot_sadt_style_write_effect_gallery.py --stage summarize','```','',
        '只读取既有COCO4000文本、K50紧凑attention/AE缓存和共享500图all-attention K32缓存，不运行模型或路径积分。']
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=('rank','plot','summarize'),required=True)
    parser.add_argument('--models',nargs='+',choices=MODELS,default=MODELS)
    parser.add_argument('--dpi',type=int,default=135)
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if args.stage=='rank':
        for model in args.models:rank_model(model)
    elif args.stage=='plot':
        for model in args.models:plot_model(model,args.dpi)
    else:summarize()


if __name__=='__main__':main()
