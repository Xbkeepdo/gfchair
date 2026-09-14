"""Offline, complete signal gallery; no model forward or detector training."""
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_ffn_input_geometry import summarize

BASE = ROOT/'outputs/ffn_all_source_paths_v1'
OUT = BASE/'signals_20260913'
MODELS = ('qwen2_5_vl_7b', 'llava_1_5_7b', 'qwen3_vl_8b', 'internvl_2_5_8b')
NAMES = ('Qwen2.5-VL', 'LLaVA-1.5', 'Qwen3-VL', 'InternVL2.5')
SOURCE = ('prompt', 'visual', 'generation')
COLORS = ('#bd6716', '#2563eb', '#16864b', '#9b43aa')


def read_csv(path):
    with path.open() as f: return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def supplement():
    """Same shared 500 images / conflict exclusion / mention weights as main study."""
    cohort = json.loads((ROOT/'outputs/ffn_output_cosine_20260909/cohort500.json').read_text())['split']
    ids, train = set(cohort['train']+cohort['test']), set(cohort['train'])
    allcurves, allpaired = [], []
    for model in MODELS:
        directory = OUT/model; directory.mkdir(parents=True, exist_ok=True)
        if (directory/'curves.csv').exists():
            allcurves += read_csv(directory/'curves.csv')
            allpaired += read_csv(directory/'paired_images.csv')
            continue
        old = torch.load(ROOT/'outputs/ffn_source_composition_v1'/model/'matrices.pt', map_location='cpu', weights_only=False)
        mentions = [m for m in old['mentions'] if m['image_id'] in ids]
        del old
        labels = defaultdict(set)
        for m in mentions: labels[m['target_key']].add(m['label'])
        mentions = [dict(m, label_conflict=False, split='train' if m['image_id'] in train else 'test')
                    for m in mentions if len(labels[m['target_key']]) == 1]
        with np.load(ROOT/'outputs/prefix_attention_gate/full/region_plots'/model/'regions.npz') as z:
            lookup = {key: i for i, key in enumerate(z['mention_ids'].tolist())}
            index = [lookup[m['mention_id']] for m in mentions]
            np.testing.assert_array_equal(z['labels'][index], [m['label'] for m in mentions])
            np.testing.assert_array_equal(z['image_ids'][index], [m['image_id'] for m in mentions])
            v = z['values'][index].astype(float)
        region = np.stack((v[..., 0]+v[..., 2], v[..., 1], v[..., 3]), axis=-1)
        np.testing.assert_allclose(region.sum(-1), v.sum(-1), atol=1e-7, rtol=1e-7)
        arrays = {source+'_'+kind: region[:, i, :, j] for i, kind in enumerate(('attention_mass', 'attention_gate_mass'))
                  for j, source in enumerate(SOURCE)}
        wanted = {m['target_key'] for m in mentions}
        targets = {}
        for ii, image in enumerate(sorted(ids), 1):
            x = torch.load(BASE/model/'shards'/f'image_{image:012d}.pt', map_location='cpu', weights_only=False, mmap=True)
            if not x['complete']: raise ValueError('Incomplete image')
            for row in x['positions']:
                if row['target_key'] not in wanted: continue
                values = defaultdict(list)
                for layer in row['tokens']:
                    kind = layer['source_type'].numpy()
                    a, e, c = (layer[k].double().numpy() for k in ('write_norm','effect_norm','direction_cosine'))
                    for i, source in enumerate(SOURCE):
                        mask = kind == i
                        aa, ee, cc = a[mask], e[mask], c[mask]
                        gain = ee[aa>1e-12]/aa[aa>1e-12]
                        for suffix, val in (
                            ('token_write_mean', np.mean(aa) if len(aa) else np.nan),
                            ('token_effect_mean', np.mean(ee) if len(ee) else np.nan),
                            ('token_gain_median', np.median(gain) if len(gain) else np.nan),
                            ('token_direction_mean', np.mean(cc[np.isfinite(cc)]) if np.isfinite(cc).any() else np.nan)):
                            values[source+'_'+suffix].append(float(val))
                targets[row['target_key']] = values
            if ii%100 == 0: print(model, 'offline token summaries', ii, '/500', flush=True)
        assert set(targets) == wanted
        for field in next(iter(targets.values())):
            arrays[field] = np.asarray([targets[m['target_key']][field] for m in mentions])
        # Exact token arithmetic should recover the already-saved generation
        # per-token gross (up to the formal FP32 scalar storage precision).
        curves, paired = summarize(model, arrays, mentions, directory)
        np.savez_compressed(directory/'metrics.npz', **arrays)
        (directory/'mentions.json').write_text(json.dumps(mentions))
        (directory/'metadata.json').write_text(json.dumps(dict(images=500, mentions=len(mentions),
            token_unit='per-target within-source mean/median first; mentions equally weighted afterwards',
            routing='existing prefix attention cache; BOS merged with prompt; same conflict-excluded cohort',
            no_model_forward=True), indent=2))
        allcurves += curves; allpaired += paired
    write_csv(OUT/'supplemental_curves.csv', allcurves)
    write_csv(OUT/'supplemental_paired_images.csv', allpaired)
    return allcurves, allpaired


def source_fields(suffix): return [(s+'_'+suffix, s.title(), COLORS[i]) for i,s in enumerate(SOURCE)]


def validate(curves):
    lookup = {(r['model'],r['metric'],r['scope'],r['label'],r['layer']):r for r in curves}
    report = {}
    for model in MODELS:
        with np.load(OUT/model/'metrics.npz') as z:
            sums = sum(z[g+'_attention_mass'] for g in SOURCE)
            # Preserve the native-precision attention cache; do not renormalize.
            np.testing.assert_allclose(sums, 1, atol=2e-3, rtol=2e-3)
            count = z['generation_token_effect_mean'].shape[0]
        mentions = json.loads((OUT/model/'mentions.json').read_text())
        assert count == len(mentions)
        errors = []
        for row in curves:
            if row['model'] != model or row['metric'] != 'generation_per_token_gross': continue
            target = lookup[(model,'generation_token_effect_mean',row['scope'],row['label'],row['layer'])]
            for field in ('mean','median','q25','q75'):
                a,b = float(row[field]),float(target[field])
                np.testing.assert_allclose(a,b,atol=1e-6,rtol=1e-6)
                errors.append(abs(a-b))
            for field in ('n_total','n_valid','n_undefined'):
                assert int(row[field]) == int(target[field])
        report[model] = dict(mentions=count,attention_mass_max_sum_error=float(np.abs(sums-1).max()),
            token_vs_existing_stats_max_abs_error=max(errors),
            metric_fields=len({r['metric'] for r in curves if r['model']==model}))
    (OUT/'validation.json').write_text(json.dumps(report,indent=2))


def residual_fields(prefix, suffix):
    return [(prefix+'_'+s+suffix, s.title(), color) for s,color in
            (('residual',COLORS[3]),('prompt',COLORS[0]),('visual',COLORS[1]),('generation',COLORS[2]))]


def figure(curves, name, title, panels, scope='all', invalid=False):
    lookup = defaultdict(list)
    for r in curves:
        if r['scope'] == scope: lookup[(r['model'], r['metric'], r['label'])].append(r)
    fig, axes = plt.subplots(len(panels), 4, figsize=(15, 2.7*len(panels)+.8), squeeze=False)
    for ri, (ylabel, fields, logarithmic) in enumerate(panels):
        for mi, model in enumerate(MODELS):
            ax = axes[ri, mi]
            for metric, label, color in fields:
                for group, style in (('REAL','-'),('HALL','--')):
                    rows = sorted(lookup[(model, metric, group)], key=lambda r:int(r['layer']))
                    if not rows: raise ValueError((model,metric,scope))
                    x = [int(r['layer']) for r in rows]; y = [float(r['mean']) for r in rows]
                    if len(fields)==1:
                        color = '#2563eb' if group=='REAL' else '#d45327'
                    ax.plot(x, y, color=color, ls=style, lw=1.35)
            if logarithmic: ax.set_yscale('log')
            if ('cos' in ylabel.lower() or 'rotation' in ylabel.lower() or 'balance' in ylabel.lower()) and ylabel != 'Visual context cosine':
                ax.axhline(0, color='black', lw=.6, alpha=.5)
            ax.grid(alpha=.16)
            if mi==0: ax.set_ylabel(ylabel, fontsize=10)
            if ri==0: ax.set_title(NAMES[mi], fontsize=11)
            if ri==len(panels)-1: ax.set_xlabel('Decoder layer')
            if mi==0:
                handles = ([Line2D([0],[0],color=c,label=n) for _,n,c in fields] if len(fields)>1 else [])
                handles += [Line2D([0],[0],color='black' if len(fields)>1 else '#2563eb',ls='-',label='REAL'),
                            Line2D([0],[0],color='black' if len(fields)>1 else '#d45327',ls='--',label='HALL')]
                ax.legend(handles=handles, fontsize=7, ncol=min(3,len(handles)), loc='best')
    warning = '\nB2 K4 DID NOT PASS FULL-DATA NUMERICAL VALIDATION; provisional curves only' if invalid else ''
    fig.suptitle(title+'\n'+scope.upper()+': shared 500-image cohort; conflict-excluded, mention-weighted means'+warning,
                 fontsize=11, color='#b91c1c' if invalid else 'black')
    fig.tight_layout(rect=(0,0,1,.95 if invalid else .96))
    for ext in ('png','pdf'): fig.savefig(OUT/f'{name}_{scope}.{ext}', dpi=160)
    plt.close(fig)


def main():
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    extra, paired_extra = supplement()
    curves = read_csv(BASE/'mechanism_curves.csv')+extra
    paired = read_csv(BASE/'mechanism_paired_images.csv')+paired_extra
    validate(curves)
    write_csv(OUT/'all_curves.csv', curves); write_csv(OUT/'all_paired_images.csv', paired)
    scalar = lambda key, name: [(key,name,'#2563eb')]
    groups = [
        ('01_routing','Attention routing and gate-weighted routing',[
            ('Raw attention mass',source_fields('attention_mass'),False),
            ('Attention x gate mass',source_fields('attention_gate_mass'),False)],False),
        ('02_strength','All-attention source magnitude and effective gain',[
            ('Gross response norm (log)',source_fields('gross_norm'),True),
            ('Net response norm (log)',source_fields('net_norm'),True),
            ('Group gain (log)',source_fields('group_gain'),True)],False),
        ('03_rotation_cancellation','Source rotation, within-source and between-source cancellation',[
            ('Group rotation cosine',source_fields('group_rotation'),False),
            ('Within-source kappa',source_fields('within_source_kappa'),False),
            ('Between-source kappa',scalar('source_group_kappa','kappa'),False)],False),
        ('04_interactions','Source-pair geometry before and after the FFN',[
            ('Input cosine',[(f'cos_{s}_in',s.upper(),COLORS[i]) for i,s in enumerate(('vg','vp','pg'))],False),
            ('Output cosine',[(f'cos_{s}_out',s.upper(),COLORS[i]) for i,s in enumerate(('vg','vp','pg'))],False),
            ('Delta cosine: out - in',[(f'delta_cos_{s}',s.upper(),COLORS[i]) for i,s in enumerate(('vg','vp','pg'))],False)],False),
        ('05_context_length','Visual context modulation and generation-length control',[
            ('Visual context cosine',scalar('visual_context_cos','Context'),False),
            ('Visual relative change (log)',scalar('visual_context_relative_change','Context'),True),
            ('Generation gross norm (log)',scalar('generation_gross_norm','Gross'),True),
            ('Generation response / token (log)',scalar('generation_per_token_gross','Per token'),True)],False),
        ('06_tokens','Token-level write, response, gain and rotation',[
            ('Mean token write norm (log)',source_fields('token_write_mean'),True),
            ('Mean token response norm (log)',source_fields('token_effect_mean'),True),
            ('Median token gain (log)',source_fields('token_gain_median'),True),
            ('Mean token direction cosine',source_fields('token_direction_mean'),False)],False)]
    for b in ('b1','b2'):
        groups += [(f'07_{b}_integration',b.upper()+': residual share and attention-residual integration',[
            ('Residual share',scalar(b+'_residual_share','Share'),False),
            ('Attention-residual kappa',scalar(b+'_attention_residual_kappa','Kappa'),False),
            ('Residual G-V balance',scalar(b+'_residual_visual_generation_balance','Balance'),False)],b=='b2'),
            (f'08_{b}_geometry',b.upper()+': source magnitude and residual-source geometry',[
            ('Source response norm (log)',residual_fields(b,'_norm')+[(b+'_attention_total_norm','Attention net','#555555')],True),
            ('Residual-source cosine',[(b+'_cos_residual_'+s,'R-'+s[0].upper(),COLORS[SOURCE.index(s)]) for s in ('visual','prompt','generation')],False),
            ('Residual-source delta cosine',[(b+'_delta_residual_'+s+'_cos','R-'+s[0].upper(),COLORS[SOURCE.index(s)]) for s in ('visual','generation')],False)],b=='b2')]
    for scope in ('all','train','test'):
        for name,title,panels,bad in groups: figure(curves,name,title,panels,scope,bad)
    # Uniform field-by-field table: these layer averages are descriptive,
    # not independent-layer tests or token-pooled population estimates.
    table=[]
    for model in MODELS:
        metrics=sorted({r['metric'] for r in curves if r['model']==model})
        for metric in metrics:
            for scope in ('all','train','test'):
                q=[r for r in curves if r['model']==model and r['metric']==metric and r['scope']==scope]
                p=[r for r in paired if r['model']==model and r['metric']==metric and r['scope']==scope]
                delta=np.array([float(r['mean_hall_minus_real']) for r in p])
                row=dict(model=model,metric=metric,scope=scope,layers=len(delta),
                    paired_positive_layers=int((delta>0).sum()),paired_negative_layers=int((delta<0).sum()),
                    paired_layer_mean=float(np.nanmean(delta)),numerical_status='B2_K4_PROVISIONAL' if metric.startswith('b2_') else 'see full numerical report')
                for label in ('REAL','HALL'):
                    selected=[r for r in q if r['label']==label]
                    row[label.lower()+'_layer_mean']=float(np.nanmean([float(r['mean']) for r in selected]))
                    row[label.lower()+'_undefined_entries']=sum(int(r['n_undefined']) for r in selected)
                table.append(row)
    write_csv(OUT/'signal_summary.csv',table)
    html=['<!doctype html><meta charset="utf-8"><title>全部信号图册</title>',
          '<style>body{font:16px sans-serif;margin:24px;max-width:1500px}img{width:100%}section{margin:40px 0}a{margin-right:14px}</style>',
          '<h1>全部信号图册</h1><p>共享500图；REAL实线、HALL虚线。按mention等权，排除冲突目标；图示均值，完整median/IQR和同图配对数据见CSV。B2 K4曲线仅作带精度限制的观察。</p>']
    for name,title,_,bad in groups:
        html += [f'<section><h2>{title}</h2>',f'<p>{"B2精度未通过，不能据此确认机制。" if bad else ""}<a href="{name}_all.pdf">PDF</a><a href="{name}_train.png">Train 400</a><a href="{name}_test.png">Test 100</a></p>',f'<img src="{name}_all.png"></section>']
    (OUT/'index.html').write_text('\n'.join(html))
    print('COMPLETE',OUT, 'metrics',len({r['metric'] for r in curves}),flush=True)


if __name__=='__main__': main()
