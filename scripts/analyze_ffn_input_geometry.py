"""Describe per-source FFN amplification and vector cancellation from v2 caches."""
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_ffn_target_consequence import parent_artifacts
from scripts.run_ffn_visual_source_attribution import EXPERIMENT

MODELS = ('qwen2_5_vl_7b', 'llava_1_5_7b', 'qwen3_vl_8b', 'internvl_2_5_8b')
OUT = ROOT / 'outputs/ffn_input_geometry_20260912'
TITLES = dict(amp_mean='Mean source gain ||e_m|| / ||a_m||',
              amp_median='Median source gain ||e_m|| / ||a_m||',
              amp_fraction_gt1='Fraction of valid sources with gain > 1',
              gross_gain='Sum ||e_m|| / sum ||a_m|| (weighted gain)',
              cancellation='||sum e_m|| / sum ||e_m|| (lower = more cancellation)',
              closure='Relative quadrature closure error',
              rotation_mean='Mean source cosine(a_m, e_m)',
              rotation_median='Median source cosine(a_m, e_m)',
              input_cancellation='||sum a_m|| / sum ||a_m||',
              cancellation_change='Output cancellation ratio - input ratio')


def divide(a, b):
    a, b = np.broadcast_arrays(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    return np.divide(a, b, out=np.full(a.shape, np.nan), where=b > 0)


def geometry(write, effect, net, closure):
    a, e = np.asarray(write, dtype=float), np.asarray(effect, dtype=float)
    if a.ndim != 2 or a.shape != e.shape or not a.shape[1]:
        raise ValueError('Expected matching nonempty [layers, visual sources] norms')
    if not np.isfinite(a).all() or not np.isfinite(e).all() or min(a.min(), e.min()) < 0:
        raise ValueError('Nonfinite/negative norms')
    for value in (net, closure):
        value = np.asarray(value, dtype=float)
        if value.shape != (a.shape[0],) or not np.isfinite(value).all() or (value < 0).any():
            raise ValueError('Invalid net norm or closure diagnostic')
    gain = divide(e, a)
    valid = np.isfinite(gain)
    count = valid.sum(-1)
    median = np.array([np.median(x[np.isfinite(x)]) if np.isfinite(x).any() else np.nan for x in gain])
    kappa = divide(net, e.sum(-1))
    if np.nanmax(kappa, initial=0) > 1.00001:
        raise ValueError('Vector cancellation exceeds triangle inequality')
    values = dict(amp_mean=divide(np.nansum(gain, axis=-1), count), amp_median=median,
                  amp_fraction_gt1=divide(((gain > 1) & valid).sum(-1), count),
                  gross_gain=divide(e.sum(-1), a.sum(-1)), cancellation=kappa,
                  closure=np.asarray(closure, dtype=float))
    return values, dict(source_entries=int(a.size), zero_write=int((a == 0).sum()),
                        zero_effect=int((e == 0).sum()),
                        zero_write_nonzero_effect=int(((a == 0) & (e > 0)).sum()),
                        zero_total_effect=int((e.sum(-1) == 0).sum()))


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader(); w.writerows(rows)


def collect(model):
    parents, _ = parent_artifacts(model)
    split = json.loads((ROOT/'outputs'/model/EXPERIMENT/'image_splits.json').read_text())
    assert len(parents) == 4000 and set(parents) == set(split['train'] + split['test'])
    train = set(split['train'])
    arrays, mentions, seen = {}, [], set()
    audit = dict(images=0, targets=0, source_entries=0, zero_write=0, zero_effect=0,
                 zero_write_nonzero_effect=0, zero_total_effect=0, conflicting_label_targets=0)
    for index, (image_id, (path, _)) in enumerate(sorted(parents.items()), 1):
        shard = torch.load(path, map_location='cpu', weights_only=False)
        assert shard['processed_image'] and shard['image_ids'] == [image_id]
        local = {}
        labels_by_target = {}
        for mention in shard['sample_table']:
            labels_by_target.setdefault(mention['target_key'], set()).add(mention['label'])
        for row in shard['positions']:
            key = row['target_key']
            assert key not in seen and key in labels_by_target
            seen.add(key)
            values, counts = geometry(row['write_mag'], row['ffn_path_gross'], row['N_vec'],
                                      row['completeness_relative_error'])
            defined = np.isfinite(values['cancellation'])
            np.testing.assert_allclose(values['cancellation'][defined], np.asarray(row['kappa_vec'])[defined], rtol=2e-6, atol=1e-7)
            local[key] = values
            for k, value in counts.items(): audit[k] += value
            audit['targets'] += 1
            audit['conflicting_label_targets'] += len(labels_by_target[key]) > 1
        for row in shard['sample_table']:
            assert row['label'] in (0, 1)
            mentions.append(dict(row, split='train' if image_id in train else 'test',
                                 label_conflict=len(labels_by_target[row['target_key']]) > 1))
            for k, value in local[row['target_key']].items(): arrays.setdefault(k, []).append(value)
        audit['images'] += 1
        if index % 500 == 0: print(model, index, '/ 4000', flush=True)
    assert len({r['mention_id'] for r in mentions}) == len(mentions)
    directory = OUT/model
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.stack(v) for k, v in arrays.items()}
    np.savez_compressed(directory/'metrics.npz', **arrays)
    (directory/'mentions.json').write_text(json.dumps(mentions))
    audit['mentions'] = len(mentions)
    (directory/'audit.json').write_text(json.dumps(audit, indent=2))
    return arrays, mentions


def summarize(model, arrays, mentions, directory):
    labels = np.array([r['label'] for r in mentions])
    images = np.array([r['image_id'] for r in mentions])
    splits = np.array([r['split'] for r in mentions])
    conflict = np.array([r['label_conflict'] for r in mentions])
    rows, paired = [], []
    # Primary curves retain the original mention weights. The paired control
    # excludes label-conflicting targets and weights each mixed-label image once.
    for scope in ('all', 'train', 'test'):
        scope_mask = np.ones(len(labels), dtype=bool) if scope == 'all' else splits == scope
        mixed = sorted(set(images[scope_mask & ~conflict & (labels == 0)]) &
                       set(images[scope_mask & ~conflict & (labels == 1)]))
        for metric, matrix in arrays.items():
            for label, name in ((1, 'REAL'), (0, 'HALL')):
                data = matrix[scope_mask & (labels == label)]
                for layer, col in enumerate(data.T, 1):
                    valid = col[np.isfinite(col)]
                    q = np.quantile(valid, [.25, .5, .75]) if len(valid) else [np.nan]*3
                    rows.append(dict(model=model, scope=scope, metric=metric, label=name, layer=layer,
                        n_total=len(col), n_valid=len(valid), n_undefined=len(col)-len(valid),
                        mean=float(valid.mean()) if len(valid) else np.nan, q25=q[0], median=q[1], q75=q[2]))
            differences = []
            for image_id in mixed:
                group = scope_mask & ~conflict & (images == image_id)
                differences.append(np.nanmean(matrix[group & (labels == 0)], axis=0) -
                                   np.nanmean(matrix[group & (labels == 1)], axis=0))
            if differences:
                for layer, col in enumerate(np.asarray(differences).T, 1):
                    valid = col[np.isfinite(col)]
                    paired.append(dict(model=model, scope=scope, metric=metric, layer=layer,
                        n_mixed_images=len(col), n_valid=len(valid),
                        mean_hall_minus_real=float(valid.mean()) if len(valid) else np.nan,
                        median_hall_minus_real=float(np.median(valid)) if len(valid) else np.nan,
                        fraction_hall_gt_real=float((valid > 0).mean()) if len(valid) else np.nan))
    write_csv(directory/'curves.csv', rows)
    if paired: write_csv(directory/'paired_images.csv', paired)
    return rows, paired


def plot(rows, scope, metrics, path):
    fig, axes = plt.subplots(4, len(metrics), figsize=(5*len(metrics), 12), squeeze=False)
    for index, model in enumerate(MODELS):
        for j, metric in enumerate(metrics):
            ax = axes[index, j]
            for label, color in (('REAL', '#2166ac'), ('HALL', '#b2182b')):
                selected = [r for r in rows if r['model']==model and r['scope']==scope and r['metric']==metric and r['label']==label]
                if not selected: continue
                x = [r['layer'] for r in selected]
                ax.plot(x, [r['mean'] for r in selected], color=color, label=label+' mean')
                ax.plot(x, [r['median'] for r in selected], color=color, ls='--', alpha=.8, label=label+' median')
            ax.set_title(model+'\n'+TITLES[metric], fontsize=9)
            ax.set_xlabel('Decoder layer'); ax.grid(alpha=.2)
            if index == 0: ax.legend(fontsize=7)
    fig.suptitle(scope.upper()+' | mention weighted; solid mean, dashed median; descriptive comparison')
    fig.tight_layout(rect=(0, 0, 1, .97))
    for suffix in ('.png', '.pdf'): fig.savefig(path.with_suffix(suffix), dpi=160)
    plt.close(fig)


def contrasts(rows, paired):
    result=[]
    for model in sorted({r['model'] for r in rows}):
        for scope in ('all','train','test'):
            for metric in sorted({r['metric'] for r in rows}):
                selected=[r for r in rows if r['model']==model and r['scope']==scope and r['metric']==metric]
                if not selected: continue
                real=np.array([r['mean'] for r in selected if r['label']=='REAL'])
                hall=np.array([r['mean'] for r in selected if r['label']=='HALL'])
                delta=hall-real
                control=[r for r in paired if r['model']==model and r['scope']==scope and r['metric']==metric]
                result.append(dict(model=model,scope=scope,metric=metric,layers=len(delta),
                    real_layer_mean=float(np.mean(real)),hall_layer_mean=float(np.mean(hall)),
                    hall_minus_real=float(np.mean(delta)),hall_higher_layers=int((delta>0).sum()),
                    mixed_images=control[0]['n_mixed_images'] if control else 0,
                    paired_mean_hall_minus_real=float(np.mean([r['mean_hall_minus_real'] for r in control])) if control else np.nan))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', choices=MODELS, default=MODELS)
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    all_rows, all_paired = [], []
    for model in args.models:
        directory = OUT/model
        if args.summarize_only:
            arrays = dict(np.load(directory/'metrics.npz'))
            mentions = json.loads((directory/'mentions.json').read_text())
        else: arrays, mentions = collect(model)
        rows, paired = summarize(model, arrays, mentions, directory)
        all_rows.extend(rows); all_paired.extend(paired)
    write_csv(OUT/'curves.csv', all_rows)
    write_csv(OUT/'paired_images.csv', all_paired)
    write_csv(OUT/'summary.csv', contrasts(all_rows,all_paired))
    for scope in ('all', 'test'):
        plot(all_rows, scope, ('amp_mean', 'amp_fraction_gt1', 'cancellation'), OUT/f'{scope}_geometry')
    print('DONE', OUT, flush=True)


if __name__ == '__main__': main()
