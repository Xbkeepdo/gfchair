"""Cached true-RMS K32 WRITE/effect/gain analysis and fixed 811 ablations."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from features.tc_fvpa_artifacts import atomic_json_save, atomic_torch_save
from scripts import search_single_mlp_811 as trainer
from scripts.analyze_all_source_paths import mention_identity, write_csv

BASE = ROOT/'outputs/ffn_all_source_paths_v1'
OUT = ROOT/'outputs/all_attention_write_gain_811_v1'
MODELS = tuple(trainer.MODELS)
REGIONS = ('prompt', 'visual', 'generation', 'all')
FIELDS = ('I', 'S', 'G', 'lambda_q25', 'lambda_median', 'lambda_q75',
          'lambda_iqr', 'lambda_cv', 'lambda_gt1', 'log_I', 'log_S', 'log_G')
CONFIG = dict(width=128, standardize=True, batch_norm=False, dropout=.3,
              activation='relu', learning_rate=.001, weight_decay=1e-5,
              batch_size=128, max_epochs=150, patience=20, lr_patience=6,
              monitor='val_loss', early_stopping=True)
SEEDS = (43, 44, 45)


def progress(model, stage, done, total, **extra):
    atomic_json_save(dict(stage=stage, completed=done, total=total,
        heartbeat=datetime.now(timezone.utc).isoformat(), **extra), OUT/model/'progress.json')


def token_statistics(a, e):
    """Token-local quantiles; no denominator offset or tiny-source exclusion."""
    a, e = np.asarray(a, dtype=np.float64), np.asarray(e, dtype=np.float64)
    if a.shape != e.shape or a.ndim != 1 or not np.isfinite([a, e]).all() or (a < 0).any() or (e < 0).any():
        raise ValueError('Invalid cached source norms')
    if ((a == 0) & (e > 0)).any():
        raise ValueError('A zero WRITE has a nonzero response')
    i, s = a.sum(), e.sum()
    gain = e[a > 0]/a[a > 0]
    q = np.quantile(gain, [.25, .5, .75]) if len(gain) else [np.nan]*3
    g = s/i if i > 0 else np.nan
    cv = gain.std()/gain.mean() if len(gain) and gain.mean() > 0 else np.nan
    logs = [np.log(x) if x > 0 else np.nan for x in (i, s, g)]
    return np.asarray([i, s, g, *q, q[2]-q[0], cv,
                       np.mean(gain > 1) if len(gain) else np.nan, *logs])


def feature_sets(values):
    # values: mentions x layers x region x statistic; never include AE or gain.
    result = {}
    for support, regions in (('PVG', (0, 1, 2)), ('ALL', (3,))):
        i, s = [np.concatenate([np.log1p(values[:, :, r, field]) for r in regions], axis=1).astype(np.float32)
                for field in (0, 1)]
        if not np.isfinite(i).all() or not np.isfinite(s).all():
            raise ValueError('Nonfinite detector strength')
        result.update({support+'/I': i, support+'/S': s,
                       support+'/I+S': np.concatenate((i, s), axis=1)})
    return result


def collect(model):
    folder = OUT/model
    source = trainer.read(trainer.SOURCES['true_rms']/model/'matrices.pt')
    split = json.loads((trainer.SOURCES['true_rms']/model/'protocol.json').read_text())['split']
    extraction = json.loads((BASE/'protocol.json').read_text())
    protocol = dict(version=1, model=model, source_fingerprint=source['fingerprint'],
        extraction_signature=extraction['signature'], path='z-A_all -> z; true Norm+FFN; GL K32; local FP32',
        regions=REGIONS, fields=FIELDS, split=split, seeds=SEEDS, config=CONFIG,
        detectors='PVG concatenated and ALL summed; each log1p(I), log1p(S), [log1p(I),log1p(S)]',
        quantiles='Within each target-layer-region first; across-mention quantiles separately',
        zeros='I=S=0 for empty regions; undefined gain NaN; zero WRITE excluded only from lambda',
        statistics='All mentions in primary curves/detectors; conflicting targets excluded from image-paired controls',
        bootstrap='2000 paired image clusters; mean of seed metric differences; nominal percentile 95% CI',
        caveat='Repeatedly inspected 811 split; exploratory; no per-feature HPO, no AE, no residual source')
    protocol = json.loads(json.dumps(protocol))
    protocol['fingerprint'] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    path = folder/'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError('Protocol changed; refusing stale results')
    atomic_json_save(protocol, path)
    cached = folder/'statistics.pt'
    if cached.exists():
        data = trainer.read(cached)
        assert data['fingerprint'] == protocol['fingerprint']
        return data
    mentions = source['mentions']
    np.testing.assert_array_equal(source['y'], [m['label'] for m in mentions])
    ids = np.asarray([m['image_id'] for m in mentions])
    assert [len(split[k]) for k in ('train', 'validation', 'test')] == [3200, 400, 400]
    assert len(set().union(*map(set, split.values()))) == 4000
    for part in split:
        np.testing.assert_array_equal(source['masks'][part], np.isin(ids, split[part]))
    by_image = defaultdict(list)
    for m in mentions: by_image[m['image_id']].append(m)
    image_ids = sorted(set().union(*map(set, split.values())))
    targets, manifest = {}, []
    audit = dict(images=0, unique_targets=0, target_layers=0, zero_write=0,
                 empty_regions=0, closure_max=0., gross_relative_max=0.)
    layers = {'qwen2_5_vl_7b':28, 'llava_1_5_7b':32, 'qwen3_vl_8b':36, 'internvl_2_5_8b':32}[model]
    assert len(list((BASE/model/'shards').glob('image_*.pt'))) == 4000
    for index, image_id in enumerate(image_ids, 1):
        path = BASE/model/'shards'/f'image_{image_id:012d}.pt'
        shard = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        assert shard['complete'] and shard['image_id'] == image_id
        assert shard['protocol_signature'] == extraction['signature'] and shard['ks']['all'] == 32
        assert Counter(map(mention_identity, shard['sample_table'])) == Counter(map(mention_identity, by_image[image_id]))
        assert {r['target_key'] for r in shard['positions']} == {m['target_key'] for m in by_image[image_id]}
        manifest.append(dict(image=image_id, bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns))
        for row in shard['positions']:
            key = row['target_key']
            assert key not in targets and len(row['tokens']) == layers and len(row['diagnostics']) == layers
            matching = next(m for m in by_image[image_id] if m['target_key'] == key)
            assert all(row[k] == matching[k] for k in ('response_index', 'target_token_id'))
            values = []
            for layer, (token, diag) in enumerate(zip(row['tokens'], row['diagnostics'])):
                a, e = [token[k].double().numpy() for k in ('write_norm', 'effect_norm')]
                kind = token['source_type'].numpy()
                assert a.shape == e.shape == kind.shape and set(kind).issubset({0, 1, 2})
                assert diag['all_closure_relative'] <= .01
                audit['closure_max'] = max(audit['closure_max'], diag['all_closure_relative'])
                audit['zero_write'] += int((a == 0).sum())
                stats = []
                for r in range(4):
                    mask = kind == r if r < 3 else np.ones(len(a), dtype=bool)
                    audit['empty_regions'] += int(not mask.any())
                    stats.append(token_statistics(a[mask], e[mask]))
                    if r < 3:
                        saved = float(row['metrics'][REGIONS[r]+'_gross_norm'][layer])
                        np.testing.assert_allclose(stats[-1][1], saved, rtol=2e-6, atol=1e-7)
                        audit['gross_relative_max'] = max(audit['gross_relative_max'], abs(stats[-1][1]-saved)/saved if saved else 0.)
                np.testing.assert_allclose(np.asarray(stats)[:3, :2].sum(0), stats[3][:2], rtol=1e-12)
                values.append(stats)
                audit['target_layers'] += 1
            targets[key] = np.asarray(values)
        audit['images'] = index
        if index % 100 == 0:
            progress(model, '缓存统计', index, 4000)
            print(model, 'cached images', index, '/4000', flush=True)
    audit['unique_targets'] = len(targets)
    values = np.stack([targets[m['target_key']] for m in mentions])
    del targets
    # Independent agreement against the previously trained true-RMS visual S block.
    old_s = np.expm1(source['groups']['visual'][:, layers:])
    np.testing.assert_allclose(values[:, :, 1, 1], old_s, rtol=2e-6, atol=2e-6)
    data = dict(values=values, mentions=mentions, y=np.asarray(source['y']), masks=source['masks'],
                fingerprint=protocol['fingerprint'], audit=audit)
    atomic_json_save(manifest, folder/'source_manifest.json')
    atomic_json_save(audit, folder/'cache_audit.json')
    atomic_torch_save(data, cached)
    return data


def summarize_mechanism(model, data):
    values, y = data['values'], data['y']
    ids = np.asarray([m['image_id'] for m in data['mentions']])
    label_sets = defaultdict(set)
    for m in data['mentions']: label_sets[m['target_key']].add(m['label'])
    clean = np.asarray([len(label_sets[m['target_key']]) == 1 for m in data['mentions']])
    curves, paired, relation = [], [], []
    scopes = dict(all=np.ones(len(y), dtype=bool), **data['masks'])
    for scope, mask in scopes.items():
        for r, region in enumerate(REGIONS):
            for f, field in enumerate(FIELDS):
                matrix = values[:, :, r, f]
                for label, name in ((1, 'REAL'), (0, 'HALL')):
                    for layer, col in enumerate(matrix[mask & (y == label)].T, 1):
                        finite = col[np.isfinite(col)]
                        q = np.quantile(finite, [.25, .5, .75]) if len(finite) else [np.nan]*3
                        curves.append(dict(model=model, scope=scope, region=region, metric=field,
                            label=name, layer=layer, n=len(col), valid=len(finite), undefined=len(col)-len(finite),
                            mean=float(finite.mean()) if len(finite) else np.nan, q25=q[0], median=q[1], q75=q[2]))
                mixed = sorted(set(ids[mask & clean & (y == 0)]) & set(ids[mask & clean & (y == 1)]))
                deltas = [np.nanmean(matrix[mask & clean & (ids == image) & (y == 0)], axis=0)
                          -np.nanmean(matrix[mask & clean & (ids == image) & (y == 1)], axis=0) for image in mixed]
                if deltas:
                    for layer, col in enumerate(np.asarray(deltas).T, 1):
                        valid = col[np.isfinite(col)]
                        paired.append(dict(model=model, scope=scope, region=region, metric=field, layer=layer,
                            mixed_images=len(col), valid=len(valid), mean_H_minus_R=float(valid.mean()) if len(valid) else np.nan))
            for layer in range(values.shape[1]):
                x = values[mask, layer, r]
                valid = np.isfinite(x[:, 9:12]).all(1)
                np.testing.assert_allclose(x[valid, 10], x[valid, 9]+x[valid, 11], atol=1e-12)
                cls = y[mask][valid]
                d = x[valid, 9:12][cls == 0].mean(0)-x[valid, 9:12][cls == 1].mean(0)
                relation.append(dict(model=model, scope=scope, region=region, layer=layer+1,
                    spearman_I_S=float(spearmanr(x[:, 0], x[:, 1]).statistic),
                    log_pearson_I_S=float(np.corrcoef(x[valid, 9:11].T)[0, 1]),
                    delta_log_I=d[0], delta_log_S=d[1], delta_log_G=d[2], valid=int(valid.sum())))
    directory = OUT/model
    write_csv(directory/'curves.csv', curves)
    write_csv(directory/'paired_images.csv', paired)
    write_csv(directory/'write_gain_decomposition.csv', relation)
    for scope in ('all', 'test'):
        for kind, fields in (('ISG', ('I', 'S', 'G')), ('lambda', ('lambda_median', 'lambda_iqr', 'lambda_cv'))):
            fig, axes = plt.subplots(4, 3, figsize=(15, 13), squeeze=False)
            for r, region in enumerate(REGIONS):
                for j, metric in enumerate(fields):
                    ax = axes[r, j]
                    for label, color in (('REAL', '#2878b5'), ('HALL', '#d95319')):
                        rows = [v for v in curves if v['scope']==scope and v['region']==region and v['metric']==metric and v['label']==label]
                        layers = [v['layer'] for v in rows]
                        ax.plot(layers, [v['mean'] for v in rows], color=color, label=label+' mean')
                        ax.plot(layers, [v['median'] for v in rows], '--', color=color, alpha=.7)
                        ax.fill_between(layers, [v['q25'] for v in rows], [v['q75'] for v in rows], color=color, alpha=.12)
                    ax.set_title(region+' | '+metric)
                    ax.set_xlabel('Decoder layer (1-based)'); ax.grid(alpha=.2)
                    if kind == 'ISG': ax.set_yscale('symlog', linthresh=.01)
            axes[0, 0].legend(fontsize=8)
            fig.suptitle(f'{model} | {scope} | true-RMS all-attention K32\nSolid: mean; dashed: median; band: across-mention IQR (not CI). Lambda stats computed within each target first.')
            fig.tight_layout(rect=(0, 0, 1, .955))
            for suffix in ('png', 'pdf'): fig.savefig(directory/f'{kind}_{scope}.{suffix}', dpi=150)
            plt.close(fig)


def run_model(model, device):
    torch.set_num_threads(1)
    data = collect(model)
    if not (OUT/model/'lambda_test.pdf').exists(): summarize_mechanism(model, data)
    groups = feature_sets(data['values'])
    masks, y = data['masks'], data['y']
    rows, checks = [], []
    for group, x in groups.items():
        for seed in SEEDS:
            path = OUT/model/group/f'seed{seed}.pt'
            if path.exists():
                result = trainer.read(path)
                assert result['fingerprint'] == data['fingerprint'] and result['config'] == CONFIG
            else:
                result = trainer.fit(x[masks['train']], y[masks['train']], x[masks['validation']],
                    y[masks['validation']], CONFIG, seed, device,
                    callback=lambda epoch: progress(model, group+f' seed{seed}', len(rows), 18, epoch=epoch))
                network = trainer.SingleMLP(result['input_dim'], CONFIG).to(device)
                network.load_state_dict(result['state_dict'])
                p = trainer.predict(network, torch.as_tensor(trainer.transform(x[masks['test']], result['mean'], result['scale']), device=device))
                result.update(fingerprint=data['fingerprint'], test_probabilities=p, test=trainer.metrics(y[masks['test']], p))
                atomic_torch_save(result, path)
            # Every checkpoint independently reloaded on CPU, including train/val predictions.
            network = trainer.SingleMLP(x.shape[1], CONFIG)
            network.load_state_dict(result['state_dict'])
            mean, scale = trainer.scale_fit(x[masks['train']], True)
            np.testing.assert_array_equal(mean, result['mean']); np.testing.assert_array_equal(scale, result['scale'])
            assert result['best_epoch'] == 1+np.argmin([h['val_loss'] for h in result['history']])
            error = 0.
            for part in ('train', 'validation', 'test'):
                p = trainer.predict(network, torch.as_tensor(trainer.transform(x[masks[part]], mean, scale)))
                saved = result[part+'_probabilities']
                np.testing.assert_allclose(p, saved, rtol=1e-5, atol=1e-6)
                error = max(error, float(np.max(abs(p-saved))))
                if part != 'train':
                    for metric, value in trainer.metrics(y[masks[part]], saved).items():
                        np.testing.assert_allclose(value, result[part if part=='validation' else 'test'][metric], atol=1e-12)
            rows.append(dict(model=model, group=group, seed=seed, input_dim=x.shape[1], best_epoch=result['best_epoch'],
                             epochs=result['epochs'], **result['test']))
            checks.append(dict(group=group, seed=seed, max_probability_error=error, status='PASS'))
            write_csv(OUT/model/'detection.csv', rows)
            atomic_json_save(checks, OUT/model/'validation.json')
            print(model, group, seed, result['test'], flush=True)
            progress(model, '检测', len(rows), 18)
    progress(model, '完成', 18, 18, status='completed')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    for model in args.models: run_model(model, args.device)


if __name__ == '__main__': main()
