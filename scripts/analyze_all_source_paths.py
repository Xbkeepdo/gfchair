"""Fixed all-source mechanism summaries, 15 probe families, and paired bootstrap."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODELS = ('qwen2_5_vl_7b', 'llava_1_5_7b', 'qwen3_vl_8b', 'internvl_2_5_8b')
SEEDS = (43, 44, 45)
LAYERS = dict(zip(MODELS, (28, 32, 36, 32)))
SOURCES = ('prompt', 'visual', 'generation')
F1_FIELDS = tuple(s+'_'+k for s in SOURCES for k in ('gross_norm', 'net_norm', 'group_gain'))
F2_FIELDS = (tuple(s+'_group_rotation' for s in SOURCES)
             + tuple('cos_'+p+'_out' for p in ('vg', 'vp', 'pg'))
             + tuple('delta_cos_'+p for p in ('vg', 'vp', 'pg'))
             + ('visual_context_cos', 'visual_context_relative_change')
             + tuple(s+'_within_source_kappa' for s in SOURCES) + ('source_group_kappa',))
F3_FIELDS = ('residual_norm', 'prompt_norm', 'visual_norm', 'generation_norm', 'residual_share')
F4_FIELDS = ('cos_residual_visual', 'cos_residual_prompt', 'cos_residual_generation',
             'residual_visual_generation_balance', 'attention_residual_kappa',
             'delta_residual_visual_cos', 'delta_residual_generation_cos')


def read(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def atomic(path, value, binary=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    os.close(descriptor)
    try:
        if binary:
            torch.save(value, name)
        else:
            Path(name).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=lambda x: x.item())+'\n')
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('')
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def old_root(model):
    return ROOT/'outputs/ffn_source_composition_v1'/model


def formal_split(model):
    value = json.loads((ROOT/'outputs'/model/'COCO4000-INSLEN-OFFICIAL-TARGET/image_splits.json').read_text())
    train_ids, test_ids = set(map(int, value['train'])), set(map(int, value['test']))
    if len(train_ids) != 3200 or len(test_ids) != 800 or train_ids & test_ids:
        raise ValueError('Expected unchanged 3200/800 image split')
    return train_ids, test_ids


def mention_identity(row):
    return tuple(row[k] for k in ('mention_id', 'target_key', 'image_id', 'response_index', 'target_token_id', 'label'))


def reference(model):
    value = read(old_root(model)/'matrices.pt')
    mentions, y, n = value['mentions'], np.asarray(value['y']), int(value['ntrain'])
    train_ids, test_ids = formal_split(model)
    if not 0 < n < len(mentions) or len({m['mention_id'] for m in mentions}) != len(mentions):
        raise ValueError('Invalid canonical mention order')
    if not all(int(m['image_id']) in train_ids for m in mentions[:n]) or not all(int(m['image_id']) in test_ids for m in mentions[n:]):
        raise ValueError('Canonical mentions do not follow original image split')
    np.testing.assert_array_equal(y, [m['label'] for m in mentions])
    if not set(y).issubset({0, 1}):
        raise ValueError('Unknown label')
    return dict(mentions=mentions, y=y, ntrain=n)


def aligned(other, ref):
    if int(other['ntrain']) != ref['ntrain']:
        raise ValueError('Baseline split mismatch')
    np.testing.assert_array_equal(other['y'], ref['y'])
    if [mention_identity(m) for m in other['mentions']] != [mention_identity(m) for m in ref['mentions']]:
        raise ValueError('Baseline mention order mismatch')


def collect(model, out, image_ids=None, progress_callback=None):
    ref = reference(model)
    train_ids, test_ids = formal_split(model)
    expected = train_ids | test_ids if image_ids is None else set(map(int, image_ids))
    if not expected <= train_ids | test_ids:
        raise ValueError('Unknown requested images')
    folder = Path(out)/model/'shards'
    if image_ids is None:
        files = list(folder.glob('image_*.pt'))
        if len(files) != 4000:
            raise ValueError(f'Incomplete formal extraction: {len(files)}/4000 image shards')
    protocol_path = Path(out)/'protocol.json'
    expected_signature = json.loads(protocol_path.read_text())['signature'] if protocol_path.exists() else None
    if image_ids is None and expected_signature is None:
        raise ValueError('Missing frozen extraction protocol')
    mentions = [m for m in ref['mentions'] if int(m['image_id']) in expected]
    by_image = defaultdict(list)
    for m in mentions:
        by_image[int(m['image_id'])].append(m)
    targets, signature, names, layers = {}, None, None, None
    for index, image_id in enumerate(sorted(expected), 1):
        path = folder/f'image_{image_id:012d}.pt'
        shard = read(path)
        if shard.get('complete') is not True or int(shard['image_id']) != image_id:
            raise ValueError(f'Incomplete/misidentified image shard {path}')
        current_signature = shard.get('protocol_signature')
        if not isinstance(current_signature, str) or not current_signature:
            raise ValueError(f'Missing protocol signature in {path}')
        signature = current_signature if signature is None else signature
        if current_signature != signature:
            raise ValueError('Mixed extraction protocol signatures')
        if expected_signature is not None and signature != expected_signature:
            raise ValueError('Shard disagrees with frozen protocol')
        if Counter(mention_identity(m) for m in shard['sample_table']) != Counter(mention_identity(m) for m in by_image[image_id]):
            raise ValueError(f'Missing or changed mentions for image {image_id}')
        required = {m['target_key'] for m in by_image[image_id]}
        if {r['target_key'] for r in shard['positions']} != required:
            raise ValueError('Target coverage mismatch')
        for row in shard['positions']:
            key = row['target_key']
            if key in targets:
                raise ValueError('Duplicate target')
            matching = next(m for m in by_image[image_id] if m['target_key'] == key)
            if any(row[k] != matching[k] for k in ('response_index', 'target_token_id')):
                raise ValueError('Target metadata mismatch')
            metric = {k: np.asarray(v, dtype=np.float64) for k, v in row['metrics'].items()}
            names = set(metric) if names is None else names
            if set(metric) != names or not metric:
                raise ValueError('Inconsistent metric schema')
            for arr in metric.values():
                layers = len(arr) if layers is None else layers
                if arr.shape != (layers,) or np.isinf(arr).any():
                    raise ValueError('Invalid layer dimension or infinite feature')
            if model in LAYERS and (layers != LAYERS[model] or len(row.get('diagnostics', [])) != layers
                                    or len(row.get('tokens', [])) != layers):
                raise ValueError('Incomplete decoder layer coverage')
            targets[key] = metric
        if progress_callback:
            progress_callback('prepare', index, len(expected))
    arrays = {name: np.stack([targets[m['target_key']][name] for m in mentions]) for name in sorted(names or [])}
    return arrays, mentions, ref, signature


def feature_groups(arrays, f0, mentions):
    """Exactly 15 frozen groups; log1p applies only to nonnegative strengths."""
    def concatenate(fields, log_fields=()):
        parts = []
        for name in fields:
            a = np.asarray(arrays[name], dtype=np.float64)
            if np.isinf(a).any():
                raise ValueError('Infinite detector feature '+name)
            if name in log_fields:
                if np.nanmin(a, initial=0) < 0:
                    raise ValueError('Negative norm '+name)
                a = np.log1p(a)
            parts.append(np.nan_to_num(a, nan=0.0).astype(np.float32))
        return np.concatenate(parts, axis=1)
    f0 = np.asarray(f0, dtype=np.float32)
    if np.isinf(f0).any():
        raise ValueError('Infinite F0')
    groups = dict(F0=np.nan_to_num(f0, nan=0.0), F1_raw=concatenate(F1_FIELDS),
                  F1_log1p=concatenate(F1_FIELDS, [s+'_'+k for s in SOURCES for k in ('gross_norm', 'net_norm')]),
                  F2=concatenate(F2_FIELDS))
    for branch in ('b1', 'b2'):
        fields = [branch+'_'+k for k in F3_FIELDS]
        groups[branch+'_F3_raw'] = concatenate(fields)
        groups[branch+'_F3_log1p'] = concatenate(fields, fields[:4])
        groups[branch+'_F4'] = concatenate([branch+'_'+k for k in F4_FIELDS])
        groups[branch+'_F5'] = np.concatenate([groups['F2'], groups[branch+'_F4']], axis=1)
        groups[branch+'_F6'] = np.concatenate([groups['F0'], groups[branch+'_F5']], axis=1)
    position = np.asarray([m['response_index'] for m in mentions], dtype=np.float32)
    groups['length'] = np.stack([position, position], axis=1)
    if len(groups) != 15 or any(x.shape[0] != len(mentions) or not np.isfinite(x).all() for x in groups.values()):
        raise ValueError('Invalid detector matrices')
    return groups


def fingerprint(groups, ref, signature):
    digest = hashlib.sha256(signature.encode())
    digest.update(json.dumps([mention_identity(m) for m in ref['mentions']]).encode())
    digest.update(str(ref['ntrain']).encode())
    for name, x in sorted(groups.items()):
        digest.update(name.encode())
        digest.update(str(x.shape).encode())
        digest.update(x.tobytes())
    return digest.hexdigest()


def mechanism(model, out):
    from scripts.analyze_ffn_input_geometry import summarize as describe
    cohort = json.loads((ROOT/'outputs/ffn_output_cosine_20260909/cohort500.json').read_text())['split']
    if len(cohort['train']) != 400 or len(cohort['test']) != 100:
        raise ValueError('Expected fixed 400/100 mechanism cohort')
    arrays, mentions, _, signature = collect(model, out, cohort['train']+cohort['test'])
    labels = defaultdict(set)
    for m in mentions:
        labels[m['target_key']].add(m['label'])
    keep = np.asarray([len(labels[m['target_key']]) == 1 for m in mentions])
    train_ids = set(cohort['train'])
    filtered = [dict(m, label_conflict=False, split='train' if m['image_id'] in train_ids else 'test')
                for m, valid in zip(mentions, keep) if valid]
    arrays = {k: v[keep] for k, v in arrays.items()}
    directory = Path(out)/model/'mechanism'
    directory.mkdir(parents=True, exist_ok=True)
    rows, paired = describe(model, arrays, filtered, directory)
    atomic(directory/'audit.json', dict(protocol_signature=signature, images=500,
           excluded_conflicting_mentions=int((~keep).sum()), retained_mentions=len(filtered),
           excluded_conflicting_targets=sum(len(v)>1 for v in labels.values()),
           statistics='Mention weighted; all primary and paired curves exclude conflicting targets; descriptive only'))
    plot_mechanism(rows, directory, model)
    (directory/'summary.md').write_text(f'# {model} 来源路径机制分析\n\n固定共享500图（400/100）；保留{len(filtered)}个mention，排除{int((~keep).sum())}个标签冲突mention。主曲线和同图配对均排除冲突；均值、中位数、IQR、all/train/test分别报告；未据此选择检测特征。B1与B2分别呈现。\n')
    return rows, paired


def plot_mechanism(rows, directory, model):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = sorted({r['metric'] for r in rows})
    for scope in ('all', 'train', 'test'):
        for page, begin in enumerate(range(0, len(names), 12), 1):
            fig, axes = plt.subplots(3, 4, figsize=(18, 11), squeeze=False)
            for ax, metric in zip(axes.flat, names[begin:begin+12]):
                for label, color in (('REAL', '#2166ac'), ('HALL', '#b2182b')):
                    selected = [r for r in rows if r['scope'] == scope and r['metric'] == metric and r['label'] == label]
                    x = [r['layer'] for r in selected]
                    ax.plot(x, [r['mean'] for r in selected], color=color, label=label+' mean')
                    ax.plot(x, [r['median'] for r in selected], '--', color=color, alpha=.7)
                    ax.fill_between(x, [r['q25'] for r in selected], [r['q75'] for r in selected], color=color, alpha=.12)
                ax.set_title(metric, fontsize=8)
                ax.set_xlabel('Decoder layer')
                ax.grid(alpha=.2)
            for ax in list(axes.flat)[len(names[begin:begin+12]):]:
                ax.axis('off')
            axes.flat[0].legend(fontsize=7)
            fig.suptitle(f'{model} | {scope} | conflict-excluded mentions; mean / median / IQR')
            fig.tight_layout()
            for extension in ('png', 'pdf'):
                fig.savefig(directory/f'{scope}_{page:02d}.{extension}', dpi=130)
            plt.close(fig)


def validate_head(value, ref, seed):
    if value['seed'] != seed:
        raise ValueError('Wrong saved head seed')
    for part, n in (('train', ref['ntrain']), ('test', len(ref['y'])-ref['ntrain'])):
        p = np.asarray(value[part+'_probabilities'])
        if p.shape != (n,) or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
            raise ValueError('Invalid saved probabilities')


def train(model, device, out, progress_callback=None):
    from scripts.train_torch_probe_feature_sets import TorchProbeConfig, train_and_evaluate_probe
    from scripts.train_ffn_ae_log1p_search import summarize_group
    arrays, mentions, ref, signature = collect(model, out, progress_callback=progress_callback)
    baseline_root = ROOT/'outputs/prefix_attention_gate/full/group_detection'/model
    baseline = read(baseline_root/'matrices.pt')
    aligned(baseline, ref)
    groups = feature_groups(arrays, baseline['groups']['gated_raw_concat'], mentions)
    identity = fingerprint(groups, ref, signature)
    directory = Path(out)/model
    matrix = dict(ref, groups=groups, protocol_signature=signature, fingerprint=identity)
    path = directory/'matrices.pt'
    if path.exists():
        previous = read(path)
        if previous.get('fingerprint') != identity:
            raise ValueError('Detector matrix changed during resume')
    else:
        atomic(path, matrix, binary=True)
    results, completed, total = {}, 0, len(groups)*len(SEEDS)
    n, y = ref['ntrain'], ref['y']
    for name, x in groups.items():
        data = dict(X_train=x[:n], y_train=y[:n], X_test=x[n:], y_test=y[n:])
        heads = []
        for seed in SEEDS:
            folder = directory/'heads'/name/f'seed{seed}'
            path = folder/'result.pt'
            if progress_callback:
                progress_callback('train:'+name, completed, total)
            if path.exists():
                value = read(path)
                if value.get('fingerprint') != identity:
                    raise ValueError('Stale saved detector head')
            elif name == 'F0':
                value = read(baseline_root/'heads/gated_raw_concat'/f'seed{seed}'/'result.pt')
                validate_head(value, ref, seed)
                value = dict(value, fingerprint=identity, reused_from=str(baseline_root))
                atomic(path, value, binary=True)
            else:
                def epoch_callback(epoch):
                    if progress_callback:
                        progress_callback('train:'+name, completed, total, epoch=epoch)
                metrics = train_and_evaluate_probe(**data, X_val=np.empty((0, x.shape[1]), np.float32),
                    y_val=np.empty(0, np.int32), config=TorchProbeConfig(seed=seed), device=torch.device(device),
                    output_dir=str(folder), return_probabilities=True, epoch_callback=epoch_callback)
                value = dict(seed=seed, fingerprint=identity,
                    train_probabilities=np.asarray(metrics.pop('train_probabilities')),
                    test_probabilities=np.asarray(metrics.pop('test_probabilities')), metrics=metrics)
                validate_head(value, ref, seed)
                atomic(path, value, binary=True)
            validate_head(value, ref, seed)
            heads.append(value)
            completed += 1
            if progress_callback:
                progress_callback('train:'+name, completed, total)
        results[name] = summarize_group(data, heads)
        atomic(directory/'detection.json', results)
    return results


def weighted_rank_metrics(y, probabilities, weights):
    """Weighted REAL AUROC and HALL AP, with exact tie groups; [B,N] weights."""
    y, p, w = np.asarray(y), np.asarray(probabilities), np.atleast_2d(weights).astype(np.float64)
    if y.shape != p.shape or w.shape[1] != len(y) or not np.isfinite(p).all():
        raise ValueError('Invalid weighted metric input')
    order = np.argsort(p, kind='stable')
    ends = np.r_[np.flatnonzero(np.diff(p[order]))+1, len(p)]
    starts = np.r_[0, ends[:-1]]
    positive = np.add.reduceat(w[:, order]*y[order], starts, axis=1)
    negative = np.add.reduceat(w[:, order]*(1-y[order]), starts, axis=1)
    total_p, total_n = positive.sum(1), negative.sum(1)
    with np.errstate(divide='ignore', invalid='ignore'):
        auc = (positive*(np.cumsum(negative, axis=1)-.5*negative)).sum(1)/(total_p*total_n)
        # Ascending REAL probability is descending HALL probability.
        precision = np.divide(np.cumsum(negative, axis=1), np.cumsum(positive+negative, axis=1),
                              out=np.zeros_like(negative), where=np.cumsum(positive+negative, axis=1)>0)
        ap = (negative*precision).sum(1)/total_n
    result = np.stack([auc, ap], axis=1)
    result[(total_p == 0)|(total_n == 0)] = np.nan
    return result


def comparison_pairs():
    pairs = [(branch+'_'+family, baseline) for branch in ('b1', 'b2') for family in ('F5', 'F6')
             for baseline in ('F0', 'source_raw', 'source_log1p', 'length')]
    pairs += [('b1_'+family, 'b2_'+family) for family in ('F3_raw', 'F3_log1p', 'F4', 'F5', 'F6')]
    return pairs


def bootstrap(model, out):
    directory = Path(out)/model
    matrix = read(directory/'matrices.pt')
    ref = reference(model)
    aligned(matrix, ref)
    names = list(matrix['groups'])+['source_raw', 'source_log1p']
    n, y = ref['ntrain'], np.asarray(ref['y'][ref['ntrain']:])
    strength_root = ROOT/'outputs/ffn_source_composition_v1/source_strength_detection'/model
    aligned(read(strength_root/'matrices.pt'), ref)
    probabilities = {}
    for name in names:
        probabilities[name] = []
        for seed in SEEDS:
            if name.startswith('source_'):
                source = 'raw_concat' if name == 'source_raw' else 'log1p_concat'
                path = strength_root/'heads'/source/f'seed{seed}'/'result.pt'
            else:
                path = directory/'heads'/name/f'seed{seed}'/'result.pt'
            value = read(path)
            validate_head(value, ref, seed)
            if not name.startswith('source_') and value.get('fingerprint') != matrix['fingerprint']:
                raise ValueError('Bootstrap head fingerprint mismatch')
            probabilities[name].append(np.asarray(value['test_probabilities']))
    digest = hashlib.sha256(matrix['fingerprint'].encode())
    for values in probabilities.values():
        for value in values:
            digest.update(value.tobytes())
    identity = digest.hexdigest()
    destination = directory/'bootstrap.json'
    if destination.exists():
        result = json.loads(destination.read_text())
        if result.get('fingerprint') != identity:
            raise ValueError('Bootstrap inputs changed')
        return result
    _, test_ids = formal_split(model)
    images = sorted(test_ids)
    image_index = {image: i for i, image in enumerate(images)}
    mention_images = np.asarray([image_index[int(m['image_id'])] for m in ref['mentions'][n:]])
    rng = np.random.default_rng(20260912)
    repetitions = 10000
    distributions = {name: np.empty((repetitions, 2)) for name in names}
    points = {name: np.mean([weighted_rank_metrics(y, p, np.ones(len(y)))[0] for p in values], axis=0)
              for name, values in probabilities.items()}
    for begin in range(0, repetitions, 128):
        size = min(128, repetitions-begin)
        draws = rng.integers(0, len(images), size=(size, len(images)))
        counts = np.zeros((size, len(images)), dtype=np.int32)
        np.add.at(counts, (np.repeat(np.arange(size), len(images)), draws.ravel()), 1)
        weights = counts[:, mention_images]
        for name, values in probabilities.items():
            distributions[name][begin:begin+size] = np.mean([weighted_rank_metrics(y, p, weights) for p in values], axis=0)
        atomic(directory/'bootstrap_progress.json', dict(completed=begin+size, total=repetitions))
    rows = []
    for left, right in comparison_pairs():
        for metric_index, metric in enumerate(('AUROC', 'HALL_AUPR')):
            delta = distributions[left][:, metric_index]-distributions[right][:, metric_index]
            delta = delta[np.isfinite(delta)]
            if not len(delta):
                raise ValueError('No two-class bootstrap draws')
            low, high = np.quantile(delta, [.025, .975])
            rows.append(dict(model=model, left=left, right=right, metric=metric,
                delta=float(points[left][metric_index]-points[right][metric_index]),
                ci95_low=float(low), ci95_high=float(high), valid_repetitions=len(delta)))
    result = dict(fingerprint=identity, repetitions=repetitions, seed=20260912,
        unit='original 800 test images; duplicate images repeat all their mentions',
        estimator='mean of 3 seed metrics, never metric of mean probabilities', comparisons=rows)
    atomic(destination, result)
    write_csv(directory/'bootstrap.csv', rows)
    return result


def summarize(out):
    out = Path(out)
    scores, comparisons, curves, paired, seed_rows = [], [], [], [], []
    for model in MODELS:
        directory = out/model
        results = json.loads((directory/'detection.json').read_text())
        if len(results) != 15:
            raise ValueError('Incomplete detector families')
        for name, value in results.items():
            for metric in ('auc', 'hall_aupr'):
                stats = value['seed_mean_std'][metric]
                scores.append(dict(model=model, family=name, metric=metric, mean=stats['mean'], std=stats['std']))
            if set(value['per_seed_metrics']) != set(map(str, SEEDS)):
                raise ValueError('Incomplete detector seeds')
            for seed, metrics in value['per_seed_metrics'].items():
                report = metrics['threshold_reports']['fixed_0.5']['test_metrics']
                seed_rows.append(dict(model=model, family=name, seed=int(seed), auc=report['auc'],
                                      hall_aupr=report['hallucination_positive']['aupr']))
        comparisons.extend(json.loads((directory/'bootstrap.json').read_text())['comparisons'])
        for name, destination in (('curves.csv', curves), ('paired_images.csv', paired)):
            with (directory/'mechanism'/name).open() as stream:
                destination.extend(csv.DictReader(stream))
    write_csv(out/'detection.csv', scores)
    write_csv(out/'seed_metrics.csv', seed_rows)
    write_csv(out/'bootstrap.csv', comparisons)
    write_csv(out/'mechanism_curves.csv', curves)
    write_csv(out/'mechanism_paired_images.csv', paired)
    lines = ['# 全注意力路径与残差路径实验', '',
             '四模型各15组、seeds 43/44/45；报告逐seed指标的均值±总体标准差，不用ensemble替代。',
             '固定4000图、3200/800划分和全部mentions；NaN置0、无标准化；norm强度分别报告raw/log1p，几何量不取log。',
             'B1是真实RMSNorm积分路径，B2是冻结端点RMS缩放后的纯FFN分解，二者单列。',
             '机制阶段共享500图（400/100），主曲线与同图配对均排除标签冲突目标；机制只作描述，不用于挑选特征。',
             '长度控制为[N_G,response_index]；当前teacher-forced定义下N_G=response_index，两列相同。',
             '成对bootstrap固定重采样原800测试图片10000次，重复图片保留全部mentions；比较的是3个seed指标差的平均。',
             'F0复用已对齐attention×gate基线，旧raw/log1p来源强度基线用于固定比较。', '',
             '| 模型 | 特征族 | AUROC | HALL-AUPR |', '|---|---|---:|---:|']
    for model in MODELS:
        for family in dict.fromkeys(r['family'] for r in scores if r['model'] == model):
            values = [next(r for r in scores if r['model']==model and r['family']==family and r['metric']==metric)
                      for metric in ('auc', 'hall_aupr')]
            lines.append(f'| {model} | {family} | '+' | '.join(f"{100*r['mean']:.2f} ± {100*r['std']:.2f}" for r in values)+' |')
    lines += ['', '指标单位为百分比；配对差值及95%区间见bootstrap.csv，机制均值/中位数/IQR与同图配对见mechanism_*.csv及各模型mechanism目录的PNG/PDF。']
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
    return scores, comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('mechanism', 'train', 'bootstrap', 'summarize'), required=True)
    parser.add_argument('--model', choices=MODELS)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--out', type=Path, default=ROOT/'outputs/ffn_all_source_paths_v1')
    parser.add_argument('--wait', action='store_true', help='For summarize, wait for all four bootstrap outputs')
    args = parser.parse_args()
    if args.stage == 'summarize':
        if args.wait:
            while not all((args.out/model/'bootstrap.json').exists() for model in MODELS):
                time.sleep(10)
        summarize(args.out)
    elif not args.model:
        parser.error('--model is required for this stage')
    elif args.stage == 'train':
        train(args.model, args.device, args.out)
    else:
        globals()[args.stage](args.model, args.out)


if __name__ == '__main__':
    main()
