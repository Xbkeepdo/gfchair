"""Cached raw-attention/evidence Top-32 overlap: standalone and added to B."""
import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_ffn_source_composition as base
from scripts.train_torch_probe_feature_sets import TorchProbeConfig

OUT = base.OUT / 'raw_ae_top32_detection'
SEEDS = (43, 44, 45)
TITLES = ('Qwen2.5', 'LLaVA', 'Qwen3', 'InternVL')


def top32_overlap(raw, evidence):
    # Match the existing curve's NumPy argsort, including its tie handling.
    p = base.normalized_mass(raw).numpy()
    q = evidence.numpy()
    k = min(32, p.shape[-1])
    ip = np.argsort(-p, axis=-1)[:, :k]
    iq = np.argsort(-q, axis=-1)[:, :k]
    return (ip[:, :, None] == iq[:, None, :]).any(-1).mean(-1).astype(np.float32)


def prepare(model):
    root = OUT / model
    if (root / 'matrices.pt').exists():
        return
    original = base.read(base.OUT / model / 'matrices.pt')
    mentions, n = original['mentions'], original['ntrain']
    files = base.feature_files(model)
    assert len(files) == 4000
    scores = {}
    for i, path in enumerate(files, 1):
        for row in base.read(path)['positions']:
            scores[row['target_key']] = top32_overlap(row['raw_attention_mean'], row['attention_evidence'])
        if i % 500 == 0:
            print('CACHE', model, i, flush=True)
    x = np.stack([scores[m['target_key']] for m in mentions])
    y = np.asarray(original['y'])
    splits = json.loads((ROOT / 'outputs' / model / base.EXPERIMENT / 'image_splits.json').read_text())
    train, test = set(splits['train']), set(splits['test'])
    assert len(train) == 3200 and len(test) == 800 and not train & test
    assert all(m['image_id'] in train for m in mentions[:n])
    assert all(m['image_id'] in test for m in mentions[n:])
    assert np.array_equal(y, [m['label'] for m in mentions])
    assert np.isfinite(x).all() and (x >= 0).all() and (x <= 1).all()
    # A direct consistency check against the previously displayed curves.
    checks = []
    with (base.OUT / model / 'curves.csv').open() as f:
        for r in csv.DictReader(f):
            if r['signal'] == 'raw_AE_top32':
                values = x[y == int(r['label']), int(r['layer']) - 1].astype(np.float64)
                checks.append(abs(values.mean() - float(r['mean'])))
    assert len(checks) == 2 * x.shape[1] and max(checks) < 1e-6
    groups = dict(raw_AE_top32=x, B_raw_AE_top32=np.concatenate([original['groups']['B'], x], axis=1))
    base.save(dict(groups=groups, y=y, ntrain=n, mentions=mentions), root / 'matrices.pt')
    base.json_save(dict(model=model, images=4000, train_images=3200, test_images=800,
        train_mentions=n, test_mentions=len(y)-n, seeds=SEEDS,
        features={'raw_AE_top32': 'all-layer |Top-k(normalized raw attention) intersect Top-k(T)|/k; k=min(32,visual tokens)',
                  'B_raw_AE_top32': '[AE, log1p(I), log1p(S_E), raw_AE_top32], each block all layers'},
        dimensions={g: a.shape[1] for g, a in groups.items()}, config=asdict(TorchProbeConfig()),
        scaling='No log or standardization of overlap; B reused exactly',
        curve_mean_max_difference=max(checks), baseline='Existing composition B; no baseline retraining',
        scope='Original 3200/800 exploratory comparison, not the later 3200/400/400 protocol'), root / 'protocol.json')


def summarize():
    rows, seed_rows, comparisons = [], [], []
    lines = ['# raw_AE_top32 幻觉检测实验', '',
        '四模型原4000图，3200训练/800测试及全部mentions；本次沿用此前B/JS/OT的82对照，不使用后来的811划分。',
        'O_l=|Top-k(P_raw)∩Top-k(T)|/k，k=min(32,视觉token数)。P_raw为视觉内归一化原始attention，T为既有gate加权证据分布。',
        '单独O为L维；B+O为4L维，B=[AE,log1p(I),log1p(S_E)]。O不取log、不做标准化。沿用旧曲线的NumPy排序及并列值处理。',
        '原MLP隐藏层128/64/32、BN、dropout .3、Adam lr .001/wd 1e-5、batch256、最多100epoch；按训练loss调度/早停/选择checkpoint，无调参。',
        '每格为seeds43/44/45均值±总体标准差，单位%。HALL-F1按各seed训练集REAL-F1阈值。B直接引用已有三头。', '',
        '| 模型 | 特征 | AUROC | HALL-AUPR | HALL-F1 |', '|---|---|---:|---:|---:|']
    for model, title in zip(base.MODELS, TITLES):
        new = json.loads((OUT / model / 'detection.json').read_text())
        old = json.loads((base.OUT / model / 'detection.json').read_text())
        groups = dict(B=old['B'], **new)
        values_by_group = {}
        for group, result in groups.items():
            values = {k: [] for k in ('AUROC', 'HALL_AUPR', 'HALL_F1')}
            for seed in SEEDS:
                reports = result['per_seed_metrics'][str(seed)]['threshold_reports']
                for rule, report in reports.items():
                    m = report['test_metrics']; h = m['hallucination_positive']
                    seed_rows.append(dict(model=model, group=group, seed=seed, threshold_rule=rule,
                        threshold=report['threshold'], AUROC=m['auc'], HALL_AUPR=h['aupr'],
                        HALL_precision=h['precision'], HALL_recall=h['recall'], HALL_F1=h['f1']))
                    if rule == 'train_f1':
                        for name, value in zip(values, (m['auc'], h['aupr'], h['f1'])):
                            values[name].append(value)
            values_by_group[group] = values
            row = dict(model=model, group=group)
            cells = []
            for metric, v in values.items():
                row[metric+'_mean'], row[metric+'_std'] = float(np.mean(v)), float(np.std(v))
                cells.append(f'{100*np.mean(v):.2f} ± {100*np.std(v):.2f}')
            rows.append(row)
            lines.append(f'| {title} | {group} | ' + ' | '.join(cells) + ' |')
        for seed_i, seed in enumerate(SEEDS):
            comparisons.append(dict(model=model, seed=seed, **{k+'_delta':
                values_by_group['B_raw_AE_top32'][k][seed_i] - values_by_group['B'][k][seed_i]
                for k in values_by_group['B']}))
    lines += ['', 'B+O相对B的均值差，单位百分点：', '',
              '| 模型 | ΔAUROC | ΔHALL-AUPR | ΔHALL-F1 |', '|---|---:|---:|---:|']
    for model, title in zip(base.MODELS, TITLES):
        subset = [r for r in comparisons if r['model'] == model]
        lines.append(f'| {title} | ' + ' | '.join(f"{100*np.mean([r[k+'_delta'] for r in subset]):+.2f}"
            for k in ('AUROC', 'HALL_AUPR', 'HALL_F1')) + ' |')
    lines += ['', '解释：本信号只衡量gate前后最高权重位置的一致性，不包含e_m的空间分布。拼接实验检验其在B之外的增量。',
        'Top32是离散排序信号，忽略权重差和空间距离；并列权重时继承旧代码的排序选择。曲线分离不能替代测试指标。',
        '800测试图曾用于多轮探索；结果不作显著性或独立数据泛化声明。与811结果不能直接横比。',
        '缓存提取的逐层类别均值已与旧raw_AE_top32曲线核对；保存精简矩阵、协议、24个新头及逐seed/双阈值CSV。无模型前向或路径积分重算。', '',
        '运行：`python scripts/train_raw_ae_top32.py --models <模型列表> --device cuda:0`；汇总：`python scripts/train_raw_ae_top32.py --summarize`。']
    base.write_csv(rows, OUT / 'detection.csv')
    base.write_csv(seed_rows, OUT / 'seed_metrics.csv')
    base.write_csv(comparisons, OUT / 'paired_deltas.csv')
    (OUT / 'summary.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', choices=base.MODELS, default=base.MODELS)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--summarize', action='store_true')
    args = parser.parse_args()
    if args.summarize:
        summarize()
    else:
        for model in args.models:
            prepare(model)
            base.train(model, args.device, root=OUT / model)
