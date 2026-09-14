"""Measure split+training seed sensitivity for true-RMS V and VP+G on four models."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from features.tc_fvpa_artifacts import atomic_json_save, atomic_torch_save
from scripts import search_single_mlp_811 as mlp

MODELS = tuple(mlp.MODELS)
GROUPS = ('visual', 'vp_generation')
SEEDS = (43, 44, 45)
OUT = ROOT/'outputs/joint_split_training_seed_811_v1'
SOURCE = ROOT/'outputs/all_attention_ae_811_v1'
SEARCH = ROOT/'outputs/single_mlp_search_811_v1'


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def image_split(model, seed):
    source = ROOT/'outputs'/model/'COCO4000-INSLEN-OFFICIAL-TARGET'/'image_splits.json'
    frozen = json.loads(source.read_text())
    image_ids = sorted({int(value) for name in ('train','test') for value in frozen[name]})
    if len(image_ids) != 4000:
        raise ValueError(f'Expected 4000 unique images, got {len(image_ids)}')
    values = np.random.default_rng(seed).permutation(image_ids)
    return dict(train=values[:3200].tolist(), validation=values[3200:3600].tolist(),
                test=values[3600:].tolist())


def masks_for(mentions, split):
    ids = np.asarray([int(row['image_id']) for row in mentions])
    masks = {name: np.isin(ids, values) for name, values in split.items()}
    if not np.all(sum(value.astype(np.int8) for value in masks.values()) == 1):
        raise ValueError('Split masks are not exhaustive and exclusive')
    return masks


def protocol():
    settings = {}
    source_fingerprints = {}
    mention_signatures = {}
    image_split_hashes = {}
    for model in MODELS:
        selection = json.loads((SEARCH/model/'selection.json').read_text())
        settings[model] = {group: selection['variants']['true_rms/'+group]['config'] for group in GROUPS}
        data = mlp.read(SOURCE/model/'matrices.pt')
        source_fingerprints[model] = data['fingerprint']
        mention_signatures[model] = hashlib.sha256(json.dumps(
            [(m['image_id'], m['target_key'], m['label']) for m in data['mentions']]).encode()).hexdigest()
        split_path = ROOT/'outputs'/model/'COCO4000-INSLEN-OFFICIAL-TARGET'/'image_splits.json'
        image_split_hashes[model] = hashlib.sha256(split_path.read_bytes()).hexdigest()
    value = dict(schema='joint-split-training-seed-811-v1', models=list(MODELS), groups=list(GROUPS),
        seeds=list(SEEDS), split='For each seed, permute all 4000 sorted image IDs into 3200/400/400',
        seed_scope='Same seed controls image split, model initialization, minibatch order and dropout',
        path='true RMS all-attention z-A_all to z; local FP32 Gauss-Legendre K32',
        features=dict(visual='[AE_V,log1p(S_V_all)]',
            vp_generation='[AE_VP,log1p(S_V_all+S_P_all),AE_G,log1p(S_G_all)]'),
        classifier='Previously validation-selected single-hidden MLP config frozen per model/group; BCE REAL=1; Adam; train-only scaling; validation early stopping; no refit',
        comparison='Existing fixed split20260912 with training seeds43/44/45 versus joint split+training seeds43/44/45',
        metrics='Per-seed test AUROC and HALL-positive AUPR; arithmetic mean and population std; no probability ensemble',
        settings=settings, source_fingerprints=source_fingerprints, mention_signatures=mention_signatures,
        image_split_hashes=image_split_hashes,
        caveat='Only three joint seeds; their test image sets differ, so variance combines cohort composition and optimization')
    value['fingerprint'] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    path = OUT/'protocol.json'
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError('Joint-seed protocol changed')
    atomic_json_save(value, path)
    return value


def progress(model, stage, completed, total, status='running', **values):
    atomic_json_save(dict(stage=stage, completed=completed, total=total, status=status,
        heartbeat=datetime.now(timezone.utc).isoformat(), **values), OUT/model/'progress.json')


def run_model(model, device):
    p = protocol(); root = OUT/model; root.mkdir(parents=True, exist_ok=True)
    with (root/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        data = mlp.read(SOURCE/model/'matrices.pt')
        if data['fingerprint'] != p['source_fingerprints'][model]: raise ValueError('Source changed')
        completed = 0; total = len(GROUPS)*len(SEEDS)
        for group in GROUPS:
            cfg = p['settings'][model][group]
            for seed in SEEDS:
                result_path = root/group/f'seed{seed}.pt'
                if result_path.exists():
                    result = mlp.read(result_path)
                else:
                    split = image_split(model, seed); masks = masks_for(data['mentions'], split)
                    for name, mask in masks.items():
                        if set(data['y'][mask].tolist()) != {0, 1}: raise ValueError(f'{name} lacks a class')
                    stage = f'联合种子 {group} seed{seed}'
                    result = mlp.fit(data['groups'][group][masks['train']], data['y'][masks['train']],
                        data['groups'][group][masks['validation']], data['y'][masks['validation']],
                        cfg, seed, device, callback=lambda epoch:progress(model, stage, completed, total, epoch=epoch))
                    net = mlp.SingleMLP(result['input_dim'], result['config']).to(device)
                    net.load_state_dict(result['state_dict'])
                    test_x = torch.as_tensor(mlp.transform(data['groups'][group][masks['test']],
                        result['mean'], result['scale']), device=device)
                    probability = mlp.predict(net, test_x); y = data['y'][masks['test']]
                    result.update(model=model, group=group, split_seed=seed,
                        training_seed=seed, split=split, split_counts={k:int(v.sum()) for k,v in masks.items()},
                        test_hall_count=int((y == 0).sum()), test_real_count=int((y == 1).sum()),
                        test_hall_prevalence=float((y == 0).mean()), test_metrics=mlp.metrics(y, probability),
                        test_probabilities=probability, fingerprint=p['fingerprint'])
                    atomic_torch_save(result, result_path)
                if result['fingerprint'] != p['fingerprint'] or result['config'] != cfg:
                    raise ValueError('Incompatible resumed result')
                completed += 1; progress(model, f'联合种子 {group} seed{seed}', completed, total)
        progress(model, '联合数据划分/训练种子完成', total, total, status='completed')


def aggregate(values, key):
    a = np.asarray([v[key] for v in values], dtype=np.float64)
    return float(a.mean()), float(a.std())


def fixed_rows(model, group):
    return [mlp.read(SEARCH/model/'final'/'true_rms'/group/f'seed{seed}/result.pt') for seed in SEEDS]


def summarize():
    p = protocol(); rows = []; per_seed = []
    for model in MODELS:
        for group in GROUPS:
            joint = [mlp.read(OUT/model/group/f'seed{seed}.pt') for seed in SEEDS]
            fixed = fixed_rows(model, group)
            for seed, value in zip(SEEDS, joint):
                per_seed.append(dict(model=model, group=group, regime='joint_split_training', seed=seed,
                    AUROC=value['test_metrics']['AUROC'], HALL_AUPR=value['test_metrics']['HALL_AUPR'],
                    test_mentions=value['test_hall_count']+value['test_real_count'],
                    test_hall_prevalence=value['test_hall_prevalence']))
            for regime, values in (('fixed_split', fixed), ('joint_split_training', joint)):
                metrics = [v['test_metrics'] for v in values]
                auc = aggregate(metrics, 'AUROC'); ap = aggregate(metrics, 'HALL_AUPR')
                rows.append(dict(model=model, group=group, regime=regime,
                    AUROC_mean=auc[0], AUROC_std=auc[1], HALL_AUPR_mean=ap[0], HALL_AUPR_std=ap[1]))
    write_csv(rows, OUT/'summary.csv'); write_csv(per_seed, OUT/'per_seed.csv')
    lookup={(r['model'],r['group'],r['regime']):r for r in rows}
    names=dict(qwen2_5_vl_7b='Qwen2.5-VL-7B',llava_1_5_7b='LLaVA-1.5-7B',
        qwen3_vl_8b='Qwen3-VL-8B',internvl_2_5_8b='InternVL2.5-8B')
    lines=['# 四模型：数据划分与训练共用随机种子的影响','',
        '设置：每个seed分别将全部4000张图随机划分为3200 train / 400 validation / 400 test；同一个seed同时控制划分、MLP初始化、minibatch顺序和dropout。seeds43/44/45。',
        '信号：真实RMS all-attention `z-A_all→z`，local FP32 Gauss–Legendre K32。V=`[AE_V,log1p(S_V^all)]`；VP+G=`[AE_VP,log1p(S_V^all+S_P^all),AE_G,log1p(S_G^all)]`。',
        '分类器：每个模型/特征组沿用旧固定划分上只按validation选出的单隐藏层MLP配置；新划分不重新搜索。validation仍用于早停，无train+validation refit。',
        '指标：每个seed在各自400张test图上单独计算，再报告算术均值±总体标准差（ddof=0），不是概率ensemble。联合种子的std同时包含测试图片构成和训练随机性，只有3个seed。','',
        '| 模型 | 特征 | 固定划分 AUROC | 联合种子 AUROC | 固定划分 HALL-AUPR | 联合种子 HALL-AUPR |',
        '|---|---|---:|---:|---:|---:|']
    for model in MODELS:
        for group in GROUPS:
            fixed=lookup[model,group,'fixed_split']; joint=lookup[model,group,'joint_split_training']
            fmt=lambda r,k:f"{100*r[k+'_mean']:.2f} ± {100*r[k+'_std']:.2f}"
            lines.append(f"| {names[model]} | {group} | {fmt(fixed,'AUROC')} | {fmt(joint,'AUROC')} | {fmt(fixed,'HALL_AUPR')} | {fmt(joint,'HALL_AUPR')} |")
    lines += ['', '## 联合种子逐次结果', '',
        '| 模型 | 特征 | seed43 AUROC/AP | seed44 AUROC/AP | seed45 AUROC/AP |',
        '|---|---|---:|---:|---:|']
    for model in MODELS:
        for group in GROUPS:
            values=[mlp.read(OUT/model/group/f'seed{seed}.pt')['test_metrics'] for seed in SEEDS]
            cells=[f"{100*v['AUROC']:.2f}/{100*v['HALL_AUPR']:.2f}" for v in values]
            lines.append(f"| {names[model]} | {group} | {' | '.join(cells)} |")
    lines += ['', '## 每次测试集构成', '',
        '同一模型内，V 与 VP+G 使用完全相同的图片和mentions。HALL比例是测试mentions中的HALL比例。', '',
        '| 模型 | seed43 mentions/HALL% | seed44 mentions/HALL% | seed45 mentions/HALL% |',
        '|---|---:|---:|---:|']
    for model in MODELS:
        values=[mlp.read(OUT/model/'visual'/f'seed{seed}.pt') for seed in SEEDS]
        cells=[f"{v['test_hall_count']+v['test_real_count']}/{100*v['test_hall_prevalence']:.2f}" for v in values]
        lines.append(f"| {names[model]} | {' | '.join(cells)} |")
    lines += ['', '## 解读', '',
        '- Qwen2.5 的 V 最敏感：联合种子AUROC为85.93/88.90/81.85，std为2.89个百分点；VP+G为88.14/88.29/87.09，std仅0.54。seed45的V验证AUROC仍为88.57而测试为81.85，主要表现为该测试图片子集上的泛化落差，不能只归因于训练没有收敛。',
        '- LLaVA最稳定：V的联合种子AUROC std只有0.04个百分点；VP+G为0.46。',
        '- Qwen3的V与VP+G联合种子均值分别比原固定测试集低1.48和2.14个百分点；对应HALL-AUPR低7.67和5.26个百分点，说明原固定测试集对Qwen3这两组特征相对有利。',
        '- InternVL的V几乎不变；VP+G联合种子均值比固定测试集低2.02 AUROC和4.13 HALL-AUPR个百分点。', '',
        '固定划分三次共享同一test集，只改变训练随机性；联合种子三次使用不同test集。每个联合test集与固定test集仅重合32–48张图，三个联合test集两两仅重合41–46张图。因此两列std回答的问题不同，不能把std变化全部解释为优化稳定性。联合均值与固定均值的差异也只是测试子集敏感性，当前只有三个新划分。','']
    (ROOT/'docs/JOINT_SPLIT_TRAINING_SEEDS_811_RESULTS.md').write_text('\n'.join(lines))
    atomic_json_save(dict(status='completed', fingerprint=p['fingerprint'], rows=len(rows), seeds=len(per_seed)), OUT/'status.json')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=MODELS)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--summarize', action='store_true')
    args=parser.parse_args(); torch.set_num_threads(1)
    if args.summarize: summarize()
    elif args.models:
        for model in args.models: run_model(model,args.device)
    else: parser.error('--models or --summarize required')
