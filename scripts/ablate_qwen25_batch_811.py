"""Controlled validation-only batch-size ablation for Qwen2.5 V and VP+G."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from features.tc_fvpa_artifacts import atomic_json_save, atomic_torch_save
from scripts import search_single_mlp_811 as mlp

MODEL = 'qwen2_5_vl_7b'
SOURCE = ROOT/'outputs/all_attention_ae_811_v1'/MODEL/'matrices.pt'
SEARCH = ROOT/'outputs/single_mlp_search_811_v1'/MODEL/'selection.json'
OUT = ROOT/'outputs/qwen25_batch_ablation_811_v1'
GROUPS = ('visual', 'vp_generation')
BATCHES = (32, 64, 128, 256, 512)
SEEDS = (43, 44, 45)


def aggregate(rows):
    return {
        metric+'_'+stat: float(fn([r['validation'][metric] for r in rows]))
        for metric in ('AUROC', 'HALL_AUPR')
        for stat, fn in (('mean', np.mean), ('std', np.std))
    }


def run(device):
    data = mlp.read(SOURCE)
    selected = json.loads(SEARCH.read_text())
    protocol = {
        'schema': 'qwen25-batch-ablation-811-v1',
        'model': MODEL,
        'split': '3200 train / 400 validation / 400 untouched test images',
        'features': {
            'visual': '[AE_V, log1p(S_V_all)]',
            'vp_generation': '[AE_VP, log1p(S_V_all+S_P_all), AE_G, log1p(S_G_all)]',
        },
        'path': 'true RMS all-attention z-A_all to z, local FP32 Gauss-Legendre K32',
        'groups': list(GROUPS), 'batch_sizes': list(BATCHES), 'seeds': list(SEEDS),
        'control': 'For each group, freeze its previously selected architecture and optimizer settings; change batch_size only.',
        'endpoint': 'validation only; no test probabilities or test metrics are computed',
        'caveat': 'Epoch/patience schedule is fixed, so optimizer-step count changes with batch size.',
        'base_selection_fingerprint': selected['fingerprint'],
        'source_fingerprint': data['fingerprint'],
    }
    path = OUT/'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError('Batch ablation protocol changed')
    atomic_json_save(protocol, path)
    train, val = data['masks']['train'], data['masks']['validation']
    total = len(GROUPS)*len(BATCHES)*len(SEEDS); done = 0
    summary = {}
    for group in GROUPS:
        variant = 'true_rms/'+group
        base = dict(selected['variants'][variant]['config'])
        summary[group] = {}
        for batch in BATCHES:
            rows = []
            for seed in SEEDS:
                result_path = OUT/group/f'batch{batch}'/f'seed{seed}.pt'
                stage = f'Qwen2.5 batch消融 {group} b{batch} seed{seed}'
                if result_path.exists():
                    result = mlp.read(result_path)
                else:
                    cfg = dict(base, batch_size=batch)
                    def callback(epoch):
                        atomic_json_save(dict(stage=stage, completed=done, total=total,
                            status='running', epoch=epoch, heartbeat=datetime.now(timezone.utc).isoformat()), OUT/'progress.json')
                    result = mlp.fit(data['groups'][group][train], data['y'][train],
                        data['groups'][group][val], data['y'][val], cfg, seed, device, callback)
                    result.update(group=group, batch_size=batch, source_fingerprint=data['fingerprint'])
                    atomic_torch_save(result, result_path)
                if result['source_fingerprint'] != data['fingerprint'] or result['config']['batch_size'] != batch:
                    raise ValueError('Incompatible resumed batch result')
                rows.append(result); done += 1
            summary[group][str(batch)] = aggregate(rows)
            atomic_json_save(summary, OUT/'results.json')
    lines = [
        '# Qwen2.5-VL-7B：batch size 受控消融', '',
        '811 图片划分：3200 train / 400 validation / 400 test；本消融只报告 validation，test 未访问。全部 mentions。',
        '特征路径：真实 RMS all-attention `z-A_all→z`，local FP32 Gauss–Legendre K32。V 与 VP+G 各自固定原已选单层 MLP 的其他参数，只改变 batch size。三种子 43/44/45 的算术均值 ± 总体标准差，不是概率 ensemble。',
        '训练仍按固定 epoch/patience 调度，因此 batch 改变时每个 epoch 的 optimizer steps 也随之改变。', '',
        '| 特征 | batch | validation AUROC | validation HALL-AUPR |',
        '|---|---:|---:|---:|',
    ]
    for group in GROUPS:
        for batch in BATCHES:
            row = summary[group][str(batch)]
            lines.append(f"| {group} | {batch} | {100*row['AUROC_mean']:.2f} ± {100*row['AUROC_std']:.2f} | {100*row['HALL_AUPR_mean']:.2f} ± {100*row['HALL_AUPR_std']:.2f} |")
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    atomic_json_save(dict(stage='Qwen2.5 batch消融完成', completed=total, total=total,
        status='completed', heartbeat=datetime.now(timezone.utc).isoformat()), OUT/'progress.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    run(args.device)
