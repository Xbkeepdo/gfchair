"""Separate and concatenated source-share probes using the frozen composition MLP."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.plot_ffn_text_sources import collect, self_check, MODELS, SOURCES, TITLES
from scripts import run_ffn_source_composition as base
from scripts.train_torch_probe_feature_sets import TorchProbeConfig

OUT = base.OUT/'source_share_detection'
SEEDS = (43, 44, 45)
CONCAT = 'prompt_generation_visual'


def run(model, device):
    root = OUT/model
    if not (root/'matrices.pt').exists():
        d = collect(model)
        original = base.read(base.OUT/model/'matrices.pt')
        mentions = original['mentions']; n = d['ntrain']
        splits = json.loads((ROOT/'outputs'/model/base.EXPERIMENT/'image_splits.json').read_text())
        train_ids, test_ids = set(splits['train']), set(splits['test'])
        assert len(train_ids) == 3200 and len(test_ids) == 800 and not train_ids & test_ids
        assert all(m['image_id'] in train_ids for m in mentions[:n])
        assert all(m['image_id'] in test_ids for m in mentions[n:])
        groups = {name: d['fraction'][:, i].astype(np.float32) for i, name in enumerate(SOURCES)}
        assert all(np.isfinite(x).all() and (x >= 0).all() and (x <= 1+1e-6).all() for x in groups.values())
        base.save(dict(groups=groups, y=d['y'], ntrain=n, mentions=mentions), root/'matrices.pt')
        base.json_save(dict(model=model, source=str(base.OUT/model/'shards/k50'),
            features={name: f'Each layer: sum_{name} ||c_j|| / gross_all' for name in SOURCES},
            denominator='All token norms + residual norm + attention output bias norm + ||FFN(0)||',
            prompt='All nonvisual prompt positions, including system/template/special tokens',
            generation='Generated prefix through query, excludes predicted target and future',
            images=4000, train_images=3200, test_images=800, train_mentions=n, test_mentions=len(mentions)-n,
            dimensions={name: x.shape[1] for name, x in groups.items()}, seeds=SEEDS,
            config=asdict(TorchProbeConfig()), input_scaling='Raw fractions; no log/AE/extra normalization',
            reporting='Three-seed mean and population std, not ensemble; original two thresholds'), root/'protocol.json')
    matrix = base.read(root/'matrices.pt')
    if CONCAT not in matrix['groups']:
        matrix['groups'][CONCAT] = np.concatenate([matrix['groups'][name] for name in SOURCES], axis=1)
        base.save(matrix, root/'matrices.pt')
        protocol = json.loads((root/'protocol.json').read_text())
        protocol['features'][CONCAT] = 'concat(prompt all layers, generation all layers, visual all layers)'
        protocol['dimensions'][CONCAT] = matrix['groups'][CONCAT].shape[1]
        base.json_save(protocol, root/'protocol.json')
    base.train(model, device, root=root)


def summarize():
    rows, seed_rows = [], []
    lines = ['# 三类来源份额的单组与拼接检测效果', '',
             '四模型原4000图、3200/800划分和全部mentions；三组分别单独训练，并新增prompt_generation_visual三组拼接。均未拼接AE或强度。',
             '拼接按[prompt全层, generation全层, visual全层]顺序，四模型维度84/96/108/96；单组L维、拼接3L维。',
             '份额=sum_group ||c_j|| / gross_all；分母含residual、attention输出偏置和FFN(0)范数。prompt包括模板，generation仅含已生成前缀。',
             '原MLP (128/64/32、dropout .3、Adam、batch 256)、原调度/早停/最低训练loss checkpoint，seeds43/44/45。',
             '每格为三seed均值±总体标准差（ddof=0），单位%；HALL-F1使用各seed训练集REAL-F1阈值。没有调参或bootstrap。',
             'F_E=AE+log1p(S_E)、F_C=AE+log1p(S_C)直接引用同划分已有结果，仅作基线，输入均为2L维。', '',
             '| 模型 | 特征 | AUROC | HALL-AUPR | HALL-F1 |', '|---|---|---:|---:|---:|']
    for model, title in zip(MODELS, TITLES):
        path = OUT/model/'detection.json'
        if not path.exists():
            continue
        groups = json.loads(path.read_text())
        old = json.loads((base.OUT/model/'detection.json').read_text())
        groups.update(F_E=old['F'], F_C=old['F_C'])
        for name, value in groups.items():
            per_seed = value['per_seed_metrics']
            reports = [per_seed[str(s)]['threshold_reports']['train_f1']['test_metrics'] for s in SEEDS]
            values = dict(AUROC=[r['auc'] for r in reports],
                          HALL_AUPR=[r['hallucination_positive']['aupr'] for r in reports],
                          HALL_F1=[r['hallucination_positive']['f1'] for r in reports])
            row = dict(model=model, group=name)
            cells = []
            for metric, v in values.items():
                row[metric+'_mean'], row[metric+'_std'] = float(np.mean(v)), float(np.std(v))
                cells.append(f'{100*np.mean(v):.3f} ± {100*np.std(v):.3f}')
            rows.append(row)
            lines.append(f'| {title} | {name} | '+ ' | '.join(cells)+' |')
            for seed in SEEDS:
                for rule, report in per_seed[str(seed)]['threshold_reports'].items():
                    metric = report['test_metrics']; hall = metric['hallucination_positive']
                    seed_rows.append(dict(model=model, group=name, seed=seed, threshold_rule=rule,
                        threshold=report['threshold'], AUROC=metric['auc'], HALL_AUPR=hall['aupr'],
                        HALL_precision=hall['precision'], HALL_recall=hall['recall'], HALL_F1=hall['f1']))
    pairs = []
    for model in MODELS:
        selected = {r['group']: r for r in rows if r['model'] == model}
        if CONCAT not in selected:
            continue
        best = max(SOURCES, key=lambda name: selected[name]['AUROC_mean'])
        c, b = selected[CONCAT], selected[best]
        pairs.append(dict(model=model, best_single=best, AUROC=c['AUROC_mean'], HALL_AUPR=c['HALL_AUPR_mean'],
                          AUROC_delta=c['AUROC_mean']-b['AUROC_mean'],
                          HALL_AUPR_delta=c['HALL_AUPR_mean']-b['HALL_AUPR_mean']))
    if pairs:
        lines += ['', '拼接相对本次最佳单组（按三seed平均AUROC）的增量，单位百分点；仅描述性比较。', '',
                  '| 模型 | 最佳单组 | 拼接ΔAUROC | 拼接ΔHALL-AUPR |', '|---|---|---:|---:|']
        for r in pairs:
            lines.append(f"| {r['model']} | {r['best_single']} | {100*r['AUROC_delta']:+.3f} | {100*r['HALL_AUPR_delta']:+.3f} |")
        base.write_csv(pairs, OUT/'concat_comparison.csv')
        lines += ['', '拼接结果：四模型平均AUROC均高于最佳单组，三seed的AUROC增量也均为正。'
                  'HALL-AUPR在Qwen2.5/Qwen3/InternVL提高，LLaVA基本持平（-0.007点）。'
                  '对照AE+S：Qwen3的AUROC/AP均高于F_E和F_C；InternVL的AUROC高于两者，AP低于F_E但高于F_C；'
                  'Qwen2.5/LLaVA两项仍低于两种基线。本次不作显著性声明。']
    lines += ['', '此前单组结果：视觉份额在Qwen2.5/LLaVA/Qwen3的三组中平均AUROC与HALL-AUPR最高；InternVL两项均以prompt份额最高。'
              '四模型三组单独输入的平均AUROC与HALL-AUPR都低于各自F_E和F_C。'
              '单组对照不支持单独替代AE+S；三组份额自身拼接结果见上。尚未检验与AE+S拼接后的增益，不声称显著性。', '',
              '结果位于本目录各模型matrices.pt、protocol.json、heads和detection.json。'
              'detection.json由原汇总函数生成，含历史兼容的ensemble字段；本报告只使用逐seed统计。',
              '训练：`python scripts/train_ffn_source_shares.py --models <模型列表> --device cuda:0`；'
              '汇总：`python scripts/train_ffn_source_shares.py --summarize`。',
              '这是来源位置的范数份额检测，不是attention检测；生成长度/位置混杂和原K50重建误差限制仍适用。']
    base.write_csv(rows, OUT/'detection.csv')
    base.write_csv(seed_rows, OUT/'seed_metrics.csv')
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=MODELS, default=MODELS)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--summarize', action='store_true')
    args = parser.parse_args()
    if args.summarize:
        summarize()
    else:
        self_check()
        for model in args.models:
            run(model, args.device)
