"""Pair matching attention/gated-attention groups with full FFN source strengths."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_ffn_source_composition as base
from scripts.train_ffn_source_strengths import MODELS, TITLES, SCALES, KINDS, SEEDS, OUT as STRENGTH
from scripts.train_prefix_attention_groups import FAMILIES, OUT as ATTENTION
from scripts.train_torch_probe_feature_sets import TorchProbeConfig

OUT = base.OUT/'attention_strength_fusion'


def pair(a, s):
    assert a.shape == s.shape
    return np.concatenate((a, s), axis=1)


def prepare(model):
    root = OUT/model
    if (root/'matrices.pt').exists():
        return
    a = base.read(ATTENTION/model/'matrices.pt')
    s = base.read(STRENGTH/model/'matrices.pt')
    assert a['ntrain'] == s['ntrain']
    np.testing.assert_array_equal(a['y'], s['y'])
    fields = ('mention_id', 'image_id', 'target_key', 'response_index', 'target_token_id', 'label')
    assert [tuple(m[k] for k in fields) for m in a['mentions']] == [tuple(m[k] for k in fields) for m in s['mentions']]
    groups = {f'{family}_{scale}_{kind}': pair(a['groups'][f'{family}_{scale}_{kind}'], s['groups'][f'{scale}_{kind}'])
              for family in FAMILIES for scale in SCALES for kind in KINDS}
    assert all(np.isfinite(x).all() for x in groups.values())
    base.save(dict(groups=groups, y=s['y'], ntrain=s['ntrain'], mentions=s['mentions']), root/'matrices.pt')
    base.json_save(dict(model=model, attention_source=str(ATTENTION/model), strength_source=str(STRENGTH/model),
        formula='concat(matching attention group vector, matching full-source S_g vector)',
        families=FAMILIES, scales=SCALES, kinds=KINDS, strength='sum_group ||c_j|| from K50 full FFN composition; visual=S_C',
        raw='[A_group, S_group] or [U_group, S_group]',
        log1p='[log1p(A_group), log1p(S_group)] or [log1p(U_group), log1p(S_group)]; no cross-scale combinations',
        concat_order='[A_P all layers, A_G all layers, A_V all layers, S_P all layers, S_G all layers, S_V all layers]; U replaces A for gated',
        visual_prompt_sum='[A_V+A_P, S_V+S_P]; gated replaces A with U; sum before log1p',
        prompt='Includes BOS/template/special tokens in both sources', gate_reference='full causal-prefix MAD, same saved gated signal',
        images=4000, train_images=3200, test_images=800, train_mentions=s['ntrain'], test_mentions=len(s['y'])-s['ntrain'],
        seeds=SEEDS, config=asdict(TorchProbeConfig()), dimensions={k: v.shape[1] for k, v in groups.items()},
        reporting='Three-seed mean and population std, original two thresholds; no ensemble in report'), root/'protocol.json')
    print('PREPARED', model, len(s['y']), 'mentions', flush=True)


def metrics(entry):
    reports = [entry['per_seed_metrics'][str(s)]['threshold_reports']['train_f1']['test_metrics'] for s in SEEDS]
    values = dict(AUROC=[r['auc'] for r in reports], HALL_AUPR=[r['hallucination_positive']['aupr'] for r in reports],
                  HALL_F1=[r['hallucination_positive']['f1'] for r in reports])
    return {f'{name}_{stat}': float(fn(v)) for name, v in values.items()
            for stat, fn in (('mean', np.mean), ('std', np.std))}


def summarize():
    rows, seeds, comparisons = [], [], []
    lines = ['# 对应attention与完整S_g配对拼接检测', '',
             '四模型原4000图/3200-800/全部mentions，原MLP/seeds43-45；只训练20融合组×4模型×3seed=240新头，原单独结果复用。',
             'attention/gated各raw/log1p两套，每套prompt/generation/visual、三组全拼接、V+P相加五组。',
             '同尺度配对：[A_g,S_g]、[U_g,S_g]、[log1p(A_g),log1p(S_g)]、[log1p(U_g),log1p(S_g)]；没有跨尺度组合。',
             '单组/相加为2L维，全拼接6L维，按[attention的P/G/V全层,S的P/G/V全层]顺序。V+P先各自求和再log1p。',
             'S_g是K50完整来源贡献范数之和，视觉为S_C；BOS并入prompt；gate沿用完整前缀MAD。MiniGPT/Shikra尚无完整S_g缓存，本轮未混用旧e_m强度。',
             '以下为三seed均值±总体std（%），HALL-F1用各seed训练REAL-F1阈值；不使用ensemble，不调参/bootstrap。', '',
             '| 模型 | 融合组 | AUROC | HALL-AUPR | HALL-F1 |', '|---|---|---:|---:|---:|']
    for model, title in zip(MODELS, TITLES):
        fused = json.loads((OUT/model/'detection.json').read_text())
        attention = json.loads((ATTENTION/model/'detection.json').read_text())
        strength = json.loads((STRENGTH/model/'detection.json').read_text())
        for name, entry in fused.items():
            values = metrics(entry); rows.append(dict(model=model, group=name, **values))
            cells = [f"{100*values[k+'_mean']:.3f} ± {100*values[k+'_std']:.3f}" for k in ('AUROC', 'HALL_AUPR', 'HALL_F1')]
            lines.append(f'| {title} | {name} | '+' | '.join(cells)+' |')
            for seed in SEEDS:
                for rule, report in entry['per_seed_metrics'][str(seed)]['threshold_reports'].items():
                    m = report['test_metrics']; h = m['hallucination_positive']
                    seeds.append(dict(model=model, group=name, seed=seed, threshold_rule=rule, threshold=report['threshold'],
                        AUROC=m['auc'], HALL_AUPR=h['aupr'], HALL_precision=h['precision'], HALL_recall=h['recall'], HALL_F1=h['f1']))
            s_name = name.split('_', 1)[1]
            for source, reference in (('gated_only' if name.startswith('gated_') else 'attention_only', attention[name]), ('strength_only', strength[s_name])):
                b = metrics(reference)
                comparisons.append(dict(model=model, group=name, baseline=source,
                    AUROC=values['AUROC_mean'], HALL_AUPR=values['HALL_AUPR_mean'],
                    baseline_AUROC=b['AUROC_mean'], baseline_HALL_AUPR=b['HALL_AUPR_mean'],
                    AUROC_delta=values['AUROC_mean']-b['AUROC_mean'], HALL_AUPR_delta=values['HALL_AUPR_mean']-b['HALL_AUPR_mean'],
                    **{f'seed{s}_AUROC_delta':entry['per_seed_metrics'][str(s)]['auc']-reference['per_seed_metrics'][str(s)]['auc'] for s in SEEDS}))
    overview = ['三组全拼接总览，AUROC（%）：', '',
                '| 模型 | A+S 原值 | U+S 原值 | log1p(A)+log1p(S) | log1p(U)+log1p(S) |', '|---|---:|---:|---:|---:|']
    lookup = {(r['model'], r['group']): r for r in rows}
    for model, title in zip(MODELS, TITLES):
        cells = []
        for key in ('attention_raw_concat', 'gated_raw_concat', 'attention_log1p_concat', 'gated_log1p_concat'):
            r = lookup[model, key]; cells.append(f"{100*r['AUROC_mean']:.3f} ± {100*r['AUROC_std']:.3f}")
        overview.append(f'| {title} | '+' | '.join(cells)+' |')
    overview += ['', '各模型预设融合组中的最高平均AUROC：', '',
                 '| 模型 | 融合组 | AUROC | HALL-AUPR |', '|---|---|---:|---:|']
    for model, title in zip(MODELS, TITLES):
        r = max((r for r in rows if r['model'] == model), key=lambda r: r['AUROC_mean'])
        overview.append(f"| {title} | {r['group']} | {100*r['AUROC_mean']:.3f} ± {100*r['AUROC_std']:.3f} | {100*r['HALL_AUPR_mean']:.3f} ± {100*r['HALL_AUPR_std']:.3f} |")
    overview += ['', '结果：LLaVA原值gate全拼接、Qwen3双方log1p的gate全拼接、InternVL原值gate prompt配对，平均AUROC/AP均超过各自两个对应单独基线，三个seed的AUROC差也均为正。'
                 'Qwen2本轮最高AUROC融合组仍低于对应gate单独使用（86.796 vs 88.032），三个seed均下降；融合没有跨模型统一增益。'
                 '最高均值只作本轮预设组的描述性对比，未据此修改训练配置或宣称显著性。', '',
                 'A为attention区域和，U为attention×gate区域和；每一列都包含对应S_g。完整单组及相加组如下。', '']
    index = lines.index('| 模型 | 融合组 | AUROC | HALL-AUPR | HALL-F1 |'); lines[index:index] = overview
    lines += ['', 'comparisons.csv逐组报告融合相对对应attention/gated单独、S_g单独的平均指标差和逐seed AUROC差；增量为原始比例单位。',
              '输入来自同一批目标但不同提取缓存，原mention顺序、目标ID/标签/划分已核对。两路源各自既有数值与gate口径限制保留；维度增加，不据提升宣称显著性或因果。',
              '运行：`python scripts/train_attention_strength_fusion.py --stage prepare`，再`--stage train --models <模型> --device cuda:0`；完成后`--stage summarize`。',
              '各模型保存matrices.pt/protocol.json/heads/detection.json，根目录detection.csv和seed_metrics.csv为均值/std与逐seed双阈值。']
    base.write_csv(rows, OUT/'detection.csv'); base.write_csv(seeds, OUT/'seed_metrics.csv')
    base.write_csv(comparisons, OUT/'comparisons.csv'); (OUT/'summary.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=MODELS, default=MODELS)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--stage', choices=('prepare', 'train', 'summarize'), default='train')
    args = parser.parse_args()
    if args.stage == 'summarize':
        summarize()
    else:
        np.testing.assert_array_equal(pair(np.array([[1., 2.]]), np.array([[3., 4.]])), [[1., 2., 3., 4.]])
        for model in args.models:
            prepare(model)
            if args.stage == 'train': base.train(model, args.device, root=OUT/model)
