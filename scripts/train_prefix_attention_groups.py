"""Grouped attention/gated attention probes from existing six-model region caches."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.plot_prefix_attention_regions import MODELS, METRICS, REGIONS, OUT as REGION_ROOT
from scripts.train_ffn_source_strengths import build_groups, check_groups, SCALES, KINDS, SEEDS
from scripts import run_ffn_source_composition as base
from scripts.train_torch_probe_feature_sets import TorchProbeConfig

OUT = ROOT/'outputs/prefix_attention_gate/full/group_detection'
FAMILIES = ('attention', 'gated')
TITLES = ('MiniGPT-4', 'Shikra', 'Qwen2.5-VL', 'LLaVA-1.5', 'Qwen3-VL', 'InternVL-2.5')


def three_regions(values):
    # Cached regions: BOS, visual, prompt excluding BOS, generated_text.
    v = np.asarray(values, dtype=np.float64)
    return np.stack((v[..., 0]+v[..., 2], v[..., 3], v[..., 1]), axis=2)


def check():
    check_groups()
    v = np.array([[[[.1, .2, .3, .4]], [[.05, .1, .15, .2]]]])
    g = three_regions(v)
    np.testing.assert_allclose(g[0, 0, :, 0], [.4, .4, .2])
    np.testing.assert_allclose(g.sum(2), v.sum(-1))


def source_root(model):
    experiment = 'COCO4000-JACOBIAN-PATH' if model in MODELS[:2] else base.EXPERIMENT
    return ROOT/'outputs'/model/experiment


def prepare(model):
    root = OUT/model
    if (root/'matrices.pt').exists():
        return
    source = REGION_ROOT/model
    meta = json.loads((source/'metadata.json').read_text())
    assert meta['metrics'] == list(METRICS) and meta['regions'] == list(REGIONS)
    split = json.loads((source_root(model)/'image_splits.json').read_text())
    train_ids, test_ids = set(split['train']), set(split['test'])
    assert len(train_ids) == 3200 and len(test_ids) == 800 and not train_ids & test_ids
    if model in MODELS[:2]:
        original = json.loads((source_root(model)/'path/training/protocol.json').read_text())
    else:
        original = base.read(base.OUT/model/'matrices.pt')
    mentions = original['mentions']
    with np.load(source/'regions.npz') as z:
        lookup = {key: i for i, key in enumerate(z['mention_ids'].tolist())}
        assert len(lookup) == len(mentions)
        order = np.array([lookup[m['mention_id']] for m in mentions])
        y, is_train, images = z['labels'][order], z['train'][order], z['image_ids'][order]
        np.testing.assert_array_equal(y, [m['label'] for m in mentions])
        np.testing.assert_array_equal(images, [m['image_id'] for m in mentions])
        n = int(is_train.sum())
        assert is_train[:n].all() and not is_train[n:].any()
        assert all(i in train_ids for i in images[:n]) and all(i in test_ids for i in images[n:])
        v = z['values'][order]
    gross = three_regions(v)
    np.testing.assert_allclose(gross.sum(2), v.astype(np.float64).sum(-1), rtol=1e-7, atol=1e-8)
    assert np.isfinite(gross).all() and (gross >= 0).all()
    assert (gross[:, 1] <= gross[:, 0]+1e-7).all()
    groups = {f'{family}_{name}': x for fi, family in enumerate(FAMILIES)
              for name, x in build_groups(gross[:, fi]).items()}
    base.save(dict(groups=groups, y=y, ntrain=n, mentions=mentions), root/'matrices.pt')
    base.json_save(dict(model=model, source=str(source/'regions.npz'), families=FAMILIES,
        formula='sum group raw_attention, or sum group (raw_attention * full-prefix gate); no renormalization/token-count division',
        prompt='BOS + cached prompt: full nonvisual prompt including template and special tokens',
        generation='Only generated prefix available through query; target/future excluded',
        gate_reference='full causal prefix, native hpre target-logit MAD sigmoid; different reference support from original visual-only AE',
        scales=SCALES, groups=KINDS, concat_order=['prompt', 'generation', 'visual'],
        sum_formula='visual+prompt first; log1p(sum) in log version',
        images=4000, train_images=3200, test_images=800, mentions=len(y), train_mentions=n, test_mentions=len(y)-n,
        dimensions={name: x.shape[1] for name, x in groups.items()}, seeds=SEEDS, config=asdict(TorchProbeConfig()),
        original_mention_order=True, head_mean=True, reporting='Three-seed mean and population std; original two thresholds'),
        root/'protocol.json')
    print('PREPARED', model, len(y), 'mentions', flush=True)


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data = {model: base.read(OUT/model/'matrices.pt') for model in MODELS}
    rows = []
    for family in FAMILIES:
        for scale in SCALES:
            fig, axes = plt.subplots(6, 4, figsize=(18, 17))
            for mi, (model, title) in enumerate(zip(MODELS, TITLES)):
                d = data[model]
                for si, kind in enumerate(('prompt', 'generation', 'visual', 'visual_prompt_sum')):
                    ax = axes[mi, si]
                    for label, name, color in ((0, 'HALL', '#d55e00'), (1, 'REAL', '#0072b2')):
                        v = d['groups'][f'{family}_{scale}_{kind}'][d['y'] == label].astype(np.float64)
                        mean, median = v.mean(0), np.median(v, axis=0)
                        q25, q75 = np.quantile(v, [.25, .75], axis=0); layers = np.arange(1, len(mean)+1)
                        ax.plot(layers, mean, color=color, label=f'{name} (n={len(v)})')
                        ax.plot(layers, median, '--', color=color, linewidth=1, alpha=.6)
                        ax.fill_between(layers, q25, q75, color=color, alpha=.12)
                        rows.extend(dict(model=model, family=family, scale=scale, group=kind, label=label, layer=int(l),
                            n=len(v), mean=float(a), median=float(b), q25=float(c), q75=float(e))
                            for l, a, b, c, e in zip(layers, mean, median, q25, q75))
                    ax.set_title(f'{title} / {kind.replace("visual_prompt_sum", "visual + prompt")}', fontsize=9)
                    ax.set_xlabel('Decoder layer'); ax.grid(alpha=.15); ax.legend(fontsize=7)
                    if si == 0: ax.set_ylabel('Group mass' if scale == 'raw' else 'log(1 + group mass)')
            fig.suptitle(f'ALL 4000 IMAGES / all mentions / {family} / {scale}: mean (solid), median (dashed), IQR (shade)', fontsize=12)
            fig.tight_layout(); fig.savefig(OUT/f'all_{family}_{scale}.png', dpi=150)
            fig.savefig(OUT/f'all_{family}_{scale}.pdf'); plt.close(fig)
    base.write_csv(rows, OUT/'curves.csv')


def summarize():
    rows, seeds, comparisons = [], [], []
    lines = ['# 分组attention与attention×gate检测', '',
             '六模型原4000图/3200-800/全部mentions，按原训练mention顺序；prompt含BOS、模板及特殊token，generation仅为已生成前缀。',
             'attention为跨头均值的原始注意力，gated为其乘完整因果前缀MAD gate；按组直接求和，不再归一化、不除token数。',
             '各自raw/log1p两套，每套prompt/generation/visual单独、三组concat、visual+prompt相加五组。先相加再log1p；未跨attention与gated混合拼接。',
             '原MLP/seeds43-45，共360头。以下三seed均值±总体std（%），HALL-F1采用每seed训练REAL-F1阈值；原双阈值见CSV。',
             '原始attention三组总量约1，V+P约等于1-G，属于近似冗余信号；不据分类器小差异声称新增信息。gate的总量则可变化，且本轮gate统计范围不同于原视觉AE。', '',
             '| 模型 | 输入 | AUROC | HALL-AUPR | HALL-F1 |', '|---|---|---:|---:|---:|']
    for model, title in zip(MODELS, TITLES):
        results = json.loads((OUT/model/'detection.json').read_text())
        if model in MODELS[:2]:
            baseline = json.loads((source_root(model)/'path/training/results.json').read_text())['F']
        else:
            baseline = json.loads((base.OUT/model/'detection.json').read_text())['F']
        results['F_E_reference'] = baseline
        for name, entry in results.items():
            reports = [entry['per_seed_metrics'][str(s)]['threshold_reports']['train_f1']['test_metrics'] for s in SEEDS]
            values = dict(AUROC=[r['auc'] for r in reports], HALL_AUPR=[r['hallucination_positive']['aupr'] for r in reports],
                          HALL_F1=[r['hallucination_positive']['f1'] for r in reports])
            row = dict(model=model, group=name); cells = []
            for metric, v in values.items():
                row[metric+'_mean'], row[metric+'_std'] = float(np.mean(v)), float(np.std(v))
                cells.append(f'{100*np.mean(v):.3f} ± {100*np.std(v):.3f}')
            rows.append(row); lines.append(f'| {title} | {name} | '+' | '.join(cells)+' |')
            for seed in SEEDS:
                for rule, report in entry['per_seed_metrics'][str(seed)]['threshold_reports'].items():
                    m = report['test_metrics']; h = m['hallucination_positive']
                    seeds.append(dict(model=model, group=name, seed=seed, threshold_rule=rule, threshold=report['threshold'],
                        AUROC=m['auc'], HALL_AUPR=h['aupr'], HALL_precision=h['precision'], HALL_recall=h['recall'], HALL_F1=h['f1']))
        selected = {r['group']: r for r in rows if r['model'] == model}
        pairs = [(f'gated_{s}_{k}', f'attention_{s}_{k}') for s in SCALES for k in KINDS]
        pairs += [(f'{f}_log1p_{k}', f'{f}_raw_{k}') for f in FAMILIES for k in KINDS]
        for f in FAMILIES:
            for s in SCALES:
                best = max(('prompt', 'generation', 'visual'), key=lambda k: selected[f'{f}_{s}_{k}']['AUROC_mean'])
                pairs.append((f'{f}_{s}_concat', f'{f}_{s}_{best}'))
        for left, right in pairs:
            comparisons.append(dict(model=model, left=left, right=right,
                AUROC_delta=selected[left]['AUROC_mean']-selected[right]['AUROC_mean'],
                HALL_AUPR_delta=selected[left]['HALL_AUPR_mean']-selected[right]['HALL_AUPR_mean']))
    overview = ['三组拼接总览，AUROC均值±总体std（%）：', '',
                '| 模型 | attention 原值 | attention×gate 原值 | attention log1p | attention×gate log1p |',
                '|---|---:|---:|---:|---:|']
    lookup = {(r['model'], r['group']): r for r in rows}
    for model, title in zip(MODELS, TITLES):
        cells = []
        for key in ('attention_raw_concat', 'gated_raw_concat', 'attention_log1p_concat', 'gated_log1p_concat'):
            r = lookup[model, key]
            cells.append(f"{100*r['AUROC_mean']:.3f} ± {100*r['AUROC_std']:.3f}")
        overview.append(f'| {title} | '+' | '.join(cells)+' |')
    overview += ['', '原值和log1p两套中，gate拼接六模型的平均AUROC/AP均高于对应attention拼接，也均高于各自最佳单组。'
                 '纯attention的原值拼接在Shikra略低于最佳单组（AUROC -0.052点，AP -0.433点），其他模型提高；log1p下Shikra仅小幅变化。'
                 'log1p没有跨模型一致收益；以上均为三seed点估计，未作显著性检验。', '', '以下为全部单组、相加及拼接结果：', '']
    index = lines.index('| 模型 | 输入 | AUROC | HALL-AUPR | HALL-F1 |')
    lines[index:index] = overview
    lines += ['', 'F_E_reference为已有AE+log1p(S_E)，仅作参照，未重新训练；本次输入不含AE或FFN强度。',
              '所有曲线为4000图全部mentions等权，先变换后汇总，阴影为IQR而非置信区间；训练评估仍只使用原800图holdout。',
              '[attention原值](all_attention_raw.png) · [attention log1p](all_attention_log1p.png) · [attention×gate原值](all_gated_raw.png) · [attention×gate log1p](all_gated_log1p.png)',
              '逐层曲线数据curves.csv；逐seed双阈值seed_metrics.csv；均值/std detection.csv；描述性配对comparisons.csv。原训练汇总JSON兼容保留ensemble字段，主表不用ensemble。',
              '运行：`python scripts/train_prefix_attention_groups.py --stage prepare`，然后`--stage train --models <列表> --device cuda:0`；完成后`--stage summarize`。',
              '保留前缀长度/位置混杂及原生attention舍入限制，不作显著性或因果声明，不调参或bootstrap。']
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
        check()
        for model in args.models:
            prepare(model)
            if args.stage == 'train': base.train(model, args.device, root=OUT/model)
        if args.stage == 'prepare': plot()
