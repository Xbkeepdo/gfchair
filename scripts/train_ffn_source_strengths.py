"""Raw/log1p full-source strengths: individual, concatenated, and visual+prompt sum."""
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

OUT = base.OUT/'source_strength_detection'
KINDS = (*SOURCES, 'concat', 'visual_prompt_sum')
SCALES = ('raw', 'log1p')
SEEDS = (43, 44, 45)


def build_groups(gross):
    # gross[N,3,L] uses prompt/generation/visual order; sum before taking log1p.
    values = {name: gross[:, i] for i, name in enumerate(SOURCES)}
    values['visual_prompt_sum'] = values['visual']+values['prompt']
    groups = {}
    for scale in SCALES:
        v = {name: (x if scale == 'raw' else np.log1p(x)).astype(np.float32)
             for name, x in values.items()}
        v['concat'] = np.concatenate([v[name] for name in SOURCES], axis=1)
        groups.update({f'{scale}_{name}': v[name] for name in KINDS})
    return groups


def check_groups():
    self_check()
    x = np.array([[[1., 2.], [3., 0.], [4., 5.]]])
    g = build_groups(x)
    np.testing.assert_array_equal(g['raw_visual_prompt_sum'], [[5., 7.]])
    np.testing.assert_allclose(g['log1p_visual_prompt_sum'], np.log1p([[5., 7.]]), rtol=1e-6)
    np.testing.assert_array_equal(g['raw_concat'], [[1., 2., 3., 0., 4., 5.]])
    np.testing.assert_allclose(g['log1p_concat'], np.log1p(g['raw_concat']), rtol=1e-6)


def run(model, device):
    root = OUT/model
    if not (root/'matrices.pt').exists():
        d = collect(model)
        old = base.read(base.OUT/model/'matrices.pt')
        np.testing.assert_array_equal(d['y'], old['y'])
        assert d['ntrain'] == old['ntrain']
        split = json.loads((ROOT/'outputs'/model/base.EXPERIMENT/'image_splits.json').read_text())
        train_ids, test_ids = set(split['train']), set(split['test'])
        n = d['ntrain']; mentions = old['mentions']
        assert len(train_ids) == 3200 and len(test_ids) == 800 and not train_ids & test_ids
        assert all(m['image_id'] in train_ids for m in mentions[:n])
        assert all(m['image_id'] in test_ids for m in mentions[n:])
        groups = build_groups(d['gross'])
        assert all(np.isfinite(x).all() and (x >= 0).all() for x in groups.values())
        base.save(dict(groups=groups, y=d['y'], ntrain=n, mentions=mentions), root/'matrices.pt')
        base.json_save(dict(model=model, source=str(base.OUT/model/'shards/k50'),
            formula='S_g=sum_{j in g} ||c_j||; no division by gross_all or token count',
            scales=list(SCALES), single_groups=list(SOURCES), concat_order=list(SOURCES),
            visual_prompt_sum='raw: S_V+S_P; log1p: log(1+S_V+S_P), sum before log',
            prompt='All nonvisual prompt positions, including system/template/special tokens',
            generation='Available generated prefix through query; excludes predicted target/future',
            images=4000, train_images=3200, test_images=800, train_mentions=n,
            test_mentions=len(mentions)-n, seeds=SEEDS, config=asdict(TorchProbeConfig()),
            dimensions={k: v.shape[1] for k, v in groups.items()},
            reporting='Three-seed mean +/- population std; original two thresholds; all-4000 curves by mention'),
            root/'protocol.json')
    base.train(model, device, root=root)


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data = {model: base.read(OUT/model/'matrices.pt') for model in MODELS}
    rows = []
    for scale in SCALES:
        fig, axes = plt.subplots(4, 4, figsize=(18, 12))
        for mi, (model, title) in enumerate(zip(MODELS, TITLES)):
            d = data[model]
            for si, name in enumerate((*SOURCES, 'visual_prompt_sum')):
                ax = axes[mi, si]
                for label, label_name, color in ((0, 'HALL', '#d55e00'), (1, 'REAL', '#0072b2')):
                    v = d['groups'][f'{scale}_{name}'][np.asarray(d['y']) == label].astype(np.float64)
                    mean, median = v.mean(0), np.median(v, axis=0)
                    q25, q75 = np.quantile(v, [.25, .75], axis=0)
                    layers = np.arange(1, len(mean)+1)
                    ax.plot(layers, mean, color=color, label=f'{label_name} (n={len(v)})')
                    ax.plot(layers, median, '--', color=color, alpha=.6, linewidth=1)
                    ax.fill_between(layers, q25, q75, color=color, alpha=.12)
                    rows.extend(dict(model=model, scale=scale, source=name, label=label, layer=int(l),
                        n=len(v), mean=float(a), median=float(b), q25=float(c), q75=float(e))
                        for l, a, b, c, e in zip(layers, mean, median, q25, q75))
                ax.set_title(f'{title} / {name.replace("visual_prompt_sum", "visual + prompt")}')
                ax.set_xlabel('Decoder layer'); ax.grid(alpha=.15); ax.legend(fontsize=7)
                if si == 0: ax.set_ylabel('S_g' if scale == 'raw' else 'log(1 + S_g)')
        fig.suptitle(f'ALL 4000 IMAGES / all mentions / {scale}: solid mean; dashed median; shade IQR (not CI)', fontsize=12)
        fig.tight_layout(); fig.savefig(OUT/f'all_{scale}.png', dpi=160)
        fig.savefig(OUT/f'all_{scale}.pdf'); plt.close(fig)
    base.write_csv(rows, OUT/'curves.csv')


def summarize():
    rows, seed_rows, comparisons = [], [], []
    lines = ['# 完整来源强度：原始值与log1p检测', '',
             'S_g=sum_group ||c_j||，不除全部来源总量或token数。分组prompt含模板和特殊token，generation仅含已生成前缀，visual为S_C。',
             '每套5组：prompt / generation / visual 单独；concat三组全层向量；visual_prompt_sum=S_V+S_P。log版本先相加再log1p。',
             '四模型原4000图、3200/800、全部mentions，MLP/seeds43/44/45保持，共120头。每格为三seed均值±总体标准差，单位%。',
             'HALL-F1使用每seed训练REAL-F1阈值；原双阈值详见CSV。无超参数搜索或bootstrap。F_E/F_C及份额拼接仅引用已有基线。', '',
             '| 模型 | 信号 | AUROC | HALL-AUPR | HALL-F1 |', '|---|---|---:|---:|---:|']
    for model, title in zip(MODELS, TITLES):
        results = json.loads((OUT/model/'detection.json').read_text())
        old = json.loads((base.OUT/model/'detection.json').read_text())
        share = json.loads((base.OUT/'source_share_detection'/model/'detection.json').read_text())
        results.update(F_E=old['F'], F_C=old['F_C'], share_concat=share['prompt_generation_visual'])
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
                    seed_rows.append(dict(model=model, group=name, seed=seed, threshold_rule=rule,
                        threshold=report['threshold'], AUROC=m['auc'], HALL_AUPR=h['aupr'],
                        HALL_precision=h['precision'], HALL_recall=h['recall'], HALL_F1=h['f1']))
        selected = {r['group']: r for r in rows if r['model'] == model}
        pairs = [(f'log1p_{name}', f'raw_{name}') for name in KINDS]
        for scale in SCALES:
            pairs += [(f'{scale}_visual_prompt_sum', f'{scale}_visual'),
                      (f'{scale}_concat', f'{scale}_visual'), (f'{scale}_concat', 'share_concat')]
        for left, right in pairs:
            comparisons.append(dict(model=model, left=left, right=right,
                AUROC_delta=selected[left]['AUROC_mean']-selected[right]['AUROC_mean'],
                HALL_AUPR_delta=selected[left]['HALL_AUPR_mean']-selected[right]['HALL_AUPR_mean']))
    lines += ['', '结果解读：四模型在raw/log1p两套中，concat平均AUROC均高于其余四组。'
              'log1p使单独prompt和visual的AUROC/AP在四模型都提高，但concat仅Qwen2的AUROC略升，另外三模型下降；concat的AP均略降或下降。'
              'V+P相加没有跨模型一致超过单独visual：raw在LLaVA/InternVL更高、两个Qwen更低；log1p仅LLaVA更高（按AUROC）。',
              '原始concat对照AE+S：Qwen3/InternVL的AUROC/AP均超过F_E/F_C；LLaVA与F_E接近且略低于F_C；Qwen2仍低于两基线。'
              '相对份额concat，原始concat在LLaVA/Qwen3/InternVL双指标提高，Qwen2下降。以上均为点估计，不宣称显著性或因果，concat输入维度为单组3倍。', '',
              '曲线为每模型4000图全部mentions等权：先对每条mention计算S或log1p(S)，再按REAL/HALL统计。',
              '![原始强度](all_raw.png)', '', '![log1p强度](all_log1p.png)', '',
              '模型的matrices.pt、protocol.json、heads、detection.json保存在各子目录；detection.csv为均值/std，seed_metrics.csv为逐seed双阈值，comparisons.csv为配对均值差。',
              '原汇总器JSON含ensemble字段，本报告不使用ensemble。长度/位置混杂及原K50来源重建误差限制仍适用。',
              '训练：`python scripts/train_ffn_source_strengths.py --models <模型列表> --device cuda:0`；完成后`--summarize`生成表格和曲线。']
    base.write_csv(rows, OUT/'detection.csv'); base.write_csv(seed_rows, OUT/'seed_metrics.csv')
    base.write_csv(comparisons, OUT/'comparisons.csv')
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n')
    plot()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=MODELS, default=MODELS)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--summarize', action='store_true')
    args = parser.parse_args()
    if args.summarize:
        summarize()
    else:
        check_groups()
        for model in args.models:
            run(model, args.device)
