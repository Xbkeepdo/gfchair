"""Plot prompt/generated-prefix FFN source norms from the existing K50 cache."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'outputs/ffn_source_composition_v1'
OUT = SOURCE/'text_source_curves'
MODELS = ('qwen2_5_vl_7b', 'llava_1_5_7b', 'qwen3_vl_8b', 'internvl_2_5_8b')
TITLES = ('Qwen2.5-VL', 'LLaVA-1.5', 'Qwen3-VL', 'InternVL-2.5')
SOURCES = ('prompt', 'generation', 'visual')


def split_norms(row):
    start, end = row['visual_range']
    query, generated = row['prediction_position'], row['response_index']
    generation_start = query-generated+1
    assert 0 <= start < end <= generation_start
    # other_token_mag keeps sequence order after removing the visual interval.
    cut = generation_start-(end-start)
    stop = cut+generated
    other = row['other_token_mag'].double().numpy()
    assert stop <= other.shape[1]
    assert not np.any(other[:, stop:]), 'Future tokens must have zero causal contribution'
    gross = np.stack((other[:, :cut].sum(-1), other[:, cut:stop].sum(-1),
                      np.asarray(row['S_C'], dtype=np.float64)))
    counts = np.array([cut, generated, end-start])
    np.testing.assert_allclose(gross.sum(0)+np.asarray(row['residual_mag'])+
                               np.asarray(row['attention_bias_mag'])+np.asarray(row['ffn_zero_norm']),
                               row['gross_all'], rtol=2e-5, atol=1e-5)
    return gross, counts


def self_check():
    # Original positions: prompt 0, visual 1:3, prompt 3, generated 4:6, future 6.
    row = dict(visual_range=[1, 3], prediction_position=5, response_index=2,
               other_token_mag=torch.tensor([[1., 2., 3., 4., 0.]]),
               S_C=[5.], residual_mag=[6.], attention_bias_mag=[0.], ffn_zero_norm=[1.], gross_all=[22.])
    gross, counts = split_norms(row)
    np.testing.assert_array_equal(gross[:, 0], [3, 7, 5])
    np.testing.assert_array_equal(counts, [2, 2, 2])
    row.update(prediction_position=3, response_index=0,
               other_token_mag=torch.tensor([[1., 2., 0., 0., 0.]]), gross_all=[15.])
    gross, counts = split_norms(row)
    assert gross[1, 0] == counts[1] == 0


def collect(model):
    cached = torch.load(SOURCE/model/'matrices.pt', map_location='cpu', weights_only=False)
    targets = {}
    files = sorted((SOURCE/model/'shards/k50').glob('image_*.pt'))
    assert len(files) == 4000
    for i, path in enumerate(files, 1):
        shard = torch.load(path, map_location='cpu', weights_only=False)
        for row in shard['positions']:
            gross, counts = split_norms(row)
            targets[row['target_key']] = (gross, counts, np.asarray(row['gross_all']))
        if i % 500 == 0:
            print(model, i, '/ 4000', flush=True)
    ordered = [targets[m['target_key']] for m in cached['mentions']]
    gross, counts, total = [np.stack([r[i] for r in ordered]) for i in range(3)]
    # Empty generation has zero gross/fraction; per-token strength is undefined.
    per_token = np.divide(gross, counts[:, :, None], out=np.full_like(gross, np.nan),
                          where=counts[:, :, None] > 0)
    fraction = np.divide(gross, total[:, None, :], out=np.full_like(gross, np.nan),
                         where=total[:, None, :] > 0)
    return dict(gross=gross, per_token=per_token, fraction=fraction, counts=counts,
                y=np.asarray(cached['y']), ntrain=cached['ntrain'])


def main():
    self_check()
    OUT.mkdir(parents=True, exist_ok=True)
    data = {model: collect(model) for model in MODELS}
    rows = []
    for split in ('all', 'train', 'test'):
        for metric, ylabel in (('gross', 'Sum of source contribution norms'),
                               ('per_token', 'Mean contribution norm per source token'),
                               ('fraction', 'Share of all-source gross (incl. residual/bias)')):
            fig, axes = plt.subplots(4, 3, figsize=(15, 12))
            for mi, (model, title) in enumerate(zip(MODELS, TITLES)):
                d = data[model]
                indices = {'all': slice(None), 'train': slice(None, d['ntrain']),
                           'test': slice(d['ntrain'], None)}[split]
                for si, source in enumerate(SOURCES):
                    ax = axes[mi, si]
                    for label, name, color in ((0, 'HALL', '#d55e00'), (1, 'REAL', '#0072b2')):
                        v = d[metric][indices, si][d['y'][indices] == label]
                        v = v[np.isfinite(v).all(1)]
                        mean, median = v.mean(0), np.median(v, axis=0)
                        q25, q75 = np.quantile(v, [.25, .75], axis=0)
                        layers = np.arange(1, len(mean)+1)
                        ax.plot(layers, mean, color=color, label=f'{name} (n={len(v)})')
                        ax.plot(layers, median, '--', color=color, alpha=.6, linewidth=1)
                        ax.fill_between(layers, q25, q75, color=color, alpha=.12)
                        rows.extend(dict(model=model, split=split, metric=metric, source=source,
                                         label=label, layer=int(l), n=len(v), mean=float(a),
                                         median=float(b), q25=float(c), q75=float(e))
                                    for l, a, b, c, e in zip(layers, mean, median, q25, q75))
                    ax.set_title(f'{title} / {source}')
                    ax.set_xlabel('Decoder layer'); ax.grid(alpha=.15); ax.legend(fontsize=7)
                    if si == 0: ax.set_ylabel(ylabel, fontsize=8)
            scope = 'ALL 4000 IMAGES / all mentions' if split == 'all' else split.upper()
            fig.suptitle(f'{scope} / {metric}: solid mean; dashed median; shade IQR (not CI)', fontsize=12)
            fig.tight_layout()
            fig.savefig(OUT/f'{split}_{metric}.png', dpi=160)
            fig.savefig(OUT/f'{split}_{metric}.pdf')
            plt.close(fig)
    with (OUT/'curves.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    metadata = {}
    for model, d in data.items():
        metadata[model] = {}
        for split, ix in (('all', slice(None)), ('train', slice(None, d['ntrain'])), ('test', slice(d['ntrain'], None))):
            for label, name in ((0, 'HALL'), (1, 'REAL')):
                counts = d['counts'][ix][d['y'][ix] == label]
                metadata[model][f'{split}_{name}'] = dict(mentions=len(counts),
                    mean_token_counts=dict(zip(SOURCES, counts.mean(0).tolist())),
                    empty_generation=int((counts[:, 1] == 0).sum()))
    (OUT/'counts.json').write_text(json.dumps(metadata, indent=2)+'\n')
    lines = ['# Prompt / generation 完整来源贡献曲线', '',
             '复用四模型原4000图K50缓存和原3200/800划分，按全部mentions统计；不重跑模型或训练检测器。', '',
             '主图合并全部4000图，每条mention等权（非每图等权）；先逐mention计算比值，再按REAL/HALL统计。另保留train/test分图。', '',
             '- prompt包含全部非视觉提示位置（含system、聊天模板及特殊token），不等同于纯instruction。',
             '- generation仅含预测目标之前已生成的前缀；包含当前query token，不含待预测目标与未来。',
             '- gross = 组内逐来源 ||c_j|| 相加；per_token = gross / 该组token数；fraction = gross / gross_all。',
             '- gross_all包含视觉、全部文本、residual、attention输出偏置和FFN(0)范数；三条来源份额不要求相加为1。',
             '- 空generation的gross和fraction为0，per_token未定义并排除；数量见counts.json。',
             '- 实线均值、虚线中位数、阴影IQR（非置信区间）；是来源位置归因，不是纯模态信息或因果效应。',
             '- 本次只有FFN贡献曲线；非视觉raw attention未保存，不能由范数恢复。原K50重建误差限制保留。', '',
             '| 模型 | 来源 | 全4000图gross：REAL均值>HALL层数 | per_token：REAL均值>HALL层数 |',
             '|---|---|---:|---:|']
    for model, title in zip(MODELS, TITLES):
        for source in SOURCES:
            cells = []
            for metric in ('gross', 'per_token'):
                groups = [[r['mean'] for r in rows if r['model'] == model and r['source'] == source
                           and r['metric'] == metric and r['split'] == 'all' and r['label'] == label] for label in (0, 1)]
                cells.append(f'{int((np.array(groups[1]) > groups[0]).sum())}/{len(groups[0])}')
            lines.append(f'| {title} | {source} | {cells[0]} | {cells[1]} |')
    lines += ['', '四模型prompt在train/test全部层均REAL均值更高。generation的gross几乎各层HALL更高，'
              '但逐样本除以前缀token数后，四模型全部层均REAL更高。HALL前缀明显更长，'
              '总强度不能直接解释成更依赖生成文本；除以长度也不等于完成位置/长度匹配或证明因果。', '',
              '| 模型 | 全量REAL平均生成前缀token数 | 全量HALL平均生成前缀token数 |', '|---|---:|---:|']
    for model, title in zip(MODELS, TITLES):
        m = metadata[model]
        lines.append(f"| {title} | {m['all_REAL']['mean_token_counts']['generation']:.2f} | "
                     f"{m['all_HALL']['mean_token_counts']['generation']:.2f} |")
    lines += ['', '![4000图贡献总强度](all_gross.png)', '', '![4000图每token强度](all_per_token.png)', '',
              '![4000图来源范数份额](all_fraction.png)', '',
              '训练/测试集对应train_*.png/pdf和test_*.png/pdf；完整逐层统计curves.csv；PNG与PDF均已保存。', '',
              '运行：`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /opt/conda/private/envs/vicr/bin/python scripts/plot_ffn_text_sources.py`。',
              '检查：合成边界例子（含空generation）、全量未来位置范数为0、分组加residual/bias/FFN(0)范数重建gross_all。']
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n')
    print('Saved', OUT, flush=True)


if __name__ == '__main__':
    main()
