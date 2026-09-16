"""Paired image-cluster intervals and reviewable reports for cached I/S ablations."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from features.tc_fvpa_artifacts import atomic_json_save
from scripts.analyze_all_attention_write_gain import OUT, MODELS, SEEDS, trainer, write_csv

PAIRS = (('S', 'I'), ('I+S', 'I'), ('I+S', 'S'))
METRICS = ('AUROC', 'HALL_AUPR')


def read_csv(path):
    with path.open() as f: return list(csv.DictReader(f))


def weighted_scores(y, p, weights):
    return [roc_auc_score(y, p, sample_weight=weights),
            average_precision_score(1-y, 1-p, sample_weight=weights)]


def bootstrap(model):
    directory = OUT/model
    protocol = json.loads((directory/'protocol.json').read_text())
    destination = directory/'bootstrap.json'
    if destination.exists():
        saved = json.loads(destination.read_text())
        assert saved['fingerprint'] == protocol['fingerprint'] and saved['replicates'] == 2000
        return saved['rows']
    data = trainer.read(directory/'statistics.pt')
    mask = data['masks']['test']; y = data['y'][mask]
    image_ids = np.asarray([m['image_id'] for m in data['mentions']])[mask]
    # Includes test images with no mentions; matches the actual 400-image sampling frame.
    images = np.asarray(sorted(protocol['split']['test']))
    mapping = {image: i for i, image in enumerate(images)}
    cluster = np.asarray([mapping[i] for i in image_ids])
    p = np.stack([np.stack([trainer.read(directory/support/feature/f'seed{seed}.pt')['test_probabilities']
                             for seed in SEEDS]) for support in ('PVG', 'ALL') for feature in ('I', 'S', 'I+S')])
    base = np.asarray([[weighted_scores(y, prediction, np.ones(len(y))) for prediction in group] for group in p]).mean(1)
    rng = np.random.default_rng(20260915)
    distributions = []
    started = time.monotonic()
    for rep in range(2000):
        counts = np.bincount(rng.integers(len(images), size=len(images)), minlength=len(images))
        weights = counts[cluster]
        if not all(weights[y == label].sum() > 0 for label in (0, 1)): continue
        scores = np.asarray([[weighted_scores(y, prediction, weights) for prediction in group] for group in p]).mean(1)
        distributions.append(scores)
        if (rep+1) % 250 == 0: print(model, 'bootstrap', rep+1, '/2000', flush=True)
    distributions = np.asarray(distributions)
    rows = []
    for support_index, support in enumerate(('PVG', 'ALL')):
        for a, b in PAIRS:
            ia, ib = [support_index*3+('I', 'S', 'I+S').index(k) for k in (a, b)]
            for metric_index, metric in enumerate(METRICS):
                delta = distributions[:, ia, metric_index]-distributions[:, ib, metric_index]
                low, high = np.quantile(delta, [.025, .975])
                rows.append(dict(model=model, support=support, comparison=a+' minus '+b, metric=metric,
                    difference=base[ia, metric_index]-base[ib, metric_index], low=low, high=high,
                    valid_replicates=len(delta), test_images=len(images), images_with_mentions=len(set(image_ids))))
    atomic_json_save(dict(fingerprint=protocol['fingerprint'], replicates=2000, seed=20260915,
        seconds=time.monotonic()-started, rows=rows), destination)
    write_csv(directory/'bootstrap.csv', rows)
    return rows


def summarize():
    detection, intervals, mechanisms = [], [], []
    for model in MODELS:
        directory = OUT/model
        rows = read_csv(directory/'detection.csv')
        assert len(rows) == 18
        checks = json.loads((directory/'validation.json').read_text())
        assert len(checks) == 18 and all(c['status']=='PASS' for c in checks)
        for group in sorted(set(r['group'] for r in rows)):
            selected = [r for r in rows if r['group'] == group]
            assert sorted(int(r['seed']) for r in selected) == list(SEEDS)
            row = dict(model=model, group=group)
            for metric in METRICS:
                values = [float(r[metric]) for r in selected]
                row.update({metric+'_mean':np.mean(values), metric+'_std':np.std(values)})
            detection.append(row)
        intervals.extend(json.loads((directory/'bootstrap.json').read_text())['rows'])
        curves = read_csv(directory/'curves.csv')
        relationships = read_csv(directory/'write_gain_decomposition.csv')
        paired = read_csv(directory/'paired_images.csv')
        for scope in ('all', 'train', 'validation', 'test'):
            for region in ('prompt', 'visual', 'generation', 'all'):
                row = dict(model=model, scope=scope, region=region)
                for metric in ('I', 'S', 'G', 'lambda_median', 'lambda_iqr', 'lambda_cv'):
                    for label in ('REAL', 'HALL'):
                        selected = [r for r in curves if r['scope']==scope and r['region']==region and r['metric']==metric and r['label']==label]
                        row[metric+'_'+label] = np.nanmean([float(r['mean']) for r in selected])
                selected = [r for r in relationships if r['scope']==scope and r['region']==region]
                for metric in ('spearman_I_S', 'log_pearson_I_S', 'delta_log_I', 'delta_log_S', 'delta_log_G'):
                    row[metric] = np.nanmean([float(r[metric]) for r in selected])
                np.testing.assert_allclose(row['delta_log_S'], row['delta_log_I']+row['delta_log_G'], atol=1e-12)
                for metric in ('log_I', 'log_S', 'log_G'):
                    selected = [r for r in paired if r['scope']==scope and r['region']==region and r['metric']==metric]
                    row['paired_delta_'+metric] = np.nanmean([float(r['mean_H_minus_R']) for r in selected])
                mechanisms.append(row)
    write_csv(OUT/'detection_summary.csv', detection)
    write_csv(OUT/'bootstrap.csv', intervals)
    write_csv(OUT/'mechanism_summary.csv', mechanisms)
    atomic_json_save(dict(detection=detection, bootstrap=intervals, mechanism=mechanisms), OUT/'summary.json')
    lines = ['# All-attention WRITE / SS / gain：前三项完成（2026-09-15）', '',
        '四模型各4000图，仅复用真实RMS全attention K32缓存；本轮没有重新提取VLM特征。路径z−A_all→z，residual保持在背景中，不是B1完整残差来源路径。', '',
        '## 数据与协议', '',
        'I=Σ||a_m||，S=Σ||e_m||，G=S/I。P/V/G/ALL分别统计。lambda分位数/IQR/CV先在同一target-layer-region的token内计算，再按mentions分REAL/HALL汇总；跨mentions IQR另存，不能混用。', '',
        '检测保留全部mentions、原811图片划分3200/400/400；输入log1p(I)、log1p(S)及其拼接，train-only逐列Z-score。统一单隐藏128/ReLU/dropout.3、无BN、Adam lr.001/wd1e-5/batch128、最多150epoch、早停20，最低val BCE checkpoint；全部模型和特征不单独调参。seeds43/44/45，报告seed指标均值±总体std，不是ensemble。PVG保留三来源后拼接，ALL先合并attention token。', '',
        '## 检测结果（%，AUROC / HALL-AUPR）', '',
        '| 模型 | 来源组织 | I-only | S-only | I+S |', '|---|---|---:|---:|---:|']
    for model in MODELS:
        for support in ('PVG', 'ALL'):
            cells = []
            for feature in ('I', 'S', 'I+S'):
                row = next(r for r in detection if r['model']==model and r['group']==support+'/'+feature)
                cells.append(' / '.join(f"{100*row[m+'_mean']:.2f}±{100*row[m+'_std']:.2f}" for m in METRICS))
            lines.append('| '+model+' | '+support+' | '+' | '.join(cells)+' |')
    lines += ['', '## 配对检测增量（百分点，名义95%图片bootstrap区间）', '',
        '固定400测试图片簇重采样2000次，每次先分别算三seed指标差再平均。包括无mention测试图片的抽样框架；没有多重比较校正，已有测试集被历史实验查看，因此仅为探索性比较。', '',
        '| 模型 | 来源组织 | 比较 | ΔAUROC [95% CI] | ΔHALL-AUPR [95% CI] |', '|---|---|---|---:|---:|']
    for model in MODELS:
        for support in ('PVG', 'ALL'):
            for a, b in PAIRS:
                cells = []
                for metric in METRICS:
                    row = next(r for r in intervals if r['model']==model and r['support']==support and r['comparison']==a+' minus '+b and r['metric']==metric)
                    cells.append(f"{100*row['difference']:+.2f} [{100*row['low']:+.2f}, {100*row['high']:+.2f}]")
                lines.append('| '+model+' | '+support+' | '+a+' − '+b+' | '+' | '.join(cells)+' |')
    lines += ['', '## WRITE与gain的差异分解（全部4000图，层等权）', '',
        '以下Δ为HALL−REAL平均log值差；逐目标满足logS=logI+logG。相关为每层跨mentions计算再平均，不能等同于因果解释或方差解释比例。具体层和train/validation/test结果另存CSV。', '',
        '| 模型 | 来源 | Spearman(I,S) | Δlog I | Δlog S | Δlog G |', '|---|---|---:|---:|---:|---:|']
    for row in mechanisms:
        if row['scope']=='all':
            lines.append('| '+row['model']+' | '+row['region']+' | '+' | '.join(f'{row[k]:.5f}' for k in ('spearman_I_S','delta_log_I','delta_log_S','delta_log_G'))+' |')
    lines += ['', '## 实际token方向上的gain离散度（REAL / HALL）', '',
        '表内先算每个target-layer内的统计量，再跨mentions和层等权平均。IQR非零表示同一条件路径上实际source directions的范数增益不同；这不是完整Jacobian奇异值谱，也不说明这种差异导致幻觉。', '',
        '| 模型 | 来源 | lambda中位数 | token内IQR | token内CV |', '|---|---|---:|---:|---:|']
    for row in mechanisms:
        if row['scope']=='all':
            cells = [f"{row[k+'_REAL']:.5f} / {row[k+'_HALL']:.5f}" for k in ('lambda_median','lambda_iqr','lambda_cv')]
            lines.append('| '+row['model']+' | '+row['region']+' | '+' | '.join(cells)+' |')
    lines += ['', '## 图与核验', '',
        'I/S/G图实线为均值、虚线为中位数、阴影为跨mentions IQR（不是CI）；lambda图分别展示token内median/IQR/CV的跨mentions分布。', '']
    for model in MODELS:
        lines.append(f'- {model}：[I/S/G全部样本]({model}/ISG_all.png)、[test]({model}/ISG_test.png)、[lambda全部样本]({model}/lambda_all.png)、[test]({model}/lambda_test.png)、[缓存核验]({model}/cache_audit.json)、[18头CPU重载]({model}/validation.json)。均有PDF。')
    lines += ['', '来源文件manifest、协议/特征指纹、逐层曲线、同图控制、log差异分解及72个checkpoint全部保存。无新VLM forward，无第四项冻结RMS对照。']
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n')
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for si, support in enumerate(('PVG', 'ALL')):
        for mi, metric in enumerate(METRICS):
            ax = axes[si, mi]
            for fi, (feature, color) in enumerate(zip(('I', 'S', 'I+S'), ('#2878b5', '#e68a2e', '#4d9f50'))):
                rows = [next(r for r in detection if r['model']==model and r['group']==support+'/'+feature) for model in MODELS]
                ax.bar(np.arange(4)+(fi-1)*.25, [100*r[metric+'_mean'] for r in rows], width=.24,
                       yerr=[100*r[metric+'_std'] for r in rows], label=feature, color=color, capsize=3)
            ax.set_xticks(np.arange(4)); ax.set_xticklabels(['Qwen2.5', 'LLaVA', 'Qwen3', 'InternVL'])
            ax.set_title(support+' | '+metric); ax.set_ylabel('%'); ax.legend(); ax.grid(axis='y', alpha=.2)
            ax.set_ylim(0, 100)
    fig.suptitle('True-RMS all-attention K32 | fixed 811 | 3-seed mean ± population std')
    fig.tight_layout()
    for ext in ('png', 'pdf'): fig.savefig(OUT/f'detection.{ext}', dpi=160)
    plt.close(fig)
    # Show source-token quartiles directly, distinct from across-mention bands.
    for scope in ('all', 'test'):
        fig, axes = plt.subplots(4, 4, figsize=(19, 14), squeeze=False)
        for mi, model in enumerate(MODELS):
            curves = read_csv(OUT/model/'curves.csv')
            for ri, region in enumerate(('prompt', 'visual', 'generation', 'all')):
                ax = axes[mi, ri]
                for label, color in (('REAL', '#2878b5'), ('HALL', '#d95319')):
                    selected = {metric: [r for r in curves if r['scope']==scope and r['region']==region and r['label']==label and r['metric']==metric]
                                for metric in ('lambda_q25', 'lambda_median', 'lambda_q75')}
                    x = [int(r['layer']) for r in selected['lambda_median']]
                    ax.plot(x, [float(r['mean']) for r in selected['lambda_median']], label=label, color=color)
                    ax.fill_between(x, [float(r['mean']) for r in selected['lambda_q25']],
                                    [float(r['mean']) for r in selected['lambda_q75']], color=color, alpha=.15)
                ax.set_title(model.replace('_vl_', '_')+' | '+region, fontsize=10)
                ax.set_yscale('log'); ax.set_xlabel('Layer'); ax.grid(alpha=.2)
        axes[0, 0].legend()
        fig.suptitle(f'True-RMS all-attention K32 | {scope} | source-direction gain lambda\nLine: mean of within-target medians; band: mean of within-target Q25 to mean of Q75 (not pooled quantiles or CI).')
        fig.tight_layout(rect=(0, 0, 1, .95))
        for ext in ('png', 'pdf'): fig.savefig(OUT/f'lambda_source_quantiles_{scope}.{ext}', dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    for model in MODELS:
        while not (OUT/model/'validation.json').exists() or len(json.loads((OUT/model/'validation.json').read_text())) != 18:
            if not args.wait: raise RuntimeError('Incomplete model: '+model)
            time.sleep(15)
        bootstrap(model)
    summarize()
    print('All four models summarized', flush=True)


if __name__ == '__main__': main()
