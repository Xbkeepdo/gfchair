"""Validate saved regional AE detectors and report all preset comparisons."""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import ParameterGrid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import train_all_attention_ae_strength as study
from scripts.run_ffn_source_composition import read, write_csv
from features.tc_fvpa_artifacts import atomic_json_save

OUT = study.OUT
NAMES = dict(zip(study.MODELS, ('Qwen2.5', 'LLaVA', 'Qwen3', 'InternVL')))
CLASSIFIERS = ('three_hidden', 'one_hidden', 'xgb')
CLASS_NAMES = dict(zip(CLASSIFIERS, ('三隐藏层 MLP', '单隐藏层 MLP', 'XGB')))
SHORT = dict(zip(study.KINDS, ('V', 'VP', 'G', 'VP + G')))


def load_json(path):
    return json.loads(path.read_text())


def validate():
    """Recompute metrics independently from all saved probabilities, including controls."""
    seed_rows, checks, selections, trials = [], [], [], []
    for model in study.MODELS:
        root = OUT/model
        data = read(root/'matrices.pt')
        original = read(ROOT/'outputs/ffn_source_composition_v1'/model/'matrices.pt')
        n, y = data['ntrain'], np.asarray(data['y'])
        assert n == original['ntrain']
        np.testing.assert_array_equal(y, original['y'])
        assert data['mentions'] == original['mentions']
        protocol = load_json(root/'protocol.json')
        status = load_json(root/'status.json')
        assert status['status'] == 'COMPLETE' and status['images'] == 4000
        assert status['signature'] == protocol['ae_signature']
        assert len(list((root/'ae_shards').glob('image_*.pt'))) == 4000
        assert set(data['groups']) == set(study.GROUPS)
        ae = data['groups']['ae_only_visual']
        old_ae = original['groups']['F'][:, :study.LAYERS[model]]
        err = np.abs(ae-old_ae)
        assert np.all(err <= 1e-4+1e-3*np.abs(old_ae)), 'Full-data legacy AE parity failed'
        train_ids = {m['image_id'] for m in data['mentions'][:n]}
        test_ids = {m['image_id'] for m in data['mentions'][n:]}
        assert not train_ids & test_ids
        assert train_ids <= set(protocol['image_split']['train']+protocol['image_split']['validation'])
        assert test_ids <= set(protocol['image_split']['test'])
        fixed = load_json(root/'detection.json')
        model_heads = 0
        for group in study.GROUPS:
            for classifier in CLASSIFIERS:
                folder = root/'heads'/group if classifier == 'three_hidden' else root/classifier/group
                entry = fixed[group] if classifier == 'three_hidden' else load_json(folder/'detection.json')
                assert set(entry['per_seed_metrics']) == {str(s) for s in study.SEEDS}
                if classifier != 'three_hidden':
                    selected = load_json(folder/'selection.json')
                    grid = list(ParameterGrid(study.search.GRID if classifier == 'one_hidden' else study.search.TREE_GRIDS['xgb']))
                    candidates = [load_json(folder/'search'/f'trial{i:02d}.json') for i in range(len(grid))]
                    for index, candidate in enumerate(candidates):
                        assert candidate['params'] == grid[index]
                        assert candidate['index'] == index and candidate['seed'] == 43
                        assert candidate['scope'] == 'inner_validation'
                        assert np.isfinite([candidate['validation_AUROC'], candidate['validation_HALL_AUPR']]).all()
                        trials.append(dict(model=model, group=group, classifier=classifier,
                            index=index, validation_AUROC=candidate['validation_AUROC'],
                            validation_HALL_AUPR=candidate['validation_HALL_AUPR'],
                            seconds=candidate['seconds'], n_iter=candidate.get('n_iter', ''),
                            convergence_warning=any(w['category']=='ConvergenceWarning' for w in candidate['warnings']),
                            params=json.dumps(candidate['params'])))
                    best = max(candidates, key=lambda c: (c['validation_AUROC'], c['validation_HALL_AUPR'], -c['index']))
                    assert selected['best'] == best and selected['candidates'] == candidates
                    selections.append(dict(model=model, group=group, classifier=classifier,
                        params=json.dumps(best['params']), validation_AUROC=best['validation_AUROC'],
                        validation_HALL_AUPR=best['validation_HALL_AUPR']))
                for seed in study.SEEDS:
                    head = read(folder/f'seed{seed}'/'result.pt')
                    assert head['seed'] == seed
                    if classifier != 'three_hidden':
                        assert head['params'] == best['params']
                    p = np.asarray(head['test_probabilities'])
                    train_p = np.asarray(head['train_probabilities'])
                    assert p.shape == y[n:].shape and train_p.shape == y[:n].shape
                    assert np.isfinite(p).all() and ((p >= 0)&(p <= 1)).all()
                    assert np.isfinite(train_p).all() and ((train_p >= 0)&(train_p <= 1)).all()
                    auc = float(roc_auc_score(y[n:], p))
                    ap = float(average_precision_score(1-y[n:], 1-p))
                    for metrics in (head['metrics'], entry['per_seed_metrics'][str(seed)]):
                        metric = metrics['threshold_reports']['fixed_0.5']['test_metrics']
                        assert abs(auc-metric['auc']) < 1e-10
                        assert abs(ap-metric['hallucination_positive']['aupr']) < 1e-10
                    seed_rows.append(dict(model=model, group=group, classifier=classifier, seed=seed,
                        AUROC=auc, HALL_AUPR=ap,
                        convergence_warning=any(w['category']=='ConvergenceWarning' for w in head.get('warnings', []))))
                    model_heads += 1
        assert model_heads == 72
        checks.append(dict(model=model, images=4000, mentions=len(y), train_mentions=n,
            test_mentions=len(y)-n, test_hall=int((y[n:]==0).sum()),
            empty_generation_mentions=sum(m['response_index']==0 for m in data['mentions']),
            formal_heads=model_heads, visual_ae_max_error=float(err.max()),
            visual_ae_mean_error=float(err.mean()), fingerprint=data['fingerprint'],
            smoke=load_json(root/'smoke8/status.json')['visual_parity']))
    assert len(seed_rows) == 288 and len(selections) == 64 and len(trials) == 960
    write_csv(seed_rows, OUT/'seed_metrics.csv')
    write_csv(selections, OUT/'selected_params.csv')
    write_csv(trials, OUT/'search_trials.csv')
    atomic_json_save(dict(status='PASS', models=checks, heads=288, candidates=960,
        validation_selections=64, metric_tolerance=1e-10,
        packages={p:version(p) for p in ('torch', 'numpy', 'scikit-learn', 'xgboost')},
        checks=['all original mentions and labels in canonical order', '4000 complete image shards/model',
                'full-data visual AE matches legacy tolerance', 'inner/outer image separation',
                'all 960 frozen grid candidates and validation-only choices',
                'all 288 saved-probability AUROC/HALL-AUPR match summaries']), OUT/'validation.json')
    return seed_rows, checks


def comparisons(rows):
    lookup = {(r['model'], r['group'], r['classifier']): r for r in rows}
    old_path = ROOT/'outputs/ffn_all_source_paths_v1/detection_comparison_20260913/historical.csv'
    with old_path.open() as stream:
        old = {(r['model'], r['experiment'], r['group'], r['classifier']): r for r in csv.DictReader(stream)}
    deltas = []
    for model in study.MODELS:
        for group in study.KINDS:
            for classifier in CLASSIFIERS:
                new = lookup[model, group, classifier]
                controls = [('AE only', lookup[model, 'ae_only_'+group, classifier])]
                if classifier != 'three_hidden':
                    controls.append(('three_hidden same feature', lookup[model, group, 'three_hidden']))
                for reference, control in controls:
                    deltas.append(dict(model=model, group=group, classifier=classifier, reference=reference,
                        AUROC_delta_pp=100*(new['AUROC_mean']-control['AUROC_mean']),
                        HALL_AUPR_delta_pp=100*(new['HALL_AUPR_mean']-control['HALL_AUPR_mean']),
                        reference_AUROC=control['AUROC_mean'], reference_HALL_AUPR=control['HALL_AUPR_mean']))
                old_classifier = 'fixed_three_hidden' if classifier == 'three_hidden' else classifier
                historical = [('legacy U + frozen-RMS S', old[model, 'old_cross_scale', 'gated_'+group, old_classifier])]
                if group == 'visual' and classifier == 'three_hidden':
                    historical.append(('legacy AE_V + visual-only S_E', old[model, 'old_composition', 'F', 'fixed_three_hidden']))
                for reference, control in historical:
                    deltas.append(dict(model=model, group=group, classifier=classifier, reference=reference,
                        AUROC_delta_pp=100*(new['AUROC_mean']-float(control['auc'])),
                        HALL_AUPR_delta_pp=100*(new['HALL_AUPR_mean']-float(control['hall_aupr'])),
                        reference_AUROC=float(control['auc']), reference_HALL_AUPR=float(control['hall_aupr'])))
    write_csv(deltas, OUT/'all_comparisons.csv')
    return deltas


def figures(rows, deltas):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    lookup = {(r['model'], r['group'], r['classifier']): r for r in rows}
    delta = {(r['model'], r['group'], r['classifier']): r for r in deltas if r['reference']=='AE only'}
    colors = ('#3976b7', '#ed963d', '#45966b')
    labels = ('3-hidden MLP', '1-hidden MLP', 'XGBoost')
    x = np.arange(4)
    for name in ('performance', 'gain_over_ae'):
        fig, axes = plt.subplots(4, 2, figsize=(12, 13), constrained_layout=True)
        for i, model in enumerate(study.MODELS):
            for j, metric in enumerate(('AUROC', 'HALL_AUPR')):
                ax = axes[i, j]
                for c, classifier in enumerate(CLASSIFIERS):
                    if name == 'performance':
                        values = [100*lookup[model,g,classifier][metric+'_mean'] for g in study.KINDS]
                        errors = [100*lookup[model,g,classifier][metric+'_std'] for g in study.KINDS]
                    else:
                        values = [delta[model,g,classifier][metric+'_delta_pp'] for g in study.KINDS]
                        errors = None
                    ax.bar(x+(c-1)*.24, values, width=.23, color=colors[c], label=labels[c],
                           yerr=errors, capsize=2, zorder=3)
                ax.set_xticks(x, list(SHORT.values()))
                ax.set_title(NAMES[model]+' | '+metric.replace('_', '-'))
                ax.set_ylabel('% (mean ± seed std)' if name=='performance' else 'Fusion minus AE only (pp)')
                ax.grid(axis='y', alpha=.25, zorder=0)
                if name == 'performance':
                    minimum = min(100*lookup[model,g,c][metric+'_mean'] for g in study.KINDS for c in CLASSIFIERS)
                    ax.set_ylim(max(0, 5*np.floor((minimum-5)/5)), 100)
                else:
                    ax.axhline(0, color='#444444', linewidth=.8)
                if i == 0 and j == 0:
                    ax.legend(fontsize=8)
        fig.suptitle('Regional AE + log1p(All-attention gross) | Original 800-image test', fontsize=14)
        fig.savefig(OUT/(name+'.png'), dpi=180)
        fig.savefig(OUT/(name+'.pdf'))
        plt.close(fig)


def report(rows, deltas, checks):
    lookup = {(r['model'], r['group'], r['classifier']): r for r in rows}
    rel = '../outputs/all_attention_ae_log_strength_v1/'
    lines = ['# 区域 AE＋All-attention gross：三隐藏层、单隐藏层 MLP 与 XGB', '',
        '本轮四种预设拼接均已完成；每种另有对应的 AE-only 对照。以下均为原 800 图测试集、三 seed 指标均值，HALL-AUPR 以幻觉为正类。', '',
        '## 特征的准确含义', '',
        '每个目标、每层先在区域 R 内归一化 attention，并在同一区域内对 hpre 的目标 raw logit 做 Gaussian MAD gate：', '',
        r'\[AE_R=\sum_{j\in R}\frac{a_j}{\sum_{k\in R}a_k}\,\sigma\!\left(\frac{\ell_j-\operatorname{median}_{R}(\ell)}{1.4826\operatorname{MAD}_{R}(\ell)+\epsilon}\right).\]', '',
        '这是原视觉 AE 的区域扩展。VP 在视觉与 prompt 的并集上重新计算，不能写成 AE_V+AE_P。prompt 含 BOS、模板和非视觉特殊 token；G 只含目标前已生成 token。空 G 的 AE 保存 NaN，检测矩阵置零，mentions 全部保留。非空区域 attention 总和≤EPS 时沿用旧实现的均匀权重回退。', '',
        r'\[e_j^{all}=\int_0^1 J_{\mathrm{FFN}\circ\mathrm{Norm}}(z-A_{all}+\alpha A_{all})a_j^{write}\,d\alpha,\qquad S_R^{all}=\sum_{j\in R}\|e_j^{all}\|_2.\]', '',
        'gross 是逐 token 响应的范数之和，保留真实 Norm 的变化；复用已完成的 All-attention K32 特征。它与组响应求和后再取范数的 net 不同。', '',
        '| 组 | 拼接内容，按每块全部层排列 | 维度 | AE-only 对照 |',
        '|---|---|---:|---|',
        '| V | `[AE_V, log1p(S_V_all)]` | 2L | `AE_V`，L |',
        '| VP | `[AE_VP, log1p(S_V_all+S_P_all)]` | 2L | `AE_VP`，L |',
        '| G | `[AE_G, log1p(S_G_all)]` | 2L | `AE_G`，L |',
        '| VP+G | `[AE_VP, log1p(S_V_all+S_P_all), AE_G, log1p(S_G_all)]` | 4L | `[AE_VP,AE_G]`，2L |', '',
        '只有非负 gross 取 log1p；VP 先相加，再取 log。L 分别为 Qwen2.5 28、LLaVA 32、Qwen3 36、InternVL 32。', '',
        '## 分类器与固定评估协议', '',
        '- 三隐藏层 MLP：`[128,64,32]`、BatchNorm、dropout 0.3、Adam、学习率 0.001、weight decay 1e-5、batch 256、最多100 epoch，沿用训练损失选 checkpoint/停止规则。',
        '- 单隐藏层 MLP：sklearn，宽度64/128/256 × 学习率0.01/0.001 × adam/sgd，max_iter500，共12候选；其他参数为既有 builder 默认值。',
        '- XGB：深度4/6/8 × 学习率0.1/0.05 × 100/200/500棵树，共18候选；复用既有 builder，n_jobs=2。',
        '- 原4000图、3200/800固定划分，全部mentions。搜索只用训练内2560/640图（划分seed20260908），候选seed43；按验证AUROC、HALL-AUPR、候选编号依次选参，再全3200图重训seeds43/44/45。无额外标准化，无test选参。', '',
        '单层与三层还存在优化器、BatchNorm/dropout、batch、正则和停止规则差异，因此这是两个实现的对照，不能把差异仅归因于层数。XGB 若三个 seed 的结果完全相同，std 为0只表示当前训练设置的确定性。', '',
        '共8组×4模型×3分类器×3seed＝288个正式头；960个内部验证候选。误差棒为三seed总体标准差，不是置信区间。', '',
        '## 全部融合结果', '',
        '每格为 **AUROC / HALL-AUPR（%）**，两个指标分别显示均值±总体std。', '',
        '| 模型 | 特征 | 三隐藏层 MLP | 单隐藏层 MLP | XGB |', '|---|---|---:|---:|---:|']
    for model in study.MODELS:
        for group in study.KINDS:
            cells = []
            for classifier in CLASSIFIERS:
                r = lookup[model,group,classifier]
                cells.append(' / '.join(f'{100*r[k+"_mean"]:.2f}±{100*r[k+"_std"]:.2f}' for k in ('AUROC','HALL_AUPR')))
            lines.append('| '+NAMES[model]+' | '+SHORT[group]+' | '+' | '.join(cells)+' |')
    lines += ['', f'![三种分类器的全部预设融合结果]({rel}performance.png)', '',
        '图中按四种特征比较三种分类器；同一子图中的柱形共享纵轴。不同模型的标签比例不同，HALL-AUPR 应主要在同一模型内比较。', '',
        '## gross 相对 AE-only 的增量', '',
        '每格为 **ΔAUROC / ΔHALL-AUPR（百分点）**；正值代表增加 gross 后更高。对照采用相同分类器及相同区域 AE，搜索方法各自只在训练内部选参。', '',
        '| 模型 | 特征 | 三隐藏层 MLP | 单隐藏层 MLP | XGB |', '|---|---|---:|---:|---:|']
    delta_lookup = {(r['model'],r['group'],r['classifier'],r['reference']): r for r in deltas}
    for model in study.MODELS:
        for group in study.KINDS:
            cells = [delta_lookup[model,group,c,'AE only'] for c in CLASSIFIERS]
            lines.append('| '+NAMES[model]+' | '+SHORT[group]+' | '+' | '.join(
                f'{r["AUROC_delta_pp"]:+.2f} / {r["HALL_AUPR_delta_pp"]:+.2f}' for r in cells)+' |')
    lines += ['', f'![加入gross相对AE-only的变化]({rel}gain_over_ae.png)', '',
        '这张图直接检验 gross 是否在对应区域 AE 之外提供检测信息。下降意味着当前拼接及分类器下未转化为收益；不能据此断言该信号没有机制意义。', '',
        '## 与历史结果对照', '',
        '视觉组固定三层 MLP 的直接对照：旧特征为 `[AE_V,log1p(S_E)]`（Visual-only 路径），新特征为 `[AE_V,log1p(S_V_all)]`。本轮重新提取的 AE 已在全数据核验旧值；两套特征的积分路径及节点数仍有差异。', '',
        '| 模型 | 旧 AUROC / HALL-AUPR | 新 AUROC / HALL-AUPR | 差值（百分点） |', '|---|---:|---:|---:|']
    for model in study.MODELS:
        d = delta_lookup[model,'visual','three_hidden','legacy AE_V + visual-only S_E']
        r = lookup[model,'visual','three_hidden']
        lines.append(f'| {NAMES[model]} | {100*d["reference_AUROC"]:.2f} / {100*d["reference_HALL_AUPR"]:.2f} | '
            f'{100*r["AUROC_mean"]:.2f} / {100*r["HALL_AUPR_mean"]:.2f} | {d["AUROC_delta_pp"]:+.2f} / {d["HALL_AUPR_delta_pp"]:+.2f} |')
    lines += ['', '另一份历史实验使用 full-prefix MAD 的 U 与冻结 RMS 来源 S。其 V/VP/G 四组与本轮形状类似，但 AE 与 gross 的定义都变了；逐分类器差值在 all_comparisons.csv 中，不能把它解释为只替换路径的消融。所有历史值取三seed均值，不混用旧总表的概率 ensemble 指标。', '',
        '## AE-only 完整结果', '', '| 模型 | AE区域 | 三隐藏层 MLP | 单隐藏层 MLP | XGB |', '|---|---|---:|---:|---:|']
    for model in study.MODELS:
        for group in study.KINDS:
            cells=[]
            for classifier in CLASSIFIERS:
                r=lookup[model,'ae_only_'+group,classifier]
                cells.append(' / '.join(f'{100*r[k+"_mean"]:.2f}±{100*r[k+"_std"]:.2f}' for k in ('AUROC','HALL_AUPR')))
            lines.append('| '+NAMES[model]+' | '+SHORT[group]+' | '+' | '.join(cells)+' |')
    lines += ['', '## 覆盖与验证', '', '| 模型 | 全部mentions | 测试mentions / HALL | 新旧视觉AE最大差 |', '|---|---:|---:|---:|']
    for c in checks:
        lines.append(f'| {NAMES[c["model"]]} | {c["mentions"]} | {c["test_mentions"]} / {c["test_hall"]} | {c["visual_ae_max_error"]:.3g} |')
    lines += ['', '四模型均保留原4000图及全部目标、全部层、全部mentions。逐图结果以原子替换保存，单模型锁防止重复写入；formal阶段要求对应签名的8图smoke通过。已独立重算288头的AUROC/HALL-AUPR，并核验960候选、64次验证选参、原始标签/mention顺序和图片划分。', '',
        '最初 AE 对齐失败的原因是 attention 分块及前向序列形状不同。修复后使用原生 eager attention，Qwen逐目标原始前缀，LLaVA/InternVL沿用原完整回答前向的因果行；未放宽原核验容差。失败smoke和日志保留。', '',
        '本轮不新增bootstrap，也不依据新测试结果筛选特征。上述差值是同一固定测试集上的描述性结果；三seed波动不能代替图片采样的不确定性。当前B2 K4已有数值问题，本轮特征未使用B2。', '',
        '## 文件', '',
        f'- [全部均值/std及绘图数据]({rel}detection.csv)、[逐seed结果]({rel}seed_metrics.csv)、[全部差值]({rel}all_comparisons.csv)。',
        f'- [64组验证选择]({rel}selected_params.csv)、[960候选及收敛警告]({rel}search_trials.csv)、[完整验证]({rel}validation.json)。',
        f'- 各模型子目录保存协议、原始区域AE、矩阵、模型、预测概率与训练日志。PNG/PDF图位于同一输出根；两个项目根进度文件保留最终状态。', '']
    (ROOT/'docs/ALL_ATTENTION_AE_STRENGTH_20260913.md').write_text('\n'.join(lines))


def main():
    study.summarize(study.MODELS)
    seed_rows, checks = validate()
    with (OUT/'detection.csv').open() as stream:
        rows = [{k:float(v) if k.endswith(('_mean','_std')) else v for k,v in r.items()} for r in csv.DictReader(stream)]
    assert len(rows) == 96
    for r in rows:
        seeds = [s for s in seed_rows if all(s[k]==r[k] for k in ('model','group','classifier'))]
        assert len(seeds) == 3
        for metric in ('AUROC','HALL_AUPR'):
            assert abs(np.mean([s[metric] for s in seeds])-r[metric+'_mean']) < 1e-10
            assert abs(np.std([s[metric] for s in seeds])-r[metric+'_std']) < 1e-10
    deltas = comparisons(rows)
    figures(rows, deltas)
    report(rows, deltas, checks)
    print('PASS: 288 heads, 960 candidates; report and figures complete', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait', action='store_true', help='Wait for all four model pipelines, then validate and report once')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'.report.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        def status(stage, **values):
            atomic_json_save(dict(stage=stage, heartbeat=datetime.now(timezone.utc).isoformat(), **values),
                             OUT/'report_progress.json')
        try:
            if args.wait:
                while True:
                    states = {m:load_json(OUT/m/'progress.json') if (OUT/m/'progress.json').exists() else {} for m in study.MODELS}
                    done = [m for m,s in states.items() if s.get('status')=='completed' and s.get('completed')==72]
                    if len(done) == 4:
                        break
                    status('等待四模型完成', status='waiting', completed_models=done,
                           failed_models=[m for m,s in states.items() if s.get('status')=='failed'])
                    time.sleep(10)
            status('核验并生成报告', status='running')
            main()
            status('报告与图表完成', status='completed', report='docs/ALL_ATTENTION_AE_STRENGTH_20260913.md')
        except BaseException as exc:
            status('报告失败', status='failed', error=str(exc))
            raise
