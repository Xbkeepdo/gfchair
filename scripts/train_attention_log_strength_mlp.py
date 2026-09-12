"""Raw attention + log1p(S): four groups and the requested shallow sklearn MLP grid."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path
import pickle
import sys
import time
import warnings

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import ParameterGrid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from detection.train import build_classifier
from scripts import run_ffn_source_composition as base
from scripts.train_ffn_source_strengths import MODELS, TITLES, OUT as STRENGTH, SEEDS
from scripts.train_prefix_attention_groups import FAMILIES, OUT as ATTENTION
from scripts.train_attention_strength_fusion import metrics
from scripts.train_ffn_consistency_alternative_heads import inner_image_split, INNER_SEED, metric_reports
from scripts.train_ffn_ae_log1p_search import summarize_group
from scripts.train_torch_probe_feature_sets import TorchProbeConfig

OUT = base.OUT/'raw_attention_log_strength_mlp'
KINDS = ('visual', 'visual_prompt_sum', 'generation', 'vp_generation')
GRID = dict(hidden_layer_sizes=[[64], [128], [256]], learning_rate_init=[.01, .001],
            solver=['adam', 'sgd'], max_iter=[500])
TREE_GRIDS = dict(
    xgb=dict(max_depth=[4, 6, 8], learning_rate=[.1, .05], n_estimators=[100, 200, 500]),
    rf=dict(max_depth=[None, 10, 20], n_estimators=[200, 400, 600]))


def build_groups(attention, strength):
    groups = {}
    for family in FAMILIES:
        for kind in KINDS[:3]:
            groups[f'{family}_{kind}'] = np.concatenate((attention[f'{family}_raw_{kind}'], strength[f'log1p_{kind}']), axis=1)
        groups[f'{family}_vp_generation'] = np.concatenate((groups[f'{family}_visual_prompt_sum'], groups[f'{family}_generation']), axis=1)
    return groups


def check_groups():
    a = {f'{f}_raw_{k}': np.array([[.2, .3]], dtype=np.float32) for f in FAMILIES for k in KINDS[:3]}
    s = {f'log1p_{k}': np.array([[1., 2.]], dtype=np.float32) for k in KINDS[:3]}
    g = build_groups(a, s)
    for f in FAMILIES:
        np.testing.assert_array_equal(g[f'{f}_visual'][:, :2], a[f'{f}_raw_visual'])
        np.testing.assert_array_equal(g[f'{f}_vp_generation'], np.concatenate((g[f'{f}_visual_prompt_sum'], g[f'{f}_generation']), axis=1))
    assert len(g) == 8 and len(list(ParameterGrid(GRID))) == 12


def prepare(model):
    root = OUT/model
    if (root/'matrices.pt').exists(): return
    a, s = base.read(ATTENTION/model/'matrices.pt'), base.read(STRENGTH/model/'matrices.pt')
    assert a['ntrain'] == s['ntrain']
    np.testing.assert_array_equal(a['y'], s['y'])
    fields = ('mention_id', 'image_id', 'target_key', 'response_index', 'target_token_id', 'label')
    assert [tuple(m[k] for k in fields) for m in a['mentions']] == [tuple(m[k] for k in fields) for m in s['mentions']]
    groups = build_groups(a['groups'], s['groups']); n = s['ntrain']; mentions = s['mentions']
    split = json.loads((ROOT/'outputs'/model/base.EXPERIMENT/'image_splits.json').read_text())
    assert len(split['train']) == 3200 and len(split['test']) == 800
    train_ids, test_ids = set(split['train']), set(split['test'])
    assert all(m['image_id'] in train_ids for m in mentions[:n])
    assert all(m['image_id'] in test_ids for m in mentions[n:])
    inner = inner_image_split(split['train'], split['test'])
    ids = np.array([m['image_id'] for m in mentions]); fit = np.isin(ids, inner['train']); val = np.isin(ids, inner['validation'])
    assert not (fit & val).any() and not fit[n:].any() and not val[n:].any()
    assert (fit[:n] | val[:n]).all()
    for mask in (fit, val): assert set(np.asarray(s['y'])[mask]) == {0, 1}
    base.save(dict(groups=groups, y=s['y'], ntrain=n, mentions=mentions, inner_fit=fit, inner_val=val), root/'matrices.pt')
    base.json_save(dict(model=model, images=4000, train_images=3200, test_images=800, mentions=len(mentions),
        train_mentions=n, test_mentions=len(mentions)-n, image_split=inner, inner_seed=INNER_SEED,
        features='X_g is raw attention or raw attention*gate, S_g is full K50 source strength; only S uses log1p',
        visual='[X_V, log1p(S_V)]', visual_prompt_sum='[X_V+X_P, log1p(S_V+S_P)]',
        generation='[X_G, log1p(S_G)]', vp_generation='[X_V+X_P, log1p(S_V+S_P), X_G, log1p(S_G)]',
        order='All layers per block; VP and G complete feature blocks concatenated; no attention log',
        dimensions={k: v.shape[1] for k, v in groups.items()}, fixed_torch=asdict(TorchProbeConfig()),
        sklearn_grid=GRID, sklearn_defaults=build_classifier('mlp', {}).get_params(), sklearn_version=version('scikit-learn'),
        selection='12 candidates each feature/model, seed43 on 2560 images; highest validation AUROC, tie HALL-AUPR then candidate index; freeze before test; refit all3200 with43/44/45',
        scaling='No additional scaler or normalization, matching repository MLP builder', seeds=SEEDS,
        distinction='sklearn one-hidden MLP has its own defaults/stopping and no BN/dropout; not a depth-only ablation of Torch MLP'), root/'protocol.json')
    print('PREPARED', model, '8 groups', flush=True)


def fit_classifier(params, seed, x, y, classifier='mlp'):
    estimator = build_classifier(classifier, params).set_params(random_state=seed)
    if classifier != 'mlp': estimator.set_params(n_jobs=2)
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        estimator.fit(x, y)
    info = dict(seconds=time.perf_counter()-start,
                warnings=[dict(category=w.category.__name__, message=str(w.message)) for w in caught])
    if classifier == 'mlp': info.update(n_iter=int(estimator.n_iter_), loss=float(estimator.loss_))
    else: info['n_estimators'] = params['n_estimators']
    return estimator, info


def search_group(model, name, classifier='mlp'):
    root = OUT/model; data = base.read(root/'matrices.pt')
    x, y, n = data['groups'][name], data['y'], data['ntrain']; fit, val = data['inner_fit'], data['inner_val']
    dest = root/('one_hidden' if classifier == 'mlp' else classifier)/name
    candidates = list(ParameterGrid(GRID if classifier == 'mlp' else TREE_GRIDS[classifier])); trials = []
    for index, params in enumerate(candidates):
        path = dest/'search'/f'trial{index:02d}.json'
        if path.exists(): trial = json.loads(path.read_text())
        else:
            estimator, info = fit_classifier(params, 43, x[fit], y[fit], classifier)
            p = estimator.predict_proba(x[val])[:, 1]
            trial = dict(index=index, params=params, seed=43, scope='inner_validation',
                validation_AUROC=float(roc_auc_score(y[val], p)), validation_HALL_AUPR=float(average_precision_score(1-y[val], 1-p)), **info)
            base.json_save(trial, path)
        trials.append(trial)
    best = max(trials, key=lambda r: (r['validation_AUROC'], r['validation_HALL_AUPR'], -r['index']))
    base.json_save(dict(best=best, candidates=trials, selection='inner validation only; refit all3200 next'), dest/'selection.json')
    heads = []
    for seed in SEEDS:
        path = dest/f'seed{seed}'/'result.pt'
        if path.exists(): value = base.read(path)
        else:
            estimator, info = fit_classifier(best['params'], seed, x[:n], y[:n], classifier)
            train_p, test_p = estimator.predict_proba(x[:n])[:, 1], estimator.predict_proba(x[n:])[:, 1]
            reports = metric_reports(y[:n], train_p, y[n:], test_p)
            value = dict(seed=seed, params=best['params'], train_probabilities=train_p, test_probabilities=test_p,
                metrics={**reports['threshold_reports']['train_f1']['test_metrics'], **reports}, **info)
            path.parent.mkdir(parents=True, exist_ok=True)
            with (path.parent/'model.pkl').open('wb') as stream: pickle.dump(estimator, stream, protocol=pickle.HIGHEST_PROTOCOL)
            base.save(value, path)
        heads.append(value)
    result = summarize_group(dict(y_train=y[:n], y_test=y[n:]), heads)
    base.json_save(result, dest/'detection.json')
    return dict(model=model, group=name, classifier=classifier, params=best['params'], validation_AUROC=best['validation_AUROC'],
                test_AUROC=result['seed_mean_std']['auc'])


def summarize():
    rows, seed_rows, selections, deltas = [], [], [], []
    lines = ['# 原始attention + log1p(S)：固定MLP与单隐藏层网格搜索', '',
        '四模型原4000图/3200-800/全部mentions。X为原始attention或原始attention×gate，只有S_g取log1p。',
        '每种X四组：visual、visual_prompt_sum、generation，以及vp_generation=[完整VP组,完整G组]。前三组2L维，最后4L维。',
        '固定对照为原Torch三隐藏层MLP；单隐藏层用项目sklearn.MLPClassifier，宽度64/128/256×学习率.01/.001×adam/sgd，max_iter500，共12候选。',
        '每个模型/特征组独立选参：原训练3200图中2560/640按图片划分，seed43验证AUROC优先，平局HALL-AUPR；选定后3200图重训43/44/45，原800图只用于最终报告。',
        '未额外标准化。sklearn与Torch还有BN/dropout、正则、batch和停止规则差异，结果不能只归因为隐藏层数。',
        '三seed均值±总体std（%），非ensemble；HALL-F1采用各seed训练REAL-F1阈值，原双阈值详见CSV。', '',
        '| 模型 | 特征 | 分类器 | AUROC | HALL-AUPR | HALL-F1 |', '|---|---|---|---:|---:|---:|']
    for model, title in zip(MODELS, TITLES):
        root = OUT/model; fixed = json.loads((root/'detection.json').read_text())
        for name, entry in fixed.items():
            shallow = json.loads((root/'one_hidden'/name/'detection.json').read_text())
            chosen = json.loads((root/'one_hidden'/name/'selection.json').read_text())['best']
            selections.append(dict(model=model, group=name, **chosen))
            for classifier, value in (('fixed_three_hidden', entry), ('searched_one_hidden', shallow)):
                m = metrics(value); rows.append(dict(model=model, group=name, classifier=classifier, **m))
                cells = [f"{100*m[k+'_mean']:.3f} ± {100*m[k+'_std']:.3f}" for k in ('AUROC','HALL_AUPR','HALL_F1')]
                lines.append(f'| {title} | {name} | {classifier} | '+' | '.join(cells)+' |')
                for seed in SEEDS:
                    for rule, report in value['per_seed_metrics'][str(seed)]['threshold_reports'].items():
                        r=report['test_metrics'];h=r['hallucination_positive']
                        seed_rows.append(dict(model=model,group=name,classifier=classifier,seed=seed,threshold_rule=rule,threshold=report['threshold'],
                            AUROC=r['auc'],HALL_AUPR=h['aupr'],HALL_precision=h['precision'],HALL_recall=h['recall'],HALL_F1=h['f1']))
            a,b=metrics(shallow),metrics(entry)
            deltas.append(dict(model=model,group=name,AUROC_delta=a['AUROC_mean']-b['AUROC_mean'],HALL_AUPR_delta=a['HALL_AUPR_mean']-b['HALL_AUPR_mean']))
    lookup={(r['model'],r['group'],r['classifier']):r for r in rows}
    brief=['八组AUROC均值总览（%，原三隐藏层 → 搜参单隐藏层；标准差见下方完整表）：', '',
           '| 特征 | Qwen2.5 | LLaVA | Qwen3 | InternVL |', '|---|---:|---:|---:|---:|']
    for family in FAMILIES:
        for kind in KINDS:
            name=f'{family}_{kind}';cells=[]
            for model in MODELS:
                a=lookup[model,name,'fixed_three_hidden'];b=lookup[model,name,'searched_one_hidden']
                cells.append(f"{100*a['AUROC_mean']:.3f} → {100*b['AUROC_mean']:.3f}")
            brief.append('| '+name+' | '+' | '.join(cells)+' |')
    brief += ['', f"单隐藏层相对原对照：{sum(r['AUROC_delta']>0 for r in deltas)}/{len(deltas)}组平均AUROC提高，{sum(r['HALL_AUPR_delta']>0 for r in deltas)}/{len(deltas)}组平均HALL-AUPR提高。只解释当前输入与实现，不将差异归因于纯隐藏层数。", '',
              '本表记录单隐藏层MLP阶段；后续XGB/RF见[四种分类器对照](trees_summary.md)。', '']
    index=lines.index('| 模型 | 特征 | 分类器 | AUROC | HALL-AUPR | HALL-F1 |');lines[index:index]=brief
    param_rows=[dict(model=r['model'],group=r['group'],hidden_width=r['params']['hidden_layer_sizes'][0],
        learning_rate_init=r['params']['learning_rate_init'],solver=r['params']['solver'],max_iter=r['params']['max_iter'],
        validation_AUROC=r['validation_AUROC'],validation_HALL_AUPR=r['validation_HALL_AUPR'],candidate_n_iter=r['n_iter']) for r in selections]
    lines += ['', 'selected_params.json及selected_params.csv保存每组验证选择；one_hidden/<group>/search逐候选记录验证指标、迭代次数、loss和收敛警告。达到max_iter的警告保留，不自动延长预算。',
              '每组最终三seed模型及概率已保存，固定与搜索方法各96个正式头；MLP内层共384次候选拟合。',
              '运行：`python scripts/train_attention_log_strength_mlp.py --stage prepare`；`--stage fixed --models <模型> --device cuda:0`；`--stage search --workers 12`；完成后`--stage summarize`。',
              '完整来源S_g目前只有四模型，不混用MiniGPT/Shikra旧S_E。保留原提取数值、gate口径及前缀长度/位置限制；没有bootstrap或独立图像验证。']
    base.write_csv(rows,OUT/'detection.csv');base.write_csv(seed_rows,OUT/'seed_metrics.csv');base.write_csv(deltas,OUT/'comparisons.csv')
    base.write_csv(param_rows,OUT/'selected_params.csv')
    base.json_save(selections,OUT/'selected_params.json');(OUT/'summary.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('prepare','fixed','search','summarize'),default='prepare')
    parser.add_argument('--models',nargs='+',choices=MODELS,default=MODELS)
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--workers',type=int,default=12)
    args=parser.parse_args()
    if args.stage=='summarize': summarize()
    else:
        check_groups()
        for model in args.models: prepare(model)
        if args.stage=='fixed':
            for model in args.models: base.train(model,args.device,root=OUT/model)
        elif args.stage=='search':
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures=[pool.submit(search_group,m,f'{f}_{k}') for m in args.models for f in FAMILIES for k in KINDS]
                for future in as_completed(futures): print('SEARCH DONE',json.dumps(future.result()),flush=True)
