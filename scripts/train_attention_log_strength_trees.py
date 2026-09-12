"""Run the user-provided XGB/RF grids on the existing eight feature groups."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import version
import json

from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.train_attention_log_strength_mlp import (
    OUT, MODELS, TITLES, FAMILIES, KINDS, SEEDS, TREE_GRIDS, prepare, search_group,
    build_classifier, ParameterGrid, metrics, base)


def prepare_trees(model):
    prepare(model)
    assert len(list(ParameterGrid(TREE_GRIDS['xgb']))) == 18
    assert len(list(ParameterGrid(TREE_GRIDS['rf']))) == 9
    assert type(build_classifier('xgb',{})).__name__ == 'XGBClassifier'
    protocol=json.loads((OUT/model/'protocol.json').read_text())
    base.json_save(dict(model=model, input_protocol=str(OUT/model/'protocol.json'),
        image_split=protocol['image_split'], inner_seed=protocol['inner_seed'], seeds=SEEDS,
        grids=TREE_GRIDS, xgboost_version=version('xgboost'), sklearn_version=version('scikit-learn'),
        defaults={c:build_classifier(c,{}).set_params(random_state=43,n_jobs=2).get_params() for c in TREE_GRIDS},
        selection='Same 2560/640 inner image split; seed43 validation AUROC, tie HALL-AUPR then candidate index; refit full3200 with43/44/45; test800 never used for selection',
        resource_threads_per_estimator=2, scaling='Reuse identical X raw + log1p(S) matrices without scaler',
        no_early_stopping=True),OUT/model/'tree_protocol.json')


def summarize():
    rows,seeds,params,comparisons=[],[],[],[]
    lines=['# 原始attention＋log1p(S)：XGB / RF网格搜索', '',
        '四模型×8组相同输入，原4000图/3200-800/全部mentions与原MLP完全一致。X保持原attention或attention×gate，仅S_g取log1p。',
        '每组分别：visual、V+P相加、generation、完整VP组与G组拼接。',
        'XGB：depth4/6/8×lr.1/.05×trees100/200/500，共18候选；RF：depth无限/10/20×trees200/400/600，共9候选。',
        '每模型/组/分类器独立在同一2560/640图片划分上用seed43验证AUROC选参，平局AP/候选顺序；选定后全3200图重训43/44/45，原800图测试。没有根据测试结果重选。',
        '使用项目原分类器构建器，CPU每估计器2线程；不增加标准化、类别权重或树模型早停。',
        '共864候选拟合、192最终树模型。两种MLP只引用已完成结果，不重训。以下为三seed均值±总体std（%），F1按训练REAL-F1阈值。', '',
        '| 模型 | 特征 | 分类器 | AUROC | HALL-AUPR | HALL-F1 |', '|---|---|---|---:|---:|---:|']
    for model,title in zip(MODELS,TITLES):
        root=OUT/model; fixed=json.loads((root/'detection.json').read_text())
        for name,entry in fixed.items():
            results=dict(three_hidden=entry,one_hidden=json.loads((root/'one_hidden'/name/'detection.json').read_text()))
            for c in TREE_GRIDS:
                results[c]=json.loads((root/c/name/'detection.json').read_text())
                best=json.loads((root/c/name/'selection.json').read_text())['best']
                params.append(dict(model=model,group=name,classifier=c,**best))
            current={}
            for c,value in results.items():
                m=metrics(value);current[c]=m;rows.append(dict(model=model,group=name,classifier=c,**m))
                cells=[f"{100*m[k+'_mean']:.3f} ± {100*m[k+'_std']:.3f}" for k in ('AUROC','HALL_AUPR','HALL_F1')]
                lines.append(f'| {title} | {name} | {c} | '+' | '.join(cells)+' |')
                for seed in SEEDS:
                    for rule,report in value['per_seed_metrics'][str(seed)]['threshold_reports'].items():
                        r=report['test_metrics'];h=r['hallucination_positive']
                        seeds.append(dict(model=model,group=name,classifier=c,seed=seed,threshold_rule=rule,threshold=report['threshold'],
                            AUROC=r['auc'],HALL_AUPR=h['aupr'],HALL_precision=h['precision'],HALL_recall=h['recall'],HALL_F1=h['f1']))
            for c in TREE_GRIDS:
                for ref in ('three_hidden','one_hidden'):
                    comparisons.append(dict(model=model,group=name,classifier=c,reference=ref,
                        AUROC_delta=current[c]['AUROC_mean']-current[ref]['AUROC_mean'],
                        HALL_AUPR_delta=current[c]['HALL_AUPR_mean']-current[ref]['HALL_AUPR_mean']))
    lookup={(r['model'],r['group'],r['classifier']):r for r in rows}
    brief=['八组平均AUROC总览（%，原三层MLP / XGB / RF）：', '',
           '| 特征 | Qwen2.5 | LLaVA | Qwen3 | InternVL |', '|---|---:|---:|---:|---:|']
    for f in FAMILIES:
        for k in KINDS:
            name=f'{f}_{k}';cells=[' / '.join(f"{100*lookup[model,name,c]['AUROC_mean']:.3f}" for c in ('three_hidden','xgb','rf')) for model in MODELS]
            brief.append('| '+name+' | '+' | '.join(cells)+' |')
    for c in TREE_GRIDS:
        v=[r for r in comparisons if r['classifier']==c and r['reference']=='three_hidden']
        brief += ['',f"{c}相对原三层：{sum(r['AUROC_delta']>0 for r in v)}/32组平均AUROC提高；{sum(r['HALL_AUPR_delta']>0 for r in v)}/32组平均HALL-AUPR提高。"]
    brief += ['', '完整指标与单层MLP对照：', '']
    i=lines.index('| 模型 | 特征 | 分类器 | AUROC | HALL-AUPR | HALL-F1 |');lines[i:i]=brief
    lines += ['', 'trees_selected_params.csv/json包含每组验证选中参数。每候选的验证指标/耗时/警告、selection、三seed模型和概率位于各模型xgb/rf子目录。',
              '原始数据与划分复用，不做SHA、bootstrap置信区间或全量checkpoint独立审计。RF自身按默认bootstrap=True训练。重复使用原holdout，报告为探索性点估计，不作显著性或因果声明。',
              '运行：`python scripts/train_attention_log_strength_trees.py --stage search --workers 24`；完成后`--stage summarize`。']
    param_rows=[dict(model=r['model'],group=r['group'],classifier=r['classifier'],max_depth=r['params']['max_depth'],
        n_estimators=r['params']['n_estimators'],learning_rate=r['params'].get('learning_rate',''),
        validation_AUROC=r['validation_AUROC'],validation_HALL_AUPR=r['validation_HALL_AUPR']) for r in params]
    base.write_csv(rows,OUT/'trees_detection.csv');base.write_csv(seeds,OUT/'trees_seed_metrics.csv')
    base.write_csv(comparisons,OUT/'trees_comparisons.csv');base.write_csv(param_rows,OUT/'trees_selected_params.csv')
    base.json_save(params,OUT/'trees_selected_params.json');(OUT/'trees_summary.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('search','summarize'),default='search')
    parser.add_argument('--models',nargs='+',choices=MODELS,default=MODELS)
    parser.add_argument('--classifiers',nargs='+',choices=list(TREE_GRIDS),default=list(TREE_GRIDS))
    parser.add_argument('--workers',type=int,default=24)
    args=parser.parse_args()
    if args.stage=='summarize': summarize()
    else:
        for model in args.models: prepare_trees(model)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(search_group,m,f'{f}_{k}',c) for m in args.models for f in FAMILIES for k in KINDS for c in args.classifiers]
            for future in as_completed(futures):print('TREE DONE',json.dumps(future.result()),flush=True)
