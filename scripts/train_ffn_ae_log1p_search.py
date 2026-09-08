#!/usr/bin/env python3
"""Dedicated AE+log1p(raw S) search; frozen image holdout, no VLM or bootstrap."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import fcntl
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import train_ffn_consistency_alternative_heads as previous
from scripts.train_torch_probe_feature_sets import TorchProbeConfig, train_and_evaluate_probe
from features.tc_fvpa_artifacts import atomic_json_save, atomic_torch_save, sha256_file, sha256_text

study = previous.study
NAME = 'ae_direct_log1p_search_20260908'
FAMILIES = ('one_hidden', 'three_hidden', 'xgboost')
SEEDS = (43, 44, 45)
SEARCH_SEED = 20260908
N_TRIALS = 48
TOP_N = 3
DIRECT = 'AE+log1p(S)'
SCALED = 'AE+log1p(S/tau)_matched_config'
PAPERS = {
    'revisiting_2021': 'https://proceedings.neurips.cc/paper/2021/file/9d86d83f925f2149e9edb0ac3b49229c-Paper.pdf',
    'mlp_config_2021': 'https://raw.githubusercontent.com/yandex-research/rtdl-revisiting-models/main/output/california_housing/mlp/tuning/0.toml',
    'xgb_config_2021': 'https://raw.githubusercontent.com/yandex-research/rtdl-revisiting-models/main/output/california_housing/xgboost/tuning/0.toml',
    'tabular_2022': 'https://papers.neurips.cc/paper_files/paper/2022/file/0378c7692da36807bdec87ab043cdadc-Paper-Datasets_and_Benchmarks.pdf',
    'random_search_2012': 'https://www.jmlr.org/papers/v13/bergstra12a.html',
}


def output_root(model):
    return study.result_root(model)/'production_k4'/NAME


def fixed_mlp():
    base = TorchProbeConfig()
    return dict(hidden_sizes=list(base.hidden_sizes), dropout=base.dropout,
                learning_rate=base.learning_rate, weight_decay=base.weight_decay)


def candidates(family):
    """12 anchors + 36 seeded draws, identical budget and lists for all models."""
    rng = np.random.default_rng(SEARCH_SEED + FAMILIES.index(family))
    if family == 'xgboost':
        values = [dict(c, min_child_weight=5., reg_lambda=1., reg_alpha=0.,
                       subsample=.8, colsample_bytree=.8) for c in previous.candidates(family)]
    else:
        values = [dict(c, weight_decay=1e-5) for c in previous.candidates('one_hidden')]
        if family == 'three_hidden':
            # Three widths of the same three-hidden-layer architecture.
            for c in values:
                width = c['hidden_sizes'][0]*4
                c['hidden_sizes'] = [width, width//2, width//4]
    while len(values) < N_TRIALS:
        if family == 'xgboost':
            c = dict(max_depth=int(rng.choice([2,3,4,5,6,8])), n_estimators=int(rng.choice([150,300,600,1000])),
                     learning_rate=float(10**rng.uniform(-3, np.log10(.3))),
                     min_child_weight=float(10**rng.uniform(0,2)), reg_lambda=float(10**rng.uniform(-2,2)),
                     reg_alpha=0. if rng.random()<.3 else float(10**rng.uniform(-5,1)),
                     subsample=float(rng.uniform(.5,1)), colsample_bytree=float(rng.uniform(.5,1)))
        else:
            width = int(rng.choice([32,64,128,256,512]))
            c = dict(hidden_sizes=[width] if family=='one_hidden' else [width,width//2,width//4],
                     dropout=0. if rng.random()<.2 else float(rng.uniform(0,.5)),
                     learning_rate=float(10**rng.uniform(-5,-2)),
                     weight_decay=0. if rng.random()<.2 else float(10**rng.uniform(-6,-3)))
        if c not in values:
            values.append(c)
    return values


def direct_features(ae, strength):
    ae, strength = np.asarray(ae), np.asarray(strength, dtype=np.float64)
    if ae.ndim!=2 or ae.shape!=strength.shape or not np.isfinite(ae).all() or not np.isfinite(strength).all() or (strength<0).any():
        raise ValueError('Invalid AE / nonnegative raw strength')
    # No tau, statistics, clipping or additional feature normalization.
    result = np.concatenate([ae, np.log1p(strength)], axis=1).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite direct-log feature')
    return result


def inner_inputs(data, split):
    ids = np.array([r['image_id'] for r in data['train_mentions']], dtype=np.int64)
    if (set(split['train'])&set(split['validation']) or
            (set(split['train'])|set(split['validation']))&set(split['test']) or
            not set(ids).issubset(set(split['train'])|set(split['validation']))):
        raise ValueError('Inner/outer image leakage')
    return {f'{kind}_{dest}': data[f'{kind}_train'][np.isin(ids, split[source])]
            for kind in ('X','y') for dest,source in (('train','train'),('test','validation'))}


def prepare(model, device):
    data = study.load_training_data(model, 'fp32_k4', True)
    baseline_path = previous.baseline_root(model)/'summary.json'
    baseline = json.loads(baseline_path.read_text())
    fingerprint = previous.original_fingerprint(data)
    if baseline['fingerprint'] != fingerprint:
        raise ValueError('Original feature/mention/protocol fingerprint changed')
    outer = json.loads((ROOT/'outputs'/model/study.EXPERIMENT/'image_splits.json').read_text())
    if len(outer['train'])!=3200 or len(outer['test'])!=800:
        raise ValueError('Expected original 3200/800 image split')
    split = previous.inner_image_split(outer['train'], outer['test'])
    direct = {k:data[k] for k in ('y_train','y_test','train_mentions','test_mentions')}
    scaled = {k:data[k] for k in ('y_train','y_test')}
    for part in ('train','test'):
        ae = data['X_'+part]['AE']
        strength = data['X_'+part]['raw_AE+S+N_vec'][:, ae.shape[1]:2*ae.shape[1]]
        direct['X_'+part] = direct_features(ae, strength)
        scaled['X_'+part] = data['X_'+part]['AE+s']
    inner = inner_inputs(direct, split)
    if any(len(np.unique(inner['y_'+part]))!=2 for part in ('train','test')):
        raise ValueError('Both labels required; never replace images based on outcomes')
    matrices = {f'{name}/{key}':sha256_text(str(value.shape)+hashlib.sha256(value.tobytes()).hexdigest())
                for name,inputs in (('direct',direct),('scaled',scaled),('inner',inner))
                for key,value in inputs.items() if key.startswith(('X_','y_'))}
    protocol = dict(version=NAME, model=model, feature='concat(full AE, log1p(raw S)); no tau or extra normalization',
        image_split=split, inner_seed=previous.INNER_SEED, search_seed=SEARCH_SEED, candidate_configs={f:candidates(f) for f in FAMILIES},
        screening_seed=43, shortlist=TOP_N, repeat_seeds=[44,45], final_seeds=list(SEEDS),
        selection='48 validation-only seed43 trials; top3 repeated with44/45; largest seed-mean validation AUROC; tie HALL-AUPR then index',
        controls='same selected configs on old train-only-scaled features; original fixed three-hidden config on direct log; no separate scaled-feature search',
        mlp_base=asdict(TorchProbeConfig()), mlp_policy='retain Adam, batch256, train-loss scheduling/stopping/checkpoint; at most100epochs',
        xgb_fixed=dict(objective='binary:logistic',eval_metric='logloss',tree_method='hist',device='cpu',n_jobs=4,scale_pos_weight=1.),
        xgb_policy='fixed candidate tree count, no validation early stopping', matrix_sha256=matrices,
        original_fingerprint=fingerprint, baseline_summary_sha256=sha256_file(baseline_path),
        numerical_exception=data['numerical_exception'], scaled_control_tau={k:v.tolist() for k,v in data['scales'].items()},
        device=device, inference_batch_size=256, checkpoint_tolerance=1e-7,
        packages={p:version(p) for p in ('torch','numpy','scikit-learn','xgboost')},
        source_sha256={str(p.relative_to(ROOT)):sha256_file(p) for p in
                      (Path(__file__),Path(previous.__file__),Path(study.__file__),ROOT/'scripts/train_torch_probe_feature_sets.py')},
        literature=PAPERS, no_bootstrap=True, independent_2000_run=False)
    protocol = json.loads(json.dumps(protocol, sort_keys=True))
    study.immutable_json(protocol, output_root(model)/'protocol.json')
    direct = {k:v for k,v in direct.items() if k.startswith(('X_','y_'))}
    return direct, scaled, inner, baseline, protocol


def run_head(family, config, seed, data, path, signature, device, epochs=100):
    """Reuse the frozen trainer/predictor; only expose extra tuning parameters."""
    path = Path(path); result_path = path/'result.pt'
    identity = sha256_text(json.dumps(dict(signature=signature,family=family,config=config,seed=seed,
                                         path=str(path),device=device,epochs=epochs),sort_keys=True))
    if result_path.exists():
        result = torch.load(result_path,map_location='cpu',weights_only=False)
        if result['fingerprint'] != identity:
            raise ValueError('Wrong head resume fingerprint')
        for name,digest in result['artifacts'].items():
            if sha256_file(name)!=digest:
                raise ValueError(f'Changed checkpoint artifact: {name}')
    else:
        attempt = path/f'attempt_{time.time_ns()}'
        attempt.mkdir(parents=True,exist_ok=False)
        started = time.perf_counter()
        if device.startswith('cuda'):
            # Memory-stat reset does not lazily initialize the CUDA allocator.
            torch.empty(0, device=device)
            torch.cuda.reset_peak_memory_stats(device)
        if family in ('one_hidden','three_hidden'):
            if len(config['hidden_sizes']) != (1 if family=='one_hidden' else 3):
                raise ValueError('Wrong number of hidden layers')
            cfg = replace(TorchProbeConfig(), **{**config,'hidden_sizes':tuple(config['hidden_sizes'])},
                          seed=seed,num_epochs=epochs,split_protocol=NAME)
            metrics = train_and_evaluate_probe(**data, X_val=np.empty((0,data['X_train'].shape[1]),dtype=np.float32),
                y_val=np.empty(0,dtype=np.int32),config=cfg,device=torch.device(device),output_dir=str(attempt),return_probabilities=True)
            train_p = np.asarray(metrics.pop('train_probabilities'),dtype=np.float32)
            test_p = np.asarray(metrics.pop('test_probabilities'),dtype=np.float32)
            checkpoint = attempt/'model.pt'
        elif family=='xgboost':
            from xgboost import XGBClassifier
            estimator = XGBClassifier(**config,random_state=seed,objective='binary:logistic',eval_metric='logloss',
                                      tree_method='hist',device='cpu',n_jobs=4,scale_pos_weight=1.)
            estimator.fit(data['X_train'],data['y_train'])
            train_p = estimator.predict_proba(data['X_train'])[:,1].astype(np.float32)
            test_p = estimator.predict_proba(data['X_test'])[:,1].astype(np.float32)
            checkpoint = attempt/'model.json'; estimator.save_model(checkpoint)
            metrics = previous.metric_reports(data['y_train'],train_p,data['y_test'],test_p)
        else:
            raise ValueError('Unknown classifier')
        result = dict(fingerprint=identity,family=family,config=config,seed=seed,checkpoint=str(checkpoint),
            train_probabilities=train_p,test_probabilities=test_p,metrics=metrics,elapsed_seconds=time.perf_counter()-started,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.startswith('cuda') and family!='xgboost' else 0,
            artifacts={str(p):sha256_file(p) for p in attempt.iterdir() if p.is_file()})
    errors = {}
    for part in ('train','test'):
        saved = result[part+'_probabilities']
        if saved.shape!=data['y_'+part].shape or not np.isfinite(saved).all() or (saved<0).any() or (saved>1).any():
            raise ValueError('Invalid saved probabilities')
        prediction = previous.predict_checkpoint(family,result['checkpoint'],config,data['X_'+part],device)
        errors[part] = float(np.max(np.abs(prediction-saved)))
        if errors[part]>1e-7:
            raise ValueError(f'Same-device checkpoint discrepancy: {errors}')
    reports = previous.metric_reports(data['y_train'],result['train_probabilities'],data['y_test'],result['test_probabilities'])
    if any(result['metrics'][key]!=reports[key] for key in ('thresholds','threshold_reports')):
        raise ValueError('Saved thresholds/metrics mismatch')
    if not result_path.exists():
        result['recomputation_max_error'] = errors
        atomic_torch_save(result,result_path)
    return result


def validation_row(index, config, values):
    reports = [v['metrics']['threshold_reports']['fixed_0.5']['test_metrics'] for v in values]
    return dict(scope='inner_validation_direct_log1p',index=index,config=config,seeds=[v['seed'] for v in values],
                auc=float(np.mean([r['auc'] for r in reports])),
                hall_aupr=float(np.mean([r['hallucination_positive']['aupr'] for r in reports])),per_seed=reports)


def rank_validation(rows):
    if not rows or any(r.get('scope')!='inner_validation_direct_log1p' for r in rows):
        raise ValueError('Selection requires inner-validation-only rows')
    if any(not np.isfinite([r['auc'],r['hall_aupr']]).all() for r in rows):
        raise ValueError('Nonfinite validation metric')
    return sorted(rows,key=lambda r:(-r['auc'],-r['hall_aupr'],r['index']))


def summarize_group(data, values):
    reports = previous.metric_reports(data['y_train'],np.mean([v['train_probabilities'] for v in values],axis=0),
                                     data['y_test'],np.mean([v['test_probabilities'] for v in values],axis=0))
    metrics = [v['metrics']['threshold_reports']['fixed_0.5']['test_metrics'] for v in values]
    scores = dict(auc=[r['auc'] for r in metrics],real_aupr=[r['real_positive']['aupr'] for r in metrics],
                  hall_aupr=[r['hallucination_positive']['aupr'] for r in metrics])
    return dict(ensemble_reports={k:r['test_metrics'] for k,r in reports['threshold_reports'].items()},
                ensemble_train_f1_threshold=reports['thresholds']['train_f1'],
                seed_mean_std={k:dict(mean=float(np.mean(v)),std=float(np.std(v,ddof=0))) for k,v in scores.items()},
                per_seed_metrics={str(v['seed']):v['metrics'] for v in values})


def run_model(model, device):
    root = output_root(model); root.mkdir(parents=True,exist_ok=True)
    with (root/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        started = time.perf_counter()
        direct, scaled, inner, baseline, protocol = prepare(model,device)
        signature = sha256_text(json.dumps(protocol,sort_keys=True))
        selected = {}
        for family in FAMILIES:
            results = []
            for index,config in enumerate(candidates(family)):
                result = run_head(family,config,43,inner,root/'search'/family/f'trial{index:02d}'/'seed43',signature,device)
                results.append(result)
                print('SEARCH',model,family,index,'validation_auc',validation_row(index,config,[result])['auc'],flush=True)
            screening = [validation_row(i,r['config'],[r]) for i,r in enumerate(results)]
            top = rank_validation(screening)[:TOP_N]
            study.immutable_json(dict(scope='inner_validation_direct_log1p',screening=screening,shortlist=top),root/f'shortlist_{family}.json')
            repeats = []
            for row in top:
                values = [results[row['index']]]
                for seed in SEEDS[1:]:
                    values.append(run_head(family,row['config'],seed,inner,
                        root/'search'/family/f"trial{row['index']:02d}"/f'seed{seed}',signature,device))
                    print('RECHECK',model,family,row['index'],seed,flush=True)
                repeats.append(validation_row(row['index'],row['config'],values))
            selected[family] = dict(winner=rank_validation(repeats)[0],finalists=repeats)
        # All three families are frozen before ANY outer-test training/evaluation.
        selection = dict(status='FROZEN_BEFORE_OUTER_EVALUATION',protocol_sha256=sha256_file(root/'protocol.json'),families=selected)
        study.immutable_json(selection,root/'selection.json')
        final_signature = sha256_text(signature+sha256_file(root/'selection.json'))
        groups = {'old_three_hidden_scaled':baseline['groups']['AE+s']}
        for family in FAMILIES:
            for label,inputs in ((DIRECT,direct),(SCALED,scaled)):
                values = []
                for seed in SEEDS:
                    values.append(run_head(family,selected[family]['winner']['config'],seed,inputs,
                        root/'final'/family/label/f'seed{seed}',final_signature,device))
                    print('FINAL',model,family,label,seed,flush=True)
                groups[family+'/'+label] = summarize_group(inputs,values)
        fixed = [run_head('three_hidden',fixed_mlp(),seed,direct,root/'final'/'three_hidden_fixed_direct'/f'seed{seed}',
                          final_signature,device) for seed in SEEDS]
        groups['three_hidden_fixed_direct'] = summarize_group(direct,fixed)
        artifacts = {str(p):sha256_file(p) for part in ('search','final') for p in (root/part).rglob('*') if p.is_file()}
        counts = {part:len(list((root/part).rglob('result.pt'))) for part in ('search','final')}
        if counts != dict(search=3*(N_TRIALS+TOP_N*2),final=21):
            raise ValueError(f'Missing or duplicate heads: {counts}')
        study.immutable_json(dict(status='PASS_ALL_HEADS_SAME_DEVICE',counts=counts,artifacts=artifacts,
            protocol_sha256=sha256_file(root/'protocol.json'),selection_sha256=sha256_file(root/'selection.json')),
            root/'verification.json')
        summary = dict(status='COMPLETE_EXPLORATORY_DIRECT_LOG_SEARCH',model=model,groups=groups,selected=selected,
            counts=counts,train_mentions=len(direct['y_train']),test_mentions=len(direct['y_test']),
            numerical_exception=protocol['numerical_exception'],verification_sha256=sha256_file(root/'verification.json'))
        study.immutable_json(summary,root/'summary.json')
        print('COMPLETE',model,'seconds',time.perf_counter()-started,flush=True)


def publish():
    paths = {m:output_root(m)/'summary.json' for m in study.MODELS}
    if not all(p.exists() for p in paths.values()):
        return False
    root = ROOT/'outputs'/study.VERSION/NAME; root.mkdir(parents=True,exist_ok=True)
    with (root/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        summaries = {m:json.loads(p.read_text()) for m,p in paths.items()}
        for model,s in summaries.items():
            if s['verification_sha256']!=sha256_file(output_root(model)/'verification.json'):
                raise ValueError('Changed model verification')
        study.immutable_json(dict(status='COMPLETE_FOUR_MODEL_DIRECT_LOG_SEARCH',models=summaries,
            sources={m:dict(path=str(p),sha256=sha256_file(p)) for m,p in paths.items()},literature=PAPERS),root/'summary.json')
        lines = ['# AE + log1p(原始 S)：单独扩大调参','',
            '四模型原3200/800图片划分；2560/640内部训练/验证。无tau、无额外特征标准化。',
            '每模型/分类器48候选(seed43)，内部前三各补seed44/45，以三seed平均验证AUROC选参。',
            '选参封存后才评估800图；最终每头seeds43/44/45。旧800图已用于研究探索，不是独立确认。',
            '同参数scaled对照仅改变输入变换，不是分别对两种变换调到最优。原三隐藏层参数的direct对照单列。',
            '保留K4数值FAIL例外及旧InternVL严格CPU复核状态；无bootstrap或新2000图实验。','',
            '## 文献依据与差异','',
            f"- [Gorishniy等，NeurIPS2021]({PAPERS['revisiting_2021']})：验证集选超参数，测试集最终评估，选定配置多seed复跑。",
            f"- [官方MLP配置]({PAPERS['mlp_config_2021']})：100次搜索；lr 1e-5至1e-2，dropout 0至.5，weight_decay允许0或1e-6至1e-3。",
            f"- [Grinsztajn等，NeurIPS2022]({PAPERS['tabular_2022']})：约400次随机搜索，比较不同搜索预算；默认配置纳入搜索。",
            f"- [Bergstra与Bengio，JMLR2012]({PAPERS['random_search_2012']})：随机搜索是固定预算下的基本对照。",
            '- 本轮48组而非100/400组；MLP保留现有BatchNorm、Adam、batch256、最多100epoch和train-loss checkpoint，不照搬论文的AdamW/验证早停/quantile变换。',
            '- XGBoost扩展深度、树数、学习率、行/列采样与正则；固定树数，无验证早停。参数预算相同不代表耗时相同。','',
            '## 选定参数','', '| 模型 | 分类器 | 配置 | 内部验证 AUROC三seed均值 |','|---|---|---|---:|']
        for model,s in summaries.items():
            for family,value in s['selected'].items():
                winner = value['winner']
                lines.append(f"| {model} | {family} | `{json.dumps(winner['config'],sort_keys=True)}` | {winner['auc']:.6f} |")
        lines += ['', '## 800图结果：三seed概率ensemble AUROC / HALL-AUPR（%）','',
                  '| 特征/分类器 | '+' | '.join(study.MODELS)+' |','|---|'+'---:|'*len(study.MODELS)]
        for key in next(iter(summaries.values()))['groups']:
            cells=[]
            for s in summaries.values():
                r=s['groups'][key]['ensemble_reports']['fixed_0.5']
                cells.append(f"{100*r['auc']:.3f} / {100*r['hallucination_positive']['aupr']:.3f}")
            lines.append('| '+key+' | '+' | '.join(cells)+' |')
        lines += ['', '逐seed、mean±std、REAL/HALL AUPR、固定0.5与训练REAL-F1阈值的P/R/F1见summary.json。']
        report = root/'summary.md'; text = '\n'.join(lines)+'\n'
        if report.exists():
            if report.read_text()!=text:
                raise ValueError('Changed completed report')
        else:
            report.write_text(text)
        print('PUBLISHED',report,flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',nargs='+',choices=study.MODELS)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--summarize-only',action='store_true')
    args = parser.parse_args()
    if args.summarize_only:
        if not publish():
            raise ValueError('Incomplete models')
        return
    if not args.models:
        parser.error('--models required')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    for model in args.models:
        try:
            run_model(model,args.device)
        except Exception:
            atomic_json_save(dict(model=model,command=sys.argv,traceback=traceback.format_exc()),
                             output_root(model)/f'failure_{time.time_ns()}.json')
            raise
    publish()


if __name__=='__main__':
    main()
