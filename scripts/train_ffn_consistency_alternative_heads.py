#!/usr/bin/env python3
"""Small train-only U_SN search; transfer one configuration to all 16 groups.

Reuse the frozen v2 features and torch trainer. Never call a VLM, change an old
artifact, or use the outer 800-image test set to select hyperparameters.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import fcntl
import hashlib
from importlib.metadata import version
from itertools import product
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import analyze_ffn_visual_source_consistency as study
from scripts.run_ffn_visual_source_attribution import result_root as v1_root
from scripts.train_torch_probe_feature_sets import (
    DGSTStyleProbe, TorchProbeConfig, train_and_evaluate_probe, _select_f1_threshold,
)
from features.tc_fvpa_artifacts import atomic_json_save, atomic_torch_save, sha256_file, sha256_text

NAME = 'head_search_20260908'
SEEDS = (43, 44, 45)
ANCHOR = 'U_SN'
INNER_SEED = 20260908


def candidates(family):
    if family == 'one_hidden':
        return [dict(hidden_sizes=[width], dropout=dropout, learning_rate=lr)
                for width, dropout, lr in product((32,64,128), (0.,.3), (3e-4,1e-3))]
    if family == 'xgboost':
        return [dict(max_depth=depth, learning_rate=lr, n_estimators=trees)
                for depth, lr, trees in product((2,3,5), (.03,.1), (100,300))]
    raise ValueError(f'Unknown family: {family}')


def output_root(model):
    return study.result_root(model)/'production_k4'/NAME


def baseline_root(model):
    return study.result_root(model)/'production_k4'/study.EXPLORATORY_TRAINING/'training/fp32_k4'


def inner_image_split(train_images, test_images, validation_count=640):
    train, test = sorted(map(int,train_images)), sorted(map(int,test_images))
    if len(train)!=len(set(train)) or len(test)!=len(set(test)) or set(train)&set(test):
        raise ValueError('Duplicate images or outer split overlap')
    if not 0 < validation_count < len(train):
        raise ValueError('Invalid inner validation size')
    order = np.random.default_rng(INNER_SEED).permutation(train).tolist()
    return dict(train=sorted(order[validation_count:]), validation=sorted(order[:validation_count]), test=test)


def inner_features(raw, image_ids, split):
    ids = np.asarray(image_ids)
    if set(split['train']) & (set(split['validation']) | set(split['test'])) or set(split['validation']) & set(split['test']):
        raise ValueError('Inner image leakage')
    scales = study.fit_scales(raw['S'],raw['I'],ids,split['train'])
    matrix = study.feature_sets(raw,scales)[ANCHOR]
    return (matrix[np.isin(ids,split['train'])], matrix[np.isin(ids,split['validation'])], scales)


def original_fingerprint(data):
    signature = hashlib.sha256()
    for spec in study.FEATURE_GROUPS:
        for split in ('train','test'):
            value=data['X_'+split][spec]
            signature.update(spec.encode()+split.encode()+str(value.shape).encode()+value.tobytes())
    for split in ('train','test'):
        signature.update(data['y_'+split].tobytes())
        signature.update(json.dumps(data[split+'_mentions'],sort_keys=True).encode())
    signature.update(json.dumps(asdict(TorchProbeConfig()),sort_keys=True).encode())
    signature.update(sha256_file(ROOT/'scripts/train_torch_probe_feature_sets.py').encode())
    signature.update(json.dumps(data['numerical_exception'],sort_keys=True).encode())
    return signature.hexdigest()


def prepare(model):
    data = study.load_training_data(model,'fp32_k4',True)
    old_root = baseline_root(model)
    old_manifest = json.loads((old_root/'manifest.json').read_text())
    old_summary = json.loads((old_root/'summary.json').read_text())
    fingerprint = original_fingerprint(data)
    if fingerprint != old_manifest['fingerprint'] or fingerprint != old_summary['fingerprint']:
        raise ValueError('Outer features do not reproduce original 3-hidden-layer inputs')
    outer = json.loads((ROOT/'outputs'/model/study.EXPERIMENT/'image_splits.json').read_text())
    if len(outer['train'])!=3200 or len(outer['test'])!=800:
        raise ValueError('Expected the original 3200/800 image split')
    split = inner_image_split(outer['train'],outer['test'])
    mentions=[]
    for path in sorted((v1_root(model)/'shards/full').glob('features*_shard_*.pt')):
        mentions.extend(torch.load(path,map_location='cpu',weights_only=False)['sample_table'])
    ids=np.array([r['image_id'] for r in mentions],dtype=np.int64)
    labels=np.array([r['label'] for r in mentions],dtype=np.int32)
    for name in ('train','test'):
        mask=np.isin(ids,outer[name])
        if [r for r,keep in zip(mentions,mask) if keep] != data[name+'_mentions']:
            raise ValueError('Inner/outer mention order mismatch')
        np.testing.assert_array_equal(labels[mask],data['y_'+name])
    x_train,x_val,scales=inner_features(data['raw'],ids,split)
    inner=dict(X_train=x_train,X_test=x_val,
               y_train=labels[np.isin(ids,split['train'])],y_test=labels[np.isin(ids,split['validation'])])
    if any(len(np.unique(inner['y_'+name]))!=2 for name in ('train','test')):
        raise ValueError('Inner split needs both labels; do not replace images based on outcomes')
    protocol=dict(version=NAME,model=model,anchor=ANCHOR,inner_seed=INNER_SEED,inner_images=split,
                  candidate_configs={f:candidates(f) for f in ('one_hidden','xgboost')},
                  search_seed=43,final_seeds=list(SEEDS),selection='inner validation AUROC; exact tie HALL-AUPR; then candidate order',
                  transfer='one config per model/family, selected on U_SN and applied unchanged to all 16 groups',
                  mlp_base=asdict(TorchProbeConfig()),
                  xgboost_fixed=dict(objective='binary:logistic',eval_metric='logloss',tree_method='hist',
                                      device='cpu',n_jobs=4,subsample=.8,colsample_bytree=.8,
                                      min_child_weight=5,reg_lambda=1.,reg_alpha=0.,scale_pos_weight=1.),
                  inner_scales={k:v.tolist() for k,v in scales.items()},
                  final_scales={k:v.tolist() for k,v in data['scales'].items()},
                  outer_feature_fingerprint=fingerprint,numerical_exception=data['numerical_exception'],
                  baseline_summary_sha256=sha256_file(old_root/'summary.json'),
                  old_internvl_cpu_verification='Original strict FAIL retained; same-device reproduction diagnosed separately',
                  probability_verification='same device, dtype and fixed batch256; absolute tolerance 1e-7; no CPU substitution',
                  packages={p:version(p) for p in ('torch','numpy','scikit-learn','xgboost')},
                  implementation={str(p.relative_to(ROOT)):sha256_file(p) for p in
                                  (Path(__file__),ROOT/'scripts/train_torch_probe_feature_sets.py',Path(study.__file__))},
                  no_bootstrap=True,independent_2000_run=False)
    # JSON round-trip makes tuple/list representation identical on resume.
    protocol=json.loads(json.dumps(protocol,sort_keys=True))
    study.immutable_json(protocol,output_root(model)/'protocol.json')
    return data,inner,old_summary,protocol


def metric_reports(y_train,train_p,y_test,test_p):
    threshold=float(_select_f1_threshold(y_train,train_p))
    thresholds={'fixed_0.5':.5,'train_f1':threshold}
    return dict(thresholds=thresholds,threshold_reports={rule:dict(
        threshold=t,train_metrics=study.probability_metrics(y_train,train_p,t),
        test_metrics=study.probability_metrics(y_test,test_p,t)) for rule,t in thresholds.items()})


def predict_checkpoint(family,path,config,matrix,device):
    if family=='xgboost':
        from xgboost import XGBClassifier
        estimator=XGBClassifier(n_jobs=4,device='cpu')
        estimator.load_model(path)
        return estimator.predict_proba(matrix)[:,1].astype(np.float32)
    estimator=DGSTStyleProbe(matrix.shape[1],config['hidden_sizes'],config['dropout'])
    estimator.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
    estimator=estimator.to(device).eval()
    with torch.no_grad():
        return np.concatenate([torch.sigmoid(estimator(torch.tensor(matrix[i:i+256],device=device))).cpu().numpy().reshape(-1)
                               for i in range(0,len(matrix),256)])


def run_head(family,config,seed,data,path,signature,device,epochs=100):
    path=Path(path); result_path=path/'result.pt'
    identity=sha256_text(json.dumps(dict(signature=signature,family=family,config=config,seed=seed,
                        path=str(path),device=device,epochs=epochs),sort_keys=True))
    if result_path.exists():
        result=torch.load(result_path,map_location='cpu',weights_only=False)
        if result['fingerprint']!=identity:
            raise ValueError('Wrong head resume fingerprint')
        for name,digest in result['artifacts'].items():
            if sha256_file(name)!=digest:
                raise ValueError(f'Changed head artifact: {name}')
    else:
        attempt=path/f'attempt_{time.time_ns()}'
        attempt.mkdir(parents=True,exist_ok=False)
        started=time.perf_counter()
        if family=='one_hidden':
            cfg=replace(TorchProbeConfig(),hidden_sizes=tuple(config['hidden_sizes']),dropout=config['dropout'],
                        learning_rate=config['learning_rate'],seed=seed,num_epochs=epochs,
                        split_protocol='train_only_inner_holdout_or_outer_refit_as_recorded_in_protocol')
            metrics=train_and_evaluate_probe(**data,X_val=np.empty((0,data['X_train'].shape[1]),dtype=np.float32),
                y_val=np.empty(0,dtype=np.int32),config=cfg,device=torch.device(device),
                output_dir=str(attempt),return_probabilities=True)
            train_p=np.asarray(metrics.pop('train_probabilities'),dtype=np.float32)
            test_p=np.asarray(metrics.pop('test_probabilities'),dtype=np.float32)
            checkpoint=attempt/'model.pt'
        elif family=='xgboost':
            from xgboost import XGBClassifier
            estimator=XGBClassifier(**config,random_state=seed,objective='binary:logistic',eval_metric='logloss',
                tree_method='hist',device='cpu',n_jobs=4,subsample=.8,colsample_bytree=.8,
                min_child_weight=5,reg_lambda=1.,reg_alpha=0.,scale_pos_weight=1.)
            estimator.fit(data['X_train'],data['y_train'])
            train_p=estimator.predict_proba(data['X_train'])[:,1].astype(np.float32)
            test_p=estimator.predict_proba(data['X_test'])[:,1].astype(np.float32)
            checkpoint=attempt/'model.json'; estimator.save_model(checkpoint)
            metrics=metric_reports(data['y_train'],train_p,data['y_test'],test_p)
        else:
            raise ValueError(f'Unknown head family: {family}')
        result=dict(fingerprint=identity,config=config,seed=seed,family=family,checkpoint=str(checkpoint),metrics=metrics,
                    train_probabilities=train_p,test_probabilities=test_p,elapsed_seconds=time.perf_counter()-started,
                    artifacts={str(p):sha256_file(p) for p in attempt.iterdir() if p.is_file()})
    errors={}
    for split in ('train','test'):
        saved=result[split+'_probabilities']
        if saved.shape!=data['y_'+split].shape or not np.isfinite(saved).all() or (saved<0).any() or (saved>1).any():
            raise ValueError('Invalid probabilities')
        prediction=predict_checkpoint(family,result['checkpoint'],config,data['X_'+split],device)
        errors[split]=float(np.max(np.abs(prediction-saved)))
        if errors[split]>1e-7:
            raise ValueError(f'Same-device checkpoint discrepancy: {family} {path} {errors}')
    reports=metric_reports(data['y_train'],result['train_probabilities'],data['y_test'],result['test_probabilities'])
    if any(result['metrics'][key]!=reports[key] for key in ('thresholds','threshold_reports')):
        raise ValueError('Saved metrics or thresholds do not reproduce saved probabilities')
    if not result_path.exists():
        result['recomputation_max_error']=errors
        atomic_torch_save(result,result_path)
    return result


def select_candidate(trials):
    # Only inner-validation reports are accepted by this selector, never outer results.
    if not trials or any(t.get('scope')!='inner_validation_U_SN' for t in trials):
        raise ValueError('Selection requires inner-validation-only trials')
    return max(range(len(trials)),key=lambda i:(trials[i]['validation']['auc'],
                 trials[i]['validation']['hallucination_positive']['aupr'],-i))


def run_model(model,device):
    root=output_root(model); root.mkdir(parents=True,exist_ok=True)
    with (root/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        started=time.perf_counter()
        data,inner,baseline,protocol=prepare(model)
        signature=sha256_text(json.dumps(protocol,sort_keys=True))
        selected={}
        # Finish and seal both searches before any outer-test refit/evaluation.
        for family in ('one_hidden','xgboost'):
            trials=[]
            for index,config in enumerate(candidates(family)):
                result=run_head(family,config,43,inner,root/'search'/family/f'trial{index:02d}',signature,device)
                report=result['metrics']['threshold_reports']['fixed_0.5']['test_metrics']
                trials.append(dict(scope='inner_validation_U_SN',config=config,validation=report))
                print('SEARCH',model,family,index,'validation_auc',report['auc'],flush=True)
            best=select_candidate(trials)
            selected[family]=dict(index=best,config=trials[best]['config'],trials=trials)
        selection=dict(protocol_sha256=sha256_file(root/'protocol.json'),families=selected,
                       selection_data='inner validation only; no outer test metric',status='FROZEN_BEFORE_OUTER_EVALUATION')
        study.immutable_json(selection,root/'selection.json')
        fit_signature=sha256_text(signature+sha256_file(root/'selection.json'))
        methods={'three_hidden_fixed':baseline['groups']}
        for family in ('one_hidden','xgboost'):
            heads={}
            for spec in study.FEATURE_GROUPS:
                heads[spec]={}
                inputs={f'{kind}_{split}':data[f'{kind}_{split}'][spec] if kind=='X' else data[f'{kind}_{split}']
                        for kind,split in product(('X','y'),('train','test'))}
                for seed in SEEDS:
                    result=run_head(family,selected[family]['config'],seed,inputs,
                                    root/'final'/family/spec/f'seed{seed}',fit_signature,device)
                    heads[spec][str(seed)]=result
                    print('FINAL',model,family,spec,seed,'test_auc',
                          result['metrics']['threshold_reports']['fixed_0.5']['test_metrics']['auc'],flush=True)
            methods[family]=study.summarize_heads(data,heads)
        artifacts={str(p):sha256_file(p) for directory in ('search','final') for p in (root/directory).rglob('*') if p.is_file()}
        verification=dict(status='PASS_96_FINAL_AND_24_SEARCH_HEADS_SAME_DEVICE',artifacts=artifacts,
                          protocol_sha256=sha256_file(root/'protocol.json'),selection_sha256=sha256_file(root/'selection.json'))
        study.immutable_json(verification,root/'verification.json')
        summary=dict(status='COMPLETE_EXPLORATORY_HEAD_COMPARISON',model=model,signature=signature,
                     numerical_exception=data['numerical_exception'],selected={f:s['config'] for f,s in selected.items()},
                     methods=methods,protocol_sha256=verification['protocol_sha256'],verification_sha256=sha256_file(root/'verification.json'),
                     train_mentions=len(data['y_train']),test_mentions=len(data['y_test']))
        study.immutable_json(summary,root/'summary.json')
        print('COMPLETE',model,'seconds',time.perf_counter()-started,flush=True)


def publish():
    paths={m:output_root(m)/'summary.json' for m in study.MODELS}
    if not all(p.exists() for p in paths.values()):
        return False
    root=ROOT/'outputs'/study.VERSION/NAME; root.mkdir(parents=True,exist_ok=True)
    with (root/'.summary.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        summaries={m:json.loads(p.read_text()) for m,p in paths.items()}
        for model,value in summaries.items():
            if value['verification_sha256']!=sha256_file(output_root(model)/'verification.json'):
                raise ValueError('Changed model verification')
        payload=dict(status='COMPLETE_FOUR_MODEL_EXPLORATORY_COMPARISON',models=summaries,
                     sources={m:dict(path=str(p),sha256=sha256_file(p)) for m,p in paths.items()},
                     no_bootstrap=True,no_independent_confirmation=True)
        study.immutable_json(payload,root/'summary.json')
        lines=['# 单隐藏层 MLP / XGBoost：训练集内部调参与16组对照','',
               '旧4000图3200/800划分；内层2560/640按图片划分，仅用U_SN和seed43各搜索12个候选。',
               '每模型/分类器选同一组参数并迁移到全部16组，最终seeds43/44/45；不是每组独立最优。',
               '新头经过内部调参，旧三隐藏层头为固定参数；差异不能单独归因于网络深度。',
               '所有值为原800图探索性点估计，非新2000图独立确认。无bootstrap，不宣称显著性。',
               '保留原K4数值FAIL例外和旧InternVL严格CPU复核状态；新头使用同设备/同batch复核。','',
               '## 选定参数','', '| 模型 | 分类器 | 参数 |','|---|---|---|']
        for model,value in summaries.items():
            for family,config in value['selected'].items():
                lines.append(f'| {model} | {family} | `{json.dumps(config,sort_keys=True)}` |')
        rows=[]
        for family in ('three_hidden_fixed','one_hidden','xgboost'):
            lines += ['',f'## {family}：AUROC / HALL-AUPR（%），三seed概率ensemble','',
                      '| 特征组 | '+' | '.join(study.MODELS)+' |','|---|'+'---:|'*len(study.MODELS)]
            for spec in study.FEATURE_GROUPS:
                cells=[]
                for model,value in summaries.items():
                    group=value['methods'][family][spec]
                    report=group['ensemble_reports']['fixed_0.5']
                    cells.append(f"{100*report['auc']:.3f} / {100*report['hallucination_positive']['aupr']:.3f}")
                    for rule,r in group['ensemble_reports'].items():
                        rows.append(dict(model=model,family=family,feature_set=spec,threshold_rule=rule,
                                         ensemble_auroc=r['auc'],ensemble_real_aupr=r['real_positive']['aupr'],
                                         ensemble_hall_aupr=r['hallucination_positive']['aupr'],
                                         real_precision=r['real_positive']['precision'],real_recall=r['real_positive']['recall'],real_f1=r['real_positive']['f1'],
                                         hall_precision=r['hallucination_positive']['precision'],hall_recall=r['hallucination_positive']['recall'],hall_f1=r['hallucination_positive']['f1'],
                                         seed_auc_mean=group['seed_mean_std']['auc']['mean'],seed_auc_std=group['seed_mean_std']['auc']['std']))
                lines.append('| '+spec+' | '+' | '.join(cells)+' |')
        text='\n'.join(lines)+'\n'
        report=root/'summary.md'
        if report.exists():
            if report.read_text()!=text:
                raise ValueError('Changed published report')
        else:
            report.write_text(text)
        import csv, io
        buffer=io.StringIO(); writer=csv.DictWriter(buffer,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        table=root/'groups.csv'
        if table.exists():
            if table.read_text()!=buffer.getvalue().replace('\r\n','\n'):
                raise ValueError('Changed published table')
        else:
            table.write_text(buffer.getvalue())
        print('PUBLISHED',report,flush=True)
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',nargs='+',choices=study.MODELS)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--summarize-only',action='store_true')
    args=parser.parse_args()
    if args.summarize_only:
        if not publish():
            raise ValueError('Not all four model summaries are complete')
        return
    if not args.models:
        parser.error('--models is required unless --summarize-only')
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
