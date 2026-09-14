"""Existing native SVAR/MetaToken heads on the exact regional-AE image 811 split."""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import pickle
import sys
import threading
import time
import warnings

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import train_baselines_811 as shared
from detection.baselines import (
    SVARMLP, _seed_everything, build_metatoken_classifier,
    raw_labels_to_hallucination_targets, sklearn_hallucination_scores,
    torch_hallucination_scores, train_torch_detector,
    select_detection_threshold, evaluate_detection_scores,
)
from features.tc_fvpa_artifacts import atomic_json_save, atomic_torch_save

OUT = ROOT/'outputs/native_baselines_811_v1'
HEADS = {'svar_native': 'svar', 'metatoken_lr': 'metatoken', 'metatoken_gb': 'metatoken'}
SEEDS = (43, 44, 45)
SVAR = dict(hidden_dim=248, epochs=50, learning_rate=0.001, batch_size=32,
            weight_decay=0.0, weighted_sampler=False, standardize=False,
            early_stopping_patience=5)


def read(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def write_progress(model, stage, completed, status='running', **values):
    atomic_json_save(dict(stage=stage, completed=completed, total=9, status=status,
        heartbeat=datetime.now(timezone.utc).isoformat(), **values), OUT/model/'progress.json')


def prepare(model):
    data = shared.prepare(model, output_root=OUT/'feature_cache')
    feature_protocol = json.loads((OUT/'feature_cache'/model/'protocol.json').read_text())
    protocol = dict(schema='native-baselines-811-v1', model=model, seeds=list(SEEDS),
        feature_fingerprint=data['fingerprint'], split=feature_protocol['split'],
        counts=feature_protocol['counts'], dimensions=feature_protocol['dimensions'],
        svar=SVAR, svar_selection='minimum validation loss; patience5; max50; no scheduler',
        metatoken=dict(lr='StandardScaler+LogisticRegression(lbfgs,max_iter=2000)',
                       gb='StandardScaler+GradientBoostingClassifier(n_estimators=100)'),
        training='Fit train3200 only; scaler fit train only; no hyperparameter search',
        threshold='Validation REAL-F1; also report fixed0.5; AUROC/AP threshold independent',
        probabilities='Saved probabilities are P(REAL); native heads train 1=HALL internally',
        scope='Existing project native classifier implementation, controlled exact shared mentions, image811 adaptation; not a new paper-dataset reproduction',
        information=feature_protocol['information'], caveat=feature_protocol['caveat'])
    protocol['fingerprint'] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    path = OUT/model/'protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == protocol, 'Native protocol changed'
    atomic_json_save(protocol, path)
    return data, protocol


def train_head(data, head, seed, device, callback=None):
    x, y, masks = data['groups'][HEADS[head]], data['y'], data['masks']
    train, val, test = (masks[k] for k in ('train', 'validation', 'test'))
    payload = dict(seed=seed, head=head)
    if head == 'svar_native':
        _seed_everything(seed)
        estimator = SVARMLP(x.shape[1], hidden_dim=SVAR['hidden_dim'])
        trained = train_torch_detector(model=estimator,
            X_train=x[train], raw_y_train=y[train], X_val=x[val], raw_y_val=y[val],
            X_test=x[test], raw_y_test=y[test], device=device, seed=seed,
            positive_class='real', strict_82_no_validation=False, epoch_callback=callback,
            **{k:v for k,v in SVAR.items() if k != 'hidden_dim'})
        payload.update(state_dict=trained.state_dict, history=trained.history,
            best_epoch=int(min(trained.history, key=lambda r:r['val_loss'])['epoch']))
        hall = {k:torch_hallucination_scores(estimator, x[m], torch.device(device)) for k,m in masks.items()}
    else:
        estimator = build_metatoken_classifier(head.rsplit('_', 1)[1], seed=seed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            estimator.fit(x[train], raw_labels_to_hallucination_targets(y[train]))
        payload['warnings'] = [str(w.message) for w in caught]
        hall = {k:sklearn_hallucination_scores(estimator, x[m]) for k,m in masks.items()}
        scaler = estimator.named_steps['standardize']
        np.testing.assert_allclose(scaler.mean_, x[train].mean(axis=0, dtype=np.float64), rtol=1e-10, atol=1e-10)
        assert int(scaler.n_samples_seen_) == int(train.sum())
    for k, p in hall.items():
        assert p.shape == (int(masks[k].sum()),) and np.isfinite(p).all()
        payload[k+'_probabilities'] = 1.0-p
    threshold = select_detection_threshold(y[val], hall['validation'], positive_class='real')
    payload['threshold'] = threshold
    payload['threshold_reports'] = {name:evaluate_detection_scores(y[test], hall['test'], t, positive_class='real')
        for name,t in [('validation_f1', threshold), ('fixed_0.5', 0.5)]}
    payload['test_metrics'] = shared.study.scores(y[test], payload['test_probabilities'])
    return payload, estimator


def summarize(model, data):
    root = OUT/model
    protocol = json.loads((root/'protocol.json').read_text())
    rows, seeds = [], []
    y = data['y'][data['masks']['test']]
    for head in HEADS:
        heads = [read(root/head/f'seed{s}'/'result.pt') for s in SEEDS]
        assert all(h['fingerprint'] == protocol['fingerprint'] for h in heads)
        p = np.stack([h['test_probabilities'] for h in heads])
        metrics = [shared.study.scores(y, v) for v in p]
        for seed,h,m in zip(SEEDS, heads, metrics):
            for k in m: assert abs(m[k]-h['test_metrics'][k]) < 1e-12
            seeds.append(dict(model=model, head=head, seed=seed, **m,
                              best_epoch=h.get('best_epoch', ''), epochs=len(h.get('history', []))))
        rows.append(dict(model=model, head=head,
            **{k+'_'+stat:float(fn([m[k] for m in metrics])) for k in metrics[0]
               for stat,fn in [('mean', np.mean), ('std', np.std)]},
            **{'ensemble_'+k:v for k,v in shared.study.scores(y, p.mean(0)).items()}))
    for name, values in [('detection', rows), ('seed_metrics', seeds)]:
        shared.study.original.base.write_csv(values, root/(name+'.csv'))
    write_progress(model, '811原生SVAR/MetaToken完成', 9, status='completed')
    return rows


def pipeline(model, device):
    root = OUT/model
    root.mkdir(parents=True, exist_ok=True)
    with (root/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = dict(stage='811原生准备', completed=0)
        stop = threading.Event()
        def heartbeat():
            while not stop.is_set():
                write_progress(model, **state)
                stop.wait(10)
        worker = threading.Thread(target=heartbeat, daemon=True)
        worker.start()
        try:
            while not (shared.AE_OUT/model/'matrices.pt').exists():
                state.update(stage='811原生等待共享划分', status='waiting')
                time.sleep(10)
            state['status'] = 'running'
            data, protocol = prepare(model)
            for head in HEADS:
                for seed in SEEDS:
                    state.update(stage=f'811原生 {head} seed{seed}', epoch=0)
                    folder = root/head/f'seed{seed}'
                    if not (folder/'result.pt').exists():
                        result, estimator = train_head(data, head, seed, device,
                            callback=lambda epoch:state.update(epoch=epoch))
                        result['fingerprint'] = protocol['fingerprint']
                        folder.mkdir(parents=True, exist_ok=True)
                        if head != 'svar_native':
                            temp = folder/'model.pkl.tmp'
                            with temp.open('wb') as f:pickle.dump(estimator, f, protocol=pickle.HIGHEST_PROTOCOL)
                            temp.replace(folder/'model.pkl')
                        atomic_torch_save(result, folder/'result.pt')
                        print('DONE', model, head, seed, result['test_metrics'], flush=True)
                    else:
                        assert read(folder/'result.pt')['fingerprint'] == protocol['fingerprint']
                    state['completed'] += 1
            stop.set();worker.join()
            summarize(model, data)
        except BaseException as exc:
            stop.set();worker.join()
            write_progress(model, '811原生失败', state['completed'], status='failed', error=str(exc)[:160])
            raise


def summarize_all():
    rows, seeds = [], []
    for model in shared.study.original.MODELS:
        with (OUT/model/'detection.csv').open() as f:rows.extend(csv.DictReader(f))
        with (OUT/model/'seed_metrics.csv').open() as f:seeds.extend(csv.DictReader(f))
    assert len(rows) == 12 and len(seeds) == 36
    for name,values in [('detection',rows),('seed_metrics',seeds)]:
        shared.study.original.base.write_csv(values,OUT/(name+'.csv'))
    lines = ['# SVAR / MetaToken 原生分类器：相同811划分', '',
        '四模型均为3200训练/400验证/400测试图片，全mentions，seeds43/44/45；均值±std为三seed指标统计，另存概率ensemble。',
        'SVAR：Linear(D,248)-ReLU-Linear(248,2)，Adam .001，batch32，最多50epochs，验证loss最优checkpoint及patience5早停；无BN/dropout/标准化/学习率调度。',
        'MetaToken：StandardScaler+LR(lbfgs,max_iter2000)和StandardScaler+GB100；标准化只拟合训练，固定参数，不搜索、不合并验证集重训。',
        '训练内部1=HALL，所有保存概率转成P(REAL)。阈值取验证REAL-F1并另报0.5，以下AUROC/HALL-AUPR不依赖阈值。',
        '使用项目已有原生分类器实现、controlled同目标特征及811适配，不代表复现论文原始数据集。MetaToken保留完整回答长度/对象span统计。旧800图已有研究使用，本轮是探索性评估。', '',
        '| 模型 | 原生分类器 | AUROC mean±std (%) | HALL-AUPR mean±std (%) |', '|---|---|---:|---:|']
    for r in rows:
        lines.append('| '+r['model']+' | '+r['head']+' | '+' | '.join(
            f"{100*float(r[k+'_mean']):.2f} ± {100*float(r[k+'_std']):.2f}" for k in ('AUROC','HALL_AUPR'))+' |')
    lines += ['', '## 与统一分类器比较', '', '每格AUROC / HALL-AUPR（%，三seed均值）；只展示已完成三个seed的统一头，不按测试结果选择分类器。', '',
              '| 模型 | 特征 | 原生 | 统一三层 | 统一单层 | 统一XGB |', '|---|---|---:|---:|---:|']
    def fmt(r):return f"{100*float(r['AUROC_mean']):.2f} / {100*float(r['HALL_AUPR_mean']):.2f}"
    for r in rows:
        model,head = r['model'],r['head'];data=read(OUT/'feature_cache'/model/'matrices.pt')
        y=data['y'][data['masks']['test']];cells=[]
        for clf in shared.CLASSIFIERS:
            paths=[shared.OUT/model/clf/HEADS[head]/f'seed{s}'/'result.pt' for s in SEEDS]
            if not all(p.exists() for p in paths):cells.append('运行中');continue
            heads=[read(p) for p in paths]
            assert all(h['fingerprint']==data['fingerprint'] for h in heads)
            v=[shared.study.scores(y,h['test_probabilities']) for h in heads]
            cells.append(fmt({k+'_mean':np.mean([m[k] for m in v]) for k in v[0]}))
        lines.append('| '+model+' | '+head+' | '+fmt(r)+' | '+' | '.join(cells)+' |')
    lines += ['', '## AE拼接参考', '',
        '同一400测试图；V为视觉AE+log1p(S_V_all)，VP+G为视觉与prompt区域块加generation区域块。列出预先指定的三层和XGB，不按测试分数择优。', '',
        '| 模型 | V 三层 | VP+G 三层 | V XGB | VP+G XGB |', '|---|---:|---:|---:|---:|']
    for model in shared.study.original.MODELS:
        data=read(shared.AE_OUT/model/'matrices.pt');y=data['y'][data['masks']['test']];cells=[]
        for clf,group in [('three_hidden','visual'),('three_hidden','vp_generation'),('xgb','visual'),('xgb','vp_generation')]:
            paths=[shared.AE_OUT/model/clf/group/f'seed{s}'/'result.pt' for s in SEEDS]
            if not all(p.exists() for p in paths):cells.append('运行中');continue
            heads=[read(p) for p in paths]
            assert all(h['fingerprint']==data['fingerprint'] for h in heads)
            v=[shared.study.scores(y,h['test_probabilities']) for h in heads]
            cells.append(fmt({k+'_mean':np.mean([m[k] for m in v]) for k in v[0]}))
        lines.append('| '+model+' | '+' | '.join(cells)+' |')
    lines += ['', '原生与统一头同时改变了结构/优化或标准化等配置，差值反映整套分类器配置变化，不能单独归因为标准化或层数。尚未计算新的配对bootstrap，不宣称差值显著。', '']
    for path in (OUT/'summary.md', ROOT/'docs/NATIVE_BASELINES_811_RESULTS.md'):path.write_text('\n'.join(lines))
    atomic_json_save(dict(status='completed', heads=36, heartbeat=datetime.now(timezone.utc).isoformat()), OUT/'summary_progress.json')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=shared.study.original.MODELS)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--summarize-all',action='store_true')
    args=parser.parse_args();torch.set_num_threads(1)
    if args.summarize_all:summarize_all()
    elif args.model:pipeline(args.model,args.device)
    else:parser.error('--model or --summarize-all required')
