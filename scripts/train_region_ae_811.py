"""Image-level 3200/400/400 regional AE study, with an explicit validation set."""
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import pickle
import sys
import threading
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import ParameterGrid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import train_all_attention_ae_strength as original
from scripts.train_torch_probe_feature_sets import TorchProbeConfig, train_and_evaluate_probe
from features.tc_fvpa_artifacts import atomic_json_save, atomic_torch_save

OUT = ROOT/'outputs/all_attention_ae_811_v1'
GROUPS = original.GROUPS+('legacy_visual',)
SEEDS = original.SEEDS
SPLIT_SEED = 20260912


def split_images(outer, mode='preserve_train'):
    rng = np.random.default_rng(SPLIT_SEED)
    if mode == 'preserve_train':
        train = sorted(outer['train'])
        rest = rng.permutation(sorted(outer['test'])).tolist()
    else:
        ids = rng.permutation(sorted(outer['train']+outer['test'])).tolist()
        train, rest = ids[:3200], ids[3200:]
    split = dict(train=train, validation=rest[:400], test=rest[400:])
    assert [len(split[k]) for k in split] == [3200, 400, 400]
    assert len(set().union(*map(set, split.values()))) == 4000
    return split


def scores(y, p):
    return dict(AUROC=float(roc_auc_score(y, p)), HALL_AUPR=float(average_precision_score(1-y, 1-p)))


def progress(model, stage, done, total, status='running', **values):
    atomic_json_save(dict(stage=stage, completed=done, total=total, status=status,
        heartbeat=datetime.now(timezone.utc).isoformat(), **values), OUT/model/'progress.json')


def prepare(model, mode):
    source = original.base.read(original.OUT/model/'matrices.pt')
    legacy = original.base.read(ROOT/'outputs/ffn_source_composition_v1'/model/'matrices.pt')
    assert source['mentions'] == legacy['mentions'] and source['ntrain'] == legacy['ntrain']
    np.testing.assert_array_equal(source['y'], legacy['y'])
    outer = json.loads((ROOT/'outputs'/model/original.base.EXPERIMENT/'image_splits.json').read_text())
    split = split_images(outer, mode)
    ids = np.asarray([m['image_id'] for m in source['mentions']])
    masks = {k:np.isin(ids, v) for k,v in split.items()}
    assert np.all(sum(m.astype(int) for m in masks.values()) == 1)
    for mask in masks.values():
        assert set(np.asarray(source['y'])[mask]) == {0, 1}
    groups = dict(source['groups'], legacy_visual=legacy['groups']['F'])
    assert set(groups) == set(GROUPS)
    cfg = TorchProbeConfig(split_protocol='image_811_validation', checkpoint_selection='minimum_val_loss')
    protocol = dict(model=model, split=split, split_seed=SPLIT_SEED, split_mode=mode,
        source_fingerprint=source['fingerprint'], fixed=asdict(cfg), groups=list(GROUPS),
        mlp_grid=original.search.GRID, xgb_grid=original.search.TREE_GRIDS['xgb'], seeds=list(SEEDS),
        selection='MLP3 validation loss checkpoint/scheduler/stopping; sklearn/XGB seed43 validation AUROC then HALL_AP then index; fit train3200 only, never refit train+validation',
        thresholds='train REAL-F1 and fixed0.5; never test threshold selection',
        comparison='original 82 heads rescored on same400 test when preserving training images',
        caveat='validation/test derive from previously inspected original800; exploratory, not independent confirmation',
        dimensions={k:v.shape[1] for k,v in groups.items()})
    protocol = json.loads(json.dumps(protocol))
    digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode())
    for k,x in sorted(groups.items()): digest.update(k.encode());digest.update(x.tobytes())
    protocol['fingerprint'] = digest.hexdigest()
    root = OUT/model
    path = root/'protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == protocol, 'Changed 811 protocol; refusing stale results'
    atomic_json_save(protocol, path)
    data = dict(groups=groups, y=np.asarray(source['y']), masks=masks, mentions=source['mentions'],
                original_ntrain=source['ntrain'], fingerprint=protocol['fingerprint'])
    if not (root/'matrices.pt').exists(): atomic_torch_save(data, root/'matrices.pt')
    return data


def fixed(model, device, data):
    done = 0
    total = len(data['groups'])*len(SEEDS)
    train, val, test = (data['masks'][k] for k in ('train','validation','test'))
    y = data['y']
    for group, x in data['groups'].items():
        for seed in SEEDS:
            folder = OUT/model/'three_hidden'/group/f'seed{seed}'
            if not (folder/'result.pt').exists():
                def callback(epoch): progress(model, '811 三层MLP '+group, done, total, epoch=epoch)
                cfg = TorchProbeConfig(seed=seed, split_protocol='image_811_validation', checkpoint_selection='minimum_val_loss')
                result = train_and_evaluate_probe(X_train=x[train], y_train=y[train], X_val=x[val], y_val=y[val],
                    X_test=x[test], y_test=y[test], config=cfg, device=torch.device(device), output_dir=str(folder),
                    return_probabilities=True, epoch_callback=callback)
                value = dict(seed=seed, fingerprint=data['fingerprint'], metrics=result)
                for part in ('train','validation','test'):
                    value[part+'_probabilities'] = np.asarray(result.pop(part+'_probabilities'))
                atomic_torch_save(value, folder/'result.pt')
            done += 1
            progress(model, '811 三层MLP '+group, done, total)


def search_group(model, group, classifier):
    data = original.base.read(OUT/model/'matrices.pt')
    x, y = data['groups'][group], data['y']
    train, val, test = (data['masks'][k] for k in ('train','validation','test'))
    root = OUT/model/classifier/group
    grid = list(ParameterGrid(original.search.GRID if classifier=='one_hidden' else original.search.TREE_GRIDS['xgb']))
    kind = 'mlp' if classifier=='one_hidden' else 'xgb'
    trials = []
    for index, params in enumerate(grid):
        path = root/'search'/f'trial{index:02d}.json'
        if path.exists(): trial = json.loads(path.read_text())
        else:
            estimator, info = original.search.fit_classifier(params, 43, x[train], y[train], kind)
            trial = dict(index=index, params=params, seed=43, fingerprint=data['fingerprint'],
                scope='validation400', **scores(y[val], estimator.predict_proba(x[val])[:,1]), **info)
            atomic_json_save(trial, path)
        assert trial['fingerprint'] == data['fingerprint'] and trial['params'] == params
        trials.append(trial)
    best = max(trials, key=lambda t:(t['AUROC'], t['HALL_AUPR'], -t['index']))
    atomic_json_save(dict(best=best, candidates=trials), root/'selection.json')
    for seed in SEEDS:
        folder = root/f'seed{seed}';path=folder/'result.pt'
        if path.exists(): continue
        estimator, info = original.search.fit_classifier(best['params'], seed, x[train], y[train], kind)
        value = dict(seed=seed, params=best['params'], fingerprint=data['fingerprint'], **info)
        for part, mask in data['masks'].items(): value[part+'_probabilities'] = estimator.predict_proba(x[mask])[:,1]
        value['metrics'] = original.search.metric_reports(y[train], value['train_probabilities'], y[test], value['test_probabilities'])
        folder.mkdir(parents=True, exist_ok=True)
        temp = folder/'model.pkl.tmp'
        with temp.open('wb') as stream: pickle.dump(estimator, stream, protocol=pickle.HIGHEST_PROTOCOL)
        temp.replace(folder/'model.pkl')
        atomic_torch_save(value, path)
    return dict(model=model, group=group, classifier=classifier, selected=best['params'])


def summarize(model, data):
    root=OUT/model;rows=[];seed_rows=[];comparisons=[]
    y=data['y'][data['masks']['test']]
    protocol=json.loads((root/'protocol.json').read_text())
    for group in GROUPS:
        for classifier in ('three_hidden','one_hidden','xgb'):
            heads=[original.base.read(root/classifier/group/f'seed{s}'/'result.pt') for s in SEEDS]
            assert all(h['fingerprint']==data['fingerprint'] for h in heads)
            p=np.stack([h['test_probabilities'] for h in heads]);assert p.shape==(3,len(y)) and np.isfinite(p).all()
            values=[scores(y,v) for v in p]
            for seed, value in zip(SEEDS, values): seed_rows.append(dict(model=model,group=group,classifier=classifier,seed=seed,**value))
            row=dict(model=model,group=group,classifier=classifier,
                **{k+'_'+stat:float(fn([v[k] for v in values])) for k in values[0] for stat,fn in [('mean',np.mean),('std',np.std)]},
                **{'ensemble_'+k:v for k,v in scores(y,p.mean(0)).items()})
            rows.append(row)
            if protocol['split_mode']=='preserve_train' and not (group=='legacy_visual' and classifier!='three_hidden'):
                previous=[]
                for seed in SEEDS:
                    if group=='legacy_visual':
                        path=ROOT/'outputs/ffn_source_composition_v1'/model/'heads/F'/f'seed{seed}'/'result.pt'
                    else:
                        path=original.OUT/model/('heads' if classifier=='three_hidden' else classifier)/group/f'seed{seed}'/'result.pt'
                    h=original.base.read(path)
                    old_p=np.asarray(h['test_probabilities'])[data['masks']['test'][data['original_ntrain']:]]
                    previous.append(scores(y,old_p))
                comparisons.append(dict(model=model,group=group,classifier=classifier,
                    **{k+'_old82_same400':float(np.mean([v[k] for v in previous])) for k in values[0]},
                    **{k+'_delta_pp':100*(row[k+'_mean']-np.mean([v[k] for v in previous])) for k in values[0]}))
    original.base.write_csv(rows,root/'detection.csv');original.base.write_csv(seed_rows,root/'seed_metrics.csv')
    if comparisons: original.base.write_csv(comparisons,root/'vs82_same400.csv')
    text=['# 区域AE 8:1:1检测','', '3200训练/400验证/400测试；全部mentions；三seed均值±std。旧800图已被查看，此轮为探索性分析。',
        '三层checkpoint/调度/早停依据验证loss；单层/XGB依据验证AUROC选参。均只拟合3200训练图，无标准化。', '',
        '| 特征 | 分类器 | AUROC | HALL-AUPR |','|---|---|---:|---:|']
    for r in rows:text.append('| '+r['group']+' | '+r['classifier']+' | '+' | '.join(f'{100*r[k+"_mean"]:.2f} ± {100*r[k+"_std"]:.2f}' for k in ('AUROC','HALL_AUPR'))+' |')
    (root/'summary.md').write_text('\n'.join(text)+'\n')
    progress(model,'811 检测完成',81,81,status='completed')


def pipeline(model, device, workers, mode):
    root=OUT/model;root.mkdir(parents=True,exist_ok=True)
    with (root/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            while True:
                path=original.OUT/model/'progress.json'
                previous=json.loads(path.read_text()) if path.exists() else {}
                if previous.get('status')=='completed': break
                progress(model,'811等待原82：'+previous.get('stage','准备中'),
                    previous.get('completed',0),previous.get('total',1),status='waiting',
                    error=previous.get('error',''));time.sleep(10)
            data=prepare(model,mode);fixed(model,device,data)
            stop=threading.Event()
            def monitor():
                while not stop.is_set():
                    n=sum(len(list((root/c).glob('*/search/trial*.json'))) for c in ('one_hidden','xgb'))
                    h=sum(len(list((root/c).glob('*/seed*/result.pt'))) for c in ('one_hidden','xgb'))
                    progress(model,f'811 选参 {n}/270 正式 {h}/54',n+h,324);stop.wait(10)
            thread=threading.Thread(target=monitor,daemon=True);thread.start()
            try:
                with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn')) as pool:
                    jobs=[pool.submit(search_group,model,g,c) for c in ('one_hidden','xgb') for g in GROUPS]
                    for job in as_completed(jobs):print('DONE',job.result(),flush=True)
            finally:stop.set();thread.join()
            summarize(model,data)
        except BaseException as exc:
            progress(model,'811 失败',0,1,status='failed',error=str(exc)[:180]);raise


def summarize_all():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'.summary.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def update(status, **values):
            atomic_json_save(dict(status=status,heartbeat=datetime.now(timezone.utc).isoformat(),**values),OUT/'summary_progress.json')
        try:
            while True:
                states={m:json.loads((OUT/m/'progress.json').read_text()) if (OUT/m/'progress.json').exists() else {} for m in original.MODELS}
                completed=[m for m,s in states.items() if s.get('status')=='completed']
                if len(completed)==4:break
                update('waiting',completed_models=completed,failed_models=[m for m,s in states.items() if s.get('status')=='failed'])
                time.sleep(10)
            update('running')
            collections={}
            for name in ('detection','seed_metrics','vs82_same400'):
                rows=[]
                for m in original.MODELS:
                    path=OUT/m/(name+'.csv')
                    if path.exists():
                        with path.open() as f:rows.extend(csv.DictReader(f))
                if rows:original.base.write_csv(rows,OUT/(name+'.csv'))
                collections[name]=rows
            assert len(collections['detection'])==108 and len(collections['seed_metrics'])==324
            lines=['# 区域AE 8:1:1检测结果','',
                '四模型各4000图，按图片分3200训练/400验证/400测试，保留全部mentions。seed20260912冻结划分，检测器seeds43/44/45；下面是均值±总体标准差。',
                '默认保留原3200训练图，将原800图分成400验证/400测试。原800图已有历史研究使用，因此本轮属于探索性评估。',
                '三层MLP保持[128,64,32]/BN/dropout.3/Adam等配置，改为验证loss选择checkpoint、学习率调度及早停。单层sklearn12候选、XGB18候选，仅根据400验证图的seed43 AUROC/AP选参；最终三seed只拟合3200训练图。',
                'V、VP、G、VP+G四组为区域原值AE＋对应All-attention gross的log1p，VP先加SV+SP再log；VP区域AE独立重算。ae_only为去掉gross的对应控制；legacy_visual为旧[AE_V,log1p(S_E)]，同样重训。',
                '新模型和旧82模型均在相同400测试图片、相同mentions评估；vs82_same400.csv保存配对点差。不能直接用新400图分数减旧800图总分。无新增bootstrap。','',
                '| 模型 | 特征 | 分类器 | AUROC | HALL-AUPR |','|---|---|---|---:|---:|']
            for r in collections['detection']:
                lines.append('| '+r['model']+' | '+r['group']+' | '+r['classifier']+' | '+' | '.join(
                    f'{100*float(r[k+"_mean"]):.2f} ± {100*float(r[k+"_std"]):.2f}' for k in ('AUROC','HALL_AUPR'))+' |')
            lines+=['','## 与旧82训练在相同400测试图上比较','',
                '下面的变化包含训练协议变化：三层使用验证loss而非训练loss；单层/XGB改用3200/400候选拟合/验证而非旧2560/640。测试图片固定。','',
                '| 模型 | 特征 | 分类器 | ΔAUROC百分点 | ΔHALL-AUPR百分点 |','|---|---|---|---:|---:|']
            for r in collections['vs82_same400']:
                lines.append('| '+r['model']+' | '+r['group']+' | '+r['classifier']+' | '+
                    f'{float(r["AUROC_delta_pp"]):+.2f} | {float(r["HALL_AUPR_delta_pp"]):+.2f} |')
            (OUT/'summary.md').write_text('\n'.join(lines)+'\n')
            (ROOT/'docs/REGION_AE_811_RESULTS.md').write_text('\n'.join(lines)+'\n')
            update('completed',heads=324,groups=108)
        except BaseException as exc:
            update('failed',error=str(exc));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=original.MODELS)
    parser.add_argument('--summarize-all',action='store_true')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--split-mode',choices=('preserve_train','reshuffle'),default='preserve_train')
    args=parser.parse_args();torch.set_num_threads(1)
    if args.summarize_all:summarize_all()
    elif args.model:pipeline(args.model,args.device,args.workers,args.split_mode)
    else:parser.error('--model or --summarize-all is required')
