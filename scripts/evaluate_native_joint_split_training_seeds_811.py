"""Joint split/training seed sensitivity for native SVAR and MetaToken heads."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from features.tc_fvpa_artifacts import atomic_json_save, atomic_torch_save
from scripts import train_native_baselines_811 as native
from scripts.evaluate_joint_split_training_seeds_811 import image_split, masks_for

MODELS = tuple(native.shared.study.original.MODELS)
HEADS = tuple(native.HEADS)
SEEDS = (43, 44, 45)
SOURCE = ROOT/'outputs/native_baselines_811_v1/feature_cache'
FIXED = ROOT/'outputs/native_baselines_811_v1'
OUT = ROOT/'outputs/native_joint_split_training_seed_811_v1'
OURS = ROOT/'outputs/joint_split_training_seed_811_v1'


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer=csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def save_pickle(value, path):
    path.parent.mkdir(parents=True, exist_ok=True); temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as stream: pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def protocol():
    fingerprints={}; mention_signatures={}; split_hashes={}
    for model in MODELS:
        data=native.read(SOURCE/model/'matrices.pt'); fingerprints[model]=data['fingerprint']
        mention_signatures[model]=hashlib.sha256(json.dumps(
            [(m['image_id'],m['target_key'],m['label']) for m in data['mentions']]).encode()).hexdigest()
        split_path=ROOT/'outputs'/model/'COCO4000-INSLEN-OFFICIAL-TARGET/image_splits.json'
        split_hashes[model]=hashlib.sha256(split_path.read_bytes()).hexdigest()
    value=dict(schema='native-joint-split-training-seed-811-v1',models=list(MODELS),heads=list(HEADS),seeds=list(SEEDS),
        split='For each seed, permute all 4000 sorted image IDs into 3200/400/400',
        seed_scope='Same seed controls image split and classifier randomness',
        svar=dict(native.SVAR,architecture='Linear(D,248)-ReLU-Linear(248,2)',optimizer='Adam',
            checkpoint='minimum validation loss; patience5; max50; no scheduler'),
        metatoken=dict(lr='train-only StandardScaler + LogisticRegression(lbfgs,max_iter2000)',
            gb='train-only StandardScaler + GradientBoostingClassifier(n_estimators=100)'),
        training='No hyperparameter search; fit new train split only; validation selects SVAR checkpoint; no train+validation refit',
        metrics='Per-seed test AUROC and HALL-positive AUPR; arithmetic mean and population std; no probability ensemble',
        comparison='Existing fixed split20260912, seeds43/44/45 versus joint split+training seeds43/44/45',
        feature_fingerprints=fingerprints,mention_signatures=mention_signatures,image_split_hashes=split_hashes,
        caveat='Only three joint seeds; test image sets differ, so variance combines cohort composition and training randomness')
    value['fingerprint']=hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
    path=OUT/'protocol.json'
    if path.exists() and json.loads(path.read_text()) != value: raise ValueError('Protocol changed')
    atomic_json_save(value,path); return value


def progress(model,stage,done,total,status='running',**values):
    atomic_json_save(dict(stage=stage,completed=done,total=total,status=status,
        heartbeat=datetime.now(timezone.utc).isoformat(),**values),OUT/model/'progress.json')


def run_model(model,device):
    p=protocol();root=OUT/model;root.mkdir(parents=True,exist_ok=True)
    with (root/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        source=native.read(SOURCE/model/'matrices.pt')
        if source['fingerprint'] != p['feature_fingerprints'][model]:raise ValueError('Feature cache changed')
        total=len(HEADS)*len(SEEDS);done=0
        for head in HEADS:
            for seed in SEEDS:
                path=root/head/f'seed{seed}/result.pt'
                if path.exists():result=native.read(path)
                else:
                    split=image_split(model,seed);masks=masks_for(source['mentions'],split)
                    for name,mask in masks.items():
                        if set(source['y'][mask].tolist()) != {0,1}:raise ValueError(f'{name} lacks a class')
                    data=dict(source,masks=masks);stage=f'联合种子 {head} seed{seed}'
                    result,estimator=native.train_head(data,head,seed,device,
                        callback=lambda epoch:progress(model,stage,done,total,epoch=epoch))
                    y=source['y'][masks['test']]
                    result.update(model=model,head=head,split_seed=seed,training_seed=seed,split=split,
                        split_counts={k:int(v.sum()) for k,v in masks.items()},test_hall_count=int((y==0).sum()),
                        test_real_count=int((y==1).sum()),test_hall_prevalence=float((y==0).mean()),
                        fingerprint=p['fingerprint'])
                    if head!='svar_native':save_pickle(estimator,root/head/f'seed{seed}/model.pkl')
                    atomic_torch_save(result,path)
                if result['fingerprint']!=p['fingerprint'] or result['head']!=head:raise ValueError('Bad resume')
                done+=1;progress(model,f'联合种子 {head} seed{seed}',done,total)
        progress(model,'联合种子SVAR/MetaToken完成',total,total,status='completed')


def aggregate(values,key):
    x=np.asarray([v[key] for v in values]);return float(x.mean()),float(x.std())


def summarize():
    p=protocol();rows=[];per_seed=[]
    for model in MODELS:
        for head in HEADS:
            joint=[native.read(OUT/model/head/f'seed{s}/result.pt') for s in SEEDS]
            fixed=[native.read(FIXED/model/head/f'seed{s}/result.pt') for s in SEEDS]
            for seed,value in zip(SEEDS,joint):
                per_seed.append(dict(model=model,head=head,regime='joint_split_training',seed=seed,
                    **value['test_metrics'],test_mentions=value['test_hall_count']+value['test_real_count'],
                    test_hall_prevalence=value['test_hall_prevalence'],best_epoch=value.get('best_epoch','')))
            for regime,values in (('fixed_split',fixed),('joint_split_training',joint)):
                metrics=[v['test_metrics'] for v in values];auc=aggregate(metrics,'AUROC');ap=aggregate(metrics,'HALL_AUPR')
                rows.append(dict(model=model,head=head,regime=regime,AUROC_mean=auc[0],AUROC_std=auc[1],
                    HALL_AUPR_mean=ap[0],HALL_AUPR_std=ap[1]))
    write_csv(rows,OUT/'summary.csv');write_csv(per_seed,OUT/'per_seed.csv')
    lookup={(r['model'],r['head'],r['regime']):r for r in rows}
    names=dict(qwen2_5_vl_7b='Qwen2.5-VL-7B',llava_1_5_7b='LLaVA-1.5-7B',
        qwen3_vl_8b='Qwen3-VL-8B',internvl_2_5_8b='InternVL2.5-8B')
    display=dict(svar_native='SVAR',metatoken_lr='MetaToken LR',metatoken_gb='MetaToken GB')
    lines=['# 四模型：SVAR/MetaToken联合数据划分与训练种子','',
        '设置：每个seed将完整4000图重新划分为3200 train / 400 validation / 400 test；同一seed同时控制划分和分类器随机性。seeds43/44/45，全mentions。',
        'SVAR：Linear(D,248)-ReLU-Linear(248,2)，Adam lr0.001、batch32、最多50 epochs、无标准化/BN/dropout/调度，validation loss checkpoint、patience5。',
        'MetaToken LR：训练集StandardScaler+LogisticRegression(lbfgs,max_iter2000)；GB：训练集StandardScaler+GradientBoosting100。固定参数，无搜索，无train+validation refit。',
        '每个seed在各自400图test上计算AUROC/HALL-AUPR；表中为算术均值±总体标准差(ddof=0)，不是概率ensemble。联合std同时包含测试图片构成与训练随机性。','',
        '| 模型 | 分类器 | 固定划分 AUROC | 联合种子 AUROC | 固定划分 HALL-AUPR | 联合种子 HALL-AUPR |',
        '|---|---|---:|---:|---:|---:|']
    fmt=lambda r,k:f"{100*r[k+'_mean']:.2f} ± {100*r[k+'_std']:.2f}"
    for model in MODELS:
        for head in HEADS:
            fixed=lookup[model,head,'fixed_split'];joint=lookup[model,head,'joint_split_training']
            lines.append(f"| {names[model]} | {display[head]} | {fmt(fixed,'AUROC')} | {fmt(joint,'AUROC')} | {fmt(fixed,'HALL_AUPR')} | {fmt(joint,'HALL_AUPR')} |")
    lines += ['','## 同一联合种子下与本方法比较','',
        '差值为本方法减原生SVAR，单位为百分点。V和VP+G均为预先定义的真实RMS all-attention K32特征；不根据本表测试结果选择特征。','',
        '| 模型 | 本方法特征 | 本方法 AUROC/AP | SVAR AUROC/AP | 差值 AUROC/AP |','|---|---|---:|---:|---:|']
    for model in MODELS:
        svar=lookup[model,'svar_native','joint_split_training']
        for group in ('visual','vp_generation'):
            values=[native.read(OURS/model/group/f'seed{s}.pt')['test_metrics'] for s in SEEDS]
            auc=float(np.mean([v['AUROC'] for v in values]));ap=float(np.mean([v['HALL_AUPR'] for v in values]))
            lines.append(f"| {names[model]} | {group} | {100*auc:.2f}/{100*ap:.2f} | {100*svar['AUROC_mean']:.2f}/{100*svar['HALL_AUPR_mean']:.2f} | {100*(auc-svar['AUROC_mean']):+.2f}/{100*(ap-svar['HALL_AUPR_mean']):+.2f} |")
    lines += ['','## 联合种子逐次测试结果','',
        '| 模型 | 分类器 | seed43 AUROC/AP | seed44 AUROC/AP | seed45 AUROC/AP |','|---|---|---:|---:|---:|']
    for model in MODELS:
        for head in HEADS:
            values=[native.read(OUT/model/head/f'seed{s}/result.pt')['test_metrics'] for s in SEEDS]
            cells=[f"{100*v['AUROC']:.2f}/{100*v['HALL_AUPR']:.2f}" for v in values]
            lines.append(f"| {names[model]} | {display[head]} | {' | '.join(cells)} |")
    lines += ['','三个联合test集彼此不同；固定划分三次共享同一test集。MetaToken LR在固定划分下跨训练seed完全相同，是因为LBFGS流程确定；联合种子下的差异来自数据划分。只有三个新划分，不作显著性结论。','']
    (ROOT/'docs/NATIVE_JOINT_SPLIT_TRAINING_SEEDS_811_RESULTS.md').write_text('\n'.join(lines))
    atomic_json_save(dict(status='completed',fingerprint=p['fingerprint'],heads=len(per_seed)),OUT/'status.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--models',nargs='+',choices=MODELS)
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--summarize',action='store_true')
    args=parser.parse_args();torch.set_num_threads(1)
    if args.summarize:summarize()
    elif args.models:
        for model in args.models:run_model(model,args.device)
    else:parser.error('--models or --summarize required')
