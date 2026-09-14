"""One-variable SVAR hidden-width 248 -> 256 comparison on fixed and joint 811 splits."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from detection.baselines import (SVARMLP,_seed_everything,train_torch_detector,
    torch_hallucination_scores,select_detection_threshold,evaluate_detection_scores)
from features.tc_fvpa_artifacts import atomic_json_save,atomic_torch_save
from scripts import train_native_baselines_811 as native
from scripts.evaluate_joint_split_training_seeds_811 import image_split,masks_for

MODELS=tuple(native.shared.study.original.MODELS)
SEEDS=(43,44,45)
REGIMES=('fixed_split','joint_split_training')
SOURCE=ROOT/'outputs/native_baselines_811_v1/feature_cache'
OLD_FIXED=ROOT/'outputs/native_baselines_811_v1'
OLD_JOINT=ROOT/'outputs/native_joint_split_training_seed_811_v1'
OUT=ROOT/'outputs/svar_hidden256_811_v1'
HIDDEN=256


def write_csv(rows,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def protocol():
    source={m:native.read(SOURCE/m/'matrices.pt')['fingerprint'] for m in MODELS}
    settings=dict(native.SVAR,hidden_dim=HIDDEN)
    value=dict(schema='svar-hidden256-811-v1',models=list(MODELS),seeds=list(SEEDS),regimes=list(REGIMES),
        comparison='Only hidden_dim changes from248 to256',input='SVAR flattened visual attention ratio; zero-based decoder layers[5,19)',
        architecture='Linear(D,256)-ReLU-Linear(256,2)',settings=settings,
        fixed_split='Existing split seed20260912: 3200/400/400 images',
        joint_split='Each seed43/44/45 independently permutes all4000 images to3200/400/400; same seed controls training',
        training='CrossEntropy; detector labels0=REAL,1=HALL; Adam; no standardization/BN/dropout/scheduler/weighted sampler; min validation-loss checkpoint; no refit',
        metrics='Per-seed test AUROC and HALL-positive AUPR; arithmetic mean and population std; no probability ensemble',
        source_fingerprints=source)
    value['fingerprint']=hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
    path=OUT/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=value:raise ValueError('Protocol changed')
    atomic_json_save(value,path);return value


def progress(model,stage,done,total,status='running',**values):
    atomic_json_save(dict(stage=stage,completed=done,total=total,status=status,
        heartbeat=datetime.now(timezone.utc).isoformat(),**values),OUT/model/'progress.json')


def fit(data,masks,seed,device,callback=None):
    x,y=data['groups']['svar'],data['y'];train,val,test=(masks[k] for k in ('train','validation','test'))
    _seed_everything(seed);model=SVARMLP(x.shape[1],hidden_dim=HIDDEN)
    settings={k:v for k,v in native.SVAR.items() if k!='hidden_dim'}
    trained=train_torch_detector(model=model,X_train=x[train],raw_y_train=y[train],X_val=x[val],raw_y_val=y[val],
        X_test=x[test],raw_y_test=y[test],device=device,seed=seed,positive_class='real',strict_82_no_validation=False,
        epoch_callback=callback,**settings)
    model.load_state_dict(trained.state_dict)
    hall={k:torch_hallucination_scores(model,x[m],torch.device(device)) for k,m in masks.items()}
    real={k:1-v for k,v in hall.items()}
    threshold=select_detection_threshold(y[val],hall['validation'],positive_class='real')
    result=dict(seed=seed,hidden_dim=HIDDEN,input_dim=x.shape[1],state_dict=trained.state_dict,history=trained.history,
        best_epoch=int(min(trained.history,key=lambda r:r['val_loss'])['epoch']),threshold=threshold,
        threshold_reports={name:evaluate_detection_scores(y[test],hall['test'],t,positive_class='real')
            for name,t in [('validation_f1',threshold),('fixed_0.5',.5)]},
        test_metrics=native.shared.study.scores(y[test],real['test']),
        train_probabilities=real['train'],validation_probabilities=real['validation'],test_probabilities=real['test'])
    return result


def run_model(model,device):
    p=protocol();root=OUT/model;root.mkdir(parents=True,exist_ok=True)
    with (root/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        data=native.read(SOURCE/model/'matrices.pt')
        if data['fingerprint']!=p['source_fingerprints'][model]:raise ValueError('Source changed')
        done=0;total=len(REGIMES)*len(SEEDS)
        for regime in REGIMES:
            for seed in SEEDS:
                path=root/regime/f'seed{seed}/result.pt'
                if path.exists():result=native.read(path)
                else:
                    split=(json.loads((SOURCE/model/'protocol.json').read_text())['split']
                           if regime=='fixed_split' else image_split(model,seed))
                    masks=(data['masks'] if regime=='fixed_split' else masks_for(data['mentions'],split))
                    stage=f'SVAR256 {regime} seed{seed}'
                    result=fit(data,masks,seed,device,lambda epoch:progress(model,stage,done,total,epoch=epoch))
                    test_y=data['y'][masks['test']]
                    result.update(model=model,regime=regime,split=split,
                        split_counts={k:int(v.sum()) for k,v in masks.items()},test_hall_count=int((test_y==0).sum()),
                        test_real_count=int((test_y==1).sum()),fingerprint=p['fingerprint'])
                    atomic_torch_save(result,path)
                if result['fingerprint']!=p['fingerprint'] or result['hidden_dim']!=HIDDEN:raise ValueError('Bad resume')
                done+=1;progress(model,f'SVAR256 {regime} seed{seed}',done,total)
        progress(model,'SVAR256两种划分完成',total,total,status='completed')


def agg(values,key):
    x=np.asarray([v[key] for v in values]);return float(x.mean()),float(x.std())


def old_results(model,regime):
    root=OLD_FIXED if regime=='fixed_split' else OLD_JOINT
    return [native.read(root/model/'svar_native'/f'seed{s}/result.pt') for s in SEEDS]


def summarize():
    p=protocol();rows=[];per_seed=[]
    for model in MODELS:
        for regime in REGIMES:
            for width,values in ((248,old_results(model,regime)),(256,[native.read(OUT/model/regime/f'seed{s}/result.pt') for s in SEEDS])):
                metrics=[v['test_metrics'] for v in values];auc=agg(metrics,'AUROC');ap=agg(metrics,'HALL_AUPR')
                rows.append(dict(model=model,regime=regime,hidden_dim=width,AUROC_mean=auc[0],AUROC_std=auc[1],
                    HALL_AUPR_mean=ap[0],HALL_AUPR_std=ap[1]))
                for seed,value in zip(SEEDS,values):per_seed.append(dict(model=model,regime=regime,hidden_dim=width,
                    seed=seed,**value['test_metrics'],best_epoch=value.get('best_epoch','')))
    write_csv(rows,OUT/'summary.csv');write_csv(per_seed,OUT/'per_seed.csv')
    lookup={(r['model'],r['regime'],r['hidden_dim']):r for r in rows}
    names=dict(qwen2_5_vl_7b='Qwen2.5-VL-7B',llava_1_5_7b='LLaVA-1.5-7B',
        qwen3_vl_8b='Qwen3-VL-8B',internvl_2_5_8b='InternVL2.5-8B')
    lines=['# SVAR隐藏层248→256维对照','',
        '唯一改动：`Linear(D,248)-ReLU-Linear(248,2)`改为`Linear(D,256)-ReLU-Linear(256,2)`。SVAR输入、Adam lr0.001、batch32、max50、validation-loss patience5、无标准化/BN/dropout/调度/加权采样均不变。',
        '固定划分为split seed20260912；联合种子中seed43/44/45分别重划全部4000图为3200/400/400，同时控制训练。每个seed单独计算test AUROC/HALL-AUPR；均值±总体std，不是ensemble。','',
        '| 模型 | 划分 | 248 AUROC/AP | 256 AUROC/AP | 256−248 AUROC/AP |','|---|---|---:|---:|---:|']
    for model in MODELS:
        for regime in REGIMES:
            a,b=lookup[model,regime,248],lookup[model,regime,256]
            cell=lambda r:f"{100*r['AUROC_mean']:.2f}±{100*r['AUROC_std']:.2f}/{100*r['HALL_AUPR_mean']:.2f}±{100*r['HALL_AUPR_std']:.2f}"
            lines.append(f"| {names[model]} | {regime} | {cell(a)} | {cell(b)} | {100*(b['AUROC_mean']-a['AUROC_mean']):+.2f}/{100*(b['HALL_AUPR_mean']-a['HALL_AUPR_mean']):+.2f} |")
    lines += ['','## 结论','',
        '- 固定划分中，InternVL提升最明显：AUROC +0.82、HALL-AUPR +1.36个百分点；Qwen3则下降0.27和1.67个百分点。',
        '- 联合种子中，Qwen2.5的HALL-AUPR提高1.61个百分点，但AUROC只提高0.04；LLaVA、Qwen3和InternVL的AUROC变化都在±0.06个百分点内。',
        '- 四模型平均后，固定划分AUROC约+0.10、HALL-AUPR约−0.12个百分点；联合种子AUROC约−0.02、HALL-AUPR约+0.17个百分点。256维没有跨模型一致收益。','',
        '差值没有用于选模型或调参。248与256在同一regime/seed使用同一数据划分和minibatch顺序，但初始化张量形状变化，所以随机权重序列不可能逐参数配对；这里衡量完整训练配置的宽度替换效果。','']
    (ROOT/'docs/SVAR_HIDDEN256_811_RESULTS.md').write_text('\n'.join(lines))
    atomic_json_save(dict(status='completed',fingerprint=p['fingerprint'],new_heads=len(MODELS)*len(REGIMES)*len(SEEDS)),OUT/'status.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--models',nargs='+',choices=MODELS)
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--summarize',action='store_true')
    args=parser.parse_args();torch.set_num_threads(1)
    if args.summarize:summarize()
    elif args.models:
        for model in args.models:run_model(model,args.device)
    else:parser.error('--models or --summarize required')
