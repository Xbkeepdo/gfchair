"""Six-model probes using raw visual/generated-text ratios, without log transform."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.plot_visual_generation_ratio import ratio,MODELS
from scripts import run_ffn_source_composition as base
from scripts.train_torch_probe_feature_sets import TorchProbeConfig

SOURCE=ROOT/'outputs/prefix_attention_gate/full/group_detection'
OUT=ROOT/'outputs/prefix_attention_gate/full/region_plots/visual_generation_ratio/detection'
SEEDS=(43,44,45)


def build_groups(groups):
    result={}
    for family in ('attention','gated'):
        values=ratio(groups[f'{family}_raw_visual'],groups[f'{family}_raw_generation'])
        if not np.isfinite(values).all():raise ValueError('Undefined ratio: refuse to drop training samples or add epsilon')
        result[family+'_ratio']=values.astype(np.float32)
    if any(not np.isfinite(v).all() for v in result.values()):raise ValueError('FP32 ratio overflow')
    return result


def prepare(model):
    root=OUT/model
    if (root/'matrices.pt').exists():return
    source=base.read(SOURCE/model/'matrices.pt')
    groups=build_groups(source['groups']);y=source['y'];n=source['ntrain'];mentions=source['mentions']
    experiment='COCO4000-JACOBIAN-PATH' if model in MODELS[:2] else base.EXPERIMENT
    split=json.loads((ROOT/'outputs'/model/experiment/'image_splits.json').read_text())
    train,test=set(split['train']),set(split['test'])
    assert len(train)==3200 and len(test)==800 and not train&test
    assert all(m['image_id'] in train for m in mentions[:n]) and all(m['image_id'] in test for m in mentions[n:])
    np.testing.assert_array_equal(y,[m['label'] for m in mentions])
    base.save(dict(groups=groups,y=y,ntrain=n,mentions=mentions),root/'matrices.pt')
    base.json_save(dict(model=model,source=str(SOURCE/model/'matrices.pt'),
        formula='For each mention/layer: visual region mass / generated-text region mass; raw attention and attention*gate separately',
        transform='none: no log/log1p, epsilon, clipping, feature standardization, or extra features',
        zero_denominator='error, no cohort changes',images=4000,train_images=3200,test_images=800,
        train_mentions=n,test_mentions=len(y)-n,input_dims={k:v.shape[1] for k,v in groups.items()},seeds=SEEDS,
        config=asdict(TorchProbeConfig()),gate_reference='full causal prefix MAD',reporting='per-seed metrics then mean and population std; no ensemble headline'),root/'protocol.json')
    print('PREPARED',model,len(y),flush=True)


def summarize():
    rows=[];seeds=[];pairs=[]
    lines=['# Visual / generated-text attention ratio detection','',
           '每个目标每层先求比值，原值直接输入MLP，不取log、不截断、不加epsilon。raw attention与attention×gate分别训练。原COCO4000/3200-800/seeds43-45；均值±总体标准差（%），F1按各seed训练REAL-F1阈值。', '',
           '| Model | Input | AUROC | HALL AUPR | HALL F1 |','|---|---|---:|---:|---:|']
    for model in MODELS:
        p=OUT/model/'detection.json'
        if not p.exists():continue
        results=json.loads(p.read_text())
        if model in MODELS[:2]:
            baseline=ROOT/'outputs'/model/'COCO4000-JACOBIAN-PATH/path/training/results.json'
        else:baseline=base.OUT/model/'detection.json'
        results['F_reference']=json.loads(baseline.read_text())['F']
        means={}
        for name,value in results.items():
            assert set(value['per_seed_metrics'])=={'43','44','45'}
            reports=[value['per_seed_metrics'][str(s)]['threshold_reports']['train_f1']['test_metrics'] for s in SEEDS]
            data={'AUROC':[r['auc'] for r in reports],'HALL_AUPR':[r['hallucination_positive']['aupr'] for r in reports],
                  'HALL_F1':[r['hallucination_positive']['f1'] for r in reports]}
            row=dict(model=model,group=name);cells=[]
            for metric,v in data.items():
                row[metric+'_mean']=float(np.mean(v));row[metric+'_std']=float(np.std(v));cells.append(f'{100*np.mean(v):.3f} ± {100*np.std(v):.3f}')
            means[name]=row;rows.append(row);lines.append('| '+' | '.join([model,name,*cells])+' |')
            for seed in SEEDS:
                for rule,report in value['per_seed_metrics'][str(seed)]['threshold_reports'].items():
                    m=report['test_metrics'];seeds.append(dict(model=model,group=name,seed=seed,rule=rule,threshold=report['threshold'],
                        AUROC=m['auc'],HALL_AUPR=m['hallucination_positive']['aupr'],HALL_F1=m['hallucination_positive']['f1']))
        for name in ('attention_ratio','gated_ratio'):
            pairs.append(dict(model=model,group=name,AUROC_vs_F=means[name]['AUROC_mean']-means['F_reference']['AUROC_mean'],
                              HALL_AUPR_vs_F=means[name]['HALL_AUPR_mean']-means['F_reference']['HALL_AUPR_mean']))
    OUT.mkdir(parents=True,exist_ok=True);base.write_csv(rows,OUT/'detection.csv');base.write_csv(seeds,OUT/'seed_metrics.csv');base.write_csv(pairs,OUT/'comparisons.csv')
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--models',nargs='+',choices=MODELS,default=MODELS);parser.add_argument('--device',default='cuda:0');parser.add_argument('--stage',choices=('prepare','train','summarize'),default='train');args=parser.parse_args();torch.set_num_threads(1)
    if args.stage=='summarize':summarize()
    else:
        for model in args.models:
            prepare(model)
            if args.stage=='train':base.train(model,args.device,root=OUT/model)
