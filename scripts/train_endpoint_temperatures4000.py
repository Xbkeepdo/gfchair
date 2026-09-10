"""Fixed 0.2/0.02 temperature contrasts for original 4000-image endpoint cosine."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.analyze_ffn_endpoint_cosine_js import MODELS, MODEL_NAMES, endpoint_cosine_js, _write_csv
from scripts.analyze_ffn_visual_source_study import _detector_matrices
from scripts.run_ffn_visual_source_attribution import result_root, EXPERIMENT
from scripts.train_q_softmax_js import q_softmax_js
from scripts import train_ffn_ae_log1p_search as trainer
from features.tc_fvpa_artifacts import sha256_file, atomic_torch_save, atomic_json_save
from scripts.run_cqb_workflow import canonical_json

OUT=ROOT/'outputs/endpoint_temperatures4000'
TEMPERATURES=(.2,.02)


def cosine_temperatures(row):
    one,audit=endpoint_cosine_js(row)
    q=torch.as_tensor(row['path_signed_q'],dtype=torch.float64)
    gross=torch.as_tensor(row['ffn_path_gross'],dtype=torch.float64)
    c=torch.where(gross>0,q/gross.clamp_min(torch.finfo(torch.float64).tiny),0.).clamp(-1,1)
    values={};sharp={}
    for tau in (1.,*TEMPERATURES):
        js,stats=q_softmax_js(c/tau,row['attention_evidence'])
        values[str(tau)]=one if tau==1 else js
        sharp[str(tau)]=stats['p_max']
    return values,sharp,audit


def prepare(model,device):
    original=result_root(model)
    data=_detector_matrices(model)
    lookup={};stats={str(t):[] for t in (1.,*TEMPERATURES)};sources={}
    zero=clipped=0
    for path in sorted((original/'shards/full').glob('features*_shard_*.pt')):
        sources[str(path)]=sha256_file(path)
        shard=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        for row in shard['positions']:
            key=row['target_key'];assert key not in lookup
            lookup[key],sharp,audit=cosine_temperatures(row)
            zero+=audit['zero_gross_entries'];clipped+=audit['clipped_cosine_entries']
            for tau,parts in stats.items():parts.append(sharp[tau])
    assert set(lookup)==set(data['train_target_keys'])|set(data['test_target_keys'])
    matrices={s:{str(t):np.stack([lookup[k][str(t)] for k in data[f'{s}_target_keys']])
                 for t in (1.,*TEMPERATURES)} for s in ('train','test')}
    split_path=ROOT/'outputs'/model/EXPERIMENT/'image_splits.json'
    oldhash=hashlib.sha256(split_path.read_bytes())
    for s in ('train','test'):
        oldhash.update(json.dumps(data[f'{s}_target_keys']).encode())
        oldhash.update(data[f'y_{s}'].tobytes())
        oldhash.update(matrices[s]['1.0'].tobytes())
    baseline_path=original/'metrics/endpoint_cosine_js_feature_results.json'
    baseline=json.loads(baseline_path.read_text())
    assert oldhash.hexdigest()==baseline['cohort_feature_sha256'], 'Temperature1 baseline mismatch'
    sharpness={}
    for tau,parts in stats.items():
        pmax=np.concatenate(parts)
        sharpness[tau]=dict(pmax_quantiles=np.quantile(pmax,[.5,.9,.99,1]).tolist(),saturation_fraction=float((pmax>.99).mean()))
    protocol=dict(model=model,device=device,temperatures=list(TEMPERATURES),images=4000,train_images=3200,test_images=800,
        definition='JS(softmax(cos(e_m,G(Z)-G(Z0))/tau),T), original v1 source, all visual tokens',
        sources=sources,split_sha=sha256_file(split_path),baseline_sha=sha256_file(baseline_path),
        baseline_matrix_sha=oldhash.hexdigest(),counts=data['counts'],
        classifier=trainer.fixed_mlp(),defaults=vars(trainer.TorchProbeConfig()),seeds=[43,44,45],
        matrix_sha={s:{t:hashlib.sha256(v.tobytes()).hexdigest() for t,v in values.items()} for s,values in matrices.items()},
        implementation={str(p):sha256_file(p) for p in [Path(__file__),ROOT/'scripts/analyze_ffn_endpoint_cosine_js.py',
            ROOT/'scripts/train_q_softmax_js.py',ROOT/'scripts/train_ffn_ae_log1p_search.py',ROOT/'scripts/train_torch_probe_feature_sets.py']})
    return data,matrices,protocol,baseline['summary'],dict(sharpness=sharpness,zero_components=zero,clipped_cosines=clipped)


def run(model,device):
    data,matrices,protocol,baseline,audit=prepare(model,device)
    root=OUT/model;root.mkdir(parents=True,exist_ok=True)
    canonical_json(protocol,root/'protocol.json');signature=sha256_file(root/'protocol.json')
    cache=root/'matrices.pt'
    if not cache.exists():atomic_torch_save(dict(matrices=matrices,y_train=data['y_train'],y_test=data['y_test']),cache)
    canonical_json(dict(sha256=sha256_file(cache)),root/'matrices_checksum.json')
    results={};replays={}
    for tau in TEMPERATURES:
        key=str(tau)
        inputs=dict(X_train=matrices['train'][key],X_test=matrices['test'][key],y_train=data['y_train'],y_test=data['y_test'])
        heads=[trainer.run_head('three_hidden',trainer.fixed_mlp(),seed,inputs,root/f'tau{key}'/f'seed{seed}',signature,device)
               for seed in (43,44,45)]
        results[key]=trainer.summarize_group(inputs,heads)
        replays[key]={str(h['seed']):h['recomputation_max_error'] for h in heads}
        print(model,'tau',tau,'AUROC',results[key]['ensemble_reports']['fixed_0.5']['auc'],flush=True)
    canonical_json(dict(model=model,status='COMPLETE',groups=results,baseline_tau1=baseline,audit=audit,replays=replays),root/'summary.json')
    rows=[];labels=np.concatenate([data['y_train'],data['y_test']])
    for tau in (1.,*TEMPERATURES):
        x=np.concatenate([matrices[s][str(tau)] for s in ('train','test')])
        for label in (0,1):
            for l in range(x.shape[1]):
                v=x[labels==label,l]
                rows.append(dict(temperature=tau,label=label,layer=l+1,n=len(v),mean=float(v.mean()),median=float(np.median(v))))
    if not (root/'curves.csv').exists():_write_csv(root/'curves.csv',rows)


def summarize():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    results={m:json.loads((OUT/m/'summary.json').read_text()) for m in MODELS}
    lines=['# 原4000图：endpoint cosine–JS温度对照', '',
        '方向d=G(Z)-G(Z0)，v1来源，原3200/800划分，固定旧MLP三seed。指标为ensemble AUROC / HALL-AUPR。', '',
        '| 模型 | tau=1 | tau=0.2 | tau=0.02 |','|---|---:|---:|---:|']
    for m,r in results.items():
        b=r['baseline_tau1'];cells=[f'{b["ensemble_auroc"]:.6f} / {b["ensemble_hall_aupr"]:.6f}']
        for tau in TEMPERATURES:
            v=r['groups'][str(tau)]['ensemble_reports']['fixed_0.5'];cells.append(f'{v["auc"]:.6f} / {v["hallucination_positive"]["aupr"]:.6f}')
        lines.append('| '+MODEL_NAMES[m]+' | '+' | '.join(cells)+' |')
    for tau in TEMPERATURES:
        fig,axes=plt.subplots(2,2,figsize=(12,7.5),sharey=True)
        for ax,m in zip(axes.flat,MODELS):
            rows=list(csv.DictReader((OUT/m/'curves.csv').open()))
            for label,name,color in [('0','HALL','#b2182b'),('1','REAL','#2166ac')]:
                selected=[v for v in rows if v['label']==label and float(v['temperature'])==tau]
                ax.plot([int(v['layer']) for v in selected],[float(v['mean']) for v in selected],label=name,color=color)
            ax.set(title=MODEL_NAMES[m],xlabel='Decoder layer',ylabel='Mean JS (nats)');ax.legend(frameon=False);ax.grid(alpha=.2)
        fig.suptitle(f'JS(softmax(cos(e_m, G(Z)-G(Z0)) / {tau}), T)')
        fig.text(.5,.01,'Means across all 4000-image train + test mentions | No uncertainty band',ha='center')
        fig.tight_layout(rect=(0,.03,1,.96))
        for ext in ['png','pdf']:fig.savefig(OUT/f'mean_curves_tau{tau}.{ext}',dpi=180)
        plt.close(fig)
    lines+=['','用户指定两个温度的探索性对照，旧测试集已反复使用，未进行独立验证集温度选择或bootstrap。']
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n')
    atomic_json_save(results,OUT/'summary.json');print('\n'.join(lines))


def self_check():
    row=dict(ffn_path_gross=torch.tensor([[2.,3.,0.]]),path_signed_q=torch.tensor([[.4,-1.2,0.]]),
             attention_evidence=torch.tensor([[.2,.3,.5]]))
    values,sharp,_=cosine_temperatures(row)
    c=torch.tensor([[.2,-.4,0.]],dtype=torch.float64)
    for tau in TEMPERATURES:
        expected,_=q_softmax_js(c/tau,row['attention_evidence'])
        np.testing.assert_allclose(values[str(tau)],expected,atol=1e-7)
    assert sharp['0.02'][0]>sharp['0.2'][0]>sharp['1.0'][0]
    print('endpoint cosine, zero component, temperatures and concentration checks PASS')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',nargs='+',choices=MODELS,default=list(MODELS));p.add_argument('--device',default='cpu')
    p.add_argument('--summarize',action='store_true');p.add_argument('--self-check',action='store_true')
    args=p.parse_args();torch.set_num_threads(1)
    if args.self_check:self_check()
    elif args.summarize:summarize()
    else:
        for m in args.models:run(m,args.device)
