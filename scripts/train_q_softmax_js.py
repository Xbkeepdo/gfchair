#!/usr/bin/env python3
"""JS(softmax(raw signed Q), T): saved FP32 K4 sources, fixed probes, no VLM."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import analyze_ffn_target_consequence as cqb
from scripts import analyze_ffn_endpoint_cosine_js as cosine
from scripts.run_cqb_workflow import canonical_json
from scripts.run_ffn_target_consequence import parent_artifacts, checked_load, EXPERIMENT
from scripts.run_ffn_visual_source_attribution import result_root as v1_root
from features.tc_fvpa_artifacts import atomic_json_save, atomic_torch_save, sha256_file

MODELS=cosine.MODELS
OUT=ROOT/'outputs/ffn_q_softmax_js_20260909'


def q_softmax_js(q,target):
    q=torch.as_tensor(q,dtype=torch.float64)
    target=torch.as_tensor(target,dtype=torch.float64)
    if q.ndim!=2 or q.shape!=target.shape or min(q.shape)==0:
        raise ValueError('Q and T must share nonempty [layer,visual_token] support')
    if not torch.isfinite(q).all() or not torch.isfinite(target).all() or (target<0).any():
        raise ValueError('Nonfinite Q/T or negative target mass')
    sums=target.sum(-1,keepdim=True)
    if (sums<=0).any() or (sums-1).abs().max()>1e-5:
        raise ValueError('Expected saved normalized T, not AE strength')
    p=torch.softmax(q,dim=-1)  # raw signed Q; temperature 1; no clipping or abs.
    t=target/sums
    mid=((p+t)*.5).clamp_min(torch.finfo(torch.float64).tiny)
    js=.5*(torch.special.xlogy(p,p/mid)+torch.special.xlogy(t,t/mid)).sum(-1)
    if not torch.isfinite(js).all() or (js < -1e-12).any() or (js > np.log(2)+1e-12).any():
        raise ValueError('JS must be finite in [0,ln(2)]')
    return js.numpy().astype(np.float32),dict(q_min=float(q.min()),q_max=float(q.max()),
        p_max=p.max(-1).values.numpy(),softmax_zero_entries=int((p==0).sum()),
        source_entries=q.numel(),p_sum_error=float((p.sum(-1)-1).abs().max()),
        target_sum_error=float((sums-1).abs().max()))


def build_groups(ae,strength,kappa,js,cos_js):
    f=cqb.trainer.direct_features(ae,strength)
    def cat(*x):return np.concatenate(x,axis=1).astype(np.float32)
    values=dict(F=f,F_K=cat(f,kappa),J_QT=js,J_cosT=cos_js,
                F_J_QT=cat(f,js),F_K_J_QT=cat(f,kappa,js))
    if any(v.ndim!=2 or len(v)!=len(f) or not np.isfinite(v).all() for v in values.values()):
        raise ValueError('Invalid feature matrix')
    return values


def prepare(model):
    parents,validation_sha=parent_artifacts(model)
    validation_path=ROOT/'outputs/ffn_visual_source_consistency_v2/old_validation_fp32_k4.json'
    exception=cqb.trainer.study.training_numerical_exception(validation_path)
    split_path=ROOT/'outputs'/model/EXPERIMENT/'image_splits.json'
    splits=json.loads(split_path.read_text())
    if len(splits['train'])!=3200 or len(splits['test'])!=800 or set(splits['train'])&set(splits['test']):
        raise ValueError('Expected original disjoint 3200/800 image split')
    if set(parents)!=set(splits['train']+splits['test']):
        raise ValueError('Parent image coverage differs')
    positions,stats,processed,empty={},[],[],[]
    for image,(path,digest) in sorted(parents.items()):
        shard=checked_load(path,digest)
        if not shard['processed_image'] or shard['image_ids']!=[image]:
            raise ValueError('Invalid processed image')
        processed.append(image)
        if not shard['positions']:empty.append(image)
        for row in shard['positions']:
            key=row['target_key']
            if key in positions:raise ValueError('Duplicate target')
            j,audit=q_softmax_js(row['path_signed_q'],row['attention_evidence'])
            jc,_=cosine.endpoint_cosine_js(row)  # Matched FP32 source; not old native-Q control.
            positions[key]=dict(AE=np.asarray(row['AE']),S=np.asarray(row['S']),
                                kappa=np.asarray(row['kappa_vec']),js=j,cos_js=jc)
            stats.append(audit)
    mentions=[]
    for p in sorted((v1_root(model)/'shards/full').glob('features*_shard_*.pt')):
        mentions.extend(torch.load(p,map_location='cpu',weights_only=False,mmap=True)['sample_table'])
    if len({m['mention_id'] for m in mentions})!=len(mentions) or set(positions)!={m['target_key'] for m in mentions}:
        raise ValueError('Original mention/target mismatch')
    train=[m for m in mentions if m['image_id'] in splits['train']]
    test=[m for m in mentions if m['image_id'] in splits['test']]
    if len(train)+len(test)!=len(mentions):raise ValueError('Unassigned mention')
    mentions=train+test
    raw={k:np.stack([positions[m['target_key']][k] for m in mentions]) for k in ('AE','S','kappa','js','cos_js')}
    groups=build_groups(raw['AE'],raw['S'],raw['kappa'],raw['js'],raw['cos_js'])
    labels=np.array([m['label'] for m in mentions],dtype=np.int32)
    if any(set(y.tolist())!={0,1} for y in (labels[:len(train)],labels[len(train):])):
        raise ValueError('Both labels required in each split')
    pmax=np.concatenate([a['p_max'] for a in stats])
    audit=dict(q_min=min(a['q_min'] for a in stats),q_max=max(a['q_max'] for a in stats),
        p_max_quantiles=np.quantile(pmax,[0,.5,.9,.99,1]).tolist(),p_max_over_099_fraction=float((pmax>.99).mean()),
        softmax_zero_entries=sum(a['softmax_zero_entries'] for a in stats),
        source_entries=sum(a['source_entries'] for a in stats),
        p_sum_error=max(a['p_sum_error'] for a in stats),target_sum_error=max(a['target_sum_error'] for a in stats),
        js_min=float(raw['js'].min()),js_max=float(raw['js'].max()),unique_targets=len(positions))
    protocol=dict(model=model,images=4000,train_images=3200,test_images=800,processed_images=processed,no_target_images=empty,
        train_mentions=len(train),test_mentions=len(test),mentions=mentions,source_validation_sha256=validation_sha,
        source_shards={str(p):h for p,h in parents.values()},split_sha256=sha256_file(split_path),numerical_exception=exception,
        definition='J_QT=JS(softmax(raw signed path_signed_q, temperature=1), saved T), all visual tokens, natural log; no top-k, Q normalization or clipping',
        precision='Q/T from saved v2 local-FP32 K4 extraction; softmax/JS statistics FP64; MLP input FP32',
        matched_control='J_cosT=JS(softmax(Q/ffn_path_gross),T) on the SAME v2 rows; not previously reported native-precision cosine experiment',
        scope='Exploratory old COCO4000, no new VLM forward, no bootstrap or independent confirmation',
        seeds=[43,44,45],classifier=cqb.trainer.fixed_mlp(),defaults=vars(cqb.trainer.TorchProbeConfig()),input_audit=audit,
        matrix_sha256={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in groups.items()},
        source_sha256={str(p):sha256_file(p) for p in (Path(__file__).resolve(),Path(cosine.__file__),
            Path(cqb.trainer.__file__),ROOT/'scripts/train_torch_probe_feature_sets.py',
            ROOT/'scripts/train_ffn_consistency_alternative_heads.py')})
    return groups,labels,len(train),protocol


def run_model(model,device):
    root=OUT/model;root.mkdir(parents=True,exist_ok=True)
    with (root/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        groups,y,n,protocol=prepare(model)
        protocol['device']=device
        canonical_json(protocol,root/'protocol.json')
        signature=sha256_file(root/'protocol.json')
        cache=root/'matrices.pt'
        if not cache.exists():atomic_torch_save(dict(groups=groups,y=y,ntrain=n),cache)
        canonical_json(dict(sha256=sha256_file(cache),protocol_sha256=signature),root/'matrices_checksum.json')
        results={}
        for name,x in groups.items():
            data=dict(X_train=x[:n],X_test=x[n:],y_train=y[:n],y_test=y[n:])
            heads=[cqb.trainer.run_head('three_hidden',cqb.trainer.fixed_mlp(),seed,data,
                   root/'heads'/name/f'seed{seed}',signature,device) for seed in (43,44,45)]
            results[name]=cqb.trainer.summarize_group(data,heads)
            print('GROUP_COMPLETE',model,name,results[name]['ensemble_reports']['fixed_0.5']['auc'],flush=True)
        value=dict(status='COMPLETE_Q_SOFTMAX_JS',model=model,protocol_sha256=signature,groups=results,input_audit=protocol['input_audit'])
        canonical_json(value,root/'summary.json')
        curve_file=root/'curve.json'
        if not curve_file.exists():
            curve=cosine.write_label_curve(model,dict(y_train=y[:n],y_test=y[n:]),groups['J_QT'][:n],groups['J_QT'][n:],root,
                    slug='q_softmax_js',ylabel='JS(softmax(Q), T), nats; median / IQR',log_scale=False)
            canonical_json(curve,curve_file)
        print('MODEL_COMPLETE',model,flush=True)


def summarize():
    results={m:json.loads((OUT/m/'summary.json').read_text()) for m in MODELS}
    if any(len(v['groups'])!=6 or v['status']!='COMPLETE_Q_SOFTMAX_JS' for v in results.values()):
        raise ValueError('Missing model/group')
    canonical_json(dict(models=results),OUT/'summary.json')
    lines=['# JS(softmax(Q_m), T)：四模型4000图检测','',
        '原3200/800图划分，seeds43/44/45，固定旧三隐藏层MLP；表格为ensemble AUROC / HALL-AUPR (%)。',
        'J_QT用原始带符号Q、温度1、全视觉token、自然对数JS；J_cosT用相同FP32 K4源的Q/||e_m||作匹配对照。F=全AE+log1p(raw S)，K=κ_vec。无bootstrap、调参或新VLM前向。','',
        '| 特征 | Qwen2.5 | LLaVA | Qwen3 | InternVL |','|---|---:|---:|---:|---:|']
    for name in ('J_QT','J_cosT','F','F_J_QT','F_K','F_K_J_QT'):
        cells=[]
        for m in MODELS:
            r=results[m]['groups'][name]['ensemble_reports']['fixed_0.5']
            cells.append(f'{100*r["auc"]:.3f} / {100*r["hallucination_positive"]["aupr"]:.3f}')
        lines.append('| '+name+' | '+' | '.join(cells)+' |')
    lines+=['','数值审计、softmax饱和率、输入来源及所有逐seed/双阈值指标保存在各模型protocol/summary。保持旧v2数值FAIL及探索性边界，不声明显著性。']
    p=OUT/'summary.md';content='\n'.join(lines)+'\n'
    if p.exists() and p.read_text()!=content:raise ValueError('Changed report on resume')
    if not p.exists():p.write_text(content)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',nargs='+',choices=MODELS,default=list(MODELS))
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--summarize',action='store_true')
    args=p.parse_args();torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.summarize:summarize();return
    for model in args.models:
        try:run_model(model,args.device)
        except BaseException as e:
            atomic_json_save(dict(error=str(e),traceback=traceback.format_exc()),OUT/model/'failures'/f'{time.time_ns()}.json')
            raise


if __name__=='__main__':main()
