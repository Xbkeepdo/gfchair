"""Fixed raw-Q softmax temperature 0.07; preserve the temperature-1 study."""
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
from scripts import train_q_softmax_js as base
from features.tc_fvpa_artifacts import atomic_json_save,atomic_torch_save,sha256_file
from scripts.run_cqb_workflow import canonical_json

OUT=base.OUT/'temperature007'
TEMPERATURE=.07
GROUPS=('J_QT','F_J_QT','F_K_J_QT')


def tempered_js(q,target,temperature=TEMPERATURE):
    if not np.isfinite(temperature) or temperature<=0:
        raise ValueError('Temperature must be finite and positive')
    return base.q_softmax_js(torch.as_tensor(q,dtype=torch.float64)/temperature,target)


def prepare(model,device):
    oldroot=base.OUT/model
    old=json.loads((oldroot/'protocol.json').read_text())
    check=json.loads((oldroot/'matrices_checksum.json').read_text())
    assert sha256_file(oldroot/'protocol.json')==check['protocol_sha256']
    assert sha256_file(oldroot/'matrices.pt')==check['sha256']
    for p,h in old['source_sha256'].items():
        if sha256_file(p)!=h:raise ValueError(f'Changed baseline implementation: {p}')
    cache=torch.load(oldroot/'matrices.pt',map_location='cpu',weights_only=False)
    np.testing.assert_array_equal(cache['y'],[m['label'] for m in old['mentions']])
    for name,x in cache['groups'].items():
        assert hashlib.sha256(x.tobytes()).hexdigest()==old['matrix_sha256'][name]
    split=ROOT/'outputs'/model/base.EXPERIMENT/'image_splits.json'
    assert sha256_file(split)==old['split_sha256']
    validation=ROOT/'outputs/ffn_visual_source_consistency_v2/old_validation_fp32_k4.json'
    assert sha256_file(validation)==old['source_validation_sha256']
    values,ones,sharp,processed={},{},[],[]
    zeros=entries=0
    max_sum_error=0.
    for p,h in old['source_shards'].items():
        shard=base.checked_load(Path(p),h)
        if not shard['processed_image']:raise ValueError('Unprocessed source')
        processed.extend(shard['image_ids'])
        for r in shard['positions']:
            key=r['target_key']
            if key in values:raise ValueError('Duplicate target')
            ones[key],_=base.q_softmax_js(r['path_signed_q'],r['attention_evidence'])
            values[key],a=tempered_js(r['path_signed_q'],r['attention_evidence'])
            sharp.append(a['p_max']);zeros+=a['softmax_zero_entries'];entries+=a['source_entries']
            max_sum_error=max(max_sum_error,a['p_sum_error'])
    assert len(processed)==4000 and set(processed)==set(old['processed_images'])
    assert set(values)=={m['target_key'] for m in old['mentions']}
    one=np.stack([ones[m['target_key']] for m in old['mentions']])
    np.testing.assert_array_equal(one,cache['groups']['J_QT'])
    js=np.stack([values[m['target_key']] for m in old['mentions']])
    g=cache['groups'];groups=dict(J_QT=js,F_J_QT=np.concatenate((g['F'],js),axis=1),
                                  F_K_J_QT=np.concatenate((g['F_K'],js),axis=1))
    pmax=np.concatenate(sharp)
    audit=dict(temperature1_matrix_exact_parity=True,processed_images=4000,unique_targets=len(values),
        pmax_quantiles=np.quantile(pmax,[0,.5,.9,.99,1]).tolist(),saturation_fraction=float((pmax>.99).mean()),
        softmax_zero_entries=zeros,source_entries=entries,source_sum_error=max_sum_error,
        js_min=float(js.min()),js_max=float(js.max()))
    protocol=dict(model=model,device=device,temperature=TEMPERATURE,groups=list(GROUPS),
        definition='JS(softmax(raw signed Q_m/0.07), T); all visual tokens, natural log; no cosine normalization, abs, clipping or top-k',
        baseline_protocol=str(oldroot/'protocol.json'),baseline_protocol_sha256=sha256_file(oldroot/'protocol.json'),
        baseline_summary_sha256=sha256_file(oldroot/'summary.json'),baseline_cache_sha256=check['sha256'],
        counts={k:old[k] for k in ('images','train_images','test_images','train_mentions','test_mentions')},
        classifier=old['classifier'],defaults=old['defaults'],seeds=old['seeds'],numerical_exception=old['numerical_exception'],
        scope='User-specified temperature on the repeatedly explored old 3200/800 split; no tuning, bootstrap or VLM forward',
        matrix_sha256={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in groups.items()},
        sources={str(p):sha256_file(p) for p in (Path(__file__).resolve(),Path(base.__file__))},audit=audit)
    return dict(groups=groups,y=cache['y'],ntrain=cache['ntrain']),protocol


def run(model,device):
    root=OUT/model;root.mkdir(parents=True,exist_ok=True)
    with (root/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        v,protocol=prepare(model,device)
        canonical_json(protocol,root/'protocol.json');signature=sha256_file(root/'protocol.json')
        p=root/'matrices.pt'
        if not p.exists():atomic_torch_save(v,p)
        canonical_json(dict(sha256=sha256_file(p),protocol_sha256=signature),root/'matrices_checksum.json')
        results={}
        for name,x in v['groups'].items():
            n=v['ntrain'];data=dict(X_train=x[:n],X_test=x[n:],y_train=v['y'][:n],y_test=v['y'][n:])
            heads=[base.cqb.trainer.run_head('three_hidden',base.cqb.trainer.fixed_mlp(),seed,data,
                root/'heads'/name/f'seed{seed}',signature,device) for seed in (43,44,45)]
            results[name]=base.cqb.trainer.summarize_group(data,heads)
            print('GROUP_COMPLETE',model,name,results[name]['ensemble_reports']['fixed_0.5']['auc'],flush=True)
        canonical_json(dict(status='COMPLETE',model=model,groups=results,audit=protocol['audit'],
            protocol_sha256=signature),root/'summary.json')
        if not (root/'curve.json').exists():
            j=v['groups']['J_QT'];n=v['ntrain']
            curve=base.cosine.write_label_curve(model,dict(y_train=v['y'][:n],y_test=v['y'][n:]),j[:n],j[n:],root,
                slug='q_softmax_js_tau007',ylabel='JS(softmax(Q / 0.07), T), nats; median / IQR',log_scale=False)
            canonical_json(curve,root/'curve.json')
        print('MODEL_COMPLETE',model,flush=True)


def summarize():
    results={m:json.loads((OUT/m/'summary.json').read_text()) for m in base.MODELS}
    old={m:json.loads((base.OUT/m/'summary.json').read_text()) for m in base.MODELS}
    assert all(len(v['groups'])==3 and v['status']=='COMPLETE' for v in results.values())
    canonical_json(dict(temperature=TEMPERATURE,models=results,baseline_temperature1=old),OUT/'summary.json')
    lines=['# 原始Q-softmax JS：温度1与0.07','',
        '四模型4000图，原3200/800划分和MLP三seed；ensemble AUROC / HALL-AUPR (%)。F=全AE+log1p(S)，K=κ_vec。',
        '仅Q的softmax温度从1改成0.07，全视觉token，T不变；不是cosine或O_FFN方向的温度实验。', '',
        '| 特征 / 温度 | Qwen2.5 | LLaVA | Qwen3 | InternVL |','|---|---:|---:|---:|---:|']
    for name in GROUPS:
        for tau,source in ((1,old),(.07,results)):
            cells=[]
            for m in base.MODELS:
                v=source[m]['groups'][name]['ensemble_reports']['fixed_0.5']
                cells.append(f'{100*v["auc"]:.3f} / {100*v["hallucination_positive"]["aupr"]:.3f}')
            lines.append('| '+name+f' / {tau} | '+' | '.join(cells)+' |')
    lines+=['','用户指定温度的探索性点估计，不做bootstrap，不称最优温度或独立验证。原温度1结果未改写；其他不受温度影响的对照复用旧结果。']
    text='\n'.join(lines)+'\n';p=OUT/'summary.md'
    if p.exists() and p.read_text()!=text:raise ValueError('Changed report')
    if not p.exists():p.write_text(text)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',nargs='+',choices=base.MODELS,default=list(base.MODELS))
    p.add_argument('--device',default='cuda:0');p.add_argument('--summarize',action='store_true')
    args=p.parse_args();torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.summarize:summarize()
    else:
        for model in args.models:
            try:run(model,args.device)
            except BaseException as e:
                atomic_json_save(dict(error=str(e),traceback=traceback.format_exc()),OUT/model/'failures'/f'{time.time_ns()}.json')
                raise
