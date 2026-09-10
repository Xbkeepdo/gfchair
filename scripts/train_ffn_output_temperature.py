"""Fixed temperature 0.2 control for the saved O_FFN cosine study."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_ffn_output_cosine as base
from scripts.analyze_ffn_endpoint_cosine_js import _write_csv, MODEL_NAMES
from features.tc_fvpa_artifacts import sha256_file, sha256_text, atomic_json_save
from scripts.run_cqb_workflow import canonical_json

OUT = base.OUT / 'cohort500/temperature02'
NAME = 'J_output_tau02'


def temperature_js(cosines, target, temperature=.2):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('Temperature must be finite and positive')
    return base.q_softmax_js(torch.as_tensor(cosines,dtype=torch.float64)/temperature, target)


def run(model, device):
    oldroot = base.OUT / 'cohort500' / model
    old = json.loads((oldroot/'training_protocol.json').read_text())
    parents, _ = base.parent_artifacts(model)
    root = OUT / model
    root.mkdir(parents=True,exist_ok=True)
    lookup, pmax, source_hashes = {}, {1.:[], .2:[]}, {}
    for i in old['cohort']['split']['train']+old['cohort']['split']['test']:
        path = oldroot/'extraction'/f'image_{i:012d}.pt'
        row = base.checked_load(path,old['sources'][str(path)])
        parent = base.checked_load(*parents[i])
        assert row['parent_sha256']==parents[i][1] and row['processed_image']
        source_hashes[str(path)] = old['sources'][str(path)]
        source_hashes[str(parents[i][0])] = parents[i][1]
        bykey = {t['target_key']:t for t in parent['positions']}
        assert {t['target_key'] for t in row['positions']}==set(bykey)
        for t in row['positions']:
            key=t['target_key']; assert key not in lookup
            target=bykey[key]['attention_evidence']
            one, audit1=temperature_js(t['cosines'],target,1.)
            new, audit2=temperature_js(t['cosines'],target,.2)
            lookup[key]=(one,new)
            pmax[1.].append(audit1['p_max']);pmax[.2].append(audit2['p_max'])
    chosen=old['mentions']; n=sum(m['image_id'] in old['cohort']['split']['train'] for m in chosen)
    assert all(m['image_id'] in old['cohort']['split']['train'] for m in chosen[:n])
    assert all(m['image_id'] in old['cohort']['split']['test'] for m in chosen[n:])
    one=np.stack([lookup[m['target_key']][0] for m in chosen])
    assert sha256_text(one.tobytes().hex())==old['matrix_sha']['J_output']
    x=np.stack([lookup[m['target_key']][1] for m in chosen])
    y=np.array([m['label'] for m in chosen],dtype=np.int32)
    protocol=dict(model=model,temperature=.2,definition='JS(softmax(cos(e_m,actual O_FFN)/0.2),T), all visual tokens',
        parent_protocol_sha=sha256_file(oldroot/'training_protocol.json'), sources=source_hashes,
        mentions=chosen,cohort=old['cohort'],matrix_sha=sha256_text(x.tobytes().hex()),
        classifier=base.trainer.fixed_mlp(),defaults=vars(base.trainer.TorchProbeConfig()),seeds=[43,44,45],device=device,
        implementation={str(p):sha256_file(p) for p in [Path(__file__),ROOT/'scripts/train_q_softmax_js.py',
            ROOT/'scripts/train_ffn_ae_log1p_search.py',ROOT/'scripts/train_torch_probe_feature_sets.py']})
    canonical_json(protocol,root/'protocol.json'); signature=sha256_file(root/'protocol.json')
    data=dict(X_train=x[:n],X_test=x[n:],y_train=y[:n],y_test=y[n:])
    heads=[base.trainer.run_head('three_hidden',base.trainer.fixed_mlp(),seed,data,root/'heads'/f'seed{seed}',signature,device)
           for seed in (43,44,45)]
    result=base.trainer.summarize_group(data,heads)
    oldsummary=json.loads((oldroot/'summary.json').read_text())['groups']['J_output']
    sharpness={}
    for tau,parts in pmax.items():
        values=np.concatenate(parts)
        sharpness[str(tau)]=dict(pmax_quantiles=np.quantile(values,[.5,.9,.99,1]).tolist(),saturation_fraction=float((values>.99).mean()))
    value=dict(model=model,temperature=.2,status='COMPLETE',summary=result,baseline_tau1=oldsummary,
        sharpness=sharpness,train_mentions=n,test_mentions=len(y)-n,test_hall=int((y[n:]==0).sum()),
        baseline_matrix_exact_match=True,js_min=float(x.min()),js_max=float(x.max()),
        replay_errors={str(h['seed']):h['recomputation_max_error'] for h in heads})
    canonical_json(value,root/'summary.json')
    rows=[]
    for label in (0,1):
        for l in range(x.shape[1]):
            v=x[y==label,l]
            rows.append(dict(label=label,layer=l+1,n=len(v),mean=float(v.mean()),median=float(np.median(v))))
    table=root/'curves.csv'
    if not table.exists():_write_csv(table,rows)
    print(model,json.dumps(dict(result=result['ensemble_reports']['fixed_0.5'],sharpness=sharpness)),flush=True)


def summarize():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    results={m:json.loads((OUT/m/'summary.json').read_text()) for m in base.MODELS}
    fig,axes=plt.subplots(2,2,figsize=(12,7.5),sharey=True)
    lines=['# O_FFN cosine–JS：固定温度0.2', '', '共享500图（400/100），同一cosine/标签/MLP与seeds43/44/45，AUROC/AP为三seed概率ensemble。', '',
           '| 模型 | tau=1 AUROC / HALL-AUPR | tau=0.2 AUROC / HALL-AUPR |','|---|---:|---:|']
    for ax,(m,r) in zip(axes.flat,results.items()):
        cells=[]
        for key in ['baseline_tau1','summary']:
            v=r[key]['ensemble_reports']['fixed_0.5'];cells.append(f'{v["auc"]:.6f} / {v["hallucination_positive"]["aupr"]:.6f}')
        lines.append('| '+MODEL_NAMES[m]+' | '+' | '.join(cells)+' |')
        rows=list(csv.DictReader((OUT/m/'curves.csv').open()))
        for label,name,color in [('0','HALL','#b2182b'),('1','REAL','#2166ac')]:
            selected=[v for v in rows if v['label']==label]
            ax.plot([int(v['layer']) for v in selected],[float(v['mean']) for v in selected],label=name,color=color)
        ax.set(title=MODEL_NAMES[m],xlabel='Decoder layer',ylabel='Mean JS (nats)');ax.legend(frameon=False);ax.grid(alpha=.2)
    fig.suptitle('JS(softmax(cos(e_m, O_FFN) / 0.2), T)')
    fig.text(.5,.01,'Means across all train + test mentions | No uncertainty band',ha='center')
    fig.tight_layout(rect=(0,.03,1,.96))
    for ext in ['png','pdf']:fig.savefig(OUT/f'mean_curves.{ext}',dpi=180)
    plt.close(fig)
    lines+=['','固定用户指定温度的一次探索性对照，未做温度搜索、bootstrap或独立cohort确认。']
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n')
    atomic_json_save(results,OUT/'summary.json')
    print('\n'.join(lines))


def self_check():
    c=torch.tensor([[.2,-.4,.8]],dtype=torch.float64);t=torch.tensor([[.2,.3,.5]],dtype=torch.float64)
    p=torch.softmax(c/.2,-1);mid=(p+t)/2
    expected=.5*(torch.special.xlogy(p,p/mid)+torch.special.xlogy(t,t/mid)).sum(-1)
    actual,audit=temperature_js(c,t)
    np.testing.assert_allclose(actual,expected.numpy(),atol=1e-8)
    assert audit['p_max'][0]>temperature_js(c,t,1.)[1]['p_max'][0]
    try:temperature_js(c,t,0)
    except ValueError:pass
    else:raise AssertionError('Zero temperature was accepted')
    print('temperature formula, concentration, validation checks PASS')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',nargs='+',choices=base.MODELS,default=list(base.MODELS))
    p.add_argument('--device',default='cpu');p.add_argument('--summarize',action='store_true');p.add_argument('--self-check',action='store_true')
    args=p.parse_args();torch.set_num_threads(1)
    if args.self_check:self_check()
    elif args.summarize:summarize()
    else:
        for m in args.models:run(m,args.device)
