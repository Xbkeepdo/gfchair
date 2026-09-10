"""Capture actual FFN outputs and compare their directions with path components."""
import argparse
import fcntl
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from features.ffn_visual_path_attribution import quadrature_rule, streaming_vector_path_statistics
from features.tc_fvpa_artifacts import atomic_json_save, atomic_torch_save, sha256_file, sha256_text
from scripts.run_ffn_target_consequence import capture_image, parent_artifacts, checked_load
from scripts.run_ffn_visual_source_consistency import local_fp32
from scripts.run_ffn_visual_source_attribution import result_root as v1_root, EXPERIMENT
from scripts.run_jffn_p_comparison import _load_inputs, _image_path
from scripts.run_cqb_workflow import canonical_json
from scripts.analyze_ffn_endpoint_cosine_js import MODELS, endpoint_cosine_js
from scripts.train_q_softmax_js import q_softmax_js
from scripts import train_ffn_ae_log1p_search as trainer
from utils.config_utils import load_config, get_extraction_model_cfg
from models import build_model

OUT = ROOT / 'outputs/ffn_output_cosine_20260909'
CONFIG = ROOT / 'configs/model_configs_inslen_official_target.yaml'


def direction_projections(fn, z, writes, output):
    """JVP/VJP duality: sum_k w_k <J_G(z_k)a_m, fixed output direction>."""
    if z.ndim != 2 or writes.ndim != 3 or writes.shape[1:] != z.shape or output.shape != z.shape:
        raise ValueError('Expected z/output [targets,width], writes [visual,targets,width]')
    if any(not torch.isfinite(v).all() for v in (z, writes, output)):
        raise ValueError('Nonfinite path input')
    aggregate = writes.sum(0)
    z0 = z - aggregate
    with torch.no_grad():
        clean, baseline = fn(z), fn(z0)
        endpoint = clean - baseline
        directions = torch.stack([output, endpoint])
        norms = directions.norm(dim=-1)
        directions = torch.where(norms[..., None] > 1e-12,
                                 directions / norms.clamp_min(1e-12)[..., None], 0.)
    integrated = torch.zeros_like(directions)
    rule = quadrature_rule('gauss_legendre', 4, device=z.device, dtype=z.dtype)
    for alpha, weight in zip(rule.nodes, rule.weights):
        point = (z0 + alpha * aggregate).detach().requires_grad_(True)
        with torch.enable_grad():
            value = fn(point)
            for i in range(2):
                grad = torch.autograd.grad(value, point, directions[i], retain_graph=(i == 0))[0]
                integrated[i].add_(grad.detach(), alpha=weight.item())
    projections = torch.einsum('btd,mtd->btm', integrated, writes)
    return projections.detach(), norms.detach(), clean.detach()


def selection(count):
    path = OUT / f'cohort{count}.json'
    splits = [json.loads((ROOT/'outputs'/m/EXPERIMENT/'image_splits.json').read_text()) for m in MODELS]
    first = splits[0]
    assert all(set(s[k]) == set(first[k]) for s in splits for k in ('train', 'test'))
    rank = lambda i: (sha256_text(f'20260909:offn:{i}'), i)
    split = {k: sorted(sorted(first[k], key=rank)[:n]) for k,n in [('train',4*count//5),('test',count//5)]}
    value = dict(count=count, split=split, models=list(MODELS),
                 selection='SHA256(20260909:offn:image_id) within original train/test; no labels or scores',
                 original_split_sha={m:sha256_file(ROOT/'outputs'/m/EXPERIMENT/'image_splits.json') for m in MODELS})
    canonical_json(value,path)
    return value


def relative(actual, expected):
    return float((actual-expected).norm() / expected.norm().clamp_min(1e-12))


def extract_image(wrapper, model, config, generations, image_id, parent, smoke):
    targets = sorted(parent['positions'], key=lambda r:r['response_index'])
    tick = time.perf_counter()
    if not targets:
        return dict(image_id=image_id, positions=[], processed_image=True, elapsed=0., smoke=smoke)
    with Image.open(_image_path(config, image_id)) as src:
        image = src.convert('RGB')
    with torch.no_grad():
        context = capture_image(wrapper,model,image,generations[image_id]['response_token_ids'],targets,
                                config.get('run',{}).get('prompt') or 'Describe this image.')
    nlayer = len(context['layers'])
    rows = [dict(target_key=t['target_key'], image_id=image_id, response_index=t['response_index'],
                 target_token_id=t['target_token_id'], cosines=[], layer_audit=[]) for t in targets]
    for index, (layer, capture, directions) in enumerate(zip(context['layers'],context['captures'],context['directions'])):
        writes = directions['a_tokens'].to(capture['h_mid'].device).float()
        z = capture['h_mid'][0,context['positions']].float()
        output = capture['o_ffn'][0,context['positions']].float()
        expected_write = torch.stack([t['write_mag'][index] for t in targets]).to(z)
        write_error = relative(writes.norm(dim=-1).T, expected_write)
        if write_error > 1e-4:
            raise ValueError(f'WRITE lineage mismatch L{index+1}: {write_error}')
        gross = torch.stack([t['ffn_path_gross'][index] for t in targets]).to(z)
        saved_q = torch.stack([t['path_signed_q'][index] for t in targets]).to(z)
        saved_net = torch.stack([t['net_strength'][index] for t in targets]).to(z)
        direct_audit = None
        with local_fp32(layer) as fn:
            projections, norms, clean = direction_projections(fn,z,writes,output)
            if smoke and index in {0,nlayer//2,nlayer-1}:
                stats = streaming_vector_path_statistics(ffn_map=fn,z=z,writes=writes,
                    integration_points=4,method='gauss_legendre',token_chunk_size=64,save_components=True)
                direct = torch.einsum('mtd,td->tm', stats.components,
                                     output / output.norm(dim=-1).clamp_min(1e-12)[:,None])
                direct_audit = dict(vjp_jvp_relative=relative(projections[0],direct),
                    gross_parent_relative=relative(stats.ffn_path_gross,gross))
                if max(direct_audit.values()) > 5e-4:
                    raise ValueError(f'VJP/JVP or saved gross parity failed: {direct_audit}')
                del stats, direct
        q_error, net_error = relative(projections[1],saved_q), relative(norms[1],saved_net)
        if q_error > 5e-4 or net_error > 5e-4:
            raise ValueError(f'Endpoint control mismatch L{index+1}: Q={q_error} norm={net_error}')
        cos = torch.where(gross > 0, projections[0] / gross.clamp_min(1e-30), 0.)
        if not torch.isfinite(cos).all() or cos.abs().max() > 1.0005:
            raise ValueError(f'Invalid output cosine L{index+1}: {cos.abs().max()}')
        for j,row in enumerate(rows):
            row['cosines'].append(cos[j].clamp(-1,1).cpu())
            row['layer_audit'].append(dict(layer=index+1, write_error=write_error,q_error=q_error,net_error=net_error,
                output_norm=float(norms[0,j]), zero_components=int((gross[j]==0).sum()),
                clipped_cosines=int((cos[j].abs()>1).sum()),direct=direct_audit,
                captured_vs_local_output_relative=relative(output[j],clean[j])))
        del writes,z,output,gross,projections,cos,clean
    for row in rows: row['cosines'] = torch.stack(row['cosines'])
    del context
    gc.collect(); torch.cuda.empty_cache()
    return dict(image_id=image_id,positions=rows,processed_image=True,elapsed=time.perf_counter()-tick,smoke=smoke)


def extract(model, device, count, smoke=False):
    config = load_config(str(CONFIG))
    cohort = selection(count)
    ids = [283] if smoke else sorted(cohort['split']['train']+cohort['split']['test'])
    root = OUT / f'cohort{count}' / model / ('smoke' if smoke else 'extraction')
    root.mkdir(parents=True,exist_ok=True)
    lock = (root/'.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    parents, _ = parent_artifacts(model)
    protocol = dict(model=model, ids=ids, count=count, smoke=smoke, device=device,
        output='Actual captured pre-residual o_ffn at target predictor position; frozen direction along integration',
        path='same v2 local-FP32 K4 path and saved component norms; FP32 VJP duality; old numerical FAIL retained',
        source_files={str(p):sha256_file(p) for p in [Path(__file__),ROOT/'models/dgst_capture.py',
            ROOT/'features/ffn_visual_path_attribution.py',ROOT/'features/visual_ffn_jacobian.py',
            ROOT/'scripts/run_ffn_target_consequence.py',ROOT/'scripts/run_ffn_visual_source_consistency.py']},
        parent_checksums={str(i):parents[i][1] for i in ids},config=config,
        input_hashes={n:sha256_file(ROOT/'outputs'/model/EXPERIMENT/n) for n in ['generations.json','labeling.json']})
    canonical_json(protocol,root/'protocol.json')
    signature = sha256_file(root/'protocol.json')
    pending=[]
    for i in ids:
        p=root/f'image_{i:012d}.pt'
        if p.exists():
            side=json.loads(p.with_suffix('.json').read_text())
            assert side['signature']==signature and sha256_file(p)==side['sha256']
        else: pending.append(i)
    if not pending: print(model,'resume complete',flush=True); return
    if not smoke:
        smoke_dir=root.parent/'smoke'
        smoke_protocol=json.loads((smoke_dir/'protocol.json').read_text())
        if smoke_protocol['source_files']!=protocol['source_files'] or smoke_protocol['config']!=protocol['config']:
            raise ValueError('Smoke implementation or config differs from extraction')
        smoke_file=smoke_dir/'image_000000000283.pt'
        smoke_meta=json.loads(smoke_file.with_suffix('.json').read_text())
        gate=checked_load(smoke_file,smoke_meta['sha256'])
        if not gate['smoke'] or not gate['positions']: raise ValueError('Missing implementation smoke')
    _, generations, _ = _load_inputs(ROOT/'outputs'/model/EXPERIMENT)
    wrapper=build_model(model,get_extraction_model_cfg(config,model),device=device)
    wrapper.model.eval().requires_grad_(False)
    for offset,i in enumerate(pending,1):
        parent=checked_load(*parents[i])
        row=extract_image(wrapper,model,config,generations,i,parent,smoke)
        row.update(signature=signature,parent_sha256=parents[i][1],image_sha256=sha256_file(_image_path(config,i)))
        p=root/f'image_{i:012d}.pt'
        atomic_torch_save(row,p)
        atomic_json_save(dict(signature=signature,sha256=sha256_file(p)),p.with_suffix('.json'))
        print(model,offset,len(pending),'image',i,'seconds',round(row['elapsed'],3),flush=True)
    del wrapper
    gc.collect();torch.cuda.empty_cache()


def train(model,device,count):
    cohort=selection(count); root=OUT/f'cohort{count}'/model
    parents,_=parent_artifacts(model)
    lookup={}; sources={}; diagnostic=[]
    for i in cohort['split']['train']+cohort['split']['test']:
        p=root/'extraction'/f'image_{i:012d}.pt'
        meta=json.loads(p.with_suffix('.json').read_text()); sources[str(p)]=meta['sha256']
        row=checked_load(p,meta['sha256']); parent=checked_load(*parents[i])
        assert row['parent_sha256']==parents[i][1]
        bykey={t['target_key']:t for t in parent['positions']}
        assert {t['target_key'] for t in row['positions']}==set(bykey)
        for t in row['positions']:
            old=bykey[t['target_key']]
            j,audit=q_softmax_js(t['cosines'],old['attention_evidence'])
            jc,_=endpoint_cosine_js(old)
            lookup[t['target_key']]=dict(J_output=j,J_endpoint=jc)
            diagnostic.append(audit['p_max'])
    mentions=[]
    for p in sorted((v1_root(model)/'shards/full').glob('features*_shard_*.pt')):
        mentions.extend(torch.load(p,map_location='cpu',weights_only=False,mmap=True)['sample_table'])
    train_rows=[m for m in mentions if m['image_id'] in cohort['split']['train']]
    test_rows=[m for m in mentions if m['image_id'] in cohort['split']['test']]
    chosen=train_rows+test_rows; n=len(train_rows)
    y=np.array([m['label'] for m in chosen],dtype=np.int32)
    groups={k:np.stack([lookup[m['target_key']][k] for m in chosen]) for k in ['J_output','J_endpoint']}
    protocol=dict(count=count,cohort=cohort,mentions=chosen,sources=sources,device=device,
        classifier=trainer.fixed_mlp(),matrix_sha={k:sha256_text(v.tobytes().hex()) for k,v in groups.items()})
    canonical_json(protocol,root/'training_protocol.json'); signature=sha256_file(root/'training_protocol.json')
    results={}
    for name,x in groups.items():
        data=dict(X_train=x[:n],X_test=x[n:],y_train=y[:n],y_test=y[n:])
        heads=[trainer.run_head('three_hidden',trainer.fixed_mlp(),seed,data,root/'heads'/name/f'seed{seed}',signature,device)
               for seed in (43,44,45)]
        results[name]=trainer.summarize_group(data,heads)
    canonical_json(dict(model=model,groups=results,pmax_quantiles=np.quantile(np.concatenate(diagnostic),[.5,.9,.99,1]).tolist()),root/'summary.json')
    from scripts.analyze_ffn_endpoint_cosine_js import _write_csv
    rows=[]
    for label in (0,1):
        for index in range(groups['J_output'].shape[1]):
            v=groups['J_output'][y==label,index]
            rows.append(dict(layer=index+1,label=label,n=len(v),mean=float(v.mean()),median=float(np.median(v))))
    _write_csv(root/'curves.csv',rows)
    print(model, json.dumps(results),flush=True)


def summarize(count):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import csv
    results={m:json.loads((OUT/f'cohort{count}'/m/'summary.json').read_text()) for m in MODELS}
    lines=['# FFN输出方向 cosine–JS', '',f'共享{count}图；同源FP32 K4归因；实际native O_FFN方向；temperature=1；旧MLP三seed。', '',
           '| 模型 | O_FFN AUROC / HALL-AUPR | endpoint AUROC / HALL-AUPR |','|---|---:|---:|']
    fig,axes=plt.subplots(2,2,figsize=(12,7.5),sharey=True)
    for ax,(m,r) in zip(axes.flat,results.items()):
        cells=[]
        for k in ['J_output','J_endpoint']:
            v=r['groups'][k]['ensemble_reports']['fixed_0.5'];cells.append(f'{v["auc"]:.6f} / {v["hallucination_positive"]["aupr"]:.6f}')
        lines.append('| '+m+' | '+' | '.join(cells)+' |')
        rows=list(csv.DictReader((OUT/f'cohort{count}'/m/'curves.csv').open()))
        for label,name,color in [('0','HALL','#b2182b'),('1','REAL','#2166ac')]:
            subset=[r for r in rows if r['label']==label]
            ax.plot([int(r['layer']) for r in subset],[float(r['mean']) for r in subset],label=name,color=color)
        ax.set(title=m,xlabel='Decoder layer',ylabel='Mean JS (nats)');ax.legend();ax.grid(alpha=.2)
    fig.suptitle('JS(softmax(cos(e_m, O_FFN)), T)');fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(OUT/f'cohort{count}'/f'mean_curves.{ext}',dpi=180)
    atomic_json_save(results,OUT/f'cohort{count}'/'summary.json')
    (OUT/f'cohort{count}'/'summary.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',nargs='+',choices=MODELS,default=list(MODELS))
    p.add_argument('--stage',choices=['smoke','extract','train','pipeline','summarize'],required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--count',type=int,choices=[500,4000],default=500)
    args=p.parse_args();torch.set_num_threads(1)
    if args.stage=='summarize':summarize(args.count)
    else:
        for m in args.models:
            if args.stage in ['smoke','extract','pipeline']:extract(m,args.device,args.count,smoke=args.stage=='smoke')
            if args.stage in ['train','pipeline']:train(m,args.device,args.count)
