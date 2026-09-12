"""Paired WRITE/path-effect geometry using the existing shared 500-image cohort."""
import argparse
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
from features.ffn_visual_path_attribution import streaming_vector_path_statistics, quadrature_rule
from features.visual_ffn_jacobian import reconstruct_visual_directions, resolve_decoder_layer_adapter
from models.dgst_capture import resolve_decoder_layers, attention_row_from_capture
from features.tc_fvpa_artifacts import atomic_torch_save
from scripts.run_ffn_source_composition import capture_full, load_wrapper, parent_paths
from scripts.run_ffn_visual_source_consistency import local_fp32
from scripts.run_jffn_p_comparison import _load_inputs, _image_path
from scripts.analyze_ffn_input_geometry import OUT, MODELS, EXPERIMENT, geometry, summarize, plot, write_csv, contrasts


def factored_components(norm, ffn, z, writes, chunk=64, *, directions=None):
    """Same K4 JVP integral, factoring the RMSNorm and gated-MLP derivatives.

    J_RMS(x)a = w*r*a - w*x*r**3*mean(x*a). Integrate the two
    gated-MLP coefficients before projecting sources, plus K rank-one terms.
    This includes both product-rule branches and the unfrozen norm derivative.
    """
    import torch.nn.functional as F
    gate, up, down = ((ffn.gate_proj,ffn.up_proj,ffn.down_proj) if hasattr(ffn,'gate_proj')
                      else (ffn.w1,ffn.w3,ffn.w2))
    rule=quadrature_rule('gauss_legendre',4,device=z.device,dtype=z.dtype)
    aggregate=writes.sum(0)
    points=z[None]-(1-rule.nodes[:,None,None])*aggregate[None]
    r=torch.rsqrt(points.square().mean(-1,keepdim=True)+norm.variance_epsilon)
    n=points*r*norm.weight
    g,u=gate(n),up(n)
    act,deriv=torch.func.jvp(ffn.act_fn,(g,),(torch.ones_like(g),))
    c=deriv*u
    weight=rule.weights[:,None,None]
    left=(weight*r*c).sum(0)
    right=(weight*r*act).sum(0)
    correction=(c*F.linear(points*norm.weight,gate.weight)+act*F.linear(points*norm.weight,up.weight))
    correction=correction*(weight*r.pow(3)/z.shape[-1])
    # The integration path ALWAYS uses original writes, even when probing Q.
    probes=writes if directions is None else directions
    if probes.ndim!=3 or probes.shape[1:]!=z.shape: raise ValueError('Invalid probe directions')
    components=torch.empty_like(probes)
    for start in range(0,len(probes),chunk):
        a=probes[start:start+chunk]
        hidden=left*F.linear(a*norm.weight,gate.weight)+right*F.linear(a*norm.weight,up.weight)
        dots=torch.einsum('ctd,ktd->kct',a,points)
        hidden-=torch.einsum('kct,kti->cti',dots,correction)
        components[start:start+chunk]=F.linear(hidden,down.weight)
    return components


def rotation(writes, effects):
    """No artificial cosine for zero vectors; zero norms are counted separately."""
    denominator = writes.norm(dim=-1)*effects.norm(dim=-1)
    cos = (writes*effects).sum(-1) / denominator
    valid = denominator > 0
    if not torch.isfinite(cos[valid]).all() or (valid.any() and cos[valid].abs().max() > 1.00001):
        raise ValueError('Invalid per-source cosine')
    return cos.T.cpu(), (~valid).sum(0).cpu()


def extract(model, device, smoke):
    cohort = json.loads((ROOT/'outputs/ffn_output_cosine_20260909/cohort500.json').read_text())
    ids = sorted(cohort['split']['train'] + cohort['split']['test'])
    parents = parent_paths(model)
    directory = OUT/'cohort500'/model/('smoke_factored' if smoke else 'extraction')
    directory.mkdir(parents=True, exist_ok=True)
    if smoke: ids = [283]
    protocol = dict(model=model, ids=ids, k=4, precision='local FP32', path='G=FFN(Norm); z0=z-sum a_m',
                    backend='factored RMSNorm plus full gated-MLP JVP; smoke compared to direct JVP',
                    selection=str(ROOT/'outputs/ffn_output_cosine_20260909/cohort500.json'),
                    rotation='cos(a_m, e_m); zero denominator undefined',
                    cancellation='actual norm of component sum / sum component norms')
    protocol_path = directory/'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('Existing extraction protocol differs')
    protocol_path.write_text(json.dumps(protocol, indent=2))
    pending = [i for i in ids if not (directory/f'image_{i:012d}.pt').exists()]
    if not pending: print(model, 'complete', flush=True); return
    if not smoke:
        check = directory.parent/'smoke_factored/image_000000000283.pt'
        if not check.exists(): raise ValueError('Run same-code smoke first')
    _, generations, _ = _load_inputs(ROOT/'outputs'/model/EXPERIMENT)
    wrapper, config = load_wrapper(model, device)
    for offset, image_id in enumerate(pending, 1):
        tick = time.perf_counter()
        parent = torch.load(parents[image_id], map_location='cpu', weights_only=False)
        targets = sorted(parent['positions'], key=lambda r:r['response_index'])
        rows = [dict(target_key=t['target_key'], target_token_id=t['target_token_id'], metrics={}, audit=[], cosines=[])
                for t in targets]
        if targets:
            with Image.open(_image_path(config,image_id)) as src: image=src.convert('RGB')
            captures, queries, start, end, grid = capture_full(wrapper,model,image,
                generations[image_id]['response_token_ids'],targets,
                config.get('run',{}).get('prompt') or 'Describe this image.')
            for li, (layer, cap) in enumerate(zip(resolve_decoder_layers(wrapper.model), captures)):
                for query in queries:
                    future = attention_row_from_capture(cap,query)[...,query+1:]
                    if future.numel() and float(future.abs().max()) > 1e-7: raise ValueError('Noncausal attention')
                direct = reconstruct_visual_directions(layer=layer,capture=cap,prediction_positions=queries,
                                                       visual_start=start,visual_end=end)
                a = direct['a_tokens'].float()
                z = cap['h_mid'][0,queries].float()
                with local_fp32(layer) as fn:
                    adapter=resolve_decoder_layer_adapter(layer)
                    with torch.no_grad():
                        effects=factored_components(adapter.ffn_norm,adapter.ffn,z,a)
                        endpoint=fn(z)-fn(z-a.sum(0))
                    parity=None
                    if smoke:
                        stats = streaming_vector_path_statistics(ffn_map=fn,z=z,writes=a,
                            method='gauss_legendre',integration_points=4,token_chunk_size=64,save_components=True)
                        parity=float((effects-stats.components).norm()/stats.components.norm().clamp_min(1e-12))
                        if parity>5e-5: raise ValueError(f'Factored/direct JVP mismatch L{li+1}: {parity}')
                        del stats
                cos, undefined = rotation(a, effects)
                a_norm = a.norm(dim=-1).T.cpu()
                e_norm = effects.norm(dim=-1).T.cpu()
                total=effects.sum(0)
                net = total.norm(dim=-1).cpu()
                closure=(total-endpoint).norm(dim=-1)/endpoint.norm(dim=-1).clamp_min(1e-12)
                for ti, (row, old) in enumerate(zip(rows, targets)):
                    row['cosines'].append(cos[ti])
                    old_a, old_e = old['write_mag'][li], old['ffn_path_gross'][li]
                    err_a = float((a_norm[ti]-old_a).norm()/old_a.norm().clamp_min(1e-12))
                    err_e = float((e_norm[ti]-old_e).norm()/old_e.norm().clamp_min(1e-12))
                    if max(err_a,err_e) > 5e-4: raise ValueError(f'Parent parity failed {model} {image_id} L{li+1}: {err_a}, {err_e}')
                    values, counts = geometry(a_norm[ti:ti+1], e_norm[ti:ti+1], net[ti:ti+1],
                                              closure[ti:ti+1].cpu())
                    valid = cos[ti][torch.isfinite(cos[ti])].numpy()
                    input_kappa = float(a[:,ti].sum(0).norm()/a_norm[ti].sum())
                    values.update(rotation_mean=[float(valid.mean()) if len(valid) else np.nan],
                                  rotation_median=[float(np.median(valid)) if len(valid) else np.nan],
                                  input_cancellation=[input_kappa],
                                  cancellation_change=[values['cancellation'][0]-input_kappa])
                    for k,value in values.items(): row['metrics'].setdefault(k,[]).append(float(value[0]))
                    row['audit'].append(dict(layer=li+1,write_relative=err_a,effect_relative=err_e,
                        direct_jvp_relative=parity,undefined_cosines=int(undefined[ti]),**counts))
                del effects,a,z,direct,cos
            del captures
        for row in rows: row['cosines'] = torch.stack(row['cosines'])
        payload = dict(image_id=image_id, positions=rows, sample_table=parent['sample_table'],
                       elapsed=time.perf_counter()-tick, complete=True, protocol=protocol)
        atomic_torch_save(payload,directory/f'image_{image_id:012d}.pt')
        print(model, offset, '/', len(pending), 'image', image_id, 'seconds', round(payload['elapsed'],2), flush=True)
    del wrapper
    gc.collect(); torch.cuda.empty_cache()


def report():
    cohort = json.loads((ROOT/'outputs/ffn_output_cosine_20260909/cohort500.json').read_text())
    ids = sorted(cohort['split']['train'] + cohort['split']['test'])
    train = set(cohort['split']['train'])
    all_rows, all_paired = [], []
    for model in MODELS:
        directory = OUT/'cohort500'/model
        arrays, mentions = {}, []
        audit=dict(images=0,targets=0,source_entries=0,undefined_cosines=0,
                   max_write_relative=0.,max_effect_relative=0.,max_abs_cosine=0.)
        for image_id in ids:
            shard = torch.load(directory/'extraction'/f'image_{image_id:012d}.pt',map_location='cpu',weights_only=False)
            assert shard['complete'] and shard['image_id'] == image_id
            bykey = {r['target_key']:r for r in shard['positions']}
            assert len(bykey)==len(shard['positions'])
            audit['images']+=1
            for target in shard['positions']:
                cos=target['cosines'].numpy()
                valid=np.isfinite(cos)
                assert not np.isinf(cos).any()
                assert cos.shape[0]==len(target['metrics']['rotation_mean'])
                np.testing.assert_allclose(np.nanmean(cos,axis=-1),target['metrics']['rotation_mean'],atol=1e-7,rtol=1e-6)
                assert int((~valid).sum())==sum(r['undefined_cosines'] for r in target['audit'])
                audit['targets']+=1
                audit['source_entries']+=int(cos.size)
                audit['undefined_cosines']+=int((~valid).sum())
                audit['max_abs_cosine']=max(audit['max_abs_cosine'],float(np.max(np.abs(cos[valid]),initial=0)))
                for row in target['audit']:
                    audit['max_write_relative']=max(audit['max_write_relative'],row['write_relative'])
                    audit['max_effect_relative']=max(audit['max_effect_relative'],row['effect_relative'])
            labels = {}
            for row in shard['sample_table']: labels.setdefault(row['target_key'],set()).add(row['label'])
            for row in shard['sample_table']:
                mentions.append(dict(row,split='train' if image_id in train else 'test',label_conflict=len(labels[row['target_key']])>1))
                for k,values in bykey[row['target_key']]['metrics'].items(): arrays.setdefault(k,[]).append(values)
        arrays = {k:np.asarray(v) for k,v in arrays.items()}
        keys=('mention_id','target_key','image_id','response_index','target_token_id','label')
        identity=lambda row:tuple(row[k] for k in keys)
        original=json.loads((OUT/model/'mentions.json').read_text())
        expected=[identity(r) for r in original if r['image_id'] in set(ids)]
        assert sorted(map(identity,mentions))==sorted(expected)
        assert len({r['mention_id'] for r in mentions})==len(mentions)
        audit['mentions']=len(mentions)
        assert audit['images']==500 and audit['max_abs_cosine']<=1.00001
        (directory/'audit.json').write_text(json.dumps(audit,indent=2))
        np.savez_compressed(directory/'metrics.npz',**arrays)
        (directory/'mentions.json').write_text(json.dumps(mentions))
        rows,paired = summarize(model,arrays,mentions,directory)
        all_rows.extend(rows); all_paired.extend(paired)
    write_csv(OUT/'cohort500/curves.csv',all_rows)
    write_csv(OUT/'cohort500/paired_images.csv',all_paired)
    write_csv(OUT/'cohort500/summary.csv',contrasts(all_rows,all_paired))
    for scope in ('all','test'):
        plot(all_rows,scope,('amp_mean','rotation_mean','cancellation'),OUT/'cohort500'/f'{scope}_geometry')
    plot(all_rows,'all',('input_cancellation','cancellation','cancellation_change'),
         OUT/'cohort500/all_cancellation_change')


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--models',nargs='+',choices=MODELS,default=MODELS)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--summarize',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1)
    if args.summarize: report()
    else:
        for model in args.models: extract(model,args.device,args.smoke)
