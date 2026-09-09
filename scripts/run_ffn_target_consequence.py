#!/usr/bin/env python3
"""C-only, three-score extraction; preserve raw per-layer visual-token C_m."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import gc
import json
from pathlib import Path
import platform
import sys
import time
import traceback

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from features.ffn_target_consequence import SCORES, detach_tree, native_suffix, fp32_suffix, fp32_cached_suffix, path_consequences
from features.ffn_visual_path_attribution import fixed_clean_competitor, target_scalar_from_logits
from features.tc_fvpa_artifacts import atomic_json_save, atomic_torch_save, sha256_file, sha256_text
from features.visual_ffn_jacobian import reconstruct_visual_directions
from models import build_model
from models.dgst_capture import resolve_decoder_layers,run_forward_with_dgst_captures,attention_row_from_capture
from scripts.run_ffn_visual_source_attribution import EXPERIMENT, MODELS
from scripts.run_ffn_visual_source_consistency import local_fp32, immutable_json
from scripts.run_jffn_p_comparison import _load_inputs, _image_path
from scripts.run_jffn_second_round_logit_causal import capture_clean, prepare_inputs
from utils.config_utils import get_extraction_model_cfg, load_config

NAME = 'ffn_target_consequence_cqb_v1'


def result_root(model):
    return ROOT / 'outputs' / model / EXPERIMENT / 'results' / NAME


def parent_artifacts(model):
    path = ROOT / 'outputs/ffn_visual_source_consistency_v2/old_validation_fp32_k4.json'
    validation = json.loads(path.read_text())
    rows = validation['models'][model]['artifacts']
    mapping = {int(Path(p).stem.split('_')[-1]): (Path(p), digest) for p, digest in rows.items()}
    if len(mapping) != 4000:
        raise ValueError('Expected complete original COCO4000 source cohort')
    return mapping, sha256_file(path)


def checked_load(path, digest):
    if sha256_file(path) != digest:
        raise ValueError(f'Checksum mismatch: {path}')
    return torch.load(path, map_location='cpu', weights_only=False)


def validate_shard(payload, signature, parent, layer_numbers, smoke=False):
    if payload['fingerprint']!=signature or not payload['processed_image']:
        raise ValueError('C shard fingerprint/completion mismatch')
    targets=sorted(parent['positions'],key=lambda r:r['response_index'])
    if smoke:targets=targets[:1]
    expected={r['target_key']:r for r in targets}
    rows=payload['positions']
    if len(rows)!=len(expected) or {r['target_key'] for r in rows}!=set(expected):
        raise ValueError('Missing/duplicate C target')
    for row in rows:
        old=expected[row['target_key']]
        if row['target_token_id']!=old['target_token_id'] or row['response_index']!=old['response_index']:
            raise ValueError('C target changed')
        if list(row['score_names'])!=list(SCORES) or row['layers']!=layer_numbers:
            raise ValueError('C scalar/layer order changed')
        value=row['C_m']
        if value.dtype!=torch.float32 or value.shape!=(3,len(layer_numbers),old['path_signed_q'].shape[-1]) or not torch.isfinite(value).all():
            raise ValueError('Invalid C map shape/dtype/value')
        if len(row['layer_statistics'])!=len(layer_numbers):
            raise ValueError('Missing C layer statistics')
        for index,stats in enumerate(row['layer_statistics']):
            if stats['layer']!=layer_numbers[index]:raise ValueError('Wrong statistics layer order')
            torch.testing.assert_close(value[:,index].sum(-1),stats['signed_sum'])
    return payload


def causal_decoder_kwargs(hidden, kwargs):
    """Restore a mask elided by the capture-only chunked attention backend.

    Stock eager attention cannot inherit that backend's implicit causality.
    Replaying its None mask unchanged would leak the changed query into prefix
    rows. These four models use an unpadded, batch-one causal prefix here.
    """
    result = detach_tree(kwargs)
    if result.get('attention_mask') is None:
        length = hidden.shape[1]
        result['attention_mask'] = torch.triu(torch.full((1,1,length,length),
            torch.finfo(hidden.dtype).min,device=hidden.device,dtype=hidden.dtype),diagonal=1)
    return result


@contextmanager
def decoder_calls(layers):
    calls = [None] * len(layers)
    handles = []
    for index, layer in enumerate(layers):
        def capture(_module, args, kwargs, index=index):
            if not args:
                raise ValueError('Expected positional hidden states')
            calls[index] = (detach_tree(args[1:]), causal_decoder_kwargs(args[0], kwargs))
        handles.append(layer.register_forward_pre_hook(capture, with_kwargs=True))
    try:
        yield calls
    finally:
        for handle in handles:
            handle.remove()


def capture_image(wrapper, model, image, response_ids, targets, prompt):
    """Match v2's full-caption capture and all-target GEMM shapes, without JVPs."""
    inputs, last_query, start, end, grid = prepare_inputs(wrapper, model, image, response_ids, prompt)
    positions=[last_query-len(response_ids)+int(row['response_index']) for row in targets]
    layers = resolve_decoder_layers(wrapper.model)
    intern=model=='internvl_2_5_8b'
    chunk=(None if intern else wrapper.dgst_attention_query_chunk_size if model=='llava_1_5_7b'
           else int(wrapper.cfg.get('dgst_attention_query_chunk_size',512)))
    with decoder_calls(layers) as calls:
        output,captures=run_forward_with_dgst_captures(wrapper.model,output_hidden_states=False,
            retain_attention_updates=True,attention_query_positions=None if intern else positions,
            capture_device=None,attention_query_chunk_size=chunk,record_model_attentions=intern,**inputs)
    logits=output.logits[0,positions].detach().float()
    del output
    if any(call is None for call in calls):
        raise ValueError('Missing native decoder invocation')
    directions=[]
    for layer,capture in zip(layers,captures):
        for query in positions:
            future=attention_row_from_capture(capture,query)[...,query+1:]
            if future.numel() and float(future.abs().max())>1e-7:
                raise ValueError('Target/future token leaked into causal capture row')
        value=reconstruct_visual_directions(layer=layer,capture=capture,
            prediction_positions=positions,visual_start=start,visual_end=end)
        value['a_tokens']=value['a_tokens'].cpu()
        directions.append(value)
    return dict(positions=positions,start=start,end=end,grid=grid,layers=layers,calls=calls,
                captures=captures,logits=logits,directions=directions,keys=[r['target_key'] for r in targets])


def extract_target(wrapper, args, image, response_ids, parent, layer_numbers, prompt, context):
    target_index, target_id = int(parent['response_index']), int(parent['target_token_id'])
    if response_ids[target_index] != target_id:
        raise ValueError('Parent/target-token mismatch')
    offset=context['keys'].index(parent['target_key'])
    query=context['positions'][offset]
    start,end,grid,layers=(context[k] for k in ('start','end','grid','layers'))
    logits=context['logits'][offset]
    captures=[{k:v[:,:query+1] for k,v in c.items() if k in ('h_prev','h_mid','o_ffn','o_attn') and v is not None}
              for c in context['captures']]
    calls=[]
    for call_args,call_kwargs in context['calls']:
        kw=dict(call_kwargs)
        for key,value in list(kw.items()):
            if key=='attention_mask' and value is not None:kw[key]=value[...,:query+1,:query+1]
            elif key in ('position_ids','cache_position') and value is not None:kw[key]=value[...,:query+1]
            elif key=='position_embeddings' and value is not None:kw[key]=tuple(v[...,:query+1,:] for v in value)
        calls.append((call_args,kw))
    if end - start != parent['path_signed_q'].shape[-1]:
        raise ValueError('C and saved Q visual-token supports differ')
    competitor = fixed_clean_competitor(logits, target_id)
    clean_scores = torch.stack([target_scalar_from_logits(logits, target_token_id=target_id,
                               competitor_token_id=competitor, scalar=s) for s in SCORES])
    rows = []
    for number in layer_numbers:
        tick = time.perf_counter()
        index, layer = number - 1, layers[number - 1]
        directions = context['directions'][index]
        writes = directions['a_tokens'][:, offset].to(captures[index]['h_mid'].device).float().contiguous()
        magnitudes=writes.norm(dim=-1)
        parent_magnitudes=parent['write_mag'][index].to(magnitudes)
        source_error=float((magnitudes-parent_magnitudes).norm()/parent_magnitudes.norm().clamp_min(1e-12))
        if source_error>1e-4:
            raise ValueError(f'C/Q source WRITE mismatch layer {number}: {source_error}')
        z = captures[index]['h_mid'][0, query].float()
        callback = {'fp32':fp32_suffix,'fp32_cached':fp32_cached_suffix,'native':native_suffix}[args.suffix_precision](model=wrapper.model, layers=layers, captures=captures, calls=calls,
            layer_index=index, prediction_position=query, target_id=target_id, competitor_id=competitor)
        with torch.no_grad():
            replay_scores = callback(captures[index]['o_ffn'][0, query])
        tolerance = .5 if captures[index]['h_mid'].dtype == torch.bfloat16 else .1
        if args.suffix_precision == 'native' and float((replay_scores - clean_scores).abs().max()) > tolerance:
            raise ValueError(f'Native clean suffix parity failed at layer {number}')
        # Current G stays FP32 until autograd completes; the suffix never calls it.
        batch = args.node_batch if args.suffix_precision=='fp32_cached' else 1
        chunk, retries = args.token_chunk, []
        while True:
            try:
                with local_fp32(layer) as function:
                    values = path_consequences(ffn_map=function,score_from_ffn=callback,z=z,writes=writes,
                        integration_points=args.k,token_chunk=chunk,node_batch=batch)
                break
            except torch.cuda.OutOfMemoryError:
                retries.append(dict(node_batch=batch,token_chunk=chunk,status='OOM'))
                if batch>1:
                    batch=max(1,batch//2)
                elif chunk>32:
                    chunk=max(32,chunk//2)
                else:
                    raise
            gc.collect()
            torch.cuda.empty_cache()
        parity = None
        if args.check_cached_parity:
            reference_callback = fp32_suffix(model=wrapper.model,layers=layers,captures=captures,calls=calls,
                layer_index=index,prediction_position=query,target_id=target_id,competitor_id=competitor)
            with local_fp32(layer) as function:
                reference = path_consequences(ffn_map=function,score_from_ffn=reference_callback,z=z,writes=writes,
                                             integration_points=args.k,token_chunk=args.token_chunk)
            relative = (values['C_m']-reference['C_m']).norm(dim=-1)/reference['C_m'].norm(dim=-1).clamp_min(1e-8)
            score_error = max(float((values[k]-reference[k]).abs().max()) for k in ('clean_scores','baseline_scores'))
            parity = dict(C_relative_l2=relative.cpu(),score_max_abs_error=score_error)
            if float(relative.max())>5e-3 or score_error>5e-4:
                raise ValueError(f'FP32 cache/full suffix parity failed: {parity}')
            del reference,reference_callback
        row = {k: v.detach().float().cpu() if isinstance(v, torch.Tensor) and v.dtype != torch.bool
               else v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in values.items()}
        row.update(layer=number, path_dtype='torch.float32', suffix_dtype='torch.float32' if args.suffix_precision.startswith('fp32') else str(captures[index]['h_mid'].dtype),
                   suffix_route=args.suffix_precision+'_checkpointed_suffix',
                   node_batch=batch,token_chunk=chunk,oom_retries=retries,
                   native_clean_scores=clean_scores.cpu(), replayed_clean_scores=replay_scores.cpu(),cache_full_parity=parity,
                   clean_suffix_max_abs_error=float((replay_scores-clean_scores).abs().max()),
                   write_reconstruction_error=float(directions['component_sum_relative_error']),
                   parent_write_map_relative_l2=source_error,
                   write_strength=float(writes.norm(dim=-1).sum()),
                   parent_write_strength=float(parent['I'][index]),
                   elapsed_seconds=time.perf_counter()-tick,
                   peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(torch.device(args.device)))
        rows.append(row)
        del writes, values, callback, directions
        print(f'{args.model} target={parent["target_key"]} layer={number} seconds={row["elapsed_seconds"]:.3f} '
              f'closure_abs={row["closure_absolute_error"].tolist()}', flush=True)
    result = dict(target_key=parent['target_key'], image_id=parent['image_id'], response_index=target_index,
                  target_token_id=target_id, competitor_token_id=competitor, visual_grid=grid,
                  visual_positions=torch.arange(start, end), layers=layer_numbers, score_names=SCORES,
                  C_m=torch.stack([r.pop('C_m') for r in rows], dim=1), layer_statistics=rows)
    del captures, calls, logits
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--config', default='configs/model_configs_inslen_official_target.yaml')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world-size', type=int, default=1)
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--token-chunk', type=int, default=256)
    parser.add_argument('--node-batch',type=int,default=4)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--suffix-precision', choices=('native','fp32','fp32_cached'), default='fp32_cached')
    parser.add_argument('--check-cached-parity',action='store_true')
    parser.add_argument('--layers', default='all')
    parser.add_argument('--image-ids', default='')
    args = parser.parse_args()
    if not 0 <= args.rank < args.world_size or args.k < 1:
        raise ValueError('Invalid sharding/quadrature')
    if not args.smoke and (args.layers != 'all' or args.image_ids or args.k != 4):
        raise ValueError('Production requires all original images/layers and K4')
    if not args.smoke and args.suffix_precision!='fp32_cached':
        raise ValueError('Native/full replay are diagnostics; production uses audited FP32 cached suffix')
    if args.check_cached_parity and not args.smoke:
        raise ValueError('Cache/full parity is an audit-only control')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = result_root(args.model) / ('smoke' if args.smoke else 'extraction')
    if args.smoke:
        implementation = sha256_text(sha256_file(Path(__file__))+sha256_file(ROOT/'features/ffn_target_consequence.py'))[:8]
        root = root / f'{args.suffix_precision}_k{args.k}_b{args.node_batch}_{args.layers.replace(",","-")}_{implementation}{"_parity" if args.check_cached_parity else ""}'
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / f'rank{args.rank}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    artifacts, validation_sha = parent_artifacts(args.model)
    ids = sorted(artifacts)
    if args.smoke:
        ids = [int(i) for i in args.image_ids.split(',') if i] or [next(i for i in ids if checked_load(*artifacts[i])['positions'])]
    model_root = ROOT / 'outputs' / args.model / EXPERIMENT
    labels, generations, splits = _load_inputs(model_root)
    if set(splits['train']) & set(splits['test']) or set(map(int, splits['train'] + splits['test'])) != set(artifacts):
        raise ValueError('Original image split/cohort mismatch')
    config = load_config(args.config)
    sources = {str(p): sha256_file(p) for p in (Path(__file__), ROOT/'features/ffn_target_consequence.py',
        ROOT/'features/ffn_visual_path_attribution.py', ROOT/'models/dgst_capture.py',
        ROOT/'features/visual_ffn_jacobian.py',
        ROOT/'scripts/run_jffn_second_round_logit_causal.py', ROOT/'scripts/run_ffn_visual_source_consistency.py')}
    sources.update({str(p):sha256_file(p) for p in (ROOT/'models').glob('*.py')})
    protocol = dict(version=NAME, model=args.model, images=ids, layers=args.layers, k=args.k,
        scores=SCORES, competitor='fixed highest clean non-target logit', path='FP32 G(z0+alpha*A), clean residual skip',
        suffix=args.suffix_precision, suffix_detail='serial VJPs and checkpointed decoder blocks; native capture, not whole-model FP32',
        capture='same full-caption/all-target native capture and write GEMM as saved v2 Q; suffix sees only causal prefix',
        save='C_m:[score,layer,visual_token] float32, no source selection or norm/absolute replacement',
        sources=sources, parent_validation_sha256=validation_sha, parent_artifacts={str(i):d for i,(_,d) in artifacts.items()},
        config_sha256=sha256_file(args.config), token_chunk=args.token_chunk,node_batch=args.node_batch,
        resolved_config_sha256=sha256_text(json.dumps(config,sort_keys=True)),cache_full_parity_requested=args.check_cached_parity,
        input_sha256={n:sha256_file(model_root/n) for n in ('generations.json','labeling.json','image_splits.json')},
        original_numerical_exception='Preserve known v2 K4 FAIL; exploratory study; no new independent confirmation')
    signature = sha256_text(json.dumps(protocol, sort_keys=True))
    immutable_json(dict(protocol, fingerprint=signature), root/'manifest.json')
    run = root/'runs'/str(time.time_ns())
    run.mkdir(parents=True)
    atomic_json_save(dict(command=sys.argv, python=sys.version, torch=torch.__version__, platform=platform.platform()), run/'environment.json')
    pending = []
    for image_id in ids[args.rank::args.world_size]:
        path = root/'shards'/f'image_{image_id:012d}.pt'
        if path.exists():
            sidecar_path=path.with_suffix('.json')
            old=checked_load(*artifacts[image_id])
            from scripts.run_ffn_visual_source_consistency import LAYER_COUNTS
            expected_layers=list(range(1,LAYER_COUNTS[args.model]+1)) if args.layers=='all' else [int(n) for n in args.layers.split(',')]
            complete=validate_shard(torch.load(path,map_location='cpu',weights_only=False),signature,old,expected_layers,args.smoke)
            if complete['parent_sha256']!=artifacts[image_id][1]:raise ValueError('C parent checksum changed')
            if not sidecar_path.exists():
                atomic_json_save(dict(fingerprint=signature,sha256=sha256_file(path),image_id=image_id,
                                      targets=len(complete['positions']),layers=expected_layers,recovered_sidecar=True),sidecar_path)
            sidecar = json.loads(sidecar_path.read_text())
            if sidecar['fingerprint'] != signature or sha256_file(path) != sidecar['sha256']:
                raise ValueError(f'Invalid completed shard: {path}')
        else:
            pending.append(image_id)
    if not pending:
        print('RESUME: all assigned images complete; no model loaded', flush=True)
        return
    tick = time.perf_counter()
    try:
        wrapper = build_model(args.model, get_extraction_model_cfg(config,args.model), device=args.device)
        wrapper.model.eval().requires_grad_(False)
        layers = resolve_decoder_layers(wrapper.model)
        numbers = list(range(1,len(layers)+1)) if args.layers=='all' else [int(n) for n in args.layers.split(',')]
        prompt = str((config.get('run') or {}).get('prompt') or 'Describe this image.')
        for image_id in pending:
            parent = checked_load(*artifacts[image_id])
            all_targets = sorted(parent['positions'], key=lambda r:r['response_index'])
            targets=all_targets
            if args.smoke:
                targets = targets[:1]
            partial_path = root/'partial'/f'image_{image_id:012d}.pt'
            partial = torch.load(partial_path, map_location='cpu', weights_only=False) if partial_path.exists() else dict(fingerprint=signature, positions=[])
            if partial['fingerprint'] != signature:
                raise ValueError('Partial extraction fingerprint changed')
            complete = {r['target_key'] for r in partial['positions']}
            with Image.open(_image_path(config,image_id)) as source:
                image = source.convert('RGB')
            context=capture_image(wrapper,args.model,image,generations[image_id]['response_token_ids'],all_targets,prompt) if any(t['target_key'] not in complete for t in targets) else None
            for target in targets:
                if target['target_key'] in complete:
                    continue
                torch.cuda.reset_peak_memory_stats(torch.device(args.device))
                partial['positions'].append(extract_target(wrapper,args,image,generations[image_id]['response_token_ids'],target,numbers,prompt,context))
                atomic_torch_save(partial,partial_path)
            if {r['target_key'] for r in partial['positions']} != {r['target_key'] for r in targets}:
                raise ValueError('Target mismatch in partial image')
            payload = dict(partial,image_id=image_id,processed_image=True,parent_sha256=artifacts[image_id][1],
                           sample_table=parent['sample_table'],image_sha256=sha256_file(_image_path(config,image_id)))
            validate_shard(payload,signature,parent,numbers,args.smoke)
            path=root/'shards'/f'image_{image_id:012d}.pt'
            if path.exists():
                raise FileExistsError(path)
            atomic_torch_save(payload,path)
            atomic_json_save(dict(fingerprint=signature,sha256=sha256_file(path),image_id=image_id,
                                  targets=len(targets),layers=numbers),path.with_suffix('.json'))
            atomic_json_save(dict(status='RUNNING',last_image=image_id,elapsed_seconds=time.perf_counter()-tick),run/'status.json')
            print('PROCESSED_IMAGE',image_id,len(targets),flush=True)
            del context
            gc.collect();torch.cuda.empty_cache()
        atomic_json_save(dict(status='COMPLETE_ASSIGNED_IMAGES',images=len(ids[args.rank::args.world_size]),
                              elapsed_seconds=time.perf_counter()-tick),run/'status.json')
    except BaseException as error:
        atomic_json_save(dict(status='FAIL',type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),
                              elapsed_seconds=time.perf_counter()-tick),run/'failure.json')
        raise


if __name__ == '__main__':
    main()
