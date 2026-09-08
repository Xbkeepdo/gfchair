#!/usr/bin/env python3
"""Versioned dual-net numerical audit; reuse the v1 capture and streaming JVP."""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import fcntl
import json
import math
from pathlib import Path
import platform
import random
import sys
import time
import traceback

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.tc_fvpa_artifacts import atomic_json_save, atomic_torch_save, sha256_file, sha256_text
from features.visual_ffn_jacobian import reconstruct_visual_directions, resolve_decoder_layer_adapter
from models import build_model
from models.dgst_capture import resolve_decoder_layers
from scripts.run_ffn_visual_source_attribution import MODELS, EXPERIMENT, result_root as v1_root
from scripts.run_ffn_visual_source_audit import _path, _compact_path
from scripts.run_jffn_p_comparison import _load_inputs, _image_path
from scripts.run_jffn_second_round_logit_causal import capture_clean, prepare_inputs, target_groups
from scripts.run_tc_fvpa_path_attribution import _case_label
from utils.config_utils import load_config, get_extraction_model_cfg

VERSION = "ffn_visual_source_consistency_v2"
SEED = 20260907
CONDITIONS = ("native_k4", "native_k64", "fp32_k4", "fp32_k64")
LAYER_COUNTS = dict(zip(("qwen2_5_vl_7b", "llava_1_5_7b", "qwen3_vl_8b", "internvl_2_5_8b"), (28, 32, 36, 32)))


def result_root(model):
    return v1_root(model).parent / VERSION


def immutable_json(payload, path):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise ValueError(f"Fingerprint/content mismatch; refusing to overwrite {path}")
    else:
        atomic_json_save(payload, path)
    return payload


def dual_net(component_sum, endpoint, gross):
    """All norms use the actual saved FP32 accumulation, never inferred errors."""
    vec = component_sum.detach().float()
    end = endpoint.detach().float()
    s = float(gross)
    nv, ne = float(vec.norm()), float(end.norm())
    error = float((vec - end).norm())
    finite = bool(torch.isfinite(vec).all() and torch.isfinite(end).all()) and all(
        math.isfinite(x) for x in (s, nv, ne, error)
    )
    if not finite or s < 0:
        raise ValueError("Nonfinite or negative dual-net statistics")
    kv, ke = (nv / s, ne / s) if s > 0 else (0., 0.)
    bound = error / s if s > 0 else None
    # Reverse triangle inequality, and epsilon_vec >= max(0, 1 - 1/kappa_end).
    lower = max(0., 1. - s / ne) if ne > 0 else None
    rel = error / ne if ne > 0 else None
    return dict(S=s, N_end=ne, N_vec=nv, kappa_end=ke, kappa_vec=kv,
                closure_absolute_error=error, closure_relative_error=rel,
                gross_zero=s == 0, endpoint_zero=ne == 0, vector_zero=nv == 0,
                gross_degenerate=s <= 1e-12, endpoint_degenerate=ne <= 1e-12,
                kappa_difference=abs(ke-kv), kappa_difference_upper_bound=bound,
                closure_relative_lower_bound=lower,
                kappa_range_pass=kv <= 1. + 2e-6,
                kappa_difference_bound_pass=bound is None or abs(ke-kv) <= bound + 2e-6,
                closure_lower_bound_pass=rel is None or rel + 2e-6 >= lower,
                finite=True)


@contextmanager
def local_fp32(layer):
    """Only current norm/FFN change; exact original tensor dtypes restored on errors."""
    adapter = resolve_decoder_layer_adapter(layer)
    modules = (adapter.ffn_norm, adapter.ffn)
    tensors = [(x, x.dtype) for module in modules for x in list(module.parameters()) + list(module.buffers())]
    old_matmul, old_cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        for module in modules:
            module.float()
        with torch.autocast(device_type=adapter.output_projection.weight.device.type, enabled=False):
            yield lambda x: adapter.ffn(adapter.ffn_norm(x))
    finally:
        for tensor, dtype in tensors:
            tensor.data = tensor.data.to(dtype=dtype)
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
        torch.backends.cudnn.allow_tf32 = old_cudnn


def measure(function, z, writes, condition, chunk, detailed):
    device = z.device
    if z.is_cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    k = int(condition.rsplit("k", 1)[1])
    stats = _path(function, z, writes, method="gauss_legendre", k=k,
                  chunk=chunk, save_components=detailed)
    if z.is_cuda:
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    row = _compact_path(stats, save_components=detailed)
    row.update(dual_net(stats.component_sum[0], stats.total_finite_effect[0], stats.gross_strength[0]))
    row.update(condition=condition, integration_points=k, path_dtype=str(z.dtype),
               accumulation_dtype=str(stats.component_sum.dtype), elapsed_seconds=elapsed,
               peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if z.is_cuda else 0,
               peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if z.is_cuda else 0)
    if detailed:
        row["component_sum"] = stats.component_sum[0].float().cpu()
    return row


def case_key(row):
    return f"{row['image_id']}:{row['response_index']}:{row['layer']}"


def select_cases(records, seed=SEED, limit=100):
    """Unique target-layer records, deterministic stratification and nearest controls."""
    groups = defaultdict(list)
    for row in records:
        groups[(row['layer'], row['label'])].append(row)
    bad = {key: [r for r in rows if r['old_kappa_end'] > 1] for key, rows in groups.items()}
    selected = {}
    for key in sorted(bad):
        if bad[key]:
            for field in ('old_kappa_end', 'old_closure_relative_error'):
                row = min(bad[key], key=lambda r: (-r[field], r['image_id'], r['response_index']))
                selected[case_key(row)] = row
    if len(selected) > limit:
        raise ValueError("Mandatory stratum extrema exceed the preregistered case limit")
    pools = {}
    for key, rows in bad.items():
        pools[key] = sorted((r for r in rows if case_key(r) not in selected), key=case_key)
        random.Random(f"{seed}:{key}").shuffle(pools[key])
    while len(selected) < limit and any(pools.values()):
        for key in sorted(pools):
            if pools[key] and len(selected) < limit:
                row = pools[key].pop()
                selected[case_key(row)] = row
    controls, used = [], set()
    for row in sorted(selected.values(), key=lambda r: (r['layer'], r['label'], r['image_id'], r['response_index'])):
        candidates = [r for r in groups[(row['layer'], row['label'])]
                      if r['old_kappa_end'] <= 1 and case_key(r) not in used]
        def distance(other):
            return (abs(math.log(max(other['old_S'], 1e-30) / max(row['old_S'], 1e-30))) +
                    abs(math.log(other['visual_token_count'] / row['visual_token_count'])),
                    other['image_id'], other['response_index'])
        if not candidates:
            raise ValueError(f"No unmatched same-layer/label control for {case_key(row)}")
        control = min(candidates, key=distance)
        used.add(case_key(control))
        controls.append(dict(control, groups=['normal'], matched_anomaly=case_key(row), matching_distance=distance(control)[0]))
    return [dict(r, groups=['anomaly']) for r in selected.values()] + controls


def prepare_cases(args):
    root = Path(args.output_root) if args.output_root else result_root(args.model)
    manifest_path = root / 'manifests/audit_cases.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for path, digest in manifest['inputs'].items():
            if sha256_file(path) != digest:
                raise ValueError(f"Changed case-selection source {path}")
        print(f"RESUME frozen manifest {manifest_path}", flush=True)
        return
    model_root = ROOT / 'outputs' / args.model / EXPERIMENT
    labels, generations, splits = _load_inputs(model_root)
    paths = sorted((v1_root(args.model) / 'shards/full').glob('features*_shard_*.pt'))
    records, targets, seen_images, inputs = [], {}, set(), {}
    for path in paths:
        shard = torch.load(path, map_location='cpu', weights_only=False)
        images = set(map(int, shard['image_ids']))
        if seen_images & images:
            raise ValueError('Duplicate v1 image shards')
        seen_images.update(images)
        inputs[str(path)] = sha256_file(path)
        for row in shard['positions']:
            image_id, response_index = int(row['image_id']), int(row['response_index'])
            groups = target_groups(labels[image_id], generations[image_id]['response_token_ids'])
            label, _ = _case_label(groups[response_index])
            target = dict(image_id=image_id, response_index=response_index,
                          target_token_id=int(row['target_token_id']), label=label,
                          visual_token_count=int(row['p_ffn'].shape[-1]))
            targets[row['target_key']] = target
            for index, (s, k, error) in enumerate(zip(row['gross_strength'].tolist(), row['kappa'].tolist(), row['completeness_relative_error'].tolist())):
                records.append(dict(target, layer=index+1, old_S=s, old_kappa_end=k, old_closure_relative_error=error))
        del shard
    if seen_images != set(map(int, splits['train'] + splits['test'])) or len(seen_images) != 4000:
        raise ValueError('Old cohort is not exactly the complete split')
    selected = {case_key(r): r for r in select_cases(records)}
    # Image-balanced all-layer controls are selected before any new measurement.
    all_layer_images = set()
    for label in (0, 1):
        eligible = sorted({r['image_id'] for r in targets.values() if r['label'] == label} - all_layer_images)
        random.Random(f'{SEED}:all_layers:{label}').shuffle(eligible)
        if len(eligible) < 10:
            raise ValueError('Insufficient label-balanced audit images')
        for image_id in eligible[:10]:
            candidates = sorted((r for r in targets.values() if r['image_id'] == image_id and r['label'] == label), key=lambda r: r['response_index'])
            target = random.Random(f'{SEED}:{image_id}').choice(candidates)
            all_layer_images.add(image_id)
            for layer in range(1, LAYER_COUNTS[args.model] + 1):
                row = dict(target, layer=layer)
                key = case_key(row)
                if key in selected:
                    selected[key]['groups'].append('all_layers')
                else:
                    selected[key] = dict(row, groups=['all_layers'])
    # Preserve only a fixed 50-case full-vector subset; compact audit maps remain available for all cases.
    detailed = sorted(selected)
    random.Random(SEED).shuffle(detailed)
    detailed = set(detailed[:50])
    cases = [dict(selected[k], case_id=k, save_components=k in detailed) for k in sorted(selected)]
    for name in ('generations.json', 'labeling.json', 'image_splits.json'):
        path = model_root / name
        inputs[str(path)] = sha256_file(path)
    manifest = dict(version=VERSION, model=args.model, seed=SEED, cases=cases, inputs=inputs,
                    all_layer_image_ids=sorted(all_layer_images), old_processed_images=len(seen_images),
                    unique_old_cases=len(records), anomaly_available=sum(r['old_kappa_end'] > 1 for r in records),
                    conditions=list(CONDITIONS), candidate_order=['native_k4', 'fp32_k4', 'fp32_k64'],
                    full_vectors_cases=len(detailed))
    immutable_json(manifest, manifest_path)
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('cases','inputs')}, indent=2), flush=True)


def audit_old_vectors(args):
    root = Path(args.output_root) if args.output_root else result_root(args.model)
    rows = []
    for path in sorted((v1_root(args.model) / 'shards/audit').glob('*.pt')):
        shard = torch.load(path, map_location='cpu', weights_only=False)
        for case in shard['cases']:
            if not case.get('fp32_audit'):
                continue
            for name, stats in [('native_k64', case['k_sweep']['64']), ('fp32_k64', case['fp32_audit']['gauss_legendre_k64'])]:
                components = stats['components'].float()
                values = dual_net(components.sum(dim=0), stats['total_finite_effect'], stats['gross_strength'])
                rows.append(dict(case_id=case['case_id'], condition=name, **values))
    if len(rows) != 100:
        raise ValueError(f'Expected 50 old cases x 2 conditions, found {len(rows)}')
    immutable_json(dict(model=args.model, scope='old50_not_tail_coverage', rows=rows), root / 'metrics/old50_dual_net.json')
    print(f'{args.model}: old50 dual-net checked', flush=True)


def audit(args):
    root = Path(args.output_root) if args.output_root else result_root(args.model)
    manifest_path = Path(args.cohort_manifest) if args.cohort_manifest else root / 'manifests/audit_cases.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['model'] != args.model or manifest['version'] != VERSION:
        raise ValueError('Wrong model/cohort manifest')
    source_paths = [Path(__file__), ROOT/'features/ffn_visual_path_attribution.py',
                    ROOT/'features/visual_ffn_jacobian.py', ROOT/'scripts/run_ffn_visual_source_audit.py',
                    ROOT/'scripts/run_jffn_second_round_logit_causal.py', Path(args.config)]
    source_paths += sorted((ROOT/'models').glob('*.py'))
    sources = {str(p.resolve()): sha256_file(p) for p in source_paths}
    fingerprint = sha256_text(json.dumps(dict(manifest=sha256_file(manifest_path), sources=sources,
                                               chunk=args.token_chunk_size), sort_keys=True))
    immutable_json(dict(fingerprint=fingerprint, sources=sources, manifest_sha256=sha256_file(manifest_path)), root/'manifests/audit_implementation.json')
    directory = root/'shards/audit'
    directory.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    for row in manifest['cases']:
        grouped[(row['image_id'], row['response_index'])].append(row)
    keys = sorted(grouped)[args.rank::args.world_size]
    if args.smoke:
        all_layers = [key for key in keys if any('all_layers' in r['groups'] for r in grouped[key])]
        keys = all_layers[:1]
    pending = []
    for key in keys:
        todo = []
        for case in grouped[key]:
            path = directory / f"case_{case['case_id'].replace(':','_')}.pt"
            if path.exists():
                if not args.resume:
                    raise FileExistsError(path)
                saved = torch.load(path, map_location='cpu', weights_only=False)
                if saved['fingerprint'] != fingerprint or saved['case'] != case:
                    raise ValueError(f'Wrong resume fingerprint/case {path}')
                if set(saved['conditions']) != set(CONDITIONS):
                    raise ValueError(f'Incomplete existing case {path}')
            else:
                todo.append(case)
        if todo:
            pending.append((key, todo))
    if not pending:
        print(f'RESUME: all {len(keys)} assigned targets complete; no model loaded', flush=True)
        return
    config = load_config(args.config)
    labels, generations, _ = _load_inputs(ROOT/'outputs'/args.model/EXPERIMENT)
    for path, digest in manifest['inputs'].items():
        if not '/shards/full/' in path and sha256_file(path) != digest:
            raise ValueError(f'Changed capture input: {path}')
    started = time.perf_counter()
    environment = dict(command=sys.argv, python=platform.python_version(), torch=torch.__version__,
                       cuda=torch.version.cuda, device=args.device, rank=args.rank, world_size=args.world_size,
                       gpu=torch.cuda.get_device_name(torch.device(args.device)), fingerprint=fingerprint)
    stamp = time.time_ns()
    atomic_json_save(environment, root/f'logs/audit_{stamp}_environment.json')
    wrapper = build_model(args.model, get_extraction_model_cfg(config, args.model), device=args.device)
    wrapper.model.requires_grad_(False).eval()
    decoder = resolve_decoder_layers(wrapper.model)
    prompt = str((config.get('run') or {}).get('prompt') or 'Describe this image.')
    completed = 0
    try:
        for (image_id, response_index), cases in pending:
            ids = [int(v) for v in generations[image_id]['response_token_ids']]
            with Image.open(_image_path(config, image_id)) as source:
                image = source.convert('RGB')
            inputs, position, start, end, grid = prepare_inputs(wrapper, args.model, image, ids[:response_index], prompt)
            captures, clean_logits = capture_clean(wrapper, inputs, position)
            del clean_logits, inputs
            keep = {c['layer']-1 for c in cases}
            for index, capture in enumerate(captures):
                if index not in keep:
                    capture.clear()
            for case in sorted(cases, key=lambda c:c['layer']):
                if ids[response_index] != case['target_token_id'] or end-start != case['visual_token_count']:
                    raise ValueError('Target/grid mismatch from frozen manifest')
                layer = decoder[case['layer']-1]
                capture = captures[case['layer']-1]
                directions = reconstruct_visual_directions(layer=layer, capture=capture,
                    prediction_positions=[position], visual_start=start, visual_end=end)
                writes = directions['a_tokens'][:,0].contiguous()
                z = capture['h_mid'][0,position].to(writes).clone()
                native_dtype = str(z.dtype)
                adapter = resolve_decoder_layer_adapter(layer)
                native = lambda x: adapter.ffn(adapter.ffn_norm(x))
                measurements = {}
                for condition in CONDITIONS[:2]:
                    measurements[condition] = measure(native, z, writes, condition, args.token_chunk_size, case['save_components'])
                with local_fp32(layer) as function:
                    for condition in CONDITIONS[2:]:
                        measurements[condition] = measure(function, z.float(), writes.float(), condition, args.token_chunk_size, case['save_components'])
                payload = dict(version=VERSION, fingerprint=fingerprint, case=case,
                               prefix_excludes_target=True, prefix_length=response_index,
                               capture_dtype=native_dtype, visual_grid=grid, conditions=measurements,
                               attention_reconstruction_relative_error=float(directions['reconstruction_relative_error']),
                               write_component_sum_relative_error=float(directions['component_sum_relative_error']))
                path = directory/f"case_{case['case_id'].replace(':','_')}.pt"
                if path.exists():
                    raise FileExistsError(path)
                atomic_torch_save(payload, path)
                atomic_json_save(dict(sha256=sha256_file(path), fingerprint=fingerprint), path.with_suffix('.sha256.json'))
                completed += 1
                print(json.dumps(dict(case=case['case_id'], completed=completed,
                    errors={k:v['closure_relative_error'] for k,v in measurements.items()},
                    elapsed_seconds=time.perf_counter()-started)), flush=True)
                capture.clear()
                del directions, writes, z, payload, measurements
            del captures
    except Exception:
        atomic_json_save(dict(status='FAIL', traceback=traceback.format_exc(), fingerprint=fingerprint,
                              case=case if 'case' in locals() else None,
                              elapsed_seconds=time.perf_counter()-started), root/f'logs/audit_{stamp}_failure.json')
        raise
    atomic_json_save(dict(status='COMPLETE_ASSIGNED_CASES', completed_now=completed,
                          assigned_targets=len(keys), elapsed_seconds=time.perf_counter()-started,
                          fingerprint=fingerprint), root/f'logs/audit_{stamp}_completion.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare-cases', 'old-vectors', 'audit'])
    parser.add_argument('--model', required=True, choices=MODELS)
    parser.add_argument('--config', default='configs/model_configs_inslen_official_target.yaml')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--cohort-manifest')
    parser.add_argument('--output-root')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world-size', type=int, default=1)
    parser.add_argument('--token-chunk-size', type=int, default=256, choices=[256,128,64,32])
    parser.add_argument('--resume', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if not 0 <= args.rank < args.world_size:
        raise ValueError('Invalid rank/world-size')
    root = Path(args.output_root) if args.output_root else result_root(args.model)
    root.mkdir(parents=True, exist_ok=True)
    # A nonblocking per-worker lock prevents two writers claiming the same shard.
    with (root/f'.{args.stage}_rank{args.rank}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        {'prepare-cases':prepare_cases, 'old-vectors':audit_old_vectors, 'audit':audit}[args.stage](args)


if __name__ == '__main__':
    main()
