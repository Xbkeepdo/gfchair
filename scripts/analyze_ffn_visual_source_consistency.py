#!/usr/bin/env python3
"""Numerical gate and fixed, train-only joint-scale feature definitions for v2."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import asdict
import inspect
import json
import os
from pathlib import Path
import sys
import random
import re
import time
import traceback
from unittest.mock import patch

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_ffn_visual_source_consistency import (
    VERSION, MODELS, CONDITIONS, LAYER_COUNTS, EXPERIMENT, result_root, immutable_json,
    dual_net, local_fp32,
)
from scripts.analyze_ffn_visual_source_study import _js, _top_overlap
from features.tc_fvpa_artifacts import atomic_json_save, atomic_torch_save, sha256_file, sha256_text

CANDIDATES = ('native_k4', 'fp32_k4', 'fp32_k64')
THRESHOLDS = dict(closure_p90=1e-3, closure_max=1e-2, spearman_median=.99,
                  js_median=.005, top32_median=.95, S_relative_p90=.01,
                  N_vec_relative_p90=.01, kappa_vec_absolute_p90=.01)
FEATURE_GROUPS = {
    'AE': ('AE',), 'AE+i': ('AE','i'), 'AE+s': ('AE','s'),
    'AE+i+s': ('AE','i','s'), 'AE+n': ('AE','n'),
    'AE+s+n': ('AE','s','n'), 'AE+s+kappa_vec': ('AE','s','kappa_vec'),
    'AE+a': ('AE','a'), 'AE+g': ('AE','g'),
    'U_SN': ('R_cos','AE','s','n'),
    'H_SN': ('R_cos','AE','D_EW','D_WF','D_EF','s','n'),
    'U_SK': ('R_cos','AE','s','kappa_vec'),
    'H_SK': ('R_cos','AE','D_EW','D_WF','D_EF','s','kappa_vec'),
    'raw_AE+S+N_vec': ('AE','S','N_vec'),
    'AE+s+L(N_end)': ('AE','s','n_end'),
    'AE+s+kappa_end': ('AE','s','kappa_end'),
}


def fit_scales(S, I, image_ids, train_images):
    """Rows are mentions, not deduplicated target positions; no test statistics."""
    S, I = np.asarray(S, dtype=np.float64), np.asarray(I, dtype=np.float64)
    mask = np.isin(np.asarray(image_ids), list(train_images))
    if S.shape != I.shape or S.ndim != 2 or len(mask) != len(S) or not mask.any():
        raise ValueError('Invalid train-only scale inputs')
    if not np.isfinite(S).all() or not np.isfinite(I).all() or (S < 0).any() or (I < 0).any():
        raise ValueError('Strengths must be finite and nonnegative')
    eps = np.finfo(np.float32).eps
    return dict(S=np.maximum(np.median(S[mask], axis=0), eps),
                I=np.maximum(np.median(I[mask], axis=0), eps))


def feature_sets(raw, scales):
    blocks = {key: np.asarray(value, dtype=np.float64) for key, value in raw.items()}
    for key in ('S','N_vec','N_end','I'):
        if (blocks[key] < 0).any():
            raise ValueError('Negative strength')
    if any(not np.isfinite(v).all() for v in blocks.values()):
        raise ValueError('Nonfinite feature')
    tau = np.asarray(scales['S'], dtype=np.float64)
    tau_i = np.asarray(scales['I'], dtype=np.float64)
    if not np.isfinite(tau).all() or not np.isfinite(tau_i).all() or (tau <= 0).any() or (tau_i <= 0).any():
        raise ValueError('Invalid frozen scale')
    S, N = blocks['S'], blocks['N_vec']
    blocks.update(s=np.log1p(S/tau), n=np.log1p(N/tau),
                  n_end=np.log1p(blocks['N_end']/tau), i=np.log1p(blocks['I']/tau_i),
                  a=np.log1p((S+N)/2/tau), g=np.log1p(np.sqrt(S)*np.sqrt(N)/tau))
    result = {name: np.concatenate([blocks[key] for key in order], axis=-1).astype(np.float32)
              for name, order in FEATURE_GROUPS.items()}
    if any(not np.isfinite(v).all() for v in result.values()):
        raise ValueError('Nonfinite transformed feature')
    return result


def compare_measurement(value, reference):
    p = np.asarray(value['p_ffn'], dtype=np.float64)
    q = np.asarray(reference['p_ffn'], dtype=np.float64)
    if p.shape != q.shape or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError('Invalid probability comparison')
    constant = np.ptp(p) == 0 or np.ptp(q) == 0
    rho = float(np.array_equal(p,q)) if constant else float(spearmanr(p,q).statistic)
    result = dict(spearman=rho, js=_js(p,q), top32_overlap=_top_overlap(p,q),
                  kappa_vec_absolute=abs(value['kappa_vec']-reference['kappa_vec']))
    for key in ('S','N_vec'):
        denom = reference[key]
        result[key+'_relative'] = abs(value[key]-denom)/denom if denom > 0 else (0. if value[key] == 0 else None)
    return result


def gate_group(rows):
    """Zero endpoint effects are separately audited; never fabricate relative PASS."""
    if not rows:
        return dict(status='NO_CASES', count=0)
    nondegenerate = [r for r in rows if not r['endpoint_degenerate']]
    degenerate = [r for r in rows if r['endpoint_degenerate']]
    summaries = {}
    checks = dict(finite=all(r['finite'] for r in rows),
                  kappa_range=all(r['kappa_range_pass'] for r in rows),
                  kappa_difference_bound=all(r['kappa_difference_bound_pass'] for r in rows),
                  closure_lower_bound=all(r['closure_lower_bound_pass'] for r in rows),
                  zero_effect_audited=all(r['closure_absolute_error'] == 0 for r in degenerate))
    specifications = (
        ('closure_relative_error','closure_p90',.90,True,nondegenerate),
        ('closure_relative_error','closure_max',1.,True,nondegenerate),
        ('spearman','spearman_median',.5,False,rows), ('js','js_median',.5,True,rows),
        ('top32_overlap','top32_median',.5,False,rows),
        ('S_relative','S_relative_p90',.9,True,rows),
        ('N_vec_relative','N_vec_relative_p90',.9,True,rows),
        ('kappa_vec_absolute','kappa_vec_absolute_p90',.9,True,rows))
    for field, name, quantile, upper, selected in specifications:
        values = [r[field] for r in selected]
        if not values and field == 'closure_relative_error':
            summaries[name] = None
            checks[name] = checks['zero_effect_audited']
        elif not values or any(v is None or not np.isfinite(v) for v in values):
            summaries[name], checks[name] = None, False
        else:
            value = float(np.quantile(values, quantile))
            summaries[name] = value
            checks[name] = value <= THRESHOLDS[name] if upper else value >= THRESHOLDS[name]
    return dict(status='PASS' if all(checks.values()) else 'FAIL', count=len(rows),
                nondegenerate_count=len(nondegenerate), zero_effect_count=len(degenerate),
                zero_effect_absolute_errors=[r['closure_absolute_error'] for r in degenerate],
                checks=checks, summaries=summaries)


def numerical_gate(models, partial=False):
    summary = dict(version=VERSION, thresholds=THRESHOLDS, reference='fp32_k64',
                   reference_is_truth=False, candidate_order=list(CANDIDATES), models={},
                   no_bootstrap=True)
    flat = []
    fingerprints = {}
    all_complete = True
    for model in models:
        root = result_root(model)
        manifest_path = root/'manifests/audit_cases.json'
        manifest = json.loads(manifest_path.read_text())
        expected_paths = {root/'shards/audit'/f"case_{case['case_id'].replace(':','_')}.pt" for case in manifest['cases']}
        actual_paths = set((root/'shards/audit').glob('*.pt'))
        if actual_paths-expected_paths or (not partial and actual_paths!=expected_paths):
            raise ValueError(f'{model}: missing or unexpected audit shards')
        impl = json.loads((root/'manifests/audit_implementation.json').read_text())
        fingerprints[model] = impl['fingerprint']
        buckets = defaultdict(list)
        count = 0
        paths = {}
        for case in manifest['cases']:
            path = root/'shards/audit'/f"case_{case['case_id'].replace(':','_')}.pt"
            if not path.exists():
                if partial:
                    continue
                raise FileNotFoundError(path)
            digest = sha256_file(path)
            sidecar = json.loads(path.with_suffix('.sha256.json').read_text())
            if sidecar != dict(sha256=digest, fingerprint=impl['fingerprint']):
                raise ValueError(f'Shard checksum/fingerprint mismatch: {path}')
            data = torch.load(path, map_location='cpu', weights_only=False)
            if data['case'] != case or data['fingerprint'] != impl['fingerprint'] or not data['prefix_excludes_target']:
                raise ValueError(f'Case mismatch: {path}')
            paths[str(path)] = digest
            count += 1
            for condition in CONDITIONS:
                value = data['conditions'][condition]
                row = {key:v for key,v in value.items() if not torch.is_tensor(v)}
                row.update(compare_measurement(value, data['conditions']['fp32_k64']))
                relative_bound = row['kappa_end']*row['closure_relative_error'] if row['closure_relative_error'] is not None and row['S']>0 else None
                row['kappa_difference_relative_upper_bound'] = relative_bound
                if relative_bound is not None and row['kappa_difference'] > relative_bound+2e-6:
                    raise ValueError('Relative-form kappa difference bound violated')
                row.update(model=model, case_id=case['case_id'], layer=case['layer'], label=case['label'], groups='|'.join(case['groups']))
                flat.append(row)
                buckets[(condition,'all')].append(row)
                for group in case['groups']:
                    buckets[(condition,group)].append(row)
                    buckets[(condition,f"{group}/layer{case['layer']}")].append(row)
                    buckets[(condition,f"{group}/label{case['label']}")].append(row)
        all_complete = all_complete and count == len(manifest['cases'])
        conditions = {}
        for condition in CONDITIONS:
            groups = {group:gate_group(rows) for (name,group),rows in buckets.items() if name == condition}
            required = ['all'] + [f'all_layers/layer{layer}' for layer in range(1,LAYER_COUNTS[model]+1)]
            for group in ('anomaly','normal'):
                if any(group in c['groups'] for c in manifest['cases']):
                    required.append(group)
            # Additional per-layer/label summaries are descriptive, not extra post-hoc gates.
            passed = all(groups.get(group,{}).get('status') == 'PASS' for group in required)
            conditions[condition] = dict(pass_thresholds=passed, required_groups=required, groups=groups)
        summary['models'][model] = dict(completed_cases=count, expected_cases=len(manifest['cases']),
                                       conditions=conditions, artifacts=paths)
    chosen = next((c for c in CANDIDATES if all(summary['models'][m]['conditions'][c]['pass_thresholds'] for m in models)), None)
    formal = all_complete and set(models) == set(MODELS) and not partial
    summary.update(status=('PASS_AUDIT' if chosen else 'FAIL_NUMERICAL_AUDIT') if formal else 'PARTIAL_NOT_A_FREEZE',
                   selected_candidate=chosen if formal else None, fingerprints=fingerprints,
                   downstream_authorized=False,
                   next_stage='old_cohort_extraction' if formal and chosen else 'STOP' if formal else 'finish_audit')
    root = ROOT/'outputs'/VERSION
    root.mkdir(parents=True, exist_ok=True)
    if partial:
        atomic_json_save(summary, root/'audit_gate_partial.json')
    else:
        immutable_json(summary, root/'audit_gate.json')
        if flat:
            path = root/'audit_cases.csv'
            if not path.exists():
                with path.open('w', newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
                    writer.writeheader()
                    writer.writerows(flat)
    print(json.dumps({k:v for k,v in summary.items() if k != 'models'}, indent=2), flush=True)
    return summary


def extract_old(args):
    """A scoped call adapter adds dual nets to the unchanged v1 full extraction route."""
    from features.ffn_visual_path_attribution import streaming_vector_path_statistics as original
    from features.jffn_experiment import mentions_for_image
    from models import build_model
    from scripts.run_ffn_visual_source_attribution import load_target_lookup, source_config, compact_position
    from scripts.run_jffn_p_comparison import _load_inputs, _extract_one_image, _image_path
    from utils.config_utils import load_config, get_dgst_t_cfg, get_extraction_model_cfg
    gate_path = ROOT/'outputs'/VERSION/'audit_gate.json'
    gate = json.loads(gate_path.read_text())
    if gate['status'] != 'PASS_AUDIT' or set(gate['models']) != set(MODELS):
        raise ValueError('All four models must pass the numerical audit before extraction')
    independent = args.stage == 'extract-new'
    frozen = checked_freeze() if independent else None
    candidate = frozen['candidate'] if independent else args.candidate or gate['selected_candidate']
    if candidate not in CANDIDATES or CANDIDATES.index(candidate) < CANDIDATES.index(gate['selected_candidate']):
        raise ValueError('Not an allowed numerical candidate')
    if not independent and candidate != gate['selected_candidate']:
        previous = CANDIDATES[CANDIDATES.index(candidate)-1]
        validation = ROOT/'outputs'/VERSION/f'old_validation_{previous}.json'
        if json.loads(validation.read_text())['status'] != 'FAIL_NUMERICAL_OLD_COHORT':
            raise ValueError('Upgrade requires the recorded failure of the preceding candidate')
    model_root = result_root(args.model)/'independent/data' if independent else ROOT/'outputs'/args.model/EXPERIMENT
    if independent:
        labels = {int(k):v for k,v in json.loads((model_root/'labeling.json').read_text()).items()}
        generations = {int(k):v for k,v in json.loads((model_root/'generations.json').read_text()).items()}
        ids = sorted(r['image_id'] for r in json.loads((ROOT/'outputs'/VERSION/'independent_cohort.json').read_text())['images'])
    else:
        labels, generations, splits = _load_inputs(model_root)
        ids = sorted(map(int, splits['train']+splits['test']))
    expected_count = 2000 if independent else 4000
    if len(ids) != expected_count or len(set(ids)) != expected_count or set(ids)-set(labels) or set(ids)-set(generations):
        raise ValueError('Incomplete old cohort')
    root = result_root(args.model)/('independent' if independent else 'old')/candidate
    config = load_config(args.config)
    input_names = ('features.pkl','generations.json','labeling.json') + (() if independent else ('image_splits.json',))
    inputs = {str(model_root/name):sha256_file(model_root/name) for name in input_names}
    dependency_paths = sorted((ROOT/'features').glob('*.py')) + sorted((ROOT/'models').glob('*.py'))
    dependency_paths += [ROOT/'scripts/run_ffn_visual_source_attribution.py', ROOT/'scripts/run_jffn_p_comparison.py']
    implementation = {fn.__name__:sha256_text(inspect.getsource(fn)) for fn in (extract_old,dual_net,local_fp32,compare_measurement,_js,_top_overlap)}
    implementation.update({str(p):sha256_file(p) for p in dependency_paths})
    provenance = dict(candidate=candidate, gate_sha256=sha256_file(gate_path), inputs=inputs,
                      config=config, chunk=args.token_chunk_size, version=VERSION,
                      full_k64_reference=bool(args.full_reference and not independent),implementation=implementation)
    signature = sha256_text(json.dumps(provenance, sort_keys=True))
    immutable_json(dict(provenance, fingerprint=signature, image_ids=ids),root/'manifest.json')
    worker_ids = ids[args.rank::args.world_size]
    if args.smoke:
        worker_ids = worker_ids[:1]
    pending = []
    for image_id in worker_ids:
        path = root/'shards'/f'image_{image_id:012d}.pt'
        if path.exists():
            saved = torch.load(path,map_location='cpu',weights_only=False)
            sidecar = json.loads(path.with_suffix('.sha256.json').read_text())
            if saved['fingerprint'] != signature or sidecar['sha256'] != sha256_file(path):
                raise ValueError(f'Wrong full-extraction resume fingerprint/checksum: {path}')
        else:
            pending.append(image_id)
    if not pending:
        print('RESUME: all assigned images complete; no model loaded',flush=True)
        return
    lookup, lookup_audit = load_target_lookup(model_root,set(pending))
    wrapper = build_model(args.model,get_extraction_model_cfg(config,args.model),device=args.device)
    wrapper.model.requires_grad_(False).eval()
    cfg = source_config(get_dgst_t_cfg(config),model=args.model,k=int(candidate.rsplit('k',1)[1]),
                        chunk=args.token_chunk_size,backend='vmap_jvp')
    started, stamp = time.perf_counter(), time.time_ns()
    dual_layers, retries = [], []
    def adapted_statistics(**kwargs):
        # Only the existing vector-path closure has this contract. No scalar/JFFN patching.
        layer = inspect.getclosurevars(kwargs['ffn_map']).nonlocals.get('layer')
        if layer is None:
            raise ValueError('Expected the v1 vector-path FFN layer closure')
        tick = time.perf_counter()
        reference = None
        phase = candidate
        try:
            if candidate.startswith('fp32'):
                with local_fp32(layer) as function:
                    parameters = dict(kwargs,ffn_map=function,z=kwargs['z'].float(),writes=kwargs['writes'].float())
                    result = original(**parameters)
                    if provenance['full_k64_reference']:
                        phase = 'fp32_k64_reference'
                        reference = result if candidate=='fp32_k64' else original(**dict(parameters,integration_points=64))
            else:
                result = original(**kwargs)
                if provenance['full_k64_reference']:
                    phase = 'fp32_k64_reference'
                    with local_fp32(layer) as function:
                        reference = original(**dict(kwargs,ffn_map=function,z=kwargs['z'].float(),writes=kwargs['writes'].float(),integration_points=64))
        except torch.cuda.OutOfMemoryError:
            retries.append(dict(status='OOM',phase=phase,chunk=kwargs['token_chunk_size']))
            raise
        values=[]
        for i in range(len(result.gross_strength)):
            row=dict(dual_net(result.component_sum[i],result.total_finite_effect[i],result.gross_strength[i]),
                     path_dtype='torch.float32' if candidate.startswith('fp32') else str(kwargs['z'].dtype),
                     capture_dtype=str(kwargs['z'].dtype), integration_points=kwargs['integration_points'],
                     token_chunk_size=result.token_chunk_size, elapsed_seconds=time.perf_counter()-tick)
            if reference is not None:
                ref=dual_net(reference.component_sum[i],reference.total_finite_effect[i],reference.gross_strength[i])
                row.update(compare_measurement(dict(row,p_ffn=result.p_ffn[i].cpu()),dict(ref,p_ffn=reference.p_ffn[i].cpu())))
                row['reference_fp32_k64']=ref
                row['reference_p_ffn']=reference.p_ffn[i].cpu()
            values.append(row)
        dual_layers.append(values)
        return result
    try:
        for offset,image_id in enumerate(pending,1):
            dual_layers.clear()
            retries.clear()
            tick = time.perf_counter()
            image_sha256 = sha256_file(_image_path(config,image_id))
            torch.cuda.reset_peak_memory_stats(torch.device(args.device))
            ids_response = generations[image_id].get('response_token_ids') or []
            mentions, indices, _ = mentions_for_image(image_id=image_id,labeling_row=labels[image_id],response_token_ids=ids_response)
            image_cfg = dict(cfg)
            image_cfg['jffn_vector_path_target_distributions'] = np.stack([lookup[(image_id,i)]['target'] for i in indices]) if indices else np.empty((0,0,0),dtype=np.float32)
            image_cfg['jffn_vector_path_evidence_strengths'] = np.stack([lookup[(image_id,i)]['strength'] for i in indices]) if indices else np.empty((0,0),dtype=np.float32)
            with patch('features.ffn_visual_path_attribution.streaming_vector_path_statistics', adapted_statistics):
                positions,returned_mentions = _extract_one_image(wrapper=wrapper,config=config,image_id=image_id,
                    label_row=labels[image_id],generation_row=generations[image_id],dgst_cfg=image_cfg,compact=False)
            if returned_mentions != mentions or (indices and len(dual_layers) != LAYER_COUNTS[args.model]):
                raise ValueError('Changed mention/layer alignment in the shared full extraction')
            compact = []
            for offset_target,position in enumerate(positions):
                row = compact_position(position,lookup)
                row['dual_net'] = [values[offset_target] for values in dual_layers]
                for name in ('S','N_end','N_vec','kappa_end','kappa_vec'):
                    row[name] = torch.tensor([v[name] for v in row['dual_net']],dtype=torch.float32)
                row['I'] = row['write_mag'].sum(-1)
                row['AE'],row['R_cos'] = row['ae_strength'],row['r_cos']
                for name in ('D_EW','D_WF','D_EF'):
                    row[name] = row['ot'][name]
                compact.append(row)
            path = root/'shards'/f'image_{image_id:012d}.pt'
            if path.exists():
                raise FileExistsError(path)
            atomic_torch_save(dict(version=VERSION,fingerprint=signature,image_ids=[image_id],positions=compact,
                                    sample_table=returned_mentions,processed_image=True,
                                    image_sha256=image_sha256,
                                    elapsed_seconds=time.perf_counter()-tick,retries=list(retries),
                                    peak_allocated_bytes=torch.cuda.max_memory_allocated(torch.device(args.device))),path)
            atomic_json_save(dict(sha256=sha256_file(path),fingerprint=signature),path.with_suffix('.sha256.json'))
            print(json.dumps(dict(model=args.model,image_id=image_id,completed_now=offset,pending=len(pending),
                                   elapsed_seconds=time.perf_counter()-started)),flush=True)
    except Exception:
        atomic_json_save(dict(status='FAIL',image_id=image_id,traceback=traceback.format_exc(),retries=list(retries)),root/f'logs/failure_{stamp}.json')
        raise
    atomic_json_save(dict(status='COMPLETE_ASSIGNED_IMAGES',elapsed_seconds=time.perf_counter()-started,
                          count=len(pending),command=sys.argv,lookup=lookup_audit),root/f'logs/complete_{stamp}.json')


def check_extracted_image(row, manifest, image_id, model, expected, indices):
    """Shared old/new acceptance: saved scalars must agree with the actual dual-net record."""
    from features.jffn_experiment import assert_finite_position_rows
    if not row['processed_image'] or row['image_ids'] != [image_id] or row['fingerprint'] != manifest['fingerprint']:
        raise ValueError('Wrong processed-image provenance')
    if row['sample_table'] != expected or [p['response_index'] for p in row['positions']] != indices:
        raise ValueError('Target/mention alignment mismatch')
    assert_finite_position_rows(row['positions'])
    for position in row['positions']:
        if len(position['dual_net']) != LAYER_COUNTS[model]:
            raise ValueError('Incomplete decoder layers')
        for name in ('AE','I','S','N_vec','N_end','kappa_vec','kappa_end','R_cos','D_EW','D_WF','D_EF'):
            value = np.asarray(position[name])
            if value.shape != (LAYER_COUNTS[model],) or not np.isfinite(value).all():
                raise ValueError(f'Invalid compact trajectory: {name}')
        for name in ('S','N_end','N_vec','kappa_end','kappa_vec'):
            saved = np.asarray([v[name] for v in position['dual_net']],dtype=np.float32)
            if not np.array_equal(np.asarray(position[name]),saved):
                raise ValueError(f'Dual-net scalar mismatch: {name}')
        for values in position['dual_net']:
            if values['S']>0 and values['closure_relative_error'] is not None:
                bound=values['kappa_end']*values['closure_relative_error']
                if values['kappa_difference']>bound+2e-6:
                    raise ValueError('Relative-form kappa difference bound violated')


def subset_directory():
    return ROOT/'outputs'/VERSION/'numerical_subset_20260907'


def prepare_subset():
    """Freeze available Qwen images and the first 500 scheduled images for the other models."""
    path=subset_directory()/'selection.json'
    if path.exists():
        return json.loads(path.read_text())
    models={}
    for model in MODELS:
        root=result_root(model)/'old/fp32_k4'
        manifest=json.loads((root/'manifest.json').read_text())
        files=sorted((root/'shards').glob('image_*.pt'))
        if any(not p.with_suffix('.sha256.json').exists() for p in files):
            raise ValueError('Finish/review orphan shards before freezing the subset')
        completed=[int(p.stem.split('_')[1]) for p in files]
        if completed!=manifest['image_ids'][:len(completed)]:
            raise ValueError('Expected the existing image-ID-ordered extraction prefix')
        ids=completed if model.startswith('qwen') else manifest['image_ids'][:500]
        if not ids or len(completed)>len(ids):
            raise ValueError('Invalid numerical subset or more than 500 non-Qwen images already computed')
        snapshots={}
        for p in files:
            digest=sha256_file(p)
            if json.loads(p.with_suffix('.sha256.json').read_text())!=dict(sha256=digest,fingerprint=manifest['fingerprint']):
                raise ValueError('Existing subset checksum/fingerprint mismatch')
            snapshots[str(p)]=dict(sha256=digest,mtime_ns=p.stat().st_mtime_ns)
        models[model]=dict(image_ids=ids,extraction_root=str(root),existing_images=len(completed),
                           manifest_sha256=sha256_file(root/'manifest.json'),existing_artifacts=snapshots,
                           selection='completed_at_user_stop' if model.startswith('qwen') else 'first_500_original_image_id_order')
    value=dict(status='FROZEN_BY_USER_SCOPE_AMENDMENT',models=models,candidate='fp32_k4',
               frozen_unix_ns=time.time_ns(),selection_uses_results=False,
               future_full_cohort_reference=False,training_image_split=[3200,800],training_on_subset=False,
               stopped_processes=dict(pipeline=560825,qwen2=556303,qwen3=556313),
               note='Only numerical K64 validation is reduced. Keep old 4000-image K4 extraction and original training split; no subset detector training.')
    immutable_json(value,path)
    print({m:len(v['image_ids']) for m,v in models.items()},flush=True)
    return value


class SubsetBoundaryReached(BaseException):
    """An intentional image-boundary stop, not a failed case or skipped target."""


def extract_subset(args):
    import fcntl
    from scripts.run_jffn_p_comparison import _extract_one_image as original
    selection=json.loads((subset_directory()/'selection.json').read_text())['models'][args.model]
    root=Path(selection['extraction_root'])
    if sha256_file(root/'manifest.json')!=selection['manifest_sha256']:
        raise ValueError('Numerical subset extraction manifest changed')
    manifest=json.loads((root/'manifest.json').read_text())
    ids=selection['image_ids']
    if ids!=manifest['image_ids'][:len(ids)] or not 0<=args.rank<args.world_size:
        raise ValueError('Bounded extraction requires the immutable image prefix and a valid rank')
    assigned=ids[args.rank::args.world_size]
    existing=list((root/'shards').glob('image_*.pt'))
    if any(int(p.stem.split('_')[1]) not in set(ids) for p in existing):
        raise ValueError('Unexpected extraction beyond the fixed subset boundary')
    complete=all((root/'shards'/f'image_{i:012d}.pt').exists() for i in assigned)
    def bounded(**kwargs):
        if kwargs['image_id'] in ids and kwargs['image_id'] not in allowed:
            raise ValueError('Extraction attempted an image assigned to another rank')
        if kwargs['image_id'] not in allowed:
            raise SubsetBoundaryReached()
        return original(**kwargs)
    allowed=set(assigned)
    if not complete:
        # The numerical extraction implementation and its old fingerprint stay unchanged.
        with (result_root(args.model)/f'.extract-old_rank{args.rank}.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with patch('scripts.run_jffn_p_comparison._extract_one_image',bounded):
                try:
                    extract_old(argparse.Namespace(**dict(vars(args),stage='extract-old',candidate='fp32_k4',full_reference=True,smoke=False)))
                except SubsetBoundaryReached:
                    pass
    for image_id in assigned:
        path=root/'shards'/f'image_{image_id:012d}.pt'
        if json.loads(path.with_suffix('.sha256.json').read_text())!=dict(sha256=sha256_file(path),fingerprint=manifest['fingerprint']):
            raise ValueError('Incomplete numerical subset image/checksum')
    status=dict(status='COMPLETE_REQUESTED_SUBSET_EXTRACTION',images=len(assigned),
                selection_sha256=sha256_file(subset_directory()/'selection.json'))
    suffix=''
    if args.world_size>1:
        status.update(rank=args.rank,world_size=args.world_size,total_subset_images=len(ids))
        suffix=f'_rank{args.rank}_of{args.world_size}'
    immutable_json(status,subset_directory()/f'{args.model}_extraction{suffix}.json')
    print(args.model,'subset extraction complete:',len(assigned),'assigned images; no image beyond the boundary computed',flush=True)


def validate_subset(model):
    from scripts.run_jffn_p_comparison import _load_inputs,_image_path
    from features.jffn_experiment import mentions_for_image
    selection_path=subset_directory()/'selection.json'
    selection=json.loads(selection_path.read_text())['models'][model]
    root=Path(selection['extraction_root'])
    if sha256_file(root/'manifest.json')!=selection['manifest_sha256']:
        raise ValueError('Changed subset source manifest')
    manifest=json.loads((root/'manifest.json').read_text())
    labels,generations,splits=_load_inputs(ROOT/'outputs'/model/EXPERIMENT)
    buckets=defaultdict(list)
    artifacts={}
    targets=mentions=0
    no_targets=[]
    for image_id in selection['image_ids']:
        path=root/'shards'/f'image_{image_id:012d}.pt'
        digest=sha256_file(path)
        if json.loads(path.with_suffix('.sha256.json').read_text())!=dict(sha256=digest,fingerprint=manifest['fingerprint']):
            raise ValueError('Subset shard checksum/fingerprint mismatch')
        previous=selection['existing_artifacts'].get(str(path))
        if previous and (previous['sha256']!=digest or previous['mtime_ns']!=path.stat().st_mtime_ns):
            raise ValueError('Pre-existing completed extraction was changed')
        data=torch.load(path,map_location='cpu',weights_only=False)
        if data['image_sha256']!=sha256_file(_image_path(manifest['config'],image_id)):
            raise ValueError('Subset image checksum mismatch')
        expected,indices,_=mentions_for_image(image_id=image_id,labeling_row=labels[image_id],response_token_ids=generations[image_id]['response_token_ids'])
        check_extracted_image(data,manifest,image_id,model,expected,indices)
        targets+=len(indices); mentions+=len(expected)
        if not indices:
            no_targets.append(image_id)
        for position in data['positions']:
            for layer,value in enumerate(position['dual_net'],1):
                row={k:v for k,v in value.items() if not torch.is_tensor(v)}
                row.update(image_id=image_id,target_key=position['target_key'],layer=layer)
                buckets['all'].append(row); buckets[f'layer{layer}'].append(row)
        artifacts[str(path)]=digest
    groups={key:gate_group(rows) for key,rows in buckets.items()}
    passed=len(groups)==LAYER_COUNTS[model]+1 and all(v['status']=='PASS' for v in groups.values())
    worst=sorted(buckets['all'],key=lambda r:r['closure_relative_error'] if r['closure_relative_error'] is not None else -1,reverse=True)[:20]
    value=dict(status='PASS_K4_NUMERICAL_SUBSET' if passed else 'FAIL_K4_NUMERICAL_SUBSET',model=model,
               processed_images=len(artifacts),unique_targets=targets,mentions=mentions,groups=groups,
               no_target_images=no_targets,worst_closure_cases=worst,artifacts=artifacts,
               selection_sha256=sha256_file(selection_path),full_4000_validation=False,training_authorized=False)
    immutable_json(value,subset_directory()/f'{model}_validation.json')
    print(model,value['status'],len(artifacts),'images',groups['all']['summaries'],flush=True)
    return value


def subset_gate():
    reports={m:json.loads((subset_directory()/f'{m}_validation.json').read_text()) for m in MODELS}
    selection_path=subset_directory()/'selection.json'
    selection=json.loads(selection_path.read_text())
    for model,value in reports.items():
        if value['model']!=model or value['selection_sha256']!=sha256_file(selection_path) or value['processed_images']!=len(selection['models'][model]['image_ids']):
            raise ValueError('Subset acceptance report belongs to another selection/model')
    passed=all(v['status']=='PASS_K4_NUMERICAL_SUBSET' for v in reports.values())
    value=dict(status='PASS_K4_NUMERICAL_SUBSET' if passed else 'FAIL_K4_NUMERICAL_SUBSET',models=reports,
               next_stage='full_4000_K4_only' if passed else 'STOP_REVIEW_NUMERICAL_FAILURE',
               candidate='fp32_k4',full_reference=False,training_on_subset=False,training_image_split=[3200,800])
    immutable_json(value,subset_directory()/'gate.json')
    print(value['status'],flush=True)
    return value


def seed_production():
    """Reuse already-computed K4 features with explicit parent provenance, never relabel the source files."""
    selection=json.loads((subset_directory()/'selection.json').read_text())
    gate=json.loads((subset_directory()/'gate.json').read_text())
    if gate['status']!='PASS_K4_NUMERICAL_SUBSET':
        raise ValueError('Subset numerical gate must precede production')
    for model,chosen in selection['models'].items():
        source=Path(chosen['extraction_root'])
        target=result_root(model)/'old/fp32_k4'
        if source==target:
            raise ValueError('Production must use its own result namespace')
        previous=json.loads((source/'manifest.json').read_text())
        if previous['implementation']['extract_old']!=sha256_text(inspect.getsource(extract_old)):
            raise ValueError('K4 extraction implementation changed; refuse unverified feature reuse')
        provenance={k:v for k,v in previous.items() if k not in ('fingerprint','image_ids')}
        if not provenance['full_k64_reference'] or provenance['candidate']!='fp32_k4':
            raise ValueError('K4 reuse requires the original K4 plus K64-reference source')
        provenance['full_k64_reference']=False
        fingerprint=sha256_text(json.dumps(provenance,sort_keys=True))
        immutable_json(dict(provenance,fingerprint=fingerprint,image_ids=previous['image_ids']),target/'manifest.json')
        links={}
        for image_id in chosen['image_ids']:
            src=source/'shards'/f'image_{image_id:012d}.pt'
            dest=target/'shards'/src.name
            digest=sha256_file(src)
            if digest!=gate['models'][model]['artifacts'][str(src)]:
                raise ValueError('K4 reuse source differs from the numerically accepted artifact')
            if json.loads(src.with_suffix('.sha256.json').read_text())!=dict(sha256=digest,fingerprint=previous['fingerprint']):
                raise ValueError('Source checksum changed before K4 reuse')
            ancestry=dict(path=str(src),sha256=digest,fingerprint=previous['fingerprint'],
                           original_included_k64_reference=True,feature_values_unchanged=True)
            if not dest.exists():
                row=torch.load(src,map_location='cpu',weights_only=False)
                row.update(fingerprint=fingerprint,reused_from=ancestry)
                atomic_torch_save(row,dest)
                immutable_json(dict(sha256=sha256_file(dest),fingerprint=fingerprint),dest.with_suffix('.sha256.json'))
            else:
                row=torch.load(dest,map_location='cpu',weights_only=False)
                if row['fingerprint']!=fingerprint or row['reused_from']!=ancestry or sha256_file(dest)!=json.loads(dest.with_suffix('.sha256.json').read_text())['sha256']:
                    raise ValueError('Wrong K4 reuse resume provenance')
            links[str(dest)]=ancestry
        immutable_json(dict(status='REUSED_K4_FEATURES_WITH_PARENT_PROVENANCE',artifacts=links,
                             subset_gate_sha256=sha256_file(subset_directory()/'gate.json')),target/'reuse_manifest.json')
        print(model,'reused',len(links),'existing images; originals unchanged',flush=True)


def validate_old(candidate):
    from scripts.run_jffn_p_comparison import _load_inputs, _image_path
    from features.jffn_experiment import mentions_for_image
    reports = {}
    for model in MODELS:
        root = result_root(model)/'old'/candidate
        manifest = json.loads((root/'manifest.json').read_text())
        expected_paths={root/'shards'/f'image_{i:012d}.pt' for i in manifest['image_ids']}
        if set((root/'shards').glob('*.pt'))!=expected_paths:
            raise ValueError(f'{model}: missing or unexpected full-cohort shards')
        labels,generations,splits = _load_inputs(ROOT/'outputs'/model/EXPERIMENT)
        if len(splits['train'])!=3200 or len(splits['test'])!=800 or set(splits['train'])&set(splits['test']):
            raise ValueError('Changed fixed 3200/800 image split')
        if set(manifest['image_ids'])!=set(splits['train']+splits['test']):
            raise ValueError('Full extraction cohort differs from the old image split')
        layers = defaultdict(list)
        count_targets = count_mentions = 0
        no_target_images = []
        paths = {}
        for image_id in manifest['image_ids']:
            path = root/'shards'/f'image_{image_id:012d}.pt'
            row = torch.load(path,map_location='cpu',weights_only=False)
            sidecar = json.loads(path.with_suffix('.sha256.json').read_text())
            digest = sha256_file(path)
            if row['fingerprint'] != manifest['fingerprint'] or sidecar['sha256'] != digest or row['image_ids'] != [image_id]:
                raise ValueError(f'Full shard mismatch: {path}')
            if row['image_sha256']!=sha256_file(_image_path(manifest['config'],image_id)):
                raise ValueError(f'Full extraction image bytes changed: {image_id}')
            expected,indices,_ = mentions_for_image(image_id=image_id,labeling_row=labels[image_id],response_token_ids=generations[image_id].get('response_token_ids') or [])
            check_extracted_image(row,manifest,image_id,model,expected,indices)
            if not indices:
                no_target_images.append(image_id)
            for position in row['positions']:
                if len(position['dual_net']) != LAYER_COUNTS[model]:
                    raise ValueError('Incomplete layers')
                for index,values in enumerate(position['dual_net']):
                    layers[index+1].append(values)
            count_targets += len(row['positions'])
            count_mentions += len(expected)
            paths[str(path)] = digest
        layer_reports = {}
        for layer,values in layers.items():
            errors = [r['closure_relative_error'] for r in values if not r['endpoint_degenerate']]
            finite = all(r['finite'] and r['kappa_range_pass'] and r['kappa_difference_bound_pass'] and r['closure_lower_bound_pass'] for r in values)
            zeros_ok = all(r['closure_absolute_error']==0 for r in values if r['endpoint_degenerate'])
            p90,maximum = (float(np.quantile(errors,.9)),max(errors)) if errors else (None,None)
            reference_gate = gate_group(values) if manifest['full_k64_reference'] else None
            passed = finite and zeros_ok and (p90 is None or p90<=1e-3 and maximum<=1e-2)
            if reference_gate:
                passed = passed and reference_gate['status']=='PASS'
            layer_reports[layer] = dict(count=len(values),closure_p90=p90,closure_max=maximum,
                                         passed=passed,reference_gate=reference_gate)
        reports[model] = dict(processed_images=len(paths),unique_targets=count_targets,mentions=count_mentions,
                             processed_image_ids=manifest['image_ids'],no_target_image_ids=no_target_images,
                             extraction_manifest_sha256=sha256_file(root/'manifest.json'),
                             full_k64_reference=manifest['full_k64_reference'],
                             layers=layer_reports,artifacts=paths,
                             passed=len(paths)==4000 and len(layers)==LAYER_COUNTS[model] and all(v['passed'] for v in layer_reports.values()))
    passed = all(r['passed'] for r in reports.values())
    payload = dict(status='PASS_NUMERICAL_OLD_COHORT' if passed else 'FAIL_NUMERICAL_OLD_COHORT',
                   candidate=candidate,models=reports,training_authorized=passed,
                   subset_gate_sha256=sha256_file(subset_directory()/'gate.json') if (subset_directory()/'gate.json').exists() else None,
                   next_candidate=CANDIDATES[CANDIDATES.index(candidate)+1] if not passed and candidate!=CANDIDATES[-1] else None)
    immutable_json(payload,ROOT/'outputs'/VERSION/f'old_validation_{candidate}.json')
    print(payload['status'],candidate,flush=True)
    return payload


EXPLORATORY_TRAINING = 'exploratory_training_20260908'


def training_numerical_exception(validation_path, *, create=False):
    """Explicit user exception for the already diagnosed K4 failure, never a PASS."""
    validation_path = Path(validation_path)
    validation = json.loads(validation_path.read_text())
    models = validation['models']
    failed = [(m, str(layer)) for m, value in models.items()
              for layer, row in value['layers'].items() if not row['passed']]
    if (validation['status'] != 'FAIL_NUMERICAL_OLD_COHORT' or validation['candidate'] != 'fp32_k4'
            or set(models) != set(MODELS) or any(v['processed_images'] != 4000 for v in models.values())
            or failed != [('qwen2_5_vl_7b', '23')]
            or models['qwen2_5_vl_7b']['layers']['23']['closure_max'] != 0.010615984949452763):
        raise ValueError('Training exception does not cover this numerical failure/cohort')
    record = dict(status='USER_AUTHORIZED_EXPLORATORY_TRAINING_WITH_NUMERICAL_FAILURE',
                  user_instruction='就先这样吧，先训练把。另外你再解释一下闭合误差是什么意思',
                  numerical_validation_sha256=sha256_file(validation_path),
                  numerical_validation_status=validation['status'], candidate='fp32_k4',
                  known_cases=['248069:17:23', '546325:30:23'],
                  feature_policy='original K4 unchanged; no K64 replacement or case removal',
                  scope='old-cohort training, checkpoint verification and exploratory metrics only',
                  independent_confirmation_authorized=False, original_gate_rewritten=False)
    path = ROOT/'outputs'/VERSION/EXPLORATORY_TRAINING/'numerical_exception.json'
    if create:
        immutable_json(record, path)
    if json.loads(path.read_text()) != record:
        raise ValueError('Changed numerical exception fingerprint or authorization')
    return dict(path=str(path), sha256=sha256_file(path), **record)


def load_training_data(model, candidate, numerical_exception=False):
    from scripts.run_ffn_visual_source_attribution import result_root as v1_root
    validation_path = ROOT/'outputs'/VERSION/f'old_validation_{candidate}.json'
    validation = json.loads(validation_path.read_text())
    exception = training_numerical_exception(validation_path) if numerical_exception else None
    if validation['status'] != 'PASS_NUMERICAL_OLD_COHORT' and exception is None:
        raise ValueError('Numerical full-cohort acceptance must precede training')
    root = result_root(model)/'old'/candidate
    fields = ('AE','I','S','N_vec','N_end','kappa_vec','kappa_end','R_cos','D_EW','D_WF','D_EF')
    positions = {}
    for name,digest in validation['models'][model]['artifacts'].items():
        if sha256_file(name) != digest:
            raise ValueError(f'Changed accepted extraction: {name}')
        shard = torch.load(name,map_location='cpu',weights_only=False)
        for row in shard['positions']:
            if row['target_key'] in positions:
                raise ValueError('Duplicate v2 target')
            positions[row['target_key']] = {key:np.asarray(row[key],dtype=np.float32) for key in fields}
    # Preserve the legacy mention order as well as its image split and duplicate mentions.
    mentions = []
    for path in sorted((v1_root(model)/'shards/full').glob('features*_shard_*.pt')):
        mentions.extend(torch.load(path,map_location='cpu',weights_only=False)['sample_table'])
    if len({r['mention_id'] for r in mentions}) != len(mentions) or set(positions) != {r['target_key'] for r in mentions}:
        raise ValueError('Old/v2 mention cohort mismatch')
    splits = json.loads((ROOT/'outputs'/model/EXPERIMENT/'image_splits.json').read_text())
    raw = {key:np.stack([positions[r['target_key']][key] for r in mentions]) for key in fields}
    image_ids = np.array([r['image_id'] for r in mentions],dtype=np.int64)
    train = np.isin(image_ids,splits['train'])
    test = np.isin(image_ids,splits['test'])
    if np.any(train & test) or not np.all(train | test):
        raise ValueError('Split leakage or missing mentions')
    scales = fit_scales(raw['S'],raw['I'],image_ids,splits['train'])
    matrices = feature_sets(raw,scales)
    labels = np.array([r['label'] for r in mentions],dtype=np.int32)
    return dict(X_train={k:v[train] for k,v in matrices.items()},X_test={k:v[test] for k,v in matrices.items()},
                y_train=labels[train],y_test=labels[test],scales=scales,raw=raw,
                train_mentions=[r for r,m in zip(mentions,train) if m],
                test_mentions=[r for r,m in zip(mentions,test) if m],validation_sha256=sha256_file(validation_path),
                numerical_exception=exception)


def checkpoint_probabilities(path, matrix):
    from scripts.train_torch_probe_feature_sets import DGSTStyleProbe
    model = DGSTStyleProbe(matrix.shape[1],(128,64,32),.3).eval()
    model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
    with torch.no_grad():
        return np.concatenate([torch.sigmoid(model(torch.from_numpy(part))).numpy().reshape(-1)
                               for part in np.array_split(matrix,max(1,(len(matrix)+255)//256))])


def probability_metrics(labels, probabilities, threshold):
    from scripts.train_torch_probe_feature_sets import _metrics_from_probs
    return _metrics_from_probs(labels,probabilities,positive_class='real',threshold=threshold)


def summarize_heads(data, heads):
    summaries = {}
    for spec in FEATURE_GROUPS:
        values = [heads[spec][str(seed)] for seed in (43,44,45)]
        probabilities = np.mean([v['test_probabilities'] for v in values],axis=0)
        train_probability = np.mean([v['train_probabilities'] for v in values],axis=0)
        from scripts.train_torch_probe_feature_sets import _select_f1_threshold
        threshold = _select_f1_threshold(data['y_train'],train_probability)
        reports = {name:probability_metrics(data['y_test'],probabilities,t) for name,t in [('fixed_0.5',.5),('train_f1',threshold)]}
        scores = {name:[] for name in ('auc','real_aupr','hall_aupr')}
        for value in values:
            report = value['metrics']['threshold_reports']['fixed_0.5']['test_metrics']
            scores['auc'].append(report['auc'])
            scores['real_aupr'].append(report['real_positive']['aupr'])
            scores['hall_aupr'].append(report['hallucination_positive']['aupr'])
        summaries[spec] = dict(seed_mean_std={k:dict(mean=float(np.mean(v)),std=float(np.std(v,ddof=0))) for k,v in scores.items()},
                               ensemble_reports=reports,ensemble_train_f1_threshold=float(threshold),
                               per_seed_metrics={str(seed):heads[spec][str(seed)]['metrics'] for seed in (43,44,45)})
    return summaries


def train(args):
    from scripts.train_torch_probe_feature_sets import TorchProbeConfig, train_and_evaluate_probe
    import hashlib
    data = load_training_data(args.model,args.candidate,getattr(args,'train_despite_known_failure',False))
    root = result_root(args.model)/'training'/args.candidate
    signature = hashlib.sha256()
    for spec in FEATURE_GROUPS:
        for split in ('train','test'):
            value = data['X_'+split][spec]
            signature.update(spec.encode()+split.encode()+str(value.shape).encode()+value.tobytes())
    for split in ('train','test'):
        signature.update(data['y_'+split].tobytes())
        signature.update(json.dumps(data[split+'_mentions'],sort_keys=True).encode())
    protocol = asdict(TorchProbeConfig())
    signature.update(json.dumps(protocol,sort_keys=True).encode())
    signature.update(sha256_file(ROOT/'scripts/train_torch_probe_feature_sets.py').encode())
    if data.get('numerical_exception'):
        signature.update(json.dumps(data['numerical_exception'],sort_keys=True).encode())
    fingerprint = signature.hexdigest()
    manifest = dict(fingerprint=fingerprint,scales={k:v.tolist() for k,v in data['scales'].items()},
                    protocol=json.loads(json.dumps(protocol)),group_order={k:list(v) for k,v in FEATURE_GROUPS.items()},
                    train_mentions=data['train_mentions'],test_mentions=data['test_mentions'],
                    numerical_validation_sha256=data['validation_sha256'],seeds=[43,44,45],
                    note='All heads trained under exactly the same protocol; no unverified AE reuse.')
    if data.get('numerical_exception'):
        manifest['numerical_exception'] = data['numerical_exception']
    immutable_json(manifest,root/'manifest.json')
    heads = {spec:{} for spec in FEATURE_GROUPS}
    started = time.perf_counter()
    for spec in FEATURE_GROUPS:
        for seed in (43,44,45):
            head_root = root/'heads'/spec/f'seed{seed}'
            result_path = head_root/'result.pt'
            if result_path.exists():
                result = torch.load(result_path,map_location='cpu',weights_only=False)
                if result['fingerprint'] != fingerprint:
                    raise ValueError('Wrong detector resume fingerprint')
                for name,digest in result['artifacts'].items():
                    if sha256_file(name) != digest:
                        raise ValueError(f'Changed completed head: {name}')
            else:
                # Incomplete attempts are preserved; trainer never overwrites their checkpoints.
                attempt = head_root/f'attempt_{time.time_ns()}'
                metrics = train_and_evaluate_probe(X_train=data['X_train'][spec],y_train=data['y_train'],
                    X_val=np.empty((0,data['X_train'][spec].shape[1]),dtype=np.float32),y_val=np.empty(0,dtype=np.int32),
                    X_test=data['X_test'][spec],y_test=data['y_test'],config=TorchProbeConfig(seed=seed),
                    device=torch.device(args.device),output_dir=str(attempt),return_probabilities=True)
                train_p = np.asarray(metrics.pop('train_probabilities'),dtype=np.float32)
                test_p = np.asarray(metrics.pop('test_probabilities'),dtype=np.float32)
                checkpoint = attempt/'model.pt'
                recomputed = checkpoint_probabilities(checkpoint,data['X_test'][spec])
                discrepancy = float(np.max(np.abs(recomputed-test_p)))
                if discrepancy > 1e-5:
                    raise ValueError(f'Checkpoint prediction mismatch {discrepancy}')
                result = dict(fingerprint=fingerprint,metrics=metrics,train_probabilities=train_p,test_probabilities=test_p,
                               checkpoint=str(checkpoint),prediction_recompute_max_error=discrepancy,
                               artifacts={str(p):sha256_file(p) for p in attempt.glob('*') if p.is_file()})
                atomic_torch_save(result,result_path)
            heads[spec][str(seed)] = result
            print(f"{args.model} {spec} seed{seed}: {result['metrics']['auc']:.6f}",flush=True)
    summary = dict(model=args.model,candidate=args.candidate,fingerprint=fingerprint,status='COMPLETE_48_HEADS',
                    groups=summarize_heads(data,heads),elapsed_seconds=time.perf_counter()-started)
    if data.get('numerical_exception'):
        summary['numerical_exception'] = data['numerical_exception']
    destination = root/'summary.json'
    if not destination.exists():
        atomic_json_save(summary,destination)
    else:
        old = json.loads(destination.read_text())
        if old['fingerprint'] != fingerprint or old['groups'] != summary['groups']:
            raise ValueError('Changed resumed summaries')
    print(f'{args.model}: complete 16 x 3 heads',flush=True)


def select_primary(summaries):
    if len(summaries)!=4:
        raise ValueError('Primary selection requires all four models')
    checks = {}
    for fusion in ('AE+a','AE+g'):
        checks[fusion] = {}
        for model,summary in summaries.items():
            joint = summary['groups']['AE+s+n']['ensemble_reports']['fixed_0.5']
            fused = summary['groups'][fusion]['ensemble_reports']['fixed_0.5']
            delta_auc = fused['auc']-joint['auc']
            delta_ap = fused['hallucination_positive']['aupr']-joint['hallucination_positive']['aupr']
            checks[fusion][model] = dict(delta_auroc=delta_auc,delta_hall_aupr=delta_ap,within_tolerance=delta_auc>=-.005-1e-12 and delta_ap>=-.01-1e-12)
    selected = next((f for f in ('AE+a','AE+g') if all(r['within_tolerance'] for r in checks[f].values())), 'AE+s+n')
    return selected,checks


def freeze(candidate):
    frozen_path = ROOT/'outputs'/VERSION/'independent_freeze.json'
    if frozen_path.exists():
        checked_freeze()
        print('RESUME: frozen protocol and artifacts unchanged',flush=True)
        return
    summaries = {}
    artifacts = {}
    for model in MODELS:
        root = result_root(model)/'training'/candidate
        summary = json.loads((root/'summary.json').read_text())
        if summary['status'] != 'COMPLETE_48_HEADS' or set(summary['groups']) != set(FEATURE_GROUPS):
            raise ValueError('Incomplete detector groups')
        verification=json.loads((root/'verification.json').read_text())
        if verification['status']!='PASS_48_CHECKPOINTS' or len(verification['recomputation'])!=48:
            raise ValueError('Independent CPU checkpoint/metric verification must precede freeze')
        summaries[model] = summary
        for path in root.rglob('*'):
            if path.is_file():
                artifacts[str(path)] = sha256_file(path)
    selected,checks = select_primary(summaries)
    for directory in ('features','models','scripts','utils','coco-labeling'):
        for path in (ROOT/directory).glob('*.py'):
                artifacts[str(path)] = sha256_file(path)
    for path in (ROOT/'configs').glob('*.yaml'):
        artifacts[str(path)] = sha256_file(path)
    for path in (ROOT/'outputs'/VERSION/'audit_gate.json',ROOT/'outputs'/VERSION/f'old_validation_{candidate}.json',
                 subset_directory()/'selection.json',subset_directory()/'gate.json'):
        if path.exists():
            artifacts[str(path)]=sha256_file(path)
    cohort_path = ROOT/'outputs'/VERSION/'independent_cohort.json'
    cohort=json.loads(cohort_path.read_text())
    if len(cohort['images'])!=2000:
        raise ValueError('Independent confirmation requires exactly 2000 frozen images')
    assert_cohort_isolation(cohort['images'],cohort['used_image_ids'],cohort['used_image_checksums'])
    for name,digest in cohort['scanned_records'].items():
        if sha256_file(name)!=digest:
            raise ValueError(f'COCO usage record changed before freeze: {name}')
    for row in cohort['images']:
        if sha256_file(row['path'])!=row['sha256']:
            raise ValueError('Selected independent image changed before freeze')
    configurations=[json.loads((result_root(m)/'old'/candidate/'manifest.json').read_text())['config'] for m in MODELS]
    if any(c!=configurations[0] for c in configurations):
        raise ValueError('Old model extraction configurations differ')
    config=configurations[0]
    for name in ('annotation_file','captions_file'):
        path=Path(config['dataset'][name])
        artifacts[str(path)]=sha256_file(path)
    artifacts[str(cohort_path)] = sha256_file(cohort_path)
    payload = dict(status='FROZEN_BEFORE_INDEPENDENT_GENERATION',candidate=candidate,primary=selected,
                   config=config,
                   frozen_time_unix_ns=time.time_ns(),generation_seed='20260907 + image_id',
                   selection_checks=checks,tolerances=dict(auroc=.005,hall_aupr=.01),
                   artifacts=artifacts,groups={k:list(v) for k,v in FEATURE_GROUPS.items()},
                   evaluation_only=True,no_bootstrap=True,
                   comparisons=[['AE+i+s','AE+i'],['AE+s+n','AE+s'],['AE+s+n','raw_AE+S+N_vec'],
                                ['AE+a','AE+s+n'],['AE+g','AE+s+n'],['H_SN','U_SN'],['H_SK','U_SK'],
                                ['AE+s+L(N_end)','AE+s+n'],['AE+s+kappa_end','AE+s+kappa_vec']])
    immutable_json(payload,frozen_path)
    print('FROZEN',selected,flush=True)


def assert_cohort_isolation(selected, used_ids, used_checksums):
    ids = [r['image_id'] for r in selected]
    hashes = [r['sha256'] for r in selected]
    if len(ids)!=len(set(ids)) or len(hashes)!=len(set(hashes)) or set(ids)&set(used_ids) or set(hashes)&set(used_checksums):
        raise ValueError('Independent cohort ID/checksum leakage or duplicate')


def prepare_independent(args):
    """Scan recognizable usage records, not the COCO annotation/caption universe."""
    from utils.config_utils import load_config
    destination = ROOT/'outputs'/VERSION/'independent_cohort.json'
    if destination.exists():
        print('RESUME: immutable independent cohort already selected',flush=True)
        return
    used, scanned = set(), {}
    def collect(value, key=''):
        if isinstance(value,dict):
            for name,child in value.items():
                if str(name).isdigit() and isinstance(child,dict) and any(k in child for k in ('generated_text','response_token_ids','caption','object_token_spans')):
                    used.add(int(name))
                collect(child,str(name))
        elif isinstance(value,list):
            if key in ('train','test','val','validation') or key.endswith('image_ids'):
                used.update(int(v) for v in value if isinstance(v,(int,str)) and str(v).isdigit())
            else:
                for child in value:
                    collect(child,key)
        elif key=='image_id' and isinstance(value,(int,str)) and str(value).isdigit():
            used.add(int(value))
        elif isinstance(value,str):
            used.update(int(v) for v in re.findall(r'COCO_(?:val|train)2014_(\d{12})',value))
    for base in (ROOT/'outputs',ROOT.parent/'token-detector'/'outputs'):
        for directory,subdirectories,files in os.walk(base):
            subdirectories[:] = [d for d in subdirectories if d != VERSION and d not in ('probes','heads','audit_vectors')]
            for name in files:
                path = Path(directory)/name
                text = str(path).lower()
                if not ('coco' in str(path.relative_to(base)).lower() or 'pope' in text):
                    continue
                if path.suffix not in ('.json','.jsonl') or not any(word in name.lower() for word in ('generation','split','manifest','cohort','selected_image')):
                    continue
                try:
                    with path.open() as handle:
                        if path.suffix=='.jsonl':
                            for line in handle:
                                if line.strip():
                                    collect(json.loads(line))
                        else:
                            collect(json.load(handle))
                except (UnicodeError,json.JSONDecodeError) as exc:
                    raise ValueError(f'Unreadable recognizable COCO usage record {path}') from exc
                scanned[str(path)] = sha256_file(path)
    config = load_config(args.config)
    dataset = config['dataset']
    image_dir = Path(dataset['coco_root'])/'val2014'
    annotation = json.loads(Path(dataset['annotation_file']).read_text())
    images = {int(row['id']):image_dir/row['file_name'] for row in annotation['images']}
    used_hashes = {sha256_file(images[i]) for i in sorted(used & set(images)) if images[i].is_file()}
    candidates = sorted(i for i in images if i not in used and images[i].is_file())
    random.Random(20260907).shuffle(candidates)
    selected,selected_hashes = [],set()
    for image_id in candidates:
        digest = sha256_file(images[image_id])
        if digest in used_hashes or digest in selected_hashes:
            continue
        selected.append(dict(image_id=image_id,path=str(images[image_id]),sha256=digest))
        selected_hashes.add(digest)
        if len(selected)==2000:
            break
    if len(selected)!=2000:
        raise ValueError('Insufficient independent local images')
    assert_cohort_isolation(selected,used,used_hashes)
    immutable_json(dict(version=VERSION,seed=20260907,images=selected,evaluation_only=True,
                         used_image_ids=sorted(used),used_image_checksums=sorted(used_hashes),
                         scanned_records=scanned,eligible_local_images=len(candidates),
                         independent_of_study_selection_not_foundation_pretraining=True),destination)
    print(json.dumps(dict(selected=2000,used_ids=len(used),scanned_records=len(scanned),eligible_local_images=len(candidates))),flush=True)


def checked_freeze():
    payload = json.loads((ROOT/'outputs'/VERSION/'independent_freeze.json').read_text())
    if payload['status'] != 'FROZEN_BEFORE_INDEPENDENT_GENERATION':
        raise ValueError('Independent generation/evaluation requires the sealed old-cohort detectors')
    for name,digest in payload['artifacts'].items():
        if sha256_file(name) != digest:
            raise ValueError(f'Frozen artifact changed: {name}')
    return payload


def generate_new(args):
    import importlib.util
    from PIL import Image
    from models import build_model
    from utils.config_utils import load_config, get_model_cfg
    frozen = checked_freeze()
    cohort = json.loads((ROOT/'outputs'/VERSION/'independent_cohort.json').read_text())
    assert_cohort_isolation(cohort['images'],cohort['used_image_ids'],cohort['used_image_checksums'])
    root = result_root(args.model)/'independent/data'
    worker = cohort['images'][args.rank::args.world_size]
    config = load_config(args.config)
    if config!=frozen['config']:
        raise ValueError('Independent generation config differs from frozen old-cohort config')
    model_cfg = get_model_cfg(config,args.model)
    prompt = str((config.get('run') or {}).get('prompt') or 'Describe this image.')
    freeze_hash=sha256_file(ROOT/'outputs'/VERSION/'independent_freeze.json')
    pending=[]
    for row in worker:
        path=root/'generation_shards'/f"image_{row['image_id']:012d}.json"
        if path.exists():
            value=json.loads(path.read_text())
            if (value['image_id']!=row['image_id'] or value['image_sha256']!=row['sha256'] or
                value['freeze_sha256']!=freeze_hash or value['generation_config']!=model_cfg or
                value['prompt']!=prompt or value['seed']!=20260907+row['image_id'] or
                sha256_file(path)!=json.loads(path.with_suffix('.sha256.json').read_text())['sha256']):
                raise ValueError('Wrong completed generation fingerprint/checksum')
        else:
            pending.append(row)
    if args.smoke:
        pending = pending[:1]
    if not pending:
        print('RESUME: assigned independent generations complete',flush=True)
        return
    wrapper = build_model(args.model,model_cfg,device=args.device)
    spec = importlib.util.spec_from_file_location('consistency_label_coco',ROOT/'coco-labeling/label_coco.py')
    labeling = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(labeling)
    evaluator = labeling.CocoChairEvaluator.from_cache(instances_file=config['dataset']['annotation_file'],
        captions_file=config['dataset']['captions_file'],cache_path=str(ROOT/'outputs/chair_cache/coco_val2014_chair.pkl'))
    started = time.perf_counter()
    for offset,row in enumerate(pending,1):
        tick=time.perf_counter()
        torch.cuda.reset_peak_memory_stats(torch.device(args.device))
        if sha256_file(row['path']) != row['sha256']:
            raise ValueError('Independent image bytes changed')
        image_id = row['image_id']
        # Per-image seed makes interrupted generation independent of worker scheduling.
        random.seed(20260907+image_id)
        np.random.seed(20260907+image_id)
        torch.manual_seed(20260907+image_id)
        torch.cuda.manual_seed_all(20260907+image_id)
        with Image.open(row['path']) as source:
            generated = wrapper.generate(source.convert('RGB'),prompt=prompt)
        caption = generated.generated_text
        token_ids = [int(v) for v in generated.response_token_ids]
        chair = evaluator.compute_chair_token(image_id,caption)
        all_spans,spans,resolution = labeling._inslen_official_token_spans(model_key=args.model,
            tokenizer=wrapper.tokenizer,caption=caption,token_ids=token_ids,chair_info=chair)
        label = labeling._compact_inslen_label_entry(image_id=image_id,model_key=args.model,caption=caption,
            all_spans=all_spans,selected_spans=spans,chair_info=chair,official_svar_samples=[],resolution_summary=resolution)
        destination=root/'generation_shards'/f'image_{image_id:012d}.json'
        immutable_json(dict(image_id=image_id,image_sha256=row['sha256'],
                             freeze_sha256=freeze_hash,
                             generation=dict(generated_text=caption,response_token_ids=token_ids),labeling=label,
                             generation_config=model_cfg,prompt=prompt,seed=20260907+image_id,
                             elapsed_seconds=time.perf_counter()-tick,
                             peak_allocated_bytes=torch.cuda.max_memory_allocated(torch.device(args.device))),destination)
        immutable_json(dict(sha256=sha256_file(destination)),destination.with_suffix('.sha256.json'))
        print(json.dumps(dict(model=args.model,image_id=image_id,completed_now=offset,total=len(pending),elapsed_seconds=time.perf_counter()-started)),flush=True)


def merge_new(args):
    frozen=checked_freeze()
    from utils.config_utils import get_model_cfg
    cohort = json.loads((ROOT/'outputs'/VERSION/'independent_cohort.json').read_text())
    root = result_root(args.model)/'independent/data'
    expected={root/'generation_shards'/f"image_{r['image_id']:012d}.json" for r in cohort['images']}
    actual={p for p in (root/'generation_shards').glob('image_*.json') if not p.name.endswith('.sha256.json')}
    if actual!=expected:
        raise ValueError('Missing/unexpected independent generation images')
    generations,labels,artifacts = {},{},{}
    for row in cohort['images']:
        path = root/'generation_shards'/f"image_{row['image_id']:012d}.json"
        value = json.loads(path.read_text())
        if sha256_file(path)!=json.loads(path.with_suffix('.sha256.json').read_text())['sha256']:
            raise ValueError('Independent generation checksum mismatch')
        if value['image_id']!=row['image_id'] or value['image_sha256']!=row['sha256'] or value['freeze_sha256']!=sha256_file(ROOT/'outputs'/VERSION/'independent_freeze.json'):
            raise ValueError('Independent generation provenance mismatch')
        if value['generation_config']!=get_model_cfg(frozen['config'],args.model) or value['seed']!=20260907+row['image_id']:
            raise ValueError('Independent generation protocol mismatch')
        generations[str(row['image_id'])] = value['generation']
        labels[str(row['image_id'])] = value['labeling']
        artifacts[str(path)] = sha256_file(path)
    immutable_json(generations,root/'generations.json')
    immutable_json(labels,root/'labeling.json')
    immutable_json(dict(evaluation_only=True,image_ids=[r['image_id'] for r in cohort['images']],
                         processed_images=len(labels),artifacts=artifacts),root/'processed_images.json')


def baseline_new(args):
    from models import build_model
    from scripts.run_ffn_visual_source_attribution import source_config
    from scripts.run_jffn_p_comparison import _extract_one_image
    from utils.config_utils import load_config,get_extraction_model_cfg,get_dgst_t_cfg
    from utils.io_utils import save_pkl
    frozen=checked_freeze()
    root = result_root(args.model)/'independent/data'
    labels = {int(k):v for k,v in json.loads((root/'labeling.json').read_text()).items()}
    generations = {int(k):v for k,v in json.loads((root/'generations.json').read_text()).items()}
    ids = sorted(labels)
    if len(ids)!=2000 or set(ids)!=set(generations):
        raise ValueError('Independent base features require all 2000 generations and labels')
    config = load_config(args.config)
    if config!=frozen['config']:
        raise ValueError('Independent AE/R_cos config differs from freeze')
    cohort=json.loads((ROOT/'outputs'/VERSION/'independent_cohort.json').read_text())
    images={r['image_id']:r for r in cohort['images']}
    if set(ids)!=set(images):
        raise ValueError('Independent AE/R_cos image cohort mismatch')
    provenance=dict(freeze_sha256=sha256_file(ROOT/'outputs'/VERSION/'independent_freeze.json'),
                     inputs={str(root/name):sha256_file(root/name) for name in ('labeling.json','generations.json')})
    fingerprint=sha256_text(json.dumps(provenance,sort_keys=True))
    immutable_json(dict(provenance,fingerprint=fingerprint),root/'baseline_manifest.json')
    cfg = source_config(get_dgst_t_cfg(config),model=args.model,k=4,chunk=256,backend='vmap_jvp')
    cfg.update(source_modes=['hpre_cos'],jffn_vector_path_only=False)
    worker = ids[args.rank::args.world_size]
    pending=[]
    for image_id in worker:
        path=root/'base_shards'/f'image_{image_id:012d}.pt'
        if path.exists():
            value=torch.load(path,map_location='cpu',weights_only=False)
            if value['fingerprint']!=fingerprint or sha256_file(path)!=json.loads(path.with_suffix('.sha256.json').read_text())['sha256']:
                raise ValueError('Wrong independent AE/R_cos resume fingerprint/checksum')
        else:
            pending.append(image_id)
    if args.smoke:
        pending = pending[:1]
    if pending:
        wrapper = build_model(args.model,get_extraction_model_cfg(config,args.model),device=args.device)
        wrapper.model.requires_grad_(False).eval()
        for offset,image_id in enumerate(pending,1):
            tick=time.perf_counter()
            torch.cuda.reset_peak_memory_stats(torch.device(args.device))
            if sha256_file(images[image_id]['path'])!=images[image_id]['sha256']:
                raise ValueError('Independent AE/R_cos image checksum mismatch')
            positions,mentions = _extract_one_image(wrapper=wrapper,config=config,image_id=image_id,
                label_row=labels[image_id],generation_row=generations[image_id],dgst_cfg=cfg,compact=False)
            rows = [dict(p['result'],image_id=image_id,response_token_idx=p['response_index']) for p in positions]
            path = root/'base_shards'/f'image_{image_id:012d}.pt'
            atomic_torch_save(dict(image_id=image_id,features=rows,mentions=mentions,fingerprint=fingerprint,
                                   image_sha256=images[image_id]['sha256'],elapsed_seconds=time.perf_counter()-tick,
                                   peak_allocated_bytes=torch.cuda.max_memory_allocated(torch.device(args.device))),path)
            atomic_json_save(dict(sha256=sha256_file(path)),path.with_suffix('.sha256.json'))
            print(f'{args.model} independent AE/R_cos {offset}/{len(pending)} image={image_id}',flush=True)
    if all((root/'base_shards'/f'image_{i:012d}.pt').exists() for i in ids):
        destination = root/'features.pkl'
        if destination.exists() and sha256_file(destination)!=json.loads(destination.with_suffix('.sha256.json').read_text())['sha256']:
            raise ValueError('Completed independent base features changed')
        if not destination.exists():
            rows = []
            for image_id in ids:
                path = root/'base_shards'/f'image_{image_id:012d}.pt'
                if sha256_file(path)!=json.loads(path.with_suffix('.sha256.json').read_text())['sha256']:
                    raise ValueError('Independent baseline shard checksum mismatch')
                rows.extend(torch.load(path,map_location='cpu',weights_only=False)['features'])
            save_pkl(rows,str(destination))
            atomic_json_save(dict(sha256=sha256_file(destination)),destination.with_suffix('.sha256.json'))


def evaluate_new(args):
    from features.jffn_experiment import mentions_for_image
    frozen = checked_freeze()
    candidate = frozen['candidate']
    training_root = result_root(args.model)/'training'/candidate
    train_manifest = json.loads((training_root/'manifest.json').read_text())
    old_summary = json.loads((training_root/'summary.json').read_text())
    cohort = json.loads((ROOT/'outputs'/VERSION/'independent_cohort.json').read_text())
    root=result_root(args.model)/'independent'/candidate
    manifest=json.loads((root/'manifest.json').read_text())
    expected_paths={root/'shards'/f"image_{r['image_id']:012d}.pt" for r in cohort['images']}
    if set((root/'shards').glob('*.pt'))!=expected_paths or set(manifest['image_ids'])!={r['image_id'] for r in cohort['images']}:
        raise ValueError('Missing/unexpected independent extraction images')
    data_root=result_root(args.model)/'independent/data'
    generations=json.loads((data_root/'generations.json').read_text())
    labeling=json.loads((data_root/'labeling.json').read_text())
    raw_rows,mentions,artifacts = [],[],{}
    keys = ('AE','I','S','N_vec','N_end','kappa_vec','kappa_end','R_cos','D_EW','D_WF','D_EF')
    for image in cohort['images']:
        path = result_root(args.model)/'independent'/candidate/'shards'/f"image_{image['image_id']:012d}.pt"
        data = torch.load(path,map_location='cpu',weights_only=False)
        if data['image_ids'] != [image['image_id']] or data['image_sha256']!=image['sha256'] or sha256_file(path)!=json.loads(path.with_suffix('.sha256.json').read_text())['sha256']:
            raise ValueError('Independent extraction shard mismatch')
        if sha256_file(image['path'])!=image['sha256']:
            raise ValueError('Independent image bytes changed at evaluation')
        expected,indices,_=mentions_for_image(image_id=image['image_id'],labeling_row=labeling[str(image['image_id'])],
                                              response_token_ids=generations[str(image['image_id'])]['response_token_ids'])
        check_extracted_image(data,manifest,image['image_id'],args.model,expected,indices)
        for position in data['positions']:
            if not all(v['finite'] and v['kappa_range_pass'] and v['kappa_difference_bound_pass'] and v['closure_lower_bound_pass'] for v in position['dual_net']):
                raise ValueError('Independent numerical finite/kappa boundary failed')
        positions = {r['target_key']:r for r in data['positions']}
        if set(positions)!={r['target_key'] for r in data['sample_table']}:
            raise ValueError('Independent target/mention mismatch')
        for row in data['sample_table']:
            mentions.append(row)
            raw_rows.append({k:np.asarray(positions[row['target_key']][k],dtype=np.float32) for k in keys})
        artifacts[str(path)] = sha256_file(path)
    if len({r['mention_id'] for r in mentions})!=len(mentions):
        raise ValueError('Duplicated independent mentions')
    raw = {k:np.stack([r[k] for r in raw_rows]) for k in keys}
    matrices = feature_sets(raw,train_manifest['scales'])
    labels = np.array([r['label'] for r in mentions],dtype=np.int32)
    summaries,predictions = {},{}
    for spec in FEATURE_GROUPS:
        seeds = {}
        p = []
        for seed in (43,44,45):
            result_path = training_root/'heads'/spec/f'seed{seed}'/'result.pt'
            head = torch.load(result_path,map_location='cpu',weights_only=False)
            prob = checkpoint_probabilities(head['checkpoint'],matrices[spec])
            p.append(prob)
            seeds[str(seed)] = {name:probability_metrics(labels,prob,threshold) for name,threshold in head['metrics']['thresholds'].items()}
        ensemble = np.mean(p,axis=0)
        scores={key:[r['fixed_0.5'][field]['aupr'] if field else r['fixed_0.5']['auc'] for r in seeds.values()]
                for key,field in [('auc',None),('real_aupr','real_positive'),('hall_aupr','hallucination_positive')]}
        summaries[spec] = dict(per_seed=seeds,
            seed_mean_std={k:dict(mean=float(np.mean(v)),std=float(np.std(v,ddof=0))) for k,v in scores.items()},ensemble_reports={
            name:probability_metrics(labels,ensemble,threshold) for name,threshold in (
                ('fixed_0.5',.5),('train_f1',old_summary['groups'][spec]['ensemble_train_f1_threshold']))})
        predictions[spec] = np.stack(p)
    immutable_json(dict(status='COMPLETE_INDEPENDENT_EVALUATION',model=args.model,processed_images=2000,
                         mentions=len(mentions),primary=frozen['primary'],groups=summaries,
                         extraction_manifest_sha256=sha256_file(root/'manifest.json'),
                         freeze_sha256=sha256_file(ROOT/'outputs'/VERSION/'independent_freeze.json'),
                         artifacts=artifacts,evaluation_only=True,no_bootstrap=True),root/'evaluation.json')
    if not (root/'predictions.pt').exists():
        atomic_torch_save(dict(predictions=predictions,mentions=mentions,labels=labels),root/'predictions.pt')
        immutable_json(dict(sha256=sha256_file(root/'predictions.pt')),(root/'predictions.sha256.json'))
    else:
        path=root/'predictions.pt'
        if sha256_file(path)!=json.loads((root/'predictions.sha256.json').read_text())['sha256']:
            raise ValueError('Saved independent predictions checksum mismatch')
        previous=torch.load(path,map_location='cpu',weights_only=False)
        if previous['mentions']!=mentions or not np.array_equal(previous['labels'],labels) or any(not np.array_equal(previous['predictions'][k],v) for k,v in predictions.items()):
            raise ValueError('Independent checkpoint predictions differ on recomputation')
        immutable_json(dict(status='PASS_INDEPENDENT_CHECKPOINT_RECOMPUTATION',heads=48,
                             predictions_sha256=sha256_file(path),evaluation_sha256=sha256_file(root/'evaluation.json')),
                       root/'prediction_verification.json')


def verify_detectors(args):
    """Read checkpoints independently on CPU; also verify both saved metric reports."""
    from scripts.train_torch_probe_feature_sets import _select_f1_threshold
    data=load_training_data(args.model,args.candidate,getattr(args,'train_despite_known_failure',False))
    root=result_root(args.model)/'training'/args.candidate
    manifest=json.loads((root/'manifest.json').read_text())
    summary=json.loads((root/'summary.json').read_text())
    if manifest.get('numerical_exception')!=data.get('numerical_exception') or summary.get('numerical_exception')!=data.get('numerical_exception'):
        raise ValueError('Detector numerical exception provenance mismatch')
    reports,heads,artifacts={},{},{}
    for spec in FEATURE_GROUPS:
        heads[spec]={}
        for seed in (43,44,45):
            path=root/'heads'/spec/f'seed{seed}'/'result.pt'
            head=torch.load(path,map_location='cpu',weights_only=False)
            if head['fingerprint']!=manifest['fingerprint']:
                raise ValueError('Detector verification fingerprint mismatch')
            for name,digest in head['artifacts'].items():
                if sha256_file(name)!=digest:
                    raise ValueError(f'Changed checkpoint artifact: {name}')
                artifacts[name]=digest
            errors={}
            for split in ('train','test'):
                probability=checkpoint_probabilities(head['checkpoint'],data['X_'+split][spec])
                saved=head[split+'_probabilities']
                error=float(np.max(np.abs(probability-saved)))
                if error>1e-5:
                    raise ValueError('Independent detector checkpoint recomputation failed')
                errors[split]=error
                for rule,threshold in head['metrics']['thresholds'].items():
                    recomputed=probability_metrics(data['y_'+split],saved,threshold)
                    if recomputed!=head['metrics']['threshold_reports'][rule][split+'_metrics']:
                        raise ValueError('Detector metrics do not match saved probabilities')
            threshold=_select_f1_threshold(data['y_train'],head['train_probabilities'])
            if threshold!=head['metrics']['thresholds']['train_f1']:
                raise ValueError('Detector threshold was not selected from train predictions')
            heads[spec][str(seed)]=head
            reports[f'{spec}/seed{seed}']=errors
            artifacts[str(path)]=sha256_file(path)
    if summarize_heads(data,heads)!=summary['groups']:
        raise ValueError('Ensemble/seed summary mismatch')
    immutable_json(dict(status='PASS_48_CHECKPOINTS',model=args.model,candidate=args.candidate,
                         recomputation=reports,artifacts=artifacts),root/'verification.json')
    print(args.model,': 48 checkpoints and all metrics independently verified',flush=True)


def summarize(candidate, independent=False, numerical_exception=False):
    """Publish all prespecified groups and paired point differences, without bootstrap."""
    summaries={}
    for model in MODELS:
        path = result_root(model)/(('independent/'+candidate+'/evaluation.json') if independent else ('training/'+candidate+'/summary.json'))
        summaries[model]=json.loads(path.read_text())
    comparisons=[('AE+i+s','AE+i'),('AE+s+n','AE+s'),('AE+s+n','raw_AE+S+N_vec'),
                 ('AE+a','AE+s+n'),('AE+g','AE+s+n'),('H_SN','U_SN'),('H_SK','U_SK'),
                 ('AE+s+L(N_end)','AE+s+n'),('AE+s+kappa_end','AE+s+kappa_vec')]
    rows,seed_rows,contrasts=[],[],[]
    for model,summary in summaries.items():
        if set(summary['groups'])!=set(FEATURE_GROUPS):
            raise ValueError('Missing prespecified group')
        for spec,group in summary['groups'].items():
            for threshold,report in group['ensemble_reports'].items():
                row=dict(model=model,feature_set=spec,threshold_rule=threshold,ensemble_auroc=report['auc'])
                for metric,value in group['seed_mean_std'].items():
                    row.update({f'seed_{metric}_{stat}':score for stat,score in value.items()})
                for label,field in [('REAL','real_positive'),('HALL','hallucination_positive')]:
                    row.update({label+'_'+metric:report[field][metric] for metric in ('aupr','precision','recall','f1')})
                rows.append(row)
            seeds=group['per_seed'] if independent else group['per_seed_metrics']
            for seed,value in seeds.items():
                reports=value if independent else {name:r['test_metrics'] for name,r in value['threshold_reports'].items()}
                for threshold,report in reports.items():
                    row=dict(model=model,feature_set=spec,seed=seed,threshold_rule=threshold,auroc=report['auc'])
                    for label,field in [('REAL','real_positive'),('HALL','hallucination_positive')]:
                        row.update({label+'_'+metric:report[field][metric] for metric in ('aupr','precision','recall','f1')})
                    seed_rows.append(row)
        for left,right in comparisons:
            a=summary['groups'][left]['ensemble_reports']['fixed_0.5']
            b=summary['groups'][right]['ensemble_reports']['fixed_0.5']
            contrasts.append(dict(model=model,left=left,right=right,delta_auroc=a['auc']-b['auc'],
                                   delta_hall_aupr=a['hallucination_positive']['aupr']-b['hallucination_positive']['aupr'],
                                   within_point_tolerances=(a['auc']-b['auc']>=-.005-1e-12 and
                                       a['hallucination_positive']['aupr']-b['hallucination_positive']['aupr']>=-.01-1e-12)
                                       if left in ('AE+a','AE+g') else None))
    root=ROOT/'outputs'/VERSION
    scope='independent' if independent else 'old_exploratory'
    payload=dict(scope=scope,candidate=candidate,models=summaries,contrasts=contrasts,no_bootstrap=True,
                  significance_claimed=False,noninferiority_claimed=False)
    if numerical_exception:
        if independent:
            raise ValueError('Training-only numerical exception does not authorize independent confirmation')
        exception=training_numerical_exception(root/f'old_validation_{candidate}.json')
        if any(s.get('numerical_exception')!=exception for s in summaries.values()):
            raise ValueError('Missing numerical exception in detector summaries')
        payload['numerical_exception']=exception
        root=root/EXPLORATORY_TRAINING
    if independent:
        payload['primary_frozen_on_old_cohort']=checked_freeze()['primary']
    immutable_json(payload,root/f'{scope}_results.json')
    for suffix,values in [('groups',rows),('seeds',seed_rows),('contrasts',contrasts)]:
        path=root/f'{scope}_{suffix}.csv'
        if not path.exists():
            with path.open('w',newline='') as handle:
                writer=csv.DictWriter(handle,fieldnames=list(values[0]))
                writer.writeheader()
                writer.writerows(values)
    lines=[f'# {scope}: {candidate}', '', '所有值为点估计；无 bootstrap，不宣称统计显著或非劣。', '',
           '| 特征组 | '+ ' | '.join(MODELS)+' |','| --- | '+' | '.join(['---:']*4)+' |']
    if numerical_exception:
        lines[2:2]=['**数值门控仍为FAIL；用户同意保留原K4特征继续探索性训练，不是正式数值验收通过。**', '']
    for spec in FEATURE_GROUPS:
        cells=[]
        for model in MODELS:
            report=summaries[model]['groups'][spec]['ensemble_reports']['fixed_0.5']
            cells.append(f"{report['auc']:.6f} / {report['hallucination_positive']['aupr']:.6f}")
        lines.append('| '+spec+' | '+' | '.join(cells)+' |')
    lines+=['','表格单元为 ensemble AUROC / HALL-AUPR；全部逐seed与双阈值P/R/F1见同名CSV/JSON。','']
    path=root/f'{scope}_summary.md'
    if not path.exists():
        path.write_text('\n'.join(lines))
    print(scope,': 64 groups, 384 seed-threshold rows',flush=True)


_raw_immutable_json = immutable_json


def immutable_json(payload, path):
    # JSON round-trips integer keys and tuples; compare the persisted representation.
    return _raw_immutable_json(json.loads(json.dumps(payload,allow_nan=False)),path)


def publish_delivery(candidate):
    """Only publish completion after all model artifacts pass their final checksums."""
    from datetime import datetime, timezone
    frozen=checked_freeze()
    if candidate!=frozen['candidate']:
        raise ValueError('Delivery candidate differs from sealed version')
    root=ROOT/'outputs'/VERSION
    validation_path=root/f'old_validation_{candidate}.json'
    validation=json.loads(validation_path.read_text())
    if validation['status']!='PASS_NUMERICAL_OLD_COHORT':
        raise ValueError('Full numerical cohort has not passed')
    old=json.loads((root/'old_exploratory_results.json').read_text())
    new=json.loads((root/'independent_results.json').read_text())
    artifacts={str(validation_path):sha256_file(validation_path)}
    for model in MODELS:
        old_model=old['models'][model]
        new_model=new['models'][model]
        if old_model['status']!='COMPLETE_48_HEADS' or new_model['status']!='COMPLETE_INDEPENDENT_EVALUATION' or new_model['processed_images']!=2000:
            raise ValueError('Incomplete detector/independent model')
        if set(old_model['groups'])!=set(FEATURE_GROUPS) or set(new_model['groups'])!=set(FEATURE_GROUPS):
            raise ValueError('Incomplete delivery feature groups')
        training=result_root(model)/'training'/candidate
        verification=json.loads((training/'verification.json').read_text())
        artifacts.update(verification['artifacts'])
        artifacts.update(validation['models'][model]['artifacts'])
        artifacts.update(new_model['artifacts'])
        predictions=result_root(model)/'independent'/candidate/'predictions.pt'
        recomputation=predictions.parent/'prediction_verification.json'
        checked=json.loads(recomputation.read_text())
        if checked['status']!='PASS_INDEPENDENT_CHECKPOINT_RECOMPUTATION' or checked['heads']!=48 or checked['predictions_sha256']!=sha256_file(predictions):
            raise ValueError('Independent checkpoints have not passed prediction recomputation')
        artifacts[str(recomputation)]=sha256_file(recomputation)
        artifacts[str(predictions)]=json.loads(predictions.with_suffix('.sha256.json').read_text())['sha256']
        artifacts[str(training/'verification.json')]=sha256_file(training/'verification.json')
    for path in root.glob('*results.json'):
        artifacts[str(path)]=sha256_file(path)
    for name,digest in artifacts.items():
        if sha256_file(name)!=digest:
            raise ValueError(f'Final artifact checksum failed: {name}')
    completion=dict(status='PASS_FOUR_MODEL_CONFIRMATION',candidate=candidate,primary=frozen['primary'],
                    models=list(MODELS),old_images_per_model=4000,new_images_per_model=2000,
                    detector_heads=192,no_bootstrap=True,artifacts=artifacts)
    immutable_json(completion,root/'delivery_verification.json')
    date=datetime.now(timezone.utc).date().isoformat()
    marker='<!-- consistency-v2-completed-delivery -->'
    lines=[marker,f'## {date}：第七点 v2 完整验收与独立确认','',
           f'数值版本 `{candidate}`；原3200/800图片划分训练/探索，共192个检测器。共享新2000图每模型均仅作评估，未重新训练、拟合尺度或阈值。',
           f'按预定四模型双容差选定主版本 `{frozen["primary"]}`；新结果不反向修改选型。以下均为点估计，无bootstrap，不作显著性或统计非劣声称。','',
           '### 旧4000图：探索结果','',(root/'old_exploratory_summary.md').read_text(),
           '### 新2000图：独立确认','',(root/'independent_summary.md').read_text(),
           '### 预定对照的独立增量','',
           '| 模型 | 比较 | ensemble ΔAUROC | ΔHALL-AUPR |','| --- | --- | ---: | ---: |']
    for row in new['contrasts']:
        lines.append(f'| {row["model"]} | {row["left"]} − {row["right"]} | {row["delta_auroc"]:+.6f} | {row["delta_hall_aupr"]:+.6f} |')
    lines+=['',f'全部seed与固定0.5/train-REAL-F1阈值的P/R/F1、均值/标准差见 `outputs/{VERSION}/` 的JSON/CSV；完整checksum验收为 `delivery_verification.json`。',
            '独立仅指未参与本研究选型，不声称COCO图片未进入基础模型预训练。v1产物未覆盖，没有提交或上传。','']
    report=ROOT/'ffn_visual_source_attribution_report.md'
    if marker not in report.read_text():
        with report.open('a') as handle:
            handle.write('\n'+'\n'.join(lines))
    task=ROOT/'docs/CURRENT_TASK.md'
    if marker not in task.read_text():
        with task.open('a') as handle:
            handle.write(f'\n{marker}\n## {date} 第七点 v2 自动执行终末验收\n\n'
                         f'- 四模型旧4000图、独立2000图和192个训练头全部验收通过；数值版本 `{candidate}`，主版本 `{frozen["primary"]}`。\n'
                         f'- 最终验收为 `outputs/{VERSION}/delivery_verification.json`，旧/新完整结果和主报告已写入。全部命令、退出码、耗时、失败及恢复记录在 `workflow_runs/`；逐图显存/耗时在各shard。\n'
                         '- 无bootstrap、无新测试集选型、无提交或上传；此前“执行中”条目为按时间保留的过程记录。\n')
    print('PASS: four-model old and independent artifacts verified and reports published locally',flush=True)


def pipeline(args):
    """Two ordinary process queues, with a hard all-model barrier at each gate."""
    import fcntl
    import subprocess
    from concurrent.futures import ThreadPoolExecutor
    root=ROOT/'outputs'/VERSION
    run_root=root/'workflow_runs'/str(time.time_ns())
    run_root.mkdir(parents=True)
    environment=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    lanes=[('cuda:0',('qwen2_5_vl_7b','llava_1_5_7b')),('cuda:1',('qwen3_vl_8b','internvl_2_5_8b'))]
    def stage(name,model=None,device=None,candidate=None,rank=0,world_size=1):
        command=[sys.executable,'-u',str(Path(__file__).resolve()),name,'--config',args.config]
        if getattr(args,'production_after_subset',False) or name=='pipeline':
            command+=['--production-after-subset']
        if getattr(args,'train_despite_known_failure',False):
            command+=['--train-despite-known-failure']
        if model:
            command+=['--model',model,'--device',device,'--rank',str(rank),'--world-size',str(world_size)]
        if candidate:
            command+=['--candidate',candidate]
        if name=='extract-old' and not args.full_reference:
            command+=['--no-full-reference']
        if name=='summarize' and model is None and device=='independent':
            command+=['--independent']
        stamp=time.time_ns()
        destination=run_root/f'{stamp}_{name}_{model or "all"}'
        record=dict(stage=name,model=model,candidate=candidate,command=command,rank=rank,world_size=world_size,
                    started_unix_ns=stamp,status='WAITING_FOR_EXISTING_WORKER',
                    environment={k:environment[k] for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS')})
        atomic_json_save(record,destination.with_suffix('.json'))
        if model:
            # ponytail: fixed two-GPU queues; use a scheduler only for a larger cluster.
            result_root(model).mkdir(parents=True,exist_ok=True)
            with (result_root(model)/f'.{name}_rank{rank}.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
            # The previous worker is finished; the CLI verifies its completed shards on resume.
        record['status']='RUNNING'
        atomic_json_save(record,destination.with_suffix('.json'))
        print('START',name,model or 'all',candidate or '',flush=True)
        tick=time.perf_counter()
        with destination.with_suffix('.log').open('w') as handle:
            result=subprocess.run(command,cwd=ROOT,env=environment,stdout=handle,stderr=subprocess.STDOUT)
        record.update(status='PASS_COMMAND' if result.returncode==0 else 'FAIL_COMMAND',returncode=result.returncode,
                      elapsed_seconds=time.perf_counter()-tick,log_sha256=sha256_file(destination.with_suffix('.log')))
        atomic_json_save(record,destination.with_suffix('.json'))
        if result.returncode:
            raise RuntimeError(f'{name} {model}: exit {result.returncode}; see {destination.with_suffix(".log")}')
        print('DONE',name,model or 'all',flush=True)
    def parallel(stages,candidate):
        def lane(device,models):
            for model in models:
                for name in stages:
                    stage(name,model,device,candidate)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(lane,*lane_args) for lane_args in lanes]
            for future in futures:
                future.result()
    try:
        mode=getattr(args,'stage','pipeline')
        if mode=='train-only':
            if not getattr(args,'train_despite_known_failure',False):
                raise ValueError('This training-only queue requires the explicit numerical exception')
            parallel(['train','verify-detectors'],'fp32_k4')
            stage('summarize',candidate='fp32_k4')
            atomic_json_save(dict(status='COMPLETE_USER_AUTHORIZED_EXPLORATORY_TRAINING_ONLY',
                                  numerical_gate='FAIL_NUMERICAL_OLD_COHORT', independent_started=False,
                                  workflow=str(run_root)),run_root/'status.json')
            return
        if mode in ('subset-pipeline','llava-dual-subset'):
            if mode=='llava-dual-subset':
                with ThreadPoolExecutor(max_workers=2) as pool:
                    futures=[pool.submit(stage,'extract-subset','llava_1_5_7b',f'cuda:{rank}','fp32_k4',rank,2) for rank in (0,1)]
                    for future in futures:
                        future.result()
                stage('extract-subset','llava_1_5_7b','cuda:0','fp32_k4')
                stage('validate-subset','llava_1_5_7b','cuda:0','fp32_k4')
            else:
                parallel(['extract-subset','validate-subset'],'fp32_k4')
            stage('subset-gate')
            if json.loads((subset_directory()/'gate.json').read_text())['status']!='PASS_K4_NUMERICAL_SUBSET':
                raise ValueError('Subset K4 validation failed; stop before full K4 production or training')
            stage('pipeline')
            atomic_json_save(dict(status='COMPLETE_USER_AMENDED_WORKFLOW',workflow=str(run_root)),run_root/'status.json')
            return
        gate=json.loads((root/'audit_gate.json').read_text())
        if gate['status']!='PASS_AUDIT':
            raise ValueError('Finish the all-model numerical audit first')
        production=getattr(args,'production_after_subset',False)
        if production:
            seed_production()
        candidates=('fp32_k4',) if production else CANDIDATES[CANDIDATES.index(gate['selected_candidate']):]
        for candidate in candidates:
            parallel(['extract-old'],candidate)
            stage('validate-old',candidate=candidate)
            validation=json.loads((root/f'old_validation_{candidate}.json').read_text())
            if validation['status']=='PASS_NUMERICAL_OLD_COHORT':
                break
        else:
            raise ValueError('Approved K4 full-cohort numerical bounds failed; stop for review' if production else
                             'Highest numerical candidate failed; no training or independent generation authorized')
        parallel(['train','verify-detectors'],candidate)
        stage('summarize',candidate=candidate)
        stage('freeze',candidate=candidate)
        parallel(['generate-new','merge-new','baseline-new','extract-new','evaluate-new','evaluate-new'],candidate)
        stage('summarize',candidate=candidate,device='independent')
        stage('publish-delivery',candidate=candidate)
        atomic_json_save(dict(status='COMPLETE_FOUR_MODEL_OLD_AND_INDEPENDENT',candidate=candidate,
                              completed_unix_ns=time.time_ns(),workflow=str(run_root)),run_root/'status.json')
    except Exception:
        atomic_json_save(dict(status='FAIL_REQUIRES_REVIEW',traceback=traceback.format_exc(),
                              failed_unix_ns=time.time_ns()),run_root/'status.json')
        raise


def main():
    global result_root
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['gate','prepare-independent','extract-old','validate-old','train','freeze',
                                         'generate-new','merge-new','baseline-new','extract-new','evaluate-new','summarize',
                                         'verify-detectors','pipeline','publish-delivery','prepare-subset',
                                         'extract-subset','validate-subset','subset-gate','subset-pipeline','llava-dual-subset','train-only'])
    parser.add_argument('--models', default=','.join(MODELS))
    parser.add_argument('--model', choices=MODELS)
    parser.add_argument('--candidate', choices=CANDIDATES)
    parser.add_argument('--config', default='configs/model_configs_inslen_official_target.yaml')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world-size', type=int, default=1)
    parser.add_argument('--token-chunk-size', type=int, default=256, choices=[256,128,64,32])
    parser.add_argument('--full-reference',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true', default=True,
                        help='Always validate and resume immutable completed artifacts; never overwrite them.')
    parser.add_argument('--partial', action='store_true')
    parser.add_argument('--independent',action='store_true')
    parser.add_argument('--production-after-subset',action='store_true',
                        help='User-amended full K4-only cohort in its own namespace; requires the subset gate.')
    parser.add_argument('--train-despite-known-failure',action='store_true',
                        help='User-authorized exploratory training on unchanged K4; retain the failed numerical gate.')
    args = parser.parse_args()
    if not 0<=args.rank<args.world_size:
        raise ValueError('Invalid rank/world-size')
    if args.production_after_subset:
        if json.loads((subset_directory()/'gate.json').read_text())['status']!='PASS_K4_NUMERICAL_SUBSET':
            raise ValueError('User-amended production requires the four-model subset gate')
        if args.candidate not in (None,'fp32_k4'):
            raise ValueError('Only K4 production was authorized after subset validation')
        args.full_reference=False
        original_result_root=result_root
        result_root=lambda model:original_result_root(model)/'production_k4'
    elif (subset_directory()/'selection.json').exists() and args.stage in ('pipeline','extract-old','train','freeze','generate-new'):
        raise ValueError('Original all-K64 pipeline superseded by user; use subset-pipeline or --production-after-subset')
    if args.train_despite_known_failure:
        if not args.production_after_subset or args.independent or args.stage not in ('train-only','train','verify-detectors','summarize'):
            raise ValueError('Numerical exception authorizes only exploratory training/verification/summary')
        args.candidate='fp32_k4'
        training_numerical_exception(ROOT/'outputs'/VERSION/'old_validation_fp32_k4.json',create=True)
        production_result_root=result_root
        result_root=lambda model:production_result_root(model)/EXPLORATORY_TRAINING
    if args.stage=='gate':
        numerical_gate(args.models.split(','),args.partial)
    elif args.stage=='prepare-independent':
        prepare_independent(args)
    elif args.stage=='validate-old':
        validate_old(args.candidate)
    elif args.stage=='freeze':
        freeze(args.candidate)
    elif args.stage=='summarize':
        summarize(args.candidate,args.independent,args.train_despite_known_failure)
    elif args.stage=='publish-delivery':
        publish_delivery(args.candidate)
    elif args.stage=='prepare-subset':
        prepare_subset()
    elif args.stage=='subset-gate':
        subset_gate()
    elif args.stage in ('pipeline','subset-pipeline','llava-dual-subset','train-only'):
        import fcntl
        directory=ROOT/'outputs'/VERSION
        directory.mkdir(parents=True,exist_ok=True)
        with (directory/f'.{args.stage}.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            pipeline(args)
    else:
        if not args.model:
            parser.error('--model is required for this stage')
        import fcntl
        root = result_root(args.model)
        root.mkdir(parents=True,exist_ok=True)
        with (root/f'.{args.stage}_rank{args.rank}.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:
                {'extract-old':extract_old,'train':train,'generate-new':generate_new,'merge-new':merge_new,
                 'baseline-new':baseline_new,'extract-new':extract_old,'evaluate-new':evaluate_new,
                 'verify-detectors':verify_detectors,'extract-subset':extract_subset,
                 'validate-subset':lambda a:validate_subset(a.model)}[args.stage](args)
            except Exception:
                atomic_json_save(dict(stage=args.stage,command=sys.argv,status='FAIL',traceback=traceback.format_exc()),
                                 root/f'logs/{args.stage}_failure_{time.time_ns()}.json')
                raise


if __name__ == '__main__':
    main()
