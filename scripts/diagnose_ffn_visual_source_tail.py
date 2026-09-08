#!/usr/bin/env python3
"""Recheck exactly two failed K4 cases through the unchanged production capture."""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import sys
import time
import traceback
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import analyze_ffn_visual_source_consistency as base
from features.ffn_visual_path_attribution import streaming_vector_path_statistics as streaming
from features.jffn_experiment import mentions_for_image
from features.tc_fvpa_artifacts import atomic_json_save, atomic_torch_save, sha256_file, sha256_text
from models import build_model
from models.dgst_capture import resolve_decoder_layers
from scripts.run_ffn_visual_source_attribution import load_target_lookup, source_config
from scripts.run_jffn_p_comparison import _extract_one_image, _load_inputs, _image_path
from utils.config_utils import get_dgst_t_cfg, get_extraction_model_cfg

MODEL = 'qwen2_5_vl_7b'
CASES = ((248069, 17, 23), (546325, 30, 23))


def snapshot(stats, index):
    row = base.dual_net(stats.component_sum[index], stats.total_finite_effect[index], stats.gross_strength[index])
    for name, value in dict(p_ffn=stats.p_ffn[index], component_sum=stats.component_sum[index],
                            endpoint=stats.total_finite_effect[index], components=stats.components[:, index]).items():
        row[name] = value.detach().float().cpu()
    row['token_chunk_size'] = stats.token_chunk_size
    return row


def reproduction(value, previous):
    errors = {name: abs(value[name]-previous[name])/max(abs(previous[name]), 1e-12)
              for name in ('S', 'N_end', 'N_vec', 'kappa_vec')}
    errors['p_ffn_relative_l2'] = float(np.linalg.norm(np.asarray(value['p_ffn'])-np.asarray(previous['p_ffn'])) /
                                     max(np.linalg.norm(np.asarray(previous['p_ffn'])), 1e-12))
    absolute = abs(value['closure_relative_error']-previous['closure_relative_error'])
    return dict(passed=max(errors.values()) <= 1e-5 and absolute <= 1e-6,
                relative_errors=errors, closure_relative_error_absolute_difference=absolute)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    model_root = ROOT/'outputs'/MODEL/base.EXPERIMENT
    source = base.result_root(MODEL)/'production_k4/old/fp32_k4'
    output = base.result_root(MODEL)/'diagnostics/tail_k4_k64_20260908'
    output.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (output/'.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = json.loads((source/'manifest.json').read_text())
    validation = ROOT/'outputs'/base.VERSION/'old_validation_fp32_k4.json'
    if json.loads(validation.read_text())['status'] != 'FAIL_NUMERICAL_OLD_COHORT':
        raise ValueError('This diagnostic is scoped to the recorded failed full-cohort gate')
    for name, digest in manifest['implementation'].items():
        actual = sha256_file(name) if Path(name).is_absolute() else sha256_text(inspect.getsource(getattr(base, name)))
        if digest != actual:
            raise ValueError(f'Production implementation changed: {name}')
    for name, digest in manifest['inputs'].items():
        if sha256_file(name) != digest:
            raise ValueError(f'Production input changed: {name}')
    originals = [source/'manifest.json', validation]
    originals += [source/'shards'/f'image_{image_id:012d}{suffix}' for image_id, _, _ in CASES
                  for suffix in ('.pt', '.sha256.json')]
    preserved = {str(p): dict(sha256=sha256_file(p), mtime_ns=p.stat().st_mtime_ns) for p in originals}
    provenance = dict(cases=CASES, original_artifacts=preserved, implementation=manifest['implementation'],
                      script_sha256=sha256_file(__file__), conditions=[4, 64],
                      capture='unchanged production full-caption causal query-row batch',
                      precision='local FP32 Norm+FFN/path/JVP; native captured states/writes',
                      scope='diagnostic only; no production replacement, threshold change, or training')
    base.immutable_json(provenance, output/'manifest.json')
    fingerprint = sha256_file(output/'manifest.json')
    completed, pending = {}, []
    for case in CASES:
        path = output/f'case_{case[0]}_{case[1]}_{case[2]}.pt'
        if path.exists():
            if json.loads(path.with_suffix('.sha256.json').read_text()) != dict(sha256=sha256_file(path), fingerprint=fingerprint):
                raise ValueError(f'Invalid diagnostic resume: {path}')
            completed[case] = torch.load(path, map_location='cpu', weights_only=False)
        else:
            pending.append(case)
    if pending:
        config = manifest['config']
        labels, generations, _ = _load_inputs(model_root)
        lookup, _ = load_target_lookup(model_root, {c[0] for c in pending})
        wrapper = build_model(MODEL, get_extraction_model_cfg(config, MODEL), device=args.device)
        wrapper.model.requires_grad_(False).eval()
        layer_ids = {id(layer): i for i, layer in enumerate(resolve_decoder_layers(wrapper.model), 1)}
        base.immutable_json(dict(command=sys.argv, torch=torch.__version__, cuda=torch.version.cuda,
                                 device=args.device, gpu=torch.cuda.get_device_name(args.device)), output/'environment.json')
        for image_id, response_index, chosen_layer in pending:
            tick = time.perf_counter()
            path = source/'shards'/f'image_{image_id:012d}.pt'
            saved = torch.load(path, map_location='cpu', weights_only=False)
            if json.loads(path.with_suffix('.sha256.json').read_text()) != dict(sha256=sha256_file(path), fingerprint=manifest['fingerprint']):
                raise ValueError('Original shard checksum mismatch')
            if saved['image_sha256'] != sha256_file(_image_path(config, image_id)):
                raise ValueError('Original image bytes changed')
            previous = next(p for p in saved['positions'] if p['response_index'] == response_index)
            mentions, indices, _ = mentions_for_image(image_id=image_id, labeling_row=labels[image_id],
                                                     response_token_ids=generations[image_id]['response_token_ids'])
            target_index = indices.index(response_index)
            cfg = source_config(get_dgst_t_cfg(config), model=MODEL, k=4, chunk=manifest['chunk'], backend='vmap_jvp')
            cfg['jffn_vector_path_target_distributions'] = np.stack([lookup[(image_id, i)]['target'] for i in indices])
            cfg['jffn_vector_path_evidence_strengths'] = np.stack([lookup[(image_id, i)]['strength'] for i in indices])
            measured, seen, retries = {}, set(), []

            def adapted(**kwargs):
                layer = inspect.getclosurevars(kwargs['ffn_map']).nonlocals['layer']
                layer_number = layer_ids[id(layer)]
                try:
                    with base.local_fp32(layer) as function:
                        parameters = dict(kwargs, ffn_map=function, z=kwargs['z'].float(), writes=kwargs['writes'].float())
                        if layer_number != chosen_layer:
                            result = streaming(**parameters)
                        else:
                            conditions = {}
                            for k in (4, 64):
                                torch.cuda.synchronize(args.device)
                                torch.cuda.reset_peak_memory_stats(args.device)
                                start = time.perf_counter()
                                stats = streaming(**dict(parameters, integration_points=k, save_components=True))
                                torch.cuda.synchronize(args.device)
                                row = snapshot(stats, target_index)
                                row.update(elapsed_seconds=time.perf_counter()-start,
                                           peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device))
                                conditions[str(k)] = row
                                if k == 4:
                                    result = stats
                            measured.update(conditions=conditions, capture_dtype=str(kwargs['z'].dtype),
                                            z=kwargs['z'][target_index].detach().cpu(),
                                            writes=kwargs['writes'][:, target_index].detach().cpu(),
                                            target_batch_size=len(indices), target_batch_index=target_index)
                    seen.add(layer_number)
                    return result
                except torch.cuda.OutOfMemoryError:
                    retries.append(dict(layer=layer_number, token_chunk_size=kwargs['token_chunk_size']))
                    raise  # The unchanged production caller retries 256/128/64/32.

            with patch('features.ffn_visual_path_attribution.streaming_vector_path_statistics', adapted):
                _, returned_mentions = _extract_one_image(wrapper=wrapper, config=config, image_id=image_id,
                    label_row=labels[image_id], generation_row=generations[image_id], dgst_cfg=cfg, compact=False)
            if returned_mentions != mentions or seen != set(range(1, 29)) or set(measured['conditions']) != {'4', '64'}:
                raise ValueError('Changed original target/layer capture alignment')
            k4, k64 = (measured['conditions'][k] for k in ('4', '64'))
            original = dict(previous['dual_net'][chosen_layer-1], p_ffn=previous['p_ffn'][chosen_layer-1])
            check = reproduction(k4, original)
            payload = dict(measured, case=[image_id, response_index, chosen_layer], fingerprint=fingerprint,
                           k4_reproduction=check, comparison_k4_vs_k64=base.compare_measurement(k4, k64),
                           identical_endpoints=torch.equal(k4['endpoint'], k64['endpoint']),
                           original_metrics={k:v for k,v in original.items() if not torch.is_tensor(v)},
                           elapsed_seconds=time.perf_counter()-tick, oom_retries=retries)
            if not payload['identical_endpoints']:
                raise ValueError('K4/K64 must evaluate identical FP32 endpoints')
            dest = output/f'case_{image_id}_{response_index}_{chosen_layer}.pt'
            atomic_torch_save(payload, dest)
            base.immutable_json(dict(sha256=sha256_file(dest), fingerprint=fingerprint), dest.with_suffix('.sha256.json'))
            completed[(image_id, response_index, chosen_layer)] = payload
            print(json.dumps(dict(case=payload['case'], reproduction=check,
                  closure_errors={k:v['closure_relative_error'] for k,v in measured['conditions'].items()})), flush=True)
    rows = []
    for case in CASES:
        value = completed[case]
        rows.append({**{k:v for k,v in value.items() if k not in ('conditions','z','writes')},
                     'conditions': {k:{n:v for n,v in r.items() if not torch.is_tensor(v)} for k,r in value['conditions'].items()}})
    for name, before in preserved.items():
        p = Path(name)
        if before != dict(sha256=sha256_file(p), mtime_ns=p.stat().st_mtime_ns):
            raise ValueError('Original production result or gate changed')
    base.immutable_json(dict(status='COMPLETE_TARGETED_DIAGNOSTIC', cases=rows,
        all_k4_reproduced=all(r['k4_reproduction']['passed'] for r in rows),
        all_k64_closure_below_one_percent=all(r['conditions']['64']['closure_relative_error'] <= .01 for r in rows),
        original_artifacts_unchanged=True, original_full_gate='FAIL_NUMERICAL_OLD_COHORT',
        training_authorized=False), output/'summary.json')
    print('COMPLETE targeted diagnostic; original full gate remains FAIL; no training started', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        output = base.result_root(MODEL)/'diagnostics/tail_k4_k64_20260908'
        atomic_json_save(dict(status='FAIL_DIAGNOSTIC', command=sys.argv, traceback=traceback.format_exc()),
                         output/f'failure_{time.time_ns()}.json')
        raise
