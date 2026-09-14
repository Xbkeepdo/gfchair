"""Extract S_V^all for MiniGPT-4/Shikra on the true-RMS all-attention path."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from features.ffn_all_source_paths import path_operator
from features.ffn_visual_path_attribution import quadrature_rule
from features.tc_fvpa_artifacts import atomic_json_save, atomic_torch_save, sha256_file
from features.visual_ffn_jacobian import resolve_decoder_layer_adapter
from features.jffn_experiment import mentions_for_image
from models import build_model
from models.base_wrapper import ExtractionRequirements, AttentionRequirement
from scripts.run_ffn_visual_source_attribution import load_target_lookup, source_config, compact_position
from scripts.run_ffn_visual_source_consistency import local_fp32
from scripts.run_jffn_p_comparison import _load_inputs, _image_path
from utils.config_utils import load_config, get_extraction_model_cfg, get_dgst_t_cfg

MODELS = ('minigpt4_7b', 'shikra_7b')
LAYERS = {'minigpt4_7b': 32, 'shikra_7b': 32}
CONFIG = ROOT/'configs/model_configs_minigpt4_shikra_path.yaml'
OUT = ROOT/'outputs/minigpt4_shikra_all_attention_v_k32_v1'
K, REFERENCE_K = 32, 64


def read(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def source_root(model):
    return ROOT/'outputs'/model/'COCO4000-JACOBIAN-PATH'


def progress(model, stage, completed, total, status='running', **values):
    atomic_json_save(dict(stage=stage, completed=completed, total=total, status=status,
        heartbeat=datetime.now(timezone.utc).isoformat(), **values), OUT/model/'progress.json')


def normalize(value, eps=1e-12):
    total = value.sum(-1, keepdim=True)
    return torch.where(total > eps, value/total.clamp_min(eps), torch.full_like(value, 1/value.shape[-1]))


def path_stats(layer, z, visual_writes, aggregate, k, chunk):
    """Integrate visual token directions along z-A_all+alpha*A_all."""
    adapter = resolve_decoder_layer_adapter(layer)
    rule = quadrature_rule('gauss_legendre', k, device=z.device, dtype=z.dtype)
    operator = path_operator(adapter.ffn_norm, adapter.ffn, z-aggregate, aggregate, rule.nodes, rule.weights)
    components = []
    for start in range(0, len(visual_writes), chunk):
        components.append(operator(visual_writes[start:start+chunk]))
    components = torch.cat(components).float()
    gross = components.norm(dim=-1).transpose(0, 1)
    gross_strength = gross.sum(-1)
    visual_sum = components.sum(0)
    net_strength = visual_sum.norm(dim=-1)
    unit = visual_sum/net_strength.clamp_min(1e-12).unsqueeze(-1)
    signed = torch.einsum('mtd,td->tm', components, unit)
    with torch.no_grad():
        endpoint = adapter.ffn(adapter.ffn_norm(z))-adapter.ffn(adapter.ffn_norm(z-aggregate))
    all_sum = operator(aggregate.unsqueeze(0))[0].float()
    closure = (all_sum-endpoint.float()).norm(dim=-1)/endpoint.float().norm(dim=-1).clamp_min(1e-12)
    return SimpleNamespace(
        ffn_path_gross=gross, p_ffn=normalize(gross), path_signed_q=signed,
        gross_strength=gross_strength, net_strength=net_strength,
        kappa=net_strength/gross_strength.clamp_min(1e-12),
        total_finite_effect=endpoint.float(), component_sum=visual_sum,
        completeness_relative_error=closure,
        gross_degenerate=gross_strength <= 1e-12, net_degenerate=net_strength <= 1e-12,
        quadrature=rule, token_chunk_size=chunk, jvp_backend='factored_true_rms', components=None)


@contextmanager
def all_attention_patch(audit_rows, chunk):
    import features.visual_ffn_jacobian as visual
    import features.ffn_visual_path_attribution as path_module
    original_reconstruct = visual.reconstruct_visual_directions
    queued = []

    def reconstruct(**kwargs):
        kwargs['return_all_sources'] = True
        value = original_reconstruct(**kwargs)
        queued.append(value['all_token_writes'].float().sum(0))
        return value

    def statistics(**kwargs):
        if not queued:
            raise RuntimeError('Missing all-attention aggregate')
        aggregate = queued.pop(0)
        layer = inspect.getclosurevars(kwargs['ffn_map']).nonlocals['layer']
        z, writes = kwargs['z'].float(), kwargs['writes'].float()
        with local_fp32(layer):
            result = path_stats(layer, z, writes, aggregate, K, chunk)
            if audit_rows is not None:
                reference = path_stats(layer, z, writes, aggregate, REFERENCE_K, chunk)
                absolute = (result.gross_strength-reference.gross_strength).abs()
                relative = absolute/reference.gross_strength.clamp_min(1e-12)
                audit_rows.extend(dict(
                    strength_relative=float(r), strength_absolute=float(a),
                    k32_closure=float(c32), k64_closure=float(c64))
                    for r, a, c32, c64 in zip(relative.cpu(), absolute.cpu(),
                        result.completeness_relative_error.cpu(), reference.completeness_relative_error.cpu()))
        return result

    with patch.object(visual, 'reconstruct_visual_directions', reconstruct), \
         patch.object(path_module, 'streaming_vector_path_statistics', statistics):
        yield queued
    if queued:
        raise RuntimeError(f'Unused all-attention aggregates: {len(queued)}')


def protocol(model):
    root = source_root(model)
    source_manifest = json.loads((root/'path/manifest.json').read_text())
    value = dict(schema='minigpt-shikra-all-attention-v-k32-v1', model=model,
        images=4000, layers=LAYERS[model], precision='native capture; local FP32; TF32 off; FP64 quadrature accumulation',
        path='z-A_all to z through FFN(RMSNorm(.)); output projection bias remains in baseline',
        statistic='S_V_all=sum over visual tokens of norm(integrated vector JVP)',
        quadrature='Gauss-Legendre K32; K64 8-image audit reference', token_chunk=64,
        thresholds=dict(strength_relative_max=.01, closure_max=.01, near_zero_absolute=1e-6),
        source_manifest_signature=source_manifest['signature'],
        source_files={name:sha256_file(root/name) for name in ('generations.json','labeling.json','image_splits.json')},
        code_sha={str(p.relative_to(ROOT)):sha256_file(p) for p in (Path(__file__),
            ROOT/'features/ffn_all_source_paths.py', ROOT/'features/visual_ffn_jacobian.py')})
    value['signature'] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    path = OUT/model/'protocol.json'
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError('Extraction protocol changed')
    atomic_json_save(value, path)
    return value


def valid(path, signature, layers):
    if not path.exists(): return False
    try: value = read(path)
    except (OSError, EOFError, RuntimeError): return False
    if not value.get('complete') or value.get('signature') != signature: return False
    return all(row['S_V_all'].shape == (layers,) and torch.isfinite(row['S_V_all']).all()
               and (row['S_V_all'] >= 0).all() for row in value['positions'])


def extract(model, device, smoke=False, chunk=64):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    p = protocol(model); base = OUT/model; base.mkdir(parents=True, exist_ok=True)
    root = source_root(model)
    labels, generations, split = _load_inputs(root)
    ids = sorted(split['train']+split['test'])
    if len(ids) != 4000 or len(set(ids)) != 4000: raise ValueError('Incomplete source cohort')
    selected = ids[:8] if smoke else ids
    folder = base/('smoke8/shards' if smoke else 'shards')
    pending = [i for i in selected if not valid(folder/f'image_{i:012d}.pt', p['signature'], LAYERS[model])]
    if not smoke:
        gate = json.loads((base/'smoke8/audit.json').read_text())
        if gate['status'] != 'PASS' or gate['signature'] != p['signature']:
            raise ValueError('Matching K32/K64 smoke audit must pass first')
    with (base/('.smoke.lock' if smoke else '.extract.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        wrapper = None
        try:
            lookup, _ = load_target_lookup(root, set(pending)) if pending else ({}, None)
            wrapper = build_model(model, get_extraction_model_cfg(load_config(str(CONFIG)), model), device=device) if pending else None
            if wrapper is not None: wrapper.model.requires_grad_(False).eval()
            cfg0 = source_config(get_dgst_t_cfg(load_config(str(CONFIG))), model=model, k=K, chunk=chunk, backend='vmap_jvp')
            requirements = ExtractionRequirements(attention=AttentionRequirement.HEAD_MEAN, logits=False,
                token_hidden_states=False, patch_hidden_states=False, response_hidden_states=False,
                visual_layout=True, dgst_capture=True)
            audit_rows = [] if smoke else None
            done = len(selected)-len(pending)
            for image_id in pending:
                tick = time.monotonic(); response = generations[image_id]['response_token_ids']
                audit_start = len(audit_rows) if audit_rows is not None else 0
                mentions, indices, targets = mentions_for_image(image_id=image_id,
                    labeling_row=labels[image_id], response_token_ids=response)
                cfg = dict(cfg0,
                    jffn_vector_path_target_distributions=np.stack([lookup[(image_id,i)]['target'] for i in indices]),
                    jffn_vector_path_evidence_strengths=np.stack([lookup[(image_id,i)]['strength'] for i in indices])) if indices else cfg0
                rows = []
                if indices:
                    with Image.open(_image_path(load_config(str(CONFIG)), image_id)) as image:
                        with all_attention_patch(audit_rows, chunk):
                            outputs = wrapper.extract_token_features_batch(image=image.convert('RGB'), response_token_ids=response,
                                response_token_indices=indices, target_token_ids=targets, cfg_dgst_t=cfg,
                                requirements=requirements, prompt=load_config(str(CONFIG))['run']['prompt'])
                    for index, target, output in zip(indices, targets, outputs):
                        compact = compact_position(dict(image_id=image_id, response_index=index,
                            target_token_id=target, visual_grid=output.visual_grid, result=output.dgst_t_result), lookup)
                        rows.append(dict(target_key=compact['target_key'], response_index=index,
                            target_token_id=target, S_V_all=compact['gross_strength'].float()))
                payload = dict(complete=True, signature=p['signature'], image_id=image_id,
                    positions=rows, sample_table=mentions, seconds=time.monotonic()-tick,
                    numerical_audit=(audit_rows[audit_start:] if audit_rows is not None else None))
                atomic_torch_save(payload, folder/f'image_{image_id:012d}.pt')
                done += 1; progress(model, 'K32/K64数值审计' if smoke else 'all-attention K32 V提取', done, len(selected))
                print(model, 'smoke' if smoke else 'extract', done, '/', len(selected), image_id,
                      'targets', len(rows), 'seconds', round(time.monotonic()-tick, 2), flush=True)
            if smoke:
                persisted = [row for image_id in selected
                    for row in read(folder/f'image_{image_id:012d}.pt').get('numerical_audit', [])]
                relative = [r['strength_relative'] for r in persisted]
                absolute = [r['strength_absolute'] for r in persisted]
                c32 = [r['k32_closure'] for r in persisted]; c64 = [r['k64_closure'] for r in persisted]
                failures = sum(r > .01 and a > 1e-6 for r,a in zip(relative,absolute))
                result = dict(status='PASS' if persisted and failures == 0 and max(c32) <= .01 else 'FAIL',
                    signature=p['signature'], images=8, target_layers=len(persisted), failures=failures,
                    strength_relative_max=max(relative, default=float('nan')),
                    strength_relative_p90=float(np.quantile(relative,.9)) if relative else float('nan'),
                    strength_absolute_max=max(absolute, default=float('nan')),
                    k32_closure_max=max(c32, default=float('nan')), k64_closure_max=max(c64, default=float('nan')))
                atomic_json_save(result, base/'smoke8/audit.json')
                progress(model, 'K32/K64数值审计', 8, 8, status='completed' if result['status']=='PASS' else 'failed')
                if result['status'] != 'PASS': raise RuntimeError(f'K32 audit failed: {result}')
            elif len(list(folder.glob('image_*.pt'))) == 4000:
                progress(model, 'all-attention K32 V提取完成', 4000, 4000, status='completed')
        except BaseException as exc:
            progress(model, 'K32/K64数值审计' if smoke else 'all-attention K32 V提取', 0, len(selected), status='failed', error=str(exc)[:180])
            raise
        finally:
            del wrapper; gc.collect()
            if torch.cuda.is_initialized(): torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--chunk', type=int, default=64)
    args = parser.parse_args()
    extract(args.model, args.device, args.smoke, args.chunk)
