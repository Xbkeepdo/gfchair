"""Forward-only, region-normalized attention times region-local hpre MAD gate."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from features.dgst_t import _gaussian_mad_gate, _renormalize
from features.tc_fvpa_artifacts import atomic_json_save, atomic_torch_save, sha256_file
from models import build_model
from models.dgst_capture import (run_forward_with_dgst_captures, attention_row_from_capture,
                                 resolve_output_embedding_layer, target_logits_and_probabilities_multi)
from scripts.extract_prefix_attention_gate import input_layout
from scripts.run_jffn_second_round_logit_causal import prepare_inputs
from scripts.run_jffn_p_comparison import _load_inputs, _image_path
from utils.config_utils import load_config, get_extraction_model_cfg

MODELS = ('qwen2_5_vl_7b', 'llava_1_5_7b', 'qwen3_vl_8b', 'internvl_2_5_8b')
LAYERS = dict(zip(MODELS, (28, 32, 36, 32)))
REGIONS = ('visual', 'visual_prompt_sum', 'generation')
OUT = ROOT/'outputs/all_attention_ae_log_strength_v1'
SMOKE_ATOL, SMOKE_RTOL = 1e-4, 1e-3


def read(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def region_ae(attention, logits, mask, epsilon=1e-6):
    """Normalize A inside the selected region, then apply that region's gate."""
    a, x = attention[mask].float(), logits[mask].float()
    if not len(a):
        return torch.tensor(float('nan'), device=attention.device)
    if not torch.isfinite(a).all() or not torch.isfinite(x).all() or (a < 0).any():
        raise ValueError('Nonfinite region inputs or negative attention')
    return (_renormalize(a)*_gaussian_mad_gate(x, epsilon=epsilon)).sum()


def _capture_prefix(wrapper, model, image, response, targets, prompt, epsilon, update):
    maximum = (len(response) if model in ('llava_1_5_7b', 'internvl_2_5_8b')
               else max(t['response_index'] for t in targets))
    inputs, last, start, end, _ = prepare_inputs(wrapper, model, image, response[:maximum], prompt)
    prompt_length = last+1-maximum
    queries = [prompt_length-1+t['response_index'] for t in targets]
    token_ids = [t['target_token_id'] for t in targets]
    with torch.no_grad():
        output, captures = run_forward_with_dgst_captures(wrapper.model, output_hidden_states=False,
            retain_attention_updates=False, attention_query_positions=queries, capture_device='cpu',
            attention_query_chunk_size=None, record_model_attentions=True, **inputs)
    del output
    length = captures[0]['h_prev'].shape[1]
    if length != last+1 or len(captures) != LAYERS[model] or end > min(queries)+1:
        raise ValueError('Invalid captured length/layers/visual range')
    layout = input_layout(wrapper, inputs, length, start, end, prompt_length)
    rows = [dict(t, ae={name: [] for name in REGIONS}, generation_count=t['response_index']) for t in targets]
    head = resolve_output_embedding_layer(wrapper.model)
    positions = torch.arange(length)
    masks = dict(visual=(positions >= start)&(positions < end), visual_prompt_sum=positions < prompt_length)
    with torch.no_grad():
        for layer, cap in enumerate(captures):
            for index, (row, query) in enumerate(zip(rows, queries)):
                attention = attention_row_from_capture(cap, query)
                if attention[:, query+1:].count_nonzero():
                    raise ValueError('Attention includes future positions')
                raw = attention.float().mean(0)
                active = dict(masks, generation=(positions >= prompt_length)&(positions <= query))
                if int(active['generation'].sum()) != row['response_index']:
                    raise ValueError('Generation prefix length mismatch')
                for name, mask in active.items():
                    if not mask.any():
                        row['ae'][name].append(float('nan'))
                        continue
                    # Preserve legacy native full-vocabulary projection and
                    # region-only chunk shape. Each region has its own MAD.
                    logits, probabilities = target_logits_and_probabilities_multi(
                        output_layer=head, states=cap['h_prev'][0][mask].float(),
                        target_token_ids=[token_ids[index]], chunk_size=64)
                    selected = raw[mask]
                    row['ae'][name].append(float(region_ae(selected, logits[:, 0],
                        torch.ones_like(selected, dtype=torch.bool), epsilon)))
                    del logits, probabilities
            captures[layer] = None
            del cap
            update(layer=layer+1)
    for row in rows:
        row['ae'] = {name: torch.tensor(values, dtype=torch.float32) for name, values in row['ae'].items()}
    return rows, dict(visual_range=[start, end], prompt_length=prompt_length,
                      expanded_length=length, explicit_bos_at_zero=layout['explicit_bos_at_zero'])


def capture(wrapper, model, image, response, targets, prompt, epsilon, update):
    # Match each original wrapper's execution shape and native eager attention.
    # Query chunking changes attention rounding enough to fail legacy parity.
    if model in ('llava_1_5_7b', 'internvl_2_5_8b'):
        return _capture_prefix(wrapper, model, image, response, targets, prompt, epsilon, update)
    rows, layout = [], None
    for index, target in enumerate(targets, 1):
        update(target=index, targets=len(targets))
        current, current_layout = _capture_prefix(wrapper, model, image, response, [target], prompt, epsilon, update)
        if layout is not None and any(layout[k] != current_layout[k] for k in ('visual_range', 'prompt_length')):
            raise ValueError('Target prefixes have different visual/prompt layout')
        layout = current_layout
        rows.extend(current)
    return rows, layout


def check_shard(shard, image_id, signature, expected, layers):
    if shard.get('complete') is not True or shard.get('image_id') != image_id or shard.get('signature') != signature:
        raise ValueError('Incomplete or incompatible AE shard')
    if shard['sample_table'] != expected:
        raise ValueError('AE shard does not preserve canonical mentions')
    targets = {m['target_key']: m for m in expected}
    if len(shard['positions']) != len(targets) or {r['target_key'] for r in shard['positions']} != set(targets):
        raise ValueError('AE target coverage mismatch')
    for row in shard['positions']:
        if any(row[k] != targets[row['target_key']][k] for k in ('response_index', 'target_token_id')):
            raise ValueError('Changed target metadata')
        if set(row['ae']) != set(REGIONS):
            raise ValueError('Missing AE region')
        for name, value in row['ae'].items():
            if value.shape != (layers,):
                raise ValueError('Missing AE decoder layers')
            empty = name == 'generation' and row['response_index'] == 0
            if empty:
                if not torch.isnan(value).all():
                    raise ValueError('Empty G must be NaN')
            elif not torch.isfinite(value).all() or ((value < 0)|(value > 1)).any():
                raise ValueError('Nonfinite or invalid AE')


def extract(model, device, smoke=False):
    torch.set_num_threads(1)
    source = ROOT/'outputs'/model/'COCO4000-INSLEN-OFFICIAL-TARGET'
    config_path = ROOT/'configs/model_configs_inslen_official_target.yaml'
    config = load_config(str(config_path))
    labels, generations, split = _load_inputs(source)
    ids = sorted(split['train']+split['test'])
    if len(ids) != 4000 or len(set(ids)) != 4000 or len(split['train']) != 3200 or len(split['test']) != 800:
        raise ValueError('Original image split is incomplete')
    old = read(ROOT/'outputs/ffn_source_composition_v1'/model/'matrices.pt')
    canonical = defaultdict(list)
    old_ae = {}
    for i, mention in enumerate(old['mentions']):
        canonical[int(mention['image_id'])].append(mention)
        if smoke:
            value = np.asarray(old['groups']['F'][i, :LAYERS[model]])
            if mention['target_key'] in old_ae:
                np.testing.assert_array_equal(value, old_ae[mention['target_key']])
            old_ae[mention['target_key']] = value
    del old
    protocol = dict(model=model, config=config, regions=list(REGIONS),
        formula='sum_region(normalize_region(mean_heads(attention)) * sigmoid((raw_hpre_target_logit-region_median)/(1.4826*region_MAD+epsilon)))',
        vp='All visual and all prompt tokens including BOS/template; excludes generation',
        logits='Native full-vocabulary projection on each region states, chunk64, target raw column; no final norm or LM-head bias',
        forward='Native eager attention; Qwen exact prefix per target; LLaVA and InternVL full response causal rows, matching original wrappers',
        empty_generation='NaN', smoke_atol=SMOKE_ATOL, smoke_rtol=SMOKE_RTOL,
        source_sha={name: sha256_file(source/name) for name in ('generations.json', 'labeling.json', 'image_splits.json')},
        code_sha={str(p.relative_to(ROOT)): sha256_file(p) for p in
                  (Path(__file__), ROOT/'features/dgst_t.py', ROOT/'models/dgst_capture.py',
                   ROOT/'scripts/extract_prefix_attention_gate.py', ROOT/'scripts/run_jffn_second_round_logit_causal.py')})
    signature = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    base = OUT/model
    directory = base/'smoke8' if smoke else base
    directory.mkdir(parents=True, exist_ok=True)
    progress = dict(model=model, stage='smoke AE' if smoke else 'extract AE', completed=0,
                    total=8 if smoke else 4000, status='running')
    def update(**values):
        progress.update(values, heartbeat=datetime.now(timezone.utc).isoformat())
        atomic_json_save(progress, directory/'extraction_progress.json')
    with (base/'.ae_extract.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        update()
        wrapper = None
        try:
            manifest = directory/'ae_protocol.json'
            if manifest.exists() and json.loads(manifest.read_text()).get('signature') != signature:
                raise ValueError('AE extraction protocol changed')
            atomic_json_save(dict(protocol, signature=signature), manifest)
            if not smoke:
                passed = json.loads((base/'smoke8/status.json').read_text())
                if passed.get('status') != 'PASS' or passed.get('signature') != signature:
                    raise ValueError('Matching 8-image visual-AE smoke must pass before formal extraction')
            selected = ids[:8] if smoke else ids
            epsilon = float(config['feature_extraction']['dgst_t'].get('relative_vll_mad_epsilon', 1e-6))
            errors, relatives, smoke_failures = [], [], 0
            for index, image_id in enumerate(selected, 1):
                path = directory/'ae_shards'/f'image_{image_id:012d}.pt'
                expected = canonical[image_id]
                update(completed=index-1, current_image=image_id)
                if path.exists():
                    shard = read(path)
                else:
                    targets = {m['target_key']: {k: m[k] for k in ('target_key', 'response_index', 'target_token_id')} for m in expected}
                    targets = sorted(targets.values(), key=lambda x: x['response_index'])
                    response = generations[image_id]['response_token_ids']
                    if any(response[t['response_index']] != t['target_token_id'] for t in targets):
                        raise ValueError('Original generation token mismatch')
                    if targets:
                        if wrapper is None:
                            wrapper = build_model(model, get_extraction_model_cfg(config, model), device=device)
                            wrapper.model.eval().requires_grad_(False)
                        with Image.open(_image_path(config, image_id)) as image:
                            rows, layout = capture(wrapper, model, image.convert('RGB'), response, targets,
                                                   config['run']['prompt'], epsilon, update)
                    else:
                        rows, layout = [], None
                    shard = dict(complete=True, image_id=image_id, positions=rows, sample_table=expected,
                                 signature=signature, layout=layout)
                    check_shard(shard, image_id, signature, expected, LAYERS[model])
                    atomic_torch_save(shard, path)
                check_shard(shard, image_id, signature, expected, LAYERS[model])
                if smoke:
                    for row in shard['positions']:
                        old_value = old_ae[row['target_key']]
                        error = np.abs(np.asarray(row['ae']['visual'])-old_value)
                        errors.extend(error.tolist())
                        relatives.extend((error/np.maximum(np.abs(old_value), 1e-12)).tolist())
                        smoke_failures += int((error > SMOKE_ATOL+SMOKE_RTOL*np.abs(old_value)).sum())
                update(completed=index)
                print(model, progress['stage'], index, '/', len(selected), image_id, flush=True)
            result = dict(status='PASS' if smoke else 'COMPLETE', signature=signature, images=len(selected),
                          mentions=sum(len(canonical[i]) for i in selected), layers=LAYERS[model])
            if smoke:
                if not errors:
                    raise ValueError('Smoke has no comparable visual target-layers')
                result['visual_parity'] = dict(target_layers=len(errors), max_absolute=max(errors),
                    p90_absolute=float(np.quantile(errors, .9)), max_relative=max(relatives),
                    atol=SMOKE_ATOL, rtol=SMOKE_RTOL, failed_target_layers=smoke_failures)
                if smoke_failures:
                    result['status'] = 'FAIL'
            atomic_json_save(result, directory/'status.json')
            if result['status'] == 'FAIL':
                raise RuntimeError('Visual AE parity failed: '+json.dumps(result['visual_parity']))
            update(status='completed', error='')
            return result
        except BaseException as error:
            update(status='failed', error=str(error)[:300])
            raise
        finally:
            del wrapper
            gc.collect()
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    print(json.dumps(extract(args.model, args.device, args.smoke), ensure_ascii=False), flush=True)
