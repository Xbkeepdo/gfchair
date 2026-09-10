#!/usr/bin/env python3
"""Isolated two-model K4 path study; no historical numerical exceptions."""
from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

import numpy as np
import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from features.tc_fvpa_artifacts import atomic_json_save, atomic_torch_save, sha256_file
from features.jffn_experiment import mentions_for_image
from features.ffn_visual_path_attribution import streaming_vector_path_statistics
from models import build_model
from models.base_wrapper import ExtractionRequirements, AttentionRequirement
from scripts.run_ffn_visual_source_attribution import load_target_lookup, source_config, compact_position
from scripts.run_ffn_visual_source_consistency import local_fp32, dual_net
from scripts import analyze_ffn_visual_source_consistency as consistency
from scripts.run_jffn_p_comparison import _load_inputs, _image_path
from utils.config_utils import load_config, get_extraction_model_cfg, get_dgst_t_cfg

MODELS = ('minigpt4_7b', 'shikra_7b')
CONFIG = ROOT/'configs/model_configs_minigpt4_shikra_path.yaml'


def save_json(value, path):
    """Immutable JSON for identities; status files use atomic_json_save directly."""
    normalized = json.loads(json.dumps(value, sort_keys=True))
    if path.exists():
        if json.loads(path.read_text()) != normalized:
            raise ValueError(f'Changed protocol: {path}')
    else:
        atomic_json_save(normalized, path)


def prepare(args):
    root = ROOT/'outputs'/args.model/'COCO4000-JACOBIAN-PATH'
    if args.smoke:
        root = root/'smoke8'
    root.mkdir(parents=True, exist_ok=True)
    config = load_config(str(CONFIG))
    if args.smoke:
        config['dataset'].update(num_images=8, train_images=6, test_images=2, shared_split_path=None)
    path = root/'config.yaml'
    if path.exists():
        if yaml.safe_load(path.read_text()) != config:
            raise ValueError('Existing resolved configuration changed')
    else:
        path.write_text(yaml.safe_dump(config, sort_keys=False))
    return root, path, config


def command(script, args, root, config_path, extra=()):
    subprocess.run([sys.executable, str(ROOT/'scripts'/script), '--model', args.model,
        '--config', str(config_path), '--output-dir', str(root), '--device', args.device, *extra],
        cwd=ROOT, check=True)


def generation(args, root, config_path):
    command('generate_and_label.py', args, root, config_path,
            ('--generation-devices', args.device, '--resume', '--chair-cache',
             str(ROOT/'outputs/chair_cache/coco_val2014_chair.pkl')))


def baselines(args, root, config_path):
    # Includes the existing attention/gate reference required by the path method.
    command('extract_features.py', args, root, config_path,
            ('--feature-devices', args.device, '--resume'))


def extract(args, root, config):
    labels, generations, splits = _load_inputs(root)
    ids = sorted(splits['train'] + splits['test'])
    expected = 8 if args.smoke else 4000
    if len(ids) != expected or len(set(ids)) != expected or set(labels) != set(ids) or set(generations) != set(ids):
        raise ValueError('Incomplete image cohort')
    if set(splits['train']) & set(splits['test']):
        raise ValueError('Image leakage')
    out = root/'path'
    paths = [p for directory in ('models','features') for p in (ROOT/directory).rglob('*.py')]
    paths += [Path(__file__), ROOT/'scripts/run_ffn_visual_source_attribution.py',
              ROOT/'scripts/run_ffn_visual_source_consistency.py']
    manifest = dict(model=args.model, images=ids, precision='local-FP32', quadrature='gauss_legendre', k=4,
        chunk=args.chunk, smoke=args.smoke, configuration=config,
        source={str(p.relative_to(ROOT)):sha256_file(p) for p in paths},
        inputs={name:sha256_file(root/name) for name in ('features.pkl','generations.json','labeling.json','image_splits.json')})
    signature = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    save_json(dict(manifest, signature=signature), out/'manifest.json')
    pending = []
    for image_id in ids:
        path = out/'shards'/f'image_{image_id:012d}.pt'
        if path.exists():
            side = json.loads(path.with_suffix('.json').read_text())
            if side['signature'] != signature or side['sha256'] != sha256_file(path):
                raise ValueError(f'Invalid resumed shard {path}')
        else:
            pending.append(image_id)
    if pending:
        lookup, _ = load_target_lookup(root, set(pending))
        wrapper = build_model(args.model, get_extraction_model_cfg(config, args.model), device=args.device)
        wrapper.model.requires_grad_(False).eval()
        cfg = source_config(get_dgst_t_cfg(config), model=args.model, k=4, chunk=args.chunk, backend='vmap_jvp')
        requirements = ExtractionRequirements(attention=AttentionRequirement.HEAD_MEAN, logits=False,
            token_hidden_states=False, patch_hidden_states=False, response_hidden_states=False,
            visual_layout=True, dgst_capture=True)
        for number, image_id in enumerate(pending, 1):
            started = time.perf_counter()
            response_ids = generations[image_id]['response_token_ids']
            mentions, indices, targets = mentions_for_image(image_id=image_id,
                labeling_row=labels[image_id], response_token_ids=response_ids)
            dual_layers = []

            def statistics(**kwargs):
                layer = inspect.getclosurevars(kwargs['ffn_map']).nonlocals['layer']
                with local_fp32(layer) as function:
                    params = dict(kwargs, ffn_map=function, z=kwargs['z'].float(), writes=kwargs['writes'].float())
                    result = streaming_vector_path_statistics(**params)
                    reference = streaming_vector_path_statistics(**dict(params, integration_points=64)) if args.smoke else None
                rows = []
                for i in range(len(result.gross_strength)):
                    row = dual_net(result.component_sum[i], result.total_finite_effect[i], result.gross_strength[i])
                    if reference is not None:
                        ref = dual_net(reference.component_sum[i], reference.total_finite_effect[i], reference.gross_strength[i])
                        row.update(consistency.compare_measurement(dict(row,p_ffn=result.p_ffn[i].cpu()),
                                                                  dict(ref,p_ffn=reference.p_ffn[i].cpu())))
                        row['reference'] = ref
                    rows.append(row)
                dual_layers.append(rows)
                return result

            rows = []
            if indices:
                image_cfg = dict(cfg,
                    jffn_vector_path_target_distributions=np.stack([lookup[(image_id,i)]['target'] for i in indices]),
                    jffn_vector_path_evidence_strengths=np.stack([lookup[(image_id,i)]['strength'] for i in indices]))
                with Image.open(_image_path(config, image_id)) as image:
                    with patch('features.ffn_visual_path_attribution.streaming_vector_path_statistics', statistics):
                        outputs = wrapper.extract_token_features_batch(image=image.convert('RGB'),
                            response_token_ids=response_ids, response_token_indices=indices, target_token_ids=targets,
                            cfg_dgst_t=image_cfg, requirements=requirements, prompt=config['run']['prompt'])
                if len(dual_layers) != wrapper.num_layers or len(outputs) != len(indices):
                    raise ValueError('Incomplete targets/layers')
                for j, (index, target, output) in enumerate(zip(indices, targets, outputs)):
                    position = dict(image_id=image_id, response_index=index, target_token_id=target,
                                    visual_grid=output.visual_grid, result=output.dgst_t_result)
                    row = compact_position(position, lookup)
                    row['dual_net'] = [layer[j] for layer in dual_layers]
                    for name in ('S','N_end','N_vec','kappa_end','kappa_vec'):
                        row[name] = torch.tensor([v[name] for v in row['dual_net']], dtype=torch.float32)
                    row.update(I=row['write_mag'].sum(-1), AE=row['ae_strength'], R_cos=row['r_cos'])
                    row.update({name:row['ot'][name] for name in ('D_EW','D_WF','D_EF')})
                    row['raw_attention_mean'] = output.text_to_patch_attn.float().mean(1).cpu()
                    if output.spatial_attention is not None:
                        row['mapped_attention_mean'] = output.spatial_attention.float().mean(1).cpu()
                        row['mapped_grid'] = output.spatial_grid
                        torch.testing.assert_close(row['mapped_attention_mean'].sum(-1), row['raw_attention_mean'].sum(-1), rtol=1e-5, atol=1e-7)
                    row['visual_positions'] = list(range(output.baseline_capture['visual_start'], output.baseline_capture['visual_end']))
                    if not torch.isfinite(row['raw_attention_mean']).all() or (row['raw_attention_mean'] < 0).any():
                        raise ValueError('Invalid raw attention')
                    rows.append(row)
            path = out/'shards'/f'image_{image_id:012d}.pt'
            atomic_torch_save(dict(signature=signature, image_id=image_id, processed_image=True, positions=rows,
                sample_table=mentions, image_sha256=sha256_file(_image_path(config,image_id)),
                elapsed_seconds=time.perf_counter()-started), path)
            atomic_json_save(dict(signature=signature,sha256=sha256_file(path)),path.with_suffix('.json'))
            print(f'PATH {args.model} {number}/{len(pending)} image={image_id} targets={len(rows)} seconds={time.perf_counter()-started:.1f}',flush=True)
    if args.smoke:
        rows = [v for path in sorted((out/'shards').glob('*.pt'))
                for p in torch.load(path,map_location='cpu',weights_only=False)['positions'] for v in p['dual_net']]
        gate = consistency.gate_group(rows)
        atomic_json_save(dict(gate, model=args.model, thresholds=consistency.THRESHOLDS,
                             signature=signature, reference='local-FP32 K64'), out/'numerical_gate.json')
        if gate['status'] != 'PASS':
            raise RuntimeError(f'K4 numerical gate failed: {gate}')


def build_groups(positions, mentions, train_images):
    from features.ffn_target_consequence import q_features
    from scripts.analyze_ffn_target_consequence import feature_groups
    from scripts.train_q_softmax_js import q_softmax_js, build_groups as qjs_groups
    from scripts.analyze_ffn_endpoint_cosine_js import endpoint_cosine_js
    from scripts.train_endpoint_dot_js import dot_js
    from scripts.train_endpoint_projection_js import projection_js
    from scripts.train_endpoint_cosine_js_decomposition import endpoint_js_decomposition
    raw_keys = ('AE','I','S','N_vec','N_end','kappa_vec','kappa_end','R_cos','D_EW','D_WF','D_EF')
    derived = {}
    for key, row in positions.items():
        value = {name:np.asarray(row[name]) for name in raw_keys}
        value.update({k:np.asarray(v) for k,v in q_features(row['path_signed_q'], row['S'], row['net_degenerate']).items()})
        for name, fn in [('J_QT',lambda r:q_softmax_js(r['path_signed_q'],r['attention_evidence'])),
                         ('J_cosT',endpoint_cosine_js),('endpoint_dot_js',dot_js),('endpoint_projection_js',projection_js)]:
            value[name] = fn(row)[0]
        value.update(endpoint_js_decomposition(row))
        derived[key] = value
    raw = {name:np.stack([derived[m['target_key']][name] for m in mentions]) for name in next(iter(derived.values()))}
    scales = consistency.fit_scales(raw['S'],raw['I'],[m['image_id'] for m in mentions],train_images)
    groups = consistency.feature_sets({k:raw[k] for k in raw_keys},scales)
    groups.update(feature_groups(raw,False))
    groups.update(qjs_groups(raw['AE'],raw['S'],raw['kappa_vec'],raw['J_QT'],raw['J_cosT']))
    groups['AE+log1p(S)'] = groups['F']
    groups['AE+R_cos+S'] = np.concatenate([raw[k] for k in ('R_cos','AE','S')],axis=1)
    for name in ('endpoint_dot_js','endpoint_projection_js','J_T','J_E'):
        groups[name] = raw[name]
    groups['endpoint_cosine_js'] = raw['J_cosT']
    groups['J_T+J_E'] = np.concatenate([raw['J_T'],raw['J_E']],axis=1)
    return {k:np.asarray(v,dtype=np.float32) for k,v in groups.items()},scales


def train(args, root):
    from scripts.train_ffn_ae_log1p_search import run_head, fixed_mlp, summarize_group
    _, _, splits = _load_inputs(root)
    manifest=json.loads((root/'path/manifest.json').read_text())
    positions, mentions, checksums, processed, numerical = {}, [], {}, [], []
    for path in sorted((root/'path/shards').glob('*.pt')):
        side = json.loads(path.with_suffix('.json').read_text())
        if side['sha256'] != sha256_file(path):
            raise ValueError(f'Changed feature shard {path}')
        checksums[str(path)] = side['sha256']
        shard = torch.load(path,map_location='cpu',weights_only=False)
        if shard['signature'] != manifest['signature'] or not shard['processed_image']:
            raise ValueError('Shard belongs to another extraction protocol')
        processed.append(shard['image_id'])
        for row in shard['positions']:
            if row['target_key'] in positions: raise ValueError('Duplicate target')
            positions[row['target_key']] = row
            numerical.extend(row['dual_net'])
        mentions.extend(shard['sample_table'])
    if len(checksums) != len(splits['train'])+len(splits['test']):
        raise ValueError('Incomplete shards')
    if sorted(processed) != sorted(splits['train']+splits['test']):
        raise ValueError('Processed image IDs differ from the frozen cohort')
    closure=[r['closure_relative_error'] for r in numerical if not r['endpoint_degenerate']]
    audit=dict(images=len(processed),unique_targets=len(positions),mentions=len(mentions),
               closure_p90=float(np.quantile(closure,.9)),closure_max=float(max(closure)),
               identities=all(r['finite'] and r['kappa_range_pass'] and r['kappa_difference_bound_pass']
                              and r['closure_lower_bound_pass'] for r in numerical),
               degenerate_closed=all(r['closure_absolute_error']==0 for r in numerical if r['endpoint_degenerate']))
    audit['status']='PASS' if (audit['identities'] and audit['degenerate_closed'] and
        audit['closure_p90']<=consistency.THRESHOLDS['closure_p90'] and
        audit['closure_max']<=consistency.THRESHOLDS['closure_max']) else 'FAIL'
    atomic_json_save(audit,root/'path/full_numerical_audit.json')
    if audit['status']!='PASS': raise ValueError(f'Full-cohort numerical audit failed: {audit}')
    train_mentions = [m for m in mentions if m['image_id'] in splits['train']]
    test_mentions = [m for m in mentions if m['image_id'] in splits['test']]
    mentions = train_mentions+test_mentions
    if set(positions) != {m['target_key'] for m in mentions}: raise ValueError('Mention/target mismatch')
    groups,scales = build_groups(positions,mentions,splits['train'])
    labels = np.array([m['label'] for m in mentions],dtype=np.int32)
    n = len(train_mentions)
    if any(set(y.tolist()) != {0,1} for y in (labels[:n],labels[n:])):
        raise ValueError('Both classes required in train and test')
    hashes = {k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in groups.items()}
    protocol = dict(sources=checksums,groups=hashes,mentions=mentions,seeds=[43,44,45],
                    scales={k:v.tolist() for k,v in scales.items()},mlp=fixed_mlp(),
                    training_sources={str(p.relative_to(ROOT)):sha256_file(p) for p in (
                        Path(__file__),ROOT/'scripts/train_ffn_ae_log1p_search.py',
                        ROOT/'scripts/train_torch_probe_feature_sets.py',
                        ROOT/'scripts/analyze_ffn_visual_source_consistency.py',
                        ROOT/'scripts/analyze_ffn_target_consequence.py',
                        ROOT/'scripts/train_q_softmax_js.py',ROOT/'scripts/analyze_ffn_endpoint_cosine_js.py',
                        ROOT/'scripts/train_endpoint_dot_js.py',ROOT/'scripts/train_endpoint_projection_js.py',
                        ROOT/'scripts/train_endpoint_cosine_js_decomposition.py')})
    save_json(protocol,root/'path/training/protocol.json')
    signature=sha256_file(root/'path/training/protocol.json')
    summaries, reused = {}, {}
    for name,matrix in groups.items():
        digest=hashes[name]
        if digest in reused:
            summaries[name]=dict(summaries[reused[digest]],same_matrix_as=reused[digest])
            continue
        data=dict(X_train=matrix[:n],X_test=matrix[n:],y_train=labels[:n],y_test=labels[n:])
        heads=[run_head('three_hidden',fixed_mlp(),seed,data,root/'path/training/heads'/digest/f'seed{seed}',signature,args.device)
               for seed in (43,44,45)]
        summaries[name]=summarize_group(data,heads)
        reused[digest]=name
        atomic_json_save(summaries,root/'path/training/results.json')
    atomic_json_save(summaries,root/'path/training/results.json')
    lines=['# '+args.model+' — Jacobian path', '', 'Seeds 43/44/45: mean ± population std (ddof=0), no ensemble.', '', '| Feature | AUROC mean ± std | HALL AUPR mean ± std |', '|---|---:|---:|']
    for name,value in summaries.items():
        reports=[value['per_seed_metrics'][str(seed)]['threshold_reports']['fixed_0.5']['test_metrics'] for seed in (43,44,45)]
        auc=[r['auc'] for r in reports];ap=[r['hallucination_positive']['aupr'] for r in reports]
        lines.append(f"| {name} | {np.mean(auc):.6f} ± {np.std(auc):.6f} | {np.mean(ap):.6f} ± {np.std(ap):.6f} |")
    (root/'path/training/summary.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=MODELS,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--stage',choices=('generate','baselines','extract','train','pipeline'),default='pipeline')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--chunk',type=int,default=32)
    args=parser.parse_args()
    os.chdir(ROOT)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    os.environ.setdefault('NLTK_DATA','/home/apulis-dev/userdata/nltk_data')
    torch.set_num_threads(1)
    root,path,config=prepare(args)
    with (root/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if not args.smoke and args.stage in ('pipeline','extract','train'):
                gate=json.loads((root/'smoke8/path/numerical_gate.json').read_text())
                if gate['status']!='PASS': raise ValueError('New-model K4 smoke gate must pass first')
                parity=json.loads((root/'smoke8/wrapper_parity.json').read_text())
                if parity['status']!='PASS': raise ValueError('Wrapper causal/mean checks must pass first')
                smoke_manifest=json.loads((root/'smoke8/path/manifest.json').read_text())
                if gate['signature'] != smoke_manifest['signature']:
                    raise ValueError('Gate does not refer to the saved smoke manifest')
                for name,digest in smoke_manifest['source'].items():
                    if name != str(Path(__file__).relative_to(ROOT)) and sha256_file(ROOT/name)!=digest:
                        raise ValueError(f'Numerically validated implementation changed: {name}')
            if args.stage in ('generate','pipeline'): generation(args,root,path)
            if args.stage in ('baselines','pipeline'): baselines(args,root,path)
            if args.stage in ('extract','pipeline'): extract(args,root,config)
            if args.stage in ('train','pipeline') and not args.smoke:
                gc.collect()
                if torch.cuda.is_initialized(): torch.cuda.empty_cache()
                train(args,root)
                gc.collect()
                if torch.cuda.is_initialized(): torch.cuda.empty_cache()
                command('train_and_eval.py',args,root,path)
            status=('SMOKE_PASS' if args.smoke and args.stage=='pipeline' else
                    'COMPLETE_EXPERIMENT' if args.stage=='pipeline' else 'COMPLETE_STAGE')
            atomic_json_save(dict(status=status,stage=args.stage),root/'status.json')
        except Exception:
            atomic_json_save(dict(status='FAILED',stage=args.stage,traceback=traceback.format_exc()),root/'status.json')
            raise


if __name__=='__main__':
    main()
