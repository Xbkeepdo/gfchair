"""Four-model all-attention/B1/B2 experiment with audited per-image resume."""
import argparse
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from features.ffn_all_source_paths import (compute_target, partition, path_operator,
    direct_integral, rel, cosine)
from features.ffn_source_composition import norm_sources, integrated_gated_jvp
from features.ffn_visual_path_attribution import quadrature_rule
from features.visual_ffn_jacobian import reconstruct_visual_directions, resolve_decoder_layer_adapter
from features.tc_fvpa_artifacts import atomic_json_save, atomic_torch_save
from models.dgst_capture import resolve_decoder_layers, attention_row_from_capture
from scripts.run_ffn_source_composition import MODELS, parent_paths, load_wrapper, capture_full, read, EXPERIMENT
from scripts.run_ffn_visual_source_consistency import local_fp32
from scripts.run_jffn_p_comparison import _load_inputs, _image_path

OUT = ROOT/'outputs/ffn_all_source_paths_v1'
COHORT = ROOT/'outputs/ffn_output_cosine_20260909/cohort500.json'
COUNTS = dict(zip(MODELS, (28, 32, 36, 32)))
KS = (4, 8, 16, 32, 64)


def utc():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


class Progress:
    def __init__(self, model, out):
        self.path = out/model/'progress.json'
        self.value = dict(model=model, stage='准备', completed=0, total=1, status='running', pid=os.getpid())
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.update()
        self.thread = threading.Thread(target=self.beat, daemon=True)
        self.thread.start()

    def update(self, stage=None, completed=None, total=None, **values):
        with self.lock:
            for key, value in dict(stage=stage, completed=completed, total=total, **values).items():
                if value is not None: self.value[key] = value
            self.value['heartbeat'] = utc()
            atomic_json_save(self.value, self.path)

    def beat(self):
        while not self.stop.wait(10): self.update()

    def finish(self, status, error=None):
        self.stop.set()
        self.thread.join(timeout=11)
        self.update(status=status, error=error or '')


@contextmanager
def model_lock(model, out):
    folder = out/model
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/'.worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def protocol(out):
    files = [Path(__file__), ROOT/'features/ffn_all_source_paths.py', COHORT]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    p = dict(version=1, models=list(MODELS), hashes=hashes, precision='native capture; local FP32; TF32 off',
        b1='true RMS sinh-adaptive GL16/32 rtol1e-4 atol1e-7 maxdepth12; FP64 accumulation',
        partition='prompt includes template/BOS; visual interval; generated prefix only; no target/future',
        features='F0 F1/raw/log F2; B1 and B2 each F3/raw/log F4 F5 F6; length; 15 total; no combined B1+B2',
        detector='original 3200/800 all mentions; raw/no scaling; fixed TorchProbeConfig seeds43/44/45',
        zero_denominator='NaN in artifacts, valid counts in mechanism, zero in detector; no dropped mentions',
        mechanism='cohort500 original400/100; all/train/test exclude label-conflicting targets',
        bootstrap=dict(replicates=10000, seed=20260912, unit='paired image clusters'))
    p['signature'] = hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()
    target = out/'protocol.json'
    out.mkdir(parents=True, exist_ok=True)
    with (out/'.protocol.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if target.exists() and json.loads(target.read_text()) != p:
            raise ValueError('Protocol/code changed: use a new output directory or explicitly resolve old artifacts')
        if not target.exists(): atomic_json_save(p, target)
    return p


def selected_cases(model):
    train_ids = set(json.loads(COHORT.read_text())['split']['train'])
    matrix = read(ROOT/'outputs/ffn_source_composition_v1'/model/'matrices.pt')
    targets = {}
    labels = defaultdict(set)
    for m in matrix['mentions']:
        labels[m['target_key']].add(m['label'])
        targets[m['target_key']] = m
    rng = np.random.default_rng(20260912)
    chosen = []
    for label in (0, 1):
        pool = sorted((m for k, m in targets.items() if labels[k] == {label} and m['image_id'] in train_ids),
                      key=lambda m: (m['response_index'], m['target_key']))
        if len(pool) < 5: raise ValueError('Insufficient audit strata')
        for indices in np.array_split(np.arange(len(pool)), 5):
            chosen.append(pool[int(rng.choice(indices))])
    layers = np.linspace(0, COUNTS[model]-1, 5).round().astype(int).tolist()
    return [dict(image_id=m['image_id'], target_key=m['target_key'], response_index=m['response_index'],
                 label=m['label'], layer=layer) for m in chosen for layer in layers]


def parity_checks(norm, ffn, z, r, bias, writes, masks):
    def close(a, b):
        # Compare the attributed D-vectors. Coordinate-relative errors are
        # undefined near zero and can reject FP32 roundoff in a valid vector.
        error = (a.double()-b.double()).norm(dim=-1)
        tolerance = 1e-6+1e-4*b.double().norm(dim=-1)
        if not bool((error <= tolerance).all()):
            raise AssertionError(f'JVP vector parity failed: {float((error/tolerance).max())}')
    causal = masks.any(0)
    writes = writes[causal]
    a = torch.stack([writes[m[causal]].sum(0) for m in masks])
    total = writes.sum(0)
    rule = quadrature_rule('gauss_legendre', 4, device=z.device, dtype=z.dtype)
    fn = lambda x: ffn(norm(x))
    fast = path_operator(norm, ffn, z-total, total, rule.nodes, rule.weights)(writes)
    direct = direct_integral(fn, z-total, total, writes, rule.nodes, rule.weights)
    close(fast, direct)
    sources = torch.cat((r[None], a), 0)
    full = sources.sum(0)
    bfast = path_operator(norm, ffn, bias, full, rule.nodes, rule.weights)(sources)
    bdirect = direct_integral(fn, bias, full, sources, rule.nodes, rule.weights)
    close(bfast, bdirect)
    n = norm(z)
    directions = norm_sources(norm, z, torch.cat((sources, bias[None]), 0))
    cfast = integrated_gated_jvp(ffn, n, rule)(directions).double()
    cdirect = direct_integral(ffn, torch.zeros_like(n), n, directions, rule.nodes, rule.weights)
    close(cfast, cdirect)
    r64 = quadrature_rule('gauss_legendre', 64, device=z.device, dtype=z.dtype)
    op = path_operator(norm, ffn, z-total, total, r64.nodes, r64.weights)
    accumulated = torch.stack([op(writes[m[causal]]).sum(0) if m.any() else torch.zeros_like(z).double() for m in masks])
    grouped = op(a)
    close(accumulated, grouped)
    return dict(all=float(rel(fast.reshape(1, -1), direct.reshape(1, -1)).max()),
                b1=float(rel(bfast.reshape(1, -1), bdirect.reshape(1, -1)).max()),
                b2=float(rel(cfast.reshape(1, -1), cdirect.reshape(1, -1)).max()),
                groups=float(rel(accumulated.reshape(1, -1), grouped.reshape(1, -1)).max()))


def capture_image(wrapper, model, config, generation, parent):
    targets = sorted(parent['positions'], key=lambda x: x['response_index'])
    if not targets: return targets, None, [], 0, 0
    with Image.open(_image_path(config, targets[0]['image_id'])) as src: image = src.convert('RGB')
    captures, queries, start, end, _ = capture_full(wrapper, model, image, generation['response_token_ids'],
        targets, config.get('run', {}).get('prompt') or 'Describe this image.')
    return targets, captures, queries, start, end


def inputs_for(layer, cap, query, target, start, end):
    future = attention_row_from_capture(cap, query)[..., query+1:]
    if future.count_nonzero(): raise ValueError('Nonzero future attention')
    direct = reconstruct_visual_directions(layer=layer, capture=cap, prediction_positions=[query],
                                           visual_start=start, visual_end=end, return_all_sources=True)
    writes = direct['all_token_writes'].float()
    z, r = cap['h_mid'][0, query:query+1].float(), cap['h_prev'][0, query:query+1].float()
    adapter = resolve_decoder_layer_adapter(layer)
    bias = adapter.output_projection.bias
    bias = torch.zeros_like(z) if bias is None else bias.float()[None]
    masks = partition(len(writes), query, target['response_index'], (start, end), z.device)
    return z, r, bias, writes, masks


def audit(model, device, out, p, progress):
    cases = selected_cases(model)
    folder = out/model/'audit'
    folder.mkdir(parents=True, exist_ok=True)
    atomic_json_save(cases, folder/'selection.json')
    def path(c): return folder/(c['target_key'].replace(':', '_')+f"_L{c['layer']+1}.pt")
    pending = [c for c in cases if not valid(path(c), p['signature'])]
    progress.update('数值审计', len(cases)-len(pending), 50)
    if pending:
        parents = parent_paths(model)
        _, generations, _ = _load_inputs(ROOT/'outputs'/model/EXPERIMENT)
        wrapper, config = load_wrapper(model, device)
        by_image = defaultdict(list)
        for c in pending: by_image[c['image_id']].append(c)
        done = len(cases)-len(pending)
        for image_id, selected in by_image.items():
            parent = read(parents[image_id])
            targets, captures, queries, start, end = capture_image(wrapper, model, config, generations[image_id], parent)
            for c in selected:
                ti = next(i for i, t in enumerate(targets) if t['target_key'] == c['target_key'])
                layer, cap = resolve_decoder_layers(wrapper.model)[c['layer']], captures[c['layer']]
                z, r, bias, writes, masks = inputs_for(layer, cap, queries[ti], targets[ti], start, end)
                adapter = resolve_decoder_layer_adapter(layer)
                with local_fp32(layer), torch.no_grad():
                    parity = parity_checks(adapter.ffn_norm, adapter.ffn, z, r, bias, writes, masks)
                    rows = {}
                    for k in KS:
                        rows[k] = compute_target(adapter.ffn_norm, adapter.ffn, z, r, bias, writes, masks,
                                                 dict(all=k, visual=k, b2=k), audit=True)
                    source = torch.cat((r[None], torch.stack([writes[m].sum(0) for m in masks])), 0)
                    plain = {}
                    reference = rows[64]['vectors']['b1'].to(z.device)
                    endpoint = adapter.ffn(adapter.ffn_norm(bias+source.sum(0)))-adapter.ffn(adapter.ffn_norm(bias))
                    for k in (*KS, 128):
                        rule = quadrature_rule('gauss_legendre', k, device=z.device, dtype=z.dtype)
                        value = path_operator(adapter.ffn_norm, adapter.ffn, bias, source.sum(0), rule.nodes, rule.weights)(source)
                        plain[k] = dict(relative_to_sinh=rel(value, reference).cpu().tolist(),
                                        closure=float(rel(value.sum(0), endpoint).max()))
                atomic_torch_save(dict(case=c, results=rows, parity=parity, b1_plain=plain,
                    protocol_signature=p['signature'], complete=True), path(c))
                done += 1; progress.update('数值审计', done, 50)
                print(model, 'AUDIT', done, '/50', c, flush=True)
            del captures
        del wrapper; gc.collect(); torch.cuda.empty_cache()
    summary = dict(model=model, cases=50, protocol_signature=p['signature'], branches={})
    allcases = [read(path(c)) for c in cases]
    for branch in ('all', 'visual', 'b2'):
        result = {}
        for k in KS:
            errors, closure, cosdiff = [], [], []
            for case in allcases:
                x, y = case['results'][k], case['results'][64]
                a, b = x['vectors'][branch], y['vectors'][branch]
                lengths = b.double().norm(dim=-1)
                err = (a.double()-b.double()).norm(dim=-1)
                errors.append(float(torch.where(lengths > 1e-6, err/lengths, err/1e-4).max()))
                cname = {'all': 'all_closure_relative', 'visual': 'visual_closure_relative', 'b2': 'b2_quadrature_relative'}[branch]
                closure.append(x['diagnostics'][cname])
                if branch == 'all':
                    ca, cb = x['direction_cosine'].double(), y['direction_cosine'].double()
                    mask = torch.isfinite(ca) & torch.isfinite(cb)
                    if mask.any(): cosdiff.append(float((ca[mask]-cb[mask]).abs().max()))
            result[k] = dict(response_max=max(errors), closure_max=max(closure), cosine_max=max(cosdiff, default=0.))
            result[k]['pass'] = result[k]['response_max'] <= .01 and result[k]['closure_max'] <= .01 and result[k]['cosine_max'] <= .01
        summary['branches'][branch] = result
    summary['complete'] = True
    atomic_json_save(summary, out/model/'audit.json')
    return summary


def select_numerics(out, signature):
    audits = {m: json.loads((out/m/'audit.json').read_text()) for m in MODELS}
    if any(not a['complete'] or a['protocol_signature'] != signature for a in audits.values()):
        raise ValueError('Invalid/incompatible audits')
    selected = {}
    for branch in ('all', 'visual', 'b2'):
        passed = [k for k in KS if all(a['branches'][branch][str(k)]['pass'] for a in audits.values())]
        if not passed: raise RuntimeError(f'No quadrature passed for {branch}; formal extraction blocked')
        selected[branch] = passed[0]
    result = dict(ks=selected, protocol_signature=signature, b1='sinh adaptive GL16/32; frozen tolerances')
    atomic_json_save(result, out/'numerics.json')
    return selected


def valid(path, signature, layers=None):
    if not path.exists(): return False
    try:
        x = read(path)
    except (EOFError, RuntimeError, OSError): return False
    if not x.get('complete') or x.get('protocol_signature') != signature: return False
    if layers is not None:
        for row in x['positions']:
            if len(row['diagnostics']) != layers or len(row['tokens']) != layers: return False
            if not all(len(v) == layers for v in row['metrics'].values()): return False
    return True


def extract(model, device, out, p, progress, cohort, ks, chunk=64):
    parents = parent_paths(model)
    _, generations, splits = _load_inputs(ROOT/'outputs'/model/EXPERIMENT)
    cohort_split = json.loads(COHORT.read_text())['split'] if cohort == 500 else splits
    ids = sorted(cohort_split['train']+cohort_split['test'])
    if len(ids) != cohort or len(set(ids)) != cohort: raise ValueError('Wrong cohort')
    folder = out/model/'shards'
    def path(i): return folder/f'image_{i:012d}.pt'
    pending = [i for i in ids if not valid(path(i), p['signature'], COUNTS[model])]
    stage = f'{cohort}图提取'
    done = len(ids)-len(pending)
    progress.update(stage, done, cohort)
    if not pending: return
    wrapper, config = load_wrapper(model, device)
    for image_id in pending:
        tick = time.monotonic()
        parent = read(parents[image_id])
        targets, captures, queries, start, end = capture_image(wrapper, model, config, generations[image_id], parent)
        rows = [dict(target_key=t['target_key'], response_index=t['response_index'],
                     target_token_id=t['target_token_id'], metrics=defaultdict(list), diagnostics=[], tokens=[]) for t in targets]
        if targets:
            for li, (layer, cap) in enumerate(zip(resolve_decoder_layers(wrapper.model), captures)):
                adapter = resolve_decoder_layer_adapter(layer)
                # One target at a time is the conservative <=4 target block;
                # capture is shared, so no extra full model forward is needed.
                for ti, target in enumerate(targets):
                    z, r, bias, writes, masks = inputs_for(layer, cap, queries[ti], target, start, end)
                    for current_chunk in sorted(set((chunk, 32, 16)), reverse=True):
                        try:
                            with local_fp32(layer), torch.no_grad():
                                value = compute_target(adapter.ffn_norm, adapter.ffn, z, r, bias, writes, masks, ks, chunk=current_chunk)
                            break
                        except torch.cuda.OutOfMemoryError:
                            gc.collect(); torch.cuda.empty_cache()
                            if current_chunk == 16: raise
                    row = rows[ti]
                    for name, v in value.pop('metrics').items(): row['metrics'][name].append(v)
                    row['diagnostics'].append(value.pop('diagnostics'))
                    row['tokens'].append(value)
                    progress.update(stage, done, cohort, current_image=image_id, layer=li+1, target=ti+1)
                    del z, r, bias, writes, masks
                captures[li] = None
            del captures
        for row in rows: row['metrics'] = {k: torch.tensor(v, dtype=torch.float64) for k, v in row['metrics'].items()}
        atomic_torch_save(dict(image_id=image_id, positions=rows, sample_table=parent['sample_table'], complete=True,
            protocol_signature=p['signature'], elapsed=time.monotonic()-tick, ks=ks), path(image_id))
        done += 1; progress.update(stage, done, cohort)
        print(model, stage, done, '/', cohort, 'image', image_id, 'seconds', round(time.monotonic()-tick, 2), flush=True)
    del wrapper; gc.collect(); torch.cuda.empty_cache()


def pipeline(args, p, progress):
    model, out = args.model, args.output
    audit(model, args.device, out, p, progress)
    progress.update('等待四模型审计', 50, 50)
    while not all((out/m/'audit.json').exists() for m in MODELS): time.sleep(10)
    with (out/'.numerics.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ks = select_numerics(out, p['signature'])
    extract(model, args.device, out, p, progress, 500, ks, args.chunk)
    from scripts.analyze_all_source_paths import mechanism, train, bootstrap
    progress.update('500图机制分析', 0, 1)
    mechanism(model, out)
    extract(model, args.device, out, p, progress, 4000, ks, args.chunk)
    train(model, args.device, out, progress_callback=progress.update)
    progress.update('图片配对bootstrap', 0, 10000)
    bootstrap(model, out)
    progress.update('全部完成', 1, 1, epoch=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=('audit', 'select', 'extract', 'pipeline', 'mechanism', 'train', 'bootstrap', 'summarize'))
    parser.add_argument('--model', choices=MODELS)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--cohort', type=int, choices=(500, 4000), default=500)
    parser.add_argument('--chunk', type=int, default=64)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    p = protocol(args.output)
    if args.stage == 'select': select_numerics(args.output, p['signature']); return
    if args.stage == 'summarize':
        from scripts.analyze_all_source_paths import summarize
        summarize(args.output); return
    if not args.model: parser.error('--model required')
    with model_lock(args.model, args.output):
        progress = Progress(args.model, args.output)
        try:
            if args.stage == 'pipeline': pipeline(args, p, progress)
            elif args.stage == 'audit': audit(args.model, args.device, args.output, p, progress)
            elif args.stage == 'extract':
                ks = json.loads((args.output/'numerics.json').read_text())['ks']
                extract(args.model, args.device, args.output, p, progress, args.cohort, ks, args.chunk)
            else:
                from scripts import analyze_all_source_paths as analysis
                if args.stage == 'train': analysis.train(args.model, args.device, args.output, progress_callback=progress.update)
                else: getattr(analysis, args.stage)(args.model, args.output)
            progress.finish('completed')
        except BaseException as error:
            progress.finish('failed', str(error)[:240])
            traceback.print_exc()
            raise


if __name__ == '__main__': main()
