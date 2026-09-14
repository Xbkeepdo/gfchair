"""811 single-hidden MLP search for MiniGPT-4/Shikra visual path features."""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from datetime import datetime, timezone
import csv
import fcntl
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from detection.baselines import baseline_vector
from features.baseline import baseline_config, get_baseline_payload, svar_training_vector
from features.tc_fvpa_artifacts import atomic_json_save, atomic_torch_save, sha256_file
from scripts import search_single_mlp_811 as mlp
from scripts import train_native_baselines_811 as native
from utils.config_utils import load_config
from utils.io_utils import load_pkl

MODELS = ('minigpt4_7b', 'shikra_7b')
SOURCE_MODE = 'visual_k4'
OUT = ROOT/'outputs/single_mlp_visual_811_minigpt_shikra_v1'
SPLIT_SEED = 20260912
SEEDS = (43, 44, 45)
HEADS = tuple(native.HEADS)


def source_root(model):
    return ROOT/'outputs'/model/'COCO4000-JACOBIAN-PATH'


def configure(source):
    global SOURCE_MODE, OUT
    SOURCE_MODE = source
    OUT = (ROOT/'outputs/single_mlp_visual_811_minigpt_shikra_v1' if source == 'visual_k4'
           else ROOT/'outputs/single_mlp_all_attention_v_811_minigpt_shikra_v1')


def progress(model, stage, completed, total, status='running', **values):
    atomic_json_save(dict(stage=stage, completed=completed, total=total, status=status,
        heartbeat=datetime.now(timezone.utc).isoformat(), **values), OUT/model/'progress.json')


def split_images(old):
    rng = np.random.default_rng(SPLIT_SEED)
    rest = rng.permutation(sorted(old['test'])).tolist()
    value = dict(train=sorted(old['train']), validation=rest[:400], test=rest[400:])
    assert [len(value[k]) for k in value] == [3200, 400, 400]
    assert not (set(value['train']) & set(value['validation']))
    assert not (set(value['train']) & set(value['test']))
    assert not (set(value['validation']) & set(value['test']))
    return value


def mention_key(row, baseline=False):
    return (int(row['image_id']), int(row['response_token_idx' if baseline else 'response_index']),
            int(row['target_token_id']), int(row['label']), str(row['token_str' if baseline else 'word']))


def align_baselines(records, mentions):
    queues = defaultdict(deque)
    for row in records:
        if row.get('metadata', {}).get('svar_protocol') != 'controlled':
            raise ValueError('Expected controlled baseline record')
        queues[mention_key(row, True)].append(row)
    ordered = []
    for mention in mentions:
        key = mention_key(mention)
        if not queues[key]:
            raise ValueError(f'Missing baseline row: {key}')
        ordered.append(queues[key].popleft())
    if any(queues.values()):
        raise ValueError('Extra baseline rows')
    return ordered


def prepare(model):
    root = source_root(model)
    old_split = json.loads((root/'image_splits.json').read_text())
    split = split_images(old_split)
    old_protocol = json.loads((root/'path/training/protocol.json').read_text())
    mentions = old_protocol['mentions']
    positions = {}
    for shard_path in sorted((root/'path/shards').glob('*.pt')):
        shard = torch.load(shard_path, map_location='cpu', weights_only=False)
        for row in shard['positions']:
            if row['target_key'] in positions:
                raise ValueError('Duplicate target position')
            positions[row['target_key']] = row
    if len(positions) != len({m['target_key'] for m in mentions}):
        raise ValueError('Incomplete visual path targets')
    ae = np.stack([np.asarray(positions[m['target_key']]['AE'], dtype=np.float32) for m in mentions])
    if SOURCE_MODE == 'visual_k4':
        strength = np.stack([np.asarray(positions[m['target_key']]['S'], dtype=np.float32) for m in mentions])
        strength_protocol = 'visual-only z-A_V to z; local FP32 Gauss-Legendre K4; not all-attention K32'
        strength_source = {}
    else:
        strict_root = ROOT/'outputs/minigpt4_shikra_all_attention_v_k32_v1'/model
        strict = {}
        strict_images = set()
        strict_protocol = json.loads((strict_root/'protocol.json').read_text())
        for shard_path in sorted((strict_root/'shards').glob('*.pt')):
            shard = torch.load(shard_path, map_location='cpu', weights_only=False)
            if (not shard.get('complete') or shard.get('signature') != strict_protocol['signature']
                    or shard['image_id'] in strict_images):
                raise ValueError('Incomplete or duplicate strict all-attention shard')
            strict_images.add(shard['image_id'])
            for row in shard['positions']:
                if row['target_key'] in strict: raise ValueError('Duplicate strict target')
                strict[row['target_key']] = row
        if len(strict_images) != 4000 or set(strict) != {m['target_key'] for m in mentions}:
            raise ValueError('Strict all-attention K32 extraction is incomplete')
        strict_audit = json.loads((strict_root/'smoke8/audit.json').read_text())
        if strict_audit['status'] != 'PASS' or strict_audit['signature'] != strict_protocol['signature']:
            raise ValueError('Strict K32/K64 audit did not pass')
        strength = np.stack([np.asarray(strict[m['target_key']]['S_V_all'], dtype=np.float32) for m in mentions])
        strength_protocol = 'true-RMS all-attention z-A_all to z; local FP32 factored Gauss-Legendre K32'
        strength_source = dict(source_mode=SOURCE_MODE, strict_protocol_signature=strict_protocol['signature'], strict_audit=strict_audit)
    if ae.shape != strength.shape or ae.shape[1] != 32 or not np.isfinite(ae).all():
        raise ValueError('Invalid 32-layer AE matrix')
    if not np.isfinite(strength).all() or (strength < 0).any():
        raise ValueError('Invalid nonnegative K4 gross matrix')
    visual = np.concatenate((ae, np.log1p(strength)), axis=1).astype(np.float32)
    y = np.asarray([m['label'] for m in mentions], dtype=np.int64)
    masks = {name:np.isin([m['image_id'] for m in mentions], ids) for name,ids in split.items()}
    if not np.all(sum(mask.astype(int) for mask in masks.values()) == 1):
        raise ValueError('Mention split is not exhaustive and exclusive')
    for mask in masks.values():
        if set(y[mask].tolist()) != {0, 1}:
            raise ValueError('Every split must contain both labels')

    baseline_records = align_baselines(load_pkl(root/'baseline/features.pkl'), mentions)
    cfg = baseline_config(load_config(str(ROOT/'configs/model_configs_minigpt4_shikra_path.yaml')))
    start, end = int(cfg['svar']['layer_start']), int(cfg['svar']['layer_end'])
    svar = np.stack([svar_training_vector(get_baseline_payload(r, 'svar'), layer_start=start,
                                          layer_end=end) for r in baseline_records]).astype(np.float32)
    metatoken = np.stack([baseline_vector(r, 'metatoken') for r in baseline_records]).astype(np.float32)
    np.testing.assert_array_equal(y, [r['label'] for r in baseline_records])
    names = baseline_records[0]['baselines']['metatoken']['feature_names']
    if any(r['baselines']['metatoken']['feature_names'] != names for r in baseline_records):
        raise ValueError('MetaToken feature names differ')
    manifest = json.loads((root/'path/manifest.json').read_text())
    audit = json.loads((root/'path/full_numerical_audit.json').read_text())
    if manifest['k'] != 4 or audit['status'] != 'PASS' or audit['images'] != 4000:
        raise ValueError('Persisted K4 visual path is not audited')
    schema = ('minigpt-shikra-visual-k4-811-single-mlp-v1' if SOURCE_MODE == 'visual_k4'
              else 'minigpt-shikra-all-attention-k32-811-single-mlp-v1')
    protocol = dict(schema=schema, model=model,
        split=split, split_seed=SPLIT_SEED, mentions=len(mentions),
        counts={k:int(v.sum()) for k,v in masks.items()}, candidates=mlp.candidates(),
        search_seed=mlp.SEARCH_SEED, shortlist_seed=43, top_n=mlp.TOP_N,
        final_seeds=list(SEEDS), ranking='validation AUROC mean, HALL-AUPR mean, candidate index',
        feature='V=[visual AE, log1p(S_visual)] across all 32 decoder layers',
        path=strength_protocol, **strength_source,
        numerical_audit=audit, source_manifest_signature=manifest['signature'],
        source_protocol_sha256=sha256_file(root/'path/training/protocol.json'),
        baseline=dict(svar_layers_zero_based=[start,end], metatoken_names=names,
                      classifier_settings=native.SVAR, heads=list(HEADS)),
        global_gate='Both model validation selections frozen before any new test evaluation',
        caveat='Old800 split has previously been viewed; exploratory, not independent blind test')
    protocol['fingerprint'] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()+visual.tobytes()+svar.tobytes()+metatoken.tobytes()).hexdigest()
    model_root = OUT/model
    model_root.mkdir(parents=True, exist_ok=True)
    protocol_path = model_root/'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('Protocol changed')
    atomic_json_save(protocol, protocol_path)
    data = dict(visual=visual, svar=svar, metatoken=metatoken, y=y, masks=masks,
                mentions=mentions, fingerprint=protocol['fingerprint'])
    matrix_path = model_root/'matrices.pt'
    if matrix_path.exists():
        saved = mlp.read(matrix_path)
        if saved['fingerprint'] != protocol['fingerprint']:
            raise ValueError('Matrix fingerprint changed')
    else:
        atomic_torch_save(data, matrix_path)
    return data, protocol


def rank(row):
    return (row['validation']['AUROC'], row['validation']['HALL_AUPR'], -row['index'])


def search(model, device):
    data, protocol = prepare(model)
    root = OUT/model
    train, val = data['masks']['train'], data['masks']['validation']
    rows = []
    done = 0
    for index, cfg in enumerate(protocol['candidates']):
        path = root/'search'/f'candidate{index:02d}'/'seed43/result.pt'
        if path.exists():
            result = mlp.read(path)
        else:
            progress(model, f'811视觉V搜索 c{index:02d} seed43', done, 30)
            result = mlp.fit(data['visual'][train], data['y'][train], data['visual'][val], data['y'][val],
                cfg, 43, device, callback=lambda epoch:progress(model, f'811视觉V搜索 c{index:02d} seed43', done, 30, epoch=epoch))
            result.update(index=index, fingerprint=protocol['fingerprint'])
            atomic_torch_save(result, path)
        if result['fingerprint'] != protocol['fingerprint'] or result['config'] != cfg:
            raise ValueError('Wrong resumed search result')
        done += 1
        rows.append(dict(index=index, seed=43, validation=result['validation']))
    shortlist = sorted(rows, key=rank, reverse=True)[:3]
    comparisons = []
    for row in shortlist:
        values = [row]
        for seed in (44,45):
            index = row['index']; cfg = protocol['candidates'][index]
            path = root/'search'/f'candidate{index:02d}'/f'seed{seed}/result.pt'
            if path.exists():
                result = mlp.read(path)
            else:
                progress(model, f'811视觉V复核 c{index:02d} seed{seed}', done, 30)
                result = mlp.fit(data['visual'][train], data['y'][train], data['visual'][val], data['y'][val],
                    cfg, seed, device, callback=lambda epoch:progress(model, f'811视觉V复核 c{index:02d} seed{seed}', done, 30, epoch=epoch))
                result.update(index=index, fingerprint=protocol['fingerprint'])
                atomic_torch_save(result, path)
            if result['fingerprint'] != protocol['fingerprint']:
                raise ValueError('Wrong resumed shortlist result')
            done += 1
            values.append(dict(index=index, seed=seed, validation=result['validation']))
        comparisons.append(dict(index=row['index'], validation={k:float(np.mean([v['validation'][k] for v in values])) for k in row['validation']}, seeds=values))
    selected = max(comparisons, key=rank)
    atomic_json_save(dict(fingerprint=protocol['fingerprint'], index=selected['index'],
        config=protocol['candidates'][selected['index']], validation=selected['validation'],
        shortlist=comparisons, seed43_trials=rows, frozen_at=datetime.now(timezone.utc).isoformat(),
        test_accessed=False), root/'selection.json')
    if done != 30:
        raise ValueError(f'Expected30 fits, got {done}')
    progress(model, '811视觉V配置已冻结，等待两模型', 30, 30, status='waiting')


def gate():
    return all((OUT/m/'selection.json').exists() for m in MODELS)


def save_pickle(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def evaluate(model, device):
    if not gate():
        raise RuntimeError('Both validation selections must be frozen')
    root = OUT/model
    data = mlp.read(root/'matrices.pt')
    selection = json.loads((root/'selection.json').read_text())
    test = data['masks']['test']; train = data['masks']['train']; y = data['y'][test]
    values = []
    for seed in SEEDS:
        trained = mlp.read(root/'search'/f"candidate{selection['index']:02d}"/f'seed{seed}/result.pt')
        net = mlp.SingleMLP(trained['input_dim'], trained['config']).to(device)
        net.load_state_dict(trained['state_dict'])
        x = torch.as_tensor(mlp.transform(data['visual'][test], trained['mean'], trained['scale']), device=device)
        probability = mlp.predict(net, x)
        result = dict(model=model, method=('visual_k4_ae_log_strength' if SOURCE_MODE == 'visual_k4'
                                           else 'all_attention_k32_ae_log_strength'), seed=seed,
            fingerprint=selection['fingerprint'], index=selection['index'], config=trained['config'],
            validation=trained['validation'], test_metrics=mlp.metrics(y, probability),
            test_probabilities=probability, best_epoch=trained['best_epoch'])
        atomic_torch_save(result, root/'final'/'visual'/f'seed{seed}/result.pt')
        values.append(result)
    baseline_data = dict(groups={'svar':data['svar'], 'metatoken':data['metatoken']},
                         y=data['y'], masks=data['masks'])
    completed = 3
    for head in HEADS:
        for seed in SEEDS:
            path = root/'final'/head/f'seed{seed}/result.pt'
            if path.exists():
                result = mlp.read(path)
            else:
                progress(model, f'811原生对照 {head} seed{seed}', completed, 12)
                result, estimator = native.train_head(baseline_data, head, seed, device,
                    callback=lambda epoch:progress(model, f'811原生对照 {head} seed{seed}', completed, 12, epoch=epoch))
                result.update(fingerprint=selection['fingerprint'])
                if head != 'svar_native':
                    save_pickle(estimator, root/'final'/head/f'seed{seed}/model.pkl')
                atomic_torch_save(result, path)
            if result['fingerprint'] != selection['fingerprint']:
                raise ValueError('Wrong baseline resume fingerprint')
            completed += 1
    summarize_model(model)
    progress(model, '811视觉V与原生对照完成', 12, 12, status='completed')


def summarize_model(model):
    root = OUT/model; data = mlp.read(root/'matrices.pt'); y = data['y'][data['masks']['test']]
    rows = []
    for method in ('visual', *HEADS):
        results = [mlp.read(root/'final'/method/f'seed{s}/result.pt') for s in SEEDS]
        metric = [mlp.metrics(y, r['test_probabilities']) for r in results]
        rows.append(dict(model=model, method=method,
            **{k+'_'+stat:float(fn([m[k] for m in metric])) for k in metric[0]
               for stat,fn in [('mean',np.mean),('std',np.std)]},
            **{'ensemble_'+k:v for k,v in mlp.metrics(y,np.mean([r['test_probabilities'] for r in results],axis=0)).items()}))
    mlp.prior.original.base.write_csv(rows, root/'detection.csv')


def pipeline(model, device):
    root = OUT/model; root.mkdir(parents=True, exist_ok=True)
    with (root/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if not (root/'selection.json').exists():
                search(model, device)
            while not gate():
                progress(model, '811视觉V配置已冻结，等待两模型', 30, 30, status='waiting')
                time.sleep(10)
            state = json.loads((root/'progress.json').read_text())
            if state.get('status') != 'completed':
                evaluate(model, device)
        except BaseException as exc:
            progress(model, '811视觉V实验失败', 0, 30, status='failed', error=str(exc)[:180])
            raise


def summarize_all():
    rows=[]
    for model in MODELS:
        with (OUT/model/'detection.csv').open() as stream:
            rows.extend(csv.DictReader(stream))
    if len(rows) != 8:
        raise ValueError('Expected eight model/method rows')
    mlp.prior.original.base.write_csv(rows, OUT/'detection.csv')
    strict = SOURCE_MODE == 'all_attention_k32'
    lines=['# MiniGPT-4 / Shikra：视觉 V 单隐藏层 811', '',
        '设置：3200/400/400图片，split seed20260912，全mentions；旧800图此前已查看，属于探索性比较。每个seed单独计算后报告seeds43/44/45均值±总体标准差，不是概率ensemble。',
        ('特征：`V=[AE_V,log1p(S_V^all)]`，32层；S_V^all来自真实RMS all-attention路径`z-A_all→z`的local-FP32 factored Gauss–Legendre K32，8图K64参考审计通过。'
         if strict else '特征：`V=[AE_V,log1p(S_V)]`，32层。这里S_V来自视觉专用真实Norm路径`z-A_V→z`的local-FP32 Gauss–Legendre K4；两个模型的K4对K64 smoke及4000图闭合审计已通过。它不是后来四模型的all-attention K32，也没有冻结RMS版。'),
        '单隐藏层设置与四模型搜索完全相同：24个固定候选seed43，验证AUROC/AP选top3补44/45，再按三seed验证均值冻结配置；两模型都冻结后才测试。原生SVAR/MetaToken使用相同811样本重新训练；SVAR为248隐藏ReLU、batch32、max50、val_loss早停5，Meta为训练StandardScaler+LR/GB100。', '',
        '| 模型 | 方法 | AUROC mean±std (%) | HALL-AUPR mean±std (%) |', '|---|---|---:|---:|']
    names={'minigpt4_7b':'MiniGPT-4','shikra_7b':'Shikra'}
    display={'visual':('V: AE+log1p(S_V^all)' if strict else 'V: AE+log1p(S_V)'),
             'svar_native':'原生SVAR','metatoken_lr':'MetaToken LR','metatoken_gb':'MetaToken GB'}
    for row in rows:
        lines.append(f"| {names[row['model']]} | {display[row['method']]} | {100*float(row['AUROC_mean']):.2f} ± {100*float(row['AUROC_std']):.2f} | {100*float(row['HALL_AUPR_mean']):.2f} ± {100*float(row['HALL_AUPR_std']):.2f} |")
    lines += ['', '选择仅看验证集。原生基线与本方法的调参预算不同，因此不作公平预算下的特征单独因果结论。完整配置见各模型protocol.json和selection.json，ensemble在CSV单列。', '']
    report = ('MINIGPT4_SHIKRA_ALL_ATTENTION_V_811_RESULTS.md' if strict
              else 'MINIGPT4_SHIKRA_VISUAL_811_RESULTS.md')
    (ROOT/'docs'/report).write_text('\n'.join(lines))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODELS)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--summarize', action='store_true')
    parser.add_argument('--source', choices=('visual_k4','all_attention_k32'), default='visual_k4')
    args=parser.parse_args(); configure(args.source); torch.set_num_threads(1)
    if args.summarize:
        summarize_all()
    elif args.model:
        pipeline(args.model, args.device)
    else:
        parser.error('--model or --summarize is required')
