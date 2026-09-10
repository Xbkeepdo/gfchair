#!/usr/bin/env python3
"""Exploratory 500-image C/Q/B_Q probes using completed shards only; no VLM."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import analyze_ffn_target_consequence as analysis
from scripts import run_ffn_target_consequence as extraction
from scripts import run_cqb_optimized as optimized
from scripts.run_cqb_workflow import canonical_json
from scripts.run_ffn_visual_source_attribution import result_root as v1_root
from scripts.run_ffn_visual_source_consistency import LAYER_COUNTS
from features.tc_fvpa_artifacts import atomic_json_save, sha256_file, sha256_text

MODELS = ('qwen2_5_vl_7b', 'qwen3_vl_8b')
OUT = ROOT/'outputs/ffn_target_consequence_cqb_v1/subset500_20260909'
SEED = 20260909


def select_images(available, split, ntrain=400, ntest=100):
    available = set(available)
    train, test = set(split['train']), set(split['test'])
    if train & test:
        raise ValueError('Original image split overlaps')
    key = lambda i: (hashlib.sha256(f'{SEED}:{i}'.encode()).hexdigest(), i)
    pools = [sorted(available&part, key=key) for part in (train, test)]
    if len(pools[0]) < ntrain or len(pools[1]) < ntest:
        raise ValueError('Insufficient completed train/test images; no substitution')
    return dict(train=sorted(pools[0][:ntrain]), test=sorted(pools[1][:ntest]))


def cohort():
    path = OUT/'cohort.json'
    if path.exists():
        value = json.loads(path.read_text())
        if value['models'] != list(MODELS) or value['seed'] != SEED:
            raise ValueError('Changed frozen subset protocol')
        for p, digest in value['split_sha256'].items():
            if sha256_file(p) != digest:
                raise ValueError('Changed original image split')
            original = json.loads(Path(p).read_text())
            for part, count in (('train',400),('test',100)):
                selected = value['split'][part]
                if len(selected)!=count or len(set(selected))!=count or not set(selected)<=set(original[part]):
                    raise ValueError('Invalid frozen subset split')
        return value
    available, splits, digests = {}, {}, {}
    for model in MODELS:
        root = ROOT/'outputs'/model/extraction.EXPERIMENT
        p = root/'image_splits.json'
        splits[model] = json.loads(p.read_text())
        digests[str(p)] = sha256_file(p)
        available[model] = sorted(int(p.stem.split('_')[-1]) for p in
            (extraction.result_root(model)/'extraction/shards').glob('image_*.pt')
            if p.with_suffix('.json').exists())
    first = splits[MODELS[0]]
    if any(set(s[p]) != set(first[p]) for s in splits.values() for p in ('train', 'test')):
        raise ValueError('Models do not share original split')
    selected = select_images(set.intersection(*(set(v) for v in available.values())), first)
    value = dict(models=list(MODELS), seed=SEED, split=selected, available_at_freeze=available,
                 split_sha256=digests, selection='SHA256(seed:image_id), common completed images, without labels or metrics',
                 scope='500 shared images: 400 original-train and 100 original-test; all mentions/targets/layers. Completion-conditioned exploratory subset, not independent confirmation.')
    canonical_json(value, path)
    return value


def load_subset(model, selection):
    ids = set(selection['split']['train']+selection['split']['test'])
    parents, validation_sha = extraction.parent_artifacts(model)
    validation_path = ROOT/'outputs/ffn_visual_source_consistency_v2/old_validation_fp32_k4.json'
    exception = analysis.trainer.study.training_numerical_exception(validation_path)
    checker, _ = optimized.lineage_checker(model)
    root = extraction.result_root(model)/'extraction'
    manifest = json.loads((root/'manifest.json').read_text())
    # Keep the original mention order; mmap reads metadata without loading all maps.
    mentions = []
    for p in sorted((v1_root(model)/'shards/full').glob('features*_shard_*.pt')):
        rows = torch.load(p, map_location='cpu', weights_only=False, mmap=True)['sample_table']
        mentions.extend(dict(r) for r in rows if r['image_id'] in ids)
    train = [r for r in mentions if r['image_id'] in selection['split']['train']]
    test = [r for r in mentions if r['image_id'] in selection['split']['test']]
    mentions = train+test
    if len({r['mention_id'] for r in mentions}) != len(mentions):
        raise ValueError('Duplicate mention')
    positions, source_hashes, closure, empty = {}, {}, [], []
    for image in sorted(ids):
        parent = extraction.checked_load(*parents[image])
        p = root/'shards'/f'image_{image:012d}.pt'
        side = json.loads(p.with_suffix('.json').read_text())
        c = extraction.checked_load(p, side['sha256'])
        if c['image_id'] != image or c['parent_sha256'] != parents[image][1]:
            raise ValueError('C parent/image mismatch')
        checker(c, manifest['fingerprint'], parent, list(range(1,LAYER_COUNTS[model]+1)))
        source_hashes[str(parents[image][0])] = parents[image][1]
        source_hashes[str(p)] = side['sha256']
        by_key = {r['target_key']:r for r in c['positions']}
        if not by_key:
            empty.append(image)
        for r in parent['positions']:
            key = r['target_key']
            if key in positions:
                raise ValueError('Duplicate source target')
            values = {k:r[k] for k in ('AE','S','kappa_vec')}
            values.update(analysis.q_features(r['path_signed_q'],r['S'],r['net_degenerate']))
            positive, negative = analysis.signed_totals(by_key[key]['C_m'])
            for i, score in enumerate(analysis.SCORES):
                values[f'C_{score}_positive'], values[f'C_{score}_negative'] = positive[i], negative[i]
            for stat in by_key[key]['layer_statistics']:
                closure.append(np.asarray(stat['closure_absolute_error'],dtype=np.float64))
            positions[key] = {k:np.asarray(v,dtype=np.float32) for k,v in values.items()}
    if set(positions) != {r['target_key'] for r in mentions}:
        raise ValueError('Subset target/mention mismatch')
    raw = {k:np.stack([positions[r['target_key']][k] for r in mentions]) for k in next(iter(positions.values()))}
    labels = np.array([r['label'] for r in mentions],dtype=np.int32)
    for y in (labels[:len(train)],labels[len(train):]):
        if set(y.tolist()) != {0,1}:
            raise ValueError('Both labels required; do not reselect images')
    closure = np.asarray(closure)
    if not np.isfinite(closure).all():
        raise ValueError('Nonfinite absolute closure error')
    provenance = dict(source_hashes=source_hashes,source_validation_sha256=validation_sha,
        numerical_exception=exception,extraction_manifest_sha256=sha256_file(root/'manifest.json'),
        mentions=mentions,processed_image_ids=sorted(ids),no_target_images=empty,
        unique_targets=len(positions),train_mentions=len(train),test_mentions=len(test),
        label_counts={part:{str(y):int((v==y).sum()) for y in (0,1)} for part,v in
                      [('train',labels[:len(train)]),('test',labels[len(train):])]},
        closure_absolute_p90_by_score=np.quantile(closure,.9,axis=0).tolist(),
        closure_absolute_max_by_score=closure.max(0).tolist(),
        numerical_status='Observed K4 errors only; not a new numerical PASS')
    return analysis.feature_groups(raw,True), labels, len(train), provenance


def run_model(model, device):
    root = OUT/model
    root.mkdir(parents=True,exist_ok=True)
    with (root/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        selection = cohort()
        groups, labels, ntrain, provenance = load_subset(model,selection)
        protocol = dict(model=model,cohort_sha256=sha256_file(OUT/'cohort.json'),scope=selection['scope'],
            provenance=provenance,device=device,seeds=[43,44,45],classifier=analysis.trainer.fixed_mlp(),
            defaults=vars(analysis.trainer.TorchProbeConfig()),bootstrap=False,
            source_sha256={str(p):sha256_file(p) for p in
                (Path(__file__).resolve(),Path(analysis.__file__),Path(analysis.trainer.__file__),
                 ROOT/'features/ffn_target_consequence.py',ROOT/'scripts/train_torch_probe_feature_sets.py')},
            feature_sha256={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in groups.items()})
        canonical_json(protocol,root/'protocol.json')
        signature = sha256_file(root/'protocol.json')
        results = {}
        for name, x in groups.items():
            data = dict(X_train=x[:ntrain],X_test=x[ntrain:],y_train=labels[:ntrain],y_test=labels[ntrain:])
            heads = [analysis.trainer.run_head('three_hidden',analysis.trainer.fixed_mlp(),seed,data,
                     root/'heads'/name/f'seed{seed}',signature,device) for seed in (43,44,45)]
            results[name] = analysis.trainer.summarize_group(data,heads)
            canonical_json(dict(groups=results),root/'progress'/f'{len(results):02d}.json')
            print('GROUP_COMPLETE',model,name,results[name]['ensemble_reports']['fixed_0.5']['auc'],flush=True)
        summary = dict(status='COMPLETE_EXPLORATORY_SUBSET500',model=model,scope=selection['scope'],
            protocol_sha256=signature,images=500,train_images=400,test_images=100,
            train_mentions=ntrain,test_mentions=len(labels)-ntrain,groups=results)
        canonical_json(summary,root/'summary.json')
        print('MODEL_COMPLETE',model,flush=True)
        return summary


def summarize():
    results = {m:json.loads((OUT/m/'summary.json').read_text()) for m in MODELS}
    if any(len(v['groups'])!=20 or v['status']!='COMPLETE_EXPLORATORY_SUBSET500' for v in results.values()):
        raise ValueError('Incomplete subset heads')
    canonical_json(dict(scope=cohort()['scope'],models=results),OUT/'summary.json')
    lines = ['# C/Q/B_Q：共享500图探索性检测','',
        '两模型共享400张原训练图+100张原测试图；全部mentions/targets/layers；三seed。样本取自已完成C提取的交集，不代表全4000图随机样本或独立确认。',
        'F=全AE+log1p(raw S)，K=κ_vec；C/Q为正负总量分别log1p，B_Q=负Q总量/S。旧三隐藏层固定协议，无bootstrap或调参。','',
        '| 特征 | Qwen2.5 AUROC / HALL-AUPR (%) | Qwen3 AUROC / HALL-AUPR (%) |',
        '|---|---:|---:|']
    for name in results[MODELS[0]]['groups']:
        cells=[]
        for m in MODELS:
            v=results[m]['groups'][name]['ensemble_reports']['fixed_0.5']
            cells.append(f'{100*v["auc"]:.3f} / {100*v["hallucination_positive"]["aupr"]:.3f}')
        lines.append('| '+name+' | '+' | '.join(cells)+' |')
    lines += ['', '完整JSON包含逐seed、均值/标准差、ensemble及固定0.5/train-REAL-F1双阈值指标。各模型protocol记录实际标签数、完整SHA、闭合误差及旧数值例外。C数值门控不因本次训练而标为PASS。']
    text='\n'.join(lines)+'\n'
    p=OUT/'summary.md'
    if p.exists() and p.read_text()!=text:
        raise ValueError('Changed report on resume')
    if not p.exists():
        p.write_text(text)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare',action='store_true')
    p.add_argument('--summarize',action='store_true')
    p.add_argument('--model',choices=MODELS)
    p.add_argument('--device',default='cuda:0')
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.summarize:
        summarize(); return
    if args.prepare:
        print(json.dumps(cohort(),ensure_ascii=False)); return
    if args.model is None:
        p.error('--model is required for training')
    try:
        run_model(args.model,args.device)
    except BaseException as error:
        atomic_json_save(dict(error=str(error),traceback=traceback.format_exc()),
                        OUT/args.model/'failures'/f'{time.time_ns()}.json')
        raise


if __name__ == '__main__':
    main()
