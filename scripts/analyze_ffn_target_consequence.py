#!/usr/bin/env python3
"""Fixed C/Q/B_Q detection contrasts, reusing saved features and legacy MLP."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from features.ffn_target_consequence import SCORES, q_features, signed_totals
from features.tc_fvpa_artifacts import atomic_json_save, sha256_file, sha256_text
from scripts import train_ffn_ae_log1p_search as trainer
from scripts.run_ffn_target_consequence import checked_load, parent_artifacts, result_root, NAME
from scripts.run_ffn_visual_source_consistency import immutable_json


def feature_groups(raw, include_c):
    def cat(*parts):
        value = np.concatenate(parts, axis=1).astype(np.float32)
        if not np.isfinite(value).all():
            raise ValueError('Nonfinite detector features')
        return value
    def log(value):
        value = np.asarray(value, dtype=np.float64)
        if (value < 0).any() or not np.isfinite(value).all():
            raise ValueError('log1p requires finite nonnegative masses')
        return np.log1p(value).astype(np.float32)
    f = trainer.direct_features(raw['AE'], raw['S'])
    fk = cat(f, raw['kappa_vec'])
    q = cat(log(raw['Q_positive']), log(raw['Q_negative']))
    b = raw['B_Q'].astype(np.float32)
    groups = dict(F=f, F_K=fk, Q=q, B_Q=b, F_Q=cat(f,q), F_B_Q=cat(f,b),
                  F_K_Q=cat(fk,q), F_K_B_Q=cat(fk,b))
    if include_c:
        for score in SCORES:
            c = cat(log(raw[f'C_{score}_positive']), log(raw[f'C_{score}_negative']))
            groups.update({f'C_{score}':c, f'F_C_{score}':cat(f,c), f'F_K_C_{score}':cat(fk,c),
                           f'F_C_{score}_Q_B_Q':cat(f,c,q,b)})
    return groups


def load_data(model, include_c):
    # This reuses and enforces the original v2 numerical-exception and mention split.
    original = trainer.study.load_training_data(model, 'fp32_k4', True)
    mentions = original['train_mentions'] + original['test_mentions']
    ntrain = len(original['train_mentions'])
    parents, validation_sha = parent_artifacts(model)
    positions, checksums, c_checksums = {}, {}, {}
    for image_id, (path, digest) in parents.items():
        shard = checked_load(path,digest)
        checksums[str(path)] = digest
        c_rows = {}
        if include_c:
            c_path = result_root(model)/'extraction/shards'/f'image_{image_id:012d}.pt'
            sidecar = json.loads(c_path.with_suffix('.json').read_text())
            c_shard = checked_load(c_path, sidecar['sha256'])
            if c_shard['parent_sha256'] != digest or not c_shard['processed_image']:
                raise ValueError('C/source cohort mismatch')
            c_rows = {r['target_key']:r for r in c_shard['positions']}
            if set(c_rows) != {r['target_key'] for r in shard['positions']}:
                raise ValueError('Incomplete C target extraction')
            c_checksums[str(c_path)] = sidecar['sha256']
        for r in shard['positions']:
            key = r['target_key']
            if key in positions:
                raise ValueError('Duplicate target')
            values = dict(AE=r['AE'], S=r['S'], kappa_vec=r['kappa_vec'])
            values.update(q_features(r['path_signed_q'],r['S'],r['net_degenerate']))
            if include_c:
                c = c_rows[key]
                if tuple(c['score_names']) != SCORES or c['layers'] != list(range(1,len(r['S'])+1)):
                    raise ValueError('C score/layer order mismatch')
                if c['C_m'].shape != (3,len(r['S']),r['path_signed_q'].shape[-1]):
                    raise ValueError('C/Q shape mismatch')
                positive, negative = signed_totals(c['C_m'])
                for i, score in enumerate(SCORES):
                    values[f'C_{score}_positive'],values[f'C_{score}_negative'] = positive[i],negative[i]
            positions[key] = {k:np.asarray(v,dtype=np.float32) for k,v in values.items()}
    if set(positions) != {m['target_key'] for m in mentions}:
        raise ValueError('Target/mention alignment failed')
    raw = {k:np.stack([positions[m['target_key']][k] for m in mentions]) for k in next(iter(positions.values()))}
    groups = feature_groups(raw,include_c)
    provenance = dict(source_validation_sha256=validation_sha, parents=checksums, c_shards=c_checksums,
                      numerical_exception=original['numerical_exception'],
                      mentions_sha256=sha256_text(json.dumps(mentions,sort_keys=True)))
    return groups, raw, ntrain, original['y_train'], original['y_test'], provenance


def run_model(model, device, include_c):
    root=result_root(model)/('training_all' if include_c else 'training_q_bq')
    root.mkdir(parents=True,exist_ok=True)
    lock=(root/'.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    groups,raw,ntrain,ytrain,ytest,provenance=load_data(model,include_c)
    protocol=dict(version=NAME,model=model,include_c=include_c,provenance=provenance,
        features='F=full AE+log1p(raw S); K=kappa_vec; C/Q=concat(log1p(positive_mass),log1p(negative_mass)); B_Q=negative_Q_mass/S',
        classifier=trainer.fixed_mlp(),trainer_defaults=vars(trainer.TorchProbeConfig()),
        seeds=[43,44,45],normalization='no extra standardization',bootstrap=False,
        q_transform_sha256=sha256_text(inspect.getsource(q_features)+inspect.getsource(signed_totals)),
        sources={str(p):sha256_file(p) for p in (Path(__file__),
          ROOT/'scripts/train_ffn_ae_log1p_search.py',ROOT/'scripts/train_torch_probe_feature_sets.py',
          ROOT/'scripts/train_ffn_consistency_alternative_heads.py')},
        matrix_sha256={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in groups.items()},
        scope='Original 3200/800 image split; exploratory, not an independent holdout claim')
    immutable_json(protocol,root/'protocol.json')
    signature=sha256_text(json.dumps(protocol,sort_keys=True))
    summaries={}
    for name,x in groups.items():
        data=dict(X_train=x[:ntrain],X_test=x[ntrain:],y_train=ytrain,y_test=ytest)
        heads=[trainer.run_head('three_hidden',trainer.fixed_mlp(),seed,data,root/'heads'/name/f'seed{seed}',signature,device)
               for seed in (43,44,45)]
        summaries[name]=trainer.summarize_group(data,heads)
        print('GROUP_COMPLETE',model,name,summaries[name]['ensemble_reports']['fixed_0.5']['auc'],flush=True)
    summary=dict(status='COMPLETE_Q_BQ_ONLY' if not include_c else 'COMPLETE_C_Q_BQ',model=model,
                 protocol_sha256=sha256_file(root/'protocol.json'),groups=summaries,
                 Q_degenerate_cases=int(raw['Q_degenerate'].sum()),
                 training_mentions=int(ntrain),test_mentions=int(len(ytest)))
    immutable_json(summary,root/'summary.json')
    report=['# C / Q / B_Q 检测对照','',protocol['scope'],'',protocol['features'],'',
            '| 特征 | Ensemble AUROC | REAL AUPR | HALL AUPR | Seed AUROC mean±std |',
            '|---|---:|---:|---:|---:|']
    for name,g in summaries.items():
        m=g['ensemble_reports']['fixed_0.5'];s=g['seed_mean_std']['auc']
        report.append(f'| {name} | {m["auc"]:.6f} | {m["real_positive"]["aupr"]:.6f} | {m["hallucination_positive"]["aupr"]:.6f} | {s["mean"]:.6f}±{s["std"]:.6f} |')
    destination=root/'summary.md'
    content='\n'.join(report)+'\n'
    if destination.exists() and destination.read_text()!=content:
        raise ValueError('Refusing changed report on resume')
    if not destination.exists():
        destination.write_text(content)
    curve=root/'layer_curves.csv'
    if not curve.exists():
        y=np.concatenate((ytrain,ytest))
        with curve.open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['signal','split','layer','label','n','mean','median','p10','p90'])
            writer.writeheader()
            for name,v in raw.items():
                if name in ('AE','S','kappa_vec','Q_degenerate'):continue
                for part,indices in (('train',np.arange(ntrain)),('test',np.arange(ntrain,len(y)))):
                    for label in (0,1):
                        rows=v[indices[y[indices]==label]]
                        for layer in range(v.shape[1]):
                            z=rows[:,layer]
                            writer.writerow(dict(signal=name,split=part,layer=layer+1,label='REAL' if label else 'HALL',n=len(z),
                                mean=float(z.mean()),median=float(np.median(z)),p10=float(np.quantile(z,.1)),p90=float(np.quantile(z,.9))))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',nargs='+',required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--q-only',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    for model in args.models:
        try:
            run_model(model,args.device,not args.q_only)
        except BaseException as error:
            atomic_json_save(dict(status='FAIL',error=str(error),traceback=traceback.format_exc()),
                result_root(model)/'training_failures'/f'{time.time_ns()}.json')
            raise


if __name__=='__main__':main()
