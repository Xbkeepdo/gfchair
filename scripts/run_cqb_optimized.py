#!/usr/bin/env python3
"""Opt-in, audited joint-score VJP execution over the unchanged C experiment."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_ffn_target_consequence as reference
from scripts import run_cqb_workflow as workflow
from features.ffn_target_consequence_fast import joint_score_adapter
from features.tc_fvpa_artifacts import atomic_json_save,sha256_file

NAME='joint_score_vjp_v1'
OUT=workflow.GLOBAL/'optimizations'/NAME


def row_digest(row):
    digest=hashlib.sha256()
    def add(value):
        if isinstance(value,torch.Tensor):
            v=value.detach().cpu().contiguous()
            digest.update(str((str(v.dtype),tuple(v.shape))).encode());digest.update(v.numpy().tobytes())
        elif isinstance(value,dict):
            for key in sorted(value):digest.update(str(key).encode());add(value[key])
        elif isinstance(value,(tuple,list)):
            digest.update(str(len(value)).encode())
            for part in value:add(part)
        else:digest.update(repr(value).encode())
    add(row)
    return digest.hexdigest()


def freeze():
    protocol=json.loads((OUT/'protocol.json').read_text())
    for source,digest in protocol['sources'].items():
        if sha256_file(source)!=digest:raise ValueError('Benchmark source changed')
    if protocol['reference_gate_sha256']!=workflow.check_gate():raise ValueError('Reference implementation changed')
    models={}
    for model in reference.MODELS:
        path=OUT/f'{model}.json';r=json.loads(path.read_text())
        if r['status']!='PASS' or not r['adopt'] or r['protocol_sha256']!=sha256_file(OUT/'protocol.json'):
            raise ValueError(f'{model}: optimization not accepted')
        models[model]=dict(benchmark=str(path),sha256=sha256_file(path),speedup=r['speedup'])
    gate=dict(status='PASS_JOINT_VJP_OPTIMIZATION',backend=NAME,models=models,
              reference_gate_sha256=protocol['reference_gate_sha256'],protocol_sha256=sha256_file(OUT/'protocol.json'),
              sources=dict(protocol['sources'],**{str(Path(__file__).resolve()):sha256_file(__file__)}))
    workflow.canonical_json(gate,OUT/'gate.json')
    return gate


def gate():
    g=json.loads((OUT/'gate.json').read_text())
    if g['status']!='PASS_JOINT_VJP_OPTIMIZATION' or g['reference_gate_sha256']!=workflow.check_gate():
        raise ValueError('Invalid optimization gate')
    for source,digest in g['sources'].items():
        if sha256_file(source)!=digest:raise ValueError(f'Changed optimization code: {source}')
    for value in g['models'].values():
        if sha256_file(value['benchmark'])!=value['sha256']:raise ValueError('Changed benchmark')
    return g


def snapshot():
    gate();models={}
    for model in reference.MODELS:
        root=reference.result_root(model)/'extraction';completed={};partial={}
        if root.exists():
            with (root/'rank0.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                for path in sorted((root/'shards').glob('image_*.pt')):
                    side=path.with_suffix('.json')
                    if not side.exists():raise ValueError('Recover missing reference sidecar before snapshot')
                    meta=json.loads(side.read_text())
                    if sha256_file(path)!=meta['sha256']:raise ValueError('Reference shard changed')
                    completed[int(meta['image_id'])]=dict(path=str(path),sha256=meta['sha256'],mtime_ns=path.stat().st_mtime_ns,
                        sidecar_sha256=sha256_file(side),sidecar_mtime_ns=side.stat().st_mtime_ns)
                for path in (root/'partial').glob('image_*.pt'):
                    if int(path.stem.split('_')[-1]) in completed:continue
                    value=torch.load(path,map_location='cpu',weights_only=False)
                    for row in value['positions']:partial[row['target_key']]=row_digest(row)
        models[model]=dict(completed=completed,partial_targets=partial)
        print('REFERENCE_SNAPSHOT',model,len(completed),'partial_targets',len(partial),flush=True)
    workflow.canonical_json(dict(status='REFERENCE_PRESERVED',gate_sha256=sha256_file(OUT/'gate.json'),models=models),OUT/'reference_snapshot.json')


def lineage_checker(model):
    g=gate();record=json.loads((OUT/'reference_snapshot.json').read_text())
    if record['gate_sha256']!=sha256_file(OUT/'gate.json'):raise ValueError('Snapshot/gate mismatch')
    preserved=record['models'][model];original=reference.validate_shard
    metadata=dict(backend=NAME,gate_sha256=sha256_file(OUT/'gate.json'),reference_gate_sha256=g['reference_gate_sha256'])
    def validate(payload,*args,**kwargs):
        original(payload,*args,**kwargs)
        old=preserved['completed'].get(str(payload['image_id']))
        if old:
            if sha256_file(old['path'])!=old['sha256']:raise ValueError('Completed serial reference overwritten')
            return payload
        for row in payload['positions']:
            if row.get('execution_backend')==metadata:continue
            if row.get('execution_backend') is None and preserved['partial_targets'].get(row['target_key'])==row_digest(row):continue
            raise ValueError(f'Unknown C execution lineage: {row["target_key"]}')
        return payload
    return validate,metadata


def run_extract(model,device):
    validate,metadata=lineage_checker(model)
    original_factory=reference.fp32_cached_suffix;original_target=reference.extract_target
    def factory(**kwargs):return joint_score_adapter(original_factory(**kwargs))
    def target(*args,**kwargs):
        result=original_target(*args,**kwargs)
        result['execution_backend']=dict(metadata)
        return result
    root=reference.result_root(model)/'extraction'
    workflow.canonical_json(dict(metadata,controller_sha256=sha256_file(__file__),
        reference_snapshot_sha256=sha256_file(OUT/'reference_snapshot.json'),
        semantics='Original manifest is the reference experiment; target rows explicitly identify the validated execution backend. Unmarked rows must be preserved serial references.'),
        root/'execution_optimizations'/f'{NAME}.json')
    sys.argv=[str(ROOT/'scripts/run_ffn_target_consequence.py'),'--model',model,'--device',device]
    with patch.object(reference,'immutable_json',workflow.canonical_json),patch.object(reference,'fp32_cached_suffix',factory),\
         patch.object(reference,'extract_target',target),patch.object(reference,'validate_shard',validate):
        reference.main()


def verify_preserved():
    s=json.loads((OUT/'reference_snapshot.json').read_text());counts={}
    for model,data in s['models'].items():
        for row in data['completed'].values():
            p=Path(row['path']);side=p.with_suffix('.json')
            if (p.stat().st_mtime_ns!=row['mtime_ns'] or sha256_file(p)!=row['sha256'] or
                side.stat().st_mtime_ns!=row['sidecar_mtime_ns'] or sha256_file(side)!=row['sidecar_sha256']):
                raise ValueError(f'Reference artifact altered: {p}')
        counts[model]=len(data['completed'])
    atomic_json_save(dict(status='PASS',unchanged_completed_images=counts),OUT/'preservation_check.json')
    print('PRESERVED',counts,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('freeze','snapshot','extract','validate','train','pipeline','verify-preserved'),required=True)
    parser.add_argument('--models',nargs='+',choices=reference.MODELS,default=list(reference.MODELS))
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    if args.stage=='freeze':freeze();return
    if args.stage=='snapshot':snapshot();return
    if args.stage=='verify-preserved':verify_preserved();return
    gate();run=OUT/'runs'/str(time.time_ns());run.mkdir(parents=True)
    atomic_json_save(dict(command=sys.argv,controller_sha256=sha256_file(__file__)),run/'command.json')
    try:
        for model in args.models:
            if args.stage=='pipeline':
                for stage in ('extract','validate','train'):
                    subprocess.run([sys.executable,__file__,'--stage',stage,'--models',model,'--device',args.device],cwd=ROOT,check=True)
            elif args.stage=='extract':run_extract(model,args.device)
            elif args.stage=='validate':
                checker,_=lineage_checker(model)
                with patch.object(reference,'validate_shard',checker):workflow.validate_extraction(model)
            else:
                subprocess.run([sys.executable,str(ROOT/'scripts/run_cqb_workflow.py'),'--stage','train','--models',model,'--device',args.device],cwd=ROOT,check=True)
        atomic_json_save(dict(status='COMPLETE_ASSIGNED_STAGE',stage=args.stage,models=args.models),run/'status.json')
    except BaseException as error:
        atomic_json_save(dict(status='FAIL',error=str(error),traceback=traceback.format_exc()),run/'failure.json')
        raise


if __name__=='__main__':main()
