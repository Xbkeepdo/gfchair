#!/usr/bin/env python3
"""Thin C/Q/B_Q workflow: canonical JSON resume, smoke gate, coverage, training."""
import argparse
import gc
import inspect
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
from scripts import run_ffn_target_consequence as extract
from scripts import analyze_ffn_target_consequence as analyze
from scripts.run_ffn_visual_source_consistency import immutable_json
from features.tc_fvpa_artifacts import atomic_json_save,sha256_file,sha256_text
from features.ffn_target_consequence import q_features,signed_totals

GLOBAL=ROOT/'outputs'/extract.NAME


def canonical_json(payload,path):
    # Tuple/list are identical JSON values. Preserve strict content checks and
    # frozen scientific/trainer source hashes without changing old artifacts.
    return immutable_json(json.loads(json.dumps(payload)),path)


def q_output_root(model):
    base=extract.result_root(model)
    path=base/'training_q_bq/protocol.json'
    expected=sha256_text(inspect.getsource(q_features)+inspect.getsource(signed_totals))
    if path.exists() and json.loads(path.read_text()).get('q_transform_sha256')!=expected:
        # Preserve the early result whose source-location introspection ran
        # while the new C helper was being edited. Do not relabel its hashes.
        return base/f'q_source_metadata_r1_{expected[:8]}'
    return base


def freeze_smoke():
    models={}
    for model in extract.MODELS:
        candidates=[]
        for path in (extract.result_root(model)/'smoke').rglob('manifest.json'):
            value=json.loads(path.read_text())
            if (value.get('layers')=='all' and value.get('k')==4 and value.get('suffix')=='fp32_cached'
                and value.get('node_batch')==4 and value.get('cache_full_parity_requested')
                and all(sha256_file(name)==digest for name,digest in value['sources'].items())):
                candidates.append((path,value))
        if len(candidates)!=1:raise ValueError(f'{model}: need exactly one completed current-code full-layer parity smoke')
        path,value=candidates[0];parents,_=extract.parent_artifacts(model)
        target_layers=0;max_relative=0.;max_score=0.;max_closure=0.;max_write_error=0.
        for image_id in value['images']:
            shard=path.parent/'shards'/f'image_{image_id:012d}.pt'
            sidecar=json.loads(shard.with_suffix('.json').read_text())
            data=extract.checked_load(shard,sidecar['sha256']);parent=extract.checked_load(*parents[image_id])
            layers=list(range(1,len(parent['positions'][0]['S'])+1))
            extract.validate_shard(data,value['fingerprint'],parent,layers,True)
            for row in data['positions']:
                for statistics in row['layer_statistics']:
                    parity=statistics['cache_full_parity']
                    max_relative=max(max_relative,float(parity['C_relative_l2'].max()))
                    max_score=max(max_score,parity['score_max_abs_error'])
                    max_closure=max(max_closure,float(statistics['closure_absolute_error'].max()))
                    max_write_error=max(max_write_error,statistics['parent_write_map_relative_l2'])
                    target_layers+=1
        if max_relative>5e-3 or max_score>5e-4 or max_write_error>1e-4:raise ValueError('Cached/full or C/Q source mismatch')
        models[model]=dict(manifest=str(path),sha256=sha256_file(path),sources=value['sources'],
            target_layers=target_layers,C_relative_l2_max=max_relative,score_max_abs_error=max_score,
            observed_closure_max_abs=max_closure)
        models[model]['parent_write_map_relative_l2_max']=max_write_error
    payload=dict(status='PASS_IMPLEMENTATION_SMOKE_NOT_FULL_COHORT_ACCEPTANCE',models=models,
                 precision='native prefix capture; current G and cached suffix FP32',k=4,node_batch=4)
    canonical_json(payload,GLOBAL/'implementation_gate.json')
    print(json.dumps(payload,indent=2),flush=True)
    return payload


def check_gate():
    path=GLOBAL/'implementation_gate.json'
    gate=json.loads(path.read_text())
    if gate['status']!='PASS_IMPLEMENTATION_SMOKE_NOT_FULL_COHORT_ACCEPTANCE':raise ValueError('No implementation gate')
    for value in gate['models'].values():
        if sha256_file(value['manifest'])!=value['sha256']:raise ValueError('Smoke manifest changed')
        for source,digest in value['sources'].items():
            if sha256_file(source)!=digest:raise ValueError('Numerical implementation changed after smoke')
    return sha256_file(path)


def validate_extraction(model):
    root=extract.result_root(model)/'extraction'
    manifest=json.loads((root/'manifest.json').read_text())
    parents,_=extract.parent_artifacts(model)
    files={int(p.stem.split('_')[-1]):p for p in (root/'shards').glob('image_*.pt')}
    if set(files)!=set(parents):raise ValueError(f'{model}: {len(files)}/4000 C images; training requires complete extraction')
    targets=0;digests={};absolute=[]
    for image_id,path in sorted(files.items()):
        sidecar=json.loads(path.with_suffix('.json').read_text());data=extract.checked_load(path,sidecar['sha256'])
        parent=extract.checked_load(*parents[image_id])
        if data['parent_sha256']!=parents[image_id][1]:raise ValueError('Parent fingerprint mismatch')
        from scripts.run_ffn_visual_source_consistency import LAYER_COUNTS
        extract.validate_shard(data,manifest['fingerprint'],parent,list(range(1,LAYER_COUNTS[model]+1)))
        targets+=len(data['positions']);digests[str(path)]=sidecar['sha256']
        for row in data['positions']:
            absolute.extend(s['closure_absolute_error'].tolist() for s in row['layer_statistics'])
    value=dict(status='COMPLETE_C_EXTRACTION',model=model,images=4000,unique_targets=targets,
               manifest_sha256=sha256_file(root/'manifest.json'),artifacts=digests,
               closure_absolute_max_by_score=torch.tensor(absolute).amax(0).tolist())
    canonical_json(value,root/'coverage.json')
    print('COVERAGE_PASS',model,targets,flush=True)
    return value


def publish_results(include_c=False):
    models={};sources={}
    for model in ('qwen2_5_vl_7b','llava_1_5_7b','qwen3_vl_8b','internvl_2_5_8b'):
        root=(extract.result_root(model)/'training_all' if include_c else q_output_root(model)/'training_q_bq')
        path=root/'summary.json'
        if not path.exists():return False
        value=json.loads(path.read_text())
        if len(value['groups'])!=(20 if include_c else 8):raise ValueError('Incomplete detector groups')
        for group in value['groups'].values():
            if set(group['per_seed_metrics'])!={'43','44','45'}:raise ValueError('Missing detector seed')
        models[model]=value;sources[model]=dict(path=str(path),sha256=sha256_file(path))
    label='c_q_bq' if include_c else 'q_bq'
    payload=dict(status='COMPLETE_C_Q_BQ' if include_c else 'COMPLETE_Q_BQ_ONLY',models=models,sources=sources,
                 scope='Exploratory COCO4000, original3200/800, no bootstrap; C is separate unless include_c is true')
    canonical_json(payload,GLOBAL/f'{label}_summary.json')
    names=list(models)
    groups=['F','F_K','Q','B_Q','F_Q','F_B_Q','F_K_Q','F_K_B_Q']
    if include_c:groups+=sorted(set(models[names[0]]['groups'])-set(groups))
    lines=['# C / Q / B_Q 检测汇总','',payload['status'],'',payload['scope'],'',
        'F=全AE+log1p(S)，K=κ_vec，Q=log1p(正Q总量)与log1p(负Q总量)拼接，B_Q=负Q总量/S。',
        '每格为三seed概率ensemble的 AUROC / HALL-AUPR（%）；逐seed、REAL-AUPR和两阈值P/R/F1见JSON。','',
        '| 组 | Qwen2.5 | LLaVA | Qwen3 | InternVL |','|---|---:|---:|---:|---:|']
    for name in groups:
        cells=[]
        for model in names:
            m=models[model]['groups'][name]['ensemble_reports']['fixed_0.5']
            cells.append(f'{100*m["auc"]:.3f} / {100*m["hallucination_positive"]["aupr"]:.3f}')
        lines.append('| '+name+' | '+' | '.join(cells)+' |')
    output=GLOBAL/f'{label}_summary.md';content='\n'.join(lines)+'\n'
    if output.exists() and output.read_text()!=content:raise ValueError('Refusing changed completed summary')
    if not output.exists():output.write_text(content)
    print('PUBLISHED_LOCAL',str(output),flush=True)
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('freeze','extract','validate','train','q-only','pipeline','report'),required=True)
    parser.add_argument('--models',nargs='+',choices=extract.MODELS,default=list(extract.MODELS))
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    if args.stage=='freeze':freeze_smoke();return
    if args.stage=='report':
        if not publish_results(False):raise ValueError('Q/B_Q results incomplete')
        publish_results(True)
        return
    GLOBAL.mkdir(parents=True,exist_ok=True)
    run=GLOBAL/'runs'/str(time.time_ns());run.mkdir(parents=True)
    atomic_json_save(dict(command=sys.argv,wrapper_sha256=sha256_file(__file__)),run/'command.json')
    started=time.perf_counter()
    try:
        for model in args.models:
            if args.stage=='pipeline':
                for stage in ('extract','validate','train'):
                    subprocess.run([sys.executable,__file__,'--stage',stage,'--models',model,'--device',args.device],cwd=ROOT,check=True)
                continue
            if args.stage!='q-only':check_gate()
            if args.stage=='extract':
                sys.argv=[str(ROOT/'scripts/run_ffn_target_consequence.py'),'--model',model,'--device',args.device]
                with patch.object(extract,'immutable_json',canonical_json):extract.main()
            elif args.stage=='validate':validate_extraction(model)
            elif args.stage in ('train','q-only'):
                if args.stage=='train':
                    coverage=json.loads((extract.result_root(model)/'extraction/coverage.json').read_text())
                    if coverage['status']!='COMPLETE_C_EXTRACTION':raise ValueError('C extraction not complete')
                with patch.object(analyze,'immutable_json',canonical_json):
                    if args.stage=='q-only':
                        with patch.object(analyze,'result_root',q_output_root):analyze.run_model(model,args.device,False)
                    else:analyze.run_model(model,args.device,True)
            gc.collect()
            torch.cuda.empty_cache()
        if args.stage in ('train','q-only'):publish_results(args.stage=='train')
        atomic_json_save(dict(status='COMPLETE_ASSIGNED_STAGE',stage=args.stage,models=args.models,elapsed_seconds=time.perf_counter()-started),run/'status.json')
    except BaseException as error:
        atomic_json_save(dict(status='FAIL',stage=args.stage,error=str(error),traceback=traceback.format_exc()),run/'failure.json')
        raise


if __name__=='__main__':main()
