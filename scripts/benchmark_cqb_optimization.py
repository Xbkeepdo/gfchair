#!/usr/bin/env python3
"""Matched real-model serial/joint C benchmark, independent of production shards."""
import argparse
import contextlib
import gc
import io
import json
from pathlib import Path
import statistics
import sys
import time
import traceback
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_ffn_target_consequence as reference
from scripts.run_cqb_workflow import check_gate,GLOBAL
from features.ffn_target_consequence_fast import joint_score_adapter
from features.tc_fvpa_artifacts import atomic_json_save,sha256_file

BACKEND='joint_score_vjp_v1'
OUT=GLOBAL/'optimizations'/BACKEND


def measure(wrapper,args,image,ids,parent,layers,prompt,context,joint):
    factory=reference.fp32_cached_suffix
    def adapted(**kwargs):return joint_score_adapter(factory(**kwargs))
    torch.cuda.synchronize(args.device)
    torch.cuda.reset_peak_memory_stats(args.device)
    started=time.perf_counter()
    with patch.object(reference,'fp32_cached_suffix',adapted if joint else factory),contextlib.redirect_stdout(io.StringIO()):
        result=reference.extract_target(wrapper,args,image,ids,parent,layers,prompt,context)
    torch.cuda.synchronize(args.device)
    return result,dict(seconds=time.perf_counter()-started,peak_allocated=torch.cuda.max_memory_allocated(args.device),
                       peak_reserved=torch.cuda.max_memory_reserved(args.device))


def parity(a,b):
    difference=(a['C_m']-b['C_m']).norm(dim=-1)
    scale=a['C_m'].norm(dim=-1)
    values=max(float((x[k]-y[k]).abs().max()) for x,y in zip(a['layer_statistics'],b['layer_statistics'])
               for k in ('node_scores','clean_scores','baseline_scores'))
    normal_batches=all(r['node_batch']==4 and not r['oom_retries'] for p in (a,b) for r in p['layer_statistics'])
    return dict(pass_parity=bool((difference<=1e-8+5e-4*scale).all()) and values<=2e-6 and normal_batches,
        C_relative_l2_max=float((difference/scale.clamp_min(1e-12)).max()),C_l2_error_max=float(difference.max()),
        score_max_abs_error=values,normal_node_batch=normal_batches)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',nargs='+',default=['qwen2_5_vl_7b','llava_1_5_7b','qwen3_vl_8b','internvl_2_5_8b'])
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--repeats',type=int,default=3)
    opts=parser.parse_args()
    if opts.repeats<3:raise ValueError('Need at least three timed pairs')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    original_gate=check_gate()
    OUT.mkdir(parents=True,exist_ok=True)
    sources={str(p):sha256_file(p) for p in (Path(__file__),ROOT/'features/ffn_target_consequence_fast.py',
        ROOT/'features/ffn_visual_path_attribution.py',ROOT/'features/ffn_target_consequence.py',
        ROOT/'scripts/run_ffn_target_consequence.py')}
    protocol=dict(backend=BACKEND,reference_gate_sha256=original_gate,sources=sources,images=[283,599],
                  targets='first and last unique target per image',layers='first, middle, last',
                  repeats=opts.repeats,minimum_speedup=1.1,maximum_peak_increment_bytes=512*1024**2,
                  C_error_bound='L2(error)<=1e-8+5e-4*L2(reference), per score and layer',score_error_bound=2e-6,
                  fixed='native original capture; FP32 G/suffix, K4 and node batch4, all three score definitions')
    reference.immutable_json(protocol,OUT/'protocol.json')
    config=reference.load_config('configs/model_configs_inslen_official_target.yaml')
    for model in opts.models:
        destination=OUT/f'{model}.json'
        if destination.exists():
            print('ALREADY_BENCHMARKED',model,flush=True);continue
        wrapper=None
        started=time.perf_counter()
        try:
            wrapper=reference.build_model(model,reference.get_extraction_model_cfg(config,model),device=opts.device)
            wrapper.model.eval().requires_grad_(False)
            args=SimpleNamespace(model=model,device=opts.device,k=4,node_batch=4,token_chunk=256,
                                 suffix_precision='fp32_cached',check_cached_parity=False)
            L=len(reference.resolve_decoder_layers(wrapper.model));layer_numbers=sorted({1,L//2,L})
            parents,_=reference.parent_artifacts(model)
            _,generations,_=reference._load_inputs(ROOT/'outputs'/model/reference.EXPERIMENT)
            prompt=str((config.get('run') or {}).get('prompt') or 'Describe this image.')
            cases=[]
            for image_id in protocol['images']:
                parent=reference.checked_load(*parents[image_id]);targets=sorted(parent['positions'],key=lambda r:r['response_index'])
                with Image.open(reference._image_path(config,image_id)) as source:image=source.convert('RGB')
                ids=generations[image_id]['response_token_ids']
                context=reference.capture_image(wrapper,model,image,ids,targets,prompt)
                chosen=[targets[0]]+([targets[-1]] if len(targets)>1 else [])
                for target in chosen:
                    timed={False:[],True:[]};checks=[]
                    for repeat in range(-1,opts.repeats):
                        results={}
                        for joint in ((False,True) if repeat%2 else (True,False)):
                            result,metrics=measure(wrapper,args,image,ids,target,layer_numbers,prompt,context,joint)
                            results[joint]=result
                            if repeat>=0:timed[joint].append(metrics)
                        check=parity(results[False],results[True]);checks.append(check)
                        if not check['pass_parity']:raise ValueError(f'C mismatch: {check}')
                    baseline=statistics.median(r['seconds'] for r in timed[False])
                    fast=statistics.median(r['seconds'] for r in timed[True])
                    record=dict(target_key=target['target_key'],layers=layer_numbers,baseline=timed[False],joint=timed[True],
                                baseline_median=baseline,joint_median=fast,speedup=baseline/fast,parity=checks)
                    cases.append(record)
                    print('BENCHMARK_CASE',model,target['target_key'],f'speedup={baseline/fast:.3f}',flush=True)
                del context,parent,targets,chosen
                gc.collect();torch.cuda.empty_cache()
            speed=sum(c['baseline_median'] for c in cases)/sum(c['joint_median'] for c in cases)
            peak_old=max(r['peak_allocated'] for c in cases for r in c['baseline'])
            peak_new=max(r['peak_allocated'] for c in cases for r in c['joint'])
            total=torch.cuda.get_device_properties(opts.device).total_memory
            adopt=speed>=1.1 and peak_new<=peak_old+512*1024**2 and peak_new<=total-768*1024**2
            result=dict(model=model,status='PASS',adopt=adopt,speedup=speed,baseline_peak=peak_old,joint_peak=peak_new,
                        cases=cases,elapsed_seconds=time.perf_counter()-started,protocol_sha256=sha256_file(OUT/'protocol.json'))
            atomic_json_save(result,destination)
            print('BENCHMARK_MODEL',model,'adopt',adopt,'speedup',speed,'peak_delta',peak_new-peak_old,flush=True)
        except BaseException as error:
            atomic_json_save(dict(model=model,status='FAIL',adopt=False,error=str(error),traceback=traceback.format_exc()),
                             OUT/f'{model}_failure_{time.time_ns()}.json')
            raise
        finally:
            del wrapper
            gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
