"""Check prefix-token alignment and saved probability products without a VLM."""
import argparse
import json
from pathlib import Path
import sys
import time

import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.extract_prefix_attention_gate import MODELS,OUT
from features.jffn_experiment import mentions_for_image
from features.tc_fvpa_artifacts import atomic_json_save


def audit(reference,smoke=False):
    results={}
    for model in MODELS:
        folder=OUT/reference/model
        if smoke:folder=folder/'smoke8'
        manifest=json.loads((folder/'manifest.json').read_text());source=Path(manifest['source'])
        labels=json.loads((source/'labeling.json').read_text());generations=json.loads((source/'generations.json').read_text())
        expected=set(manifest['images']);files=sorted((folder/'shards').glob('*.pt'))
        if len(files)!=len(expected):raise ValueError(f'{model}: incomplete shards')
        seen=set();targets=mentions_count=0;max_error=0.;minimum=None;maximum=0;has_bos=None
        layers=28 if model=='qwen2_5_vl_7b' else 36 if model=='qwen3_vl_8b' else 32
        for path in files:
            saved=torch.load(path,map_location='cpu',weights_only=False);i=saved['image_id']
            if i in seen or i not in expected or saved['signature']!=manifest['signature']:raise ValueError('Wrong source/image signature')
            seen.add(i);ids=generations[str(i)]['response_token_ids']
            mentions,indices,target_ids=mentions_for_image(image_id=i,labeling_row=labels[str(i)],response_token_ids=ids)
            assert saved['sample_table']==mentions
            assert [r['response_index'] for r in saved['positions']]==indices
            assert [r['target_token_id'] for r in saved['positions']]==target_ids
            mentions_count+=len(mentions)
            if not indices:continue
            layout=saved['layout'];prompt=layout['prompt_length'];start,end=layout['visual_range']
            tokens=layout['token_ids'];assert all(t is None for t in tokens[start:end])
            assert len(tokens)==len(layout['position_types'])==len(layout['token_pieces'])==len(layout['is_special'])
            has_bos=layout['explicit_bos_at_zero']
            for row in saved['positions']:
                idx=row['response_index'];length=row['prefix_length'];assert length==prompt+idx==row['prediction_position']+1
                assert tokens[prompt:length]==ids[:idx]
                a,b=row['raw_attention'],row['attention_x_gate'];assert a.dtype==b.dtype==torch.float32
                assert a.shape==b.shape==(layers,length)
                assert torch.isfinite(a).all() and torch.isfinite(b).all() and (a>=0).all() and (b>=0).all() and (b<=a+1e-7).all()
                assert row['gate_median'].shape==row['gate_mad'].shape==(layers,)
                assert torch.isfinite(row['gate_median']).all() and torch.isfinite(row['gate_mad']).all() and (row['gate_mad']>=0).all()
                error=float((a.sum(-1)-1).abs().max());assert error<=layout['raw_mass_tolerance']
                max_error=max(max_error,error);targets+=1;minimum=length if minimum is None else min(minimum,length);maximum=max(maximum,length)
        results[model]=dict(status='PASS',images=len(seen),targets=targets,mentions=mentions_count,
            layers=layers,prefix_length_min=minimum,prefix_length_max=maximum,max_raw_mass_error=max_error,explicit_bos_at_zero=has_bos)
    atomic_json_save(dict(status='PASS',gate_reference=reference,models=results),OUT/reference/('smoke_audit.json' if smoke else 'audit.json'))
    return results


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gate-reference',choices=('full','visual'),default='full');parser.add_argument('--smoke',action='store_true');parser.add_argument('--watch',action='store_true');args=parser.parse_args();torch.set_num_threads(1)
    if args.watch:
        while True:
            statuses={}
            for m in MODELS:
                p=OUT/args.gate_reference/m/'status.json';statuses[m]=json.loads(p.read_text()) if p.exists() else dict(status='QUEUED')
            atomic_json_save(dict(models=statuses),OUT/args.gate_reference/'progress.json')
            if any(v['status']=='FAILED' for v in statuses.values()):raise RuntimeError('An extraction job failed')
            if all(v['status']=='COMPLETE' for v in statuses.values()):break
            time.sleep(60)
    print(json.dumps(audit(args.gate_reference,args.smoke),indent=2),flush=True)


if __name__=='__main__':main()
