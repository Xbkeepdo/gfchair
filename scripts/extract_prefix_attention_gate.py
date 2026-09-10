"""Save full causal-prefix raw attention and attention times hpre MAD gate."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import fcntl

import torch
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from models import build_model
from models.dgst_capture import run_forward_with_dgst_captures,attention_row_from_capture,resolve_output_embedding_layer,target_logits_multi
from scripts.run_jffn_second_round_logit_causal import prepare_inputs
from scripts.run_jffn_p_comparison import _load_inputs,_image_path
from features.jffn_experiment import mentions_for_image
from features.dgst_t import _gaussian_mad_gate,GAUSSIAN_MAD_SCALE
from features.tc_fvpa_artifacts import atomic_json_save,atomic_torch_save,sha256_file
from utils.config_utils import load_config,get_extraction_model_cfg
from scripts.run_minigpt4_shikra_path import save_json

MODELS=('minigpt4_7b','shikra_7b','qwen2_5_vl_7b','llava_1_5_7b','qwen3_vl_8b','internvl_2_5_8b')
OUT=ROOT/'outputs/prefix_attention_gate'


def mass_tolerance(dtype):
    # Native softmax probabilities are rounded to FP16/BF16 before capture.
    # Summed relative rounding error is bounded by half machine epsilon.
    return max(1e-5, .5*torch.finfo(dtype).eps+1e-5)


def gate(values,visual_range,reference='full',epsilon=1e-6):
    values=values.float()
    if values.ndim!=1 or not len(values) or not torch.isfinite(values).all():raise ValueError('Nonfinite or empty prefix logits')
    selected=values if reference=='full' else values[visual_range[0]:visual_range[1]]
    if not len(selected):raise ValueError('Empty gate reference')
    median=selected.median();mad=(selected-median).abs().median()
    if reference=='full':g=_gaussian_mad_gate(values,epsilon=epsilon)
    elif reference=='visual':g=torch.sigmoid((values-median)/(float(GAUSSIAN_MAD_SCALE)*mad+max(epsilon,1e-12)))
    else:raise ValueError('Unknown gate reference')
    return g,median,mad


def prefix_arrays(attention,logits,prediction,visual_range,reference,epsilon):
    if attention.ndim!=2 or attention.shape[-1]!=len(logits):raise ValueError('Attention/logits positions differ')
    if not 0<=prediction<len(logits):raise ValueError('Invalid prediction row')
    if not torch.isfinite(attention).all() or (attention<0).any():raise ValueError('Invalid raw attention')
    if attention[:,prediction+1:].numel() and torch.count_nonzero(attention[:,prediction+1:]):raise ValueError('Attention sees target/future positions')
    raw=attention[:,:prediction+1].float().mean(0)
    if abs(float(raw.sum())-1)>mass_tolerance(attention.dtype):raise ValueError('Unpadded causal attention mass differs from one beyond native rounding')
    g,median,mad=gate(logits[:prediction+1],visual_range,reference,epsilon)
    weighted=raw*g
    if not torch.isfinite(weighted).all() or (weighted<0).any() or (weighted>raw+1e-7).any():raise ValueError('Invalid gated attention')
    return raw,weighted,median,mad


def input_layout(wrapper,inputs,length,start,end,prompt_length):
    ids=inputs['input_ids'][0].tolist()
    if len(ids)!=length:
        if len(ids)+(end-start)-1!=length:raise ValueError('Cannot align expanded decoder positions')
        ids=ids[:start]+[None]*(end-start)+ids[start+1:]
    ids=[None if start<=i<end else token for i,token in enumerate(ids)]
    kinds=['visual' if start<=i<end else 'response_text' if i>=prompt_length else 'prompt_text' for i in range(length)]
    special=set(wrapper.tokenizer.all_special_ids)
    return dict(token_ids=ids,token_pieces=[f'<visual:{i-start}>' if token is None else wrapper.tokenizer.decode([token],skip_special_tokens=False) for i,token in enumerate(ids)],
                position_types=kinds,is_special=[token in special for token in ids],visual_range=[start,end],
                prompt_length=prompt_length,bos_token_id=wrapper.tokenizer.bos_token_id,
                explicit_bos_at_zero=wrapper.tokenizer.bos_token_id is not None and ids[0]==wrapper.tokenizer.bos_token_id)


def capture_image(wrapper,model,image,response_ids,indices,targets,reference,epsilon,prompt):
    maximum=max(indices)
    family='llava_1_5_7b' if model in ('minigpt4_7b','shikra_7b') else model
    inputs,last,start,end,grid=prepare_inputs(wrapper,family,image,response_ids[:maximum],prompt)
    if model in ('minigpt4_7b','shikra_7b'):grid=wrapper._visual_grid_for_output(inputs,start,end)
    prompt_length=last+1-maximum
    queries=[prompt_length-1+i for i in indices]
    if end>min(queries)+1:raise ValueError('Visual range extends past the earliest prediction')
    intern=model=='internvl_2_5_8b'
    output,captures=run_forward_with_dgst_captures(wrapper.model,output_hidden_states=False,retain_attention_updates=False,
        attention_query_positions=queries,capture_device='cpu',attention_query_chunk_size=None if intern else 64,
        record_model_attentions=intern,**inputs)
    length=int(captures[0]['h_prev'].shape[1])
    if length!=last+1:raise ValueError('Decoder length/prediction mapping differs')
    layout=input_layout(wrapper,inputs,length,start,end,prompt_length);layout['visual_grid']=grid
    layout['native_attention_dtype']=str(captures[0]['attn_weights'].dtype)
    layout['raw_mass_tolerance']=mass_tolerance(captures[0]['attn_weights'].dtype)
    del output
    head=resolve_output_embedding_layer(wrapper.model)
    if any(not 0<=t<head.weight.shape[0] for t in targets):raise ValueError('Invalid target token ID')
    rows=[dict(response_index=i,target_token_id=t,prediction_position=q,prefix_length=q+1,raw_attention=[],attention_x_gate=[],gate_median=[],gate_mad=[])
          for i,t,q in zip(indices,targets,queries)]
    for li,cap in enumerate(captures):
        if str(cap['attn_weights'].dtype)!=layout['native_attention_dtype']:raise ValueError('Mixed native attention precision')
        logits=target_logits_multi(output_layer=head,states=cap['h_prev'][0].float(),target_token_ids=targets,chunk_size=64)
        for j,(row,q) in enumerate(zip(rows,queries)):
            a=attention_row_from_capture(cap,q)
            raw,weighted,median,mad=prefix_arrays(a,logits[:,j],q,(start,end),reference,epsilon)
            for name,value in (('raw_attention',raw),('attention_x_gate',weighted),('gate_median',median),('gate_mad',mad)):
                row[name].append(value.cpu())
        captures[li]=None
        del logits,cap
    for row in rows:
        for name in ('raw_attention','attention_x_gate','gate_median','gate_mad'):row[name]=torch.stack(row[name]).float()
    return rows,layout


def run(args):
    two=args.model in ('minigpt4_7b','shikra_7b')
    source=ROOT/'outputs'/args.model/('COCO4000-JACOBIAN-PATH' if two else 'COCO4000-INSLEN-OFFICIAL-TARGET')
    config=load_config(str(source/'config.yaml' if two else ROOT/'configs/model_configs_inslen_official_target.yaml'))
    labels,generations,splits=_load_inputs(source)
    ids=sorted(splits['train']+splits['test'])
    if len(ids)!=4000 or len(set(ids))!=4000 or set(ids)-set(labels) or set(ids)-set(generations):raise ValueError('Incomplete original cohort')
    if args.smoke:ids=ids[:8]
    dest=OUT/args.gate_reference/args.model
    if args.smoke:dest=dest/'smoke8'
    dest.mkdir(parents=True,exist_ok=True)
    with (dest/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest=dict(model=args.model,images=ids,source=str(source),config=config,gate_reference=args.gate_reference,
            dtype='float32',head_reduction='mean',scope='actual input position 0 through causal prediction row inclusive; target/future excluded',
            gate_formula='sigmoid((hpre @ W_U[target] - median) / (1.4826 * MAD + epsilon)); no final norm or LM-head bias',
            attention_normalization='none after native softmax; attention_x_gate is raw product',
            source_sha={name:sha256_file(source/name) for name in ('generations.json','labeling.json','image_splits.json')},
            code_sha={str(p.relative_to(ROOT)):sha256_file(p) for p in [Path(__file__),ROOT/'models/dgst_capture.py',ROOT/'features/dgst_t.py',ROOT/'scripts/run_jffn_second_round_logit_causal.py']})
        signature=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
        save_json(dict(manifest,signature=signature),dest/'manifest.json')
        if not args.smoke:
            passed=json.loads((dest/'smoke8/status.json').read_text())
            if passed['status']!='COMPLETE':raise ValueError('Model smoke must complete first')
        pending=[]
        for image_id in ids:
            path=dest/'shards'/f'image_{image_id:012d}.pt'
            if path.exists():
                old=torch.load(path,map_location='cpu',weights_only=False)
                if old['signature']!=signature or old['image_id']!=image_id:raise ValueError('Wrong resumed image')
            else:pending.append(image_id)
        atomic_json_save(dict(status='RUNNING',images=len(ids),pending=len(pending)),dest/'status.json')
        try:
            wrapper=build_model(args.model,get_extraction_model_cfg(config,args.model),device=args.device) if pending else None
            if wrapper:wrapper.model.eval().requires_grad_(False)
            epsilon=float(config['feature_extraction']['dgst_t'].get('relative_vll_mad_epsilon',1e-6))
            for offset,image_id in enumerate(pending,1):
                tick=time.perf_counter();response=generations[image_id]['response_token_ids']
                mentions,indices,targets=mentions_for_image(image_id=image_id,labeling_row=labels[image_id],response_token_ids=response)
                rows=[];layout=None
                if indices:
                    with Image.open(_image_path(config,image_id)) as source_image:
                        rows,layout=capture_image(wrapper,args.model,source_image.convert('RGB'),response,indices,targets,args.gate_reference,epsilon,config['run']['prompt'])
                    for row in rows:row.update(image_id=image_id,target_key=f"{image_id}:{row['response_index']}")
                atomic_torch_save(dict(signature=signature,image_id=image_id,processed_image=True,positions=rows,sample_table=mentions,
                    layout=layout,gate_reference=args.gate_reference,gate_epsilon=epsilon,elapsed_seconds=time.perf_counter()-tick),
                    dest/'shards'/f'image_{image_id:012d}.pt')
                print(args.model,offset,'/',len(pending),'image',image_id,'targets',len(rows),'seconds',round(time.perf_counter()-tick,2),flush=True)
                atomic_json_save(dict(status='RUNNING',images=len(ids),completed=len(ids)-len(pending)+offset),dest/'status.json')
            atomic_json_save(dict(status='COMPLETE',images=len(ids),completed=len(ids)),dest/'status.json')
        except Exception:
            atomic_json_save(dict(status='FAILED',traceback=traceback.format_exc()),dest/'status.json');raise
        finally:
            gc.collect()
            if torch.cuda.is_initialized():torch.cuda.empty_cache()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--model',choices=MODELS,required=True);parser.add_argument('--device',default='cuda:0');parser.add_argument('--gate-reference',choices=('full','visual'),default='full');parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    torch.set_num_threads(1);run(args)
