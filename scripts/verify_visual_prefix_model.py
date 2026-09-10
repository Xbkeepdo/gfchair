"""Real-model causal and raw-head-mean checks on a saved smoke caption."""
import argparse
import json
from pathlib import Path
import sys

import torch
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from models import build_model
from models.base_wrapper import ExtractionRequirements
from utils.config_utils import load_config,get_extraction_model_cfg
from features.tc_fvpa_artifacts import atomic_json_save
from scripts.run_jffn_p_comparison import _image_path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',required=True)
    parser.add_argument('--device',required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    root=ROOT/'outputs'/args.model/'COCO4000-JACOBIAN-PATH/smoke8'
    config=load_config(str(root/'config.yaml'))
    generations=json.loads((root/'generations.json').read_text())
    image_id=min(map(int,generations))
    ids=generations[str(image_id)]['response_token_ids']
    indices=[0,len(ids)//2]
    wrapper=build_model(args.model,get_extraction_model_cfg(config,args.model),device=args.device)
    req=ExtractionRequirements(attention='per_head',logits=True,token_hidden_states=False,
        patch_hidden_states=False,response_hidden_states=False,dgst_capture=False)
    with Image.open(_image_path(config,image_id)) as source:
        image=source.convert('RGB')
    batch=wrapper.extract_token_features_batch(image,ids,indices,requirements=req)
    means=wrapper.extract_token_features_batch(image,ids,indices,requirements=ExtractionRequirements(
        attention='head_mean',logits=False,token_hidden_states=False,patch_hidden_states=False,dgst_capture=False))
    changed=list(ids)
    changed[indices[-1]:]=[wrapper.tokenizer.eos_token_id]*len(changed[indices[-1]:])
    future=wrapper.extract_token_features_batch(image,changed,indices,requirements=req)
    rows=[]
    for j,index in enumerate(indices):
        prefix=wrapper.extract_token_features(image,ids[:index],index,target_token_id=ids[index],requirements=req)
        logit_error=float((batch[j].token_logits-prefix.token_logits).abs().max())
        attention_error=float((batch[j].text_to_patch_attn-prefix.text_to_patch_attn).abs().max())
        future_error=float((batch[j].token_logits-future[j].token_logits).abs().max())
        mean_error=float((means[j].text_to_patch_attn[:,0]-batch[j].text_to_patch_attn.float().mean(1)).abs().max())
        # Native FP16 GEMM uses different shapes for prefix vs full-caption.
        # GPU vs CPU FP32 reductions may differ by one rounding unit.
        if future_error!=0 or mean_error>torch.finfo(torch.float32).eps:
            raise ValueError((logit_error,attention_error,future_error,mean_error))
        rows.append(dict(response_index=index,prediction_position=batch[j].baseline_capture['prediction_position'],
            prefix_logit_max_abs=logit_error,prefix_attention_max_abs=attention_error,
            future_logit_max_abs=future_error,mean_max_abs=mean_error))
    # Distinguish length-dependent FP16 rounding from a prefix alignment bug.
    # Reuse exactly the same visual embeddings and verify CPU FP32 arithmetic.
    cached=wrapper.processor(text=wrapper._format_prompt(wrapper.resolve_prompt()),images=image)
    cached={k:v.detach().cpu() for k,v in cached.items()}
    wrapper.processor=lambda **kwargs: {k:v.clone() for k,v in cached.items()}
    wrapper.model.cpu().float()
    wrapper.device='cpu'
    torch.cuda.empty_cache()
    torch.set_num_threads(4)
    full=wrapper.extract_token_features_batch(image,ids,indices,requirements=req)
    fp32=[]
    for j,index in enumerate(indices):
        prefix=wrapper.extract_token_features(image,ids[:index],index,target_token_id=ids[index],requirements=req)
        logits=float((full[j].token_logits-prefix.token_logits).abs().max())
        attention=float((full[j].text_to_patch_attn-prefix.text_to_patch_attn).abs().max())
        if logits>1e-4 or attention>1e-5:
            raise ValueError(f'FP32 causal-prefix discrepancy: {logits}, {attention}')
        fp32.append(dict(response_index=index,logits_max_abs=logits,attention_max_abs=attention))
    atomic_json_save(dict(status='PASS',model=args.model,image_id=image_id,checks=rows,
                         fp32_same_visual_prefix=fp32),root/'wrapper_parity.json')
    print(json.dumps(rows),flush=True)


if __name__=='__main__':main()
