"""Small COCO4000 composition experiment; plain per-image resume, no hashes."""
import argparse
import csv
import gc
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from features.ffn_source_composition import compose_ffn, normalized_mass, relative_error
from features.visual_ffn_jacobian import reconstruct_visual_directions, resolve_decoder_layer_adapter
from models import build_model
from models.dgst_capture import resolve_decoder_layers, run_forward_with_dgst_captures, attention_row_from_capture
from scripts.run_ffn_visual_source_attribution import EXPERIMENT, result_root as old_root
from scripts.run_ffn_visual_source_consistency import local_fp32
from scripts.run_jffn_p_comparison import _load_inputs, _image_path
from scripts.run_jffn_second_round_logit_causal import prepare_inputs, capture_clean
from utils.config_utils import load_config, get_extraction_model_cfg

MODELS = ('qwen2_5_vl_7b', 'llava_1_5_7b', 'qwen3_vl_8b', 'internvl_2_5_8b')
OUT = ROOT/'outputs/ffn_source_composition_v1'
CONFIG = ROOT/'configs/model_configs_inslen_official_target.yaml'


def read(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value, path)


def json_save(value, path):
    def convert(x):
        if torch.is_tensor(x): return x.tolist()
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        return str(x)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=convert)+'\n')


def parent_paths(model):
    # Use the existing path list only; deliberately do not recompute checksums.
    value = json.loads((ROOT/'outputs/ffn_visual_source_consistency_v2/old_validation_fp32_k4.json').read_text())
    return {int(Path(p).stem.split('_')[-1]): Path(p) for p in value['models'][model]['artifacts']}


def load_wrapper(model, device):
    config = load_config(str(CONFIG))
    wrapper = build_model(model, get_extraction_model_cfg(config, model), device=device)
    wrapper.model.eval().requires_grad_(False)
    return wrapper, config


def capture_full(wrapper, model, image, response_ids, targets, prompt):
    inputs, last, start, end, grid = prepare_inputs(wrapper,model,image,response_ids,prompt)
    positions = [last-len(response_ids)+int(t['response_index']) for t in targets]
    intern = model == 'internvl_2_5_8b'
    chunk = None if intern else (wrapper.dgst_attention_query_chunk_size if model=='llava_1_5_7b'
                                else int(wrapper.cfg.get('dgst_attention_query_chunk_size',512)))
    with torch.no_grad():
        output, captures = run_forward_with_dgst_captures(wrapper.model, output_hidden_states=False,
            retain_attention_updates=True, attention_query_positions=None if intern else positions,
            capture_device=None, attention_query_chunk_size=chunk, record_model_attentions=intern, **inputs)
    del output
    return captures, positions, start, end, grid


def distance_ot(p, q, states):
    from features.dgst_t import _topk_union_indices, _prepare_transport_problem_for_state_cost, _solve_transport_problem
    support = _topk_union_indices(p,q,64)
    problem = _prepare_transport_problem_for_state_cost(source_dist=p,target_dist=q,states=states,
                                                       support=support,sqrt_cosine=True)
    return _solve_transport_problem(problem,'emd')


def js(p,q):
    p,q = p.double(),q.double()
    mid = ((p+q)*.5).clamp_min(1e-300)
    return (.5*(torch.special.xlogy(p,p/mid)+torch.special.xlogy(q,q/mid)).sum(-1)).float()


def extract_image(wrapper, model, config, generation, parent, ks, audit=False, chunk=64, factored=True, examples=False):
    targets = sorted(parent['positions'], key=lambda x:x['response_index'])
    if not targets: return {k:dict(image_id=parent['image_ids'][0],positions=[]) for k in ks}
    image_id = targets[0]['image_id']
    with Image.open(_image_path(config,image_id)) as src: image=src.convert('RGB')
    captures, queries, start, end, grid = capture_full(wrapper,model,image,
        generation['response_token_ids'],targets,config.get('run',{}).get('prompt') or 'Describe this image.')
    results = {k:[dict(target_key=t['target_key'],image_id=image_id,response_index=t['response_index'],
                      target_token_id=t['target_token_id'],visual_grid=grid,visual_range=[start,end],
                      prediction_position=query,layers=[]) for t,query in zip(targets,queries)] for k in ks}
    layers = resolve_decoder_layers(wrapper.model)
    for li,(layer,cap) in enumerate(zip(layers,captures)):
        direct = reconstruct_visual_directions(layer=layer,capture=cap,prediction_positions=queries,
                    visual_start=start,visual_end=end,return_all_sources=True)
        adapter = resolve_decoder_layer_adapter(layer)
        z = cap['h_mid'][0,queries].float()
        residual = cap['h_prev'][0,queries].float()
        bias = adapter.output_projection.bias
        bias = torch.zeros_like(z) if bias is None else bias.float().expand_as(z)
        sources = torch.cat((direct['all_token_writes'].float(),residual[None],bias[None]),0)
        raw = direct['raw_attention_mean'].cpu()
        input_error = relative_error(sources.sum(0),z).cpu()
        if audit:
            reference = torch.stack([attention_row_from_capture(cap,q)[:,start:end].float().mean(0) for q in queries]).cpu()
            torch.testing.assert_close(raw,reference)
        states = cap['h_prev'][0,start:end].float()
        with local_fp32(layer):
            for k in ks:
                value = compose_ffn(adapter.ffn_norm,adapter.ffn,z,sources,(start,end),k,chunk,
                                    save_vectors=audit and (ks==(4,16) or examples) and li in (0,len(layers)//2,len(layers)-1),factored=factored)
                for ti,t in enumerate(targets):
                    row = {name:v[ti] for name,v in value.items() if name!='vectors'}
                    row.update(raw_attention_mean=raw[ti],raw_attention_mass=raw[ti].sum(),
                               attention_reconstruction=direct['component_sum_relative_error'],
                               input_relative=input_error[ti],norm_class=type(adapter.ffn_norm).__name__,
                               norm_eps=adapter.ffn_norm.variance_epsilon)
                    if audit:
                        row['write_parent_relative'] = relative_error(direct['a_tokens'][:,ti].float().norm(dim=-1),t['write_mag'][li].to(z)).cpu()
                        if 'vectors' in value:
                            row['vectors']={name:(v[:,ti] if name in ('c','n','sources') else v[ti]) for name,v in value['vectors'].items()}
                    else:
                        p = row['p_c'].to(z)
                        pw = t['p_write'][li].to(z)
                        target = t['attention_evidence'][li].to(z)
                        row['ot_TC'] = distance_ot(target,p,states)
                        row['ot_WC'] = distance_ot(pw,p,states)
                    results[k][ti]['layers'].append(row)
        del sources,direct,value
    payloads = {}
    for k,rows in results.items():
        for row,t in zip(rows,targets):
            records = row.pop('layers')
            row.update({name:torch.stack([r[name] for r in records]) for name in records[0]
                        if torch.is_tensor(records[0][name])})
            for name in ('ot_TC','ot_WC'):
                if name in records[0]: row[name]=torch.tensor([r[name] for r in records])
            row['norm_class'],row['norm_eps'] = records[0]['norm_class'],records[0]['norm_eps']
            row['attention_reconstruction'] = torch.tensor([r['attention_reconstruction'] for r in records])
            if audit: row['example_vectors']={i+1:r['vectors'] for i,r in enumerate(records) if 'vectors' in r}
            row.update({key:t[key] for key in ('AE','I','S','p_write','p_ffn','attention_evidence')})
            row['ae_token_weight'] = t['attention_evidence']*t['AE'][:,None]
            row['ot_TE'] = t['ot']['D_EF']
            row['js_TE'] = js(t['attention_evidence'],t['p_ffn'])
            row['js_TC'] = js(t['attention_evidence'],row['p_c'])
            row['k'] = k
        payloads[k] = dict(image_id=image_id,positions=rows)
    return payloads


def extract(model,device,audit=False,limit=None,chunk=64,extended=False):
    root = OUT/model
    parents = parent_paths(model)
    _,generations,splits = _load_inputs(ROOT/'outputs'/model/EXPERIMENT)
    images = sorted(splits['train'])[:8] if audit else sorted(parents)
    if limit: images=images[:limit]
    ks = ((16,32,50) if extended else (4,16)) if audit else (json.loads((OUT/'numerics.json').read_text())['k'],)
    folder = root/('audit50' if extended else 'audit') if audit else root/'shards'
    pending=[i for i in images if not all((folder/f'k{k}'/f'image_{i:012d}.pt').exists() for k in ks)]
    if not pending: return
    wrapper,config=load_wrapper(model,device)
    tick=time.time()
    for num,image_id in enumerate(pending,1):
        payloads=extract_image(wrapper,model,config,generations[image_id],read(parents[image_id]),ks,audit,chunk,
                               factored=(not audit or extended))
        for k,payload in payloads.items(): save(payload,folder/f'k{k}'/f'image_{image_id:012d}.pt')
        json_save(dict(stage='audit' if audit else 'extract',completed=len(images)-len(pending)+num,
                       total=len(images),last_image=image_id,elapsed=time.time()-tick),root/'progress.json')
        print(model,'audit' if audit else 'extract',num,'/',len(pending),'image',image_id,
              'seconds',round(time.time()-tick,1),flush=True)
    del wrapper
    gc.collect()
    torch.cuda.empty_cache()


def select_k50():
    results={}
    for model in MODELS:
        results[model]={}
        for k in (16,32,50):
            closure,strength,quadrature,source=[],[],[],[]
            paths=sorted((OUT/model/'audit50'/f'k{k}').glob('*.pt'))
            assert len(paths)==8
            for p in paths:
                a,b=read(p),read(OUT/model/'audit50/k50'/p.name)
                for x,y in zip(a['positions'],b['positions']):
                    closure.extend(x['closure_relative'].tolist())
                    strength.extend(((x['S_C']-y['S_C']).abs()/y['S_C'].clamp_min(1e-12)).tolist())
                    quadrature.extend(x['quadrature_relative'].tolist())
                    source.extend(x['source_output_relative'].tolist())
            results[model][k]=dict(closure_max=max(closure),closure_p90=float(np.quantile(closure,.9)),
                strength_relative_p90=float(np.quantile(strength,.9)),quadrature_max=max(quadrature),
                source_output_max=max(source),target_layers=len(closure))
    passing=[k for k in (16,32,50) if all(r[k]['closure_max']<=.01 and r[k]['strength_relative_p90']<=.01 for r in results.values())]
    k=passing[0] if passing else 50
    json_save(dict(k=k,models=results,all_sample_closure_pass=bool(passing),
                   selection='K16/32/50; cap50 requested by user; no repair of source rounding or old e_m'),OUT/'numerics.json')
    print('SELECTED_K',k,results,flush=True)


def select_k():
    results={}
    for model in MODELS:
        closure,strength,other,norm,formula,write=[],[],[],[],[],[]
        for p in sorted((OUT/model/'audit/k4').glob('*.pt')):
            a,b=read(p),read(OUT/model/'audit/k16'/p.name)
            for x,y in zip(a['positions'],b['positions']):
                closure.extend(x['closure_relative'].tolist())
                other.extend(y['closure_relative'].tolist())
                strength.extend(((x['S_C']-y['S_C']).abs()/y['S_C'].clamp_min(1e-12)).tolist())
                norm.extend(x['norm_relative'].tolist());formula.extend(x['norm_formula_relative'].tolist())
                write.extend(x['write_parent_relative'].tolist())
        assert len(list((OUT/model/'audit/k4').glob('*.pt')))==8
        results[model]=dict(k4_closure_max=max(closure),k16_closure_max=max(other),
            strength_relative_p90=float(np.quantile(strength,.9)),norm_sum_max=max(norm),
            norm_formula_max=max(formula),write_parent_max=max(write),target_layers=len(closure))
    k=16 if any(r['k4_closure_max']>.01 or r['strength_relative_p90']>.01 for r in results.values()) else 4
    json_save(dict(k=k,models=results),OUT/'numerics.json')
    print('SELECTED_K',k,results,flush=True)


def verify_fast(model,device,chunk):
    parents=parent_paths(model)
    _,generations,splits=_load_inputs(ROOT/'outputs'/model/EXPERIMENT)
    image_id=next(i for i in sorted(splits['train'])[:8] if read(parents[i])['positions'])
    wrapper,config=load_wrapper(model,device)
    fast=extract_image(wrapper,model,config,generations[image_id],read(parents[image_id]),(4,16),True,chunk,True)
    comparisons=[]
    for k in (4,16):
        reference=read(OUT/model/'audit'/f'k{k}'/f'image_{image_id:012d}.pt')
        for a,b in zip(fast[k]['positions'],reference['positions']):
            errors=relative_error(a['c_mag'],b['c_mag'])
            vec=[float(relative_error(a['example_vectors'][li]['c'],v['c']).max())
                 for li,v in b['example_vectors'].items()]
            # Zero directions need an absolute check; use a global vector norm.
            vector_global=[float((a['example_vectors'][li]['c']-v['c']).norm()/v['c'].norm().clamp_min(1e-12))
                           for li,v in b['example_vectors'].items()]
            comparisons.append(dict(k=k,target=a['target_key'],magnitude_relative_max=float(errors.max()),
                                    vector_global_max=max(vector_global),per_source_relative_max=max(vec)))
    assert max(max(r['magnitude_relative_max'],r['vector_global_max']) for r in comparisons)<1e-4
    json_save(comparisons,OUT/model/'fast_jvp_check.json')
    print('FACTORED_JVP_CHECK',model,comparisons,flush=True)
    del wrapper
    gc.collect();torch.cuda.empty_cache()


def build_groups(ae,i,se,sc,jse,jsc,ote,otc):
    def cat(*v): return np.concatenate(v,axis=1).astype(np.float32)
    i,se,sc=(np.log1p(np.asarray(x,dtype=np.float64)) for x in (i,se,sc))
    b=cat(ae,i,se)
    return dict(AE_I=cat(ae,i),B=b,AE_I_SC=cat(ae,i,sc),B_SC=cat(b,sc),
                B_JS_TE=cat(b,jse),B_JS_TC=cat(b,jsc),B_OT_TE=cat(b,ote),B_OT_TC=cat(b,otc),
                F=cat(ae,se),F_C=cat(ae,sc))


def save_example_vectors(model,device,chunk):
    k=json.loads((OUT/'numerics.json').read_text())['k']
    path=OUT/model/f'example_vectors_k{k}.pt'
    if path.exists(): return
    parents=parent_paths(model)
    _,generations,splits=_load_inputs(ROOT/'outputs'/model/EXPERIMENT)
    image_id=next(i for i in sorted(splits['train']) if read(parents[i])['positions'])
    wrapper,config=load_wrapper(model,device)
    payload=extract_image(wrapper,model,config,generations[image_id],read(parents[image_id]),
                          (k,),audit=True,chunk=chunk,examples=True)[k]
    row=payload['positions'][0]
    # Clone this one target's views so torch.save does not retain other targets' storage.
    row['example_vectors']={layer:{name:v.clone() for name,v in vectors.items()}
                            for layer,vectors in row['example_vectors'].items()}
    save(dict(model=model,k=k,position=row),path)
    print('EXAMPLE_VECTORS',model,row['target_key'],flush=True)
    del wrapper
    gc.collect();torch.cuda.empty_cache()


def feature_files(model):
    k=json.loads((OUT/'numerics.json').read_text())['k']
    return sorted((OUT/model/'shards'/f'k{k}').glob('image_*.pt'))


def mentions_for_model(model):
    mentions=[]
    for path in sorted((old_root(model)/'shards/full').glob('*.pt')):
        mentions.extend(read(path)['sample_table'])
    return mentions


def distribution_comparison(p,q):
    from scipy.stats import rankdata
    p,q=np.asarray(p),np.asarray(q)
    rp,rq=rankdata(p,axis=-1),rankdata(q,axis=-1)
    rp-=rp.mean(-1,keepdims=True);rq-=rq.mean(-1,keepdims=True)
    denom=np.linalg.norm(rp,axis=-1)*np.linalg.norm(rq,axis=-1)
    corr=np.divide((rp*rq).sum(-1),denom,out=np.zeros(len(p)),where=denom>0)
    corr[(denom==0)&np.all(p==q,axis=-1)]=1
    k=min(32,p.shape[-1])
    ip,iq=np.argsort(-p,axis=-1)[:,:k],np.argsort(-q,axis=-1)[:,:k]
    overlap=(ip[:,:,None]==iq[:,None,:]).any(-1).mean(-1)
    return dict(js=js(torch.as_tensor(p),torch.as_tensor(q)).numpy(),
                spearman=corr,top32=overlap,tv=np.abs(p-q).sum(-1)*.5)


def write_csv(rows,path):
    if not rows: return
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def plot_example(model,mention,row):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    config=load_config(str(CONFIG))
    from scripts.tc_fvpa_common import FORMAL_LAYERS_BY_MODEL
    levels=FORMAL_LAYERS_BY_MODEL[model]
    maps=dict(raw_attention=row['raw_attention_mean'],AE_token=row['ae_token_weight'],
              WRITE=row['p_write'],conditional=row['p_ffn'],composition=row['p_c'],
              norm_sources=row['p_n'],diff_E_W=row['p_ffn']-row['p_write'],
              diff_C_W=row['p_c']-row['p_write'],diff_C_N=row['p_c']-row['p_n'])
    fig,axes=plt.subplots(len(levels),len(maps)+1,figsize=(24,10))
    with Image.open(_image_path(config,mention['image_id'])) as image:
        for ax in axes[:,0]: ax.imshow(image);ax.axis('off')
    for col,(name,value) in enumerate(maps.items(),1):
        for ri,layer in enumerate(levels):
            ax=axes[ri,col]
            if name in ('composition','norm_sources','diff_C_W','diff_C_N') and bool(row['degenerate'][layer-1]):
                ax.text(.5,.5,'undefined: zero mass',ha='center');ax.axis('off');continue
            grid=value[layer-1].reshape(row['visual_grid']).numpy()
            if name.startswith('diff'):
                limit=max(float(np.abs(grid).max()),1e-12)
                im=ax.imshow(grid,cmap='RdBu_r',vmin=-limit,vmax=limit)
            else: im=ax.imshow(grid,cmap='magma')
            ax.set_title(f'{name} L{layer}',fontsize=8);ax.axis('off');fig.colorbar(im,ax=ax,fraction=.045)
    fig.suptitle(f'{model} image {mention["image_id"]} / {mention.get("word","")} / label {mention["label"]}')
    fig.tight_layout()
    dest=OUT/model/f'example_label{mention["label"]}'
    fig.savefig(str(dest)+'.png',dpi=140);fig.savefig(str(dest)+'.pdf');plt.close(fig)
    save(dict(mention=mention,maps=maps,visual_grid=row['visual_grid']),str(dest)+'.pt')


def analyze(model):
    root=OUT/model
    files=feature_files(model)
    assert len(files)==4000, f'{model}: {len(files)}/4000 images'
    mentions=mentions_for_model(model)
    examples={label:min((m for m in mentions if m['label']==label),key=lambda m:(m['image_id'],m['response_index'])) for label in (0,1)}
    targets={};curve_values={};errors=[];input_errors=[];norm_errors=[];quadrature=[];source_errors=[];degenerate=0;empty=0
    for file in files:
        shard=read(file)
        empty+=not bool(shard['positions'])
        for r in shard['positions']:
            key=r['target_key']
            targets[key]={name:np.asarray(r[name]) for name in ('AE','I','S','S_C','js_TE','js_TC','ot_TE','ot_TC')}
            signals={name:np.asarray(r[name]) for name in ('AE','I','S','S_C','N_C','kappa_C','visual_fraction',
                     'residual_mag','attention_bias_mag','ffn_zero_norm','raw_attention_mass')}
            signals['other_gross']=r['other_token_mag'].sum(-1).numpy()
            signals['gain_E']=(r['S']/r['I'].clamp_min(1e-12)).numpy()
            signals['gain_C']=(r['S_C']/r['I'].clamp_min(1e-12)).numpy()
            for name,p,q in (('WE',r['p_write'],r['p_ffn']),('WC',r['p_write'],r['p_c']),
                             ('EC',r['p_ffn'],r['p_c']),('NC',r['p_n'],r['p_c']),
                             ('raw_AE',normalized_mass(r['raw_attention_mean']),r['attention_evidence'])):
                signals.update({f'{name}_{metric}':v for metric,v in distribution_comparison(p,q).items()})
            signals.update(ot_TC=r['ot_TC'].numpy(),ot_WC=r['ot_WC'].numpy(),ot_TE=r['ot_TE'].numpy(),
                           js_TC=r['js_TC'].numpy(),js_TE=r['js_TE'].numpy())
            support=r['attention_evidence'].topk(min(32,r['p_c'].shape[-1]),dim=-1).indices
            for name,delta in (('E_W',r['p_ffn']-r['p_write']),('C_W',r['p_c']-r['p_write']),('C_N',r['p_c']-r['p_n'])):
                selected=delta.gather(-1,support)
                signals[name+'_evidence_gain']=selected.clamp_min(0).sum(-1).numpy()
                signals[name+'_evidence_loss']=(-selected).clamp_min(0).sum(-1).numpy()
                assert float(delta.sum(-1).abs().max())<1e-5
            curve_values[key]=signals
            errors.extend(r['closure_relative'].tolist());input_errors.extend(r['input_relative'].tolist())
            norm_errors.extend(r['norm_relative'].tolist());degenerate+=int(r['degenerate'].sum())
            quadrature.extend(r['quadrature_relative'].tolist());source_errors.extend(r['source_output_relative'].tolist())
            for example in examples.values():
                if key==example['target_key']: plot_example(model,example,r)
    splits=json.loads((ROOT/'outputs'/model/EXPERIMENT/'image_splits.json').read_text())
    train_set=set(splits['train'])
    ordered=[m for m in mentions if m['image_id'] in train_set]+[m for m in mentions if m['image_id'] not in train_set]
    ntrain=sum(m['image_id'] in train_set for m in mentions)
    raw={name:np.stack([targets[m['target_key']][name] for m in ordered]) for name in next(iter(targets.values()))}
    groups=build_groups(*(raw[name] for name in ('AE','I','S','S_C','js_TE','js_TC','ot_TE','ot_TC')))
    labels=np.array([m['label'] for m in ordered],dtype=np.int32)
    assert all(np.isfinite(x).all() for x in groups.values())
    save(dict(groups=groups,y=labels,ntrain=ntrain,mentions=ordered),root/'matrices.pt')
    curves=[]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=list(next(iter(curve_values.values())))
    fig,axes=plt.subplots((len(names)+3)//4,4,figsize=(18,3*((len(names)+3)//4)),squeeze=False)
    for name,ax in zip(names,axes.flat):
        for label,color in ((0,'tab:red'),(1,'tab:blue')):
            x=np.stack([curve_values[m['target_key']][name] for m in ordered if m['label']==label])
            mean,median=x.mean(0),np.median(x,axis=0)
            q25,q75=np.quantile(x,[.25,.75],axis=0)
            layers=np.arange(1,len(mean)+1)
            ax.plot(layers,median,color=color,label='HALL' if label==0 else 'REAL')
            ax.fill_between(layers,q25,q75,color=color,alpha=.15)
            for i in range(len(mean)):
                curves.append(dict(signal=name,label=label,layer=i+1,mean=mean[i],median=median[i],q25=q25[i],q75=q75[i]))
        ax.set_title(name);ax.legend(fontsize=7)
    for ax in list(axes.flat)[len(names):]:ax.axis('off')
    fig.tight_layout();fig.savefig(root/'curves.png',dpi=130);fig.savefig(root/'curves.pdf');plt.close(fig)
    write_csv(curves,root/'curves.csv')
    stats=dict(images=len(files),no_target_images=empty,unique_targets=len(targets),train_mentions=ntrain,
               test_mentions=len(ordered)-ntrain,target_layers=len(errors),degenerate=degenerate,
               closure_median=float(np.median(errors)),closure_p90=float(np.quantile(errors,.9)),
               closure_max=max(errors),closure_over_1pct=sum(e>.01 for e in errors),
               input_relative_max=max(input_errors),norm_relative_max=max(norm_errors),
               quadrature_relative_p90=float(np.quantile(quadrature,.9)),quadrature_relative_max=max(quadrature),
               source_output_relative_p90=float(np.quantile(source_errors,.9)),source_output_relative_max=max(source_errors))
    json_save(stats,root/'analysis.json');print('ANALYSIS',model,stats,flush=True)


def train(model,device,root=None):
    from scripts.train_torch_probe_feature_sets import TorchProbeConfig,train_and_evaluate_probe
    from scripts.train_ffn_ae_log1p_search import summarize_group
    root=OUT/model if root is None else Path(root)
    matrix=read(root/'matrices.pt');y=matrix['y'];n=matrix['ntrain'];results={}
    for name,x in matrix['groups'].items():
        data=dict(X_train=x[:n],y_train=y[:n],X_test=x[n:],y_test=y[n:])
        heads=[]
        for seed in (43,44,45):
            folder=root/'heads'/name/f'seed{seed}';path=folder/'result.pt'
            if path.exists(): value=read(path)
            else:
                metrics=train_and_evaluate_probe(**data,X_val=np.empty((0,x.shape[1]),np.float32),
                    y_val=np.empty(0,np.int32),config=TorchProbeConfig(seed=seed),device=torch.device(device),
                    output_dir=str(folder),return_probabilities=True)
                value=dict(seed=seed,train_probabilities=np.asarray(metrics.pop('train_probabilities')),
                           test_probabilities=np.asarray(metrics.pop('test_probabilities')),metrics=metrics)
                save(value,path)
            heads.append(value)
        results[name]=summarize_group(data,heads)
        print('TRAIN',model,name,results[name]['ensemble_reports']['fixed_0.5']['auc'],flush=True)
        json_save(results,root/'detection.json')


def intervene(model,device,limit=None):
    from features.ffn_visual_interactions import regular_grid_regions
    from scripts.run_ffn_visual_source_counterfactuals import _forward_intervention,_scores
    from scripts.tc_fvpa_common import FORMAL_LAYERS_BY_MODEL
    root=OUT/model;k=json.loads((OUT/'numerics.json').read_text())['k']
    targets={}
    for path in sorted((old_root(model)/'shards/counterfactuals').glob('*.pt')):
        r=read(path)['rows'][0]
        targets[r['image_id']]=r
    assert len(targets)==100
    pending=[i for i in sorted(targets) if not (root/'fixed_qk'/f'image_{i:012d}.pt').exists()]
    if limit: pending=pending[:limit]
    if pending:
        wrapper,config=load_wrapper(model,device)
        _,generations,_=_load_inputs(ROOT/'outputs'/model/EXPERIMENT)
        layers=resolve_decoder_layers(wrapper.model)
        for num,image_id in enumerate(pending,1):
            t=targets[image_id];response_index=t['response_index'];target_id=t['target_token_id']
            row=next(r for r in read(root/'shards'/f'k{k}'/f'image_{image_id:012d}.pt')['positions'] if r['response_index']==response_index)
            with Image.open(_image_path(config,image_id)) as src:image=src.convert('RGB')
            inputs,query,start,end,grid=prepare_inputs(wrapper,model,image,
                generations[image_id]['response_token_ids'][:response_index],config.get('run',{}).get('prompt') or 'Describe this image.')
            captures,logits=capture_clean(wrapper,inputs,query)
            masked=logits.clone();masked[target_id]=-torch.inf;competitor=int(masked.argmax())
            clean=_scores(logits,target_id,competitor);rows=[]
            regions=regular_grid_regions(int(grid[0]),int(grid[1]),8)
            for number in FORMAL_LAYERS_BY_MODEL[model]:
                layer=layers[number-1]
                direct=reconstruct_visual_directions(layer=layer,capture=captures[number-1],prediction_positions=[query],visual_start=start,visual_end=end)
                choices={name:max(range(8),key=lambda j:float(row[field][number-1,regions[j]].sum()))
                         for name,field in (('WRITE','p_write'),('conditional','p_ffn'),('composition','p_c'))}
                choices['random']=random.Random(20260829+image_id+response_index+number).randrange(8)
                observed={}
                for region in set(choices.values()):
                    scores,_=_forward_intervention(model=wrapper.model,inputs=inputs,layer=layer,prediction_position=query,
                        target_id=target_id,competitor_id=competitor,attention_subtract=direct['a_tokens'][regions[region],0].sum(0))
                    observed[region]=scores
                for strategy,region in choices.items():
                    delta=observed[region]['log_probability']-clean['log_probability']
                    rows.append(dict(image_id=image_id,response_index=response_index,label=t['label'],layer=number,
                        strategy=strategy,region=region,log_probability_delta=delta,absolute_log_probability_delta=abs(delta),
                        target_logit_delta=observed[region]['target_logit']-clean['target_logit']))
            save(dict(rows=rows),root/'fixed_qk'/f'image_{image_id:012d}.pt')
            print('FIXED_QK',model,num,'/',len(pending),flush=True)
        del wrapper
        gc.collect();torch.cuda.empty_cache()
    rows=[r for p in sorted((root/'fixed_qk').glob('*.pt')) for r in read(p)['rows']]
    write_csv(rows,root/'fixed_qk.csv')
    paired=[]
    for left,right in (('composition','WRITE'),('composition','conditional'),('composition','random'),('conditional','WRITE')):
        for metric in ('absolute_log_probability_delta','log_probability_delta'):
            differences=[]
            for image_id in sorted({r['image_id'] for r in rows}):
                a=np.mean([r[metric] for r in rows if r['image_id']==image_id and r['strategy']==left])
                b=np.mean([r[metric] for r in rows if r['image_id']==image_id and r['strategy']==right])
                differences.append(a-b)
            paired.append(dict(left=left,right=right,metric=metric,images=len(differences),mean=float(np.mean(differences)),
                               median=float(np.median(differences)),win_fraction=float(np.mean(np.array(differences)>0))))
    json_save(paired,root/'fixed_qk_summary.json')


def summarize():
    rows=[];deltas=[];seed_rows=[];numerical=[];interventions=[];agreement=[]
    for model in MODELS:
        results=json.loads((OUT/model/'detection.json').read_text())
        numerical.append(dict(model=model,**json.loads((OUT/model/'analysis.json').read_text())))
        interventions.extend(dict(model=model,**r) for r in json.loads((OUT/model/'fixed_qk_summary.json').read_text()))
        selected={}
        with (OUT/model/'fixed_qk.csv').open() as source:
            for r in csv.DictReader(source):
                selected.setdefault((r['image_id'],r['layer']),{})[r['strategy']]=r['region']
        agreement.append(dict(model=model,cases=len(selected),
            composition_conditional_same=sum(r['composition']==r['conditional'] for r in selected.values())/len(selected),
            composition_write_same=sum(r['composition']==r['WRITE'] for r in selected.values())/len(selected)))
        for group,value in results.items():
            metric=value['ensemble_reports']['fixed_0.5']
            rows.append(dict(model=model,group=group,AUROC=metric['auc'],HALL_AUPR=metric['hallucination_positive']['aupr']))
            for seed,entry in value['per_seed_metrics'].items():
                for rule,report in entry['threshold_reports'].items():
                    metric=report['test_metrics'];hall=metric['hallucination_positive']
                    seed_rows.append(dict(model=model,group=group,seed=seed,threshold_rule=rule,threshold=report['threshold'],
                        AUROC=metric['auc'],HALL_AUPR=hall['aupr'],HALL_precision=hall['precision'],
                        HALL_recall=hall['recall'],HALL_F1=hall['f1']))
        for left,right in (('B','AE_I'),('B','F'),('F_C','F'),('AE_I_SC','B'),('B_SC','B'),
                           ('B_JS_TC','B_JS_TE'),('B_OT_TC','B_OT_TE'),('B_JS_TC','B'),('B_OT_TC','B')):
            a,b=results[left],results[right]
            delta=[a['per_seed_metrics'][str(s)]['auc']-b['per_seed_metrics'][str(s)]['auc'] for s in (43,44,45)]
            ma,mb=(r['ensemble_reports']['fixed_0.5'] for r in (a,b))
            deltas.append(dict(model=model,left=left,right=right,AUROC_delta=ma['auc']-mb['auc'],
                HALL_AUPR_delta=ma['hallucination_positive']['aupr']-mb['hallucination_positive']['aupr'],
                seed43_delta=delta[0],seed44_delta=delta[1],seed45_delta=delta[2]))
    write_csv(rows,OUT/'detection.csv')
    write_csv(seed_rows,OUT/'seed_metrics.csv');write_csv(deltas,OUT/'paired_detection.csv')
    write_csv(numerical,OUT/'numerical_summary.csv');write_csv(interventions,OUT/'fixed_qk_summary.csv')
    write_csv(agreement,OUT/'region_agreement.csv')
    lines=['# FFN 来源分解验证结果','', '原4000图3200/800划分；10组×3seeds（原9组加F_C=AE+log1p(S_C)）；探索性结果，无bootstrap。','',
           '| 模型 | 组 | AUROC | HALL-AUPR |','|---|---|---:|---:|']
    lines += [f'| {r["model"]} | {r["group"]} | {r["AUROC"]:.6f} | {r["HALL_AUPR"]:.6f} |' for r in rows]
    lines+=['','## 配对检测增量','','| 模型 | 比较 | AUROC增量 | HALL-AUPR增量 | 逐seed AUROC差 |','|---|---|---:|---:|---|']
    lines += [f'| {r["model"]} | {r["left"]} − {r["right"]} | {r["AUROC_delta"]:+.6f} | {r["HALL_AUPR_delta"]:+.6f} | {r["seed43_delta"]:+.6f}/{r["seed44_delta"]:+.6f}/{r["seed45_delta"]:+.6f} |' for r in deltas]
    lines+=['','## 数值闭合','','| 模型 | 中位数 | P90 | 最大值 | 超过1% / target-layer |','|---|---:|---:|---:|---:|']
    lines += [f'| {r["model"]} | {r["closure_median"]:.6g} | {r["closure_p90"]:.6g} | {r["closure_max"]:.6g} | {r["closure_over_1pct"]}/{r["target_layers"]} |' for r in numerical]
    lines+=['','## fixed-QK配对差','','图片内先平均四层，再计算策略之间的差；无bootstrap。','','| 模型 | 比较 | 指标 | 均值差 | 中位数差 | 胜率 |','|---|---|---|---:|---:|---:|']
    lines += [f'| {r["model"]} | {r["left"]} − {r["right"]} | {r["metric"]} | {r["mean"]:+.6f} | {r["median"]:+.6f} | {r["win_fraction"]:.3f} |' for r in interventions]
    lines+=['','## 相同区域比例','','| 模型 | composition=conditional | composition=WRITE |','|---|---:|---:|']
    lines += [f'| {r["model"]} | {100*r["composition_conditional_same"]:.2f}% | {100*r["composition_write_same"]:.2f}% |' for r in agreement]
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('audit','audit50','verify-fast','select-k','select-k50','extract','analyze','train','intervene','examples','summarize','pipeline'),required=True)
    parser.add_argument('--models',nargs='+',choices=MODELS,default=list(MODELS))
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--limit',type=int)
    parser.add_argument('--chunk',type=int,default=64)
    args=parser.parse_args()
    torch.set_num_threads(1)
    if args.stage=='select-k': select_k(); return
    if args.stage=='select-k50': select_k50(); return
    if args.stage=='summarize': summarize(); return
    for model in args.models:
        if args.stage=='examples': save_example_vectors(model,args.device,args.chunk)
        if args.stage=='verify-fast': verify_fast(model,args.device,args.chunk)
        if args.stage in ('audit','audit50','extract','pipeline'): extract(model,args.device,args.stage in ('audit','audit50'),args.limit,args.chunk,args.stage=='audit50')
        if args.stage in ('analyze','pipeline'): analyze(model)
        if args.stage in ('train','pipeline'): train(model,args.device)
        if args.stage in ('intervene','pipeline'): intervene(model,args.device,args.limit)
        if args.stage=='pipeline': save_example_vectors(model,args.device,args.chunk)


if __name__=='__main__': main()
