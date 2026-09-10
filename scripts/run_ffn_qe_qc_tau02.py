"""QE/QC signed projections -> softmax(tau=.2) -> JS(T) -> matched F heads."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_ffn_source_composition as base
from scripts.train_q_softmax_js import q_softmax_js
from features.ffn_source_composition import composition_q

COMPOSITION=base.OUT
OUT=COMPOSITION/'q_softmax_tau02'
MODELS=base.MODELS
TAU=.2


def q_js(q,target):
    # Reuse the established all-support, natural-log JS; division occurs in FP64.
    return q_softmax_js(torch.as_tensor(q).double()/TAU,target)


def extract_image(wrapper,model,config,generation,parent,reference=None):
    image_id=parent['image_ids'][0]
    targets=sorted(parent['positions'],key=lambda r:r['response_index'])
    if not targets: return dict(image_id=image_id,positions=[])
    with Image.open(base._image_path(config,image_id)) as src: image=src.convert('RGB')
    captures,queries,start,end,grid=base.capture_full(wrapper,model,image,generation['response_token_ids'],targets,
                                                   config.get('run',{}).get('prompt') or 'Describe this image.')
    rows=[dict(target_key=t['target_key'],image_id=image_id,response_index=t['response_index'],
               target_token_id=t['target_token_id'],visual_grid=grid,q_c=[],reference_norm=[],degenerate=[]) for t in targets]
    checks=[]
    for li,(layer,cap) in enumerate(zip(base.resolve_decoder_layers(wrapper.model),captures)):
        directions=base.reconstruct_visual_directions(layer=layer,capture=cap,prediction_positions=queries,
                                                      visual_start=start,visual_end=end)
        z=cap['h_mid'][0,queries].float()
        adapter=base.resolve_decoder_layer_adapter(layer)
        with base.local_fp32(layer):
            q,length,bad=composition_q(adapter.ffn_norm,adapter.ffn,z,directions['a_tokens'].float(),k=50)
        q,length,bad=q.cpu(),length.cpu(),bad.cpu()
        for ti,row in enumerate(rows):
            row['q_c'].append(q[ti]);row['reference_norm'].append(length[ti]);row['degenerate'].append(bad[ti])
        if reference is not None and li+1 in reference['example_vectors']:
            ti=next(i for i,t in enumerate(targets) if t['target_key']==reference['target_key'])
            v=reference['example_vectors'][li+1]
            delta=v['output']-v['baseline']
            direct=v['c'][start:end]@(delta/delta.norm().clamp_min(1e-12))
            relative=float((q[ti]-direct).norm()/direct.norm().clamp_min(1e-12))
            assert relative<1e-4, (model,li+1,relative)
            checks.append(dict(layer=li+1,projection_relative_error=relative,
                               projection_max_absolute_error=float((q[ti]-direct).abs().max())))
    for row,parent_row in zip(rows,targets):
        for key in ('q_c','reference_norm','degenerate'): row[key]=torch.stack(row[key])
        for name,q in (('E',parent_row['path_signed_q']),('C',row['q_c'])):
            value,stats=q_js(q,parent_row['attention_evidence'])
            row['J_'+name]=torch.from_numpy(value)
            row['Pmax_'+name]=torch.from_numpy(stats['p_max']).float()
            row['softmax_zeros_'+name]=stats['softmax_zero_entries']
    return dict(image_id=image_id,positions=rows,projection_check=checks)


def extract(model,device,limit=None):
    parents=base.parent_paths(model)
    images=sorted(parents)
    if limit: images=images[:limit]
    root=OUT/model
    pending=[i for i in images if not (root/'shards'/f'image_{i:012d}.pt').exists()]
    if not pending: return
    _,generations,_=base._load_inputs(ROOT/'outputs'/model/base.EXPERIMENT)
    example=base.read(COMPOSITION/model/'example_vectors_k50.pt')['position']
    wrapper,config=base.load_wrapper(model,device)
    tick=time.time()
    for number,image_id in enumerate(pending,1):
        reference=example if image_id==example['image_id'] else None
        result=extract_image(wrapper,model,config,generations[image_id],base.read(parents[image_id]),reference)
        base.save(result,root/'shards'/f'image_{image_id:012d}.pt')
        if result.get('projection_check'): base.json_save(result['projection_check'],root/'projection_check.json')
        base.json_save(dict(stage='extract_QC',completed=len(images)-len(pending)+number,total=len(images),
                            last_image=image_id,elapsed=time.time()-tick),root/'progress.json')
        print('EXTRACT_QC',model,number,'/',len(pending),image_id,round(time.time()-tick,1),flush=True)
    del wrapper
    gc.collect();torch.cuda.empty_cache()


def prepare(model):
    root=OUT/model
    files=sorted((root/'shards').glob('image_*.pt'))
    assert len(files)==4000, (model,len(files))
    features={};pmax={'E':[],'C':[]};zeros={'E':0,'C':0};degenerate=0
    for path in files:
        for row in base.read(path)['positions']:
            features[row['target_key']]={name:row['J_'+name].numpy() for name in ('E','C')}
            for name in ('E','C'):
                pmax[name].extend(row['Pmax_'+name].tolist())
                zeros[name]+=row['softmax_zeros_'+name]
            degenerate+=int(row['degenerate'].sum())
    old=base.read(COMPOSITION/model/'matrices.pt')
    js={name:np.stack([features[m['target_key']][name] for m in old['mentions']]) for name in ('E','C')}
    groups={f'F_{name}_Q02':np.concatenate((old['groups']['F' if name=='E' else 'F_C'],js[name]),axis=1).astype(np.float32)
            for name in ('E','C')}
    base.save(dict(groups=groups,y=old['y'],ntrain=old['ntrain']),root/'matrices.pt')
    base.json_save(dict(temperature=TAU,images=len(files),unique_targets=len(features),
        train_mentions=old['ntrain'],test_mentions=len(old['y'])-old['ntrain'],qc_degenerate=degenerate,
        QE='saved v2 K4: <e_m, unit(G(z)-G(z-A))>',
        QC='K50: <c_m, unit(FFN(n)-FFN(0))>; full endpoint, not visual sum; no cosine division',
        features='F_E_Q02=[AE,log1p(S_E),JS(softmax(QE/0.2),T)]; F_C_Q02 analogous with S_C/QC',
        pmax_quantiles={name:np.quantile(v,[.5,.9,.99,1]).tolist() for name,v in pmax.items()},
        pmax_above_099_fraction={name:float(np.mean(np.asarray(v)>.99)) for name,v in pmax.items()},
        softmax_zero_entries=zeros),root/'settings.json')
    # Simple matching layer curves; class labels never enter the feature formula.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4));curves=[]
    for ax,(name,values) in zip(axes,js.items()):
        for label,color in ((0,'tab:red'),(1,'tab:blue')):
            x=values[old['y']==label];mean=x.mean(0);median=np.median(x,axis=0)
            q25,q75=np.quantile(x,[.25,.75],axis=0);layers=np.arange(1,len(mean)+1)
            ax.plot(layers,median,color=color,label='HALL' if label==0 else 'REAL')
            ax.fill_between(layers,q25,q75,color=color,alpha=.15)
            curves.extend(dict(Q=name,label=label,layer=i+1,mean=mean[i],median=median[i],q25=q25[i],q75=q75[i]) for i in range(len(mean)))
        ax.set_title(f'JS(softmax(Q{name}/0.2),T)');ax.set_xlabel('Layer');ax.legend()
    fig.tight_layout();fig.savefig(root/'curves.png',dpi=150);fig.savefig(root/'curves.pdf');plt.close(fig)
    base.write_csv(curves,root/'curves.csv')


def summarize():
    rows=[];paired=[];seeds=[]
    for model in MODELS:
        old=json.loads((COMPOSITION/model/'detection.json').read_text())
        new=json.loads((OUT/model/'detection.json').read_text())
        results=dict(F_E=old['F'],F_E_Q02=new['F_E_Q02'],F_C=old['F_C'],F_C_Q02=new['F_C_Q02'])
        for name,r in results.items():
            metric=r['ensemble_reports']['fixed_0.5']
            rows.append(dict(model=model,group=name,AUROC=metric['auc'],HALL_AUPR=metric['hallucination_positive']['aupr']))
            for seed,value in r['per_seed_metrics'].items():
                for rule,report in value['threshold_reports'].items():
                    m=report['test_metrics'];hall=m['hallucination_positive']
                    seeds.append(dict(model=model,group=name,seed=seed,rule=rule,threshold=report['threshold'],
                        AUROC=m['auc'],HALL_AUPR=hall['aupr'],HALL_precision=hall['precision'],HALL_recall=hall['recall'],HALL_F1=hall['f1']))
        for left,right in (('F_E_Q02','F_E'),('F_C_Q02','F_C'),('F_C_Q02','F_E_Q02')):
            a,b=(results[n]['ensemble_reports']['fixed_0.5'] for n in (left,right))
            paired.append(dict(model=model,left=left,right=right,AUROC_delta=a['auc']-b['auc'],
                HALL_AUPR_delta=a['hallucination_positive']['aupr']-b['hallucination_positive']['aupr'],
                **{f'seed{s}_AUROC_delta':results[left]['per_seed_metrics'][str(s)]['auc']-results[right]['per_seed_metrics'][str(s)]['auc'] for s in (43,44,45)}))
    base.write_csv(rows,OUT/'detection.csv');base.write_csv(seeds,OUT/'seed_metrics.csv');base.write_csv(paired,OUT/'paired.csv')
    lines=['# QE / QC softmax温度0.2与对应AE+S融合','',
        '原4000图3200/800及全部mentions，原MLP/seeds43/44/45。Q保留符号，是投影长度；不除以分量范数。',
        'softmax后与T做全视觉support自然对数JS，再分别拼接AE+log1p(S_E)或AE+log1p(S_C)。原F/F_C复用，只训练24新头。','',
        '| 模型 | 组 | ensemble AUROC | HALL-AUPR |','|---|---|---:|---:|']
    lines += [f'| {r["model"]} | {r["group"]} | {r["AUROC"]:.6f} | {r["HALL_AUPR"]:.6f} |' for r in rows]
    lines += ['','| 模型 | 比较 | AUROC差 | HALL-AUPR差 | 逐seed AUROC差 |','|---|---|---:|---:|---|']
    lines += [f'| {r["model"]} | {r["left"]} − {r["right"]} | {r["AUROC_delta"]:+.6f} | {r["HALL_AUPR_delta"]:+.6f} | '+ '/'.join(f'{r[f"seed{s}_AUROC_delta"]:+.6f}' for s in (43,44,45))+' |' for r in paired]
    lines += ['','旧测试集探索性比较，无bootstrap、新图像确认或超参数搜索。QE沿用旧K4数值口径，QC与S_C沿用K50，源重建限制保留。']
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('extract','train','pipeline','summarize'),required=True)
    p.add_argument('--models',nargs='+',choices=MODELS,default=list(MODELS))
    p.add_argument('--device',default='cuda:0');p.add_argument('--limit',type=int)
    args=p.parse_args();torch.set_num_threads(1)
    if args.stage=='summarize': summarize();return
    for model in args.models:
        if args.stage in ('extract','pipeline'): extract(model,args.device,args.limit)
        if args.stage in ('train','pipeline'):
            prepare(model);base.train(model,args.device,root=OUT/model)
            base.json_save(dict(stage='complete',completed=4000,total=4000,new_heads=6),OUT/model/'progress.json')


if __name__=='__main__': main()
