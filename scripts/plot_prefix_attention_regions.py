"""Disjoint BOS/vision/prompt/generated-text mass curves from saved prefixes."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'outputs/prefix_attention_gate/full'
OUT=SOURCE/'region_plots'
MODELS=('minigpt4_7b','shikra_7b','qwen2_5_vl_7b','llava_1_5_7b','qwen3_vl_8b','internvl_2_5_8b')
REGIONS=('BOS','visual','prompt','generated_text')
METRICS=('raw_attention','attention_x_gate')


def region_masks(layout,length):
    kinds=layout['position_types'][:length]
    if len(kinds)!=length or any(k not in ('visual','prompt_text','response_text') for k in kinds):raise ValueError('Invalid prefix position types')
    bos=np.zeros(length,dtype=bool)
    if layout['explicit_bos_at_zero']:
        if not length or layout['token_ids'][0]!=layout['bos_token_id'] or kinds[0]!='prompt_text':raise ValueError('BOS mapping mismatch')
        bos[0]=True
    masks=np.stack([bos,np.array([k=='visual' for k in kinds]),np.array([k=='prompt_text' for k in kinds])&~bos,np.array([k=='response_text' for k in kinds])])
    if not np.all(masks.sum(0)==1):raise ValueError('Regions must partition the prefix')
    return masks


def region_sums(row,layout):
    masks=region_masks(layout,row['prefix_length'])
    values=[];error=0.
    for name in METRICS:
        a=np.asarray(row[name],dtype=np.float64)
        if a.ndim!=2 or a.shape[1]!=masks.shape[1] or not np.isfinite(a).all() or (a<0).any():raise ValueError('Invalid attention array')
        groups=np.stack([a[:,mask].sum(-1) for mask in masks],axis=-1)
        error=max(error,float(np.max(np.abs(groups.sum(-1)-a.sum(-1)))))
        values.append(groups.astype(np.float32))
    if error>1e-10:raise ValueError('Region masses do not conserve total')
    return np.stack(values),error


def collect(model):
    dest=OUT/model;dest.mkdir(parents=True,exist_ok=True)
    cache=dest/'regions.npz'
    if cache.exists():
        with np.load(cache) as z:return z['values'],z['labels'],z['train'],json.loads((dest/'metadata.json').read_text())
    root=SOURCE/model;manifest=json.loads((root/'manifest.json').read_text())
    split=json.loads((Path(manifest['source'])/'image_splits.json').read_text())
    train_ids,test_ids=set(split['train']),set(split['test'])
    files=sorted((root/'shards').glob('*.pt'));expected=set(manifest['images']);seen=set()
    assert len(files)==4000 and len(train_ids)==3200 and len(test_ids)==800 and not train_ids&test_ids
    values=[];labels=[];images=[];keys=[];bos_flags=set();error=0.;empty_images=0;targets=0
    for n,path in enumerate(files,1):
        shard=torch.load(path,map_location='cpu',weights_only=False);image=shard['image_id']
        if image in seen or image not in expected or shard['signature']!=manifest['signature']:raise ValueError('Wrong or duplicate shard')
        seen.add(image)
        lookup={}
        for row in shard['positions']:
            sums,e=region_sums(row,shard['layout']);error=max(error,e)
            lookup[row['target_key']]=sums;targets+=1
            bos_flags.add(shard['layout']['explicit_bos_at_zero'])
        if not lookup:empty_images+=1
        for mention in shard['sample_table']:
            values.append(lookup[mention['target_key']]);labels.append(mention['label']);images.append(image);keys.append(mention['mention_id'])
        if n%500==0:print(model,'loaded',n,flush=True)
    if seen!=train_ids|test_ids:raise ValueError('Image coverage differs')
    if len(keys)!=len(set(keys)):raise ValueError('Duplicate mention')
    values=np.stack(values);labels=np.array(labels);images=np.array(images)
    train=np.isin(images,list(train_ids))
    metadata=dict(model=model,images=4000,unique_targets=targets,mentions=len(labels),train_mentions=int(train.sum()),test_mentions=int((~train).sum()),
        metrics=METRICS,regions=REGIONS,layers=values.shape[2],explicit_bos=bool(True in bos_flags),empty_images=empty_images,
        source_manifest_signature=manifest['signature'],max_partition_error=error,
        definition='sum within disjoint regions, no renormalization or division by region length; actual initial BOS only; prompt includes remaining template/special tokens; generated_text excludes target/future',
        weighting='all original mentions equally weighted; empty region contributes zero; train/test separate; IQR is not confidence interval')
    if len(bos_flags)>1:raise ValueError('Mixed BOS conventions within one model')
    np.savez_compressed(cache,values=values,labels=labels,train=train,image_ids=images,mention_ids=np.array(keys))
    (dest/'metadata.json').write_text(json.dumps(metadata,indent=2))
    return values,labels,train,metadata


def plot_model(model):
    values,labels,train,meta=collect(model);dest=OUT/model;records=[]
    for split,selection in [('train',train),('test',~train)]:
        fig,axes=plt.subplots(2,4,figsize=(18,7),squeeze=False)
        for mi,metric in enumerate(METRICS):
            for gi,region in enumerate(REGIONS):
                ax=axes[mi,gi]
                for label,name,color in [(0,'HALL','#d55e00'),(1,'REAL','#0072b2')]:
                    x=values[selection&(labels==label),mi,:,gi].astype(np.float64)
                    if not len(x):raise ValueError('Missing label class')
                    mean=x.mean(0);median=np.median(x,axis=0);q25,q75=np.quantile(x,[.25,.75],axis=0);layers=np.arange(1,len(mean)+1)
                    if region!='BOS' or meta['explicit_bos']:
                        ax.plot(layers,mean,color=color,label=f'{name} (n={len(x)})');ax.fill_between(layers,q25,q75,color=color,alpha=.13)
                    for l in range(len(mean)):
                        records.append(dict(model=model,split=split,metric=metric,region=region,label=label,n=len(x),layer=l+1,mean=mean[l],median=median[l],q25=q25[l],q75=q75[l]))
                if region=='BOS' and not meta['explicit_bos']:
                    ax.text(.5,.5,'No explicit initial BOS\n(region mass = 0)',ha='center',va='center',transform=ax.transAxes)
                else:ax.legend(fontsize=7)
                ax.set_title(region);ax.set_xlabel('Layer (1-based)');ax.set_ylabel(metric+' mass');ax.set_xlim(1,len(mean));ax.grid(alpha=.15);ax.set_ylim(bottom=0)
        fig.suptitle(f'{model} / {split}: mean + IQR, mention-weighted (not confidence interval)');fig.tight_layout()
        fig.savefig(dest/f'{split}.png',dpi=160);fig.savefig(dest/f'{split}.pdf');plt.close(fig)
    with (dest/'curves.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    print(model,'DONE',flush=True)


def overview():
    for split in ('train','test'):
        for metric in METRICS:
            fig,axes=plt.subplots(6,4,figsize=(18,18),squeeze=False)
            for ri,model in enumerate(MODELS):
                with (OUT/model/'curves.csv').open() as f:rows=list(csv.DictReader(f))
                meta=json.loads((OUT/model/'metadata.json').read_text())
                for gi,region in enumerate(REGIONS):
                    ax=axes[ri,gi]
                    for label,name,color in [('0','HALL','#d55e00'),('1','REAL','#0072b2')]:
                        subset=sorted([r for r in rows if r['split']==split and r['metric']==metric and r['region']==region and r['label']==label],key=lambda r:int(r['layer']))
                        layers=[int(r['layer']) for r in subset]
                        if region!='BOS' or meta['explicit_bos']:
                            ax.plot(layers,[float(r['mean']) for r in subset],color=color,label=name)
                            ax.fill_between(layers,[float(r['q25']) for r in subset],[float(r['q75']) for r in subset],color=color,alpha=.13)
                    if region=='BOS' and not meta['explicit_bos']:ax.text(.5,.5,'No explicit BOS',ha='center',transform=ax.transAxes)
                    else:ax.legend(fontsize=7)
                    ax.set_title(f'{model} / {region}',fontsize=9);ax.set_xlabel('Layer');ax.set_xlim(1,max(layers));ax.set_ylim(bottom=0);ax.grid(alpha=.15)
            fig.suptitle(f'{metric} / {split}: region SUM; mean + IQR (not CI); panel y-scales differ');fig.tight_layout()
            fig.savefig(OUT/f'{split}_{metric}.png',dpi=140);fig.savefig(OUT/f'{split}_{metric}.pdf');plt.close(fig)
    lines=['# 完整前缀四区域注意力曲线','', '区域互斥：真实起点BOS、视觉、其余prompt、目标之前的生成文本。无BOS模型不伪造BOS；其起点模板token计入prompt。各区域直接求和，不除以token数、不再归一化。全部mentions等权，空生成文本区域记0；实线均值、阴影IQR，非置信区间。', '',
           '[测试集 raw attention](test_raw_attention.png) · [测试集 attention×gate](test_attention_x_gate.png)',
           '[训练集 raw attention](train_raw_attention.png) · [训练集 attention×gate](train_attention_x_gate.png)', '']
    for m in MODELS:lines.append(f'- {m}：[测试集]({m}/test.png) · [训练集]({m}/train.png) · [逐层数据]({m}/curves.csv)')
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--models',nargs='+',choices=MODELS,default=list(MODELS));parser.add_argument('--overview',action='store_true');args=parser.parse_args();torch.set_num_threads(1)
    if args.overview:overview()
    else:
        for model in args.models:plot_model(model)
