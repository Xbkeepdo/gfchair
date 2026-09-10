"""Raw visual attention mass curves and R + log1p(S) detection."""
import argparse
import csv
import fcntl
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.train_ffn_ae_log1p_search import direct_features,run_head,summarize_group
from scripts.run_minigpt4_shikra_path import save_json
from features.tc_fvpa_artifacts import atomic_json_save,atomic_torch_save,sha256_file

OUT=ROOT/'outputs/raw_attention_strength'
FOUR_MODELS=('qwen2_5_vl_7b','llava_1_5_7b','qwen3_vl_8b','internvl_2_5_8b')
MODELS=('minigpt4_7b','shikra_7b',*FOUR_MODELS)


def signals(attention,strength):
    a=np.asarray(attention,dtype=np.float64)
    s=np.asarray(strength,dtype=np.float64)
    if a.ndim!=2 or not a.size or s.shape!=a.shape[:1]:raise ValueError('Expected attention[L,V], S[L]')
    if not np.isfinite(a).all() or not np.isfinite(s).all() or (a<0).any() or (s<0).any():raise ValueError('Invalid attention/strength')
    mass=a.sum(-1)
    if (mass>1+1e-5).any():raise ValueError('Visual probability mass exceeds one')
    return mass.astype(np.float32),np.log1p(s).astype(np.float32)


def plot_curves(model,r,y,ntrain,dest,ylabel='Raw visual attention mass (head mean, visual sum)'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    csv_rows=[]
    for ax,part,values,labels in zip(axes,('train','test'),(r[:ntrain],r[ntrain:]),(y[:ntrain],y[ntrain:])):
        for label,name,color in ((0,'HALL','#d55e00'),(1,'REAL','#0072b2')):
            v=values[labels==label]
            if not len(v):raise ValueError('Missing class for curves')
            mean=v.mean(0);median=np.median(v,axis=0);q25,q75=np.quantile(v,[.25,.75],axis=0)
            layers=np.arange(1,len(mean)+1)
            ax.plot(layers,mean,color=color,label=f'{name} mean (n={len(v)})')
            ax.plot(layers,median,color=color,linestyle='--',alpha=.65)
            ax.fill_between(layers,q25,q75,color=color,alpha=.12)
            for i in range(len(mean)):
                csv_rows.append(dict(model=model,split=part,label=label,layer=i+1,n=len(v),mean=float(mean[i]),median=float(median[i]),q25=float(q25[i]),q75=float(q75[i])))
        ax.set_title(f'{model} / {part}');ax.set_xlabel('Decoder layer (1-based)');ax.legend(fontsize=8);ax.grid(alpha=.15)
    axes[0].set_ylabel(ylabel)
    fig.suptitle('Solid: mean; dashed: median; shade: IQR (not confidence interval)',fontsize=10)
    fig.tight_layout();fig.savefig(dest/'curves.png',dpi=180);fig.savefig(dest/'curves.pdf');plt.close(fig)
    with (dest/'curves.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)


def run(model,device):
    if model in FOUR_MODELS:
        return run_four(model,device)
    root=ROOT/'outputs'/model/'COCO4000-JACOBIAN-PATH';dest=OUT/model;dest.mkdir(exist_ok=True,parents=True)
    with (dest/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        original=json.loads((root/'path/training/protocol.json').read_text())
        split=json.loads((root/'image_splits.json').read_text());mentions=original['mentions']
        train_ids,test_ids=set(split['train']),set(split['test'])
        assert len(train_ids)==3200 and len(test_ids)==800 and not train_ids&test_ids
        ntrain=sum(m['image_id'] in train_ids for m in mentions)
        assert all(m['image_id'] in train_ids for m in mentions[:ntrain]) and all(m['image_id'] in test_ids for m in mentions[ntrain:])
        positions={};processed=set()
        for i,(filename,digest) in enumerate(original['sources'].items(),1):
            path=Path(filename)
            if not path.is_absolute():path=ROOT/path
            if sha256_file(path)!=digest:raise ValueError(f'Changed source {path}')
            shard=torch.load(path,map_location='cpu',weights_only=False)
            if shard['image_id'] in processed or not shard['processed_image']:raise ValueError('Invalid image')
            processed.add(shard['image_id'])
            for row in shard['positions']:
                key=row['target_key']
                if key in positions:raise ValueError('Duplicate target')
                r,logs=signals(row['raw_attention_mean'],row['S'])
                positions[key]=(r,logs,direct_features(np.asarray(row['AE'])[None],np.asarray(row['S'])[None])[0])
            if i%500==0:print(model,'loaded',i,flush=True)
        if processed!=train_ids|test_ids or set(positions)!={m['target_key'] for m in mentions}:raise ValueError('Cohort mismatch')
        r,logs,base=[np.stack([positions[m['target_key']][i] for m in mentions]) for i in range(3)]
        del positions
        if hashlib.sha256(base.tobytes()).hexdigest()!=original['groups']['F']:raise ValueError('Original F mismatch')
        groups=dict(R=r,logS=logs,R_logS=np.concatenate([r,logs],axis=1))
        y=np.array([m['label'] for m in mentions],dtype=np.int32)
        plot_curves(model,r,y,ntrain,dest)
        protocol=dict(model=model,formula='R_l=sum_visual(mean_heads(raw post-softmax attention)); concat(R, log1p(raw path S))',
            images=4000,train_images=3200,test_images=800,train_mentions=ntrain,test_mentions=len(y)-ntrain,
            curve_unit='all labeled mentions; train/test separate; IQR is not a confidence interval',
            seeds=[43,44,45],mlp=original['mlp'],source_protocol_sha=sha256_file(root/'path/training/protocol.json'),
            code_sha=sha256_file(Path(__file__)),matrix_sha={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in groups.items()})
        save_json(protocol,dest/'protocol.json')
        if not (dest/'matrices.pt').exists():atomic_torch_save(dict(groups=groups,y=y,ntrain=ntrain),dest/'matrices.pt')
        fit_groups(model,device,dest,groups,y,ntrain,protocol,
                   json.loads((root/'path/training/results.json').read_text())['F'])


def fit_groups(model,device,dest,groups,y,ntrain,protocol,baseline):
    results={}
    for name,x in groups.items():
        data=dict(X_train=x[:ntrain],X_test=x[ntrain:],y_train=y[:ntrain],y_test=y[ntrain:])
        heads=[run_head('three_hidden',protocol['mlp'],seed,data,dest/name/f'seed{seed}',sha256_file(dest/'protocol.json'),device) for seed in (43,44,45)]
        results[name]=summarize_group(data,heads)
        print(model,'trained',name,flush=True)
    results['AE_logS']=baseline
    atomic_json_save(dict(status='COMPLETE',protocol=protocol,groups=results),dest/'results.json')
    summarize()


def run_four(model,device):
    from scripts.train_ffn_ae_log1p_search import fixed_mlp
    source=ROOT/'outputs/ffn_source_composition_v1'/model
    dest=OUT/model;dest.mkdir(exist_ok=True,parents=True)
    with (dest/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cached=torch.load(source/'matrices.pt',map_location='cpu',weights_only=False)
        mentions=cached['mentions'];y=np.asarray(cached['y']);ntrain=cached['ntrain']
        split=json.loads((ROOT/'outputs'/model/'COCO4000-INSLEN-OFFICIAL-TARGET/image_splits.json').read_text())
        train_ids,test_ids=set(split['train']),set(split['test'])
        assert len(train_ids)==3200 and len(test_ids)==800 and not train_ids&test_ids
        assert all(m['image_id'] in train_ids for m in mentions[:ntrain]) and all(m['image_id'] in test_ids for m in mentions[ntrain:])
        np.testing.assert_array_equal(y,[m['label'] for m in mentions])
        positions={};processed=set()
        files=sorted((source/'shards/k50').glob('image_*.pt'))
        if len(files)!=4000:raise ValueError('Incomplete composition image cohort')
        for i,path in enumerate(files,1):
            shard=torch.load(path,map_location='cpu',weights_only=False)
            if shard['image_id'] in processed:raise ValueError('Duplicate image')
            processed.add(shard['image_id'])
            for row in shard['positions']:
                key=row['target_key']
                if key in positions or row['k']!=50:raise ValueError('Duplicate target or mixed source version')
                r,logs=signals(row['raw_attention_mean'],row['S'])
                positions[key]=(r,logs,np.asarray(row['AE'],dtype=np.float32))
            if i%500==0:print(model,'loaded',i,flush=True)
        if processed!=train_ids|test_ids or set(positions)!={m['target_key'] for m in mentions}:raise ValueError('Cohort mismatch')
        r,logs,ae=[np.stack([positions[m['target_key']][i] for m in mentions]) for i in range(3)]
        np.testing.assert_array_equal(np.concatenate([ae,logs],axis=1),cached['groups']['F'])
        del positions,cached
        groups=dict(R=r,logS=logs,R_logS=np.concatenate([r,logs],axis=1))
        plot_curves(model,r,y,ntrain,dest)
        protocol=dict(model=model,formula='R_l=sum_visual(mean_heads(raw attention)); concat(R,log1p(original path S_E)), not S_C',
            source=str(source),images=4000,train_images=3200,test_images=800,train_mentions=ntrain,test_mentions=len(y)-ntrain,
            source_note='Raw attention captured in composition k50; S_E and AE are preserved parent v2 K4 signals; historical numerical limitations unchanged',
            curve_unit='all labeled mentions; train/test separate; IQR is not confidence interval',seeds=[43,44,45],mlp=fixed_mlp(),
            code_sha=sha256_file(Path(__file__)),matrix_sha={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in groups.items()})
        save_json(protocol,dest/'protocol.json')
        if not (dest/'matrices.pt').exists():atomic_torch_save(dict(groups=groups,y=y,ntrain=ntrain),dest/'matrices.pt')
        fit_groups(model,device,dest,groups,y,ntrain,protocol,json.loads((source/'detection.json').read_text())['F'])


def summarize():
    lines=['# Raw attention mass + log1p(S)', '', 'Seeds 43/44/45: mean ± population std (ddof=0), no ensemble. HALL F1 uses each seed train-REAL-F1 threshold.', '',
           '| Model | Features | AUROC | HALL AUPR | HALL F1 |','|---|---|---:|---:|---:|']
    for model in MODELS:
        path=OUT/model/'results.json'
        if not path.exists():continue
        for name,value in json.loads(path.read_text())['groups'].items():
            reports=[value['per_seed_metrics'][str(seed)]['threshold_reports']['train_f1']['test_metrics'] for seed in (43,44,45)]
            cols=([r['auc'] for r in reports],[r['hallucination_positive']['aupr'] for r in reports],[r['hallucination_positive']['f1'] for r in reports])
            lines.append('| '+' | '.join([model,name,*[f'{np.mean(v):.6f} ± {np.std(v):.6f}' for v in cols]])+' |')
    OUT.mkdir(exist_ok=True,parents=True);(OUT/'summary.md').write_text('\n'.join(lines)+'\n')
    if all((OUT/model/'curves.csv').exists() for model in FOUR_MODELS):
        plot_four_test()


def plot_four_test():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharey=True)
    for model,ax in zip(FOUR_MODELS,axes.flat):
        with (OUT/model/'curves.csv').open() as f:rows=list(csv.DictReader(f))
        for label,name,color in (('0','HALL','#d55e00'),('1','REAL','#0072b2')):
            selected=sorted((r for r in rows if r['split']=='test' and r['label']==label),key=lambda r:int(r['layer']))
            x=[int(r['layer']) for r in selected]
            ax.plot(x,[float(r['mean']) for r in selected],color=color,label=name)
            ax.fill_between(x,[float(r['q25']) for r in selected],[float(r['q75']) for r in selected],color=color,alpha=.12)
        ax.set_title(model);ax.set_xlabel('Decoder layer (1-based)');ax.set_xlim(1,max(x));ax.set_ylabel('Raw visual attention mass R');ax.legend();ax.grid(alpha=.15)
    fig.suptitle('Test mentions: mean and interquartile range (not confidence interval)')
    fig.tight_layout();fig.savefig(OUT/'four_models_test_curves.png',dpi=180);fig.savefig(OUT/'four_models_test_curves.pdf');plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--model',choices=MODELS);parser.add_argument('--device',default='cuda:0');parser.add_argument('--summarize',action='store_true');args=parser.parse_args()
    torch.set_num_threads(1)
    if args.summarize:summarize()
    elif args.model:run(args.model,args.device)
    else:parser.error('--model or --summarize required')
