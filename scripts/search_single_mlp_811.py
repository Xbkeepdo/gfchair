"""Fixed-budget single-hidden-layer search; seal validation choices before test access."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from features.tc_fvpa_artifacts import atomic_json_save, atomic_torch_save
from scripts import train_region_ae_811 as prior
from detection.baselines import select_detection_threshold, evaluate_detection_scores

OUT=ROOT/'outputs/single_mlp_search_811_v1'
SOURCES={'true_rms':ROOT/'outputs/all_attention_ae_811_v1','frozen_rms':ROOT/'outputs/frozen_region_ae_811_v1'}
MODELS=prior.original.MODELS
GROUPS=prior.original.KINDS
SEEDS=(43,44,45)
SEARCH_SEED=20260914
N_CANDIDATES=24
TOP_N=3
VARIANTS=tuple(f'{path}/{group}' for path in SOURCES for group in GROUPS)


def read(path):return torch.load(path,map_location='cpu',weights_only=False)


def metrics(y,p):return dict(AUROC=float(roc_auc_score(y,p)),HALL_AUPR=float(average_precision_score(1-y,1-p)))


def candidates():
    values=[]
    def add(**kw):
        c=dict(width=248,standardize=False,batch_norm=True,dropout=.1,activation='relu',
               learning_rate=.001,weight_decay=1e-5,batch_size=256,
               max_epochs=150,patience=20,lr_patience=6,monitor='val_loss')
        c.update(kw)
        if c not in values:values.append(c)
    for width in (128,248,512):
        for scaled in (False,True):add(width=width,standardize=scaled)
    for scaled in (False,True):
        add(width=248,standardize=scaled,batch_norm=False,dropout=0.,weight_decay=0.,batch_size=32)
        add(width=128,standardize=scaled,dropout=.3)
    rng=np.random.default_rng(SEARCH_SEED)
    while len(values)<N_CANDIDATES:
        add(width=int(rng.choice([64,128,248,256,512,1024])),standardize=bool(rng.integers(2)),
            batch_norm=bool(rng.integers(2)),dropout=float(rng.choice([0.,.1,.3,.5])),
            activation=str(rng.choice(['relu','gelu'])),learning_rate=float(rng.choice([.0001,.0003,.001,.003,.01])),
            weight_decay=float(rng.choice([0.,1e-6,1e-4,.001])),batch_size=int(rng.choice([64,128,256,512])),
            monitor=str(rng.choice(['val_loss','val_auroc'])))
    return values


class SingleMLP(nn.Module):
    def __init__(self,dim,cfg):
        super().__init__()
        parts=[nn.Linear(dim,cfg['width'])]
        if cfg['batch_norm']:parts.append(nn.BatchNorm1d(cfg['width']))
        parts.extend([nn.ReLU() if cfg['activation']=='relu' else nn.GELU(),nn.Dropout(cfg['dropout']),nn.Linear(cfg['width'],1)])
        self.net=nn.Sequential(*parts)
    def forward(self,x):return self.net(x).squeeze(-1)


def batches(indices,size):
    chunks=list(indices.split(size))
    if len(chunks)>1 and len(chunks[-1])==1:chunks[-2]=torch.cat((chunks[-2],chunks[-1]));chunks.pop()
    return chunks


def scale_fit(x,enabled,block_sizes=None):
    x=np.asarray(x,dtype=np.float64)
    if enabled and block_sizes is not None:
        if sum(block_sizes)!=x.shape[1] or any(size<=0 for size in block_sizes):
            raise ValueError(f'Invalid block sizes {block_sizes} for input dimension {x.shape[1]}')
        mean=np.empty(x.shape[1],dtype=np.float64);scale=np.empty(x.shape[1],dtype=np.float64)
        start=0
        for size in block_sizes:
            stop=start+size;block=x[:,start:stop]
            mean[start:stop]=block.mean();scale[start:stop]=block.std();start=stop
    else:
        mean=x.mean(0) if enabled else np.zeros(x.shape[1])
        scale=x.std(0) if enabled else np.ones(x.shape[1])
    scale[scale<1e-12]=1.
    return mean,scale


def transform(x,mean,scale):
    value=((np.asarray(x,dtype=np.float64)-mean)/scale).astype(np.float32)
    if not np.isfinite(value).all():raise ValueError('Nonfinite transformed features')
    return value


@torch.no_grad()
def predict(model,x):
    model.eval();return torch.cat([model(a).sigmoid() for a in x.split(4096)]).cpu().numpy()


def fit(train_x,train_y,val_x,val_y,cfg,seed,device,callback=None):
    # Search API deliberately has no test arguments.
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    mean,scale=scale_fit(train_x,cfg['standardize'],cfg.get('block_sizes'))
    x=torch.as_tensor(transform(train_x,mean,scale),device=device)
    v=torch.as_tensor(transform(val_x,mean,scale),device=device)
    y=torch.as_tensor(train_y,dtype=torch.float32,device=device)
    vy=torch.as_tensor(val_y,dtype=torch.float32,device=device)
    model=SingleMLP(x.shape[1],cfg).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    mode='min' if cfg['monitor']=='val_loss' else 'max'
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode=mode,factor=.5,patience=cfg['lr_patience'],min_lr=1e-6)
    criterion=nn.BCEWithLogitsLoss();best=float('inf') if mode=='min' else -float('inf')
    state=None;stale=0;history=[];started=time.monotonic()
    for epoch in range(1,cfg['max_epochs']+1):
        model.train();loss_sum=torch.zeros((),device=device)
        for ix in batches(torch.randperm(len(x),device=device),cfg['batch_size']):
            opt.zero_grad(set_to_none=True);loss=criterion(model(x[ix]),y[ix]);loss.backward();opt.step()
            loss_sum+=loss.detach()*len(ix)
        model.eval()
        with torch.no_grad():
            logits=model(v);val_loss=float(criterion(logits,vy));vp=logits.sigmoid().cpu().numpy()
        score=metrics(val_y,vp);monitor=val_loss if mode=='min' else score['AUROC']
        better=monitor<best if mode=='min' else monitor>best
        if better:
            best=monitor;stale=0;best_epoch=epoch
            state={k:a.detach().cpu().clone() for k,a in model.state_dict().items()}
        else:stale+=1
        history.append(dict(epoch=epoch,train_loss=float(loss_sum/len(x)),val_loss=val_loss,**score,lr=opt.param_groups[0]['lr'],is_best=better))
        scheduler.step(monitor)
        if callback:callback(epoch)
        if cfg.get('early_stopping',True) and stale>=cfg['patience']:break
    assert state is not None
    model.load_state_dict(state)
    train_p,val_p=predict(model,x),predict(model,v)
    result=dict(seed=seed,config=cfg,input_dim=int(x.shape[1]),state_dict=state,mean=mean,scale=scale,
        best_epoch=best_epoch,epochs=len(history),history=history,validation=metrics(val_y,val_p),
        train_probabilities=train_p,validation_probabilities=val_p,seconds=time.monotonic()-started)
    del x,y,v,vy,model,opt
    return result


def prepare(model):
    data={p:read(root/model/'matrices.pt') for p,root in SOURCES.items()}
    reference=data['true_rms']
    for d in data.values():
        assert d['mentions']==reference['mentions'];np.testing.assert_array_equal(d['y'],reference['y'])
        for k in reference['masks']:np.testing.assert_array_equal(d['masks'][k],reference['masks'][k])
    protocol=dict(schema='single-hidden-811-search-v1',model=model,candidates=candidates(),search_seed=SEARCH_SEED,
        shortlist_seed=43,top_n=TOP_N,final_seeds=list(SEEDS),variants=list(VARIANTS),
        split=json.loads((SOURCES['true_rms']/model/'protocol.json').read_text())['split'],
        source_fingerprints={k:d['fingerprint'] for k,d in data.items()},
        ranking='AUROC mean desc, HALL_AUPR mean desc, candidate index asc; same rule for seed43 shortlist',
        champion='Select among all8 variant finalists by validation three-seed AUROC/AP, then fixed variant order; freeze before test',
        test_gate='All four model selections must exist and share the exact search settings before any new test evaluation',
        protocol='Image3200/400/400, original shared mentions, train-only scaling and optimization; no train+val refit',
        architecture='Exactly one hidden Linear -> optional BN -> ReLU/GELU -> dropout -> output Linear; BCE REAL=1, Adam',
        budget=dict(candidates_per_variant=24,shortlist_per_variant=3,fits_per_model=240,total_fits=960),
        primary_baselines='Previously completed native SVAR, MetaToken LR and GB, all three reported; not equally retuned',
        caveat='Previously viewed400 test; exploratory search. Same split used repeatedly. No guarantee of beating baselines, no significant/generalization claim without independent evaluation.')
    protocol['fingerprint']=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    root=OUT/model;path=root/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol,'Search protocol changed'
    atomic_json_save(protocol,path)
    train,val=reference['masks']['train'],reference['masks']['validation']
    search={f'{p}/{g}':dict(train_x=d['groups'][g][train],val_x=d['groups'][g][val]) for p,d in data.items() for g in GROUPS}
    return search,reference['y'][train],reference['y'][val],protocol


def rank(row):return (row['validation']['AUROC'],row['validation']['HALL_AUPR'],-row['index'])


def run_search(model,device):
    root=OUT/model;search,ty,vy,protocol=prepare(model);done=0;total=240
    def progress(stage,epoch=None,status='running',**other):
        atomic_json_save(dict(stage=stage,completed=done,total=total,status=status,epoch=epoch,
            heartbeat=datetime.now(timezone.utc).isoformat(),**other),root/'progress.json')
    def one(variant,index,seed):
        nonlocal done
        folder=root/variant/f'candidate{index:02d}'/f'seed{seed}';path=folder/'result.pt'
        if path.exists():
            r=read(path);assert r['fingerprint']==protocol['fingerprint']
        else:
            phase=f'811单层搜索 {variant} c{index:02d} seed{seed}'
            progress(phase)
            r=fit(**search[variant],train_y=ty,val_y=vy,cfg=protocol['candidates'][index],seed=seed,device=device,
                  callback=lambda epoch:progress(phase,epoch))
            r.update(fingerprint=protocol['fingerprint'],index=index,variant=variant)
            atomic_torch_save(r,path)
            print('FIT',model,variant,index,seed,r['validation'],round(r['seconds'],1),flush=True)
        done+=1
        return dict(index=index,validation=r['validation'],seed=seed)
    by_variant={v:[] for v in VARIANTS}
    # Same candidate list/budget for every variant, independent of any test scores.
    for index in range(N_CANDIDATES):
        for variant in VARIANTS:by_variant[variant].append(one(variant,index,43))
    selected={}
    for variant in VARIANTS:
        shortlist=sorted(by_variant[variant],key=rank,reverse=True)[:TOP_N]
        comparisons=[]
        for row in shortlist:
            rs=[row]+[one(variant,row['index'],s) for s in (44,45)]
            comparisons.append(dict(index=row['index'],validation={k:float(np.mean([r['validation'][k] for r in rs])) for k in rs[0]['validation']},seeds=rs))
        best=max(comparisons,key=rank)
        selected[variant]=dict(**best,config=protocol['candidates'][best['index']],shortlist=comparisons,seed43_trials=by_variant[variant])
    champion=max(enumerate(VARIANTS),key=lambda a:(selected[a[1]]['validation']['AUROC'],selected[a[1]]['validation']['HALL_AUPR'],-a[0]))[1]
    atomic_json_save(dict(fingerprint=protocol['fingerprint'],champion=champion,variants=selected,
        frozen_at=datetime.now(timezone.utc).isoformat(),test_accessed=False),root/'selection.json')
    assert done==240
    progress('811单层配置已冻结，等待四模型后评估',status='waiting')
    return protocol


def test_gate():
    for model in MODELS:
        if not (OUT/model/'selection.json').exists():return False
        p=json.loads((OUT/model/'protocol.json').read_text());s=json.loads((OUT/model/'selection.json').read_text())
        assert p['candidates']==candidates() and p['variants']==list(VARIANTS) and s['fingerprint']==p['fingerprint']
    return True


def evaluate(model,device):
    if not test_gate():raise RuntimeError('All-model validation selection gate is not sealed')
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=OUT/model;selection=json.loads((root/'selection.json').read_text());protocol=json.loads((root/'protocol.json').read_text())
    data={p:read(d/model/'matrices.pt') for p,d in SOURCES.items()};rows=[];seeds=[]
    for variant in VARIANTS:
        p,g=variant.split('/');d=data[p];assert d['fingerprint']==protocol['source_fingerprints'][p]
        test=d['masks']['test'];y=d['y'][test];index=selection['variants'][variant]['index'];values=[];probs=[]
        for seed in SEEDS:
            trained=read(root/variant/f'candidate{index:02d}'/f'seed{seed}'/'result.pt')
            assert trained['fingerprint']==selection['fingerprint']
            net=SingleMLP(trained['input_dim'],trained['config']).to(device);net.load_state_dict(trained['state_dict'])
            x=torch.as_tensor(transform(d['groups'][g][test],trained['mean'],trained['scale']),device=device)
            pp=predict(net,x);value=metrics(y,pp);values.append(value);probs.append(pp)
            threshold=select_detection_threshold(d['y'][d['masks']['train']],1-trained['train_probabilities'],positive_class='real')
            final=dict(seed=seed,variant=variant,index=index,fingerprint=selection['fingerprint'],config=trained['config'],
                best_epoch=trained['best_epoch'],validation=trained['validation'],test_metrics=value,test_probabilities=pp,
                threshold=threshold,threshold_reports={name:evaluate_detection_scores(y,1-pp,t,positive_class='real') for name,t in [('train_f1',threshold),('fixed_0.5',.5)]})
            atomic_torch_save(final,root/'final'/variant/f'seed{seed}'/'result.pt')
            seeds.append(dict(model=model,variant=variant,seed=seed,**value))
        rows.append(dict(model=model,variant=variant,champion=variant==selection['champion'],candidate=index,
            **{k+'_'+stat:float(fn([v[k] for v in values])) for k in values[0] for stat,fn in [('mean',np.mean),('std',np.std)]},
            **{'ensemble_'+k:v for k,v in metrics(y,np.mean(probs,axis=0)).items()}))
    prior.original.base.write_csv(rows,root/'detection.csv');prior.original.base.write_csv(seeds,root/'seed_metrics.csv')
    atomic_json_save(dict(stage='811单层搜索及测试完成',completed=24,total=24,status='completed',heartbeat=datetime.now(timezone.utc).isoformat()),root/'progress.json')


def pipeline(model,device):
    root=OUT/model;root.mkdir(parents=True,exist_ok=True)
    with (root/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            _,_,_,protocol=prepare(model)
            if (root/'selection.json').exists():
                assert json.loads((root/'selection.json').read_text())['fingerprint']==protocol['fingerprint']
            else:run_search(model,device)
            while not test_gate():
                atomic_json_save(dict(stage='811单层配置已冻结，等待四模型后评估',completed=240,total=240,status='waiting',heartbeat=datetime.now(timezone.utc).isoformat()),root/'progress.json');time.sleep(10)
            state=json.loads((root/'progress.json').read_text()) if (root/'progress.json').exists() else {}
            if state.get('status')!='completed':evaluate(model,device)
        except BaseException as e:
            atomic_json_save(dict(stage='811单层搜索失败',completed=0,total=240,status='failed',error=str(e)[:180],heartbeat=datetime.now(timezone.utc).isoformat()),root/'progress.json');raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',choices=MODELS,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args();torch.set_num_threads(1)
    if args.prepare_only:prepare(args.model)
    else:pipeline(args.model,args.device)
