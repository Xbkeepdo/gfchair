"""Regional AE + true-RMS all-attention gross: fixed MLP, shallow MLP, XGB."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import sys
import threading

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import train_attention_log_strength_mlp as search
from scripts import run_ffn_source_composition as base
from scripts.analyze_all_source_paths import MODELS,LAYERS,F1_FIELDS
from scripts.train_torch_probe_feature_sets import TorchProbeConfig,train_and_evaluate_probe
from scripts.train_ffn_ae_log1p_search import summarize_group
from scripts.train_attention_strength_fusion import metrics
from features.tc_fvpa_artifacts import atomic_json_save,atomic_torch_save

OUT=ROOT/'outputs/all_attention_ae_log_strength_v1'
KINDS=('visual','visual_prompt_sum','generation','vp_generation')
GROUPS=KINDS+tuple('ae_only_'+k for k in KINDS)
SEEDS=(43,44,45)


def build_groups(ae,strength):
    """Keep AE raw; VP gross is summed before log; undefined AE -> zero only here."""
    if any(np.isinf(x).any() for x in ae.values()):
        raise ValueError('Infinite AE')
    a={k:np.nan_to_num(np.asarray(ae[k],dtype=np.float32),nan=0.) for k in KINDS[:3]}
    s={k:np.asarray(strength[k],dtype=np.float64) for k in ('visual','prompt','generation')}
    if any(not np.isfinite(x).all() or (x<0).any() for x in s.values()):raise ValueError('Invalid gross')
    s['visual_prompt_sum']=s['visual']+s['prompt']
    groups={k:np.concatenate((a[k],np.log1p(s[k]).astype(np.float32)),axis=1) for k in KINDS[:3]}
    groups['vp_generation']=np.concatenate((groups['visual_prompt_sum'],groups['generation']),axis=1)
    groups.update({'ae_only_'+k:a[k] for k in KINDS[:3]})
    groups['ae_only_vp_generation']=np.concatenate((a['visual_prompt_sum'],a['generation']),axis=1)
    if any(not np.isfinite(x).all() for x in groups.values()):raise ValueError('Nonfinite detector matrix')
    return groups


def prepare(model):
    dest=OUT/model
    ref=base.read(ROOT/'outputs/ffn_all_source_paths_v1'/model/'matrices.pt')
    n,y,mentions=ref['ntrain'],ref['y'],ref['mentions'];layers=LAYERS[model]
    split=json.loads((ROOT/'outputs'/model/base.EXPERIMENT/'image_splits.json').read_text())
    expected=set(split['train']+split['test']);assert len(expected)==4000
    shards=list((dest/'ae_shards').glob('image_*.pt'));assert len(shards)==4000
    values={};signatures=set();seen=set()
    for path in shards:
        shard=base.read(path);assert shard['complete'] and shard['image_id'] not in seen
        seen.add(shard['image_id']);signatures.add(shard.get('signature',shard.get('protocol_signature')))
        for row in shard['positions']:
            key=row['target_key'];assert key not in values
            values[key]=row
    assert seen==expected and len(signatures)==1 and None not in signatures
    assert set(values)=={m['target_key'] for m in mentions}
    for m in mentions:
        row=values[m['target_key']]
        assert row['response_index']==m['response_index'] and row['target_token_id']==m['target_token_id']
    ae={k:np.stack([np.asarray(values[m['target_key']]['ae'][k],dtype=np.float32) for m in mentions]) for k in KINDS[:3]}
    assert all(x.shape==(len(y),layers) for x in ae.values())
    assert all(not np.isinf(x).any() and ((x[np.isfinite(x)]>=0)&(x[np.isfinite(x)]<=1+1e-6)).all() for x in ae.values())
    empty=np.array([m['response_index']==0 for m in mentions])
    assert np.isnan(ae['generation'][empty]).all() and np.isfinite(ae['generation'][~empty]).all()
    strength={s:ref['groups']['F1_raw'][:,F1_FIELDS.index(s+'_gross_norm')*layers:(F1_FIELDS.index(s+'_gross_norm')+1)*layers] for s in ('visual','prompt','generation')}
    groups=build_groups(ae,strength)
    inner=search.inner_image_split(split['train'],split['test'])
    ids=np.asarray([m['image_id'] for m in mentions]);fit=np.isin(ids,inner['train']);val=np.isin(ids,inner['validation'])
    assert not (fit&val).any() and not fit[n:].any() and not val[n:].any() and (fit[:n]|val[:n]).all()
    digest=hashlib.sha256()
    for k,x in sorted(groups.items()):digest.update(k.encode());digest.update(x.tobytes())
    digest.update(np.asarray(y).tobytes());digest.update(json.dumps([m['mention_id'] for m in mentions]).encode())
    signature=digest.hexdigest()
    matrix=dict(groups=groups,y=y,ntrain=n,mentions=mentions,inner_fit=fit,inner_val=val,fingerprint=signature)
    path=dest/'matrices.pt'
    if path.exists():assert base.read(path)['fingerprint']==signature,'Changed features; refusing stale heads'
    else:atomic_torch_save(matrix,path)
    old=base.read(ROOT/'outputs/ffn_source_composition_v1'/model/'matrices.pt')
    assert [m['mention_id'] for m in old['mentions']]==[m['mention_id'] for m in mentions]
    old_ae=old['groups']['F'][:,:layers]
    diff=np.abs(ae['visual']-old_ae)
    np.savez_compressed(dest/'signals.npz',**{'ae_'+k:v for k,v in ae.items()},**{'gross_'+k:v for k,v in strength.items()})
    protocol=dict(model=model,images=4000,train_images=3200,test_images=800,mentions=len(y),ntrain=n,
        fingerprint=signature,ae_signature=next(iter(signatures)),all_attention_signature=ref['protocol_signature'],
        feature_groups=list(GROUPS),dimensions={k:x.shape[1] for k,x in groups.items()},
        ae='per-region normalized native attention x sigmoid((hpre target logit - region median)/(1.4826 region MAD+epsilon)); raw AE',
        strength='All-attention true RMS path z-Aall -> z, K32 token gross; VP=sum gross then log1p; never B2 K4',
        undefined='Empty generation AE=NaN in signals, zero in detector; preserve every mention',
        old_visual_ae_comparison=dict(max_absolute=float(diff.max()),mean_absolute=float(diff.mean())),
        image_split=inner,inner_seed=search.INNER_SEED,mlp_grid=search.GRID,xgb_grid=search.TREE_GRIDS['xgb'],
        seeds=SEEDS,selection='inner 2560/640 images seed43 AUROC then HALL_AP then candidate index; refit3200 seeds43/44/45; no test selection',
        controls='AE-only on same regions, each with all three classifiers; no scaler; not a depth-only ablation')
    atomic_json_save(protocol,dest/'protocol.json')
    print('PREPARED',model,'visual AE vs old',protocol['old_visual_ae_comparison'],flush=True)


def progress(model,stage,done,total,status='running',**kwargs):
    atomic_json_save(dict(stage=stage,completed=done,total=total,status=status,
        heartbeat=datetime.now(timezone.utc).isoformat(),**kwargs),OUT/model/'progress.json')


def fixed(model,device):
    dest=OUT/model;data=base.read(dest/'matrices.pt');n,y=data['ntrain'],data['y'];results={};done=0
    for name,x in data['groups'].items():
        heads=[]
        for seed in SEEDS:
            folder=dest/'heads'/name/f'seed{seed}';path=folder/'result.pt'
            if path.exists():value=base.read(path)
            else:
                def callback(epoch):progress(model,'三层MLP '+name,done,24,epoch=epoch)
                result=train_and_evaluate_probe(X_train=x[:n],y_train=y[:n],X_test=x[n:],y_test=y[n:],
                    X_val=np.empty((0,x.shape[1]),np.float32),y_val=np.empty(0,np.int32),
                    config=TorchProbeConfig(seed=seed),device=torch.device(device),output_dir=str(folder),
                    return_probabilities=True,epoch_callback=callback)
                value=dict(seed=seed,train_probabilities=np.asarray(result.pop('train_probabilities')),
                    test_probabilities=np.asarray(result.pop('test_probabilities')),metrics=result)
                atomic_torch_save(value,path)
            heads.append(value);done+=1
            progress(model,'三层MLP '+name,done,24)
        results[name]=summarize_group(dict(y_train=y[:n],y_test=y[n:]),heads)
        atomic_json_save(results,dest/'detection.json')
    print('FIXED COMPLETE',model,flush=True)


def search_job(model,group,classifier):
    search.OUT=OUT
    return search.search_group(model,group,classifier)


def searched(model,workers):
    stop=threading.Event()
    def update():
        while not stop.is_set():
            root=OUT/model
            trials=sum(len(list((root/folder).glob('*/search/trial*.json'))) for folder in ('one_hidden','xgb'))
            heads=sum(len(list((root/folder).glob('*/seed*/result.pt'))) for folder in ('one_hidden','xgb'))
            progress(model,f'单层/XGB 选参 {trials}/240 正式 {heads}/48',trials+heads,288)
            stop.wait(10)
    thread=threading.Thread(target=update,daemon=True);thread.start()
    try:
        with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn')) as pool:
            jobs=[pool.submit(search_job,model,g,c) for c in ('mlp','xgb') for g in GROUPS]
            for job in as_completed(jobs):print('SEARCH COMPLETE',json.dumps(job.result()),flush=True)
    finally:stop.set();thread.join()


def summarize(models):
    rows=[];selected=[];comparisons=[]
    for model in models:
        root=OUT/model;fixed_results=json.loads((root/'detection.json').read_text())
        for group in GROUPS:
            for classifier in ('three_hidden','one_hidden','xgb'):
                result=fixed_results[group] if classifier=='three_hidden' else json.loads((root/classifier/group/'detection.json').read_text())
                rows.append(dict(model=model,group=group,classifier=classifier,**metrics(result)))
                if classifier!='three_hidden':
                    best=json.loads((root/classifier/group/'selection.json').read_text())['best']
                    selected.append(dict(model=model,group=group,classifier=classifier,params=json.dumps(best['params']),validation_AUROC=best['validation_AUROC']))
        progress(model,'全部检测完成',72,72,status='completed')
    lookup={(r['model'],r['group'],r['classifier']):r for r in rows}
    for model in models:
        for group in KINDS:
            for classifier in ('three_hidden','one_hidden','xgb'):
                a,b=lookup[model,group,classifier],lookup[model,'ae_only_'+group,classifier]
                comparisons.append(dict(model=model,group=group,classifier=classifier,reference='corresponding AE only',
                    AUROC_delta=a['AUROC_mean']-b['AUROC_mean'],HALL_AUPR_delta=a['HALL_AUPR_mean']-b['HALL_AUPR_mean']))
    # Model jobs save independent summaries; only --stage summarize writes shared tables.
    destination=OUT if len(models)==4 else OUT/models[0]
    base.write_csv(rows,destination/'detection.csv');base.write_csv(selected,destination/'selected_params.csv')
    base.write_csv(comparisons,destination/'comparisons.csv')
    lines=['# 区域 AE + All-attention gross：三层MLP / 单层MLP / XGB','',
        '原4000图3200/800、全部mentions、三seed均值±总体std；每域独立MAD gate与attention归一化，强度为真实RMS All-attention K32 gross。',
        '单层MLP 12候选、XGB18候选；训练内2560/640选参，原800图最终评估。AE-only为相同分类器对照；仅S取log1p，无标准化。','',
        '| 模型 | 特征 | 分类器 | AUROC | HALL-AUPR |','|---|---|---|---:|---:|']
    for r in rows:
        lines.append('| '+r['model']+' | '+r['group']+' | '+r['classifier']+' | '+
            ' | '.join(f'{100*r[k+"_mean"]:.3f} ± {100*r[k+"_std"]:.3f}' for k in ('AUROC','HALL_AUPR'))+' |')
    (destination/'summary.md').write_text('\n'.join(lines)+'\n')


def pipeline(model,device,workers):
    (OUT/model).mkdir(parents=True,exist_ok=True)
    with (OUT/model/'.pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            from scripts.extract_region_ae import extract
            stop=threading.Event();state={'smoke':True}
            def extraction_update():
                while not stop.is_set():
                    folder=OUT/model/('smoke8' if state['smoke'] else '')
                    path=folder/'extraction_progress.json'
                    value=json.loads(path.read_text()) if path.exists() else {}
                    progress(model,'区域AE核对' if state['smoke'] else '区域AE提取',
                        value.get('completed',0),8 if state['smoke'] else 4000,
                        error=value.get('error',''))
                    stop.wait(10)
            thread=threading.Thread(target=extraction_update,daemon=True);thread.start()
            try:
                extract(model,device,smoke=True)
                state['smoke']=False
                extract(model,device,smoke=False)
            finally:stop.set();thread.join()
            prepare(model);fixed(model,device);searched(model,workers);summarize([model])
        except Exception as exc:
            progress(model,'失败',0,1,status='failed',error=str(exc)[:160]);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('prepare','fixed','search','pipeline','summarize'),required=True)
    parser.add_argument('--models',nargs='+',choices=MODELS,default=MODELS)
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    torch.set_num_threads(1)
    if args.stage=='summarize':summarize(args.models)
    else:
        for model in args.models:
            if args.stage=='pipeline':pipeline(model,args.device,args.workers)
            elif args.stage=='prepare':prepare(model)
            elif args.stage=='fixed':fixed(model,args.device)
            else:searched(model,args.workers)
