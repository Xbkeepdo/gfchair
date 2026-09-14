"""SVAR and MetaToken features with the same 811 detectors and exact mentions."""
import argparse
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import pickle
import sys
import threading
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import train_region_ae_811 as study
from detection.baselines import baseline_vector
from features.baseline import baseline_config, get_baseline_payload, svar_training_vector
from utils.config_utils import load_config
from features.tc_fvpa_artifacts import atomic_json_save, atomic_torch_save

AE_OUT=study.OUT
OUT=ROOT/'outputs/baselines_811_v1'
METHODS=('svar','metatoken')
CLASSIFIERS=('three_hidden','one_hidden','xgb')


def setup():
    study.OUT=OUT


def key(record, mention=False):
    return (int(record['image_id']),int(record['response_index' if mention else 'response_token_idx']),
            int(record['target_token_id']),int(record['label']),record['word' if mention else 'token_str'])


def align_records(records, mentions):
    queues=defaultdict(deque)
    for r in records:
        if r.get('metadata',{}).get('svar_protocol')!='controlled':
            raise ValueError('Expected shared-mention controlled SVAR, not official alternate samples')
        queues[key(r)].append(r)
    ordered=[]
    for m in mentions:
        if not queues[key(m,True)]:raise ValueError(f'Missing baseline mention: {m["mention_id"]}')
        ordered.append(queues[key(m,True)].popleft())
    if any(queues.values()):raise ValueError('Extra baseline mentions')
    return ordered


def prepare(model, output_root=None):
    source=study.original.base.read(AE_OUT/model/'matrices.pt')
    ae_protocol=json.loads((AE_OUT/model/'protocol.json').read_text())
    path=ROOT/'outputs'/model/study.original.base.EXPERIMENT/'baseline/features.pkl'
    with path.open('rb') as stream:records=pickle.load(stream)
    assert len(records)==len(source['mentions'])
    records=align_records(records,source['mentions'])
    cfg=baseline_config(load_config(str(ROOT/'configs/model_configs_inslen_official_target.yaml')))
    start,end=int(cfg['svar']['layer_start']),int(cfg['svar']['layer_end'])
    groups=dict(svar=np.stack([svar_training_vector(get_baseline_payload(r,'svar'),layer_start=start,layer_end=end) for r in records]),
                metatoken=np.stack([baseline_vector(r,'metatoken') for r in records]))
    assert all(x.ndim==2 and np.isfinite(x).all() for x in groups.values())
    np.testing.assert_array_equal(source['y'],[r['label'] for r in records])
    # Duplicate annotations remain separate mentions. Their semantic key must
    # not ambiguously map to different cached feature vectors.
    seen={}
    for i,r in enumerate(records):
        k=key(r)
        if k in seen:
            for method in METHODS:np.testing.assert_array_equal(groups[method][i],groups[method][seen[k]])
        seen[k]=i
    for part,mask in source['masks'].items():
        assert set(np.asarray(source['y'])[mask])=={0,1}
        assert all(m['image_id'] in set(ae_protocol['split'][part]) for i,m in enumerate(source['mentions']) if mask[i])
    names=records[0]['baselines']['metatoken']['feature_names']
    assert all(r['baselines']['metatoken']['feature_names']==names for r in records)
    protocol=dict(model=model,split=ae_protocol['split'],split_mode=ae_protocol['split_mode'],split_seed=ae_protocol['split_seed'],
        ae_fingerprint=source['fingerprint'],source=str(path.relative_to(ROOT)),mentions=len(records),
        counts={k:int(v.sum()) for k,v in source['masks'].items()},svar_layers_zero_based=[start,end],
        svar='Per-layer/per-head visual attention mass, flattened; controlled exact response offsets',
        metatoken_feature_names=names,dimensions={k:x.shape[1] for k,x in groups.items()},
        fixed=asdict(study.TorchProbeConfig(split_protocol='image_811_validation',checkpoint_selection='minimum_val_loss')),
        mlp_grid=study.original.search.GRID,xgb_grid=study.original.search.TREE_GRIDS['xgb'],seeds=list(study.SEEDS),
        scope='Feature comparison using common three-hidden/sklearn-one-hidden/XGB classifiers; not paper-native heads',
        scaling='No added scaling, matching AE 811 classifiers; native MetaToken scaler+LR/GB is not used here',
        information='MetaToken retains full response length and object-span statistics; not strictly pre-target-only',
        selection=ae_protocol['selection'],caveat=ae_protocol['caveat'])
    protocol=json.loads(json.dumps(protocol));digest=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode())
    for k,x in groups.items():digest.update(k.encode());digest.update(x.tobytes())
    protocol['fingerprint']=digest.hexdigest()
    root=(OUT if output_root is None else Path(output_root))/model;path=root/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol,'Baseline 811 protocol changed'
    atomic_json_save(protocol,path)
    data=dict(groups=groups,y=source['y'],masks=source['masks'],mentions=source['mentions'],fingerprint=protocol['fingerprint'])
    path=root/'matrices.pt'
    if not path.exists():atomic_torch_save(data,path)
    print('PREPARED',model,protocol['dimensions'],protocol['counts'],flush=True)
    return data


def search_job(model,method,classifier):
    setup()
    return study.search_group(model,method,classifier)


def summarize(model,data):
    rows=[];seed_rows=[];comparisons=[];root=OUT/model;y=data['y'][data['masks']['test']]
    for method in METHODS:
        for classifier in CLASSIFIERS:
            heads=[study.original.base.read(root/classifier/method/f'seed{s}'/'result.pt') for s in study.SEEDS]
            assert all(h['fingerprint']==data['fingerprint'] for h in heads)
            p=np.stack([h['test_probabilities'] for h in heads]);assert p.shape==(3,len(y)) and np.isfinite(p).all()
            values=[study.scores(y,v) for v in p]
            for seed,v in zip(study.SEEDS,values):seed_rows.append(dict(model=model,group=method,classifier=classifier,seed=seed,**v))
            rows.append(dict(model=model,group=method,classifier=classifier,
                **{k+'_'+stat:float(fn([v[k] for v in values])) for k in values[0] for stat,fn in [('mean',np.mean),('std',np.std)]},
                **{'ensemble_'+k:v for k,v in study.scores(y,p.mean(0)).items()}))
    with (AE_OUT/model/'detection.csv').open() as f:ae=list(csv.DictReader(f))
    for b in rows:
        for a in ae:
            if a['classifier']!=b['classifier'] or a['group'] not in study.original.KINDS+('legacy_visual',):continue
            comparisons.append(dict(model=model,classifier=b['classifier'],ae_group=a['group'],baseline=b['group'],
                AUROC_delta_pp=100*(float(a['AUROC_mean'])-b['AUROC_mean']),
                HALL_AUPR_delta_pp=100*(float(a['HALL_AUPR_mean'])-b['HALL_AUPR_mean'])))
    for name,values in [('detection',rows),('seed_metrics',seed_rows),('ae_vs_baselines',comparisons)]:study.original.base.write_csv(values,root/(name+'.csv'))
    write_report(rows,root/'summary.md')
    study.progress(model,'811 SVAR/MetaToken完成',18,18,status='completed')


def write_report(rows,path):
    lines=['# SVAR/MetaToken：相同811划分与分类器对照','',
        '逐模型严格复用区域AE的3200训练/400验证/400测试图片、全部mentions、原标签及seeds43/44/45。SVAR为controlled样本、零基索引[5,19)层各head视觉attention mass；MetaToken为原10+H维特征。',
        '三层[128,64,32]/BN/dropout.3，验证loss选择checkpoint/调度/早停；sklearn单层12候选、XGB18候选按验证AUROC/AP选参。无额外标准化。这是统一分类器的特征对照，不是SVAR原生248隐藏单层或MetaToken原生Scaler+LR/GB。',
        'MetaToken保留完整回答长度和对象span统计，信息范围比严格pre-target信号更宽。原800图已有研究使用，本轮为探索性评估；无新bootstrap。', '',
        '| 模型 | 特征 | 分类器 | AUROC均值±std | HALL-AUPR均值±std |','|---|---|---|---:|---:|']
    for r in rows:lines.append('| '+r['model']+' | '+r['group']+' | '+r['classifier']+' | '+' | '.join(
        f'{100*float(r[k+"_mean"]):.2f} ± {100*float(r[k+"_std"]):.2f}' for k in ('AUROC','HALL_AUPR'))+' |')
    lines+=['','逐seed及概率ensemble见CSV/保存的result.pt；ae_vs_baselines.csv中的差值为AE组减baseline，始终固定相同分类器及400测试图。正值表示AE组更高，不据此宣称显著性。','']
    lines+=['## 同分类器下与AE拼接直接比较','', '每格AUROC / HALL-AUPR（%，三seed均值）；V、VP、G均指区域AE＋对应All-attention gross的log1p。', '',
        '| 模型 | 分类器 | SVAR | MetaToken | V | VP | G | VP+G |','|---|---|---:|---:|---:|---:|---:|---:|']
    lookup={(r['model'],r['group'],r['classifier']):r for r in rows}
    for model in dict.fromkeys(r['model'] for r in rows):
        with (AE_OUT/model/'detection.csv').open() as f:
            ae={(r['group'],r['classifier']):r for r in csv.DictReader(f)}
        for classifier in CLASSIFIERS:
            values=[lookup[model,m,classifier] for m in METHODS]+[ae[g,classifier] for g in study.original.KINDS]
            lines.append('| '+model+' | '+classifier+' | '+' | '.join(
                f'{100*float(r["AUROC_mean"]):.2f} / {100*float(r["HALL_AUPR_mean"]):.2f}' for r in values)+' |')
    path.write_text('\n'.join(lines))


def pipeline(model,device,workers):
    setup();root=OUT/model;root.mkdir(parents=True,exist_ok=True)
    with (root/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            study.progress(model,'811基线准备',0,18)
            while True:
                p=AE_OUT/model/'progress.json';state=json.loads(p.read_text()) if p.exists() else {}
                if state.get('status')=='completed':break
                study.progress(model,'基线等待AE811：'+state.get('stage','准备'),state.get('completed',0),state.get('total',1),status='waiting')
                time.sleep(10)
            data=prepare(model);study.fixed(model,device,data)
            stop=threading.Event()
            def monitor():
                while not stop.is_set():
                    n=sum(len(list((root/c).glob('*/search/trial*.json'))) for c in ('one_hidden','xgb'))
                    h=sum(len(list((root/c).glob('*/seed*/result.pt'))) for c in ('one_hidden','xgb'))
                    study.progress(model,f'811基线选参 {n}/60 正式 {h}/12',n+h,72);stop.wait(10)
            thread=threading.Thread(target=monitor,daemon=True);thread.start()
            try:
                with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn')) as pool:
                    jobs=[pool.submit(search_job,model,m,c) for c in ('one_hidden','xgb') for m in METHODS]
                    for job in as_completed(jobs):print('DONE',job.result(),flush=True)
            finally:stop.set();thread.join()
            summarize(model,data)
        except BaseException as exc:
            study.progress(model,'811基线失败',0,1,status='failed',error=str(exc)[:180]);raise


def summarize_all():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'.summary.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            done=[m for m in study.original.MODELS if (OUT/m/'progress.json').exists() and
                  json.loads((OUT/m/'progress.json').read_text()).get('status')=='completed']
            atomic_json_save(dict(status='waiting',completed_models=done,heartbeat=datetime.now(timezone.utc).isoformat()),OUT/'summary_progress.json')
            if len(done)==4:break
            time.sleep(10)
        for name in ('detection','seed_metrics','ae_vs_baselines'):
            rows=[]
            for m in study.original.MODELS:
                with (OUT/m/(name+'.csv')).open() as f:rows.extend(csv.DictReader(f))
            study.original.base.write_csv(rows,OUT/(name+'.csv'))
            if name=='detection':
                assert len(rows)==24
                write_report(rows,OUT/'summary.md');write_report(rows,ROOT/'docs/SVAR_METATOKEN_811_RESULTS.md')
        atomic_json_save(dict(status='completed',heads=72,heartbeat=datetime.now(timezone.utc).isoformat()),OUT/'summary_progress.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=study.original.MODELS)
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--summarize-all',action='store_true')
    args=parser.parse_args();torch.set_num_threads(1)
    if args.summarize_all:summarize_all()
    elif args.model:pipeline(args.model,args.device,args.workers)
    else:parser.error('--model or --summarize-all is required')
