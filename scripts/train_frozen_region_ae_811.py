"""Same regional AE 811 detectors, replacing true-RMS gross with frozen-RMS K50 gross."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import sys
import threading
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import train_region_ae_811 as study
from features.tc_fvpa_artifacts import atomic_json_save, atomic_torch_save

AE_OUT=study.OUT
OUT=ROOT/'outputs/frozen_region_ae_811_v1'
SOURCE=ROOT/'outputs/ffn_source_composition_v1'
GROUPS=study.original.KINDS
CLASSIFIERS=('three_hidden','one_hidden','xgb')


def setup():
    study.OUT=OUT


def prepare(model):
    ae=study.original.base.read(AE_OUT/model/'matrices.pt')
    source=study.original.base.read(SOURCE/'source_strength_detection'/model/'matrices.pt')
    assert ae['mentions']==source['mentions']
    np.testing.assert_array_equal(ae['y'],source['y'])
    protocol_ae=json.loads((AE_OUT/model/'protocol.json').read_text())
    numerical=json.loads((SOURCE/model/'analysis.json').read_text())
    selection=json.loads((SOURCE/'numerics.json').read_text())
    assert selection['k']==50 and numerical['images']==4000
    assert numerical['quadrature_relative_max']<.01, 'Frozen K50 pure quadrature failed'
    assert len(list((SOURCE/model/'shards/k50').glob('image_*.pt')))==4000
    strength={k:source['groups']['raw_'+k] for k in ('visual','prompt','generation')}
    regional={k:ae['groups']['ae_only_'+k] for k in GROUPS[:3]}
    assert all(np.isfinite(v).all() and (v>=0).all() for v in strength.values())
    empty=np.array([m['response_index']==0 for m in ae['mentions']])
    assert np.all(strength['generation'][empty]==0)
    combined=study.original.build_groups(regional,strength)
    for k in GROUPS:
        np.testing.assert_array_equal(combined['ae_only_'+k],ae['groups']['ae_only_'+k])
    groups={k:combined[k] for k in GROUPS}
    assert all(groups[k].shape==ae['groups'][k].shape for k in GROUPS)
    protocol=dict(model=model,split=protocol_ae['split'],split_mode=protocol_ae['split_mode'],
        split_seed=protocol_ae['split_seed'],counts={k:int(v.sum()) for k,v in ae['masks'].items()},seeds=list(study.SEEDS),
        fixed=protocol_ae['fixed'],mlp_grid=protocol_ae['mlp_grid'],xgb_grid=protocol_ae['xgb_grid'],
        selection=protocol_ae['selection'],thresholds=protocol_ae['thresholds'],caveat=protocol_ae['caveat'],
        source_ae_fingerprint=ae['fingerprint'],source_strength=str(SOURCE/'source_strength_detection'/model/'matrices.pt'),
        features=list(GROUPS),dimensions={k:x.shape[1] for k,x in groups.items()},
        ae='Exact same existing per-region AE arrays as true-RMS 811; no full-prefix gated-mass substitution',
        strength='Frozen endpoint RMS D(z); c_j=integral J_FFN(alpha*Norm(z)) D(z)a_j d_alpha; S_D=sum_j_in_D norm(c_j); K50 localFP32',
        full_sources='attention token writes, residual, attention output bias; residual/bias participate in path but are not additional detector features',
        vp='concat(AE_VP,log1p(S_V+S_P)); VP+G concatenates complete VP and G blocks',
        scaling='Only gross uses log1p; no added scaler; empty generation is zero; all mentions retained',
        numerical=numerical,quadrature_selection=selection['models'][model],
        limitation='Historical K50 pure quadrature passed; native-precision source reconstruction residual remains. Reuse cached captures; do not claim exact source closure or a fresh identical-capture numerical ablation.')
    protocol=json.loads(json.dumps(protocol));digest=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode())
    for k,x in groups.items():digest.update(k.encode());digest.update(x.tobytes())
    protocol['fingerprint']=digest.hexdigest();root=OUT/model
    if (root/'protocol.json').exists():assert json.loads((root/'protocol.json').read_text())==protocol,'Frozen protocol changed'
    atomic_json_save(protocol,root/'protocol.json')
    data=dict(groups=groups,y=ae['y'],masks=ae['masks'],mentions=ae['mentions'],fingerprint=protocol['fingerprint'])
    if (root/'matrices.pt').exists():assert study.original.base.read(root/'matrices.pt')['fingerprint']==data['fingerprint']
    else:atomic_torch_save(data,root/'matrices.pt')
    print('PREPARED',model,protocol['dimensions'],'pure_integral_max',numerical['quadrature_relative_max'],'closure_max',numerical['closure_max'],flush=True)
    return data


def search_job(model,group,classifier):
    setup();return study.search_group(model,group,classifier)


def summarize(model,data):
    rows=[];seeds=[];comparisons=[];y=data['y'][data['masks']['test']];root=OUT/model
    true=study.original.base.read(AE_OUT/model/'matrices.pt')
    assert data['mentions']==true['mentions']
    np.testing.assert_array_equal(data['y'],true['y'])
    for part in data['masks']:np.testing.assert_array_equal(data['masks'][part],true['masks'][part])
    for group in GROUPS:
        for clf in CLASSIFIERS:
            heads=[study.original.base.read(root/clf/group/f'seed{s}'/'result.pt') for s in study.SEEDS]
            assert all(h['fingerprint']==data['fingerprint'] for h in heads)
            p=np.stack([h['test_probabilities'] for h in heads]);assert p.shape==(3,len(y)) and np.isfinite(p).all()
            v=[study.scores(y,a) for a in p]
            seeds.extend(dict(model=model,group=group,classifier=clf,seed=s,**m) for s,m in zip(study.SEEDS,v))
            row=dict(model=model,group=group,classifier=clf,
                **{k+'_'+stat:float(fn([m[k] for m in v])) for k in v[0] for stat,fn in [('mean',np.mean),('std',np.std)]},
                **{'ensemble_'+k:value for k,value in study.scores(y,p.mean(0)).items()})
            rows.append(row)
            originals=[study.original.base.read(AE_OUT/model/clf/group/f'seed{s}'/'result.pt') for s in study.SEEDS]
            assert all(h['fingerprint']==true['fingerprint'] for h in originals)
            t=[study.scores(y,h['test_probabilities']) for h in originals]
            comparisons.append(dict(model=model,group=group,classifier=clf,
                **{k+'_true_mean':float(np.mean([m[k] for m in t])) for k in v[0]},
                **{k+'_frozen_mean':row[k+'_mean'] for k in v[0]},
                **{k+'_delta_pp':100*(row[k+'_mean']-np.mean([m[k] for m in t])) for k in v[0]}))
    for name,values in [('detection',rows),('seed_metrics',seeds),('vs_true_rms',comparisons)]:
        study.original.base.write_csv(values,root/(name+'.csv'))
    study.progress(model,'811冻结RMS完成',36,36,status='completed')


def pipeline(model,device,workers):
    setup();root=OUT/model;root.mkdir(parents=True,exist_ok=True)
    with (root/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            study.progress(model,'811冻结RMS对齐检查',0,36)
            data=prepare(model);study.fixed(model,device,data)
            stop=threading.Event()
            def monitor():
                while not stop.is_set():
                    trials=sum(len(list((root/c).glob('*/search/trial*.json'))) for c in ('one_hidden','xgb'))
                    heads=sum(len(list((root/c).glob('*/seed*/result.pt'))) for c in ('one_hidden','xgb'))
                    study.progress(model,f'811冻结选参 {trials}/120 正式 {heads}/24',trials+heads,144);stop.wait(10)
            thread=threading.Thread(target=monitor,daemon=True);thread.start()
            try:
                with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn')) as pool:
                    jobs=[pool.submit(search_job,model,g,c) for g in GROUPS for c in ('one_hidden','xgb')]
                    for job in as_completed(jobs):print('DONE',job.result(),flush=True)
            finally:stop.set();thread.join()
            summarize(model,data)
        except BaseException as exc:
            study.progress(model,'811冻结RMS失败',0,36,status='failed',error=str(exc)[:180]);raise


def report():
    rows=[];seeds=[];pairs=[]
    for model in study.original.MODELS:
        for name,values in [('detection',rows),('seed_metrics',seeds),('vs_true_rms',pairs)]:
            with (OUT/model/(name+'.csv')).open() as f:values.extend(csv.DictReader(f))
    assert len(rows)==48 and len(seeds)==144 and len(pairs)==48
    for name,values in [('detection',rows),('seed_metrics',seeds),('vs_true_rms',pairs)]:
        study.original.base.write_csv(values,OUT/(name+'.csv'))
    lines=['# 冻结RMS完整来源分解：区域AE＋gross的811对照','',
        '设置：图片3200训练/400验证/400测试；保留原3200训练，原800按seed20260912分半。全部mentions；seeds43/44/45。指标AUROC/HALL-AUPR为三seed指标均值，另存std与概率ensemble。',
        '区域AE与真实Norm 811逐值相同；只将S替换为旧K50冻结端点RMS的全来源FFN积分gross。V=[AE_V,log1p(S_V)]；VP=[AE_VP,log1p(S_V+S_P)]；G=[AE_G,log1p(S_G)]；VP+G拼接完整后两块。残差/bias参与全分解，未额外进入这些检测特征。',
        '归因定义差异：真实Norm是integral J_(FFN∘Norm)(z-A_all+alpha*A_all)a_j；冻结全分解是integral J_FFN(alpha*Norm(z))D(z)a_j。后者不仅冻结RMS，还将共同路径基线改为FFN的零输入，因此本比较不是仅删除RMS导数的单因素消融。',
        '三层128/64/32、BN/dropout.3、Adam .001、wd1e-5、batch256/max100，验证loss选checkpoint/调度/早停；单层sklearn12候选、XGB18候选，seed43验证AUROC/AP选参。均只训练3200图片，不加标准化。',
        '数值边界：K50纯积分诊断通过，来源重构/总闭合仍有原生精度误差，protocol逐模型保存；不使用B2 K4，也不宣称全部source闭合。复用历史capture；本次是缓存特征对照。旧800图曾被查看，无新bootstrap，不宣称显著性。','',
        '| 模型 | 分类器 | V | VP | G | VP+G |','|---|---|---:|---:|---:|---:|']
    lookup={(r['model'],r['group'],r['classifier']):r for r in rows}
    for model in study.original.MODELS:
        for clf in CLASSIFIERS:
            values=[lookup[model,g,clf] for g in GROUPS]
            lines.append('| '+model+' | '+clf+' | '+' | '.join(f"{100*float(r['AUROC_mean']):.2f} / {100*float(r['HALL_AUPR_mean']):.2f}" for r in values)+' |')
    lines+=['','## 冻结RMS减真实Norm的差值','', '保持相同特征组、分类器、测试mentions及seed均值口径，单位百分点。', '',
            '| 模型 | 组 | 分类器 | ΔAUROC | ΔHALL-AUPR |','|---|---|---|---:|---:|']
    for r in pairs:lines.append('| '+r['model']+' | '+r['group']+' | '+r['classifier']+f" | {float(r['AUROC_delta_pp']):+.2f} | {float(r['HALL_AUPR_delta_pp']):+.2f} |")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,8))
    index=[(m,c) for m in study.original.MODELS for c in CLASSIFIERS]
    paired={(r['model'],r['group'],r['classifier']):r for r in pairs}
    for ax,metric in zip(axes,('AUROC','HALL_AUPR')):
        values=np.array([[float(paired[m,g,c][metric+'_delta_pp']) for g in GROUPS] for m,c in index])
        limit=max(float(np.abs(values).max()),.01)
        plot=ax.imshow(values,cmap='RdBu',vmin=-limit,vmax=limit,aspect='auto')
        ax.set_xticks(range(4),['V','VP','G','VP+G'])
        ax.set_yticks(range(len(index)),[m.replace('_vl_7b','').replace('_vl_8b','')+' / '+c for m,c in index],fontsize=8)
        ax.set_title(metric+' change (percentage points)')
        for i in range(values.shape[0]):
            for j in range(4):ax.text(j,i,f'{values[i,j]:+.2f}',ha='center',va='center',fontsize=8,
                                      color='white' if abs(values[i,j])>.65*limit else 'black')
        fig.colorbar(plot,ax=ax,shrink=.7)
    fig.suptitle('Frozen RMS K50 minus true Norm K32 | image 811 | three-seed metric means')
    fig.tight_layout();fig.savefig(OUT/'delta_heatmap.png',dpi=180);fig.savefig(OUT/'delta_heatmap.pdf');plt.close(fig)
    lines+=['','差值热图：正值表示冻结版分数更高；色块不是显著性标记。',
            '![冻结减真实Norm](../outputs/frozen_region_ae_811_v1/delta_heatmap.png)']
    for path in (OUT/'summary.md',ROOT/'docs/FROZEN_REGION_AE_811_RESULTS.md'):
        content='\n'.join(lines)+'\n'
        if path.parent==OUT:content=content.replace('../outputs/frozen_region_ae_811_v1/delta_heatmap.png','delta_heatmap.png')
        path.write_text(content)
    atomic_json_save(dict(status='completed',heads=144,heartbeat=datetime.now(timezone.utc).isoformat()),OUT/'summary_progress.json')


def wait_report():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'.summary.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            done=[];failed=[]
            for model in study.original.MODELS:
                path=OUT/model/'progress.json';s=json.loads(path.read_text()) if path.exists() else {}
                if s.get('status')=='completed':done.append(model)
                if s.get('status')=='failed':failed.append(model)
            atomic_json_save(dict(status='waiting',completed_models=done,failed_models=failed,heartbeat=datetime.now(timezone.utc).isoformat()),OUT/'summary_progress.json')
            if len(done)==4:break
            time.sleep(10)
        report()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',choices=study.original.MODELS)
    p.add_argument('--device',default='cpu');p.add_argument('--workers',type=int,default=4)
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--summarize-all',action='store_true')
    args=p.parse_args();torch.set_num_threads(1)
    if args.summarize_all:wait_report()
    elif args.model:
        if args.prepare_only:prepare(args.model)
        else:pipeline(args.model,args.device,args.workers)
    else:p.error('--model or --summarize-all required')
