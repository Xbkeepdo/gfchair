"""Cached generation gross / response_index versus gross and length controls."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from features.tc_fvpa_artifacts import atomic_json_save, atomic_torch_save
from scripts import analyze_all_attention_write_gain as prior
from scripts.summarize_all_attention_write_gain import weighted_scores, read_csv

OUT = ROOT/'outputs/generation_per_position_811_v1'
MODELS, SEEDS, CONFIG = prior.MODELS, prior.SEEDS, dict(prior.CONFIG)
TRAINER = prior.trainer
FEATURES = ('S_G', 'S_G_per_position', 'position')
PAIRS = (('S_G_per_position', 'S_G'), ('S_G_per_position', 'position'), ('S_G', 'position'))


def build_features(s, position):
    s, position = np.asarray(s, dtype=np.float64), np.asarray(position, dtype=np.float64)
    if s.ndim != 2 or position.shape != (len(s),): raise ValueError('Unaligned shapes')
    if not np.isfinite(s).all() or not np.isfinite(position).all() or (s < 0).any() or (position < 0).any() or (position != np.floor(position)).any():
        raise ValueError('Invalid strength or token position')
    ratio = np.full_like(s, np.nan)
    np.divide(s, position[:, None], out=ratio, where=position[:, None] > 0)
    groups = dict(S_G=np.log1p(s).astype(np.float32),
                  S_G_per_position=np.log1p(np.where(np.isfinite(ratio), ratio, 0.)).astype(np.float32),
                  position=np.log1p(position[:, None]).astype(np.float32))
    return ratio, groups


def progress(model, stage, completed, **kwargs):
    atomic_json_save(dict(stage=stage, completed=completed, total=9,
        heartbeat=datetime.now(timezone.utc).isoformat(), **kwargs), OUT/model/'progress.json')


def prepare(model):
    source = TRAINER.read(prior.OUT/model/'statistics.pt')
    old_protocol = json.loads((prior.OUT/model/'protocol.json').read_text())
    assert source['fingerprint'] == old_protocol['fingerprint']
    v = source['values']
    digest = hashlib.sha256(v.tobytes()).hexdigest()
    assert digest == json.loads((prior.OUT/'statistics_validation.json').read_text())[model]['statistics_sha256']
    s = v[:, :, prior.REGIONS.index('generation'), prior.FIELDS.index('S')]
    position = np.asarray([m['response_index'] for m in source['mentions']])
    ref = TRAINER.read(TRAINER.SOURCES['true_rms']/model/'matrices.pt')
    assert source['mentions'] == ref['mentions']
    np.testing.assert_array_equal(source['y'], ref['y'])
    for part in source['masks']:
        np.testing.assert_array_equal(source['masks'][part], ref['masks'][part])
        np.testing.assert_array_equal(source['masks'][part], np.isin([m['image_id'] for m in source['mentions']], old_protocol['split'][part]))
    np.testing.assert_allclose(s, np.expm1(ref['groups']['generation'][:, s.shape[1]:]), rtol=2e-6, atol=2e-6)
    ratio, groups = build_features(s, position)
    protocol = dict(version=1, model=model, source_fingerprint=source['fingerprint'], statistics_sha256=digest,
        split=old_protocol['split'], config=CONFIG, seeds=list(SEEDS), features=list(FEATURES),
        definitions='S_G=sum generation-token ||e_m||; true-RMS All-attention K32 z-A_all->z; position=response_index=N_G',
        transform='log1p(S_G), log1p(S_G/position), log1p(position); per-column train-only Z-score',
        zero_position='raw ratio NaN; detector value zero; retain every mention',
        dimensions={k:x.shape[1] for k,x in groups.items()},
        diagnostics=dict(mentions=len(s), zero_position=int((position==0).sum()),
            min_position=int(position.min()), max_position=int(position.max()), undefined_ratio=int(np.isnan(ratio).sum())),
        bootstrap=dict(replicates=2000, seed=20260915, pairs=PAIRS, unit='400 test images including empty images; mean seed metric differences'),
        caveat='Previously inspected test; exploratory; nominal CIs without multiple-comparison correction; position-only has one input column')
    protocol = json.loads(json.dumps(protocol))
    h = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode())
    for name, x in groups.items():h.update(name.encode());h.update(x.tobytes())
    protocol['fingerprint'] = h.hexdigest()
    path = OUT/model/'protocol.json'
    if path.exists(): assert json.loads(path.read_text()) == protocol, 'Changed protocol'
    atomic_json_save(protocol, path)
    if not (OUT/model/'features.pt').exists():
        atomic_torch_save(dict(groups=groups, raw_s=s.copy(), raw_ratio=ratio, position=position,
            mentions=source['mentions'], y=source['y'], masks=source['masks'], fingerprint=protocol['fingerprint']), OUT/model/'features.pt')
    return source, groups, ratio, protocol


def run_model(model, device):
    torch.set_num_threads(1)
    data, groups, ratio, protocol = prepare(model)
    y, masks = data['y'], data['masks']
    rows, checks, results = [], [], {}
    for group, x in groups.items():
        for seed in SEEDS:
            path = OUT/model/group/f'seed{seed}.pt'
            if path.exists():
                result = TRAINER.read(path)
                assert result['fingerprint']==protocol['fingerprint'] and result['config']==CONFIG
            else:
                result = TRAINER.fit(x[masks['train']], y[masks['train']], x[masks['validation']], y[masks['validation']],
                    CONFIG, seed, device, callback=lambda epoch:progress(model, group+f' seed{seed}', len(rows), epoch=epoch))
                net = TRAINER.SingleMLP(x.shape[1], CONFIG).to(device);net.load_state_dict(result['state_dict'])
                p = TRAINER.predict(net, torch.as_tensor(TRAINER.transform(x[masks['test']],result['mean'],result['scale']),device=device))
                result.update(fingerprint=protocol['fingerprint'],test_probabilities=p,test=TRAINER.metrics(y[masks['test']],p))
                atomic_torch_save(result,path)
            net = TRAINER.SingleMLP(x.shape[1],CONFIG);net.load_state_dict(result['state_dict'])
            mean,scale=TRAINER.scale_fit(x[masks['train']],True)
            np.testing.assert_array_equal(mean,result['mean']);np.testing.assert_array_equal(scale,result['scale'])
            assert result['best_epoch']==1+np.argmin([r['val_loss'] for r in result['history']])
            error=0.
            for part in ('train','validation','test'):
                p=TRAINER.predict(net,torch.as_tensor(TRAINER.transform(x[masks[part]],mean,scale)))
                saved=result[part+'_probabilities'];np.testing.assert_allclose(p,saved,rtol=1e-5,atol=1e-6)
                error=max(error,float(np.max(abs(p-saved))))
                if part!='train':
                    for k,value in TRAINER.metrics(y[masks[part]],saved).items():np.testing.assert_allclose(value,result[part][k],atol=1e-12)
            rows.append(dict(model=model,feature=group,seed=seed,input_dim=x.shape[1],best_epoch=result['best_epoch'],epochs=result['epochs'],**result['test']))
            checks.append(dict(feature=group,seed=seed,status='PASS',max_probability_error=error))
            results[group,seed]=result
            prior.write_csv(OUT/model/'detection.csv',rows);atomic_json_save(checks,OUT/model/'validation.json')
            print(model,group,seed,result['test'],flush=True)
    curve_rows=[]
    for scope,mask in dict(all=np.ones(len(y),dtype=bool),**masks).items():
        for label,name in ((1,'REAL'),(0,'HALL')):
            for layer,col in enumerate(ratio[mask & (y==label)].T,1):
                valid=col[np.isfinite(col)];q=np.quantile(valid,[.25,.5,.75]) if len(valid) else [np.nan]*3
                curve_rows.append(dict(model=model,scope=scope,label=name,layer=layer,valid=len(valid),undefined=len(col)-len(valid),
                    mean=float(valid.mean()) if len(valid) else np.nan,q25=q[0],median=q[1],q75=q[2]))
    prior.write_csv(OUT/model/'ratio_curves.csv',curve_rows)
    bootstrap(model,data,results,protocol)
    progress(model,'完成',9,status='completed')


def bootstrap(model,data,results,protocol):
    path=OUT/model/'bootstrap.json'
    if path.exists():
        assert json.loads(path.read_text())['fingerprint']==protocol['fingerprint'];return
    y=data['y'][data['masks']['test']]
    images=sorted(protocol['split']['test']);lookup={image:i for i,image in enumerate(images)}
    cluster=np.asarray([lookup[m['image_id']] for m,keep in zip(data['mentions'],data['masks']['test']) if keep])
    p=np.stack([[results[feature,seed]['test_probabilities'] for seed in SEEDS] for feature in FEATURES])
    base=np.asarray([[weighted_scores(y,ps,np.ones(len(y))) for ps in group] for group in p]).mean(1)
    rng=np.random.default_rng(20260915);sample=[]
    for rep in range(2000):
        weights=np.bincount(rng.integers(len(images),size=len(images)),minlength=len(images))[cluster]
        if not all(weights[y==label].sum()>0 for label in (0,1)):continue
        sample.append(np.asarray([[weighted_scores(y,ps,weights) for ps in group] for group in p]).mean(1))
        if (rep+1)%250==0:progress(model,'bootstrap',9,replicates=rep+1);print(model,'bootstrap',rep+1,flush=True)
    sample=np.asarray(sample);rows=[]
    for a,b in PAIRS:
        ia,ib=FEATURES.index(a),FEATURES.index(b)
        for mi,metric in enumerate(('AUROC','HALL_AUPR')):
            low,high=np.quantile(sample[:,ia,mi]-sample[:,ib,mi],[.025,.975])
            rows.append(dict(model=model,comparison=a+' minus '+b,metric=metric,difference=base[ia,mi]-base[ib,mi],low=low,high=high,replicates=len(sample)))
    atomic_json_save(dict(fingerprint=protocol['fingerprint'],rows=rows),path)
    prior.write_csv(OUT/model/'bootstrap.csv',rows)


def summarize():
    rows=[];intervals=[];validation={}
    for model in MODELS:
        directory=OUT/model;seeds=read_csv(directory/'detection.csv');assert len(seeds)==9
        checks=json.loads((directory/'validation.json').read_text());assert len(checks)==9 and all(x['status']=='PASS' for x in checks)
        validation[model]=dict(heads=9,max_probability_error=max(x['max_probability_error'] for x in checks),status='PASS')
        intervals.extend(json.loads((directory/'bootstrap.json').read_text())['rows'])
        for feature in FEATURES:
            selected=[r for r in seeds if r['feature']==feature];assert sorted(int(r['seed']) for r in selected)==list(SEEDS)
            row=dict(model=model,feature=feature)
            for metric in ('AUROC','HALL_AUPR'):
                x=[float(r[metric]) for r in selected];row[metric+'_mean']=np.mean(x);row[metric+'_std']=np.std(x)
            rows.append(row)
    prior.write_csv(OUT/'detection_summary.csv',rows);prior.write_csv(OUT/'bootstrap.csv',intervals)
    atomic_json_save(dict(detection=rows,bootstrap=intervals),OUT/'summary.json');atomic_json_save(validation,OUT/'validation.json')
    lines=['# 生成来源 S_G / position：固定811检测结果','',
        '真RMS All-attention K32缓存，position=response_index=N_G；S_G为逐生成token响应范数和。无新VLM提取。',
        '四模型原4000图全部mentions，固定3200/400/400，seeds43/44/45。先求比，再log1p和train-only逐列Z-score；位置单独组为log1p(position)一个输入维度。统一单隐藏128/ReLU/dropout.3，无BN，Adam lr.001/wd1e-5/batch128，最多150epoch/早停20/最低val BCE checkpoint，无HPO。',
        'position=0时raw ratio=NaN、检测输入置0，保留样本。已有测试集曾被查看，以下为探索性比较。','',
        '| 模型 | S_G-only AUROC/AP | S_G/position-only AUROC/AP | position-only AUROC/AP |','|---|---:|---:|---:|']
    for model in MODELS:
        cells=[]
        for feature in FEATURES:
            row=next(r for r in rows if r['model']==model and r['feature']==feature)
            cells.append(' / '.join(f"{100*row[m+'_mean']:.2f}±{100*row[m+'_std']:.2f}" for m in ('AUROC','HALL_AUPR')))
        lines.append('| '+model+' | '+' | '.join(cells)+' |')
    lines+=['','数值为百分比、三seed指标均值±总体std，不是ensemble。AP以HALL为正类。','',
        '| 模型 | 比较 | 指标 | 差值pp [名义95% CI] |','|---|---|---|---:|']
    for r in intervals:lines.append(f"| {r['model']} | {r['comparison']} | {r['metric']} | {100*r['difference']:+.2f} [{100*r['low']:+.2f},{100*r['high']:+.2f}] |")
    lines+=['','每模型2000次测试图片簇配对bootstrap，包括无mention图片的400图抽样框架。每次分别算3seed指标差再平均，区间未多重校正。',
        '36头CPU重载、train-only scaler、最佳val-loss epoch、train/validation/test概率和指标复算均通过。比例归一化不等同于完全消除位置相关性或控制所有长度混杂。','',
        '[检测图](detection.png) · [REAL/HALL比值曲线](ratio_curves.png) · [核验](validation.json)']
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for mi,metric in enumerate(('AUROC','HALL_AUPR')):
        for fi,feature in enumerate(FEATURES):
            selected=[next(r for r in rows if r['model']==model and r['feature']==feature) for model in MODELS]
            axes[mi].bar(np.arange(4)+(fi-1)*.25,[100*r[metric+'_mean'] for r in selected],width=.24,
                yerr=[100*r[metric+'_std'] for r in selected],label=feature,capsize=3)
        axes[mi].set_xticks(range(4));axes[mi].set_xticklabels(['Qwen2.5','LLaVA','Qwen3','InternVL'])
        axes[mi].set_title(metric);axes[mi].set_ylim(0,100);axes[mi].legend();axes[mi].grid(axis='y',alpha=.2)
    fig.suptitle('True-RMS K32 | fixed 811 | 3-seed mean and population std');fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(OUT/f'detection.{ext}',dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(4,2,figsize=(13,14))
    for mi,model in enumerate(MODELS):
        curves=read_csv(OUT/model/'ratio_curves.csv')
        for si,scope in enumerate(('all','test')):
            ax=axes[mi,si]
            for label,color in (('REAL','#2878b5'),('HALL','#d95319')):
                selected=[r for r in curves if r['scope']==scope and r['label']==label];x=[int(r['layer']) for r in selected]
                ax.plot(x,[float(r['median']) for r in selected],label=label,color=color)
                ax.fill_between(x,[float(r['q25']) for r in selected],[float(r['q75']) for r in selected],color=color,alpha=.15)
            ax.set_title(model+' | '+scope);ax.set_xlabel('Decoder layer');ax.set_yscale('symlog',linthresh=.01);ax.grid(alpha=.2)
    axes[0,0].legend();fig.suptitle('S_G / position | raw ratio | mention median and IQR (not CI)');fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(OUT/f'ratio_curves.{ext}',dpi=150)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--models',nargs='+',choices=MODELS,default=list(MODELS))
    parser.add_argument('--device',default='cpu');parser.add_argument('--summarize',action='store_true');args=parser.parse_args()
    if args.summarize:summarize()
    else:
        for model in args.models:run_model(model,args.device)
