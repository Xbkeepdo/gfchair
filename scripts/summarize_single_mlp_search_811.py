"""Audit the sealed 811 single-hidden-layer search and report every finalist."""
import argparse
import csv
import hashlib
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import search_single_mlp_811 as run
from features.tc_fvpa_artifacts import atomic_json_save

OUT=run.OUT
NATIVE=ROOT/'outputs/native_baselines_811_v1'
HEADS=('svar_native','metatoken_lr','metatoken_gb')
NAMES=dict(qwen2_5_vl_7b='Qwen2.5',llava_1_5_7b='LLaVA',qwen3_vl_8b='Qwen3',internvl_2_5_8b='InternVL')
REGIONS=dict(visual='V',visual_prompt_sum='VP',generation='G',vp_generation='VP+G')


def csv_read(path):
    with path.open() as f:return list(csv.DictReader(f))


def load(path):
    # Shared NFS can briefly advertise completion before every filename is visible.
    for attempt in range(6):
        try:return run.read(path)
        except FileNotFoundError:
            if attempt==5:raise
            time.sleep(2)


def state(status,**kw):
    atomic_json_save(dict(status=status,heartbeat=datetime.now(timezone.utc).isoformat(),**kw),OUT/'summary_progress.json')


def audit_model(model):
    root=OUT/model;protocol=json.loads((root/'protocol.json').read_text());selection=json.loads((root/'selection.json').read_text())
    contents={k:v for k,v in protocol.items() if k!='fingerprint'}
    assert hashlib.sha256(json.dumps(contents,sort_keys=True).encode()).hexdigest()==protocol['fingerprint']
    assert selection['fingerprint']==protocol['fingerprint']
    data={p:load(folder/model/'matrices.pt') for p,folder in run.SOURCES.items()}
    reference=data['true_rms'];ty=reference['y'][reference['masks']['train']];vy=reference['y'][reference['masks']['validation']];y=reference['y'][reference['masks']['test']]
    native_data=load(NATIVE/'feature_cache'/model/'matrices.pt')
    native_protocol=json.loads((NATIVE/model/'protocol.json').read_text())
    assert native_protocol['feature_fingerprint']==native_data['fingerprint']
    assert native_data['mentions']==reference['mentions']
    np.testing.assert_array_equal(native_data['y'],reference['y'])
    for key in reference['masks']:np.testing.assert_array_equal(native_data['masks'][key],reference['masks'][key])
    latest_freeze=max(datetime.fromisoformat(json.loads((OUT/m/'selection.json').read_text())['frozen_at']).timestamp() for m in run.MODELS)
    all_trials=[];rows=[];seed_rows=[];params=[];champions=[];count=0
    for variant in run.VARIANTS:
        path,group=variant.split('/');d=data[path];chosen=selection['variants'][variant]
        assert d['fingerprint']==protocol['source_fingerprints'][path]
        original=[]
        def check(index,seed):
            nonlocal count
            f=root/variant/f'candidate{index:02d}'/f'seed{seed}'/'result.pt';r=load(f)
            assert r['fingerprint']==protocol['fingerprint'] and r['config']==protocol['candidates'][index]
            assert r['index']==index and r['seed']==seed and r['variant']==variant
            assert not any('test' in k for k in r)
            m=run.metrics(vy,r['validation_probabilities'])
            for key in m:assert abs(m[key]-r['validation'][key])<1e-12
            cfg=r['config'];h=r['history']
            best=min(h,key=lambda a:a['val_loss']) if cfg['monitor']=='val_loss' else max(h,key=lambda a:a['AUROC'])
            assert r['best_epoch']==best['epoch']
            mean,scale=run.scale_fit(d['groups'][group][d['masks']['train']],cfg['standardize'])
            np.testing.assert_array_equal(mean,r['mean']);np.testing.assert_array_equal(scale,r['scale'])
            count+=1
            all_trials.append(dict(model=model,variant=variant,index=index,seed=seed,**m,best_epoch=r['best_epoch'],epochs=r['epochs'],seconds=r['seconds']))
            return r
        for index in range(run.N_CANDIDATES):
            r=check(index,43);original.append(dict(index=index,validation=r['validation'],seed=43))
        assert original==chosen['seed43_trials']
        expected=sorted(original,key=run.rank,reverse=True)[:run.TOP_N]
        comparisons=[]
        for row in expected:
            rs=[row]+[dict(index=row['index'],seed=s,validation=check(row['index'],s)['validation']) for s in (44,45)]
            comparisons.append(dict(index=row['index'],validation={k:float(np.mean([a['validation'][k] for a in rs])) for k in rs[0]['validation']},seeds=rs))
        assert comparisons==chosen['shortlist']
        best=max(comparisons,key=run.rank)
        assert best['index']==chosen['index'] and best['validation']==chosen['validation']
        values=[];probs=[];epochs=[]
        for seed in run.SEEDS:
            final_path=root/'final'/variant/f'seed{seed}'/'result.pt';r=load(final_path)
            assert final_path.stat().st_mtime>=latest_freeze-2 # allow subsecond host-clock skew
            assert r['index']==chosen['index'] and r['config']==chosen['config'] and r['fingerprint']==protocol['fingerprint']
            trained=load(root/variant/f"candidate{chosen['index']:02d}"/f'seed{seed}'/'result.pt')
            net=run.SingleMLP(trained['input_dim'],trained['config']);net.load_state_dict(trained['state_dict'])
            pp=run.predict(net,torch.as_tensor(run.transform(d['groups'][group][d['masks']['test']],trained['mean'],trained['scale'])))
            np.testing.assert_allclose(pp,r['test_probabilities'],rtol=1e-5,atol=1e-6)
            m=run.metrics(y,r['test_probabilities'])
            for k in m:assert abs(m[k]-r['test_metrics'][k])<1e-12
            values.append(m);probs.append(r['test_probabilities']);epochs.append(r['best_epoch'])
            seed_rows.append(dict(model=model,variant=variant,seed=seed,**m))
        row=dict(model=model,variant=variant,champion=variant==selection['champion'],candidate=chosen['index'],
            validation_AUROC=chosen['validation']['AUROC'],validation_HALL_AUPR=chosen['validation']['HALL_AUPR'],
            **{k+'_'+stat:float(fn([a[k] for a in values])) for k in values[0] for stat,fn in [('mean',np.mean),('std',np.std)]},
            **{'ensemble_'+k:v for k,v in run.metrics(y,np.mean(probs,axis=0)).items()})
        rows.append(row);params.append(dict(model=model,variant=variant,index=chosen['index'],best_epochs='/'.join(map(str,epochs)),**chosen['config']))
    expected_champion=max(enumerate(run.VARIANTS),key=lambda a:(selection['variants'][a[1]]['validation']['AUROC'],selection['variants'][a[1]]['validation']['HALL_AUPR'],-a[0]))[1]
    assert selection['champion']==expected_champion and count==240
    assert len(list(root.glob('*/*/candidate*/seed*/result.pt')))==240
    baseline=[]
    for head in HEADS:
        hs=[load(NATIVE/model/head/f'seed{s}'/'result.pt') for s in run.SEEDS]
        for seed,h in zip(run.SEEDS,hs):
            assert h['fingerprint']==native_protocol['fingerprint'] and h['seed']==seed and h['head']==head
        ms=[run.metrics(y,h['test_probabilities']) for h in hs]
        for h,m in zip(hs,ms):
            for k in m:assert abs(m[k]-h['test_metrics'][k])<1e-12
        baseline.append(dict(model=model,head=head,**{k+'_'+stat:float(fn([a[k] for a in ms])) for k in ms[0] for stat,fn in [('mean',np.mean),('std',np.std)]}))
    state('auditing',model=model,validated_fits=count,validated_test_heads=24)
    return rows,seed_rows,params,baseline,all_trials


def label(variant):
    path,group=variant.split('/');return ('真实 RMS' if path=='true_rms' else '冻结 RMS')+' '+REGIONS[group]


def summary():
    assert run.test_gate()
    protocols=[json.loads((OUT/m/'protocol.json').read_text()) for m in run.MODELS]
    for p in protocols[1:]:
        for key in ('candidates','search_seed','shortlist_seed','top_n','final_seeds','variants','split','ranking','champion','test_gate','architecture','budget'):
            assert p[key]==protocols[0][key],f'Four-model protocol differs: {key}'
    rows=[];seeds=[];params=[];base=[];trials=[]
    for model in run.MODELS:
        parts=audit_model(model)
        for target,source in zip((rows,seeds,params,base,trials),parts):target.extend(source)
    deltas=[];previous=[]
    for r in rows:
        for b in base:
            if r['model']==b['model']:
                deltas.append(dict(model=r['model'],variant=r['variant'],champion=r['champion'],baseline=b['head'],
                    **{k+'_delta_pp':100*(r[k+'_mean']-b[k+'_mean']) for k in ('AUROC','HALL_AUPR')}))
        path,group=r['variant'].split('/')
        old=next(a for a in csv_read(run.SOURCES[path]/'detection.csv') if a['model']==r['model'] and a['group']==group and a['classifier']=='one_hidden')
        previous.append(dict(model=r['model'],variant=r['variant'],
            **{k+'_delta_pp':100*(r[k+'_mean']-float(old[k+'_mean'])) for k in ('AUROC','HALL_AUPR')}))
    write=run.prior.original.base.write_csv
    for name,value in [('detection',rows),('seed_metrics',seeds),('selected_parameters',params),('native_baselines',base),('vs_native',deltas),('vs_previous_single',previous),('validation_trials',trials)]:write(value,OUT/(name+'.csv'))
    plot(rows,base)
    report(rows,params,base,deltas,previous)
    atomic_json_save(dict(status='PASS',fits=960,final_heads=96,native_heads=36,variants=32,
        checks=['image/mention/label/mask alignment','train-only scaler','all validation metrics recomputed',
                'checkpoint selection from validation history','fixed shortlist and three-seed winner',
                'validation-only champion','global test gate timestamps','CPU checkpoint test probabilities reproduced',
                'test and baseline rank metrics recomputed'],timestamp=datetime.now(timezone.utc).isoformat()),OUT/'validation.json')
    state('completed',fits=960,heads=96,groups=32,report='docs/SINGLE_MLP_SEARCH_811_RESULTS.md')


def plot(rows,base):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for metric,lower,filename in [('AUROC',70,'auroc'),('HALL_AUPR',0,'hall_aupr')]:
        fig,axes=plt.subplots(2,2,figsize=(16,9),sharey=True)
        for ax,model in zip(axes.flat,run.MODELS):
            rs=[r for r in rows if r['model']==model];x=np.arange(8)
            ax.bar(x,[100*r[metric+'_mean'] for r in rs],yerr=[100*r[metric+'_std'] for r in rs],
                   color=['#195b87' if r['champion'] else '#83b6d4' for r in rs],capsize=3)
            for b,color in zip([b for b in base if b['model']==model],['#ad3535','#987727','#565656']):
                ax.axhline(100*b[metric+'_mean'],label=b['head'],color=color,ls='--',lw=1.5)
            ax.set_xticks(x,[('True' if r['variant'].startswith('true') else 'Frozen')+'\n'+REGIONS[r['variant'].split('/')[1]] for r in rs],fontsize=8)
            ax.set_title(NAMES[model]);ax.set_ylim(lower,100);ax.set_ylabel('Test '+metric+' (%)');ax.grid(axis='y',alpha=.15);ax.legend(fontsize=8)
        fig.suptitle('811 single-hidden-layer search | mean +/- SD of seeds 43/44/45\nDark bar: validation-AUROC-selected champion; native baselines have a different tuning budget')
        fig.tight_layout()
        for ext in ('png','pdf'):fig.savefig(OUT/f'{filename}_vs_native.{ext}',dpi=180)
        plt.close(fig)


def report(rows,params,base,deltas,previous):
    lines=['# 811 单隐藏层 MLP 固定预算搜索结果','',
        '实验设置：按图片 3200 训练 / 400 验证 / 400 测试，固定 split seed=20260912，保留全部 mentions。沿用原训练集，原 800 图留出集分半；该留出集此前已被查看，本轮属于探索性比较。',
        '分类器严格单隐藏层：Linear → 可选 BatchNorm → ReLU/GELU → Dropout → Linear，BCE（REAL=1）、Adam；输入标准化若启用，仅拟合训练集。最多150 epochs、早停 patience=20、ReduceLROnPlateau factor=0.5/patience=6/min_lr=1e-6。预注册候选中 checkpoint/调度/早停依据为 val_loss 或 val_AUROC，见参数表。',
        '每个模型、每组24个固定候选先跑 seed43；验证 AUROC 排名前3补44/45，再按三 seed 验证 AUROC 均值选配置。并列依次看验证 HALL-AUPR、较小候选序号。每模型再按相同指标选一个跨8组 champion；四模型选择全部冻结后才计算本轮测试指标。总计960次训练、96个测试头，不 train+val 重训。',
        '**下列 AUROC / HALL-AUPR 均为 seeds43/44/45 分别计算后取均值，单位 %；不是概率 ensemble。** 完整结果表与 CSV 另列总体标准差，ensemble 单独保存在 CSV。排名指标无阈值依赖；附带阈值结果采用 train REAL-F1 和固定0.5。原生基线阈值采用 val REAL-F1。','',
        '## 特征与路径','',
        '| 组 | 每层输入 |', '|---|---|',
        '| V | `[AE_V, log1p(S_V)]` |',
        '| VP | `[AE_VP, log1p(S_V+S_P)]` |',
        '| G | `[AE_G, log1p(S_G)]` |',
        '| VP+G | 完整拼接 VP 与 G 两块 |','',
        'AE_VP 在视觉+prompt 区域重新归一化并计算 gate，不是 AE_V+AE_P。空 generation 在检测输入置零，保留该 mention。四模型层数为28/32/36/32，V/VP/G维度2L，VP+G维度4L。',
        '真实 RMS：S 是逐 token 积分响应范数之和（gross），路径 z−A_all → z，J_(FFN∘Norm)，K32。冻结 RMS：固定 clean endpoint 的 D(z)，纯 FFN 的 αNorm(z) 路径，K50；完整分解包含 residual、attention、bias，这四组仅输入 attention gross。两版 AE 逐值相同；两条路径和基点也不同，不能把差值全部归因于 RMS 导数。',
        '继承的数值限制：冻结K50纯积分闭合已通过，但 Qwen2.5/Qwen3 原生来源重构的总闭合存在尾部误差（最大约9.51%/11.62%）；本轮只搜索分类器，不改变提取结果。','',
        '## 验证集选出的代表配置与原生基线','',
        '| 模型 | 验证选出的组 | 本方法 AUROC / AP | SVAR AUROC / AP | Meta LR AUROC / AP | Meta GB AUROC / AP | 对SVAR差值 AUROC / AP (pp) |','|---|---|---|---|---|---|---|']
    def pair(r):return f"{100*r['AUROC_mean']:.2f} / {100*r['HALL_AUPR_mean']:.2f}"
    for model in run.MODELS:
        r=next(r for r in rows if r['model']==model and r['champion']);bs=[next(b for b in base if b['model']==model and b['head']==head) for head in HEADS]
        lines.append(f"| {NAMES[model]} | {label(r['variant'])} | {pair(r)} | "+' | '.join(pair(b) for b in bs)+f" | {100*(r['AUROC_mean']-bs[0]['AUROC_mean']):+.2f} / {100*(r['HALL_AUPR_mean']-bs[0]['HALL_AUPR_mean']):+.2f} |")
    wins=[r['model'] for r in rows if r['champion'] and all(r['AUROC_mean']>b['AUROC_mean'] for b in base if b['model']==r['model'])]
    lines.extend(['',f"验证集选出的代表配置在测试 AUROC 上超过全部三种原生基线的模型：{', '.join(NAMES[m] for m in wins) or '无'}（{len(wins)}/4）。这只表示当前测试集的均值比较，没有显著性检验或独立测试的推广保证。",'',
        '原生 SVAR：248单隐藏ReLU、Adam lr=.001/batch32/max50、val_loss/patience5，无BN/dropout/scaler。MetaToken：训练集StandardScaler+LR(lbfgs,max2000)或GB100。原生基线没有获得本轮相同的搜索预算；本表比较方法连同分类器配置，不证明公平预算下的特征单独优势。基线使用项目已有原生分类器和同目标 controlled 特征，并非论文原始数据集复现；MetaToken 保留完整回答长度/对象span统计，不能称严格前缀信号的公平因果对比。详见 [原生基线设置](NATIVE_BASELINES_811_RESULTS.md)。','',
        '## 全部8组结果','',
        '| 模型 | 组 | 验证AUROC | 测试AUROC ± SD | 测试HALL-AUPR ± SD | Δ旧单层 AUROC / AP (pp) |','|---|---|---:|---:|---:|---:|'])
    for r in rows:
        old=next(v for v in previous if v['model']==r['model'] and v['variant']==r['variant'])
        lines.append(f"| {NAMES[r['model']]} | {label(r['variant'])}{' ★' if r['champion'] else ''} | {100*r['validation_AUROC']:.2f} | {100*r['AUROC_mean']:.2f} ± {100*r['AUROC_std']:.2f} | {100*r['HALL_AUPR_mean']:.2f} ± {100*r['HALL_AUPR_std']:.2f} | {old['AUROC_delta_pp']:+.2f} / {old['HALL_AUPR_delta_pp']:+.2f} |")
    lines.extend(['','★ 仅表示验证集选出的 champion，不表示按测试集选出的最佳组。旧单层为 sklearn 12候选（宽度64/128/256、lr .01/.001、Adam/SGD、max500）、无额外标准化；新旧差值包含训练实现、正则化、早停与预算差异，不能单独归因某一超参数。','',
        '## 选出的具体参数','',
        '| 模型 | 组 | c编号 | 宽度 | 标准化 | BN | dropout | 激活 | lr | wd | batch | checkpoint | 最佳epoch 43/44/45 |','|---|---|---:|---:|---|---|---:|---|---:|---:|---:|---|---|'])
    for p in params:
        lines.append(f"| {NAMES[p['model']]} | {label(p['variant'])} | {p['index']} | {p['width']} | {p['standardize']} | {p['batch_norm']} | {p['dropout']} | {p['activation']} | {p['learning_rate']} | {p['weight_decay']} | {p['batch_size']} | {p['monitor']} | {p['best_epochs']} |")
    lines.extend(['','## 为什么可能改善，哪些解释尚不能成立','',
        '本轮允许标准化、BN、正则化、学习率和宽度共同匹配 AE 与 log1p(gross) 的尺度，并用验证集控制 checkpoint；这些设置可能改善优化和泛化。实际效果见完整表，不能保证每个模型、每组或两个指标同时改善。',
        '这些解释是机制假设。搜索不是逐因素消融实验，选出的参数也不等于每个参数都必要；不能根据赢家配置断言“BN带来多少百分点”或“冻结RMS必然更好”。要估计某个设置的独立影响，需要固定其他条件的另行对照。',
        '本轮没有按测试结果继续扩大网格，也没有隐藏落后组。三seed标准差描述训练随机性，不是图片抽样置信区间。','',
        '## 图与可复核数据','',
        '![811单层AUROC与原生基线](../outputs/single_mlp_search_811_v1/auroc_vs_native.png)','',
        '![811单层HALL-AUPR与原生基线](../outputs/single_mlp_search_811_v1/hall_aupr_vs_native.png)','',
        '图中深色柱为验证选出的组，误差线为三个seed标准差，虚线是原生基线；AUROC纵轴70–100%，HALL-AUPR纵轴0–100%；阅读差值时以表中百分点为准。',
        '- [所有组指标及ensemble](../outputs/single_mlp_search_811_v1/detection.csv)',
        '- [96个seed指标](../outputs/single_mlp_search_811_v1/seed_metrics.csv)',
        '- [具体参数](../outputs/single_mlp_search_811_v1/selected_parameters.csv)',
        '- [对三种原生基线的全部差值](../outputs/single_mlp_search_811_v1/vs_native.csv)',
        '- [960次验证搜索记录](../outputs/single_mlp_search_811_v1/validation_trials.csv)',
        '- [核验记录](../outputs/single_mlp_search_811_v1/validation.json)',
        '- 每模型目录的 protocol.json 与 selection.json 保存完整预注册候选、来源指纹、验证选择及冻结时间；candidate*/seed*/result.pt 保存可恢复checkpoint。',
        '- 日志 outputs/single811_{qwen2,llava,qwen3,internvl}.log；双机进度文件每10秒覆写，最终状态保留。',
        '- InternVL 曾在 progress.json 的 NFS inode 验证上阻塞；确认旧worker退出及锁释放后恢复，并同字节原子重发进度文件。240次训练和冻结选择保持不变，未重训；恢复日志 outputs/single811_internvl_resume1.log。全960 fits / 96测试头 / 36原生头复算核验PASS。',''])
    (ROOT/'docs/SINGLE_MLP_SEARCH_811_RESULTS.md').write_text('\n'.join(lines))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--wait',action='store_true');args=parser.parse_args();torch.set_num_threads(1)
    try:
        while True:
            statuses={m:json.loads((OUT/m/'progress.json').read_text()) if (OUT/m/'progress.json').exists() else {} for m in run.MODELS}
            failed=[m for m,s in statuses.items() if s.get('status')=='failed']
            if failed:raise RuntimeError('Model search failed: '+','.join(failed))
            if all(s.get('status')=='completed' for s in statuses.values()):break
            if not args.wait:raise RuntimeError('Four model searches are not completed')
            state('waiting',models=statuses);time.sleep(10)
        summary()
    except BaseException as e:
        state('failed',error=str(e));raise
