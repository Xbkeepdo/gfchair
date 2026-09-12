"""Per-target visual/generated-text attention ratios from region caches."""
import csv
import json
from pathlib import Path
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'outputs/prefix_attention_gate/full/region_plots'
OUT=SOURCE/'visual_generation_ratio'
MODELS=('minigpt4_7b','shikra_7b','qwen2_5_vl_7b','llava_1_5_7b','qwen3_vl_8b','internvl_2_5_8b')
METRICS=('raw_attention','attention_x_gate')


def ratio(visual,generation):
    v,g=np.asarray(visual,dtype=np.float64),np.asarray(generation,dtype=np.float64)
    if v.shape!=g.shape or v.ndim!=2 or not np.isfinite(v).all() or not np.isfinite(g).all() or (v<0).any() or (g<0).any():
        raise ValueError('Expected finite nonnegative [mentions,layers] masses')
    out=np.full_like(v,np.nan)
    np.divide(v,g,out=out,where=g>0)
    if np.isinf(out).any():raise ValueError('Ratio overflow')
    return out


def main():
    OUT.mkdir(parents=True,exist_ok=True);records=[];metadata={}
    for model in MODELS:
        meta=json.loads((SOURCE/model/'metadata.json').read_text())
        assert list(meta['metrics'])==list(METRICS)
        vi=meta['regions'].index('visual');gi=meta['regions'].index('generated_text')
        with np.load(SOURCE/model/'regions.npz') as cache:
            values=cache['values'];labels=cache['labels'];train=cache['train']
        metadata[model]=dict(mentions=len(labels),layers=values.shape[2])
        for mi,metric in enumerate(METRICS):
            ratios=ratio(values[:,mi,:,vi],values[:,mi,:,gi])
            for split,selection in [('all',np.ones(len(labels),dtype=bool)),('train',train),('test',~train)]:
                for label in (0,1):
                    x=ratios[selection&(labels==label)]
                    for layer in range(x.shape[1]):
                        column=x[:,layer];valid=column[np.isfinite(column)]
                        stats=dict(mean=None,median=None,q25=None,q75=None,min=None,max=None)
                        if len(valid):
                            q25,median,q75=np.quantile(valid,[.25,.5,.75])
                            stats=dict(mean=float(valid.mean()),median=float(median),q25=float(q25),q75=float(q75),min=float(valid.min()),max=float(valid.max()))
                        records.append(dict(model=model,metric=metric,split=split,label=label,layer=layer+1,n_total=len(column),n_valid=len(valid),n_denominator_zero=int(np.isnan(column).sum()),n_zero_ratio=int((valid==0).sum()),**stats))
    with (OUT/'curves.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    for metric in METRICS:
        for split in ('all','train','test'):
            fig,axes=plt.subplots(3,2,figsize=(13,11),squeeze=False)
            for model,ax in zip(MODELS,axes.flat):
                for label,name,color in ((0,'HALL','#d55e00'),(1,'REAL','#0072b2')):
                    rows=[r for r in records if r['model']==model and r['metric']==metric and r['split']==split and r['label']==label]
                    layers=[r['layer'] for r in rows]
                    # Zero remains zero in statistics; it cannot be displayed on a log axis.
                    def positive(key):return [r[key] if r[key] is not None and r[key]>0 else np.nan for r in rows]
                    ax.plot(layers,positive('mean'),color=color,label=name+' mean')
                    ax.plot(layers,positive('median'),color=color,linestyle='--',alpha=.7)
                    ax.fill_between(layers,positive('q25'),positive('q75'),color=color,alpha=.13)
                ax.set_yscale('log');ax.set_xlim(1,max(layers));ax.set_xlabel('Decoder layer (1-based)');ax.set_ylabel('Visual / generated-text mass (log axis)')
                ax.set_title(model);ax.legend(fontsize=8);ax.grid(alpha=.15,which='both')
            fig.suptitle(f'{metric} / {split}: per-mention ratio; solid mean, dashed median, IQR shade\nZero denominators excluded and counted in CSV; no epsilon; panel y-scales differ',fontsize=11)
            fig.tight_layout();fig.savefig(OUT/f'{split}_{metric}.png',dpi=170);fig.savefig(OUT/f'{split}_{metric}.pdf');plt.close(fig)
    protocol=dict(models=metadata,definition='For each mention and layer: sum_visual(attention) / sum_generated_text(attention), separately for raw and attention*gate',
        denominator_zero='undefined; exclude only from that layer/label statistic and report counts; no epsilon or cap',
        statistic='ratio first, then mean/median/IQR; not ratio of mean masses and not mean(log ratio)',
        plotting='logarithmic y-axis; valid zero ratios remain in CSV/statistics, not drawable on log scale; IQR not CI')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    lines=['# 视觉注意力 / 生成文本注意力','', '先逐目标、逐层求比值，再按HALL/REAL统计。使用原全部mentions，分别提供all/train/test。分母为0的比值未定义，仅在对应层统计中排除并在CSV计数，不加epsilon、不截断极值。',
           '实线均值、虚线中位数、阴影IQR（非置信区间）。纵轴为对数刻度；统计的是原始比值，不是log比值。各面板纵轴独立。合法的零比值保留在CSV及统计中，但不在对数图显示。','']
    for split in ('all','train','test'):
        lines.append(f'- {split}：[raw attention]({split}_raw_attention.png) · [attention×gate]({split}_attention_x_gate.png)')
    lines+=['','[逐层数据与无效计数](curves.csv) · [协议](protocol.json)']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    print('Saved',len(records),'curve rows and 6 PNG/PDF figures to',OUT)


if __name__=='__main__':main()
