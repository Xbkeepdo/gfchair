"""Compare saved detectors using seed means throughout; no training or selection."""
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_all_source_paths import MODELS, LAYERS, F1_FIELDS, F2_FIELDS, F3_FIELDS, F4_FIELDS

BASE = ROOT/'outputs/ffn_all_source_paths_v1'
OUT = BASE/'detection_comparison_20260913'
NAMES = ('Qwen2.5', 'LLaVA', 'Qwen3', 'InternVL')
GROUPS = ['F0','F1_raw','F1_log1p','F2'] + [b+'_'+g for b in ('b1','b2')
    for g in ('F3_raw','F3_log1p','F4','F5','F6')] + ['length']


def csv_read(path):
    with path.open() as f: return list(csv.DictReader(f))


def csv_write(path, rows):
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def record(model, experiment, group, classifier, auc, auc_std, ap, ap_std, source):
    return dict(model=model,experiment=experiment,group=group,classifier=classifier,
                auc=float(auc),auc_std=float(auc_std),hall_aupr=float(ap),hall_aupr_std=float(ap_std),
                status='B2_K4_PROVISIONAL' if experiment=='current' and group.startswith('b2_') else 'see source numerical limits',source=str(source))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    current=[];historical=[];aggregation=[]
    for model in MODELS:
        for experiment, path in [('current',BASE/model/'detection.json'),
            ('old_composition',ROOT/'outputs/ffn_source_composition_v1'/model/'detection.json')]:
            for group,value in json.loads(path.read_text()).items():
                stats=value['seed_mean_std']
                row=record(model,experiment,group,'fixed_three_hidden',stats['auc']['mean'],stats['auc']['std'],
                    stats['hall_aupr']['mean'],stats['hall_aupr']['std'],path.relative_to(ROOT))
                (current if experiment=='current' else historical).append(row)
                if experiment=='old_composition' and group=='F':
                    ensemble=value['ensemble_reports']['fixed_0.5']
                    aggregation.append(dict(model=model,seed_mean_auc=row['auc'],seed_mean_ap=row['hall_aupr'],
                        ensemble_auc=ensemble['auc'],ensemble_ap=ensemble['hallucination_positive']['aupr']))
    paths=[('old_gate','outputs/prefix_attention_gate/full/group_detection/detection.csv'),
        ('old_strength','outputs/ffn_source_composition_v1/source_strength_detection/detection.csv'),
        ('old_fusion','outputs/ffn_source_composition_v1/attention_strength_fusion/detection.csv'),
        ('old_cross_scale','outputs/ffn_source_composition_v1/raw_attention_log_strength_mlp/trees_detection.csv')]
    for experiment,path in paths:
        for r in csv_read(ROOT/path):
            if r['model'] not in MODELS: continue
            classifier=r.get('classifier','fixed_three_hidden')
            if classifier=='three_hidden':classifier='fixed_three_hidden'
            historical.append(record(r['model'],experiment,r['group'],classifier,
                r['AUROC_mean'],r['AUROC_std'],r['HALL_AUPR_mean'],r['HALL_AUPR_std'],path))
    current_lookup={(r['model'],r['group']):r for r in current}
    old_lookup={(r['model'],r['experiment'],r['group'],r['classifier']):r for r in historical}
    for model in MODELS:
        a=current_lookup[(model,'F0')];b=old_lookup[(model,'old_gate','gated_raw_concat','fixed_three_hidden')]
        for key in ('auc','auc_std','hall_aupr','hall_aupr_std'):assert abs(a[key]-b[key])<1e-12
    csv_write(OUT/'current.csv',current);csv_write(OUT/'historical.csv',historical)
    csv_write(OUT/'old_ensemble_vs_seed_mean.csv',aggregation)
    selections=[('旧 Visual-only：AE + log1p(S_E)','old_composition','F','fixed_three_hidden'),
        ('旧 U 原值三来源拼接 = 本次 F0','old_gate','gated_raw_concat','fixed_three_hidden'),
        ('旧 log1p(U) 三来源拼接','old_gate','gated_log1p_concat','fixed_three_hidden'),
        ('旧 S 三来源原值','old_strength','raw_concat','fixed_three_hidden'),
        ('旧 log1p(S) 三来源','old_strength','log1p_concat','fixed_three_hidden'),
        ('旧 [U,S] 三来源拼接','old_fusion','gated_raw_concat','fixed_three_hidden'),
        ('旧 [log1p(U),log1p(S)] 三来源','old_fusion','gated_log1p_concat','fixed_three_hidden'),
        ('旧 [U_P,S_P] 仅 prompt','old_fusion','gated_raw_prompt','fixed_three_hidden'),
        ('旧 [U_VP,log1p(S_VP),U_G,log1p(S_G)] MLP','old_cross_scale','gated_vp_generation','fixed_three_hidden'),
        ('同上特征，搜参 XGB（分类器不同）','old_cross_scale','gated_vp_generation','xgb')]
    selected=[];deltas=[]
    for title,experiment,group,classifier in selections:
        for model in MODELS:
            row=old_lookup[(model,experiment,group,classifier)]
            selected.append(dict(title=title,**row))
            for branch in ('b1_F6','b2_F6'):
                new=current_lookup[(model,branch)]
                deltas.append(dict(model=model,current_group=branch,previous=title,
                    delta_auc_pp=100*(new['auc']-row['auc']),delta_hall_ap_pp=100*(new['hall_aupr']-row['hall_aupr']),
                    inference='descriptive point difference; no new bootstrap; B2 provisional' if branch.startswith('b2') else 'descriptive point difference; see existing bootstrap for registered pairs'))
    csv_write(OUT/'selected_history.csv',selected);csv_write(OUT/'f6_vs_history.csv',deltas)
    fields={'F0':['prompt_attention_gate_mass','generation_attention_gate_mass','visual_attention_gate_mass'],
        'F1_raw':list(F1_FIELDS),'F1_log1p':list(F1_FIELDS),'F2':list(F2_FIELDS)}
    for branch in ('b1','b2'):
        fields[branch+'_F3_raw']=[branch+'_'+f for f in F3_FIELDS]
        fields[branch+'_F3_log1p']=fields[branch+'_F3_raw']
        fields[branch+'_F4']=[branch+'_'+f for f in F4_FIELDS]
        fields[branch+'_F5']=fields['F2']+fields[branch+'_F4']
        fields[branch+'_F6']=fields['F0']+fields[branch+'_F5']
    fields['length']=['N_G=response_index','response_index']
    (OUT/'feature_fields.json').write_text(json.dumps({g:dict(fields=fields[g],
        dimensions={m:len(fields[g])*LAYERS[m] if g!='length' else 2 for m in MODELS},
        log1p_fields=([s+'_'+k for s in ('prompt','visual','generation') for k in ('gross_norm','net_norm')]
            if g=='F1_log1p' else fields[g][:4] if g.endswith('F3_log1p') else [])) for g in GROUPS},indent=2))
    def table(title, rows, keys, fmt):
        lines=[title,'','| 组 | '+' | '.join(NAMES)+' |','|---|'+'---:|'*4]
        for label,values in rows:
            lines.append('| '+label+' | '+' | '.join(fmt(values[m],keys) for m in MODELS)+' |')
        return '\n'.join(lines)+'\n'
    current_rows=[(g+' †' if g.startswith('b2_') else g,{m:current_lookup[m,g] for m in MODELS}) for g in GROUPS]
    fmt=lambda r,k:f'{100*r[k]:.2f} ± {100*r[k+"_std"]:.2f}'
    tables=table('## 全部 15 组 AUROC（%，三 seed 均值 ± 总体标准差）',current_rows,'auc',fmt)
    tables+='\n'+table('## 全部 15 组 HALL-AUPR（%，三 seed 均值 ± 总体标准差）',current_rows,'hall_aupr',fmt)
    history_rows=[(title,{m:old_lookup[m,ex,g,cl] for m in MODELS}) for title,ex,g,cl in selections]
    history_rows += [('本次 B1-F6',{m:current_lookup[m,'b1_F6'] for m in MODELS}),('本次 B2-F6 †',{m:current_lookup[m,'b2_F6'] for m in MODELS})]
    tables+='\n'+table('## 与之前结果比较（每格 AUROC / HALL-AUPR，三 seed 均值 %）',history_rows,None,
        lambda r,k:f'{100*r["auc"]:.2f} / {100*r["hall_aupr"]:.2f}')
    tables+='\n† B2 K4 未通过全量数值复核，分数保留，方法优劣待修复后确认。历史对照完整 std 见 selected_history.csv。\n'
    bs=csv_read(BASE/'bootstrap.csv')
    tables+='\n## 预注册配对：B1-F6 相对 F0 及来源强度\n\n单位：百分点，括号为 10,000 次图片级配对 bootstrap 的 95% 区间；比较三 seed 指标差的平均。\n\n| 模型 | 对照 | ΔAUROC [95% CI] | ΔHALL-AUPR [95% CI] |\n|---|---|---:|---:|\n'
    for model,name in zip(MODELS,NAMES):
        for baseline in ('F0','source_raw','source_log1p','length'):
            values=[]
            for metric in ('AUROC','HALL_AUPR'):
                r=next(r for r in bs if r['model']==model and r['left']=='b1_F6' and r['right']==baseline and r['metric']==metric)
                values.append(f'{100*float(r["delta"]):+.2f} [{100*float(r["ci95_low"]):+.2f}, {100*float(r["ci95_high"]):+.2f}]')
            tables+='| '+name+' | '+baseline+' | '+' | '.join(values)+' |\n'
    (OUT/'tables.md').write_text(tables)
    assert len(current)==60 and all(len(fields[g]) in (2,3,5,7,9,15,22,25) for g in GROUPS)
    print('Complete:',OUT,'current groups',len(current),'historical groups',len(historical),'F0 matches all four old baselines')


if __name__=='__main__':main()
