"""Watch the two persisted jobs and write their completed comparison table."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time

ROOT=Path(__file__).resolve().parents[1]
MODELS=('minigpt4_7b','shikra_7b')
OUT=ROOT/'outputs/minigpt4_shikra_path_summary'


def read(path):
    return json.loads(path.read_text())


def write(value,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False))
    tmp.replace(path)


def summarize_reports(reports):
    result={}
    values={'auc':[r['auc'] for r in reports]}
    for label in ('real_positive','hallucination_positive'):
        for metric in ('precision','recall','f1','aupr'):
            values[label+'_'+metric]=[r[label][metric] for r in reports]
    for name,v in values.items():
        result[name]=dict(mean=statistics.mean(v),std=statistics.pstdev(v),values=v)
    return result


def collect():
    statuses={}
    for model in MODELS:
        root=ROOT/'outputs'/model/'COCO4000-JACOBIAN-PATH'
        status=read(root/'status.json') if (root/'status.json').exists() else {'status':'RUNNING'}
        status.update(generated_images=len(read(root/'generations.json')) if (root/'generations.json').exists() else 0,
                      path_shards=len(list((root/'path/shards').glob('*.pt'))))
        statuses[model]=status
    write(dict(updated_utc=datetime.now(timezone.utc).isoformat(),models=statuses),OUT/'progress.json')
    if any(v['status']=='FAILED' for v in statuses.values()):
        raise RuntimeError('A pipeline failed; inspect its status.json and run.log')
    if not all(v['status']=='COMPLETE_EXPERIMENT' for v in statuses.values()):
        return False
    rows,metadata=[],{}
    for model in MODELS:
        root=ROOT/'outputs'/model/'COCO4000-JACOBIAN-PATH'
        metadata[model]=dict(chair=read(root/'chair_summary.json'),numerical=read(root/'path/full_numerical_audit.json'),
            ads_protocol='Q-Former mapped ADS + query CGC' if model=='minigpt4_7b' else 'native patch ADS+CGC')
        methods=read(root/'path/training/results.json')
        added=ROOT/'outputs/ae_s_qjs02'/model/'results.json'
        if added.exists():
            addition=read(added)
            if addition['status']!='COMPLETE':raise ValueError('Incomplete JS fusion experiment')
            methods['F+JS(Q,tau=0.2)']=addition['F_QJS02']
        cosine=ROOT/'outputs/ae_s_cosinejs02'/model/'results.json'
        if cosine.exists():
            addition=read(cosine)
            if addition['status']!='COMPLETE':raise ValueError('Incomplete cosine JS fusion experiment')
            methods['F+JS(endpoint_cos,tau=0.2)']=addition['F_CosineJS02']
        attention=ROOT/'outputs/raw_attention_strength'/model/'results.json'
        if attention.exists():
            addition=read(attention)
            if addition['status']!='COMPLETE':raise ValueError('Incomplete raw attention experiment')
            for key,name in (('R','R_raw'),('logS','log1p(S)'),('R_logS','R_raw+log1p(S)')):
                methods[name]=addition['groups'][key]
        raw_js=ROOT/'outputs/cosine_raw_attention_js02'/model/'results.json'
        if raw_js.exists():
            addition=read(raw_js)
            if addition['status']!='COMPLETE':raise ValueError('Incomplete cosine/raw-attention JS experiment')
            methods['JS(endpoint_cos,raw_attention,tau=0.2)']=addition['groups']['J_raw']
            methods['F+JS(endpoint_cos,raw_attention,tau=0.2)']=addition['groups']['F_J_raw']
        for name,value in methods.items():
            if set(value['per_seed_metrics'])!={'43','44','45'}:raise ValueError('Incomplete method seeds')
            for rule in ('fixed_0.5','train_f1'):
                metrics=summarize_reports([value['per_seed_metrics'][str(s)]['threshold_reports'][rule]['test_metrics'] for s in (43,44,45)])
                rows.append(dict(model=model,family='jacobian_path',method=name,threshold=rule,metrics=metrics))
        baseline=read(root/f'baseline/results/comparison/{model}_baselines_native_vs_shared_mlp_3seed.json')
        if baseline['seeds']!=[43,44,45]:raise ValueError('Incomplete baseline seeds')
        for row in baseline['rows']:
            m=row['test_metrics'];metrics={'auc':m['auc']}
            for label in ('real_positive','hallucination_positive'):
                for key in ('precision','recall','f1','aupr'):metrics[label+'_'+key]=m[label][key]
            rows.append(dict(model=model,family=row['trainer'],method=row['display_name'],threshold=row['threshold_mode'],metrics=metrics))
        ads=[read(root/f'results/seed{s}/{model}_selected_feature_sets.json')['ads+cgc']['torch_probe'] for s in (43,44,45)]
        for rule in ('fixed_0.5','train_f1'):
            rows.append(dict(model=model,family='shared_torch_mlp',method='ADS+CGC',threshold=rule,
                             metrics=summarize_reports([x['threshold_reports'][rule]['test_metrics'] for x in ads])))
    write(dict(status='COMPLETE',metadata=metadata,rows=rows),OUT/'results.json')
    lines=['# MiniGPT-4 / Shikra COCO4000', '', '分别计算seeds 43/44/45的指标，再报告均值±总体标准差（ddof=0）；主表不使用ensemble。完整双阈值见results.json。F1阈值由各seed训练集REAL-F1选择。',
           'MiniGPT-4 使用映射版 ADS + 查询空间 CGC。', '',
           '| 模型 | 方法/训练器 | AUROC | HALL AUPR | HALL F1（train-F1阈值） |', '|---|---|---:|---:|---:|']
    for row in rows:
        if row['threshold']!='train_f1':continue
        values=[row['metrics'][k] for k in ('auc','hallucination_positive_aupr','hallucination_positive_f1')]
        text=[f"{v['mean']:.4f} ± {v['std']:.4f}" for v in values]
        lines.append('| '+ ' | '.join([row['model'],row['method']+' / '+row['family'],*text])+' |')
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n')
    for model in MODELS:
        selected=[line for line in lines[7:] if line.startswith('| '+model+' |') and ' / jacobian_path |' in line]
        (ROOT/'outputs'/model/'COCO4000-JACOBIAN-PATH/path/training/summary.md').write_text(
            '\n'.join([f'# {model} — Jacobian path', '', lines[2], '', lines[5], lines[6], *selected])+'\n')
    return True


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch',action='store_true')
    args=parser.parse_args()
    while not collect():
        if not args.watch:break
        time.sleep(60)
