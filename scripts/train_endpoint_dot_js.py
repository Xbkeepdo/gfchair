"""Direct e_m dot d -> softmax -> JS(T), matched to the v1 cosine study."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from features.tc_fvpa_artifacts import atomic_json_save, atomic_torch_save, sha256_file
from scripts.analyze_ffn_endpoint_cosine_js import MODELS, MODEL_NAMES, _write_csv
from scripts.analyze_ffn_visual_source_study import _detector_matrices, SEEDS
from scripts.analyze_ffn_visual_source_signal_ablation import train_matrices
from scripts.run_ffn_visual_source_attribution import result_root
from scripts.train_q_softmax_js import q_softmax_js
from scripts.train_ffn_consistency_alternative_heads import predict_checkpoint

SPEC = 'endpoint_dot_js'


def dot_js(row):
    q = torch.as_tensor(row['path_signed_q'], dtype=torch.float64)
    norm = torch.as_tensor(row['net_strength'], dtype=torch.float64)
    if q.ndim != 2 or norm.shape != q.shape[:1] or not torch.isfinite(norm).all() or (norm < 0).any():
        raise ValueError('Expected finite nonnegative endpoint norm per layer')
    return q_softmax_js(q * norm[:, None], row['attention_evidence'])


def run(model, device):
    torch.set_num_threads(1)
    root = result_root(model)
    data = _detector_matrices(model)
    lookup, audits, sources = {}, [], {}
    for path in sorted((root / 'shards/full').glob('features*_shard_*.pt')):
        sources[str(path.relative_to(ROOT))] = sha256_file(path)
        shard = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        for row in shard['positions']:
            key = row['target_key']
            if key in lookup:
                raise ValueError('Duplicate target')
            lookup[key], audit = dot_js(row)
            audit['net_degenerate_layers'] = int(torch.as_tensor(row['net_degenerate']).sum())
            audits.append(audit)
    assert set(lookup) == set(data['train_target_keys']) | set(data['test_target_keys'])
    matrices = {s: {SPEC: np.stack([lookup[k] for k in data[f'{s}_target_keys']])} for s in ('train', 'test')}
    digest = hashlib.sha256(json.dumps(sources, sort_keys=True).encode())
    digest.update(str(device).encode())
    for s in ('train', 'test'):
        digest.update(json.dumps(data[f'{s}_target_keys']).encode())
        digest.update(data[f'y_{s}'].tobytes())
        digest.update(matrices[s][SPEC].tobytes())
    signature = digest.hexdigest()
    progress_path = root / f'metrics/{SPEC}_progress.pt'
    result_path = root / f'metrics/{SPEC}_results.json'
    if progress_path.exists():
        progress = torch.load(progress_path, map_location='cpu', weights_only=False)
        assert progress['cohort_feature_sha256'] == signature
    else:
        atomic_torch_save(dict(metrics={}, predictions={}, cohort_feature_sha256=signature), progress_path)
    if result_path.exists():
        result = json.loads(result_path.read_text())
        assert result['signature'] == signature
        for path, checksum in result['artifacts'].items():
            assert sha256_file(ROOT / path) == checksum, path
        print(model, 'resume verified; no writes', flush=True)
        return
    progress, ensembles, summaries = train_matrices(model, data, matrices, (SPEC,), device=device,
        resume=True, progress_name=progress_path.name, probe_directory=f'metrics/probes_{SPEC}')
    summary = summaries[SPEC]
    summary['ensemble_hall_aupr'] = float(average_precision_score(1-data['y_test'], 1-ensembles[SPEC]))
    replay, configs, seed_rows = {}, {}, []
    for seed in SEEDS:
        path = root / f'metrics/probes_{SPEC}/{SPEC}/seed{seed}'
        config = json.loads((path / 'config.json').read_text())
        configs[str(seed)] = config
        p = predict_checkpoint('torch', path / 'model.pt', config, matrices['test'][SPEC], device)
        error = float(np.max(np.abs(p-progress['predictions'][SPEC][seed])))
        assert np.isfinite(error) and error <= 1e-7
        replay[str(seed)] = error
        for rule, report in progress['metrics'][SPEC][seed]['threshold_reports'].items():
            metric = report['test_metrics']
            seed_rows.append(dict(seed=seed, rule=rule, threshold=report['threshold'], auroc=metric['auc'],
                **{f'{label}_{name}':metric[key][name] for label,key in [('real','real_positive'),('hall','hallucination_positive')]
                   for name in ('aupr','precision','recall','f1')}))
    _write_csv(root / f'tables/{SPEC}_seeds.csv', seed_rows)
    x = np.concatenate([matrices[s][SPEC] for s in ('train','test')])
    labels = np.concatenate([data[f'y_{s}'] for s in ('train','test')])
    rows = []
    for label, name in [(0,'HALL'), (1,'REAL')]:
        selected = x[labels==label]
        for layer in range(x.shape[1]):
            v = selected[:,layer]
            rows.append(dict(model=model, label=name, layer=layer+1, n=len(v), mean=float(v.mean()),
                median=float(np.median(v)), q25=float(np.quantile(v,.25)), q75=float(np.quantile(v,.75))))
    _write_csv(root / f'tables/{SPEC}_curves.csv', rows)
    pmax = np.concatenate([a['p_max'] for a in audits])
    audit = dict(dot_min=min(a['q_min'] for a in audits), dot_max=max(a['q_max'] for a in audits),
        saturation_fraction=float((pmax>.99).mean()), pmax_quantiles=np.quantile(pmax,[0,.5,.9,.99,1]).tolist(),
        zero_softmax_entries=sum(a['softmax_zero_entries'] for a in audits),
        net_degenerate_layers=sum(a['net_degenerate_layers'] for a in audits),
        js_min=float(x.min()), js_max=float(x.max()))
    artifacts = [progress_path, Path(__file__).resolve(), root / f'tables/{SPEC}_seeds.csv', root / f'tables/{SPEC}_curves.csv']
    artifacts += list((root / f'metrics/probes_{SPEC}').glob('*/seed*/*'))
    result = dict(model=model, status='COMPLETE', signature=signature, summary=summary, audit=audit,
        definition='JS(softmax(Q_m * N_end),T); Q_m=<e_m,d/||d||>; N_end=||G(Z)-G(Z0)||; temperature=1',
        source='v1 native extraction, same source as endpoint cosine; FP64 dot/softmax/JS, FP32 probe input',
        configs=configs, device=str(device), counts=data['counts'], replay=replay, sources=sources,
        cosine_baseline=json.loads((root/'metrics/endpoint_cosine_js_feature_results.json').read_text())['summary'],
        artifacts={str(p.relative_to(ROOT)):sha256_file(p) for p in artifacts})
    atomic_json_save(result, result_path)
    print(model, json.dumps(dict(summary=summary, audit=audit)), flush=True)


def summarize():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    results = {m:json.loads((result_root(m)/f'metrics/{SPEC}_results.json').read_text()) for m in MODELS}
    fig, axes = plt.subplots(2,2,figsize=(12,7.5),sharey=True)
    lines = ['# 直接内积 e_m·d 的 softmax–JS 检测', '',
        '同源v1，原3200/800图片，seeds43/44/45，温度1、全视觉support、旧MLP。AUROC/AP为概率ensemble；F1为train-REAL-F1阈值下逐seed均值。', '',
        '| 模型 | 内积 AUROC | HALL-AUPR | Mean HALL-F1 | Cosine AUROC | Pmax>0.99比例 |',
        '|---|---:|---:|---:|---:|---:|']
    for ax, (m,r) in zip(axes.flat, results.items()):
        rows = list(csv.DictReader((result_root(m)/f'tables/{SPEC}_curves.csv').open()))
        for label,color in [('REAL','#2166ac'),('HALL','#b2182b')]:
            selected = [v for v in rows if v['label']==label]
            ax.plot([int(v['layer']) for v in selected],[float(v['mean']) for v in selected],color=color,label=label)
        ax.set(title=MODEL_NAMES[m],xlabel='Decoder layer',ylabel='Mean JS (nats)')
        ax.grid(alpha=.2); ax.legend(frameon=False)
        s=r['summary']
        lines.append(f"| {MODEL_NAMES[m]} | {s['ensemble_auroc']:.6f} | {s['ensemble_hall_aupr']:.6f} | {s['mean_hall_f1']:.6f} | {r['cosine_baseline']['ensemble_auroc']:.6f} | {r['audit']['saturation_fraction']:.2%} |")
    fig.suptitle('JS(softmax(e_m dot [G(Z)-G(Z0)]), T)')
    fig.text(.5,.01,'Arithmetic means | All train + test mentions | No uncertainty band',ha='center')
    fig.tight_layout(rect=(0,.03,1,.95))
    for ext in ('png','pdf'): fig.savefig(ROOT/f'outputs/{SPEC}_mean_curves.{ext}',dpi=180)
    plt.close(fig)
    lines += ['', '这是旧测试集上的探索性结果，未做温度选择、bootstrap或独立确认。']
    (ROOT/f'outputs/{SPEC}_summary.md').write_text('\n'.join(lines)+'\n')
    atomic_json_save(results, ROOT/f'outputs/{SPEC}_summary.json')
    print('\n'.join(lines))


def self_check():
    e = torch.tensor([[2.,0.],[-1.,3.],[0.,0.]],dtype=torch.float64)
    d = torch.tensor([3.,4.],dtype=torch.float64)
    q = e @ (d/d.norm())
    row = dict(path_signed_q=q[None], net_strength=d.norm()[None], attention_evidence=torch.tensor([[.2,.3,.5]]))
    np.testing.assert_allclose(dot_js(row)[0],q_softmax_js((e@d)[None],row['attention_evidence'])[0],atol=1e-8)
    row['net_strength'] = torch.tensor([0.])
    np.testing.assert_allclose(dot_js(row)[0],q_softmax_js(torch.zeros(1,3),row['attention_evidence'])[0])
    row['net_strength'] = torch.tensor([1e6])
    assert np.isfinite(dot_js(row)[0]).all()
    print('dot identity / zero direction / extreme softmax checks PASS')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',nargs='+',choices=MODELS,default=list(MODELS))
    p.add_argument('--device',default='cpu')
    p.add_argument('--summarize',action='store_true')
    p.add_argument('--self-check',action='store_true')
    args=p.parse_args()
    if args.self_check: self_check()
    elif args.summarize: summarize()
    else:
        for model in args.models: run(model,torch.device(args.device))
