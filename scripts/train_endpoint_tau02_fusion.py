"""Fuse the existing v1 endpoint cosine-JS tau0.2 trajectories with matched F."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import train_ffn_ae_log1p_search as trainer
from scripts.analyze_ffn_visual_source_study import _detector_matrices
from scripts.analyze_ffn_endpoint_cosine_js import MODELS, MODEL_NAMES, _write_csv
from scripts.run_ffn_visual_source_attribution import result_root, EXPERIMENT
from features.tc_fvpa_artifacts import sha256_file, atomic_torch_save, atomic_json_save
from scripts.run_cqb_workflow import canonical_json

PARENT=ROOT/'outputs/endpoint_temperatures4000'
OUT=ROOT/'outputs/endpoint_tau02_fusion4000'


def build_groups(formal_f, js, definition):
    formal_f=np.asarray(formal_f,dtype=np.float32)
    js=np.asarray(js,dtype=np.float32)
    if formal_f.ndim!=2 or js.ndim!=2 or formal_f.shape!=(len(js),3*js.shape[1]):
        raise ValueError('F and JS must have the same rows and layers')
    if not np.isfinite(formal_f).all() or not np.isfinite(js).all():
        raise ValueError('Nonfinite F/JS')
    ae,strength,kappa=np.split(formal_f,3,axis=1)
    if definition=='log_s': f=trainer.direct_features(ae,strength)
    elif definition=='raw_s_kappa': f=formal_f
    else: raise ValueError('Unknown F definition')
    return dict(F=f,F_J=np.concatenate([f,js],axis=1).astype(np.float32))


def prepare(model,device,definition):
    parent=PARENT/model
    old=json.loads((parent/'protocol.json').read_text())
    for p,h in old['sources'].items():
        if sha256_file(p)!=h:raise ValueError(f'Changed JS source {p}')
    checksum=json.loads((parent/'matrices_checksum.json').read_text())['sha256']
    if sha256_file(parent/'matrices.pt')!=checksum:raise ValueError('Changed JS cache')
    cache=torch.load(parent/'matrices.pt',map_location='cpu',weights_only=False)
    data=_detector_matrices(model)
    split=ROOT/'outputs'/model/EXPERIMENT/'image_splits.json'
    identity=hashlib.sha256(split.read_bytes())
    groups={}
    for s in ('train','test'):
        np.testing.assert_array_equal(cache[f'y_{s}'],data[f'y_{s}'])
        for t,x in cache['matrices'][s].items():
            assert hashlib.sha256(x.tobytes()).hexdigest()==old['matrix_sha'][s][t]
        identity.update(json.dumps(data[f'{s}_target_keys']).encode())
        identity.update(data[f'y_{s}'].tobytes())
        identity.update(cache['matrices'][s]['1.0'].tobytes())
        groups[s]=build_groups(data[f'X_{s}']['F'],cache['matrices'][s]['0.2'],definition)
    assert identity.hexdigest()==old['baseline_matrix_sha'], 'F/JS mention order differs'
    protocol=dict(model=model,device=device,f_definition=definition,
        f_formula='AE+log1p(raw S)' if definition=='log_s' else 'AE+raw S+kappa',
        source='All F blocks and temperature0.2 JS use the same v1 saved extraction; not the v2 Q-softmax experiment',
        js_formula='JS(softmax(cos(e_m,G(Z)-G(Z0))/0.2),T), full support',
        counts=data['counts'],split_sha=sha256_file(split),parent_protocol_sha=sha256_file(parent/'protocol.json'),
        parent_cache_sha=checksum,source_shards=old['sources'],target_order_sha=identity.hexdigest(),
        classifier=trainer.fixed_mlp(),defaults=vars(trainer.TorchProbeConfig()),seeds=[43,44,45],
        matrices_sha={s:{k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in gs.items()} for s,gs in groups.items()},
        implementation={str(p):sha256_file(p) for p in [Path(__file__),ROOT/'scripts/train_ffn_ae_log1p_search.py',
            ROOT/'scripts/train_torch_probe_feature_sets.py',ROOT/'scripts/train_ffn_consistency_alternative_heads.py']})
    return groups,data['y_train'],data['y_test'],protocol


def run(model,device,definition):
    groups,ytrain,ytest,protocol=prepare(model,device,definition)
    root=OUT/definition/model;root.mkdir(parents=True,exist_ok=True)
    canonical_json(protocol,root/'protocol.json'); signature=sha256_file(root/'protocol.json')
    path=root/'matrices.pt'
    if not path.exists():atomic_torch_save(dict(groups=groups,y_train=ytrain,y_test=ytest),path)
    canonical_json(dict(sha256=sha256_file(path)),root/'matrices_checksum.json')
    results={}; replay={}
    for name in ['F','F_J']:
        data=dict(X_train=groups['train'][name],X_test=groups['test'][name],y_train=ytrain,y_test=ytest)
        heads=[trainer.run_head('three_hidden',trainer.fixed_mlp(),seed,data,root/'heads'/name/f'seed{seed}',signature,device)
               for seed in (43,44,45)]
        results[name]=trainer.summarize_group(data,heads)
        replay[name]={str(h['seed']):h['recomputation_max_error'] for h in heads}
        print(model,name,json.dumps(results[name]['ensemble_reports']['fixed_0.5']),flush=True)
    js=json.loads((PARENT/model/'summary.json').read_text())['groups']['0.2']
    value=dict(status='COMPLETE',model=model,f_formula=protocol['f_formula'],groups=dict(J=js,**results),replay=replay,
        delta_ensemble_auc=results['F_J']['ensemble_reports']['fixed_0.5']['auc']-results['F']['ensemble_reports']['fixed_0.5']['auc'],
        delta_ensemble_hall_aupr=results['F_J']['ensemble_reports']['fixed_0.5']['hallucination_positive']['aupr']-results['F']['ensemble_reports']['fixed_0.5']['hallucination_positive']['aupr'])
    canonical_json(value,root/'summary.json')


def summarize(definition):
    root=OUT/definition
    results={m:json.loads((root/m/'summary.json').read_text()) for m in MODELS}
    lines=['# 温度0.2 endpoint JS与F融合', '', 'F='+results[MODELS[0]]['f_formula']+'；全部采用同源v1特征。原3200/800图片，旧MLP三seed；表中为ensemble AUROC / HALL-AUPR。', '',
           '| 模型 | J (tau=.2) | F | F+J | AUROC增量 | HALL-AUPR增量 |','|---|---:|---:|---:|---:|---:|']
    rows=[]
    for m,r in results.items():
        cells=[]
        for name in ['J','F','F_J']:
            v=r['groups'][name]['ensemble_reports']['fixed_0.5']
            cells.append(f'{v["auc"]:.6f} / {v["hallucination_positive"]["aupr"]:.6f}')
            for rule,report in r['groups'][name]['ensemble_reports'].items():
                rows.append(dict(model=m,group=name,threshold_rule=rule,auroc=report['auc'],
                    **{f'{label}_{metric}':report[key][metric] for label,key in [('real','real_positive'),('hall','hallucination_positive')]
                       for metric in ('aupr','precision','recall','f1')}))
        lines.append('| '+MODEL_NAMES[m]+' | '+' | '.join(cells)+f' | {r["delta_ensemble_auc"]:+.6f} | {r["delta_ensemble_hall_aupr"]:+.6f} |')
    lines+=['','F与F+J同配置重训；J复用已验收温度0.2结果。已反复使用旧测试集，结果为探索性点估计，未做bootstrap或验证集调参。']
    (root/'summary.md').write_text('\n'.join(lines)+'\n')
    atomic_json_save(results,root/'summary.json');_write_csv(root/'ensemble_metrics.csv',rows)
    print('\n'.join(lines))


def self_check():
    f=np.array([[.2,.3,3.,8.,.4,.5]],dtype=np.float32);j=np.array([[.1,.6]],dtype=np.float32)
    log=build_groups(f,j,'log_s');raw=build_groups(f,j,'raw_s_kappa')
    np.testing.assert_allclose(log['F'],[[.2,.3,np.log(4),np.log(9)]],atol=1e-7)
    np.testing.assert_array_equal(log['F_J'][:,:4],log['F']);np.testing.assert_array_equal(log['F_J'][:,4:],j)
    np.testing.assert_array_equal(raw['F'],f)
    try:build_groups(f,j[:,:1],'log_s')
    except ValueError:pass
    else:raise AssertionError('Layer mismatch accepted')
    print('F definition, block concatenation and shape checks PASS')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',nargs='+',choices=MODELS,default=list(MODELS));p.add_argument('--device',default='cpu')
    p.add_argument('--f-definition',choices=['log_s','raw_s_kappa'])
    p.add_argument('--summarize',action='store_true');p.add_argument('--self-check',action='store_true')
    args=p.parse_args();torch.set_num_threads(1)
    if args.self_check:self_check()
    elif args.f_definition is None:p.error('--f-definition is required')
    elif args.summarize:summarize(args.f_definition)
    else:
        for m in args.models:run(m,args.device,args.f_definition)
