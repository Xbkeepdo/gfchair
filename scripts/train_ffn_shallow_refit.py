#!/usr/bin/env python3
"""Refit the frozen shallow-search groups on all 3200 images; no new search."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import io
import json
from pathlib import Path
import statistics
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import train_ffn_shallow_standardized_search as search

study = search.study
NAME = 'shallow_refit3200_20260908'


def output_root(model):
    return study.result_root(model)/'production_k4'/NAME


def digest(array):
    return hashlib.sha256(str(array.shape).encode()+array.tobytes()).hexdigest()


def fixed_epochs(records):
    if sorted(r['seed'] for r in records)!=list(search.SEEDS):
        raise ValueError('Exactly the three original validation seeds required')
    values = [r['best_epoch'] for r in records]
    if any(type(v) is not int or not 1<=v<=search.MAX_EPOCHS for v in values):
        raise ValueError('Invalid original validation epoch')
    return int(statistics.median(values))


def prepare(model, device):
    parent = search.output_root(model)
    previous = json.loads((parent/'protocol.json').read_text())
    selection = json.loads((parent/'selection.json').read_text())
    if selection['protocol_sha256']!=search.sha256_file(parent/'protocol.json'):
        raise ValueError('Changed parent protocol')
    for name,sha in previous['source_sha256'].items():
        if search.sha256_file(ROOT/name)!=sha:
            raise ValueError('Changed frozen search implementation')
    data = study.load_training_data(model,'fp32_k4',True)
    if search.old.previous.original_fingerprint(data)!=previous['original_fingerprint']:
        raise ValueError('Changed original features/mentions')
    splits_path = ROOT/'outputs'/model/study.EXPERIMENT/'image_splits.json'
    splits = json.loads(splits_path.read_text())
    if (len(splits['train']),len(splits['test']))!=(3200,800):
        raise ValueError('Original 3200/800 split required')
    if search.old.previous.inner_image_split(splits['train'],splits['test'])!=previous['image_split']:
        raise ValueError('Changed image split')
    direct = {k:data[k] for k in ('train_mentions','test_mentions','y_train','y_test')}
    for part in ('train','test'):
        ae = data['X_'+part]['AE']
        strength = data['X_'+part]['raw_AE+S+N_vec'][:,ae.shape[1]:2*ae.shape[1]]
        direct['X_'+part] = search.old.direct_features(ae,strength)
    inner = {k.replace('_test','_val'):v for k,v in search.old.inner_inputs(direct,previous['image_split']).items()}
    matrices = {**inner,'X_test':direct['X_test'],'y_test':direct['y_test']}
    if {k:digest(v) for k,v in matrices.items()}!=previous['matrix_sha256']:
        raise ValueError('Changed search/test matrices or row order')
    provenance = {str(parent/name):search.sha256_file(parent/name)
                  for name in ('protocol.json','selection.json','summary.json','stage1.json','verification.json')}
    groups = {}
    for name,cfg in selection['group_configs'].items():
        records = []
        for seed in search.SEEDS:
            hid = search.key(dict(config=cfg,seed=seed))[:20]
            path = parent/'heads'/hid/'result.json'
            r = json.loads(path.read_text())
            if r['config']!=cfg or r['seed']!=seed or r['head_id']!=hid or hid not in selection['final_head_ids']:
                raise ValueError('Wrong parent selected head')
            for artifact,sha in r['artifacts'].items():
                if search.sha256_file(artifact)!=sha:
                    raise ValueError('Changed parent head artifact')
            history = json.loads((Path(r['checkpoint']).parent/'history.json').read_text())
            best = max(history,key=lambda v:(v['validation_auc'],v['validation_hall_aupr'],-v['epoch']))
            if r['best_epoch']!=best['epoch']:
                raise ValueError('Parent epoch is not validation-selected')
            provenance[str(path)] = search.sha256_file(path)
            records.append(r)
        groups[name] = dict(config=cfg,epochs=fixed_epochs(records),
                            original_best_epochs=[r['best_epoch'] for r in records])
    if len(groups)!=10 or groups['fixed_arch5']['config']['hidden_sizes']!=[128,64,32]:
        raise ValueError('All ten frozen groups, including three-hidden control, required')
    scaler = search.fit_scaler(direct['X_train'])
    protocol = dict(version=NAME,model=model,feature=previous['feature'],feature_dim=previous['feature_dim'],
        image_split=splits,source_image_splits_sha256=search.sha256_file(splits_path),
        mention_sha256={p:search.key(direct[p+'_mentions']) for p in ('train','test')},
        original_fingerprint=previous['original_fingerprint'],parent_artifacts=provenance,
        groups=groups,primary_group=f"tuned_arch{selection['best_tuned_mlp']['index']}",seeds=list(search.SEEDS),
        epoch_rule='Integer median of original three validation-best epochs for each config, same for refit43/44/45; no step-count rescaling',
        training=dict(optimizer='AdamW',checkpoint='fixed final epoch; fresh initialization, no early stopping',
            train_images=3200,test_images=800,validation_images=0,normalization='refit StandardScaler on all3200 training mentions only',
            threshold='REAL-F1 on all3200 training mentions',dtype='FP32',AMP=False,TF32=False,
            scheduler=None,class_weight=None,resampling=False,inference_batch_size=256),
        scaler={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in scaler.items()},
        matrix_sha256={k:digest(v) for k,v in direct.items() if k.startswith(('X_','y_'))},
        numerical_exception=data['numerical_exception'],device=device,packages=previous['packages'],
        source_sha256={**previous['source_sha256'],str(Path(search.__file__).relative_to(ROOT)):search.sha256_file(search.__file__),
                       str(Path(__file__).relative_to(ROOT)):search.sha256_file(__file__)},
        no_bootstrap=True,independent_2000_run=False,
        limitation='Follow-up on previously explored800, not independent confirmation; fixed three-layer control was not stage2-tuned; numerical FAIL retained')
    study.immutable_json(protocol,output_root(model)/'protocol.json')
    frozen = dict(status='FROZEN_BEFORE_OUTER_EVALUATION',protocol_sha256=search.sha256_file(output_root(model)/'protocol.json'),
        group_configs=selection['group_configs'],group_epochs={k:v['epochs'] for k,v in groups.items()},
        primary_group=protocol['primary_group'],final_head_ids=sorted(selection['final_head_ids']))
    study.immutable_json(frozen,output_root(model)/'selection.json')
    return {k:v for k,v in direct.items() if k.startswith(('X_','y_'))},scaler,protocol


def run_head(cfg, seed, train, scaler, root, signature, device, epochs):
    """No validation/test arguments: only the frozen final epoch can be saved."""
    if set(train)!={'X_train','y_train'} or type(epochs) is not int or not 1<=epochs<=search.MAX_EPOCHS:
        raise ValueError('Only training data and a frozen positive epoch budget accepted')
    hid = search.key(dict(config=cfg,seed=seed))[:20]
    path = Path(root)/'heads'/hid; result_path = path/'result.pt'
    identity = search.key(dict(signature=signature,config=cfg,seed=seed,device=device,epochs=epochs))
    x = search.transform(train['X_train'],scaler,cfg['standardize'])
    if result_path.exists():
        result = torch.load(result_path,map_location='cpu',weights_only=False)
        if result['fingerprint']!=identity:
            raise ValueError('Wrong refit resume fingerprint')
        for name,sha in result['artifacts'].items():
            if search.sha256_file(name)!=sha:
                raise ValueError('Changed completed refit artifact')
    else:
        search._set_seed(seed)
        attempt = path/f'attempt_{time.time_ns()}'; attempt.mkdir(parents=True,exist_ok=False)
        started = time.perf_counter()
        if str(device).startswith('cuda'):
            torch.empty(0,device=device); torch.cuda.reset_peak_memory_stats(device)
        model = search.make_probe(x.shape[1],cfg).to(device)
        optimizer = torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        dataset = search.MatrixDataset(x,train['y_train']); criterion = torch.nn.BCEWithLogitsLoss()
        history = []
        for epoch in range(1,epochs+1):
            loss = search._train_epoch(model,search.epoch_loader(dataset,cfg['batch_size']),optimizer,criterion,torch.device(device))
            if not np.isfinite(loss):
                raise ValueError('Nonfinite refit loss')
            history.append(dict(epoch=epoch,train_loss=loss))
        state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        pred = search.probabilities(model,x,device)
        if not np.isfinite(pred).all():
            raise ValueError('Nonfinite refit probabilities')
        checkpoint = attempt/'model.pt'
        search.atomic_torch_save(dict(state_dict=state,config=cfg,input_dim=x.shape[1],scaler=scaler,
            fingerprint=identity,seed=seed,epochs=epochs),checkpoint)
        search.atomic_json_save(history,attempt/'history.json')
        result = dict(head_id=hid,fingerprint=identity,seed=seed,config=cfg,checkpoint=str(checkpoint),
            epochs_run=epochs,train_probabilities=pred,parameter_count=sum(v.numel() for v in model.parameters()),
            elapsed_seconds=time.perf_counter()-started,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if str(device).startswith('cuda') else 0,
            artifacts={str(p):search.sha256_file(p) for p in attempt.iterdir() if p.is_file()})
    ckpt = torch.load(result['checkpoint'],map_location='cpu',weights_only=False)
    if (ckpt['fingerprint']!=identity or ckpt['config']!=cfg or ckpt['seed']!=seed or ckpt['epochs']!=epochs or
            result['epochs_run']!=epochs or result['seed']!=seed or result['config']!=cfg or result['head_id']!=hid):
        raise ValueError('Wrong refit checkpoint metadata')
    for k in scaler:
        np.testing.assert_array_equal(ckpt['scaler'][k],scaler[k])
    replica = search.make_probe(ckpt['input_dim'],cfg).to(device); replica.load_state_dict(ckpt['state_dict'])
    pred = search.probabilities(replica,search.transform(train['X_train'],ckpt['scaler'],cfg['standardize']),device)
    error = float(np.max(np.abs(pred-result['train_probabilities'])))
    if not np.isfinite(pred).all() or error>1e-7:
        raise ValueError('Refit checkpoint probability discrepancy')
    if not result_path.exists():
        result['recomputation_max_error'] = dict(train=error)
        search.atomic_torch_save(result,result_path)
        search.atomic_json_save({k:v for k,v in result.items() if not k.endswith('_probabilities')},path/'result.json')
    return result


def run_model(model, device):
    root = output_root(model); root.mkdir(parents=True,exist_ok=True)
    with (root/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        started = time.perf_counter()
        data,scaler,protocol = prepare(model,device)
        signature = search.key(protocol)
        train = {k:data[k] for k in ('X_train','y_train')}
        outer = {k:data[k] for k in ('X_test','y_test')}
        results = {}
        for name,g in protocol['groups'].items():
            results[name] = []
            for seed in search.SEEDS:
                value = run_head(g['config'],seed,train,scaler,root,signature,device,g['epochs'])
                results[name].append(value)
                print('REFIT',model,name,seed,'epochs',g['epochs'],flush=True)
        # All thirty fresh fits finish before the first new outer evaluation.
        groups = {}
        for name,values in results.items():
            evaluated = [search.evaluate(v,train,outer,root,device) for v in values]
            groups[name] = search.old.summarize_group(data,evaluated)
            groups[name].update(protocol['groups'][name])
            print('FINAL',model,name,groups[name]['ensemble_reports']['fixed_0.5']['auc'],flush=True)
        previous = json.loads((search.output_root(model)/'summary.json').read_text())
        artifacts = {str(p):search.sha256_file(p) for part in ('heads','evaluation') for p in (root/part).rglob('*') if p.is_file()}
        heads = {v['head_id']:v for values in results.values() for v in values}
        study.immutable_json(dict(status='PASS_SAME_DEVICE_CHECKPOINTS_AND_METRICS',artifacts=artifacts,
            trained_unique_heads=len(heads),evaluated_unique_heads=len(heads),
            protocol_sha256=search.sha256_file(root/'protocol.json'),selection_sha256=search.sha256_file(root/'selection.json')),
            root/'verification.json')
        summary = dict(status='COMPLETE_EXPLORATORY_3200_REFIT',model=model,groups=groups,
            primary_group=protocol['primary_group'],feature_dim=protocol['feature_dim'],
            train_mentions=len(data['y_train']),test_mentions=len(data['y_test']),train_images=3200,test_images=800,
            previous_2560_groups=previous['groups'],trained_unique_heads=len(heads),evaluated_unique_heads=len(heads),
            fit_seconds=sum(v['elapsed_seconds'] for v in heads.values()),
            peak_cuda_allocated_bytes=max(v['peak_cuda_allocated_bytes'] for v in heads.values()),
            numerical_exception=protocol['numerical_exception'],verification_sha256=search.sha256_file(root/'verification.json'))
        study.immutable_json(summary,root/'summary.json')
        print('COMPLETE',model,'seconds',time.perf_counter()-started,flush=True)


def immutable_text(value, path):
    if path.exists():
        if path.read_text()!=value:
            raise ValueError('Completed text/table changed')
    else:
        path.write_text(value)


def publish():
    paths = {m:output_root(m)/'summary.json' for m in study.MODELS}
    if not all(p.exists() for p in paths.values()):
        return False
    root = ROOT/'outputs'/study.VERSION/NAME; root.mkdir(parents=True,exist_ok=True)
    with (root/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        summaries = {m:json.loads(p.read_text()) for m,p in paths.items()}
        for m,s in summaries.items():
            if s['verification_sha256']!=search.sha256_file(output_root(m)/'verification.json'):
                raise ValueError('Changed refit verification')
        study.immutable_json(dict(status='COMPLETE_FOUR_MODEL_3200_REFIT',models=summaries,
            sources={m:dict(path=str(p),sha256=search.sha256_file(p)) for m,p in paths.items()}),root/'summary.json')
        lines = ['# AE + log1p(S)：冻结配置后3200图重训','',
            '四模型上一轮10组各3seed，共120个头；StandardScaler只在完整3200训练图mentions重新拟合。',
            '各组轮数固定为上一轮三seed最佳验证epoch的中位数，保存末轮；不再搜索、不以800图挑epoch或结构。',
            '以下均为3seed概率ensemble，AUROC / HALL-AUPR（%）。旧800图已被探索，非独立确认；数值FAIL保留。','',
            '| 组 | '+' | '.join(study.MODELS)+' |','|---|'+'---:|'*4]
        for name in [f'fixed_arch{i}' for i in range(6)]+['primary','matched_BN_on','matched_scaler_off','previous_primary','old_three_hidden_direct_reference']:
            cells = []
            for s in summaries.values():
                if name in ('previous_primary','old_three_hidden_direct_reference'):
                    g = s['previous_2560_groups'][s['primary_group'] if name=='previous_primary' else name]
                else:
                    g = s['groups'][s['primary_group'] if name=='primary' else name]
                r = g['ensemble_reports']['fixed_0.5']
                cells.append(f"{r['auc']*100:.3f} / {r['hallucination_positive']['aupr']*100:.3f}")
            lines.append('| '+name+' | '+' | '.join(cells)+' |')
        lines += ['', '## 冻结配置与训练轮数','']
        for m,s in summaries.items():
            p = s['groups'][s['primary_group']]
            lines.append(f"- {m}：主组`{s['primary_group']}`，`{json.dumps(p['config'],sort_keys=True)}`，{p['epochs']}epochs；三层固定组{ s['groups']['fixed_arch5']['epochs']}epochs。")
        lines += ['', '完整10组、逐seed、均值/标准差、REAL/HALL AUPR与双阈值P/R/F1见summary.json和groups.csv。',
            '三层只参加了上一轮第一阶段固定参数比较，没有进入前二精搜；本轮没有为三层追加精搜。',
            '旧三层参考虽然也用3200图，但BN/Adam/train-loss checkpoint等不同；与本轮不是仅改变深度的对照。']
        immutable_text('\n'.join(lines)+'\n',root/'summary.md')
        rows = []
        for m,s in summaries.items():
            for run,groups in (('refit3200',s['groups']),('previous',s['previous_2560_groups'])):
                for name,g in groups.items():
                    records = [(seed,rule,r['threshold'],r['test_metrics']) for seed,v in g['per_seed_metrics'].items()
                               for rule,r in v['threshold_reports'].items()]
                    records += [('ensemble',rule,.5 if rule=='fixed_0.5' else g['ensemble_train_f1_threshold'],r)
                                for rule,r in g['ensemble_reports'].items()]
                    for seed,rule,threshold,r in records:
                        row = dict(model=m,run=run,group=name,seed=seed,threshold_rule=rule,threshold=threshold,
                            train_images=3200 if run=='refit3200' or name=='old_three_hidden_direct_reference' else 2560,
                            epochs=g.get('epochs',''),config=json.dumps(g.get('config',{}),sort_keys=True),auc=r['auc'])
                        for label in ('real','hallucination'):
                            row.update({label+'_'+k:r[label+'_positive'][k] for k in ('aupr','precision','recall','f1')})
                        rows.append(row)
        buffer = io.StringIO(newline=''); writer = csv.DictWriter(buffer,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
        immutable_text(buffer.getvalue(),root/'groups.csv')
        print('PUBLISHED_LOCAL',root,flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',nargs='+',choices=study.MODELS)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--summarize-only',action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    if args.summarize_only:
        if not publish():
            raise ValueError('Incomplete models')
        return
    if not args.models:
        parser.error('--models required')
    for model in args.models:
        try:
            run_model(model,args.device)
        except Exception:
            search.atomic_json_save(dict(model=model,command=sys.argv,traceback=traceback.format_exc()),
                                    output_root(model)/f'failure_{time.time_ns()}.json')
            raise
    publish()


if __name__=='__main__':
    main()
