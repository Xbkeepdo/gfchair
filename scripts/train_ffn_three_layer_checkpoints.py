#!/usr/bin/env python3
"""Paired three-layer checkpoint comparison with the legacy train-loss scheduler."""
from __future__ import annotations

import argparse
import csv
import fcntl
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
sys.path.insert(0,str(ROOT))
from scripts import train_ffn_shallow_standardized_search as search
from scripts import train_ffn_shallow_refit as refit

study = search.study
NAME = 'three_layer_scheduler_checkpoints_20260908'
LEGACY = search.old.TorchProbeConfig()
CFG = search.config((128,64,32))


def output_root(model):
    return study.result_root(model)/'production_k4'/NAME


def prepare(model, device):
    parent = search.output_root(model)
    previous = json.loads((parent/'protocol.json').read_text())
    for name,sha in previous['source_sha256'].items():
        if search.sha256_file(ROOT/name)!=sha:
            raise ValueError('Changed frozen source')
    data = study.load_training_data(model,'fp32_k4',True)
    if search.old.previous.original_fingerprint(data)!=previous['original_fingerprint']:
        raise ValueError('Changed original features/mentions')
    split_path = ROOT/'outputs'/model/study.EXPERIMENT/'image_splits.json'
    splits = json.loads(split_path.read_text())
    if (len(splits['train']),len(splits['test']))!=(3200,800) or search.old.previous.inner_image_split(splits['train'],splits['test'])!=previous['image_split']:
        raise ValueError('Changed original image split')
    direct = {k:data[k] for k in ('train_mentions','test_mentions','y_train','y_test')}
    for part in ('train','test'):
        ae = data['X_'+part]['AE']
        direct['X_'+part] = search.old.direct_features(ae,data['X_'+part]['raw_AE+S+N_vec'][:,ae.shape[1]:2*ae.shape[1]])
    inner = {k.replace('_test','_val'):v for k,v in search.old.inner_inputs(direct,previous['image_split']).items()}
    outer = {k:direct[k] for k in ('X_test','y_test')}
    full = {k:direct[k] for k in ('X_train','y_train')}
    if {k:refit.digest(v) for k,v in {**inner,**outer}.items()}!=previous['matrix_sha256']:
        raise ValueError('Changed original matrices/order')
    scalers = {name:search.fit_scaler(x['X_train']) for name,x in (('inner',inner),('full',full))}
    sources = [parent/'protocol.json',parent/'summary.json',refit.output_root(model)/'summary.json']
    protocol = dict(version=NAME,model=model,config=CFG,seeds=list(search.SEEDS),image_split=previous['image_split'],
        original_fingerprint=previous['original_fingerprint'],feature=previous['feature'],feature_dim=previous['feature_dim'],
        matrix_sha256={scope:{k:refit.digest(v) for k,v in x.items()} for scope,x in (('inner',inner),('full',full),('outer',outer))},
        scalers={name:{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in scaler.items()} for name,scaler in scalers.items()},
        scheduler=dict(type='ReduceLROnPlateau',monitor='train_loss',mode='min',factor=LEGACY.lr_factor,patience=LEGACY.lr_patience,
            default_threshold=1e-4,threshold_mode='rel',cooldown=0,min_lr=0,eps=1e-8),scheduler_controls=[False,True],
        common_training=dict(max_epochs=LEGACY.num_epochs,stop_monitor='train_loss',stop_patience=LEGACY.early_stopping_patience,
            optimizer='AdamW',loss='unweighted BCE; train-mode minibatch mean as legacy',no_validation_stopping=True),
        checkpoints=['max_val_auc','min_train_loss'],
        full_refit='Freeze median inner max-val-AUC epoch; all3200 scaler; fixed-epoch AUC transfer versus direct3200 min-train-loss with100/10. Same-seed common training prefixes must match.',
        evaluation='All choices and fits precede800-image test; train-REAL-F1 threshold uses corresponding optimization cohort; also0.5',
        device=device,packages={p:search.version(p) for p in ('torch','numpy','scikit-learn')},
        AMP=False,TF32=False,inference_batch_size=256,
        numerical_exception=data['numerical_exception'],source_sha256={**previous['source_sha256'],
            **{str(p.relative_to(ROOT)):search.sha256_file(p) for p in (Path(__file__),Path(search.__file__),Path(refit.__file__))}},
        parent_sha256={str(p):search.sha256_file(p) for p in sources},no_bootstrap=True,independent_2000_run=False,
        limitation='Previously explored800; exploratory only; not a batch/architecture search or old-recipe replication')
    study.immutable_json(protocol,output_root(model)/'protocol.json')
    return inner,full,outer,scalers,protocol


def trajectory(seed, train, scaler, path, signature, device, use_scheduler, validation=None,
               fixed_epoch=None, max_epochs=LEGACY.num_epochs, patience=LEGACY.early_stopping_patience):
    """Shared train trajectory; no test data. Fixed refits never use validation."""
    if set(train)!={'X_train','y_train'} or (validation is not None and set(validation)!={'X_val','y_val'}):
        raise ValueError('Trainer accepts only explicit train/validation inputs')
    if fixed_epoch is not None and (validation is not None or type(fixed_epoch) is not int or not 1<=fixed_epoch<=max_epochs):
        raise ValueError('Fixed refit epoch must be frozen, positive and validation-free')
    path = Path(path); destination = path/'result.pt'
    identity = search.key(dict(signature=signature,seed=seed,path=str(path),device=device,scheduler=use_scheduler,
        validation=validation is not None,fixed_epoch=fixed_epoch,max_epochs=max_epochs,patience=patience,config=CFG))
    x = search.transform(train['X_train'],scaler)
    val_x = search.transform(validation['X_val'],scaler) if validation is not None else None
    if destination.exists():
        result = torch.load(destination,map_location='cpu',weights_only=False)
        if result['fingerprint']!=identity:
            raise ValueError('Wrong trajectory resume fingerprint')
        for name,sha in result['artifacts'].items():
            if search.sha256_file(name)!=sha:
                raise ValueError('Changed completed trajectory')
    else:
        search._set_seed(seed)
        attempt = path/f'attempt_{time.time_ns()}'; attempt.mkdir(parents=True,exist_ok=False)
        started = time.perf_counter()
        if str(device).startswith('cuda'):
            torch.empty(0,device=device); torch.cuda.reset_peak_memory_stats(device)
        net = search.make_probe(x.shape[1],CFG).to(device)
        optimizer = torch.optim.AdamW(net.parameters(),lr=CFG['learning_rate'],weight_decay=CFG['weight_decay'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=LEGACY.lr_factor,
                                                               patience=LEGACY.lr_patience) if use_scheduler else None
        dataset = search.MatrixDataset(x,train['y_train']); criterion = torch.nn.BCEWithLogitsLoss()
        best_loss,best_auc,stale = float('inf'),(-float('inf'),-float('inf')),0
        selected,history = {},[]
        for epoch in range(1,(fixed_epoch or max_epochs)+1):
            lr_before = float(optimizer.param_groups[0]['lr'])
            loss = search._train_epoch(net,search.epoch_loader(dataset,CFG['batch_size']),optimizer,criterion,torch.device(device))
            if not np.isfinite(loss):
                raise ValueError('Nonfinite train loss')
            loss_improved = loss<best_loss
            if loss_improved:
                best_loss,stale = loss,0
            else:
                stale += 1
            score = search.validation_score(validation['y_val'],search.probabilities(net,val_x,device)) if validation is not None else None
            auc_improved = score is not None and score>best_auc
            if auc_improved:
                best_auc = score
            rules = []
            if fixed_epoch is not None:
                if epoch==fixed_epoch:
                    rules.append('val_epoch_transfer')
            else:
                if loss_improved:
                    rules.append('min_train_loss')
                if auc_improved:
                    rules.append('max_val_auc')
            for rule in rules:
                selected[rule] = dict(epoch=epoch,state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()})
            if scheduler:
                scheduler.step(loss)
            history.append(dict(epoch=epoch,train_loss=loss,validation_auc=score[0] if score else None,
                validation_hall_aupr=score[1] if score else None,lr_before=lr_before,lr_after=float(optimizer.param_groups[0]['lr']),
                is_best_loss=loss_improved,is_best_auc=auc_improved,stale_train_epochs=stale))
            if fixed_epoch is None and stale>=patience:
                break
        values = {}
        for rule,item in selected.items():
            net.load_state_dict(item['state'])
            checkpoint = attempt/f'{rule}.pt'
            hid = search.key(dict(trajectory=identity,rule=rule))[:20]
            ck_identity = search.key(dict(trajectory=identity,rule=rule,epoch=item['epoch']))
            search.atomic_torch_save(dict(state_dict=item['state'],config=CFG,input_dim=x.shape[1],scaler=scaler,
                fingerprint=ck_identity,seed=seed,epoch=item['epoch'],rule=rule),checkpoint)
            v = dict(head_id=hid,fingerprint=ck_identity,seed=seed,config=CFG,checkpoint=str(checkpoint),
                selected_epoch=item['epoch'],train_probabilities=search.probabilities(net,x,device))
            if validation is not None:
                v['val_probabilities'] = search.probabilities(net,val_x,device)
                v['validation_auc'],v['validation_hall_aupr'] = search.validation_score(validation['y_val'],v['val_probabilities'])
            values[rule] = v
        search.atomic_json_save(history,attempt/'history.json')
        result = dict(fingerprint=identity,seed=seed,scheduler=use_scheduler,fixed_epoch=fixed_epoch,
            epochs_run=len(history),history=str(attempt/'history.json'),checkpoints=values,
            elapsed_seconds=time.perf_counter()-started,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if str(device).startswith('cuda') else 0,
            artifacts={str(p):search.sha256_file(p) for p in attempt.iterdir() if p.is_file()})
    h = json.loads(Path(result['history']).read_text())
    expected = {'val_epoch_transfer':fixed_epoch} if fixed_epoch is not None else {'min_train_loss':min(h,key=lambda r:r['train_loss'])['epoch']}
    if validation is not None:
        expected['max_val_auc'] = max(h,key=lambda r:(r['validation_auc'],r['validation_hall_aupr'],-r['epoch']))['epoch']
    if set(result['checkpoints'])!=set(expected) or result['epochs_run']!=len(h):
        raise ValueError('Wrong saved trajectory selection')
    errors = {}
    for rule,v in result['checkpoints'].items():
        c = torch.load(v['checkpoint'],weights_only=False,map_location='cpu')
        if c['fingerprint']!=v['fingerprint'] or c['epoch']!=expected[rule] or v['selected_epoch']!=expected[rule] or c['config']!=CFG:
            raise ValueError('Wrong selected checkpoint')
        for k in scaler:
            np.testing.assert_array_equal(scaler[k],c['scaler'][k])
        net = search.make_probe(c['input_dim'],CFG).to(device); net.load_state_dict(c['state_dict'])
        for part,matrix in [('train',x)]+([('val',val_x)] if validation is not None else []):
            pred = search.probabilities(net,matrix,device)
            error = float(np.max(np.abs(pred-v[part+'_probabilities'])))
            if not np.isfinite(pred).all() or error>1e-7:
                raise ValueError('Checkpoint recomputation discrepancy')
            errors[rule+'/'+part] = error
        if validation is not None and search.validation_score(validation['y_val'],v['val_probabilities'])!=(v['validation_auc'],v['validation_hall_aupr']):
            raise ValueError('Changed saved validation metrics')
    if not destination.exists():
        result['recomputation_max_error'] = errors
        search.atomic_torch_save(result,destination)
        compact = {**result,'checkpoints':{k:{a:b for a,b in v.items() if not a.endswith('_probabilities')} for k,v in result['checkpoints'].items()}}
        search.atomic_json_save(compact,path/'result.json')
    return result


def run_model(model, device):
    root = output_root(model); root.mkdir(parents=True,exist_ok=True)
    with (root/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        started = time.perf_counter()
        inner,full,outer,scalers,protocol = prepare(model,device)
        signature = search.key(protocol)
        train = {k:inner[k] for k in ('X_train','y_train')}; val = {k:inner[k] for k in ('X_val','y_val')}
        runs,groups,epochs = {},{},{}
        for scheduled in (False,True):
            label = 'schedule_on' if scheduled else 'schedule_off'
            values = []
            for seed in search.SEEDS:
                path = root/'trajectories'/'inner'/label/f'seed{seed}'
                r = trajectory(seed,train,scalers['inner'],path,signature,device,scheduled,validation=val)
                runs[str(path)] = r; values.append(r)
                print('INNER',model,label,seed,r['epochs_run'],{k:v['selected_epoch'] for k,v in r['checkpoints'].items()},flush=True)
            epochs[label] = int(statistics.median(r['checkpoints']['max_val_auc']['selected_epoch'] for r in values))
            for rule in ('max_val_auc','min_train_loss'):
                groups['inner/'+label+'/'+rule] = [r['checkpoints'][rule] for r in values]
        study.immutable_json(dict(status='FROZEN_BEFORE_FULL_REFIT_AND_OUTER_EVALUATION',epochs=epochs,
            protocol_sha256=search.sha256_file(root/'protocol.json'),inner_artifacts={p:search.sha256_file(Path(p)/'result.pt') for p in runs}),root/'epoch_freeze.json')
        for scheduled in (False,True):
            label = 'schedule_on' if scheduled else 'schedule_off'
            for rule in ('val_epoch_transfer','min_train_loss'):
                values = []
                for seed in search.SEEDS:
                    path = root/'trajectories'/'full'/label/rule/f'seed{seed}'
                    r = trajectory(seed,full,scalers['full'],path,signature,device,scheduled,
                                   fixed_epoch=epochs[label] if rule=='val_epoch_transfer' else None)
                    runs[str(path)] = r; values.append(r['checkpoints'][rule])
                    print('REFIT',model,label,rule,seed,r['epochs_run'],r['checkpoints'][rule]['selected_epoch'],flush=True)
                groups['full/'+label+'/'+rule] = values
            for seed in search.SEEDS:
                pair=[runs[str(root/'trajectories'/'full'/label/r/f'seed{seed}')] for r in ('val_epoch_transfer','min_train_loss')]
                histories=[json.loads(Path(r['history']).read_text()) for r in pair]
                length=min(map(len,histories))
                if histories[0][:length]!=histories[1][:length]:
                    raise ValueError('Same-seed full-refit training prefixes differ')
        study.immutable_json(dict(status='FROZEN_BEFORE_OUTER_EVALUATION',epochs=epochs,
            protocol_sha256=search.sha256_file(root/'protocol.json'),
            final_head_ids=sorted(v['head_id'] for values in groups.values() for v in values)),root/'selection.json')
        summaries = {}
        for name,values in groups.items():
            inputs = train if name.startswith('inner/') else full
            evaluated = [search.evaluate(v,inputs,outer,root,device) for v in values]
            summaries[name] = search.old.summarize_group({**inputs,**outer},evaluated)
            summaries[name].update(selected_epochs=[v['selected_epoch'] for v in values],config=CFG,
                training_images=2560 if name.startswith('inner/') else 3200)
            print('FINAL',model,name,summaries[name]['ensemble_reports']['fixed_0.5']['auc'],flush=True)
        artifacts = {str(p):search.sha256_file(p) for part in ('trajectories','evaluation') for p in (root/part).rglob('*') if p.is_file()}
        study.immutable_json(dict(status='PASS_CHECKPOINTS_METRICS_AND_SHARED_TRAJECTORIES',artifacts=artifacts,
            trajectories=len(runs),evaluated_checkpoints=sum(map(len,groups.values())),
            protocol_sha256=search.sha256_file(root/'protocol.json'),selection_sha256=search.sha256_file(root/'selection.json')),
            root/'verification.json')
        old_search = json.loads((search.output_root(model)/'summary.json').read_text())
        old_refit = json.loads((refit.output_root(model)/'summary.json').read_text())
        references=dict(previous_2560_three=old_search['groups']['fixed_arch5'],previous_3200_three=old_refit['groups']['fixed_arch5'],
                        legacy_3200_three=old_search['groups']['old_three_hidden_direct_reference'])
        summary = dict(status='COMPLETE_EXPLORATORY_THREE_LAYER_CHECKPOINTS',model=model,groups=summaries,epochs=epochs,
            references=references,trajectories=len(runs),evaluated_checkpoints=sum(map(len,groups.values())),
            fit_seconds=sum(r['elapsed_seconds'] for r in runs.values()),
            peak_cuda_allocated_bytes=max(r['peak_cuda_allocated_bytes'] for r in runs.values()),
            numerical_exception=protocol['numerical_exception'],verification_sha256=search.sha256_file(root/'verification.json'))
        study.immutable_json(summary,root/'summary.json')
        print('COMPLETE',model,'seconds',time.perf_counter()-started,flush=True)


def publish():
    paths = {m:output_root(m)/'summary.json' for m in study.MODELS}
    if not all(p.exists() for p in paths.values()):
        return False
    root = ROOT/'outputs'/study.VERSION/NAME; root.mkdir(parents=True,exist_ok=True)
    with (root/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        summaries = {m:json.loads(p.read_text()) for m,p in paths.items()}
        for m,s in summaries.items():
            if search.sha256_file(output_root(m)/'verification.json')!=s['verification_sha256']:
                raise ValueError('Changed verification')
        study.immutable_json(dict(status='COMPLETE_FOUR_MODEL_CHECKPOINT_COMPARISON',models=summaries,
            sources={m:dict(path=str(p),sha256=search.sha256_file(p)) for m,p in paths.items()}),root/'summary.json')
        lines=['# 三层：学习率调度与checkpoint选择','',
            '固定全AE+log1p(S)、[128,64,32]、StandardScaler、BN-off、AdamW、dropout.1、lr.001、wd1e-5、batch128。',
            '内层同轨比较：2560优化/640验证；最高验证AUROC与最低训练损失；共同100epochs上限、train-loss patience10。',
            '调度开/关对照；开启时精确复用旧ReduceLROnPlateau(train_loss,factor.5,patience5)。',
            '完整3200重训：迁移内层max-AUC epoch中位数、固定末轮；对比直接3200最小训练损失。不是在800图上选AUC。',
            '所有新结果为旧800图的探索性评价；数值FAIL保留，无bootstrap、新2000图或batch网格。','',
            '## 三seed概率ensemble AUROC / HALL-AUPR（%）','',
            '| 组 | '+' | '.join(study.MODELS)+' |','|---|'+'---:|'*4]
        names=list(next(iter(summaries.values()))['groups'])+list(next(iter(summaries.values()))['references'])
        rows=[]
        for name in names:
            cells=[]
            for m,s in summaries.items():
                g={**s['groups'],**s['references']}[name];r=g['ensemble_reports']['fixed_0.5']
                cells.append(f"{100*r['auc']:.3f} / {100*r['hallucination_positive']['aupr']:.3f}")
                for seed,v in list(g['per_seed_metrics'].items())+ [('ensemble',None)]:
                    for rule in ('fixed_0.5','train_f1'):
                        metric=g['ensemble_reports'][rule] if v is None else v['threshold_reports'][rule]['test_metrics']
                        t=(.5 if rule=='fixed_0.5' else g['ensemble_train_f1_threshold']) if v is None else v['threshold_reports'][rule]['threshold']
                        row=dict(model=m,group=name,seed=seed,threshold_rule=rule,threshold=t,auc=metric['auc'])
                        for label in ('real','hallucination'):
                            row.update({label+'_'+k:metric[label+'_positive'][k] for k in ('aupr','precision','recall','f1')})
                        rows.append(row)
            lines.append('| '+name+' | '+' | '.join(cells)+' |')
        lines+=['','## 完整3200重训的固定AUC迁移轮数','']
        lines += [f"- {m}: {json.dumps(s['epochs'])}" for m,s in summaries.items()]
        lines+=['','72条训练轨迹、96个checkpoint评价；完整逐seed/均值标准差、两阈值及实际epoch/LR在JSON/CSV和各模型history中。']
        refit.immutable_text('\n'.join(lines)+'\n',root/'summary.md')
        buffer=io.StringIO(newline=''); writer=csv.DictWriter(buffer,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows);refit.immutable_text(buffer.getvalue(),root/'groups.csv')
        print('PUBLISHED_LOCAL',root,flush=True)
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',nargs='+',choices=study.MODELS);parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--summarize-only',action='store_true');args=parser.parse_args()
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
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
