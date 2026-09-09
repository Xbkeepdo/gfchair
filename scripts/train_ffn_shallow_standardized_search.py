#!/usr/bin/env python3
"""Two-stage shallow probes on existing AE + log1p(raw S), image-held-out selection."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
from importlib.metadata import version
from itertools import product
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import BatchSampler, DataLoader, RandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import train_ffn_ae_log1p_search as old
from scripts.train_torch_probe_feature_sets import DGSTStyleProbe, MatrixDataset, _set_seed, _train_epoch
from features.tc_fvpa_artifacts import atomic_json_save, atomic_torch_save, sha256_file, sha256_text

study = old.study
NAME = 'shallow_standardized_search_20260908'
ARCHITECTURES = ((), (128,), (256,), (128,64), (256,128), (128,64,32))
SEEDS = (43,44,45)
MAX_EPOCHS = 150
PATIENCE = 15


def output_root(model):
    return study.result_root(model)/'production_k4'/NAME


def config(hidden, **updates):
    value = dict(hidden_sizes=list(hidden), learning_rate=.001, weight_decay=1e-5,
                 dropout=.1 if hidden else 0., batch_size=128, batch_norm=False, standardize=True)
    value.update(updates)
    if not hidden:
        value.update(dropout=0., batch_norm=False)
    return value


def grid(hidden):
    return [config(hidden,learning_rate=lr,dropout=drop,weight_decay=wd)
            for lr,drop,wd in product((1e-4,3e-4,1e-3,3e-3),
                                     (0.,.1,.2,.3) if hidden else (0.,), (0.,1e-5,1e-4))]


def key(value):
    return sha256_text(json.dumps(value,sort_keys=True))


def make_probe(dim, cfg):
    model = DGSTStyleProbe(dim,cfg['hidden_sizes'],cfg['dropout'])
    if not cfg['batch_norm']:
        model.net = nn.Sequential(*(m for m in model.net if not isinstance(m,nn.BatchNorm1d)))
    return model


def fit_scaler(x):
    if x.ndim!=2 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Invalid training features for StandardScaler')
    scaler = StandardScaler().fit(x)
    return dict(mean=scaler.mean_,scale=scaler.scale_,var=scaler.var_,n_samples=int(scaler.n_samples_seen_))


def transform(x, scaler, enabled=True):
    if not np.isfinite(x).all() or x.ndim!=2 or x.shape[1]!=len(scaler['mean']):
        raise ValueError('Invalid feature matrix or scaler dimension')
    # Match sklearn's float32 transform (including its two rounding steps).
    result = np.asarray(x,dtype=np.float32).copy()
    if enabled:
        result -= scaler['mean']
        result /= scaler['scale']
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite standardized features')
    return result


def probabilities(model, x, device):
    model.eval()
    with torch.no_grad():
        return np.concatenate([torch.sigmoid(model(torch.from_numpy(x[i:i+256]).to(device)))
                               .flatten().cpu().numpy() for i in range(0,len(x),256)])


def validation_score(y, p):
    if len(np.unique(y))!=2 or not np.isfinite(p).all():
        raise ValueError('Finite validation predictions and both labels required')
    return float(roc_auc_score(y,p)), float(average_precision_score(1-y,1-p))


def epoch_loader(dataset, batch_size):
    batches = list(BatchSampler(RandomSampler(dataset),batch_size,drop_last=False))
    # Keep every mention; merge a last singleton so BN-on never drops a sample.
    if len(batches)>1 and len(batches[-1])==1:
        batches[-2].extend(batches.pop())
    return DataLoader(dataset,batch_sampler=batches)


def rank(rows):
    if not rows or any(r.get('scope')!='inner_validation' for r in rows):
        raise ValueError('Selection accepts inner-validation-only rows')
    if any(not np.isfinite([r['auc'],r['hall_aupr']]).all() for r in rows):
        raise ValueError('Nonfinite selection scores')
    return sorted(rows,key=lambda r:(-r['auc'],-r['hall_aupr'],r['index']))


def row(index, cfg, values):
    return dict(scope='inner_validation',index=index,config=cfg,seeds=[v['seed'] for v in values],
                auc=float(np.mean([v['validation_auc'] for v in values])),
                hall_aupr=float(np.mean([v['validation_hall_aupr'] for v in values])),
                best_epochs=[v['best_epoch'] for v in values],head_ids=[v['head_id'] for v in values])


def prepare(model, device):
    data = study.load_training_data(model,'fp32_k4',True)
    original = json.loads((old.previous.baseline_root(model)/'summary.json').read_text())
    if old.previous.original_fingerprint(data)!=original['fingerprint']:
        raise ValueError('Original features/mentions/protocol changed')
    splits = json.loads((ROOT/'outputs'/model/study.EXPERIMENT/'image_splits.json').read_text())
    if (len(splits['train']),len(splits['test']))!=(3200,800):
        raise ValueError('Expected original image split')
    split = old.previous.inner_image_split(splits['train'],splits['test'])
    direct = {k:data[k] for k in ('train_mentions','test_mentions','y_train','y_test')}
    for part in ('train','test'):
        ae = data['X_'+part]['AE']
        raw_s = data['X_'+part]['raw_AE+S+N_vec'][:,ae.shape[1]:2*ae.shape[1]]
        direct['X_'+part] = old.direct_features(ae,raw_s)
    inner_old = old.inner_inputs(direct,split)
    inner = {k.replace('_test','_val'):v for k,v in inner_old.items()}
    if any(len(np.unique(inner['y_'+p]))!=2 for p in ('train','val')):
        raise ValueError('Both labels required; do not replace images')
    scaler = fit_scaler(inner['X_train'])
    baseline_path = old.output_root(model)/'summary.json'
    baseline = json.loads(baseline_path.read_text())['groups']['three_hidden_fixed_direct']
    protocol = dict(version=NAME,model=model,feature='concat(full AE, log1p(raw S))',
        feature_dim=direct['X_train'].shape[1], image_split=split,split_seed=old.previous.INNER_SEED,
        architectures=[list(a) for a in ARCHITECTURES],stage1=[config(a) for a in ARCHITECTURES],
        stage2='Top two stage1 architectures by seed43 validation AUROC, then HALL-AUPR, then listed order; 48 configs each (12 for linear); batch64/128/256 on each winner',
        stage2_grids={str(list(a)):grid(a) for a in ARCHITECTURES},seeds=list(SEEDS),
        selection='stage1/grid/batch screening seed43; selected per-architecture winners repeated44/45; best MLP chosen by three-seed mean validation AUROC, tie HALL-AUPR',
        controls='All six fixed stage1 architectures repeated44/45 for equal-protocol capacity control; BN-on and scaler-off separately at best tuned MLP config, not used to reselect primary',
        training=dict(optimizer='AdamW',loss='unweighted BCEWithLogitsLoss',max_epochs=MAX_EPOCHS,
            patience=PATIENCE,checkpoint='maximum validation AUROC; exact tie HALL-AUPR; then earliest epoch',
            scheduler=None,final_refit=False,final_training_images=2560,validation_images=640,test_images=800,
            normalization='StandardScaler fit on optimization/train mentions only, never on validation/test',
            training_threshold='REAL-F1 on optimization/train mentions only',inference_batch_size=256,
            dtype='FP32',AMP=False,TF32=False,singleton_batch='merge into previous batch; no dropped samples'),
        scaler={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in scaler.items()},
        matrix_sha256={k:hashlib.sha256(str(v.shape).encode()+v.tobytes()).hexdigest()
                       for k,v in {**inner,'X_test':direct['X_test'],'y_test':direct['y_test']}.items()},
        original_fingerprint=original['fingerprint'],numerical_exception=data['numerical_exception'],
        old_baseline_sha256=sha256_file(baseline_path),
        source_sha256={str(p.relative_to(ROOT)):sha256_file(p) for p in
                      (Path(__file__),Path(old.__file__),Path(old.previous.__file__),Path(study.__file__),
                       ROOT/'scripts/train_torch_probe_feature_sets.py')},
        packages={p:version(p) for p in ('torch','numpy','scikit-learn')},device=device,
        no_bootstrap=True,independent_2000_run=False,
        limitation='Previously explored 800-image test set; not independent confirmation. Old reference used 3200 optimization images and a different training protocol.')
    protocol = json.loads(json.dumps(protocol))
    study.immutable_json(protocol,output_root(model)/'protocol.json')
    return inner,dict(X_test=direct['X_test'],y_test=direct['y_test']),scaler,baseline,protocol


def run_head(cfg, seed, inner, scaler, root, signature, device, epochs=MAX_EPOCHS, patience=PATIENCE):
    """Train/validate only: this function never receives the outer test matrix."""
    head_id = key(dict(config=cfg,seed=seed))[:20]
    path = Path(root)/'heads'/head_id
    identity = key(dict(signature=signature,config=cfg,seed=seed,device=device,epochs=epochs,patience=patience))
    result_path = path/'result.pt'
    x = {p:transform(inner['X_'+p],scaler,cfg['standardize']) for p in ('train','val')}
    if result_path.exists():
        result = torch.load(result_path,map_location='cpu',weights_only=False)
        if result['fingerprint']!=identity:
            raise ValueError('Wrong head resume fingerprint')
        for name,digest in result['artifacts'].items():
            if sha256_file(name)!=digest:
                raise ValueError('Changed completed head artifact')
    else:
        _set_seed(seed)
        attempt = path/f'attempt_{time.time_ns()}'; attempt.mkdir(parents=True,exist_ok=False)
        started = time.perf_counter()
        if str(device).startswith('cuda'):
            torch.empty(0,device=device); torch.cuda.reset_peak_memory_stats(device)
        model = make_probe(x['train'].shape[1],cfg).to(device)
        optimizer = torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        dataset = MatrixDataset(x['train'],inner['y_train'])
        criterion = nn.BCEWithLogitsLoss()
        best, stale, history, state, best_epoch = (-np.inf,-np.inf),0,[],None,0
        for epoch in range(1,epochs+1):
            loss = _train_epoch(model,epoch_loader(dataset,cfg['batch_size']),optimizer,criterion,torch.device(device))
            val_p = probabilities(model,x['val'],device)
            score = validation_score(inner['y_val'],val_p)
            improved = score>best
            if improved:
                best,stale,best_epoch = score,0,epoch
                state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else:
                stale += 1
            history.append(dict(epoch=epoch,train_loss=loss,validation_auc=score[0],validation_hall_aupr=score[1],is_best=improved))
            if stale>=patience:
                break
        if state is None:
            raise ValueError('No finite validation-selected checkpoint')
        model.load_state_dict(state)
        saved = {p:probabilities(model,x[p],device) for p in ('train','val')}
        if validation_score(inner['y_val'],saved['val'])!=best:
            raise ValueError('Best-epoch probabilities do not reproduce validation score')
        checkpoint = attempt/'model.pt'
        atomic_torch_save(dict(state_dict=state,config=cfg,input_dim=x['train'].shape[1],scaler=scaler,
                              fingerprint=identity,seed=seed),checkpoint)
        atomic_json_save(history,attempt/'history.json')
        result = dict(head_id=head_id,fingerprint=identity,seed=seed,config=cfg,checkpoint=str(checkpoint),
            train_probabilities=saved['train'],val_probabilities=saved['val'],validation_auc=best[0],
            validation_hall_aupr=best[1],best_epoch=best_epoch,epochs_run=len(history),
            parameter_count=sum(v.numel() for v in model.parameters()),elapsed_seconds=time.perf_counter()-started,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if str(device).startswith('cuda') else 0,
            artifacts={str(p):sha256_file(p) for p in attempt.iterdir() if p.is_file()})
    checkpoint = torch.load(result['checkpoint'],map_location='cpu',weights_only=False)
    if checkpoint['fingerprint']!=identity or checkpoint['config']!=cfg:
        raise ValueError('Wrong checkpoint identity')
    model = make_probe(checkpoint['input_dim'],cfg).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    errors = {}
    for part in ('train','val'):
        matrix = transform(inner['X_'+part],checkpoint['scaler'],cfg['standardize'])
        pred = probabilities(model,matrix,device)
        errors[part] = float(np.max(np.abs(pred-result[part+'_probabilities'])))
        if not np.isfinite(pred).all() or errors[part]>1e-7:
            raise ValueError('Same-device checkpoint recomputation failed')
    if (validation_score(inner['y_val'],result['val_probabilities']) !=
            (result['validation_auc'],result['validation_hall_aupr']) or
            result['config']!=cfg or result['seed']!=seed or result['head_id']!=head_id):
        raise ValueError('Saved validation metrics or head metadata changed')
    if not result_path.exists():
        result['recomputation_max_error'] = errors
        atomic_torch_save(result,result_path)
        compact = {k:v for k,v in result.items() if not k.endswith('_probabilities')}
        atomic_json_save(compact,path/'result.json')
    return result


def evaluate(result, inner, outer, root, device):
    selection_path = Path(root)/'selection.json'
    selection = json.loads(selection_path.read_text())
    if selection['status']!='FROZEN_BEFORE_OUTER_EVALUATION' or result['head_id'] not in selection['final_head_ids']:
        raise ValueError('Outer evaluation requires frozen head selection')
    identity = key(dict(selection=sha256_file(selection_path),head=result['fingerprint'],device=device))
    destination = Path(root)/'evaluation'/result['head_id']/'result.pt'
    saved = torch.load(destination,map_location='cpu',weights_only=False) if destination.exists() else None
    if saved and saved['fingerprint']!=identity:
        raise ValueError('Wrong evaluation resume fingerprint')
    ckpt = torch.load(result['checkpoint'],map_location='cpu',weights_only=False)
    model = make_probe(ckpt['input_dim'],ckpt['config']).to(device)
    model.load_state_dict(ckpt['state_dict'])
    pred = probabilities(model,transform(outer['X_test'],ckpt['scaler'],ckpt['config']['standardize']),device)
    metrics = old.previous.metric_reports(inner['y_train'],result['train_probabilities'],outer['y_test'],pred)
    if saved:
        if (float(np.max(np.abs(saved['test_probabilities']-pred)))>1e-7 or saved['metrics']!=metrics or
                not np.array_equal(saved['train_probabilities'],result['train_probabilities'])):
            raise ValueError('Changed test predictions or metrics')
    else:
        saved = dict(fingerprint=identity,seed=result['seed'],config=result['config'],head_id=result['head_id'],
                     train_probabilities=result['train_probabilities'],test_probabilities=pred,metrics=metrics)
        # A separate model instance checks the newly saved checkpoint on the outer matrix too.
        replica = make_probe(ckpt['input_dim'],ckpt['config']).to(device)
        replica.load_state_dict(torch.load(result['checkpoint'],map_location='cpu',weights_only=False)['state_dict'])
        error = float(np.max(np.abs(probabilities(replica,transform(outer['X_test'],ckpt['scaler'],ckpt['config']['standardize']),device)-pred)))
        if error>1e-7 or not np.isfinite(pred).all():
            raise ValueError('Outer checkpoint verification failed')
        saved['recomputation_max_error'] = error
        atomic_torch_save(saved,destination)
    return saved


def run_model(model, device):
    root = output_root(model); root.mkdir(parents=True,exist_ok=True)
    with (root/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        started = time.perf_counter()
        inner,outer,scaler,baseline,protocol = prepare(model,device)
        signature = key(protocol)
        def fit(cfg, seed=43):
            value = run_head(cfg,seed,inner,scaler,root,signature,device)
            print('HEAD',model,value['head_id'],seed,cfg['hidden_sizes'],value['best_epoch'],value['validation_auc'],flush=True)
            return value
        first = [row(i,config(a),[fit(config(a))]) for i,a in enumerate(ARCHITECTURES)]
        top = rank(first)[:2]
        study.immutable_json(dict(stage1=first,top_two=top),root/'stage1.json')
        tuned = []
        for chosen in top:
            rows = [row(i,c,[fit(c)]) for i,c in enumerate(grid(chosen['config']['hidden_sizes']))]
            winner = rank(rows)[0]
            batch_rows = [row(i,{**winner['config'],'batch_size':b},[fit({**winner['config'],'batch_size':b})])
                          for i,b in enumerate((64,128,256))]
            best = rank(batch_rows)[0]
            values = [fit(best['config'],seed) for seed in SEEDS]
            tuned.append(row(chosen['index'],best['config'],values))
            study.immutable_json(dict(grid=rows,batch_sweep=batch_rows,selected=tuned[-1]),root/f"stage2_arch{chosen['index']}.json")
        # Equal-parameter capacity controls, including linear, irrespective of shortlist.
        group_configs = {f'fixed_arch{i}':config(a) for i,a in enumerate(ARCHITECTURES)}
        group_configs.update({f"tuned_arch{r['index']}":r['config'] for r in tuned})
        mlp = rank([r for r in tuned if r['config']['hidden_sizes']])[0]
        group_configs['matched_BN_on'] = {**mlp['config'],'batch_norm':True}
        group_configs['matched_scaler_off'] = {**mlp['config'],'standardize':False}
        final_heads = {name:[fit(cfg,seed) for seed in SEEDS] for name,cfg in group_configs.items()}
        selection = dict(status='FROZEN_BEFORE_OUTER_EVALUATION',protocol_sha256=sha256_file(root/'protocol.json'),
            tuned=tuned,best_tuned=rank(tuned)[0],best_tuned_mlp=mlp,group_configs=group_configs,
            final_head_ids=sorted({v['head_id'] for values in final_heads.values() for v in values}))
        study.immutable_json(selection,root/'selection.json')
        groups = {}
        for name,values in final_heads.items():
            evaluated = [evaluate(v,inner,outer,root,device) for v in values]
            groups[name] = old.summarize_group(dict(y_train=inner['y_train'],y_test=outer['y_test']),evaluated)
            groups[name]['config'] = group_configs[name]
            print('FINAL',model,name,groups[name]['ensemble_reports']['fixed_0.5']['auc'],flush=True)
        groups['old_three_hidden_direct_reference'] = baseline
        all_heads = [torch.load(p,map_location='cpu',weights_only=False) for p in sorted((root/'heads').glob('*/result.pt'))]
        artifacts = {str(p):sha256_file(p) for part in ('heads','evaluation') for p in (root/part).rglob('*') if p.is_file()}
        study.immutable_json(dict(status='PASS_SAME_DEVICE_CHECKPOINTS_AND_METRICS',artifacts=artifacts,
            trained_unique_heads=len(all_heads),evaluated_unique_heads=len(selection['final_head_ids']),
            protocol_sha256=sha256_file(root/'protocol.json'),selection_sha256=sha256_file(root/'selection.json')),
            root/'verification.json')
        summary = dict(status='COMPLETE_EXPLORATORY_SHALLOW_SEARCH',model=model,feature_dim=protocol['feature_dim'],
            groups=groups,selection=selection,train_mentions=len(inner['y_train']),validation_mentions=len(inner['y_val']),
            test_mentions=len(outer['y_test']),trained_unique_heads=len(all_heads),
            fit_seconds=sum(v['elapsed_seconds'] for v in all_heads),peak_cuda_allocated_bytes=max(v['peak_cuda_allocated_bytes'] for v in all_heads),
            numerical_exception=protocol['numerical_exception'],verification_sha256=sha256_file(root/'verification.json'))
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
            if s['verification_sha256']!=sha256_file(output_root(m)/'verification.json'):
                raise ValueError('Changed verification')
        study.immutable_json(dict(status='COMPLETE_FOUR_MODEL_SHALLOW_SEARCH',models=summaries,
            sources={m:dict(path=str(p),sha256=sha256_file(p)) for m,p in paths.items()}),root/'summary.json')
        lines = ['# AE + log1p(S)：StandardScaler 与浅层 probe 两阶段搜索','',
            '原4000图按图片分为2560训练/640验证/800测试。只在2560图拟合StandardScaler。',
            'AdamW、最多150epochs、patience15，validation AUROC最佳checkpoint；平局HALL-AUPR，再取最早epoch。',
            '六结构固定参数对照，验证最好的两个结构做48组网格（Linear为12个不重复配置），再比较batch64/128/256。',
            '全部最终组seeds43/44/45；冻结后才评价800图；不合并训练与验证集重训。',
            'BN-on与scaler-off是同参数单因素控制，不用于重新选择主分类器。',
            '旧800图已经用于研究探索，不是独立确认；保留数值FAIL例外，不做bootstrap。',
            '旧三隐藏层reference使用3200图训练、BatchNorm、Adam及训练损失checkpoint，不能把差异仅归因于深度或标准化。','',
            '## 结构对照：ensemble AUROC / HALL-AUPR（%）','',
            '| 组 | '+' | '.join(study.MODELS)+' |','|---|'+'---:|'*4]
        fixed = [f'fixed_arch{i}' for i in range(6)]+['best_tuned','best_tuned_mlp','matched_BN_on','matched_scaler_off','old_three_hidden_direct_reference']
        for name in fixed:
            cells=[]
            for s in summaries.values():
                group = f"tuned_arch{s['selection'][name]['index']}" if name.startswith('best_tuned') else name
                r = s['groups'][group]['ensemble_reports']['fixed_0.5']
                cells.append(f"{r['auc']*100:.3f} / {r['hallucination_positive']['aupr']*100:.3f}")
            lines.append('| '+name+' | '+' | '.join(cells)+' |')
        lines += ['', '## 验证集选定参数','']
        for m,s in summaries.items():
            lines.append(f"- {m}，{s['feature_dim']}维：`{json.dumps(s['selection']['best_tuned']['config'],sort_keys=True)}`。")
        lines += ['', '完整逐seed、mean±std、REAL/HALL AUPR、P/R/F1和阈值见summary.json；各模型stage1/stage2文件公开所有验证候选。']
        text = '\n'.join(lines)+'\n'; path = root/'summary.md'
        if path.exists() and path.read_text()!=text:
            raise ValueError('Completed report changed')
        if not path.exists():
            path.write_text(text)
        print('PUBLISHED',path,flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',nargs='+',choices=study.MODELS)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--summarize-only',action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
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
            atomic_json_save(dict(model=model,command=sys.argv,traceback=traceback.format_exc()),
                             output_root(model)/f'failure_{time.time_ns()}.json')
            raise
    publish()


if __name__=='__main__':
    main()
