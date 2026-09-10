"""JS(endpoint-cosine softmax/.2, normalized raw visual attention)."""
import argparse
import csv
import fcntl
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.train_raw_attention_strength import MODELS,FOUR_MODELS,plot_curves
from scripts.train_endpoint_temperatures4000 import cosine_temperatures
from scripts.train_ffn_ae_log1p_search import direct_features,fixed_mlp,run_head,summarize_group
from scripts.run_minigpt4_shikra_path import save_json
from features.tc_fvpa_artifacts import atomic_json_save,atomic_torch_save,sha256_file

OUT=ROOT/'outputs/cosine_raw_attention_js02'


def raw_js(row,attention):
    a=torch.as_tensor(attention,dtype=torch.float64)
    q=torch.as_tensor(row['path_signed_q'])
    if a.ndim!=2 or a.shape!=q.shape or not torch.isfinite(a).all() or (a<0).any():
        raise ValueError('Invalid raw attention or mismatched cosine support')
    mass=a.sum(-1,keepdim=True)
    if (mass<=0).any() or (mass>1+1e-5).any():raise ValueError('Raw visual mass must be in (0,1]')
    target=a/mass
    values,_,audit=cosine_temperatures(dict(row,attention_evidence=target))
    return values['0.2'],audit


def run(model,device):
    dest=OUT/model;dest.mkdir(exist_ok=True,parents=True)
    with (dest/'.lock').open('a') as lock:
        four=model in FOUR_MODELS
        if four:
            source=ROOT/'outputs/ffn_source_composition_v1'/model
            cache=torch.load(source/'matrices.pt',map_location='cpu',weights_only=False)
            mentions=cache['mentions'];base=np.asarray(cache['groups']['F']);y=np.asarray(cache['y']);n=cache['ntrain']
            from scripts.run_ffn_source_composition import parent_paths
            paths=parent_paths(model)
            raw_root=source/'shards/k50'
            baseline=json.loads((source/'detection.json').read_text())['F']
            split_path=ROOT/'outputs'/model/'COCO4000-INSLEN-OFFICIAL-TARGET/image_splits.json'
            mlp=fixed_mlp()
        else:
            source=ROOT/'outputs'/model/'COCO4000-JACOBIAN-PATH'
            protocol=json.loads((source/'path/training/protocol.json').read_text())
            mentions=protocol['mentions'];y=np.asarray([m['label'] for m in mentions],dtype=np.int32)
            split_path=source/'image_splits.json';split=json.loads(split_path.read_text());n=sum(m['image_id'] in split['train'] for m in mentions)
            base=None;paths={int(Path(p).stem.split('_')[-1]):Path(p) for p in protocol['sources']}
            baseline=json.loads((source/'path/training/results.json').read_text())['F'];mlp=protocol['mlp']
        split=json.loads(split_path.read_text());train_ids,test_ids=set(split['train']),set(split['test'])
        assert len(train_ids)==3200 and len(test_ids)==800 and not train_ids&test_ids
        assert set(paths)==train_ids|test_ids
        assert all(m['image_id'] in train_ids for m in mentions[:n]) and all(m['image_id'] in test_ids for m in mentions[n:])
        np.testing.assert_array_equal(y,[m['label'] for m in mentions])
        positions={};clipped=zero_norm=0
        for i,(image,path) in enumerate(sorted(paths.items()),1):
            if not path.is_absolute():path=ROOT/path
            shard=torch.load(path,map_location='cpu',weights_only=False)
            rows=shard['positions']
            if four:
                raw_shard=torch.load(raw_root/path.name,map_location='cpu',weights_only=False)
                assert raw_shard['image_id']==image
                raw={r['target_key']:r for r in raw_shard['positions']}
                if set(raw)!={r['target_key'] for r in rows}:raise ValueError('Parent/composition target mismatch')
            for row in rows:
                key=row['target_key']
                if key in positions:raise ValueError('Duplicate target')
                if four:
                    current=raw[key]
                    assert int(current['target_token_id'])==int(row['target_token_id'])
                    np.testing.assert_array_equal(current['AE'],row['AE']);np.testing.assert_array_equal(current['S'],row['S'])
                    attention=current['raw_attention_mean']
                else:attention=row['raw_attention_mean']
                js,audit=raw_js(row,attention)
                f=direct_features(np.asarray(row['AE'])[None],np.asarray(row['S'])[None])[0]
                positions[key]=(js,f)
                clipped+=audit['clipped_cosine_entries'];zero_norm+=audit['zero_gross_entries']
            if i%500==0:print(model,'loaded',i,flush=True)
        if set(positions)!={m['target_key'] for m in mentions}:raise ValueError('Mention/target mismatch')
        j=np.stack([positions[m['target_key']][0] for m in mentions])
        rebuilt=np.stack([positions[m['target_key']][1] for m in mentions])
        if four:np.testing.assert_array_equal(rebuilt,base)
        else:
            assert hashlib.sha256(rebuilt.tobytes()).hexdigest()==protocol['groups']['F']
        base=rebuilt;groups=dict(J_raw=j,F_J_raw=np.concatenate([base,j],axis=1))
        plot_curves(model,j,y,n,dest,ylabel='JS(cosine softmax / 0.2, normalized raw attention)')
        identity=dict(model=model,formula='JS(softmax(cos(e_m,G(Z)-G(Z0))/.2), raw_visual_attention / sum_visual(raw_visual_attention)), natural log; F=AE+log1p(S_E)',
            images=4000,train_images=3200,test_images=800,train_mentions=n,test_mentions=len(y)-n,
            source=str(source),split_sha=sha256_file(split_path),seeds=[43,44,45],mlp=mlp,
            cosine_clipped_entries=clipped,zero_component_norm_entries=zero_norm,
            zero_visual_mass_policy='error; never substitute uniform',
            precision_note='Q/norms from saved path extraction; four-model raw attention from composition capture with exact target/AE/S alignment; existing numerical limitations retained',
            matrix_sha={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in groups.items()},code_sha=sha256_file(Path(__file__)))
        save_json(identity,dest/'protocol.json')
        if not (dest/'matrices.pt').exists():atomic_torch_save(dict(groups=groups,y=y,ntrain=n),dest/'matrices.pt')
        results={'F':baseline}
        for name,x in groups.items():
            data=dict(X_train=x[:n],X_test=x[n:],y_train=y[:n],y_test=y[n:])
            heads=[run_head('three_hidden',mlp,seed,data,dest/name/f'seed{seed}',sha256_file(dest/'protocol.json'),device) for seed in (43,44,45)]
            results[name]=summarize_group(data,heads);print(model,'trained',name,flush=True)
        atomic_json_save(dict(status='COMPLETE',protocol=identity,groups=results),dest/'results.json')
        summarize()


def summarize():
    lines=['# Cosine softmax (tau=.2) vs normalized raw attention: JS', '',
           'Three seeds 43/44/45: mean ± population std (ddof=0), no ensemble. HALL F1 uses train-REAL-F1 thresholds.', '',
           '| Model | Input | AUROC | HALL AUPR | HALL F1 |','|---|---|---:|---:|---:|']
    for model in MODELS:
        p=OUT/model/'results.json'
        if not p.exists():continue
        for name,v in json.loads(p.read_text())['groups'].items():
            reports=[v['per_seed_metrics'][str(s)]['threshold_reports']['train_f1']['test_metrics'] for s in (43,44,45)]
            cols=([r['auc'] for r in reports],[r['hallucination_positive']['aupr'] for r in reports],[r['hallucination_positive']['f1'] for r in reports])
            lines.append('| '+' | '.join([model,name,*[f'{np.mean(x):.6f} ± {np.std(x):.6f}' for x in cols]])+' |')
    OUT.mkdir(exist_ok=True,parents=True);(OUT/'summary.md').write_text('\n'.join(lines)+'\n')
    if all((OUT/m/'curves.csv').exists() for m in MODELS):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(3,2,figsize=(12,11),sharey=True)
        for m,ax in zip(MODELS,axes.flat):
            with (OUT/m/'curves.csv').open() as f:rows=list(csv.DictReader(f))
            for label,name,color in (('0','HALL','#d55e00'),('1','REAL','#0072b2')):
                selected=sorted((r for r in rows if r['split']=='test' and r['label']==label),key=lambda r:int(r['layer']))
                x=[int(r['layer']) for r in selected];ax.plot(x,[float(r['mean']) for r in selected],color=color,label=name)
                ax.fill_between(x,[float(r['q25']) for r in selected],[float(r['q75']) for r in selected],color=color,alpha=.12)
            ax.set_title(m);ax.set_xlim(1,max(x));ax.set_xlabel('Layer (1-based)');ax.set_ylabel('JS (natural log)');ax.legend();ax.grid(alpha=.15)
        fig.suptitle('Test mentions: mean and IQR (not confidence interval)');fig.tight_layout()
        fig.savefig(OUT/'test_curves.png',dpi=170);fig.savefig(OUT/'test_curves.pdf');plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--model',choices=MODELS);parser.add_argument('--device',default='cuda:0');parser.add_argument('--summarize',action='store_true');args=parser.parse_args()
    torch.set_num_threads(1)
    if args.summarize:summarize()
    elif args.model:run(args.model,args.device)
    else:parser.error('--model or --summarize required')
