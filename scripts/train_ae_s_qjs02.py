"""Concatenate AE, log1p(S), and JS(softmax(raw signed Q / .2), T)."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from features.tc_fvpa_artifacts import atomic_json_save, atomic_torch_save, sha256_file
from scripts.train_ffn_ae_log1p_search import direct_features, run_head, summarize_group
from scripts.run_minigpt4_shikra_path import save_json
from scripts.train_q_softmax_js import q_softmax_js

MODELS = ('minigpt4_7b', 'shikra_7b')
OUT = ROOT/'outputs/ae_s_qjs02'
COSINE_OUT = ROOT/'outputs/ae_s_cosinejs02'


def features(ae, strength, q, target, gross=None):
    q = torch.as_tensor(q, dtype=torch.float64)
    if q.ndim != 2 or not q.numel() or not torch.isfinite(q).all():
        raise ValueError('Expected finite nonempty [layer,visual_token] signed Q')
    base = direct_features(np.asarray(ae)[None], np.asarray(strength)[None])[0]
    if base.size != 2*q.shape[0]:
        raise ValueError('AE/S/Q layers do not align')
    if gross is None:
        js, _ = q_softmax_js(q/.2, target)
    else:
        from scripts.train_endpoint_temperatures4000 import cosine_temperatures
        values, _, _ = cosine_temperatures(dict(path_signed_q=q, ffn_path_gross=gross, attention_evidence=target))
        js = values['0.2']
    value = np.concatenate([base, js])
    if not np.isfinite(value).all(): raise ValueError('Nonfinite concatenated feature')
    return base, value


def run(model, device, cosine=False):
    root = ROOT/'outputs'/model/'COCO4000-JACOBIAN-PATH'
    dest = (COSINE_OUT if cosine else OUT)/model
    dest.mkdir(parents=True, exist_ok=True)
    with (dest/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        original = json.loads((root/'path/training/protocol.json').read_text())
        split = json.loads((root/'image_splits.json').read_text())
        mentions = original['mentions']
        train_ids, test_ids = set(split['train']), set(split['test'])
        ntrain = sum(m['image_id'] in train_ids for m in mentions)
        assert len(train_ids)==3200 and len(test_ids)==800 and not train_ids & test_ids
        assert all(m['image_id'] in train_ids for m in mentions[:ntrain])
        assert all(m['image_id'] in test_ids for m in mentions[ntrain:])
        if json.loads((root/'path/full_numerical_audit.json').read_text())['status']!='PASS':
            raise ValueError('Original full-cohort numerical audit has not passed')
        positions, ids = {}, set()
        for offset, (filename, digest) in enumerate(original['sources'].items(), 1):
            path = Path(filename)
            if not path.is_absolute(): path = ROOT/path
            if sha256_file(path)!=digest: raise ValueError(f'Changed source: {path}')
            shard = torch.load(path, map_location='cpu', weights_only=False)
            if shard['image_id'] in ids or not shard['processed_image']: raise ValueError('Invalid image shard')
            ids.add(shard['image_id'])
            for row in shard['positions']:
                key = row['target_key']
                if key in positions: raise ValueError('Duplicate target')
                positions[key] = features(row['AE'], row['S'], row['path_signed_q'], row['attention_evidence'],
                                          row['ffn_path_gross'] if cosine else None)
            if offset % 500 == 0: print(model, 'loaded', offset, flush=True)
        if ids != train_ids | test_ids or set(positions)!={m['target_key'] for m in mentions}:
            raise ValueError('Cohort/target mismatch')
        baseline = np.stack([positions[m['target_key']][0] for m in mentions])
        matrix = np.stack([positions[m['target_key']][1] for m in mentions])
        del positions
        if hashlib.sha256(baseline.tobytes()).hexdigest()!=original['groups']['F']:
            raise ValueError('Rebuilt F differs from the original baseline')
        y = np.asarray([m['label'] for m in mentions], dtype=np.int32)
        definition=('concat(AE, log1p(raw S), JS(softmax(cos(e_m,G(Z)-G(Z0))/0.2),T)), all layers, full visual support, natural log'
                    if cosine else 'concat(all-layer AE, log1p(raw S), all-layer JS(softmax(raw signed Q / 0.2, visual axis), saved T), natural log)')
        protocol = dict(model=model, definition=definition,
            input_dim=matrix.shape[1], images=4000, train_images=3200, test_images=800,
            train_mentions=ntrain, test_mentions=len(y)-ntrain, seeds=[43,44,45], mlp=original['mlp'],
            source_protocol_sha256=sha256_file(root/'path/training/protocol.json'),
            implementation_sha256=sha256_file(Path(__file__)),
            matrix_sha256=hashlib.sha256(matrix.tobytes()).hexdigest(), baseline_sha256=original['groups']['F'])
        save_json(protocol, dest/'protocol.json')
        if not (dest/'matrices.pt').exists():
            atomic_torch_save(dict(X=matrix,y=y,ntrain=ntrain),dest/'matrices.pt')
        data=dict(X_train=matrix[:ntrain],X_test=matrix[ntrain:],y_train=y[:ntrain],y_test=y[ntrain:])
        heads=[run_head('three_hidden',original['mlp'],seed,data,dest/f'seed{seed}',
                        sha256_file(dest/'protocol.json'),device) for seed in (43,44,45)]
        result=summarize_group(data,heads)
        base_result=json.loads((root/'path/training/results.json').read_text())['F']
        key='F_CosineJS02' if cosine else 'F_QJS02'
        atomic_json_save(dict(status='COMPLETE',protocol=protocol,F=base_result,**{key:result}),dest/'results.json')
        summarize(cosine)


def summarize(cosine=False):
    output=COSINE_OUT if cosine else OUT
    key='F_CosineJS02' if cosine else 'F_QJS02'
    results={m:json.loads((output/m/'results.json').read_text()) for m in MODELS if (output/m/'results.json').exists()}
    lines=['# AE + log1p(S) + JS(softmax(Q/0.2), T)', '',
           '每层计算softmax(raw signed Q/0.2)与同源T的JS散度（自然对数），再与AE/log1p(S)拼接。固定COCO4000，分别计算seeds 43/44/45的指标，再报告均值±总体标准差（ddof=0），不使用ensemble。F1阈值由各seed训练集REAL-F1选择。', '',
           '| Model | Input | AUROC mean ± std | HALL AUPR mean ± std | HALL F1 mean ± std (train-REAL-F1) |',
           '|---|---|---:|---:|---:|']
    for model,r in results.items():
        for name in ('F',key):
            value=r[name]
            reports=[value['per_seed_metrics'][str(seed)]['threshold_reports']['train_f1']['test_metrics'] for seed in (43,44,45)]
            columns=([v['auc'] for v in reports], [v['hallucination_positive']['aupr'] for v in reports],
                     [v['hallucination_positive']['f1'] for v in reports])
            cells=[f'{np.mean(v):.6f} ± {np.std(v,ddof=0):.6f}' for v in columns]
            lines.append('| '+' | '.join([model,name,*cells])+' |')
    if cosine:
        lines[0]='# AE + log1p(S) + JS(softmax(cos(e_m,G(Z)-G(Z0))/0.2), T)'
        lines[2]=lines[2].replace('softmax(raw signed Q/0.2)','softmax(cos(e_m,G(Z)-G(Z0))/0.2)')
    output.mkdir(exist_ok=True,parents=True)
    (output/'summary.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=MODELS)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--summarize',action='store_true')
    parser.add_argument('--cosine',action='store_true',help='Use endpoint cosine instead of raw signed Q')
    args=parser.parse_args()
    torch.set_num_threads(1)
    if args.summarize: summarize(args.cosine)
    elif args.model: run(args.model,args.device,args.cosine)
    else: parser.error('--model or --summarize is required')
