"""Test scalar-identity behavior on the existing 500-image visual WRITE cache."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import analyze_ffn_input_geometry as base
from scripts.run_ffn_source_composition import parent_paths

OUT=ROOT/'outputs/ffn_scalar_identity_20260912'
TITLES=dict(cos_mean='Mean cosine(a_m, e_m)',parallel_gain_mean='Mean signed parallel gain',
           parallel_gain_median='Median signed parallel gain',parallel_gain_std='Within-target source gain std',
           orthogonal_ratio_mean='Mean ||e_m - g_m a_m|| / ||e_m||',
           orthogonal_ratio_median='Median orthogonal ratio',common_c='Best shared scalar c (WRITE weighted)',
           shared_scalar_error='||E-cA||_F / ||E||_F',
           individual_scalar_error='Per-source scalar-fit residual / ||E||_F',
           gain_nonuniform_error='Gain nonuniformity contribution / ||E||_F')


def token_metrics(a,e,cos):
    a,e,cos=[np.asarray(x,dtype=np.float64) for x in (a,e,cos)]
    if a.ndim!=2 or a.shape!=e.shape or a.shape!=cos.shape: raise ValueError('Expected matching [L,M] arrays')
    if not np.isfinite(a).all() or not np.isfinite(e).all() or (a<0).any() or (e<0).any():
        raise ValueError('Invalid norms')
    valid=(a>0)&(e>0)
    if not np.isfinite(cos[valid]).all() or (np.abs(cos[valid])>1+1e-6).any(): raise ValueError('Invalid cosine')
    if ((a==0)&(e>0)).any(): raise ValueError('A linear map cannot map a zero write to a nonzero effect')
    cossafe=np.where(valid,np.clip(cos,-1,1),0.)
    g=base.divide(e*cossafe,a)
    orth=np.where(valid,np.sqrt(np.maximum(0,1-cossafe**2)),np.nan)
    dot=a*e*cossafe
    a2=(a*a).sum(-1);e2=(e*e).sum(-1)
    common=base.divide(dot.sum(-1),a2)
    individual_sq=(e*e*(1-cossafe**2)).sum(-1)
    varying_sq=np.nansum(a*a*(g-common[:,None])**2,axis=-1)
    shared_sq=e2-base.divide(dot.sum(-1)**2,a2)
    if np.any(shared_sq < -1e-10*np.maximum(e2,1)): raise ValueError('Negative scalar residual')
    np.testing.assert_allclose(shared_sq,individual_sq+varying_sq,rtol=1e-8,atol=1e-10)
    def stat(x,fn):
        return np.array([fn(v[np.isfinite(v)]) if np.isfinite(v).any() else np.nan for v in x])
    values=dict(cos_mean=stat(np.where(valid,cos,np.nan),np.mean),parallel_gain_mean=stat(g,np.mean),
        parallel_gain_median=stat(g,np.median),parallel_gain_std=stat(g,np.std),
        orthogonal_ratio_mean=stat(orth,np.mean),orthogonal_ratio_median=stat(orth,np.median),
        common_c=common,shared_scalar_error=np.sqrt(np.maximum(0,base.divide(shared_sq,e2))),
        individual_scalar_error=np.sqrt(np.maximum(0,base.divide(individual_sq,e2))),
        gain_nonuniform_error=np.sqrt(np.maximum(0,base.divide(varying_sq,e2))))
    return values


def main():
    torch.set_num_threads(1)
    base.TITLES.update(TITLES)
    allrows,allpaired=[],[]
    for model in base.MODELS:
        parents=parent_paths(model)
        old=base.OUT/'cohort500'/model
        mentions=json.loads((old/'mentions.json').read_text())
        targets={};nsource=0
        for path in sorted((old/'extraction').glob('image*.pt')):
            shard=torch.load(path,map_location='cpu',weights_only=False)
            parent=torch.load(parents[shard['image_id']],map_location='cpu',weights_only=False)
            lookup={p['target_key']:p for p in parent['positions']}
            assert shard['sample_table']==parent['sample_table']
            for row in shard['positions']:
                key=row['target_key'];p=lookup[key]
                assert key not in targets and row['target_token_id']==p['target_token_id']
                targets[key]=token_metrics(p['write_mag'],p['ffn_path_gross'],row['cosines'])
                nsource+=row['cosines'].numel()
        values={k:np.stack([targets[r['target_key']][k] for r in mentions]) for k in TITLES}
        directory=OUT/'tokens500'/model;directory.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(directory/'metrics.npz',**values)
        (directory/'mentions.json').write_text(json.dumps(mentions))
        (directory/'audit.json').write_text(json.dumps(dict(targets=len(targets),mentions=len(mentions),source_entries=nsource)))
        rows,paired=base.summarize(model,values,mentions,directory)
        allrows+=rows;allpaired+=paired
        print(model,'DONE',len(mentions),flush=True)
    for name,rows in [('curves',allrows),('paired_images',allpaired),('summary',base.contrasts(allrows,allpaired))]:
        base.write_csv(OUT/'tokens500'/f'{name}.csv',rows)
    for scope in ('all','test'):
        base.plot(allrows,scope,('cos_mean','parallel_gain_mean','orthogonal_ratio_mean'),OUT/'tokens500'/f'{scope}_metrics')
    base.plot(allrows,'all',('common_c','shared_scalar_error','parallel_gain_std'),OUT/'tokens500/all_scalar_fit')


if __name__=='__main__':main()
