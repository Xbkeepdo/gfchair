"""Full numerical visual-write subspace study, all targets/layers on COCO500."""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from features.ffn_scalar_subspace import visual_basis,scalar_subspace
from features.ffn_visual_path_attribution import quadrature_rule
from features.tc_fvpa_artifacts import atomic_torch_save
from features.visual_ffn_jacobian import reconstruct_visual_directions,resolve_decoder_layer_adapter
from models.dgst_capture import resolve_decoder_layers,attention_row_from_capture
from scripts.extract_ffn_input_rotation import factored_components
from scripts.run_ffn_source_composition import capture_full,load_wrapper,parent_paths
from scripts.run_ffn_visual_source_consistency import local_fp32
from scripts.run_jffn_p_comparison import _load_inputs,_image_path
from scripts.analyze_ffn_scalar_identity import OUT
from scripts import analyze_ffn_input_geometry as base

TITLES=dict(span_c='Best c on orthonormal visual-write basis',
    scalar_error_full='||Y-cQ||_F / ||Y||_F (including leakage)',
    scalar_error_projected='||B-cI||_F / ||B||_F',leakage_ratio='||Y-QB||_F / ||Y||_F',
    offdiag_ratio='Off-diagonal Frobenius fraction (basis dependent)',
    sigma_B_cv='CV of singular values of B',sigma_B_p90_p10='B singular values: P90/P10',
    sigma_Y_cv='CV of full-response singular values',sigma_Y_p90_p10='Y singular values: P90/P10',
    rank='Numerical rank of visual WRITE span')


def pending_images(ids,completed,parity=None):
    if parity not in (None,0,1):raise ValueError('Image parity must be 0, 1, or None')
    return [i for i in ids if i not in completed and (parity is None or i%2==parity)]


def extract(model,device,smoke=False,image_parity=None):
    cohort=json.loads((ROOT/'outputs/ffn_output_cosine_20260909/cohort500.json').read_text())
    ids=[283] if smoke else sorted(cohort['split']['train']+cohort['split']['test'])
    directory=OUT/'subspace500'/model/('smoke_fp64_spectrum' if smoke else 'extraction')
    directory.mkdir(parents=True,exist_ok=True)
    protocol=dict(model=model,ids=ids,k=4,precision='FP32 fixed-path JVP; FP64 basis and projections',
        basis='FP64 normalized-column Gram eigensolve then thin QR, all singular directions above rtol',
        basis_rtol=1e-5,spectrum='FP64 Gram spectra of B and Y; direct FP64 SVD(B) smoke control',
        path='original z0=z-sum a_m; replacing probes by Q does not alter the path')
    pp=directory/'protocol.json'
    if pp.exists() and json.loads(pp.read_text())!=protocol:raise ValueError('Protocol changed')
    pp.write_text(json.dumps(protocol,indent=2))
    completed={i for i in ids if (directory/f'image_{i:012d}.pt').exists()}
    pending=pending_images(ids,completed,None if smoke else image_parity)
    if not pending:print(model,'COMPLETE',flush=True);return
    if not smoke and not (directory.parent/'smoke_fp64_spectrum/image_000000000283.pt').exists():
        raise ValueError('Smoke required before full extraction')
    parents=parent_paths(model)
    _,generations,_=_load_inputs(ROOT/'outputs'/model/base.EXPERIMENT)
    wrapper,config=load_wrapper(model,device)
    for offset,image_id in enumerate(pending,1):
        tick=time.perf_counter()
        parent=torch.load(parents[image_id],map_location='cpu',weights_only=False)
        targets=sorted(parent['positions'],key=lambda x:x['response_index'])
        cosine_dir='smoke_factored' if smoke else 'extraction'
        oldcos=torch.load(base.OUT/'cohort500'/model/cosine_dir/f'image_{image_id:012d}.pt',map_location='cpu',weights_only=False)
        oldlookup={r['target_key']:r for r in oldcos['positions']}
        rows=[dict(target_key=t['target_key'],target_token_id=t['target_token_id'],layers=[]) for t in targets]
        if targets:
            with Image.open(_image_path(config,image_id)) as src:image=src.convert('RGB')
            caps,queries,start,end,_=capture_full(wrapper,model,image,generations[image_id]['response_token_ids'],
                targets,config.get('run',{}).get('prompt') or 'Describe this image.')
            for li,(layer,cap) in enumerate(zip(resolve_decoder_layers(wrapper.model),caps)):
                lt=time.perf_counter()
                for query in queries:
                    future=attention_row_from_capture(cap,query)[...,query+1:]
                    if future.numel() and float(future.abs().max())>1e-7:raise ValueError('Noncausal capture')
                a=reconstruct_visual_directions(layer=layer,capture=cap,prediction_positions=queries,
                    visual_start=start,visual_end=end)['a_tokens'].float()
                z=cap['h_mid'][0,queries].float()
                with local_fp32(layer) as fn:
                    adapter=resolve_decoder_layer_adapter(layer)
                    for ti,(row,old) in enumerate(zip(rows,targets)):
                        aa=a[:,ti].T
                        q,basis=visual_basis(aa)
                        with torch.no_grad():
                            y=factored_components(adapter.ffn_norm,adapter.ffn,z[ti:ti+1],a[:,ti:ti+1],
                                                  directions=q.T.float()[:,None,:])[:,0].T
                        stats,b,sb,sy=scalar_subspace(q,y,reference_svd=smoke)
                        reconstructed=y.double()@(q.T@aa.double())
                        norms=reconstructed.norm(dim=0)
                        an=aa.double().norm(dim=0)
                        write_error=float((an-old['write_mag'][li].to(an)).norm()/an.norm())
                        norm_error=float((norms-old['ffn_path_gross'][li].to(norms)).norm()/norms.norm())
                        cos=(aa.double()*reconstructed).sum(0)/(an*norms)
                        prev=oldlookup[row['target_key']]['cosines'][li].to(cos)
                        valid=torch.isfinite(prev)&torch.isfinite(cos)
                        cosine_error=float((cos[valid]-prev[valid]).abs().mean())
                        if max(write_error,norm_error,cosine_error)>5e-4:
                            raise ValueError(f'Parent reconstruction mismatch {model} {image_id} L{li+1}: {write_error} {norm_error} {cosine_error}')
                        stats.update(write_parent_relative=write_error,effect_parent_relative=norm_error,
                                     cosine_parent_mean_absolute=cosine_error)
                        if smoke:
                            # Independent direct JVP probes at the SAME original path points.
                            indices=sorted({0,q.shape[1]//2,q.shape[1]-1})
                            probe=q[:,indices].T.float()[:,None,:]
                            reference=torch.zeros_like(probe)
                            rule=quadrature_rule('gauss_legendre',4,device=z.device,dtype=z.dtype)
                            aggregate=a[:,ti:ti+1].sum(0)
                            for alpha,weight in zip(rule.nodes,rule.weights):
                                point=z[ti:ti+1]-aggregate+alpha*aggregate
                                for j,direction in enumerate(probe):
                                    with torch.enable_grad():
                                        value=torch.func.jvp(fn,(point,),(direction,))[1]
                                    reference[j]+=weight*value.detach()
                            error=float((reference[:,0].T-y[:,indices]).norm()/reference.norm().clamp_min(1e-12))
                            if error>5e-5:raise ValueError(f'Basis JVP mismatch {error}')
                            stats['direct_basis_jvp_relative']=error
                            # Gram basis is checked against direct FP64 SVD at fixed boundary/middle layers.
                            if li in (0,len(caps)//2,len(caps)-1):
                                unit=aa.double()[:,an>0]/an[an>0]
                                u,s,_=torch.linalg.svd(unit,full_matrices=False,driver='gesvdj')
                                rr=int((s>s[0]*1e-5).sum())
                                if rr!=q.shape[1]:raise ValueError('Gram/SVD rank mismatch')
                                basis_error=float((u[:,:rr]-q@(q.T@u[:,:rr])).norm()/rr**.5)
                                if basis_error>1e-4:raise ValueError(f'Gram/SVD span mismatch {basis_error}')
                                stats['direct_svd_span_relative']=basis_error
                        record=dict(layer=li+1,**stats,**basis,sigma_B=sb,sigma_Y=sy)
                        if smoke:record['B']=b.float().cpu()
                        row['layers'].append(record)
                        del q,y,b,reconstructed
                del a,z
                if smoke:print(model,'smoke layer',li+1,'seconds',round(time.perf_counter()-lt,2),flush=True)
            del caps
        atomic_torch_save(dict(image_id=image_id,positions=rows,sample_table=parent['sample_table'],device=device,image_parity=image_parity,
            complete=True,protocol=protocol,elapsed=time.perf_counter()-tick),directory/f'image_{image_id:012d}.pt')
        progress=(f'parity{image_parity} {offset}/{len(pending)}' if image_parity is not None
                  else f'{len(ids)-len(pending)+offset}/{len(ids)}')
        print(model,progress,'image',image_id,'seconds',round(time.perf_counter()-tick,2),flush=True)
    del wrapper
    gc.collect();torch.cuda.empty_cache()


def summarize():
    base.TITLES.update(TITLES)
    allrows,allpaired=[],[]
    cohort=json.loads((ROOT/'outputs/ffn_output_cosine_20260909/cohort500.json').read_text())
    expected_images=set(cohort['split']['train']+cohort['split']['test'])
    identity=lambda row:tuple(row[k] for k in ('mention_id','target_key','image_id','response_index','target_token_id','label'))
    for model in base.MODELS:
        directory=OUT/'subspace500'/model
        mentions=json.loads((base.OUT/'cohort500'/model/'mentions.json').read_text())
        expected_by_image={i:sorted(identity(r) for r in mentions if r['image_id']==i) for i in expected_images}
        expected_layers=np.load(base.OUT/'cohort500'/model/'metrics.npz')['rotation_mean'].shape[1]
        targets={};seen_images=set();audit=dict(images=0,targets=0,target_layers=0,rank_min=100000,rank_max=0,
            rank_drop_cases=0,rank_threshold_sensitive_cases=0,max_energy_identity_error=0.,
            zero_projected_response_cases=0,
            max_orthogonality=0.,max_parent_error=0.,max_projection_error=0.)
        spectrum_rows=[]
        for path in sorted((directory/'extraction').glob('image*.pt')):
            s=torch.load(path,map_location='cpu',weights_only=False);assert s['complete']
            image_id=s['image_id'];assert image_id not in seen_images and image_id in expected_images
            seen_images.add(image_id)
            assert path.name==f'image_{image_id:012d}.pt'
            assert sorted(identity(r) for r in s['sample_table'])==expected_by_image[image_id]
            assert s['protocol']['model']==model and set(s['protocol']['ids'])==expected_images
            audit['images']+=1
            for row in s['positions']:
                key=row['target_key'];assert key not in targets
                assert [r['layer'] for r in row['layers']]==list(range(1,expected_layers+1))
                targets[key]={name:np.array([r[name] for r in row['layers']]) for name in TITLES}
                audit['targets']+=1
                for r in row['layers']:
                    rank=r['rank'];assert rank==r['ranks']['1e-05']
                    sb,sy=r['sigma_B'].numpy(),r['sigma_Y'].numpy()
                    for values in (sb,sy):
                        assert values.shape==(rank,) and np.isfinite(values).all()
                        assert (values>=0).all() and (np.diff(values)<=0).all()
                    be,ye=float(sb@sb),float(sy@sy)
                    assert be>=0 and ye>0
                    identities=[abs(r['scalar_error_full']**2-(1-rank*r['span_c']**2/ye)),
                                abs(r['leakage_ratio']**2-(1-be/ye))]
                    if be>0:identities.append(abs(r['scalar_error_projected']**2-(1-rank*r['span_c']**2/be)))
                    assert max(identities)<1e-8
                    for k in ('scalar_error_full','scalar_error_projected','leakage_ratio','offdiag_ratio'):
                        if be==0 and k in ('scalar_error_projected','offdiag_ratio'):
                            assert np.isnan(r[k]);continue
                        assert np.isfinite(r[k]) and -1e-8<=r[k]<=1+1e-8
                    if be>0:assert r['offdiag_ratio']<=r['scalar_error_projected']+1e-8
                    audit['zero_projected_response_cases']+=be==0
                    audit['max_energy_identity_error']=max(audit['max_energy_identity_error'],*identities)
                    audit['target_layers']+=1
                    audit['rank_min']=min(audit['rank_min'],rank);audit['rank_max']=max(audit['rank_max'],rank)
                    audit['rank_drop_cases']+=rank<len(r['source_singular_values'])
                    audit['rank_threshold_sensitive_cases']+=r['ranks']['0.0001']!=r['ranks']['1e-06']
                    audit['max_orthogonality']=max(audit['max_orthogonality'],r['orthogonality_error'])
                    audit['max_parent_error']=max(audit['max_parent_error'],r['effect_parent_relative'])
                    audit['max_projection_error']=max(audit['max_projection_error'],r['write_projection_relative'])
                    spectrum_rows.append(dict(target_key=key,layer=r['layer'],rank=r['rank'],
                        rank_1e4=r['ranks']['0.0001'],rank_1e5=r['ranks']['1e-05'],rank_1e6=r['ranks']['1e-06'],
                        **{k:r[k] for k in r if k.startswith('sigma_') and not torch.is_tensor(r[k])}))
        assert seen_images==expected_images and audit['images']==500 and set(targets)=={r['target_key'] for r in mentions}
        assert audit['max_parent_error']<5e-4 and audit['max_orthogonality']<1e-9
        values={k:np.stack([targets[r['target_key']][k] for r in mentions]) for k in TITLES}
        np.savez_compressed(directory/'metrics.npz',**values)
        (directory/'mentions.json').write_text(json.dumps(mentions))
        (directory/'audit.json').write_text(json.dumps(audit,indent=2))
        base.write_csv(directory/'spectra.csv',spectrum_rows)
        rows,paired=base.summarize(model,values,mentions,directory)
        allrows+=rows;allpaired+=paired
    for name,rows in [('curves',allrows),('paired_images',allpaired),('summary',base.contrasts(allrows,allpaired))]:
        base.write_csv(OUT/'subspace500'/f'{name}.csv',rows)
    base.plot(allrows,'all',('scalar_error_full','leakage_ratio','sigma_Y_p90_p10'),OUT/'subspace500/all_subspace')
    base.plot(allrows,'all',('scalar_error_projected','offdiag_ratio','sigma_B_p90_p10'),OUT/'subspace500/all_projected')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=('smoke','extract','summarize'),required=True)
    p.add_argument('--models',nargs='+',choices=base.MODELS,default=base.MODELS)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--image-parity',type=int,choices=(0,1),help='Disjoint even/odd image-ID workers; protocol cohort stays 500')
    args=p.parse_args();torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    if args.stage=='summarize':summarize()
    else:
        for model in args.models:extract(model,args.device,args.stage=='smoke',args.image_parity)
