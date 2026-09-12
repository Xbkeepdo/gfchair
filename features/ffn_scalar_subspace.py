"""Numerical visual-write bases and complete (including leakage) cI tests."""
import torch
import numpy as np


def visual_basis(a,rtol=1e-5):
    """Column normalization preserves span; FP64 Gram eigensolve plus QR.

    Retain EVERY singular direction above the declared numerical threshold,
    not a fixed top-k. FP64 is required because Gram squares conditioning.
    """
    a=a.double()
    norms=a.norm(dim=0)
    if a.ndim!=2 or not torch.isfinite(a).all() or not (norms>0).any():
        raise ValueError('Expected nonzero finite [D,M] writes')
    unit=a[:,norms>0]/norms[norms>0]
    gram=unit.T@unit
    eigen,v=torch.linalg.eigh(gram)
    if float(eigen.min()) < -1e-10*float(eigen.max()): raise ValueError('Invalid Gram spectrum')
    sigma=eigen.clamp_min(0).sqrt().flip(0)
    v=v.flip(1)
    ranks={str(t):int((sigma>sigma[0]*t).sum()) for t in (1e-4,1e-5,1e-6)}
    rank=int((sigma>sigma[0]*rtol).sum())
    if rank==0: raise ValueError('Empty numerical span')
    qraw=(unit@v[:,:rank])/sigma[:rank]
    q=torch.linalg.qr(qraw,mode='reduced')[0]
    orth=float((q.T@q-torch.eye(rank,device=a.device,dtype=a.dtype)).norm())
    error=float((a-q@(q.T@a)).norm()/a.norm())
    if orth>1e-9: raise ValueError(f'Nonorthogonal basis {orth}')
    return q,dict(rank=rank,rtol=rtol,ranks=ranks,orthogonality_error=orth,
                  write_projection_relative=error,zero_writes=int((norms==0).sum()),
                  source_singular_values=sigma.cpu())


def scalar_subspace(q,y,*,reference_svd=False):
    """B is a compression, not a complete restriction unless leakage is zero."""
    q,y=q.double(),y.double()
    if q.shape!=y.shape or q.ndim!=2 or not torch.isfinite(y).all(): raise ValueError('Invalid Q/Y')
    rank=q.shape[1]
    eye=torch.eye(rank,device=q.device,dtype=q.dtype)
    if float((q.T@q-eye).norm())>1e-8: raise ValueError('Q is not orthonormal')
    b=q.T@y
    c=b.trace()/rank
    outside=y-q@b
    bnorm,ynorm=b.norm(),y.norm()
    if ynorm==0: raise ValueError('Zero operator: relative diagnostics undefined')
    full=(y-c*q).norm()/ynorm
    leakage=outside.norm()/ynorm
    projected=(b-c*eye).norm()
    decomp=abs(float(full**2-leakage**2-(projected/ynorm)**2))
    if decomp>1e-9: raise ValueError('Projection residual identity failed')
    # FP64 Gram spectra avoid the FP32 Jacobi SVD's observed ~4e-5 error.
    # Direct FP64 SVD remains the independent smoke reference. Tiny tail
    # singular values inherit Gram conditioning; report robust P90/P10 too.
    driver='gesvdj' if b.is_cuda else None
    sb=torch.linalg.eigvalsh(b.T@b).clamp_min(0).sqrt().flip(0).cpu()
    sy=torch.linalg.eigvalsh(y.T@y).clamp_min(0).sqrt().flip(0).cpu()
    def spectrum(s):
        # These vectors are saved on CPU anyway; avoid repeated GPU sorting/sync.
        s=s.numpy();lo,hi=np.quantile(s,[.1,.9])
        return dict(min=float(s.min()),max=float(s.max()),mean=float(s.mean()),
                    cv=float(s.std()/s.mean()) if s.mean()>0 else float('nan'),
                    p90_p10=float(hi/lo) if lo>0 else (float('inf') if hi>0 else float('nan')))
    stats=dict(span_c=float(c),scalar_error_full=float(full),leakage_ratio=float(leakage),
               scalar_error_projected=float(projected/bnorm) if bnorm>0 else float('nan'),
               offdiag_ratio=float((b-torch.diag(b.diag())).norm()/bnorm) if bnorm>0 else float('nan'),
               projected_response_fraction=float(bnorm/ynorm),decomposition_error=decomp,
               **{f'sigma_B_{k}':v for k,v in spectrum(sb).items()},
               **{f'sigma_Y_{k}':v for k,v in spectrum(sy).items()})
    if reference_svd:
        reference=torch.linalg.svdvals(b,driver=driver)
        relative=float((reference.cpu()-sb).norm()/reference.norm().cpu().clamp_min(1e-30))
        if relative>1e-5: raise ValueError(f'Gram/direct FP64 SVD mismatch {relative}')
        stats['gram_direct_svd_relative']=relative
    return stats,b,sb.cpu(),sy.cpu()
