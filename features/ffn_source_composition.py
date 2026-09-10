"""Fixed-endpoint RMSNorm sources and a zero-to-input FFN JVP path."""
import torch
import torch.nn.functional as F

from features.ffn_visual_path_attribution import quadrature_rule


def normalized_mass(mass):
    total = mass.sum(-1, keepdim=True)
    return torch.where(total > 1e-12, mass / total.clamp_min(1e-12),
                       torch.full_like(mass, 1. / mass.shape[-1]))


def relative_error(actual, expected):
    return (actual - expected).norm(dim=-1) / expected.norm(dim=-1).clamp_min(1e-12)


def norm_sources(norm, z, sources):
    """All four supported local implementations use weight and variance_epsilon.

    The scale is frozen at z, NOT the RMSNorm Jacobian. Read the real layer's
    epsilon (Qwen 1e-6, Llama/InternLM2 1e-5), rather than a shared default.
    """
    scale = torch.rsqrt(z.square().mean(-1, keepdim=True) + norm.variance_epsilon)
    return sources * scale.unsqueeze(0) * norm.weight


def integrated_gated_jvp(ffn, n, rule):
    """Factor the SAME full FFN JVP; integrate both product-rule coefficients.

    Linearity lets source projections and the output projection run once, not
    K times. Projection biases participate in each path point, never in a
    directional projection. The actual model activation supplies its JVP.
    """
    if hasattr(ffn, 'gate_proj'):
        gate, up, down = ffn.gate_proj, ffn.up_proj, ffn.down_proj
    else:
        gate, up, down = ffn.w1, ffn.w3, ffn.w2
    with torch.no_grad():
        alpha, weight = rule.nodes[:,None,None], rule.weights[:,None,None]
        g = alpha * F.linear(n,gate.weight)
        u = alpha * F.linear(n,up.weight)
        if gate.bias is not None: g += gate.bias
        if up.bias is not None: u += up.bias
        activation, derivative = torch.func.jvp(ffn.act_fn, (g,), (torch.ones_like(g),))
        a = (weight*derivative*u).sum(0)
        b = (weight*activation).sum(0)
    def apply(directions):
        return F.linear(a*F.linear(directions,gate.weight) + b*F.linear(directions,up.weight),down.weight)
    return apply


def composition_q(norm, ffn, z, writes, k=50):
    """Signed <c_m, unit(FFN(n)-FFN(0))>, via the shared linear map's VJP.

    No division by ||c_m||: this is a projection length, not cosine. The
    direction is the full FFN endpoint difference, not the visual vector sum.
    """
    with torch.no_grad():
        n = norm(z)
        delta = ffn(n) - ffn(torch.zeros_like(n))
        length = delta.norm(dim=-1)
        degenerate = length <= 1e-12
        unit = torch.where(degenerate[:,None], torch.zeros_like(delta),
                           delta / length.clamp_min(1e-12)[:,None])
        directions = norm_sources(norm,z,writes)
        rule = quadrature_rule('gauss_legendre',k,device=z.device,dtype=z.dtype)
        linear = integrated_gated_jvp(ffn,n,rule)
    with torch.enable_grad():
        probe = torch.zeros_like(n,requires_grad=True)
        pullback = torch.autograd.grad(linear(probe),probe,unit)[0]
    q = torch.einsum('mtd,td->tm',directions,pullback).detach()
    return q,length.detach(),degenerate.detach()


def compose_ffn(norm, ffn, z, sources, visual_slice, k=4, chunk=64, save_vectors=False, factored=False):
    """sources[S,T,D]: all attention writes, residual skip, then output bias.

    Integrate at alpha*norm(z), even when finite-precision source sums differ
    from z. Report that discrepancy; never manufacture a residual contributor.
    """
    with torch.no_grad():
        n = norm(z)
        directions = norm_sources(norm, z, sources)
        output, baseline = ffn(n), ffn(torch.zeros_like(n))
    rule = quadrature_rule('gauss_legendre', k, device=z.device, dtype=z.dtype)
    fast_jvp = integrated_gated_jvp(ffn,n,rule) if factored else None
    magnitudes = torch.empty(z.shape[0], len(sources), device=z.device, dtype=z.dtype)
    total = torch.zeros_like(output)
    visual_sum = torch.zeros_like(output)
    other_sum = torch.zeros_like(output)
    saved = []
    vstart, vend = visual_slice
    for start in range(0, len(sources), chunk):
        stop = min(start + chunk, len(sources))
        block = directions[start:stop]
        contribution = torch.zeros_like(block)
        if factored:
            with torch.no_grad(): contribution = fast_jvp(block)
        else:
            for alpha, weight in zip(rule.nodes, rule.weights):
                with torch.enable_grad():
                    response = torch.vmap(lambda d: torch.func.jvp(ffn, (alpha*n,), (d,))[1])(block)
                contribution.add_(response.detach(), alpha=float(weight))
        magnitudes[:, start:stop] = contribution.norm(dim=-1).T
        total += contribution.sum(0)
        lo, hi = max(start, vstart), min(stop, vend)
        if lo < hi:
            visual_sum += contribution[lo-start:hi-start].sum(0)
        for lo, hi in ((start, min(stop, vstart)), (max(start, vend), min(stop, len(sources)-2))):
            if lo < hi:
                other_sum += contribution[lo-start:hi-start].sum(0)
        if save_vectors:
            saved.append(contribution.cpu())
    visual_mag = magnitudes[:, vstart:vend]
    strength = visual_mag.sum(-1)
    all_gross = magnitudes.sum(-1) + baseline.norm(dim=-1)
    error = (total + baseline - output).norm(dim=-1)
    result = dict(
        c_mag=visual_mag.cpu(), p_c=normalized_mass(visual_mag).cpu(),
        p_n=normalized_mass(directions[vstart:vend].norm(dim=-1).T).cpu(),
        S_C=strength.cpu(), N_C=visual_sum.norm(dim=-1).cpu(),
        kappa_C=(visual_sum.norm(dim=-1)/strength.clamp_min(1e-12)).cpu(),
        visual_fraction=(strength/all_gross.clamp_min(1e-12)).cpu(),
        other_token_mag=torch.cat((magnitudes[:,:vstart], magnitudes[:,vend:-2]), -1).cpu(),
        residual_mag=magnitudes[:,-2].cpu(), attention_bias_mag=magnitudes[:,-1].cpu(),
        ffn_zero_norm=baseline.norm(dim=-1).cpu(), gross_all=all_gross.cpu(),
        closure_relative=(error/output.norm(dim=-1).clamp_min(1e-12)).cpu(),
        closure_absolute=error.cpu(), output_norm=output.norm(dim=-1).cpu(),
        input_relative=relative_error(sources.sum(0),z).cpu(),
        norm_relative=relative_error(directions.sum(0),n).cpu(),
        norm_formula_relative=relative_error(norm_sources(norm,z,z.unsqueeze(0))[0],n).cpu(),
        degenerate=(strength<=1e-12).cpu())
    if factored:
        with torch.no_grad():
            radial_sum = fast_jvp(n)
        result['quadrature_relative'] = relative_error(radial_sum+baseline,output).cpu()
        result['source_output_relative'] = ((total-radial_sum).norm(dim=-1)/output.norm(dim=-1).clamp_min(1e-12)).cpu()
    if save_vectors:
        result['vectors'] = dict(c=torch.cat(saved), n=directions.cpu(),
                                 sources=sources.cpu(), z=z.cpu(), n_endpoint=n.cpu(),
                                 output=output.cpu(), baseline=baseline.cpu(),
                                 visual_sum=visual_sum.cpu(), other_sum=other_sum.cpu())
    return result
