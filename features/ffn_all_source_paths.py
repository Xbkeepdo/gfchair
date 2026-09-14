"""Streaming attention paths and two explicitly different residual decompositions."""
import math

import numpy as np
import torch
import torch.nn.functional as F

from features.ffn_source_composition import norm_sources, integrated_gated_jvp
from features.ffn_visual_path_attribution import quadrature_rule

SOURCES = ('prompt', 'visual', 'generation')


def ratio(a, b):
    return torch.where(b > 1e-12, a / b.clamp_min(1e-12), torch.full_like(a, float('nan')))


def cosine(a, b):
    a, b = a.double(), b.double()
    return ratio((a*b).sum(-1), a.norm(dim=-1)*b.norm(dim=-1)).clamp(-1, 1)


def rel(a, b):
    return (a.double()-b.double()).norm(dim=-1) / b.double().norm(dim=-1).clamp_min(1e-12)


def partition(length, query, response_index, visual_range, device=None):
    start, end = visual_range
    generation_start = query-response_index+1
    if not 0 <= start < end <= generation_start <= query+1 <= length:
        raise ValueError('Invalid causal source boundaries')
    pos = torch.arange(length, device=device)
    masks = torch.stack(((pos < generation_start) & ~((pos >= start) & (pos < end)),
                         (pos >= start) & (pos < end),
                         (pos >= generation_start) & (pos <= query)))
    if not torch.equal(masks.sum(0), (pos <= query).long()):
        raise ValueError('Sources do not partition causal prefix')
    return masks


def path_operator(norm, ffn, baseline, delta, nodes, weights):
    """Factor full RMSNorm+gated-FFN JVP at supplied alpha nodes.

    Input baseline/delta are [T,D]. Directions are [S,T,D]. Both gated
    product-rule terms and the RMS rank-one correction are included.
    Biases affect path values, never directional projections.
    """
    gate, up, down = ((ffn.gate_proj, ffn.up_proj, ffn.down_proj)
                      if hasattr(ffn, 'gate_proj') else (ffn.w1, ffn.w3, ffn.w2))
    with torch.no_grad():
        x = baseline[None] + nodes.to(baseline)[:, None, None]*delta[None]
        rms = torch.rsqrt(x.square().mean(-1, keepdim=True)+norm.variance_epsilon)
        n = x*rms*norm.weight
        g, u = gate(n), up(n)
        act, derivative = torch.func.jvp(ffn.act_fn, (g,), (torch.ones_like(g),))
        c = derivative*u
        w = weights.to(device=x.device, dtype=torch.float64)[:, None, None]
        left = (w*rms.double()*c.double()).sum(0).to(x)
        right = (w*rms.double()*act.double()).sum(0).to(x)
        correction = c*F.linear(x*norm.weight, gate.weight)+act*F.linear(x*norm.weight, up.weight)
        correction = correction.double()*(w*rms.double().pow(3)/x.shape[-1])

    def apply(directions):
        with torch.no_grad():
            a = directions.to(x)
            hidden = left*F.linear(a*norm.weight, gate.weight)+right*F.linear(a*norm.weight, up.weight)
            dots = torch.einsum('std,ktd->kst', a, x).double()
            hidden = hidden.double()-torch.einsum('kst,kti->sti', dots, correction)
            return F.linear(hidden.to(x), down.weight).double()
    return apply


def direct_integral(function, baseline, delta, directions, nodes, weights, chunk=16):
    result = torch.zeros_like(directions, dtype=torch.float64)
    for begin in range(0, len(directions), chunk):
        block = directions[begin:begin+chunk]
        for node, weight in zip(nodes, weights):
            with torch.enable_grad():
                v = torch.vmap(lambda d: torch.func.jvp(function, (baseline+node*delta,), (d,))[1])(block)
            result[begin:begin+len(block)] += v.detach().double()*weight.double()
    return result


def radial_operator(norm, ffn, delta, nodes, weights):
    """Same zero-bias radial derivative without subtracting its parallel terms.

    J_RMS(alpha*S)a = w*r*(a_perp + eps*r**2*a_parallel).
    This is algebraically exact including epsilon; the radial derivative is
    NOT set to zero under an approximate scale-invariance assumption.
    """
    gate, up, down = ((ffn.gate_proj, ffn.up_proj, ffn.down_proj)
                      if hasattr(ffn, 'gate_proj') else (ffn.w1, ffn.w3, ffn.w2))
    with torch.no_grad():
        x = nodes.to(delta)[:, None, None]*delta[None]
        rms = torch.rsqrt(x.square().mean(-1, keepdim=True)+norm.variance_epsilon)
        n = x*rms*norm.weight
        g, u = gate(n), up(n)
        act, derivative = torch.func.jvp(ffn.act_fn, (g,), (torch.ones_like(g),))
        common = weights.double()[:, None, None]*rms.double()
        c = (derivative*u).double()
        parallel = norm.variance_epsilon*rms.double().square()
        left, right = (common*c).sum(0), (common*act.double()).sum(0)
        lparallel, rparallel = (common*parallel*c).sum(0), (common*parallel*act.double()).sum(0)
        square = delta.double().square().sum(-1)

    def apply(directions):
        a = directions.double()
        coefficient = (a*delta.double()).sum(-1)/square.clamp_min(1e-300)
        along = coefficient[..., None]*delta.double()
        across = a-along
        # Projection inputs remain the original local model precision, while
        # coefficients and the parallel/perpendicular combination use FP64.
        h = left*F.linear((across*norm.weight).to(delta), gate.weight).double()
        h += right*F.linear((across*norm.weight).to(delta), up.weight).double()
        h += lparallel*F.linear((along*norm.weight).to(delta), gate.weight).double()
        h += rparallel*F.linear((along*norm.weight).to(delta), up.weight).double()
        return F.linear(h.to(delta), down.weight).double()
    return apply


def adaptive_b1(norm, ffn, residual_sources, bias, *, rtol=1e-4, atol=1e-7, max_depth=12):
    """True path, sinh-distributed alpha; adaptive 16/32 GL per target.

    Returns four group vectors (R,P,V,G), the independent radial integral,
    and diagnostics. FP64 accumulation does not imply an FP64 model.
    """
    if residual_sources.shape[1] != 1:
        raise ValueError('Adaptive B1 operates on one target at a time')
    svec = residual_sources.sum(0)
    probes = torch.cat((residual_sources, svec[None]), 0)
    magnitude = float(svec.double().square().mean().sqrt())
    if magnitude == 0:
        value = direct_integral(lambda x: ffn(norm(x)), bias, svec, probes,
                                torch.zeros(1, device=bias.device), torch.ones(1, device=bias.device))
        return value[:4], value[4], dict(evaluations=1, panels=1, max_depth=0, converged=True)
    scale = math.sqrt(norm.variance_epsilon)/magnitude
    stop = math.asinh(1/scale)
    rules = {k: np.polynomial.legendre.leggauss(k) for k in (16, 32)}
    stats = dict(evaluations=0, panels=0, max_depth=0, converged=True)

    def panel(lo, hi, k):
        nodes, weights = rules[k]
        t = torch.tensor(lo+(nodes+1)*(hi-lo)/2, dtype=torch.float64, device=bias.device)
        alpha = scale*torch.sinh(t)
        weight = torch.tensor(weights*(hi-lo)/2, dtype=torch.float64, device=bias.device)*scale*torch.cosh(t)
        stats['evaluations'] += k
        if not bias.count_nonzero():
            return radial_operator(norm, ffn, svec, alpha, weight)(probes)
        return path_operator(norm, ffn, bias, svec, alpha, weight)(probes)

    def integrate(lo, hi, depth):
        low, high = panel(lo, hi, 16), panel(lo, hi, 32)
        error = (low-high).norm(dim=-1)
        tolerance = atol*(hi-lo)/stop+rtol*high.norm(dim=-1)
        if bool((error <= tolerance).all()):
            stats['panels'] += 1
            stats['max_depth'] = max(stats['max_depth'], depth)
            return high
        if depth >= max_depth:
            raise RuntimeError(f'B1 sinh quadrature did not converge: depth={depth}, worst={float((error/tolerance).max()):.3g}')
        mid = (lo+hi)/2
        return integrate(lo, mid, depth+1)+integrate(mid, hi, depth+1)

    result = integrate(0., stop, 0)
    endpoint = (ffn(norm(bias+svec))-ffn(norm(bias))).double()
    stats['quadrature_relative'] = float(rel(result[4], endpoint).max())
    stats['closure_relative'] = float(rel(result[:4].sum(0), endpoint).max())
    stats['source_sum_relative'] = float(rel(result[:4].sum(0), result[4]).max())
    if stats['quadrature_relative'] > .01 and float((result[4]-endpoint).norm()) > 1e-6:
        raise RuntimeError(f'B1 radial closure failed: {stats}')
    if stats['closure_relative'] > .01 and float((result[:4].sum(0)-endpoint).norm()) > 1e-6:
        raise RuntimeError(f'B1 four-source closure failed: {stats}')
    return result[:4], result[4], stats


def residual_geometry(inputs, effects, prefix):
    # Shared order R,P,V,G; the optional B2 bias is excluded from C_A.
    r, p, v, g = effects[:4]
    attention = p+v+g
    rn, an = r.norm(dim=-1), attention.norm(dim=-1)
    metrics = dict(residual_norm=rn, prompt_norm=p.norm(dim=-1), visual_norm=v.norm(dim=-1),
                   generation_norm=g.norm(dim=-1), attention_total_norm=an,
                   residual_share=ratio(rn, rn+an), cos_residual_visual=cosine(r, v),
                   cos_residual_prompt=cosine(r, p), cos_residual_generation=cosine(r, g),
                   attention_residual_kappa=ratio((r+attention).norm(dim=-1), rn+an))
    metrics['residual_visual_generation_balance'] = metrics['cos_residual_generation']-metrics['cos_residual_visual']
    metrics['delta_residual_visual_cos'] = metrics['cos_residual_visual']-cosine(inputs[0], inputs[2])
    metrics['delta_residual_generation_cos'] = metrics['cos_residual_generation']-cosine(inputs[0], inputs[3])
    return {prefix+'_'+k: value for k, value in metrics.items()}


def compute_target(norm, ffn, z, residual, bias, writes, masks, ks, *, chunk=64, audit=False):
    """One target, all sources. Only chunk-sized token response vectors live."""
    if z.shape[0] != 1 or writes.shape[1] != 1:
        raise ValueError('Expected one target')
    masks = masks.to(writes.device)
    if writes[~masks.any(0)].count_nonzero():
        raise ValueError('Nonzero future write')
    causal = masks.any(0)
    positions = causal.nonzero().flatten()
    writes = writes[causal].float()
    masks = masks[:, causal]
    a = torch.stack([writes[m].sum(0) for m in masks])
    total = writes.sum(0)
    rule = quadrature_rule('gauss_legendre', ks['all'], device=z.device, dtype=z.dtype)
    op = path_operator(norm, ffn, z-total, total, rule.nodes, rule.weights)
    effects = torch.zeros_like(a, dtype=torch.float64)
    norms, write_norms, cosines, vectors = [], [], [], []
    gross = torch.zeros(3, 1, dtype=torch.float64, device=z.device)
    for lo in range(0, len(writes), chunk):
        block = writes[lo:lo+chunk]
        response = op(block)
        en = response.norm(dim=-1)
        norms.append(en[:, 0].cpu()); write_norms.append(block.norm(dim=-1)[:, 0].cpu())
        cosines.append(cosine(block, response)[:, 0].cpu())
        for gi in range(3):
            mask = masks[gi, lo:lo+len(block)]
            effects[gi] += response[mask].sum(0)
            gross[gi] += en[mask].sum(0)
        if audit: vectors.append(response.cpu())
    metrics = {}
    for i, name in enumerate(SOURCES):
        net = effects[i].norm(dim=-1)
        metrics.update({name+'_gross_norm': gross[i], name+'_net_norm': net,
                        name+'_group_gain': ratio(net, a[i].double().norm(dim=-1)),
                        name+'_group_rotation': cosine(a[i], effects[i]),
                        name+'_within_source_kappa': ratio(net, gross[i]),
                        name+'_token_count': masks[i].sum().reshape(1)})
    for i, j, name in ((1, 2, 'vg'), (1, 0, 'vp'), (0, 2, 'pg')):
        before, after = cosine(a[i], a[j]), cosine(effects[i], effects[j])
        metrics.update({'cos_'+name+'_in': before, 'cos_'+name+'_out': after,
                        'delta_cos_'+name: after-before})
    metrics['source_group_kappa'] = ratio(effects.sum(0).norm(dim=-1), effects.norm(dim=-1).sum(0))
    metrics['generation_per_token_gross'] = ratio(gross[2], masks[2].sum().reshape(1))
    vrule = quadrature_rule('gauss_legendre', ks['visual'], device=z.device, dtype=z.dtype)
    visual = path_operator(norm, ffn, z-a[1], a[1], vrule.nodes, vrule.weights)(a[1:2])[0]
    metrics['visual_context_cos'] = cosine(visual, effects[1])
    metrics['visual_context_relative_change'] = ratio((effects[1]-visual).norm(dim=-1), visual.norm(dim=-1))
    sources = torch.cat((residual[None], a), 0)
    b1, radial, b1stats = adaptive_b1(norm, ffn, sources, bias)
    metrics.update(residual_geometry(sources.double(), b1, 'b1'))
    n = norm(z)
    directions = norm_sources(norm, z, torch.cat((sources, bias[None]), 0))
    brule = quadrature_rule('gauss_legendre', ks['b2'], device=z.device, dtype=z.dtype)
    bop = integrated_gated_jvp(ffn, n, brule)
    b2 = bop(directions).double()
    metrics.update(residual_geometry(directions.double(), b2, 'b2'))
    endpoint = (ffn(norm(z))-ffn(norm(z-total))).double()
    b2endpoint = (ffn(n)-ffn(torch.zeros_like(n))).double()
    diagnostics = dict(all_closure_relative=float(rel(effects.sum(0), endpoint).max()),
        all_baseline_reconstruction_relative=float(rel(z-total, residual+bias).max()),
        input_reconstruction_relative=float(rel(sources.sum(0)+bias, z).max()),
        visual_closure_relative=float(rel(visual, ffn(norm(z))-ffn(norm(z-a[1]))).max()),
        b1=b1stats, b1_endpoint_output_difference=float(rel(ffn(norm(sources.sum(0)+bias)), ffn(norm(z))).max()),
        b2_quadrature_relative=float(rel(bop(n), b2endpoint).max()),
        b2_closure_relative=float(rel(b2.sum(0), b2endpoint).max()),
        b2_source_output_relative=float(rel(b2.sum(0), bop(n)).max()),
        b2_norm_reconstruction_relative=float(rel(directions.sum(0), n).max()),
        b2_bias_norm=float(b2[4].norm()), ffn_zero_norm=float(ffn(torch.zeros_like(n)).norm()))
    row = dict(metrics={k: float(v.item()) for k, v in metrics.items()}, diagnostics=diagnostics,
        source_position=positions.cpu(), source_type=masks.long().argmax(0).cpu().to(torch.uint8),
        write_norm=torch.cat(write_norms).float(), effect_norm=torch.cat(norms).float(),
        direction_cosine=torch.cat(cosines).float())
    if audit:
        row['vectors'] = dict(all=torch.cat(vectors), groups=effects.cpu(), visual=visual.cpu(),
                              b1=b1.cpu(), b2=b2.cpu(), writes=writes.cpu())
    return row
