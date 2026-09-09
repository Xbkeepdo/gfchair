"""Signed path target consequences; all visual sources share each score VJP."""
from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint

from features.ffn_visual_path_attribution import quadrature_rule, target_scalar_from_logits
from models.dgst_capture import resolve_decoder_final_norm, resolve_output_embedding_layer

SCORES = ('logit', 'margin', 'log_probability')


def signed_totals(values: torch.Tensor):
    values = torch.as_tensor(values).float()
    if values.ndim < 1 or not torch.isfinite(values).all():
        raise ValueError('Signed maps must be finite with a source axis')
    return values.clamp_min(0).sum(-1), (-values).clamp_min(0).sum(-1)


def q_features(q, strength, endpoint_degenerate):
    q, strength = torch.as_tensor(q).float(), torch.as_tensor(strength).float()
    bad = torch.as_tensor(endpoint_degenerate).bool()
    if q.shape[:-1] != strength.shape or bad.shape != strength.shape:
        raise ValueError('Q/S/degenerate shapes differ')
    if not torch.isfinite(strength).all() or (strength < 0).any():
        raise ValueError('Invalid gross strength')
    positive, negative = signed_totals(q)
    ratio = torch.where(strength > 0, negative / strength.clamp_min(1e-12), 0.)
    if (ratio > 1 + 2e-6).any():
        raise ValueError('Negative Q projection exceeds gross strength')
    return dict(Q_positive=positive, Q_negative=negative, B_Q=ratio,
                Q_degenerate=bad | (strength <= 1e-12))


def _fp32_call(module, args, kwargs=None):
    state = {name: value.float() if value.is_floating_point() else value
             for name, value in list(module.named_parameters()) + list(module.named_buffers())}
    return torch.func.functional_call(module, state, args, kwargs or {})


def path_consequences(*, ffn_map, score_from_ffn, z, writes, integration_points=4, token_chunk=256, node_batch=1):
    """Return [3,M] C, not a clean-gradient-times-integrated-vector proxy.

    The score callback returns the three scalars at the actual path point.
    Only the current G/path and attribution accumulation are required FP32;
    the caller records the downstream score model's own precision separately.
    """
    if z.ndim != 1 or writes.ndim != 2 or writes.shape[1] != z.numel() or not writes.shape[0]:
        raise ValueError('Expected z:[D], writes:[M,D]')
    if z.dtype not in (torch.float32, torch.float64) or writes.dtype != z.dtype:
        raise TypeError('C path requires FP32/FP64 z and writes')
    if token_chunk <= 0 or node_batch <= 0 or not torch.isfinite(z).all() or not torch.isfinite(writes).all():
        raise ValueError('Invalid path input')
    z, writes = z.detach(), writes.detach()
    aggregate = writes.sum(0)
    baseline = z - aggregate
    rule = quadrature_rule('gauss_legendre', integration_points, device=z.device, dtype=z.dtype)
    result = torch.zeros((len(SCORES), len(writes)), device=z.device, dtype=z.dtype)
    direct = torch.zeros(len(SCORES), device=z.device, dtype=z.dtype)
    node_scores = []
    for offset in range(0,len(rule.nodes),node_batch):
        nodes, weights = rule.nodes[offset:offset+node_batch], rule.weights[offset:offset+node_batch]
        with torch.enable_grad():
            point = (baseline + nodes[:,None] * aggregate).detach()
            if len(nodes)==1:
                point=point[0]
            point.requires_grad_(True)
            scores = score_from_ffn(ffn_map(point))
            expected = (3,) if len(nodes)==1 else (len(nodes),3)
            if scores.shape != expected or not torch.isfinite(scores).all():
                raise ValueError('Score callback must return three finite scalars')
            totals=scores if scores.ndim==1 else scores.sum(0)
            pullbacks = torch.stack([torch.autograd.grad(value, point, retain_graph=i < 2)[0]
                                     for i, value in enumerate(totals)]).detach().reshape(3,len(nodes),-1)
            averaged=torch.einsum('skd,k->sd',pullbacks,weights)
        for start in range(0, len(writes), token_chunk):
            result[:, start:start + token_chunk].add_(averaged @ writes[start:start + token_chunk].T)
        direct.add_(averaged @ aggregate)
        node_scores.append(scores.detach().reshape(-1,3))
        del point, scores, pullbacks, averaged, totals
    with torch.no_grad():
        clean_scores = score_from_ffn(ffn_map(z)).detach()
        baseline_scores = score_from_ffn(ffn_map(baseline)).detach()
    finite_effect = clean_scores - baseline_scores
    sums = result.sum(-1)
    absolute = (sums - finite_effect).abs()
    zero = finite_effect.abs() <= 1e-12
    relative = torch.where(zero, torch.full_like(absolute, torch.nan), absolute / finite_effect.abs())
    if not all(torch.isfinite(x).all() for x in (result, direct, clean_scores, baseline_scores, absolute)):
        raise ValueError('Nonfinite target consequence')
    positive, negative = signed_totals(result)
    return dict(C_m=result, positive=positive, negative=negative, signed_sum=sums,
                clean_scores=clean_scores, baseline_scores=baseline_scores,
                finite_score_effect=finite_effect, closure_absolute_error=absolute,
                closure_relative_error=relative, score_effect_degenerate=zero,
                source_sum_vjp_error=(sums - direct).abs(),
                quadrature_nodes=rule.nodes, quadrature_weights=rule.weights,
                node_scores=torch.cat(node_scores))


def detach_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, tuple):
        return tuple(detach_tree(x) for x in value)
    if isinstance(value, list):
        return [detach_tree(x) for x in value]
    if isinstance(value, dict):
        return {k: detach_tree(x) for k, x in value.items()}
    if value is not None and not isinstance(value, (str, int, float, bool)):
        raise TypeError(f'Unsupported cached decoder argument: {type(value).__name__}')
    return value


def native_suffix(*, model, layers, captures, calls, layer_index, prediction_position,
                  target_id, competitor_id, use_checkpoint=True):
    """Replay only the native downstream blocks, with clean causal prefix rows.

    Keeping original decoder arguments and full GEMM shapes supports all four
    families without a new hand-written attention/rotary implementation.
    Current-layer residual skip stays clean. Cached prefixes are constants:
    the intervention cannot causally affect rows before the final query.
    """
    current = captures[layer_index]
    q = int(prediction_position)
    if q != current['h_mid'].shape[1] - 1:
        raise ValueError('Suffix requires target-excluding prefix ending at prediction row')
    norm, head = resolve_decoder_final_norm(model), resolve_output_embedding_layer(model)
    native_dtype = current['h_mid'].dtype
    final_prefix = (captures[-1]['h_mid'] + captures[-1]['o_ffn'])[:, :q].detach()

    def scores(replacement):
        query = current['h_mid'][:, q] + replacement.to(native_dtype).reshape(1, -1)
        for index in range(layer_index + 1, len(layers)):
            prefix = captures[index]['h_prev'][:, :q]
            args, kwargs = calls[index]
            def step(value, layer=layers[index], prefix=prefix, args=args, kwargs=kwargs):
                output = layer(torch.cat((prefix, value.unsqueeze(1)), dim=1), *args, **kwargs)
                hidden = output[0] if isinstance(output, (tuple, list)) else output
                return hidden[:, q]
            query = (checkpoint(step, query, use_reentrant=False)
                     if use_checkpoint and torch.is_grad_enabled() else step(query))
        hidden = torch.cat((final_prefix, query.unsqueeze(1)), dim=1)
        logits = head(norm(hidden))[0, q].float()
        return torch.stack([target_scalar_from_logits(logits, target_token_id=target_id,
                           competitor_token_id=competitor_id, scalar=name) for name in SCORES])
    return scores


def fp32_suffix(*, model, layers, captures, calls, layer_index, prediction_position,
                target_id, competitor_id, use_checkpoint=True):
    """FP32 diagnostic suffix without a second resident FP32 model.

    Captured prefix/current residual states remain native-model observations.
    Downstream blocks are replayed in FP32; checkpoint recomputation limits
    FP32 weight-copy lifetime. This is NOT whole-VLM FP32 recapture.
    Qwen3's clean, visual-only deepstack additions are retained as frozen
    effective residual additions measured from the native capture.
    """
    q = int(prediction_position)
    current = captures[layer_index]
    if q != current['h_mid'].shape[1] - 1:
        raise ValueError('FP32 suffix requires a target-excluding prefix')
    norm, head = resolve_decoder_final_norm(model), resolve_output_embedding_layer(model)
    prefix = (current['h_mid'] + current['o_ffn'])[:, :q].float()
    def promote(value):
        if isinstance(value, torch.Tensor):
            return value.float() if value.is_floating_point() else value
        if isinstance(value, tuple):
            return tuple(promote(x) for x in value)
        if isinstance(value, dict):
            return {k: promote(x) for k, x in value.items()}
        return value
    call = _fp32_call
    def scores(replacement):
        query = current['h_mid'][:, q].float() + replacement.float().reshape(1, -1)
        hidden = torch.cat((prefix, query.unsqueeze(1)), dim=1)
        for index in range(layer_index + 1, len(layers)):
            previous = captures[index-1]
            extra = (captures[index]['h_prev'] - (previous['h_mid'] + previous['o_ffn'])).float()
            if extra[:, q].abs().max() != 0:
                raise ValueError('Unexpected nonvisual suffix residual addition')
            hidden = hidden + extra
            args, kwargs = calls[index]
            def step(value, layer=layers[index], args=promote(args), kwargs=promote(kwargs)):
                output = call(layer, (value, *args), kwargs)
                return output[0] if isinstance(output, (tuple, list)) else output
            hidden = (checkpoint(step, hidden, use_reentrant=False)
                      if use_checkpoint and torch.is_grad_enabled() else step(hidden))
        logits = call(head, (call(norm, (hidden[:, q],)),))[0]
        return torch.stack([target_scalar_from_logits(logits, target_token_id=target_id,
                           competitor_token_id=competitor_id, scalar=name) for name in SCORES])
    return scores


class _PrefixKV:
    """Minimal native attention Cache.update contract; query reads never append."""
    def __init__(self, states=None):
        self.states = states
    def update(self, key, value, layer_idx, cache_kwargs=None):
        if self.states is None:
            self.states = (key.detach(), value.detach())
            return key, value
        return tuple(torch.cat((prefix.expand(current.shape[0],-1,-1,-1), current), dim=-2)
                     for prefix, current in zip(self.states, (key, value)))


def fp32_cached_suffix(*, model, layers, captures, calls, layer_index, prediction_position,
                       target_id, competitor_id, use_checkpoint=True):
    """Same FP32 suffix estimand, reusing native attention's immutable prefix K/V.

    Prefixes cannot depend on an intervention in the final query row. Compute
    their FP32 trajectory once, then use the existing decoder/cache interfaces
    for query-only evaluations. No custom attention or rotary algebra.
    """
    q = int(prediction_position)
    current = captures[layer_index]
    if q < 1 or q != current['h_mid'].shape[1] - 1:
        raise ValueError('Cached suffix requires a nonempty causal prefix')
    norm, head = resolve_decoder_final_norm(model), resolve_output_embedding_layer(model)
    prefix = (current['h_mid'] + current['o_ffn'])[:, :q].float()
    cached = []
    def arguments(original, query):
        kw = dict(original)
        for key, value in list(kw.items()):
            if key in ('past_key_values','past_key_value'):
                kw.pop(key)
            elif key == 'position_embeddings' and value is not None:
                kw[key] = tuple(v[..., q:q+1, :].float() if query else v[..., :q, :].float() for v in value)
            elif key in ('position_ids','cache_position') and value is not None:
                kw[key] = value[..., q:q+1] if query else value[..., :q]
            elif key == 'attention_mask' and value is not None:
                kw[key] = value[..., q:q+1, :q+1].float() if query else value[..., :q, :q].float()
        if kw.get('attention_mask') is None:
            kw['attention_mask'] = (torch.zeros((1,1,1,q+1),device=prefix.device) if query else
                torch.triu(torch.full((1,1,q,q),torch.finfo(torch.float32).min,device=prefix.device),diagonal=1))
        kw['use_cache'] = True
        kw['output_attentions'] = False
        return kw
    with torch.no_grad():
        for index in range(layer_index+1,len(layers)):
            previous = captures[index-1]
            extra = (captures[index]['h_prev']-(previous['h_mid']+previous['o_ffn'])).float()
            if extra[:,q].abs().max()!=0:
                raise ValueError('Unexpected query residual addition')
            prefix = prefix + extra[:,:q]
            args, original = calls[index]
            if args:
                raise ValueError('Cached suffix requires keyword decoder metadata')
            layer = layers[index]
            intern = hasattr(layer,'feed_forward')
            slot = None if intern else _PrefixKV()
            kw = arguments(original,False)
            kw['past_key_value' if intern else 'past_key_values'] = slot
            output = _fp32_call(layer,(prefix,),kw)
            prefix = output[0] if isinstance(output,(tuple,list)) else output
            states = output[-1] if intern else slot.states
            if not isinstance(states,tuple) or len(states)!=2 or any(v.shape[-2]!=q for v in states):
                raise ValueError('Native decoder did not expose expected prefix K/V')
            query_kw = arguments(original,True)
            query_kw['past_key_value' if intern else 'past_key_values'] = states if intern else _PrefixKV(states)
            cached.append((layer,query_kw))
    def scores(replacement):
        scalar_input = replacement.ndim == 1
        query = current['h_mid'][:,q].float() + replacement.float().reshape(-1,current['h_mid'].shape[-1])
        for layer, kwargs in cached:
            def step(value,layer=layer,kwargs=kwargs):
                kw=dict(kwargs)
                if isinstance(kw.get('attention_mask'),torch.Tensor) and kw['attention_mask'].ndim==4:
                    kw['attention_mask']=kw['attention_mask'].expand(value.shape[0],-1,-1,-1)
                if isinstance(kw.get('past_key_value'),tuple):
                    kw['past_key_value']=tuple(v.expand(value.shape[0],-1,-1,-1) for v in kw['past_key_value'])
                output = _fp32_call(layer,(value.unsqueeze(1),),kw)
                return (output[0] if isinstance(output,(tuple,list)) else output)[:,0]
            query = checkpoint(step,query,use_reentrant=False) if use_checkpoint and torch.is_grad_enabled() else step(query)
        logits = _fp32_call(head,(_fp32_call(norm,(query,)),))
        values=torch.stack([logits[:,target_id],logits[:,target_id]-logits[:,competitor_id],
                            torch.log_softmax(logits,dim=-1)[:,target_id]],dim=-1)
        return values[0] if scalar_input else values
    return scores
