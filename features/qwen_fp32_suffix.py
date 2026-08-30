"""Exact causal query-row FP32 suffix execution for Qwen2.5-VL/Qwen3-VL."""

from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from features.ffn_visual_path_attribution import (
    batched_scalar_gradients,
    target_scalar_from_logits,
)


QWEN_MODELS = {"qwen2_5_vl_7b", "qwen3_vl_8b"}


def _family(model_name: str) -> str:
    if model_name == "qwen2_5_vl_7b":
        return "qwen2"
    if model_name == "qwen3_vl_8b":
        return "qwen3"
    raise ValueError(f"Unsupported Qwen FP32 suffix model {model_name}")


def _repeat_kv(states: torch.Tensor, groups: int) -> torch.Tensor:
    if int(groups) == 1:
        return states
    return states.repeat_interleave(int(groups), dim=1)


def qwen_query_layer(
    *,
    layer: Any,
    clean_context: torch.Tensor,
    query_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    family: str,
    query_batch_size: int | None = None,
) -> torch.Tensor:
    """Run one decoder layer only at the final causal query row in FP32.

    ``clean_context`` contains the already-FP32 states strictly before the
    query.  Causality makes those states invariant to the query intervention,
    so a batch of query variants can share their keys and values exactly.
    """
    if clean_context.ndim != 2 or query_states.ndim != 2:
        raise ValueError("Expected context [S,D] and query states [B,D]")
    if clean_context.shape[1] != query_states.shape[1]:
        raise ValueError("Context/query hidden widths differ")
    if clean_context.dtype != query_states.dtype:
        raise TypeError("Qwen suffix context/query dtypes differ")

    attention = layer.self_attn
    context_length = int(clean_context.shape[0])
    batch_size = int(query_states.shape[0])
    microbatch_size = (
        batch_size if query_batch_size is None else int(query_batch_size)
    )
    if microbatch_size <= 0:
        raise ValueError("query_batch_size must be positive")
    microbatch_size = min(microbatch_size, batch_size)
    head_dim = int(attention.head_dim)
    num_heads = int(attention.q_proj.out_features) // head_dim
    num_kv_heads = int(attention.k_proj.out_features) // head_dim
    groups = num_heads // num_kv_heads

    normalized_context = layer.input_layernorm(clean_context)
    normalized_query = layer.input_layernorm(query_states)
    query = attention.q_proj(normalized_query).view(
        batch_size, 1, num_heads, head_dim
    ).transpose(1, 2)
    key_context = attention.k_proj(normalized_context).view(
        1, context_length, num_kv_heads, head_dim
    ).transpose(1, 2)
    value_context = attention.v_proj(normalized_context).view(
        1, context_length, num_kv_heads, head_dim
    ).transpose(1, 2)
    key_query = attention.k_proj(normalized_query).view(
        batch_size, 1, num_kv_heads, head_dim
    ).transpose(1, 2)
    value_query = attention.v_proj(normalized_query).view(
        batch_size, 1, num_kv_heads, head_dim
    ).transpose(1, 2)

    if family == "qwen2":
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            apply_multimodal_rotary_pos_emb,
        )

        query_cos = cos[:, :, context_length : context_length + 1]
        query_sin = sin[:, :, context_length : context_length + 1]
        query, key_query = apply_multimodal_rotary_pos_emb(
            query,
            key_query,
            query_cos,
            query_sin,
            attention.rope_scaling["mrope_section"],
        )
        _unused, key_context = apply_multimodal_rotary_pos_emb(
            key_context,
            key_context,
            cos[:, :, :context_length],
            sin[:, :, :context_length],
            attention.rope_scaling["mrope_section"],
        )
    elif family == "qwen3":
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            apply_rotary_pos_emb,
        )

        query = attention.q_norm(query.transpose(1, 2)).transpose(1, 2)
        key_context = attention.k_norm(key_context.transpose(1, 2)).transpose(1, 2)
        key_query = attention.k_norm(key_query.transpose(1, 2)).transpose(1, 2)
        query, key_query = apply_rotary_pos_emb(
            query,
            key_query,
            cos[:, context_length : context_length + 1],
            sin[:, context_length : context_length + 1],
        )
        _unused, key_context = apply_rotary_pos_emb(
            key_context,
            key_context,
            cos[:, :context_length],
            sin[:, :context_length],
        )
    else:
        raise ValueError(f"Unknown Qwen family {family}")

    # The causal prefix K/V is identical for every intervention query.  Keep a
    # single head-expanded copy and only materialize it for a bounded query
    # microbatch.  Expanding the full 225--241 intervention grid before GQA
    # head repetition otherwise consumes more than a 24 GiB card by itself.
    key_context = _repeat_kv(key_context, groups)
    value_context = _repeat_kv(value_context, groups)
    key_query = _repeat_kv(key_query, groups)
    value_query = _repeat_kv(value_query, groups)
    outputs = []
    for start in range(0, batch_size, microbatch_size):
        stop = min(start + microbatch_size, batch_size)
        chunk_size = stop - start
        chunk_query = query[start:stop]
        key = torch.cat(
            (
                key_context.expand(chunk_size, -1, -1, -1),
                key_query[start:stop],
            ),
            dim=2,
        )
        value = torch.cat(
            (
                value_context.expand(chunk_size, -1, -1, -1),
                value_query[start:stop],
            ),
            dim=2,
        )
        weights = torch.matmul(chunk_query, key.transpose(2, 3)) * float(
            attention.scaling
        )
        weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attended = torch.matmul(weights, value)
        attended = attended.transpose(1, 2).reshape(
            chunk_size, num_heads * head_dim
        )
        hidden = query_states[start:stop] + attention.o_proj(attended)
        outputs.append(hidden + layer.mlp(layer.post_attention_layernorm(hidden)))
    return torch.cat(outputs, dim=0)


def qwen_query_layer_fp32(**kwargs) -> torch.Tensor:
    clean_context = kwargs["clean_context"]
    query_states = kwargs["query_states"]
    if clean_context.dtype != torch.float32 or query_states.dtype != torch.float32:
        raise TypeError("Qwen FP32 suffix execution requires true FP32 states")
    return qwen_query_layer(**kwargs)


def _causal_mask(
    length: int, device: torch.device, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    mask = torch.full(
        (length, length),
        torch.finfo(dtype).min,
        dtype=dtype,
        device=device,
    )
    mask = torch.triu(mask, diagonal=1)
    return mask.view(1, 1, length, length)


def _full_context_layer_fp32(
    *,
    layer: Any,
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    family: str,
) -> torch.Tensor:
    length = int(hidden_states.shape[0])
    mask = _causal_mask(length, hidden_states.device)
    if family == "qwen2":
        output = layer(
            hidden_states.unsqueeze(0),
            attention_mask=mask,
            position_embeddings=(cos[:, :, :length], sin[:, :, :length]),
            output_attentions=False,
            use_cache=False,
        )[0]
    else:
        output = layer(
            hidden_states.unsqueeze(0),
            attention_mask=mask,
            position_embeddings=(cos[:, :length], sin[:, :length]),
            use_cache=False,
        )
    return output[0]


def _position_embeddings_fp32(
    *,
    model: Any,
    inputs: Mapping[str, Any],
    reference_hidden: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    base = model.model
    position_ids, _rope_delta = base.get_rope_index(
        input_ids=inputs["input_ids"],
        image_grid_thw=inputs.get("image_grid_thw"),
        video_grid_thw=inputs.get("video_grid_thw"),
        attention_mask=inputs.get("attention_mask"),
    )
    return base.language_model.rotary_emb(
        reference_hidden.unsqueeze(0), position_ids
    )


def qwen2_cached_text_replay_inputs(
    *, model: Any, inputs: Mapping[str, Any]
) -> dict[str, torch.Tensor]:
    """Cache Qwen2.5 multimodal embeddings for repeated decoder replays."""
    base = model.model
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    with torch.no_grad():
        inputs_embeds = base.get_input_embeddings()(input_ids)
        pixel_values = inputs.get("pixel_values")
        if pixel_values is not None:
            image_features = base.get_image_features(
                pixel_values, inputs.get("image_grid_thw")
            )
            image_features = torch.cat(image_features, dim=0).to(
                device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )
            image_mask, _video_mask = base.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                image_features=image_features,
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                image_mask, image_features
            )
        position_ids, _rope_deltas = base.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=inputs.get("image_grid_thw"),
            video_grid_thw=inputs.get("video_grid_thw"),
            second_per_grid_ts=inputs.get("second_per_grid_ts"),
            attention_mask=attention_mask,
        )
    result = {
        "inputs_embeds": inputs_embeds.detach(),
        "position_ids": position_ids.detach(),
    }
    if attention_mask is not None:
        result["attention_mask"] = attention_mask
    return result


def qwen_native_suffix_gradients_at_replacement(
    *,
    model: Any,
    model_name: str,
    layers: Sequence[Any],
    captures: Sequence[Mapping[str, torch.Tensor]],
    intervention_layer_index: int,
    prediction_position: int,
    replacement: torch.Tensor,
    final_norm: Any,
    output_embedding: Any,
    inputs: Mapping[str, Any],
    target_token_id: int,
    competitor_token_id: int,
    target_scalars: Sequence[str],
) -> tuple[dict[str, torch.Tensor], dict[str, float], torch.Tensor, dict[str, Any]]:
    """Exact native-dtype Qwen suffix with cached causal prefix states."""
    family = _family(model_name)
    current_capture = captures[intervention_layer_index]
    h_mid = current_capture["h_mid"][0, : prediction_position + 1]
    if prediction_position != int(h_mid.shape[0]) - 1:
        raise ValueError("Qwen native suffix requires the final causal query row")
    leaf = replacement.detach().to(h_mid.dtype).requires_grad_(True)
    query = h_mid[prediction_position] + leaf
    cos, sin = _position_embeddings_fp32(
        model=model,
        inputs=inputs,
        reference_hidden=h_mid,
    )
    for downstream_index in range(intervention_layer_index + 1, len(layers)):
        context = captures[downstream_index]["h_prev"][
            0, :prediction_position
        ].to(device=query.device, dtype=query.dtype)
        query = qwen_query_layer(
            layer=layers[downstream_index],
            clean_context=context,
            query_states=query.unsqueeze(0),
            cos=cos,
            sin=sin,
            family=family,
        )[0]
    logits = output_embedding(final_norm(query)).float()
    scalar_names = list(target_scalars)
    gradients: dict[str, torch.Tensor] = {}
    values: dict[str, float] = {}
    for offset, scalar in enumerate(scalar_names):
        value = target_scalar_from_logits(
            logits,
            target_token_id=target_token_id,
            competitor_token_id=competitor_token_id,
            scalar=scalar,
        )
        values[scalar] = float(value.detach())
        gradients[scalar] = torch.autograd.grad(
            value,
            leaf,
            retain_graph=offset + 1 < len(scalar_names),
        )[0].detach()
    audit = {
        "route": "qwen_exact_causal_query_suffix",
        "downstream_layer_count": len(layers) - intervention_layer_index - 1,
        "cached_context_dtype": str(h_mid.dtype),
    }
    return gradients, values, logits.detach(), audit


def qwen_native_full_row_suffix_gradients(
    *,
    model: Any,
    model_name: str,
    layers: Sequence[Any],
    captures: Sequence[Mapping[str, torch.Tensor]],
    intervention_layer_index: int,
    prediction_position: int,
    replacements: torch.Tensor,
    final_norm: Any,
    output_embedding: Any,
    inputs: Mapping[str, Any],
    target_token_id: int,
    competitor_token_id: int,
    target_scalars: Sequence[str],
    compute_gradients: bool = True,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor, dict[str, Any]]:
    """Native suffix retaining the full-sequence GEMM numerical semantics.

    Only downstream decoder blocks are executed.  At each layer, prefix rows
    come from the clean capture and the intervened final query row is inserted
    into the same full sequence shape used by the original model.  This keeps
    BF16 GEMM accumulation order aligned with a full multimodal replay while
    skipping the vision tower and all upstream decoder layers.
    """
    family = _family(model_name)
    current_capture = captures[intervention_layer_index]
    current_h_mid = current_capture["h_mid"][0, : prediction_position + 1]
    if prediction_position != int(current_h_mid.shape[0]) - 1:
        raise ValueError("Qwen full-row suffix requires the final causal query row")
    leaf = replacements.detach().to(current_h_mid.dtype)
    if leaf.ndim == 1:
        leaf = leaf.unsqueeze(0)
    if compute_gradients:
        leaf.requires_grad_(True)
    batch_size = int(leaf.shape[0])
    current_prefix = (
        current_h_mid[:prediction_position]
        + current_capture["o_ffn"][0, :prediction_position].to(
            device=current_h_mid.device, dtype=current_h_mid.dtype
        )
    )
    query = current_h_mid[prediction_position].unsqueeze(0) + leaf
    hidden = torch.cat(
        (
            current_prefix.unsqueeze(0).expand(batch_size, -1, -1),
            query.unsqueeze(1),
        ),
        dim=1,
    )
    cos, sin = _position_embeddings_fp32(
        model=model,
        inputs=inputs,
        reference_hidden=current_h_mid,
    )
    sequence_length = prediction_position + 1
    mask = _causal_mask(
        sequence_length, current_h_mid.device, dtype=current_h_mid.dtype
    )
    gradient_context = torch.enable_grad() if compute_gradients else torch.no_grad()
    with gradient_context:
        for downstream_index in range(intervention_layer_index + 1, len(layers)):
            layer = layers[downstream_index]
            if family == "qwen2":
                hidden = layer(
                    hidden,
                    attention_mask=mask,
                    position_embeddings=(cos, sin),
                    output_attentions=False,
                    use_cache=False,
                )[0]
            else:
                hidden = layer(
                    hidden,
                    attention_mask=mask,
                    position_embeddings=(cos, sin),
                    use_cache=False,
                )
        query = hidden[:, prediction_position]
        logits = output_embedding(final_norm(query)).float()
        scalar_tensors = {
            scalar: torch.stack(
                [
                    target_scalar_from_logits(
                        row,
                        target_token_id=target_token_id,
                        competitor_token_id=competitor_token_id,
                        scalar=scalar,
                    )
                    for row in logits
                ]
            )
            for scalar in target_scalars
        }
        gradients = (
            batched_scalar_gradients(scalar_tensors, leaf)
            if compute_gradients
            else {}
        )
        values = {
            scalar: scalar_values.detach()
            for scalar, scalar_values in scalar_tensors.items()
        }
    audit = {
        "route": "qwen_exact_full_row_causal_suffix",
        "downstream_layer_count": len(layers) - intervention_layer_index - 1,
        "batch_size": batch_size,
        "native_dtype": str(current_h_mid.dtype),
        "compute_gradients": bool(compute_gradients),
    }
    return gradients, values, logits.detach(), audit


def qwen_native_full_row_suffix_at_replacement(**kwargs):
    replacement = kwargs.pop("replacement")
    gradients, values, logits, audit = qwen_native_full_row_suffix_gradients(
        replacements=replacement.unsqueeze(0), **kwargs
    )
    return (
        {name: value[0] for name, value in gradients.items()},
        {name: float(value[0]) for name, value in values.items()},
        logits[0],
        audit,
    )


def qwen_fp32_suffix_logits_and_gradients(
    *,
    model: Any,
    model_name: str,
    layers: Sequence[Any],
    intervention_layer_index: int,
    h_mid: torch.Tensor,
    prediction_position: int,
    branch_outputs: torch.Tensor,
    current_ffn_norm: Any,
    current_ffn: Any,
    final_norm: Any,
    lm_weight: torch.Tensor,
    lm_bias: torch.Tensor | None,
    inputs: Mapping[str, Any],
    target_token_id: int,
    competitor_token_id: int,
    target_scalars: Sequence[str],
    compute_gradients: bool = True,
    query_batch_size: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
    """Evaluate a branch-output batch and exact scalar gradients in FP32.

    All context rows are recomputed through every downstream decoder layer in
    FP32.  The intervened query row shares those causal states, so its batched
    execution is algebraically identical to full-sequence replay without
    repeating the invariant prefix for each intervention.
    """
    family = _family(model_name)
    if prediction_position != int(h_mid.shape[0]) - 1:
        raise ValueError("Qwen FP32 suffix currently requires the final causal query row")
    if branch_outputs.dtype != torch.float32 or h_mid.dtype != torch.float32:
        raise TypeError("Qwen FP32 suffix inputs must already be float32")
    device = h_mid.device
    cos, sin = _position_embeddings_fp32(
        model=model,
        inputs=inputs,
        reference_hidden=h_mid,
    )
    cos = cos.float()
    sin = sin.float()

    with torch.no_grad():
        prefix_mid = h_mid[:prediction_position]
        context = prefix_mid + current_ffn(current_ffn_norm(prefix_mid))
        queries = h_mid[prediction_position].unsqueeze(0) + branch_outputs
        contexts: list[torch.Tensor] = []
        base_query_inputs: list[torch.Tensor] = []
        for layer in layers[intervention_layer_index + 1 :]:
            if compute_gradients:
                contexts.append(context.detach())
                base_query_inputs.append(queries[0].detach())
            layer32 = copy.deepcopy(layer).to(device=device, dtype=torch.float32).eval()
            queries = qwen_query_layer_fp32(
                layer=layer32,
                clean_context=context,
                query_states=queries,
                cos=cos,
                sin=sin,
                family=family,
                query_batch_size=query_batch_size,
            )
            context = _full_context_layer_fp32(
                layer=layer32,
                hidden_states=context,
                cos=cos,
                sin=sin,
                family=family,
            )
            del layer32
        logits = F.linear(final_norm(queries), lm_weight, lm_bias)

    if not compute_gradients:
        audit = {
            "route": "qwen_causal_query_suffix_true_fp32",
            "downstream_layer_count": len(layers) - intervention_layer_index - 1,
            "context_recomputed_fp32": True,
            "query_batch_size": int(branch_outputs.shape[0]),
            "query_microbatch_size": min(
                int(branch_outputs.shape[0]),
                int(query_batch_size or branch_outputs.shape[0]),
            ),
        }
        return logits.detach(), {}, audit

    final_leaf = queries[0].detach().requires_grad_(True)
    final_logits = F.linear(final_norm(final_leaf), lm_weight, lm_bias)
    incoming: dict[str, torch.Tensor] = {}
    scalar_names = list(target_scalars)
    for offset, scalar in enumerate(scalar_names):
        value = target_scalar_from_logits(
            final_logits,
            target_token_id=target_token_id,
            competitor_token_id=competitor_token_id,
            scalar=scalar,
        )
        incoming[scalar] = torch.autograd.grad(
            value,
            final_leaf,
            retain_graph=offset + 1 < len(scalar_names),
        )[0].detach()

    downstream_layers = list(layers[intervention_layer_index + 1 :])
    for layer, context, base_query in reversed(
        list(zip(downstream_layers, contexts, base_query_inputs))
    ):
        layer32 = copy.deepcopy(layer).to(device=device, dtype=torch.float32).eval()
        leaf = base_query.detach().requires_grad_(True)
        output = qwen_query_layer_fp32(
            layer=layer32,
            clean_context=context,
            query_states=leaf.unsqueeze(0),
            cos=cos,
            sin=sin,
            family=family,
            query_batch_size=query_batch_size,
        )[0]
        next_incoming: dict[str, torch.Tensor] = {}
        for offset, scalar in enumerate(scalar_names):
            next_incoming[scalar] = torch.autograd.grad(
                output,
                leaf,
                grad_outputs=incoming[scalar],
                retain_graph=offset + 1 < len(scalar_names),
            )[0].detach()
        incoming = next_incoming
        del layer32, output, leaf

    audit = {
        "route": "qwen_causal_query_suffix_true_fp32",
        "downstream_layer_count": len(downstream_layers),
        "context_recomputed_fp32": True,
        "query_batch_size": int(branch_outputs.shape[0]),
        "query_microbatch_size": min(
            int(branch_outputs.shape[0]),
            int(query_batch_size or branch_outputs.shape[0]),
        ),
    }
    return logits.detach(), incoming, audit
