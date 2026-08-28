"""Exact visual-token attention contributions and FFN Jacobian responses.

The production path keeps two operations separate: reconstruct the residual
write attributable to every visual key, then apply the local FFN Jacobian to
all directions with ``vmap(jvp)``.  Llama/Qwen separate-QKV layers and
InternVL's InternLM2 packed-QKV layers are supported.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from models.dgst_capture import attention_row_from_capture, resolve_decoder_layers


JFFN_SOURCE_MODE = "jffn"
JFFN_ENTROPY_MATCHED_SOURCE_MODE = "jffn_entropy_matched"
JFFN_SOURCE_MODES = (JFFN_SOURCE_MODE, JFFN_ENTROPY_MATCHED_SOURCE_MODE)


@dataclass(frozen=True)
class DecoderLayerAdapter:
    family: str
    attention: Any
    attention_norm: Any
    ffn_norm: Any
    ffn: Any
    output_projection: Any
    value_projection: Any
    num_attention_heads: int
    num_key_value_heads: int
    num_key_value_groups: int
    head_dim: int


def resolve_decoder_layer_adapter(layer: Any) -> DecoderLayerAdapter:
    """Resolve one supported decoder layer without relying on class names."""
    if hasattr(layer, "self_attn"):
        attention = layer.self_attn
        attention_norm = getattr(layer, "input_layernorm", None)
        ffn_norm = getattr(layer, "post_attention_layernorm", None)
        ffn = getattr(layer, "mlp", None)
        value_projection = getattr(attention, "v_proj", None)
        output_projection = getattr(attention, "o_proj", None)
        family = "separate_qkv"
    elif hasattr(layer, "attention"):
        attention = layer.attention
        attention_norm = getattr(layer, "attention_norm", None)
        ffn_norm = getattr(layer, "ffn_norm", None)
        ffn = getattr(layer, "feed_forward", None)
        value_projection = getattr(attention, "wqkv", None)
        output_projection = getattr(attention, "wo", None)
        family = "internlm2_packed_qkv"
    else:
        raise TypeError(
            "Unsupported decoder layer: expected self_attn or attention, got "
            f"{type(layer).__name__}."
        )

    missing = [
        name
        for name, value in (
            ("attention_norm", attention_norm),
            ("ffn_norm", ffn_norm),
            ("ffn", ffn),
            ("value_projection", value_projection),
            ("output_projection", output_projection),
        )
        if value is None
    ]
    if missing:
        raise TypeError(
            f"Unsupported {type(layer).__name__}; missing modules {missing}."
        )

    config = getattr(attention, "config", None)
    num_attention_heads = int(
        getattr(attention, "num_heads", getattr(config, "num_attention_heads", 0))
    )
    num_key_value_heads = int(
        getattr(
            attention,
            "num_key_value_heads",
            getattr(config, "num_key_value_heads", 0),
        )
    )
    head_dim = int(
        getattr(
            attention,
            "head_dim",
            (
                int(getattr(config, "hidden_size", 0)) // num_attention_heads
                if num_attention_heads
                else 0
            ),
        )
    )
    if num_attention_heads <= 0 or num_key_value_heads <= 0 or head_dim <= 0:
        raise ValueError(
            "Invalid attention dimensions: "
            f"H={num_attention_heads}, Hkv={num_key_value_heads}, Dh={head_dim}."
        )
    groups = int(
        getattr(
            attention,
            "num_key_value_groups",
            num_attention_heads // num_key_value_heads,
        )
    )
    if num_key_value_heads * groups != num_attention_heads:
        raise ValueError(
            "GQA dimensions do not tile query heads: "
            f"{num_key_value_heads} * {groups} != {num_attention_heads}."
        )
    return DecoderLayerAdapter(
        family=family,
        attention=attention,
        attention_norm=attention_norm,
        ffn_norm=ffn_norm,
        ffn=ffn,
        output_projection=output_projection,
        value_projection=value_projection,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        num_key_value_groups=groups,
        head_dim=head_dim,
    )


def ffn_from_residual(layer: Any, value: torch.Tensor) -> torch.Tensor:
    """Evaluate ``FFN(Norm(value))`` for either supported layer layout."""
    if hasattr(layer, "post_attention_layernorm") and hasattr(layer, "mlp"):
        return layer.mlp(layer.post_attention_layernorm(value))
    if hasattr(layer, "ffn_norm") and hasattr(layer, "feed_forward"):
        return layer.feed_forward(layer.ffn_norm(value))
    raise TypeError(
        "Unsupported FFN layer: expected post_attention_layernorm+mlp or "
        f"ffn_norm+feed_forward, got {type(layer).__name__}."
    )


def _project_values(
    adapter: DecoderLayerAdapter,
    normalized_states: torch.Tensor,
) -> torch.Tensor:
    """Return repeated value states as ``[H,S,Dh]``."""
    sequence_length = int(normalized_states.shape[0])
    projected = adapter.value_projection(normalized_states)
    if adapter.family == "internlm2_packed_qkv":
        packed_groups = 2 + adapter.num_key_value_groups
        expected = adapter.num_key_value_heads * packed_groups * adapter.head_dim
        if int(projected.shape[-1]) != expected:
            raise ValueError(
                "Unexpected InternLM2 packed wqkv width: "
                f"{projected.shape[-1]} != {expected}."
            )
        values = projected.view(
            sequence_length,
            adapter.num_key_value_heads,
            packed_groups,
            adapter.head_dim,
        )[:, :, -1, :].permute(1, 0, 2)
    else:
        expected = adapter.num_key_value_heads * adapter.head_dim
        if int(projected.shape[-1]) != expected:
            raise ValueError(
                f"Unexpected v_proj width: {projected.shape[-1]} != {expected}."
            )
        values = projected.view(
            sequence_length,
            adapter.num_key_value_heads,
            adapter.head_dim,
        ).permute(1, 0, 2)
    if adapter.num_key_value_groups > 1:
        values = values.repeat_interleave(adapter.num_key_value_groups, dim=0)
    return values


def reconstruct_visual_directions(
    *,
    layer: Any,
    capture: dict[str, Any],
    prediction_positions: Sequence[int],
    visual_start: int,
    visual_end: int,
) -> dict[str, torch.Tensor | float | str]:
    """Reconstruct per-visual-token residual writes as ``a_tokens[M,T,D]``.

    The output projection bias is included in complete-attention reconstruction
    but excluded from every individual visual-token contribution.
    """
    positions = [int(value) for value in prediction_positions]
    if not positions:
        raise ValueError("prediction_positions must not be empty")
    h_prev = capture["h_prev"][0]
    h_mid = capture["h_mid"][0]
    sequence_length = int(h_prev.shape[0])
    if not 0 <= int(visual_start) < int(visual_end) <= sequence_length:
        raise ValueError(
            f"Invalid visual range [{visual_start}, {visual_end}) for sequence "
            f"length {sequence_length}."
        )
    if any(position < 0 or position >= sequence_length for position in positions):
        raise ValueError("A prediction position is outside the captured sequence")

    adapter = resolve_decoder_layer_adapter(layer)
    model_device = adapter.output_projection.weight.device
    model_dtype = adapter.output_projection.weight.dtype
    h_prev = h_prev.to(device=model_device, dtype=model_dtype)
    h_mid = h_mid.to(device=model_device, dtype=model_dtype)

    with torch.no_grad():
        normalized = adapter.attention_norm(h_prev)
        values = _project_values(adapter, normalized)
        attention = torch.stack(
            [
                attention_row_from_capture(capture, position).to(
                    device=model_device, dtype=model_dtype
                )
                for position in positions
            ],
            dim=0,
        )
        if int(attention.shape[1]) != int(values.shape[0]):
            raise ValueError(
                "Attention/value head mismatch: "
                f"{tuple(attention.shape)} vs {tuple(values.shape)}"
            )

        all_heads = torch.einsum("ths,hsd->thd", attention, values)
        reconstructed = F.linear(
            all_heads.reshape(len(positions), -1),
            adapter.output_projection.weight,
            adapter.output_projection.bias,
        )
        position_index = torch.tensor(
            positions, dtype=torch.long, device=model_device
        )
        captured_attention_update = capture.get("o_attn")
        if captured_attention_update is not None:
            actual = captured_attention_update[0].to(
                device=model_device, dtype=model_dtype
            ).index_select(0, position_index)
        else:
            # Production drops o_attn to save memory.  h_mid-h_prev is
            # algebraically identical, but BF16 cancellation makes it a less
            # precise smoke-test reference, so validation retains o_attn.
            actual = h_mid.index_select(0, position_index) - h_prev.index_select(
                0, position_index
            )

        # Project every source token separately once.  The visual slice is the
        # formal a_j payload; summing the complete set plus exactly one output
        # bias gives a real component-decomposition diagnostic rather than a
        # hard-coded zero.
        pre_projection_all_tokens = (
            attention.permute(2, 0, 1).unsqueeze(-1)
            * values.permute(1, 0, 2).unsqueeze(1)
        ).reshape(sequence_length, len(positions), -1)
        all_token_writes = F.linear(
            pre_projection_all_tokens,
            adapter.output_projection.weight,
            bias=None,
        )
        a_tokens = all_token_writes[
            int(visual_start) : int(visual_end)
        ].contiguous()
        a_visual = a_tokens.sum(dim=0)
        component_sum = all_token_writes.sum(dim=0)
        if adapter.output_projection.bias is not None:
            component_sum = component_sum + adapter.output_projection.bias
        visual_attention_scores = attention[
            :, :, int(visual_start) : int(visual_end)
        ].float().sum(dim=1)

    reconstructed_float = reconstructed.float()
    actual_float = actual.float()
    reconstruction_relative_error = float(
        (reconstructed_float - actual_float).norm()
        / actual_float.norm().clamp_min(1e-12)
    )
    reconstruction_cosine = float(
        F.cosine_similarity(
            reconstructed_float.reshape(1, -1),
            actual_float.reshape(1, -1),
            dim=-1,
        ).item()
    )
    component_sum_relative_error = float(
        (component_sum.float() - actual_float).norm()
        / actual_float.norm().clamp_min(1e-12)
    )
    return {
        "adapter_family": adapter.family,
        "a_tokens": a_tokens.detach(),
        "a_visual": a_visual.detach(),
        "visual_attention_scores": visual_attention_scores.detach(),
        "reconstruction_relative_error": reconstruction_relative_error,
        "reconstruction_cosine": reconstruction_cosine,
        "component_sum_relative_error": component_sum_relative_error,
    }


def reconstruct_llama_visual_directions(**kwargs: Any) -> dict[str, Any]:
    """Backward-compatible alias used by the original LLaVA benchmark."""
    result = reconstruct_visual_directions(**kwargs)
    if result["adapter_family"] != "separate_qkv":
        raise TypeError("Expected a Llama-style separate-QKV decoder layer")
    return result


def exact_visual_token_jvps(
    *,
    layer: Any,
    z: torch.Tensor,
    a_tokens: torch.Tensor,
    chunk_size: int | None,
    synchronize: bool = True,
    measure_time: bool = True,
) -> tuple[torch.Tensor, float]:
    """Return exact ``J_f(z) a_j`` for every leading visual direction."""
    if a_tokens.ndim != z.ndim + 1 or tuple(a_tokens.shape[1:]) != tuple(z.shape):
        raise ValueError(
            f"Expected directions [M,{','.join(map(str, z.shape))}], got "
            f"{tuple(a_tokens.shape)}"
        )
    if chunk_size is not None and int(chunk_size) <= 0:
        raise ValueError("chunk_size must be positive or None")

    def one_direction(direction: torch.Tensor) -> torch.Tensor:
        return torch.func.jvp(
            lambda value: ffn_from_residual(layer, value),
            (z,),
            (direction,),
        )[1]

    should_sync = bool(synchronize and z.is_cuda)
    if should_sync:
        torch.cuda.synchronize(z.device)
    start = time.perf_counter() if measure_time else 0.0
    with torch.enable_grad():
        responses = torch.vmap(one_direction, chunk_size=chunk_size)(a_tokens)
    if should_sync:
        torch.cuda.synchronize(z.device)
    elapsed = float(time.perf_counter() - start) if measure_time else 0.0
    return responses.detach(), elapsed


def estimate_jvp_increment_bytes(
    *,
    layer: Any,
    num_directions: int,
    num_targets: int,
    dtype: torch.dtype,
    workspace_factor: float = 3.0,
) -> int:
    """Conservative estimate used only to choose a vmap chunk."""
    adapter = resolve_decoder_layer_adapter(layer)
    hidden = int(adapter.output_projection.weight.shape[0])
    intermediate = int(
        getattr(adapter.ffn, "intermediate_size", 0)
        or getattr(getattr(adapter.ffn, "gate_proj", None), "out_features", 0)
        or getattr(getattr(adapter.ffn, "w1", None), "out_features", 0)
        or hidden * 4
    )
    element_size = torch.empty((), dtype=dtype).element_size()
    elements = 4 * intermediate + 4 * hidden
    return int(
        max(float(workspace_factor), 1.0)
        * int(num_directions)
        * int(num_targets)
        * elements
        * element_size
    )


def choose_adaptive_jvp_chunk_size(
    *,
    layer: Any,
    num_visual_tokens: int,
    num_targets: int,
    dtype: torch.dtype,
    max_increment_bytes: int = 1536 * 1024**2,
    fallback_chunks: Sequence[int] = (256, 128, 64),
) -> int | None:
    """Return None for full parallelism, otherwise a bounded chunk size."""
    count = int(num_visual_tokens)
    if count <= 0 or int(num_targets) <= 0:
        raise ValueError("num_visual_tokens and num_targets must be positive")
    if estimate_jvp_increment_bytes(
        layer=layer,
        num_directions=count,
        num_targets=int(num_targets),
        dtype=dtype,
    ) <= int(max_increment_bytes):
        return None
    for value in fallback_chunks:
        candidate = min(int(value), count)
        if candidate <= 0:
            continue
        if estimate_jvp_increment_bytes(
            layer=layer,
            num_directions=candidate,
            num_targets=int(num_targets),
            dtype=dtype,
        ) <= int(max_increment_bytes):
            return candidate
    return min(64, count)


def _retry_chunk_sequence(
    initial_chunk_size: int | None,
    num_visual_tokens: int,
) -> tuple[int | None, ...]:
    candidates: list[int | None] = [initial_chunk_size]
    start = num_visual_tokens if initial_chunk_size is None else int(initial_chunk_size)
    for value in (256, 128, 64):
        candidate = min(value, int(num_visual_tokens))
        if candidate < start and candidate not in candidates:
            candidates.append(candidate)
    return tuple(candidates)


def exact_visual_token_jvps_with_oom_fallback(
    *,
    layer: Any,
    z: torch.Tensor,
    a_tokens: torch.Tensor,
    chunk_size: int | None,
) -> tuple[torch.Tensor, int | None]:
    """Run production JVP and retry 256/128/64 chunks after CUDA OOM."""
    failures: list[str] = []
    for candidate in _retry_chunk_sequence(chunk_size, int(a_tokens.shape[0])):
        try:
            responses, _ = exact_visual_token_jvps(
                layer=layer,
                z=z,
                a_tokens=a_tokens,
                chunk_size=candidate,
                synchronize=False,
                measure_time=False,
            )
            return responses, candidate
        except torch.OutOfMemoryError as exc:
            failures.append("all" if candidate is None else str(candidate))
            del exc
            if z.is_cuda:
                torch.cuda.empty_cache()
    raise torch.OutOfMemoryError(
        "JFFN vmap exhausted chunk candidates " + ",".join(failures)
    )


def _renormalize_energy(energy: torch.Tensor, eps: float) -> torch.Tensor:
    finite = torch.nan_to_num(energy.float(), nan=0.0, posinf=0.0, neginf=0.0)
    finite = finite.clamp_min(0.0)
    denominator = finite.sum(dim=-1, keepdim=True)
    uniform = torch.full_like(finite, 1.0 / max(int(finite.shape[-1]), 1))
    return torch.where(
        denominator > float(eps),
        finite / denominator.clamp_min(float(eps)),
        uniform,
    )


def entropy_matched_jffn_distribution(
    energy: torch.Tensor,
    beta: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Apply frozen layer temperature to log-energy without changing ranks."""
    beta_value = float(beta)
    if not math.isfinite(beta_value) or beta_value <= 0.0:
        raise ValueError(f"JFFN entropy beta must be finite and positive, got {beta}")
    return torch.softmax(
        torch.log(energy.float().clamp_min(float(eps))) / beta_value,
        dim=-1,
    )


def distribution_statistics(distribution: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return row-wise P diagnostics for a ``[T,M]`` distribution."""
    if distribution.ndim != 2:
        raise ValueError("distribution_statistics expects [T,M]")
    values = _renormalize_energy(distribution, 1e-12)
    count = int(values.shape[-1])
    safe = values.clamp_min(1e-30)
    natural_entropy = -(safe * safe.log()).sum(dim=-1)
    sorted_values = torch.sort(values, dim=-1).values
    ranks = torch.arange(
        1, count + 1, dtype=values.dtype, device=values.device
    )
    result = {
        "normalized_entropy": natural_entropy / math.log(max(count, 2)),
        "effective_token_count": torch.exp(natural_entropy),
        "max_over_uniform": values.max(dim=-1).values * count,
        "gini": (
            2.0 * (sorted_values * ranks).sum(dim=-1) / count
            - (count + 1.0) / count
        ),
    }
    for top_k in (1, 5, 10, 32):
        result[f"top{top_k}_mass"] = torch.topk(
            values, k=min(top_k, count), dim=-1
        ).values.sum(dim=-1)
    return result


def union_topk_aggregate_diagnostics(
    *,
    a_tokens: torch.Tensor,
    responses: torch.Tensor,
    source_distribution: torch.Tensor,
    target_distribution: torch.Tensor,
    side_top_k: int = 32,
    eps: float = 1e-12,
) -> dict[str, torch.Tensor]:
    """Aggregate exact visual directions inside source/target Union Top-K.

    ``a_tokens`` and ``responses`` are ``[M,T,D]`` tensors.  Unlike the
    tokenwise ``sum(||delta_j||) / sum(||a_j||)`` diagnostic, this function
    sums the selected vectors first and therefore preserves constructive and
    destructive interference between visual-token directions.
    """
    if a_tokens.ndim != 3 or responses.shape != a_tokens.shape:
        raise ValueError(
            "a_tokens/responses must have matching [visual,target,hidden] shapes"
        )
    source = torch.as_tensor(
        source_distribution, device=a_tokens.device, dtype=torch.float32
    )
    target = torch.as_tensor(
        target_distribution, device=a_tokens.device, dtype=torch.float32
    )
    expected = (int(a_tokens.shape[1]), int(a_tokens.shape[0]))
    if tuple(source.shape) != expected or tuple(target.shape) != expected:
        raise ValueError(
            f"Union distributions must have shape {expected}, got "
            f"{tuple(source.shape)}/{tuple(target.shape)}"
        )
    source = torch.nan_to_num(source, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    source = source / source.sum(dim=-1, keepdim=True).clamp_min(float(eps))
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(float(eps))
    k = min(max(int(side_top_k), 1), int(a_tokens.shape[0]))
    source_index = torch.argsort(
        source, dim=-1, descending=True, stable=True
    )[:, :k]
    target_index = torch.argsort(
        target, dim=-1, descending=True, stable=True
    )[:, :k]
    mask = torch.zeros(expected, device=a_tokens.device, dtype=torch.bool)
    mask.scatter_(1, source_index, True)
    mask.scatter_(1, target_index, True)
    selected = mask.transpose(0, 1).unsqueeze(-1)
    directions32 = a_tokens.float()
    responses32 = responses.float()
    aggregate_input = torch.where(selected, directions32, 0.0).sum(dim=0)
    aggregate_response = torch.where(selected, responses32, 0.0).sum(dim=0)
    input_norm = aggregate_input.norm(dim=-1)
    response_norm = aggregate_response.norm(dim=-1)
    token_input_norm_sum = torch.where(
        mask, directions32.norm(dim=-1).transpose(0, 1), 0.0
    ).sum(dim=-1)
    token_response_norm_sum = torch.where(
        mask, responses32.norm(dim=-1).transpose(0, 1), 0.0
    ).sum(dim=-1)
    return {
        "input_norm": input_norm.detach(),
        "response_norm": response_norm.detach(),
        "gain": (response_norm / input_norm.clamp_min(float(eps))).detach(),
        "direction_cosine": F.cosine_similarity(
            aggregate_response, aggregate_input, dim=-1
        ).detach(),
        "input_cancellation_ratio": (
            input_norm / token_input_norm_sum.clamp_min(float(eps))
        ).detach(),
        "response_cancellation_ratio": (
            response_norm / token_response_norm_sum.clamp_min(float(eps))
        ).detach(),
        "union_size": mask.sum(dim=-1).detach(),
    }


def build_jffn_source_payload(
    *,
    model: Any,
    captures: Sequence[dict[str, Any]],
    prediction_positions: Sequence[int],
    visual_start: int,
    visual_end: int,
    entropy_beta_by_layer: Sequence[float] | None = None,
    max_increment_bytes: int = 1536 * 1024**2,
    fixed_chunk_size: int | None = None,
    force_full_parallel: bool = False,
    validate_linearity: bool = False,
    finite_difference_eta: float | None = None,
    parity_chunk_size: int | None = None,
    include_second_round_diagnostics: bool = False,
    union_target_distributions: Any | None = None,
    union_side_top_k: int = 32,
    eps: float = 1e-12,
) -> list[dict[str, Any]]:
    """Compute all JFFN source matrices once before the inference-only OT path."""
    layers = resolve_decoder_layers(model)
    if len(layers) != len(captures):
        raise ValueError(
            f"Decoder layer/capture mismatch: {len(layers)} vs {len(captures)}"
        )
    if entropy_beta_by_layer is not None and len(entropy_beta_by_layer) != len(layers):
        raise ValueError(
            "entropy_beta_by_layer must contain exactly one value per decoder layer"
        )
    union_targets = None
    if union_target_distributions is not None:
        if not include_second_round_diagnostics:
            raise ValueError(
                "union_target_distributions requires second-round diagnostics"
            )
        union_targets = torch.as_tensor(union_target_distributions)
        expected_prefix = (len(prediction_positions), len(layers))
        if union_targets.ndim != 3 or tuple(union_targets.shape[:2]) != expected_prefix:
            raise ValueError(
                "union_target_distributions must have shape "
                f"[targets,layers,visual], got {tuple(union_targets.shape)}"
            )
    positions = [int(value) for value in prediction_positions]
    payload: list[dict[str, Any]] = []
    for layer_index, (layer, capture) in enumerate(zip(layers, captures)):
        directions = reconstruct_visual_directions(
            layer=layer,
            capture=capture,
            prediction_positions=positions,
            visual_start=int(visual_start),
            visual_end=int(visual_end),
        )
        adapter = resolve_decoder_layer_adapter(layer)
        position_index = torch.tensor(
            positions,
            dtype=torch.long,
            device=adapter.output_projection.weight.device,
        )
        z = capture["h_mid"][0].to(
            device=adapter.output_projection.weight.device,
            dtype=adapter.output_projection.weight.dtype,
        ).index_select(0, position_index)
        a_tokens = directions["a_tokens"]
        requested_chunk = fixed_chunk_size
        if requested_chunk is None and not force_full_parallel:
            requested_chunk = choose_adaptive_jvp_chunk_size(
                layer=layer,
                num_visual_tokens=int(a_tokens.shape[0]),
                num_targets=len(positions),
                dtype=a_tokens.dtype,
                max_increment_bytes=int(max_increment_bytes),
            )
        responses, used_chunk = exact_visual_token_jvps_with_oom_fallback(
            layer=layer,
            z=z,
            a_tokens=a_tokens,
            chunk_size=requested_chunk,
        )
        energy = responses.float().norm(dim=-1).transpose(0, 1).contiguous()
        distribution = _renormalize_energy(energy, eps)
        distributions: dict[str, torch.Tensor] = {JFFN_SOURCE_MODE: distribution}
        if entropy_beta_by_layer is not None:
            distributions[JFFN_ENTROPY_MATCHED_SOURCE_MODE] = (
                entropy_matched_jffn_distribution(
                    energy,
                    beta=float(entropy_beta_by_layer[layer_index]),
                    eps=float(eps),
                )
            )

        a_visual = directions["a_visual"].float()
        response_visual = responses.float().sum(dim=0)
        response_sum_relative_error = torch.zeros(
            len(positions), device=response_visual.device, dtype=torch.float32
        )
        # Validation-only values stay finite in production shards; the
        # explicit flag below distinguishes "not computed" from a true zero.
        local_fd_cosine = torch.zeros_like(response_sum_relative_error)
        local_fd_relative_error = torch.zeros_like(response_sum_relative_error)
        if validate_linearity or finite_difference_eta is not None:
            # BF16 GEMMs legitimately choose different kernels for full-vmap
            # and chunked-vmap, and a tiny BF16 finite difference is dominated
            # by quantization.  Smoke validation therefore uses a temporary
            # FP32 copy of this layer's Norm+FFN.  The formal source E/P above
            # remains the model-dtype torch.func.jvp used by the experiment.
            validation = validate_jffn_numerics_float32(
                layer=layer,
                z=z,
                a_tokens=a_tokens,
                finite_difference_eta=finite_difference_eta,
                parity_chunk_size=parity_chunk_size,
                eps=float(eps),
            )
            aggregate_response = validation["aggregate_response"]
            validation_responses = validation["responses"]
            validation_sum = validation_responses.sum(dim=0)
            response_sum_relative_error = (
                (validation_sum - aggregate_response).norm(dim=-1)
                / aggregate_response.norm(dim=-1).clamp_min(float(eps))
            )
            if finite_difference_eta is not None:
                central = validation["finite_difference"]
                local_fd_cosine = F.cosine_similarity(
                    central, aggregate_response, dim=-1
                )
                local_fd_relative_error = (
                    (central - aggregate_response).norm(dim=-1)
                    / central.norm(dim=-1).clamp_min(float(eps))
                )
            parity_relative_error = float(validation["parallel_chunk_relative_error"])
            del validation_responses, validation
        else:
            parity_relative_error = 0.0
        if parity_chunk_size is not None and not (
            validate_linearity or finite_difference_eta is not None
        ):
            parity_chunk = min(int(parity_chunk_size), int(a_tokens.shape[0]))
            compared, _ = exact_visual_token_jvps(
                layer=layer,
                z=z,
                a_tokens=a_tokens,
                chunk_size=parity_chunk,
                synchronize=False,
                measure_time=False,
            )
            parity_relative_error = float(
                (compared.float() - responses.float()).norm()
                / responses.float().norm().clamp_min(float(eps))
            )
            del compared
        input_norm = a_visual.norm(dim=-1)
        response_norm = response_visual.norm(dim=-1)
        layer_payload: dict[str, Any] = {
                "distributions": {
                    key: value.detach() for key, value in distributions.items()
                },
                "energy": energy.detach(),
                "input_norm": input_norm.detach(),
                "response_norm": response_norm.detach(),
                "gain": (
                    response_norm / input_norm.clamp_min(float(eps))
                ).detach(),
                "direction_cosine": F.cosine_similarity(
                    response_visual, a_visual, dim=-1
                ).detach(),
                "response_sum_relative_error": response_sum_relative_error.detach(),
                "local_fd_cosine": local_fd_cosine.detach(),
                "local_fd_relative_error": local_fd_relative_error.detach(),
                "parallel_chunk_relative_error": parity_relative_error,
                "validation_computed": bool(
                    validate_linearity
                    or finite_difference_eta is not None
                    or parity_chunk_size is not None
                ),
                "reconstruction_relative_error": float(
                    directions["reconstruction_relative_error"]
                ),
                "reconstruction_cosine": float(
                    directions["reconstruction_cosine"]
                ),
                "component_sum_relative_error": float(
                    directions["component_sum_relative_error"]
                ),
                "jvp_chunk_size": (
                    int(a_tokens.shape[0])
                    if used_chunk is None
                    else int(used_chunk)
                ),
                "jvp_full_parallel": bool(used_chunk is None),
                "distribution_statistics": {
                    source: distribution_statistics(value)
                    for source, value in distributions.items()
                },
            }
        if include_second_round_diagnostics:
            write_energy = a_tokens.float().norm(dim=-1).transpose(0, 1).contiguous()
            response_unit = response_visual / response_norm.unsqueeze(-1).clamp_min(
                float(eps)
            )
            signed_q = torch.einsum(
                "mtd,td->tm", responses.float(), response_unit
            ).contiguous()
            q_sum = signed_q.sum(dim=-1)
            layer_payload["second_round"] = {
                "attention_distribution": _renormalize_energy(
                    directions["visual_attention_scores"], float(eps)
                ).detach(),
                "write_energy": write_energy.detach(),
                "write_distribution": _renormalize_energy(
                    write_energy, float(eps)
                ).detach(),
                "signed_q": signed_q.detach(),
                "signed_q_conservation_relative_error": (
                    (q_sum - response_norm).abs()
                    / response_norm.clamp_min(float(eps))
                ).detach(),
                "cancellation_ratio": (
                    response_norm / energy.sum(dim=-1).clamp_min(float(eps))
                ).detach(),
            }
            if union_targets is not None:
                layer_target = union_targets[:, layer_index, :]
                union_values = union_topk_aggregate_diagnostics(
                    a_tokens=a_tokens,
                    responses=responses,
                    source_distribution=distribution,
                    target_distribution=layer_target,
                    side_top_k=int(union_side_top_k),
                    eps=float(eps),
                )
                layer_payload["second_round"]["union_aggregate"] = union_values
        payload.append(layer_payload)
        del responses, a_tokens, directions, z
    return payload


def validate_jffn_numerics_float32(
    *,
    layer: Any,
    z: torch.Tensor,
    a_tokens: torch.Tensor,
    finite_difference_eta: float | None,
    parity_chunk_size: int | None,
    eps: float,
) -> dict[str, torch.Tensor | float]:
    """Numerically validate JVP identities without BF16 quantization noise."""
    adapter = resolve_decoder_layer_adapter(layer)
    norm = copy.deepcopy(adapter.ffn_norm).to(
        device=z.device, dtype=torch.float32
    )
    ffn = copy.deepcopy(adapter.ffn).to(device=z.device, dtype=torch.float32)
    z32 = z.float()
    directions32 = a_tokens.float()

    def function(value: torch.Tensor) -> torch.Tensor:
        return ffn(norm(value))

    def one_direction(direction: torch.Tensor) -> torch.Tensor:
        return torch.func.jvp(function, (z32,), (direction,))[1]

    with torch.enable_grad():
        responses = torch.vmap(one_direction, chunk_size=None)(directions32)
        aggregate = torch.func.jvp(
            function, (z32,), (directions32.sum(dim=0),)
        )[1]
        if parity_chunk_size is not None:
            chunked = torch.vmap(
                one_direction,
                chunk_size=min(int(parity_chunk_size), int(directions32.shape[0])),
            )(directions32)
            parity = float(
                (chunked - responses).norm()
                / responses.norm().clamp_min(float(eps))
            )
            del chunked
        else:
            parity = float("nan")

    finite = torch.full_like(aggregate, float("nan"))
    if finite_difference_eta is not None:
        eta = float(finite_difference_eta)
        if not math.isfinite(eta) or eta <= 0.0:
            raise ValueError("finite_difference_eta must be positive")
        visual_direction = directions32.sum(dim=0)
        with torch.no_grad():
            plus = function(z32 + eta * visual_direction)
            minus = function(z32 - eta * visual_direction)
        finite = (plus - minus) / (2.0 * eta)

    result = {
        "responses": responses.detach(),
        "aggregate_response": aggregate.detach(),
        "finite_difference": finite.detach(),
        "parallel_chunk_relative_error": parity,
    }
    del norm, ffn, z32, directions32
    return result


def aggregate_visual_jvp(
    *, layer: Any, z: torch.Tensor, a_visual: torch.Tensor
) -> torch.Tensor:
    """Compute the one-direction aggregate visual JVP."""
    with torch.enable_grad():
        return torch.func.jvp(
            lambda value: ffn_from_residual(layer, value),
            (z,),
            (a_visual,),
        )[1].detach()


def summarize_visual_jvps(
    *,
    layer: Any,
    z: torch.Tensor,
    a_visual: torch.Tensor,
    token_responses: torch.Tensor,
    finite_eta: float = 0.05,
) -> dict[str, torch.Tensor | float]:
    """Reduce exact responses and run expensive finite-difference validation."""
    aggregate_response = aggregate_visual_jvp(
        layer=layer, z=z, a_visual=a_visual
    )
    token_response_norm = token_responses.float().norm(dim=-1).transpose(0, 1)
    distribution = _renormalize_energy(token_response_norm, 1e-12)
    stats = distribution_statistics(distribution)
    summed_response = token_responses.float().sum(dim=0)
    aggregate_float = aggregate_response.float()
    response_sum_relative_error = (
        (summed_response - aggregate_float).norm(dim=-1)
        / aggregate_float.norm(dim=-1).clamp_min(1e-12)
    )
    input_norm = a_visual.float().norm(dim=-1)
    response_norm = aggregate_float.norm(dim=-1)
    gain = response_norm / input_norm.clamp_min(1e-12)
    direction_cosine = F.cosine_similarity(
        aggregate_float, a_visual.float(), dim=-1
    )

    eta = float(finite_eta)
    with torch.no_grad():
        plus = ffn_from_residual(layer, z + eta * a_visual)
        minus = ffn_from_residual(layer, z - eta * a_visual)
        center = ffn_from_residual(layer, z)
        without_visual = ffn_from_residual(layer, z - a_visual)
    central = (plus.float() - minus.float()) / (2.0 * eta)
    finite_delete = center.float() - without_visual.float()
    local_fd_relative_error = (
        (central - aggregate_float).norm(dim=-1)
        / central.norm(dim=-1).clamp_min(1e-12)
    )
    return {
        "token_response_norm": token_response_norm.detach().cpu(),
        "distribution": distribution.detach().cpu(),
        "input_norm": input_norm.detach().cpu(),
        "response_norm": response_norm.detach().cpu(),
        "gain": gain.detach().cpu(),
        "direction_cosine": direction_cosine.detach().cpu(),
        "normalized_entropy": stats["normalized_entropy"].detach().cpu(),
        "top32_mass": stats["top32_mass"].detach().cpu(),
        "max_over_uniform": stats["max_over_uniform"].detach().cpu(),
        "response_sum_relative_error": response_sum_relative_error.detach().cpu(),
        "local_fd_relative_error": (
            (central - aggregate_float).norm(dim=-1)
            / central.norm(dim=-1).clamp_min(1e-12)
        ).detach().cpu(),
        "local_fd_cosine": F.cosine_similarity(
            central, aggregate_float, dim=-1
        ).detach().cpu(),
        "delete_relative_error": (
            (finite_delete - aggregate_float).norm(dim=-1)
            / finite_delete.norm(dim=-1).clamp_min(1e-12)
        ).detach().cpu(),
        "delete_cosine": F.cosine_similarity(
            finite_delete, aggregate_float, dim=-1
        ).detach().cpu(),
    }
