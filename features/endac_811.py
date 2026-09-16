"""Compact feature math for the four-method ENDAC-811 extraction pass."""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn.functional as F

from features.dgst_t import _gaussian_mad_gate, _renormalize
from features.ffn_all_source_paths import partition, path_operator
from features.ffn_visual_path_attribution import quadrature_rule
from features.visual_ffn_jacobian import (
    reconstruct_visual_directions,
    resolve_decoder_layer_adapter,
)
from models.dgst_capture import attention_row_from_capture


def _region_ae(attention: torch.Tensor, logits: torch.Tensor, mask: torch.Tensor, epsilon: float) -> torch.Tensor:
    if not bool(mask.any()):
        return torch.zeros((), dtype=torch.float32, device=attention.device)
    selected_attention = attention[mask].float()
    selected_logits = logits[mask].float()
    return (
        _renormalize(selected_attention)
        * _gaussian_mad_gate(selected_logits, epsilon=float(epsilon))
    ).sum()


def _target_raw_logits(
    output_layer: Any,
    states: torch.Tensor,
    target_token_ids: Sequence[int],
) -> torch.Tensor:
    """Project only requested raw-logit columns; LM-head bias is excluded."""

    weight = output_layer.weight
    indices = torch.tensor(
        [int(value) for value in target_token_ids],
        dtype=torch.long,
        device=weight.device,
    )
    selected = weight.index_select(0, indices)
    return F.linear(states.to(device=weight.device, dtype=weight.dtype), selected).float()


def extract_layer_features(
    *,
    layer: Any,
    capture: dict,
    prediction_positions: Sequence[int],
    response_indices: Sequence[int],
    target_token_ids: Sequence[int],
    visual_start: int,
    visual_end: int,
    output_layer: Any,
    visual_points: int = 4,
    all_attention_points: int = 32,
    token_chunk_size: int = 32,
    mad_epsilon: float = 1e-6,
    reconstructed_directions: dict | None = None,
) -> list[dict[str, torch.Tensor]]:
    """Derive SVAR/MetaToken attention inputs and both path feature families."""

    positions = [int(value) for value in prediction_positions]
    response = [int(value) for value in response_indices]
    targets = [int(value) for value in target_token_ids]
    if not (len(positions) == len(response) == len(targets)):
        raise ValueError("Target positions, response indices and token IDs must align")

    directions = reconstructed_directions or reconstruct_visual_directions(
        layer=layer,
        capture=capture,
        prediction_positions=positions,
        visual_start=int(visual_start),
        visual_end=int(visual_end),
        return_all_sources=True,
    )
    writes = directions["all_token_writes"].float()
    sequence_length = int(writes.shape[0])
    h_prev = capture["h_prev"][0]
    logits = _target_raw_logits(output_layer, h_prev, targets)
    rows = []

    adapter = resolve_decoder_layer_adapter(layer)
    visual_rule = quadrature_rule(
        "gauss_legendre",
        int(visual_points),
        device=adapter.output_projection.weight.device,
        dtype=torch.float32,
    )
    all_rule = quadrature_rule(
        "gauss_legendre",
        int(all_attention_points),
        device=adapter.output_projection.weight.device,
        dtype=torch.float32,
    )
    for target_index, (query, response_index) in enumerate(zip(positions, response)):
        target_writes = writes[:, target_index : target_index + 1]
        masks = partition(
            sequence_length,
            query,
            response_index,
            (int(visual_start), int(visual_end)),
            target_writes.device,
        )
        causal = masks.any(dim=0)
        causal_writes = target_writes[causal]
        causal_masks = masks[:, causal]
        z = capture["h_mid"][0, query : query + 1].float()

        total = causal_writes.sum(dim=0)
        all_operator = path_operator(
            adapter.ffn_norm,
            adapter.ffn,
            z - total,
            total,
            all_rule.nodes,
            all_rule.weights,
        )
        gross = torch.zeros(3, dtype=torch.float64, device=z.device)
        step = max(1, int(token_chunk_size))
        for start in range(0, len(causal_writes), step):
            block = causal_writes[start : start + step]
            response_norm = all_operator(block).norm(dim=-1)[:, 0]
            for source_index in range(3):
                selected = causal_masks[source_index, start : start + len(block)]
                gross[source_index] += response_norm[selected].sum()

        visual_mask = masks[1]
        visual_writes = target_writes[visual_mask]
        visual_total = visual_writes.sum(dim=0)
        visual_operator = path_operator(
            adapter.ffn_norm,
            adapter.ffn,
            z - visual_total,
            visual_total,
            visual_rule.nodes,
            visual_rule.weights,
        )
        visual_gross = torch.zeros((), dtype=torch.float64, device=z.device)
        for start in range(0, len(visual_writes), step):
            block = visual_writes[start : start + step]
            visual_gross += visual_operator(block).norm(dim=-1)[:, 0].sum()

        raw_attention = attention_row_from_capture(capture, query).float()
        if bool(raw_attention[:, query + 1 :].count_nonzero()):
            raise ValueError("Attention row contains nonzero future weights")
        mean_attention = raw_attention.mean(dim=0)
        position_index = torch.arange(sequence_length, device=mean_attention.device)
        generation_start = query - response_index + 1
        region_masks = {
            "visual": (position_index >= int(visual_start))
            & (position_index < int(visual_end)),
            "visual_prompt": position_index < generation_start,
            "generation": (position_index >= generation_start)
            & (position_index <= query),
        }
        target_logits = logits[:, target_index].to(mean_attention.device)
        ae = {
            name: _region_ae(
                mean_attention,
                target_logits,
                mask,
                float(mad_epsilon),
            )
            for name, mask in region_masks.items()
        }
        rows.append(
            {
                "visual_attention": raw_attention[:, int(visual_start) : int(visual_end)].detach(),
                "ae_visual": ae["visual"].detach(),
                "ae_visual_prompt": ae["visual_prompt"].detach(),
                "ae_generation": ae["generation"].detach(),
                "visual_only_gross": visual_gross.detach(),
                "all_prompt_gross": gross[0].detach(),
                "all_visual_gross": gross[1].detach(),
                "all_generation_gross": gross[2].detach(),
            }
        )
    return rows
