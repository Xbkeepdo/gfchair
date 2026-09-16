"""Semantic-attention Top-K matrices and compact ablation features."""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn.functional as F

from models.dgst_capture import (
    apply_decoder_final_norm,
    attention_row_from_capture,
)


FEATURE_NAMES = (
    "semantic_only",
    "attention_only",
    "cosine_only",
    "semantic_attention",
    "cosine_attention",
    "semantic_attention_positive_cosine",
)


@torch.no_grad()
def target_probabilities(
    visual_states: torch.Tensor,
    target_token_ids: Sequence[int],
    output_layer: Any,
    final_norm: Any,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return no-Norm and final-Norm target probabilities as ``[M, N_V]``."""

    weight = output_layer.weight
    targets = torch.tensor(
        [int(value) for value in target_token_ids],
        dtype=torch.long,
        device=weight.device,
    )
    raw_parts, norm_parts = [], []
    for start in range(0, len(visual_states), max(1, int(chunk_size))):
        raw = visual_states[start : start + int(chunk_size)].to(
            device=weight.device,
            dtype=weight.dtype,
        )
        normed = apply_decoder_final_norm(final_norm, raw)
        states = torch.cat((raw, normed), dim=0)
        logits = F.linear(states, weight, bias=None)
        log_partition = torch.logsumexp(logits.float(), dim=-1, keepdim=True)
        selected = logits.index_select(-1, targets).float()
        probabilities = torch.exp(selected - log_partition)
        raw_probability, norm_probability = probabilities.split(len(raw), dim=0)
        raw_parts.append(raw_probability.transpose(0, 1))
        norm_parts.append(norm_probability.transpose(0, 1))
        del raw, normed, states, logits, log_partition, selected, probabilities
    return torch.cat(raw_parts, dim=1), torch.cat(norm_parts, dim=1)


@torch.no_grad()
def layer_matrices(
    *,
    capture: dict,
    prediction_positions: Sequence[int],
    target_token_ids: Sequence[int],
    visual_start: int,
    visual_end: int,
    output_layer: Any,
    final_norm: Any,
    vocab_chunk_size: int,
) -> dict[str, torch.Tensor]:
    """Build head-mean attention, two probability variants, and cosine matrices."""

    positions = [int(value) for value in prediction_positions]
    h_prev = capture["h_prev"][0]
    visual = h_prev[int(visual_start) : int(visual_end)]
    position_index = torch.tensor(positions, dtype=torch.long, device=h_prev.device)
    query = h_prev.index_select(0, position_index)

    attention = torch.stack(
        [
            attention_row_from_capture(capture, position)
            .float()
            .mean(dim=0)[int(visual_start) : int(visual_end)]
            for position in positions
        ],
        dim=0,
    )
    query_unit = F.normalize(query.float(), dim=-1)
    visual_unit = F.normalize(visual.float(), dim=-1)
    cosine = query_unit @ visual_unit.transpose(0, 1)
    probability_raw, probability_norm = target_probabilities(
        visual,
        target_token_ids,
        output_layer,
        final_norm,
        int(vocab_chunk_size),
    )
    return {
        "attention": attention,
        "object_probability_raw": probability_raw.to(attention.device),
        "object_probability_norm": probability_norm.to(attention.device),
        "cosine_similarity": cosine.to(attention.device),
    }


@torch.no_grad()
def topk_features(
    probability: torch.Tensor,
    attention: torch.Tensor,
    cosine: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Top-K indices and the six requested per-layer scalar features."""

    if int(probability.shape[-1]) < int(top_k):
        raise ValueError(
            f"Top-{top_k} requires at least {top_k} visual tokens, got "
            f"{probability.shape[-1]}"
        )
    semantic, indices = probability.topk(int(top_k), dim=-1, sorted=True)
    selected_attention = attention.gather(-1, indices)
    selected_cosine = cosine.gather(-1, indices)
    semantic_only = semantic.mean(dim=-1)
    attention_only = selected_attention.sum(dim=-1)
    cosine_only = selected_cosine.mean(dim=-1)
    semantic_attention = (semantic * selected_attention).sum(dim=-1)
    cosine_attention = cosine_only * attention_only
    joint = (
        semantic * selected_attention * selected_cosine.clamp_min(0.0)
    ).sum(dim=-1)
    features = torch.stack(
        (
            semantic_only,
            attention_only,
            cosine_only,
            semantic_attention,
            cosine_attention,
            joint,
        ),
        dim=-1,
    )
    return indices, features
