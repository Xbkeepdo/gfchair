"""Shared full-caption query-row extraction for Qwen2.5-VL and Qwen3-VL."""

from __future__ import annotations

from typing import Any, Sequence

import torch

from features.dgst_t import compute_dgst_t_batch_from_captures
from models.base_wrapper import (
    AttentionRequirement,
    ExtractionRequirements,
    ModelOutput,
)
from models.dgst_capture import (
    attention_row_from_capture,
    final_normalized_hidden_slice,
    hidden_states_from_captures,
    run_forward_with_dgst_captures,
)
from models.prompt_support import resolve_prompt_support_positions
from models.prompt_target import _dgst_options


def extract_qwen_shared_caption(
    *,
    wrapper: Any,
    prompt_inputs: Any,
    response_token_ids: Sequence[int],
    response_token_indices: Sequence[int],
    target_token_ids: Sequence[int],
    cfg_dgst_t: dict[str, Any],
    requirements: ExtractionRequirements,
    model_name: str,
) -> list[ModelOutput]:
    """Run one causal full-caption forward and retain only requested rows."""
    response_ids, requested_indices, targets = wrapper.validate_causal_batch_request(
        response_token_ids=response_token_ids,
        response_token_indices=response_token_indices,
        target_token_ids=target_token_ids,
    )
    if not requested_indices:
        return []
    prompt_length = int(prompt_inputs["input_ids"].shape[1])
    inputs = append_response_ids(
        prompt_inputs,
        response_token_ids=response_ids,
        device=wrapper.device,
    )
    input_ids = inputs["input_ids"][0]
    visual_start, visual_end = wrapper._find_vision_token_range(input_ids)
    visual_grid = wrapper._resolve_visual_grid(inputs, visual_end - visual_start)
    prediction_positions = [
        prompt_length - 1 + int(index) for index in requested_indices
    ]

    out, captures = run_forward_with_dgst_captures(
        wrapper.model,
        output_hidden_states=False,
        retain_attention_updates=bool(
            cfg_dgst_t.get("compute_ffn_injection_features", False)
            or cfg_dgst_t.get("jffn_validate_linearity", False)
        ),
        attention_query_positions=prediction_positions,
        capture_device=None,
        attention_query_chunk_size=int(
            wrapper.cfg.get("dgst_attention_query_chunk_size", 512)
        ),
        record_model_attentions=False,
        **inputs,
    )
    prompt_positions = resolve_prompt_support_positions(
        tokenizer=wrapper.tokenizer,
        full_input_ids=input_ids.tolist(),
        prompt_tokenized_length=prompt_length,
        image_token_id=int(wrapper._image_token_id),
        visual_start=visual_start,
        visual_end=visual_end,
        cfg_dgst_t=cfg_dgst_t,
        model_name=model_name,
    )
    dgst_results = compute_dgst_t_batch_from_captures(
        model=wrapper.model,
        full_input_ids=input_ids.tolist(),
        prompt_tokenized_length=prompt_length,
        captures=captures,
        visual_start=visual_start,
        visual_end=visual_end,
        visual_grid=visual_grid,
        image_token_id=int(wrapper._image_token_id),
        target_token_ids=targets,
        prediction_positions=prediction_positions,
        support_scope=wrapper.cfg.get("dgst_t_support_scope", "visual"),
        semantic_chunk_size=int(cfg_dgst_t.get("semantic_chunk_size", 64)),
        prompt_positions_override=prompt_positions,
        **_dgst_options(cfg_dgst_t),
        release_layer_captures=False,
    )
    if len(dgst_results) != len(requested_indices):
        raise RuntimeError(
            f"Shared Qwen forward returned {len(dgst_results)} DGST rows for "
            f"{len(requested_indices)} targets."
        )

    response_hidden = None
    if requirements.response_hidden_states:
        response_hidden = final_normalized_hidden_slice(
            model=wrapper.model,
            out=out,
            dgst_captures=captures,
            start=prompt_length,
            end=prompt_length + len(response_ids),
        ).cpu()

    outputs: list[ModelOutput] = []
    sequence_length = int(input_ids.shape[0])
    for response_index, prediction_position, dgst_result in zip(
        requested_indices, prediction_positions, dgst_results
    ):
        if requirements.attention is not AttentionRequirement.NONE:
            patch_attention, text_attention = attention_features_at_position(
                captures=captures,
                prediction_position=prediction_position,
                visual_start=visual_start,
                visual_end=visual_end,
                sequence_length=sequence_length,
            )
            if requirements.attention is AttentionRequirement.HEAD_MEAN:
                patch_attention = patch_attention.mean(dim=1, keepdim=True)
                text_attention = text_attention.mean(dim=1, keepdim=True)
        else:
            patch_attention = torch.empty(0)
            text_attention = torch.empty(0)

        if requirements.token_hidden_states or requirements.patch_hidden_states:
            token_states, patch_states = hidden_states_from_captures(
                captures,
                token_position=prediction_position,
                visual_start=visual_start,
                visual_end=visual_end,
            )
            if not requirements.token_hidden_states:
                token_states = torch.empty(0, device=patch_states.device)
            if not requirements.patch_hidden_states:
                patch_states = torch.empty(0, device=token_states.device)
        else:
            token_states = torch.empty(0)
            patch_states = torch.empty(0)

        logits = out.logits[0, prediction_position]
        predicted_id = int(logits.argmax())
        outputs.append(
            ModelOutput(
                token_id=predicted_id,
                token_str=wrapper.tokenizer.decode(
                    [predicted_id], skip_special_tokens=False
                ),
                text_to_patch_attn=patch_attention.cpu(),
                text_to_text_attn=text_attention.cpu(),
                token_hidden_states=token_states.cpu(),
                patch_hidden_states=patch_states.cpu(),
                response_token_idx=int(response_index),
                token_logits=(logits.float().cpu() if requirements.logits else None),
                dgst_t_raw=None,
                dgst_t_result=dgst_result,
                visual_grid=visual_grid if requirements.visual_layout else None,
                response_hidden_states=response_hidden,
                baseline_capture={
                    "prediction_position": int(prediction_position),
                    "visual_start": int(visual_start),
                    "visual_end": int(visual_end),
                    "attention_requirement": requirements.attention.value,
                    "qwen_forward_mode": "shared_caption_query_rows",
                },
            )
        )
    out.logits = None
    out.attentions = None
    for capture in captures:
        for key in ("h_prev", "o_attn", "h_mid", "o_ffn", "attn_weights"):
            capture[key] = None
    return outputs


def append_response_ids(
    prompt_inputs: Any,
    *,
    response_token_ids: Sequence[int],
    device: str,
) -> dict[str, Any]:
    inputs = {
        key: value.to(device=device) if torch.is_tensor(value) else value
        for key, value in dict(prompt_inputs).items()
    }
    suffix = torch.tensor(
        [int(value) for value in response_token_ids],
        dtype=inputs["input_ids"].dtype,
        device=inputs["input_ids"].device,
    ).unsqueeze(0)
    if suffix.numel() == 0:
        return inputs
    inputs["input_ids"] = torch.cat((inputs["input_ids"], suffix), dim=1)
    if "attention_mask" in inputs:
        suffix_mask = torch.ones(
            (inputs["attention_mask"].shape[0], suffix.shape[1]),
            dtype=inputs["attention_mask"].dtype,
            device=inputs["attention_mask"].device,
        )
        inputs["attention_mask"] = torch.cat(
            (inputs["attention_mask"], suffix_mask), dim=1
        )
    for key in ("position_ids", "cache_position", "rope_deltas"):
        inputs.pop(key, None)
    return inputs


def attention_features_at_position(
    *,
    captures: Sequence[dict[str, Any]],
    prediction_position: int,
    visual_start: int,
    visual_end: int,
    sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    visual_set = set(range(int(visual_start), int(visual_end)))
    text_positions = [
        position
        for position in range(int(prediction_position) + 1)
        if position not in visual_set and position != int(prediction_position)
    ]
    patch_layers = []
    text_layers = []
    for capture in captures:
        row = attention_row_from_capture(capture, int(prediction_position))
        patch_layers.append(row[:, int(visual_start) : int(visual_end)])
        text_index = torch.tensor(
            text_positions, dtype=torch.long, device=row.device
        )
        text_layers.append(row.index_select(1, text_index))
    return torch.stack(patch_layers), torch.stack(text_layers)
