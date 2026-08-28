#!/usr/bin/env python3
"""Independent numerical/method audit for the LLaVA visual-FFN Jacobian.

This script intentionally does not modify the production JFFN-P path.  It
reconstructs the attention decomposition independently, evaluates the same
local function in model precision and in a temporary FP32 Norm+FFN copy, and
writes machine-readable evidence used by ``jacobian_visual_ffn_validation_report.md``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.visual_ffn_jacobian import (  # noqa: E402
    exact_visual_token_jvps,
    ffn_from_residual,
    reconstruct_visual_directions,
    resolve_decoder_layer_adapter,
)
from models import build_model  # noqa: E402
from models.dgst_capture import (  # noqa: E402
    attention_row_from_capture,
    pre_token_prediction_positions,
    resolve_decoder_layers,
    resolve_prompt_positions,
    run_forward_with_dgst_captures,
)
from models.llava_wrapper import _append_prefix_token_ids  # noqa: E402
from utils.config_utils import (  # noqa: E402
    get_dataset_cfg,
    get_extraction_model_cfg,
    load_config,
)
from utils.io_utils import load_json  # noqa: E402


HALL = 0
REAL = 1
LABEL_NAMES = {HALL: "hall", REAL: "real"}
DEFAULT_IMAGES = (408805, 459408, 163155)
DEFAULT_LAYERS = (1, 8, 16, 24, 32)
DEFAULT_EPSILONS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)
DEFAULT_REMOVAL_FRACTIONS = (0.1, 0.25, 0.5, 0.75, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/model_configs_inslen_official_target.yaml"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/llava_1_5_7b/COCO4000-INSLEN-OFFICIAL-TARGET",
    )
    parser.add_argument(
        "--result-dir",
        default=(
            "outputs/llava_1_5_7b/COCO4000-INSLEN-OFFICIAL-TARGET/"
            "results/jacobian_visual_ffn_validation"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--image-ids", default=",".join(str(value) for value in DEFAULT_IMAGES)
    )
    parser.add_argument(
        "--layers", default=",".join(str(value) for value in DEFAULT_LAYERS)
    )
    parser.add_argument("--max-targets-per-image", type=int, default=2)
    parser.add_argument("--random-directions", type=int, default=32)
    parser.add_argument("--jvp-chunk-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument(
        "--run-image-controls",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--run-prefix-cache-check",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def _parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _finite_float(value: Any) -> float:
    result = float(value)
    return result if math.isfinite(result) else float("nan")


def _relative_error(actual: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return (actual.float() - reference.float()).norm(dim=-1) / reference.float().norm(
        dim=-1
    ).clamp_min(1e-12)


def _cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(left.float(), right.float(), dim=-1)


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size < 3 or np.std(left) <= 1e-20 or np.std(right) <= 1e-20:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    return _safe_corr(_rankdata(left), _rankdata(right))


def _distribution(values: torch.Tensor) -> torch.Tensor:
    finite = torch.nan_to_num(values.float(), nan=0.0, posinf=0.0, neginf=0.0)
    finite = finite.clamp_min(0.0)
    return finite / finite.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def _js(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = _distribution(left).clamp_min(1e-30)
    right = _distribution(right).clamp_min(1e-30)
    middle = 0.5 * (left + right)
    return 0.5 * (
        (left * (left.log() - middle.log())).sum(dim=-1)
        + (right * (right.log() - middle.log())).sum(dim=-1)
    )


def _expand_decoder_ids(
    tokenized_ids: Sequence[int], image_token_id: int, visual_count: int
) -> list[int]:
    ids = [int(value) for value in tokenized_ids]
    image_positions = [
        index for index, token_id in enumerate(ids) if token_id == int(image_token_id)
    ]
    if len(image_positions) == 1 and int(visual_count) > 1:
        position = image_positions[0]
        return (
            ids[:position]
            + [int(image_token_id)] * int(visual_count)
            + ids[position + 1 :]
        )
    return ids


def _project_values_independently(adapter: Any, normalized: torch.Tensor) -> torch.Tensor:
    sequence_length = int(normalized.shape[0])
    projected = adapter.value_projection(normalized)
    if adapter.family == "internlm2_packed_qkv":
        group_width = 2 + int(adapter.num_key_value_groups)
        values = projected.view(
            sequence_length,
            int(adapter.num_key_value_heads),
            group_width,
            int(adapter.head_dim),
        )[:, :, -1, :]
    else:
        values = projected.view(
            sequence_length,
            int(adapter.num_key_value_heads),
            int(adapter.head_dim),
        )
    values = values.permute(1, 0, 2)
    if int(adapter.num_key_value_groups) > 1:
        values = values.repeat_interleave(
            int(adapter.num_key_value_groups), dim=0
        )
    return values


@torch.no_grad()
def independent_attention_decomposition(
    *,
    layer: Any,
    capture: dict[str, Any],
    prediction_positions: Sequence[int],
    visual_start: int,
    visual_end: int,
    expanded_ids: Sequence[int],
    special_token_ids: set[int],
) -> dict[str, Any]:
    """Reconstruct every key contribution without calling the production helper."""
    adapter = resolve_decoder_layer_adapter(layer)
    device = adapter.output_projection.weight.device
    dtype = adapter.output_projection.weight.dtype
    h_prev = capture["h_prev"][0].to(device=device, dtype=dtype)
    sequence_length = int(h_prev.shape[0])
    if len(expanded_ids) != sequence_length:
        raise ValueError(
            f"Expanded ID length {len(expanded_ids)} != decoder length {sequence_length}"
        )
    normalized = adapter.attention_norm(h_prev)
    values = _project_values_independently(adapter, normalized)
    attention = torch.stack(
        [
            attention_row_from_capture(capture, int(position)).to(
                device=device, dtype=dtype
            )
            for position in prediction_positions
        ],
        dim=0,
    )
    # [S,T,H,Dh] -> [S,T,D]
    weighted_heads = (
        attention.permute(2, 0, 1).unsqueeze(-1)
        * values.permute(1, 0, 2).unsqueeze(1)
    )
    all_token_updates = F.linear(
        weighted_heads.reshape(sequence_length, len(prediction_positions), -1),
        adapter.output_projection.weight,
        bias=None,
    )
    visual_updates = all_token_updates[int(visual_start) : int(visual_end)]
    visual_sum = visual_updates.sum(dim=0)

    visual_mask = torch.zeros(sequence_length, dtype=torch.bool, device=device)
    visual_mask[int(visual_start) : int(visual_end)] = True
    nonvisual_sum = all_token_updates[~visual_mask].sum(dim=0)
    text_mask_values = [
        (not bool(visual_mask[index].item()))
        and int(token_id) not in special_token_ids
        for index, token_id in enumerate(expanded_ids)
    ]
    text_mask = torch.tensor(text_mask_values, dtype=torch.bool, device=device)
    text_sum = all_token_updates[text_mask].sum(dim=0)

    bias = adapter.output_projection.bias
    if bias is None:
        projected_bias = torch.zeros(
            (1, int(adapter.output_projection.weight.shape[0])),
            dtype=dtype,
            device=device,
        )
    else:
        projected_bias = bias.to(device=device, dtype=dtype).unsqueeze(0)
    reconstructed = all_token_updates.sum(dim=0) + projected_bias
    position_index = torch.tensor(
        [int(value) for value in prediction_positions],
        dtype=torch.long,
        device=device,
    )
    actual = capture["o_attn"][0].to(device=device, dtype=dtype).index_select(
        0, position_index
    )
    component_sum = visual_sum + nonvisual_sum + projected_bias

    visual_attention = attention[:, :, int(visual_start) : int(visual_end)]
    attention_mean = visual_attention.float().mean(dim=1)
    attention_mass = attention_mean.sum(dim=-1)
    return {
        "adapter": adapter,
        "a_all_tokens": all_token_updates.detach(),
        "a_tokens": visual_updates.detach(),
        "a_visual": visual_sum.detach(),
        "a_nonvisual": nonvisual_sum.detach(),
        "a_text": text_sum.detach(),
        "attention_visual_mean": attention_mean.detach(),
        "attention_visual_mass": attention_mass.detach(),
        "actual_attention_update": actual.detach(),
        "reconstructed_attention_update": reconstructed.detach(),
        "reconstruction_relative_error": _relative_error(
            reconstructed, actual
        ).detach(),
        "reconstruction_cosine": _cosine(reconstructed, actual).detach(),
        "component_sum_relative_error": _relative_error(
            component_sum, actual
        ).detach(),
        "all_token_sum_relative_error": _relative_error(
            all_token_updates.sum(dim=0) + projected_bias, actual
        ).detach(),
        "output_bias_norm": float(projected_bias.float().norm()),
        "text_position_count": int(text_mask.sum().item()),
        "nonvisual_position_count": int((~visual_mask).sum().item()),
    }


def _fp32_function(layer: Any, device: torch.device):
    adapter = resolve_decoder_layer_adapter(layer)
    norm = copy.deepcopy(adapter.ffn_norm).to(device=device, dtype=torch.float32).eval()
    ffn = copy.deepcopy(adapter.ffn).to(device=device, dtype=torch.float32).eval()
    norm.requires_grad_(False)
    ffn.requires_grad_(False)

    def function(value: torch.Tensor) -> torch.Tensor:
        return ffn(norm(value))

    return norm, ffn, function


def _vmap_jvp(
    function: Any,
    z: torch.Tensor,
    directions: torch.Tensor,
    *,
    chunk_size: int | None,
) -> torch.Tensor:
    def one(direction: torch.Tensor) -> torch.Tensor:
        return torch.func.jvp(function, (z,), (direction,))[1]

    with torch.enable_grad():
        return torch.vmap(one, chunk_size=chunk_size)(directions).detach()


def _one_jvp(function: Any, z: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    with torch.enable_grad():
        return torch.func.jvp(function, (z,), (direction,))[1].detach()


def _target_groups(
    label_row: dict[str, Any], response_ids: Sequence[int]
) -> dict[int, list[dict[str, Any]]]:
    result: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for span in label_row.get("object_token_spans", []):
        indices = span.get("token_indices") or []
        label = int(span.get("label", -1))
        if not indices or label not in (HALL, REAL):
            continue
        index = int(indices[0])
        if index < 0 or index >= len(response_ids):
            raise IndexError(f"Invalid response index {index}")
        target_id = int(span.get("target_token_id", response_ids[index]))
        if target_id != int(response_ids[index]):
            raise ValueError(f"Target ID mismatch at response index {index}")
        result[index].append(span)
    return dict(result)


def _select_targets(
    groups: dict[int, list[dict[str, Any]]], limit: int
) -> list[int]:
    if limit <= 0 or len(groups) <= limit:
        return sorted(groups)
    labels_by_index = {
        index: {int(span["label"]) for span in spans}
        for index, spans in groups.items()
    }
    selected: list[int] = []
    for label in (REAL, HALL):
        candidates = sorted(
            index for index, labels in labels_by_index.items() if label in labels
        )
        if candidates:
            # Prefer a late hallucination and an early real token to cover
            # substantially different autoregressive positions.
            selected.append(candidates[-1] if label == HALL else candidates[0])
    for index in sorted(groups):
        if len(selected) >= limit:
            break
        if index not in selected:
            selected.append(index)
    return sorted(selected[:limit])


def _target_metadata(
    image_id: int,
    response_indices: Sequence[int],
    groups: dict[int, list[dict[str, Any]]],
    response_ids: Sequence[int],
    tokenizer: Any,
) -> list[dict[str, Any]]:
    rows = []
    for index in response_indices:
        spans = groups[int(index)]
        labels = sorted({int(span["label"]) for span in spans})
        words = sorted(
            {
                str(span.get("official_detected_word") or span.get("word") or "")
                for span in spans
            }
        )
        rows.append(
            {
                "image_id": int(image_id),
                "response_index": int(index),
                "target_token_id": int(response_ids[int(index)]),
                "decoded_token": tokenizer.decode(
                    [int(response_ids[int(index)])], skip_special_tokens=False
                ),
                "labels": "+".join(LABEL_NAMES[value] for value in labels),
                "words": "+".join(words),
            }
        )
    return rows


def _prepare_capture(
    *,
    wrapper: Any,
    image: Image.Image,
    response_ids: Sequence[int],
    response_indices: Sequence[int],
    prompt: str,
) -> dict[str, Any]:
    prompt_text = wrapper._format_prompt(wrapper.resolve_prompt(prompt))
    prefix_inputs = wrapper.processor(
        text=prompt_text, images=image, return_tensors="pt"
    )
    prompt_tokenized_length = int(prefix_inputs["input_ids"].shape[1])
    full_inputs = _append_prefix_token_ids(
        prefix_inputs,
        prefix_token_ids=response_ids,
        device=wrapper.device,
        dtype=torch.float16,
    )
    tokenized_ids = [int(value) for value in full_inputs["input_ids"][0].tolist()]
    image_token_id = int(wrapper._image_token_id())
    visual_start, visual_end = wrapper._find_visual_token_range(
        full_inputs, image_token_id
    )
    visual_count = int(visual_end - visual_start)
    prompt_positions = resolve_prompt_positions(
        full_input_ids=tokenized_ids,
        prompt_tokenized_length=prompt_tokenized_length,
        image_token_id=image_token_id,
        visual_start=visual_start,
        visual_end=visual_end,
    )
    prediction_positions = pre_token_prediction_positions(
        full_input_ids=tokenized_ids,
        prompt_tokenized_length=prompt_tokenized_length,
        response_token_indices=response_indices,
        image_token_id=image_token_id,
        visual_token_count=visual_count,
        prompt_positions=prompt_positions,
    )
    out, captures = run_forward_with_dgst_captures(
        wrapper.model,
        output_hidden_states=False,
        retain_attention_updates=True,
        attention_query_positions=prediction_positions,
        capture_device=None,
        attention_query_chunk_size=512,
        record_model_attentions=False,
        **full_inputs,
    )
    selected_logits = torch.stack(
        [out.logits[0, int(position)].float().cpu() for position in prediction_positions]
    )
    expanded_ids = _expand_decoder_ids(
        tokenized_ids, image_token_id=image_token_id, visual_count=visual_count
    )
    result = {
        "captures": captures,
        "prediction_positions": prediction_positions,
        "selected_logits": selected_logits,
        "tokenized_ids": tokenized_ids,
        "expanded_ids": expanded_ids,
        "prompt_tokenized_length": prompt_tokenized_length,
        "visual_start": int(visual_start),
        "visual_end": int(visual_end),
        "visual_count": visual_count,
        "image_token_id": image_token_id,
        "expanded_sequence_length": int(out.logits.shape[1]),
        "full_inputs": full_inputs,
    }
    out.logits = None
    out.attentions = None
    del out, prefix_inputs
    return result


def _release_prepared(prepared: dict[str, Any]) -> None:
    for capture in prepared.get("captures", []):
        for key in tuple(capture):
            capture[key] = None
    prepared.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _intermediate_size(adapter: Any) -> int:
    for module_name in ("gate_proj", "w1"):
        module = getattr(adapter.ffn, module_name, None)
        if module is not None and hasattr(module, "out_features"):
            return int(module.out_features)
    return int(getattr(adapter.ffn, "intermediate_size", 0))


def _cache_summary(past_key_values: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"type": type(past_key_values).__name__}
    get_length = getattr(past_key_values, "get_seq_length", None)
    if callable(get_length):
        result["sequence_length"] = int(get_length())
    try:
        first = past_key_values[0]
        if isinstance(first, (tuple, list)) and len(first) >= 2:
            result["first_key_shape"] = list(first[0].shape)
            result["first_value_shape"] = list(first[1].shape)
    except Exception:
        layers = getattr(past_key_values, "layers", None)
        if layers:
            first_layer = layers[0]
            for name in ("keys", "key_cache"):
                value = getattr(first_layer, name, None)
                if torch.is_tensor(value):
                    result["first_key_shape"] = list(value.shape)
                    break
            for name in ("values", "value_cache"):
                value = getattr(first_layer, name, None)
                if torch.is_tensor(value):
                    result["first_value_shape"] = list(value.shape)
                    break
    return result


def _prefix_and_cache_checks(
    *,
    wrapper: Any,
    image: Image.Image,
    prompt: str,
    response_ids: Sequence[int],
    response_indices: Sequence[int],
    layer_indices: Sequence[int],
    full_prepared: dict[str, Any],
    special_token_ids: set[int],
    jvp_chunk_size: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    layers = resolve_decoder_layers(wrapper.model)
    for target_offset, response_index in enumerate(response_indices):
        prefix_ids = [int(value) for value in response_ids[: int(response_index)]]
        prefix_prepared = _prepare_capture(
            wrapper=wrapper,
            image=image,
            response_ids=prefix_ids,
            response_indices=[int(response_index)],
            prompt=prompt,
        )
        full_position = int(full_prepared["prediction_positions"][target_offset])
        prefix_position = int(prefix_prepared["prediction_positions"][0])
        logits_full = full_prepared["selected_logits"][target_offset]
        logits_prefix = prefix_prepared["selected_logits"][0]
        base = {
            "response_index": int(response_index),
            "full_prediction_position": full_position,
            "prefix_prediction_position": prefix_position,
            "full_sequence_length": int(full_prepared["expanded_sequence_length"]),
            "prefix_sequence_length": int(prefix_prepared["expanded_sequence_length"]),
            "logits_relative_error_full_vs_prefix": float(
                (logits_full - logits_prefix).norm()
                / logits_prefix.norm().clamp_min(1e-12)
            ),
            "logits_max_abs_full_vs_prefix": float(
                (logits_full - logits_prefix).abs().max()
            ),
            "logits_argmax_match_full_vs_prefix": bool(
                int(logits_full.argmax()) == int(logits_prefix.argmax())
            ),
        }

        # Native cache is not used by the production extractor.  This check only
        # establishes that a normal cached full-prefix forward agrees at the same
        # prediction row and that visual K/V are present in the returned cache.
        cache_inputs = prefix_prepared["full_inputs"]
        with torch.no_grad():
            cache_out = wrapper.model(
                **cache_inputs,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
                use_cache=True,
            )
        cache_logits = cache_out.logits[0, prefix_position].float().cpu()
        base.update(
            {
                "logits_relative_error_no_cache_vs_cache": float(
                    (logits_prefix - cache_logits).norm()
                    / logits_prefix.norm().clamp_min(1e-12)
                ),
                "logits_max_abs_no_cache_vs_cache": float(
                    (logits_prefix - cache_logits).abs().max()
                ),
                "logits_argmax_match_no_cache_vs_cache": bool(
                    int(logits_prefix.argmax()) == int(cache_logits.argmax())
                ),
                "cache": _cache_summary(cache_out.past_key_values),
            }
        )
        cache_out.logits = None
        del cache_out, cache_logits

        for layer_index in layer_indices:
            layer = layers[layer_index]
            full_capture = full_prepared["captures"][layer_index]
            prefix_capture = prefix_prepared["captures"][layer_index]
            full_index = torch.tensor(
                [full_position], dtype=torch.long, device=full_capture["h_mid"].device
            )
            prefix_index = torch.tensor(
                [prefix_position],
                dtype=torch.long,
                device=prefix_capture["h_mid"].device,
            )
            full_hmid = full_capture["h_mid"][0].index_select(0, full_index)
            prefix_hmid = prefix_capture["h_mid"][0].index_select(0, prefix_index)
            full_decomp = independent_attention_decomposition(
                layer=layer,
                capture=full_capture,
                prediction_positions=[full_position],
                visual_start=int(full_prepared["visual_start"]),
                visual_end=int(full_prepared["visual_end"]),
                expanded_ids=full_prepared["expanded_ids"],
                special_token_ids=special_token_ids,
            )
            prefix_decomp = independent_attention_decomposition(
                layer=layer,
                capture=prefix_capture,
                prediction_positions=[prefix_position],
                visual_start=int(prefix_prepared["visual_start"]),
                visual_end=int(prefix_prepared["visual_end"]),
                expanded_ids=prefix_prepared["expanded_ids"],
                special_token_ids=special_token_ids,
            )
            full_responses, _ = exact_visual_token_jvps(
                layer=layer,
                z=full_hmid,
                a_tokens=full_decomp["a_tokens"],
                chunk_size=int(jvp_chunk_size),
                synchronize=False,
                measure_time=False,
            )
            prefix_responses, _ = exact_visual_token_jvps(
                layer=layer,
                z=prefix_hmid,
                a_tokens=prefix_decomp["a_tokens"],
                chunk_size=int(jvp_chunk_size),
                synchronize=False,
                measure_time=False,
            )
            energy_full = full_responses.float().norm(dim=-1).squeeze(1)
            energy_prefix = prefix_responses.float().norm(dim=-1).squeeze(1)
            p_full = _distribution(energy_full.unsqueeze(0)).squeeze(0)
            p_prefix = _distribution(energy_prefix.unsqueeze(0)).squeeze(0)
            rows.append(
                {
                    **base,
                    "layer": int(layer_index + 1),
                    "hmid_relative_error": float(
                        _relative_error(full_hmid, prefix_hmid).item()
                    ),
                    "hmid_cosine": float(_cosine(full_hmid, prefix_hmid).item()),
                    "a_visual_relative_error": float(
                        _relative_error(
                            full_decomp["a_visual"], prefix_decomp["a_visual"]
                        ).item()
                    ),
                    "p_js_full_vs_prefix": float(
                        _js(p_full.unsqueeze(0), p_prefix.unsqueeze(0)).item()
                    ),
                    "p_cosine_full_vs_prefix": float(
                        _cosine(p_full.unsqueeze(0), p_prefix.unsqueeze(0)).item()
                    ),
                }
            )
            del (
                full_responses,
                prefix_responses,
                energy_full,
                energy_prefix,
                p_full,
                p_prefix,
                full_decomp,
                prefix_decomp,
            )
        _release_prepared(prefix_prepared)
    return rows


def _patch_shuffle(image: Image.Image, seed: int, grid: int = 8) -> Image.Image:
    array = np.asarray(image.convert("RGB"))
    height, width = array.shape[:2]
    y_edges = np.linspace(0, height, grid + 1, dtype=int)
    x_edges = np.linspace(0, width, grid + 1, dtype=int)
    tiles = [
        array[y_edges[y] : y_edges[y + 1], x_edges[x] : x_edges[x + 1]].copy()
        for y in range(grid)
        for x in range(grid)
    ]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(tiles))
    result = np.empty_like(array)
    # Resize only when unequal edge rounding gives a different tile shape.
    for output_index, source_index in enumerate(order):
        y, x = divmod(output_index, grid)
        target_h = y_edges[y + 1] - y_edges[y]
        target_w = x_edges[x + 1] - x_edges[x]
        tile = Image.fromarray(tiles[int(source_index)]).resize(
            (target_w, target_h), Image.Resampling.BILINEAR
        )
        result[y_edges[y] : y_edges[y + 1], x_edges[x] : x_edges[x + 1]] = np.asarray(tile)
    return Image.fromarray(result)


def _image_control_checks(
    *,
    wrapper: Any,
    prompt: str,
    base_image_id: int,
    base_image: Image.Image,
    unrelated_image: Image.Image,
    response_ids: Sequence[int],
    response_indices: Sequence[int],
    layer_indices: Sequence[int],
    special_token_ids: set[int],
    baseline_profiles: dict[tuple[int, int], dict[str, Any]],
    chunk_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    variants = {
        "black_image": Image.new("RGB", base_image.size, color=(0, 0, 0)),
        "patch_shuffle_8x8": _patch_shuffle(base_image, seed=seed),
        "unrelated_image": unrelated_image.convert("RGB"),
    }
    layers = resolve_decoder_layers(wrapper.model)
    for variant_name, variant_image in variants.items():
        prepared = _prepare_capture(
            wrapper=wrapper,
            image=variant_image,
            response_ids=response_ids,
            response_indices=response_indices,
            prompt=prompt,
        )
        position_index = torch.tensor(
            prepared["prediction_positions"],
            dtype=torch.long,
            device=prepared["captures"][0]["h_mid"].device,
        )
        for layer_index in layer_indices:
            layer = layers[layer_index]
            capture = prepared["captures"][layer_index]
            decomposition = independent_attention_decomposition(
                layer=layer,
                capture=capture,
                prediction_positions=prepared["prediction_positions"],
                visual_start=int(prepared["visual_start"]),
                visual_end=int(prepared["visual_end"]),
                expanded_ids=prepared["expanded_ids"],
                special_token_ids=special_token_ids,
            )
            z = capture["h_mid"][0].index_select(0, position_index)
            responses, _ = exact_visual_token_jvps(
                layer=layer,
                z=z,
                a_tokens=decomposition["a_tokens"],
                chunk_size=int(chunk_size),
                synchronize=False,
                measure_time=False,
            )
            energy = responses.float().norm(dim=-1).transpose(0, 1)
            p = _distribution(energy)
            aggregate = responses.float().sum(dim=0)
            gain = aggregate.norm(dim=-1) / decomposition["a_visual"].float().norm(
                dim=-1
            ).clamp_min(1e-12)
            for target_offset, response_index in enumerate(response_indices):
                baseline = baseline_profiles[(int(response_index), int(layer_index))]
                baseline_p = baseline["p"]
                current_p = p[target_offset].detach().cpu()
                rows.append(
                    {
                        "base_image_id": int(base_image_id),
                        "variant": variant_name,
                        "response_index": int(response_index),
                        "layer": int(layer_index + 1),
                        "p_js_vs_original": float(
                            _js(
                                current_p.unsqueeze(0),
                                baseline_p.unsqueeze(0),
                            ).item()
                        ),
                        "p_tv_vs_original": float(
                            0.5
                            * (current_p - baseline_p)
                            .abs()
                            .sum()
                        ),
                        "p_cosine_vs_original": float(
                            _cosine(
                                current_p.unsqueeze(0),
                                baseline_p.unsqueeze(0),
                            ).item()
                        ),
                        "gain": float(gain[target_offset]),
                        "gain_ratio_vs_original": float(
                            gain[target_offset] / max(float(baseline["gain"]), 1e-12)
                        ),
                        "visual_attention_mass": float(
                            decomposition["attention_visual_mass"][target_offset]
                        ),
                        "attention_mass_ratio_vs_original": float(
                            decomposition["attention_visual_mass"][target_offset]
                            / max(float(baseline["attention_mass"]), 1e-12)
                        ),
                    }
                )
            del responses, decomposition, energy, p, aggregate, gain, z
        _release_prepared(prepared)
    return rows


def _plot_diagnostics(
    *,
    epsilon_rows: Sequence[dict[str, Any]],
    case_rows: Sequence[dict[str, Any]],
    image_control_rows: Sequence[dict[str, Any]],
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    epsilons = sorted({float(row["epsilon"]) for row in epsilon_rows})
    central = [
        float(
            np.nanmedian(
                [
                    float(row["central_relative_error"])
                    for row in epsilon_rows
                    if float(row["epsilon"]) == epsilon
                ]
            )
        )
        for epsilon in epsilons
    ]
    forward = [
        float(
            np.nanmedian(
                [
                    float(row["forward_relative_error"])
                    for row in epsilon_rows
                    if float(row["epsilon"]) == epsilon
                ]
            )
        )
        for epsilon in epsilons
    ]
    axes[0, 0].loglog(epsilons, central, marker="o", label="central FD")
    axes[0, 0].loglog(epsilons, forward, marker="s", label="forward FD")
    axes[0, 0].set_title("FP32 finite-difference relative error")
    axes[0, 0].set_xlabel("epsilon")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    layer_values = sorted({int(row["layer"]) for row in case_rows})
    visual_s = [
        np.nanmedian(
            [float(row["visual_gain_fp32"]) for row in case_rows if int(row["layer"]) == layer]
        )
        for layer in layer_values
    ]
    random_s = [
        np.nanmedian(
            [float(row["random_gain_median"]) for row in case_rows if int(row["layer"]) == layer]
        )
        for layer in layer_values
    ]
    axes[0, 1].plot(layer_values, visual_s, marker="o", label="visual direction")
    axes[0, 1].plot(layer_values, random_s, marker="s", label="matched random")
    axes[0, 1].set_title("Visual gain vs matched-norm random control")
    axes[0, 1].set_xlabel("decoder layer")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    pearson = [
        np.nanmedian(
            [float(row["attention_energy_pearson"]) for row in case_rows if int(row["layer"]) == layer]
        )
        for layer in layer_values
    ]
    spearman = [
        np.nanmedian(
            [float(row["attention_energy_spearman"]) for row in case_rows if int(row["layer"]) == layer]
        )
        for layer in layer_values
    ]
    axes[1, 0].plot(layer_values, pearson, marker="o", label="Pearson")
    axes[1, 0].plot(layer_values, spearman, marker="s", label="Spearman")
    axes[1, 0].set_ylim(-1.0, 1.0)
    axes[1, 0].set_title("Per-token attention vs JFFN energy")
    axes[1, 0].set_xlabel("decoder layer")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    if image_control_rows:
        names = sorted({str(row["variant"]) for row in image_control_rows})
        js_values = [
            np.nanmedian(
                [
                    float(row["p_js_vs_original"])
                    for row in image_control_rows
                    if row["variant"] == name
                ]
            )
            for name in names
        ]
        axes[1, 1].bar(range(len(names)), js_values)
        axes[1, 1].set_xticks(range(len(names)), names, rotation=18, ha="right")
        axes[1, 1].set_title("Image control: median JS(P, original P)")
    else:
        axes[1, 1].axis("off")
    fig.suptitle("LLaVA visual-FFN Jacobian independent validation")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _aggregate_numeric(rows: Sequence[dict[str, Any]], keys: Iterable[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        result[key] = {
            "count": int(values.size),
            "finite_count": int(finite.size),
            "mean": float(np.mean(finite)) if finite.size else float("nan"),
            "median": float(np.median(finite)) if finite.size else float("nan"),
            "min": float(np.min(finite)) if finite.size else float("nan"),
            "max": float(np.max(finite)) if finite.size else float("nan"),
        }
    return result


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed) % (2**32))
    config = load_config(args.config)
    model_cfg = get_extraction_model_cfg(config, "llava_1_5_7b")
    dataset_cfg = get_dataset_cfg(config)
    prompt = str((config.get("run") or {}).get("prompt") or "Describe this image.")
    source_output = Path(args.output_dir).resolve()
    result_dir = Path(args.result_dir).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    labels = {
        int(key): value
        for key, value in load_json(str(source_output / "labeling.json")).items()
    }
    generations = {
        int(key): value
        for key, value in load_json(str(source_output / "generations.json")).items()
    }
    images_dir = Path(dataset_cfg["coco_root"]) / "val2014"
    image_ids = _parse_ints(args.image_ids)
    requested_layers = _parse_ints(args.layers)
    if not image_ids or not requested_layers:
        raise ValueError("--image-ids and --layers must be non-empty")

    print(f"[audit] loading LLaVA on {args.device}", flush=True)
    wrapper = build_model("llava_1_5_7b", model_cfg, device=args.device)
    wrapper.model.requires_grad_(False)
    wrapper.model.eval()
    layers = resolve_decoder_layers(wrapper.model)
    layer_indices = [int(value) - 1 for value in requested_layers]
    if any(index < 0 or index >= len(layers) for index in layer_indices):
        raise IndexError(f"Invalid layers {requested_layers}; model has {len(layers)}")
    device = torch.device(args.device)
    special_token_ids = {int(value) for value in wrapper.tokenizer.all_special_ids}

    case_rows: list[dict[str, Any]] = []
    epsilon_rows: list[dict[str, Any]] = []
    lambda_rows: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []
    prefix_rows: list[dict[str, Any]] = []
    image_control_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    sequence_audit: list[dict[str, Any]] = []
    efficiency_rows: list[dict[str, Any]] = []
    baseline_profiles: dict[tuple[int, int], dict[str, Any]] = {}
    base_image_for_controls: Image.Image | None = None
    base_response_ids: list[int] = []
    base_response_indices: list[int] = []
    base_image_id = int(image_ids[0])

    model_allocated = int(torch.cuda.memory_allocated(device))
    model_reserved = int(torch.cuda.memory_reserved(device))
    failures: list[dict[str, Any]] = []

    for image_offset, image_id in enumerate(image_ids):
        try:
            response_ids = [
                int(value)
                for value in generations[int(image_id)].get("response_token_ids", [])
            ]
            groups = _target_groups(labels[int(image_id)], response_ids)
            response_indices = _select_targets(
                groups, limit=int(args.max_targets_per_image)
            )
            if not response_indices:
                raise ValueError(f"Image {image_id} has no official target")
            metadata = _target_metadata(
                image_id,
                response_indices,
                groups,
                response_ids,
                wrapper.tokenizer,
            )
            target_rows.extend(metadata)
            image_path = images_dir / f"COCO_val2014_{int(image_id):012d}.jpg"
            image = Image.open(image_path).convert("RGB")
            if image_offset == 0:
                base_image_for_controls = image.copy()
                base_response_ids = list(response_ids)
                base_response_indices = list(response_indices)

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            forward_before = int(torch.cuda.memory_allocated(device))
            prepared = _prepare_capture(
                wrapper=wrapper,
                image=image,
                response_ids=response_ids,
                response_indices=response_indices,
                prompt=prompt,
            )
            torch.cuda.synchronize(device)
            forward_peak = int(torch.cuda.max_memory_allocated(device))
            sequence_audit.append(
                {
                    "image_id": int(image_id),
                    "tokenized_sequence_length": len(prepared["tokenized_ids"]),
                    "expanded_sequence_length": int(
                        prepared["expanded_sequence_length"]
                    ),
                    "captured_sequence_length": int(
                        prepared["captures"][0]["h_prev"].shape[1]
                    ),
                    "prompt_tokenized_length": int(
                        prepared["prompt_tokenized_length"]
                    ),
                    "visual_start": int(prepared["visual_start"]),
                    "visual_end_exclusive": int(prepared["visual_end"]),
                    "visual_token_count": int(prepared["visual_count"]),
                    "image_placeholder_count_tokenized": int(
                        sum(
                            token_id == int(prepared["image_token_id"])
                            for token_id in prepared["tokenized_ids"]
                        )
                    ),
                    "bos_token_id": wrapper.tokenizer.bos_token_id,
                    "first_token_id": int(prepared["tokenized_ids"][0]),
                    "prediction_positions": list(prepared["prediction_positions"]),
                    "response_indices": list(response_indices),
                    "forward_peak_gib": forward_peak / 2**30,
                    "forward_increment_gib": (forward_peak - forward_before) / 2**30,
                }
            )

            if image_offset == 0 and bool(args.run_prefix_cache_check):
                prefix_rows.extend(
                    _prefix_and_cache_checks(
                        wrapper=wrapper,
                        image=image,
                        prompt=prompt,
                        response_ids=response_ids,
                        response_indices=response_indices,
                        layer_indices=layer_indices,
                        full_prepared=prepared,
                        special_token_ids=special_token_ids,
                        jvp_chunk_size=int(args.jvp_chunk_size),
                    )
                )

            position_index = torch.tensor(
                prepared["prediction_positions"],
                dtype=torch.long,
                device=device,
            )
            for layer_index in layer_indices:
                layer = layers[layer_index]
                capture = prepared["captures"][layer_index]
                independent = independent_attention_decomposition(
                    layer=layer,
                    capture=capture,
                    prediction_positions=prepared["prediction_positions"],
                    visual_start=int(prepared["visual_start"]),
                    visual_end=int(prepared["visual_end"]),
                    expanded_ids=prepared["expanded_ids"],
                    special_token_ids=special_token_ids,
                )
                production = reconstruct_visual_directions(
                    layer=layer,
                    capture=capture,
                    prediction_positions=prepared["prediction_positions"],
                    visual_start=int(prepared["visual_start"]),
                    visual_end=int(prepared["visual_end"]),
                )
                z_model = capture["h_mid"][0].index_select(0, position_index)
                o_ffn_actual = capture["o_ffn"][0].index_select(0, position_index)
                o_ffn_recomputed = ffn_from_residual(layer, z_model)
                a_tokens_model = independent["a_tokens"]
                a_visual_model = independent["a_visual"]

                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                jvp_before = int(torch.cuda.memory_allocated(device))
                model_responses, model_seconds = exact_visual_token_jvps(
                    layer=layer,
                    z=z_model,
                    a_tokens=a_tokens_model,
                    chunk_size=int(args.jvp_chunk_size),
                )
                model_peak = int(torch.cuda.max_memory_allocated(device))
                model_energy = model_responses.float().norm(dim=-1).transpose(0, 1)
                model_p = _distribution(model_energy)
                model_response_sum = model_responses.float().sum(dim=0)

                norm32, ffn32, function32 = _fp32_function(layer, device)
                z32 = z_model.float()
                a_tokens32 = a_tokens_model.float()
                a_visual32 = a_visual_model.float()
                fp32_responses = _vmap_jvp(
                    function32,
                    z32,
                    a_tokens32,
                    chunk_size=int(args.jvp_chunk_size),
                )
                fp32_aggregate = _one_jvp(function32, z32, a_visual32)
                fp32_sum = fp32_responses.sum(dim=0)
                fp32_energy = fp32_responses.norm(dim=-1).transpose(0, 1)
                fp32_p = _distribution(fp32_energy)
                center32 = function32(z32)
                normalized32 = norm32(z32)
                text32 = independent["a_text"].float()
                nonvisual32 = independent["a_nonvisual"].float()
                text_response = _one_jvp(function32, z32, text32)
                nonvisual_response = _one_jvp(function32, z32, nonvisual32)

                random_count = int(args.random_directions)
                random_directions = torch.randn(
                    (random_count, *z32.shape),
                    dtype=torch.float32,
                    device=device,
                )
                visual_norm = a_visual32.norm(dim=-1)
                random_directions = random_directions / random_directions.norm(
                    dim=-1, keepdim=True
                ).clamp_min(1e-12)
                random_directions = random_directions * visual_norm.unsqueeze(0).unsqueeze(-1)
                random_responses = _vmap_jvp(
                    function32,
                    z32,
                    random_directions,
                    chunk_size=min(16, random_count),
                )
                random_gain = random_responses.norm(dim=-1) / random_directions.norm(
                    dim=-1
                ).clamp_min(1e-12)

                # Explicit JVP linearity on two independent matched-norm random
                # directions.  This is separate from per-token additivity.
                alpha, beta = 0.37, -1.21
                d1, d2 = random_directions[0], random_directions[1]
                jd1 = random_responses[0]
                jd2 = random_responses[1]
                combined = _one_jvp(function32, z32, alpha * d1 + beta * d2)
                combined_reference = alpha * jd1 + beta * jd2
                scalar_linearity = _relative_error(combined, combined_reference)

                for target_offset, (target_meta, response_index) in enumerate(
                    zip(metadata, response_indices)
                ):
                    attention_values = independent["attention_visual_mean"][
                        target_offset
                    ].float()
                    input_per_token = a_tokens32[:, target_offset, :].norm(dim=-1)
                    energy_per_token = fp32_energy[target_offset]
                    gain_per_token = energy_per_token / input_per_token.clamp_min(1e-12)
                    delta_total = fp32_aggregate[target_offset]
                    signed = (
                        fp32_responses[:, target_offset, :]
                        * delta_total.unsqueeze(0)
                    ).sum(dim=-1) / delta_total.square().sum().clamp_min(1e-12)
                    cancellation_ratio = delta_total.norm() / energy_per_token.sum().clamp_min(
                        1e-12
                    )
                    random_here = random_gain[:, target_offset]
                    visual_gain = fp32_aggregate[target_offset].norm() / visual_norm[
                        target_offset
                    ].clamp_min(1e-12)
                    text_gain = text_response[target_offset].norm() / text32[
                        target_offset
                    ].norm().clamp_min(1e-12)
                    nonvisual_gain = nonvisual_response[target_offset].norm() / nonvisual32[
                        target_offset
                    ].norm().clamp_min(1e-12)
                    row = {
                        **target_meta,
                        "layer": int(layer_index + 1),
                        "hidden_size": int(z_model.shape[-1]),
                        "intermediate_size": _intermediate_size(independent["adapter"]),
                        "num_attention_heads": int(
                            independent["adapter"].num_attention_heads
                        ),
                        "num_key_value_heads": int(
                            independent["adapter"].num_key_value_heads
                        ),
                        "head_dim": int(independent["adapter"].head_dim),
                        "visual_tokens": int(prepared["visual_count"]),
                        "model_dtype": str(z_model.dtype),
                        "model_eval": bool(not layer.training),
                        "z_requires_grad": bool(z_model.requires_grad),
                        "response_requires_grad_after_detach": bool(
                            model_responses.requires_grad
                        ),
                        "ffn_recompute_relative_error": float(
                            _relative_error(o_ffn_recomputed, o_ffn_actual)[
                                target_offset
                            ]
                        ),
                        "ffn_recompute_cosine": float(
                            _cosine(o_ffn_recomputed, o_ffn_actual)[target_offset]
                        ),
                        "attention_reconstruction_relative_error": float(
                            independent["reconstruction_relative_error"][target_offset]
                        ),
                        "attention_reconstruction_cosine": float(
                            independent["reconstruction_cosine"][target_offset]
                        ),
                        "component_sum_relative_error_recomputed": float(
                            independent["component_sum_relative_error"][target_offset]
                        ),
                        "component_sum_relative_error_original_field": float(
                            production["component_sum_relative_error"]
                        ),
                        "production_a_tokens_relative_error": float(
                            _relative_error(
                                production["a_tokens"][:, target_offset, :],
                                a_tokens_model[:, target_offset, :],
                            ).mean()
                        ),
                        "production_a_visual_relative_error": float(
                            _relative_error(
                                production["a_visual"][target_offset].unsqueeze(0),
                                a_visual_model[target_offset].unsqueeze(0),
                            ).item()
                        ),
                        "output_bias_norm": float(independent["output_bias_norm"]),
                        "visual_attention_mass": float(
                            independent["attention_visual_mass"][target_offset]
                        ),
                        "visual_input_norm": float(visual_norm[target_offset]),
                        "visual_response_norm_fp32": float(delta_total.norm()),
                        "visual_gain_fp32": float(visual_gain),
                        "visual_direction_cosine_fp32": float(
                            _cosine(
                                delta_total.unsqueeze(0),
                                a_visual32[target_offset].unsqueeze(0),
                            ).item()
                        ),
                        "visual_input_over_z_norm": float(
                            visual_norm[target_offset]
                            / z32[target_offset].norm().clamp_min(1e-12)
                        ),
                        "visual_input_over_normalized_z_norm": float(
                            visual_norm[target_offset]
                            / normalized32[target_offset].norm().clamp_min(1e-12)
                        ),
                        "visual_response_over_ffn_output_norm": float(
                            delta_total.norm()
                            / center32[target_offset].norm().clamp_min(1e-12)
                        ),
                        "model_vs_fp32_response_relative_error": float(
                            _relative_error(
                                model_response_sum[target_offset].unsqueeze(0),
                                delta_total.unsqueeze(0),
                            ).item()
                        ),
                        "model_vs_fp32_response_cosine": float(
                            _cosine(
                                model_response_sum[target_offset].unsqueeze(0),
                                delta_total.unsqueeze(0),
                            ).item()
                        ),
                        "model_vs_fp32_p_js": float(
                            _js(
                                model_p[target_offset].unsqueeze(0),
                                fp32_p[target_offset].unsqueeze(0),
                            ).item()
                        ),
                        "fp32_token_additivity_relative_error": float(
                            _relative_error(
                                fp32_sum[target_offset].unsqueeze(0),
                                delta_total.unsqueeze(0),
                            ).item()
                        ),
                        "fp32_scalar_linearity_relative_error": float(
                            scalar_linearity[target_offset]
                        ),
                        "attention_energy_pearson": _safe_corr(
                            attention_values.detach().cpu().numpy(),
                            energy_per_token.detach().cpu().numpy(),
                        ),
                        "attention_energy_spearman": _safe_spearman(
                            attention_values.detach().cpu().numpy(),
                            energy_per_token.detach().cpu().numpy(),
                        ),
                        "attention_input_norm_pearson": _safe_corr(
                            attention_values.detach().cpu().numpy(),
                            input_per_token.detach().cpu().numpy(),
                        ),
                        "input_energy_pearson": _safe_corr(
                            input_per_token.detach().cpu().numpy(),
                            energy_per_token.detach().cpu().numpy(),
                        ),
                        "per_token_gain_mean": float(gain_per_token.mean()),
                        "per_token_gain_std": float(gain_per_token.std()),
                        "signed_contribution_sum": float(signed.sum()),
                        "signed_contribution_abs_sum": float(signed.abs().sum()),
                        "signed_contribution_negative_fraction": float(
                            (signed < 0).float().mean()
                        ),
                        "response_cancellation_ratio": float(cancellation_ratio),
                        "text_gain_fp32": float(text_gain),
                        "nonvisual_gain_fp32": float(nonvisual_gain),
                        "random_gain_mean": float(random_here.mean()),
                        "random_gain_std": float(random_here.std()),
                        "random_gain_median": float(random_here.median()),
                        "visual_gain_over_random_median": float(
                            visual_gain / random_here.median().clamp_min(1e-12)
                        ),
                        "visual_gain_random_percentile": float(
                            (random_here <= visual_gain).float().mean()
                        ),
                        "jvp_seconds_model_dtype_layer_batch": float(model_seconds),
                        "jvp_increment_mib_model_dtype": (
                            model_peak - jvp_before
                        )
                        / 2**20,
                    }
                    case_rows.append(row)

                    if image_offset == 0:
                        baseline_profiles[(int(response_index), int(layer_index))] = {
                            "p": model_p[target_offset].detach().cpu(),
                            "gain": float(
                                model_response_sum[target_offset].norm()
                                / a_visual_model[target_offset]
                                .float()
                                .norm()
                                .clamp_min(1e-12)
                            ),
                            "attention_mass": float(
                                independent["attention_visual_mass"][target_offset]
                            ),
                        }

                    for token_index in range(int(prepared["visual_count"])):
                        token_rows.append(
                            {
                                "image_id": int(image_id),
                                "response_index": int(response_index),
                                "labels": target_meta["labels"],
                                "layer": int(layer_index + 1),
                                "visual_token_index": int(token_index),
                                "attention_mean": float(attention_values[token_index]),
                                "input_norm": float(input_per_token[token_index]),
                                "response_energy": float(energy_per_token[token_index]),
                                "per_token_gain": float(gain_per_token[token_index]),
                                "signed_contribution": float(signed[token_index]),
                            }
                        )

                    for epsilon in DEFAULT_EPSILONS:
                        with torch.no_grad():
                            plus = function32(
                                z32[target_offset : target_offset + 1]
                                + float(epsilon)
                                * a_visual32[target_offset : target_offset + 1]
                            )
                            minus = function32(
                                z32[target_offset : target_offset + 1]
                                - float(epsilon)
                                * a_visual32[target_offset : target_offset + 1]
                            )
                            center = center32[target_offset : target_offset + 1]
                        central = (plus - minus) / (2.0 * float(epsilon))
                        forward = (plus - center) / float(epsilon)
                        reference = delta_total.unsqueeze(0)
                        epsilon_rows.append(
                            {
                                "image_id": int(image_id),
                                "response_index": int(response_index),
                                "labels": target_meta["labels"],
                                "layer": int(layer_index + 1),
                                "epsilon": float(epsilon),
                                "relative_input_perturbation": float(
                                    float(epsilon)
                                    * visual_norm[target_offset]
                                    / z32[target_offset].norm().clamp_min(1e-12)
                                ),
                                "central_relative_error": float(
                                    _relative_error(central, reference).item()
                                ),
                                "central_cosine": float(
                                    _cosine(central, reference).item()
                                ),
                                "forward_relative_error": float(
                                    _relative_error(forward, reference).item()
                                ),
                                "forward_cosine": float(
                                    _cosine(forward, reference).item()
                                ),
                            }
                        )

                    for removed_fraction in DEFAULT_REMOVAL_FRACTIONS:
                        with torch.no_grad():
                            removed = function32(
                                z32[target_offset : target_offset + 1]
                                - float(removed_fraction)
                                * a_visual32[target_offset : target_offset + 1]
                            )
                        exact_change = center32[target_offset : target_offset + 1] - removed
                        linear_change = float(removed_fraction) * delta_total.unsqueeze(0)
                        lambda_rows.append(
                            {
                                "image_id": int(image_id),
                                "response_index": int(response_index),
                                "labels": target_meta["labels"],
                                "layer": int(layer_index + 1),
                                "removed_fraction": float(removed_fraction),
                                "relative_input_perturbation": float(
                                    float(removed_fraction)
                                    * visual_norm[target_offset]
                                    / z32[target_offset].norm().clamp_min(1e-12)
                                ),
                                "exact_change_norm": float(exact_change.norm()),
                                "linear_change_norm": float(linear_change.norm()),
                                "linear_relative_error": float(
                                    _relative_error(linear_change, exact_change).item()
                                ),
                                "linear_cosine": float(
                                    _cosine(linear_change, exact_change).item()
                                ),
                            }
                        )

                if image_offset == 0 and layer_index == layer_indices[0]:
                    # Warmed full-vmap / chunked / small sequential timing.  The
                    # one-by-one estimate is based on 8 real visual directions.
                    torch.cuda.synchronize(device)
                    start = time.perf_counter()
                    full_parallel, _ = exact_visual_token_jvps(
                        layer=layer,
                        z=z_model,
                        a_tokens=a_tokens_model,
                        chunk_size=None,
                    )
                    full_seconds = time.perf_counter() - start
                    torch.cuda.synchronize(device)
                    start = time.perf_counter()
                    chunked, _ = exact_visual_token_jvps(
                        layer=layer,
                        z=z_model,
                        a_tokens=a_tokens_model,
                        chunk_size=int(args.jvp_chunk_size),
                    )
                    chunked_seconds = time.perf_counter() - start
                    torch.cuda.synchronize(device)
                    start = time.perf_counter()
                    sequential_parts = []
                    for token_index in range(8):
                        value, _ = exact_visual_token_jvps(
                            layer=layer,
                            z=z_model,
                            a_tokens=a_tokens_model[token_index : token_index + 1],
                            chunk_size=None,
                        )
                        sequential_parts.append(value)
                    torch.cuda.synchronize(device)
                    sequential_8_seconds = time.perf_counter() - start
                    sequential = torch.cat(sequential_parts, dim=0)
                    efficiency_rows.append(
                        {
                            "image_id": int(image_id),
                            "layer": int(layer_index + 1),
                            "directions": int(a_tokens_model.shape[0]),
                            "targets": int(z_model.shape[0]),
                            "full_parallel_seconds": float(full_seconds),
                            "chunked_seconds": float(chunked_seconds),
                            "sequential_8_seconds": float(sequential_8_seconds),
                            "sequential_576_extrapolated_seconds": float(
                                sequential_8_seconds
                                * int(a_tokens_model.shape[0])
                                / 8.0
                            ),
                            "full_vs_chunk_relative_error": float(
                                (full_parallel.float() - chunked.float()).norm()
                                / full_parallel.float().norm().clamp_min(1e-12)
                            ),
                            "first8_vs_sequential_relative_error": float(
                                (
                                    full_parallel[:8].float()
                                    - sequential.float()
                                ).norm()
                                / full_parallel[:8].float().norm().clamp_min(1e-12)
                            ),
                        }
                    )
                    del full_parallel, chunked, sequential, sequential_parts

                del (
                    norm32,
                    ffn32,
                    function32,
                    z32,
                    a_tokens32,
                    a_visual32,
                    fp32_responses,
                    fp32_aggregate,
                    fp32_sum,
                    fp32_energy,
                    fp32_p,
                    center32,
                    normalized32,
                    text32,
                    nonvisual32,
                    text_response,
                    nonvisual_response,
                    random_directions,
                    random_responses,
                    random_gain,
                    independent,
                    production,
                    model_responses,
                    model_energy,
                    model_p,
                    model_response_sum,
                    z_model,
                    o_ffn_actual,
                    o_ffn_recomputed,
                    a_tokens_model,
                    a_visual_model,
                )
                torch.cuda.empty_cache()

            _release_prepared(prepared)
            image.close()
            print(
                f"[audit] image {image_offset + 1}/{len(image_ids)}: {image_id}",
                flush=True,
            )
        except Exception as exc:
            failures.append({"image_id": int(image_id), "error": repr(exc)})
            raise

    if bool(args.run_image_controls):
        if base_image_for_controls is None:
            raise RuntimeError("Missing base image for controls")
        unrelated_id = int(image_ids[-1])
        unrelated_path = images_dir / f"COCO_val2014_{unrelated_id:012d}.jpg"
        with Image.open(unrelated_path) as unrelated_source:
            image_control_rows = _image_control_checks(
                wrapper=wrapper,
                prompt=prompt,
                base_image_id=base_image_id,
                base_image=base_image_for_controls,
                unrelated_image=unrelated_source.convert("RGB"),
                response_ids=base_response_ids,
                response_indices=base_response_indices,
                layer_indices=layer_indices,
                special_token_ids=special_token_ids,
                baseline_profiles=baseline_profiles,
                chunk_size=int(args.jvp_chunk_size),
                seed=int(args.seed),
            )
        base_image_for_controls.close()

    metadata = {
        "protocol": "independent_jacobian_visual_ffn_audit_v1",
        "model": "llava_1_5_7b",
        "image_ids": image_ids,
        "layers": requested_layers,
        "num_targets": len(target_rows),
        "num_case_rows": len(case_rows),
        "num_token_rows": len(token_rows),
        "model_allocated_gib": model_allocated / 2**30,
        "model_reserved_gib": model_reserved / 2**30,
        "formal_model_dtype": str(next(wrapper.model.parameters()).dtype),
        "validation_dtype": "torch.float32 temporary Norm+FFN copies",
        "production_uses_cache": False,
        "failures": failures,
        "original_implementation_defects": [
            {
                "field": "component_sum_relative_error",
                "location": "features/visual_ffn_jacobian.py:289",
                "description": "Current production helper returns literal 0.0 instead of measuring visual+nonvisual+bias decomposition.",
                "affects_energy_or_p": False,
            }
        ],
        "sequence_audit": sequence_audit,
        "targets": target_rows,
        "case_summary": _aggregate_numeric(
            case_rows,
            (
                "ffn_recompute_relative_error",
                "attention_reconstruction_relative_error",
                "component_sum_relative_error_recomputed",
                "production_a_tokens_relative_error",
                "fp32_token_additivity_relative_error",
                "fp32_scalar_linearity_relative_error",
                "model_vs_fp32_response_relative_error",
                "model_vs_fp32_p_js",
                "visual_input_over_z_norm",
                "visual_response_over_ffn_output_norm",
                "visual_gain_fp32",
                "visual_gain_over_random_median",
                "visual_gain_random_percentile",
                "attention_energy_pearson",
                "attention_energy_spearman",
                "input_energy_pearson",
                "signed_contribution_sum",
                "signed_contribution_abs_sum",
                "response_cancellation_ratio",
            ),
        ),
        "epsilon_summary": _aggregate_numeric(
            epsilon_rows,
            (
                "central_relative_error",
                "central_cosine",
                "forward_relative_error",
                "forward_cosine",
            ),
        ),
        "full_removal_summary": _aggregate_numeric(
            [row for row in lambda_rows if float(row["removed_fraction"]) == 1.0],
            ("linear_relative_error", "linear_cosine"),
        ),
        "prefix_summary": _aggregate_numeric(
            prefix_rows,
            (
                "logits_relative_error_full_vs_prefix",
                "logits_relative_error_no_cache_vs_cache",
                "hmid_relative_error",
                "a_visual_relative_error",
                "p_js_full_vs_prefix",
            ),
        )
        if prefix_rows
        else {},
        "efficiency": efficiency_rows,
    }

    _write_csv(result_dir / "jacobian_visual_ffn_cases.csv", case_rows)
    _write_csv(result_dir / "jacobian_visual_ffn_epsilon_sweep.csv", epsilon_rows)
    _write_csv(result_dir / "jacobian_visual_ffn_removal_curve.csv", lambda_rows)
    _write_csv(result_dir / "jacobian_visual_ffn_token_diagnostics.csv", token_rows)
    _write_csv(result_dir / "jacobian_visual_ffn_prefix_cache.csv", prefix_rows)
    _write_csv(result_dir / "jacobian_visual_ffn_image_controls.csv", image_control_rows)
    _write_csv(result_dir / "jacobian_visual_ffn_efficiency.csv", efficiency_rows)
    metrics_path = result_dir / "jacobian_visual_ffn_validation_metrics.json"
    metrics_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    _plot_diagnostics(
        epsilon_rows=epsilon_rows,
        case_rows=case_rows,
        image_control_rows=image_control_rows,
        path=result_dir / "jacobian_visual_ffn_validation_diagnostics.png",
    )
    print(f"[audit] wrote {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
