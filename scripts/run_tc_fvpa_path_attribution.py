#!/usr/bin/env python3
"""Sharded real-model local-Riesz and finite-path attribution extraction.

The four primary VLMs share the same exact-token causal-prefix contract while
retaining architecture-specific prompt construction and visual-token ranges.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.ffn_visual_path_attribution import (  # noqa: E402
    batched_ffn_jvps,
    batched_scalar_gradients,
    ffn_pullback,
    quadrature_rule,
    signed_mass_statistics,
    target_scalar_from_logits,
    vector_path_components,
)
from features.qwen_fp32_suffix import (  # noqa: E402
    QWEN_MODELS,
    qwen2_cached_text_replay_inputs,
    qwen_native_full_row_suffix_gradients,
)
from features.tc_fvpa_artifacts import (  # noqa: E402
    ExperimentLayout,
    atomic_json_save,
    initialize_manifests,
    update_stage_status,
    write_case_shard,
    write_token_map_shard,
)
from features.visual_ffn_jacobian import (  # noqa: E402
    reconstruct_visual_directions,
    resolve_decoder_layer_adapter,
)
from models import build_model  # noqa: E402
from models.dgst_capture import (  # noqa: E402
    resolve_decoder_final_norm,
    resolve_decoder_layers,
    resolve_output_embedding_layer,
)
from scripts.run_jffn_p_comparison import (  # noqa: E402
    _available_image_ids,
    _image_path,
    _load_inputs,
    build_spatial_context,
    patch_overlap_fraction,
)
from scripts.run_jffn_second_round_logit_causal import (  # noqa: E402
    capture_clean,
    prepare_inputs,
    target_groups,
)
from scripts.tc_fvpa_common import (  # noqa: E402
    add_common_arguments,
    exact_command,
    input_paths_for_model,
    print_dry_run,
    shell_command,
    validate_common_args,
)
from utils.config_utils import get_extraction_model_cfg, load_config  # noqa: E402


SUPPORTED_MODELS = {
    "llava_1_5_7b",
    "internvl_2_5_8b",
    "qwen2_5_vl_7b",
    "qwen3_vl_8b",
}
EPS = 1e-12
QWEN_SUFFIX_PARITY = {
    torch.bfloat16: {
        "full_vocab_abs": 1.0,
        "selected_abs": 0.5,
        "scalar_abs": 0.5,
        "gradient_relative": 0.05,
        "gradient_cosine": 0.999,
    },
    torch.float16: {
        "full_vocab_abs": 0.25,
        "selected_abs": 0.10,
        "scalar_abs": 0.10,
        "gradient_relative": 0.02,
        "gradient_cosine": 0.9995,
    },
}
QWEN_PREVALIDATED_SUFFIX_LAYERS = {
    # Frozen from the multilayer real-model parity gate before formal runs.
    # Qwen2 remains on its original per-case gradient gate because route
    # acceptance varies with the prefix; final layers use the exact suffix.
    "qwen2_5_vl_7b": set(),
    "qwen3_vl_8b": {9, 18, 27},
}


def parse_args() -> argparse.Namespace:
    parser = add_common_arguments(
        argparse.ArgumentParser(description=__doc__), include_integration=True
    )
    parser.add_argument("--num-images", type=int, default=0)
    parser.add_argument("--max-targets-per-image", type=int, default=2)
    parser.add_argument(
        "--quadrature", default="trapezoid,gauss_legendre"
    )
    parser.add_argument("--audit-vector-cases", type=int, default=1)
    parser.add_argument(
        "--path-batch-size",
        type=int,
        default=1,
        help=(
            "Maximum number of unique path nodes evaluated together by an exact "
            "suffix adapter. The conservative default preserves native "
            "low-precision GEMM shapes across models."
        ),
    )
    parser.add_argument(
        "--attribution-mode",
        choices=("local", "path", "both"),
        default="both",
        help="Run local Riesz only, finite path/LOO (including local), or the legacy combined route.",
    )
    parser.add_argument("--prompt", default="")
    return parser.parse_args()


def _selected_images(labels, generations, splits, *, count: int, seed: int) -> list[int]:
    available = _available_image_ids(labels, generations, splits)
    rng = random.Random(int(seed))
    rng.shuffle(available)

    def attributes(image_id: int):
        response = generations[image_id].get("response_token_ids") or ()
        groups = target_groups(labels[image_id], response)
        hall = any(
            int(span["label"]) == 0
            for spans in groups.values()
            for span in spans
        )
        real = any(
            int(span["label"]) == 1
            for spans in groups.values()
            for span in spans
        )
        return hall, real

    # A deterministic round-robin keeps hallucination-containing and REAL-only
    # images represented without looking at intervention outcomes.
    buckets = {"both": [], "hall": [], "real": []}
    for image_id in available:
        hall, real = attributes(image_id)
        key = "both" if hall and real else ("hall" if hall else "real")
        buckets[key].append(image_id)
    selected = []
    while len(selected) < count and any(buckets.values()):
        for key in ("both", "hall", "real"):
            if buckets[key] and len(selected) < count:
                selected.append(buckets[key].pop())
    return selected


def _choose_target_indices(
    groups: Mapping[int, Sequence[Mapping[str, Any]]], limit: int
) -> list[int]:
    selected = []
    for label in (0, 1):
        candidates = [
            index
            for index, spans in sorted(groups.items())
            if any(int(span["label"]) == label for span in spans)
        ]
        if candidates and candidates[0] not in selected:
            selected.append(candidates[0])
    for index in sorted(groups):
        if len(selected) >= int(limit):
            break
        if index not in selected:
            selected.append(index)
    return selected[: int(limit)]


def downstream_gradients_at_replacement(
    *,
    model: Any,
    layer: Any,
    inputs: Mapping[str, torch.Tensor],
    prediction_position: int,
    target_token_id: int,
    competitor_token_id: int,
    replacement: torch.Tensor,
    target_scalars: Sequence[str],
) -> tuple[dict[str, torch.Tensor], dict[str, float], torch.Tensor]:
    """Replace one FFN branch row and differentiate all requested scalars."""
    model_device = next(model.parameters()).device
    index = torch.tensor([int(prediction_position)], device=model_device)
    leaf = replacement.detach().clone().requires_grad_(True)

    def hook(_module, _args, output):
        if output.ndim != 3 or int(output.shape[0]) != 1:
            raise ValueError(f"Unexpected FFN output shape {tuple(output.shape)}")
        row = leaf.reshape(1, 1, -1).to(device=output.device, dtype=output.dtype)
        return output.index_copy(1, index.to(output.device), row)

    handle = resolve_decoder_layer_adapter(layer).ffn.register_forward_hook(hook)
    try:
        with torch.enable_grad():
            output = model(
                **inputs,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
                use_cache=False,
            )
            logits = output.logits[0, int(prediction_position)]
            scalar_tensors = {
                scalar: target_scalar_from_logits(
                    logits.float(),
                    target_token_id=int(target_token_id),
                    competitor_token_id=int(competitor_token_id),
                    scalar=scalar,
                )
                for scalar in target_scalars
            }
            gradients = batched_scalar_gradients(scalar_tensors, leaf)
            values = {
                scalar: float(value.detach().cpu())
                for scalar, value in scalar_tensors.items()
            }
            logits_detached = logits.detach().float().cpu()
        return gradients, values, logits_detached
    finally:
        handle.remove()


def downstream_scores_at_replacement(
    *,
    model: Any,
    layer: Any,
    inputs: Mapping[str, torch.Tensor],
    prediction_position: int,
    target_token_id: int,
    competitor_token_id: int,
    replacement: torch.Tensor,
    target_scalars: Sequence[str],
) -> tuple[dict[str, float], torch.Tensor]:
    """Evaluate a replacement without constructing three unused VJP graphs."""
    model_device = next(model.parameters()).device
    index = torch.tensor([int(prediction_position)], device=model_device)
    fixed = replacement.detach()

    def hook(_module, _args, output):
        if output.ndim != 3 or int(output.shape[0]) != 1:
            raise ValueError(f"Unexpected FFN output shape {tuple(output.shape)}")
        row = fixed.reshape(1, 1, -1).to(device=output.device, dtype=output.dtype)
        return output.index_copy(1, index.to(output.device), row)

    handle = resolve_decoder_layer_adapter(layer).ffn.register_forward_hook(hook)
    try:
        with torch.no_grad():
            output = model(
                **inputs,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
                use_cache=False,
            )
            logits = output.logits[0, int(prediction_position)].float()
            values = {
                scalar: float(
                    target_scalar_from_logits(
                        logits,
                        target_token_id=int(target_token_id),
                        competitor_token_id=int(competitor_token_id),
                        scalar=scalar,
                    ).cpu()
                )
                for scalar in target_scalars
            }
        return values, logits.cpu()
    finally:
        handle.remove()


def final_block_gradients_at_replacements(
    *,
    z: torch.Tensor,
    final_norm: Any,
    output_embedding: Any,
    target_token_id: int,
    competitor_token_id: int,
    replacements: torch.Tensor,
    target_scalars: Sequence[str],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
    """Exact final-block suffix: ``z + replacement -> norm -> LM head``.

    There are no decoder blocks after the final FFN.  Replaying the image tower,
    multimodal projector and all decoder layers for every path node is therefore
    mathematically redundant; this route preserves autograd while evaluating
    only the true suffix.
    """
    leaf = replacements.detach().clone()
    if leaf.ndim == 1:
        leaf = leaf.unsqueeze(0)
    leaf.requires_grad_(True)
    with torch.enable_grad():
        logits = output_embedding(final_norm(z.unsqueeze(0) + leaf))
        scalar_tensors = {
            scalar: torch.stack(
                [
                    target_scalar_from_logits(
                        row.float(),
                        target_token_id=int(target_token_id),
                        competitor_token_id=int(competitor_token_id),
                        scalar=scalar,
                    )
                    for row in logits
                ]
            )
            for scalar in target_scalars
        }
        gradients = batched_scalar_gradients(scalar_tensors, leaf)
        values = {
            scalar: value.detach()
            for scalar, value in scalar_tensors.items()
        }
    return gradients, values, logits.detach().float().cpu()


def final_block_gradients_at_replacement(
    *,
    z: torch.Tensor,
    final_norm: Any,
    output_embedding: Any,
    target_token_id: int,
    competitor_token_id: int,
    replacement: torch.Tensor,
    target_scalars: Sequence[str],
) -> tuple[dict[str, torch.Tensor], dict[str, float], torch.Tensor]:
    gradients, values, logits = final_block_gradients_at_replacements(
        z=z,
        final_norm=final_norm,
        output_embedding=output_embedding,
        target_token_id=target_token_id,
        competitor_token_id=competitor_token_id,
        replacements=replacement,
        target_scalars=target_scalars,
    )
    return (
        {name: value[0] for name, value in gradients.items()},
        {name: float(value[0]) for name, value in values.items()},
        logits[0],
    )


def final_block_scores_at_replacement(
    *,
    z: torch.Tensor,
    final_norm: Any,
    output_embedding: Any,
    target_token_id: int,
    competitor_token_id: int,
    replacement: torch.Tensor,
    target_scalars: Sequence[str],
) -> tuple[dict[str, float], torch.Tensor]:
    with torch.no_grad():
        logits = output_embedding(final_norm(z + replacement.detach())).float()
        values = {
            scalar: float(
                target_scalar_from_logits(
                    logits,
                    target_token_id=int(target_token_id),
                    competitor_token_id=int(competitor_token_id),
                    scalar=scalar,
                ).cpu()
            )
            for scalar in target_scalars
        }
    return values, logits.cpu()


def _normalized(values: torch.Tensor) -> torch.Tensor:
    finite = torch.nan_to_num(values.float(), nan=0.0, posinf=0.0, neginf=0.0)
    finite = finite.clamp_min(0.0)
    total = finite.sum()
    if float(total) <= EPS:
        return torch.full_like(finite, 1.0 / max(int(finite.numel()), 1))
    return finite / total


def _case_label(spans: Sequence[Mapping[str, Any]]) -> tuple[int, str]:
    labels = {int(span["label"]) for span in spans}
    # Conflicting official mention labels at one prediction position are rare.
    # Persist the conflict rather than resolving from any attribution outcome.
    value = 1 if 1 in labels else 0
    return value, "REAL" if value == 1 else "HALL"


def _decode_target(wrapper: Any, target_id: int) -> str:
    tokenizer = getattr(wrapper, "tokenizer", None)
    if tokenizer is None:
        processor = getattr(wrapper, "processor", None)
        tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return ""
    try:
        return str(tokenizer.decode([int(target_id)]))
    except Exception:
        return ""


def main() -> None:
    args = parse_args()
    common = validate_common_args(args)
    quadratures = [item for item in args.quadrature.split(",") if item]
    if set(quadratures) - {"trapezoid", "gauss_legendre"}:
        raise ValueError(f"Unknown quadrature list {quadratures}")
    do_path = args.attribution_mode in {"path", "both"}
    formal_default = 200 if do_path else 500
    count = int(
        args.num_images
        or (formal_default if args.formal else (3 if args.smoke else 12))
    )
    if (
        count <= 0
        or int(args.max_targets_per_image) <= 0
        or int(args.path_batch_size) <= 0
    ):
        raise ValueError(
            "num-images, max-targets-per-image, and path-batch-size must be positive"
        )
    formal_minimum = 200 if do_path else 500
    if args.formal and count < formal_minimum:
        raise ValueError(
            f"Formal {args.attribution_mode} attribution requires at least "
            f"{formal_minimum} images"
        )
    extra = {
        "num_images": count,
        "max_targets_per_image": int(args.max_targets_per_image),
        "quadrature": quadratures,
        "audit_vector_cases": int(args.audit_vector_cases),
        "path_batch_size": int(args.path_batch_size),
        "attribution_mode": args.attribution_mode,
        "target_protocol": "InsLen official first subtoken, first occurrence; target excluded from prefix",
        "competitor_rule": "fixed highest non-target token from clean logits",
        "path_baseline": "current-block zero visual write: z0=z-sum_m a_m; not a no-image baseline",
        "calibration_cohort": "none during extraction; downstream calibration must be train-image-only",
    }
    if args.dry_run:
        print_dry_run("path_attribution", common, extra)
        return

    layout = ExperimentLayout.create(common["output_dir"])
    manifest_path = layout.path("manifests/experiment_manifest.json")
    if not manifest_path.exists():
        initialize_manifests(
            layout=layout,
            repo_root=ROOT,
            experiment_config={**common, **extra},
            input_paths=input_paths_for_model(args.model, args.config),
            exact_command=exact_command(),
        )
    stage_name = "path_attribution" if do_path else "local_riesz"
    stage = f"{stage_name}:{args.model}:shard{args.shard_id}"
    resume_command = shell_command()
    cohort = "path" if do_path else "local"
    case_stem = "case" if cohort == "path" else "local_case"
    token_stem = "token_maps" if cohort == "path" else "local_token_maps"
    case_path = layout.path(
        f"shards/{case_stem}_rank{args.shard_id:02d}_shard_00000.pt"
    )
    token_path = layout.path(
        f"shards/{token_stem}_rank{args.shard_id:02d}_shard_00000.pt"
    )
    if args.resume and case_path.exists() and token_path.exists():
        update_stage_status(
            layout=layout,
            stage=stage,
            status="PASS",
            details={"resumed_existing": True, "case_shard": str(case_path)},
            resume_command=resume_command,
        )
        print(f"[TC-FVPA {args.attribution_mode}] reuse {case_path} and {token_path}")
        return
    if not args.resume and (case_path.exists() or token_path.exists()):
        raise FileExistsError("--no-resume refuses to overwrite existing shards")
    update_stage_status(
        layout=layout,
        stage=stage,
        status="RUNNING",
        details={"command": exact_command()},
        resume_command=resume_command,
    )

    if args.model not in SUPPORTED_MODELS:
        error = (
            f"{args.model} {args.attribution_mode} extraction is BLOCKED: "
            "no audited causal-prefix branch-replacement adapter is registered."
        )
        update_stage_status(
            layout=layout,
            stage=stage,
            status="BLOCKED",
            details={"error": error},
            resume_command=resume_command,
        )
        raise RuntimeError(error)

    config = load_config(args.config)
    model_root = (
        ROOT
        / "outputs"
        / args.model
        / "COCO4000-INSLEN-OFFICIAL-TARGET"
    )
    labels, generations, splits = _load_inputs(model_root)
    spatial_context = build_spatial_context(config)
    image_ids = _selected_images(
        labels, generations, splits, count=count, seed=int(args.seed)
    )
    image_ids = image_ids[int(args.shard_id) :: int(args.num_shards)]
    wrapper = None
    case_rows: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    audit_vectors = []
    start_time = time.perf_counter()
    try:
        wrapper = build_model(
            args.model,
            get_extraction_model_cfg(config, args.model),
            device=args.device,
        )
        wrapper.model.requires_grad_(False)
        wrapper.model.eval()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(torch.device(args.device))
        layers = resolve_decoder_layers(wrapper.model)
        layer_numbers = common["layers"]
        if max(layer_numbers) > len(layers):
            raise ValueError(
                f"Requested layer {max(layer_numbers)} for {len(layers)}-layer model"
            )
        prompt = args.prompt or str(
            (config.get("run") or {}).get("prompt") or "Describe this image."
        )
        output_embedding = resolve_output_embedding_layer(wrapper.model)
        final_norm = resolve_decoder_final_norm(wrapper.model)
        vector_audits_remaining = int(args.audit_vector_cases)

        for image_offset, image_id in enumerate(image_ids, 1):
            response_ids = [
                int(value)
                for value in generations[image_id]["response_token_ids"]
            ]
            groups = target_groups(labels[image_id], response_ids)
            selected_targets = _choose_target_indices(
                groups, int(args.max_targets_per_image)
            )
            try:
                from PIL import Image

                with Image.open(_image_path(config, image_id)) as source:
                    image = source.convert("RGB")
                for response_index in selected_targets:
                    spans = groups[response_index]
                    label_integer, label_string = _case_label(spans)
                    canonical_object = str(
                        spans[0].get("canonical_object") or ""
                    )
                    target_id = int(response_ids[response_index])
                    prefix = response_ids[:response_index]
                    inputs, prediction_position, visual_start, visual_end, grid = (
                        prepare_inputs(
                            wrapper, args.model, image, prefix, prompt
                        )
                    )
                    replay_inputs = (
                        qwen2_cached_text_replay_inputs(
                            model=wrapper.model, inputs=inputs
                        )
                        if args.model == "qwen2_5_vl_7b"
                        else inputs
                    )
                    captures, clean_logits = capture_clean(
                        wrapper, replay_inputs, prediction_position
                    )
                    competing = clean_logits.clone()
                    competing[target_id] = -torch.inf
                    competitor_id = int(competing.argmax())
                    if response_index != len(prefix):
                        raise AssertionError("Target token entered its own prefix")

                    for layer_number in layer_numbers:
                        case_started = time.perf_counter()
                        try:
                            layer = layers[layer_number - 1]
                            capture = captures[layer_number - 1]
                            adapter = resolve_decoder_layer_adapter(layer)
                            directions = reconstruct_visual_directions(
                                layer=layer,
                                capture=capture,
                                prediction_positions=[prediction_position],
                                visual_start=visual_start,
                                visual_end=visual_end,
                            )
                            writes = directions["a_tokens"][:, 0].contiguous()
                            z = capture["h_mid"][0, prediction_position].to(
                                device=writes.device, dtype=writes.dtype
                            )
                            aggregate = writes.sum(dim=0)
                            z0 = z - aggregate

                            def ffn_map(value: torch.Tensor) -> torch.Tensor:
                                if adapter.family == "separate_qkv":
                                    return layer.mlp(layer.post_attention_layernorm(value))
                                return layer.feed_forward(layer.ffn_norm(value))

                            clean_m = ffn_map(z)
                            baseline_m = ffn_map(z0)
                            pre_visual_states = capture["h_prev"][
                                0, visual_start:visual_end
                            ].to(device=z.device, dtype=torch.float32)
                            post_visual_states = capture["h_mid"][
                                0, visual_start:visual_end
                            ].to(device=z.device, dtype=torch.float32)

                            def full_downstream(replacement: torch.Tensor):
                                return downstream_gradients_at_replacement(
                                    model=wrapper.model,
                                    layer=layer,
                                    inputs=replay_inputs,
                                    prediction_position=prediction_position,
                                    target_token_id=target_id,
                                    competitor_token_id=competitor_id,
                                    replacement=replacement,
                                    target_scalars=common["target_scalars"],
                                )

                            def full_downstream_scores(replacement: torch.Tensor):
                                return downstream_scores_at_replacement(
                                    model=wrapper.model,
                                    layer=layer,
                                    inputs=replay_inputs,
                                    prediction_position=prediction_position,
                                    target_token_id=target_id,
                                    competitor_token_id=competitor_id,
                                    replacement=replacement,
                                    target_scalars=common["target_scalars"],
                                )

                            def serial_downstream_many(replacements: torch.Tensor):
                                outputs = [
                                    downstream(replacement)
                                    for replacement in replacements
                                ]
                                return (
                                    {
                                        scalar: torch.stack(
                                            [output[0][scalar] for output in outputs]
                                        )
                                        for scalar in common["target_scalars"]
                                    },
                                    {
                                        scalar: torch.as_tensor(
                                            [output[1][scalar] for output in outputs],
                                            device=replacements.device,
                                        )
                                        for scalar in common["target_scalars"]
                                    },
                                    torch.stack([output[2] for output in outputs]),
                                )

                            downstream = full_downstream
                            downstream_many = serial_downstream_many
                            downstream_scores = full_downstream_scores
                            downstream_supports_batch = False
                            downstream_route = (
                                "qwen2_cached_multimodal_embeddings_full_text_replay"
                                if args.model == "qwen2_5_vl_7b"
                                else "full_multimodal_replay"
                            )
                            suffix_parity_max_abs_error = None
                            suffix_parity_selected_max_abs_error = None
                            suffix_parity_scalar_max_abs_error = None
                            suffix_parity_argmax_match = None
                            suffix_parity_tolerance = None
                            suffix_parity_full_vocab_tolerance = None
                            suffix_parity_gradient_max_relative_error = None
                            suffix_parity_gradient_min_cosine = None
                            suffix_parity_gradient_relative_tolerance = None
                            suffix_parity_gradient_cosine_tolerance = None
                            suffix_parity_gate_source = None
                            if layer_number == len(layers):
                                suffix_parity_gate_source = (
                                    "exact_final_block_per_case_clean_forward_gate"
                                )
                                def final_downstream(replacement: torch.Tensor):
                                    return final_block_gradients_at_replacement(
                                        z=z,
                                        final_norm=final_norm,
                                        output_embedding=output_embedding,
                                        target_token_id=target_id,
                                        competitor_token_id=competitor_id,
                                        replacement=replacement,
                                        target_scalars=common["target_scalars"],
                                    )

                                def final_downstream_many(replacements: torch.Tensor):
                                    return final_block_gradients_at_replacements(
                                        z=z,
                                        final_norm=final_norm,
                                        output_embedding=output_embedding,
                                        target_token_id=target_id,
                                        competitor_token_id=competitor_id,
                                        replacements=replacements,
                                        target_scalars=common["target_scalars"],
                                    )

                                def final_downstream_scores(replacement: torch.Tensor):
                                    return final_block_scores_at_replacement(
                                        z=z,
                                        final_norm=final_norm,
                                        output_embedding=output_embedding,
                                        target_token_id=target_id,
                                        competitor_token_id=competitor_id,
                                        replacement=replacement,
                                        target_scalars=common["target_scalars"],
                                    )

                                direct_clean = final_downstream(clean_m)
                                suffix_parity_max_abs_error = float(
                                    (
                                        direct_clean[2]
                                        - clean_logits.detach().float().cpu()
                                    )
                                    .abs()
                                    .max()
                                )
                                clean_reference_logits = (
                                    clean_logits.detach().float().cpu()
                                )
                                selected_indices = [target_id, competitor_id]
                                suffix_parity_selected_max_abs_error = float(
                                    (
                                        direct_clean[2][selected_indices]
                                        - clean_reference_logits[selected_indices]
                                    )
                                    .abs()
                                    .max()
                                )
                                clean_reference_scores = {
                                    scalar: float(
                                        target_scalar_from_logits(
                                            clean_reference_logits,
                                            target_token_id=target_id,
                                            competitor_token_id=competitor_id,
                                            scalar=scalar,
                                        )
                                    )
                                    for scalar in common["target_scalars"]
                                }
                                suffix_parity_scalar_max_abs_error = max(
                                    abs(
                                        direct_clean[1][scalar]
                                        - clean_reference_scores[scalar]
                                    )
                                    for scalar in common["target_scalars"]
                                )
                                suffix_parity_argmax_match = int(
                                    direct_clean[2].argmax()
                                ) == int(clean_reference_logits.argmax())
                                suffix_parity_tolerance = {
                                    torch.bfloat16: 0.25,
                                    torch.float16: 0.05,
                                }.get(z.dtype, 1e-4)
                                if (
                                    math.isfinite(suffix_parity_max_abs_error)
                                    and suffix_parity_max_abs_error
                                    <= suffix_parity_tolerance
                                    and suffix_parity_selected_max_abs_error
                                    <= suffix_parity_tolerance
                                    and suffix_parity_scalar_max_abs_error
                                    <= suffix_parity_tolerance
                                    and suffix_parity_argmax_match
                                ):
                                    downstream = final_downstream
                                    downstream_many = final_downstream_many
                                    downstream_scores = final_downstream_scores
                                    downstream_supports_batch = True
                                    downstream_route = "exact_final_block_suffix"
                                    clean_gradients, clean_scores, branch_clean_logits = direct_clean
                                else:
                                    clean_gradients, clean_scores, branch_clean_logits = downstream(
                                        clean_m
                                    )
                            elif args.model in QWEN_MODELS:
                                def qwen_downstream_many(replacements: torch.Tensor):
                                    gradients, values, logits, _audit = (
                                        qwen_native_full_row_suffix_gradients(
                                            model=wrapper.model,
                                            model_name=args.model,
                                            layers=layers,
                                            captures=captures,
                                            intervention_layer_index=layer_number - 1,
                                            prediction_position=prediction_position,
                                            replacements=replacements,
                                            final_norm=final_norm,
                                            output_embedding=output_embedding,
                                            inputs=inputs,
                                            target_token_id=target_id,
                                            competitor_token_id=competitor_id,
                                            target_scalars=common["target_scalars"],
                                        )
                                    )
                                    return gradients, values, logits

                                def qwen_downstream(replacement: torch.Tensor):
                                    gradients, values, logits = qwen_downstream_many(
                                        replacement.unsqueeze(0)
                                    )
                                    return (
                                        {
                                            name: value[0]
                                            for name, value in gradients.items()
                                        },
                                        {
                                            name: float(value[0])
                                            for name, value in values.items()
                                        },
                                        logits[0],
                                    )

                                def qwen_downstream_scores(replacement: torch.Tensor):
                                    _gradients, values, logits, _audit = (
                                        qwen_native_full_row_suffix_gradients(
                                            model=wrapper.model,
                                            model_name=args.model,
                                            layers=layers,
                                            captures=captures,
                                            intervention_layer_index=layer_number - 1,
                                            prediction_position=prediction_position,
                                            replacements=replacement.unsqueeze(0),
                                            final_norm=final_norm,
                                            output_embedding=output_embedding,
                                            inputs=inputs,
                                            target_token_id=target_id,
                                            competitor_token_id=competitor_id,
                                            target_scalars=common["target_scalars"],
                                            compute_gradients=False,
                                        )
                                    )
                                    return (
                                        {
                                            name: float(value[0])
                                            for name, value in values.items()
                                        },
                                        logits[0],
                                    )

                                if layer_number in QWEN_PREVALIDATED_SUFFIX_LAYERS[
                                    args.model
                                ]:
                                    # Gradient parity was frozen on the real-model
                                    # multilayer audit before formal extraction.
                                    # Repeating a full multimodal backward for every
                                    # formal case is redundant and OOMs on rare long
                                    # prefixes.  Continue to enforce per-case native
                                    # clean-logit/scalar/argmax parity.
                                    suffix_parity_gate_source = (
                                        "prevalidated_multilayer_gradient_gate_"
                                        "plus_per_case_clean_forward_gate"
                                    )
                                    direct_clean = qwen_downstream(clean_m)
                                    clean_reference_logits = (
                                        clean_logits.detach().float().cpu()
                                    )
                                    candidate_logits = direct_clean[2].float().cpu()
                                    suffix_parity_max_abs_error = float(
                                        (candidate_logits - clean_reference_logits)
                                        .abs()
                                        .max()
                                    )
                                    selected_indices = [target_id, competitor_id]
                                    suffix_parity_selected_max_abs_error = float(
                                        (
                                            candidate_logits[selected_indices]
                                            - clean_reference_logits[selected_indices]
                                        )
                                        .abs()
                                        .max()
                                    )
                                    clean_reference_scores = {
                                        scalar: float(
                                            target_scalar_from_logits(
                                                clean_reference_logits,
                                                target_token_id=target_id,
                                                competitor_token_id=competitor_id,
                                                scalar=scalar,
                                            )
                                        )
                                        for scalar in common["target_scalars"]
                                    }
                                    suffix_parity_scalar_max_abs_error = max(
                                        abs(
                                            direct_clean[1][scalar]
                                            - clean_reference_scores[scalar]
                                        )
                                        for scalar in common["target_scalars"]
                                    )
                                    suffix_parity_argmax_match = int(
                                        candidate_logits.argmax()
                                    ) == int(clean_reference_logits.argmax())
                                    tolerance = QWEN_SUFFIX_PARITY.get(
                                        z.dtype,
                                        {
                                            "full_vocab_abs": 1e-4,
                                            "selected_abs": 1e-4,
                                            "scalar_abs": 1e-4,
                                            "gradient_relative": 1e-4,
                                            "gradient_cosine": 0.99999,
                                        },
                                    )
                                    suffix_parity_tolerance = tolerance["selected_abs"]
                                    suffix_parity_full_vocab_tolerance = tolerance[
                                        "full_vocab_abs"
                                    ]
                                    suffix_parity_gradient_relative_tolerance = tolerance[
                                        "gradient_relative"
                                    ]
                                    suffix_parity_gradient_cosine_tolerance = tolerance[
                                        "gradient_cosine"
                                    ]
                                    if (
                                        math.isfinite(suffix_parity_max_abs_error)
                                        and suffix_parity_max_abs_error
                                        <= suffix_parity_full_vocab_tolerance
                                        and suffix_parity_selected_max_abs_error
                                        <= suffix_parity_tolerance
                                        and suffix_parity_scalar_max_abs_error
                                        <= tolerance["scalar_abs"]
                                        and suffix_parity_argmax_match
                                    ):
                                        downstream = qwen_downstream
                                        downstream_many = qwen_downstream_many
                                        downstream_scores = qwen_downstream_scores
                                        downstream_supports_batch = True
                                        downstream_route = (
                                            "qwen_exact_full_row_causal_suffix"
                                        )
                                        (
                                            clean_gradients,
                                            clean_scores,
                                            branch_clean_logits,
                                        ) = direct_clean
                                    else:
                                        suffix_parity_gate_source += "_fallback_full_replay"
                                        (
                                            clean_gradients,
                                            clean_scores,
                                            branch_clean_logits,
                                        ) = downstream(clean_m)
                                else:
                                    suffix_parity_gate_source = (
                                        "per_case_clean_forward_and_gradient_gate"
                                    )
                                    direct_clean = qwen_downstream(clean_m)
                                    full_clean = full_downstream(clean_m)
                                    clean_reference_logits = full_clean[2].float().cpu()
                                    candidate_logits = direct_clean[2].float().cpu()
                                    suffix_parity_max_abs_error = float(
                                        (candidate_logits - clean_reference_logits)
                                        .abs()
                                        .max()
                                    )
                                    selected_indices = [target_id, competitor_id]
                                    suffix_parity_selected_max_abs_error = float(
                                        (
                                            candidate_logits[selected_indices]
                                            - clean_reference_logits[selected_indices]
                                        )
                                        .abs()
                                        .max()
                                    )
                                    clean_reference_scores = full_clean[1]
                                    suffix_parity_scalar_max_abs_error = max(
                                        abs(
                                            direct_clean[1][scalar]
                                            - clean_reference_scores[scalar]
                                        )
                                        for scalar in common["target_scalars"]
                                    )
                                    gradient_relative_errors = []
                                    gradient_cosines = []
                                    for scalar in common["target_scalars"]:
                                        candidate_gradient = direct_clean[0][scalar].float()
                                        reference_gradient = full_clean[0][scalar].float()
                                        gradient_relative_errors.append(
                                            float(
                                                (candidate_gradient - reference_gradient).norm()
                                                / reference_gradient.norm().clamp_min(EPS)
                                            )
                                        )
                                        gradient_cosines.append(
                                            float(
                                                torch.nn.functional.cosine_similarity(
                                                    candidate_gradient.reshape(1, -1),
                                                    reference_gradient.reshape(1, -1),
                                                )[0]
                                            )
                                        )
                                    suffix_parity_gradient_max_relative_error = max(
                                        gradient_relative_errors
                                    )
                                    suffix_parity_gradient_min_cosine = min(
                                        gradient_cosines
                                    )
                                    suffix_parity_argmax_match = int(
                                        candidate_logits.argmax()
                                    ) == int(clean_reference_logits.argmax())
                                    tolerance = QWEN_SUFFIX_PARITY.get(
                                        z.dtype,
                                        {
                                            "full_vocab_abs": 1e-4,
                                            "selected_abs": 1e-4,
                                            "scalar_abs": 1e-4,
                                            "gradient_relative": 1e-4,
                                            "gradient_cosine": 0.99999,
                                        },
                                    )
                                    suffix_parity_tolerance = tolerance["selected_abs"]
                                    suffix_parity_full_vocab_tolerance = tolerance[
                                        "full_vocab_abs"
                                    ]
                                    suffix_parity_gradient_relative_tolerance = tolerance[
                                        "gradient_relative"
                                    ]
                                    suffix_parity_gradient_cosine_tolerance = tolerance[
                                        "gradient_cosine"
                                    ]
                                    if (
                                        math.isfinite(suffix_parity_max_abs_error)
                                        and suffix_parity_max_abs_error
                                        <= suffix_parity_full_vocab_tolerance
                                        and suffix_parity_selected_max_abs_error
                                        <= suffix_parity_tolerance
                                        and suffix_parity_scalar_max_abs_error
                                        <= tolerance["scalar_abs"]
                                        and suffix_parity_gradient_max_relative_error
                                        <= suffix_parity_gradient_relative_tolerance
                                        and suffix_parity_gradient_min_cosine
                                        >= suffix_parity_gradient_cosine_tolerance
                                        and suffix_parity_argmax_match
                                    ):
                                        downstream = qwen_downstream
                                        downstream_many = qwen_downstream_many
                                        downstream_scores = qwen_downstream_scores
                                        downstream_supports_batch = True
                                        downstream_route = (
                                            "qwen_exact_full_row_causal_suffix"
                                        )
                                        (
                                            clean_gradients,
                                            clean_scores,
                                            branch_clean_logits,
                                        ) = direct_clean
                                    else:
                                        (
                                            clean_gradients,
                                            clean_scores,
                                            branch_clean_logits,
                                        ) = full_clean
                            else:
                                clean_gradients, clean_scores, branch_clean_logits = downstream(
                                    clean_m
                                )
                            baseline_scores = {}
                            baseline_gradients = {}
                            if do_path:
                                baseline_gradients, baseline_scores, _ = downstream(
                                    baseline_m
                                )
                            responses = batched_ffn_jvps(
                                ffn_map, z, writes, chunk_size=None
                            )
                            write_energy = writes.float().norm(dim=-1)
                            response_energy = responses.float().norm(dim=-1)
                            gain = response_energy / write_energy.clamp_min(EPS)
                            response_total = responses.float().sum(dim=0)
                            response_unit = response_total / response_total.norm().clamp_min(EPS)
                            signed_q = responses.float() @ response_unit
                            attention = _normalized(
                                directions["visual_attention_scores"][0]
                            )
                            methods: dict[str, torch.Tensor] = {
                                "ATTN": attention.cpu(),
                                "WRITE": write_energy.cpu(),
                                "JFFN": response_energy.cpu(),
                                "GAIN": gain.cpu(),
                                "SIGNED_Q": signed_q.cpu(),
                            }
                            clean_update32 = clean_m.float().reshape(1, -1)
                            expanded_update = clean_update32.expand_as(
                                pre_visual_states
                            )
                            methods.update(
                                {
                                    "RAW_COS_FFN_PRE_MHSA": torch.nn.functional.cosine_similarity(
                                        expanded_update, pre_visual_states, dim=-1
                                    ).cpu(),
                                    "RAW_COS_FFN_POST_MHSA": torch.nn.functional.cosine_similarity(
                                        expanded_update, post_visual_states, dim=-1
                                    ).cpu(),
                                    "RAW_COS_FFN_WRITE": torch.nn.functional.cosine_similarity(
                                        expanded_update, writes.float(), dim=-1
                                    ).cpu(),
                                    "RAW_COS_RESPONSE_FFN": torch.nn.functional.cosine_similarity(
                                        responses.float(), expanded_update, dim=-1
                                    ).cpu(),
                                    "STATE_AFFINITY_CHANGE": (
                                        torch.nn.functional.cosine_similarity(
                                            (z.float() + clean_m.float())
                                            .reshape(1, -1)
                                            .expand_as(pre_visual_states),
                                            pre_visual_states,
                                            dim=-1,
                                        )
                                        - torch.nn.functional.cosine_similarity(
                                            z.float()
                                            .reshape(1, -1)
                                            .expand_as(pre_visual_states),
                                            pre_visual_states,
                                            dim=-1,
                                        )
                                    ).cpu(),
                                }
                            )
                            local_scores = {}
                            pullbacks = {}
                            duality_errors = {}
                            for scalar in common["target_scalars"]:
                                gradient = clean_gradients[scalar].to(z)
                                pullback = ffn_pullback(ffn_map, z, gradient)
                                scores = writes @ pullback
                                direct = responses.to(gradient) @ gradient
                                local_scores[scalar] = scores
                                pullbacks[scalar] = pullback
                                duality_errors[scalar] = float(
                                    (scores - direct).abs().max().detach().cpu()
                                )
                                methods[f"LOCAL_{scalar.upper()}"] = scores.float().cpu()
                            unembedding = output_embedding.weight[target_id].detach().to(
                                device=responses.device, dtype=responses.dtype
                            )
                            methods["DIRECT_UNEMBEDDING"] = (
                                responses @ unembedding
                            ).float().cpu()

                            point_cache: dict[
                                float,
                                tuple[
                                    dict[str, torch.Tensor],
                                    dict[str, torch.Tensor],
                                ],
                            ] = {
                                1.0: (
                                    clean_gradients,
                                    {
                                        scalar: pullbacks[scalar].detach()
                                        for scalar in common["target_scalars"]
                                    },
                                )
                            }
                            if do_path:
                                point_cache[0.0] = (
                                    baseline_gradients,
                                    {
                                        scalar: ffn_pullback(
                                            ffn_map,
                                            z0,
                                            baseline_gradients[scalar].to(z0),
                                        ).detach()
                                        for scalar in common["target_scalars"]
                                    },
                                )
                            convergence = {}
                            frozen_write_interventions = []
                            path_batch_oom_fallbacks = 0
                            path_effective_batch_size = (
                                int(args.path_batch_size)
                                if downstream_supports_batch
                                else 1
                            )
                            maximum_k = max(common["integration_points"])
                            if do_path:
                                path_rules = {}
                                path_nodes: dict[float, float] = {}
                                for quadrature in quadratures:
                                    for integration_points in common["integration_points"]:
                                        rule = quadrature_rule(
                                            quadrature,
                                            integration_points,
                                            device=z.device,
                                            dtype=z.dtype,
                                        )
                                        path_rules[(quadrature, integration_points)] = rule
                                        for alpha in rule.nodes:
                                            alpha_value = float(alpha)
                                            key = round(alpha_value, 15)
                                            if key not in point_cache:
                                                path_nodes.setdefault(key, alpha_value)

                                pending_nodes = list(path_nodes.items())
                                node_start = 0

                                def try_downstream_many(
                                    replacements: torch.Tensor,
                                ):
                                    # Keep the OOM exception in this short-lived
                                    # frame.  Catching it around the subsequent
                                    # single-node retry retains the failed graph
                                    # through the traceback and prevents CUDA
                                    # memory from being reclaimed.
                                    try:
                                        return downstream_many(replacements)
                                    except torch.cuda.OutOfMemoryError:
                                        return None

                                while node_start < len(pending_nodes):
                                    batch_nodes = pending_nodes[
                                        node_start : node_start
                                        + path_effective_batch_size
                                    ]
                                    points = torch.stack(
                                        [
                                            z0 + alpha_value * aggregate
                                            for _key, alpha_value in batch_nodes
                                        ]
                                    )
                                    replacements = ffn_map(points)
                                    batch_output = try_downstream_many(replacements)
                                    if batch_output is None:
                                        if path_effective_batch_size <= 1:
                                            raise torch.cuda.OutOfMemoryError(
                                                "Path suffix exhausted CUDA memory at batch size 1"
                                            )
                                        path_batch_oom_fallbacks += 1
                                        path_effective_batch_size = 1
                                        del replacements, points, batch_nodes
                                        torch.cuda.empty_cache()
                                        continue
                                    gradients_batch, _scores, _logits = batch_output
                                    for offset, (key, _alpha_value) in enumerate(
                                        batch_nodes
                                    ):
                                        point = points[offset]
                                        gradients = {
                                            scalar: gradients_batch[scalar][offset]
                                            for scalar in common["target_scalars"]
                                        }
                                        pulled = {
                                            scalar: ffn_pullback(
                                                ffn_map,
                                                point,
                                                gradients[scalar].to(point),
                                            ).detach()
                                            for scalar in common["target_scalars"]
                                        }
                                        point_cache[key] = (gradients, pulled)
                                    node_start += len(batch_nodes)

                                for quadrature in quadratures:
                                    for integration_points in common["integration_points"]:
                                        rule = path_rules[
                                            (quadrature, integration_points)
                                        ]
                                        for scalar in common["target_scalars"]:
                                            scores = torch.zeros(
                                                writes.shape[0],
                                                device=z.device,
                                                dtype=z.dtype,
                                            )
                                            for alpha, weight in zip(
                                                rule.nodes, rule.weights
                                            ):
                                                key = round(float(alpha), 15)
                                                _gradients, pulled = point_cache[key]
                                                scores = scores + weight * (
                                                    writes @ pulled[scalar]
                                                )
                                            name = (
                                                f"PATH_{scalar.upper()}_"
                                                f"{quadrature.upper()}_K{integration_points}"
                                            )
                                            methods[name] = scores.float().cpu()
                                            finite = (
                                                clean_scores[scalar]
                                                - baseline_scores[scalar]
                                            )
                                            convergence[name] = {
                                                "score_sum": float(scores.sum()),
                                                "finite_effect": float(finite),
                                                "completeness_absolute_error": abs(
                                                    float(scores.sum()) - float(finite)
                                                ),
                                                "completeness_relative_error": abs(
                                                    float(scores.sum()) - float(finite)
                                                )
                                                / max(abs(float(finite)), EPS),
                                            }

                                primary_scalar = (
                                    "log_probability"
                                    if "log_probability" in common["target_scalars"]
                                    else common["target_scalars"][0]
                                )
                                primary_name = (
                                    f"PATH_{primary_scalar.upper()}_"
                                    f"GAUSS_LEGENDRE_K{maximum_k}"
                                )
                                if primary_name not in methods:
                                    primary_name = (
                                        f"PATH_{primary_scalar.upper()}_"
                                        f"{quadratures[0].upper()}_K{maximum_k}"
                                    )
                                primary_path = methods[primary_name].to(z)
                                local_primary = local_scores[primary_scalar]
                                positive_order = torch.argsort(
                                    primary_path, descending=True, stable=True
                                )
                                rng = random.Random(
                                    f"{args.seed}:{args.model}:{image_id}:"
                                    f"{response_index}:{layer_number}"
                                )
                                intervention_regions: list[tuple[str, list[int]]] = [
                                    (
                                        "aggregate_visual_write",
                                        list(range(int(writes.shape[0]))),
                                    ),
                                    ("highest_path", [int(primary_path.argmax())]),
                                    (
                                        "highest_local_riesz",
                                        [int(local_primary.argmax())],
                                    ),
                                    ("highest_write", [int(write_energy.argmax())]),
                                    ("highest_jffn", [int(response_energy.argmax())]),
                                    ("highest_attention", [int(attention.argmax())]),
                                    (
                                        "most_negative_path",
                                        [int(primary_path.argmin())],
                                    ),
                                    (
                                        "random_token",
                                        [rng.randrange(int(writes.shape[0]))],
                                    ),
                                ]
                                for top_k in (4, 16, 32):
                                    intervention_regions.append(
                                        (
                                            f"top{top_k}_positive_path",
                                            [
                                                int(value)
                                                for value in positive_order[
                                                    : min(
                                                        top_k,
                                                        int(writes.shape[0]),
                                                    )
                                                ].tolist()
                                                if float(
                                                    primary_path[int(value)]
                                                )
                                                > 0.0
                                            ],
                                        )
                                    )
                                loo_score_cache: dict[
                                    tuple[int, ...], dict[str, float]
                                ] = {
                                    tuple(range(int(writes.shape[0]))): baseline_scores
                                }
                                for strategy, region in intervention_regions:
                                    region = sorted(set(region))
                                    if not region:
                                        frozen_write_interventions.append(
                                            {
                                                "strategy": strategy,
                                                "measurement_status": "NOT_RUN",
                                                "error": (
                                                    "No positive path tokens in requested region"
                                                ),
                                            }
                                        )
                                        continue
                                    region_key = tuple(region)
                                    if region_key not in loo_score_cache:
                                        region_write = writes[region].sum(dim=0)
                                        replacement = ffn_map(z - region_write)
                                        loo_scores, _loo_logits = (
                                            downstream_scores(replacement)
                                        )
                                        loo_score_cache[region_key] = loo_scores
                                    loo_scores = loo_score_cache[region_key]
                                    frozen_write_interventions.append(
                                        {
                                            "strategy": strategy,
                                            "token_indices": region,
                                            "region_size": len(region),
                                            "predicted_path_effect": {
                                                scalar: float(
                                                    methods[
                                                        f"PATH_{scalar.upper()}_"
                                                        f"GAUSS_LEGENDRE_K{maximum_k}"
                                                        if (
                                                            f"PATH_{scalar.upper()}_"
                                                            f"GAUSS_LEGENDRE_K{maximum_k}"
                                                        )
                                                        in methods
                                                        else (
                                                            f"PATH_{scalar.upper()}_"
                                                            f"{quadratures[0].upper()}_"
                                                            f"K{maximum_k}"
                                                        )
                                                    ][region].sum()
                                                )
                                                for scalar in common["target_scalars"]
                                            },
                                            "observed_frozen_write_effect": {
                                                scalar: float(
                                                    clean_scores[scalar]
                                                    - loo_scores[scalar]
                                                )
                                                for scalar in common["target_scalars"]
                                            },
                                            "measurement_status": "MEASURED",
                                        }
                                    )

                            box_overlap = None
                            box_overlap_status = "NOT_APPLICABLE_HALL"
                            if label_integer == 1:
                                target_boxes = spatial_context["boxes"].get(
                                    (int(image_id), canonical_object), ()
                                )
                                image_size = spatial_context["image_size"].get(
                                    int(image_id)
                                )
                                if target_boxes and image_size is not None:
                                    box_overlap = patch_overlap_fraction(
                                        model=args.model,
                                        image_size=image_size,
                                        boxes=target_boxes,
                                        grid=(int(grid[0]), int(grid[1])),
                                    )
                                    box_overlap_status = "MEASURED"
                                else:
                                    box_overlap_status = (
                                        "BLOCKED_NO_MATCHING_COCO_BOX"
                                    )

                            vector_audit = None
                            if do_path and vector_audits_remaining > 0:
                                vector_result = vector_path_components(
                                    ffn_map=ffn_map,
                                    z=z,
                                    writes=writes,
                                    method="gauss_legendre",
                                    integration_points=maximum_k,
                                )
                                vector_audit = {
                                    "z": z.detach().float().cpu(),
                                    "z0": z0.detach().float().cpu(),
                                    "writes": writes.detach().float().cpu(),
                                    "local_responses": responses.detach().float().cpu(),
                                    "path_components": vector_result.components.detach().float().cpu(),
                                    "ffn_clean": clean_m.detach().float().cpu(),
                                    "ffn_baseline": baseline_m.detach().float().cpu(),
                                    "target_gradients": {
                                        key: value.detach().float().cpu()
                                        for key, value in clean_gradients.items()
                                    },
                                    "pullbacks": {
                                        key: value.detach().float().cpu()
                                        for key, value in pullbacks.items()
                                    },
                                    "clean_logits": branch_clean_logits,
                                    "box_overlap": box_overlap,
                                    "box_overlap_status": box_overlap_status,
                                    "vector_completeness_relative_error": vector_result.completeness_relative_error,
                                }
                                audit_vectors.append(
                                    {
                                        "case_id": f"{args.model}:{image_id}:{response_index}:{layer_number}",
                                        **vector_audit,
                                    }
                                )
                                vector_audits_remaining -= 1

                            aggregate_gain = float(
                                response_total.norm()
                                / aggregate.float().norm().clamp_min(EPS)
                            )
                            aggregate_stats = {
                                scalar: signed_mass_statistics(local_scores[scalar])
                                for scalar in common["target_scalars"]
                            }
                            clean_target_logit = float(branch_clean_logits[target_id])
                            clean_margin = float(
                                branch_clean_logits[target_id]
                                - branch_clean_logits[competitor_id]
                            )
                            clean_log_probability = float(
                                torch.log_softmax(branch_clean_logits, dim=-1)[target_id]
                            )
                            measured_fields = [
                                "clean_target_logit",
                                "clean_margin",
                                "clean_log_probability",
                                "aggregate_write_norm",
                                "aggregate_response_norm",
                                "aggregate_gain",
                            ]
                            row = {
                                "model": args.model,
                                "dataset": "COCO4000-CHAIR",
                                "image_id": int(image_id),
                                "case_id": f"{args.model}:{image_id}:{response_index}:{layer_number}",
                                "mention_id": f"{image_id}:{response_index}",
                                "response_index": int(response_index),
                                "prediction_position": int(prediction_position),
                                "prefix_excludes_target": True,
                                "target_text": _decode_target(wrapper, target_id),
                                "canonical_object": canonical_object,
                                "target_token_id": int(target_id),
                                "competitor_token_id": int(competitor_id),
                                "label_string": label_string,
                                "label_integer": label_integer,
                                "label_direction": "HALL=0,REAL=1",
                                "layer": int(layer_number),
                                "visual_token_count": int(writes.shape[0]),
                                "grid_height": int(grid[0]),
                                "grid_width": int(grid[1]),
                                "clean_target_logit": clean_target_logit,
                                "clean_margin": clean_margin,
                                "clean_log_probability": clean_log_probability,
                                "aggregate_attention_mass": float(
                                    directions["visual_attention_scores"][0].sum()
                                ),
                                "aggregate_write_norm": float(aggregate.float().norm()),
                                "aggregate_response_norm": float(response_total.norm()),
                                "aggregate_gain": aggregate_gain,
                                "direction_cosine": float(
                                    torch.nn.functional.cosine_similarity(
                                        response_total.reshape(1, -1),
                                        aggregate.float().reshape(1, -1),
                                    )
                                ),
                                "vector_cancellation": float(
                                    response_total.norm()
                                    / response_energy.sum().clamp_min(EPS)
                                ),
                                "local_signed_statistics": aggregate_stats,
                                "local_duality_max_abs_error": duality_errors,
                                "path_convergence": convergence,
                                "frozen_write_interventions": frozen_write_interventions,
                                "attention_reconstruction_relative_error": float(
                                    directions["reconstruction_relative_error"]
                                ),
                                "component_sum_relative_error": float(
                                    directions["component_sum_relative_error"]
                                ),
                                "native_dtype": str(z.dtype),
                                "downstream_dtype": str(next(wrapper.model.parameters()).dtype),
                                "attribution_mode": args.attribution_mode,
                                "downstream_route": downstream_route,
                                "suffix_parity_gate_source": suffix_parity_gate_source,
                                "path_batch_size_requested": int(args.path_batch_size),
                                "path_batch_size_effective": path_effective_batch_size,
                                "path_batch_oom_fallbacks": path_batch_oom_fallbacks,
                                "path_unique_node_count": max(len(point_cache) - 2, 0),
                                "suffix_parity_max_abs_error": suffix_parity_max_abs_error,
                                "suffix_parity_selected_max_abs_error": suffix_parity_selected_max_abs_error,
                                "suffix_parity_scalar_max_abs_error": suffix_parity_scalar_max_abs_error,
                                "suffix_parity_argmax_match": suffix_parity_argmax_match,
                                "suffix_parity_tolerance": suffix_parity_tolerance,
                                "suffix_parity_full_vocab_tolerance": suffix_parity_full_vocab_tolerance,
                                "suffix_parity_gradient_max_relative_error": suffix_parity_gradient_max_relative_error,
                                "suffix_parity_gradient_min_cosine": suffix_parity_gradient_min_cosine,
                                "suffix_parity_gradient_relative_tolerance": suffix_parity_gradient_relative_tolerance,
                                "suffix_parity_gradient_cosine_tolerance": suffix_parity_gradient_cosine_tolerance,
                                "runtime_seconds": time.perf_counter() - case_started,
                                "measurement_status": "MEASURED",
                                "measured_fields": measured_fields,
                                "available_experiments": {
                                    "local_riesz": True,
                                    "path_riesz": do_path,
                                    "fp32_downstream": False,
                                    "finite_loo": do_path,
                                    "pixel_counterfactual": False,
                                    "shapley": False,
                                },
                                "box_overlap_status": box_overlap_status,
                            }
                            token_row = {
                                "model": args.model,
                                "image_id": int(image_id),
                                "case_id": row["case_id"],
                                "response_index": int(response_index),
                                "prediction_position": int(prediction_position),
                                "target_token_id": int(target_id),
                                "label_string": label_string,
                                "label_integer": label_integer,
                                "layer": int(layer_number),
                                "visual_grid": [int(grid[0]), int(grid[1])],
                                "validity_mask": torch.ones(
                                    writes.shape[0], dtype=torch.bool
                                ),
                                "box_overlap": box_overlap,
                                "box_overlap_status": box_overlap_status,
                                "methods": methods,
                            }
                            case_rows.append(row)
                            token_rows.append(token_row)
                            del directions, writes, responses, point_cache
                        except Exception as exc:
                            failures.append(
                                {
                                    "model": args.model,
                                    "image_id": int(image_id),
                                    "response_index": int(response_index),
                                    "layer": int(layer_number),
                                    "error": repr(exc),
                                    "traceback": traceback.format_exc(),
                                }
                            )
                    for capture in captures:
                        capture.clear()
                    del captures, inputs
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            except Exception as exc:
                failures.append(
                    {
                        "model": args.model,
                        "image_id": int(image_id),
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
            print(
                f"[TC-FVPA {args.attribution_mode}] "
                f"{image_offset}/{len(image_ids)} image={image_id} "
                f"cases={len(case_rows)} failures={len(failures)}",
                flush=True,
            )

        provenance = {
            **common,
            **extra,
            "target_protocol": "inslen_official_first_token_first_occurrence",
            "label_direction": "HALL=0,REAL=1",
            "current_block_baseline": "z0=z-sum_visual_clean_value_writes",
            "cohort": cohort,
            "elapsed_seconds": time.perf_counter() - start_time,
            "peak_gpu_memory_bytes": (
                int(torch.cuda.max_memory_allocated(torch.device(args.device)))
                if torch.cuda.is_available()
                else 0
            ),
        }
        write_case_shard(
            layout=layout,
            rank=int(args.shard_id),
            shard_id=0,
            rows=case_rows,
            failures=failures,
            provenance=provenance,
            cohort=cohort,
        )
        write_token_map_shard(
            layout=layout,
            rank=int(args.shard_id),
            shard_id=0,
            rows=token_rows,
            failures=failures,
            provenance=provenance,
            cohort=cohort,
        )
        if audit_vectors:
            audit_path = layout.path(
                f"audit_vectors/{args.model}_rank{args.shard_id:02d}_audit_cases.pt"
            )
            from features.tc_fvpa_artifacts import atomic_torch_save

            atomic_torch_save(
                {
                    "model": args.model,
                    "dtype": "FP32 persisted from native computation",
                    "cases": audit_vectors,
                },
                audit_path,
            )
        status = "PASS" if case_rows and not failures else ("FAIL" if failures else "BLOCKED")
        update_stage_status(
            layout=layout,
            stage=stage,
            status=status,
            details={
                "images": len(image_ids),
                "measured_cases": len(case_rows),
                "failed_cases": len(failures),
                "elapsed_seconds": time.perf_counter() - start_time,
                "peak_gpu_memory_bytes": provenance["peak_gpu_memory_bytes"],
            },
            resume_command=resume_command,
        )
        if status != "PASS":
            raise RuntimeError(
                f"{args.attribution_mode} extraction ended with status {status}; "
                f"failures={len(failures)}"
            )
    except Exception as exc:
        current_status = json.loads(
            layout.path("manifests/run_status.json").read_text(encoding="utf-8")
        ).get("stages", {}).get(stage, {}).get("status")
        if current_status not in {"BLOCKED", "FAIL"}:
            update_stage_status(
                layout=layout,
                stage=stage,
                status="FAIL",
                details={"error": repr(exc), "traceback": traceback.format_exc()},
                resume_command=resume_command,
            )
        raise
    finally:
        if wrapper is not None:
            del wrapper
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
