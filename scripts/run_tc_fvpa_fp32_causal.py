#!/usr/bin/env python3
"""True-FP32 symmetric causal validation for the final decoder block.

For the final block, downstream execution after FFN-branch replacement is
exactly ``clean z + replacement -> final norm -> LM head``.  This launcher
copies the block FFN norm/MLP, decoder final norm, and LM head weights to true
FP32 before constructing or inserting interventions.  It does not claim that
casting already-produced low-precision logits resolves precision.

Layers below the final block require architecture-specific full-sequence
downstream decoder execution.  Until that adapter passes prefix/full parity,
requested lower layers are persisted as BLOCKED rather than approximated.
"""

from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import gzip
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.ffn_visual_path_attribution import (  # noqa: E402
    batched_ffn_jvps,
    target_scalar_from_logits,
)
from features.qwen_fp32_suffix import (  # noqa: E402
    QWEN_MODELS,
    qwen_fp32_suffix_logits_and_gradients,
)
from features.tc_fvpa_artifacts import (  # noqa: E402
    ExperimentLayout,
    atomic_json_save,
    atomic_torch_save,
    initialize_manifests,
    update_stage_status,
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
from scripts.run_jffn_p_comparison import _image_path, _load_inputs  # noqa: E402
from scripts.run_jffn_second_round_logit_causal import (  # noqa: E402
    capture_clean,
    prepare_inputs,
    target_groups,
)
from scripts.run_tc_fvpa_path_attribution import (  # noqa: E402
    SUPPORTED_MODELS,
    _case_label,
    _choose_target_indices,
    _selected_images,
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


ETAS = (0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.4)
EPS = 1e-12
PROTOCOL = "tc_fvpa_true_fp32_causal_suffix_v3_microbatched"


def parse_args() -> argparse.Namespace:
    parser = add_common_arguments(
        argparse.ArgumentParser(description=__doc__), include_integration=True
    )
    parser.add_argument("--num-images", type=int, default=0)
    parser.add_argument("--max-targets-per-image", type=int, default=2)
    parser.add_argument(
        "--logit-batch-size",
        type=int,
        default=32,
        help=(
            "Maximum batch for the FP32 vocabulary projection and Qwen "
            "intervention-query suffix replay"
        ),
    )
    parser.add_argument("--path-result-dir")
    return parser.parse_args()


def _load_path_maps(path_result_dir: Path | None) -> dict[str, torch.Tensor]:
    if path_result_dir is None or not path_result_dir.exists():
        return {}
    result = {}
    for path in sorted((path_result_dir / "shards").glob("token_maps_rank*_shard_*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        for row in payload.get("token_maps", ()):
            methods = row.get("methods") or {}
            candidates = sorted(
                name
                for name in methods
                if name.startswith("PATH_LOG_PROBABILITY_GAUSS_LEGENDRE_K")
            )
            if candidates:
                name = max(candidates, key=lambda item: int(item.rsplit("K", 1)[-1]))
                result[str(row["case_id"])] = torch.as_tensor(methods[name]).float()
    return result


def _fp32_modules(model: Any, layer: Any, device: torch.device):
    ffn_norm, ffn = _fp32_branch_modules(layer, device)
    final_norm, lm_weight, lm_bias = _fp32_output_modules(model, device)
    return ffn_norm, ffn, final_norm, lm_weight, lm_bias


def _fp32_branch_modules(layer: Any, device: torch.device):
    adapter = resolve_decoder_layer_adapter(layer)
    ffn_norm = copy.deepcopy(adapter.ffn_norm).to(device=device, dtype=torch.float32).eval()
    ffn = copy.deepcopy(adapter.ffn).to(device=device, dtype=torch.float32).eval()
    return ffn_norm, ffn


def _fp32_output_modules(model: Any, device: torch.device):
    final_norm = copy.deepcopy(resolve_decoder_final_norm(model)).to(
        device=device, dtype=torch.float32
    ).eval()
    output = resolve_output_embedding_layer(model)
    lm_weight = output.weight.detach().to(device=device, dtype=torch.float32).clone()
    lm_bias = getattr(output, "bias", None)
    lm_bias32 = (
        lm_bias.detach().to(device=device, dtype=torch.float32).clone()
        if lm_bias is not None
        else None
    )
    return final_norm, lm_weight, lm_bias32


def _all_scalar_values(
    logits: torch.Tensor, target_id: int, competitor_id: int
) -> dict[str, torch.Tensor]:
    return {
        scalar: target_scalar_from_logits(
            logits,
            target_token_id=target_id,
            competitor_token_id=competitor_id,
            scalar=scalar,
        )
        for scalar in ("log_probability", "margin", "logit")
    }


def _write_csv_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with gzip.open(temporary, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _consolidate_fp32_shards(
    *,
    layout: ExperimentLayout,
    num_shards: int,
    query_microbatch_size: int,
) -> dict[str, Any]:
    """Atomically rebuild shared CSV/metrics from rank-local PT shards."""
    lock_path = layout.path("manifests/.fp32_consolidate.lock")
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        payloads = []
        completed_ranks = []
        for rank in range(int(num_shards)):
            shard_path = layout.path(
                f"shards/fp32_causal_rank{rank:02d}_shard_00000.pt"
            )
            if not shard_path.exists():
                continue
            payload = torch.load(
                shard_path, map_location="cpu", weights_only=False
            )
            if (
                payload.get("protocol") != PROTOCOL
                or int(payload.get("query_microbatch_size", -1))
                != int(query_microbatch_size)
            ):
                continue
            payloads.append(payload)
            completed_ranks.append(rank)

        rows = [row for payload in payloads for row in payload.get("rows", ())]
        blocked_rows = [
            row
            for payload in payloads
            for row in payload.get("blocked_rows", ())
        ]
        failures = [
            failure
            for payload in payloads
            for failure in payload.get("failures", ())
        ]
        if rows or blocked_rows:
            _write_csv_gz(
                layout.path("tables/interventions.csv.gz"),
                rows + blocked_rows,
            )
        summary = {
            "protocol": PROTOCOL,
            "model": payloads[0].get("model") if payloads else None,
            "measured_rows": len(rows),
            "blocked_layer_rows": len(blocked_rows),
            "failures": failures,
            "completed_ranks": completed_ranks,
            "expected_ranks": int(num_shards),
            "all_ranks_present": len(completed_ranks) == int(num_shards),
            "aggregate_gpu_seconds": sum(
                float(payload.get("elapsed_seconds", 0.0))
                for payload in payloads
            ),
            "peak_gpu_memory_bytes": max(
                (
                    int(payload.get("peak_gpu_memory_bytes", 0))
                    for payload in payloads
                ),
                default=0,
            ),
            "query_microbatch_size": int(query_microbatch_size),
        }
        atomic_json_save(summary, layout.path("metrics/fp32_causal_summary.json"))
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return summary


def _evaluate_layer_case(
    *,
    wrapper: Any,
    model_name: str,
    layers: Any,
    layer_index: int,
    captures: list[dict[str, Any]],
    inputs: Mapping[str, Any],
    prediction_position: int,
    visual_start: int,
    visual_end: int,
    target_id: int,
    competitor_id: int,
    image_id: int,
    response_index: int,
    label_integer: int,
    label_string: str,
    scalar_names: list[str],
    path_maps: Mapping[str, torch.Tensor],
    previous_wrong_image_direction: tuple[int, torch.Tensor] | None,
    rng: random.Random,
    device: torch.device,
    final_norm32: Any,
    lm_weight32: torch.Tensor,
    lm_bias32: torch.Tensor | None,
    logit_batch_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], torch.Tensor]:
    layer = layers[layer_index]
    layer_number = layer_index + 1
    capture = captures[layer_index]
    ffn_norm32, ffn32 = _fp32_branch_modules(layer, device)

    def ffn_map32(value: torch.Tensor) -> torch.Tensor:
        return ffn32(ffn_norm32(value))

    directions = reconstruct_visual_directions(
        layer=layer,
        capture=capture,
        prediction_positions=[prediction_position],
        visual_start=visual_start,
        visual_end=visual_end,
    )
    writes32 = directions["a_tokens"][:, 0].float().to(device)
    z32 = capture["h_mid"][0, prediction_position].float().to(device)
    responses32 = batched_ffn_jvps(ffn_map32, z32, writes32)
    m0 = ffn_map32(z32).detach()

    def final_logits_from_m(batch_m: torch.Tensor) -> torch.Tensor:
        normalized = final_norm32(z32.unsqueeze(0) + batch_m)
        outputs = []
        for start in range(0, int(normalized.shape[0]), int(logit_batch_size)):
            outputs.append(
                F.linear(
                    normalized[start : start + int(logit_batch_size)],
                    lm_weight32,
                    lm_bias32,
                )
            )
        return torch.cat(outputs, dim=0)

    if layer_index == len(layers) - 1:
        m_leaf = m0.detach().requires_grad_(True)
        clean_logits32 = final_logits_from_m(m_leaf.unsqueeze(0))[0]
        scalar_tensors = _all_scalar_values(clean_logits32, target_id, competitor_id)
        gradients = {}
        for offset, scalar in enumerate(scalar_names):
            gradients[scalar] = torch.autograd.grad(
                scalar_tensors[scalar],
                m_leaf,
                retain_graph=offset + 1 < len(scalar_names),
            )[0].detach()
        downstream_audit = {
            "route": "final_block_true_fp32",
            "downstream_layer_count": 0,
            "context_recomputed_fp32": True,
        }
    elif model_name in QWEN_MODELS:
        clean_batch, gradients, downstream_audit = (
            qwen_fp32_suffix_logits_and_gradients(
                model=wrapper.model,
                model_name=model_name,
                layers=layers,
                intervention_layer_index=layer_index,
                h_mid=capture["h_mid"][0, : prediction_position + 1].float().to(device),
                prediction_position=prediction_position,
                branch_outputs=m0.unsqueeze(0),
                current_ffn_norm=ffn_norm32,
                current_ffn=ffn32,
                final_norm=final_norm32,
                lm_weight=lm_weight32,
                lm_bias=lm_bias32,
                inputs=inputs,
                target_token_id=target_id,
                competitor_token_id=competitor_id,
                target_scalars=scalar_names,
                query_batch_size=logit_batch_size,
            )
        )
        clean_logits32 = clean_batch[0]
    else:
        raise RuntimeError(
            f"No lower-layer true-FP32 suffix adapter for {model_name} layer {layer_number}"
        )

    local_primary = responses32 @ gradients[
        "log_probability" if "log_probability" in gradients else scalar_names[0]
    ]
    case_id = f"{model_name}:{image_id}:{response_index}:{layer_number}"
    path_scores = path_maps.get(case_id)
    path_ranking = path_scores.to(device) if path_scores is not None else None
    write_energy = writes32.norm(dim=-1)
    response_energy = responses32.norm(dim=-1)
    attention = directions["visual_attention_scores"][0].float().to(device)
    aggregate = responses32.sum(dim=0)
    positive = (
        responses32[local_primary > 0].sum(dim=0)
        if bool((local_primary > 0).any())
        else torch.zeros_like(aggregate)
    )
    negative = (
        responses32[local_primary < 0].sum(dim=0)
        if bool((local_primary < 0).any())
        else torch.zeros_like(aggregate)
    )
    token_norm_reference = float(response_energy.median())
    random_index = rng.randrange(int(responses32.shape[0]))
    random_token = responses32[random_index]
    random_token = random_token * (
        token_norm_reference / float(random_token.norm().clamp_min(EPS))
    )
    gaussian = torch.randn_like(aggregate)
    gaussian = gaussian * (
        float(aggregate.norm()) / float(gaussian.norm().clamp_min(EPS))
    )
    strategy_directions: list[tuple[str, torch.Tensor, list[int]]] = [
        ("aggregate_visual_response", aggregate, list(range(int(responses32.shape[0])))),
        ("aggregate_positive_riesz", positive, torch.where(local_primary > 0)[0].tolist()),
        ("aggregate_negative_riesz", negative, torch.where(local_primary < 0)[0].tolist()),
        ("highest_local_riesz", responses32[int(local_primary.argmax())], [int(local_primary.argmax())]),
        ("highest_write", responses32[int(write_energy.argmax())], [int(write_energy.argmax())]),
        ("highest_jffn", responses32[int(response_energy.argmax())], [int(response_energy.argmax())]),
        ("highest_attention", responses32[int(attention.argmax())], [int(attention.argmax())]),
        ("random_norm_matched_token", random_token, [random_index]),
        ("random_gaussian_matched_norm", gaussian, []),
    ]
    blocked: list[dict[str, Any]] = []
    if path_ranking is not None:
        highest_index = int(path_ranking.argmax())
        negative_index = int(path_ranking.argmin())
        strategy_directions.extend(
            [
                ("highest_path", responses32[highest_index], [highest_index]),
                ("most_negative_path", responses32[negative_index], [negative_index]),
            ]
        )
        positive_order = torch.argsort(path_ranking, descending=True, stable=True)
        positive_indices = positive_order[path_ranking[positive_order] > 0]
        for top_k in (4, 16, 32):
            indices = positive_indices[: min(top_k, int(positive_indices.numel()))]
            if int(indices.numel()):
                strategy_directions.append(
                    (
                        f"top{top_k}_positive_path",
                        responses32[indices].sum(dim=0),
                        [int(value) for value in indices.tolist()],
                    )
                )
    else:
        blocked.append(
            {
                "model": model_name,
                "case_id": case_id,
                "layer": layer_number,
                "measurement_status": "BLOCKED",
                "error": (
                    "Path token map is unavailable; path-ranked FP32 interventions "
                    "were not substituted with local rankings"
                ),
            }
        )
    if (
        previous_wrong_image_direction is not None
        and previous_wrong_image_direction[0] != int(image_id)
    ):
        wrong = previous_wrong_image_direction[1].to(device)
        wrong = wrong * (
            float(aggregate.norm()) / float(wrong.norm().clamp_min(EPS))
        )
        strategy_directions.append(("wrong_image_matched_direction", wrong, []))

    perturbed_m = [m0]
    perturbation_index = {}
    for strategy, direction, _indices in strategy_directions:
        if float(direction.norm()) <= EPS:
            continue
        for eta in ETAS:
            plus_index = len(perturbed_m)
            perturbed_m.append(m0 + float(eta) * direction)
            minus_index = len(perturbed_m)
            perturbed_m.append(m0 - float(eta) * direction)
            perturbation_index[(strategy, eta)] = (plus_index, minus_index)
    branch_batch = torch.stack(perturbed_m)
    if layer_index == len(layers) - 1:
        batch_logits = final_logits_from_m(branch_batch)
    else:
        batch_logits, _unused_gradients, downstream_audit = (
            qwen_fp32_suffix_logits_and_gradients(
                model=wrapper.model,
                model_name=model_name,
                layers=layers,
                intervention_layer_index=layer_index,
                h_mid=capture["h_mid"][0, : prediction_position + 1].float().to(device),
                prediction_position=prediction_position,
                branch_outputs=branch_batch,
                current_ffn_norm=ffn_norm32,
                current_ffn=ffn32,
                final_norm=final_norm32,
                lm_weight=lm_weight32,
                lm_bias=lm_bias32,
                inputs=inputs,
                target_token_id=target_id,
                competitor_token_id=competitor_id,
                target_scalars=scalar_names,
                compute_gradients=False,
                query_batch_size=logit_batch_size,
            )
        )
    batch_scalars = {
        scalar: torch.stack(
            [
                target_scalar_from_logits(
                    logit_row,
                    target_token_id=target_id,
                    competitor_token_id=competitor_id,
                    scalar=scalar,
                )
                for logit_row in batch_logits
            ]
        )
        for scalar in scalar_names
    }
    strategy_indices = {
        strategy: indices for strategy, _direction, indices in strategy_directions
    }
    result_rows: list[dict[str, Any]] = []
    for strategy, direction, _indices in strategy_directions:
        if float(direction.norm()) <= EPS:
            continue
        for scalar in scalar_names:
            center_value = float(batch_scalars[scalar][0])
            predicted = float(torch.dot(gradients[scalar], direction))
            for eta in ETAS:
                plus_index, minus_index = perturbation_index[(strategy, eta)]
                plus = float(batch_scalars[scalar][plus_index])
                minus = float(batch_scalars[scalar][minus_index])
                symmetric = (plus - minus) / (2.0 * float(eta))
                result_rows.append(
                    {
                        "model": model_name,
                        "case_id": case_id,
                        "image_id": int(image_id),
                        "response_index": int(response_index),
                        "prediction_position": int(prediction_position),
                        "prefix_excludes_target": True,
                        "label_string": label_string,
                        "label_integer": label_integer,
                        "layer": layer_number,
                        "intervention_family": "ffn_output_true_fp32",
                        "downstream_route": downstream_audit["route"],
                        "downstream_layer_count": downstream_audit["downstream_layer_count"],
                        "context_recomputed_fp32": downstream_audit["context_recomputed_fp32"],
                        "strategy": strategy,
                        "token_or_region_ids": json.dumps(strategy_indices[strategy]),
                        "region_size": len(strategy_indices[strategy]),
                        "eta": float(eta),
                        "target_scalar": scalar,
                        "predicted_derivative": predicted,
                        "predicted_finite_change": float(eta) * predicted,
                        "observed_plus": plus,
                        "observed_minus": minus,
                        "symmetric_derivative": symmetric,
                        "observed_finite_effect": plus - center_value,
                        "sign_match": int(
                            math.copysign(1.0, predicted)
                            == math.copysign(1.0, symmetric)
                        )
                        if predicted != 0.0 and symmetric != 0.0
                        else 0,
                        "native_dtype": str(capture["h_mid"].dtype),
                        "intervention_dtype": "torch.float32",
                        "downstream_dtype": "torch.float32",
                        "zero_observed_flag": plus == center_value and minus == center_value,
                        "resolution_floor": 0.0,
                        "curvature": abs(plus - 2.0 * center_value + minus)
                        / (float(eta) ** 2),
                        "relative_error": abs(symmetric - predicted)
                        / max(abs(predicted), EPS),
                        "measurement_status": "MEASURED",
                        "error_flags": "",
                    }
                )
    del ffn_norm32, ffn32, directions, responses32, batch_logits
    return result_rows, blocked, aggregate.detach().cpu()


def main() -> None:
    args = parse_args()
    common = validate_common_args(args)
    count = int(args.num_images or (100 if args.formal else (3 if args.smoke else 12)))
    if args.formal and count < 100:
        raise ValueError("Formal FP32 causal validation requires at least 100 images")
    if int(args.logit_batch_size) <= 0:
        raise ValueError("logit-batch-size must be positive")
    extra = {
        "num_images": count,
        "max_targets_per_image": int(args.max_targets_per_image),
        "query_microbatch_size": int(args.logit_batch_size),
        "etas": list(ETAS),
        "route": "true_fp32_causal_suffix (Qwen all requested layers; other models final block)",
        "requested_layers": common["layers"],
    }
    if args.dry_run:
        print_dry_run("fp32_causal", common, extra)
        return

    layout = ExperimentLayout.create(common["output_dir"])
    if not layout.path("manifests/experiment_manifest.json").exists():
        initialize_manifests(
            layout=layout,
            repo_root=ROOT,
            experiment_config={**common, **extra},
            input_paths=input_paths_for_model(args.model, args.config),
            exact_command=exact_command(),
        )
    stage = f"fp32_causal:{args.model}:shard{args.shard_id}"
    resume_command = shell_command()
    output_path = layout.path(
        f"shards/fp32_causal_rank{args.shard_id:02d}_shard_00000.pt"
    )
    if args.resume and output_path.exists():
        existing = torch.load(output_path, map_location="cpu", weights_only=False)
        existing_failures = list(existing.get("failures") or ())
        existing_blocked = list(existing.get("blocked_rows") or ())
        reusable = (
            existing.get("protocol") == PROTOCOL
            and not existing_failures
            and int(existing.get("query_microbatch_size", -1))
            == int(args.logit_batch_size)
        )
        if reusable:
            resumed_status = "BLOCKED" if existing_blocked else "PASS"
            update_stage_status(
                layout=layout,
                stage=stage,
                status=resumed_status,
                details={
                    "resumed_existing": True,
                    "path": str(output_path),
                    "measured_rows": len(existing.get("rows") or ()),
                    "blocked_layers": sorted(
                        {
                            row["layer"]
                            for row in existing_blocked
                            if "layer" in row
                        }
                    ),
                    "failures": 0,
                    "query_microbatch_size": int(args.logit_batch_size),
                },
                resume_command=resume_command,
            )
            _consolidate_fp32_shards(
                layout=layout,
                num_shards=int(args.num_shards),
                query_microbatch_size=int(args.logit_batch_size),
            )
            print(f"[TC-FVPA FP32] reuse {output_path} status={resumed_status}")
            return
        print(
            f"[TC-FVPA FP32] recompute non-reusable shard {output_path} "
            f"protocol={existing.get('protocol')} failures={len(existing_failures)}",
            flush=True,
        )
    if not args.resume and output_path.exists():
        raise FileExistsError("--no-resume refuses to overwrite FP32 shard")
    update_stage_status(
        layout=layout,
        stage=stage,
        status="RUNNING",
        details={"command": exact_command()},
        resume_command=resume_command,
    )
    if args.model not in SUPPORTED_MODELS:
        error = f"{args.model} FP32 branch replacement is not parity-validated"
        update_stage_status(
            layout=layout,
            stage=stage,
            status="BLOCKED",
            details={"error": error},
            resume_command=resume_command,
        )
        raise RuntimeError(error)

    config = load_config(args.config)
    model_root = ROOT / "outputs" / args.model / "COCO4000-INSLEN-OFFICIAL-TARGET"
    labels, generations, splits = _load_inputs(model_root)
    image_ids = _selected_images(
        labels, generations, splits, count=count, seed=int(args.seed)
    )[int(args.shard_id) :: int(args.num_shards)]
    path_result_dir = Path(args.path_result_dir).resolve() if args.path_result_dir else layout.root
    path_maps = _load_path_maps(path_result_dir)
    rows: list[dict[str, Any]] = []
    blocked_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    wrapper = None
    started = time.perf_counter()
    previous_wrong_image_directions: dict[int, tuple[int, torch.Tensor]] = {}
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
        final_layer_number = len(layers)
        requested = common["layers"]
        for layer_number in requested:
            if layer_number != final_layer_number and args.model not in QWEN_MODELS:
                blocked_rows.append(
                    {
                        "model": args.model,
                        "layer": layer_number,
                        "measurement_status": "BLOCKED",
                        "error": (
                            "True-FP32 downstream decoder adapter is not yet validated "
                            "for layers below the final block"
                        ),
                    }
                )
        if final_layer_number not in requested and args.model not in QWEN_MODELS:
            raise RuntimeError(
                f"Requested layers {requested} exclude final layer {final_layer_number}; "
                "no true-FP32 route is available"
            )
        device = torch.device(args.device)
        final_norm32, lm_weight32, lm_bias32 = _fp32_output_modules(
            wrapper.model, device
        )

        prompt = str((config.get("run") or {}).get("prompt") or "Describe this image.")
        rng = random.Random(int(args.seed))
        for image_offset, image_id in enumerate(image_ids, 1):
            response_ids = [int(value) for value in generations[image_id]["response_token_ids"]]
            groups = target_groups(labels[image_id], response_ids)
            for response_index in _choose_target_indices(
                groups, int(args.max_targets_per_image)
            ):
                try:
                    from PIL import Image

                    with Image.open(_image_path(config, image_id)) as source:
                        image = source.convert("RGB")
                    spans = groups[response_index]
                    label_integer, label_string = _case_label(spans)
                    target_id = response_ids[response_index]
                    inputs, prediction_position, visual_start, visual_end, _grid = prepare_inputs(
                        wrapper,
                        args.model,
                        image,
                        response_ids[:response_index],
                        prompt,
                    )
                    captures, clean_native_logits = capture_clean(
                        wrapper, inputs, prediction_position
                    )
                    competitor_logits = clean_native_logits.clone()
                    competitor_logits[target_id] = -torch.inf
                    competitor_id = int(competitor_logits.argmax())
                    scalar_names = list(common["target_scalars"])
                    runnable = [
                        layer_number
                        for layer_number in requested
                        if layer_number == final_layer_number or args.model in QWEN_MODELS
                    ]
                    for layer_number in runnable:
                        try:
                            layer_rows, layer_blocked, aggregate = _evaluate_layer_case(
                                wrapper=wrapper,
                                model_name=args.model,
                                layers=layers,
                                layer_index=layer_number - 1,
                                captures=captures,
                                inputs=inputs,
                                prediction_position=prediction_position,
                                visual_start=visual_start,
                                visual_end=visual_end,
                                target_id=target_id,
                                competitor_id=competitor_id,
                                image_id=int(image_id),
                                response_index=int(response_index),
                                label_integer=label_integer,
                                label_string=label_string,
                                scalar_names=scalar_names,
                                path_maps=path_maps,
                                previous_wrong_image_direction=(
                                    previous_wrong_image_directions.get(layer_number)
                                ),
                                rng=rng,
                                device=device,
                                final_norm32=final_norm32,
                                lm_weight32=lm_weight32,
                                lm_bias32=lm_bias32,
                                logit_batch_size=int(args.logit_batch_size),
                            )
                            rows.extend(layer_rows)
                            blocked_rows.extend(layer_blocked)
                            previous_wrong_image_directions[layer_number] = (
                                int(image_id),
                                aggregate,
                            )
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
                    for item in captures:
                        item.clear()
                    del captures, inputs
                    torch.cuda.empty_cache()
                except Exception as exc:
                    failures.append(
                        {
                            "model": args.model,
                            "image_id": int(image_id),
                            "response_index": int(response_index),
                            "error": repr(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )
            print(
                f"[TC-FVPA FP32] {image_offset}/{len(image_ids)} image={image_id} "
                f"rows={len(rows)} failures={len(failures)}",
                flush=True,
            )
        payload = {
            "protocol": PROTOCOL,
            "route": (
                "Qwen FP32 current FFN + exact causal query-row downstream decoder "
                "+ final norm + LM head; final-block exact suffix for other models"
            ),
            "model": args.model,
            "query_microbatch_size": int(args.logit_batch_size),
            "rows": rows,
            "blocked_rows": blocked_rows,
            "failures": failures,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_gpu_memory_bytes": (
                int(torch.cuda.max_memory_allocated(torch.device(args.device)))
                if torch.cuda.is_available()
                else 0
            ),
        }
        atomic_torch_save(payload, output_path)
        _consolidate_fp32_shards(
            layout=layout,
            num_shards=int(args.num_shards),
            query_microbatch_size=int(args.logit_batch_size),
        )
        status = "BLOCKED" if blocked_rows else ("FAIL" if failures else "PASS")
        update_stage_status(
            layout=layout,
            stage=stage,
            status=status,
            details={
                "measured_rows": len(rows),
                "blocked_layers": sorted({row["layer"] for row in blocked_rows}),
                "failures": len(failures),
                "route": payload["route"],
                "peak_gpu_memory_bytes": payload["peak_gpu_memory_bytes"],
                "query_microbatch_size": int(args.logit_batch_size),
            },
            resume_command=resume_command,
        )
        if failures:
            raise RuntimeError(f"FP32 causal run retained {len(failures)} failed cases")
    except Exception as exc:
        status_payload = json.loads(
            layout.path("manifests/run_status.json").read_text(encoding="utf-8")
        )
        current = status_payload.get("stages", {}).get(stage, {}).get("status")
        if current not in {"BLOCKED", "FAIL"}:
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
