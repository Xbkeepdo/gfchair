#!/usr/bin/env python3
"""Region-level Monte-Carlo Shapley for the true-FP32 final-block route."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.ffn_visual_interactions import (  # noqa: E402
    aggregate_region_writes,
    regular_grid_regions,
)
from features.ffn_visual_path_attribution import target_scalar_from_logits  # noqa: E402
from features.tc_fvpa_artifacts import (  # noqa: E402
    ExperimentLayout,
    atomic_json_save,
    atomic_torch_save,
    initialize_manifests,
    update_stage_status,
)
from features.visual_ffn_jacobian import reconstruct_visual_directions  # noqa: E402
from models import build_model  # noqa: E402
from models.dgst_capture import resolve_decoder_layers  # noqa: E402
from scripts.run_jffn_p_comparison import _image_path, _load_inputs  # noqa: E402
from scripts.run_jffn_second_round_logit_causal import (  # noqa: E402
    capture_clean,
    prepare_inputs,
    target_groups,
)
from scripts.run_tc_fvpa_fp32_causal import _fp32_modules  # noqa: E402
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


PROTOCOL = "tc_fvpa_shapley_final_block_fp32_v1"


def _consolidate_shapley_shards(
    *, layout: ExperimentLayout, num_shards: int
) -> dict[str, Any]:
    """Rebuild shared Shapley tables from rank-local authoritative shards."""
    lock_path = layout.path("manifests/.shapley_consolidate.lock")
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        payloads = []
        completed_ranks = []
        for rank in range(int(num_shards)):
            shard_path = layout.path(
                f"shards/shapley_rank{rank:02d}_shard_00000.pt"
            )
            if not shard_path.exists():
                continue
            payload = torch.load(
                shard_path, map_location="cpu", weights_only=False
            )
            if payload.get("protocol") != PROTOCOL:
                continue
            payloads.append(payload)
            completed_ranks.append(rank)
        case_rows = [
            row for payload in payloads for row in payload.get("cases", ())
        ]
        running_rows = [
            row
            for payload in payloads
            for row in payload.get("running_estimates", ())
        ]
        not_in_scope_layers = [
            row
            for payload in payloads
            for row in payload.get("not_in_scope_layers", ())
        ]
        failures = [
            row for payload in payloads for row in payload.get("failures", ())
        ]
        atomic_torch_save(case_rows, layout.path("tables/shapley_cases.pt"))
        atomic_torch_save(
            running_rows, layout.path("tables/shapley_running_estimates.pt")
        )
        summary = {
            "protocol": PROTOCOL,
            "model": payloads[0].get("model") if payloads else None,
            "measured_rows": len(case_rows),
            "blocked_layers": [],
            "not_in_scope_layers": not_in_scope_layers,
            "failures": failures,
            "permutations": (
                int(payloads[0].get("permutations", 0)) if payloads else 0
            ),
            "completed_ranks": completed_ranks,
            "expected_ranks": int(num_shards),
            "all_ranks_present": len(completed_ranks) == int(num_shards),
            "peak_gpu_memory_bytes": max(
                (
                    int(payload.get("peak_gpu_memory_bytes", 0))
                    for payload in payloads
                ),
                default=0,
            ),
        }
        atomic_json_save(summary, layout.path("metrics/shapley_summary.json"))
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return summary
from utils.config_utils import get_extraction_model_cfg, load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = add_common_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--num-images", type=int, default=0)
    parser.add_argument("--max-targets-per-image", type=int, default=1)
    parser.add_argument("--region-counts", default="8,16")
    parser.add_argument("--permutations", type=int, default=128)
    parser.add_argument("--coalition-batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    common = validate_common_args(args)
    region_counts = [int(value) for value in args.region_counts.split(",") if value]
    if not region_counts or set(region_counts) - {8, 16}:
        raise ValueError("region-counts must be 8,16 or a subset")
    if int(args.permutations) < (128 if args.formal else 1):
        raise ValueError("Formal Shapley requires at least 128 permutations")
    count = int(args.num_images or (50 if args.formal else (2 if args.smoke else 12)))
    if args.formal and count < 50:
        raise ValueError("Formal Shapley requires at least 50 target-layer cases")
    extra = {
        "num_images": count,
        "region_counts": region_counts,
        "permutations": int(args.permutations),
        "route": "final_block_true_fp32",
    }
    if args.dry_run:
        print_dry_run("shapley", common, extra)
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
    stage = f"shapley:{args.model}:shard{args.shard_id}"
    output_path = layout.path(
        f"shards/shapley_rank{args.shard_id:02d}_shard_00000.pt"
    )
    if args.resume and output_path.exists():
        _consolidate_shapley_shards(
            layout=layout, num_shards=int(args.num_shards)
        )
        print(f"[TC-FVPA Shapley] reuse {output_path}")
        return
    if not args.resume and output_path.exists():
        raise FileExistsError("--no-resume refuses to overwrite Shapley shard")
    if args.model not in SUPPORTED_MODELS:
        error = f"{args.model} Shapley branch adapter is not parity-validated"
        update_stage_status(
            layout=layout,
            stage=stage,
            status="BLOCKED",
            details={"error": error},
            resume_command=shell_command(),
        )
        raise RuntimeError(error)
    update_stage_status(
        layout=layout,
        stage=stage,
        status="RUNNING",
        details={"command": exact_command()},
        resume_command=shell_command(),
    )
    config = load_config(args.config)
    model_root = ROOT / "outputs" / args.model / "COCO4000-INSLEN-OFFICIAL-TARGET"
    labels, generations, splits = _load_inputs(model_root)
    image_ids = _selected_images(labels, generations, splits, count=count, seed=args.seed)
    image_ids = image_ids[args.shard_id :: args.num_shards]
    wrapper = None
    case_rows = []
    running_rows = []
    failures = []
    not_in_scope_layers = []
    started = time.perf_counter()
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
        not_in_scope_layers = [
            layer for layer in common["layers"] if layer != final_layer_number
        ]
        if final_layer_number not in common["layers"]:
            raise RuntimeError("Requested layers exclude the only validated FP32 Shapley layer")
        layer = layers[-1]
        device = torch.device(args.device)
        ffn_norm32, ffn32, final_norm32, lm_weight32, lm_bias32 = _fp32_modules(
            wrapper.model, layer, device
        )
        prompt = str((config.get("run") or {}).get("prompt") or "Describe this image.")
        rng = random.Random(int(args.seed) + int(args.shard_id))
        for image_offset, image_id in enumerate(image_ids, 1):
            response_ids = [int(value) for value in generations[image_id]["response_token_ids"]]
            groups = target_groups(labels[image_id], response_ids)
            selected = _choose_target_indices(groups, int(args.max_targets_per_image))
            for response_index in selected:
                try:
                    from PIL import Image

                    with Image.open(_image_path(config, image_id)) as source:
                        image = source.convert("RGB")
                    target_id = response_ids[response_index]
                    spans = groups[response_index]
                    label_integer, label_string = _case_label(spans)
                    inputs, prediction_position, visual_start, visual_end, grid = prepare_inputs(
                        wrapper,
                        args.model,
                        image,
                        response_ids[:response_index],
                        prompt,
                    )
                    captures, native_logits = capture_clean(wrapper, inputs, prediction_position)
                    competitors = native_logits.clone()
                    competitors[target_id] = -torch.inf
                    competitor_id = int(competitors.argmax())
                    capture = captures[-1]
                    directions = reconstruct_visual_directions(
                        layer=layer,
                        capture=capture,
                        prediction_positions=[prediction_position],
                        visual_start=visual_start,
                        visual_end=visual_end,
                    )
                    writes = directions["a_tokens"][:, 0].float().to(device)
                    clean_z = capture["h_mid"][0, prediction_position].float().to(device)
                    z0 = clean_z - writes.sum(dim=0)
                    case_id = f"{args.model}:{image_id}:{response_index}:{final_layer_number}"
                    for region_count in region_counts:
                        regions = regular_grid_regions(int(grid[0]), int(grid[1]), region_count)
                        region_writes = aggregate_region_writes(writes, regions)
                        permutations = []
                        mask_ids = {0, (1 << region_count) - 1}
                        for _ in range(int(args.permutations)):
                            order = list(range(region_count))
                            rng.shuffle(order)
                            permutations.append(order)
                            mask = 0
                            for region in order:
                                mask |= 1 << region
                                mask_ids.add(mask)
                        sorted_masks = sorted(mask_ids)
                        score_by_mask = {
                            scalar: {} for scalar in common["target_scalars"]
                        }
                        for batch_start in range(0, len(sorted_masks), int(args.coalition_batch_size)):
                            batch_masks = sorted_masks[
                                batch_start : batch_start + int(args.coalition_batch_size)
                            ]
                            active = torch.tensor(
                                [
                                    [(mask >> region) & 1 for region in range(region_count)]
                                    for mask in batch_masks
                                ],
                                device=device,
                                dtype=torch.float32,
                            )
                            points = z0.unsqueeze(0) + active @ region_writes
                            branch_outputs = ffn32(ffn_norm32(points))
                            normalized = final_norm32(clean_z.unsqueeze(0) + branch_outputs)
                            logits = F.linear(normalized, lm_weight32, lm_bias32)
                            for row_offset, mask in enumerate(batch_masks):
                                for scalar in common["target_scalars"]:
                                    score_by_mask[scalar][mask] = float(
                                        target_scalar_from_logits(
                                            logits[row_offset],
                                            target_token_id=target_id,
                                            competitor_token_id=competitor_id,
                                            scalar=scalar,
                                        )
                                    )
                        full_mask = (1 << region_count) - 1
                        for scalar in common["target_scalars"]:
                            contributions = torch.zeros(
                                int(args.permutations), region_count, dtype=torch.float64
                            )
                            running_mean = torch.zeros(region_count, dtype=torch.float64)
                            for permutation_index, order in enumerate(permutations):
                                mask = 0
                                previous = score_by_mask[scalar][0]
                                for region in order:
                                    mask |= 1 << region
                                    current = score_by_mask[scalar][mask]
                                    contributions[permutation_index, region] = current - previous
                                    previous = current
                                running_mean = contributions[: permutation_index + 1].mean(dim=0)
                                running_rows.append(
                                    {
                                        "model": args.model,
                                        "case_id": case_id,
                                        "layer": final_layer_number,
                                        "target_scalar": scalar,
                                        "region_definition": f"regular_grid_{region_count}",
                                        "permutation_count": permutation_index + 1,
                                        "running_estimate": running_mean.clone(),
                                    }
                                )
                            values = contributions.mean(dim=0)
                            standard_errors = contributions.std(
                                dim=0, unbiased=True
                            ) / math.sqrt(int(args.permutations))
                            total_effect = (
                                score_by_mask[scalar][full_mask]
                                - score_by_mask[scalar][0]
                            )
                            case_rows.append(
                                {
                                    "model": args.model,
                                    "case_id": case_id,
                                    "image_id": image_id,
                                    "response_index": response_index,
                                    "label_string": label_string,
                                    "label_integer": label_integer,
                                    "layer": final_layer_number,
                                    "target_scalar": scalar,
                                    "region_definition": f"regular_grid_{region_count}",
                                    "permutation_count": int(args.permutations),
                                    "shapley_values": values,
                                    "standard_errors": standard_errors,
                                    "total_game_effect": total_effect,
                                    "completeness_absolute_error": abs(
                                        float(values.sum()) - total_effect
                                    ),
                                    "measurement_status": "MEASURED",
                                }
                            )
                    for item in captures:
                        item.clear()
                    del captures, inputs, directions
                    torch.cuda.empty_cache()
                except Exception as exc:
                    failures.append(
                        {
                            "model": args.model,
                            "image_id": image_id,
                            "response_index": response_index,
                            "error": repr(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )
            print(
                f"[TC-FVPA Shapley] {image_offset}/{len(image_ids)} image={image_id} "
                f"cases={len(case_rows)} failures={len(failures)}",
                flush=True,
            )
        payload = {
            "protocol": PROTOCOL,
            "scope": "preregistered final-decoder-layer subset",
            "model": args.model,
            "permutations": int(args.permutations),
            "cases": case_rows,
            "running_estimates": running_rows,
            "blocked_layers": [],
            "not_in_scope_layers": not_in_scope_layers,
            "failures": failures,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_gpu_memory_bytes": (
                int(torch.cuda.max_memory_allocated(torch.device(args.device)))
                if torch.cuda.is_available()
                else 0
            ),
        }
        atomic_torch_save(payload, output_path)
        _consolidate_shapley_shards(
            layout=layout, num_shards=int(args.num_shards)
        )
        update_stage_status(
            layout=layout,
            stage=stage,
            status="FAIL" if failures else "PASS",
            details={
                "measured_rows": len(case_rows),
                "scope": "preregistered final-decoder-layer subset",
                "not_in_scope_layers": not_in_scope_layers,
                "failures": len(failures),
            },
            resume_command=shell_command(),
        )
        if failures:
            raise RuntimeError(f"Shapley retained {len(failures)} failures")
    except Exception as exc:
        update_stage_status(
            layout=layout,
            stage=stage,
            status="FAIL",
            details={"error": repr(exc), "traceback": traceback.format_exc()},
            resume_command=shell_command(),
        )
        raise
    finally:
        if wrapper is not None:
            del wrapper
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
