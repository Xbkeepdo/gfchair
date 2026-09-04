#!/usr/bin/env python3
"""Run gated fixed-QK, activation-patch, and pixel counterfactuals."""

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

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.ffn_visual_interactions import regular_grid_regions  # noqa: E402
from features.ffn_visual_path_attribution import (  # noqa: E402
    streaming_vector_path_statistics,
)
from features.tc_fvpa_artifacts import (  # noqa: E402
    ExperimentLayout,
    atomic_json_save,
    atomic_torch_save,
    checksums_for_paths,
    initialize_manifests,
    update_stage_status,
)
from features.visual_ffn_jacobian import (  # noqa: E402
    reconstruct_visual_directions,
    resolve_decoder_layer_adapter,
)
from models import build_model  # noqa: E402
from models.dgst_capture import resolve_decoder_layers  # noqa: E402
from scripts.analyze_ffn_visual_source_study import _load_full_model  # noqa: E402
from scripts.run_ffn_visual_source_attribution import (  # noqa: E402
    EXPERIMENT,
    MODELS,
    result_root,
)
from scripts.run_jffn_p_comparison import _image_path, _load_inputs  # noqa: E402
from scripts.run_jffn_second_round_logit_causal import (  # noqa: E402
    capture_clean,
    prepare_inputs,
)
from scripts.tc_fvpa_common import (  # noqa: E402
    FORMAL_LAYERS_BY_MODEL,
    exact_command,
    shell_command,
)
from utils.config_utils import get_extraction_model_cfg, load_config  # noqa: E402


SCHEMA_VERSION = "ffn-visual-source-counterfactual-v1"
EPS = 1e-12
STRATEGIES = ("highest_p_ffn_region", "highest_p_write_region", "random_region")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument(
        "--config", default="configs/model_configs_inslen_official_target.yaml"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-images", type=int, default=100)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _balanced_targets(
    positions: Sequence[Mapping[str, Any]],
    mentions: Sequence[Mapping[str, Any]],
    *,
    count: int,
    seed: int,
) -> list[tuple[int, int, int]]:
    available = {str(row["target_key"]): row for row in positions}
    buckets = {0: [], 1: []}
    seen = set()
    for mention in mentions:
        key = (int(mention["image_id"]), int(mention["response_index"]), int(mention["label"]))
        if key in seen or str(mention["target_key"]) not in available:
            continue
        seen.add(key)
        buckets[key[2]].append(key)
    for label in (0, 1):
        random.Random(f"{seed}:{label}").shuffle(buckets[label])
    selected = []
    used_images = set()
    while len(selected) < count and any(buckets.values()):
        for label in (0, 1):
            while buckets[label] and buckets[label][-1][0] in used_images:
                buckets[label].pop()
            if buckets[label] and len(selected) < count:
                row = buckets[label].pop()
                selected.append(row)
                used_images.add(row[0])
    if len(selected) != count:
        raise ValueError(f"Only {len(selected)} distinct label-balanced images for {count}")
    label_counts = {label: sum(row[2] == label for row in selected) for label in (0, 1)}
    if abs(label_counts[0] - label_counts[1]) > 1:
        raise AssertionError(f"Target selection is not label balanced: {label_counts}")
    return selected


def _region_choices(
    position: Mapping[str, Any], layer_index: int, seed: int
) -> tuple[list[list[int]], dict[str, int]]:
    grid = position["visual_grid"]
    regions = regular_grid_regions(int(grid[0]), int(grid[1]), 8)
    p_ffn = torch.as_tensor(position["p_ffn"])[layer_index]
    p_write = torch.as_tensor(position["p_write"])[layer_index]
    ffn_index = max(range(8), key=lambda index: float(p_ffn[regions[index]].sum()))
    write_index = max(range(8), key=lambda index: float(p_write[regions[index]].sum()))
    random_index = random.Random(int(seed)).randrange(8)
    return regions, {
        "highest_p_ffn_region": ffn_index,
        "highest_p_write_region": write_index,
        "random_region": random_index,
    }


def _rectangle(
    region: Sequence[int], grid: Sequence[int], image_size: Sequence[int]
) -> tuple[int, int, int, int]:
    height, width = map(int, grid)
    image_width, image_height = map(int, image_size)
    rows = [int(index) // width for index in region]
    columns = [int(index) % width for index in region]
    x0 = math.floor(min(columns) * image_width / width)
    x1 = math.ceil((max(columns) + 1) * image_width / width)
    y0 = math.floor(min(rows) * image_height / height)
    y1 = math.ceil((max(rows) + 1) * image_height / height)
    return x0, y0, x1, y1


def _mean_filled(image: Image.Image, rectangle: Sequence[int]) -> Image.Image:
    values = np.asarray(image.convert("RGB"), dtype=np.uint8)
    output = values.copy()
    mean = np.rint(values.reshape(-1, 3).mean(axis=0)).astype(np.uint8)
    x0, y0, x1, y1 = map(int, rectangle)
    output[y0:y1, x0:x1] = mean
    result = Image.fromarray(output, mode="RGB")
    if result.size != image.size:
        raise AssertionError("Pixel counterfactual changed image dimensions")
    return result


def _scores(logits: torch.Tensor, target: int, competitor: int) -> dict[str, float]:
    logits = logits.float().cpu()
    return {
        "target_logit": float(logits[target]),
        "fixed_clean_competitor_margin": float(logits[target] - logits[competitor]),
        "log_probability": float(torch.log_softmax(logits, dim=-1)[target]),
    }


def _replace_first(output: Any, value: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (value, *output[1:])
    return value


def _forward_intervention(
    *,
    model: Any,
    inputs: Mapping[str, torch.Tensor],
    layer: Any,
    prediction_position: int,
    target_id: int,
    competitor_id: int,
    attention_subtract: torch.Tensor | None = None,
    visual_positions: torch.Tensor | None = None,
    patch_states: torch.Tensor | None = None,
) -> tuple[dict[str, float], torch.Tensor]:
    captured = {}
    handles = []
    if attention_subtract is not None:
        attention = resolve_decoder_layer_adapter(layer).attention

        def attention_hook(_module, _args, output):
            value = output[0] if isinstance(output, tuple) else output
            changed = value.clone()
            changed[:, int(prediction_position)] -= attention_subtract.to(changed)
            return _replace_first(output, changed)

        handles.append(attention.register_forward_hook(attention_hook))
    if patch_states is not None:
        if visual_positions is None:
            raise ValueError("Activation patch requires visual positions")

        def layer_pre_hook(_module, args, kwargs):
            if args:
                hidden = args[0].clone()
                hidden[:, visual_positions.to(hidden.device)] = patch_states.to(hidden)
                return (hidden, *args[1:]), kwargs
            hidden = kwargs["hidden_states"].clone()
            hidden[:, visual_positions.to(hidden.device)] = patch_states.to(hidden)
            return args, {**kwargs, "hidden_states": hidden}

        handles.append(layer.register_forward_pre_hook(layer_pre_hook, with_kwargs=True))

    def layer_hook(_module, _args, output):
        value = output[0] if isinstance(output, tuple) else output
        captured["row"] = value[0, int(prediction_position)].detach().float().cpu()

    handles.append(layer.register_forward_hook(layer_hook))
    try:
        with torch.no_grad():
            output = model(
                **inputs,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
                use_cache=False,
            )
            logits = output.logits[0, int(prediction_position)].detach()
    finally:
        for handle in handles:
            handle.remove()
    if "row" not in captured:
        raise RuntimeError("Current-layer output hook did not fire")
    return _scores(logits, target_id, competitor_id), captured["row"]


def _cosine_and_relative(predicted: torch.Tensor, observed: torch.Tensor) -> dict[str, float]:
    predicted = predicted.float().reshape(1, -1)
    observed = observed.float().reshape(1, -1)
    return {
        "cosine": float(torch.nn.functional.cosine_similarity(predicted, observed)[0]),
        "norm_relative_error": float(
            (predicted - observed).norm() / observed.norm().clamp_min(EPS)
        ),
    }


def _bootstrap_mean_differences(rows: Sequence[Mapping[str, Any]], replicates: int) -> dict[str, Any]:
    metrics = ("target_logit_delta", "fixed_clean_competitor_margin_delta", "log_probability_delta")
    comparisons = (
        ("highest_p_ffn_region", "random_region"),
        ("highest_p_write_region", "random_region"),
        ("highest_p_ffn_region", "highest_p_write_region"),
    )
    results = {}
    for family in ("fixed_qk", "activation_patching", "pixel_counterfactual"):
        family_rows = [row for row in rows if row["intervention_family"] == family]
        lookup = {
            (int(row["image_id"]), int(row["layer"]), strategy): row
            for row in family_rows
            for strategy in row["strategies"]
        }
        images = sorted({int(row["image_id"]) for row in family_rows})
        for metric in metrics:
            for new, baseline in comparisons:
                per_image = []
                for image_id in images:
                    values = []
                    for layer in sorted(
                        {key[1] for key in lookup if key[0] == image_id}
                    ):
                        left = lookup[(image_id, layer, new)][metric]
                        right = lookup[(image_id, layer, baseline)][metric]
                        values.append(abs(float(left)) - abs(float(right)))
                    per_image.append(float(np.mean(values)))
                values = np.asarray(per_image, dtype=np.float64)
                rng = np.random.default_rng(
                    20260829 + len(results)
                )
                sampled = rng.integers(0, len(values), size=(replicates, len(values)))
                estimates = values[sampled].mean(axis=1)
                results[f"{family}:{metric}:{new}__minus__{baseline}"] = {
                    "images": len(images),
                    "effective_resamples": int(replicates),
                    "mean_difference": float(values.mean()),
                    "ci95": [
                        float(np.quantile(estimates, 0.025)),
                        float(np.quantile(estimates, 0.975)),
                    ],
                }
    return results


def _validate_counterfactual_rows(
    rows: Sequence[Mapping[str, Any]], requested_images: int, layers_per_image: int = 4
) -> int:
    def finite(value: Any) -> bool:
        if torch.is_tensor(value):
            return bool(torch.isfinite(value).all())
        if isinstance(value, Mapping):
            return all(finite(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return all(finite(item) for item in value)
        return not isinstance(value, float) or math.isfinite(value)

    images = {int(row["image_id"]) for row in rows}
    cases = {(int(row["image_id"]), int(row["layer"])) for row in rows}
    if len(images) != requested_images or len(cases) != requested_images * layers_per_image:
        raise AssertionError("Counterfactual image or target-layer coverage is incomplete")
    if not all(finite(row) for row in rows):
        raise ValueError("Counterfactual rows contain non-finite values")
    for case in cases:
        case_rows = [
            row
            for row in rows
            if (int(row["image_id"]), int(row["layer"])) == case
        ]
        for family in ("fixed_qk", "activation_patching", "pixel_counterfactual"):
            strategies = [
                strategy
                for row in case_rows
                if row["intervention_family"] == family
                for strategy in row["strategies"]
            ]
            if sorted(strategies) != sorted(STRATEGIES):
                raise AssertionError(
                    f"Incomplete or duplicate strategy map for {case} {family}"
                )
    return len(cases)


def _analyze_if_complete(root: Path, requested_images: int, replicates: int) -> None:
    paths = sorted((root / "shards/counterfactuals").glob("counterfactual_rank*_image_*.pt"))
    shards = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    if len({int(shard["image_id"]) for shard in shards}) != requested_images:
        return
    rows = [row for shard in shards for row in shard["rows"]]
    if not rows:
        raise AssertionError("Completed counterfactual shards contain no rows")
    target_layer_cases = _validate_counterfactual_rows(rows, requested_images)
    bootstrap = _bootstrap_mean_differences(rows, replicates)
    estimand_gaps = {
        "fixed_qk_e_r_vs_branch_delta_mean_norm_relative_error": float(
            np.mean(
                [
                    row["e_r_vs_ffn_branch_delta"]["norm_relative_error"]
                    for row in rows
                    if row["intervention_family"] == "fixed_qk"
                ]
            )
        ),
        "fixed_qk_a_plus_e_vs_block_delta_mean_norm_relative_error": float(
            np.mean(
                [
                    row["a_plus_e_vs_block_delta"]["norm_relative_error"]
                    for row in rows
                    if row["intervention_family"] == "fixed_qk"
                ]
            )
        ),
    }
    atomic_json_save(
        {
            "schema_version": SCHEMA_VERSION,
            "images": requested_images,
            "target_layer_cases": target_layer_cases,
            "bootstrap": bootstrap,
            "estimand_gaps": estimand_gaps,
        },
        root / "metrics/counterfactual_results.json",
    )


def main() -> None:
    args = parse_args()
    if args.world_size <= 0 or not 0 <= args.rank < args.world_size:
        raise ValueError(f"Invalid rank/world-size {args.rank}/{args.world_size}")
    if args.num_images <= 0 or args.bootstrap_resamples <= 0:
        raise ValueError("Image and bootstrap counts must be positive")
    if args.smoke:
        args.num_images = min(args.num_images, 1)
    root = result_root(args.model)
    layout = ExperimentLayout.create(root)
    gate_path = root / "tables/preregistered_gate.json"
    if not gate_path.exists():
        raise FileNotFoundError(f"Detector gate has not been evaluated: {gate_path}")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    stage = f"counterfactuals:{args.model}:rank{args.rank}"
    if not gate.get("passed"):
        update_stage_status(
            layout=layout,
            stage=stage,
            status="NOT_RUN",
            details={"reason": "NOT_RUN_BY_PREREGISTERED_GATE"},
        )
        print("NOT_RUN_BY_PREREGISTERED_GATE")
        return
    initialize_manifests(
        layout=layout,
        repo_root=ROOT,
        experiment_config={
            "schema_version": SCHEMA_VERSION,
            "model": args.model,
            "images": args.num_images,
            "layers": FORMAL_LAYERS_BY_MODEL[args.model],
            "regions": 8,
            "strategies": STRATEGIES,
            "bootstrap_resamples": args.bootstrap_resamples,
        },
        input_paths=[
            args.config,
            ROOT / "outputs" / args.model / EXPERIMENT / "generations.json",
            ROOT / "outputs" / args.model / EXPERIMENT / "labeling.json",
        ],
        exact_command=exact_command(),
    )
    positions, mentions, _processed = _load_full_model(args.model)
    selected = _balanced_targets(
        positions, mentions, count=args.num_images, seed=args.seed
    )
    worker = selected[args.rank :: args.world_size]
    by_target = {str(row["target_key"]): row for row in positions}
    frozen_k = int(
        json.loads((root / "tables/frozen_k.json").read_text(encoding="utf-8"))[
            "frozen_k"
        ]
    )
    config = load_config(args.config)
    model_root = ROOT / "outputs" / args.model / EXPERIMENT
    labels, generations, _splits = _load_inputs(model_root)
    shard_dir = root / "shards/counterfactuals"
    shard_dir.mkdir(parents=True, exist_ok=True)
    existing = {
        int(path.stem.rsplit("_", 1)[-1])
        for path in shard_dir.glob(f"counterfactual_rank{args.rank:02d}_image_*.pt")
    }
    pending = [row for row in worker if row[0] not in existing]
    wrapper = None
    failure = None
    status = "FAIL"
    started = time.perf_counter()
    update_stage_status(
        layout=layout,
        stage=stage,
        status="RUNNING",
        details={"worker_images": len(worker), "frozen_k": frozen_k},
        resume_command=shell_command(),
    )
    try:
        wrapper = build_model(
            args.model,
            get_extraction_model_cfg(config, args.model),
            device=args.device,
        )
        wrapper.model.requires_grad_(False)
        wrapper.model.eval()
        layers = resolve_decoder_layers(wrapper.model)
        prompt = str((config.get("run") or {}).get("prompt") or "Describe this image.")
        for offset, (image_id, response_index, label) in enumerate(pending, 1):
            try:
                response_ids = [
                    int(value) for value in generations[image_id]["response_token_ids"]
                ]
                target_id = response_ids[response_index]
                prefix = response_ids[:response_index]
                position = by_target[f"{image_id}:{response_index}"]
                with Image.open(_image_path(config, image_id)) as source:
                    clean_image = source.convert("RGB")
                inputs, prediction_position, visual_start, visual_end, grid = prepare_inputs(
                    wrapper, args.model, clean_image, prefix, prompt
                )
                if list(map(int, grid)) != list(map(int, position["visual_grid"])):
                    raise AssertionError("Full-attribution and counterfactual grids differ")
                clean_captures, clean_logits = capture_clean(
                    wrapper, inputs, prediction_position
                )
                competitor_logits = clean_logits.clone()
                competitor_logits[target_id] = -torch.inf
                competitor_id = int(competitor_logits.argmax())
                clean_scores = _scores(clean_logits, target_id, competitor_id)
                pixel_cache = {}
                rows = []
                for layer_number in FORMAL_LAYERS_BY_MODEL[args.model]:
                    layer_index = layer_number - 1
                    layer = layers[layer_index]
                    capture = clean_captures[layer_index]
                    directions = reconstruct_visual_directions(
                        layer=layer,
                        capture=capture,
                        prediction_positions=[prediction_position],
                        visual_start=visual_start,
                        visual_end=visual_end,
                    )
                    writes = directions["a_tokens"][:, 0].contiguous()
                    z = capture["h_mid"][0, prediction_position].to(writes)
                    adapter = resolve_decoder_layer_adapter(layer)

                    def ffn_map(value: torch.Tensor) -> torch.Tensor:
                        return adapter.ffn(adapter.ffn_norm(value))

                    path = streaming_vector_path_statistics(
                        ffn_map=ffn_map,
                        z=z.unsqueeze(0),
                        writes=writes.unsqueeze(1),
                        method="gauss_legendre",
                        integration_points=frozen_k,
                        token_chunk_size=256,
                        save_components=True,
                    )
                    regions, strategy_map = _region_choices(
                        position,
                        layer_index,
                        args.seed + image_id + response_index + layer_number,
                    )
                    unique_regions: dict[int, list[str]] = {}
                    for strategy, region_index in strategy_map.items():
                        unique_regions.setdefault(region_index, []).append(strategy)
                    clean_layer_row = (
                        capture["h_mid"][0, prediction_position]
                        + capture["o_ffn"][0, prediction_position]
                    ).float().cpu()
                    for region_index, strategies in unique_regions.items():
                        region = regions[region_index]
                        region_write = writes[region].sum(dim=0)
                        e_region = path.components[region, 0].sum(dim=0).float().cpu()
                        with torch.no_grad():
                            branch_delta = (
                                ffn_map(z) - ffn_map(z - region_write)
                            ).float().cpu()
                        block_delta = region_write.float().cpu() + branch_delta
                        fixed_scores, fixed_row = _forward_intervention(
                            model=wrapper.model,
                            inputs=inputs,
                            layer=layer,
                            prediction_position=prediction_position,
                            target_id=target_id,
                            competitor_id=competitor_id,
                            attention_subtract=region_write,
                        )
                        rectangle = _rectangle(region, grid, clean_image.size)
                        cache_key = tuple(rectangle)
                        if cache_key not in pixel_cache:
                            mean_image = _mean_filled(clean_image, rectangle)
                            cf_inputs, cf_position, cf_start, cf_end, cf_grid = prepare_inputs(
                                wrapper, args.model, mean_image, prefix, prompt
                            )
                            if (
                                cf_position != prediction_position
                                or cf_start != visual_start
                                or cf_end != visual_end
                                or list(map(int, cf_grid)) != list(map(int, grid))
                            ):
                                raise AssertionError("Pixel counterfactual changed token alignment")
                            cf_captures, cf_logits = capture_clean(
                                wrapper, cf_inputs, cf_position
                            )
                            pixel_cache[cache_key] = {
                                "scores": _scores(cf_logits, target_id, competitor_id),
                                "layers": {
                                    index: {
                                        "h_prev_visual": cf_captures[index]["h_prev"][
                                            0, visual_start:visual_end
                                        ].detach().cpu(),
                                        "row": (
                                            cf_captures[index]["h_mid"][
                                                0, prediction_position
                                            ]
                                            + cf_captures[index]["o_ffn"][
                                                0, prediction_position
                                            ]
                                        ).float().cpu(),
                                    }
                                    for index in (
                                        number - 1
                                        for number in FORMAL_LAYERS_BY_MODEL[args.model]
                                    )
                                },
                            }
                            for cf_capture in cf_captures:
                                cf_capture.clear()
                            del cf_captures, cf_logits, cf_inputs
                            torch.cuda.empty_cache()
                        cached_pixel = pixel_cache[cache_key]
                        visual_positions = torch.tensor(
                            [visual_start + index for index in region],
                            device=writes.device,
                            dtype=torch.long,
                        )
                        patch_states = cached_pixel["layers"][layer_index][
                            "h_prev_visual"
                        ][region]
                        patch_scores, patch_row = _forward_intervention(
                            model=wrapper.model,
                            inputs=inputs,
                            layer=layer,
                            prediction_position=prediction_position,
                            target_id=target_id,
                            competitor_id=competitor_id,
                            visual_positions=visual_positions,
                            patch_states=patch_states,
                        )
                        pixel_scores = cached_pixel["scores"]
                        pixel_row = cached_pixel["layers"][layer_index]["row"]

                        def deltas(scores):
                            return {
                                f"{name}_delta": clean_scores[name] - scores[name]
                                for name in clean_scores
                            }

                        common = {
                            "model": args.model,
                            "image_id": image_id,
                            "response_index": response_index,
                            "target_token_id": target_id,
                            "competitor_token_id": competitor_id,
                            "label": label,
                            "layer": layer_number,
                            "region_index": region_index,
                            "region_tokens": list(region),
                            "pixel_rectangle": list(rectangle),
                            "strategies": strategies,
                            "clean_scores": clean_scores,
                        }
                        rows.extend(
                            [
                                {
                                    **common,
                                    "intervention_family": "fixed_qk",
                                    **deltas(fixed_scores),
                                    "current_layer_vector_delta": clean_layer_row - fixed_row,
                                    "e_region": e_region,
                                    "ffn_branch_delta": branch_delta,
                                    "a_region": region_write.float().cpu(),
                                    "block_residual_delta": block_delta,
                                    "e_r_vs_ffn_branch_delta": _cosine_and_relative(
                                        e_region, branch_delta
                                    ),
                                    "a_plus_e_vs_block_delta": _cosine_and_relative(
                                        region_write.float().cpu() + e_region,
                                        block_delta,
                                    ),
                                },
                                {
                                    **common,
                                    "intervention_family": "activation_patching",
                                    **deltas(patch_scores),
                                    "current_layer_vector_delta": clean_layer_row - patch_row,
                                },
                                {
                                    **common,
                                    "intervention_family": "pixel_counterfactual",
                                    **deltas(pixel_scores),
                                    "current_layer_vector_delta": clean_layer_row - pixel_row,
                                },
                            ]
                        )
                    del directions, writes, z, path
                for capture in clean_captures:
                    capture.clear()
                atomic_torch_save(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "model": args.model,
                        "image_id": image_id,
                        "rows": rows,
                        "strategy_names": STRATEGIES,
                    },
                    shard_dir
                    / f"counterfactual_rank{args.rank:02d}_image_{image_id}.pt",
                )
                torch.cuda.empty_cache()
                print(
                    f"[counterfactual] {offset}/{len(pending)} image={image_id} "
                    f"rows={len(rows)}",
                    flush=True,
                )
            except Exception as exc:
                failure = {
                    "image_id": image_id,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
                break
        if failure:
            raise RuntimeError(f"Counterfactual stopped at first failure: {failure}")
        complete = existing | {
            int(path.stem.rsplit("_", 1)[-1])
            for path in shard_dir.glob(f"counterfactual_rank{args.rank:02d}_image_*.pt")
        }
        if not {row[0] for row in worker}.issubset(complete):
            raise RuntimeError("Counterfactual processed-image manifest is incomplete")
        status = "PASS"
    except Exception:
        status = "FAIL"
        raise
    finally:
        details = {
            "requested_images": len(worker),
            "pending_at_start": len(pending),
            "frozen_k": frozen_k,
            "failure": failure,
            "elapsed_seconds": time.perf_counter() - started,
        }
        atomic_json_save(
            details, root / f"manifests/counterfactual_rank{args.rank:02d}.json"
        )
        if status == "PASS":
            atomic_json_save(
                checksums_for_paths(
                    shard_dir.glob(
                        f"counterfactual_rank{args.rank:02d}_image_*.pt"
                    )
                ),
                root
                / f"manifests/counterfactual_rank{args.rank:02d}_checksums.json",
            )
        update_stage_status(
            layout=layout,
            stage=stage,
            status=status,
            details=details,
            resume_command=shell_command(),
        )
        if wrapper is not None:
            del wrapper
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if args.rank == 0 and args.world_size == 1:
        _analyze_if_complete(root, args.num_images, args.bootstrap_resamples)


if __name__ == "__main__":
    main()
