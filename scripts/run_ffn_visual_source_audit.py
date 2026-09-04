#!/usr/bin/env python3
"""Run the preregistered Vector FFN numerical and mechanism audit."""

from __future__ import annotations

import argparse
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.ffn_visual_interactions import (  # noqa: E402
    aggregate_region_writes,
    regular_grid_regions,
    sampled_vector_shapley,
)
from features.ffn_visual_path_attribution import (  # noqa: E402
    source_scaled_residual,
    streaming_vector_path_statistics,
)
from features.jffn_experiment import mentions_for_image  # noqa: E402
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
from scripts.run_ffn_visual_source_attribution import (  # noqa: E402
    EXPERIMENT,
    MODELS,
    result_root,
)
from scripts.run_jffn_p_comparison import (  # noqa: E402
    _image_path,
    _load_inputs,
)
from scripts.run_jffn_second_round_logit_causal import (  # noqa: E402
    capture_clean,
    prepare_inputs,
    target_groups,
)
from scripts.run_tc_fvpa_path_attribution import (  # noqa: E402
    _case_label,
    _choose_target_indices,
    _selected_images,
)
from scripts.tc_fvpa_common import (  # noqa: E402
    FORMAL_LAYERS_BY_MODEL,
    exact_command,
    shell_command,
)
from utils.config_utils import get_extraction_model_cfg, load_config  # noqa: E402


SCHEMA_VERSION = "ffn-visual-source-audit-v1"
K_VALUES = (1, 4, 8, 16, 32, 64)
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25)
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument(
        "--config", default="configs/model_configs_inslen_official_target.yaml"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-images", type=int, default=200)
    parser.add_argument("--audit-cases", type=int, default=50)
    parser.add_argument("--permutations", type=int, default=128)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--token-chunk-size", type=int, default=256)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _stratified_case_keys(
    *,
    labels: Mapping[int, Any],
    generations: Mapping[int, Any],
    image_ids: Sequence[int],
    layers: Sequence[int],
    count: int,
    seed: int,
) -> set[tuple[int, int, int]]:
    buckets: dict[tuple[int, int], list[tuple[int, int, int]]] = {
        (label, layer): [] for label in (0, 1) for layer in layers
    }
    for image_id in image_ids:
        response_ids = generations[image_id].get("response_token_ids") or ()
        groups = target_groups(labels[image_id], response_ids)
        for response_index in _choose_target_indices(groups, 2):
            label, _name = _case_label(groups[response_index])
            for layer in layers:
                buckets[(label, layer)].append((image_id, response_index, layer))
    for key, values in buckets.items():
        random.Random(f"{seed}:{key[0]}:{key[1]}").shuffle(values)
    selected: list[tuple[int, int, int]] = []
    while len(selected) < count and any(buckets.values()):
        for key in sorted(buckets):
            if buckets[key] and len(selected) < count:
                selected.append(buckets[key].pop())
    if len(selected) != count:
        raise ValueError(f"Only {len(selected)} stratified cases available for {count}")
    return set(selected)


def _fp32_ffn(layer: Any, device: torch.device) -> tuple[Callable, list[Any], torch.dtype]:
    adapter = resolve_decoder_layer_adapter(layer)
    norm = adapter.ffn_norm
    ffn = adapter.ffn
    original_dtype = adapter.output_projection.weight.dtype
    converted = []
    try:
        for module in (norm, ffn):
            module.to(device=device, dtype=torch.float32).eval()
            converted.append(module)
    except Exception:
        for module in converted:
            module.to(device=device, dtype=original_dtype)
        raise

    def function(value: torch.Tensor) -> torch.Tensor:
        return ffn(norm(value))

    return function, [norm, ffn], original_dtype


def _path(
    function: Callable,
    z: torch.Tensor,
    writes: torch.Tensor,
    *,
    method: str,
    k: int,
    chunk: int,
    backend: str = "vmap_jvp",
    save_components: bool = False,
    oom_fallback: bool = True,
):
    candidates = []
    for value in (int(chunk), 256, 128, 64, 32):
        value = min(value, int(writes.shape[0]))
        if value > 0 and value <= int(chunk) and value not in candidates:
            candidates.append(value)

    def attempt(chunk_size: int):
        try:
            return streaming_vector_path_statistics(
                ffn_map=function,
                z=z.unsqueeze(0),
                writes=writes.unsqueeze(1),
                method=method,
                integration_points=k,
                token_chunk_size=chunk_size,
                jvp_backend=backend,
                save_components=save_components,
            )
        except torch.cuda.OutOfMemoryError:
            return None

    for chunk_size in candidates:
        result = attempt(chunk_size)
        if result is not None:
            return result
        if not oom_fallback:
            break
        torch.cuda.empty_cache()
    raise torch.cuda.OutOfMemoryError(
        f"Audit vector path exhausted token chunks {candidates}"
    )


def _compact_path(result: Any, *, save_components: bool = False) -> dict[str, Any]:
    row = {
        "ffn_path_gross": result.ffn_path_gross[0].float().cpu(),
        "p_ffn": result.p_ffn[0].float().cpu(),
        "path_signed_q": result.path_signed_q[0].float().cpu(),
        "gross_strength": float(result.gross_strength[0]),
        "net_strength": float(result.net_strength[0]),
        "kappa": float(result.kappa[0]),
        "completeness_relative_error": float(
            result.completeness_relative_error[0]
        ),
        "gross_degenerate": bool(result.gross_degenerate[0]),
        "net_degenerate": bool(result.net_degenerate[0]),
        "token_chunk_size": int(result.token_chunk_size),
        "jvp_backend": str(result.jvp_backend),
    }
    if save_components:
        row["components"] = result.components[:, 0].float().cpu()
        row["total_finite_effect"] = result.total_finite_effect[0].float().cpu()
    return row


def _backend_benchmark(
    function: Callable,
    z: torch.Tensor,
    writes: torch.Tensor,
    *,
    chunk: int,
) -> dict[str, Any]:
    measurements = {}
    results = {}
    for backend in ("vmap_jvp", "linearize"):
        try:
            _path(
                function,
                z,
                writes,
                method="gauss_legendre",
                k=1,
                chunk=chunk,
                backend=backend,
                oom_fallback=False,
            )
            if z.is_cuda:
                torch.cuda.synchronize(z.device)
                torch.cuda.empty_cache()
                baseline = int(torch.cuda.memory_allocated(z.device))
                torch.cuda.reset_peak_memory_stats(z.device)
            else:
                baseline = 0
            started = time.perf_counter()
            result = _path(
                function,
                z,
                writes,
                method="gauss_legendre",
                k=4,
                chunk=chunk,
                backend=backend,
                save_components=True,
                oom_fallback=False,
            )
            if z.is_cuda:
                torch.cuda.synchronize(z.device)
                peak = int(torch.cuda.max_memory_allocated(z.device))
            else:
                peak = 0
            measurements[backend] = {
                "status": "PASS",
                "seconds": time.perf_counter() - started,
                "increment_bytes": max(peak - baseline, 0),
            }
            results[backend] = result.components.float().cpu()
        except Exception as exc:
            measurements[backend] = {"status": "FAIL", "error": repr(exc)}
    if len(results) == 2:
        reference = results["vmap_jvp"]
        measurements["relative_l2_error"] = float(
            (results["linearize"] - reference).norm()
            / reference.norm().clamp_min(EPS)
        )
    return measurements


def _synthetic_benchmark(device: torch.device) -> dict[str, Any]:
    torch.manual_seed(20260829)
    function = torch.nn.Sequential(
        torch.nn.LayerNorm(64, device=device, dtype=torch.float32),
        torch.nn.Linear(64, 192, device=device, dtype=torch.float32),
        torch.nn.SiLU(),
        torch.nn.Linear(192, 64, device=device, dtype=torch.float32),
    ).eval()
    z = torch.randn(64, device=device, dtype=torch.float32)
    writes = 0.02 * torch.randn(96, 64, device=device, dtype=torch.float32)
    result = _backend_benchmark(function, z, writes, chunk=64)
    del function, z, writes
    return result


def _audit_fp32_case(
    *,
    function: Callable,
    z: torch.Tensor,
    writes: torch.Tensor,
    grid: Sequence[int],
    seed: int,
    permutations: int,
    benchmark: bool,
) -> dict[str, Any]:
    chunk = min(256, int(writes.shape[0]))
    gauss = _path(
        function,
        z,
        writes,
        method="gauss_legendre",
        k=64,
        chunk=chunk,
        save_components=True,
    )
    trapezoid = _path(
        function,
        z,
        writes,
        method="trapezoid",
        k=64,
        chunk=chunk,
        save_components=True,
    )
    components = gauss.components[:, 0]
    p_ffn = gauss.p_ffn[0]
    regions8 = regular_grid_regions(int(grid[0]), int(grid[1]), 8)
    top_region = max(regions8, key=lambda region: float(p_ffn[region].sum()))
    rng = random.Random(int(seed))
    random_region = regions8[rng.randrange(len(regions8))]
    strategies = {
        "highest_p_ffn_token": [int(p_ffn.argmax())],
        "highest_8_region": list(top_region),
        "deterministic_random_region": list(random_region),
    }
    z0 = z - writes.sum(dim=0)
    scaling = {}
    with torch.no_grad():
        for name, indices in strategies.items():
            scaling[name] = []
            for value in LAMBDAS:
                scaled = source_scaled_residual(z, writes, indices, value)
                effect = function(scaled) - function(z0)
                scaling[name].append(
                    {
                        "lambda": value,
                        "finite_effect": effect.float().cpu(),
                        "finite_effect_norm": float(effect.float().norm()),
                    }
                )

    shapley = {}
    for region_count in (8, 16):
        regions = regular_grid_regions(int(grid[0]), int(grid[1]), region_count)
        region_writes = aggregate_region_writes(writes, regions)

        def game(mask: torch.Tensor) -> torch.Tensor:
            selected = region_writes[mask.to(region_writes.device)]
            point = z0 + (
                selected.sum(dim=0) if selected.numel() else torch.zeros_like(z)
            )
            return function(point)

        estimate = sampled_vector_shapley(
            value_from_active_mask=game,
            region_count=region_count,
            permutations=permutations,
            seed=seed + region_count,
            persist_every=16,
        )
        path_regions = torch.stack(
            [components[region].sum(dim=0) for region in regions]
        ).float().cpu()
        cosine = torch.nn.functional.cosine_similarity(
            path_regions, estimate.values.float(), dim=-1
        )
        norm_relative_error = (
            (path_regions - estimate.values.float()).norm(dim=-1)
            / estimate.values.float().norm(dim=-1).clamp_min(EPS)
        )
        shapley[str(region_count)] = {
            "permutation_count": estimate.permutation_count,
            "values": estimate.values.float(),
            "standard_errors": estimate.standard_errors.float(),
            "running_estimates": estimate.running_estimates.float(),
            "total_game_effect": estimate.total_game_effect.float(),
            "vector_completeness_relative_error": (
                estimate.completeness_relative_error
            ),
            "path_region_values": path_regions,
            "path_shapley_cosine": cosine,
            "path_shapley_norm_relative_error": norm_relative_error,
        }
    return {
        "gauss_legendre_k64": _compact_path(gauss, save_components=True),
        "trapezoid_k64": _compact_path(trapezoid, save_components=True),
        "source_scaling": scaling,
        "source_scaling_regions": strategies,
        "shapley": shapley,
        "linearize_benchmark": (
            _backend_benchmark(function, z, writes, chunk=min(64, len(writes)))
            if benchmark
            else None
        ),
    }


def main() -> None:
    args = parse_args()
    if args.world_size <= 0 or not 0 <= args.rank < args.world_size:
        raise ValueError(f"Invalid rank/world-size {args.rank}/{args.world_size}")
    if min(args.num_images, args.audit_cases, args.permutations, args.token_chunk_size) <= 0:
        raise ValueError("Audit counts and token chunk size must be positive")
    if args.smoke:
        args.num_images = min(args.num_images, 1)
        args.audit_cases = min(args.audit_cases, 1)
        args.permutations = min(args.permutations, 2)
    config = load_config(args.config)
    model_root = ROOT / "outputs" / args.model / EXPERIMENT
    labels, generations, splits = _load_inputs(model_root)
    image_ids = _selected_images(
        labels, generations, splits, count=args.num_images, seed=args.seed
    )
    layers = FORMAL_LAYERS_BY_MODEL[args.model]
    audit_keys = _stratified_case_keys(
        labels=labels,
        generations=generations,
        image_ids=image_ids,
        layers=layers,
        count=args.audit_cases,
        seed=args.seed,
    )
    worker_ids = image_ids[args.rank :: args.world_size]
    root = result_root(args.model)
    layout = ExperimentLayout.create(root)
    stage_kind = "audit_smoke" if args.smoke else "audit"
    stage = f"{stage_kind}:{args.model}:rank{args.rank}"
    initialize_manifests(
        layout=layout,
        repo_root=ROOT,
        experiment_config={
            "schema_version": SCHEMA_VERSION,
            "model": args.model,
            "images": args.num_images,
            "layers": layers,
            "k_values": K_VALUES,
            "fp32_cases": args.audit_cases,
            "region_counts": [8, 16],
            "shapley_permutations": args.permutations,
            "seed": args.seed,
        },
        input_paths=[
            args.config,
            model_root / "generations.json",
            model_root / "labeling.json",
            model_root / "image_splits.json",
        ],
        exact_command=exact_command(),
    )
    update_stage_status(
        layout=layout,
        stage=stage,
        status="RUNNING",
        details={"worker_images": len(worker_ids)},
        resume_command=shell_command(),
    )
    shard_dir = root / ("shards/audit_smoke" if args.smoke else "shards/audit")
    shard_dir.mkdir(parents=True, exist_ok=True)
    existing = {
        int(path.stem.rsplit("_", 1)[-1])
        for path in shard_dir.glob(f"audit_rank{args.rank:02d}_image_*.pt")
    }
    if not args.resume and existing:
        raise FileExistsError(f"Refusing to overwrite {shard_dir}")
    pending = [image_id for image_id in worker_ids if image_id not in existing]
    wrapper = None
    failure = None
    benchmark_pending = args.rank == 0
    started = time.perf_counter()
    try:
        device = torch.device(args.device)
        synthetic = _synthetic_benchmark(device)
        wrapper = build_model(
            args.model,
            get_extraction_model_cfg(config, args.model),
            device=args.device,
        )
        wrapper.model.requires_grad_(False)
        wrapper.model.eval()
        decoder_layers = resolve_decoder_layers(wrapper.model)
        prompt = str((config.get("run") or {}).get("prompt") or "Describe this image.")
        for offset, image_id in enumerate(pending, 1):
            image_rows = []
            response_ids = [
                int(value) for value in generations[image_id]["response_token_ids"]
            ]
            groups = target_groups(labels[image_id], response_ids)
            selected_targets = _choose_target_indices(groups, 2)
            try:
                with Image.open(_image_path(config, image_id)) as source:
                    image = source.convert("RGB")
                for response_index in selected_targets:
                    label, label_name = _case_label(groups[response_index])
                    target_id = int(response_ids[response_index])
                    prefix = response_ids[:response_index]
                    inputs, prediction_position, visual_start, visual_end, grid = (
                        prepare_inputs(wrapper, args.model, image, prefix, prompt)
                    )
                    if response_index != len(prefix):
                        raise AssertionError("Target entered its own causal prefix")
                    captures, _clean_logits = capture_clean(
                        wrapper, inputs, prediction_position
                    )
                    representative_indices = {value - 1 for value in layers}
                    for capture_index, unused_capture in enumerate(captures):
                        if capture_index not in representative_indices:
                            unused_capture.clear()
                    for layer_number in layers:
                        layer = decoder_layers[layer_number - 1]
                        capture = captures[layer_number - 1]
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

                        def native_ffn(value: torch.Tensor) -> torch.Tensor:
                            return adapter.ffn(adapter.ffn_norm(value))

                        key = (image_id, response_index, layer_number)
                        detailed = key in audit_keys
                        convergence = {}
                        for k in K_VALUES:
                            value = _path(
                                native_ffn,
                                z,
                                writes,
                                method="gauss_legendre",
                                k=k,
                                chunk=args.token_chunk_size,
                                save_components=detailed and k == 64,
                            )
                            convergence[str(k)] = _compact_path(
                                value, save_components=detailed and k == 64
                            )
                        del value
                        fp32 = None
                        if detailed:
                            torch.cuda.empty_cache()
                            function32, modules32, original_dtype = _fp32_ffn(
                                layer, z.device
                            )
                            try:
                                fp32 = _audit_fp32_case(
                                    function=function32,
                                    z=z.float(),
                                    writes=writes.float(),
                                    grid=grid,
                                    seed=(
                                        args.seed
                                        + image_id
                                        + response_index
                                        + layer_number
                                    ),
                                    permutations=args.permutations,
                                    benchmark=benchmark_pending,
                                )
                                benchmark_pending = False
                            finally:
                                for module in modules32:
                                    module.to(device=z.device, dtype=original_dtype)
                            del modules32, function32
                            torch.cuda.empty_cache()
                        image_rows.append(
                            {
                                "case_id": (
                                    f"{args.model}:{image_id}:{response_index}:"
                                    f"{layer_number}"
                                ),
                                "image_id": int(image_id),
                                "response_index": int(response_index),
                                "target_token_id": target_id,
                                "label": label,
                                "label_string": label_name,
                                "layer": layer_number,
                                "visual_grid": [int(value) for value in grid],
                                "visual_token_count": int(writes.shape[0]),
                                "prefix_excludes_target": True,
                                "attention_reconstruction_relative_error": float(
                                    directions["reconstruction_relative_error"]
                                ),
                                "component_sum_relative_error": float(
                                    directions["component_sum_relative_error"]
                                ),
                                "k_sweep": convergence,
                                "fp32_audit": fp32,
                            }
                        )
                        del directions, writes, z
                    for capture in captures:
                        capture.clear()
                    del captures, inputs
                    torch.cuda.empty_cache()
                atomic_torch_save(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "model": args.model,
                        "image_id": int(image_id),
                        "processed_image": True,
                        "cases": image_rows,
                        "synthetic_linearize_benchmark": (
                            synthetic if offset == 1 else None
                        ),
                    },
                    shard_dir / f"audit_rank{args.rank:02d}_image_{image_id}.pt",
                )
                print(
                    f"[FFN audit] {offset}/{len(pending)} image={image_id} "
                    f"cases={len(image_rows)}",
                    flush=True,
                )
            except Exception as exc:
                failure = {
                    "image_id": int(image_id),
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
                break
        if failure:
            raise RuntimeError(f"Audit stopped at first failure: {failure}")
        complete = existing | {
            int(path.stem.rsplit("_", 1)[-1])
            for path in shard_dir.glob(f"audit_rank{args.rank:02d}_image_*.pt")
        }
        if not set(worker_ids).issubset(complete):
            raise RuntimeError("Audit processed-image manifest is incomplete")
        status = "PASS"
    except Exception:
        status = "FAIL"
        raise
    finally:
        details = {
            "requested_images": len(worker_ids),
            "already_complete": len(existing),
            "pending_at_start": len(pending),
            "stratified_fp32_cases_global": len(audit_keys),
            "failure": failure,
            "elapsed_seconds": time.perf_counter() - started,
        }
        atomic_json_save(
            details, root / f"manifests/{stage_kind}_rank{args.rank:02d}.json"
        )
        if status == "PASS":
            atomic_json_save(
                checksums_for_paths(
                    shard_dir.glob(f"audit_rank{args.rank:02d}_image_*.pt")
                ),
                root / f"manifests/{stage_kind}_rank{args.rank:02d}_checksums.json",
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


if __name__ == "__main__":
    main()
