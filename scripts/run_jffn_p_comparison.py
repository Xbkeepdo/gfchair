#!/usr/bin/env python3
"""Run the sharded four-model JFFN-P comparison without replacing old outputs."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from features.jffn_experiment import (  # noqa: E402
    COSTS,
    GATES,
    SOURCES,
    atomic_json_save,
    atomic_torch_save,
    assert_finite_position_rows,
    build_all_training_matrices_from_shards,
    build_jffn_dgst_config,
    compact_position_result,
    completed_image_ids,
    distribution_pair_metrics,
    feature_specs,
    fit_entropy_betas,
    load_shards,
    mentions_for_image,
    normalized_entropy,
    select_train_calibration_images,
)
from features.visual_ffn_jacobian import distribution_statistics  # noqa: E402
from models import build_model  # noqa: E402
from models.base_wrapper import AttentionRequirement, ExtractionRequirements  # noqa: E402
from scripts.train_torch_probe_feature_sets import (  # noqa: E402
    TorchProbeConfig,
    train_and_evaluate_probe,
)
from utils.config_utils import (  # noqa: E402
    get_dataset_cfg,
    get_dgst_t_cfg,
    get_extraction_model_cfg,
    load_config,
)
from utils.io_utils import load_json  # noqa: E402


MODELS = (
    "llava_1_5_7b",
    "internvl_2_5_8b",
    "qwen2_5_vl_7b",
    "qwen3_vl_8b",
)
SCHEMA_VERSION = "jffn-p-comparison-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=(*MODELS, "all"))
    parser.add_argument(
        "--config", default="configs/model_configs_inslen_official_target.yaml"
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--stage",
        choices=("smoke", "calibrate", "extract", "train", "analyze", "all", "summarize"),
        default="all",
    )
    parser.add_argument("--num-images", type=int, default=0)
    parser.add_argument("--calibration-images", type=int, default=0)
    parser.add_argument("--shard-images", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--training-device", default="auto")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.world_size <= 0 or args.rank < 0 or args.rank >= args.world_size:
        raise ValueError(
            f"Invalid image-shard rank/world-size: {args.rank}/{args.world_size}"
        )
    config = load_config(args.config)
    if args.model == "all":
        if args.stage != "summarize":
            raise ValueError("--model all is supported only with --stage summarize")
        write_cross_model_summary(config)
        return
    output_dir = Path(
        args.output_dir
        or ROOT / "outputs" / args.model / "COCO4000-INSLEN-OFFICIAL-TARGET"
    ).resolve()
    result_dir = output_dir / "results" / "jffn_p_comparison"
    result_dir.mkdir(parents=True, exist_ok=True)

    if args.stage == "smoke":
        run_extraction(
            args=args,
            config=config,
            output_dir=output_dir,
            result_dir=result_dir / "smoke_5images",
            smoke=True,
        )
        return
    if args.stage in {"calibrate", "all"}:
        run_calibration(args, config, output_dir, result_dir)
    if args.stage in {"extract", "all"}:
        run_extraction(
            args=args,
            config=config,
            output_dir=output_dir,
            result_dir=result_dir,
            smoke=False,
        )
    if args.stage in {"train", "all"}:
        run_training(args, config, output_dir, result_dir)
    if args.stage in {"analyze", "all"}:
        run_analysis(args, config, output_dir, result_dir)
    if args.stage in {"train", "analyze", "all"}:
        write_model_report(args.model, result_dir)
        write_cross_model_summary(config)


def _load_inputs(output_dir: Path):
    labels = {
        int(key): value
        for key, value in load_json(str(output_dir / "labeling.json")).items()
    }
    generations = {
        int(key): value
        for key, value in load_json(str(output_dir / "generations.json")).items()
    }
    splits = load_json(str(output_dir / "image_splits.json"))
    return labels, generations, splits


def _available_image_ids(labels, generations, splits) -> list[int]:
    split_ids = {int(value) for name in ("train", "test") for value in splits[name]}
    available = split_ids & set(labels) & set(generations)
    return sorted(
        image_id
        for image_id in available
        if labels[image_id].get("object_token_spans")
        and generations[image_id].get("response_token_ids")
    )


def _requirements() -> ExtractionRequirements:
    return ExtractionRequirements(
        attention=AttentionRequirement.NONE,
        logits=False,
        token_hidden_states=False,
        patch_hidden_states=False,
        response_hidden_states=False,
        visual_layout=True,
        dgst_capture=True,
    )


def _image_path(config: Mapping[str, Any], image_id: int) -> Path:
    dataset = get_dataset_cfg(config)
    return (
        Path(dataset["coco_root"])
        / "val2014"
        / f"COCO_val2014_{int(image_id):012d}.jpg"
    )


def _extract_one_image(
    *,
    wrapper: Any,
    config: Mapping[str, Any],
    image_id: int,
    label_row: Mapping[str, Any],
    generation_row: Mapping[str, Any],
    dgst_cfg: Mapping[str, Any],
    compact: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    response_ids = [int(value) for value in generation_row["response_token_ids"]]
    mentions, response_indices, target_ids = mentions_for_image(
        image_id=image_id,
        labeling_row=label_row,
        response_token_ids=response_ids,
    )
    if not response_indices:
        return [], []
    prompt = str((config.get("run") or {}).get("prompt") or "Describe this image.")
    with Image.open(_image_path(config, image_id)) as source:
        image = source.convert("RGB")
        outputs = wrapper.extract_token_features_batch(
            image=image,
            response_token_ids=response_ids,
            response_token_indices=response_indices,
            target_token_ids=target_ids,
            cfg_dgst_t=dict(dgst_cfg),
            prompt=prompt,
            requirements=_requirements(),
        )
    if len(outputs) != len(response_indices):
        raise RuntimeError(
            f"Image {image_id}: {len(outputs)} outputs for {len(response_indices)} targets"
        )
    positions = []
    for response_index, target_id, output in zip(
        response_indices, target_ids, outputs
    ):
        if output.dgst_t_result is None:
            raise RuntimeError(f"Image {image_id}: wrapper returned no DGST result")
        if compact:
            positions.append(
                compact_position_result(
                    image_id=image_id,
                    response_index=response_index,
                    target_token_id=target_id,
                    visual_grid=output.visual_grid,
                    result=output.dgst_t_result,
                )
            )
        else:
            positions.append(
                {
                    "image_id": image_id,
                    "response_index": response_index,
                    "target_token_id": target_id,
                    "visual_grid": [int(value) for value in output.visual_grid or ()],
                    "result": output.dgst_t_result,
                }
            )
    return positions, mentions


def run_calibration(args, config, output_dir: Path, result_dir: Path) -> None:
    calibration_path = result_dir / "entropy_calibration.json"
    if calibration_path.exists() and args.resume:
        existing = load_json(str(calibration_path))
        if (
            existing.get("selection_split") == "train_only"
            and int(existing.get("test_leakage_count", -1)) == 0
            and existing.get("betas")
        ):
            print(f"[JFFN calibration] reuse {calibration_path}", flush=True)
            return
    labels, generations, splits = _load_inputs(output_dir)
    section = config.get("jffn_p_comparison") or {}
    entropy_cfg = section.get("entropy_matching") or {}
    requested = int(
        args.calibration_images
        or entropy_cfg.get("calibration_images", 500)
    )
    seed = int(entropy_cfg.get("selection_seed", 20260818))
    available = set(_available_image_ids(labels, generations, splits))
    selected = select_train_calibration_images(
        train_image_ids=splits["train"],
        available_image_ids=available,
        count=requested,
        seed=seed,
    )
    test_ids = {int(value) for value in splits["test"]}
    leaked = sorted(set(selected) & test_ids)
    if leaked:
        raise AssertionError(f"Calibration selection leaked test IDs: {leaked[:5]}")

    model_cfg = get_extraction_model_cfg(config, args.model)
    wrapper = build_model(args.model, model_cfg, device=args.device)
    wrapper.model.requires_grad_(False)
    dgst_cfg = build_jffn_dgst_config(
        get_dgst_t_cfg(config),
        entropy_beta_by_layer=None,
        calibration_only=True,
        force_full_parallel=args.model in {"llava_1_5_7b", "internvl_2_5_8b"},
    )
    jvp_cfg = section.get("jvp") or {}
    dgst_cfg["jffn_max_increment_gib"] = float(
        jvp_cfg.get("max_increment_gib", 1.5)
    )
    layer_count = int(wrapper.num_layers)
    energies_by_layer: list[list[torch.Tensor]] = [
        [] for _ in range(layer_count)
    ]
    target_entropy_sums = np.zeros(layer_count, dtype=np.float64)
    target_entropy_counts = np.zeros(layer_count, dtype=np.int64)
    failures = []

    prior_diag = os.environ.get("DGST_UNION_COSINE_DIAGNOSTICS")
    prior_only = os.environ.get("DGST_UNION_COSINE_DIAGNOSTICS_ONLY")
    os.environ["DGST_UNION_COSINE_DIAGNOSTICS"] = "1"
    os.environ["DGST_UNION_COSINE_DIAGNOSTICS_ONLY"] = "1"
    try:
        for offset, image_id in enumerate(selected, start=1):
            try:
                positions, _mentions = _extract_one_image(
                    wrapper=wrapper,
                    config=config,
                    image_id=image_id,
                    label_row=labels[image_id],
                    generation_row=generations[image_id],
                    dgst_cfg=dgst_cfg,
                    compact=False,
                )
                for position in positions:
                    result = position["result"]
                    energy = result["dgst_t_jffn_energy_per_layer"].float().cpu()
                    old_hpre = result[
                        "dgst_t_source_hpre_cos_dist_per_layer"
                    ].float().cpu()
                    for layer_index in range(layer_count):
                        energies_by_layer[layer_index].append(
                            energy[layer_index : layer_index + 1].clone()
                        )
                        entropy = normalized_entropy(
                            old_hpre[layer_index : layer_index + 1]
                        )
                        target_entropy_sums[layer_index] += float(entropy.sum())
                        target_entropy_counts[layer_index] += int(entropy.numel())
            except Exception as exc:
                failures.append({"image_id": image_id, "error": repr(exc)})
                raise
            if offset % 10 == 0 or offset == len(selected):
                print(
                    f"[JFFN calibration] {offset}/{len(selected)} images",
                    flush=True,
                )
    finally:
        _restore_environment(
            "DGST_UNION_COSINE_DIAGNOSTICS", prior_diag
        )
        _restore_environment(
            "DGST_UNION_COSINE_DIAGNOSTICS_ONLY", prior_only
        )
    target_entropies = (
        target_entropy_sums / np.maximum(target_entropy_counts, 1)
    ).tolist()
    betas, layer_audit = fit_entropy_betas(
        energies_by_layer=energies_by_layer,
        target_entropy_by_layer=target_entropies,
        beta_min=float(entropy_cfg.get("beta_min", 0.02)),
        beta_max=float(entropy_cfg.get("beta_max", 50.0)),
        steps=int(entropy_cfg.get("bisection_steps", 60)),
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "selection_seed": seed,
        "requested_images": requested,
        "processed_images": len(selected),
        "selection_image_ids": selected,
        "selection_split": "train_only",
        "test_leakage_count": 0,
        "betas": betas,
        "layers": layer_audit,
        "failures": failures,
    }
    atomic_json_save(payload, calibration_path)
    print(f"[JFFN calibration] saved {len(betas)} layer betas", flush=True)
    del wrapper
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _restore_environment(name: str, prior: str | None) -> None:
    if prior is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = prior


def _load_betas(result_dir: Path) -> list[float]:
    path = result_dir / "entropy_calibration.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}; run --stage calibrate before full extraction."
        )
    return [float(value) for value in load_json(str(path))["betas"]]


def run_extraction(
    *,
    args,
    config,
    output_dir: Path,
    result_dir: Path,
    smoke: bool,
) -> None:
    existing_audit_path = result_dir / "extraction_audit.json"
    if existing_audit_path.exists() and args.resume:
        existing_audit = load_json(str(existing_audit_path))
        if smoke and existing_audit.get("validation"):
            validate_smoke_thresholds(existing_audit["validation"])
            print(f"[JFFN smoke] reuse {existing_audit_path}", flush=True)
            return
        if not smoke and bool(existing_audit.get("complete", False)):
            print(f"[JFFN extract] reuse complete {existing_audit_path}", flush=True)
            return
    labels, generations, splits = _load_inputs(output_dir)
    all_available_image_ids = _available_image_ids(labels, generations, splits)
    image_ids = list(all_available_image_ids)
    default_count = 5 if smoke else 0
    count = int(args.num_images or default_count)
    if count > 0:
        image_ids = image_ids[:count]
    global_requested_images = list(image_ids)
    # Stable striding gives disjoint image ownership without communicating
    # tensors or sharing model memory between the two GPU processes.
    image_ids = image_ids[args.rank :: args.world_size]
    section = config.get("jffn_p_comparison") or {}
    shard_images = int(args.shard_images or section.get("shard_images", 50))
    if shard_images <= 0:
        raise ValueError("shard_images must be positive")

    model_cfg = get_extraction_model_cfg(config, args.model)
    wrapper = build_model(args.model, model_cfg, device=args.device)
    wrapper.model.requires_grad_(False)
    if smoke:
        betas = [1.0] * int(wrapper.num_layers)
    else:
        betas = _load_betas(result_dir)
    dgst_cfg = build_jffn_dgst_config(
        get_dgst_t_cfg(config),
        entropy_beta_by_layer=betas,
        smoke_validation=smoke,
        force_full_parallel=args.model in {"llava_1_5_7b", "internvl_2_5_8b"},
    )
    jvp_cfg = section.get("jvp") or {}
    dgst_cfg["jffn_max_increment_gib"] = float(
        jvp_cfg.get("max_increment_gib", 1.5)
    )
    qwen_decision = None
    if args.model.startswith("qwen"):
        qwen_decision = run_qwen_shared_forward_gate(
            args=args,
            config=config,
            wrapper=wrapper,
            labels=labels,
            generations=generations,
            image_ids=all_available_image_ids,
            dgst_cfg=dgst_cfg,
            result_dir=result_dir,
        )
        dgst_cfg["jffn_qwen_shared_forward"] = bool(qwen_decision["enabled"])

    shard_dir = result_dir / "shards"
    existing_shards = list(shard_dir.glob("features*_shard_*.pt"))
    if not args.resume and existing_shards:
        raise FileExistsError(
            f"--no-resume refuses to append duplicate rows to non-empty {shard_dir}; "
            "use a fresh --output-dir or move the existing comparison directory."
        )
    done = completed_image_ids(shard_dir) if args.resume else set()
    pending = [image_id for image_id in image_ids if image_id not in done]
    shard_prefix = (
        f"features_rank{args.rank:02d}_shard"
        if args.world_size > 1
        else "features_shard"
    )
    existing_indices = [
        int(path.stem.rsplit("_", 1)[-1])
        for path in shard_dir.glob(f"{shard_prefix}_*.pt")
    ]
    shard_index = max(existing_indices, default=-1) + 1
    batch_positions: list[dict[str, Any]] = []
    batch_mentions: list[dict[str, Any]] = []
    batch_images: list[int] = []
    memory_audit = []
    failures = []

    for offset, image_id in enumerate(pending, start=1):
        try:
            if torch.cuda.is_available() and str(args.device).startswith("cuda"):
                device = torch.device(args.device)
                torch.cuda.reset_peak_memory_stats(device)
                before = int(torch.cuda.memory_allocated(device))
            else:
                device = None
                before = 0
            positions, mentions = _extract_one_image(
                wrapper=wrapper,
                config=config,
                image_id=image_id,
                label_row=labels[image_id],
                generation_row=generations[image_id],
                dgst_cfg=dgst_cfg,
            )
            if device is not None:
                peak = int(torch.cuda.max_memory_allocated(device))
                memory_audit.append(
                    {
                        "image_id": image_id,
                        "before_bytes": before,
                        "peak_bytes": peak,
                        "increment_bytes": peak - before,
                    }
                )
            batch_positions.extend(positions)
            batch_mentions.extend(mentions)
            batch_images.append(image_id)
        except Exception as exc:
            failures.append({"image_id": image_id, "error": repr(exc)})
            raise
        if len(batch_images) >= shard_images or offset == len(pending):
            shard_path = shard_dir / f"{shard_prefix}_{shard_index:05d}.pt"
            assert_finite_position_rows(batch_positions)
            atomic_torch_save(
                {
                    "schema_version": SCHEMA_VERSION,
                    "model": args.model,
                    "image_ids": list(batch_images),
                    "positions": batch_positions,
                    "sample_table": batch_mentions,
                    "provenance": {
                        "target_protocol": (
                            "inslen_official_first_token_first_occurrence"
                        ),
                        "source_top_k": 32,
                        "target_top_k": 32,
                        "union_top_k": 64,
                        "entropy_betas": betas,
                        "image_shard_rank": int(args.rank),
                        "image_shard_world_size": int(args.world_size),
                    },
                },
                shard_path,
            )
            print(
                f"[JFFN extract] shard {shard_index}: "
                f"{len(batch_images)} images, {len(batch_positions)} positions, "
                f"{len(batch_mentions)} mentions"
            )
            shard_index += 1
            batch_positions = []
            batch_mentions = []
            batch_images = []

    all_image_id_set: set[int] = set()
    num_shards = 0
    num_unique_positions = 0
    num_official_mentions = 0
    smoke_shards = []
    for shard in load_shards(shard_dir):
        num_shards += 1
        all_image_id_set.update(int(value) for value in shard["image_ids"])
        num_unique_positions += len(shard["positions"])
        num_official_mentions += len(shard["sample_table"])
        if smoke:
            smoke_shards.append(shard)
        else:
            del shard
    all_image_ids = sorted(all_image_id_set)
    audit = {
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "global_requested_images": len(global_requested_images),
        "worker_requested_images": len(image_ids),
        "image_shard_rank": int(args.rank),
        "image_shard_world_size": int(args.world_size),
        "completed_images": len(all_image_ids),
        "completed_image_ids": all_image_ids,
        "num_shards": num_shards,
        "num_unique_positions": num_unique_positions,
        "num_official_mentions": num_official_mentions,
        "memory": memory_audit,
        "failures": failures,
        "qwen_forward_mode": (
            (
                "shared_caption_query_rows"
                if qwen_decision and qwen_decision["enabled"]
                else "prefix_fallback"
            )
            if args.model.startswith("qwen")
            else "shared_caption"
        ),
    }
    if smoke:
        audit["validation"] = validate_smoke_shards(smoke_shards)
        audit["validation"]["peak_memory_bytes"] = max(
            (int(item["peak_bytes"]) for item in memory_audit),
            default=0,
        )
        validate_smoke_thresholds(audit["validation"])
    audit_name = (
        f"extraction_audit_rank{args.rank:02d}.json"
        if args.world_size > 1
        else "extraction_audit.json"
    )
    atomic_json_save(audit, result_dir / audit_name)
    del wrapper
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_qwen_shared_forward_gate(
    *,
    args,
    config,
    wrapper,
    labels,
    generations,
    image_ids: Sequence[int],
    dgst_cfg: Mapping[str, Any],
    result_dir: Path,
) -> dict[str, Any]:
    """Accept shared Qwen forward only if memory and numerical gates pass."""
    decision_path = result_dir / "qwen_shared_forward_decision.json"
    lock_path = result_dir / ".qwen_shared_forward_gate.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        # In a multi-GPU extraction, exactly one worker evaluates the gate and
        # every other worker consumes that same model-wide decision.  This also
        # applies to --no-resume so modes can never be mixed across workers.
        if decision_path.exists() and (args.resume or args.world_size > 1):
            return load_json(str(decision_path))
        decision = _evaluate_qwen_shared_forward_gate(
            args=args,
            config=config,
            wrapper=wrapper,
            labels=labels,
            generations=generations,
            image_ids=image_ids,
            dgst_cfg=dgst_cfg,
            gate_cfg=(config.get("jffn_p_comparison") or {}).get(
                "qwen_shared_forward_gate"
            )
            or {},
        )
        atomic_json_save(decision, decision_path)
        print(
            f"[Qwen shared gate] enabled={decision['enabled']}, "
            f"reason={decision['reason']}"
        )
        return decision


def _evaluate_qwen_shared_forward_gate(
    *, args, config, wrapper, labels, generations, image_ids, dgst_cfg, gate_cfg
) -> dict[str, Any]:
    """Evaluate the model-wide Qwen shared-forward gate under a file lock."""
    if not bool(gate_cfg.get("enabled", True)):
        return {"enabled": False, "reason": "disabled_by_config", "samples": []}
    scored = []
    for image_id in image_ids:
        response_ids = generations[image_id].get("response_token_ids") or []
        _mentions, indices, _target_ids = mentions_for_image(
            image_id=image_id,
            labeling_row=labels[image_id],
            response_token_ids=response_ids,
        )
        if indices:
            scored.append((len(indices), len(response_ids), image_id))
    selected = [row[2] for row in sorted(scored, reverse=True)[:3]]
    tolerance = float(gate_cfg.get("relative_error", 1e-3))
    memory_limit = int(float(gate_cfg.get("peak_memory_gib", 22.0)) * 1024**3)
    samples = []
    enabled = True
    reason = "accepted"
    prompt = str((config.get("run") or {}).get("prompt") or "Describe this image.")
    requirements = ExtractionRequirements(
        attention=AttentionRequirement.NONE,
        logits=False,
        token_hidden_states=True,
        patch_hidden_states=False,
        response_hidden_states=False,
        visual_layout=True,
        dgst_capture=True,
    )
    for image_id in selected:
        response_ids = [
            int(value) for value in generations[image_id]["response_token_ids"]
        ]
        _mentions, indices, targets = mentions_for_image(
            image_id=image_id,
            labeling_row=labels[image_id],
            response_token_ids=response_ids,
        )
        with Image.open(_image_path(config, image_id)) as source:
            image = source.convert("RGB")
            shared_cfg = dict(dgst_cfg)
            shared_cfg["jffn_qwen_shared_forward"] = True
            prefix_cfg = dict(dgst_cfg)
            prefix_cfg["jffn_qwen_shared_forward"] = False
            for candidate_cfg in (shared_cfg, prefix_cfg):
                candidate_cfg["jffn_validate_linearity"] = False
                candidate_cfg["jffn_finite_difference_eta"] = None
                candidate_cfg["jffn_parity_chunk_size"] = None
            try:
                device = torch.device(args.device)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                shared_outputs = wrapper.extract_token_features_batch(
                    image=image,
                    response_token_ids=response_ids,
                    response_token_indices=indices,
                    target_token_ids=targets,
                    cfg_dgst_t=shared_cfg,
                    prompt=prompt,
                    requirements=requirements,
                )
                shared_peak = int(torch.cuda.max_memory_allocated(device))
                prefix_outputs = wrapper.extract_token_features_batch(
                    image=image,
                    response_token_ids=response_ids,
                    response_token_indices=indices,
                    target_token_ids=targets,
                    cfg_dgst_t=prefix_cfg,
                    prompt=prompt,
                    requirements=requirements,
                )
                errors = compare_qwen_forward_outputs(
                    shared_outputs, prefix_outputs
                )
                sample_passed = (
                    shared_peak <= memory_limit
                    and max(errors.values(), default=0.0) <= tolerance
                )
                samples.append(
                    {
                        "image_id": image_id,
                        "target_count": len(indices),
                        "response_length": len(response_ids),
                        "shared_peak_bytes": shared_peak,
                        "relative_errors": errors,
                        "passed": sample_passed,
                    }
                )
                if not sample_passed:
                    enabled = False
                    reason = "memory_or_parity_threshold_failed"
            except torch.OutOfMemoryError as exc:
                enabled = False
                reason = "shared_forward_oom"
                samples.append(
                    {"image_id": image_id, "passed": False, "error": repr(exc)}
                )
                torch.cuda.empty_cache()
                break
            except Exception as exc:
                enabled = False
                reason = "shared_forward_error"
                samples.append(
                    {"image_id": image_id, "passed": False, "error": repr(exc)}
                )
                torch.cuda.empty_cache()
                break
    decision = {
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "enabled": bool(enabled),
        "reason": reason,
        "memory_limit_bytes": memory_limit,
        "relative_error_tolerance": tolerance,
        "samples": samples,
        "policy": "one_model_one_mode_no_mixing",
    }
    return decision


def compare_qwen_forward_outputs(
    shared_outputs: Sequence[Any], prefix_outputs: Sequence[Any]
) -> dict[str, float]:
    if len(shared_outputs) != len(prefix_outputs):
        return {"output_count": float("inf")}
    errors: dict[str, float] = {}

    def relative(left: Any, right: Any) -> float:
        left_tensor = torch.as_tensor(left).float()
        right_tensor = torch.as_tensor(right).float()
        return float(
            (left_tensor - right_tensor).norm()
            / right_tensor.norm().clamp_min(1e-12)
        )

    fields = (
        "dgst_t_source_dist_per_layer",
        "dgst_t_source_hpre_cos_dist_per_layer",
        "dgst_t_source_jffn_dist_per_layer",
        "dgst_t_source_jffn_entropy_matched_dist_per_layer",
        "dgst_t_hpre_raw_logit_gauss_source_jffn_risk_sqrt_hpre_per_layer",
    )
    for index, (shared, prefix) in enumerate(zip(shared_outputs, prefix_outputs)):
        errors[f"target_{index}_hidden"] = relative(
            shared.token_hidden_states, prefix.token_hidden_states
        )
        for field in fields:
            errors[f"target_{index}_{field}"] = relative(
                shared.dgst_t_result[field], prefix.dgst_t_result[field]
            )
    return errors


def validate_smoke_shards(shards: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    rows = [row for shard in shards for row in shard["positions"]]
    if not rows:
        raise RuntimeError("Smoke extraction produced no target positions")
    sums = []
    finite = []
    reconstruction_error = []
    reconstruction_cosine = []
    linearity = []
    parity = []
    fd_cosine = []
    for row in rows:
        for source in SOURCES:
            p = row["sources"][source]
            sums.extend(p.sum(dim=-1).tolist())
            finite.append(bool(torch.isfinite(p).all()))
        diagnostics = row["jffn_diagnostics"]
        reconstruction_error.extend(
            diagnostics["attention_reconstruction_relative_error"].tolist()
        )
        reconstruction_cosine.extend(
            diagnostics["attention_reconstruction_cosine"].tolist()
        )
        linearity.extend(diagnostics["response_sum_relative_error"].tolist())
        parity.extend(diagnostics["parallel_chunk_relative_error"].tolist())
        fd_cosine.extend(diagnostics["local_fd_cosine"].tolist())
    return {
        "num_positions": len(rows),
        "all_finite": bool(all(finite)),
        "max_distribution_sum_error": float(
            np.max(np.abs(np.asarray(sums) - 1.0))
        ),
        "max_attention_reconstruction_relative_error": float(
            np.nanmax(reconstruction_error)
        ),
        "min_attention_reconstruction_cosine": float(
            np.nanmin(reconstruction_cosine)
        ),
        "max_linearity_relative_error": float(np.nanmax(linearity)),
        "max_parallel_chunk_relative_error": float(np.nanmax(parity)),
        "median_finite_difference_cosine": float(np.nanmedian(fd_cosine)),
        "peak_memory_bytes": 0,
    }


def validate_smoke_thresholds(metrics: Mapping[str, Any]) -> None:
    checks = {
        "all_finite": bool(metrics["all_finite"]),
        "P sum": float(metrics["max_distribution_sum_error"]) <= 1e-5,
        "attention relative error": float(
            metrics["max_attention_reconstruction_relative_error"]
        ) <= 1e-2,
        "attention cosine": float(
            metrics["min_attention_reconstruction_cosine"]
        ) >= 0.999,
        "JVP linearity": float(metrics["max_linearity_relative_error"]) <= 1e-2,
        "parallel/chunk parity": float(
            metrics["max_parallel_chunk_relative_error"]
        ) <= 5e-3,
        "finite difference cosine": float(
            metrics["median_finite_difference_cosine"]
        ) >= 0.98,
        "peak memory": int(metrics["peak_memory_bytes"]) <= 22 * 1024**3,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"JFFN smoke acceptance failed: {failed}; {metrics}")


def _training_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def run_training(args, config, output_dir: Path, result_dir: Path) -> None:
    finalize_full_extraction(config, args.model, output_dir, result_dir)
    final_path = result_dir / "training_results.json"
    if final_path.exists() and args.resume:
        print(f"[JFFN train] reuse {final_path}", flush=True)
        return
    shard_dir = result_dir / "shards"
    splits = load_json(str(output_dir / "image_splits.json"))
    train_ids = {int(value) for value in splits["train"]}
    test_ids = {int(value) for value in splits["test"]}
    section = config.get("jffn_p_comparison") or {}
    comparison_training = section.get("training") or {}
    inherited = (config.get("training") or {}).get("torch_probe") or {}
    seeds = [int(value) for value in comparison_training.get("seeds", [43, 44, 45])]
    device = _training_device(args.training_device)
    progress_path = result_dir / "training_progress.pt"
    if progress_path.exists() and args.resume:
        progress = torch.load(progress_path, map_location="cpu", weights_only=False)
        results: dict[str, Any] = dict(progress.get("feature_sets") or {})
        predictions: dict[str, dict[int, dict[str, Any]]] = defaultdict(
            dict,
            {
                str(name): {int(seed): value for seed, value in rows.items()}
                for name, rows in (progress.get("predictions") or {}).items()
            },
        )
        print(
            f"[JFFN train] resume {len(results)} feature sets from {progress_path}",
            flush=True,
        )
    else:
        results = {}
        predictions = defaultdict(dict)
    specs = feature_specs()
    matrices = build_all_training_matrices_from_shards(
        shard_dir=shard_dir,
        specs=specs,
        train_image_ids=train_ids,
        test_image_ids=test_ids,
    )

    for spec in specs:
        X_train, y_train, _train_images, _train_mentions = matrices[spec["name"]][
            "train"
        ]
        X_test, y_test, test_images, test_mentions = matrices[spec["name"]][
            "test"
        ]
        if X_train.size == 0 or X_test.size == 0:
            raise RuntimeError(f"No train/test rows for {spec['name']}")
        if set(np.unique(y_train)) != {0, 1} or set(np.unique(y_test)) != {0, 1}:
            raise RuntimeError(f"Non-binary split for {spec['name']}")
        feature_results = list(results.get(spec["name"], {}).get("seeds", ()))
        completed_seeds = {
            int(row["seed"])
            for row in feature_results
            if int(row["seed"]) in predictions.get(spec["name"], {})
        }
        for seed in seeds:
            if seed in completed_seeds:
                continue
            probe_cfg = TorchProbeConfig(
                hidden_sizes=tuple(
                    int(value)
                    for value in comparison_training.get(
                        "hidden_sizes", inherited.get("hidden_sizes", [128, 64, 32])
                    )
                ),
                dropout=float(inherited.get("dropout", 0.3)),
                drop_last=bool(inherited.get("drop_last", True)),
                batch_size=int(comparison_training.get("batch_size", 256)),
                num_epochs=int(comparison_training.get("max_epochs", 100)),
                learning_rate=float(inherited.get("learning_rate", 1e-3)),
                weight_decay=float(inherited.get("weight_decay", 1e-5)),
                lr_factor=float(inherited.get("lr_factor", 0.5)),
                lr_patience=int(inherited.get("lr_patience", 5)),
                early_stopping_patience=int(
                    inherited.get("early_stopping_patience", 10)
                ),
                seed=seed,
                positive_class="real",
                split_protocol="strict_82_no_validation",
                threshold_selection="train_f1",
                fixed_threshold=float(inherited.get("fixed_threshold", 0.5)),
                checkpoint_selection="minimum_train_loss",
            )
            artifacts = result_dir / "training" / spec["name"] / f"seed_{seed}"
            metrics = train_and_evaluate_probe(
                X_train=X_train,
                y_train=y_train,
                X_val=np.empty((0, X_train.shape[1]), dtype=np.float32),
                y_val=np.empty((0,), dtype=np.int32),
                X_test=X_test,
                y_test=y_test,
                config=probe_cfg,
                device=device,
                output_dir=str(artifacts),
                return_probabilities=True,
            )
            test_probs = np.asarray(metrics.pop("test_probabilities"), dtype=np.float32)
            metrics.pop("train_probabilities", None)
            metrics["seed"] = seed
            metrics["num_features"] = int(X_train.shape[1])
            feature_results.append(metrics)
            predictions[spec["name"]][seed] = {
                "probabilities": test_probs,
                "labels": y_test.copy(),
                "image_ids": test_images.copy(),
                "mention_ids": list(test_mentions),
            }
            print(
                f"[JFFN train] {spec['name']} seed={seed} "
                f"AUC={metrics['auc']:.4f} F1={metrics['f1']:.4f}"
            )
            results[spec["name"]] = {
                "spec": spec,
                "seeds": feature_results,
                "aggregate": aggregate_seed_metrics(feature_results),
            }
            atomic_torch_save(
                {
                    "schema_version": SCHEMA_VERSION,
                    "model": args.model,
                    "feature_sets": results,
                    "predictions": dict(predictions),
                },
                progress_path,
            )
        if len(feature_results) != len(seeds):
            raise RuntimeError(
                f"Incomplete seeds for {spec['name']}: "
                f"{[row['seed'] for row in feature_results]}"
            )
        results[spec["name"]] = {
            "spec": spec,
            "seeds": feature_results,
            "aggregate": aggregate_seed_metrics(feature_results),
        }

    bootstrap = main_comparison_bootstrap(
        predictions=predictions,
        replicates=int(comparison_training.get("bootstrap_replicates", 10000)),
        seed=int(comparison_training.get("bootstrap_seed", 20260819)),
    )
    atomic_json_save(
        {
            "schema_version": SCHEMA_VERSION,
            "model": args.model,
            "sample_counts": {
                "train_mentions": int(len(next(iter(matrices.values()))["train"][1])),
                "test_mentions": int(len(next(iter(matrices.values()))["test"][1])),
            },
            "feature_sets": results,
            "main_comparison_bootstrap": bootstrap,
        },
        final_path,
    )


def aggregate_seed_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    paths = {
        "auroc": lambda row: row["auc"],
        "accuracy": lambda row: row["accuracy"],
        "real_f1": lambda row: row["real_positive"]["f1"],
        "real_aupr": lambda row: row["real_positive"]["aupr"],
        "hall_f1": lambda row: row["hallucination_positive"]["f1"],
        "hall_aupr": lambda row: row["hallucination_positive"]["aupr"],
    }
    result = {}
    for name, getter in paths.items():
        values = np.asarray([float(getter(row)) for row in rows], dtype=np.float64)
        result[name] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "values": values.tolist(),
        }
    return result


def _main_spec(source: str, feature: str = "risk_ev") -> str:
    return f"{source}__hpre_raw_logit_gauss__sqrt_matched_state__{feature}"


def main_comparison_bootstrap(
    *, predictions: Mapping[str, Mapping[int, Mapping[str, Any]]], replicates: int, seed: int
) -> dict[str, Any]:
    comparisons = (
        ("new_jffn", "old_hpre_cos"),
        ("new_jffn_entropy_matched", "old_hpre_cos"),
    )
    output = {}
    for new_source, old_source in comparisons:
        new_name = _main_spec(new_source)
        old_name = _main_spec(old_source)
        common_seeds = sorted(set(predictions[new_name]) & set(predictions[old_name]))
        new_probs = np.mean(
            [predictions[new_name][value]["probabilities"] for value in common_seeds],
            axis=0,
        )
        old_probs = np.mean(
            [predictions[old_name][value]["probabilities"] for value in common_seeds],
            axis=0,
        )
        reference = predictions[new_name][common_seeds[0]]
        labels = np.asarray(reference["labels"], dtype=np.int32)
        image_ids = np.asarray(reference["image_ids"], dtype=np.int64)
        unique_images = np.unique(image_ids)
        by_image = {value: np.flatnonzero(image_ids == value) for value in unique_images}
        rng = np.random.default_rng(int(seed))
        differences = []
        for _ in range(int(replicates)):
            sampled = rng.choice(unique_images, size=len(unique_images), replace=True)
            indices = np.concatenate([by_image[value] for value in sampled])
            if np.unique(labels[indices]).size < 2:
                continue
            differences.append(
                float(roc_auc_score(labels[indices], new_probs[indices]))
                - float(roc_auc_score(labels[indices], old_probs[indices]))
            )
        values = np.asarray(differences, dtype=np.float64)
        key = f"{new_source}_vs_{old_source}"
        output[key] = {
            "new_feature_set": new_name,
            "old_feature_set": old_name,
            "seed_averaged_new_auroc": float(roc_auc_score(labels, new_probs)),
            "seed_averaged_old_auroc": float(roc_auc_score(labels, old_probs)),
            "difference": float(
                roc_auc_score(labels, new_probs) - roc_auc_score(labels, old_probs)
            ),
            "bootstrap_replicates": int(values.size),
            "ci95": [
                float(np.quantile(values, 0.025)),
                float(np.quantile(values, 0.975)),
            ],
        }
    return output


class Running:
    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.square = 0.0

    def add(self, value: float) -> None:
        if math.isfinite(float(value)):
            self.count += 1
            self.total += float(value)
            self.square += float(value) ** 2

    def result(self) -> dict[str, float]:
        mean = self.total / max(self.count, 1)
        variance = max(self.square / max(self.count, 1) - mean * mean, 0.0)
        return {"count": self.count, "mean": mean, "std": math.sqrt(variance)}


def run_analysis(args, config, output_dir: Path, result_dir: Path) -> None:
    finalize_full_extraction(config, args.model, output_dir, result_dir)
    final_path = result_dir / "p_analysis.json"
    if final_path.exists() and args.resume:
        print(f"[JFFN analysis] reuse {final_path}", flush=True)
        return
    distribution_acc: dict[tuple[str, str, int, str], Running] = defaultdict(Running)
    specificity_acc: dict[tuple[str, int, str], Running] = defaultdict(Running)
    spatial_acc: dict[tuple[str, int, str], Running] = defaultdict(Running)
    spatial_audit = {
        "real_mentions": 0,
        "with_matching_boxes": 0,
        "without_matching_boxes": 0,
        "grid_mismatch": 0,
    }
    spatial_context = build_spatial_context(config)
    for shard in load_shards(result_dir / "shards"):
        positions = {
            str(row["target_key"]): row for row in shard.get("positions", ())
        }
        mentions = shard.get("sample_table", ())
        for mention in mentions:
            label = "real" if int(mention["label"]) == 1 else "hall"
            position = positions[str(mention["target_key"])]
            for source in SOURCES:
                stats = distribution_statistics(position["sources"][source])
                for statistic, values in stats.items():
                    for layer, value in enumerate(values.tolist(), start=1):
                        distribution_acc[(source, label, layer, statistic)].add(value)

        by_image: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for position in positions.values():
            by_image[int(position["image_id"])].append(position)
        for image_positions in by_image.values():
            for left_index in range(len(image_positions)):
                for right_index in range(left_index + 1, len(image_positions)):
                    left = image_positions[left_index]
                    right = image_positions[right_index]
                    for source in SOURCES:
                        left_p = left["sources"][source]
                        right_p = right["sources"][source]
                        if tuple(left_p.shape) != tuple(right_p.shape):
                            continue
                        for layer in range(int(left_p.shape[0])):
                            metrics = distribution_pair_metrics(
                                left_p[layer], right_p[layer]
                            )
                            for metric, value in metrics.items():
                                specificity_acc[(source, layer + 1, metric)].add(value)
        accumulate_spatial_analysis(
            model=args.model,
            positions=positions,
            mentions=mentions,
            spatial_context=spatial_context,
            accumulator=spatial_acc,
            audit=spatial_audit,
        )
        del shard, positions
    distribution_rows = _running_rows(
        distribution_acc, ("source", "label", "layer", "metric")
    )
    specificity_rows = _running_rows(
        specificity_acc, ("source", "layer", "metric")
    )
    spatial_rows = _running_rows(
        spatial_acc, ("source", "layer", "metric")
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "distribution": distribution_rows,
        "same_image_different_target": specificity_rows,
        "spatial": spatial_rows,
        "spatial_audit": spatial_audit,
    }
    atomic_json_save(payload, final_path)
    _write_analysis_csv(result_dir / "p_distribution.csv", distribution_rows)
    _write_analysis_csv(result_dir / "p_target_specificity.csv", specificity_rows)
    _write_analysis_csv(result_dir / "p_spatial_validation.csv", spatial_rows)
    plot_analysis(payload, result_dir)


def finalize_full_extraction(
    config: Mapping[str, Any], model: str, output_dir: Path, result_dir: Path
) -> dict[str, Any]:
    """Verify that all two-GPU shards form exactly the official cohort."""
    del config  # Reserved for future protocol metadata; cohort comes from frozen files.
    labels, generations, splits = _load_inputs(output_dir)
    expected_images = _available_image_ids(labels, generations, splits)
    expected_mentions: set[str] = set()
    expected_positions: set[str] = set()
    for image_id in expected_images:
        mentions, _indices, _targets = mentions_for_image(
            image_id=image_id,
            labeling_row=labels[image_id],
            response_token_ids=generations[image_id]["response_token_ids"],
        )
        expected_mentions.update(str(row["mention_id"]) for row in mentions)
        expected_positions.update(str(row["target_key"]) for row in mentions)

    actual_mention_set: set[str] = set()
    actual_position_set: set[str] = set()
    referenced_positions: set[str] = set()
    actual_images: set[int] = set()
    for shard in load_shards(result_dir / "shards"):
        for row in shard.get("positions", ()):
            key = str(row["target_key"])
            if key in actual_position_set:
                raise ValueError(f"Duplicate position row across JFFN shards: {key}")
            actual_position_set.add(key)
            actual_images.add(int(row["image_id"]))
        for row in shard.get("sample_table", ()):
            mention_id = str(row["mention_id"])
            if mention_id in actual_mention_set:
                raise ValueError(
                    f"Duplicate official mention row across JFFN shards: {mention_id}"
                )
            actual_mention_set.add(mention_id)
            referenced_positions.add(str(row["target_key"]))
            actual_images.add(int(row["image_id"]))
        del shard
    missing_references = referenced_positions - actual_position_set
    if missing_references:
        raise ValueError(
            "Sample table refers to missing position rows: "
            f"{sorted(missing_references)[:20]}"
        )
    differences = {
        "missing_images": sorted(set(expected_images) - actual_images),
        "extra_images": sorted(actual_images - set(expected_images)),
        "missing_positions": sorted(expected_positions - actual_position_set),
        "extra_positions": sorted(actual_position_set - expected_positions),
        "missing_mentions": sorted(expected_mentions - actual_mention_set),
        "extra_mentions": sorted(actual_mention_set - expected_mentions),
    }
    incomplete = {name: values[:20] for name, values in differences.items() if values}
    audit = {
        "schema_version": SCHEMA_VERSION,
        "model": model,
        "complete": not bool(incomplete),
        "expected_images": len(expected_images),
        "actual_images": len(actual_images),
        "expected_unique_positions": len(expected_positions),
        "actual_unique_positions": len(actual_position_set),
        "expected_official_mentions": len(expected_mentions),
        "actual_official_mentions": len(actual_mention_set),
        "differences_first20": incomplete,
    }
    atomic_json_save(audit, result_dir / "extraction_audit.json")
    if incomplete:
        raise RuntimeError(
            "JFFN shard cohort is incomplete or mixed; see extraction_audit.json: "
            f"{incomplete}"
        )
    return audit


def _running_rows(accumulator, names: Sequence[str]) -> list[dict[str, Any]]:
    rows = []
    for key, running in sorted(accumulator.items()):
        rows.append({**dict(zip(names, key)), **running.result()})
    return rows


def _write_analysis_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def spatial_analysis(*, config, model, positions, mentions):
    context = build_spatial_context(config)
    accumulator: dict[tuple[str, int, str], Running] = defaultdict(Running)
    audit = {
        "real_mentions": 0,
        "with_matching_boxes": 0,
        "without_matching_boxes": 0,
        "grid_mismatch": 0,
    }
    accumulate_spatial_analysis(
        model=model,
        positions=positions,
        mentions=mentions,
        spatial_context=context,
        accumulator=accumulator,
        audit=audit,
    )
    return accumulator, audit


def build_spatial_context(config):
    dataset = get_dataset_cfg(config)
    with Path(dataset["annotation_file"]).open("r", encoding="utf-8") as handle:
        annotations = json.load(handle)
    category_by_id = {
        int(row["id"]): str(row["name"]) for row in annotations["categories"]
    }
    image_size = {
        int(row["id"]): (int(row["width"]), int(row["height"]))
        for row in annotations["images"]
    }
    boxes: dict[tuple[int, str], list[tuple[float, float, float, float]]] = defaultdict(list)
    for annotation in annotations["annotations"]:
        if int(annotation.get("iscrowd", 0)):
            continue
        x, y, width, height = (float(value) for value in annotation["bbox"])
        boxes[(int(annotation["image_id"]), category_by_id[int(annotation["category_id"])])].append(
            (x, y, x + width, y + height)
        )
    return {"image_size": image_size, "boxes": boxes}


def accumulate_spatial_analysis(
    *, model, positions, mentions, spatial_context, accumulator, audit
) -> None:
    image_size = spatial_context["image_size"]
    boxes = spatial_context["boxes"]
    for mention in mentions:
        if int(mention["label"]) != 1:
            continue
        audit["real_mentions"] += 1
        image_id = int(mention["image_id"])
        category = str(mention.get("canonical_object") or "")
        target_boxes = boxes.get((image_id, category), ())
        if not target_boxes:
            audit["without_matching_boxes"] += 1
            continue
        position = positions[str(mention["target_key"])]
        grid = tuple(int(value) for value in position.get("visual_grid") or ())
        if len(grid) != 2 or grid[0] * grid[1] != int(
            position["sources"]["old_hmid_cos"].shape[-1]
        ):
            audit["grid_mismatch"] += 1
            continue
        audit["with_matching_boxes"] += 1
        overlap = patch_overlap_fraction(
            model=model,
            image_size=image_size[image_id],
            boxes=target_boxes,
            grid=grid,
        )
        uniform_mass = float(overlap.mean())
        labels = overlap > 0
        for source in SOURCES:
            p = position["sources"][source]
            for layer_index, layer_p in enumerate(p, start=1):
                mass = float((layer_p * overlap).sum())
                enrichment = mass / max(uniform_mass, 1e-12)
                pointing = float(bool(labels[int(torch.argmax(layer_p))]))
                aupr = binary_average_precision(labels, layer_p)
                for metric, value in (
                    ("bbox_mass", mass),
                    ("mass_enrichment", enrichment),
                    ("top1_pointing_accuracy", pointing),
                    ("patch_aupr", aupr),
                ):
                    accumulator[(source, layer_index, metric)].add(value)


def patch_overlap_fraction(*, model, image_size, boxes, grid) -> torch.Tensor:
    width, height = image_size
    grid_h, grid_w = grid
    transformed = []
    if model == "llava_1_5_7b":
        scale = 336.0 / min(width, height)
        resized_w, resized_h = width * scale, height * scale
        offset_x = (resized_w - 336.0) / 2.0
        offset_y = (resized_h - 336.0) / 2.0
        for x1, y1, x2, y2 in boxes:
            transformed.append(
                (
                    (x1 * scale - offset_x) / 336.0 * grid_w,
                    (y1 * scale - offset_y) / 336.0 * grid_h,
                    (x2 * scale - offset_x) / 336.0 * grid_w,
                    (y2 * scale - offset_y) / 336.0 * grid_h,
                )
            )
    else:
        transformed = [
            (
                x1 / width * grid_w,
                y1 / height * grid_h,
                x2 / width * grid_w,
                y2 / height * grid_h,
            )
            for x1, y1, x2, y2 in boxes
        ]
    values = []
    for row in range(grid_h):
        for column in range(grid_w):
            clipped = []
            for x1, y1, x2, y2 in transformed:
                rectangle = (
                    max(float(column), x1),
                    max(float(row), y1),
                    min(float(column + 1), x2),
                    min(float(row + 1), y2),
                )
                if rectangle[2] > rectangle[0] and rectangle[3] > rectangle[1]:
                    clipped.append(rectangle)
            values.append(rectangle_union_area(clipped))
    return torch.tensor(values, dtype=torch.float32).clamp(0.0, 1.0)


def rectangle_union_area(rectangles) -> float:
    if not rectangles:
        return 0.0
    x_values = sorted({value for rect in rectangles for value in (rect[0], rect[2])})
    area = 0.0
    for left, right in zip(x_values, x_values[1:]):
        if right <= left:
            continue
        intervals = sorted(
            (rect[1], rect[3])
            for rect in rectangles
            if rect[0] < right and rect[2] > left
        )
        covered = 0.0
        if intervals:
            start, end = intervals[0]
            for next_start, next_end in intervals[1:]:
                if next_start > end:
                    covered += end - start
                    start, end = next_start, next_end
                else:
                    end = max(end, next_end)
            covered += end - start
        area += (right - left) * covered
    return area


def binary_average_precision(labels: torch.Tensor, scores: torch.Tensor) -> float:
    labels = labels.to(dtype=torch.bool)
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = torch.argsort(scores, descending=True, stable=True)
    ranked = labels.index_select(0, order).float()
    precision = ranked.cumsum(dim=0) / torch.arange(
        1, len(ranked) + 1, dtype=torch.float32
    )
    return float((precision * ranked).sum() / positives)


def plot_analysis(payload: Mapping[str, Any], result_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = (
        ("distribution", "normalized_entropy", "Normalized entropy", "p_distribution_entropy.png"),
        ("same_image_different_target", "js", "Same-image target JS", "p_target_specificity_js.png"),
        ("spatial", "mass_enrichment", "BBox mass enrichment", "p_spatial_enrichment.png"),
    )
    for section, metric, title, filename in plots:
        rows = [row for row in payload[section] if row["metric"] == metric]
        if not rows:
            continue
        figure, axis = plt.subplots(figsize=(9, 5))
        grouping = defaultdict(list)
        for row in rows:
            label = str(row.get("label", "all"))
            grouping[(row["source"], label)].append(row)
        for (source, label), group in sorted(grouping.items()):
            group = sorted(group, key=lambda row: int(row["layer"]))
            axis.plot(
                [int(row["layer"]) for row in group],
                [float(row["mean"]) for row in group],
                label=f"{source}/{label}",
            )
        axis.set_xlabel("Decoder layer")
        axis.set_ylabel(title)
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7, ncol=2)
        figure.tight_layout()
        figure.savefig(result_dir / filename, dpi=180)
        plt.close(figure)

    _plot_metric_grid(
        rows=payload["distribution"],
        metrics=(
            "normalized_entropy",
            "effective_token_count",
            "top1_mass",
            "top5_mass",
            "top10_mass",
            "top32_mass",
            "max_over_uniform",
            "gini",
        ),
        group_field="label",
        title="P distribution diagnostics by REAL/HALL",
        path=result_dir / "p_distribution_all_metrics.png",
    )
    _plot_metric_grid(
        rows=payload["same_image_different_target"],
        metrics=("raw_cosine", "centered_cosine", "js", "tv", "top32_overlap"),
        group_field=None,
        title="Same-image different-target specificity",
        path=result_dir / "p_target_specificity_all_metrics.png",
    )
    _plot_metric_grid(
        rows=payload["spatial"],
        metrics=(
            "bbox_mass",
            "mass_enrichment",
            "top1_pointing_accuracy",
            "patch_aupr",
        ),
        group_field=None,
        title="COCO spatial localization",
        path=result_dir / "p_spatial_all_metrics.png",
    )


def _plot_metric_grid(*, rows, metrics, group_field, title, path) -> None:
    if not rows:
        return
    import matplotlib.pyplot as plt

    columns = 2
    row_count = math.ceil(len(metrics) / columns)
    figure, axes = plt.subplots(
        row_count, columns, figsize=(12, 3.6 * row_count), squeeze=False
    )
    for axis, metric in zip(axes.flat, metrics):
        selected = [row for row in rows if row["metric"] == metric]
        grouping = defaultdict(list)
        for row in selected:
            group = str(row.get(group_field, "all")) if group_field else "all"
            grouping[(str(row["source"]), group)].append(row)
        for (source, group), values in sorted(grouping.items()):
            values = sorted(values, key=lambda row: int(row["layer"]))
            label = source if group == "all" else f"{source}/{group}"
            axis.plot(
                [int(row["layer"]) for row in values],
                [float(row["mean"]) for row in values],
                label=label,
            )
        axis.set_title(metric)
        axis.set_xlabel("Decoder layer")
        axis.grid(alpha=0.25)
    for axis in axes.flat[len(metrics) :]:
        axis.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="upper center", ncol=4, fontsize=7)
    figure.suptitle(title, y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_model_report(model: str, result_dir: Path) -> None:
    training_path = result_dir / "training_results.json"
    analysis_path = result_dir / "p_analysis.json"
    lines = [f"# {model} JFFN-P comparison", ""]
    smoke_path = result_dir / "smoke_5images" / "extraction_audit.json"
    calibration_path = result_dir / "entropy_calibration.json"
    extraction_path = result_dir / "extraction_audit.json"
    qwen_gate_path = result_dir / "qwen_shared_forward_decision.json"
    if smoke_path.exists():
        smoke = load_json(str(smoke_path))["validation"]
        lines.extend(
            [
                "## Numerical acceptance",
                "",
                f"- finite/P-normalized: `{smoke['all_finite']}` / max sum error "
                f"`{smoke['max_distribution_sum_error']:.3e}`",
                f"- attention reconstruction: min cosine "
                f"`{smoke['min_attention_reconstruction_cosine']:.6f}`, max relative "
                f"error `{smoke['max_attention_reconstruction_relative_error']:.3e}`",
                f"- JVP: max linearity error `{smoke['max_linearity_relative_error']:.3e}`, "
                f"max full/chunk error `{smoke['max_parallel_chunk_relative_error']:.3e}`, "
                f"median finite-difference cosine "
                f"`{smoke['median_finite_difference_cosine']:.6f}`",
                f"- peak memory: `{smoke['peak_memory_bytes'] / 1024**3:.2f} GiB`",
                "",
            ]
        )
    if calibration_path.exists():
        calibration = load_json(str(calibration_path))
        max_entropy_error = max(
            (float(row["absolute_error"]) for row in calibration["layers"]),
            default=float("nan"),
        )
        lines.extend(
            [
                "## Cohort and calibration",
                "",
                f"- entropy beta fit: {calibration['processed_images']} fixed train-only "
                f"images; test leakage `{calibration['test_leakage_count']}`; max layer "
                f"entropy error `{max_entropy_error:.3e}`.",
            ]
        )
        if extraction_path.exists():
            extraction = load_json(str(extraction_path))
            lines.append(
                f"- extracted official cohort: {extraction.get('actual_images', '?')} "
                f"images, {extraction.get('actual_unique_positions', '?')} unique "
                f"positions, {extraction.get('actual_official_mentions', '?')} mentions; "
                f"complete=`{extraction.get('complete', False)}`."
            )
        if qwen_gate_path.exists():
            decision = load_json(str(qwen_gate_path))
            lines.append(
                "- Qwen forward mode: "
                + (
                    "shared-caption query rows"
                    if decision["enabled"]
                    else f"prefix fallback ({decision['reason']})"
                )
                + "."
            )
        lines.append("")
    if training_path.exists():
        training = load_json(str(training_path))
        lines.extend(
            [
                "## Preregistered downstream comparison",
                "",
                "| Comparison | New AUROC | Old AUROC | ΔAUROC | 95% image-bootstrap CI |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for name, row in training["main_comparison_bootstrap"].items():
            ci = row["ci95"]
            lines.append(
                f"| {name} | {row['seed_averaged_new_auroc']:.4f} | "
                f"{row['seed_averaged_old_auroc']:.4f} | {row['difference']:+.4f} | "
                f"[{ci[0]:+.4f}, {ci[1]:+.4f}] |"
            )
        lines.extend(
            [
                "",
                "### All 32 feature sets (mean ± population std over seeds 43/44/45)",
                "",
                "| Feature set | AUROC | Real F1 | Hall F1 | Real AUPR | Hall AUPR | Accuracy |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for feature_name, row in training["feature_sets"].items():
            aggregate = row["aggregate"]

            def formatted(metric: str) -> str:
                values = aggregate[metric]
                return f"{values['mean']:.4f} ± {values['std']:.4f}"

            lines.append(
                f"| `{feature_name}` | {formatted('auroc')} | "
                f"{formatted('real_f1')} | {formatted('hall_f1')} | "
                f"{formatted('real_aupr')} | {formatted('hall_aupr')} | "
                f"{formatted('accuracy')} |"
            )
        lines.extend(
            [
                "",
                "### Preregistered per-seed AUROC deltas",
                "",
                "| Comparison | seed 43 | seed 44 | seed 45 |",
                "|---|---:|---:|---:|",
            ]
        )
        for new_source in ("new_jffn", "new_jffn_entropy_matched"):
            new_rows = training["feature_sets"][_main_spec(new_source)]["seeds"]
            old_rows = training["feature_sets"][_main_spec("old_hpre_cos")]["seeds"]
            new_by_seed = {int(row["seed"]): float(row["auc"]) for row in new_rows}
            old_by_seed = {int(row["seed"]): float(row["auc"]) for row in old_rows}
            deltas = [new_by_seed[seed] - old_by_seed[seed] for seed in (43, 44, 45)]
            lines.append(
                f"| {new_source} vs old_hpre_cos | "
                + " | ".join(f"{value:+.4f}" for value in deltas)
                + " |"
            )
        lines.extend(["", "机器可读的逐 seed 全指标见 `training_results.json`。", ""])
    if analysis_path.exists():
        analysis = load_json(str(analysis_path))
        audit = analysis["spatial_audit"]
        lines.extend(
            [
                "## P 本身的验证",
                "",
                f"- 空间验证：{audit['with_matching_boxes']} 个 REAL mention 有同类别 COCO box；"
                f"{audit['without_matching_boxes']} 个无匹配 box；{audit['grid_mismatch']} 个网格不匹配。",
                "- 分布曲线：`p_distribution_entropy.png`。",
                "- 同图不同目标特异性：`p_target_specificity_js.png`。",
                "- COCO 空间富集：`p_spatial_enrichment.png`。",
                "- 全指标面板：`p_distribution_all_metrics.png`、"
                "`p_target_specificity_all_metrics.png`、`p_spatial_all_metrics.png`。",
                "",
            ]
        )
    (result_dir / f"{model}_jffn_p_comparison_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def write_cross_model_summary(config: Mapping[str, Any]) -> None:
    rows = []
    for model in MODELS:
        root = (
            ROOT
            / "outputs"
            / model
            / "COCO4000-INSLEN-OFFICIAL-TARGET"
            / "results"
            / "jffn_p_comparison"
        )
        path = root / "training_results.json"
        if not path.exists():
            continue
        payload = load_json(str(path))
        for comparison, values in payload["main_comparison_bootstrap"].items():
            rows.append({"model": model, "comparison": comparison, **values})
    if not rows:
        return
    output = ROOT / "outputs" / "jffn_p_comparison_4model_summary.md"
    lines = [
        "# Four-model JFFN-P comparison",
        "",
        "| Model | Comparison | ΔAUROC | 95% CI |",
        "|---|---|---:|---:|",
    ]
    for row in rows:
        ci = row["ci95"]
        lines.append(
            f"| {row['model']} | {row['comparison']} | {row['difference']:+.4f} | "
            f"[{ci[0]:+.4f}, {ci[1]:+.4f}] |"
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
