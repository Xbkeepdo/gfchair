#!/usr/bin/env python3
"""Extract WRITE, signed-Q, and attention controls without overwriting round one."""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.jffn_experiment import (  # noqa: E402
    atomic_json_save,
    atomic_torch_save,
    assert_finite_position_rows,
    completed_image_ids,
    mentions_for_image,
)
from models import build_model  # noqa: E402
from scripts.run_jffn_p_comparison import (  # noqa: E402
    _available_image_ids,
    _image_path,
    _load_inputs,
    _requirements,
)
from utils.config_utils import (  # noqa: E402
    get_dgst_t_cfg,
    get_extraction_model_cfg,
    load_config,
)
from utils.io_utils import load_pkl  # noqa: E402


MODELS = ("llava_1_5_7b", "internvl_2_5_8b")
SCHEMA_VERSION = "jffn-second-round-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument(
        "--config", default="configs/model_configs_inslen_official_target.yaml"
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--result-dir")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-images", type=int, default=0)
    parser.add_argument("--shard-images", type=int, default=50)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--union-aggregate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Compute exact vector-sum aggregate S inside "
            "Top32(P_JFFN) union Top32(Q_raw)."
        ),
    )
    return parser.parse_args()


def _second_round_config(base: Mapping[str, Any], model: str) -> dict[str, Any]:
    cfg = dict(base)
    cfg.update(
        {
            "enabled": True,
            "feature_output_profile": "four_gate_vv",
            "target_gate_mode": "four_gate",
            "four_gate_methods": ["hpre_raw_logit_gauss"],
            "source_modes": ["jffn"],
            "support_modes": ["vv"],
            "cost_modes": ["sqrt_matched_state"],
            "compute_capped_topmass_085": False,
            "compute_ffn_injection_features": False,
            "compute_prompt_cafe": False,
            "jffn_second_round_diagnostics": True,
            "jffn_second_round_only": True,
            "jffn_validate_linearity": False,
            "jffn_finite_difference_eta": None,
            "jffn_parity_chunk_size": None,
            "jffn_force_full_parallel": model in MODELS,
            "jffn_max_increment_gib": 1.5,
        }
    )
    return cfg


def _cpu_float(value: Any) -> torch.Tensor:
    return torch.as_tensor(value).detach().to(device="cpu", dtype=torch.float32)


def _raw_target_lookup_for_images(
    model_root: Path, image_ids: set[int]
) -> tuple[dict[tuple[int, int], np.ndarray], dict[str, Any]]:
    """Stream compact feature parts and retain raw-gate Q for one worker."""
    lookup: dict[tuple[int, int], np.ndarray] = {}
    duplicates = 0
    maximum_duplicate_error = 0.0
    rows_seen = 0
    selected_rows = 0
    parts = sorted(model_root.glob("features.part*.pkl"))
    if not parts:
        raise FileNotFoundError(f"No feature parts under {model_root}")
    for part in parts:
        rows = load_pkl(str(part))
        for row in rows:
            rows_seen += 1
            image_id = int(row["image_id"])
            if image_id not in image_ids:
                continue
            selected_rows += 1
            key = (image_id, int(row["response_token_idx"]))
            attention = np.asarray(
                row["dgst_t_attention_support_per_layer"], dtype=np.float32
            )
            gate = np.asarray(
                row["dgst_t_hpre_raw_logit_gauss_gate_per_layer"], dtype=np.float32
            )
            target = np.maximum(
                np.nan_to_num(attention * gate, nan=0.0, posinf=0.0, neginf=0.0),
                0.0,
            )
            total = target.sum(axis=-1, keepdims=True)
            target = np.divide(
                target,
                np.maximum(total, 1.0e-12),
                out=np.full_like(target, 1.0 / float(target.shape[-1])),
                where=total > 0.0,
            ).astype(np.float32, copy=False)
            previous = lookup.get(key)
            if previous is None:
                lookup[key] = target
            else:
                duplicates += 1
                error = float(np.max(np.abs(previous - target)))
                maximum_duplicate_error = max(maximum_duplicate_error, error)
                if error > 1.0e-6:
                    raise AssertionError(f"Conflicting target Q for {key}")
        del rows
        gc.collect()
    return lookup, {
        "formula": "normalize(attention_support * hpre_raw_logit_gauss_gate)",
        "feature_parts": len(parts),
        "rows_seen": rows_seen,
        "selected_rows": selected_rows,
        "unique_positions": len(lookup),
        "duplicate_rows": duplicates,
        "maximum_duplicate_error": maximum_duplicate_error,
    }


def compact_position(
    *,
    image_id: int,
    response_index: int,
    target_token_id: int,
    visual_grid: list[int] | tuple[int, ...] | None,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    position = {
        "target_key": f"{int(image_id)}:{int(response_index)}",
        "image_id": int(image_id),
        "response_index": int(response_index),
        "target_token_id": int(target_token_id),
        "visual_grid": [int(value) for value in visual_grid or ()],
        "attention_distribution": _cpu_float(
            result["dgst_t_second_round_attention_dist_per_layer"]
        ),
        "write_energy": _cpu_float(
            result["dgst_t_second_round_write_energy_per_layer"]
        ),
        "write_distribution": _cpu_float(
            result["dgst_t_second_round_write_dist_per_layer"]
        ),
        "jffn_energy": _cpu_float(result["dgst_t_jffn_energy_per_layer"]),
        "jffn_distribution": _cpu_float(
            result["dgst_t_source_jffn_dist_per_layer"]
        ),
        "signed_q": _cpu_float(
            result["dgst_t_second_round_signed_q_per_layer"]
        ),
        "diagnostics": {
            "input_norm": _cpu_float(result["dgst_t_jffn_input_norm_per_layer"]),
            "response_norm": _cpu_float(
                result["dgst_t_jffn_response_norm_per_layer"]
            ),
            "gain": _cpu_float(result["dgst_t_jffn_gain_per_layer"]),
            "direction_cosine": _cpu_float(
                result["dgst_t_jffn_direction_cosine_per_layer"]
            ),
            "q_conservation_relative_error": _cpu_float(
                result["dgst_t_second_round_q_conservation_error_per_layer"]
            ),
            "cancellation_ratio": _cpu_float(
                result["dgst_t_second_round_cancellation_ratio_per_layer"]
            ),
            "attention_reconstruction_relative_error": _cpu_float(
                result[
                    "dgst_t_jffn_attention_reconstruction_relative_error_per_layer"
                ]
            ),
            "component_sum_relative_error": _cpu_float(
                result["dgst_t_jffn_component_sum_relative_error_per_layer"]
            ),
        },
    }
    union_prefix = "dgst_t_union_aggregate_"
    union_fields = (
        "input_norm",
        "response_norm",
        "gain",
        "direction_cosine",
        "input_cancellation_ratio",
        "response_cancellation_ratio",
        "union_size",
    )
    if f"{union_prefix}gain_per_layer" in result:
        position["union_aggregate"] = {
            field: (
                torch.as_tensor(result[f"{union_prefix}{field}_per_layer"])
                .detach()
                .to(
                    device="cpu",
                    dtype=(torch.int64 if field == "union_size" else torch.float32),
                )
            )
            for field in union_fields
        }
    return position


def extract_one_image(
    *,
    wrapper: Any,
    config: Mapping[str, Any],
    image_id: int,
    label_row: Mapping[str, Any],
    generation_row: Mapping[str, Any],
    dgst_cfg: Mapping[str, Any],
    target_lookup: Mapping[tuple[int, int], np.ndarray] | None = None,
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
    image_dgst_cfg = dict(dgst_cfg)
    if target_lookup is not None:
        missing = [
            (int(image_id), int(response_index))
            for response_index in response_indices
            if (int(image_id), int(response_index)) not in target_lookup
        ]
        if missing:
            raise KeyError(f"Missing raw target Q for {missing}")
        image_dgst_cfg["jffn_union_target_distributions"] = np.stack(
            [
                target_lookup[(int(image_id), int(response_index))]
                for response_index in response_indices
            ]
        )
        image_dgst_cfg["jffn_union_side_top_k"] = 32
    with Image.open(_image_path(config, image_id)) as source:
        image = source.convert("RGB")
        outputs = wrapper.extract_token_features_batch(
            image=image,
            response_token_ids=response_ids,
            response_token_indices=response_indices,
            target_token_ids=target_ids,
            cfg_dgst_t=image_dgst_cfg,
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
        positions.append(
            compact_position(
                image_id=image_id,
                response_index=response_index,
                target_token_id=target_id,
                visual_grid=output.visual_grid,
                result=output.dgst_t_result,
            )
        )
    return positions, mentions


def main() -> None:
    args = parse_args()
    if args.world_size <= 0 or not 0 <= args.rank < args.world_size:
        raise ValueError(f"Invalid rank/world-size {args.rank}/{args.world_size}")
    if args.shard_images <= 0:
        raise ValueError("--shard-images must be positive")
    config = load_config(args.config)
    output_dir = Path(
        args.output_dir
        or ROOT / "outputs" / args.model / "COCO4000-INSLEN-OFFICIAL-TARGET"
    ).resolve()
    default_result_name = (
        "jffn_union_aggregate" if args.union_aggregate else "jffn_second_round"
    )
    result_dir = Path(
        args.result_dir or output_dir / "results" / default_result_name
    ).resolve()
    shard_dir = result_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    labels, generations, splits = _load_inputs(output_dir)
    all_image_ids = _available_image_ids(labels, generations, splits)
    if args.num_images > 0:
        all_image_ids = all_image_ids[: args.num_images]
    worker_image_ids = all_image_ids[args.rank :: args.world_size]
    done = completed_image_ids(shard_dir) if args.resume else set()
    if not args.resume and any(shard_dir.glob("features*_shard_*.pt")):
        raise FileExistsError(f"Refusing to append to non-empty {shard_dir}")
    pending = [image_id for image_id in worker_image_ids if image_id not in done]

    target_lookup = None
    target_audit = None
    if args.union_aggregate:
        target_lookup, target_audit = _raw_target_lookup_for_images(
            output_dir, set(worker_image_ids)
        )

    model_cfg = get_extraction_model_cfg(config, args.model)
    wrapper = build_model(args.model, model_cfg, device=args.device)
    wrapper.model.requires_grad_(False)
    wrapper.model.eval()
    dgst_cfg = _second_round_config(get_dgst_t_cfg(config), args.model)

    prefix = (
        f"features_rank{args.rank:02d}_shard"
        if args.world_size > 1
        else "features_shard"
    )
    indices = [
        int(path.stem.rsplit("_", 1)[-1])
        for path in shard_dir.glob(f"{prefix}_*.pt")
    ]
    shard_index = max(indices, default=-1) + 1
    batch_images: list[int] = []
    batch_positions: list[dict[str, Any]] = []
    batch_mentions: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    memory: list[dict[str, int]] = []

    for offset, image_id in enumerate(pending, start=1):
        try:
            device = torch.device(args.device)
            torch.cuda.reset_peak_memory_stats(device)
            before = int(torch.cuda.memory_allocated(device))
            positions, mentions = extract_one_image(
                wrapper=wrapper,
                config=config,
                image_id=image_id,
                label_row=labels[image_id],
                generation_row=generations[image_id],
                dgst_cfg=dgst_cfg,
                target_lookup=target_lookup,
            )
            peak = int(torch.cuda.max_memory_allocated(device))
            memory.append(
                {
                    "image_id": int(image_id),
                    "before_bytes": before,
                    "peak_bytes": peak,
                    "increment_bytes": peak - before,
                }
            )
            batch_images.append(int(image_id))
            batch_positions.extend(positions)
            batch_mentions.extend(mentions)
        except Exception as exc:
            failures.append({"image_id": int(image_id), "error": repr(exc)})
            raise

        if len(batch_images) >= args.shard_images or offset == len(pending):
            assert_finite_position_rows(batch_positions)
            path = shard_dir / f"{prefix}_{shard_index:05d}.pt"
            atomic_torch_save(
                {
                    "schema_version": (
                        "jffn-union-aggregate-v1"
                        if args.union_aggregate
                        else SCHEMA_VERSION
                    ),
                    "model": args.model,
                    "image_ids": list(batch_images),
                    "positions": batch_positions,
                    "sample_table": batch_mentions,
                    "provenance": {
                        "target_protocol": (
                            "inslen_official_first_token_first_occurrence"
                        ),
                        "round": (
                            "jffn_union_aggregate"
                            if args.union_aggregate
                            else "jffn_second_round"
                        ),
                        "rank": int(args.rank),
                        "world_size": int(args.world_size),
                    },
                },
                path,
            )
            print(
                f"[second round] shard {shard_index}: {len(batch_images)} images, "
                f"{len(batch_positions)} positions, {len(batch_mentions)} mentions",
                flush=True,
            )
            shard_index += 1
            batch_images = []
            batch_positions = []
            batch_mentions = []

    worker_complete = set(worker_image_ids).issubset(done | set(pending)) and not failures
    audit = {
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "global_requested_images": len(all_image_ids),
        "worker_requested_images": len(worker_image_ids),
        "rank": int(args.rank),
        "world_size": int(args.world_size),
        "pending_at_start": len(pending),
        "worker_complete": bool(worker_complete),
        "failures": failures,
        "peak_memory_bytes": max(
            (entry["peak_bytes"] for entry in memory), default=0
        ),
        "memory": memory,
        "union_aggregate": bool(args.union_aggregate),
        "target_audit": target_audit,
    }
    atomic_json_save(audit, result_dir / f"extraction_audit_rank{args.rank:02d}.json")
    print(f"[second round] worker rank {args.rank} complete", flush=True)


if __name__ == "__main__":
    main()
