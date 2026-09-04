#!/usr/bin/env python3
"""Resumable all-layer Vector FFN source-attribution extraction."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.jffn_experiment import (  # noqa: E402
    assert_finite_position_rows,
    completed_image_ids,
    mentions_for_image,
)
from features.tc_fvpa_artifacts import (  # noqa: E402
    ExperimentLayout,
    atomic_json_save,
    atomic_torch_save,
    checksums_for_paths,
    initialize_manifests,
    update_stage_status,
)
from models import build_model  # noqa: E402
from scripts.run_jffn_p_comparison import (  # noqa: E402
    _extract_one_image,
    _load_inputs,
)
from scripts.run_tc_fvpa_path_attribution import _selected_images  # noqa: E402
from scripts.tc_fvpa_common import exact_command, shell_command  # noqa: E402
from utils.config_utils import (  # noqa: E402
    get_dgst_t_cfg,
    get_extraction_model_cfg,
    load_config,
)
from utils.io_utils import load_pkl  # noqa: E402


MODELS = (
    "llava_1_5_7b",
    "internvl_2_5_8b",
    "qwen2_5_vl_7b",
    "qwen3_vl_8b",
)
EXPERIMENT = "COCO4000-INSLEN-OFFICIAL-TARGET"
SCHEMA_VERSION = "ffn-visual-source-attribution-v1"
EPS = 1e-12


def result_root(model: str) -> Path:
    return (
        ROOT
        / "outputs"
        / model
        / EXPERIMENT
        / "results"
        / "ffn_visual_source_attribution_v1"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument(
        "--config", default="configs/model_configs_inslen_official_target.yaml"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-images", type=int, default=0)
    parser.add_argument("--shard-images", type=int, default=10)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--token-chunk-size", type=int, default=256)
    parser.add_argument(
        "--jvp-backend", choices=("auto", "vmap_jvp", "linearize"), default="auto"
    )
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _normalized(values: np.ndarray) -> np.ndarray:
    finite = np.maximum(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    total = finite.sum(axis=-1, keepdims=True)
    return np.divide(
        finite,
        np.maximum(total, EPS),
        out=np.full_like(finite, 1.0 / float(finite.shape[-1])),
        where=total > 0.0,
    ).astype(np.float32, copy=False)


def formal_image_ids(labels, generations, splits) -> list[int]:
    """Keep every fixed-split image, including images without target spans."""
    split_ids = {
        int(value) for name in ("train", "test") for value in splits[name]
    }
    return sorted(split_ids & set(labels) & set(generations))


def load_target_lookup(
    model_root: Path, image_ids: set[int]
) -> tuple[dict[tuple[int, int], dict[str, np.ndarray]], dict[str, Any]]:
    """Load the exact existing hpre-raw target and old-Rcos trajectory once."""
    rows = load_pkl(str(model_root / "features.pkl"))
    lookup: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    duplicates = 0
    maximum_duplicate_error = 0.0
    for row in rows:
        image_id = int(row["image_id"])
        if image_id not in image_ids:
            continue
        key = (image_id, int(row["response_token_idx"]))
        attention = np.asarray(
            row["dgst_t_attention_support_per_layer"], dtype=np.float32
        )
        gate = np.asarray(
            row["dgst_t_hpre_raw_logit_gauss_gate_per_layer"], dtype=np.float32
        )
        evidence = np.maximum(
            np.nan_to_num(attention * gate, nan=0.0, posinf=0.0, neginf=0.0),
            0.0,
        )
        value = {
            "target": _normalized(evidence),
            "strength": evidence.sum(axis=-1).astype(np.float32),
            "r_cos": np.asarray(
                row[
                    "dgst_t_hpre_raw_logit_gauss_source_hpre_cos_"
                    "risk_sqrt_hpre_per_layer"
                ],
                dtype=np.float32,
            ),
        }
        previous = lookup.get(key)
        if previous is None:
            lookup[key] = value
        else:
            duplicates += 1
            error = max(
                float(np.max(np.abs(previous[name] - value[name])))
                for name in value
            )
            maximum_duplicate_error = max(maximum_duplicate_error, error)
            if error > 1e-6:
                raise AssertionError(f"Conflicting existing features for target {key}")
    row_count = len(rows)
    del rows
    gc.collect()
    return lookup, {
        "feature_rows": row_count,
        "selected_images": len(image_ids),
        "unique_targets": len(lookup),
        "duplicate_mentions": duplicates,
        "maximum_duplicate_error": maximum_duplicate_error,
        "target_formula": (
            "normalize(dgst_t_attention_support_per_layer * "
            "dgst_t_hpre_raw_logit_gauss_gate_per_layer)"
        ),
        "r_cos_origin": (
            "old_hpre_cos source, hpre_raw_logit_gauss target, "
            "sqrt_matched_state OT"
        ),
    }


def source_config(
    base: Mapping[str, Any], *, model: str, k: int, chunk: int, backend: str
) -> dict[str, Any]:
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
            "jffn_qwen_shared_forward": model.startswith("qwen"),
            "jffn_vector_path_only": True,
            "jffn_vector_path_integration_points": int(k),
            "jffn_vector_path_method": "gauss_legendre",
            "jffn_vector_path_chunk_size": int(chunk),
            "jffn_vector_path_jvp_backend": str(backend),
        }
    )
    return cfg


def compact_position(
    position: Mapping[str, Any], lookup: Mapping[tuple[int, int], Mapping[str, np.ndarray]]
) -> dict[str, Any]:
    result = position["result"]
    key = (int(position["image_id"]), int(position["response_index"]))
    old = lookup[key]
    compact = {
        "target_key": f"{key[0]}:{key[1]}",
        "image_id": key[0],
        "response_index": key[1],
        "target_token_id": int(position["target_token_id"]),
        "visual_grid": [int(value) for value in position.get("visual_grid") or ()],
        "attention_evidence": result["ffn_source_attention_evidence"].float(),
        "ae_strength": result["ffn_source_ae_strength"].float(),
        "write_mag": result["ffn_source_write_energy"].float(),
        "p_write": result["ffn_source_write_distribution"].float(),
        "ffn_path_gross": result["ffn_source_ffn_path_gross"].float(),
        "p_ffn": result["ffn_source_ffn_distribution"].float(),
        "path_signed_q": result["ffn_source_path_signed_q"].float(),
        "gross_strength": result["ffn_source_gross_strength"].float(),
        "net_strength": result["ffn_source_net_strength"].float(),
        "kappa": result["ffn_source_kappa"].float(),
        "completeness_relative_error": result[
            "ffn_source_completeness_relative_error"
        ].float(),
        "gross_degenerate": result["ffn_source_gross_degenerate"].bool(),
        "net_degenerate": result["ffn_source_net_degenerate"].bool(),
        "js": {
            name: result[f"ffn_source_js_{name}"].float()
            for name in ("D_EW", "D_WF", "D_EF")
        },
        "ot": {
            name: result[f"ffn_source_ot_{name}"].float()
            for name in ("D_EW", "D_WF", "D_EF")
        },
        "r_cos": torch.as_tensor(old["r_cos"], dtype=torch.float32),
        "token_chunk_size": result["ffn_source_token_chunk_size"].int(),
        "attention_reconstruction_relative_error": result[
            "ffn_source_attention_reconstruction_relative_error"
        ].float(),
        "component_sum_relative_error": result[
            "ffn_source_component_sum_relative_error"
        ].float(),
    }
    assert_finite_position_rows([compact])
    return compact


def resolve_frozen_settings(root: Path, k: int, backend: str) -> tuple[int, str]:
    frozen_k = int(k)
    if frozen_k <= 0:
        path = root / "tables/frozen_k.json"
        if not path.exists():
            raise FileNotFoundError(f"Run analyze_ffn_visual_source_study.py first: {path}")
        frozen_k = int(json.loads(path.read_text(encoding="utf-8"))["frozen_k"])
    selected_backend = backend
    if backend == "auto":
        path = root / "tables/linearize_decision.json"
        selected_backend = "vmap_jvp"
        if path.exists():
            selected_backend = str(
                json.loads(path.read_text(encoding="utf-8")).get(
                    "selected_backend", "vmap_jvp"
                )
            )
    return frozen_k, selected_backend


def main() -> None:
    args = parse_args()
    if args.world_size <= 0 or not 0 <= args.rank < args.world_size:
        raise ValueError(f"Invalid rank/world-size {args.rank}/{args.world_size}")
    if args.shard_images <= 0 or args.token_chunk_size <= 0:
        raise ValueError("shard-images and token-chunk-size must be positive")
    root = result_root(args.model)
    layout = ExperimentLayout.create(root)
    k, backend = resolve_frozen_settings(root, args.k, args.jvp_backend)
    if backend not in {"vmap_jvp", "linearize"}:
        raise ValueError(f"Invalid frozen JVP backend {backend!r}")
    model_root = ROOT / "outputs" / args.model / EXPERIMENT
    config = load_config(args.config)
    labels, generations, splits = _load_inputs(model_root)
    available = formal_image_ids(labels, generations, splits)
    if args.smoke:
        image_ids = _selected_images(
            labels, generations, splits, count=int(args.num_images or 1), seed=20260829
        )
    else:
        image_ids = available[: int(args.num_images or len(available))]
    worker_ids = image_ids[args.rank :: args.world_size]
    stage_kind = "smoke" if args.smoke else "full_attribution"
    stage = f"{stage_kind}:{args.model}:rank{args.rank}"
    initialize_manifests(
        layout=layout,
        repo_root=ROOT,
        experiment_config={
            "schema_version": SCHEMA_VERSION,
            "model": args.model,
            "frozen_k": k,
            "quadrature": "gauss_legendre",
            "jvp_backend": backend,
            "token_chunk_fallback": [256, 128, 64, 32],
            "output_root": str(root),
        },
        input_paths=[
            args.config,
            model_root / "features.pkl",
            model_root / "generations.json",
            model_root / "labeling.json",
            model_root / "image_splits.json",
        ],
        exact_command=exact_command(),
    )
    if args.rank == 0 and not (root / "manifests/full_input_checksums.json").exists():
        atomic_json_save(
            checksums_for_paths(
                [
                    args.config,
                    model_root / "features.pkl",
                    model_root / "generations.json",
                    model_root / "labeling.json",
                    model_root / "image_splits.json",
                ]
            ),
            root / "manifests/full_input_checksums.json",
        )
    update_stage_status(
        layout=layout,
        stage=stage,
        status="RUNNING",
        details={"worker_images": len(worker_ids), "k": k, "backend": backend},
        resume_command=shell_command(),
    )
    shard_dir = root / ("shards/smoke" if args.smoke else "shards/full")
    shard_dir.mkdir(parents=True, exist_ok=True)
    done = completed_image_ids(shard_dir) if args.resume else set()
    if not args.resume and any(shard_dir.glob("features*_shard_*.pt")):
        raise FileExistsError(f"Refusing to overwrite {shard_dir}")
    pending = [image_id for image_id in worker_ids if image_id not in done]
    lookup, lookup_audit = load_target_lookup(model_root, set(worker_ids))
    missing = []
    for image_id in worker_ids:
        response_ids = generations[image_id].get("response_token_ids") or ()
        _mentions, indices, _targets = mentions_for_image(
            image_id=image_id,
            labeling_row=labels[image_id],
            response_token_ids=response_ids,
        )
        missing.extend(
            (image_id, index) for index in indices if (image_id, index) not in lookup
        )
    if missing:
        raise KeyError(f"Existing hpre target lookup misses {len(missing)} targets: {missing[:5]}")

    wrapper = None
    failures = []
    batch_images: list[int] = []
    batch_positions: list[dict[str, Any]] = []
    batch_mentions: list[dict[str, Any]] = []
    prefix = f"features_rank{args.rank:02d}_shard"
    existing_indices = [
        int(path.stem.rsplit("_", 1)[-1])
        for path in shard_dir.glob(f"{prefix}_*.pt")
    ]
    shard_index = max(existing_indices, default=-1) + 1
    started = time.perf_counter()
    status = "FAIL"
    try:
        wrapper = build_model(
            args.model,
            get_extraction_model_cfg(config, args.model),
            device=args.device,
        )
        wrapper.model.requires_grad_(False)
        wrapper.model.eval()
        cfg = source_config(
            get_dgst_t_cfg(config),
            model=args.model,
            k=k,
            chunk=args.token_chunk_size,
            backend=backend,
        )
        for offset, image_id in enumerate(pending, 1):
            try:
                response_ids = [
                    int(value) for value in generations[image_id]["response_token_ids"]
                ]
                mentions, indices, _targets = mentions_for_image(
                    image_id=image_id,
                    labeling_row=labels[image_id],
                    response_token_ids=response_ids,
                )
                image_cfg = dict(cfg)
                image_cfg["jffn_vector_path_target_distributions"] = np.stack(
                    [lookup[(image_id, index)]["target"] for index in indices]
                ) if indices else np.empty((0, 0, 0), dtype=np.float32)
                image_cfg["jffn_vector_path_evidence_strengths"] = np.stack(
                    [lookup[(image_id, index)]["strength"] for index in indices]
                ) if indices else np.empty((0, 0), dtype=np.float32)
                positions, returned_mentions = _extract_one_image(
                    wrapper=wrapper,
                    config=config,
                    image_id=image_id,
                    label_row=labels[image_id],
                    generation_row=generations[image_id],
                    dgst_cfg=image_cfg,
                    compact=False,
                )
                if len(returned_mentions) != len(mentions):
                    raise AssertionError("Mention alignment changed during extraction")
                batch_images.append(int(image_id))
                batch_positions.extend(compact_position(row, lookup) for row in positions)
                batch_mentions.extend(returned_mentions)
            except Exception as exc:
                failures.append(
                    {
                        "image_id": int(image_id),
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                break
            if len(batch_images) >= args.shard_images or offset == len(pending):
                path = shard_dir / f"{prefix}_{shard_index:05d}.pt"
                atomic_torch_save(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "model": args.model,
                        "image_ids": batch_images,
                        "positions": batch_positions,
                        "sample_table": batch_mentions,
                        "provenance": {
                            "frozen_k": k,
                            "quadrature": "gauss_legendre",
                            "jvp_backend": backend,
                            "rank": args.rank,
                            "world_size": args.world_size,
                        },
                    },
                    path,
                )
                print(
                    f"[FFN source] rank={args.rank} shard={shard_index} "
                    f"images={len(batch_images)} positions={len(batch_positions)}",
                    flush=True,
                )
                shard_index += 1
                batch_images, batch_positions, batch_mentions = [], [], []
        if failures:
            raise RuntimeError(f"Formal extraction stopped at first failure: {failures[0]}")
        completed = completed_image_ids(shard_dir)
        if not set(worker_ids).issubset(completed):
            raise RuntimeError(
                f"Incomplete worker manifest: {len(set(worker_ids) - completed)} images"
            )
        status = "PASS"
    except Exception:
        status = "FAIL"
        raise
    finally:
        elapsed = time.perf_counter() - started
        audit = {
            "schema_version": SCHEMA_VERSION,
            "model": args.model,
            "rank": args.rank,
            "world_size": args.world_size,
            "requested_images": len(worker_ids),
            "already_complete": len(done & set(worker_ids)),
            "pending_at_start": len(pending),
            "failures": failures,
            "lookup": lookup_audit,
            "elapsed_seconds": elapsed,
        }
        atomic_json_save(
            audit,
            root / f"manifests/{stage_kind}_rank{args.rank:02d}.json",
        )
        if status == "PASS":
            atomic_json_save(
                checksums_for_paths(shard_dir.glob(f"{prefix}_*.pt")),
                root / f"manifests/{stage_kind}_rank{args.rank:02d}_checksums.json",
            )
        update_stage_status(
            layout=layout,
            stage=stage,
            status=status,
            details=audit,
            resume_command=shell_command(),
        )
        if wrapper is not None:
            del wrapper
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
