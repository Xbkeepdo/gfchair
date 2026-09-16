#!/usr/bin/env python3
"""Extract ENDAC first-token semantic-attention Top-K matrices and features."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.semantic_attention_topk import FEATURE_NAMES, layer_matrices, topk_features
from models import build_model
from models.dgst_capture import (
    resolve_decoder_final_norm,
    resolve_decoder_layers,
    resolve_output_embedding_layer,
)
from scripts.extract_endac_811 import (
    MODELS,
    _capture,
    _load_json,
    _split_lookup,
    _targets,
)
from utils.config_utils import get_dataset_cfg, get_extraction_model_cfg, load_config
from utils.io_utils import append_pkl, load_pkl, save_pkl

DEFAULT_CONFIG = ROOT / "configs/model_configs_endac_811.yaml"


def _root(config: dict, key: str) -> Path:
    value = Path(config[key]["output_root"])
    return value if value.is_absolute() else ROOT / value


def _save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def extract_image(
    *,
    wrapper,
    model: str,
    config: dict,
    image_id: int,
    label_row: dict,
    generation_row: dict,
    split_name: str,
) -> list[dict]:
    response_ids = [int(value) for value in generation_row["response_token_ids"]]
    targets = _targets(label_row, response_ids)
    if not targets:
        return []
    dataset = get_dataset_cfg(config)
    image_path = (
        Path(dataset["coco_root"])
        / "val2014"
        / f"COCO_val2014_{int(image_id):012d}.jpg"
    )
    prompt = str(config.get("run", {}).get("prompt") or "Describe this image.")
    with Image.open(image_path) as source:
        captures, positions, visual_start, visual_end, visual_grid, _ = _capture(
            wrapper,
            model,
            source.convert("RGB"),
            response_ids,
            targets,
            prompt,
        )

    settings = config["semantic_attention_topk"]
    layers = resolve_decoder_layers(wrapper.model)
    expected_layers = int(config["endac_811"]["models"][model]["num_layers"])
    if len(layers) != expected_layers or len(captures) != expected_layers:
        raise ValueError(f"{model}: expected {expected_layers} decoder layers")
    output_layer = resolve_output_embedding_layer(wrapper.model)
    final_norm = resolve_decoder_final_norm(wrapper.model)
    top_ks = tuple(int(value) for value in settings["top_k"])
    accumulated = [
        {
            "matrices": {
                "attention": [],
                "object_probability_raw": [],
                "object_probability_norm": [],
                "cosine_similarity": [],
            },
            "top_indices": {
                f"{variant}_top{k}": []
                for variant in ("raw", "norm")
                for k in top_ks
            },
            "features": {
                f"{variant}_top{k}": []
                for variant in ("raw", "norm")
                for k in top_ks
            },
        }
        for _ in targets
    ]

    for layer_index, capture in enumerate(captures):
        matrices = layer_matrices(
            capture=capture,
            prediction_positions=positions,
            target_token_ids=[row["target_token_id"] for row in targets],
            visual_start=visual_start,
            visual_end=visual_end,
            output_layer=output_layer,
            final_norm=final_norm,
            vocab_chunk_size=int(settings["vocab_chunk_size"]),
        )
        for target_index, item in enumerate(accumulated):
            attention = matrices["attention"][target_index]
            cosine = matrices["cosine_similarity"][target_index]
            for name in item["matrices"]:
                item["matrices"][name].append(
                    matrices[name][target_index].float().cpu()
                )
            for variant in ("raw", "norm"):
                probability = matrices[f"object_probability_{variant}"][target_index]
                for top_k in top_ks:
                    indices, features = topk_features(
                        probability,
                        attention,
                        cosine,
                        top_k,
                    )
                    key = f"{variant}_top{top_k}"
                    item["top_indices"][key].append(indices.to(torch.int16).cpu())
                    item["features"][key].append(features.float().cpu())
        captures[layer_index] = None
        del capture, matrices

    rows = []
    for target, item in zip(targets, accumulated):
        matrices = {
            name: torch.stack(values).numpy()
            for name, values in item["matrices"].items()
        }
        top_indices = {
            name: torch.stack(values).numpy()
            for name, values in item["top_indices"].items()
        }
        features = {
            name: torch.stack(values).numpy().astype(np.float32, copy=False)
            for name, values in item["features"].items()
        }
        if any(not np.isfinite(value).all() for value in (*matrices.values(), *features.values())):
            raise ValueError(f"{target['sample_id']}: non-finite semantic-attention value")
        rows.append(
            {
                "sample_id": target["sample_id"],
                "image_id": int(image_id),
                "response_token_idx": target["response_index"],
                "target_token_id": target["target_token_id"],
                "token_indices": target["token_indices"],
                "token": target["token"],
                "canonical_object": target["canonical_object"],
                "label": target["label"],
                "split": split_name,
                "visual_grid": list(visual_grid),
                "feature_names": list(FEATURE_NAMES),
                "matrices": matrices,
                "top_indices": top_indices,
                "features": features,
            }
        )
    return rows


def merge(model: str, part_paths: list[Path], target: Path, summary_path: Path) -> None:
    rows, seen = [], set()
    for path in (target, *part_paths):
        if not path.exists():
            continue
        for row in load_pkl(path):
            sample_id = str(row["sample_id"])
            if sample_id not in seen:
                rows.append(row)
                seen.add(sample_id)
    rows.sort(key=lambda row: (int(row["image_id"]), str(row["sample_id"])))
    save_pkl(rows, target)
    summary = {
        "model": model,
        "images": len({int(row["image_id"]) for row in rows}),
        "samples": len(rows),
        "layers": sorted({int(row["matrices"]["attention"].shape[0]) for row in rows}),
        "feature_names": list(FEATURE_NAMES),
        "matrix_dtype": "float32",
        "top_k": [16, 32],
        "split_samples": {
            name: sum(row["split"] == name for row in rows)
            for name in ("train", "val", "test")
        },
    }
    _save_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def run(args: argparse.Namespace) -> None:
    config = load_config(str(args.config))
    source_root = _root(config, "endac_811")
    output_root = _root(config, "semantic_attention_topk")
    source = source_root / args.model
    output = output_root / args.model
    output.mkdir(parents=True, exist_ok=True)
    labels = {int(key): value for key, value in _load_json(source / "labeling.json").items()}
    generations = {
        int(key): value for key, value in _load_json(source / "generations.json").items()
    }
    split_lookup = _split_lookup(_load_json(source_root / "image_splits.json"))
    image_ids = [
        image_id
        for image_id in sorted(split_lookup)
        if labels[image_id].get("object_token_spans")
    ]
    prefix = "semantic_attention.smoke" if args.smoke_images else "semantic_attention"
    if args.smoke_images:
        image_ids = image_ids[: int(args.smoke_images)]
    part_path = output / f"{prefix}.part{args.part_index}.pkl"
    target = output / f"{prefix}.pkl"
    summary_path = output / f"{prefix}_summary.json"
    progress_path = output / (
        f"{prefix}_progress.json"
        if args.num_parts == 1
        else f"{prefix}_progress.part{args.part_index}.json"
    )
    part_paths = sorted(output.glob(f"{prefix}.part*.pkl"))
    if args.merge_only:
        merge(args.model, part_paths, target, summary_path)
        return

    done_image_ids = set()
    for path in (target, *part_paths):
        if path.exists():
            done_image_ids.update(int(row["image_id"]) for row in load_pkl(path))
    assigned = image_ids[args.part_index :: args.num_parts]
    pending = [image_id for image_id in assigned if image_id not in done_image_ids]
    completed_before = len(assigned) - len(pending)
    print(
        f"[semantic-attention] {args.model} part {args.part_index}/{args.num_parts}: "
        f"{completed_before} assigned done, {len(pending)} pending",
        flush=True,
    )
    wrapper = None
    started = time.monotonic()
    try:
        for offset, image_id in enumerate(pending, 1):
            tick = time.monotonic()
            if wrapper is None:
                wrapper = build_model(
                    args.model,
                    get_extraction_model_cfg(config, args.model),
                    device=args.device,
                )
                wrapper.model.eval().requires_grad_(False)
            rows = extract_image(
                wrapper=wrapper,
                model=args.model,
                config=config,
                image_id=image_id,
                label_row=labels[image_id],
                generation_row=generations[image_id],
                split_name=split_lookup[image_id],
            )
            append_pkl(rows, part_path)
            completed = completed_before + offset
            _save_json(
                progress_path,
                {
                    "model": args.model,
                    "completed": completed,
                    "total": len(assigned),
                    "part_index": args.part_index,
                    "num_parts": args.num_parts,
                    "last_image": image_id,
                    "elapsed_seconds": time.monotonic() - started,
                },
            )
            print(
                f"{args.model} part{args.part_index} {completed}/{len(assigned)} "
                f"image={image_id} samples={len(rows)} seconds={time.monotonic() - tick:.1f}",
                flush=True,
            )
        if args.num_parts == 1 and not args.no_merge:
            merge(
                args.model,
                sorted(output.glob(f"{prefix}.part*.pkl")),
                target,
                summary_path,
            )
    finally:
        del wrapper
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke-images", type=int, default=0)
    parser.add_argument("--part-index", type=int, default=0)
    parser.add_argument("--num-parts", type=int, default=1)
    parser.add_argument("--no-merge", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()
    if args.num_parts < 1 or not 0 <= args.part_index < args.num_parts:
        parser.error("--part-index must be in [0, --num-parts)")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    run(args)


if __name__ == "__main__":
    main()
