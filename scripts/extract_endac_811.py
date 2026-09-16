#!/usr/bin/env python3
"""Extract baselines, path features, and optional semantic Top-K in one pass."""

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

from features.baseline.metatoken import compute_metatoken_features_from_stats
from features.endac_811 import extract_layer_features
from features.semantic_attention_topk import FEATURE_NAMES, layer_matrices, topk_features
from models import build_model
from models.base_wrapper import compact_response_logit_statistics
from models.dgst_capture import resolve_decoder_final_norm, resolve_decoder_layers, resolve_output_embedding_layer, run_forward_with_dgst_captures
from models.llava_wrapper import _append_prefix_token_ids
from scripts.run_jffn_second_round_logit_causal import prepare_inputs
from scripts.run_ffn_visual_source_consistency import local_fp32
from utils.config_utils import get_dataset_cfg, get_extraction_model_cfg, load_config
from utils.io_utils import append_pkl, load_pkl, save_pkl
from features.visual_ffn_jacobian import reconstruct_visual_directions

MODELS = ("qwen2_5_vl_7b", "llava_1_5_7b", "qwen3_vl_8b", "internvl_2_5_8b", "minigpt4_7b", "shikra_7b")
DEFAULT_CONFIG = ROOT / "configs/model_configs_endac_811.yaml"


def _load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _output_root(config: dict) -> Path:
    path = Path(config["endac_811"]["output_root"])
    return path if path.is_absolute() else ROOT / path


def _split_lookup(split: dict) -> dict[int, str]:
    result = {}
    for name in ("train", "val", "test"):
        for image_id in split[name]:
            result[int(image_id)] = name
    if len(result) != 4000:
        raise ValueError("ENDAC-811 split must contain 4000 unique image IDs")
    return result


def _targets(label_row: dict, response_ids: list[int]) -> list[dict]:
    targets = []
    for mention_index, span in enumerate(label_row.get("object_token_spans") or ()):
        token_indices = [int(value) for value in span.get("token_indices") or ()]
        if not token_indices or int(span.get("label", -1)) not in (0, 1):
            continue
        if min(token_indices) < 0 or max(token_indices) >= len(response_ids):
            raise ValueError("Exact object span is outside the saved response")
        response_index = token_indices[0]
        targets.append(
            {
                "sample_id": f"{int(label_row['image_id'])}:{mention_index}",
                "response_index": response_index,
                "target_token_id": int(response_ids[response_index]),
                "token_indices": token_indices,
                "token": str(span.get("surface") or span.get("word") or ""),
                "canonical_object": str(span.get("canonical_object") or span.get("word") or ""),
                "occurrence_count": int(span.get("occurrence_count", 1)),
                "label": int(span["label"]),
            }
        )
    return targets


def _capture(
    wrapper,
    model: str,
    image: Image.Image,
    response_ids: list[int],
    targets: list[dict],
    prompt: str,
):
    if model in {"minigpt4_7b", "shikra_7b"}:
        prompt_inputs = wrapper.processor(
            text=wrapper._format_prompt(wrapper.resolve_prompt(prompt)), images=image
        )
        inputs = _append_prefix_token_ids(
            prompt_inputs,
            prefix_token_ids=response_ids,
            device=wrapper.device,
            dtype=torch.float16,
        )
        last = int(inputs["input_ids"].shape[1]) - 1
        visual_start, visual_end = wrapper._find_visual_token_range(
            inputs, wrapper._image_token_id()
        )
        visual_grid = list(wrapper.grid) if model == "shikra_7b" else []
    else:
        inputs, last, visual_start, visual_end, visual_grid = prepare_inputs(
            wrapper, model, image, response_ids, prompt
        )
    response_start = int(last) - len(response_ids)
    prediction_positions = [response_start + int(row["response_index"]) for row in targets]
    response_positions = [response_start + index for index in range(len(response_ids))]
    internvl = model == "internvl_2_5_8b"
    chunk_size = None
    if not internvl:
        chunk_size = (
            wrapper.dgst_attention_query_chunk_size
            if model == "llava_1_5_7b"
            else int(wrapper.cfg.get("dgst_attention_query_chunk_size", 512))
        )
    with torch.no_grad():
        output, captures = run_forward_with_dgst_captures(
            wrapper.model,
            output_hidden_states=False,
            retain_attention_updates=True,
            attention_query_positions=None if internvl else prediction_positions,
            capture_device=None,
            attention_query_chunk_size=chunk_size,
            record_model_attentions=internvl,
            **inputs,
        )
        response_logits = output.logits[0, response_positions].detach()
        output.logits = None
        statistics = compact_response_logit_statistics(
            response_logits, response_token_ids=response_ids
        )
        statistics.pop("response_token_ids", None)
    del output, response_logits, inputs
    return captures, prediction_positions, int(visual_start), int(visual_end), visual_grid, statistics


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
        captures, positions, visual_start, visual_end, visual_grid, statistics = _capture(
            wrapper, model, source.convert("RGB"), response_ids, targets, prompt
        )

    settings = config["endac_811"]
    extraction = settings["extraction"]
    semantic_enabled = bool(settings.get("extract_semantic_attention", False))
    semantic_settings = config["semantic_attention_topk"] if semantic_enabled else None
    layers = resolve_decoder_layers(wrapper.model)
    expected_layers = int(settings["models"][model]["num_layers"])
    if len(layers) != expected_layers or len(captures) != expected_layers:
        raise ValueError(f"{model}: expected {expected_layers} decoder layers")
    output_layer = resolve_output_embedding_layer(wrapper.model)
    final_norm = resolve_decoder_final_norm(wrapper.model) if semantic_enabled else None
    accumulated = [
        {
            "svar": [],
            "last_visual_attention": None,
            "visual_only": [],
            "all_attention_v": [],
            "all_attention_vp_generation": [],
            "visual_only_gross_raw": [],
            "semantic_matrices": {
                name: [] for name in (
                    "attention", "object_probability_raw",
                    "object_probability_norm", "cosine_similarity",
                )
            },
            "semantic_top_indices": {
                f"{variant}_top{k}": []
                for variant in ("raw", "norm")
                for k in (semantic_settings["top_k"] if semantic_enabled else ())
            },
            "semantic_features": {
                f"{variant}_top{k}": []
                for variant in ("raw", "norm")
                for k in (semantic_settings["top_k"] if semantic_enabled else ())
            },
        }
        for _ in targets
    ]
    for layer_index, (layer, capture) in enumerate(zip(layers, captures)):
        if semantic_enabled:
            matrices = layer_matrices(
                capture=capture,
                prediction_positions=positions,
                target_token_ids=[row["target_token_id"] for row in targets],
                visual_start=visual_start,
                visual_end=visual_end,
                output_layer=output_layer,
                final_norm=final_norm,
                vocab_chunk_size=int(semantic_settings["vocab_chunk_size"]),
            )
            for target_index, item in enumerate(accumulated):
                attention = matrices["attention"][target_index]
                cosine = matrices["cosine_similarity"][target_index]
                for name, values in item["semantic_matrices"].items():
                    values.append(matrices[name][target_index].float().cpu())
                for variant in ("raw", "norm"):
                    probability = matrices[f"object_probability_{variant}"][target_index]
                    for k in semantic_settings["top_k"]:
                        indices, feature = topk_features(probability, attention, cosine, int(k))
                        key = f"{variant}_top{k}"
                        item["semantic_top_indices"][key].append(indices.to(torch.int16).cpu())
                        item["semantic_features"][key].append(feature.float().cpu())
            del matrices
        directions = reconstruct_visual_directions(
            layer=layer,
            capture=capture,
            prediction_positions=positions,
            visual_start=visual_start,
            visual_end=visual_end,
            return_all_sources=True,
        )
        with local_fp32(layer), torch.no_grad():
            values = extract_layer_features(
                layer=layer,
                capture=capture,
                prediction_positions=positions,
                response_indices=[row["response_index"] for row in targets],
                target_token_ids=[row["target_token_id"] for row in targets],
                visual_start=visual_start,
                visual_end=visual_end,
                output_layer=output_layer,
                visual_points=int(extraction["visual_quadrature_points"]),
                all_attention_points=int(extraction["all_attention_quadrature_points"]),
                token_chunk_size=int(extraction["token_chunk_size"]),
                mad_epsilon=float(extraction["mad_epsilon"]),
                reconstructed_directions=directions,
            )
        for target_index, value in enumerate(values):
            item = accumulated[target_index]
            visual_attention = value["visual_attention"].float().cpu()
            item["svar"].append(visual_attention.sum(dim=-1))
            if layer_index == expected_layers - 1:
                item["last_visual_attention"] = visual_attention
            item["visual_only"].append(
                torch.stack(
                    (
                        value["ae_visual"].float().cpu(),
                        torch.log1p(value["visual_only_gross"].float()).cpu(),
                    )
                )
            )
            if semantic_enabled:
                item["visual_only_gross_raw"].append(
                    value["visual_only_gross"].float().cpu()
                )
            item["all_attention_v"].append(
                torch.stack(
                    (
                        value["ae_visual"].float().cpu(),
                        torch.log1p(value["all_visual_gross"].float()).cpu(),
                    )
                )
            )
            item["all_attention_vp_generation"].append(
                torch.stack(
                    (
                        value["ae_visual_prompt"].float().cpu(),
                        torch.log1p(
                            value["all_prompt_gross"].float()
                            + value["all_visual_gross"].float()
                        ).cpu(),
                        value["ae_generation"].float().cpu(),
                        torch.log1p(value["all_generation_gross"].float()).cpu(),
                    )
                )
            )
        captures[layer_index] = None
        del directions, values, capture

    layer_start, layer_end = map(int, settings["models"][model]["svar_layers"])
    rows = []
    for target, item in zip(targets, accumulated):
        svar = torch.stack(item["svar"])[layer_start:layer_end].reshape(-1)
        meta = compute_metatoken_features_from_stats(
            response_token_ids=response_ids,
            visual_attention=item["last_visual_attention"],
            span_start=min(target["token_indices"]),
            span_end=max(target["token_indices"]),
            occurrence_count=target["occurrence_count"],
            **statistics,
        ).vector
        features = {
            "svar": svar.numpy().astype(np.float32, copy=False),
            "metatoken": meta.astype(np.float32, copy=False),
            "visual_only": torch.stack(item["visual_only"]).reshape(-1).numpy(),
            "all_attention_v": torch.stack(item["all_attention_v"]).reshape(-1).numpy(),
            "all_attention_vp_generation": torch.stack(
                item["all_attention_vp_generation"]
            ).reshape(-1).numpy(),
        }
        if semantic_enabled:
            features["visual_only_gross_raw"] = torch.stack(
                item["visual_only_gross_raw"]
            ).numpy()
        if any(not np.isfinite(value).all() for value in features.values()):
            raise ValueError(f"{target['sample_id']}: non-finite feature value")
        row = {
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
                "features": features,
            }
        if semantic_enabled:
            semantic_matrices = {
                name: torch.stack(values).numpy()
                for name, values in item["semantic_matrices"].items()
            }
            semantic_features = {
                name: torch.stack(values).numpy()
                for name, values in item["semantic_features"].items()
            }
            if any(not np.isfinite(value).all() for value in (*semantic_matrices.values(), *semantic_features.values())):
                raise ValueError(f"{target['sample_id']}: non-finite semantic-attention feature")
            row["semantic_attention"] = {
                "feature_names": list(FEATURE_NAMES),
                "matrices": semantic_matrices,
                "top_indices": {
                    name: torch.stack(values).numpy()
                    for name, values in item["semantic_top_indices"].items()
                },
                "features": semantic_features,
            }
        rows.append(row)
    return rows


def merge(model: str, part_paths: list[Path], target: Path, summary_path: Path) -> list[dict]:
    rows = []
    seen = set()
    for path in (target, *part_paths):
        if not path.exists():
            continue
        for row in load_pkl(path):
            key = str(row["sample_id"])
            if key not in seen:
                rows.append(row)
                seen.add(key)
    rows.sort(key=lambda row: (int(row["image_id"]), str(row["sample_id"])))
    dimensions = {}
    names = ("svar", "metatoken", "visual_only", "all_attention_v", "all_attention_vp_generation")
    if rows and "visual_only_gross_raw" in rows[0]["features"]:
        names += ("visual_only_gross_raw",)
    for name in names:
        widths = {int(np.asarray(row["features"][name]).size) for row in rows}
        if len(widths) != 1:
            raise ValueError(f"{model}: inconsistent {name} dimensions {widths}")
        dimensions[name] = widths.pop()
    save_pkl(rows, target)
    summary = {
        "model": model,
        "images": len({int(row["image_id"]) for row in rows}),
        "samples": len(rows),
        "dimensions": dimensions,
        "split_samples": {
            name: sum(row["split"] == name for row in rows)
            for name in ("train", "val", "test")
        },
    }
    _save_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False))
    return rows


def run(args: argparse.Namespace) -> None:
    config = load_config(str(args.config))
    output_root = _output_root(config)
    model_output = output_root / args.model
    labels = {int(key): value for key, value in _load_json(model_output / "labeling.json").items()}
    generations = {
        int(key): value for key, value in _load_json(model_output / "generations.json").items()
    }
    split = _load_json(output_root / "image_splits.json")
    split_lookup = _split_lookup(split)
    image_ids = [
        image_id
        for image_id in sorted(split_lookup)
        if labels[image_id].get("object_token_spans")
    ]
    if args.smoke_images:
        image_ids = image_ids[: int(args.smoke_images)]
        part_prefix = "features.smoke"
        target = model_output / "features.smoke.pkl"
        summary_path = model_output / "feature_smoke_summary.json"
        progress_prefix = "feature_smoke_progress"
    else:
        part_prefix = "features"
        target = model_output / "features.pkl"
        summary_path = model_output / "feature_summary.json"
        progress_prefix = "feature_progress"
    part_path = model_output / f"{part_prefix}.part{args.part_index}.pkl"
    progress_path = model_output / (
        f"{progress_prefix}.json"
        if args.num_parts == 1
        else f"{progress_prefix}.part{args.part_index}.json"
    )
    part_paths = sorted(model_output.glob(f"{part_prefix}.part*.pkl"))
    done_image_ids = set()
    for path in (target, *part_paths):
        if path.exists():
            done_image_ids.update(int(row["image_id"]) for row in load_pkl(path))
    if args.merge_only:
        merge(args.model, part_paths, target, summary_path)
        return
    assigned = image_ids[args.part_index :: args.num_parts]
    pending = [image_id for image_id in assigned if image_id not in done_image_ids]
    completed_before = len(assigned) - len(pending)
    print(
        f"[ENDAC features] {args.model} part {args.part_index}/{args.num_parts}: "
        f"{completed_before} assigned done, {len(pending)} pending"
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
                f"{args.model} part{args.part_index} {completed}/{len(assigned)} image={image_id} "
                f"samples={len(rows)} seconds={time.monotonic() - tick:.1f}",
                flush=True,
            )
        if args.num_parts == 1 and not args.no_merge:
            merge(args.model, sorted(model_output.glob(f"{part_prefix}.part*.pkl")), target, summary_path)
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
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--part-index", type=int, default=0)
    parser.add_argument("--num-parts", type=int, default=1)
    parser.add_argument("--no-merge", action="store_true")
    args = parser.parse_args()
    if args.num_parts < 1 or not 0 <= args.part_index < args.num_parts:
        parser.error("--part-index must be in [0, --num-parts)")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    run(args)


if __name__ == "__main__":
    main()
