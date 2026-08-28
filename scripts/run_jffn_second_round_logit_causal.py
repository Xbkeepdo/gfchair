#!/usr/bin/env python3
"""Target-logit/margin attribution and FFN-output causal interventions.

The target token is never inserted into the prefix.  Gradients and
interventions operate on the FFN output at the exact causal query row that
predicts the official InsLen first subtoken.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.jffn_experiment import mentions_for_image  # noqa: E402
from features.visual_ffn_jacobian import (  # noqa: E402
    exact_visual_token_jvps,
    reconstruct_visual_directions,
    resolve_decoder_layer_adapter,
)
from models import build_model  # noqa: E402
from models.dgst_capture import (  # noqa: E402
    pre_token_prediction_positions,
    resolve_decoder_layers,
    resolve_prompt_positions,
    run_forward_with_dgst_captures,
)
from models.llava_wrapper import _append_prefix_token_ids  # noqa: E402
from scripts.run_jffn_p_comparison import (  # noqa: E402
    _image_path,
    _load_inputs,
    binary_average_precision,
    build_spatial_context,
    patch_overlap_fraction,
)
from utils.config_utils import get_extraction_model_cfg, load_config  # noqa: E402


MODELS = ("llava_1_5_7b", "internvl_2_5_8b")
EXPERIMENT = "COCO4000-INSLEN-OFFICIAL-TARGET"
DEFAULT_LAYERS = (8, 16, 24, 32)
ETAS = (0.05, 0.10, 0.25)
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument(
        "--config", default="configs/model_configs_inslen_official_target.yaml"
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--result-dir")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-images", type=int, default=12)
    parser.add_argument("--max-targets-per-image", type=int, default=2)
    parser.add_argument("--layers", default=",".join(map(str, DEFAULT_LAYERS)))
    parser.add_argument("--max-intervention-target-layers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260819)
    return parser.parse_args()


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def target_groups(label_row: Mapping[str, Any], response_ids: Sequence[int]):
    groups: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for span in label_row.get("object_token_spans", ()):
        indices = span.get("token_indices") or ()
        if not indices or int(span.get("label", -1)) not in (0, 1):
            continue
        index = int(indices[0])
        target = int(span.get("target_token_id", response_ids[index]))
        if target != int(response_ids[index]):
            raise AssertionError(f"Target token mismatch at response index {index}")
        groups[index].append(span)
    return dict(groups)


def choose_targets(groups: Mapping[int, Sequence[Mapping[str, Any]]], limit: int):
    if limit <= 0:
        return sorted(groups)
    selected: list[int] = []
    for label in (1, 0):
        candidates = [
            index for index, spans in sorted(groups.items())
            if any(int(span["label"]) == label for span in spans)
        ]
        if candidates:
            selected.append(candidates[0] if label == 1 else candidates[-1])
    for index in sorted(groups):
        if len(selected) >= limit:
            break
        if index not in selected:
            selected.append(index)
    return sorted(selected[:limit])


def select_images(labels, generations, splits, count: int, seed: int):
    candidates = []
    for image_id in map(int, splits["test"]):
        if image_id not in labels or image_id not in generations:
            continue
        response = generations[image_id].get("response_token_ids") or ()
        groups = target_groups(labels[image_id], response)
        if groups:
            has_hall = any(
                int(span["label"]) == 0
                for spans in groups.values() for span in spans
            )
            categories = {
                str(span.get("canonical_object") or "")
                for spans in groups.values() for span in spans
            }
            candidates.append((not has_hall, -len(categories), image_id))
    rng = random.Random(int(seed))
    rng.shuffle(candidates)
    candidates.sort(key=lambda row: (row[0], row[1]))
    # Alternate hallucination-rich and remaining cases after a deterministic
    # shuffle so both labels and multiple categories are represented.
    return [row[2] for row in candidates[: min(count, len(candidates))]]


def prepare_inputs(wrapper, model_name: str, image: Image.Image, prefix: Sequence[int], prompt: str):
    if model_name == "llava_1_5_7b":
        text = wrapper._format_prompt(wrapper.resolve_prompt(prompt))
        prompt_inputs = wrapper.processor(text=text, images=image, return_tensors="pt")
        prompt_length = int(prompt_inputs["input_ids"].shape[1])
        inputs = _append_prefix_token_ids(
            prompt_inputs, prefix_token_ids=prefix, device=wrapper.device, dtype=torch.float16
        )
        tokenized = [int(value) for value in inputs["input_ids"][0].tolist()]
        image_token_id = int(wrapper._image_token_id())
        visual_start, visual_end = wrapper._find_visual_token_range(inputs, image_token_id)
        prompt_positions = resolve_prompt_positions(
            full_input_ids=tokenized,
            prompt_tokenized_length=prompt_length,
            image_token_id=image_token_id,
            visual_start=visual_start,
            visual_end=visual_end,
        )
        positions = pre_token_prediction_positions(
            full_input_ids=tokenized,
            prompt_tokenized_length=prompt_length,
            response_token_indices=[len(prefix)],
            image_token_id=image_token_id,
            visual_token_count=visual_end - visual_start,
            prompt_positions=prompt_positions,
        )
        return inputs, int(positions[0]), int(visual_start), int(visual_end), [24, 24]

    pixel_values = wrapper._preprocess_image(image)
    prompt_ids, _, _ = wrapper._build_input_ids_with_image(
        pixel_values, prefix_token_ids=[], user_prompt=wrapper.resolve_prompt(prompt)
    )
    input_ids, visual_start, visual_end = wrapper._build_input_ids_with_image(
        pixel_values, prefix_token_ids=prefix, user_prompt=wrapper.resolve_prompt(prompt)
    )
    inputs = {
        "input_ids": input_ids.to(wrapper.device),
        "attention_mask": torch.ones_like(input_ids).to(wrapper.device),
        "pixel_values": pixel_values.to(wrapper.device),
        "image_flags": torch.ones(
            pixel_values.shape[0], dtype=torch.long, device=wrapper.device
        ),
    }
    tokenized = [int(value) for value in input_ids[0].tolist()]
    image_token_id = int(wrapper._img_ctx_id)
    prompt_positions = resolve_prompt_positions(
        full_input_ids=tokenized,
        prompt_tokenized_length=int(prompt_ids.shape[1]),
        image_token_id=image_token_id,
        visual_start=visual_start,
        visual_end=visual_end,
    )
    positions = pre_token_prediction_positions(
        full_input_ids=tokenized,
        prompt_tokenized_length=int(prompt_ids.shape[1]),
        response_token_indices=[len(prefix)],
        image_token_id=image_token_id,
        visual_token_count=visual_end - visual_start,
        prompt_positions=prompt_positions,
    )
    side = int(round(math.sqrt(visual_end - visual_start)))
    return inputs, int(positions[0]), int(visual_start), int(visual_end), [side, side]


def capture_clean(wrapper, inputs, prediction_position: int):
    is_internvl = type(wrapper).__name__ == "InternVLWrapper"
    with torch.no_grad():
        out, captures = run_forward_with_dgst_captures(
            wrapper.model,
            output_hidden_states=False,
            retain_attention_updates=True,
            # InternLM2's remote-code attention does not dispatch through the
            # Transformers attention registry used by the compact query-row
            # implementation.  Its official eager path already returns the
            # weights, matching InternVLWrapper's formal extraction route.
            attention_query_positions=(
                None if is_internvl else [prediction_position]
            ),
            capture_device=None,
            attention_query_chunk_size=None if is_internvl else 512,
            record_model_attentions=is_internvl,
            **inputs,
        )
    logits = out.logits[0, prediction_position].float().detach()
    out.logits = None
    return captures, logits


def gradient_at_ffn_output(
    model, layer, inputs, prediction_position: int, target_id: int, competitor_id: int
):
    holder: dict[str, torch.Tensor] = {}
    index = torch.tensor([prediction_position], dtype=torch.long, device=next(model.parameters()).device)

    def hook(_module, _args, output):
        leaf = output.index_select(1, index).detach().clone().requires_grad_(True)
        holder["leaf"] = leaf
        return output.detach().index_copy(1, index, leaf)

    handle = resolve_decoder_layer_adapter(layer).ffn.register_forward_hook(hook)
    try:
        with torch.enable_grad():
            out = model(
                **inputs, output_attentions=False, output_hidden_states=False,
                return_dict=True, use_cache=False,
            )
            logits = out.logits[0, prediction_position]
            target = logits[int(target_id)]
            margin = target - logits[int(competitor_id)]
            grad_logit = torch.autograd.grad(target, holder["leaf"], retain_graph=True)[0][0, 0]
            grad_margin = torch.autograd.grad(margin, holder["leaf"])[0][0, 0]
            clean = {
                "target_logit": float(target.detach()),
                "competitor_logit": float(logits[int(competitor_id)].detach()),
                "margin": float(margin.detach()),
            }
        return grad_logit.detach(), grad_margin.detach(), clean
    finally:
        handle.remove()


def intervene_ffn_output(model, layer, inputs, prediction_position: int, target_id: int, competitor_id: int, delta: torch.Tensor, eta: float):
    index = torch.tensor([prediction_position], dtype=torch.long, device=delta.device)

    def hook(_module, _args, output):
        row = output.index_select(1, index) - float(eta) * delta.reshape(1, 1, -1).to(output)
        return output.index_copy(1, index, row)

    handle = resolve_decoder_layer_adapter(layer).ffn.register_forward_hook(hook)
    try:
        with torch.no_grad():
            out = model(
                **inputs, output_attentions=False, output_hidden_states=False,
                return_dict=True, use_cache=False,
            )
        logits = out.logits[0, prediction_position].float()
        return float(logits[target_id]), float(logits[target_id] - logits[competitor_id])
    finally:
        handle.remove()


def spatial_values(p: torch.Tensor, overlap: torch.Tensor):
    labels = overlap > 0
    return {
        "bbox_mass": float((p * overlap).sum()),
        "top1_pointing": float(bool(labels[int(p.argmax())])),
        "patch_aupr": binary_average_precision(labels, p),
    }


def safe_corr(x: Sequence[float], y: Sequence[float], rank: bool = False):
    left = np.asarray(x, dtype=np.float64); right = np.asarray(y, dtype=np.float64)
    if rank:
        left = np.argsort(np.argsort(left)); right = np.argsort(np.argsort(right))
    if left.size < 3 or left.std() <= 1e-20 or right.std() <= 1e-20:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = Path(args.output_dir or ROOT / "outputs" / args.model / EXPERIMENT).resolve()
    result_dir = Path(args.result_dir or output_dir / "results/jffn_second_round/logit_causal").resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    labels, generations, splits = _load_inputs(output_dir)
    image_ids = select_images(labels, generations, splits, args.num_images, args.seed)
    wrapper = build_model(
        args.model, get_extraction_model_cfg(config, args.model), device=args.device
    )
    wrapper.model.requires_grad_(False); wrapper.model.eval()
    layers = resolve_decoder_layers(wrapper.model)
    layer_numbers = [int(value) for value in args.layers.split(",") if value]
    if any(value < 1 or value > len(layers) for value in layer_numbers):
        raise ValueError(f"Invalid layers {layer_numbers}; model has {len(layers)}")
    prompt = str((config.get("run") or {}).get("prompt") or "Describe this image.")
    spatial_context = build_spatial_context(config)
    rng = random.Random(args.seed)

    attribution_rows: list[dict[str, Any]] = []
    spatial_rows: list[dict[str, Any]] = []
    intervention_rows: list[dict[str, Any]] = []
    saved_cases: list[dict[str, Any]] = []
    intervention_case_count = 0
    started = time.perf_counter()

    for image_offset, image_id in enumerate(image_ids, 1):
        response_ids = [int(value) for value in generations[image_id]["response_token_ids"]]
        groups = target_groups(labels[image_id], response_ids)
        target_indices = choose_targets(groups, args.max_targets_per_image)
        with Image.open(_image_path(config, image_id)) as source:
            image = source.convert("RGB")
        for response_index in target_indices:
            spans = groups[response_index]
            label = 1 if any(int(span["label"]) == 1 for span in spans) else 0
            category = str(spans[0].get("canonical_object") or "")
            target_id = int(response_ids[response_index])
            prefix = response_ids[:response_index]
            inputs, prediction_position, visual_start, visual_end, grid = prepare_inputs(
                wrapper, args.model, image, prefix, prompt
            )
            captures, clean_logits = capture_clean(wrapper, inputs, prediction_position)
            competitor_logits = clean_logits.clone(); competitor_logits[target_id] = -torch.inf
            competitor_id = int(competitor_logits.argmax())
            clean_target = float(clean_logits[target_id]); clean_margin = float(clean_logits[target_id] - clean_logits[competitor_id])
            if response_index != len(prefix):
                raise AssertionError("Causal row includes the target token")

            boxes = spatial_context["boxes"].get((image_id, category), ()) if label == 1 else ()
            overlap = None
            if boxes and grid[0] * grid[1] == visual_end - visual_start:
                overlap = patch_overlap_fraction(
                    model=args.model,
                    image_size=spatial_context["image_size"][image_id],
                    boxes=boxes,
                    grid=tuple(grid),
                )
            for layer_number in layer_numbers:
                layer = layers[layer_number - 1]
                capture = captures[layer_number - 1]
                reconstructed = reconstruct_visual_directions(
                    layer=layer, capture=capture,
                    prediction_positions=[prediction_position],
                    visual_start=visual_start, visual_end=visual_end,
                )
                z = capture["h_mid"][0, prediction_position].reshape(1, -1)
                delta, _ = exact_visual_token_jvps(
                    layer=layer, z=z, a_tokens=reconstructed["a_tokens"],
                    chunk_size=None, synchronize=True, measure_time=False,
                )
                delta = delta[:, 0].float()
                a_tokens = reconstructed["a_tokens"][:, 0].float()
                write_e = a_tokens.norm(dim=-1); energy = delta.norm(dim=-1)
                grad_logit, grad_margin, gradient_clean = gradient_at_ffn_output(
                    wrapper.model, layer, inputs, prediction_position, target_id, competitor_id
                )
                a_logit = delta @ grad_logit.float()
                a_margin = delta @ grad_margin.float()
                visual_delta = delta.sum(0)
                logit_sum_error = float((a_logit.sum() - grad_logit.float().dot(visual_delta)).abs() / grad_logit.float().dot(visual_delta).abs().clamp_min(EPS))
                margin_sum_error = float((a_margin.sum() - grad_margin.float().dot(visual_delta)).abs() / grad_margin.float().dot(visual_delta).abs().clamp_min(EPS))
                positive = a_margin.clamp_min(0); negative = (-a_margin).clamp_min(0)
                row = {
                    "model": args.model, "image_id": image_id,
                    "response_index": response_index, "label": "REAL" if label else "HALL",
                    "category": category, "layer": layer_number,
                    "target_token_id": target_id, "competitor_token_id": competitor_id,
                    "prefix_excludes_target": True,
                    "prediction_position": prediction_position,
                    "clean_target_logit": clean_target, "clean_margin": clean_margin,
                    "gradient_clean_target_logit_abs_error": abs(gradient_clean["target_logit"] - clean_target),
                    "gradient_clean_margin_abs_error": abs(gradient_clean["margin"] - clean_margin),
                    "a_logit_total": float(a_logit.sum()),
                    "a_margin_total": float(a_margin.sum()),
                    "a_margin_positive_mass": float(positive.sum()),
                    "a_margin_negative_mass": float(negative.sum()),
                    "a_margin_positive_minus_negative": float(positive.sum() - negative.sum()),
                    "a_margin_positive_fraction": float((a_margin > 0).float().mean()),
                    "a_margin_max_positive": float(a_margin.max()),
                    "a_margin_minimum": float(a_margin.min()),
                    "a_margin_positive_support_nonzero": bool(float(positive.sum()) > EPS),
                    "a_margin_cancellation_ratio": float(a_margin.sum().abs() / a_margin.abs().sum().clamp_min(EPS)),
                    "logit_additivity_relative_error": logit_sum_error,
                    "margin_additivity_relative_error": margin_sum_error,
                    "visual_token_count": int(delta.shape[0]),
                }
                attribution_rows.append(row)

                q_total = visual_delta
                q = delta @ (q_total / q_total.norm().clamp_min(EPS))
                methods = {
                    "WRITE": write_e / write_e.sum().clamp_min(EPS),
                    "JFFN": energy / energy.sum().clamp_min(EPS),
                    "Q_POSITIVE": q.clamp_min(0) / q.clamp_min(0).sum().clamp_min(EPS),
                    "LOGIT_MARGIN_POSITIVE": positive / positive.sum().clamp_min(EPS),
                }
                if overlap is not None:
                    for method, p in methods.items():
                        if method == "LOGIT_MARGIN_POSITIVE" and float(positive.sum()) <= EPS:
                            # There is no positive-support distribution in
                            # this case; recording an arbitrary all-zero map
                            # as a pointing failure would be misleading.
                            continue
                        for metric, value in spatial_values(p.cpu(), overlap).items():
                            spatial_rows.append({
                                "model": args.model, "image_id": image_id,
                                "response_index": response_index, "layer": layer_number,
                                "method": method, "metric": metric, "value": value,
                            })

                case = {
                    "image_id": image_id, "response_index": response_index,
                    "label": label, "category": category, "layer": layer_number,
                    "grid": grid, "write": methods["WRITE"].cpu(),
                    "jffn": methods["JFFN"].cpu(), "q_positive": methods["Q_POSITIVE"].cpu(),
                    "logit_margin_positive": methods["LOGIT_MARGIN_POSITIVE"].cpu(),
                    "a_logit": a_logit.cpu(), "a_margin": a_margin.cpu(),
                }
                saved_cases.append(case)

                if intervention_case_count < args.max_intervention_target_layers:
                    # Use the exact same no-grad FFN-hook execution path at
                    # eta=0 as the causal baseline.  Comparing against the
                    # custom-attention capture forward can otherwise expose a
                    # model-dtype kernel difference larger than the tiny
                    # intervention itself (0.015625 was observed in FP16).
                    intervention_clean_target, intervention_clean_margin = (
                        intervene_ffn_output(
                            wrapper.model,
                            layer,
                            inputs,
                            prediction_position,
                            target_id,
                            competitor_id,
                            delta.sum(0),
                            0.0,
                        )
                    )
                    selected = {
                        "top_positive_margin": int(a_margin.argmax()),
                        "highest_E": int(energy.argmax()),
                        "highest_I": int(write_e.argmax()),
                        "random": rng.randrange(int(delta.shape[0])),
                        "most_negative_margin": int(a_margin.argmin()),
                    }
                    selected_norms = torch.tensor([float(delta[index].norm()) for index in selected.values()])
                    reference_norm = float(selected_norms.median())
                    for strategy, token_index in selected.items():
                        raw_delta = delta[token_index]
                        matched_delta = raw_delta * (reference_norm / float(raw_delta.norm().clamp_min(EPS)))
                        predicted_unit = float(grad_margin.float().dot(matched_delta))
                        for eta in ETAS:
                            target_new, margin_new = intervene_ffn_output(
                                wrapper.model, layer, inputs, prediction_position,
                                target_id, competitor_id, matched_delta, eta,
                            )
                            intervention_rows.append({
                                "model": args.model, "image_id": image_id,
                                "response_index": response_index, "label": "REAL" if label else "HALL",
                                "category": category, "layer": layer_number,
                                "strategy": strategy, "token_index": token_index,
                                "eta": eta, "matched_delta_norm": reference_norm,
                                "intervention_baseline_target_logit": intervention_clean_target,
                                "intervention_baseline_margin": intervention_clean_margin,
                                "intervention_baseline_margin_vs_capture": intervention_clean_margin - clean_margin,
                                "predicted_delta_margin": -eta * predicted_unit,
                                "observed_delta_margin": margin_new - intervention_clean_margin,
                                "observed_delta_logit": target_new - intervention_clean_target,
                            })
                    aggregate = delta.sum(0)
                    predicted_unit = float(grad_margin.float().dot(aggregate))
                    for eta in ETAS:
                        target_new, margin_new = intervene_ffn_output(
                            wrapper.model, layer, inputs, prediction_position,
                            target_id, competitor_id, aggregate, eta,
                        )
                        intervention_rows.append({
                            "model": args.model, "image_id": image_id,
                            "response_index": response_index, "label": "REAL" if label else "HALL",
                            "category": category, "layer": layer_number,
                            "strategy": "aggregate_visual_response", "token_index": -1,
                            "eta": eta, "matched_delta_norm": float(aggregate.norm()),
                            "intervention_baseline_target_logit": intervention_clean_target,
                            "intervention_baseline_margin": intervention_clean_margin,
                            "intervention_baseline_margin_vs_capture": intervention_clean_margin - clean_margin,
                            "predicted_delta_margin": -eta * predicted_unit,
                            "observed_delta_margin": margin_new - intervention_clean_margin,
                            "observed_delta_logit": target_new - intervention_clean_target,
                        })
                    intervention_case_count += 1
                del delta, a_tokens, grad_logit, grad_margin
            for capture in captures:
                capture.clear()
            del captures, inputs
            torch.cuda.empty_cache()
        print(f"[logit causal] {image_offset}/{len(image_ids)} image={image_id}", flush=True)

    write_csv(result_dir / "logit_attribution_cases.csv", attribution_rows)
    write_csv(result_dir / "logit_positive_spatial.csv", spatial_rows)
    write_csv(result_dir / "causal_interventions.csv", intervention_rows)
    torch.save(saved_cases, result_dir / "logit_attribution_token_maps.pt")

    intervention_summary = []
    for eta in ETAS:
        rows = [row for row in intervention_rows if float(row["eta"]) == eta]
        predicted = [float(row["predicted_delta_margin"]) for row in rows]
        observed = [float(row["observed_delta_margin"]) for row in rows]
        errors = np.asarray(predicted) - np.asarray(observed)
        predicted_array = np.asarray(predicted, dtype=np.float64)
        observed_array = np.asarray(observed, dtype=np.float64)
        stable = np.abs(predicted_array) > 1e-5
        slope = float(np.polyfit(predicted, observed, 1)[0]) if len(rows) >= 2 else float("nan")
        intervention_summary.append({
            "eta": eta, "count": len(rows),
            "pearson": safe_corr(predicted, observed),
            "spearman": safe_corr(predicted, observed, rank=True),
            "sign_accuracy": float(np.mean(np.sign(predicted) == np.sign(observed))) if rows else float("nan"),
            "regression_slope": slope,
            "mean_absolute_prediction_error": float(np.mean(np.abs(errors))) if rows else float("nan"),
            "mean_relative_error_where_abs_prediction_gt_1e-5": (
                float(np.mean(np.abs(errors[stable]) / np.abs(predicted_array[stable])))
                if np.any(stable) else float("nan")
            ),
        })

    strategy_summary = []
    for strategy in sorted({str(row["strategy"]) for row in intervention_rows}):
        for eta in ETAS:
            rows = [
                row for row in intervention_rows
                if str(row["strategy"]) == strategy and float(row["eta"]) == eta
            ]
            if not rows:
                continue
            strategy_summary.append({
                "strategy": strategy,
                "eta": eta,
                "count": len(rows),
                "predicted_delta_margin_mean": float(np.mean([row["predicted_delta_margin"] for row in rows])),
                "observed_delta_margin_mean": float(np.mean([row["observed_delta_margin"] for row in rows])),
                "sign_accuracy": float(np.mean([
                    np.sign(row["predicted_delta_margin"]) == np.sign(row["observed_delta_margin"])
                    for row in rows
                ])),
            })

    labels_np = np.asarray([row["label"] == "HALL" for row in attribution_rows], dtype=np.int64)
    real_hall = {}
    for metric in ("a_logit_total", "a_margin_total", "a_margin_positive_mass", "a_margin_negative_mass", "a_margin_cancellation_ratio"):
        values = np.asarray([float(row[metric]) for row in attribution_rows])
        real_hall[metric] = {
            "real_mean": float(values[labels_np == 0].mean()) if np.any(labels_np == 0) else float("nan"),
            "hall_mean": float(values[labels_np == 1].mean()) if np.any(labels_np == 1) else float("nan"),
            "hall_auroc_direction_free": max(float(roc_auc_score(labels_np, values)), float(roc_auc_score(labels_np, -values))) if np.unique(labels_np).size == 2 else float("nan"),
        }
    summary = {
        "protocol": "jffn_second_round_logit_causal_v1",
        "model": args.model,
        "sample": {
            "images": len(image_ids), "image_ids": image_ids,
            "target_layer_cases": len(attribution_rows),
            "real_cases": int(np.sum(labels_np == 0)), "hall_cases": int(np.sum(labels_np == 1)),
            "intervention_target_layer_cases": intervention_case_count,
            "intervention_rows": len(intervention_rows),
            "layers_one_based": layer_numbers,
        },
        "causal_row_audit": {
            "prefix_excludes_target_all": all(bool(row["prefix_excludes_target"]) for row in attribution_rows),
            "max_gradient_clean_target_logit_abs_error": max((row["gradient_clean_target_logit_abs_error"] for row in attribution_rows), default=float("nan")),
            "max_gradient_clean_margin_abs_error": max((row["gradient_clean_margin_abs_error"] for row in attribution_rows), default=float("nan")),
        },
        "additivity": {
            "max_logit_relative_error": max((row["logit_additivity_relative_error"] for row in attribution_rows), default=float("nan")),
            "max_margin_relative_error": max((row["margin_additivity_relative_error"] for row in attribution_rows), default=float("nan")),
        },
        "real_hall": real_hall,
        "intervention": intervention_summary,
        "intervention_by_strategy": strategy_summary,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (result_dir / "logit_causal_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_results(result_dir, attribution_rows, intervention_rows)
    print(f"[logit causal] wrote {result_dir}", flush=True)


def plot_results(result_dir: Path, attribution_rows, intervention_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axis = plt.subplots(figsize=(7, 4.5))
    for label in ("REAL", "HALL"):
        rows = [row for row in attribution_rows if row["label"] == label]
        by_layer = defaultdict(list)
        for row in rows: by_layer[int(row["layer"])].append(float(row["a_margin_total"]))
        layers = sorted(by_layer)
        means = np.asarray([np.mean(by_layer[layer]) for layer in layers])
        ci = np.asarray([
            1.96 * np.std(by_layer[layer]) / math.sqrt(max(len(by_layer[layer]), 1))
            for layer in layers
        ])
        axis.plot(layers, means, marker="o", label=label)
        axis.fill_between(layers, means - ci, means + ci, alpha=.16)
    axis.axhline(0, color="black", lw=.8); axis.set_xlabel("Layer"); axis.set_ylabel(r"$A^{margin,total}$")
    axis.set_title("Target-margin-aligned visual FFN support"); axis.grid(alpha=.2); axis.legend()
    fig.tight_layout(); fig.savefig(result_dir / "figure5_margin_attribution_real_hall.png", dpi=220); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for axis, eta in zip(axes, ETAS):
        rows = [row for row in intervention_rows if float(row["eta"]) == eta]
        x = [row["predicted_delta_margin"] for row in rows]; y = [row["observed_delta_margin"] for row in rows]
        axis.scatter(x, y, s=10, alpha=.55)
        if x:
            low = min(min(x), min(y)); high = max(max(x), max(y)); axis.plot([low, high], [low, high], "k--", lw=1)
        axis.set_title(f"eta={eta}"); axis.set_xlabel("Predicted"); axis.set_ylabel("Observed")
    fig.suptitle("FFN-output intervention: margin change")
    fig.tight_layout(); fig.savefig(result_dir / "figure6_predicted_vs_observed_intervention.png", dpi=220); plt.close(fig)


if __name__ == "__main__":
    main()
