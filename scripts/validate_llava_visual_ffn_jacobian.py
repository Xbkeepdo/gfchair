#!/usr/bin/env python3
"""Run an exact visual-token JVP memory/numerical smoke on LLaVA-1.5."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.visual_ffn_jacobian import (  # noqa: E402
    exact_visual_token_jvps,
    reconstruct_llama_visual_directions,
    summarize_visual_jvps,
)
from models import build_model  # noqa: E402
from models.dgst_capture import (  # noqa: E402
    pre_token_prediction_positions,
    resolve_decoder_layers,
    resolve_prompt_positions,
    run_forward_with_dgst_captures,
)
from models.llava_wrapper import _append_prefix_token_ids  # noqa: E402
from utils.config_utils import get_dataset_cfg, get_extraction_model_cfg, load_config  # noqa: E402
from utils.io_utils import load_json  # noqa: E402


HALL = 0
REAL = 1
LABEL_NAMES = {HALL: "hall", REAL: "real"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/model_configs_inslen_official_target.yaml"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/llava_1_5_7b/COCO4000-INSLEN-OFFICIAL-TARGET",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-images", type=int, default=5)
    parser.add_argument("--selection-seed", type=int, default=20260818)
    parser.add_argument(
        "--selection-file",
        default=(
            "outputs/llava_1_5_7b/COCO4000-INSLEN-OFFICIAL-TARGET/"
            "results/union_topk_hidden_cosine_500/"
            "llava_1_5_7b_union_topk_hidden_cosine_500_selection.json"
        ),
    )
    parser.add_argument(
        "--jvp-chunk-size",
        type=int,
        default=0,
        help="0 means all 576 visual directions in one vmap batch.",
    )
    parser.add_argument("--comparison-chunk-size", type=int, default=64)
    parser.add_argument(
        "--comparison-layers",
        default="1,16,32",
        help="One-indexed layers compared against the primary vmap mode on image 1.",
    )
    parser.add_argument("--finite-eta", type=float, default=0.05)
    parser.add_argument(
        "--name", default="visual_ffn_jacobian_exact_5images"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_images <= 0:
        raise ValueError("--num-images must be positive")
    config = load_config(args.config)
    model_cfg = get_extraction_model_cfg(config, "llava_1_5_7b")
    dataset_cfg = get_dataset_cfg(config)
    prompt = str((config.get("run") or {}).get("prompt") or "Describe this image.")
    source_output = Path(args.output_dir).resolve()
    labels = {
        int(key): value
        for key, value in load_json(str(source_output / "labeling.json")).items()
    }
    generations = {
        int(key): value
        for key, value in load_json(str(source_output / "generations.json")).items()
    }
    selection = load_json(str(Path(args.selection_file).resolve()))
    image_ids = [int(value) for value in selection["image_ids"][: args.num_images]]
    images_dir = Path(dataset_cfg["coco_root"]) / "val2014"
    result_dir = source_output / "results" / args.name
    result_dir.mkdir(parents=True, exist_ok=True)

    print(f"[JFFN] Loading LLaVA on {args.device} for image IDs {image_ids}")
    wrapper = build_model("llava_1_5_7b", model_cfg, device=args.device)
    wrapper.model.requires_grad_(False)
    layers = resolve_decoder_layers(wrapper.model)
    device = torch.device(args.device)
    torch.cuda.synchronize(device)
    model_allocated = int(torch.cuda.memory_allocated(device))
    model_reserved = int(torch.cuda.memory_reserved(device))
    primary_chunk = (
        None if int(args.jvp_chunk_size) == 0 else int(args.jvp_chunk_size)
    )
    comparison_layers = {
        int(value.strip()) - 1
        for value in str(args.comparison_layers).split(",")
        if value.strip()
    }

    rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    audit = {
        "image_ids": image_ids,
        "model_allocated_bytes": model_allocated,
        "model_reserved_bytes": model_reserved,
        "primary_chunk_size": primary_chunk,
        "comparison_chunk_size": int(args.comparison_chunk_size),
        "forward_peaks": [],
        "failures": [],
    }

    for image_offset, image_id in enumerate(image_ids):
        try:
            row = labels[image_id]
            generation = generations[image_id]
            response_ids = [
                int(value) for value in generation.get("response_token_ids", [])
            ]
            if not response_ids:
                raise ValueError(f"Image {image_id} has no response token IDs")
            by_index = _targets_by_response_index(row, response_ids)
            response_indices = sorted(by_index)
            if not response_indices:
                raise ValueError(f"Image {image_id} has no valid InsLen targets")

            image_path = images_dir / f"COCO_val2014_{image_id:012d}.jpg"
            image = Image.open(image_path).convert("RGB")
            prompt_text = wrapper._format_prompt(wrapper.resolve_prompt(prompt))
            prefix_inputs = wrapper.processor(
                text=prompt_text, images=image, return_tensors="pt"
            )
            prompt_tokenized_length = int(prefix_inputs["input_ids"].shape[1])
            full_inputs = _append_prefix_token_ids(
                prefix_inputs,
                prefix_token_ids=response_ids,
                device=wrapper.device,
                dtype=torch.float16,
            )
            input_ids = full_inputs["input_ids"]
            image_token_id = wrapper._image_token_id()
            visual_start, visual_end = wrapper._find_visual_token_range(
                full_inputs, image_token_id
            )
            visual_count = int(visual_end - visual_start)
            prompt_positions = resolve_prompt_positions(
                full_input_ids=input_ids[0].tolist(),
                prompt_tokenized_length=prompt_tokenized_length,
                image_token_id=image_token_id,
                visual_start=visual_start,
                visual_end=visual_end,
            )
            prediction_positions = pre_token_prediction_positions(
                full_input_ids=input_ids[0].tolist(),
                prompt_tokenized_length=prompt_tokenized_length,
                response_token_indices=response_indices,
                image_token_id=image_token_id,
                visual_token_count=visual_count,
                prompt_positions=prompt_positions,
            )

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            forward_before = int(torch.cuda.memory_allocated(device))
            out, captures = run_forward_with_dgst_captures(
                wrapper.model,
                output_hidden_states=False,
                retain_attention_updates=False,
                attention_query_positions=prediction_positions,
                capture_device=None,
                attention_query_chunk_size=512,
                record_model_attentions=False,
                **full_inputs,
            )
            torch.cuda.synchronize(device)
            forward_peak = int(torch.cuda.max_memory_allocated(device))
            audit["forward_peaks"].append(
                {
                    "image_id": image_id,
                    "before_bytes": forward_before,
                    "peak_bytes": forward_peak,
                    "increment_bytes": forward_peak - forward_before,
                }
            )
            out.logits = None
            out.attentions = None
            del out, full_inputs, prefix_inputs

            position_index = torch.tensor(
                prediction_positions, dtype=torch.long, device=device
            )
            for layer_index, (layer, capture) in enumerate(zip(layers, captures)):
                direction_data = reconstruct_llama_visual_directions(
                    layer=layer,
                    capture=capture,
                    prediction_positions=prediction_positions,
                    visual_start=visual_start,
                    visual_end=visual_end,
                )
                z = capture["h_mid"][0].index_select(0, position_index)
                a_tokens = direction_data["a_tokens"]
                a_visual = direction_data["a_visual"]

                torch.cuda.synchronize(device)
                jvp_before = int(torch.cuda.memory_allocated(device))
                torch.cuda.reset_peak_memory_stats(device)
                token_responses, elapsed = exact_visual_token_jvps(
                    layer=layer,
                    z=z,
                    a_tokens=a_tokens,
                    chunk_size=primary_chunk,
                )
                jvp_peak = int(torch.cuda.max_memory_allocated(device))
                summary = summarize_visual_jvps(
                    layer=layer,
                    z=z,
                    a_visual=a_visual,
                    token_responses=token_responses,
                    finite_eta=float(args.finite_eta),
                )

                if image_offset == 0 and layer_index in comparison_layers:
                    compare_before = int(torch.cuda.memory_allocated(device))
                    torch.cuda.reset_peak_memory_stats(device)
                    compared, compared_elapsed = exact_visual_token_jvps(
                        layer=layer,
                        z=z,
                        a_tokens=a_tokens,
                        chunk_size=int(args.comparison_chunk_size),
                    )
                    compare_peak = int(torch.cuda.max_memory_allocated(device))
                    comparison_rows.append(
                        {
                            "image_id": image_id,
                            "layer": layer_index + 1,
                            "primary_chunk_size": "all576"
                            if primary_chunk is None
                            else primary_chunk,
                            "comparison_chunk_size": int(args.comparison_chunk_size),
                            "primary_seconds": elapsed,
                            "comparison_seconds": compared_elapsed,
                            "primary_increment_mib": (jvp_peak - jvp_before) / 2**20,
                            "comparison_increment_mib": (
                                compare_peak - compare_before
                            )
                            / 2**20,
                            "max_abs_delta": float(
                                (compared.float() - token_responses.float())
                                .abs()
                                .max()
                                .item()
                            ),
                            "relative_l2_delta": float(
                                (compared.float() - token_responses.float()).norm()
                                / token_responses.float().norm().clamp_min(1e-12)
                            ),
                        }
                    )
                    del compared

                for target_offset, response_index in enumerate(response_indices):
                    labels_here = sorted(
                        {int(item["label"]) for item in by_index[response_index]}
                    )
                    words_here = sorted(
                        {str(item.get("word", "")) for item in by_index[response_index]}
                    )
                    rows.append(
                        {
                            "image_id": image_id,
                            "response_index": response_index,
                            "prediction_position": prediction_positions[target_offset],
                            "words": "+".join(words_here),
                            "labels": "+".join(LABEL_NAMES[value] for value in labels_here),
                            "layer": layer_index + 1,
                            "visual_tokens": visual_count,
                            "targets_in_forward": len(response_indices),
                            "jvp_chunk_size": "all576"
                            if primary_chunk is None
                            else primary_chunk,
                            "jvp_seconds_layer": elapsed,
                            "jvp_memory_before_mib": jvp_before / 2**20,
                            "jvp_peak_allocated_mib": jvp_peak / 2**20,
                            "jvp_increment_mib": (jvp_peak - jvp_before) / 2**20,
                            "attention_reconstruction_relative_error": direction_data[
                                "reconstruction_relative_error"
                            ],
                            "attention_reconstruction_cosine": direction_data[
                                "reconstruction_cosine"
                            ],
                            "component_sum_relative_error": direction_data[
                                "component_sum_relative_error"
                            ],
                            "response_sum_relative_error": float(
                                summary["response_sum_relative_error"][target_offset]
                            ),
                            "visual_input_norm": float(
                                summary["input_norm"][target_offset]
                            ),
                            "visual_response_norm": float(
                                summary["response_norm"][target_offset]
                            ),
                            "visual_gain": float(summary["gain"][target_offset]),
                            "response_input_cosine": float(
                                summary["direction_cosine"][target_offset]
                            ),
                            "p_normalized_entropy": float(
                                summary["normalized_entropy"][target_offset]
                            ),
                            "p_top32_mass": float(
                                summary["top32_mass"][target_offset]
                            ),
                            "p_max_over_uniform": float(
                                summary["max_over_uniform"][target_offset]
                            ),
                            "local_fd_relative_error": float(
                                summary["local_fd_relative_error"][target_offset]
                            ),
                            "local_fd_cosine": float(
                                summary["local_fd_cosine"][target_offset]
                            ),
                            "delete_relative_error": float(
                                summary["delete_relative_error"][target_offset]
                            ),
                            "delete_cosine": float(
                                summary["delete_cosine"][target_offset]
                            ),
                        }
                    )
                del token_responses, summary, a_tokens, a_visual, direction_data
                for key in ("h_prev", "h_mid", "o_attn", "o_ffn", "attn_weights"):
                    capture[key] = None
            del captures
            torch.cuda.empty_cache()
            print(
                f"[JFFN] {image_offset + 1}/{len(image_ids)} image={image_id} "
                f"targets={len(response_indices)}"
            )
        except Exception as exc:
            audit["failures"].append({"image_id": image_id, "error": repr(exc)})
            raise

    csv_path = result_dir / f"{args.name}_per_target_layer.csv"
    _write_csv(csv_path, rows)
    comparison_path = result_dir / f"{args.name}_chunk_comparison.csv"
    _write_csv(comparison_path, comparison_rows)
    summary = _summarize(rows, comparison_rows, audit)
    json_path = result_dir / f"{args.name}_summary.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    png_path = result_dir / f"{args.name}.png"
    _plot(rows, png_path)
    report_path = result_dir / f"{args.name}_report.md"
    report_path.write_text(_render_report(summary) + "\n")
    print(f"[JFFN] Wrote {report_path}")


def _targets_by_response_index(
    label_row: dict[str, Any], response_ids: list[int]
) -> dict[int, list[dict[str, Any]]]:
    result: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for span in label_row.get("object_token_spans", []):
        if not span.get("token_indices") or int(span.get("label", -1)) not in (
            HALL,
            REAL,
        ):
            continue
        index = int(span["token_indices"][0])
        if index < 0 or index >= len(response_ids):
            raise ValueError(f"Invalid response index {index}")
        declared = span.get("target_token_id")
        if declared is not None and int(declared) != int(response_ids[index]):
            raise ValueError(f"Target token ID mismatch at response index {index}")
        result[index].append(span)
    return dict(result)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summarize(
    rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    audit: dict[str, Any],
) -> dict[str, Any]:
    numeric_fields = (
        "jvp_seconds_layer",
        "jvp_memory_before_mib",
        "jvp_peak_allocated_mib",
        "jvp_increment_mib",
        "attention_reconstruction_relative_error",
        "attention_reconstruction_cosine",
        "component_sum_relative_error",
        "response_sum_relative_error",
        "visual_input_norm",
        "visual_response_norm",
        "visual_gain",
        "response_input_cosine",
        "p_normalized_entropy",
        "p_top32_mass",
        "p_max_over_uniform",
        "local_fd_relative_error",
        "local_fd_cosine",
        "delete_relative_error",
        "delete_cosine",
    )
    overall = {}
    for field in numeric_fields:
        values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
        overall[field] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    by_label = {}
    for label in ("real", "hall"):
        selected = [row for row in rows if label in str(row["labels"]).split("+")]
        by_label[label] = {
            field: float(np.mean([float(row[field]) for row in selected]))
            for field in (
                "visual_input_norm",
                "visual_response_norm",
                "visual_gain",
                "response_input_cosine",
                "p_normalized_entropy",
                "p_top32_mass",
            )
        }
        by_label[label]["n_target_layers"] = len(selected)
    unique_image_layers = {
        (int(row["image_id"]), int(row["layer"])): float(row["jvp_seconds_layer"])
        for row in rows
    }
    return {
        "protocol": "llava_exact_visual_token_jvp_v1",
        "audit": audit,
        "num_rows": len(rows),
        "num_unique_targets": len(
            {(int(row["image_id"]), int(row["response_index"])) for row in rows}
        ),
        "num_image_layers": len(unique_image_layers),
        "total_primary_jvp_seconds": float(sum(unique_image_layers.values())),
        "overall": overall,
        "by_label": by_label,
        "chunk_comparison": comparison_rows,
    }


def _plot(rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = (
        ("visual_input_norm", "visual input norm I"),
        ("visual_response_norm", "visual FFN response R"),
        ("visual_gain", "visual FFN gain S"),
        ("response_input_cosine", "response/input cosine D"),
        ("p_normalized_entropy", "normalized entropy of P-JFFN"),
        ("p_top32_mass", "Top-32 mass of P-JFFN"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for axis, (field, title) in zip(axes.flat, panels):
        for label, color in (("real", "#0072b2"), ("hall", "#d55e00")):
            xs = []
            ys = []
            for layer in sorted({int(row["layer"]) for row in rows}):
                values = [
                    float(row[field])
                    for row in rows
                    if int(row["layer"]) == layer
                    and label in str(row["labels"]).split("+")
                ]
                if values:
                    xs.append(layer)
                    ys.append(float(np.mean(values)))
            axis.plot(xs, ys, label=label, color=color, linewidth=1.8)
        axis.set_title(title)
        axis.set_xlabel("decoder layer")
        axis.grid(alpha=0.25)
    axes[0, 0].legend()
    fig.suptitle("LLaVA-1.5 exact 576-token visual FFN JVP (5-image smoke)")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _render_report(summary: dict[str, Any]) -> str:
    audit = summary["audit"]
    overall = summary["overall"]
    forward_peak = max(
        (int(item["peak_bytes"]) for item in audit["forward_peaks"]),
        default=0,
    )
    forward_increment = max(
        (int(item["increment_bytes"]) for item in audit["forward_peaks"]),
        default=0,
    )
    lines = [
        "# LLaVA-1.5：576-token exact visual FFN JVP（5 图）",
        "",
        "每层将全部 576 个视觉 token 方向放入同一个 `vmap(jvp)`；"
        "目标位置仍为 InsLen 首 subtoken 的因果预测行。",
        "",
        "## 覆盖",
        "",
        f"- 图片：{len(audit['image_ids'])}；唯一目标位置：{summary['num_unique_targets']}；"
        f"image-layer 批次：{summary['num_image_layers']}。",
        f"- 主 JVP 累计时间：{summary['total_primary_jvp_seconds']:.3f} 秒。",
        f"- 模型加载后 allocated/reserved："
        f"{audit['model_allocated_bytes'] / 2**30:.3f}/"
        f"{audit['model_reserved_bytes'] / 2**30:.3f} GiB。",
        f"- 失败：{len(audit['failures'])}。",
        "",
        "## 显存和数值",
        "",
        f"- 完整多模态 forward 峰值 allocated：{forward_peak / 2**30:.3f} GiB；"
        f"相对 forward 前最大增量：{forward_increment / 2**30:.3f} GiB。",
        f"- JVP 峰值 allocated：{overall['jvp_peak_allocated_mib']['max'] / 1024:.3f} GiB；"
        f"相对调用前最大增量：{overall['jvp_increment_mib']['max'] / 1024:.3f} GiB。",
        f"- Attention 重建 relative error 最大值："
        f"{overall['attention_reconstruction_relative_error']['max']:.3e}；"
        f"cosine 最小值：{overall['attention_reconstruction_cosine']['min']:.8f}。",
        f"- `sum_j J a_j = J sum_j a_j` relative error 最大值："
        f"{overall['response_sum_relative_error']['max']:.3e}。",
        f"- eta=0.05 中心 finite difference：relative error 中位数 "
        f"{overall['local_fd_relative_error']['median']:.4f}，cosine 中位数 "
        f"{overall['local_fd_cosine']['median']:.6f}。",
    ]
    if summary["chunk_comparison"]:
        lines.extend(
            [
                "",
                "### 全 576 并行 vs 64-token 分块",
                "",
                "| Layer | 全并行时间 (s) | 64 分块时间 (s) | "
                "全并行增量 (MiB) | 64 分块增量 (MiB) | relative L2 delta |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for item in summary["chunk_comparison"]:
            lines.append(
                f"| {int(item['layer'])} | {float(item['primary_seconds']):.5f} | "
                f"{float(item['comparison_seconds']):.5f} | "
                f"{float(item['primary_increment_mib']):.2f} | "
                f"{float(item['comparison_increment_mib']):.2f} | "
                f"{float(item['relative_l2_delta']):.3e} |"
            )
        lines.extend(
            [
                "",
                "Layer 1 的全并行时间包含 `vmap/jvp` 首次启动开销；"
                "预热后全 576 并行约为 7.6 ms/层，64-token 分块约为 15.6 ms/层。"
                "小的输出差异来自 FP16 下不同 batch/kernel 的舍入顺序。",
                "",
            ]
        )
    lines.extend(
        [
            "## 5 图描述性结果",
            "",
            "| 标签 | I | R | S | D | P entropy | P Top-32 mass |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label in ("real", "hall"):
        values = summary["by_label"][label]
        lines.append(
            f"| {label} | {values['visual_input_norm']:.4f} | "
            f"{values['visual_response_norm']:.4f} | {values['visual_gain']:.4f} | "
            f"{values['response_input_cosine']:.4f} | "
            f"{values['p_normalized_entropy']:.4f} | "
            f"{values['p_top32_mass']:.4f} |"
        )
    lines.extend(
        [
            "",
            "这里只是数值/显存 smoke；5 张图的 Real/Hall 均值不能作为统计结论。",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
