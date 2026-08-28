#!/usr/bin/env python3
"""Join scalar/JFFN/logit quantities for the same second-round target cases.

The second-round production shards intentionally retain normalized attention
maps but not the *unnormalized total visual-attention mass*.  This analyzer
therefore leaves that conceptual stage explicitly NOT RUN instead of
substituting a different quantity.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.jffn_experiment import load_shards  # noqa: E402


EXPERIMENT = "COCO4000-INSLEN-OFFICIAL-TARGET"
MODELS = ("llava_1_5_7b", "internvl_2_5_8b")
VARIABLES = (
    "I_visual_write_norm",
    "R_ffn_response_norm",
    "S_ffn_sensitivity",
    "D_direction_cosine",
    "R_signed_q_sum",
    "A_logit_total",
    "A_margin_total",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--outputs-root", default="outputs")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or float(left.std()) == 0.0 or float(right.std()) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def main() -> None:
    args = parse_args()
    result_dir = (
        Path(args.outputs_root)
        / args.model
        / EXPERIMENT
        / "results/jffn_second_round"
    )
    logit_rows = read_csv(result_dir / "logit_causal/logit_attribution_cases.csv")
    requested: dict[tuple[int, int], dict[int, dict[str, str]]] = {}
    for row in logit_rows:
        key = (int(row["image_id"]), int(row["response_index"]))
        requested.setdefault(key, {})[int(row["layer"])] = row

    joined: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int]] = set()
    for shard in load_shards(result_dir / "shards"):
        for position in shard["positions"]:
            key = (int(position["image_id"]), int(position["response_index"]))
            layer_rows = requested.get(key)
            if not layer_rows:
                continue
            diagnostics = position["diagnostics"]
            signed_q = position["signed_q"].float()
            for layer, logit in layer_rows.items():
                index = layer - 1
                row_key = (*key, layer)
                if row_key in seen:
                    raise RuntimeError(f"Duplicate conceptual-stage row: {row_key}")
                seen.add(row_key)
                joined.append(
                    {
                        "model": args.model,
                        "image_id": key[0],
                        "response_index": key[1],
                        "layer": layer,
                        "label": logit["label"],
                        "category": logit["category"],
                        "attention_visual_mass": "",
                        "attention_visual_mass_status": "NOT RUN: raw mass not persisted",
                        "I_visual_write_norm": float(diagnostics["input_norm"][index]),
                        "R_ffn_response_norm": float(diagnostics["response_norm"][index]),
                        "S_ffn_sensitivity": float(diagnostics["gain"][index]),
                        "D_direction_cosine": float(diagnostics["direction_cosine"][index]),
                        "R_signed_q_sum": float(signed_q[index].sum()),
                        "A_logit_total": float(logit["a_logit_total"]),
                        "A_margin_total": float(logit["a_margin_total"]),
                    }
                )
    expected = len(logit_rows)
    if len(joined) != expected or len(seen) != expected:
        missing = sorted(
            (image_id, response_index, layer)
            for (image_id, response_index), layers in requested.items()
            for layer in layers
            if (image_id, response_index, layer) not in seen
        )
        raise RuntimeError(
            f"Conceptual-stage join incomplete: {len(joined)}/{expected}; "
            f"first missing={missing[:5]}"
        )
    joined.sort(key=lambda row: (row["image_id"], row["response_index"], row["layer"]))
    write_csv(result_dir / "conceptual_stage_cases.csv", joined)

    correlation_rows: list[dict[str, Any]] = []
    for label in ("REAL", "HALL"):
        subset = [row for row in joined if row["label"] == label]
        for left in VARIABLES:
            for right in VARIABLES:
                x = np.asarray([float(row[left]) for row in subset], dtype=np.float64)
                y = np.asarray([float(row[right]) for row in subset], dtype=np.float64)
                correlation_rows.append(
                    {
                        "model": args.model,
                        "label": label,
                        "left": left,
                        "right": right,
                        "pearson": safe_corr(x, y),
                        "count": len(subset),
                    }
                )
    write_csv(result_dir / "conceptual_stage_correlations.csv", correlation_rows)
    conservation = np.asarray(
        [
            abs(float(row["R_signed_q_sum"]) - float(row["R_ffn_response_norm"]))
            / max(abs(float(row["R_ffn_response_norm"])), 1e-12)
            for row in joined
        ],
        dtype=np.float64,
    )
    payload = {
        "protocol": "jffn_second_round_conceptual_stages_v1",
        "model": args.model,
        "cases": len(joined),
        "real_cases": sum(row["label"] == "REAL" for row in joined),
        "hall_cases": sum(row["label"] == "HALL" for row in joined),
        "attention_visual_mass": {
            "status": "NOT RUN",
            "reason": "unnormalized total visual-attention mass was not persisted",
        },
        "variables": list(VARIABLES),
        "max_signed_q_vs_R_relative_error": float(conservation.max()),
        "all_values_finite": all(
            math.isfinite(float(row[name])) for row in joined for name in VARIABLES
        ),
    }
    (result_dir / "conceptual_stage_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[conceptual stages] {args.model}: {len(joined)} rows", flush=True)


if __name__ == "__main__":
    main()
