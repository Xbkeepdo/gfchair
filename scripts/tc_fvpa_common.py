"""Shared CLI and lifecycle helpers for TC-FVPA launchers."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_ROOT = ROOT / "outputs" / "tc_fvpa_comprehensive_v1"
MODELS = (
    "llava_1_5_7b",
    "internvl_2_5_8b",
    "qwen2_5_vl_7b",
    "qwen3_vl_8b",
)
TARGET_SCALARS = ("log_probability", "margin", "logit")
FORMAL_LAYERS_BY_MODEL = {
    "llava_1_5_7b": [8, 16, 24, 32],
    "internvl_2_5_8b": [8, 16, 24, 32],
    # Frozen depth-quartile equivalents for architectures whose decoder depth
    # differs from 32.  Indices are one-based throughout TC-FVPA.
    "qwen2_5_vl_7b": [7, 14, 21, 28],
    "qwen3_vl_8b": [9, 18, 27, 36],
}


def comma_strings(value: str | Sequence[str]) -> list[str]:
    raw = [value] if isinstance(value, str) else list(value)
    return [item.strip() for block in raw for item in str(block).split(",") if item.strip()]


def comma_ints(value: str | Sequence[int]) -> list[int]:
    return [int(item) for item in comma_strings(value)]


def add_common_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_integration: bool = True,
) -> argparse.ArgumentParser:
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--output-dir")
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--layers", default="8,16,24,32")
    parser.add_argument(
        "--target-scalars", default=",".join(TARGET_SCALARS)
    )
    if include_integration:
        parser.add_argument("--integration-points", default="1,4,8,16,32")
    parser.add_argument("--dry-run", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--formal", action="store_true")
    parser.add_argument(
        "--config", default="configs/model_configs_inslen_official_target.yaml"
    )
    return parser


def validate_common_args(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.num_shards) <= 0 or not 0 <= int(args.shard_id) < int(args.num_shards):
        raise ValueError(
            f"Invalid shard-id/num-shards {args.shard_id}/{args.num_shards}"
        )
    layers = comma_ints(args.layers)
    if not layers or min(layers) <= 0 or len(set(layers)) != len(layers):
        raise ValueError(f"Layers must be unique positive one-based indices: {layers}")
    scalars = comma_strings(args.target_scalars)
    unknown = sorted(set(scalars) - set(TARGET_SCALARS))
    if unknown or not scalars:
        raise ValueError(f"Unknown or empty target scalars: {unknown}")
    points = comma_ints(getattr(args, "integration_points", "1"))
    if not points or min(points) <= 0 or len(set(points)) != len(points):
        raise ValueError(f"Integration points must be unique positive values: {points}")
    if args.formal:
        expected_layers = FORMAL_LAYERS_BY_MODEL[args.model]
        # The shared parser default denotes the preregistered depth quartiles;
        # translate it for non-32-layer Qwen decoders before any outcomes exist.
        if layers == [8, 16, 24, 32] and expected_layers != layers:
            layers = list(expected_layers)
        if layers != expected_layers:
            expected = ",".join(map(str, expected_layers))
            raise ValueError(
                f"Formal layers for {args.model} are frozen to {expected}"
            )
        required_points = {1, 4, 8, 16, 32}
        if set(points) != required_points:
            raise ValueError(
                "Formal integration points are frozen to 1,4,8,16,32"
            )
        if set(scalars) != set(TARGET_SCALARS):
            raise ValueError("Formal runs require log_probability,margin,logit")
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (
            ROOT
            / "outputs"
            / args.model
            / "COCO4000-INSLEN-OFFICIAL-TARGET"
            / "results"
            / "tc_fvpa_comprehensive_v1"
        ).resolve()
    )
    return {
        "model": args.model,
        "device": args.device,
        "devices": comma_strings(args.devices),
        "resume": bool(args.resume),
        "shard_id": int(args.shard_id),
        "num_shards": int(args.num_shards),
        "output_dir": str(output_dir),
        "seed": int(args.seed),
        "layers": layers,
        "target_scalars": scalars,
        "integration_points": points,
        "mode": "formal" if args.formal else ("smoke" if args.smoke else "development"),
        "config": str(Path(args.config).resolve()),
    }


def exact_command() -> list[str]:
    return [sys.executable, *sys.argv]


def shell_command(argv: Sequence[str] | None = None) -> str:
    return " ".join(shlex.quote(item) for item in (argv or exact_command()))


def input_paths_for_model(model: str, config: str | Path) -> list[Path]:
    experiment = (
        ROOT
        / "outputs"
        / model
        / "COCO4000-INSLEN-OFFICIAL-TARGET"
    )
    return [
        Path(config).resolve(),
        experiment / "generations.json",
        experiment / "labeling.json",
        experiment / "image_splits.json",
    ]


def print_dry_run(stage: str, config: Mapping[str, Any], extra: Mapping[str, Any] | None = None) -> None:
    print(
        json.dumps(
            {"stage": stage, "config": dict(config), "extra": dict(extra or {})},
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    )
