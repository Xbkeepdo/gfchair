#!/usr/bin/env python3
"""Label saved COCO generations with ENDAC exact spans and make 8:1:1 splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.config_utils import load_config

DEFAULT_CONFIG = ROOT / "configs/model_configs_endac_811.yaml"


def _settings(config_path: Path):
    config = load_config(str(config_path))
    settings = dict(config["endac_811"])
    output_root = Path(settings["output_root"])
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    return config, settings, output_root


def _json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def label_model(model: str, config_path: Path) -> Path:
    _config, settings, output_root = _settings(config_path)
    model_settings = settings["models"][model]
    output = output_root / model
    label_path = output / "labeling.json"
    if label_path.exists():
        labels = _json(label_path)
        if len(labels) == 4000 and all(
            row.get("labeling_protocol", {}).get("primary_locator")
            == "exact_response_offsets"
            for row in labels.values()
        ):
            print(f"[ENDAC] {model}: reusing 4000 exact labels")
            return output

    command = [
        sys.executable,
        str(ROOT / "scripts/chair_label_coco.py"),
        "--model",
        model,
        "--config",
        str(config_path),
        "--output-dir",
        str(output),
    ]
    if settings.get("chair_cache"):
        command.extend(("--chair-cache", str(settings["chair_cache"])))
    generations_path = output / "generations.json"
    if not generations_path.exists():
        output.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            Path(model_settings["generation_source"]) / "generations.json",
            generations_path,
        )
        print(f"[ENDAC] {model}: copied existing generations.json")
    command.append("--resume")
    subprocess.run(command, cwd=ROOT, check=True)
    return output


def build_split(config_path: Path) -> dict[str, list[int]]:
    _config, settings, output_root = _settings(config_path)
    qwen_source = Path(settings["models"]["qwen3_vl_8b"]["generation_source"])
    outer = _json(qwen_source / "image_splits.json")
    train = sorted(map(int, outer["train"]))
    held_out = sorted(map(int, outer["test"]))
    rng = np.random.default_rng(int(settings["split_seed"]))
    held_out = [int(value) for value in rng.permutation(held_out)]
    split = {
        "train": train,
        "val": held_out[:400],
        "test": held_out[400:],
    }
    if [len(split[name]) for name in ("train", "val", "test")] != [3200, 400, 400]:
        raise ValueError("Expected a 3200/400/400 image split")
    flattened = split["train"] + split["val"] + split["test"]
    if len(set(flattened)) != 4000:
        raise ValueError("The 8:1:1 image split overlaps or is incomplete")

    _write_json(output_root / "image_splits.json", split)
    for model in settings["models"]:
        model_output = output_root / model
        labels = _json(model_output / "labeling.json")
        generations = _json(model_output / "generations.json")
        if set(map(int, labels)) != set(flattened) or set(map(int, generations)) != set(
            flattened
        ):
            raise ValueError(f"{model}: labels/generations do not cover the 4000 images")
        _write_json(model_output / "image_splits.json", split)
    return split


def write_summary(config_path: Path) -> dict:
    _config, settings, output_root = _settings(config_path)
    split = _json(output_root / "image_splits.json")
    result = {"split": {name: len(values) for name, values in split.items()}, "models": {}}
    for model in settings["models"]:
        labels = _json(output_root / model / "labeling.json")
        all_mentions = sum(len(row.get("all_object_token_spans") or ()) for row in labels.values())
        controlled = sum(len(row.get("object_token_spans") or ()) for row in labels.values())
        official = sum(len(row.get("official_svar_samples") or ()) for row in labels.values())
        result["models"][model] = {
            "images": len(labels),
            "all_mentions": all_mentions,
            "controlled_samples": controlled,
            "official_svar_samples": official,
        }
    _write_json(output_root / "labeling_summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--model",
        default="all",
    )
    parser.add_argument("--split-only", action="store_true")
    args = parser.parse_args()
    _config, settings, _output_root = _settings(args.config)
    if args.model != "all" and args.model not in settings["models"]:
        parser.error(f"unknown model {args.model!r}; choose from {list(settings['models'])}")
    if not args.split_only:
        selected = settings["models"] if args.model == "all" else (args.model,)
        for model in selected:
            label_model(model, args.config)
    if args.model == "all" or args.split_only:
        build_split(args.config)
        write_summary(args.config)


if __name__ == "__main__":
    main()
