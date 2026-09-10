#!/usr/bin/env python3
"""Ablate T concentration from endpoint-cosine JS using saved v1 features."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.tc_fvpa_artifacts import (  # noqa: E402
    atomic_json_save,
    atomic_torch_save,
    sha256_file,
)
from scripts.analyze_ffn_endpoint_cosine_js import (  # noqa: E402
    MODELS,
    endpoint_cosine_js,
)
from scripts.analyze_ffn_visual_source_signal_ablation import (  # noqa: E402
    train_matrices,
    write_label_curve,
)
from scripts.analyze_ffn_visual_source_study import (  # noqa: E402
    SEEDS,
    _detector_matrices,
    _json_ready,
    _load_full_model,
)
from scripts.run_ffn_visual_source_attribution import EXPERIMENT, result_root  # noqa: E402
from scripts.train_ffn_consistency_alternative_heads import (  # noqa: E402
    predict_checkpoint,
)


SPECS = ("J_T", "J_E", "J_T+J_E")
SLUG = "endpoint_cosine_js_decomposition"


def endpoint_js_decomposition(position: dict) -> dict[str, np.ndarray]:
    """J_T=JS(U,T), J_E=JS(P,T)-J_T, with P from endpoint cosine."""
    endpoint, _audit = endpoint_cosine_js(position)
    target = torch.as_tensor(position["attention_evidence"], dtype=torch.float64)
    if target.ndim != 2 or not bool(torch.isfinite(target).all()) or bool((target < 0).any()):
        raise ValueError("T must be a finite non-negative [layers,visual_tokens] matrix")
    target = target / target.sum(dim=-1, keepdim=True)
    uniform = torch.full_like(target, 1.0 / target.shape[-1])
    midpoint = 0.5 * (uniform + target)
    uniform_js = 0.5 * (
        torch.special.xlogy(uniform, uniform / midpoint)
        + torch.special.xlogy(target, target / midpoint)
    ).sum(dim=-1)
    uniform_js = uniform_js.numpy().astype(np.float32)
    residual = endpoint - uniform_js
    values = {
        "J_T": uniform_js,
        "J_E": residual,
        "J_PT": endpoint,
    }
    if not all(value.ndim == 1 and np.isfinite(value).all() for value in values.values()):
        raise ValueError("endpoint JS decomposition must produce finite layer trajectories")
    return values


def decomposition_matrices(data: dict, positions: list[dict]) -> tuple[dict, dict]:
    lookup = {}
    maximum_identity_error = 0.0
    for position in positions:
        key = str(position["target_key"])
        if key in lookup:
            raise AssertionError(f"duplicate target {key}")
        values = endpoint_js_decomposition(position)
        maximum_identity_error = max(
            maximum_identity_error,
            float(np.max(np.abs(values["J_T"] + values["J_E"] - values["J_PT"]))),
        )
        lookup[key] = values
    expected = set(data["train_target_keys"]) | set(data["test_target_keys"])
    if set(lookup) != expected:
        raise AssertionError("decomposition targets differ from formal endpoint cohort")

    raw = {
        split: {
            name: np.stack([lookup[key][name] for key in data[f"{split}_target_keys"]])
            for name in ("J_T", "J_E", "J_PT")
        }
        for split in ("train", "test")
    }
    matrices = {
        split: {
            "J_T": raw[split]["J_T"],
            "J_E": raw[split]["J_E"],
            "J_T+J_E": np.concatenate(
                [raw[split]["J_T"], raw[split]["J_E"]], axis=1
            ).astype(np.float32),
        }
        for split in ("train", "test")
    }
    return matrices, {
        "maximum_float32_identity_error": maximum_identity_error,
        "ranges": {
            split: {
                name: [float(value.min()), float(value.max())]
                for name, value in raw[split].items()
            }
            for split in ("train", "test")
        },
    }


def run_model(model: str, device: torch.device) -> dict:
    started = time.monotonic()
    torch.set_num_threads(1)
    data = _detector_matrices(model)
    positions, _mentions, _images = _load_full_model(model)
    matrices, audit = decomposition_matrices(data, positions)
    del positions, _mentions, _images
    gc.collect()

    root = result_root(model)
    split_path = ROOT / "outputs" / model / EXPERIMENT / "image_splits.json"
    digest = hashlib.sha256(split_path.read_bytes())
    for split in ("train", "test"):
        digest.update(json.dumps(data[f"{split}_target_keys"]).encode())
        digest.update(data[f"y_{split}"].tobytes())
        for spec in SPECS:
            digest.update(matrices[split][spec].tobytes())
    signature = digest.hexdigest()
    progress_path = root / f"metrics/{SLUG}_progress.pt"
    result_path = root / f"metrics/{SLUG}_results.json"
    if progress_path.exists():
        progress = torch.load(progress_path, map_location="cpu", weights_only=False)
        if progress.get("cohort_feature_sha256") != signature:
            raise AssertionError("decomposition resume cohort/features differ")
    else:
        atomic_torch_save(
            {"metrics": {}, "predictions": {}, "cohort_feature_sha256": signature},
            progress_path,
        )
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result["cohort_feature_sha256"] != signature:
            raise AssertionError("completed decomposition cohort/features differ")
        for path, checksum in result["artifact_checksums"].items():
            if sha256_file(ROOT / path) != checksum:
                raise AssertionError(f"changed decomposition artifact: {path}")
        print(f"[{model}] COMPLETE: verified resume without training or writes", flush=True)
        return result

    curves = {
        name: write_label_curve(
            model,
            data,
            matrices["train"][name],
            matrices["test"][name],
            root,
            slug=slug,
            ylabel=ylabel,
            log_scale=False,
        )
        for name, slug, ylabel in (
            ("J_T", "endpoint_uniform_js", "JS(Uniform,T), nats; median / IQR"),
            ("J_E", "endpoint_js_residual", "JS(P,T)-JS(Uniform,T); median / IQR"),
        )
    }
    progress, ensembles, summaries = train_matrices(
        model,
        data,
        matrices,
        SPECS,
        device=device,
        resume=True,
        progress_name=progress_path.name,
        probe_directory=f"metrics/probes_{SLUG}",
    )
    for spec, summary in summaries.items():
        seed_aurocs = list(summary["per_seed_auroc"].values())
        summary.update(
            mean_seed_auroc=float(np.mean(seed_aurocs)),
            std_seed_auroc=float(np.std(seed_aurocs)),
            ensemble_real_aupr=float(
                average_precision_score(data["y_test"], ensembles[spec])
            ),
            ensemble_hall_aupr=float(
                average_precision_score(1 - data["y_test"], 1 - ensembles[spec])
            ),
        )

    baseline = json.loads(
        (root / "metrics/endpoint_cosine_js_feature_results.json").read_text()
    )["summary"]
    comparisons = {
        f"{spec}__minus__J_PT": {
            "ensemble_auroc_delta": summary["ensemble_auroc"]
            - baseline["ensemble_auroc"],
            "ensemble_hall_aupr_delta": summary["ensemble_hall_aupr"]
            - baseline["ensemble_hall_aupr"],
        }
        for spec, summary in summaries.items()
    }

    replay = {}
    for spec in SPECS:
        replay[spec] = {}
        for seed in SEEDS:
            directory = root / f"metrics/probes_{SLUG}/{spec}/seed{seed}"
            config = json.loads((directory / "config.json").read_text())
            probabilities = predict_checkpoint(
                "torch", directory / "model.pt", config, matrices["test"][spec], device
            )
            difference = float(
                np.max(np.abs(probabilities - progress["predictions"][spec][seed]))
            )
            if difference > 1e-7:
                raise AssertionError(
                    f"checkpoint replay differs for {spec}/seed{seed}: {difference}"
                )
            replay[spec][str(seed)] = difference

    artifact_paths = [
        progress_path,
        Path(__file__),
        ROOT / "scripts/analyze_ffn_endpoint_cosine_js.py",
        ROOT / "scripts/analyze_ffn_visual_source_signal_ablation.py",
        ROOT / "scripts/train_torch_probe_feature_sets.py",
    ]
    artifact_paths += [
        ROOT / curve[key]
        for curve in curves.values()
        for key in ("figure", "table")
    ]
    artifact_paths += list((root / f"metrics/probes_{SLUG}").glob("*/seed*/*"))
    result = {
        "model": model,
        "status": "EXPLORATORY_ENDPOINT_COSINE_JS_DECOMPOSITION_COMPLETE",
        "definitions": {
            "J_PT": "existing JS(softmax(cos(e_m,d)),T), d=G(Z)-G(Z0)",
            "J_T": "JS(Uniform,T) on all visual tokens",
            "J_E": "J_PT-J_T, signed endpoint-cosine correction",
            "J_T+J_E": "concatenate the two full layer trajectories",
        },
        "scope": "same v1 saved source, formal 3200/800 split; no VLM/bootstrap/tuning",
        "cohort_feature_sha256": signature,
        "counts": data["counts"],
        "input_audit": audit,
        "baseline_J_PT": baseline,
        "summaries": summaries,
        "comparisons": comparisons,
        "curves": curves,
        "checkpoint_replay_max_abs": replay,
        "artifact_checksums": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in artifact_paths
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_json_save(_json_ready(result), result_path)
    print(
        f"[{model}] COMPLETE "
        + " ".join(
            f"{spec}={summaries[spec]['ensemble_auroc']:.6f}" for spec in SPECS
        ),
        flush=True,
    )
    return result


def summarize() -> None:
    results = {
        model: json.loads(
            (result_root(model) / f"metrics/{SLUG}_results.json").read_text()
        )
        for model in MODELS
    }
    lines = [
        "# Endpoint-cosine JS：T集中度与endpoint修正消融",
        "",
        "每格为三seed概率ensemble AUROC / HALL-AUPR（%）；J_PT是原信号，未重训。",
        "",
        "| 特征 | Qwen2.5 | LLaVA | Qwen3 | InternVL |",
        "|---|---:|---:|---:|---:|",
    ]
    for spec in ("J_PT", *SPECS):
        cells = []
        for model in MODELS:
            summary = (
                results[model]["baseline_J_PT"]
                if spec == "J_PT"
                else results[model]["summaries"][spec]
            )
            cells.append(
                f"{100 * summary['ensemble_auroc']:.3f} / "
                f"{100 * summary['ensemble_hall_aupr']:.3f}"
            )
        lines.append(f"| {spec} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "J_T只含T相对均匀分布的集中度；J_E是有符号差值，不是JS且可为负。固定旧MLP与原split，无bootstrap或独立确认。",
    ]
    path = ROOT / f"outputs/{SLUG}_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    atomic_json_save(
        {"status": "COMPLETE", "models": results, "markdown": str(path.relative_to(ROOT))},
        ROOT / f"outputs/{SLUG}_summary.json",
    )
    print(path.read_text(), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize()
        return
    for model in args.models:
        run_model(model, torch.device(args.device))


if __name__ == "__main__":
    main()
