#!/usr/bin/env python3
"""Evaluate JS(softmax(cos(e_m, G(Z)-G(Z0))), T) as a layerwise detector."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import sys
import time
from dataclasses import asdict
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
from scripts.run_ffn_visual_source_attribution import (  # noqa: E402
    EXPERIMENT,
    result_root,
)
from scripts.train_ffn_consistency_alternative_heads import (  # noqa: E402
    predict_checkpoint,
)
from scripts.train_torch_probe_feature_sets import TorchProbeConfig  # noqa: E402


MODELS = ("qwen2_5_vl_7b", "llava_1_5_7b", "qwen3_vl_8b", "internvl_2_5_8b")
MODEL_NAMES = {
    "qwen2_5_vl_7b": "Qwen2.5-VL-7B",
    "llava_1_5_7b": "LLaVA-1.5-7B",
    "qwen3_vl_8b": "Qwen3-VL-8B",
    "internvl_2_5_8b": "InternVL2.5-8B",
}
SPEC = "endpoint_cosine_js"


def endpoint_cosine_js(position: dict) -> tuple[np.ndarray, dict]:
    """Recover cosine from saved Q/||e||, softmax it, then compute JS to T."""
    gross = torch.as_tensor(position["ffn_path_gross"], dtype=torch.float64)
    signed = torch.as_tensor(position["path_signed_q"], dtype=torch.float64)
    target = torch.as_tensor(position["attention_evidence"], dtype=torch.float64)
    if gross.ndim != 2 or gross.shape != signed.shape or gross.shape != target.shape:
        raise ValueError("gross, signed-Q and T must share [layers,visual_tokens] shape")
    if gross.shape[1] == 0 or not bool(
        torch.isfinite(gross).all()
        and torch.isfinite(signed).all()
        and torch.isfinite(target).all()
    ):
        raise ValueError("endpoint-cosine inputs must be finite with visual support")
    if bool((gross < 0).any()) or bool((target < -2e-7).any()):
        raise ValueError("gross and T must be non-negative")

    positive = gross > 0
    cosine = torch.where(positive, signed / gross.clamp_min(torch.finfo(gross.dtype).tiny), 0.0)
    maximum_absolute_cosine = float(cosine.abs().max())
    if maximum_absolute_cosine > 1.001:
        raise ValueError(f"saved Q/||e|| violates the cosine bound: {maximum_absolute_cosine}")
    clipped_entries = int((cosine.abs() > 1.0).sum())
    cosine = cosine.clamp(-1.0, 1.0)
    source = torch.softmax(cosine, dim=-1)

    saved_target_sum_error = float((target.sum(dim=-1) - 1).abs().max())
    target = target.clamp_min(0.0)
    target_sum = target.sum(dim=-1, keepdim=True)
    if bool((target_sum <= 0).any()):
        raise ValueError("T must have positive mass in every layer")
    target = target / target_sum
    midpoint = 0.5 * (source + target)
    js = 0.5 * (
        (source * (source.log() - midpoint.log())).sum(dim=-1)
        + torch.where(
            target > 0,
            target * (target.clamp_min(torch.finfo(target.dtype).tiny).log() - midpoint.log()),
            0.0,
        ).sum(dim=-1)
    )
    if not bool(torch.isfinite(js).all()) or bool((js < -1e-12).any()) or bool(
        (js > np.log(2.0) + 1e-10).any()
    ):
        raise AssertionError("natural-log JS must be finite in [0,ln(2)]")
    return js.numpy().astype(np.float32), {
        "layers": int(gross.shape[0]),
        "visual_tokens": int(gross.shape[1]),
        "source_entries": int(gross.numel()),
        "zero_gross_entries": int((~positive).sum()),
        "clipped_cosine_entries": clipped_entries,
        "maximum_absolute_preclip_cosine": maximum_absolute_cosine,
        "maximum_source_sum_error": float((source.sum(dim=-1) - 1).abs().max()),
        "maximum_saved_target_sum_error": saved_target_sum_error,
        "minimum_js": float(js.min()),
        "maximum_js": float(js.max()),
    }


def endpoint_cosine_js_matrices(data: dict, positions: list[dict]) -> tuple[dict, dict]:
    lookup, audits = {}, []
    for position in positions:
        key = str(position["target_key"])
        if key in lookup:
            raise AssertionError(f"duplicate target {key}")
        lookup[key], audit = endpoint_cosine_js(position)
        audits.append(audit)
    expected = set(data["train_target_keys"]) | set(data["test_target_keys"])
    if set(lookup) != expected:
        raise AssertionError("endpoint-cosine targets differ from the formal cohort")
    matrices = {
        split: {SPEC: np.stack([lookup[key] for key in data[f"{split}_target_keys"]])}
        for split in ("train", "test")
    }
    return matrices, {
        "unique_targets": len(lookup),
        "target_layers": sum(row["layers"] for row in audits),
        "source_entries": sum(row["source_entries"] for row in audits),
        "zero_gross_entries": sum(row["zero_gross_entries"] for row in audits),
        "clipped_cosine_entries": sum(row["clipped_cosine_entries"] for row in audits),
        "maximum_absolute_preclip_cosine": max(
            row["maximum_absolute_preclip_cosine"] for row in audits
        ),
        "maximum_source_sum_error": max(row["maximum_source_sum_error"] for row in audits),
        "maximum_saved_target_sum_error": max(
            row["maximum_saved_target_sum_error"] for row in audits
        ),
        "minimum_js": min(row["minimum_js"] for row in audits),
        "maximum_js": max(row["maximum_js"] for row in audits),
        "visual_token_counts": sorted({row["visual_tokens"] for row in audits}),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_model(model: str, device: torch.device) -> dict:
    started = time.monotonic()
    torch.set_num_threads(1)
    data = _detector_matrices(model)
    positions, _mentions, _images = _load_full_model(model)
    matrices, input_audit = endpoint_cosine_js_matrices(data, positions)
    del positions, _mentions, _images
    gc.collect()

    root = result_root(model)
    split_path = ROOT / "outputs" / model / EXPERIMENT / "image_splits.json"
    digest = hashlib.sha256(split_path.read_bytes())
    for split in ("train", "test"):
        values = matrices[split][SPEC]
        digest.update(json.dumps(data[f"{split}_target_keys"]).encode())
        digest.update(data[f"y_{split}"].tobytes())
        digest.update(values.tobytes())
    signature = digest.hexdigest()
    progress_path = root / f"metrics/{SPEC}_training_progress.pt"
    result_path = root / f"metrics/{SPEC}_feature_results.json"
    if progress_path.exists():
        progress = torch.load(progress_path, map_location="cpu", weights_only=False)
        if progress.get("cohort_feature_sha256") != signature:
            raise AssertionError("endpoint-cosine JS resume cohort/features differ")
    else:
        progress = {"metrics": {}, "predictions": {}, "cohort_feature_sha256": signature}
        atomic_torch_save(progress, progress_path)
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result["cohort_feature_sha256"] != signature:
            raise AssertionError("completed endpoint-cosine JS cohort/features differ")
        for path, checksum in result["artifact_checksums"].items():
            if sha256_file(ROOT / path) != checksum:
                raise AssertionError(f"changed endpoint-cosine JS artifact: {path}")
        print(f"[{model}] COMPLETE: verified resume without training or writes", flush=True)
        return result

    curve = write_label_curve(
        model,
        data,
        matrices["train"][SPEC],
        matrices["test"][SPEC],
        root,
        slug=SPEC,
        ylabel="JS(P_endpoint-cos, T), nats (median; IQR)",
        log_scale=False,
    )
    progress, ensembles, summaries = train_matrices(
        model,
        data,
        matrices,
        (SPEC,),
        device=device,
        resume=True,
        progress_name=progress_path.name,
        probe_directory=f"metrics/probes_{SPEC}",
    )
    summary = summaries[SPEC]
    seed_aurocs = list(summary["per_seed_auroc"].values())
    summary.update(
        mean_seed_auroc=float(np.mean(seed_aurocs)),
        std_seed_auroc=float(np.std(seed_aurocs)),
        ensemble_real_aupr=float(
            average_precision_score(data["y_test"], ensembles[SPEC])
        ),
        ensemble_hall_aupr=float(
            average_precision_score(1 - data["y_test"], 1 - ensembles[SPEC])
        ),
    )

    replay, seed_rows = {}, []
    for seed in SEEDS:
        directory = root / f"metrics/probes_{SPEC}/{SPEC}/seed{seed}"
        config = json.loads((directory / "config.json").read_text())
        probabilities = predict_checkpoint(
            "torch", directory / "model.pt", config, matrices["test"][SPEC], device
        )
        difference = float(
            np.max(np.abs(probabilities - progress["predictions"][SPEC][seed]))
        )
        if difference > 1e-7:
            raise AssertionError(f"checkpoint replay differs for seed {seed}: {difference}")
        replay[str(seed)] = {"maximum_probability_difference": difference}
        for rule, report in progress["metrics"][SPEC][seed]["threshold_reports"].items():
            metrics = report["test_metrics"]
            seed_rows.append(
                {
                    "seed": seed,
                    "threshold_rule": rule,
                    "decision_threshold": report["threshold"],
                    "auroc": metrics["auc"],
                    **{
                        f"{label}_{name}": metrics[key][name]
                        for label, key in (
                            ("real", "real_positive"),
                            ("hall", "hallucination_positive"),
                        )
                        for name in ("aupr", "precision", "recall", "f1")
                    },
                }
            )
    seed_table = root / f"tables/{SPEC}_seed_metrics.csv"
    _write_csv(seed_table, seed_rows)

    artifact_paths = [
        progress_path,
        seed_table,
        ROOT / curve["figure"],
        ROOT / curve["table"],
        Path(__file__),
        ROOT / "features/ffn_visual_path_attribution.py",
        ROOT / "scripts/analyze_ffn_visual_source_signal_ablation.py",
        ROOT / "scripts/train_torch_probe_feature_sets.py",
    ]
    artifact_paths += list((root / f"metrics/probes_{SPEC}").glob("*/seed*/*"))
    result = {
        "model": model,
        "status": "EXPLORATORY_ENDPOINT_COSINE_JS_COMPLETE",
        "definition": {
            "endpoint": "d=G(Z)-G(Z0)",
            "saved_identity": "path_signed_q=<e_m,d/||d||>; ffn_path_gross=||e_m||",
            "cosine": "path_signed_q/ffn_path_gross; zero when ||e_m||=0 or d is degenerate",
            "source_distribution": "softmax(cosine) over all visual tokens; temperature=1",
            "target_distribution": "saved normalized attention_evidence T over the same visual tokens",
            "signal": "natural-log JS(P_endpoint_cosine,T), one scalar per decoder layer",
        },
        "protocol": {
            **_json_ready(asdict(TorchProbeConfig())),
            "seeds": list(SEEDS),
            "training_device": str(device),
            "standardization": "none",
            "class_weighting_or_resampling": "none",
            "bootstrap": "NOT_RUN",
            "inference_boundary": "saved compact source-attribution shards only; no VLM forward",
        },
        "counts": data["counts"],
        "cohort_feature_sha256": signature,
        "input_audit": input_audit,
        "curve": curve,
        "summary": summary,
        "checkpoint_replay": replay,
        "artifact_checksums": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in artifact_paths
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_json_save(_json_ready(result), result_path)
    print(
        f"[{model}] COMPLETE AUROC={summary['ensemble_auroc']:.6f} "
        f"HallAUPR={summary['ensemble_hall_aupr']:.6f}",
        flush=True,
    )
    return result


def summarize(models: tuple[str, ...]) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = {
        model: json.loads(
            (result_root(model) / f"metrics/{SPEC}_feature_results.json").read_text()
        )
        for model in models
    }
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharey=True)
    for axis, (model, result) in zip(axes.flat, results.items()):
        with (ROOT / result["curve"]["table"]).open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        for label, color in (("REAL", "#2166ac"), ("HALL", "#b2182b")):
            selected = [row for row in rows if row["label"] == label]
            layers = [int(row["layer"]) for row in selected]
            axis.plot(
                layers,
                [float(row["median"]) for row in selected],
                color=color,
                label=f"{label} (n={selected[0]['n_mentions']})",
            )
            axis.fill_between(
                layers,
                [float(row["q25"]) for row in selected],
                [float(row["q75"]) for row in selected],
                color=color,
                alpha=0.14,
            )
        axis.set(title=MODEL_NAMES[model], xlabel="Decoder layer", ylabel="JS (nats)")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=9)
    for axis in list(axes.flat)[len(results) :]:
        axis.set_visible(False)
    fig.suptitle("JS(softmax(cos(e_m, G(Z)-G(Z0))), T): REAL / HALL", fontsize=14)
    fig.text(
        0.5,
        0.014,
        "Medians and IQR (not CI) | All formal train + test mentions | Natural-log JS",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    for extension in ("png", "pdf"):
        fig.savefig(ROOT / f"outputs/ffn_{SPEC}_real_hall.{extension}", dpi=180)
    plt.close(fig)

    lines = [
        "# Endpoint-cosine distribution vs T：逐层 JS 幻觉检测",
        "",
        "每格为三 seed 概率 ensemble 的 AUROC / HALL-AUPR；F1 是各 seed 在训练集 REAL-F1 阈值下的测试 HALL-F1 均值。",
        "",
        "| Model | AUROC | HALL-AUPR | Mean HALL-F1 |",
        "|---|---:|---:|---:|",
    ]
    for model, result in results.items():
        summary = result["summary"]
        lines.append(
            f"| {MODEL_NAMES[model]} | {summary['ensemble_auroc']:.6f} | "
            f"{summary['ensemble_hall_aupr']:.6f} | {summary['mean_hall_f1']:.6f} |"
        )
    lines += [
        "",
        "输入仅从已保存的 `path_signed_q`、`ffn_path_gross` 和 `attention_evidence` 恢复；未运行VLM。无bootstrap或独立cohort，结果为探索性。",
    ]
    markdown_path = ROOT / f"outputs/ffn_{SPEC}_summary.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = {
        "status": "EXPLORATORY_ENDPOINT_COSINE_JS_COMPLETE",
        "models": results,
        "figure": f"outputs/ffn_{SPEC}_real_hall.png",
        "markdown": str(markdown_path.relative_to(ROOT)),
    }
    atomic_json_save(payload, ROOT / f"outputs/ffn_{SPEC}_summary.json")
    print(markdown_path.read_text(), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--training-device", default="cpu")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    models = tuple(filter(None, map(str.strip, args.models.split(","))))
    if not models or len(set(models)) != len(models) or set(models) - set(MODELS):
        raise ValueError("choose unique supported model names")
    if args.summarize_only:
        summarize(models)
        return
    for model in models:
        run_model(model, torch.device(args.training_device))


if __name__ == "__main__":
    main()
