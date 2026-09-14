"""Evaluate True-RMS V/VP+G over model-specific curve phases at batch 128."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.tc_fvpa_artifacts import atomic_json_save, atomic_torch_save
from scripts import evaluate_ours_batchsize_vs_svar_811 as batch_sweep
from scripts import search_single_mlp_811 as search


MODELS = tuple(search.MODELS)
GROUPS = ("visual", "vp_generation")
SEEDS = (43, 44, 45)
BATCH_SIZE = 128
RANGE_NAMES = ("segment_1", "segment_2", "segment_3", "segment_4", "all")
CURVE_NAMES = ("AE_V", "logS_V", "AE_VP", "logS_VP", "AE_G", "logS_G")
MIN_SEGMENT_LAYERS = 3
OUT = ROOT / "outputs/ours_curve_layer_ranges_batch128_811_v1"


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def training_curves(source):
    """Six label-free train-set mean curves used to define model phases."""
    train = source["masks"]["train"]
    visual = np.asarray(source["groups"]["visual"][train], dtype=np.float64)
    combined = np.asarray(source["groups"]["vp_generation"][train], dtype=np.float64)
    total_layers = visual.shape[1] // 2
    if visual.shape[1] != 2 * total_layers or combined.shape[1] != 4 * total_layers:
        raise ValueError("Unexpected feature block layout")
    blocks = (
        visual[:, :total_layers],
        visual[:, total_layers:],
        combined[:, :total_layers],
        combined[:, total_layers:2 * total_layers],
        combined[:, 2 * total_layers:3 * total_layers],
        combined[:, 3 * total_layers:],
    )
    raw = np.stack([block.mean(axis=0) for block in blocks], axis=1)
    mean = raw.mean(axis=0, keepdims=True)
    scale = raw.std(axis=0, keepdims=True)
    scale[scale < 1e-12] = 1.0
    standardized = (raw - mean) / scale
    return raw, standardized


def curve_ranges(standardized):
    """Minimum-SSE four-phase segmentation with at least three layers per phase."""
    total_layers, channel_count = standardized.shape
    prefix = np.vstack([np.zeros((1, channel_count)), np.cumsum(standardized, axis=0)])
    prefix_sq = np.concatenate([[0.0], np.cumsum(np.square(standardized).sum(axis=1))])

    def cost(start, end):
        summed = prefix[end] - prefix[start]
        return prefix_sq[end] - prefix_sq[start] - float(summed @ summed) / (end - start)

    segments = 4
    scores = np.full((segments + 1, total_layers + 1), np.inf)
    previous = np.full((segments + 1, total_layers + 1), -1, dtype=np.int64)
    scores[0, 0] = 0.0
    for segment in range(1, segments + 1):
        for end in range(segment * MIN_SEGMENT_LAYERS, total_layers + 1):
            for start in range((segment - 1) * MIN_SEGMENT_LAYERS, end - MIN_SEGMENT_LAYERS + 1):
                candidate = scores[segment - 1, start] + cost(start, end)
                if candidate < scores[segment, end]:
                    scores[segment, end] = candidate
                    previous[segment, end] = start
    bounds = [total_layers]
    end = total_layers
    for segment in range(segments, 0, -1):
        end = int(previous[segment, end])
        bounds.append(end)
    bounds.sort()
    if bounds[0] != 0 or any(b - a < MIN_SEGMENT_LAYERS for a, b in zip(bounds, bounds[1:])):
        raise ValueError(f"Invalid curve segmentation: {bounds}")
    ranges = [(RANGE_NAMES[i], bounds[i], bounds[i + 1]) for i in range(segments)]
    ranges.append(("all", 0, total_layers))
    return ranges, float(scores[segments, total_layers])


def save_curve_artifacts(model_root, raw, standardized, ranges, objective):
    rows = []
    for layer in range(len(raw)):
        row = {"layer": layer}
        for index, name in enumerate(CURVE_NAMES):
            row[f"{name}_train_mean"] = raw[layer, index]
            row[f"{name}_z_over_layers"] = standardized[layer, index]
        rows.append(row)
    write_csv(rows, model_root / "layer_curves.csv")
    atomic_json_save(
        {
            "curve_source": "all training mentions, labels unused",
            "channels": list(CURVE_NAMES),
            "normalization": "independent Z-score over layers per curve",
            "algorithm": "four-piece constant dynamic programming; equal channel weights; minimum 3 layers",
            "objective_sse": objective,
            "ranges": [dict(name=name, start=start, end_exclusive=end) for name, start, end in ranges],
        },
        model_root / "curve_segmentation.json",
    )
    figure, axes = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    for axis, indices, title in (
        (axes[0], (0, 2, 4), "AE train-mean curves (standardized over layers)"),
        (axes[1], (1, 3, 5), "log-strength train-mean curves (standardized over layers)"),
    ):
        for index in indices:
            axis.plot(np.arange(len(raw)), standardized[:, index], marker="o", markersize=2.5, label=CURVE_NAMES[index])
        for segment_index, (_, start, end) in enumerate(ranges[:4]):
            axis.axvspan(start - 0.5, end - 0.5, color="0.2", alpha=0.035 if segment_index % 2 == 0 else 0.08)
            if start:
                axis.axvline(start - 0.5, color="black", linestyle="--", linewidth=0.8)
        axis.axhline(0, color="0.6", linewidth=0.6)
        axis.set_ylabel("layer-wise z-score")
        axis.set_title(title)
        axis.legend(ncol=3, fontsize=8)
    axes[1].set_xlabel("decoder layer (zero-based)")
    figure.tight_layout()
    figure.savefig(model_root / "layer_curve_segmentation.png", dpi=180)
    plt.close(figure)


def slice_layers(matrix, group, total_layers, start, end):
    block_count = 2 if group == "visual" else 4
    if matrix.ndim != 2 or matrix.shape[1] != block_count * total_layers:
        raise ValueError(f"Unexpected {group} matrix shape {matrix.shape} for L={total_layers}")
    return np.concatenate(
        [matrix[:, block * total_layers + start:block * total_layers + end] for block in range(block_count)],
        axis=1,
    )


def protocol_for(model_name, source, configs, total_layers, ranges, objective):
    protocol = {
        "schema": "ours-curve-layer-ranges-batch128-811-v1",
        "model": model_name,
        "source": "True-RMS all-attention z-A_all -> z; local FP32 Gauss-Legendre K32",
        "source_fingerprint": source["fingerprint"],
        "features": {
            "visual": "[AE_V,log1p(S_V_all)]",
            "vp_generation": "[AE_VP,log1p(S_V_all+S_P_all),AE_G,log1p(S_G_all)]",
        },
        "column_layout": "V has 2 complete L-column blocks; VP+G has 4 complete L-column blocks",
        "range_selection_data": "All training mentions only; labels unused",
        "range_selection_curves": list(CURVE_NAMES),
        "range_selection": "Per-curve layer Z-score, then four-piece constant minimum-SSE dynamic programming; equal curve weights; minimum three layers per segment",
        "range_selection_objective_sse": objective,
        "total_layers": total_layers,
        "ranges": [dict(name=name, start=start, end_exclusive=end) for name, start, end in ranges],
        "base_configs": {group: configs[(model_name, group)] for group in GROUPS},
        "batch_size": BATCH_SIZE,
        "seeds": list(SEEDS),
        "split_seed": 20260912,
        "split": "3200 train / 400 validation / 400 test images; all mentions",
        "normalization": "Per-column train-only population Z-score after layer slicing",
        "training": "No BatchNorm; all 150 epochs; minimum validation BCE checkpoint",
        "status": "Exploratory layer-range follow-up after test-set access; all ranges reported.",
    }
    protocol["fingerprint"] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    return protocol


def run_model(model_name, device):
    configs = batch_sweep.base_configs()
    for group in GROUPS:
        configs[(model_name, group)] = dict(configs[(model_name, group)], batch_size=BATCH_SIZE)
    source = search.read(search.SOURCES["true_rms"] / model_name / "matrices.pt")
    total_layers = source["groups"]["visual"].shape[1] // 2
    raw_curves, standardized_curves = training_curves(source)
    ranges, objective = curve_ranges(standardized_curves)
    protocol = protocol_for(model_name, source, configs, total_layers, ranges, objective)
    model_root = OUT / model_name
    atomic_json_save(protocol, model_root / "protocol.json")
    save_curve_artifacts(model_root, raw_curves, standardized_curves, ranges, objective)
    train, validation, test = (source["masks"][part] for part in ("train", "validation", "test"))
    rows = []
    for group in GROUPS:
        config = configs[(model_name, group)]
        full_matrix = source["groups"][group]
        for range_name, start, end in ranges:
            matrix = slice_layers(full_matrix, group, total_layers, start, end)
            for seed in SEEDS:
                path = model_root / group / range_name / f"seed{seed}" / "result.pt"
                if path.exists():
                    result = search.read(path)
                    assert result["protocol_fingerprint"] == protocol["fingerprint"]
                    assert result["config"] == config
                elif range_name == "all":
                    prior_path = (
                        batch_sweep.OUT / model_name / group / f"batch{BATCH_SIZE}" /
                        f"seed{seed}" / "result.pt"
                    )
                    result = dict(search.read(prior_path))
                    assert result["config"] == config
                    result.update(
                        protocol_fingerprint=protocol["fingerprint"],
                        range_name=range_name,
                        layer_start=start,
                        layer_end_exclusive=end,
                        reused_from=str(prior_path.relative_to(ROOT)),
                    )
                    atomic_torch_save(result, path)
                else:
                    result = search.fit(
                        train_x=matrix[train],
                        train_y=source["y"][train],
                        val_x=matrix[validation],
                        val_y=source["y"][validation],
                        cfg=config,
                        seed=seed,
                        device=device,
                    )
                    network = search.SingleMLP(result["input_dim"], config).to(device)
                    network.load_state_dict(result["state_dict"])
                    test_x = torch.as_tensor(
                        search.transform(matrix[test], result["mean"], result["scale"]), device=device
                    )
                    probabilities = search.predict(network, test_x)
                    result.update(
                        protocol_fingerprint=protocol["fingerprint"],
                        model=model_name,
                        group=group,
                        range_name=range_name,
                        layer_start=start,
                        layer_end_exclusive=end,
                        test=search.metrics(source["y"][test], probabilities),
                        test_probabilities=probabilities,
                    )
                    atomic_torch_save(result, path)
                rows.append({
                    "model": model_name,
                    "group": group,
                    "range": range_name,
                    "layer_start": start,
                    "layer_end_exclusive": end,
                    "selected_layers": end - start,
                    "input_dim": int(matrix.shape[1]),
                    "batch_size": BATCH_SIZE,
                    "seed": seed,
                    "best_epoch": result["best_epoch"],
                    "validation_AUROC": result["validation"]["AUROC"],
                    "validation_HALL_AUPR": result["validation"]["HALL_AUPR"],
                    "test_AUROC": result["test"]["AUROC"],
                    "test_HALL_AUPR": result["test"]["HALL_AUPR"],
                })
                print(model_name, group, range_name, seed, result["test"], flush=True)
    summaries = []
    for group in GROUPS:
        for range_name, start, end in ranges:
            selected = [row for row in rows if row["group"] == group and row["range"] == range_name]
            summaries.append({
                "model": model_name,
                "group": group,
                "range": range_name,
                "layer_start": start,
                "layer_end_exclusive": end,
                "selected_layers": end - start,
                "input_dim": selected[0]["input_dim"],
                "AUROC_mean": float(np.mean([row["test_AUROC"] for row in selected])),
                "AUROC_std": float(np.std([row["test_AUROC"] for row in selected])),
                "HALL_AUPR_mean": float(np.mean([row["test_HALL_AUPR"] for row in selected])),
                "HALL_AUPR_std": float(np.std([row["test_HALL_AUPR"] for row in selected])),
                "best_epochs": [row["best_epoch"] for row in selected],
            })
    write_csv(rows, model_root / "seed_metrics.csv")
    atomic_json_save({"model": model_name, "summaries": summaries}, model_root / "summary.json")


def summarize_all():
    summaries, rows = [], []
    for model_name in MODELS:
        summaries.extend(json.loads((OUT / model_name / "summary.json").read_text())["summaries"])
        with (OUT / model_name / "seed_metrics.csv").open() as handle:
            rows.extend(csv.DictReader(handle))
    atomic_json_save({"schema": "ours-curve-layer-ranges-batch128-811-v1", "summaries": summaries}, OUT / "summary.json")
    write_csv(rows, OUT / "seed_metrics.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.summarize:
        summarize_all()
    elif args.models:
        for model_name in args.models:
            run_model(model_name, args.device)
    else:
        parser.error("Provide --models or --summarize")


if __name__ == "__main__":
    main()
