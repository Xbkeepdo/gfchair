"""Search the established 24-candidate Torch MLP grid on legacy Visual-only paths."""

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from detection.baselines import evaluate_detection_scores, select_detection_threshold
from features.tc_fvpa_artifacts import atomic_json_save, atomic_torch_save
from scripts import search_single_mlp_811 as search


OUT = ROOT / "outputs/legacy_visual_single_mlp_search_811_v1"
SOURCE = ROOT / "outputs/all_attention_ae_811_v1"
OLD_RESULTS = SOURCE / "detection.csv"
SVAR_RESULTS = ROOT / "outputs/svar_llava_5_18_proportional_811_v1/summary.json"
MODELS = tuple(search.MODELS)
SEEDS = tuple(search.SEEDS)
GROUP = "legacy_visual"
TOP_N = search.TOP_N


def _now():
    return datetime.now(timezone.utc).isoformat()


def _read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_source(model):
    data = search.read(SOURCE / model / "matrices.pt")
    matrix = np.asarray(data["groups"][GROUP], dtype=np.float32)
    y = np.asarray(data["y"], dtype=np.int64)
    masks = {name: np.asarray(mask, dtype=bool) for name, mask in data["masks"].items()}
    if matrix.ndim != 2 or len(matrix) != len(y):
        raise ValueError(f"Invalid {model} {GROUP} matrix shape {matrix.shape}")
    if not np.isfinite(matrix).all() or set(np.unique(y)) != {0, 1}:
        raise ValueError(f"Invalid finite values or labels for {model}")
    if set(masks) != {"train", "validation", "test"}:
        raise ValueError(f"Unexpected masks for {model}: {sorted(masks)}")
    membership = sum(mask.astype(np.int8) for mask in masks.values())
    if not np.all(membership == 1):
        raise ValueError(f"Non-exclusive split masks for {model}")
    return data, matrix, y, masks


def make_protocol(model, source):
    source_protocol = json.loads((SOURCE / model / "protocol.json").read_text())
    protocol = {
        "schema": "legacy-visual-single-hidden-search-811-v1",
        "model": model,
        "feature": "legacy_visual=[AE_V, log1p(S_E)]",
        "path": "Visual-only conditional path; true RMS is differentiated along z-A_V -> z",
        "candidates": search.candidates(),
        "search_seed": search.SEARCH_SEED,
        "shortlist_seed": 43,
        "top_n": TOP_N,
        "final_seeds": list(SEEDS),
        "ranking": "validation AUROC desc, HALL_AUPR desc, candidate index asc",
        "split": source_protocol["split"],
        "source_fingerprint": source["fingerprint"],
        "source_path": str((SOURCE / model / "matrices.pt").relative_to(ROOT)),
        "architecture": (
            "One hidden Linear -> optional BN -> ReLU/GELU -> dropout -> output Linear; "
            "BCE REAL=1, Adam"
        ),
        "budget": {
            "seed43_candidates": len(search.candidates()),
            "shortlist": TOP_N,
            "extra_seeds_per_shortlist": 2,
            "fits_per_model": len(search.candidates()) + TOP_N * 2,
            "total_fits": len(MODELS) * (len(search.candidates()) + TOP_N * 2),
        },
        "test_gate": "All four validation selections must be frozen before test evaluation",
        "caveat": (
            "The fixed 400-image test split was used by earlier studies; this is exploratory "
            "and test metrics do not select hyperparameters."
        ),
    }
    protocol["fingerprint"] = hashlib.sha256(
        json.dumps(protocol, sort_keys=True).encode()
    ).hexdigest()
    return protocol


def prepare(model):
    source, matrix, y, masks = load_source(model)
    protocol = make_protocol(model, source)
    path = OUT / model / "protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise ValueError(f"Protocol changed for {model}")
    else:
        atomic_json_save(protocol, path)
    return (
        matrix[masks["train"]],
        y[masks["train"]],
        matrix[masks["validation"]],
        y[masks["validation"]],
        protocol,
    )


def rank(row):
    return (
        row["validation"]["AUROC"],
        row["validation"]["HALL_AUPR"],
        -row["index"],
    )


def run_search(model, device):
    root = OUT / model
    train_x, train_y, val_x, val_y, protocol = prepare(model)
    candidates = protocol["candidates"]
    done = 0
    total = protocol["budget"]["fits_per_model"]

    def progress(stage, **extra):
        atomic_json_save(
            {
                "stage": stage,
                "completed": done,
                "total": total,
                "status": "running",
                "heartbeat": _now(),
                **extra,
            },
            root / "progress.json",
        )

    def fit_one(index, seed):
        nonlocal done
        path = root / "search" / f"candidate{index:02d}" / f"seed{seed}" / "result.pt"
        if path.exists():
            result = search.read(path)
            if result["fingerprint"] != protocol["fingerprint"]:
                raise ValueError(f"Fingerprint mismatch in {path}")
        else:
            progress(f"candidate{index:02d}/seed{seed}", index=index, seed=seed)
            result = search.fit(
                train_x,
                train_y,
                val_x,
                val_y,
                candidates[index],
                seed,
                device,
                callback=lambda epoch: progress(
                    f"candidate{index:02d}/seed{seed}", index=index, seed=seed, epoch=epoch
                ),
            )
            result.update(
                fingerprint=protocol["fingerprint"],
                index=index,
                feature=GROUP,
            )
            atomic_torch_save(result, path)
            print(
                "FIT",
                model,
                index,
                seed,
                result["validation"],
                round(result["seconds"], 2),
                flush=True,
            )
        done += 1
        return {"index": index, "seed": seed, "validation": result["validation"]}

    seed43 = [fit_one(index, 43) for index in range(len(candidates))]
    shortlist = sorted(seed43, key=rank, reverse=True)[:TOP_N]
    comparisons = []
    for row in shortlist:
        seeds = [row, fit_one(row["index"], 44), fit_one(row["index"], 45)]
        comparisons.append(
            {
                "index": row["index"],
                "validation": {
                    key: float(np.mean([item["validation"][key] for item in seeds]))
                    for key in row["validation"]
                },
                "seeds": seeds,
            }
        )
    best = max(comparisons, key=rank)
    selection = {
        "fingerprint": protocol["fingerprint"],
        "index": best["index"],
        "config": candidates[best["index"]],
        "validation": best["validation"],
        "shortlist": comparisons,
        "seed43_trials": seed43,
        "frozen_at": _now(),
        "test_accessed": False,
    }
    atomic_json_save(selection, root / "selection.json")
    atomic_json_save(
        {
            "stage": "validation selection frozen",
            "completed": total,
            "total": total,
            "status": "selected",
            "heartbeat": _now(),
        },
        root / "progress.json",
    )
    if done != total:
        raise AssertionError((done, total))


def test_gate():
    for model in MODELS:
        protocol_path = OUT / model / "protocol.json"
        selection_path = OUT / model / "selection.json"
        if not protocol_path.exists() or not selection_path.exists():
            return False
        protocol = json.loads(protocol_path.read_text())
        selection = json.loads(selection_path.read_text())
        if protocol["candidates"] != search.candidates():
            raise ValueError(f"Candidate mismatch for {model}")
        if selection["fingerprint"] != protocol["fingerprint"]:
            raise ValueError(f"Selection mismatch for {model}")
    return True


def evaluate(model, device):
    if not test_gate():
        raise RuntimeError("All four validation selections must be frozen before test evaluation")
    source, matrix, y, masks = load_source(model)
    root = OUT / model
    protocol = json.loads((root / "protocol.json").read_text())
    selection = json.loads((root / "selection.json").read_text())
    index = selection["index"]
    test_x = matrix[masks["test"]]
    test_y = y[masks["test"]]
    rows = []
    probabilities = []
    for seed in SEEDS:
        trained = search.read(
            root / "search" / f"candidate{index:02d}" / f"seed{seed}" / "result.pt"
        )
        if trained["fingerprint"] != protocol["fingerprint"]:
            raise ValueError(f"Training fingerprint mismatch for {model} seed{seed}")
        network = search.SingleMLP(trained["input_dim"], trained["config"]).to(device)
        network.load_state_dict(trained["state_dict"])
        transformed = torch.as_tensor(
            search.transform(test_x, trained["mean"], trained["scale"]), device=device
        )
        probability = search.predict(network, transformed)
        value = search.metrics(test_y, probability)
        threshold = select_detection_threshold(
            y[masks["train"]], 1 - trained["train_probabilities"], positive_class="real"
        )
        final = {
            "schema": "legacy-visual-single-hidden-final-811-v1",
            "model": model,
            "feature": GROUP,
            "seed": seed,
            "candidate": index,
            "fingerprint": protocol["fingerprint"],
            "config": trained["config"],
            "best_epoch": trained["best_epoch"],
            "validation": trained["validation"],
            "test_metrics": value,
            "test_probabilities": probability,
            "test_labels": test_y,
            "threshold": threshold,
            "threshold_reports": {
                name: evaluate_detection_scores(
                    test_y, 1 - probability, selected, positive_class="real"
                )
                for name, selected in (("train_f1", threshold), ("fixed_0.5", 0.5))
            },
        }
        atomic_torch_save(final, root / "final" / f"seed{seed}" / "result.pt")
        rows.append({"model": model, "seed": seed, **value})
        probabilities.append(probability)
    summary = {
        "model": model,
        "feature": GROUP,
        "candidate": index,
        "config": selection["config"],
        "validation": selection["validation"],
        **{
            f"{key}_{stat}": float(fn([row[key] for row in rows]))
            for key in ("AUROC", "HALL_AUPR")
            for stat, fn in (("mean", np.mean), ("std", np.std))
        },
        "ensemble": search.metrics(test_y, np.mean(probabilities, axis=0)),
    }
    _write_csv(rows, root / "seed_metrics.csv")
    atomic_json_save(summary, root / "summary.json")
    selection["test_accessed"] = True
    selection["test_accessed_at"] = _now()
    atomic_json_save(selection, root / "selection.json")
    atomic_json_save(
        {
            "stage": "complete",
            "completed": 3,
            "total": 3,
            "status": "completed",
            "heartbeat": _now(),
        },
        root / "progress.json",
    )


def summarize():
    summaries = [json.loads((OUT / model / "summary.json").read_text()) for model in MODELS]
    old_rows = {
        (row["model"], row["classifier"]): row
        for row in _read_csv(OLD_RESULTS)
        if row["group"] == GROUP
    }
    svar = {
        row["model"]: row
        for row in json.loads(SVAR_RESULTS.read_text())["summaries"]
    }
    rows = []
    for new in summaries:
        model = new["model"]
        for classifier, label in (
            ("torch_searched", "Torch single MLP (24-candidate search)"),
            ("three_hidden", "Earlier three-hidden MLP"),
            ("one_hidden", "Earlier sklearn single MLP"),
            ("xgb", "Earlier searched XGB"),
            ("svar", "Native SVAR proportional layers"),
        ):
            if classifier == "torch_searched":
                auroc, auroc_std = new["AUROC_mean"], new["AUROC_std"]
                ap, ap_std = new["HALL_AUPR_mean"], new["HALL_AUPR_std"]
            elif classifier == "svar":
                item = svar[model]
                auroc, auroc_std = item["AUROC_mean"], item["AUROC_std"]
                ap, ap_std = item["HALL_AUPR_mean"], item["HALL_AUPR_std"]
            else:
                item = old_rows[(model, classifier)]
                auroc, auroc_std = float(item["AUROC_mean"]), float(item["AUROC_std"])
                ap, ap_std = float(item["HALL_AUPR_mean"]), float(item["HALL_AUPR_std"])
            rows.append(
                {
                    "model": model,
                    "classifier": classifier,
                    "label": label,
                    "AUROC_mean": auroc,
                    "AUROC_std": auroc_std,
                    "HALL_AUPR_mean": ap,
                    "HALL_AUPR_std": ap_std,
                }
            )
    _write_csv(rows, OUT / "comparison.csv")
    atomic_json_save(
        {
            "schema": "legacy-visual-single-hidden-search-811-summary-v1",
            "summaries": summaries,
            "comparison": rows,
        },
        OUT / "summary.json",
    )

    titles = {
        "qwen2_5_vl_7b": "Qwen2.5-VL-7B",
        "llava_1_5_7b": "LLaVA-1.5-7B",
        "qwen3_vl_8b": "Qwen3-VL-8B",
        "internvl_2_5_8b": "InternVL2.5-8B",
    }
    lines = [
        "# Visual-only 条件路径：Torch 单层 MLP 搜参（811）",
        "",
        "固定3200/400/400图片划分及全部mentions。特征为旧Visual-only条件路径 "
        "`[AE_V, log1p(S_E)]`；复用缓存，不重新提取。24候选先以seed43验证集排序，"
        "前三名补seeds44/45，再以三seed验证AUROC/HALL-AUPR冻结配置；四模型全部冻结后才读取测试集。",
        "",
        "表中均为三个训练seed的测试均值 ± 总体标准差，单位%。XGB为同一811实验中已有的18候选验证搜参结果。",
        "",
        "| 模型 | Torch单层搜参 AUROC/AP | 早期三层MLP | 早期sklearn单层 | 搜参XGB | 原生SVAR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    by_key = {(row["model"], row["classifier"]): row for row in rows}

    def cell(model, classifier):
        row = by_key[(model, classifier)]
        return (
            f"{100*row['AUROC_mean']:.2f}±{100*row['AUROC_std']:.2f} / "
            f"{100*row['HALL_AUPR_mean']:.2f}±{100*row['HALL_AUPR_std']:.2f}"
        )

    for model in MODELS:
        lines.append(
            f"| {titles[model]} | {cell(model, 'torch_searched')} | "
            f"{cell(model, 'three_hidden')} | {cell(model, 'one_hidden')} | "
            f"{cell(model, 'xgb')} | {cell(model, 'svar')} |"
        )
    lines.extend(["", "## 验证集选出的参数", ""])
    for item in summaries:
        lines.append(
            f"- {titles[item['model']]}：candidate {item['candidate']}，"
            f"`{json.dumps(item['config'], ensure_ascii=False, sort_keys=True)}`"
        )
    lines.extend(
        [
            "",
            "本实验沿用已经多次访问过的固定测试集，属于探索性比较。不同分类器的搜参预算也不同，"
            "因此只能比较当前完整管线，不能把差值完全归因于特征或分类器。",
            "",
            "完整逐seed指标见 `outputs/legacy_visual_single_mlp_search_811_v1/*/seed_metrics.csv`。",
        ]
    )
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")


def validate():
    if not test_gate():
        raise RuntimeError("Selection gate is incomplete")
    checked = 0
    max_probability_error = 0.0
    for model in MODELS:
        source, matrix, y, masks = load_source(model)
        summary = json.loads((OUT / model / "summary.json").read_text())
        selection = json.loads((OUT / model / "selection.json").read_text())
        values = []
        for seed in SEEDS:
            final = search.read(OUT / model / "final" / f"seed{seed}" / "result.pt")
            trained = search.read(
                OUT
                / model
                / "search"
                / f"candidate{selection['index']:02d}"
                / f"seed{seed}"
                / "result.pt"
            )
            network = search.SingleMLP(trained["input_dim"], trained["config"])
            network.load_state_dict(trained["state_dict"])
            test_x = torch.as_tensor(
                search.transform(matrix[masks["test"]], trained["mean"], trained["scale"])
            )
            probability = search.predict(network, test_x)
            error = float(np.max(np.abs(probability - final["test_probabilities"])))
            max_probability_error = max(max_probability_error, error)
            value = search.metrics(y[masks["test"]], probability)
            for key in value:
                if abs(value[key] - final["test_metrics"][key]) > 1e-12:
                    raise AssertionError((model, seed, key, value[key], final["test_metrics"][key]))
            values.append(value)
            checked += 1
        for key in ("AUROC", "HALL_AUPR"):
            if abs(float(np.mean([v[key] for v in values])) - summary[f"{key}_mean"]) > 1e-12:
                raise AssertionError((model, key))
    report = {
        "schema": "legacy-visual-single-hidden-search-811-validation-v1",
        "status": "PASS",
        "checked_heads": checked,
        "max_probability_abs_error": max_probability_error,
        "all_source_matrices_finite": True,
        "selection_gate_complete": True,
        "validated_at": _now(),
    }
    atomic_json_save(report, OUT / "validation.json")
    print(json.dumps(report, indent=2), flush=True)


def locked(model, action, device):
    root = OUT / model
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        action(model, device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "search", "evaluate", "summarize", "validate"), required=True)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.stage in {"prepare", "search", "evaluate"} and args.model is None:
        parser.error(f"--model is required for --stage {args.stage}")
    if args.stage == "prepare":
        prepare(args.model)
    elif args.stage == "search":
        locked(args.model, run_search, args.device)
    elif args.stage == "evaluate":
        locked(args.model, evaluate, args.device)
    elif args.stage == "summarize":
        summarize()
    else:
        validate()


if __name__ == "__main__":
    main()
