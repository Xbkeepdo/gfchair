"""Four-model COCO4000 Operator Visual-write Innovation Ratio experiment."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.ffn_all_source_paths import partition, path_operator
from features.ffn_visual_path_attribution import quadrature_rule
from features.ovir import (
    operator_visual_write_innovation,
    response_from_token_matrix,
    visual_write_energy_basis,
)
from features.tc_fvpa_artifacts import atomic_json_save, atomic_torch_save
from features.visual_ffn_jacobian import reconstruct_visual_directions, resolve_decoder_layer_adapter
from models.dgst_capture import attention_row_from_capture, resolve_decoder_layers
from scripts.analyze_all_source_paths import mention_identity
from scripts.run_ffn_source_composition import (
    EXPERIMENT,
    MODELS,
    capture_full,
    load_wrapper,
    parent_paths,
    read,
)
from scripts.run_ffn_visual_source_consistency import local_fp32
from scripts.run_jffn_p_comparison import _image_path, _load_inputs


OUT = ROOT / "outputs/ovir_all_attention_4000_v1"
BASE = ROOT / "outputs/ffn_all_source_paths_v1"
MATRIX_BASE = ROOT / "outputs/all_attention_ae_811_v1"
LAYERS = dict(zip(MODELS, (28, 32, 36, 32)))
EXPECTED_TARGETS = dict(zip(MODELS, (8654, 14951, 14704, 11630)))
METRICS = (
    "ovir",
    "rank95",
    "retained_energy",
    "previous_retained_energy",
    "projection_residual_energy",
    "visual_tokens",
    "source_energy",
    "operator_total_energy",
    "operator_inside_energy",
    "operator_outside_energy",
    "basis_orthogonality_error",
    "basis_energy_identity_error",
    "operator_energy_identity_error",
    "max_singular_value",
    "min_kept_singular_value",
    "zero_input",
    "zero_operator",
)
CURVE_METRICS = ("ovir", "rank95", "retained_energy")
ENERGY_THRESHOLD = 0.95
QUADRATURE_POINTS = 32
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260915


def utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Progress:
    def __init__(self, model: str, out: Path):
        self.path = out / model / "progress.json"
        self.value = dict(
            model=model,
            stage="准备",
            completed=0,
            total=1,
            status="running",
            pid=os.getpid(),
        )
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.update()
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.thread.start()

    def update(self, stage=None, completed=None, total=None, **values):
        with self.lock:
            for key, value in dict(stage=stage, completed=completed, total=total, **values).items():
                if value is not None:
                    self.value[key] = value
            self.value["heartbeat"] = utc()
            atomic_json_save(self.value, self.path)

    def _heartbeat(self):
        while not self.stop.wait(10):
            self.update()

    def finish(self, status: str, error: str = ""):
        self.stop.set()
        self.thread.join(timeout=11)
        self.update(status=status, error=error)


@contextmanager
def model_lock(model: str, out: Path):
    folder = out / model
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def protocol(out: Path) -> dict:
    base_protocol = json.loads((BASE / "protocol.json").read_text())
    files = (Path(__file__), ROOT / "features/ovir.py")
    value = dict(
        version=1,
        models=list(MODELS),
        code_hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        source_protocol_signature=base_protocol["signature"],
        cohort="four models x original COCO4000; all targets and decoder layers",
        visual_subspace="raw A_visual columns; no column normalization; smallest FP64 Gram/SVD rank with >=95% energy",
        operator="true RMSNorm+FFN average Jacobian on z-A_all -> z; Gauss-Legendre K32; local FP32; TF32 off",
        ovir="||(I-UU^T) Jbar U||_F^2 / ||Jbar U||_F^2; direct orthogonal energy decomposition",
        undefined="zero visual-WRITE input or zero operator response is NaN and counted; never imputed",
        labels="labels are used only after extraction; conflicting target labels excluded from mechanism comparisons",
        split="fixed 811 split: 3200 train / 400 validation / 400 test; extraction always all 4000",
        bootstrap=dict(replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED,
                       unit="mixed-label image; HALL-minus-REAL within image", interval="nominal percentile 95%"),
        no_target_conditioning=True,
        no_detector_training=True,
    )
    value["signature"] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    path = out / "protocol.json"
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".protocol.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError("OVIR protocol changed; use a new output root")
        if not path.exists():
            atomic_json_save(value, path)
    return value


def fixed_split(model: str) -> dict[str, list[int]]:
    value = json.loads((MATRIX_BASE / model / "protocol.json").read_text())["split"]
    if [len(value[k]) for k in ("train", "validation", "test")] != [3200, 400, 400]:
        raise ValueError(f"Invalid fixed 811 split for {model}")
    if len(set().union(*(set(value[k]) for k in value))) != 4000:
        raise ValueError(f"Fixed split does not cover 4000 unique images for {model}")
    return value


def source_snapshot(model: str) -> dict:
    parents = parent_paths(model)
    if len(parents) != 4000:
        raise ValueError(f"Expected 4000 source shards for {model}")
    rows = []
    for image_id, path in sorted(parents.items()):
        stat = path.stat()
        rows.append(dict(image_id=image_id, path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns))
    experiment = ROOT / "outputs" / model / EXPERIMENT
    extras = []
    for name in ("generations.json", "labeling.json", "image_splits.json"):
        path = experiment / name
        stat = path.stat()
        extras.append(dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns))
    return dict(model=model, parents=rows, extras=extras)


def ensure_source_snapshot(model: str, out: Path) -> dict:
    path = out / model / "source_manifest.json"
    current = source_snapshot(model)
    if path.exists():
        if json.loads(path.read_text()) != current:
            raise ValueError(f"Source artifacts changed for {model}")
    else:
        atomic_json_save(current, path)
    return current


def _relative(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual.double() - expected.double()).norm() / expected.double().norm().clamp_min(1e-30))


def _apply_basis(operator, basis: torch.Tensor, chunk: int = 16) -> torch.Tensor:
    if basis.shape[1] == 0:
        return basis.clone()
    parts = []
    probes = basis.T[:, None, :]
    for start in range(0, len(probes), int(chunk)):
        parts.append(operator(probes[start:start + int(chunk)])[:, 0].T)
    return torch.cat(parts, dim=1)


def _direct_svd_span_error(a: torch.Tensor, basis: torch.Tensor, rank: int) -> float:
    source = a.detach().double().cpu()
    u, singular, _ = torch.linalg.svd(source, full_matrices=False)
    energy = singular.square().cumsum(0) / singular.square().sum()
    direct_rank = int(torch.searchsorted(energy, energy.new_tensor(ENERGY_THRESHOLD)).item()) + 1
    if direct_rank != rank:
        raise ValueError(f"Gram/direct SVD rank mismatch: {rank} vs {direct_rank}")
    q = basis.detach().double().cpu()
    return float((u[:, :rank] - q @ (q.T @ u[:, :rank])).norm() / math.sqrt(rank))


def _audit_artifact(model: str, case: dict) -> Path:
    key = case["target_key"].replace(":", "_")
    return BASE / model / "audit" / f"{key}_L{int(case['layer']) + 1}.pt"


def _run_audit(model: str, device: str, out: Path, progress: Progress) -> dict:
    cases = json.loads((BASE / model / "audit/selection.json").read_text())
    if len(cases) != 50:
        raise ValueError(f"Expected 50 existing audit cases for {model}")
    parents = parent_paths(model)
    _, generations, _ = _load_inputs(ROOT / "outputs" / model / EXPERIMENT)
    wrapper, config = load_wrapper(model, device)
    layers = resolve_decoder_layers(wrapper.model)
    by_image = defaultdict(list)
    for index, case in enumerate(cases):
        by_image[int(case["image_id"])].append((index, case))
    maxima = dict(
        live_saved_write_relative=0.0,
        direct_formula_k32_relative=0.0,
        direct_formula_k64_relative=0.0,
        k32_k64_response_relative=0.0,
        k32_k64_ovir_absolute=0.0,
        gram_direct_svd_span_relative=0.0,
        basis_orthogonality_error=0.0,
        basis_energy_identity_error=0.0,
        operator_energy_identity_error=0.0,
    )
    ranks = []
    done = 0
    rule32 = quadrature_rule("gauss_legendre", 32, device=torch.device(device), dtype=torch.float32)
    rule64 = quadrature_rule("gauss_legendre", 64, device=torch.device(device), dtype=torch.float32)
    for image_id, selected in sorted(by_image.items()):
        parent = read(parents[image_id])
        targets = sorted(parent["positions"], key=lambda row: row["response_index"])
        with Image.open(_image_path(config, image_id)) as source:
            image = source.convert("RGB")
        captures, queries, start, end, _ = capture_full(
            wrapper,
            model,
            image,
            generations[image_id]["response_token_ids"],
            targets,
            config.get("run", {}).get("prompt") or "Describe this image.",
        )
        for case_index, case in selected:
            target_index = next(i for i, row in enumerate(targets) if row["target_key"] == case["target_key"])
            layer_index = int(case["layer"])
            layer, capture = layers[layer_index], captures[layer_index]
            query = queries[target_index]
            future = attention_row_from_capture(capture, query)[..., query + 1:]
            if future.count_nonzero():
                raise ValueError("Noncausal attention in OVIR audit")
            direct = reconstruct_visual_directions(
                layer=layer,
                capture=capture,
                prediction_positions=[query],
                visual_start=start,
                visual_end=end,
                return_all_sources=True,
            )
            all_writes = direct["all_token_writes"].float()
            masks = partition(len(all_writes), query, targets[target_index]["response_index"], (start, end), all_writes.device)
            causal = masks.any(0)
            a = all_writes[masks[1], 0].T.contiguous()
            total = all_writes[causal].sum(0)
            z = capture["h_mid"][0, query:query + 1].float()
            basis, basis_audit, factors = visual_write_energy_basis(a, energy_threshold=ENERGY_THRESHOLD)
            ranks.append(basis_audit["rank95"])
            adapter = resolve_decoder_layer_adapter(layer)
            with local_fp32(layer), torch.no_grad():
                operator32 = path_operator(adapter.ffn_norm, adapter.ffn, z - total, total, rule32.nodes, rule32.weights)
                operator64 = path_operator(adapter.ffn_norm, adapter.ffn, z - total, total, rule64.nodes, rule64.weights)
                y32 = _apply_basis(operator32, basis)
                y64 = _apply_basis(operator64, basis)

            saved = torch.load(_audit_artifact(model, case), map_location="cpu", weights_only=False, mmap=True)
            old32, old64 = saved["results"][32], saved["results"][64]
            visual32 = old32["source_type"] == 1
            visual64 = old64["source_type"] == 1
            if not torch.equal(visual32, visual64):
                raise ValueError("K32/K64 visual source masks differ")
            old_a = old32["vectors"]["writes"][visual32, 0].T.to(a)
            write_error = _relative(a, old_a)
            e32 = old32["vectors"]["all"][visual32, 0].T.to(a.device)
            e64 = old64["vectors"]["all"][visual64, 0].T.to(a.device)
            formula32 = response_from_token_matrix(e32, factors)
            formula64 = response_from_token_matrix(e64, factors)
            metric32 = operator_visual_write_innovation(basis, y32)
            metric64 = operator_visual_write_innovation(basis, y64)
            values = dict(
                live_saved_write_relative=write_error,
                direct_formula_k32_relative=_relative(y32, formula32),
                direct_formula_k64_relative=_relative(y64, formula64),
                k32_k64_response_relative=_relative(y32, y64),
                k32_k64_ovir_absolute=abs(metric32["ovir"] - metric64["ovir"]),
                basis_orthogonality_error=basis_audit["basis_orthogonality_error"],
                basis_energy_identity_error=basis_audit["basis_energy_identity_error"],
                operator_energy_identity_error=max(
                    metric32["operator_energy_identity_error"],
                    metric64["operator_energy_identity_error"],
                ),
            )
            if case_index in (0, 2, 4):
                values["gram_direct_svd_span_relative"] = _direct_svd_span_error(
                    a, basis, basis_audit["rank95"]
                )
            for name, value in values.items():
                maxima[name] = max(maxima[name], float(value))
            done += 1
            progress.update("OVIR数值审计", done, len(cases), current_image=image_id, layer=layer_index + 1)
            del saved, direct, all_writes, a, total, z, basis, factors, y32, y64, formula32, formula64
        del captures
    del wrapper
    gc.collect()
    torch.cuda.empty_cache()
    tolerances = dict(
        live_saved_write_relative=5e-4,
        direct_formula_k32_relative=5e-4,
        direct_formula_k64_relative=5e-4,
        k32_k64_response_relative=1e-2,
        k32_k64_ovir_absolute=1e-3,
        gram_direct_svd_span_relative=1e-8,
        basis_orthogonality_error=1e-9,
        basis_energy_identity_error=1e-9,
        operator_energy_identity_error=1e-9,
    )
    passed = all(maxima[name] <= limit for name, limit in tolerances.items())
    return dict(
        model=model,
        cases=len(cases),
        complete=True,
        passed=passed,
        maxima=maxima,
        tolerances=tolerances,
        rank95_min=min(ranks),
        rank95_max=max(ranks),
    )


def audit(model: str, device: str, out: Path, progress: Progress) -> dict:
    path = out / model / "audit.json"
    if path.exists():
        value = json.loads(path.read_text())
        if value.get("complete") and value.get("passed"):
            return value
    try:
        value = _run_audit(model, device, out, progress)
        atomic_json_save(value, path)
        if not value["passed"]:
            raise RuntimeError(f"OVIR audit failed for {model}: {value['maxima']}")
        return value
    except Exception as error:
        atomic_json_save(
            dict(model=model, complete=True, passed=False, error=repr(error), traceback=traceback.format_exc()),
            path,
        )
        raise


def validate_all_audits(out: Path) -> dict:
    values = {}
    for model in MODELS:
        path = out / model / "audit.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing OVIR audit: {model}")
        values[model] = json.loads(path.read_text())
    if not all(value.get("complete") and value.get("passed") for value in values.values()):
        raise RuntimeError("At least one OVIR model audit failed; formal extraction is blocked")
    result = dict(complete=True, passed=True, models=values)
    atomic_json_save(result, out / "audit_summary.json")
    return result


def wait_for_audits(out: Path, progress: Progress):
    while True:
        present = sum((out / model / "audit.json").exists() for model in MODELS)
        progress.update("等待四模型审计", present, len(MODELS))
        if present == len(MODELS):
            validate_all_audits(out)
            return
        time.sleep(10)


def valid_shard(path: Path, signature: str, layers: int) -> bool:
    if not path.exists():
        return False
    try:
        value = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except (EOFError, OSError, RuntimeError):
        return False
    if not value.get("complete") or value.get("protocol_signature") != signature:
        return False
    return all(
        set(row.get("metrics", {})) == set(METRICS)
        and all(len(values) == layers for values in row["metrics"].values())
        for row in value.get("positions", [])
    )


def pending_images(ids, completed, parity=None):
    if parity not in (None, 0, 1):
        raise ValueError("Image parity must be 0, 1, or None")
    return [image_id for image_id in ids if image_id not in completed and (parity is None or image_id % 2 == parity)]


def _metric_record(audit: dict, operator: dict) -> dict[str, float]:
    combined = {**audit, **operator}
    return {name: float(combined[name]) for name in METRICS}


def extract(model: str, device: str, out: Path, p: dict, progress: Progress,
            image_parity=None, smoke: bool = False):
    ensure_source_snapshot(model, out)
    split = fixed_split(model)
    ids = sorted(set().union(*(set(split[name]) for name in split)))
    if smoke:
        ids = [283]
        image_parity = None
    folder = out / model / ("smoke" if smoke else "shards")
    folder.mkdir(parents=True, exist_ok=True)
    complete = {
        image_id for image_id in ids
        if valid_shard(folder / f"image_{image_id:012d}.pt", p["signature"], LAYERS[model])
    }
    pending = pending_images(ids, complete, image_parity)
    stage = "单图全层smoke" if smoke else "4000图OVIR提取"
    progress.update(stage, len(complete), len(ids),
                    **({} if smoke else dict(images_completed=len(complete), images_total=len(ids))))
    if not pending:
        return
    parents = parent_paths(model)
    _, generations, _ = _load_inputs(ROOT / "outputs" / model / EXPERIMENT)
    wrapper, config = load_wrapper(model, device)
    layers = resolve_decoder_layers(wrapper.model)
    rule = quadrature_rule("gauss_legendre", QUADRATURE_POINTS, device=torch.device(device), dtype=torch.float32)
    for offset, image_id in enumerate(pending, 1):
        tick = time.monotonic()
        parent = read(parents[image_id])
        targets = sorted(parent["positions"], key=lambda row: row["response_index"])
        rows = [dict(
            target_key=target["target_key"],
            response_index=target["response_index"],
            target_token_id=target["target_token_id"],
            metrics=defaultdict(list),
        ) for target in targets]
        if targets:
            with Image.open(_image_path(config, image_id)) as source:
                image = source.convert("RGB")
            captures, queries, start, end, _ = capture_full(
                wrapper,
                model,
                image,
                generations[image_id]["response_token_ids"],
                targets,
                config.get("run", {}).get("prompt") or "Describe this image.",
            )
            for layer_index, (layer, capture) in enumerate(zip(layers, captures)):
                for query in queries:
                    future = attention_row_from_capture(capture, query)[..., query + 1:]
                    if future.count_nonzero():
                        raise ValueError("Noncausal attention in OVIR extraction")
                direct = reconstruct_visual_directions(
                    layer=layer,
                    capture=capture,
                    prediction_positions=queries,
                    visual_start=start,
                    visual_end=end,
                    return_all_sources=True,
                )
                all_writes = direct["all_token_writes"].float()
                z_all = capture["h_mid"][0, queries].float()
                adapter = resolve_decoder_layer_adapter(layer)
                for target_index, (row, target, query) in enumerate(zip(rows, targets, queries)):
                    writes = all_writes[:, target_index:target_index + 1]
                    masks = partition(len(writes), query, target["response_index"], (start, end), writes.device)
                    causal = masks.any(0)
                    a = writes[masks[1], 0].T.contiguous()
                    total = writes[causal].sum(0)
                    z = z_all[target_index:target_index + 1]
                    basis, basis_audit, _ = visual_write_energy_basis(a, energy_threshold=ENERGY_THRESHOLD)
                    if basis.shape[1]:
                        with local_fp32(layer), torch.no_grad():
                            operator = path_operator(
                                adapter.ffn_norm,
                                adapter.ffn,
                                z - total,
                                total,
                                rule.nodes,
                                rule.weights,
                            )
                            response = _apply_basis(operator, basis)
                        operator_audit = operator_visual_write_innovation(basis, response)
                        del response, operator
                    else:
                        operator_audit = operator_visual_write_innovation(basis, basis)
                    record = _metric_record(basis_audit, operator_audit)
                    for name, value in record.items():
                        row["metrics"][name].append(value)
                    progress.update(
                        stage,
                        len(complete) + offset - 1,
                        len(ids),
                        **({} if smoke else dict(images_completed=len(complete) + offset - 1,
                                                images_total=len(ids))),
                        current_image=image_id,
                        layer=layer_index + 1,
                        target=target_index + 1,
                    )
                    del writes, a, total, z, basis
                captures[layer_index] = None
                del direct, all_writes, z_all
            del captures
        for row in rows:
            row["metrics"] = {
                name: torch.tensor(values, dtype=torch.float64)
                for name, values in row["metrics"].items()
            }
        atomic_torch_save(
            dict(
                image_id=image_id,
                positions=rows,
                sample_table=parent["sample_table"],
                complete=True,
                protocol_signature=p["signature"],
                elapsed=time.monotonic() - tick,
            ),
            folder / f"image_{image_id:012d}.pt",
        )
        done = len(complete) + offset
        progress.update(stage, done, len(ids), current_image=image_id, layer=0, target=0,
                        **({} if smoke else dict(images_completed=done, images_total=len(ids))))
        print(model, done, "/", len(ids), "image", image_id, "seconds", round(time.monotonic() - tick, 2), flush=True)
    del wrapper
    gc.collect()
    torch.cuda.empty_cache()
    ensure_source_snapshot(model, out)


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing empty CSV: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _bootstrap_deltas(deltas: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    count = len(deltas)
    if count == 0:
        return np.full(deltas.shape[1], np.nan), np.full(deltas.shape[1], np.nan)
    samples = np.empty((BOOTSTRAP_REPLICATES, deltas.shape[1]), dtype=np.float64)
    for begin in range(0, BOOTSTRAP_REPLICATES, 100):
        size = min(100, BOOTSTRAP_REPLICATES - begin)
        indices = rng.integers(0, count, size=(size, count))
        samples[begin:begin + size] = np.nanmean(deltas[indices], axis=1)
    return np.nanquantile(samples, 0.025, axis=0), np.nanquantile(samples, 0.975, axis=0)


def analyze(model: str, out: Path, p: dict, progress: Progress | None = None) -> dict:
    ensure_source_snapshot(model, out)
    source = torch.load(MATRIX_BASE / model / "matrices.pt", map_location="cpu", weights_only=False)
    mentions = source["mentions"]
    labels = np.asarray(source["y"])
    image_ids = np.asarray([row["image_id"] for row in mentions])
    masks = {name: np.asarray(mask, dtype=bool) for name, mask in source["masks"].items()}
    split = fixed_split(model)
    for name in split:
        np.testing.assert_array_equal(masks[name], np.isin(image_ids, split[name]))
    by_image = defaultdict(list)
    for mention in mentions:
        by_image[mention["image_id"]].append(mention)
    targets = {}
    folder = out / model / "shards"
    audit = dict(
        images=0,
        unique_targets=0,
        target_layers=0,
        mentions=len(mentions),
        conflicting_targets=0,
        undefined_ovir=0,
        rank95_min=10**9,
        rank95_max=0,
        retained_energy_min=1.0,
        retained_energy_max=0.0,
        max_basis_orthogonality_error=0.0,
        max_basis_energy_identity_error=0.0,
        max_operator_energy_identity_error=0.0,
    )
    expected_ids = sorted(set().union(*(set(split[name]) for name in split)))
    if len(expected_ids) != 4000:
        raise ValueError(f"Expected a 4000-image fixed split for {model}")
    for index, image_id in enumerate(expected_ids, 1):
        path = folder / f"image_{image_id:012d}.pt"
        if not valid_shard(path, p["signature"], LAYERS[model]):
            raise ValueError(f"Invalid OVIR shard: {path}")
        shard = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        if Counter(map(mention_identity, shard["sample_table"])) != Counter(map(mention_identity, by_image[image_id])):
            raise ValueError(f"Mention identity mismatch for {model} image {image_id}")
        if {row["target_key"] for row in shard["positions"]} != {row["target_key"] for row in by_image[image_id]}:
            raise ValueError(f"Target identity mismatch for {model} image {image_id}")
        for row in shard["positions"]:
            key = row["target_key"]
            if key in targets:
                raise ValueError(f"Duplicate OVIR target {model} {key}")
            values = {name: row["metrics"][name].double().numpy() for name in METRICS}
            if any(value.shape != (LAYERS[model],) for value in values.values()):
                raise ValueError(f"Wrong layer count for {model} {key}")
            finite = np.isfinite(values["ovir"])
            if ((values["ovir"][finite] < -1e-12) | (values["ovir"][finite] > 1 + 1e-12)).any():
                raise ValueError(f"OVIR outside [0,1] for {model} {key}")
            nonzero = values["zero_input"] == 0
            if (values["retained_energy"][nonzero] < ENERGY_THRESHOLD - 1e-10).any():
                raise ValueError(f"Energy threshold failure for {model} {key}")
            minimal = nonzero & (values["rank95"] > 1)
            if (values["previous_retained_energy"][minimal] >= ENERGY_THRESHOLD + 1e-10).any():
                raise ValueError(f"Nonminimal rank95 for {model} {key}")
            audit["undefined_ovir"] += int((~finite).sum())
            audit["rank95_min"] = min(audit["rank95_min"], int(values["rank95"][nonzero].min(initial=10**9)))
            audit["rank95_max"] = max(audit["rank95_max"], int(values["rank95"].max(initial=0)))
            audit["retained_energy_min"] = min(audit["retained_energy_min"], float(values["retained_energy"][nonzero].min(initial=1)))
            audit["retained_energy_max"] = max(audit["retained_energy_max"], float(values["retained_energy"][nonzero].max(initial=0)))
            for name in ("basis_orthogonality_error", "basis_energy_identity_error", "operator_energy_identity_error"):
                audit["max_" + name] = max(audit["max_" + name], float(np.nanmax(values[name], initial=0)))
            targets[key] = values
            audit["target_layers"] += LAYERS[model]
        audit["images"] = index
        if progress and index % 100 == 0:
            progress.update("结果重载验证", index, 4000)
    if len(targets) != EXPECTED_TARGETS[model]:
        raise ValueError(f"Wrong target count for {model}: {len(targets)}")
    audit["unique_targets"] = len(targets)
    label_sets = defaultdict(set)
    for mention in mentions:
        label_sets[mention["target_key"]].add(mention["label"])
    audit["conflicting_targets"] = sum(len(value) > 1 for value in label_sets.values())
    clean = np.asarray([len(label_sets[row["target_key"]]) == 1 for row in mentions])
    arrays = {
        name: np.stack([targets[row["target_key"]][name] for row in mentions])
        for name in METRICS
    }
    del targets
    data = dict(
        values={name: value.astype(np.float32) for name, value in arrays.items()},
        mentions=mentions,
        y=labels,
        masks=masks,
        clean=clean,
        protocol_signature=p["signature"],
        audit=audit,
    )
    atomic_torch_save(data, out / model / "matrices.pt")

    curves, paired, bootstrap_rows = [], [], []
    bootstrap_json = dict(model=model, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED, scopes={})
    scopes = dict(all=np.ones(len(mentions), dtype=bool), **masks)
    for scope, scope_mask in scopes.items():
        selected_scope = scope_mask & clean
        for metric in CURVE_METRICS:
            matrix = arrays[metric]
            for label, label_name in ((1, "REAL"), (0, "HALL")):
                selected = matrix[selected_scope & (labels == label)]
                for layer, column in enumerate(selected.T, 1):
                    valid = column[np.isfinite(column)]
                    quartiles = np.quantile(valid, (0.25, 0.5, 0.75)) if len(valid) else (np.nan,) * 3
                    curves.append(dict(
                        model=model, scope=scope, metric=metric, label=label_name, layer=layer,
                        total=len(column), valid=len(valid), undefined=len(column) - len(valid),
                        mean=float(valid.mean()) if len(valid) else np.nan,
                        q25=quartiles[0], median=quartiles[1], q75=quartiles[2],
                    ))
        ovir = arrays["ovir"]
        mixed = sorted(
            set(image_ids[selected_scope & (labels == 0)])
            & set(image_ids[selected_scope & (labels == 1)])
        )
        deltas = np.asarray([
            np.nanmean(ovir[selected_scope & (image_ids == image_id) & (labels == 0)], axis=0)
            - np.nanmean(ovir[selected_scope & (image_ids == image_id) & (labels == 1)], axis=0)
            for image_id in mixed
        ])
        if len(deltas):
            lower, upper = _bootstrap_deltas(deltas, BOOTSTRAP_SEED + list(scopes).index(scope))
            point = np.nanmean(deltas, axis=0)
            for layer, column in enumerate(deltas.T, 1):
                valid = column[np.isfinite(column)]
                row = dict(
                    model=model, scope=scope, metric="ovir", layer=layer,
                    mixed_images=len(column), valid=len(valid),
                    mean_H_minus_R=float(point[layer - 1]),
                    median_H_minus_R=float(np.nanmedian(column)),
                    fraction_H_gt_R=float(np.mean(valid > 0)) if len(valid) else np.nan,
                    ci95_low=float(lower[layer - 1]), ci95_high=float(upper[layer - 1]),
                )
                paired.append(row)
                bootstrap_rows.append(row.copy())
            image_layer_means = np.nanmean(deltas, axis=1, keepdims=True)
            cross_lower, cross_upper = _bootstrap_deltas(
                image_layer_means,
                BOOTSTRAP_SEED + 100 + list(scopes).index(scope),
            )
            bootstrap_json["scopes"][scope] = dict(
                mixed_images=len(mixed),
                layer_point=point.tolist(),
                layer_ci95_low=lower.tolist(),
                layer_ci95_high=upper.tolist(),
                across_layer_point=float(np.nanmean(image_layer_means)),
                across_layer_ci95=[float(cross_lower[0]), float(cross_upper[0])],
            )
    write_csv(out / model / "curves.csv", curves)
    write_csv(out / model / "paired_images.csv", paired)
    write_csv(out / model / "bootstrap.csv", bootstrap_rows)
    atomic_json_save(bootstrap_json, out / model / "bootstrap.json")
    atomic_json_save(audit, out / model / "validation.json")
    _plot_model(model, curves, paired, out / model)
    return dict(model=model, audit=audit, bootstrap=bootstrap_json)


def _plot_model(model: str, curves: list[dict], paired: list[dict], directory: Path):
    for scope in ("all", "test"):
        fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
        ax = axes[0]
        for label, color in (("REAL", "#2878b5"), ("HALL", "#d95319")):
            rows = [row for row in curves if row["scope"] == scope and row["metric"] == "ovir" and row["label"] == label]
            x = [row["layer"] for row in rows]
            ax.plot(x, [row["mean"] for row in rows], color=color, label=label + " mean")
            ax.plot(x, [row["median"] for row in rows], color=color, linestyle="--", alpha=0.8, label=label + " median")
            ax.fill_between(x, [row["q25"] for row in rows], [row["q75"] for row in rows], color=color, alpha=0.12)
        ax.set_ylabel("OVIR")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        delta = [row for row in paired if row["scope"] == scope]
        x = [row["layer"] for row in delta]
        axes[1].axhline(0, color="black", linewidth=0.8)
        axes[1].plot(x, [row["mean_H_minus_R"] for row in delta], color="#6a3d9a")
        axes[1].fill_between(x, [row["ci95_low"] for row in delta], [row["ci95_high"] for row in delta], color="#6a3d9a", alpha=0.18)
        axes[1].set_ylabel("HALL − REAL OVIR")
        axes[1].set_xlabel("Decoder layer (1-based)")
        axes[1].grid(alpha=0.2)
        fig.suptitle(f"{model} | {scope} | raw-WRITE 95% subspace | All-attention K32")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        for suffix in ("png", "pdf"):
            fig.savefig(directory / f"ovir_{scope}.{suffix}", dpi=160)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for index, metric in enumerate(("rank95", "retained_energy")):
        ax = axes[index]
        for label, color in (("REAL", "#2878b5"), ("HALL", "#d95319")):
            rows = [row for row in curves if row["scope"] == "all" and row["metric"] == metric and row["label"] == label]
            ax.plot([row["layer"] for row in rows], [row["mean"] for row in rows], color=color, label=label)
        ax.set_title(metric)
        ax.set_xlabel("Decoder layer (1-based)")
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle(model + " | 95% visual-WRITE basis diagnostics")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for suffix in ("png", "pdf"):
        fig.savefig(directory / f"rank_energy.{suffix}", dpi=160)
    plt.close(fig)


def summarize(out: Path, p: dict, wait: bool = False) -> dict:
    while wait and not all((out / model / "validation.json").exists() for model in MODELS):
        time.sleep(10)
    validate_all_audits(out)
    curves, paired, models = [], [], {}
    for model in MODELS:
        validation = json.loads((out / model / "validation.json").read_text())
        bootstrap = json.loads((out / model / "bootstrap.json").read_text())
        if validation["images"] != 4000 or validation["unique_targets"] != EXPECTED_TARGETS[model]:
            raise ValueError(f"Incomplete OVIR validation for {model}")
        with (out / model / "curves.csv").open() as handle:
            curves.extend(csv.DictReader(handle))
        with (out / model / "paired_images.csv").open() as handle:
            paired.extend(csv.DictReader(handle))
        all_scope = bootstrap["scopes"]["all"]
        test_scope = bootstrap["scopes"]["test"]
        model_curves = [row for row in curves if row["model"] == model and row["scope"] == "all" and row["metric"] == "ovir"]
        real = np.asarray([float(row["mean"]) for row in model_curves if row["label"] == "REAL"])
        hall = np.asarray([float(row["mean"]) for row in model_curves if row["label"] == "HALL"])
        models[model] = dict(
            images=validation["images"],
            unique_targets=validation["unique_targets"],
            mentions=validation["mentions"],
            target_layers=validation["target_layers"],
            conflicting_targets=validation["conflicting_targets"],
            undefined_ovir=validation["undefined_ovir"],
            rank95=[validation["rank95_min"], validation["rank95_max"]],
            all_real_layer_mean=float(real.mean()),
            all_hall_layer_mean=float(hall.mean()),
            all_hall_minus_real=float(hall.mean() - real.mean()),
            all_hall_higher_layers=int((hall > real).sum()),
            all_paired_across_layer=all_scope,
            test_paired_across_layer=test_scope,
        )
    summary = dict(
        protocol_signature=p["signature"],
        complete=True,
        models=models,
        totals=dict(
            images=sum(value["images"] for value in models.values()),
            unique_targets=sum(value["unique_targets"] for value in models.values()),
            mentions=sum(value["mentions"] for value in models.values()),
            target_layers=sum(value["target_layers"] for value in models.values()),
        ),
    )
    if summary["totals"] != dict(images=16000, unique_targets=49939, mentions=50812, target_layers=1622248):
        raise ValueError(f"Unexpected OVIR totals: {summary['totals']}")
    atomic_json_save(summary, out / "summary.json")
    write_csv(out / "curves.csv", curves)
    write_csv(out / "paired_images.csv", paired)
    _plot_all_models(curves, paired, out)
    return summary


def _plot_all_models(curves: list[dict], paired: list[dict], out: Path):
    fig, axes = plt.subplots(4, 2, figsize=(13, 14), squeeze=False)
    for index, model in enumerate(MODELS):
        for label, color in (("REAL", "#2878b5"), ("HALL", "#d95319")):
            rows = [row for row in curves if row["model"] == model and row["scope"] == "all"
                    and row["metric"] == "ovir" and row["label"] == label]
            x = [int(row["layer"]) for row in rows]
            axes[index, 0].plot(x, [float(row["mean"]) for row in rows], color=color, label=label)
        delta = [row for row in paired if row["model"] == model and row["scope"] == "all"]
        x = [int(row["layer"]) for row in delta]
        axes[index, 1].axhline(0, color="black", linewidth=0.8)
        axes[index, 1].plot(x, [float(row["mean_H_minus_R"]) for row in delta], color="#6a3d9a")
        axes[index, 1].fill_between(
            x,
            [float(row["ci95_low"]) for row in delta],
            [float(row["ci95_high"]) for row in delta],
            color="#6a3d9a",
            alpha=0.18,
        )
        axes[index, 0].set_title(model + " | OVIR")
        axes[index, 1].set_title(model + " | paired HALL − REAL")
        for ax in axes[index]:
            ax.set_xlabel("Decoder layer")
            ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("COCO4000 | raw-WRITE 95% subspace | All-attention K32")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"all_models_ovir.{suffix}", dpi=160)
    plt.close(fig)


def _progress_bar(completed: int, total: int, width: int = 28) -> str:
    fraction = min(max(completed / total if total else 0.0, 0.0), 1.0)
    filled = int(round(width * fraction))
    return "█" * filled + "░" * (width - filled)


def watch(out: Path):
    target = ROOT / "OVIR_EXPERIMENT_PROGRESS.md"
    locations = {
        "qwen2_5_vl_7b": "本机 GPU0",
        "llava_1_5_7b": "本机 GPU1",
        "qwen3_vl_8b": "32678 GPU0",
        "internvl_2_5_8b": "32678 GPU1",
    }
    while True:
        rows, terminal = [], True
        for model in MODELS:
            path = out / model / "progress.json"
            value = json.loads(path.read_text()) if path.exists() else dict(
                model=model, stage="等待启动", completed=0, total=1, status="pending", heartbeat=""
            )
            terminal &= value.get("status") in ("completed", "failed")
            rows.append(value)
        completed = sum(int(value.get("images_completed", 0)) for value in rows)
        total = 4000 * len(MODELS)
        lines = ["# OVIR COCO4000 实验进度", "", f"更新时间：{utc()}", "",
                 f"总进度：`{_progress_bar(completed, total)}` {completed}/{total} 张模型-图片",
                 "", "| 模型 | 运行位置 | 图片进度 | 当前阶段 | 状态 |", "|---|---|---:|---|---|"]
        for value in rows:
            images = int(value.get("images_completed", 0))
            model = value["model"]
            lines.append(f"| {model} | {locations[model]} | {images}/4000 | "
                         f"{value.get('stage')} | {value.get('status')} |")
        lines.extend(["", "进度文件由 watcher 每 10 秒原子覆写；详细心跳保存在各模型 `progress.json`。"])
        temporary = target.with_suffix(target.suffix + f".tmp.{os.getpid()}")
        temporary.write_text("\n".join(lines) + "\n")
        os.replace(temporary, target)
        if terminal:
            return
        time.sleep(10)


def pipeline(model: str, device: str, out: Path, p: dict, progress: Progress):
    audit(model, device, out, progress)
    extract(model, device, out, p, progress, smoke=True)
    wait_for_audits(out, progress)
    extract(model, device, out, p, progress)
    progress.update("结果分析", 0, 1)
    analyze(model, out, p, progress)
    progress.update("全部完成", 1, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True,
                        choices=("audit", "select", "smoke", "extract", "analyze", "summarize", "pipeline", "watch"))
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--image-parity", type=int, choices=(0, 1))
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    p = protocol(args.output)
    if args.stage == "select":
        validate_all_audits(args.output)
        return
    if args.stage == "summarize":
        summarize(args.output, p, args.wait)
        return
    if args.stage == "watch":
        watch(args.output)
        return
    if not args.model:
        parser.error("--model is required")
    with model_lock(args.model, args.output):
        progress = Progress(args.model, args.output)
        try:
            if args.stage == "audit":
                audit(args.model, args.device, args.output, progress)
            elif args.stage == "smoke":
                audit(args.model, args.device, args.output, progress)
                extract(args.model, args.device, args.output, p, progress, smoke=True)
            elif args.stage == "extract":
                validate_all_audits(args.output)
                extract(args.model, args.device, args.output, p, progress, args.image_parity)
            elif args.stage == "analyze":
                analyze(args.model, args.output, p, progress)
            else:
                pipeline(args.model, args.device, args.output, p, progress)
            progress.finish("completed")
        except Exception as error:
            progress.finish("failed", traceback.format_exc())
            raise


if __name__ == "__main__":
    main()
