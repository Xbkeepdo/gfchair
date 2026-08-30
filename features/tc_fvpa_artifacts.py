"""Auditable, resumable artifacts for the TC-FVPA experiment family."""

from __future__ import annotations

import gzip
import fcntl
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch


SCHEMA_VERSION = "tc-fvpa-comprehensive-v1"
DIRECTORIES = (
    "manifests",
    "configs",
    "logs",
    "schemas",
    "shards",
    "audit_vectors",
    "tables",
    "metrics",
    "figures",
    "reports",
    "handoff_bundle",
)
VALID_MEASUREMENT_STATUS = {"MEASURED", "FAILED", "BLOCKED", "NOT_RUN"}
VALID_CASE_COHORTS = {"path", "local"}


def _cohort_stem(cohort: str, canonical: str) -> str:
    value = str(cohort)
    if value not in VALID_CASE_COHORTS:
        raise ValueError(
            f"Invalid case cohort {value!r}; expected one of {sorted(VALID_CASE_COHORTS)}"
        )
    return canonical if value == "path" else f"local_{canonical}"


@dataclass(frozen=True)
class ExperimentLayout:
    root: Path

    @classmethod
    def create(cls, root: str | Path) -> "ExperimentLayout":
        resolved = Path(root).resolve()
        for name in DIRECTORIES:
            (resolved / name).mkdir(parents=True, exist_ok=True)
        return cls(resolved)

    def path(self, relative: str) -> Path:
        return self.root / relative


def atomic_json_save(payload: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def atomic_torch_save(payload: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def sha256_file(path: str | Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(int(chunk_bytes))
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _command(command: Sequence[str], cwd: Path) -> dict[str, Any]:
    started = time.time()
    try:
        completed = subprocess.run(
            list(command),
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        return {
            "command": list(command),
            "exit_code": int(completed.returncode),
            "output": completed.stdout,
            "elapsed_seconds": time.time() - started,
        }
    except Exception as exc:
        return {
            "command": list(command),
            "exit_code": None,
            "output": "",
            "error": repr(exc),
            "elapsed_seconds": time.time() - started,
        }


def git_snapshot(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    status = _command(("git", "status", "--short", "--branch"), root)
    revision = _command(("git", "rev-parse", "HEAD"), root)
    diff = _command(("git", "diff", "--binary"), root)
    diff_stat = _command(("git", "diff", "--stat"), root)
    return {
        "status": status,
        "revision": revision,
        "diff_stat": diff_stat,
        "diff_sha256": sha256_text(diff.get("output", "")),
    }


def environment_snapshot(repo_root: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Capture the executing Python environment and visible accelerator state."""
    root = Path(repo_root).resolve()
    freeze = _command((sys.executable, "-m", "pip", "freeze"), root)
    nvidia = _command(("nvidia-smi",), root)
    torch_info = {
        "version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cudnn_version": torch.backends.cudnn.version(),
    }
    hardware = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "nvidia_smi": nvidia,
        "torch_devices": [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "total_memory": torch.cuda.get_device_properties(index).total_memory,
                "capability": list(torch.cuda.get_device_capability(index)),
            }
            for index in range(torch.cuda.device_count())
        ],
    }
    environment = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "pip_freeze": freeze,
        "torch": torch_info,
        "argv": list(sys.argv),
        "cwd": str(Path.cwd()),
        "environment_keys": sorted(
            key
            for key in os.environ
            if not any(term in key.upper() for term in ("TOKEN", "SECRET", "PASSWORD", "KEY"))
        ),
    }
    return environment, hardware


def checksums_for_paths(paths: Iterable[str | Path]) -> dict[str, dict[str, Any]]:
    result = {}
    for raw_path in sorted({str(Path(path).resolve()) for path in paths}):
        path = Path(raw_path)
        if not path.is_file():
            result[raw_path] = {"exists": False}
            continue
        result[raw_path] = {
            "exists": True,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return result


def initialize_manifests(
    *,
    layout: ExperimentLayout,
    repo_root: str | Path,
    experiment_config: Mapping[str, Any],
    input_paths: Sequence[str | Path],
    exact_command: Sequence[str],
) -> None:
    lock_path = layout.path("manifests/.initialize.lock")
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        manifest_path = layout.path("manifests/experiment_manifest.json")
        if not manifest_path.exists():
            environment, hardware = environment_snapshot(repo_root)
            git = git_snapshot(repo_root)
            config = dict(experiment_config)
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "created_unix": time.time(),
                "repository": git,
                "exact_command": list(exact_command),
                "config": config,
            }
            atomic_json_save(manifest, manifest_path)
            atomic_json_save(environment, layout.path("manifests/environment.json"))
            atomic_json_save(hardware, layout.path("manifests/hardware.json"))
            atomic_json_save(
                checksums_for_paths(input_paths),
                layout.path("manifests/input_checksums.json"),
            )
            for raw_path in input_paths:
                source = Path(raw_path).resolve()
                if source.is_file() and source.suffix.lower() in {
                    ".yaml",
                    ".yml",
                    ".json",
                } and "config" in source.name.lower():
                    shutil.copy2(source, layout.path(f"configs/{source.name}"))
        run_status_path = layout.path("manifests/run_status.json")
        if not run_status_path.exists():
            atomic_json_save(
                {
                    "schema_version": SCHEMA_VERSION,
                    "created_unix": time.time(),
                    "updated_unix": time.time(),
                    "stages": {},
                    "completed_shards": [],
                    "failed_shards": [],
                    "failed_cases": [],
                    "resume_commands": [],
                },
                run_status_path,
            )
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def load_run_status(layout: ExperimentLayout) -> dict[str, Any]:
    path = layout.path("manifests/run_status.json")
    if not path.exists():
        raise FileNotFoundError(f"Run status has not been initialized: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def update_stage_status(
    *,
    layout: ExperimentLayout,
    stage: str,
    status: str,
    details: Mapping[str, Any] | None = None,
    resume_command: str | None = None,
) -> None:
    if status not in {"RUNNING", "PASS", "FAIL", "BLOCKED", "NOT_RUN"}:
        raise ValueError(f"Invalid stage status {status!r}")
    lock_path = layout.path("manifests/.run_status.lock")
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        payload = load_run_status(layout)
        now = time.time()
        previous = dict(payload.get("stages", {}).get(stage, {}))
        started = previous.get("started_unix", now)
        payload.setdefault("stages", {})[stage] = {
            "status": status,
            "started_unix": started,
            "updated_unix": now,
            "details": dict(details or {}),
        }
        if resume_command:
            commands = payload.setdefault("resume_commands", [])
            if resume_command not in commands:
                commands.append(resume_command)
        payload["updated_unix"] = now
        atomic_json_save(payload, layout.path("manifests/run_status.json"))
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def case_key(row: Mapping[str, Any]) -> str:
    required = ("model", "image_id", "response_index", "layer")
    missing = [name for name in required if name not in row]
    if missing:
        raise KeyError(f"Case row misses uniqueness fields {missing}")
    return ":".join(str(row[name]) for name in required)


def validate_measurement_row(row: Mapping[str, Any]) -> None:
    """Reject ambiguous rows and non-finite measured values."""
    status = str(row.get("measurement_status", ""))
    if status not in VALID_MEASUREMENT_STATUS:
        raise ValueError(
            "Every case needs measurement_status in "
            f"{sorted(VALID_MEASUREMENT_STATUS)}, got {status!r}"
        )
    if status == "MEASURED":
        if row.get("error"):
            raise ValueError("A measured row cannot also carry an error")
        measured_fields = row.get("measured_fields")
        if not measured_fields:
            raise ValueError("Measured rows must enumerate measured_fields")
        for name in measured_fields:
            if name not in row:
                raise KeyError(f"Measured field {name!r} is absent")
            value = row[name]
            tensor = torch.as_tensor(value) if isinstance(value, (float, int, torch.Tensor)) else None
            if tensor is not None and tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"Measured field {name!r} is non-finite")
    elif not row.get("error") and status in {"FAILED", "BLOCKED"}:
        raise ValueError(f"{status} rows must persist a non-empty error")


def write_case_shard(
    *,
    layout: ExperimentLayout,
    rank: int,
    shard_id: int,
    rows: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
    cohort: str = "path",
) -> Path:
    seen = set()
    normalized_rows = []
    for source in rows:
        row = dict(source)
        validate_measurement_row(row)
        key = case_key(row)
        if key in seen:
            raise ValueError(f"Duplicate case inside shard: {key}")
        seen.add(key)
        normalized_rows.append(row)
    stem = _cohort_stem(cohort, "case")
    path = layout.path(
        f"shards/{stem}_rank{int(rank):02d}_shard_{int(shard_id):05d}.pt"
    )
    if path.exists():
        raise FileExistsError(f"Atomic shard already exists: {path}")
    atomic_torch_save(
        {
            "schema_version": SCHEMA_VERSION,
            "rank": int(rank),
            "shard_id": int(shard_id),
            "cohort": str(cohort),
            "rows": normalized_rows,
            "failures": [dict(row) for row in failures],
            "provenance": dict(provenance),
        },
        path,
    )
    return path


def write_token_map_shard(
    *,
    layout: ExperimentLayout,
    rank: int,
    shard_id: int,
    rows: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
    cohort: str = "path",
) -> Path:
    """Atomically persist dense maps without expanding them into a CSV."""
    keys = set()
    normalized = []
    for source in rows:
        row = dict(source)
        key = case_key(row)
        if key in keys:
            raise ValueError(f"Duplicate token map inside shard: {key}")
        keys.add(key)
        label_pair = (str(row.get("label_string")), int(row.get("label_integer", -1)))
        if label_pair not in {("HALL", 0), ("REAL", 1)}:
            raise ValueError(f"Invalid label direction in token map: {label_pair}")
        methods = row.get("methods")
        if not isinstance(methods, Mapping) or not methods:
            raise ValueError("A token map must contain at least one method")
        widths = {
            int(torch.as_tensor(value).reshape(-1).numel())
            for value in methods.values()
        }
        if len(widths) != 1:
            raise ValueError(f"Token-map method widths differ: {sorted(widths)}")
        width = next(iter(widths))
        validity = torch.as_tensor(row.get("validity_mask")).reshape(-1)
        if int(validity.numel()) != width:
            raise ValueError("Token-map validity mask width differs from methods")
        for name, value in methods.items():
            tensor = torch.as_tensor(value)
            if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"Token-map method {name!r} is non-finite")
        normalized.append(row)
    stem = _cohort_stem(cohort, "token_maps")
    path = layout.path(
        f"shards/{stem}_rank{int(rank):02d}_shard_{int(shard_id):05d}.pt"
    )
    if path.exists():
        raise FileExistsError(f"Atomic token-map shard already exists: {path}")
    atomic_torch_save(
        {
            "schema_version": SCHEMA_VERSION,
            "rank": int(rank),
            "shard_id": int(shard_id),
            "cohort": str(cohort),
            "token_maps": normalized,
            "failures": [dict(row) for row in failures],
            "provenance": dict(provenance),
        },
        path,
    )
    return path


def iter_case_shards(layout: ExperimentLayout, *, cohort: str = "path"):
    stem = _cohort_stem(cohort, "case")
    for path in sorted(layout.path("shards").glob(f"{stem}_rank*_shard_*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Unexpected shard schema in {path}")
        yield path, payload


def audit_case_shards(
    layout: ExperimentLayout, *, cohort: str = "path"
) -> dict[str, Any]:
    cases = {}
    failures = []
    shard_rows = []
    for path, payload in iter_case_shards(layout, cohort=cohort):
        checksum = sha256_file(path)
        for row in payload.get("rows", ()):
            validate_measurement_row(row)
            key = case_key(row)
            if key in cases:
                raise ValueError(f"Duplicate case across shards: {key}")
            cases[key] = row
        failures.extend(dict(row) for row in payload.get("failures", ()))
        shard_rows.append(
            {
                "path": str(path.relative_to(layout.root)),
                "sha256": checksum,
                "rows": len(payload.get("rows", ())),
                "failures": len(payload.get("failures", ())),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "cohort": str(cohort),
        "shards": shard_rows,
        "unique_cases": len(cases),
        "failures": failures,
    }


def write_output_checksums(layout: ExperimentLayout) -> dict[str, Any]:
    excluded = {
        layout.path("manifests/output_checksums.json").resolve(),
        layout.path("handoff_bundle.tar.gz").resolve(),
    }
    files = [
        path
        for path in sorted(layout.root.rglob("*"))
        if path.is_file()
        and path.resolve() not in excluded
        # The bundle has a closed checksum manifest of its own.  Including it
        # here would create a recursive checksum cycle because it carries a
        # copy of this output manifest.
        and "handoff_bundle" not in path.relative_to(layout.root).parts
    ]
    payload = {
        str(path.relative_to(layout.root)): {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    }
    atomic_json_save(payload, layout.path("manifests/output_checksums.json"))
    return payload


def write_json_gzip(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    """Portable fallback when parquet dependencies are unavailable."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".tmp.", dir=str(destination.parent)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with gzip.open(temporary, "wt", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True))
                handle.write("\n")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_handoff_bundle(layout: ExperimentLayout) -> dict[str, Any]:
    """Copy only compact review artifacts and produce a checksummed bundle."""
    bundle = layout.path("handoff_bundle")
    selected = []
    for relative_root in (
        "reports",
        "metrics",
        "schemas",
        "manifests",
        "configs",
        "tables",
        "figures",
    ):
        source_root = layout.path(relative_root)
        for source in sorted(source_root.rglob("*")):
            if not source.is_file():
                continue
            if source.name.startswith("."):
                continue
            if source.stat().st_size > 64 * 1024 * 1024:
                continue
            relative = source.relative_to(layout.root)
            destination = bundle / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            selected.append(destination)
    loader_source = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "load_tc_fvpa_results.py"
    )
    if loader_source.is_file():
        loader_destination = bundle / "scripts" / loader_source.name
        loader_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(loader_source, loader_destination)
        selected.append(loader_destination)
    handoff_source = layout.path("reports/HANDOFF_TO_CHATGPT.md")
    if handoff_source.exists():
        destination = bundle / "HANDOFF_TO_CHATGPT.md"
        shutil.copy2(handoff_source, destination)
        selected.append(destination)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_unix": time.time(),
        "files": {
            str(path.relative_to(bundle)): {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(set(selected))
        },
    }
    atomic_json_save(manifest, bundle / "HANDOFF_MANIFEST.json")
    checksum_lines = [
        f"{entry['sha256']}  {name}"
        for name, entry in sorted(manifest["files"].items())
    ]
    checksum_path = bundle / "SHA256SUMS"
    temporary = checksum_path.with_suffix(f".tmp.{os.getpid()}")
    temporary.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    os.replace(temporary, checksum_path)
    return manifest
