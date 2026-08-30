#!/usr/bin/env python3
"""Coordinator for sharded TC-FVPA extraction, causal validation and reports."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.tc_fvpa_common import (  # noqa: E402
    add_common_arguments,
    print_dry_run,
    validate_common_args,
)


def parse_args() -> argparse.Namespace:
    parser = add_common_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument(
        "--phases", default="local,path,fp32,shapley,analyze,counterfactuals,reports"
    )
    parser.add_argument("--local-images", type=int, default=0)
    parser.add_argument("--path-images", type=int, default=0)
    parser.add_argument("--fp32-images", type=int, default=0)
    parser.add_argument("--shapley-images", type=int, default=0)
    return parser.parse_args()


def _base_args(args, common, *, shard_id=None, device=None):
    values = [
        "--model",
        args.model,
        "--device",
        device or args.device,
        "--devices",
        ",".join(common["devices"]),
        "--output-dir",
        common["output_dir"],
        "--seed",
        str(common["seed"]),
        "--layers",
        ",".join(map(str, common["layers"])),
        "--target-scalars",
        ",".join(common["target_scalars"]),
        "--integration-points",
        ",".join(map(str, common["integration_points"])),
        "--config",
        common["config"],
        "--num-shards",
        str(args.num_shards),
        "--shard-id",
        str(args.shard_id if shard_id is None else shard_id),
    ]
    values.append("--resume" if args.resume else "--no-resume")
    if args.smoke:
        values.append("--smoke")
    if args.formal:
        values.append("--formal")
    return values


def _run(command, log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("COMMAND: " + " ".join(command) + "\n")
        handle.flush()
        completed = subprocess.run(
            command,
            cwd=str(ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit {completed.returncode}; see {log_path}"
        )


def _run_parallel(commands):
    processes = []
    handles = []
    try:
        for command, log_path in commands:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("a", encoding="utf-8")
            handle.write("COMMAND: " + " ".join(command) + "\n")
            handle.flush()
            handles.append(handle)
            processes.append(
                (command, log_path, subprocess.Popen(
                    command,
                    cwd=str(ROOT),
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                ))
            )
        failures = []
        for command, log_path, process in processes:
            return_code = process.wait()
            if return_code:
                failures.append((return_code, command, log_path))
        if failures:
            raise RuntimeError(
                "Parallel phase failed: "
                + "; ".join(
                    f"exit={code} log={log}" for code, _command, log in failures
                )
            )
    finally:
        for handle in handles:
            handle.close()


def main() -> None:
    args = parse_args()
    common = validate_common_args(args)
    phases = [value for value in args.phases.split(",") if value]
    known = {
        "local",
        "path",
        "fp32",
        "shapley",
        "analyze",
        "counterfactuals",
        "reports",
    }
    if set(phases) - known:
        raise ValueError(f"Unknown phases {sorted(set(phases) - known)}")
    devices = common["devices"]
    plan = []
    shard_count = int(args.num_shards)
    if "local" in phases:
        for shard in range(shard_count):
            command = [
                sys.executable,
                "scripts/run_tc_fvpa_path_attribution.py",
                *_base_args(args, common, shard_id=shard, device=devices[shard % len(devices)]),
                "--attribution-mode",
                "local",
                "--audit-vector-cases",
                "0",
                "--num-images",
                str(args.local_images or (500 if args.formal else (3 if args.smoke else 12))),
            ]
            plan.append(("local", shard, command))
    if "path" in phases:
        for shard in range(shard_count):
            command = [
                sys.executable,
                "scripts/run_tc_fvpa_path_attribution.py",
                *_base_args(args, common, shard_id=shard, device=devices[shard % len(devices)]),
                "--attribution-mode",
                "path",
                "--num-images",
                str(args.path_images or (200 if args.formal else (3 if args.smoke else 12))),
            ]
            plan.append(("path", shard, command))
    if "fp32" in phases:
        for shard in range(shard_count):
            command = [
                sys.executable,
                "scripts/run_tc_fvpa_fp32_causal.py",
                *_base_args(args, common, shard_id=shard, device=devices[shard % len(devices)]),
                "--num-images",
                str(args.fp32_images or (100 if args.formal else (3 if args.smoke else 12))),
            ]
            plan.append(("fp32", shard, command))
    if "shapley" in phases:
        for shard in range(shard_count):
            command = [
                sys.executable,
                "scripts/run_tc_fvpa_shapley.py",
                *_base_args(args, common, shard_id=shard, device=devices[shard % len(devices)]),
                "--num-images",
                str(args.shapley_images or (50 if args.formal else (2 if args.smoke else 12))),
            ]
            plan.append(("shapley", shard, command))
    for phase, script in (
        ("analyze", "scripts/analyze_tc_fvpa_comprehensive.py"),
        ("counterfactuals", "scripts/run_tc_fvpa_counterfactuals.py"),
        ("reports", "scripts/build_tc_fvpa_reports.py"),
    ):
        if phase in phases:
            plan.append((phase, 0, [sys.executable, script, *_base_args(args, common)]))
    if args.dry_run:
        print_dry_run(
            "comprehensive",
            common,
            {"phases": phases, "commands": [command for _phase, _shard, command in plan]},
        )
        return
    log_root = Path(common["output_dir"]) / "logs"
    for phase in phases:
        phase_commands = [
            (command, log_root / f"{phase}_rank{shard:02d}.log")
            for item_phase, shard, command in plan
            if item_phase == phase
        ]
        if not phase_commands:
            continue
        if phase in {"local", "path", "fp32", "shapley"} and len(phase_commands) > 1:
            _run_parallel(phase_commands)
        else:
            for command, log_path in phase_commands:
                _run(command, log_path)
        print(f"[TC-FVPA coordinator] phase {phase} complete", flush=True)


if __name__ == "__main__":
    main()
