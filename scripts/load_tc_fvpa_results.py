#!/usr/bin/env python3
"""Verified loader for TC-FVPA case shards and complete token maps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.tc_fvpa_artifacts import SCHEMA_VERSION, sha256_file  # noqa: E402


TOKEN_MAP_REQUIRED_FIELDS = (
    "model",
    "image_id",
    "case_id",
    "response_index",
    "prediction_position",
    "target_token_id",
    "label_string",
    "label_integer",
    "layer",
    "visual_grid",
    "validity_mask",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def expected_checksums(result_root: str | Path) -> dict[str, Any]:
    root = Path(result_root).resolve()
    path = root / "manifests" / "output_checksums.json"
    return _load_json(path) if path.exists() else {}


def verify_checksum(path: Path, root: Path, checksums: Mapping[str, Any]) -> None:
    relative = str(path.relative_to(root))
    expected = checksums.get(relative)
    if expected is None:
        raise KeyError(f"No persisted checksum for {relative}")
    observed = sha256_file(path)
    if observed != str(expected["sha256"]):
        raise ValueError(
            f"Checksum mismatch for {relative}: {observed} != {expected['sha256']}"
        )


def token_map_key(row: Mapping[str, Any]) -> str:
    return ":".join(
        str(row[name])
        for name in ("model", "image_id", "response_index", "layer")
    )


def validate_token_map(row: Mapping[str, Any], required_fields: Sequence[str]) -> None:
    missing = [field for field in required_fields if field not in row]
    if missing:
        raise KeyError(f"Token map misses fields {missing}")
    label_string = str(row["label_string"])
    label_integer = int(row["label_integer"])
    if (label_string, label_integer) not in {("HALL", 0), ("REAL", 1)}:
        raise ValueError(
            "Label direction must be HALL=0, REAL=1; got "
            f"{label_string}={label_integer}"
        )
    grid = tuple(int(value) for value in row["visual_grid"])
    if len(grid) != 2 or min(grid) <= 0:
        raise ValueError(f"Invalid visual grid {grid}")
    visual_count = grid[0] * grid[1]
    mask = torch.as_tensor(row["validity_mask"]).reshape(-1)
    if int(mask.numel()) != visual_count:
        raise ValueError(
            f"Validity mask has {mask.numel()} entries for grid {grid}"
        )


def iter_token_maps(
    result_root: str | Path,
    *,
    verify_checksums: bool = True,
    required_fields: Sequence[str] = TOKEN_MAP_REQUIRED_FIELDS,
    cohorts: Sequence[str] = ("path", "local"),
) -> Iterable[dict[str, Any]]:
    root = Path(result_root).resolve()
    checksums = expected_checksums(root) if verify_checksums else {}
    cohort_patterns = {
        "path": "token_maps_rank*_shard_*.pt",
        "local": "local_token_maps_rank*_shard_*.pt",
    }
    unknown = set(cohorts) - set(cohort_patterns)
    if unknown:
        raise ValueError(f"Unknown token-map cohorts {sorted(unknown)}")
    cohort_paths = [
        (cohort, sorted((root / "shards").glob(cohort_patterns[cohort])))
        for cohort in cohorts
    ]
    if not any(paths for _cohort, paths in cohort_paths):
        raise FileNotFoundError(f"No token-map shards under {root / 'shards'}")
    # Cohorts are priority ordered.  The default loads complete path maps first
    # and fills only cases absent from the larger local-only cohort.  Duplicate
    # cases within one cohort remain a hard error.
    emitted: set[str] = set()
    for cohort, paths in cohort_paths:
        seen_in_cohort: set[str] = set()
        for path in paths:
            if verify_checksums:
                verify_checksum(path, root, checksums)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if payload.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(f"Unexpected schema in {path}")
            rows = payload.get("token_maps")
            if not isinstance(rows, list):
                raise TypeError(f"{path} has no token_maps list")
            for source in rows:
                row = dict(source)
                validate_token_map(row, required_fields)
                key = token_map_key(row)
                if key in seen_in_cohort:
                    raise ValueError(f"Duplicate token-map case {key}")
                seen_in_cohort.add(key)
                if key in emitted:
                    continue
                emitted.add(key)
                yield row


def select_token_maps(
    result_root: str | Path,
    *,
    model: str | None = None,
    layer: int | None = None,
    method: str | None = None,
    verify_checksums: bool = True,
) -> list[dict[str, Any]]:
    rows = []
    for row in iter_token_maps(result_root, verify_checksums=verify_checksums):
        if model is not None and str(row["model"]) != str(model):
            continue
        if layer is not None and int(row["layer"]) != int(layer):
            continue
        selected = row
        if method is not None:
            methods = row.get("methods") or {}
            if method not in methods:
                raise KeyError(f"Case {token_map_key(row)} has no method {method!r}")
            selected = {
                **{key: row[key] for key in TOKEN_MAP_REQUIRED_FIELDS},
                "method": method,
                "scores": methods[method],
            }
        rows.append(selected)
    return rows


def token_maps_dataframe(rows: Sequence[Mapping[str, Any]]):
    """Return one row per case-token when pandas is available."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required for DataFrame output") from exc
    flattened = []
    for row in rows:
        method = row.get("method")
        if method is None:
            raise ValueError("Select one method before flattening token maps")
        scores = torch.as_tensor(row["scores"]).reshape(-1)
        valid = torch.as_tensor(row["validity_mask"]).bool().reshape(-1)
        for token_index in range(int(scores.numel())):
            flattened.append(
                {
                    "model": row["model"],
                    "image_id": row["image_id"],
                    "case_id": row["case_id"],
                    "response_index": row["response_index"],
                    "layer": row["layer"],
                    "label_string": row["label_string"],
                    "label_integer": row["label_integer"],
                    "method": method,
                    "visual_token_index": token_index,
                    "score": float(scores[token_index]),
                    "valid": bool(valid[token_index]),
                }
            )
    return pd.DataFrame(flattened)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root")
    parser.add_argument("--model")
    parser.add_argument("--layer", type=int)
    parser.add_argument("--method")
    parser.add_argument("--no-verify-checksums", action="store_true")
    parser.add_argument("--summary-json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = select_token_maps(
        args.result_root,
        model=args.model,
        layer=args.layer,
        method=args.method,
        verify_checksums=not args.no_verify_checksums,
    )
    summary = {
        "rows": len(rows),
        "models": sorted({str(row["model"]) for row in rows}),
        "layers": sorted({int(row["layer"]) for row in rows}),
        "method": args.method,
    }
    if args.summary_json:
        Path(args.summary_json).write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
