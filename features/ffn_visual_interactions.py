"""Region-level sampled Shapley and interaction helpers for visual writes."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Sequence

import torch


@dataclass(frozen=True)
class SampledShapleyResult:
    values: torch.Tensor
    standard_errors: torch.Tensor
    running_estimates: torch.Tensor
    permutation_count: int
    total_game_effect: float
    completeness_absolute_error: float


@dataclass(frozen=True)
class SampledVectorShapleyResult:
    values: torch.Tensor
    standard_errors: torch.Tensor
    running_estimates: torch.Tensor
    permutation_count: int
    total_game_effect: torch.Tensor
    completeness_vector: torch.Tensor
    completeness_relative_error: float


def aggregate_region_writes(
    writes: torch.Tensor, regions: Sequence[Sequence[int]]
) -> torch.Tensor:
    if writes.ndim != 2:
        raise ValueError("writes must have shape [M,D]")
    assigned = set()
    result = []
    for region in regions:
        indices = [int(value) for value in region]
        if not indices:
            raise ValueError("regions must not be empty")
        if min(indices) < 0 or max(indices) >= int(writes.shape[0]):
            raise IndexError(f"Invalid region indices {indices}")
        overlap = assigned & set(indices)
        if overlap:
            raise ValueError(f"Regions overlap at tokens {sorted(overlap)}")
        assigned.update(indices)
        result.append(writes[indices].sum(dim=0))
    return torch.stack(result)


def regular_grid_regions(height: int, width: int, region_count: int) -> list[list[int]]:
    """Deterministically partition a visual grid into 8 or 16 rectangles."""
    height = int(height)
    width = int(width)
    region_count = int(region_count)
    if height <= 0 or width <= 0:
        raise ValueError("grid dimensions must be positive")
    if region_count not in {8, 16}:
        raise ValueError("region_count must be 8 or 16")
    row_groups, column_groups = ((2, 4) if region_count == 8 else (4, 4))
    regions = [[] for _ in range(region_count)]
    for row in range(height):
        row_group = min(row_groups - 1, row * row_groups // height)
        for column in range(width):
            column_group = min(column_groups - 1, column * column_groups // width)
            regions[row_group * column_groups + column_group].append(row * width + column)
    if any(not region for region in regions):
        raise ValueError(
            f"Grid {height}x{width} is too small for {region_count} nonempty regions"
        )
    return regions


def sampled_shapley(
    *,
    value_from_active_mask: Callable[[torch.Tensor], torch.Tensor],
    region_count: int,
    permutations: int,
    seed: int,
    persist_every: int = 1,
) -> SampledShapleyResult:
    """Monte-Carlo permutation Shapley with running estimates and SEs.

    The value function receives a boolean mask of active regions.  Callers are
    responsible for defining the baseline and for never deriving regions from
    intervention outcomes.
    """
    count = int(region_count)
    samples = int(permutations)
    if count <= 0 or samples <= 0:
        raise ValueError("region_count and permutations must be positive")
    if int(persist_every) <= 0:
        raise ValueError("persist_every must be positive")
    rng = random.Random(int(seed))
    contributions = torch.empty(samples, count, dtype=torch.float64)
    running = []
    empty = torch.zeros(count, dtype=torch.bool)
    full = torch.ones(count, dtype=torch.bool)
    baseline = float(value_from_active_mask(empty).detach().cpu())
    complete = float(value_from_active_mask(full).detach().cpu())
    for sample_index in range(samples):
        order = list(range(count))
        rng.shuffle(order)
        active = torch.zeros(count, dtype=torch.bool)
        previous = baseline
        for region in order:
            active[region] = True
            current = float(value_from_active_mask(active.clone()).detach().cpu())
            contributions[sample_index, region] = current - previous
            previous = current
        if (sample_index + 1) % int(persist_every) == 0 or sample_index + 1 == samples:
            running.append(contributions[: sample_index + 1].mean(dim=0).clone())
    values = contributions.mean(dim=0)
    if samples > 1:
        standard_errors = contributions.std(dim=0, unbiased=True) / math.sqrt(samples)
    else:
        standard_errors = torch.full_like(values, float("nan"))
    total_effect = complete - baseline
    return SampledShapleyResult(
        values=values,
        standard_errors=standard_errors,
        running_estimates=torch.stack(running),
        permutation_count=samples,
        total_game_effect=total_effect,
        completeness_absolute_error=abs(float(values.sum()) - total_effect),
    )


def sampled_vector_shapley(
    *,
    value_from_active_mask: Callable[[torch.Tensor], torch.Tensor],
    region_count: int,
    permutations: int,
    seed: int,
    persist_every: int = 1,
    eps: float = 1e-12,
) -> SampledVectorShapleyResult:
    """Monte-Carlo permutation Shapley for a vector-valued game."""
    count = int(region_count)
    samples = int(permutations)
    every = int(persist_every)
    if count <= 0 or samples <= 0:
        raise ValueError("region_count and permutations must be positive")
    if every <= 0:
        raise ValueError("persist_every must be positive")
    empty = torch.zeros(count, dtype=torch.bool)
    full = torch.ones(count, dtype=torch.bool)
    baseline = value_from_active_mask(empty).detach().cpu().to(torch.float64)
    complete = value_from_active_mask(full).detach().cpu().to(torch.float64)
    if baseline.ndim != 1 or complete.shape != baseline.shape:
        raise ValueError("vector game values must be matching [D] tensors")

    rng = random.Random(int(seed))
    contributions = torch.empty(samples, count, int(baseline.numel()), dtype=torch.float64)
    running = []
    for sample_index in range(samples):
        order = list(range(count))
        rng.shuffle(order)
        active = torch.zeros(count, dtype=torch.bool)
        previous = baseline
        for region in order:
            active[region] = True
            current = (
                value_from_active_mask(active.clone())
                .detach()
                .cpu()
                .to(torch.float64)
            )
            if current.shape != baseline.shape:
                raise ValueError("vector game returned inconsistent shapes")
            contributions[sample_index, region] = current - previous
            previous = current
        if (sample_index + 1) % every == 0 or sample_index + 1 == samples:
            running.append(contributions[: sample_index + 1].mean(dim=0).clone())

    values = contributions.mean(dim=0)
    standard_errors = (
        contributions.std(dim=0, unbiased=True) / math.sqrt(samples)
        if samples > 1
        else torch.full_like(values, float("nan"))
    )
    total_effect = complete - baseline
    completeness_vector = values.sum(dim=0) - total_effect
    completeness_relative_error = float(
        completeness_vector.norm() / total_effect.norm().clamp_min(float(eps))
    )
    return SampledVectorShapleyResult(
        values=values,
        standard_errors=standard_errors,
        running_estimates=torch.stack(running),
        permutation_count=samples,
        total_game_effect=total_effect,
        completeness_vector=completeness_vector,
        completeness_relative_error=completeness_relative_error,
    )
